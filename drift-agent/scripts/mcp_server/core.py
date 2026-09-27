"""MCP server exposing dataset-drift-detection tools. Stateless and
read-only: every tool takes two CSV paths directly (reference_path,
current_path) — there is no run_id, no data/runs directory, no
train/apply/export cycle, and nothing here ever writes to either input file.

PSI (population stability index) is the primary drift signal, not the
p-value: a KS-test/chi-square p-value gets more "significant" as row counts
grow and less powered as they shrink, so the same real-world shift can flip
significance purely from sample size. PSI compares bin *proportions*, so it
stays comparable across dataset sizes — that's why production drift
monitoring conventionally gates severity on PSI thresholds (<0.1 none,
0.1-0.25 moderate, >=0.25 severe) while still surfacing the p-value for
context, not the other way round.

Run: python -m mcp_server.server  (cwd=scripts/; agent.py does this)

Shared kernel of the package: imports, constants, run storage, every
helper and the `mcp` instance. Tools live in one module per skill."""

import json

import sys
from pathlib import Path

import numpy as np

import pandas as pd

from mcp.server.fastmcp import FastMCP

from scipy import stats

mcp = FastMCP("drift-agent")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for core.*
from core import toolguard  # noqa: E402
from core.data_quality import identifier_columns  # noqa: E402
from core.datasource import read_table  # noqa: E402

toolguard.install(
    mcp
)  # every tool's paths/models checked before it runs (core/toolguard.py)

PSI_MODERATE = 0.1

PSI_SEVERE = 0.25

N_BINS = 10

# Under no drift, PSI over B bins behaves like (1/n_ref + 1/n_cur) x a
# chi-square with B-1 degrees of freedom: with 200 current rows and 10 bins
# pure sampling noise reaches ~0.1 — a "moderate drift" that is not there.
# A PSI below the 99th percentile of that noise is reported, not flagged.
NOISE_QUANTILE = 0.99
SMALL_SAMPLE_ROWS = 500

# A column going null is the most common production failure (a broken join,
# a feed that stopped) and is invisible to a PSI computed on non-null values.
MISSING_MODERATE_PTS = 5.0
MISSING_SEVERE_PTS = 20.0

# Domain classifier (reference vs current): cross-validated ROC-AUC. 0.5 =
# the two tables are indistinguishable; it rises with ANY shift, including
# joint ones no single column shows.
MV_MODERATE_AUC = 0.6
MV_SEVERE_AUC = 0.75
MV_MAX_ROWS = 20_000


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _read_csv_or_error(
    path: str, label: str
) -> tuple[pd.DataFrame | None, dict | None]:
    """Read a data file in any supported format (csv, parquet, compressed csv,
    json, xlsx, archives) — the name is historical."""
    file = Path(path)
    if not file.exists():
        return None, {"error": f"no {label} file at '{path}'"}
    df = read_table(file)
    if df.empty:
        return None, {"error": f"{label} file '{path}' has no rows"}
    return df, None


def _structural_columns(df: pd.DataFrame) -> dict:
    """Columns whose 'drift' is guaranteed and meaningless: timestamps (the
    current data is later by definition) and identifiers (new customers,
    new transaction ids). Excluded from the verdict, listed with the reason."""
    out = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = "timestamp — current data is later by definition"
        elif s.dtype == object and s.notna().any():
            sample = s.dropna().astype(str).head(200)
            if sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
                out[col] = "timestamp — current data is later by definition"
    for col, signal in identifier_columns(df.drop(columns=list(out)), None).items():
        if signal["strength"] == "strong":
            out[col] = f"identifier — {signal['reason']}"
    return out


def psi_noise_floor(n_ref: int, n_cur: int, bins: int) -> float:
    return float(stats.chi2.ppf(NOISE_QUANTILE, max(bins - 1, 1)) * (1 / max(n_ref, 1) + 1 / max(n_cur, 1)))


def _severity(psi: float) -> str:
    if psi >= PSI_SEVERE:
        return "severe"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "none"


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    """Population Stability Index over matching bins. A small floor keeps a
    zero-proportion bin from producing -inf/NaN via log(0) or divide-by-zero
    — an empty-vs-nonempty bin is exactly the kind of shift PSI exists to
    catch, not a case to error out on."""
    eps = 1e-4
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _numeric_drift(ref: pd.Series, cur: pd.Series) -> dict:
    ref, cur = ref.dropna(), cur.dropna()
    if ref.empty or cur.empty:
        return {"dtype": "numeric", "psi": 0.0, "test": "ks_2samp", "statistic": None, "p_value": None,
                "bins_used": 1, "note": "no non-missing values on one side — see the missing share"}
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, N_BINS + 1)))
    if len(edges) < 3:
        # ponytail: a (near-)constant reference column degrades to one bin —
        # PSI still returns a number instead of crashing, but loses power to
        # spot a current column that moved off that constant. Add a
        # binary same-vs-different-value bin here if that case ever matters.
        edges = np.array(
            [ref.min(), ref.max() if ref.max() > ref.min() else ref.min() + 1.0]
        )
    edges = edges.astype(float)
    edges[0], edges[-1] = (
        -np.inf,
        np.inf,
    )  # outermost bins catch new tail values, not just the ref range
    # Quantile edges collapse wherever the reference has heavy ties. A column
    # that is 90% zeros — call counts, data volume, revenue: the telecom norm
    # — yields a handful of distinct edges, clears the len(edges) < 3 guard
    # above, and then reports a PSI computed over 3 bins as though it had 10.
    # That is a real loss of resolution, and silent, so say it.
    n_bins_effective = len(edges) - 1
    tie_note = None
    if n_bins_effective < N_BINS:
        modal_share = (
            float(ref.value_counts(normalize=True).iloc[0]) if len(ref) else 0.0
        )
        tie_note = (
            f"reference column collapsed to {n_bins_effective} quantile bin(s) instead of {N_BINS} "
            f"(most common value holds {modal_share:.0%} of rows) — PSI here has reduced power to detect a "
            "shift WITHIN the dominant value's mass; it still catches movement between the value and the rest."
        )

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_counts / max(len(ref), 1)
    cur_pct = cur_counts / max(len(cur), 1)
    psi = _psi(ref_pct, cur_pct)
    ks_stat, ks_p = stats.ks_2samp(ref, cur)
    outer_tail_pct = (
        float(cur_pct[0] + cur_pct[-1]) if len(cur_pct) > 1 else float(cur_pct[0])
    )
    result = {
        "dtype": "numeric",
        "psi": round(psi, 4),
        "test": "ks_2samp",
        "statistic": round(float(ks_stat), 4),
        "p_value": round(float(ks_p), 4),
        "reference_mean": round(float(ref.mean()), 4),
        "current_mean": round(float(cur.mean()), 4),
        "reference_std": round(float(ref.std()), 4),
        "current_std": round(float(cur.std()), 4),
        "current_pct_in_outer_bins": round(outer_tail_pct, 4),
        "bins_used": n_bins_effective,
    }
    if tie_note:
        result["low_resolution_note"] = tie_note
    if outer_tail_pct > 0.5:
        result["note"] = (
            f"{outer_tail_pct:.0%} of current rows fall in the unbounded outer bins — "
            "PSI may understate a large tail shift compressed into one bin."
        )
    return result


def _categorical_drift(ref: pd.Series, cur: pd.Series) -> dict:
    ref, cur = ref.dropna(), cur.dropna()
    if ref.empty or cur.empty:
        return {"dtype": "categorical", "psi": 0.0, "test": "chi2_contingency", "categories_compared": 1,
                "new_categories": [], "new_category_prevalence": 0.0, "statistic": None, "p_value": None,
                "note": "no non-missing values on one side — see the missing share"}
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy()
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy()
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    psi = _psi(ref_pct, cur_pct)
    new_categories = sorted(set(cur.unique()) - set(ref.unique()))
    new_category_prevalence = (
        float(cur.isin(new_categories).sum() / max(len(cur), 1))
        if new_categories
        else 0.0
    )
    result = {
        "dtype": "categorical",
        "psi": round(psi, 4),
        "test": "chi2_contingency",
        "categories_compared": len(categories),
        "new_categories": new_categories,
        "new_category_prevalence": round(new_category_prevalence, 4),
    }
    # eps-flooring in _psi dampens a brand-new category's PSI contribution
    # more than its real-world prevalence warrants — floor severity directly
    # instead of trusting PSI alone for this case.
    if new_category_prevalence >= 0.01 and _severity(psi) == "none":
        result["severity_override"] = "moderate"
        result["note"] = (
            f"new categor{'y' if len(new_categories) == 1 else 'ies'} "
            f"{new_categories} at {new_category_prevalence:.1%} of current rows; "
            "flagged moderate regardless of PSI, which under-weights new categories via eps-flooring."
        )
    try:
        chi2, p, _, _ = stats.chi2_contingency(np.array([ref_counts, cur_counts]))
        result["statistic"] = round(float(chi2), 4)
        result["p_value"] = round(float(p), 4)
    except ValueError as exc:
        # Degenerate contingency table (e.g. only one shared category) — PSI
        # above is still a valid verdict, the chi-square just doesn't apply.
        result["statistic"] = None
        result["p_value"] = None
        result["note"] = f"chi2_contingency not applicable: {exc}"
    return result


def _missing_severity(ref_pct: float, cur_pct: float) -> str:
    change = abs(cur_pct - ref_pct)
    if change >= MISSING_SEVERE_PTS:
        return "severe"
    if change >= MISSING_MODERATE_PTS and (min(ref_pct, cur_pct) == 0 or max(ref_pct, cur_pct) / min(ref_pct, cur_pct) >= 2):
        return "moderate"
    return "none"


def _column_drift(ref: pd.Series, cur: pd.Series) -> dict:
    """One column: its distribution (PSI, severity only above the sampling
    noise floor), its missing share, and its type."""
    ref_numeric, cur_numeric = pd.api.types.is_numeric_dtype(ref), pd.api.types.is_numeric_dtype(cur)
    if ref_numeric and cur_numeric:
        result = _numeric_drift(ref, cur)
        bins = result["bins_used"]
    else:
        result = _categorical_drift(ref.astype(str).where(ref.notna()), cur.astype(str).where(cur.notna()))
        bins = result["categories_compared"]
    severity = _severity(result["psi"])
    floor = psi_noise_floor(int(ref.notna().sum()), int(cur.notna().sum()), bins)
    result["psi_noise_floor"] = round(floor, 4)
    if severity != "none" and result["psi"] < floor:
        severity = "none"
        result["noise_note"] = (f"PSI {result['psi']} is within sampling noise for these sample sizes "
                                f"(99th percentile under no drift: {floor:.3f}) — not flagged")
    severity = result.pop("severity_override", None) or severity
    reasons = ["distribution"] if severity != "none" else []

    ref_missing, cur_missing = round(float(ref.isna().mean()) * 100, 2), round(float(cur.isna().mean()) * 100, 2)
    result["reference_missing_pct"], result["current_missing_pct"] = ref_missing, cur_missing
    missing = _missing_severity(ref_missing, cur_missing)
    if missing != "none":
        reasons.append(f"missing share {ref_missing}% -> {cur_missing}%")
        severity = max(severity, missing, key=SEVERITY_ORDER.get)
    if ref_numeric != cur_numeric:
        result["type_change"] = f"{'numeric' if ref_numeric else 'text'} -> {'numeric' if cur_numeric else 'text'}"
        reasons.append(f"type changed {result['type_change']} — a model fitted on the old type cannot use it")
        severity = "severe"
    result["severity"] = severity
    result["drifted"] = severity != "none"
    result["drift_reasons"] = reasons
    return result


SEVERITY_ORDER = {"none": 0, "moderate": 1, "severe": 2}


def _overall_verdict(
    columns: dict, key_columns: list[str] | None, target_column: str | None
) -> dict:
    """The verdict is driven by the columns the MODEL ACTUALLY USES.

    Max-severity-over-every-column is what this used to do, and on a
    200-column telecom extract it means one drifting column nobody models
    outranks 199 stable predictive ones — so "retrain now" fires on almost
    every check and stops carrying information. When key_columns is known
    (the exporting agent's own explainer named them), unmodelled columns are
    still reported, just not allowed to set the verdict on their own.

    The target is scored separately: a shift in the label is prior/label
    shift, a different problem with a different fix from covariate shift on
    the features, and averaging them into one number hides which happened."""
    feature_cols = {c: r for c, r in columns.items() if c != target_column}
    key = [c for c in (key_columns or []) if c in feature_cols]
    deciding = {c: feature_cols[c] for c in key} if key else feature_cols

    worst = max((SEVERITY_ORDER[r["severity"]] for r in deciding.values()), default=0)
    severity = {v: k for k, v in SEVERITY_ORDER.items()}[worst]
    drifted_deciding = sorted(c for c, r in deciding.items() if r["drifted"])

    basis = (
        f"the {len(key)} column(s) this model actually leans on ({', '.join(key)})"
        if key
        else "every shared feature column — no key_columns were supplied, so nothing distinguishes a column the model "
        "depends on from one it ignores; pass profile_path or key_columns to sharpen this"
    )

    # Drift among columns the model doesn't use is context, not a verdict.
    incidental = (
        sorted(
            c
            for c, r in feature_cols.items()
            if r["drifted"] and (not key or c not in key)
        )
        if key
        else []
    )

    target_drift = None
    if target_column and target_column in columns:
        t = columns[target_column]
        target_drift = {
            "column": target_column,
            "severity": t["severity"],
            "psi": t["psi"],
            "interpretation": (
                "the LABEL distribution moved — this is prior/label shift, not covariate shift. The features can "
                "be perfectly stable and this still invalidates a tuned decision threshold, because the threshold "
                "was chosen against the old base rate. Re-tune the threshold before concluding the model is stale."
                if t["drifted"]
                else "label distribution is stable — the base rate a tuned threshold was chosen against still holds."
            ),
        }

    recommendation = {
        "severe": "retrain now — a column this model depends on has moved substantially; treat its current "
        "predictions as unreliable until it is refit or re-validated.",
        "moderate": "consider retraining — real shift in a column the model uses, not yet extreme; keep monitoring.",
        "none": "no meaningful shift in the columns this model uses.",
    }[severity]

    return {
        "overall_severity": severity,
        "verdict_basis": basis,
        "drifted_key_columns": drifted_deciding,
        "drifted_incidental_columns": incidental,
        "target_drift": target_drift,
        "recommendation": recommendation,
        # Said every time, including when the verdict is "none". This tool
        # compares two tables of INPUTS. It never scores the model, never sees
        # a label for the current period, and therefore cannot observe the
        # failure where inputs look identical and the relationship between
        # inputs and outcome has changed underneath them.
        "what_this_did_not_check": [
            "prediction drift — the model was never run on the current data here",
            "concept drift — the input/outcome relationship can change with every input distribution unchanged",
            "performance — no current-period labels were supplied, so accuracy/recall now is unknown",
        ],
        "scope_caveat": "this is a DATA drift check. 'No drift' means the inputs look like training data; it does "
        "not mean the model is still accurate.",
    }


def _read_profile(profile_path: str) -> tuple[dict | None, dict | None]:
    """A monitoring profile is the plain-JSON sidecar a training agent writes
    next to its exported .pkl (see classification-agent's export_model). It
    names the training fold it saved, the target, and the columns the model's
    own explainer said it leans on.

    JSON on purpose, not the pickle: this agent has no sklearn dependency and
    no copy of the exporting agent's transformer classes, so unpickling a
    model here would fail. The contract travels as data."""
    file = Path(profile_path)
    if not file.exists():
        return None, {"error": f"no monitoring profile at '{profile_path}'"}
    try:
        profile = json.loads(file.read_text())
    except json.JSONDecodeError as exc:
        return None, {"error": f"'{profile_path}' is not valid JSON: {exc}"}
    if not profile.get("reference_path"):
        return None, {
            "error": f"'{profile_path}' has no reference_path — it isn't a monitoring profile"
        }
    return profile, None


def domain_classifier(ref: pd.DataFrame, cur: pd.DataFrame, columns: list, seed: int = 42) -> dict:
    """Can a model tell the two tables apart? A gradient-boosted classifier
    is trained to separate reference rows from current rows (balanced samples
    of up to MV_MAX_ROWS each) and scored by 3-fold cross-validated ROC-AUC.
    0.5 = indistinguishable. It sees every column at once, so it catches
    shifts in how columns move TOGETHER that no per-column PSI can — and its
    permutation importance names the columns that give the difference away."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

    n = min(len(ref), len(cur), MV_MAX_ROWS)
    both = pd.concat([ref[columns].sample(n, random_state=seed), cur[columns].sample(n, random_state=seed)],
                     ignore_index=True)
    y = np.r_[np.zeros(n), np.ones(n)]
    for col in columns:
        if not pd.api.types.is_numeric_dtype(both[col]) or pd.api.types.is_bool_dtype(both[col]):
            values = both[col].astype(str).where(both[col].notna())
            both[col] = values.astype("category") if values.nunique() <= 250 else values.astype("category").cat.codes
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, categorical_features="from_dtype",
                                         random_state=seed)
    proba = cross_val_predict(clf, both, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    X_fit, X_hold, y_fit, y_hold = train_test_split(both, y, test_size=0.3, stratify=y, random_state=seed)
    imp = permutation_importance(clf.fit(X_fit, y_fit), X_hold, y_hold, scoring="roc_auc", n_repeats=3,
                                 random_state=seed)
    ranked = sorted(zip(columns, imp.importances_mean), key=lambda kv: -kv[1])
    severity = "severe" if auc >= MV_SEVERE_AUC else "moderate" if auc >= MV_MODERATE_AUC else "none"
    return {"auc": round(auc, 4), "severity": severity, "rows_per_side": n,
            "top_columns": [{"column": c, "auc_drop": round(float(v), 4)} for c, v in ranked[:5] if v > 0.002]}
