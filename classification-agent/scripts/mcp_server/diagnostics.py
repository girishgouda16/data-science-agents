"""The checks a data scientist runs before and after training that aren't
"clean the data" (data-cleaning), "look at the raw data" (eda), or "pick/
tune a model" (modeling): is any column suspiciously predictive (leakage),
how much better than guessing is the model really (baseline), where does it
fail (error analysis / segments), and is its score a fluke of one
train/test split (stability). See the `diagnostics` skill."""

import json
import re

import joblib
import numpy as np
import pandas as pd
from core.datasource import read_table
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from .core import (
    _build_pipeline,
    _cv_sample,
    _cv_scoring,
    _evaluate,
    _load_meta,
    _load_split,
    _run_dir,
    ECE_CONCERN,
    _save_meta,
    _validation_folds,
    _validation_proba,
    positive_index,
    positive_label,
    mcp,
    split_keys,
)
from .data_cleaning import identifier_columns


@mcp.tool()
def detect_data_leakage(run_id: str) -> str:
    """Flags leakage/non-feature smells on the run's training fold: (1) any
    column — numeric or categorical — that nearly decides the label ALONE
    (`near_perfect_predictors`: |corr| > 0.98, or single-column out-of-fold
    ROC-AUC >= 0.98; almost always a proxy/copy of the label or a field
    recorded after the outcome — the leakage gate FAILS on these until they
    are dropped or acknowledged), (2) `possible_post_event_columns` by name
    (weak, report and ask), plus every
    identifier-shaped column, using the SAME shared heuristic
    propose_drop_columns uses (data_cleaning.identifier_columns): near-unique
    any dtype, high-cardinality categorical by absolute count, an
    identifier-shaped categorical by uniqueness RATIO (the ticket/order/
    invoice/claim-number shape that clears both other bars), and a
    name-pattern match.

    Each flagged column carries a `strength`: "strong" (a cardinality
    signal — it cannot generalize through a one-hot encoder, only memorize)
    or "weak" (name match alone, which can be a false positive on a real
    feature). The readiness gate blocks on strong signals only.

    Read-only, heuristic — a flagged column still needs domain judgment
    before dropping, but leaving a STRONG one in the model is what the
    readiness gate will fail the run for unless it is explicitly
    acknowledged."""
    train, _, meta = _load_split(run_id)
    target = meta["target"]
    # The split keys never reach the model (_build_pipeline drops them), so
    # they are not leakage candidates — screening them would only tempt a
    # drop that breaks grouped/temporal validation.
    keys = split_keys(meta)
    train = train.drop(columns=keys, errors="ignore")
    numeric = train.select_dtypes(include="number").drop(
        columns=[target], errors="ignore"
    )
    suspects = {}
    if not numeric.empty:
        y = (
            pd.factorize(train[target])[0]
            if not pd.api.types.is_numeric_dtype(train[target])
            else train[target]
        )
        corr = numeric.corrwith(pd.Series(y, index=train.index)).abs()
        suspects = {c: round(float(v), 4) for c, v in corr[corr > 0.98].items()}

    signals = identifier_columns(train, target)
    by_rule = lambda rule: [c for c, s in signals.items() if s["rule"] == rule]  # noqa: E731
    strong_ids = {c for c, s in signals.items() if s["strength"] == "strong"}
    single = _single_column_auc(
        train,
        target,
        positive_label(meta, train[target]),
        [c for c in train.columns if c != target and c not in strong_ids],
    )
    near_perfect = sorted(
        set(suspects) | {c for c, a in single.items() if a >= NEAR_PERFECT_AUC}
    )
    post_event = [
        c for c in train.columns if c != target and _POST_EVENT_NAME.search(c)
    ]
    result = {
        "run_id": run_id,
        # Not screened: kept for grouped/temporal validation, excluded from the model.
        "split_keys": keys,
        "near_perfect_correlation_with_target": suspects,
        # One column that almost decides the label on its own — any dtype,
        # scored out-of-fold so a category->label lookup can't flatter itself.
        # Almost always a copy/proxy of the label or a field filled in after
        # the outcome. The leakage gate FAILS on these until dropped or
        # acknowledged with a domain reason.
        "near_perfect_predictors": near_perfect,
        "single_column_auc_top": dict(
            sorted(single.items(), key=lambda kv: -kv[1])[:5]
        ),
        # Weak, name-only: a field that sounds recorded AFTER the outcome.
        # Report it and ask where it comes from; it doesn't gate.
        "possible_post_event_columns": post_event,
        "duplicates_dropped_before_split": meta.get("duplicates_dropped"),
        "identifier_columns": signals,
        "strong_identifier_columns": [
            c for c, s in signals.items() if s["strength"] == "strong"
        ],
        # Legacy per-rule lists — kept so existing report sections and callers
        # keep working now that all four rules come from one shared function.
        "id_like_columns": by_rule("near_unique"),
        "high_cardinality_categorical_columns": by_rule("high_cardinality"),
        "identifier_ratio_columns": by_rule("identifier_ratio"),
        "name_flagged_columns": by_rule("name_pattern"),
        "clear": not suspects and not signals and not near_perfect and not post_event,
    }
    meta["leakage"] = result
    _save_meta(run_id, meta)
    return json.dumps(result)


NEAR_PERFECT_AUC = 0.98
_POST_EVENT_NAME = re.compile(
    r"(^|_)(after|post|outcome|result|resolution|resolved|closed|refund(ed)?|chargeback|cancel(l)?ed|"
    r"final|settled|disposition|recovered|written_off)(_|$)",
    re.IGNORECASE,
)


def _single_column_auc(
    train: pd.DataFrame, target: str, pos_label, columns: list[str]
) -> dict:
    """ROC-AUC of each column ALONE as a predictor of the label. Binary:
    numeric columns by their raw value (either direction), anything else by
    an out-of-fold target-rate encoding, so a category seen once can't
    predict itself. Multiclass: every column by an out-of-fold class-rate
    encoding (numerics in quantile bins, so a label copy stored as codes in
    any order is still caught), scored as one-vs-rest macro AUC. Capped at
    LABEL_RULE_SAMPLE_ROWS rows."""
    if train[target].nunique() < 2:
        return {}
    if train[target].nunique() > 2:
        return _single_column_auc_multiclass(train, target, columns)
    df = (
        train
        if len(train) <= LABEL_RULE_SAMPLE_ROWS
        else train.sample(LABEL_RULE_SAMPLE_ROWS, random_state=42)
    )
    y = (df[target] == pos_label).astype(int).to_numpy()
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, y)
    )
    scores = {}
    for col in columns:
        x = df[col]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            auc = roc_auc_score(y, x.fillna(x.median()).to_numpy())
            scores[col] = round(float(max(auc, 1 - auc)), 4)
            continue
        encoded = np.empty(len(df))
        keys = x.astype(str).to_numpy()
        for fit_idx, val_idx in folds:
            rates = pd.Series(y[fit_idx]).groupby(keys[fit_idx]).mean()
            encoded[val_idx] = (
                pd.Series(keys[val_idx]).map(rates).fillna(y[fit_idx].mean()).to_numpy()
            )
        scores[col] = round(float(roc_auc_score(y, encoded)), 4)
    return scores



def _single_column_auc_multiclass(train: pd.DataFrame, target: str, columns: list[str]) -> dict:
    df = train if len(train) <= LABEL_RULE_SAMPLE_ROWS else train.sample(LABEL_RULE_SAMPLE_ROWS, random_state=42)
    classes, codes = np.unique(df[target].astype(str).to_numpy(), return_inverse=True)
    onehot = np.eye(len(classes))[codes]
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, codes))
    scores = {}
    for col in columns:
        x = df[col]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x) and x.nunique() > 20:
            # Many distinct values: 20 quantile bins. Few (a label stored as codes): the values themselves,
            # since equal-frequency bins would cut straight through imbalanced classes.
            keys = pd.qcut(x.rank(method="first"), q=20, labels=False, duplicates="drop").astype(str).to_numpy()
        else:
            keys = x.astype(str).to_numpy()
        encoded = np.empty_like(onehot)
        for fit_idx, val_idx in folds:
            rates = pd.DataFrame(onehot[fit_idx]).groupby(keys[fit_idx]).mean()
            prior = onehot[fit_idx].mean(axis=0)
            encoded[val_idx] = rates.reindex(keys[val_idx]).fillna(pd.Series(prior)).to_numpy()
        encoded = encoded / encoded.sum(axis=1, keepdims=True)
        scores[col] = round(float(roc_auc_score(codes, encoded, multi_class="ovr", average="macro")), 4)
    return scores

@mcp.tool()
def train_baseline(run_id: str) -> str:
    """Fits a DummyClassifier (stratified — samples predictions from the
    training class distribution) on this run and evaluates it on the held
    -out test fold. The reference point every real model must beat; doesn't
    touch pipeline.pkl, so it never becomes this run's "current" model —
    call train_model separately for that. Result is persisted to
    meta["baseline_comparison"] so generate_report can show the delta
    against the real model without a live conversation."""
    train, test, meta = _load_split(run_id)
    target = meta["target"]
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    result = {
        "run_id": run_id,
        "model": "dummy_baseline",
        **_evaluate(dummy, X_test, y_test, meta),
    }
    meta["baseline_comparison"] = result
    _save_meta(run_id, meta)
    return json.dumps(result)


@mcp.tool()
def evaluate_model(run_id: str) -> str:
    """Re-evaluates this run's CURRENT fitted pipeline (tuned if
    tune_hyperparams ran) on the held-out test fold, on demand — the same
    metrics train_model/tune_hyperparams already returned, without
    retraining. Useful after a step that doesn't retrain but you still want
    the numbers restated (e.g. after apply_drop_columns from
    propose_feature_selection, before deciding whether to re-run
    train_model)."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    pipeline = joblib.load(pipeline_path)
    _, test, meta = _load_split(run_id)
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]
    return json.dumps(
        {
            "run_id": run_id,
            "model": meta["model"],
            **_evaluate(pipeline, X_test, y_test, meta),
        }
    )


@mcp.tool()
def error_analysis(run_id: str, top_n: int = 10) -> str:
    """This run's CURRENT model's most-confident WRONG predictions on the
    held-out test fold — the rows worth looking at by hand, not a bulk
    confusion-matrix count (generate_report already has that). Binary and
    multiclass both supported; confidence is the model's predict_proba for
    the class it (wrongly) picked."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    pipeline = joblib.load(pipeline_path)
    _, test, meta = _load_split(run_id)
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]
    y_pred = pipeline.predict(X_test)
    wrong = y_pred != y_test.values
    if not wrong.any():
        result = {"run_id": run_id, "n_errors": 0, "errors": []}
        meta["error_analysis"] = result
        _save_meta(run_id, meta)
        return json.dumps(result)

    rows = X_test.loc[wrong].copy()
    rows["actual"] = y_test.values[wrong]
    rows["predicted"] = y_pred[wrong]
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)[wrong]
        rows["confidence"] = proba.max(axis=1).round(4)
        rows = rows.sort_values("confidence", ascending=False)
    result = {
        "run_id": run_id,
        "n_errors": int(wrong.sum()),
        "error_rate": round(float(wrong.mean()), 4),
        "top_errors": rows.head(top_n).to_dict(orient="records"),
    }
    meta["error_analysis"] = result
    _save_meta(run_id, meta)
    return json.dumps(result, default=str)


@mcp.tool()
def analyze_segments(run_id: str, segment_column: str) -> str:
    """This run's CURRENT model's precision/recall/F1 broken out by each
    value of segment_column (a categorical column from the original data,
    e.g. region/customer_type/channel) on the held-out test fold — catches
    a model that looks fine in aggregate but fails one segment (the classic
    "works overall, breaks for the minority region" case aggregate metrics
    hide)."""
    from sklearn.metrics import classification_report

    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    pipeline = joblib.load(pipeline_path)
    _, test, meta = _load_split(run_id)
    if segment_column not in test.columns:
        return json.dumps({"error": f"no column '{segment_column}' in the test fold"})
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]
    y_pred = pipeline.predict(X_test)

    segments = {}
    for value, idx in test.groupby(segment_column).groups.items():
        mask = test.index.isin(idx)
        if mask.sum() < 2 or y_test[mask].nunique() < 2:
            segments[str(value)] = {
                "n": int(mask.sum()),
                "note": "too few rows or only one class present",
            }
            continue
        report = classification_report(
            y_test[mask], y_pred[mask], output_dict=True, zero_division=0
        )
        segments[str(value)] = {"n": int(mask.sum()), **report.get("weighted avg", {})}
    result = {"run_id": run_id, "segment_column": segment_column, "segments": segments}
    # `or {}`, not setdefault: runs retrained before _invalidate_model_derived
    # switched to pop() still carry segment_analysis: null on disk.
    meta["segment_analysis"] = {
        **(meta.get("segment_analysis") or {}),
        segment_column: result,
    }
    _save_meta(run_id, meta)
    return json.dumps(result)


@mcp.tool()
def check_model_stability(run_id: str, n_repeats: int = 3) -> str:
    """Re-runs 5-fold CV on this run's training fold, with the same folds
    every other CV here uses (forward-chained on a temporal run, grouped on
    an entity run, stratified otherwise), n_repeats times with different
    shuffles, using the CURRENT model type from meta.json, and reports the
    spread of the run's CV metric (PR-AUC when the target is imbalanced,
    else f1_weighted) across all folds x repeats. A single train_model
    call's CV mean/std is one split's story; a high spread here means that
    number was partly luck of the split, not just the model. On a temporal
    run the forward folds have no randomness, so they run once and the
    spread is across time — often the more useful number: a model whose
    score falls fold after fold is decaying. Above CV_SAMPLE_MAX_ROWS rows,
    runs on the same structure-keeping sample as every CV."""
    train, _, meta = _load_split(run_id)
    target = meta["target"]
    model = meta.get("model")
    if not model:
        return json.dumps(
            {"error": "no trained model for this run yet — call train_model first"}
        )
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_cv, y_cv, cv_n_rows = _cv_sample(X_train, y_train, meta)
    scoring = _cv_scoring(y_train)

    scores = []
    for seed in range(1 if meta.get("time_column") else n_repeats):
        pipeline = _build_pipeline(meta, model, X_cv, y_cv)
        cv, cv_scheme = _validation_folds(X_cv, y_cv, meta, seed=seed)
        scores.extend(
            cross_val_score(pipeline, X_cv, y_cv, cv=cv, scoring=scoring).tolist()
        )
    scores = np.array(scores)
    result = {
        "run_id": run_id,
        "model": model,
        "cv_metric": scoring,
        "cv_scheme": cv_scheme,
        "cv_computed_on_n_rows": cv_n_rows,
        "n_scores": len(scores),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "min": round(float(scores.min()), 4),
        "max": round(float(scores.max()), 4),
        "high_variance_warning": bool(scores.std() > 0.1),
    }
    meta["stability"] = result
    _save_meta(run_id, meta)
    return json.dumps(result)


@mcp.tool()
def check_calibration(run_id: str, n_bins: int = 10) -> str:
    """Measures whether this run's CURRENT model's predicted probabilities
    mean what they say — of the rows it scores 0.8, are ~80% actually
    positive? Reports the Brier score (lower is better; a proper scoring
    rule combining calibration and discrimination) and the expected
    calibration error (ECE — the average gap between predicted probability
    and observed frequency, weighted by bin size), plus the per-bin table
    behind it.

    This is not a nicety. `tune_threshold` picks a cutoff by reading
    precision and recall off the predicted-probability distribution, and
    every cost/alert-budget mode does arithmetic on those probabilities as
    if they were probabilities. Random forests are systematically
    over-confident at the extremes and boosted trees under-confident in the
    middle, so a threshold tuned on uncalibrated scores is solving the
    right problem with the wrong numbers — it still ships a cutoff, just not
    the one the stated objective implies.

    Call it after train_model/tune_hyperparams and BEFORE tune_threshold. If
    ECE comes back above 0.05, call calibrate_model and re-tune: the
    ranking-quality metrics (ROC-AUC, PR-AUC) will barely move, because
    calibration changes what the scores mean, not their order.

    The visualization agent's plot_calibration_curve draws the shape behind
    these two numbers; this one is what the readiness gate records.

    Measured on out-of-fold predictions for the training fold, because this
    number DECIDES things (calibrate or not, which method). The test fold's
    ECE is reported alongside as `test_expected_calibration_error` — read it,
    don't iterate on it."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    pipeline = joblib.load(pipeline_path)
    train, test, meta = _load_split(run_id)
    target = meta["target"]
    X_test, y_test = test.drop(columns=[target]), test[target]
    if y_test.nunique() != 2:
        return json.dumps(
            {
                "skipped": "calibration is measured for binary targets only",
                "reason": "multiclass calibration needs a per-class treatment this tool doesn't implement",
            }
        )
    if not hasattr(pipeline, "predict_proba"):
        return json.dumps(
            {
                "error": "this model has no predict_proba — there are no probabilities to calibrate"
            }
        )

    pos_label = positive_label(meta, y_test)
    test_proba = pipeline.predict_proba(X_test)[:, positive_index(meta, y=y_test)]
    test_ece = _ece((y_test.values == pos_label).astype(int), test_proba, n_bins)[0]
    y_val, proba, validation = _validation_proba(
        pipeline, train.drop(columns=[target]), train[target], meta
    )
    y_true = (y_val.values == pos_label).astype(int)

    brier = float(brier_score_loss(y_true, proba))
    ece, bins = _ece(y_true, proba, n_bins)

    # A model that already went through calibrate_model is reported as such,
    # so "ECE 0.02" can't be mistaken for a raw model that happened to be good.
    calibrated_by = meta.get("calibration_method")
    result = {
        "run_id": run_id,
        "model": meta.get("model"),
        "positive_label": str(pos_label),
        "measured_on": validation,
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(float(ece), 4),
        "test_expected_calibration_error": round(float(test_ece), 4),
        "n_bins_populated": len(bins),
        "bins": bins,
        "calibrated_by": calibrated_by,
        "miscalibration_warning": bool(ece > ECE_CONCERN),
        "interpretation": (
            f"ECE {ece:.4f} exceeds {ECE_CONCERN} — these probabilities are not trustworthy as probabilities. "
            "Call calibrate_model, then re-run tune_threshold: any threshold tuned on the current scores "
            "encodes a precision/recall tradeoff the stated objective didn't ask for."
            if ece > ECE_CONCERN
            else f"ECE {ece:.4f} is within {ECE_CONCERN} — predicted probabilities track observed frequencies closely "
            "enough to tune a threshold against."
        ),
    }
    meta["calibration"] = result
    _save_meta(run_id, meta)
    return json.dumps(result)


def _ece(y_true, proba, n_bins: int) -> tuple[float, list[dict]]:
    """Expected calibration error and its per-bin table.

    Equal-width bins over [0,1]: the question is "when it says 0.8, is it
    0.8", which is a statement about probability VALUES, so the bins are
    over the probability axis, not over quantiles of the predictions."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(proba[mask].mean())
        observed = float(y_true[mask].mean())
        ece += (n / len(y_true)) * abs(mean_pred - observed)
        bins.append(
            {
                "bin": f"[{edges[b]:.2f}, {edges[b + 1]:.2f})",
                "n": n,
                "mean_predicted": round(mean_pred, 4),
                "observed_positive_rate": round(observed, 4),
                "gap": round(observed - mean_pred, 4),
            }
        )

    return float(ece), bins


@mcp.tool()
def assess_fairness(
    run_id: str, protected_attribute: str, min_group_size: int = 20
) -> str:
    """Formal fairness/bias screening for this run's CURRENT model — a
    different question from analyze_segments's plain performance breakdown:
    this computes parity metrics against a stated protected_attribute
    (demographic parity difference, disparate impact ratio, and an
    equal-opportunity/true-positive-rate gap — the standard trio for a
    binary classifier), not just precision/recall per group. Binary targets
    only (these definitions assume one favorable/positive outcome). Groups
    with fewer than min_group_size test rows are marked
    insufficient_sample and excluded from the parity math rather than
    silently folded in — a ratio computed on 3 rows is noise, not evidence.
    This is a screening tool, not a legal/regulatory audit: a clean read
    doesn't certify compliance, and a flagged gap needs a human's domain
    judgment on which definition of "fair" actually applies here —
    demographic parity and equal opportunity can trade off against each
    other. Persisted to meta["fairness"]; rendered in generate_report's
    Fairness section."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    pipeline = joblib.load(pipeline_path)
    _, test, meta = _load_split(run_id)
    if protected_attribute not in test.columns:
        return json.dumps(
            {"error": f"no column '{protected_attribute}' in the test fold"}
        )
    target = meta["target"]
    y_test = test[target]
    if y_test.nunique() != 2:
        return json.dumps(
            {"error": "fairness metrics here are defined for a binary target only"}
        )
    X_test = test.drop(columns=[target])
    y_pred = pipeline.predict(X_test)
    pos_label = positive_label(meta, y_test)

    groups = {}
    for value, idx in test.groupby(protected_attribute).groups.items():
        mask = test.index.isin(idx)
        n = int(mask.sum())
        if n < min_group_size:
            groups[str(value)] = {"n": n, "insufficient_sample": True}
            continue
        pred_positive = y_pred[mask] == pos_label
        actual_positive = y_test[mask].values == pos_label
        groups[str(value)] = {
            "n": n,
            "insufficient_sample": False,
            "selection_rate": round(float(pred_positive.mean()), 4),
            "true_positive_rate": (
                round(float(pred_positive[actual_positive].mean()), 4)
                if actual_positive.sum() > 0
                else None
            ),
        }

    eligible = {k: v for k, v in groups.items() if not v["insufficient_sample"]}
    metrics, flags, note = None, None, None
    if len(eligible) < 2:
        note = "fewer than 2 groups have >= min_group_size rows — cannot compute parity metrics"
    else:
        rates = {k: v["selection_rate"] for k, v in eligible.items()}
        tprs = {
            k: v["true_positive_rate"]
            for k, v in eligible.items()
            if v["true_positive_rate"] is not None
        }
        dpd = round(max(rates.values()) - min(rates.values()), 4)
        dir_ratio = (
            round(min(rates.values()) / max(rates.values()), 4)
            if max(rates.values()) > 0
            else None
        )
        eod = (
            round(max(tprs.values()) - min(tprs.values()), 4)
            if len(tprs) >= 2
            else None
        )
        metrics = {
            "demographic_parity_difference": dpd,
            "disparate_impact_ratio": dir_ratio,
            "equal_opportunity_difference": eod,
        }
        flags = {
            "demographic_parity_concern": dpd > 0.1,
            "disparate_impact_concern": dir_ratio is not None
            and dir_ratio < 0.8,  # standard 80% (4/5ths) rule
            "equal_opportunity_concern": eod is not None and eod > 0.1,
        }

    result = {
        "run_id": run_id,
        "protected_attribute": protected_attribute,
        "positive_label": str(pos_label),
        "groups": groups,
        "metrics": metrics,
        "flags": flags,
        "note": note,
        "limitations": [
            "screening-level metrics only, not a substitute for a domain/legal fairness review",
            "small-sample groups are excluded, not averaged in — a real disparity in a rare group can still be invisible here",
        ],
    }
    meta["fairness"] = result
    _save_meta(run_id, meta)
    return json.dumps(result)


# The ten dimensions modeling/SKILL.md's reflection checklist asks the agent
# to walk before reporting — a fixed list so record_reflection can flag a
# skipped dimension explicitly ("not_assessed") instead of the report
# silently having a shorter checklist than the one specified.
REFLECTION_CHECKS = (
    "business_objective_and_target_validity",
    "data_quality_and_leakage",
    "validation_strategy",
    "feature_engineering",
    "imbalance_handling",
    "model_selection_and_overfitting",
    "evaluation_and_error_analysis",
    "explainability",
    "fairness_assessment_status",
    "unresolved_risks_and_assumptions",
)


# Which deterministic gate backs each self-assessed reflection dimension. A
# dimension the agent marks "clear" while its backing gate failed or never ran
# is a contradiction, and gets recorded as one.
#
# This is the check that would have caught the original reflection failure: a
# run marked `data_quality_and_leakage: clear` citing "both dropped on user
# approval, final detect_data_leakage returned clear=true" — true, but only
# because a HUMAN had spotted the leak in the SHAP output and asked for the
# drop. Reflection ratified a hole someone else patched and reported it as its
# own clean bill of health. Self-assessment cannot audit self-assessment;
# artifacts can.
REFLECTION_GATE_BACKING = {
    "data_quality_and_leakage": ("leakage_screened", "no_identifier_in_model"),
    "model_selection_and_overfitting": ("no_overfitting",),
    "evaluation_and_error_analysis": ("error_analysis",),
    "explainability": ("explainability_run", "explainability_clean"),
    "fairness_assessment_status": ("fairness_assessed",),
    "business_objective_and_target_validity": ("business_context",),
}


def _contradictions(run_id: str, checks_dict: dict) -> list[str]:
    """Claims in this reflection that the run's own artifacts refute."""
    from .gates import compute_readiness

    try:
        gates = compute_readiness(run_id)["checks"]
    except Exception:
        return []  # never let the cross-check break recording the reflection itself
    found = []
    for dimension, gate_names in REFLECTION_GATE_BACKING.items():
        if (checks_dict.get(dimension) or {}).get("status") != "clear":
            continue
        for gate_name in gate_names:
            gate = gates.get(gate_name) or {}
            if gate.get("status") in ("fail", "not_run"):
                found.append(
                    f"reflection marks `{dimension}` as clear, but the `{gate_name}` gate is "
                    f"{gate['status']}: {gate.get('evidence')}"
                )
    return found


@mcp.tool()
def record_reflection(
    run_id: str,
    checks: str,
    critical_issues: str = "[]",
    recommended_actions: str = "[]",
) -> str:
    """Persists the modeling skill's self-critique checklist as structured,
    auditable evidence instead of prose that only lives in the chat
    transcript. Call once, near the end of an autonomous run, after
    explain_model/error_analysis/check_model_stability (and assess_fairness,
    if run) have produced something to reflect on.

    checks: JSON object keyed by check name — one entry per name in
    REFLECTION_CHECKS (business_objective_and_target_validity,
    data_quality_and_leakage, validation_strategy, feature_engineering,
    imbalance_handling, model_selection_and_overfitting,
    evaluation_and_error_analysis, explainability,
    fairness_assessment_status, unresolved_risks_and_assumptions). Any name
    left out is recorded as "not_assessed" rather than silently dropped.
    Each value: {"status": "clear"|"concern"|"not_applicable", "evidence":
    str} — evidence should cite an actual tool result (e.g.
    "detect_data_leakage: clear, no flagged columns"), not a restated
    opinion. Keep evidence to one or two sentences — this stores concise
    conclusions, not the full chain-of-thought that produced them.

    critical_issues: JSON list of strings — concerns serious enough that a
    human must review before this model ships (empty if none).
    recommended_actions: JSON list of strings — concrete next steps, in
    priority order.

    Every "clear" you record is cross-checked against the deterministic gate
    that backs it, and any claim the run's own artifacts refute comes back in
    `contradictions` (and fails the `reflection_recorded` readiness gate).
    Self-assessment cannot audit self-assessment: a previous run marked
    `data_quality_and_leakage` clear on the strength of a leak that a HUMAN
    had spotted in the SHAP output and asked to be dropped — reflection
    ratified someone else's fix and reported it as its own clean result. So
    record what the tools actually returned; claiming clear over a failed or
    never-run gate now makes the report worse, not better.
    """
    meta = _load_meta(run_id)
    try:
        checks_dict = json.loads(checks)
        critical_list = json.loads(critical_issues)
        actions_list = json.loads(recommended_actions)
    except json.JSONDecodeError as e:
        return json.dumps(
            {"error": f"checks/critical_issues/recommended_actions must be JSON: {e}"}
        )

    for name in REFLECTION_CHECKS:
        checks_dict.setdefault(name, {"status": "not_assessed", "evidence": ""})

    if critical_list:
        overall_status = "blocked"
    elif any(c.get("status") == "concern" for c in checks_dict.values()):
        overall_status = "concerns_noted"
    else:
        overall_status = "clear"

    contradictions = _contradictions(run_id, checks_dict)

    result = {
        "checks": checks_dict,
        "critical_issues": critical_list,
        "recommended_actions": actions_list,
        "overall_status": overall_status,
        "contradictions": contradictions,
    }
    meta["reflection"] = result
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, **result})


# How close a trivial rule has to get to the real model before the labels
# themselves become the suspect. Not a hard science: at 0.90 the stump is
# reproducing almost everything the model knows, which for a hand-built rule
# is implausible unless the rule wrote the labels.
LABEL_RULE_SUSPICION_RATIO = 0.90
LABEL_RULE_SAMPLE_ROWS = 200_000


@mcp.tool()
def check_label_rule(run_id: str) -> str:
    """Asks whether a TRIVIAL rule can reproduce this run's labels — one or
    two thresholds on a single column. Read-only. Call it after train_model,
    alongside train_baseline.

    train_baseline establishes the FLOOR (what no skill scores). This
    establishes the CEILING from the other direction: if a 2-split decision
    tree on one column recovers most of the full model's PR-AUC, the most
    likely explanation is not that the model is elegant — it is that the
    labels were generated by a rule on that column, and the model has
    rediscovered it. Everything downstream then measures agreement with a
    rule rather than detection of the real phenomenon: PR-AUC is inflated,
    error analysis describes disagreements with the rule, and the recall
    floor is met against labels that already encode the answer.

    Nothing else in this agent can see this. detect_data_leakage looks for
    near-perfect correlation and identifier shape, but a rule is usually a
    threshold on a RATIO — correlation with the target can sit at 0.6 while a
    stump on the same column reaches 0.97. This is the check that would have
    caught a Wangiri run whose top feature was the distinct-callee ratio that
    operators' own detection rules are built on.

    A high ratio is NOT proof. Some domains genuinely have one dominant
    physical signal. It is a question to put to whoever produced the labels,
    and `business-understanding`'s label_provenance field is where the answer
    belongs. Numeric columns only — a categorical would need encoding, which
    would make the "rule" no longer trivial."""
    train, test, meta = _load_split(run_id)
    target = meta["target"]
    if train[target].nunique() != 2:
        return json.dumps(
            {
                "skipped": "binary targets only — a rule ceiling is not well defined for multiclass"
            }
        )

    numeric = train.drop(columns=[target]).select_dtypes(include="number")
    if numeric.empty:
        return json.dumps(
            {"skipped": "no numeric columns to build a single-column rule from"}
        )

    sample = (
        train
        if len(train) <= LABEL_RULE_SAMPLE_ROWS
        else train.sample(LABEL_RULE_SAMPLE_ROWS, random_state=42)
    )
    pos_label = positive_label(meta, train[target])
    y_train = (sample[target] == pos_label).astype(int)
    y_test = (test[target] == pos_label).astype(int)

    per_column = {}
    for col in numeric.columns:
        x_tr = sample[[col]].fillna(sample[col].median())
        x_te = test[[col]].fillna(sample[col].median())
        for depth in (1, 2):
            stump = DecisionTreeClassifier(
                max_depth=depth, class_weight="balanced", random_state=42
            )
            stump.fit(x_tr, y_train)
            score = float(
                average_precision_score(y_test, stump.predict_proba(x_te)[:, 1])
            )
            best = per_column.get(col)
            if best is None or score > best["pr_auc"]:
                per_column[col] = {"pr_auc": round(score, 4), "depth": depth}

    ranked = sorted(per_column.items(), key=lambda kv: -kv[1]["pr_auc"])
    top_col, top = ranked[0]
    model_metrics = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
    model_pr_auc = (model_metrics.get("auc") or {}).get("pr_auc")
    ratio = round(top["pr_auc"] / model_pr_auc, 4) if model_pr_auc else None

    suspicious = bool(ratio is not None and ratio >= LABEL_RULE_SUSPICION_RATIO)
    result = {
        "best_single_column_rule": {
            "column": top_col,
            "tree_depth": top["depth"],
            "pr_auc": top["pr_auc"],
        },
        "model_pr_auc": model_pr_auc,
        "rule_to_model_ratio": ratio,
        "label_rule_suspicion": suspicious,
        "top_5": [{"column": c, **v} for c, v in ranked[:5]],
        "interpretation": (
            f"a depth-{top['depth']} tree on `{top_col}` alone reaches {ratio:.0%} of the full model's PR-AUC. "
            "Ask whoever produced the labels whether a rule on this column generated them. If it did, this run "
            "measures agreement with that rule, not detection of the underlying behaviour, and the column must "
            "be excluded before the score means anything."
            if suspicious
            else (
                f"the best single-column rule reaches {ratio:.0%} of the model's PR-AUC — the model is doing real "
                "work beyond any one threshold, which is what you want to see."
                if ratio is not None
                else "no model PR-AUC on file yet — train a model first for the comparison to mean anything"
            )
        ),
    }
    meta["label_rule_check"] = result
    _save_meta(run_id, meta)
    return json.dumps(result)


@mcp.tool()
def declare_feature_engineering_not_applicable(run_id: str, reason: str) -> str:
    """Record WHY no business features were engineered for this run, when
    that is a genuine conclusion rather than a skipped step.

    Feature engineering is the step most easily skipped without anyone
    noticing, because a run with no engineered features looks exactly like a
    run where none were warranted. This makes the difference visible: the
    readiness gate accepts applied features OR a recorded reason, and nothing
    else. Legitimate reasons exist — the file is already pre-aggregated with
    the domain ratios computed, every archetype needs a column this schema
    lacks, the row grain forbids the aggregation. "Didn't get to it" is not
    one; in that case go and do the step."""
    if not reason.strip():
        return json.dumps(
            {
                "error": "a reason is required — an empty justification is the same as skipping"
            }
        )
    meta = _load_meta(run_id)
    meta["feature_engineering_not_applicable"] = reason
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "feature_engineering_not_applicable": reason})


@mcp.tool()
def backtest_on_new_data(run_id: str, path: str) -> str:
    """Score this run's CURRENT model against a LATER labelled file and
    report the metrics. Read-only for the run's own artifacts.

    The test fold answers "does this work on data I held out". It cannot
    answer "does this still work", and in an adversarial domain those are
    different questions with different answers — the pattern changes once it
    is detected. check_model_stability cannot see this either: it resamples
    randomly, so it mixes periods together and reports stability for a model
    that has already decayed.

    path: a CSV shaped like the training data, WITH the target column, drawn
    from a period after the training window. Metrics are computed at the
    run's operating point when one exists and at 0.5 otherwise, so the number
    is the one the deployed model would actually produce."""
    from pathlib import Path

    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    if not Path(path).exists():
        return json.dumps({"error": f"no data file at '{path}'"})
    meta = _load_meta(run_id)
    target = meta["target"]
    df = read_table(path)
    if target not in df.columns:
        return json.dumps(
            {
                "error": f"'{path}' has no '{target}' column — a backtest needs labels to score against"
            }
        )

    pipeline = joblib.load(pipeline_path)
    X, y = df.drop(columns=[target]), df[target]
    if y.nunique() < 2:
        return json.dumps(
            {"error": f"'{path}' has only one class — nothing to measure"}
        )

    op = meta.get("operating_point")
    threshold = float(op["threshold"]) if op else 0.5
    proba = pipeline.predict_proba(X)[:, positive_index(meta, y=y)]
    y_bin = (y == positive_label(meta, y)).astype(int)
    y_pred = (proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_bin == 1)).sum())
    fp = int(((y_pred == 1) & (y_bin == 0)).sum())
    fn = int(((y_pred == 0) & (y_bin == 1)).sum())

    new_pr_auc = round(float(average_precision_score(y_bin, proba)), 4)
    held_out = (
        (meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}).get("auc")
        or {}
    ).get("pr_auc")
    decay = round(new_pr_auc - held_out, 4) if held_out else None

    result = {
        "source": path,
        "n_rows": len(df),
        "positive_rate": round(float(y_bin.mean()), 6),
        "threshold_applied": threshold,
        "pr_auc": new_pr_auc,
        "held_out_pr_auc": held_out,
        "pr_auc_change": decay,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "alert_rate": round(float(y_pred.mean()), 6),
        # A shifted positive rate is its own finding: the model may be intact
        # while the world changed underneath it, which calls for different
        # action than a model that simply stopped working.
        "note": "compare positive_rate against the training fold's — a large move means the population shifted, "
        "which is a different problem from the model degrading",
    }
    meta["backtest"] = result
    _save_meta(run_id, meta)
    return json.dumps(result)
