"""drift agent — `drift` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import (
    _column_drift,
    _json_default,
    _overall_verdict,
    _read_csv_or_error,
    _read_profile,
    _structural_columns,
)  # noqa: F401


@mcp.tool()
def detect_drift(
    reference_path: str = "",
    current_path: str = "",
    columns: str = "",
    profile_path: str = "",
    key_columns: str = "",
    target_column: str = "",
) -> str:
    """Compares a reference (training-time) dataset against a current
    (production-time) one, column by column. Numeric columns get a
    quantile-binned PSI + KS-test; categorical columns get a
    category-proportion PSI + chi-square. Severity is driven by PSI (stable
    across sample sizes), not the p-value. Reports only — it never retrains.

    **profile_path** is the preferred way to call this: point it at the
    `*.monitoring.json` a training agent wrote beside its exported .pkl and
    reference_path, key_columns and target_column are all filled in from it —
    and current_path too, when omitted: the serving agent's predictions log
    for that model, i.e. everything it has scored in production.
    That matters for two reasons. The run directory holding the original
    train.csv is swept after a retention window, so the profile's own copy is
    the only baseline that still exists for a model more than a few days old.
    And the profile names the columns the model's explainer said it actually
    uses, which is what the verdict should hinge on.

    key_columns: comma-separated columns the model depends on. The overall
    verdict is decided by THESE; drift in other columns is reported as
    `drifted_incidental_columns` rather than driving a retrain call. Without
    them every column votes equally, and on a wide table one drifting
    unmodelled column produces a permanent "retrain now".

    target_column: scored and reported SEPARATELY, as label/prior shift.
    A moved label distribution invalidates a tuned decision threshold even
    when every feature is stable — a different finding with a different fix,
    so it does not get averaged into the feature verdict.

    columns: optional comma-separated subset to compare at all; empty checks
    every shared column.

    Scope: this compares two tables of inputs. It does not run the model, so
    it cannot see prediction drift, concept drift, or current accuracy — the
    result says so explicitly every time, including when it finds nothing."""
    profile = None
    if profile_path:
        profile, err = _read_profile(profile_path)
        if err:
            return json.dumps(err)
        reference_path = reference_path or profile["reference_path"]
        # No current file given: what the model has actually scored so far.
        current_path = current_path or profile.get("predictions_log") or ""
        key_columns = key_columns or ",".join(profile.get("key_columns") or [])
        target_column = target_column or (profile.get("target") or "")
    if not reference_path or not current_path:
        return json.dumps(
            {"error": "pass current_path plus either reference_path or profile_path"}
        )

    ref_df, err = _read_csv_or_error(reference_path, "reference")
    if err:
        return json.dumps(err)
    cur_df, err = _read_csv_or_error(current_path, "current")
    if err:
        return json.dumps(err)

    if columns.strip():
        requested = [c.strip() for c in columns.split(",") if c.strip()]
        missing = [
            c for c in requested if c not in ref_df.columns or c not in cur_df.columns
        ]
        if missing:
            return json.dumps(
                {"error": f"column(s) not found in both files: {missing}"}
            )
        shared = requested
    else:
        shared = [c for c in ref_df.columns if c in cur_df.columns]
        if not shared:
            return json.dumps(
                {"error": "reference_path and current_path share no column names"}
            )
    key_list = [c.strip() for c in key_columns.split(",") if c.strip()]
    # Timestamps and ids always "drift"; asked for by name, they are compared anyway.
    structural = {c: why for c, why in _structural_columns(ref_df[shared]).items()
                  if c not in key_list and c != target_column and not columns.strip()}
    shared = [c for c in shared if c not in structural]

    results = {col: _column_drift(ref_df[col], cur_df[col]) for col in shared}
    target = target_column.strip() or None
    verdict = _overall_verdict(results, key_list, target)
    drifted_columns = [c for c, r in results.items() if r["drifted"]]

    # A key column the model needs that ISN'T in the current file is not a
    # clean comparison — it is a broken one, and silently comparing the rest
    # would report "no drift" about a dataset the model cannot even score.
    missing_key = [c for c in key_list if c not in cur_df.columns]

    return json.dumps(
        {
            "reference_path": reference_path,
            "reference_shape": ref_df.shape,
            "current_shape": cur_df.shape,
            "profile_used": profile_path or None,
            "model_path": (profile or {}).get("model_path"),
            "columns_compared": shared,
            "columns_only_in_reference": [
                c for c in ref_df.columns if c not in cur_df.columns
            ],
            "columns_only_in_current": [
                c for c in cur_df.columns if c not in ref_df.columns
            ],
            "missing_key_columns": missing_key,
            **(
                {
                    "blocking_note": f"the model depends on {missing_key}, absent from the current file — it could not "
                    "score this data at all; fix the feed before reading anything below as a drift verdict"
                }
                if missing_key
                else {}
            ),
            "columns": results,
            "drifted_columns": drifted_columns,
            "excluded_structural_columns": structural,
            "missingness_changes": {c: f"{r['reference_missing_pct']}% -> {r['current_missing_pct']}%"
                                    for c, r in results.items()
                                    if any(x.startswith("missing") for x in r["drift_reasons"])},
            "type_changes": {c: r["type_change"] for c, r in results.items() if r.get("type_change")},
            **({"small_sample_warning": f"only {len(cur_df)} current rows — PSI is noisy at this size; drift is "
                "flagged only above each column's sampling-noise floor (psi_noise_floor)"}
               if len(cur_df) < SMALL_SAMPLE_ROWS else {}),
            **verdict,
        },
        default=_json_default,
    )


@mcp.tool()
def compare_distributions(reference_path: str, current_path: str, column: str) -> str:
    """Single-column deep dive — use after detect_drift flags a column, to
    see the actual summary stats/proportions behind its PSI. Read-only."""
    ref_df, err = _read_csv_or_error(reference_path, "reference")
    if err:
        return json.dumps(err)
    cur_df, err = _read_csv_or_error(current_path, "current")
    if err:
        return json.dumps(err)
    if column not in ref_df.columns or column not in cur_df.columns:
        return json.dumps({"error": f"column '{column}' not found in both files"})

    drift = _column_drift(ref_df[column], cur_df[column])
    if drift["dtype"] == "numeric":
        ref_summary = ref_df[column].describe().round(4).to_dict()
        cur_summary = cur_df[column].describe().round(4).to_dict()
    else:
        ref_summary = (
            ref_df[column]
            .astype(str)
            .value_counts(normalize=True)
            .round(4)
            .head(10)
            .to_dict()
        )
        cur_summary = (
            cur_df[column]
            .astype(str)
            .value_counts(normalize=True)
            .round(4)
            .head(10)
            .to_dict()
        )

    return json.dumps(
        {
            "column": column,
            "reference_summary": ref_summary,
            "current_summary": cur_summary,
            **drift,
        },
        default=_json_default,
    )


def _resolve(reference_path, current_path, profile_path, columns):
    """(ref_df, cur_df, columns, key columns, target, error) — the shared
    profile / path handling of the tools below."""
    profile, key, target = None, [], None
    if profile_path:
        profile, err = _read_profile(profile_path)
        if err:
            return None, None, None, None, None, err
        reference_path = reference_path or profile["reference_path"]
        current_path = current_path or profile.get("predictions_log") or ""
        key, target = list(profile.get("key_columns") or []), profile.get("target")
    if not reference_path or not current_path:
        return None, None, None, None, None, {"error": "pass current_path plus either reference_path or profile_path"}
    ref_df, err = _read_csv_or_error(reference_path, "reference")
    if err:
        return None, None, None, None, None, err
    cur_df, err = _read_csv_or_error(current_path, "current")
    if err:
        return None, None, None, None, None, err
    requested = [c.strip() for c in columns.split(",") if c.strip()]
    missing = [c for c in requested if c not in ref_df.columns or c not in cur_df.columns]
    if missing:
        return None, None, None, None, None, {"error": f"column(s) not found in both files: {missing}"}
    return ref_df, cur_df, requested, key, target, None


@mcp.tool()
def detect_multivariate_drift(reference_path: str = "", current_path: str = "", profile_path: str = "",
                              columns: str = "") -> str:
    """Has the data changed AS A WHOLE? A classifier is trained to tell
    reference rows from current rows; its cross-validated ROC-AUC is the
    verdict — 0.5: indistinguishable, >= 0.6 moderate, >= 0.75 severe — and
    the columns that give the difference away are named. Catches what
    per-column PSI cannot: two columns each stable on their own whose
    relationship changed (usage now high for the plan that used to be low),
    and many small shifts that add up. Timestamps, identifiers and the
    target are left out (they always differ, or they are label shift).
    columns: comma-separated subset; default the model's key columns from
    profile_path, else every shared column. Read-only."""
    ref_df, cur_df, requested, key, target, err = _resolve(reference_path, current_path, profile_path, columns)
    if err:
        return json.dumps(err)
    shared = requested or [c for c in (key or ref_df.columns) if c in cur_df.columns and c != target]
    structural = {} if requested else _structural_columns(ref_df[shared])
    use = [c for c in shared if c not in structural]
    if not use:
        return json.dumps({"error": "no comparable columns left after excluding timestamps and identifiers"})
    if min(len(ref_df), len(cur_df)) < 50:
        return json.dumps({"error": "need at least 50 rows on each side to train the classifier"})
    result = domain_classifier(ref_df, cur_df, use)
    result.update({
        "columns_used": use,
        "excluded_structural_columns": structural,
        "interpretation": {
            "none": "a model cannot tell the current rows from the reference rows — no joint shift either",
            "moderate": "the tables are distinguishable: a real shift, look at top_columns (compare_distributions)",
            "severe": "the current data is clearly different from the reference — treat the model's predictions "
                      "as unvalidated until it is re-evaluated on recent labelled data",
        }[result["severity"]],
        "scope_caveat": "a DATA drift check — it does not run the model or measure its accuracy",
    })
    return json.dumps(result, default=_json_default)


@mcp.tool()
def drift_over_time(reference_path: str = "", current_path: str = "", time_column: str = "",
                    freq: str = "W", profile_path: str = "", columns: str = "") -> str:
    """WHEN did it start? Splits the current data into periods of `freq`
    ("D", "W", "MS") on time_column and scores each period against the
    reference, per column (PSI above the noise floor, missing share). A
    step that appears in one period and stays is a change (a release, a
    tariff, a broken feed on that date); a slow climb is gradual drift; a
    one-period spike is an event. Periods under 100 rows are skipped.
    columns: default the model's key columns (profile) or every shared
    non-structural column (the 12 that drift most over the whole window)."""
    ref_df, cur_df, requested, key, target, err = _resolve(reference_path, current_path, profile_path, columns)
    if err:
        return json.dumps(err)
    if time_column not in cur_df.columns:
        return json.dumps({"error": f"time_column '{time_column}' not in the current data"})
    stamps = pd.to_datetime(cur_df[time_column], errors="coerce", format="mixed")
    if stamps.isna().all():
        return json.dumps({"error": f"time_column '{time_column}' does not parse as dates"})
    shared = [c for c in ref_df.columns if c in cur_df.columns and c != time_column]
    structural = _structural_columns(ref_df[shared])
    candidates = requested or [c for c in (key or shared) if c in shared and c not in structural]
    if not requested and not key and len(candidates) > 12:
        whole = {c: _column_drift(ref_df[c], cur_df[c]) for c in candidates}
        candidates = sorted(candidates, key=lambda c: -(whole[c]["psi"] or 0))[:12]
    periods = stamps.dt.to_period(freq)
    timeline, first = [], None
    for period, rows in cur_df.groupby(periods, sort=True):
        if len(rows) < 100:
            continue
        per_col = {c: _column_drift(ref_df[c], rows[c]) for c in candidates}
        drifted = sorted(c for c, r in per_col.items() if r["drifted"])
        worst = max((SEVERITY_ORDER[r["severity"]] for r in per_col.values()), default=0)
        entry = {"period": str(period), "rows": int(len(rows)),
                 "severity": {v: k for k, v in SEVERITY_ORDER.items()}[worst], "drifted_columns": drifted,
                 "psi": {c: r["psi"] for c, r in per_col.items()},
                 "missing_pct": {c: r["current_missing_pct"] for c, r in per_col.items() if r["current_missing_pct"]}}
        timeline.append(entry)
        if drifted and first is None:
            first = {"period": str(period), "columns": drifted}
    return json.dumps({
        "columns": candidates, "freq": freq, "periods": len(timeline), "timeline": timeline,
        "first_drift": first, "excluded_structural_columns": structural,
        "latest_severity": timeline[-1]["severity"] if timeline else None,
        "reading": "a step that persists = a change on that date; a steady climb = gradual drift; a single "
                   "spike = an event. Compare like with like: a December reference against a January current "
                   "shows seasonality, not decay.",
    }, default=_json_default)
