"""clustering agent — `data_cleaning` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from core.data_quality import identifier_columns  # noqa: E402
from .core import _load_split, _run_dir, _save_meta, _to_native  # noqa: F401


@mcp.tool()
def propose_imputation(run_id: str) -> str:
    """Suggest a per-column imputation strategy (median/mode/none), computed
    from the fit fold only — does not modify anything."""
    fit, _, _ = _load_split(run_id)
    proposals = {}
    for col in fit.columns:
        missing_pct = round(fit[col].isna().mean() * 100, 2)
        if missing_pct == 0:
            continue
        if missing_pct > 50:
            strategy = "drop_column"
        elif pd.api.types.is_numeric_dtype(fit[col]):
            strategy = "median"
        else:
            strategy = "mode"
        proposals[col] = {"missing_pct": missing_pct, "suggested_strategy": strategy}
    return json.dumps({"proposals": proposals})


@mcp.tool()
def apply_imputation(run_id: str, strategies: str) -> str:
    """Apply user-approved per-column strategies. strategies: JSON string
    {col: "median"|"mode"|"drop_column"|<literal value>}."""
    fit, holdout, meta = _load_split(run_id)
    plan = json.loads(strategies)
    dropped, values = [], {}
    for col, strategy in plan.items():
        if strategy == "drop_column":
            dropped.append(col)
            continue
        if strategy == "median":
            value = float(fit[col].median())
        elif strategy == "mode":
            mode = fit[col].mode(dropna=True)
            value = _to_native(mode.iloc[0]) if not mode.empty else None
        else:
            value = strategy
        values[col] = value
        fit[col] = fit[col].fillna(value)
        holdout[col] = holdout[col].fillna(value)

    if dropped:
        fit = fit.drop(columns=dropped)
        holdout = holdout.drop(columns=[c for c in dropped if c in holdout.columns])
        meta["dropped_columns"] = sorted(set(meta["dropped_columns"]) | set(dropped))

    meta["imputation"] = {**meta["imputation"], **values}
    fit.to_csv(_run_dir(run_id) / "train.csv", index=False)
    holdout.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "dropped_columns": dropped,
            "imputed_columns": list(values),
            "fit_shape": fit.shape,
        }
    )


@mcp.tool()
def propose_drop_columns(run_id: str) -> str:
    """Suggest columns to drop — constant, or identifier-shaped by the shared
    rules (core/data_quality: near-unique, repeating high-cardinality,
    identifier-ratio, name pattern) — computed from the fit fold; does not
    modify anything. Distance on an identifier is meaningless: it splits
    every row into its own cluster."""
    fit, _, meta = _load_split(run_id)
    keep = set(meta.get("profile_columns") or [])
    proposals = {c: "constant column, no signal" for c in fit.columns
                 if c not in keep and fit[c].nunique(dropna=True) <= 1}
    for col, signal in identifier_columns(fit, None).items():
        if col not in keep:
            proposals.setdefault(col, f"{signal['reason']} ({signal['strength']})")
    return json.dumps({"proposals": proposals})


@mcp.tool()
def apply_drop_columns(run_id: str, columns: str) -> str:
    """Drop approved columns from both fit and holdout. columns: a JSON list
    or comma-separated names."""
    fit, holdout, meta = _load_split(run_id)
    try:
        parsed = json.loads(columns)
    except (json.JSONDecodeError, TypeError):
        parsed = [c.strip() for c in str(columns).split(",") if c.strip()]
    parsed = [parsed] if isinstance(parsed, str) else list(parsed)
    cols = [c for c in parsed if c in fit.columns]
    fit = fit.drop(columns=cols)
    holdout = holdout.drop(columns=[c for c in cols if c in holdout.columns])
    meta["dropped_columns"] = sorted(set(meta["dropped_columns"]) | set(cols))
    fit.to_csv(_run_dir(run_id) / "train.csv", index=False)
    holdout.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "dropped_columns": cols, "fit_shape": fit.shape})


@mcp.tool()
def set_profile_columns(run_id: str, columns: str) -> str:
    """Columns that DESCRIBE the segments but must not DEFINE them — kept in
    the data, excluded from the distance, and reported per segment by
    explain_model (with eta^2, the share of their variance the segmentation
    explains):
      outcomes     churn flag, revenue, ARPU, NPS — clustering on the outcome
                   finds the outcome back and calls it a segment; profiling
                   by it shows whether behaviour-based segments matter
      protected    age, gender, ethnicity, a postcode that proxies them —
                   segments that drive offers or treatment should not be
                   built on them; profiling shows whether they correlate anyway
    columns: comma-separated or a JSON list; replaces the previous set.
    Retrain afterwards."""
    fit, _, meta = _load_split(run_id)
    try:
        parsed = json.loads(columns)
    except (json.JSONDecodeError, TypeError):
        parsed = [c.strip() for c in str(columns).split(",") if c.strip()]
    parsed = [parsed] if isinstance(parsed, str) else list(parsed)
    unknown = [c for c in parsed if c not in fit.columns]
    if unknown:
        return json.dumps({"error": f"not in the fit fold: {unknown}"})
    meta["profile_columns"] = parsed
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "profile_columns": parsed,
                       "next": "retrain — the current model (if any) was fit with these in the distance"})
