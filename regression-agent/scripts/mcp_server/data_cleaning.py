"""regression agent — `data_cleaning` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _invalidate_model_derived, _load_split, _run_dir, _save_meta, _to_native  # noqa: F401


@mcp.tool()
def propose_imputation(run_id: str) -> str:
    """Suggest a per-column imputation strategy (median/mode/none), computed
    from the full dataset (train+test) so missingness isn't missed when NaNs
    happen to land only in one fold — does not modify anything. Present to
    the user before apply_imputation."""
    train, test, meta = _load_split(run_id)
    combined = pd.concat([train, test], axis=0)
    proposals = {}
    for col in train.columns:
        if col == meta["target"]:
            continue
        missing_pct = round(combined[col].isna().mean() * 100, 2)
        if missing_pct == 0:
            continue
        if missing_pct > 50:
            strategy = "drop_column"
        elif pd.api.types.is_numeric_dtype(combined[col]):
            strategy = "median"
        else:
            strategy = "mode"
        proposals[col] = {"missing_pct": missing_pct, "suggested_strategy": strategy}
    return json.dumps({"proposals": proposals})


@mcp.tool()
def apply_imputation(run_id: str, strategies: str) -> str:
    """Apply user-approved per-column strategies. strategies: JSON string
    {col: "median"|"mode"|"drop_column"|<literal value>}. Fill values are
    computed from the training fold only, then applied to both train and
    test — the test fold never influences what value gets filled in."""
    train, test, meta = _load_split(run_id)
    plan = json.loads(strategies)
    dropped, values = [], {}
    for col, strategy in plan.items():
        if strategy == "drop_column":
            dropped.append(col)
            continue
        if strategy == "median":
            value = float(train[col].median())
        elif strategy == "mode":
            value = _to_native(train[col].mode().iloc[0])
        else:
            value = strategy
        values[col] = value
        train[col] = train[col].fillna(value)
        test[col] = test[col].fillna(value)

    if dropped:
        train = train.drop(columns=dropped)
        test = test.drop(columns=[c for c in dropped if c in test.columns])
        meta["dropped_columns"] = sorted(set(meta["dropped_columns"]) | set(dropped))

    meta["imputation"] = {**meta["imputation"], **values}
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "dropped_columns": dropped,
            "imputed_columns": list(values),
            "train_shape": train.shape,
        }
    )


@mcp.tool()
def propose_drop_columns(run_id: str) -> str:
    """Suggest columns to drop — constant, or identifier-shaped by the shared
    rules (core/data_quality: near-unique, high-cardinality with repeats,
    identifier-ratio, name pattern) — computed from the training fold; does
    not modify anything. The run's split keys (group/time column) are never
    proposed: validation needs them and the model pipeline excludes them."""
    train, _, meta = _load_split(run_id)
    keys = split_keys(meta)
    proposals = {}
    for col in train.columns:
        if col == meta["target"] or col in keys:
            continue
        if train[col].nunique(dropna=True) <= 1:
            proposals[col] = "constant column, no signal"
    for col, signal in identifier_columns(train, meta["target"]).items():
        if col not in keys:
            proposals.setdefault(col, f"{signal['reason']} ({signal['strength']})")
    return json.dumps({"proposals": proposals, "split_keys_kept": keys})


@mcp.tool()
def apply_drop_columns(run_id: str, columns: str) -> str:
    """Drop approved columns from both train and test. columns: a JSON list
    or comma-separated names. Split keys are kept (kept_split_keys)."""
    train, test, meta = _load_split(run_id)
    try:
        parsed = json.loads(columns)
    except (json.JSONDecodeError, TypeError):
        parsed = [c.strip() for c in str(columns).split(",") if c.strip()]
    parsed = [parsed] if isinstance(parsed, str) else list(parsed)
    keys = split_keys(meta)
    kept_keys = [c for c in parsed if c in keys]
    cols = [c for c in parsed if c in train.columns and c not in keys]
    train = train.drop(columns=cols)
    test = test.drop(columns=[c for c in cols if c in test.columns])
    meta["dropped_columns"] = sorted(set(meta["dropped_columns"]) | set(cols))
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "dropped_columns": cols, "train_shape": train.shape,
                       **({"kept_split_keys": kept_keys} if kept_keys else {})})


@mcp.tool()
def propose_datetime_features(run_id: str) -> str:
    """Detects training-fold columns that are datetimes (by dtype, or by
    >=95% of non-null values date-parsing) and proposes decomposing each
    into month/day/dayofweek/hour (and year, except for the run's own time
    column) numeric columns. Read-only."""
    train, _, meta = _load_split(run_id)
    skip = {meta["target"], *(meta.get("dropped_columns") or [])}
    candidates = []
    for col in train.columns:
        if col in skip:
            continue
        if pd.api.types.is_datetime64_any_dtype(train[col]):
            candidates.append(col)
        elif train[col].dtype == object and shared_validation._parse_dates(train[col]).notna().mean() >= 0.95:
            candidates.append(col)
    return json.dumps({"run_id": run_id, "datetime_columns": candidates})


@mcp.tool()
def apply_datetime_features(run_id: str, columns: str) -> str:
    """Adds <col>_month/_day/_dayofweek/_hour (and _year) numeric columns for
    each approved column, on both train and test, and drops the raw column —
    except the run's own time_column, which is KEPT (it orders temporal
    validation; the pipeline excludes it) and gets no _year: on a forward
    split the test period's year is one the model never saw."""
    train, test, meta = _load_split(run_id)
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    time_key = meta.get("time_column")
    for df in (train, test):
        for col in cols:
            if col not in df.columns:
                continue
            parsed = shared_validation._parse_dates(df[col])
            if col != time_key:
                df[f"{col}_year"] = parsed.dt.year
            df[f"{col}_month"] = parsed.dt.month
            df[f"{col}_day"] = parsed.dt.day
            df[f"{col}_dayofweek"] = parsed.dt.dayofweek
            df[f"{col}_hour"] = parsed.dt.hour
            if col != time_key:
                df.drop(columns=[col], inplace=True)
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    meta["datetime_features"] = sorted(set(meta.get("datetime_features") or []) | set(cols))
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "decomposed_columns": cols, "train_shape": train.shape})


@mcp.tool()
def apply_target_transform(run_id: str, transform: str = "log1p") -> str:
    """Model the target in a transformed space — "log1p" or "none". For a
    non-negative, right-skewed target (revenue, ARPU, spend, claim amount,
    data usage) a squared-error model on the raw scale spends its capacity
    on the few huge values and leaves every typical row badly fitted;
    log1p makes errors roughly proportional. The model fits log1p(y), and
    predictions and every metric come back in the original units (expm1).
    Decide from prepare_dataset's target_skew (> 1) and target_min (>= 0);
    record it as an assumption. Changes the model: retrain after it."""
    train, _, meta = _load_split(run_id)
    y = train[meta["target"]]
    if transform not in ("log1p", "none"):
        return json.dumps({"error": "transform must be 'log1p' or 'none'"})
    if transform == "log1p" and meta.get("objective") == "poisson":
        return json.dumps({"error": "a poisson objective already models the log of the mean — "
                                    "log1p on top would model it twice"})
    if transform == "log1p" and float(y.min()) < 0:
        return json.dumps({"error": f"log1p needs a non-negative target; the minimum is {float(y.min())}"})
    meta["target_transform"] = None if transform == "none" else transform
    stale = _invalidate_model_derived(meta)
    _save_meta(run_id, meta)
    return json.dumps({
        "run_id": run_id,
        "target_transform": meta["target_transform"],
        "skew_raw": round(float(y.skew()), 3),
        "skew_transformed": round(float(np.log1p(y).skew()), 3) if transform == "log1p" else None,
        "next": "retrain (train_model / compare_models) — metrics stay in the original units",
        **({"invalidated": stale} if stale else {}),
    })
