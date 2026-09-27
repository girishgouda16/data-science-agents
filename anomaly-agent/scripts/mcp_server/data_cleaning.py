"""anomaly agent — `data_cleaning` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import split_keys  # noqa: F401
from .core import (
    _feature_frame,
    _json_default,
    _load_split,
    _looks_id_like,
    _run_dir,
    _save_meta,
    _to_native,
)  # noqa: F401


@mcp.tool()
def propose_imputation(run_id: str) -> str:
    """Suggest per-column imputation strategies from the training fold only.
    Read-only — pairs with apply_imputation."""
    train, _, meta = _load_split(run_id)
    X_train = _feature_frame(train, meta)
    proposals = {}
    for col in X_train.columns:
        missing_pct = round(float(X_train[col].isna().mean() * 100), 2)
        if missing_pct == 0:
            continue
        if missing_pct > 50:
            strategy = "drop_column"
        elif pd.api.types.is_numeric_dtype(X_train[col]):
            strategy = "median"
        else:
            strategy = "mode"
        proposals[col] = {"missing_pct": missing_pct, "suggested_strategy": strategy}
    return json.dumps({"run_id": run_id, "proposals": proposals})


@mcp.tool()
def apply_imputation(run_id: str, strategies: str) -> str:
    """Apply user-approved imputation/drop decisions. strategies is a JSON
    object: {column: "median"|"mode"|"drop_column"|literal_value}."""
    train, test, meta = _load_split(run_id)
    label_column = meta.get("label_column")
    plan = json.loads(strategies)
    dropped, values = [], {}
    for col, strategy in plan.items():
        if col not in train.columns or col == label_column:
            continue
        if strategy == "drop_column":
            dropped.append(col)
            continue
        if strategy == "median":
            value = float(train[col].median())
        elif strategy == "mode":
            value = _to_native(train[col].mode(dropna=True).iloc[0])
        else:
            value = strategy
        values[col] = value
        train[col] = train[col].fillna(value)
        test[col] = test[col].fillna(value)

    if dropped:
        train = train.drop(columns=dropped)
        test = test.drop(columns=[c for c in dropped if c in test.columns])
        meta["dropped_columns"] = sorted(
            set(meta.get("dropped_columns") or []) | set(dropped)
        )

    meta["imputation"] = {**meta.get("imputation", {}), **values}
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "dropped_columns": dropped,
            "imputed_columns": list(values),
            "train_shape": train.shape,
        },
        default=_json_default,
    )


@mcp.tool()
def propose_drop_columns(run_id: str) -> str:
    """Suggest columns to drop (constant, near-unique / ID-like) from the
    training fold only. Read-only — pairs with apply_drop_columns."""
    train, _, meta = _load_split(run_id)
    X_train = _feature_frame(train, meta).drop(columns=split_keys(meta))  # split keys stay: features use them
    proposals = {}
    n = len(X_train)
    for col in X_train.columns:
        series = X_train[col]
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            proposals[col] = "constant column, no signal"
        elif _looks_id_like(series, n):
            proposals[col] = "near-unique, likely an ID column"
    return json.dumps({"run_id": run_id, "proposals": proposals})


@mcp.tool()
def apply_drop_columns(run_id: str, columns: str) -> str:
    """Drop user-approved columns from both train and test. columns is a
    JSON list of column names."""
    train, test, meta = _load_split(run_id)
    label_column = meta.get("label_column")
    try:
        parsed = json.loads(columns)
    except (json.JSONDecodeError, TypeError):
        parsed = [c.strip() for c in str(columns).split(",") if c.strip()]
    parsed = [parsed] if isinstance(parsed, str) else list(parsed)
    keep = {label_column, *split_keys(meta)}
    kept_keys = [c for c in parsed if c in split_keys(meta)]
    cols = [c for c in parsed if c in train.columns and c not in keep]
    train = train.drop(columns=cols)
    test = test.drop(columns=[c for c in cols if c in test.columns])
    meta["dropped_columns"] = sorted(set(meta.get("dropped_columns") or []) | set(cols))
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps(
        {"run_id": run_id, "dropped_columns": cols, "train_shape": train.shape,
         **({"kept_split_keys": kept_keys, "note": "split keys stay in the data (features are built from them) "
             "and never reach the detector"} if kept_keys else {})},
        default=_json_default,
    )
