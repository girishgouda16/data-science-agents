"""forecasting agent — `data_cleaning` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _load_split, _run_dir, _save_meta  # noqa: F401


@mcp.tool()
def propose_imputation(run_id: str) -> str:
    """Suggests forward-fill ("ffill") for any column with missing values,
    computed from the training fold — never median/mode. A time series has
    an order; filling a gap with the dataset's overall median ignores that
    the value right before the gap is almost always the best guess, and
    (worse) a global median computed before the chronological split would
    see into the test period. ffill for the target itself is a legitimate,
    common practice for short gaps — surfaced as a warning, not a block, so
    the human decides."""
    train, _, meta = _load_split(run_id)
    proposals = {}
    for col in train.columns:
        if col == meta["date_column"]:
            continue
        missing_pct = round(train[col].isna().mean() * 100, 2)
        if missing_pct == 0:
            continue
        if col != meta["target"] and missing_pct > 50:
            proposals[col] = {
                "missing_pct": missing_pct,
                "suggested_strategy": "drop_column",
            }
        elif col == meta["target"]:
            y = train[col].dropna()
            counts = bool((y >= 0).all() and (y == y.round()).all() and (y == 0).any())
            proposals[col] = {
                "missing_pct": missing_pct,
                "suggested_strategy": "0" if counts else "interpolate",
                "why": ("a count series that already has zero periods: a missing period most often means nothing "
                        "happened — confirm the feed was up" if counts else
                        "a level series: interpolate between the neighbours; ffill for a value that holds until it "
                        "changes (a price, a tariff)"),
                "warning": "This is the target — any fill fabricates values. Ask before filling more than a few "
                           "periods, and never fill a long outage: cut the series after it instead.",
            }
        else:
            proposals[col] = {"missing_pct": missing_pct, "suggested_strategy": "ffill"}
    return json.dumps({"proposals": proposals})


def _apply_panel_imputation(run_id, train, test, meta, plan) -> str:
    """Multi-series runs: every fill runs WITHIN each series, in time order,
    never across series (a gap in cell A is not filled from cell B)."""
    s_col, d_col = meta["series_column"], meta["date_column"]
    both = pd.concat([train.assign(_fold=0), test.assign(_fold=1)], ignore_index=True)
    both = both.sort_values([s_col, d_col], kind="mergesort")
    applied = {}
    for col, strategy in plan.items():
        if col not in both.columns or col in (s_col, d_col):
            continue
        g = both.groupby(s_col)[col]
        if strategy == "ffill":
            both[col] = g.ffill()
        elif strategy == "interpolate":
            both[col] = g.transform(lambda v: v.interpolate(limit_direction="forward"))
        else:
            try:
                both[col] = both[col].fillna(float(strategy))
            except (TypeError, ValueError):
                return json.dumps({"error": f"'{strategy}' is not ffill, interpolate or a number"})
        applied[col] = strategy
    for fold, name in ((0, "train.csv"), (1, "test.csv")):
        both[both["_fold"] == fold].drop(columns="_fold").to_csv(_run_dir(run_id) / name, index=False)
    meta["imputation"] = {**meta["imputation"], **applied}
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "imputed_columns": list(applied), "within": s_col,
                       "still_missing": int(both[list(applied)].isna().sum().sum()) if applied else 0})


@mcp.tool()
def apply_imputation(run_id: str, strategies: str) -> str:
    """Apply user-approved per-column strategies. strategies: JSON string
    {col: "ffill"|"interpolate"|"drop_column"|<literal value>}. "interpolate"
    is linear between the neighbours within each split; a gap at the very
    start of test is bridged from train's last value. "ffill" fills each split
    independently (train and test both only ever look backward in their own
    frame) except a gap at the very start of test is seeded from train's
    last known value — that's carrying forward a known PAST value across
    the split boundary, not leaking anything from test into train."""
    train, test, meta = _load_split(run_id)
    plan = json.loads(strategies)
    if meta.get("panel"):
        return _apply_panel_imputation(run_id, train, test, meta, plan)
    dropped, applied = [], {}
    for col, strategy in plan.items():
        if strategy == "drop_column":
            dropped.append(col)
            continue
        if strategy == "ffill":
            train[col] = train[col].ffill()
            seed = pd.concat([train[[col]].tail(1), test[[col]]], ignore_index=True)
            test[col] = seed[col].ffill().iloc[1:].reset_index(drop=True)
            applied[col] = "ffill"
        elif strategy == "interpolate":
            train[col] = train[col].interpolate(limit_direction="forward")
            seed = pd.concat([train[[col]].tail(1), test[[col]]], ignore_index=True)
            test[col] = seed[col].interpolate(limit_direction="forward").iloc[1:].reset_index(drop=True)
            applied[col] = "interpolate"
        else:
            value = strategy
            if pd.api.types.is_numeric_dtype(train[col]):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return json.dumps({"error": f"'{strategy}' is not a number — {col} is numeric"})
            train[col] = train[col].fillna(value)
            test[col] = test[col].fillna(value)
            applied[col] = value

    if dropped:
        train = train.drop(columns=dropped)
        test = test.drop(columns=[c for c in dropped if c in test.columns])
        meta["dropped_columns"] = sorted(set(meta["dropped_columns"]) | set(dropped))

    meta["imputation"] = {**meta["imputation"], **applied}
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "dropped_columns": dropped,
            "imputed_columns": list(applied),
            "train_shape": train.shape,
        }
    )


@mcp.tool()
def propose_drop_columns(run_id: str) -> str:
    """Suggest columns to drop (near-constant, near-unique/ID-like),
    computed from the training fold. Never proposes date_column or target."""
    train, _, meta = _load_split(run_id)
    skip = {meta["date_column"], meta["target"]}
    proposals = {}
    n = len(train)
    for col in train.columns:
        if col in skip:
            continue
        nunique = train[col].nunique(dropna=True)
        if nunique <= 1:
            proposals[col] = "constant column, no signal"
        elif nunique >= n * 0.98:
            proposals[col] = "near-unique, likely an ID column"
    return json.dumps({"proposals": proposals})


@mcp.tool()
def apply_drop_columns(run_id: str, columns: str) -> str:
    """Drop user-approved columns from both train and test. columns: JSON list."""
    train, test, meta = _load_split(run_id)
    cols = [
        c
        for c in json.loads(columns)
        if c in train.columns and c not in (meta["date_column"], meta["target"])
    ]
    train = train.drop(columns=cols)
    test = test.drop(columns=[c for c in cols if c in test.columns])
    meta["dropped_columns"] = sorted(set(meta["dropped_columns"]) | set(cols))
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps(
        {"run_id": run_id, "dropped_columns": cols, "train_shape": train.shape}
    )
