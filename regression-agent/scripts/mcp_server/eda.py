"""regression agent — `eda` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _cleanup_old_runs, _load_split, _run_dir, _save_meta  # noqa: F401
from core import data_tools
from core import validation as shared_validation


@mcp.tool()
def eda(path: str) -> str:
    """Shape, dtypes, missingness, and target-agnostic summary stats for a CSV.
    Read-only — call this on the raw file before prepare_dataset."""
    df = read_table(path)
    return json.dumps(
        {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
            "describe": json.loads(df.describe(include="all").to_json()),
        }
    )


@mcp.tool()
def propose_group_column(path: str, target: str) -> str:
    """Columns that look like a repeating ENTITY id (a customer, subscriber,
    store, device) — the same identifier signals as propose_drop_columns,
    but only for columns that actually repeat. If one entity's rows can land
    in both train and test, the model is partly graded on entities it
    already saw, and every metric is optimistic. Read-only; ask the user."""
    df = read_table(path)
    n = len(df)
    candidates = {
        col: {"distinct_values": s["distinct"], "why_flagged": s["reason"], "strength": s["strength"],
              "rows_sharing_a_repeated_value": int(n - (df[col].value_counts() == 1).sum())}
        for col, s in identifier_columns(df, target).items() if s["distinct"] < n
    }
    return json.dumps({"path": path, "group_column_candidates": candidates})


@mcp.tool()
def propose_time_column(path: str) -> str:
    """Columns that order the rows in TIME (a date/timestamp, or an integer
    period index whose name says it is one), so the split can go forward in
    time. A random split on time-ordered rows trains on the future and
    tests on the past; every lag or trend feature then crosses the boundary.
    Read-only."""
    df = read_table(path)
    candidates = {}
    for col in df.columns:
        parsed = None
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            parsed = df[col]
        elif df[col].dtype == object:
            as_dt = shared_validation._parse_dates(df[col])
            if as_dt.notna().mean() >= 0.95:
                parsed = as_dt
        elif pd.api.types.is_numeric_dtype(df[col]) and any(
                p in col.lower() for p in ("date", "time", "day", "week", "month", "year", "period", "epoch")):
            parsed = df[col]
        if parsed is not None and parsed.notna().any():
            ordered = parsed.dropna()
            candidates[col] = {"distinct_values": int(ordered.nunique()),
                               "span": [str(ordered.min()), str(ordered.max())]}
    return json.dumps({"path": path, "time_column_candidates": candidates})


@mcp.tool()
def prepare_dataset(path: str, target: str, test_size: float = 0.2, group_column: str = "",
                    time_column: str = "", immature_after: str = "") -> str:
    """Train/test split, BEFORE any imputation/encoding — every downstream
    tool fits only on the training fold. Returns a run_id; every other tool
    takes run_id from here on.

    group_column (from propose_group_column — ask first): split by entity, so
    no entity is on both sides. time_column (from propose_time_column): split
    FORWARD in time — the newest test_size fraction is the test fold; pass
    both for entity-per-period data (entity_overlap_pct reports how many test
    rows belong to entities seen in training). immature_after (needs
    time_column): drop rows LATER than this before the split, because their
    target has not settled yet (revenue still accruing, claims still open,
    next-month usage not yet observed).

    Exact duplicate rows are dropped first. The split keys stay in the data
    for validation and never reach the model."""
    _cleanup_old_runs()
    df = read_table(path)
    if df.empty:
        return json.dumps({"error": f"'{path}' has no rows"})
    if target not in df.columns:
        return json.dumps({"error": f"target column '{target}' not found. Columns: {list(df.columns)}"})
    if df[target].isna().all():
        return json.dumps({"error": f"target column '{target}' is entirely missing"})
    if not pd.api.types.is_numeric_dtype(df[target]):
        return json.dumps({"error": f"target column '{target}' is not numeric — regression needs a continuous target"})
    for col in (group_column, time_column):
        if col and col not in df.columns:
            return json.dumps({"error": f"column '{col}' not found. Columns: {list(df.columns)}"})
    duplicates_dropped = int(df.duplicated().sum())
    if duplicates_dropped:
        df = df.drop_duplicates().reset_index(drop=True)
    df = df[df[target].notna()].reset_index(drop=True)
    immature_dropped = 0
    order = None
    if time_column:
        order = df[time_column] if pd.api.types.is_numeric_dtype(df[time_column]) \
            else shared_validation._parse_dates(df[time_column])
        if order.isna().any():
            return json.dumps({"error": f"time_column '{time_column}' has {int(order.isna().sum())} unorderable value(s)"})
        if immature_after:
            cutoff = float(immature_after) if pd.api.types.is_numeric_dtype(order) else pd.Timestamp(immature_after)
            late = order > cutoff
            immature_dropped = int(late.sum())
            df, order = df[~late].reset_index(drop=True), order[~late].reset_index(drop=True)
    elif immature_after:
        return json.dumps({"error": "immature_after needs time_column — it is a cutoff on that column"})
    if len(df) < 10:
        return json.dumps({"error": f"only {len(df)} row(s) — regression needs at least 10"})

    entity_overlap_pct = None
    if time_column:
        df = df.assign(_order=order).sort_values("_order", kind="mergesort").drop(columns="_order")
        cut = int(len(df) * (1 - test_size))
        train, test = df.iloc[:cut], df.iloc[cut:]
        if group_column:
            seen = set(train[group_column])
            entity_overlap_pct = round(float(test[group_column].isin(seen).mean()) * 100, 2)
    elif group_column:
        from sklearn.model_selection import GroupShuffleSplit

        fit_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                                 .split(df, groups=df[group_column].astype(str)))
        train, test = df.iloc[fit_idx], df.iloc[test_idx]
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=42)

    run_id = str(uuid.uuid4())
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True)
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)
    split_type = "temporal" if time_column else ("grouped" if group_column else "random")
    meta = {
        "target": target,
        "source_path": path,
        "group_column": group_column or None,
        "time_column": time_column or None,
        "split_type": split_type,
        "entity_overlap_pct": entity_overlap_pct,
        "duplicates_dropped": duplicates_dropped,
        "immature_after": immature_after or None,
        "immature_rows_dropped": immature_dropped,
        "dropped_columns": [],
        "imputation": {},
        "model": None,
        "best_params": None,
        "baseline_metrics": None,
        "tuned_metrics": None,
    }
    features = Path(f"{path}.features.json")  # aggregate_events' feature definitions
    if features.exists():
        meta["engineered_features"] = {k: v for k, v in json.loads(features.read_text()).items()
                                       if k in df.columns and k != target}
    lineage = data_tools.lineage_for(path)  # load_dataset's source record
    if lineage:
        meta["data_source"] = lineage
    _save_meta(run_id, meta)
    y = train[target]
    return json.dumps({
        "run_id": run_id,
        "train_shape": train.shape,
        "test_shape": test.shape,
        "split_type": split_type,
        "entity_overlap_pct": entity_overlap_pct,
        "duplicates_dropped": duplicates_dropped,
        "immature_rows_dropped": immature_dropped,
        "target_summary": y.describe().round(4).to_dict(),
        # > 1 on a non-negative target (money, usage, counts): consider apply_target_transform("log1p")
        "target_skew": round(float(y.skew()), 3),
        "target_min": float(y.min()),
    }, default=str)


@mcp.tool()
def detect_outliers(run_id: str) -> str:
    """Flag numeric-column outliers via z-score (|z| > 3) in the run's
    training fold. Feature columns are reported in outlier_counts_by_column
    (target excluded, same as classification-agent). target_outlier_count is
    a separate signal, specific to regression: an extreme target value (e.g.
    a $50M house in a pricing dataset) is a legitimate outlier concern in its
    own right, not something to fold silently into the feature report."""
    train, _, meta = _load_split(run_id)
    numeric = train.drop(columns=[meta["target"]]).select_dtypes(include="number")
    z = numeric.apply(lambda col: stats.zscore(col, nan_policy="omit"))
    flags = (z.abs() > 3).sum().to_dict()
    target_z = stats.zscore(train[meta["target"]], nan_policy="omit")
    target_outlier_count = int((np.abs(target_z) > 3).sum())
    return json.dumps(
        {
            "outlier_counts_by_column": flags,
            "target_outlier_count": target_outlier_count,
        }
    )


_intake = data_tools.register(mcp)
list_data_sources = _intake["list_data_sources"]
load_dataset = _intake["load_dataset"]
