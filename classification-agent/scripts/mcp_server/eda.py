"""Everything that runs on a CSV before or right after the train/test split:
shape/dtypes/missingness, the split itself, class balance, and numeric
outliers. Nothing here fits a learned statistic — prepare_dataset is the
one line after which every other skill's tools take run_id, not a path."""

import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd
from core import data_tools
from core.datasource import read_table
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .core import (
    _cleanup_old_runs,
    _load_split,
    _run_dir,
    _save_meta,
    _to_native,
    infer_positive_label,
    mcp,
    parse_dates,
)
from .data_cleaning import identifier_columns

# Substrings that make an INTEGER column a plausible clock rather than just a
# sortable number. Deliberately short: a false positive here would offer a
# temporal split on an arbitrary counter, which is worse than not offering one.
_TEMPORAL_NAME_PATTERNS = (
    "date",
    "time",
    "day",
    "week",
    "month",
    "year",
    "period",
    "dt",
    "epoch",
)


def _NAME_LOOKS_TEMPORAL(col: str) -> bool:
    return any(p in col.lower() for p in _TEMPORAL_NAME_PATTERNS)


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
    """Detects columns that look like a repeating ENTITY id (a customer,
    caller, account...) — reusing the same identifier heuristics as
    data-cleaning's propose_drop_columns (name pattern or high-cardinality-
    categorical), but only when the column actually repeats (nunique < row
    count — a fully-unique id has no group-leakage risk to begin with,
    that's data-cleaning's territory instead). Call this BEFORE
    prepare_dataset: if the same entity's rows can land in both train and
    test (e.g. one row per caller per day, a common shape for pre-
    aggregated telco/transaction data), a plain random split lets the model
    partially recognize an entity it already saw in training, inflating
    every metric downstream. Read-only — the agent should surface any
    candidate and ask the user whether to group-split on it (via
    prepare_dataset's group_column) rather than assume any particular
    column name; a dataset's entity id is different every time."""
    df = read_table(path)
    n = len(df)
    candidates = {}
    for col, signal in identifier_columns(df, target).items():
        nunique = signal["distinct"]
        if nunique >= n:
            continue  # every value unique — no repeats, no group-leakage risk
        candidates[col] = {
            "distinct_values": nunique,
            "rows_sharing_a_repeated_value": int(
                n - (df[col].value_counts() == 1).sum()
            ),
            "why_flagged": signal["reason"],
            "strength": signal["strength"],
        }
    return json.dumps({"path": path, "group_column_candidates": candidates})


@mcp.tool()
def propose_time_column(path: str) -> str:
    """Detects columns that order the rows in TIME (a date/timestamp, or an
    integer period/day index), so the agent can offer a forward-chained
    split instead of a random one. Call this BEFORE prepare_dataset,
    alongside propose_group_column.

    Why this matters: a stratified random split on time-ordered data puts
    later rows in train and earlier rows in test, so any trailing-window
    feature (a 7-day lag, a rolling mean, a trend) is computed partly from
    rows the model also trained on, and the test fold stops being a
    forecast of the future. propose_group_column catches the same class of
    problem for entities; nothing catches this one, which is why it exists.
    Read-only."""
    df = read_table(path)
    candidates = {}
    for col in df.columns:
        parsed = None
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            parsed = df[col]
        elif df[col].dtype == object:
            as_dt = parse_dates(df[col])
            if as_dt.notna().mean() >= 0.95:
                parsed = as_dt
        elif pd.api.types.is_numeric_dtype(df[col]) and _NAME_LOOKS_TEMPORAL(col):
            # an integer period index (day_number, week, month) only counts
            # when the NAME says it is time — any integer column is sortable,
            # which is not the same as being a clock.
            parsed = df[col]
        if parsed is None:
            continue
        ordered = parsed.dropna()
        candidates[col] = {
            "distinct_values": int(ordered.nunique()),
            "already_sorted": bool(ordered.is_monotonic_increasing),
            "span": [str(ordered.min()), str(ordered.max())] if len(ordered) else [],
        }
    return json.dumps({"path": path, "time_column_candidates": candidates})


@mcp.tool()
def prepare_dataset(
    path: str,
    target: str,
    test_size: float = 0.2,
    group_column: str = "",
    time_column: str = "",
    positive_label: str = "",
    immature_after: str = "",
) -> str:
    """Stratified train/test split, BEFORE any imputation/encoding/SMOTE —
    every downstream tool fits only on the training fold, so nothing about
    the held-out test set ever leaks into a preprocessing decision. Returns
    a run_id; every other tool from here on takes run_id, not a file path.

    group_column: optional, from propose_group_column's candidates (ask the
    user first — don't guess a column name). When set, splits by that
    column's distinct values (StratifiedGroupKFold, approximating the
    requested test_size) instead of by row, so the same entity's rows can
    never straddle train and test — leave empty for a plain per-row
    stratified split when no entity id is present or the user declines.

    time_column: optional, from propose_time_column's candidates. When set,
    the split is FORWARD-CHAINED: rows are sorted by this column and the
    last test_size fraction becomes the test fold, so training data is
    strictly older than test data. Use this whenever the rows are ordered in
    time and any feature looks backwards (a lag, a rolling window, a trend,
    a "days since" recency) — under a random split those features are
    computed partly from rows that also sit in training, and the test score
    stops measuring generalization to the future. A temporal split cannot be
    stratified (the class mix of the future is not ours to choose), so this
    verifies both classes survive in both folds and errors out rather than
    handing back a single-class fold.

    Passing BOTH is allowed and is the honest option for entity-per-period
    data: the split stays temporal, and the result reports how much entity
    overlap remains across the boundary (entity_overlap_pct) instead of
    silently purging rows. That overlap is usually legitimate — a production
    scorer does see yesterday's callers again — but it inflates metrics
    relative to unseen entities, so it is surfaced rather than hidden.

    positive_label: for a binary target, WHICH of the two labels is the event
    being predicted — the class that recall_positive, the tuned operating
    point, fairness TPR, PR-AUC and SHAP all refer to. Left empty it is
    inferred: the sorted-max label ("Yes" over "No", 1 over 0), except where
    that label names the absence of the event ("no_churn" sorts above
    "churn", "legit" above "fraud"), which flips it. The inference and
    whether it flipped are BOTH reported back as positive_label /
    positive_label_inferred — check them; on a target whose labels the
    word list doesn't cover ("A"/"B", "class1"/"class2") the guess is just
    sorting, and passing the right one explicitly is the only way to be
    sure every positive-class number describes the class you care about.

    immature_after: with time_column, drop rows LATER than this before the
    split, because their labels have not settled — fraud still undiscovered,
    first payment not yet due, churn window still open. Left in, the newest
    period's negatives are just un-investigated positives, and they land in
    the test fold of a temporal split, where they flatter precision. Set it
    to (newest date - how long outcomes take to settle)."""
    _cleanup_old_runs()
    df = read_table(path)
    if df.empty:
        return json.dumps({"error": f"'{path}' has no rows"})
    # Exact duplicate rows (every column, target included) go before the
    # split: otherwise a copy lands in train and its twin in test, and the
    # model is graded on rows it memorised. Decided and recorded, not asked —
    # identical rows carry no information a second time.
    duplicates_dropped = int(df.duplicated().sum())
    if duplicates_dropped:
        df = df.drop_duplicates().reset_index(drop=True)
    immature_dropped = 0
    if immature_after:
        if not time_column or time_column not in df.columns:
            return json.dumps(
                {
                    "error": "immature_after needs time_column — it is a cutoff on that column"
                }
            )
        numeric = pd.api.types.is_numeric_dtype(df[time_column])
        stamp = df[time_column] if numeric else parse_dates(df[time_column])
        cutoff = float(immature_after) if numeric else pd.Timestamp(immature_after)
        late = (
            stamp > cutoff
        )  # unparseable stays in, so the temporal split below reports it
        immature_dropped = int(late.sum())
        df = df[~late].reset_index(drop=True)
    if target not in df.columns:
        return json.dumps(
            {
                "error": f"target column '{target}' not found. Columns: {list(df.columns)}"
            }
        )
    if df[target].isna().all():
        return json.dumps({"error": f"target column '{target}' is entirely missing"})
    class_counts = df[target].value_counts()
    if class_counts.shape[0] < 2:
        return json.dumps(
            {
                "error": f"target column '{target}' has only {class_counts.shape[0]} class(es) — classification needs at least 2"
            }
        )
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        return json.dumps(
            {
                "error": f"class '{class_counts.idxmin()}' has only {min_class_count} row(s) — "
                "a stratified split needs at least 2 rows per class",
            }
        )

    if time_column and time_column not in df.columns:
        return json.dumps(
            {
                "error": f"time_column '{time_column}' not found. Columns: {list(df.columns)}"
            }
        )
    if group_column and group_column not in df.columns:
        return json.dumps(
            {
                "error": f"group_column '{group_column}' not found. Columns: {list(df.columns)}"
            }
        )

    entity_overlap_pct = None
    if time_column:
        order = df[time_column]
        if not pd.api.types.is_numeric_dtype(order):
            order = parse_dates(order)
        if order.isna().any():
            return json.dumps(
                {
                    "error": f"time_column '{time_column}' has {int(order.isna().sum())} value(s) that are neither a parseable date nor a number — a forward-chained split needs every row to be orderable"
                }
            )
        # mergesort keeps rows with the same timestamp in their original
        # order, so a tie at the cut point is resolved deterministically
        # rather than by whatever order sort happens to produce.
        df = (
            df.assign(_split_order=order)
            .sort_values("_split_order", kind="mergesort")
            .drop(columns="_split_order")
        )
        cut = int(len(df) * (1 - test_size))
        if cut == 0 or cut == len(df):
            return json.dumps(
                {
                    "error": f"test_size {test_size} leaves one fold empty for {len(df)} rows"
                }
            )
        train, test = df.iloc[:cut], df.iloc[cut:]
        for fold_name, fold in (("train", train), ("test", test)):
            if fold[target].nunique(dropna=True) < 2:
                return json.dumps(
                    {
                        "error": f"a forward-chained split on '{time_column}' leaves the {fold_name} fold with only "
                        f"one class — the positives are not spread across time in this file. Use a random or "
                        f"grouped split, or aggregate to a coarser period, but do not ignore this: it means "
                        f"the label itself is time-localised.",
                    }
                )
        if group_column:
            overlap = set(train[group_column]) & set(test[group_column])
            entity_overlap_pct = round(
                float(test[group_column].isin(overlap).mean()) * 100, 2
            )
    elif group_column:
        # ponytail: StratifiedGroupKFold approximates the requested test_size
        # via n_splits = round(1/test_size) and takes one fold as the test
        # set — sklearn has no single-shot grouped+stratified train_test_split,
        # this is the standard workaround. Exact split size will vary
        # slightly with group sizes, unlike the plain row-level split below.
        n_splits = max(2, round(1 / test_size))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        train_idx, test_idx = next(
            splitter.split(df, df[target], groups=df[group_column])
        )
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        shared_groups = set(train[group_column]) & set(test[group_column])
        if shared_groups:
            return json.dumps(
                {
                    "error": f"internal: {len(shared_groups)} group(s) leaked across the split — this should be impossible with StratifiedGroupKFold"
                }
            )
    else:
        train, test = train_test_split(
            df, test_size=test_size, stratify=df[target], random_state=42
        )

    run_id = str(uuid.uuid4())
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True)
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)
    labels = sorted(pd.unique(df[target].dropna()))
    if positive_label:
        matched = [l for l in labels if str(l) == positive_label]
        if not matched:
            return json.dumps(
                {
                    "error": f"positive_label '{positive_label}' is not a value of '{target}' — "
                    f"its labels are {[str(l) for l in labels]}"
                }
            )
        resolved_positive, inferred = _to_native(matched[0]), False
    else:
        resolved_positive, inferred = infer_positive_label(df[target]), True

    meta = {
        "target": target,
        "source_path": path,
        "positive_label": resolved_positive,
        "positive_label_inferred": inferred,
        "group_column": group_column or None,
        "time_column": time_column or None,
        "split_type": (
            "temporal" if time_column else ("grouped" if group_column else "random")
        ),
        "entity_overlap_pct": entity_overlap_pct,
        "duplicates_dropped": duplicates_dropped,
        "immature_after": immature_after or None,
        "immature_rows_dropped": immature_dropped,
        # Lineage: which bytes this run was trained on, and which exact test
        # fold its numbers come from — two runs are only comparable on test
        # when these fingerprints match.
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256((run_dir / "test.csv").read_bytes()).hexdigest(),
        "dropped_columns": {},
        "imputation": {},
        "use_smote": False,
        "smote_sampling_strategy": 1.0,
        "model": None,
        "best_params": None,
        "baseline_metrics": None,
        "tuned_metrics": None,
    }
    # aggregate_events leaves its feature definitions beside the file it wrote
    sidecar = Path(f"{path}.features.json")
    if sidecar.exists():
        meta["engineered_features"] = {
            k: v
            for k, v in json.loads(sidecar.read_text()).items()
            if k in df.columns and k != target
        }
    # load_dataset leaves where the rows came from (source name, query, when)
    lineage = Path(f"{path}.source.json")
    if lineage.exists():
        meta["data_source"] = json.loads(lineage.read_text())
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "train_shape": train.shape,
            "test_shape": test.shape,
            "train_class_counts": train[target].value_counts().to_dict(),
            "test_class_counts": test[target].value_counts().to_dict(),
            "group_column": group_column or None,
            "time_column": time_column or None,
            "split_type": (
                "temporal" if time_column else ("grouped" if group_column else "random")
            ),
            "entity_overlap_pct": entity_overlap_pct,
            "duplicates_dropped": duplicates_dropped,
            "immature_rows_dropped": immature_dropped,
            "positive_label": resolved_positive,
            "positive_label_inferred": inferred,
        }
    )


@mcp.tool()
def inspect_target(path: str, target: str) -> str:
    """Target-column shape check on the RAW file, before prepare_dataset:
    dtype, class count/labels, binary vs multiclass, and class balance %.
    Call this alongside eda to decide stratification and which metrics
    (binary ROC/PR vs macro-averaged) will apply downstream. Read-only."""
    df = read_table(path)
    if target not in df.columns:
        return json.dumps({"error": f"no column '{target}' in {path}"})
    counts = df[target].value_counts()
    return json.dumps(
        {
            "dtype": str(df[target].dtype),
            "n_classes": int(counts.shape[0]),
            "is_binary": bool(counts.shape[0] == 2),
            "class_counts": counts.to_dict(),
            "class_pct": (counts / len(df) * 100).round(2).to_dict(),
            "missing_target_rows": int(df[target].isna().sum()),
        }
    )


@mcp.tool()
def check_imbalance(run_id: str) -> str:
    """Class distribution and imbalance ratio for the run's training fold —
    the fold the model will actually be fit on."""
    train, _, meta = _load_split(run_id)
    counts = train[meta["target"]].value_counts()
    ratio = counts.min() / counts.max()
    return json.dumps(
        {
            "counts": counts.to_dict(),
            "imbalance_ratio": round(float(ratio), 4),
            "is_imbalanced": bool(ratio < 0.2),
        }
    )


@mcp.tool()
def detect_outliers(run_id: str) -> str:
    """Flag numeric-column outliers via z-score (|z| > 3) in the run's
    training fold, excluding the target."""
    train, _, meta = _load_split(run_id)
    numeric = train.drop(columns=[meta["target"]]).select_dtypes(include="number")
    z = numeric.apply(lambda col: stats.zscore(col, nan_policy="omit"))
    flags = (z.abs() > 3).sum().to_dict()
    return json.dumps({"outlier_counts_by_column": flags})


_intake = data_tools.register(mcp)
list_data_sources = _intake["list_data_sources"]
load_dataset = _intake["load_dataset"]
