"""anomaly agent — `eda` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import (
    _cleanup_old_runs,
    _json_default,
    _label_classes,
    _label_to_binary,
    _run_dir,
    _save_meta,
    _to_native,
)  # noqa: F401
from core import data_tools
from core import validation as shared_validation


@mcp.tool()
def eda(path: str) -> str:
    """Shape, dtypes, missingness, and target-agnostic summary stats for a
    data file in any supported format. Read-only — call this on the raw file
    before prepare_dataset."""
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
def prepare_dataset(path: str, label_column: str = "", test_size: float = 0.2,
                    time_column: str = "", group_column: str = "") -> str:
    """Train/test split before any learned preprocessing. label_column is
    optional: give one for held-out evaluation of the (still unsupervised)
    detector, leave it empty for pure unsupervised mode.
    time_column: split by time — the detector is judged on LATER rows, as it
    will be used (fraud and faults drift; a random split mixes next month's
    patterns into training). group_column: the entity (subscriber, device,
    card) — with no time_column the split keeps each entity on one side.
    Both stay in the data for feature building and never reach the detector.
    Exact duplicate rows are dropped first."""
    _cleanup_old_runs()
    df = read_table(path)
    if df.empty:
        return json.dumps({"error": f"'{path}' has no rows"})
    if not 0 < float(test_size) < 1:
        return json.dumps({"error": "test_size must be between 0 and 1"})
    for col in (time_column, group_column):
        if col and col not in df.columns:
            return json.dumps({"error": f"column '{col}' not found. Columns: {list(df.columns)}"})
    duplicates_dropped = int(df.duplicated().sum())
    if duplicates_dropped:
        df = df.drop_duplicates().reset_index(drop=True)

    meta = {
        "source_path": path,
        "label_column": label_column,
        "target": label_column or None,  # the shared feature tools keep it out of every feature
        "evaluation_available": bool(label_column),
        "normal_label": None,
        "anomaly_label": None,
        "group_column": group_column or None,
        "time_column": time_column or None,
        "duplicates_dropped": duplicates_dropped,
        "dropped_columns": [],
        "imputation": {},
        "algorithm": None,
        "contamination": None,
        "metrics": None,
        "artifact_path": None,
        "feature_importance": None,
    }

    if label_column:
        if label_column not in df.columns:
            return json.dumps({"error": f"label column '{label_column}' not found. Columns: {list(df.columns)}"})
        if df[label_column].isna().all():
            return json.dumps({"error": f"label column '{label_column}' is entirely missing"})
        if df[label_column].isna().any():
            return json.dumps({"error": f"label column '{label_column}' has missing values — provide a fully labeled "
                                        "binary column for held-out evaluation, or leave label_column empty for pure "
                                        "unsupervised mode"})
        non_null = df[label_column].dropna()
        unique_values = list(non_null.unique())
        if len(unique_values) != 2:
            return json.dumps({"error": f"label column '{label_column}' has {len(unique_values)} unique non-null "
                                        "values — anomaly evaluation needs exactly 2"})
        class_counts = non_null.value_counts()
        if int(class_counts.min()) < 2:
            return json.dumps({"error": f"class '{class_counts.idxmin()}' has only {int(class_counts.min())} row(s) — "
                                        "evaluation needs at least 2 rows in the minority class"})
        normal_label, anomaly_label = _label_classes(non_null)
        meta["normal_label"] = _to_native(normal_label)
        meta["anomaly_label"] = _to_native(anomaly_label)

    if time_column:
        order = df[time_column] if pd.api.types.is_numeric_dtype(df[time_column]) \
            else shared_validation._parse_dates(df[time_column])
        if order.isna().any():
            return json.dumps({"error": f"time_column '{time_column}' has {int(order.isna().sum())} unorderable value(s)"})
        df = df.assign(_order=order).sort_values("_order", kind="mergesort").drop(columns="_order")
        cut = int(len(df) * (1 - test_size))
        train, test = df.iloc[:cut], df.iloc[cut:]
        split_type = "temporal"
    elif group_column:
        from sklearn.model_selection import GroupShuffleSplit

        fit_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                                 .split(df, groups=df[group_column].astype(str)))
        train, test = df.iloc[fit_idx], df.iloc[test_idx]
        split_type = "grouped"
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=42,
                                       stratify=df[label_column] if label_column else None)
        split_type = "stratified" if label_column else "random"
    meta["split_type"] = split_type
    if label_column and _label_to_binary(test[label_column], meta).sum() == 0:
        return json.dumps({"error": f"the {split_type} split left no labelled anomaly in the test fold — "
                                    "a larger test_size, or a random split, is needed to evaluate"})

    run_id = str(uuid.uuid4())
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True)
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)
    features = Path(f"{path}.features.json")  # aggregate_events' feature definitions
    if features.exists():
        meta["engineered_features"] = {k: v for k, v in json.loads(features.read_text()).items() if k in df.columns}
    lineage = data_tools.lineage_for(path)  # load_dataset's source record
    if lineage:
        meta["data_source"] = lineage
    _save_meta(run_id, meta)

    result = {"run_id": run_id, "train_shape": train.shape, "test_shape": test.shape,
              "split_type": split_type, "duplicates_dropped": duplicates_dropped,
              "evaluation_available": bool(label_column)}
    if label_column:
        y_train = _label_to_binary(train[label_column], meta)
        result["train_label_counts"] = train[label_column].value_counts().to_dict()
        result["train_anomaly_prevalence"] = round(float(y_train.mean()), 4)
        result["anomaly_label"] = meta["anomaly_label"]
    return json.dumps(result, default=_json_default)


_intake = data_tools.register(mcp)
list_data_sources = _intake["list_data_sources"]
load_dataset = _intake["load_dataset"]
