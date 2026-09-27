"""clustering agent — `eda` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _cleanup_old_runs, _run_dir, _save_meta  # noqa: F401
from core import data_tools
from core.datasource import read_table


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
def prepare_dataset(path: str, test_size: float = 0.2) -> str:
    """Plain fit/holdout split, BEFORE any learned preprocessing. Returns a
    run_id; every other tool from here on takes run_id, not a file path."""
    _cleanup_old_runs()
    df = read_table(path)
    if df.empty:
        return json.dumps({"error": f"'{path}' has no rows"})
    if len(df) < 20:
        return json.dumps(
            {
                "error": f"'{path}' has only {len(df)} rows — clustering needs at least 20 to be meaningful"
            }
        )

    duplicates_dropped = int(df.duplicated().sum())
    if duplicates_dropped:  # a duplicated row pulls a centroid toward itself twice
        df = df.drop_duplicates().reset_index(drop=True)
    fit, holdout = train_test_split(df, test_size=test_size, random_state=42)
    run_id = str(uuid.uuid4())
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True)
    fit.to_csv(run_dir / "train.csv", index=False)
    holdout.to_csv(run_dir / "test.csv", index=False)
    meta = {
        "source_path": path,
        "data_source": data_tools.lineage_for(path),
        "dropped_columns": [],
        "imputation": {},
        "algorithm": None,
        "algorithm_params": None,
        "training_metrics": None,
        "cluster_centroids": None,
    }
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "fit_shape": fit.shape,
            "holdout_shape": holdout.shape,
            "duplicates_dropped": duplicates_dropped,
            "message": "Split into fit and holdout folds before learned preprocessing to keep scaling/imputation leakage-safe.",
        }
    )


_intake = data_tools.register(mcp)
list_data_sources = _intake["list_data_sources"]
load_dataset = _intake["load_dataset"]
