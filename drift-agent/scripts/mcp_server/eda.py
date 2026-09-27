"""drift agent — `eda` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage


@mcp.tool()
def eda(path: str) -> str:
    """Shape, dtypes, missingness, and summary stats for one data file in any
    supported format. Read-only — call this on either file before
    detect_drift if you want a first look."""
    df = read_table(path)
    return json.dumps(
        {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
            "describe": json.loads(df.describe(include="all").to_json()),
        }
    )


from core import data_tools  # noqa: E402

_intake = data_tools.register(mcp)
list_data_sources = _intake["list_data_sources"]
load_dataset = _intake["load_dataset"]
