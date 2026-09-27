"""Data intake tools shared by every agent: what can be loaded, and loading
it — a file in any supported format or a named database / warehouse /
object-store source (core/datasource.py) — into a parquet snapshot every
other tool reads, with its lineage beside it. Nothing here depends on an
agent's runs; each agent registers these on its own server:

    globals().update(data_tools.register(mcp))
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import datasource, runtime, toolguard


def list_data_sources() -> str:
    """What can be loaded: the named sources this deployment configures
    (names only — their URLs and credentials live in the environment and
    never reach the conversation) and the file formats every tool reads.
    Read-only; call it when the user points at a database, warehouse or
    bucket rather than a file."""
    return json.dumps({
        "named_sources": datasource.configured_sources(),
        "file_formats": sorted(datasource.DATA_SUFFIXES),
        "how": "files: pass the path to any tool, or to load_dataset for a snapshot; "
               "sql:<name> / clickhouse:<name>: load_dataset(source, query=SELECT ...); "
               "store:<name>: load_dataset(source, query=<path or glob under the store>)",
    })


def load_dataset(source: str, query: str = "", name: str = "", max_rows: int = datasource.MAX_ROWS_DEFAULT) -> str:
    """Bring data in from where it lives and freeze it as a parquet SNAPSHOT
    that every other tool reads — so the run is reproducible after the table
    changes. Call it before eda/prepare_dataset whenever the data is not
    already a plain file, or when a big/compressed file will be read many
    times.

    source:
      a file path  csv/tsv/txt (optionally .gz/.bz2/.xz/.zst), parquet,
                   feather, json/jsonl, xlsx, or a tar/tgz/zip archive whose
                   members share one schema (stacked)
      sql:<name>   a configured SQLAlchemy database (Postgres, MySQL,
                   Snowflake, BigQuery, Trino...) — query is ONE SELECT/WITH
      clickhouse:<name>  a configured ClickHouse — query is ONE SELECT/WITH,
                   run in the server's readonly mode
      store:<name> a configured object-store prefix (s3://, gs://, az://,
                   file://) — query is a path or glob under it
    Names come from list_data_sources; never ask the user for a password or
    a connection string — a source that is not configured is an ops request.

    Queries are read-only and capped at max_rows; the result says whether the
    cap cut the data (`truncated`), which matters: a LIMIT-ed sample of a
    table ordered by time is the OLDEST rows, not a random sample. Push
    filtering, joins and heavy aggregation into the query — the database does
    them better than a pandas process can. The snapshot's lineage (source,
    query, rows, time) is carried into the run by prepare_dataset."""
    try:
        if datasource.is_named_source(source):
            df = datasource.fetch(source, query, max_rows + 1)
        else:
            toolguard.check_input(source)  # a file follows the same access rules as every tool's `path`
            df = datasource.read_table(source).head(max_rows + 1)
    except toolguard.Refused as refused:
        return json.dumps({"error": f"refused: {refused}"})
    except Exception as e:  # driver, network, SQL and parse errors all come back readable
        return json.dumps({"error": f"could not load {source}: {type(e).__name__}: {e}"})
    if df.empty:
        return json.dumps({"error": f"{source} returned no rows"})
    truncated = len(df) > max_rows
    df = df.head(max_rows)
    stem = name or f"{source.replace(':', '_').replace('/', '_')}_{hashlib.sha256(f'{source}|{query}'.encode()).hexdigest()[:8]}"
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(stem).name).lstrip(".") or "dataset"
    snapshot = toolguard.data_dir() / "uploads" / toolguard.user_folder(runtime.user()) / f"{stem}.parquet"
    try:
        snapshot = toolguard.check_output(str(snapshot), "load_dataset")
    except toolguard.Refused as refused:
        return json.dumps({"error": f"refused: {refused}"})
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(snapshot, index=False)
    lineage = {
        "source": source,
        "query": query or None,
        "rows": len(df),
        "truncated_at_max_rows": truncated,
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    Path(f"{snapshot}.source.json").write_text(json.dumps(lineage, indent=2))
    return json.dumps({
        "path": str(snapshot),
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "truncated": truncated,
        "next": "every tool takes this path — start with eda(path)",
    })


def lineage_for(path) -> dict | None:
    """The snapshot lineage load_dataset left beside `path`, if any — for
    prepare_dataset to carry into the run."""
    sidecar = Path(f"{path}.source.json")
    return json.loads(sidecar.read_text()) if sidecar.exists() else None


TOOLS = (list_data_sources, load_dataset)


def register(mcp) -> dict:
    return {fn.__name__: mcp.tool()(fn) for fn in TOOLS}

