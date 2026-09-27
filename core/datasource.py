"""Where data comes from, the same way for every agent: files in the common
formats, and NAMED database / warehouse / object-store sources.

Credentials never pass through a model. A source is a name — `sql:warehouse`,
`clickhouse:events`, `store:lake` — resolved here from the environment
(AGENTIC_ML_SQL_WAREHOUSE, AGENTIC_ML_CLICKHOUSE_EVENTS, AGENTIC_ML_STORE_LAKE).
The model, the execution log and the report only ever see the name and the
query. Queries are read-only by construction: one SELECT/WITH statement,
run inside a transaction that is always rolled back (SQL) or with the
server's readonly mode (ClickHouse), capped at `max_rows`.
"""
import gzip
import io
import os
import re
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

import pandas as pd

SOURCE_KINDS = {"sql": "AGENTIC_ML_SQL_", "clickhouse": "AGENTIC_ML_CLICKHOUSE_", "store": "AGENTIC_ML_STORE_"}
MAX_ROWS_DEFAULT = 5_000_000

_COMPRESSED = {".gz": "gzip", ".bz2": "bz2", ".xz": "xz", ".zst": "zstd"}
_TABULAR = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".feather", ".arrow", ".json", ".jsonl", ".ndjson",
            ".xlsx", ".xls"}
_ARCHIVES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".zip")
DATA_SUFFIXES = _TABULAR | set(_COMPRESSED) | {".tar", ".tgz", ".zip"}

# First bytes of each binary format — what an upload must start with to be
# accepted under that name. Text formats instead must not contain NUL bytes.
_MAGIC = {".parquet": b"PAR1", ".pq": b"PAR1", ".feather": b"ARROW1", ".arrow": b"ARROW1", ".gz": b"\x1f\x8b",
          ".tgz": b"\x1f\x8b", ".bz2": b"BZh", ".xz": b"\xfd7zXZ\x00", ".zst": b"\x28\xb5\x2f\xfd",
          ".zip": b"PK\x03\x04", ".xlsx": b"PK\x03\x04", ".xls": b"\xd0\xcf\x11\xe0"}


def is_data_file(path) -> bool:
    return Path(str(path)).suffix.lower() in DATA_SUFFIXES


def looks_like(name: str, head: bytes) -> bool:
    """Whether the first bytes of an upload match the format its name
    claims — a renamed pickle is not a parquet file."""
    suffix = Path(name).suffix.lower()
    if suffix not in DATA_SUFFIXES:
        return False
    if suffix == ".tar":
        return head[257:262] == b"ustar"
    if suffix in _MAGIC:
        return head.startswith(_MAGIC[suffix])
    return b"\0" not in head[:8192]


def _read_one(src, name: str) -> pd.DataFrame:
    """One file (a path or a bytes buffer) by its name's format, with an
    optional .gz/.bz2/.xz/.zst wrapper."""
    lower, compression = name.lower(), None
    for ext, codec in _COMPRESSED.items():
        if lower.endswith(ext):
            lower, compression = lower[: -len(ext)], codec
            break
    suffix = Path(lower).suffix
    if suffix in (".parquet", ".pq", ".feather", ".arrow", ".xlsx", ".xls") and compression:
        # Columnar and Excel formats compress internally; a gzip around them is rare but readable.
        src = io.BytesIO(gzip.decompress(src.read() if hasattr(src, "read") else Path(src).read_bytes())) \
            if compression == "gzip" else src
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(src)
    if suffix in (".feather", ".arrow"):
        return pd.read_feather(src)
    if suffix in (".jsonl", ".ndjson", ".json"):
        return pd.read_json(src, lines=suffix != ".json", compression=compression)
    if suffix in (".xlsx", ".xls"):
        try:
            return pd.read_excel(src)
        except ImportError as e:
            raise ValueError(f"reading {suffix} needs openpyxl (xlsx) or xlrd (xls): {e}") from e
    if suffix in (".csv", ".tsv", ".txt"):
        sep = {".csv": ",", ".tsv": "\t"}.get(suffix)
        return pd.read_csv(src, sep=sep, engine=None if sep else "python", compression=compression)
    raise ValueError(f"unsupported format '{name}' — readable: {sorted(DATA_SUFFIXES)}")


def _concat(frames: list, names: list, where: str) -> pd.DataFrame:
    """Archive members become one table only when they share a schema —
    daily dumps of one export, not unrelated files."""
    if not frames:
        raise ValueError(f"{where} holds no readable data file")
    first = list(frames[0].columns)
    odd = [n for n, f in zip(names, frames) if set(f.columns) != set(first)]
    if odd:
        raise ValueError(f"{where}: members {odd} have different columns from {names[0]} — "
                         "unrelated files cannot be stacked; extract the one you need")
    return pd.concat([f[first] for f in frames], ignore_index=True)


def read_table(path) -> pd.DataFrame:
    """Any supported file: csv/tsv/txt (optionally .gz/.bz2/.xz/.zst),
    parquet, feather, json/jsonl, xlsx/xls, and tar/tgz/zip archives whose
    data members share one schema (stacked in name order). Archive members
    are read in memory, never extracted to disk."""
    name = str(path)
    lower = name.lower()
    if lower.endswith(_ARCHIVES[:-1]):
        with tarfile.open(name) as tar:
            members = sorted((m for m in tar.getmembers() if m.isfile() and is_data_file(m.name)
                              and not Path(m.name).name.startswith(".")), key=lambda m: m.name)
            frames = [_read_one(io.BytesIO(tar.extractfile(m).read()), m.name) for m in members]
        return _concat(frames, [m.name for m in members], name)
    if lower.endswith(".zip"):
        with zipfile.ZipFile(name) as z:
            members = sorted(m for m in z.namelist() if is_data_file(m) and not m.endswith("/")
                             and not Path(m).name.startswith("."))
            frames = [_read_one(io.BytesIO(z.read(m)), m) for m in members]
        return _concat(frames, members, name)
    return _read_one(name, name)


def configured_sources() -> list[str]:
    """Every named source the environment defines — names only, never URLs."""
    return sorted(f"{kind}:{key[len(prefix):].lower()}" for kind, prefix in SOURCE_KINDS.items()
                  for key in os.environ if key.startswith(prefix) and len(key) > len(prefix))


def is_named_source(source: str) -> bool:
    return source.partition(":")[0] in SOURCE_KINDS and "://" not in source


def _url(source: str) -> tuple[str, str]:
    kind, _, name = source.partition(":")
    env = SOURCE_KINDS[kind] + re.sub(r"[^A-Za-z0-9]", "_", name).upper()
    url = os.environ.get(env)
    if not name or not url:
        raise ValueError(f"source '{source}' is not configured — its URL goes in the environment variable {env} "
                         f"(configured: {configured_sources() or 'none'})")
    return kind, url


_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)


def read_only_statement(query: str) -> str:
    """One SELECT/WITH statement, comments stripped — this loader reads, it
    never writes. (A data-modifying CTE still passes this check; the SQL
    path's rollback and ClickHouse's readonly mode are what stop it.)"""
    body = _COMMENTS.sub(" ", query).strip().rstrip(";").strip()
    if ";" in body:
        raise ValueError("one statement per query")
    if not re.match(r"(select|with)\b", body, re.I):
        raise ValueError("only SELECT / WITH queries — data sources are read, never written")
    return body


def fetch(source: str, query: str, max_rows: int = MAX_ROWS_DEFAULT) -> pd.DataFrame:
    """Rows from a named source: `sql:<name>` (any SQLAlchemy URL — Postgres,
    MySQL, Snowflake, BigQuery, Trino, ... once its driver is installed),
    `clickhouse:<name>` (the HTTP interface, http(s)://user:pass@host:port/?database=db),
    or `store:<name>` (an fsspec URL prefix — s3://, gs://, az://, file:// —
    with `query` as the path or glob under it)."""
    kind, url = _url(source)
    if kind == "store":
        return _fetch_store(url, query, max_rows)
    body = read_only_statement(query)
    return _fetch_clickhouse(url, body, max_rows) if kind == "clickhouse" else _fetch_sql(url, body, max_rows)


def _fetch_sql(url: str, body: str, max_rows: int) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    frames, rows = [], 0
    try:
        with engine.connect() as conn:
            transaction = conn.begin()
            try:
                for chunk in pd.read_sql(text(body), conn, chunksize=100_000):
                    frames.append(chunk)
                    rows += len(chunk)
                    if rows >= max_rows:
                        break
            finally:
                transaction.rollback()  # whatever the statement did, nothing persists
    finally:
        engine.dispose()
    return pd.concat(frames, ignore_index=True).head(max_rows) if frames else pd.DataFrame()


def _fetch_clickhouse(url: str, body: str, max_rows: int) -> pd.DataFrame:
    import requests

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("a ClickHouse source is its HTTP interface: http(s)://user:pass@host:port/?database=db")
    port = parts.port or (8443 if parts.scheme == "https" else 8123)
    params = {**dict(parse_qsl(parts.query)), "readonly": "2", "max_result_rows": str(max_rows),
              "result_overflow_mode": "break"}
    headers = {}
    if parts.username:
        headers["X-ClickHouse-User"] = unquote(parts.username)
    if parts.password:
        headers["X-ClickHouse-Key"] = unquote(parts.password)
    response = requests.post(f"{parts.scheme}://{parts.hostname}:{port}/", params=params, headers=headers,
                             data=f"{body} FORMAT Parquet".encode(), timeout=600)
    if response.status_code != 200:
        raise ValueError(f"ClickHouse refused the query ({response.status_code}): {response.text[:500]}")
    return pd.read_parquet(io.BytesIO(response.content)).head(max_rows)


def _fetch_store(url: str, pattern: str, max_rows: int) -> pd.DataFrame:
    import fsspec

    if not pattern or ".." in pattern.split("/"):
        raise ValueError("a store query is a path or glob under the store's prefix, e.g. 'events/2026-09/*.parquet'")
    # ponytail: every matched file is read whole before the row cap; stream per file if lakes get huge
    files = sorted(fsspec.open_files(url.rstrip("/") + "/" + pattern.lstrip("/"), mode="rb"), key=lambda f: f.path)
    readable = [f for f in files if is_data_file(f.path)]
    frames = []
    for f in readable:
        with f as handle:
            frames.append(_read_one(io.BytesIO(handle.read()), f.path))
    return _concat(frames, [f.path for f in readable], pattern).head(max_rows)
