"""Data intake: every file format read the same way, named database /
ClickHouse / object-store sources that never expose credentials and never
write, and load_dataset's snapshot + lineage reaching the run.
No pytest fixtures — CI also runs this file as a plain script.

Run: python test_data_sources.py
"""
import os as _os
import tempfile as _tempfile

_os.environ.setdefault("AGENTIC_ML_DATA_DIR", _tempfile.mkdtemp(prefix="agentic-ml-test-"))

import gzip
import io
import json
import sqlite3
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd

import mcp_server as m
from core import datasource

FRAME = pd.DataFrame({"sim": ["A", "B", "C", "D"], "calls": [1, 2, 3, 4], "label": [0, 1, 0, 1]})


def _env(**values):
    """Set env vars for the block, restore after — without pytest's monkeypatch."""
    class _Scope:
        def __enter__(self):
            self.saved = {k: _os.environ.get(k) for k in values}
            _os.environ.update(values)

        def __exit__(self, *exc):
            for k, v in self.saved.items():
                _os.environ.pop(k, None) if v is None else _os.environ.__setitem__(k, v)
    return _Scope()


def test_every_file_format_reads_to_the_same_table(tmp_path):
    csv = FRAME.to_csv(index=False).encode()
    files = {
        "a.csv": csv,
        "a.csv.gz": gzip.compress(csv),
        "a.tsv": FRAME.to_csv(index=False, sep="\t").encode(),
        "a.jsonl": FRAME.to_json(orient="records", lines=True).encode(),
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    FRAME.to_parquet(tmp_path / "a.parquet", index=False)
    FRAME.to_feather(tmp_path / "a.feather")
    for name in [*files, "a.parquet", "a.feather"]:
        got = datasource.read_table(tmp_path / name)
        pd.testing.assert_frame_equal(got, FRAME, check_dtype=False, obj=name)


def test_archives_stack_members_that_share_a_schema_and_refuse_those_that_do_not(tmp_path):
    day1, day2 = FRAME.iloc[:2], FRAME.iloc[2:]
    with tarfile.open(tmp_path / "dumps.tar.gz", "w:gz") as tar:
        for name, part in (("day2.csv", day2), ("day1.csv", day1), (".hidden.csv", FRAME)):
            data = part.to_csv(index=False).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    got = datasource.read_table(tmp_path / "dumps.tar.gz")
    assert got["sim"].tolist() == ["A", "B", "C", "D"], "members stacked in name order, dotfiles skipped"

    with zipfile.ZipFile(tmp_path / "mixed.zip", "w") as z:
        z.writestr("a.csv", FRAME.to_csv(index=False))
        z.writestr("b.csv", "other,cols\n1,2\n")
    try:
        datasource.read_table(tmp_path / "mixed.zip")
        raise AssertionError("different schemas must not be stacked")
    except ValueError as e:
        assert "different columns" in str(e)


def test_an_upload_must_be_the_format_its_name_claims(tmp_path):
    parquet = io.BytesIO()
    FRAME.to_parquet(parquet, index=False)
    assert datasource.looks_like("x.parquet", parquet.getvalue()[:8192])
    assert datasource.looks_like("x.csv", b"a,b\n1,2\n")
    assert not datasource.looks_like("x.parquet", b"\x80\x04\x95pickle")  # a renamed pickle
    assert not datasource.looks_like("x.csv", b"a,b\n\x00\x01")
    assert not datasource.looks_like("model.pkl", b"\x80\x04")


def test_sql_source_reads_by_name_and_never_writes(tmp_path):
    db = tmp_path / "warehouse.db"
    with sqlite3.connect(db) as conn:
        FRAME.to_sql("subs", conn, index=False)
    with _env(AGENTIC_ML_SQL_WAREHOUSE=f"sqlite:///{db}"):
        assert "sql:warehouse" in datasource.configured_sources()
        got = datasource.fetch("sql:warehouse", "-- latest\nSELECT * FROM subs WHERE calls > 1", max_rows=2)
        assert len(got) == 2, "capped at max_rows"
        for bad in ("DELETE FROM subs", "SELECT 1; DROP TABLE subs", "update subs set calls = 0"):
            try:
                datasource.fetch("sql:warehouse", bad)
                raise AssertionError(f"must refuse: {bad}")
            except ValueError:
                pass
        result = json.loads(m.load_dataset("sql:warehouse", "SELECT * FROM subs", name="subs"))
        assert "error" not in result and result["shape"] == [4, 3], result
        assert "sqlite" not in json.dumps(result), "the URL never comes back to the model"
    with sqlite3.connect(db) as conn:
        assert conn.execute("select count(*) from subs").fetchone()[0] == 4
    missing = json.loads(m.load_dataset("sql:nowhere", "SELECT 1"))
    assert "AGENTIC_ML_SQL_NOWHERE" in missing["error"], "names the env var to set, never asks for a password"


def test_clickhouse_source_uses_readonly_parquet_over_http_and_keeps_credentials_out_of_the_url(tmp_path):
    seen = {}

    class Fake(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["params"] = parse_qs(urlsplit(self.path).query)
            seen["user"] = self.headers.get("X-ClickHouse-User")
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"])).decode()
            buf = io.BytesIO()
            FRAME.to_parquet(buf, index=False)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(buf.getvalue())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://analyst:s3cret@127.0.0.1:{server.server_port}/?database=cdr"
        with _env(AGENTIC_ML_CLICKHOUSE_EVENTS=url):
            got = datasource.fetch("clickhouse:events", "SELECT * FROM calls", max_rows=3)
    finally:
        server.shutdown()
    assert len(got) == 3
    assert seen["params"]["readonly"] == ["2"] and seen["params"]["database"] == ["cdr"]
    assert "s3cret" not in json.dumps(seen["params"]) and seen["user"] == "analyst"
    assert seen["body"].endswith("FORMAT Parquet")


def test_store_source_reads_a_glob_under_its_prefix(tmp_path):
    lake = tmp_path / "lake" / "events"
    lake.mkdir(parents=True)
    FRAME.iloc[:2].to_parquet(lake / "part-0.parquet", index=False)
    FRAME.iloc[2:].to_parquet(lake / "part-1.parquet", index=False)
    with _env(AGENTIC_ML_STORE_LAKE=f"file://{tmp_path / 'lake'}"):
        got = datasource.fetch("store:lake", "events/*.parquet")
        assert got["sim"].tolist() == ["A", "B", "C", "D"]
        try:
            datasource.fetch("store:lake", "../../etc/*")
            raise AssertionError("must not escape the store prefix")
        except ValueError:
            pass


def test_load_dataset_snapshot_carries_its_lineage_into_the_run(tmp_path):
    rows = pd.concat([FRAME] * 10, ignore_index=True).assign(calls=range(40))
    with tarfile.open(tmp_path / "export.tgz", "w:gz") as tar:
        data = rows.to_csv(index=False).encode()
        info = tarfile.TarInfo("export.csv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    loaded = json.loads(m.load_dataset(str(tmp_path / "export.tgz"), name="export"))
    assert loaded["path"].endswith("export.parquet") and loaded["truncated"] is False, loaded
    run = json.loads(m.prepare_dataset(loaded["path"], "label"))
    _, _, meta = m._load_split(run["run_id"])
    assert meta["data_source"]["source"].endswith("export.tgz") and meta["data_source"]["rows"] == 40
    capped = json.loads(m.load_dataset(str(tmp_path / "export.tgz"), name="export_small", max_rows=5))
    assert capped["truncated"] is True and capped["shape"][0] == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with _tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok {name}")
    print("all data-source checks passed")
