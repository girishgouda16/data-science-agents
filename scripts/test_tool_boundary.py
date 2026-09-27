"""The tool boundary (core/toolguard.py), attacked the way a prompt-injected
model or a curious colleague would: every negative case must fail closed with
a clear error, and the legitimate flow must still work.

The pickle payload is harmless — it only writes a marker file; the test
asserts the marker never appears.

Run:  python -m pytest -q scripts/test_tool_boundary.py
"""

import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mlops_all_agents import (
    _agent,
    _tabular,
)  # noqa: E402 — sets temp data dir + MLflow store
from test_multi_user import _upload, as_user  # noqa: E402

from core import toolguard  # noqa: E402

import pytest  # noqa: E402

DATA = toolguard.data_dir()


class _Payload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "code ran")


def _refused(result: str) -> str:
    error = json.loads(result).get("error", "")
    assert error.startswith("refused:"), result
    return error


@pytest.fixture(scope="module")
def regression():
    with _agent("regression") as module:
        yield module


@pytest.fixture(scope="module")
def serving():
    with _agent("serving") as module:
        yield module


@pytest.fixture(scope="module")
def alice_export(regression):
    """alice's trained, exported model — what the legitimate flow produces."""
    m = regression
    with as_user("alice"):
        run_id = json.loads(m.prepare_dataset(_upload("bills"), "spend"))["run_id"]
        assert "error" not in json.loads(m.train_model(run_id, "linear_regression"))
        out = str(DATA / "artifacts" / f"alice_{run_id}.pkl")
        exported = json.loads(m.export_model(run_id, out, force=True))
        assert "error" not in exported, exported
    return run_id, out


def test_the_legitimate_flow_still_works(regression, serving, alice_export):
    _, pkl = alice_export
    with as_user("alice"):
        data = _upload("new_bills")
        assert "prediction_summary" in json.loads(regression.predict(pkl, data))
        assert "error" not in json.loads(serving.predict(pkl, data_path=data))
        shared = DATA / "shared_sample.csv"
        _tabular().to_csv(shared, index=False)
        assert "shape" in json.loads(regression.eda(str(shared))), (
            "shared datasets stay readable"
        )


def test_paths_outside_the_users_data_are_refused(regression, tmp_path):
    outside = tmp_path / "secret.csv"
    _tabular().to_csv(outside, index=False)
    with as_user("mallory"):
        own = Path(_upload("mine"))
        _refused(regression.eda(str(outside)))  # absolute
        _refused(
            regression.eda(str(own.parent / ".." / ".." / "runs" / "x.csv"))
        )  # ../ escape
        link = own.parent / "link.csv"
        link.symlink_to(outside)
        _refused(regression.eda(str(link)))  # symlink out
        _refused(regression.eda("/etc/passwd"))  # not a CSV
        _refused(regression.prepare_dataset(str(outside), "spend"))
        assert "shape" in json.loads(regression.eda(str(own))), (
            "own upload still readable"
        )


def test_a_colleagues_upload_is_refused(regression):
    with as_user("alice"):
        hers = _upload("private")
    with as_user("mallory"):
        _refused(regression.eda(hers))
        _refused(regression.prepare_dataset(hers, "spend"))


def test_a_crafted_pickle_never_loads(regression, serving, tmp_path):
    marker = tmp_path / "pwned"
    with as_user("mallory"):
        uploaded = Path(_upload("x")).with_name("model.pkl")
        uploaded.write_bytes(pickle.dumps(_Payload(marker)))
        planted = DATA / "artifacts" / "planted.pkl"
        planted.write_bytes(pickle.dumps(_Payload(marker)))
        data = _upload("rows")
        _refused(serving.predict(str(uploaded), data_path=data))  # outside artifacts
        _refused(serving.inspect_model(str(planted)))  # no export record
        _refused(regression.predict(str(planted), data))
    # No user (trusted caller) still needs the record: a pickle is code.
    _refused(serving.inspect_model(str(planted)))
    assert not marker.exists(), "the payload ran"


def test_a_tampered_export_is_refused(serving, alice_export, tmp_path):
    _, pkl = alice_export
    tampered = Path(pkl).with_name("tampered.pkl")
    tampered.write_bytes(Path(pkl).read_bytes())
    toolguard.trust(tampered, "alice")
    marker = tmp_path / "pwned"
    tampered.write_bytes(pickle.dumps(_Payload(marker)))
    with as_user("alice"):
        assert "changed after it was exported" in _refused(
            serving.inspect_model(str(tampered))
        )
    assert not marker.exists()


def test_a_colleagues_private_model_is_refused_but_team_models_load(
    serving, alice_export
):
    _, pkl = alice_export
    with as_user("mallory"):
        assert "belongs to 'alice'" in _refused(serving.inspect_model(pkl))
    team = Path(pkl).with_name("regression-x-spend-v1.pkl")
    team.write_bytes(Path(pkl).read_bytes())
    toolguard.trust(team, "alice")
    with as_user("mallory"):
        assert "error" not in json.loads(serving.inspect_model(str(team))), (
            "registered versions are team assets"
        )


def test_writes_outside_the_allowed_folders_are_refused(
    regression, serving, alice_export
):
    run_id, pkl = alice_export
    with as_user("alice"):
        _refused(
            regression.export_model(run_id, str(DATA.parent / "evil.pkl"), force=True)
        )
        _refused(
            regression.export_model(
                run_id, str(DATA / "artifacts" / "x.csv"), force=True
            )
        )
        _refused(
            serving.predict(
                pkl,
                data_path=_upload("rows"),
                out_path=str(DATA / "artifacts" / "x.reference.csv"),
            )
        )
    with as_user("mallory"):
        _refused(
            regression.export_model(run_id, pkl, force=True)
        )  # alice's run and alice's file


def test_run_ids_cannot_name_other_folders(regression):
    with as_user("mallory"):
        assert "not a run id" in _refused(
            regression.train_model("../../artifacts", "linear_regression")
        )


def test_developer_tools_are_not_agent_tools():
    with _agent("classification") as m, as_user("mallory"):
        assert "developer tool" in _refused(
            m.generate_ci_workflow(".github/workflows/x.yml")
        )
        assert "developer tool" in _refused(m.dvc_track("data"))


def test_other_agents_are_guarded_too(tmp_path):
    outside = tmp_path / "secret.csv"
    _tabular().to_csv(outside, index=False)
    with as_user("mallory"):
        with _agent("drift") as drift:
            _refused(drift.eda(str(outside)))
            _refused(drift.compare_distributions(str(outside), str(outside), "x1"))
        with _agent("visualization") as viz:
            _refused(viz.plot_histogram(str(outside), "x1", "h.png"))
        with _agent("serving") as srv:
            _refused(srv.list_exported_models(str(tmp_path)))
