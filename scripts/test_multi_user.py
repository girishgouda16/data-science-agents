"""Many data scientists on one deployment: the tool/registry side of it.

  * a run belongs to whoever created it — a colleague's tool calls on it are
    refused before the tool runs, and the visualization agent can't read it;
  * registry names can't collide by accident: a team project only when a person
    named it for the chat, otherwise a personal <user>.<dataset> namespace — two
    people who both upload bills.csv never share a champion;
  * champion/challenger only move off, or onto, a version the caller owns
    (admins excepted), and `approved_by` is the authenticated user, not text;
  * concurrent registrations never collide on a version number;
  * turns beyond an agent's cap queue; task state survives a restart.

The user is what core/runtime.py reads in a tool process: AGENTIC_ML_USER.
No LLM, no network. Run: python -m pytest -q scripts/test_multi_user.py
"""

import asyncio
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mlops_all_agents import (
    ROOT,
    _agent,
    _csv,
    _tabular,
)  # noqa: E402 — sets temp data dir + MLflow store

from core import registry, runtime, toolguard  # noqa: E402


@contextlib.contextmanager
def as_user(name: str, admins: str = "", project: str = ""):
    """What the gateway + runtime give a tool process for one chat."""
    os.environ.update(
        AGENTIC_ML_USER=name, AGENTIC_ML_ADMINS=admins, AGENTIC_ML_PROJECT=project
    )
    try:
        yield
    finally:
        for key in ("AGENTIC_ML_USER", "AGENTIC_ML_ADMINS", "AGENTIC_ML_PROJECT"):
            os.environ.pop(key)


def _upload(dataset: str) -> str:
    """Where the gateway puts the current user's upload — the only folder
    (besides shared datasets) their tools may read."""
    folder = toolguard.data_dir() / "uploads" / toolguard.user_folder(runtime.user())
    folder.mkdir(parents=True, exist_ok=True)
    return _csv(folder, dataset, _tabular())


def _trained(m, dataset: str) -> str:
    run_id = json.loads(m.prepare_dataset(_upload(dataset), "spend"))["run_id"]
    context = json.loads(
        m.record_business_context(
            run_id,
            domain="telecom",
            business_objective="price plans",
            target_definition="monthly spend",
        )
    )
    assert "error" not in context, context
    m.detect_data_leakage(run_id)
    assert "error" not in json.loads(m.train_model(run_id, "linear_regression"))
    return run_id


def _version(m, run_id: str) -> tuple[str, int]:
    logged = json.loads(m.log_run_to_mlflow(run_id))
    return logged["name"], logged["registered"]["version"]


def test_a_run_belongs_to_its_creator():
    with _agent("regression") as m:
        with as_user("alice"):
            run_id = json.loads(m.prepare_dataset(_upload("bills"), "spend"))["run_id"]
        assert m._load_meta(run_id)["owner"] == "alice"

        with as_user("bob"):
            refused = json.loads(m.train_model(run_id, "linear_regression"))
        assert "belongs to 'alice'" in refused["error"], refused
        assert m._load_meta(run_id).get("model") is None, (
            "the refused tool must not have run"
        )

        with as_user("bob", admins="bob"):
            assert "error" not in json.loads(
                m.train_model(run_id, "linear_regression")
            ), "an admin may"
        log = m._load_meta(run_id)["execution_log"]
        assert log[-1]["user"] == "bob" and m._load_meta(run_id)["owner"] == "alice", (
            "ownership never transfers"
        )

    with _agent("visualization") as viz:
        with as_user("bob"):
            assert viz._run_meta(run_id) is None, "a colleague's run reads as absent"
        with as_user("alice"):
            assert viz._run_meta(run_id)["owner"] == "alice"


def test_registry_names_never_collide_by_accident():
    with _agent("regression") as m:
        with as_user("alice"):
            alice = _version(m, _trained(m, "bills"))[0]
        with as_user("bob"):
            bob = _version(m, _trained(m, "bills"))[0]
        assert (alice, bob) == (
            "regression-alice.bills-spend",
            "regression-bob.bills-spend",
        ), (
            "same file name, same target, different people: separate models, separate champions"
        )
        with as_user("alice", project="retention"):
            team = _version(m, _trained(m, "bills_2024_10"))[0]
        assert team == "regression-retention-spend", (
            "a project the person named is the shared namespace"
        )
        assert {alice, bob, team} <= set(registry.find_names("regression", "spend"))


def test_aliases_move_only_for_the_owner_or_an_admin():
    with _agent("regression") as m:
        with as_user("alice", project="shared"):
            name, v_alice = _version(m, _trained(m, "shared_a"))
        with as_user("bob", project="shared"):
            _, v_bob = _version(m, _trained(m, "shared_b"))
            refused = json.loads(m.promote_model(v_alice, "challenger", name=name))
            assert "belongs to 'alice'" in refused["error"], refused

        with as_user("alice"):
            promoted = json.loads(
                m.promote_model(v_alice, "challenger", name=name, approved_by="mallory")
            )
            assert "error" not in promoted, promoted
        tags = json.loads(m.list_model_versions(name=name))["versions"]
        tags = next(row["version_tags"] for row in tags if row["version"] == v_alice)
        assert tags["owner"] == "alice" and tags["challenger.approved_by"] == "alice", (
            "approver is who asked, not typed text"
        )

        with as_user("bob"):
            assert (
                "belongs to 'alice'"
                in json.loads(m.promote_model(v_bob, "challenger", name=name))["error"]
            ), "moving challenger off alice's version is alice's call"
            assert (
                "belongs to 'alice'"
                in json.loads(m.demote_model("challenger", name=name))["error"]
            )
        with as_user("lead", admins="lead"):
            assert "error" not in json.loads(
                m.promote_model(v_bob, "challenger", name=name)
            ), "a team lead may"


_REGISTER = r"""
import os, sys, tempfile
sys.path.insert(0, sys.argv[1])
import mlflow
from core import registry
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
with mlflow.start_run() as run:
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "weights.txt"), "w").write("x")
        mlflow.log_artifacts(tmp, "model")
print(registry.register(run.info.run_id, "race-model")["version"])
"""


def test_concurrent_registrations_get_distinct_versions(tmp_path):
    # MLFLOW_EXPERIMENT_*: set by earlier in-process mlflow.set_experiment calls, and naming an
    # experiment this fresh store doesn't have.
    env = {
        **{
            k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_EXPERIMENT")
        },
        "MLFLOW_TRACKING_URI": f"file:{tmp_path / 'mlruns'}",
    }
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _REGISTER, str(ROOT)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(6)
    ]
    outputs = [p.communicate(timeout=180) for p in procs]
    versions = sorted(
        out.strip().splitlines()[-1] if out.strip() else f"crashed: {err[-400:]}"
        for out, err in outputs
    )
    assert versions == ["1", "2", "3", "4", "5", "6"], versions


def test_turns_beyond_the_cap_wait():
    os.environ["CAPTEST_MAX_CONCURRENT_TURNS"] = "2"
    running, peak = [0], [0]

    async def turn():
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        await asyncio.sleep(0.05)
        running[0] -= 1

    async def six_users():
        await asyncio.gather(*(runtime.limited("captest", turn()) for _ in range(6)))

    asyncio.run(six_users())
    assert peak[0] == 2, peak


_READ_TASK = r"""
import asyncio, sys
sys.path.insert(0, sys.argv[1])
from core import runtime
task = asyncio.run(runtime.task_store("durability").get("t1"))
print(task.status.state.value if task else "missing")
"""


def test_task_state_survives_a_restart():
    """A question the agent asked before a restart can still be answered after it."""
    from a2a.types import Task, TaskState, TaskStatus

    store = runtime.task_store("durability")
    assert type(store).__name__ == "DatabaseTaskStore", (
        "the SQL extra is missing — tasks would die with the process"
    )
    asyncio.run(
        store.save(
            Task(
                id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.input_required),
            )
        )
    )
    restarted = subprocess.run(
        [sys.executable, "-c", _READ_TASK, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert restarted.stdout.strip().splitlines()[-1] == "input-required", (
        restarted.stdout + restarted.stderr
    )
