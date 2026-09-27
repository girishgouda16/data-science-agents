"""Which readiness verdict gates promotion of a REGISTERED version.

Shipped: v2 was logged `incomplete`, then its baseline gate ran and the live
verdict became `ready` — while the registry kept saying `incomplete`. The
dangerous mirror image is a version logged `ready` whose same model later FAILS
a gate: the snapshot-only promotion gate would have let it through.

Run: python test_registry_readiness.py
"""

import os as _os
import tempfile as _tempfile

# Tests never write into the real data/ (runs, artifacts, predictions) that
# the orchestrator reports as "recent runs"; CI may point this elsewhere.
_os.environ.setdefault(
    "AGENTIC_ML_DATA_DIR", _tempfile.mkdtemp(prefix="agentic-ml-test-")
)
# ...nor register test models into the real MLflow registry (the UI on :5000).
_os.environ.setdefault(
    "MLFLOW_TRACKING_URI", "file:" + _tempfile.mkdtemp(prefix="agentic-ml-test-mlruns-")
)

from mcp_server.mlops import _effective_readiness as eff

# Same pipeline provably still in the run dir -> the live gate wins, both ways.
assert (
    eff("incomplete", "ready", "2", "2") == "ready"
)  # the v2 case: evidence added after logging
assert (
    eff("ready", "blocked", "2", "2") == "blocked"
)  # same model, defect found after logging

# Refit since logging -> live describes a different model; never let a blocked
# verdict from EITHER side slip through.
assert eff("ready", "blocked", "2", "3") == "blocked"
assert eff("blocked", "ready", "2", "3") == "blocked"
assert (
    eff("incomplete", "ready", "2", "3") == "incomplete"
)  # can't claim the new fit's pass for the old artifact

# Logged before pipeline_version was recorded (v1-v5), or run dir swept.
assert eff("ready", "blocked", None, "2") == "blocked"
assert eff("incomplete", None, "2", None) == "incomplete"
assert eff("blocked", None, None, None) == "blocked"

print(
    "OK — effective readiness: live when provably the same fit, otherwise the worse verdict"
)


# ── champion export: only the exact registered pipeline may be exported ──────
import json
import os
import shutil
import tempfile
from pathlib import Path

store = Path(tempfile.mkdtemp(prefix="champion-export-"))
os.environ["MLFLOW_TRACKING_URI"] = f"file:{store / 'mlruns'}"
try:
    import joblib
    import mlflow
    from sklearn.dummy import DummyClassifier
    from mcp_server import mlops

    run_dir = store / "run"
    run_dir.mkdir()
    joblib.dump(DummyClassifier().fit([[0], [1]], [0, 1]), run_dir / "pipeline.pkl")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run() as run:
        mlflow.set_tag("agent_run_id", "agent-run")
        mlflow.log_artifact(str(run_dir / "pipeline.pkl"))
        mlflow.sklearn.log_model(joblib.load(run_dir / "pipeline.pkl"), "model")
    version = mlops.registry.register(run.info.run_id, "clf-t")["version"]

    exported = []
    mlops.export_model = lambda rid, out, force=False: (
        exported.append((rid, out, force)) or json.dumps({"out_path": out})
    )
    mlops.registry.ARTIFACTS_DIR = store / "artifacts"

    mlops._run_dir = lambda rid: run_dir
    ok = mlops._export_version("clf-t", version, force=True)
    assert exported == [
        ("agent-run", str(store / "artifacts" / f"clf-t-v{version}.pkl"), True)
    ], exported

    joblib.dump(
        DummyClassifier(strategy="uniform").fit([[0], [1]], [0, 1]),
        run_dir / "pipeline.pkl",
    )
    assert "refit" in mlops._export_version("clf-t", version, force=False)["error"]

    mlops._run_dir = lambda rid: store / "swept"
    assert "gone" in mlops._export_version("clf-t", version, force=False)["error"]
    assert len(exported) == 1, "a refused version must never reach export_model"

    # A champion with no recorded reason is a production change nobody can audit.
    refused = json.loads(
        mlops.promote_model(version, "champion", name="clf-t", confirmed=True)
    )
    assert "reason" in refused["error"], refused
finally:
    shutil.rmtree(store, ignore_errors=True)

print("OK — champion export: registered pipeline exported; refit or swept runs refused")
