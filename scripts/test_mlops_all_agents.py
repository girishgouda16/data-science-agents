"""EDA-to-MLOps, the senior-data-scientist way, for every non-classification agent.

For regression, clustering, anomaly and forecasting: train a small model,
log it to MLflow (a registry VERSION), promote it to challenger, refuse
champion without the user's confirmation and reason, promote it to champion
(which exports it and links data/artifacts/<name>-champion.pkl), serve
predictions from that link through the serving agent, and demote it (the
link goes, the versioned export stays). Forecasting additionally: the served
champion must forecast from the END of the data, not from the train/test cut.

No LLM, no network: the MCP tool functions are called directly, against a
temp data dir and a temp MLflow store.

Run:  python -m pytest -q scripts/test_mlops_all_agents.py
"""

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["AGENTIC_ML_DATA_DIR"] = tempfile.mkdtemp(prefix="agentic-ml-mlops-")
os.environ["MLFLOW_TRACKING_URI"] = (
    f"file:{tempfile.mkdtemp(prefix='agentic-ml-mlruns-')}"
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import registry  # noqa: E402


@contextlib.contextmanager
def _agent(agent: str):
    """An agent's mcp_server with ITS pipeline_transformers importable for the
    whole block — models are pickled and unpickled by that module name, and
    each agent ships its own copy (see test_swarm_e2e._load)."""
    agent_dir = str(ROOT / f"{agent}-agent" / "scripts")
    package = Path(agent_dir) / "mcp_server"
    sys.modules.pop("pipeline_transformers", None)
    # Another test file in the same pytest process may have loaded this agent
    # under the same name; its stale submodules would shadow ours.
    for stale in [
        k for k in sys.modules if k == f"{agent}_ms" or k.startswith(f"{agent}_ms.")
    ]:
        del sys.modules[stale]
    sys.path.insert(0, agent_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            f"{agent}_ms",
            package / "__init__.py",
            submodule_search_locations=[str(package)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{agent}_ms"] = (
            module  # before exec: its relative imports resolve through it
        )
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(agent_dir)
        sys.modules.pop("pipeline_transformers", None)


def _csv(tmp: Path, name: str, df: pd.DataFrame) -> str:
    path = tmp / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _tabular(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(5, 2, n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "channel": rng.choice(["a", "b", "c"], n),
            "spend": (3 * x1 + x2 + rng.normal(0, 0.5, n)).round(3),
        }
    )


def _train(agent: str, m, tmp: Path) -> tuple[str, str, pd.DataFrame]:
    """(run_id, scoring input path, full source frame) for a run taken through
    the senior workflow: business context and a success bar FIRST, the leakage
    screen before fitting, a baseline where the domain has one, train, explain,
    then the readiness gate and the report."""
    rng = np.random.default_rng(0)
    if agent == "regression":
        df = _tabular()
        run_id = json.loads(m.prepare_dataset(_csv(tmp, "reg", df), "spend"))["run_id"]
        bar = dict(success_metric="r2", success_threshold=0.5)
    elif agent == "clustering":
        centers = np.array([[0, 0], [8, 8], [0, 8]])
        pts = np.vstack(
            [c + rng.normal(0, 0.8, (100, 2)) for c in centers]
        )  # three real segments
        df = pd.DataFrame(pts, columns=["recency", "frequency"])
        run_id = json.loads(m.prepare_dataset(_csv(tmp, "clu", df)))["run_id"]
        bar = dict(success_metric="silhouette_score", success_threshold=0.5)
    elif agent == "anomaly":
        df = _tabular().drop(columns=["spend"])
        run_id = json.loads(m.prepare_dataset(_csv(tmp, "ano", df)))["run_id"]
        bar = {}
    else:
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        t = np.arange(200)
        df = pd.DataFrame(
            {
                "date": dates,
                "sales": 100
                + 0.8 * t
                + 10 * np.sin(t * 2 * np.pi / 7)
                + rng.normal(0, 1, 200),
            }
        )
        run_id = json.loads(m.prepare_dataset(_csv(tmp, "fc", df), "date", "sales"))[
            "run_id"
        ]
        bar = dict(success_metric="rmse", success_threshold=50)
    context = json.loads(
        m.record_business_context(
            run_id,
            business_objective=f"test {agent} objective",
            target_definition="what one row / the output means",
            domain="test",
            **bar,
        )
    )
    assert "error" not in context, context
    m.detect_data_leakage(run_id)
    if agent == "regression":
        m.train_baseline(run_id)
        m.train_model(run_id, "random_forest")
        m.check_residuals(run_id)
        m.check_model_stability(run_id)
    elif agent == "clustering":
        m.train_model(run_id, "kmeans", n_clusters=3)
    elif agent == "anomaly":
        m.propose_contamination(run_id)
        m.train_model(run_id, "iforest", contamination=0.05)
    else:
        m.train_baseline(run_id)
        m.train_model(run_id, "exponential_smoothing")
    m.explain_model(run_id)
    readiness = json.loads(m.check_readiness(run_id))
    assert readiness["overall_status"] != "blocked", readiness["failed_gates"]
    assert readiness["overall_status"] == "ready", {
        k: v
        for k, v in readiness["checks"].items()
        if v["status"] in ("fail", "not_run")
    }
    report = m.generate_report(run_id)
    verdict = {
        "ready": "Proceed to human review",
        "incomplete": "Incomplete",
        "blocked": "Blocked",
    }[readiness["overall_status"]]
    assert (
        report.startswith(f"# {agent.title()} report")
        and verdict in report.split("## Business")[0]
    ), report[:400]
    new_data = (
        ""
        if agent == "forecasting"
        else _csv(
            tmp,
            f"{agent}_new",
            df.drop(columns=[c for c in ("spend",) if c in df]).head(20),
        )
    )
    return run_id, new_data, df


@pytest.fixture(scope="module")
def serving():
    with _agent("serving") as module:
        return module


@pytest.fixture(scope="module")
def visualization(tmp_path_factory):
    with _agent("visualization") as module:
        module.core.CHARTS_DIR = tmp_path_factory.mktemp("charts")
        return module


EVAL_CHART = {
    "regression": "plot_regression_diagnostics",
    "clustering": "plot_clusters",
    "anomaly": "plot_anomaly_scores",
    "forecasting": "plot_forecast",
}


@pytest.mark.parametrize(
    "agent", ["regression", "clustering", "anomaly", "forecasting"]
)
def test_eda_to_champion_to_serving(agent, tmp_path, serving, visualization):
    with _agent(agent) as m:
        _eda_to_champion_to_serving(agent, m, tmp_path, serving, visualization)


def _eda_to_champion_to_serving(agent, m, tmp_path, serving, visualization):
    run_id, new_data, source = _train(agent, m, tmp_path)

    # The visualization agent reads this agent's run artifacts as they are —
    # a renamed meta key or file breaks the chart, not just the report.
    charts = [EVAL_CHART[agent]] + (
        ["plot_feature_importance"] if agent in ("regression", "anomaly") else []
    )
    for tool in charts:
        chart = json.loads(
            getattr(visualization, tool)(run_id=run_id, out_path=f"{tool}.png")
        )
        assert "error" not in chart and Path(chart["out_path"]).exists(), chart
    assert len(
        list(
            (registry.ARTIFACTS_DIR.parent / "runs" / run_id / "charts").glob("pv*.png")
        )
    ) == len(charts)
    if agent != "regression":
        assert "error" in json.loads(
            visualization.plot_regression_diagnostics(run_id, "wrong.png")
        ), "wrong run type must refuse"

    logged = json.loads(m.log_run_to_mlflow(run_id))
    assert "version" in logged["registered"], logged
    name, version = logged["name"], logged["registered"]["version"]
    assert (
        name.startswith(f"{agent}-")
        and m._load_meta(run_id)["registry"]["version"] == version
    ), "run must be pinned"
    assert logged["experiment"] == name, (
        "the default experiment is the registered model's name"
    )
    row = json.loads(m.list_model_versions(name=name))["versions"][0]
    assert row["agent_run_id"] == run_id and row["metrics"], row

    assert "error" not in json.loads(m.promote_model(version, "challenger", name=name))
    assert (
        "confirmation"
        in json.loads(m.promote_model(version, "champion", name=name))["error"]
    )
    assert (
        "reason"
        in json.loads(m.promote_model(version, "champion", name=name, confirmed=True))[
            "error"
        ]
    )
    champion = json.loads(
        m.promote_model(
            version,
            "champion",
            run_id=run_id,
            confirmed=True,
            reason="test",
            force=True,
        )
    )
    assert "error" not in champion, champion
    served = Path(champion["serving_path"])
    assert served == registry.champion_path(name) and served.exists(), champion
    tags = json.loads(m.list_model_versions(name=name))["versions"][0]["version_tags"]
    assert tags["champion.reason"] == "test", tags

    if agent == "forecasting":
        # The champion must forecast from the end of the data, not from the
        # train/test cut 20% of history ago.
        assert champion["export"]["history_ends_at"].startswith(
            str(source["date"].max().date())
        ), champion["export"]
        # One process, two copies of ForecastModel: the pickle resolves to the
        # forecasting agent's class here. In production serving runs in its own
        # process and unpickles with its own (identical) class.
        serving.serving.ForecastModel = (
            m.ForecastModel
        )  # patched where predict() reads it
        out = json.loads(serving.predict(str(served), horizon=7))
        assert "error" not in out, out
        assert pd.to_datetime(out["preview"][0]["date"]) > source["date"].max(), out[
            "preview"
        ][0]
        assert (
            out["model"]["run_id"] == run_id and out["readiness_at_export"] == "ready"
        ), "every forecast must name its model"
    else:
        assert served.with_suffix(".monitoring.json").exists(), (
            "drift needs the profile beside the champion"
        )
        out = json.loads(serving.predict(str(served), data_path=new_data))
        assert "error" not in out, out
        assert out["model"]["run_id"] == run_id, (
            "every prediction must name the model that made it"
        )
        assert out["readiness_at_export"] == "ready", (
            "the gate verdict travels with every prediction"
        )

    demoted = json.loads(m.demote_model("champion", name=name, confirmed=True))
    assert demoted["was_pointing_at"] == version and not served.exists()
    assert (registry.ARTIFACTS_DIR / f"{name}-v{version}.pkl").exists(), (
        "rollback needs the versioned export"
    )
