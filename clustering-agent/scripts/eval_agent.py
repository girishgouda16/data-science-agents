"""Agent-level evals for the clustering agent: does the LLM keep outcomes and
protected attributes out of the distance, drop identifiers, choose k inside
what the business can act on, and admit when there is no stable structure —
graded from the run's artifacts. Runner and shared checks: core/agent_eval.py.

Costs real LLM calls (CLUSTERING_AGENT_MODEL via LLM_PROVIDER), so it is not
in CI:

    python eval_agent.py                      # all scenarios
    python eval_agent.py outcome_kept_out     # one
Writes nothing into the repo's data/ or mlruns/ (both go to a temp dir).
"""
import os
import sys
import tempfile
from pathlib import Path

WORKDIR = Path(tempfile.mkdtemp(prefix="agent-eval-"))
os.environ["AGENTIC_ML_DATA_DIR"] = str(WORKDIR / "data")  # inherited by the MCP server subprocess
os.environ["MLFLOW_TRACKING_URI"] = f"file:{WORKDIR / 'mlruns'}"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import make_blobs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import agent  # noqa: E402
import mcp_server as m  # noqa: E402
from core.agent_eval import Scenario, main, out_of_model, ran  # noqa: E402

REQUEST = ("Dataset: {path}\nSegment these subscribers. Run the full pipeline including the readiness gates, "
           "then log the run to MLflow.")


def _write(df: pd.DataFrame, d: Path, stem: str) -> Path:
    path = d / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def _subscribers(d: Path, k: int = 4, seed: int = 21) -> Path:
    rng = np.random.default_rng(seed)
    X, truth = make_blobs(2000, centers=k, n_features=3, cluster_std=1.2, random_state=seed)
    return _write(pd.DataFrame({
        "subscriber_id": [f"SUB{i:06d}" for i in range(2000)],
        "calls_per_day": X[:, 0].round(3), "data_share": X[:, 1].round(3), "night_share": X[:, 2].round(3),
        "monthly_spend": rng.lognormal(3 + 0.3 * truth, 0.6).round(2),
        "churned": (rng.random(2000) < 0.05 + 0.1 * truth).astype(int),
        "age": rng.integers(18, 80, 2000),
    }), d, "subscribers")


def _structureless(d: Path) -> Path:
    rng = np.random.default_rng(22)
    return _write(pd.DataFrame(rng.normal(0, 1, (2000, 6)), columns=[f"usage_{i}" for i in range(6)]),
                  d, "structureless")


def _mixed_plans(d: Path, seed: int = 23) -> Path:
    """Segments carried by plan, roaming and call volume; three noise usage
    columns that standardised kmeans weighs as much as the plan."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, 2000)
    plans = np.array(["prepaid", "postpaid", "business"])
    return _write(pd.DataFrame({
        "subscriber_id": [f"SUB{i:06d}" for i in range(2000)],
        "plan": np.where(rng.random(2000) < 0.85, plans[truth], rng.choice(plans, 2000)),
        "roaming": np.where(rng.random(2000) < np.array([0.05, 0.2, 0.85])[truth], "yes", "no"),
        "calls_per_day": rng.normal(np.array([10, 18, 26])[truth], 6).round(2),
        **{f"usage_{i}": rng.normal(0, 1, 2000).round(3) for i in range(3)},
    }), d, "mixed_plans")


def _monthly(d: Path, seed: int = 24) -> Path:
    """Six monthly snapshots of 400 subscribers in three usage segments; a
    fifth of the heavy users slide to the light segment from April."""
    rng = np.random.default_rng(seed)
    home = rng.integers(0, 3, 400)
    sliders = (home == 2) & (rng.random(400) < 0.2)
    centres = np.array([[2, 1], [6, 5], [12, 9]])
    rows = []
    for i, month in enumerate(pd.period_range("2026-01", periods=6, freq="M").astype(str)):
        seg = np.where(sliders & (i >= 3), 0, home)
        xy = centres[seg] + rng.normal(0, 0.8, (400, 2))
        rows += [{"subscriber_id": f"SUB{j:05d}", "month": month, "data_gb": round(xy[j, 0], 2),
                  "voice_hours": round(xy[j, 1], 2)} for j in range(400)]
    return _write(pd.DataFrame(rows), d, "monthly_usage")


def algorithm_is(name):
    def check(meta, questions):
        return f"ships {name}", meta.get("algorithm") == name, f"algorithm={meta.get('algorithm')}"
    return check


def tracked_migration(meta, questions):
    mig = meta.get("migration") or {}
    return "measures migration between months", bool(mig), f"stay_rate={mig.get('stay_rate')}"


def profiles(*columns):
    def check(meta, questions):
        kept = set(meta.get("profile_columns") or [])
        missing = [c for c in columns if c not in kept]
        return (f"keeps {list(columns)} out of the distance as profile columns", not missing,
                f"profile_columns={sorted(kept)}")
    return check


def k_in_range(low, high):
    def check(meta, questions):
        k = (meta.get("algorithm_params") or {}).get("n_clusters")
        return f"chooses k within the actionable {low}-{high}", k is not None and low <= k <= high, f"k={k}"
    return check


def stable(meta, questions):
    ari = ((meta.get("training_metrics") or {}).get("stability") or {}).get("stability_ari_mean")
    return "ships a stable segmentation", ari is not None and ari >= 0.7, f"stability ARI {ari}"


def admits_no_structure(meta, questions):
    status = {g: c.get("status") for g, c in (m.compute_readiness(meta["_run_id"])["checks"]).items()} \
        if meta.get("training_metrics") else {}
    failed = status.get("holdout_stability") == "fail" or status.get("structure_found") == "fail"
    return "lets the gates report there is no stable structure", failed, f"gates={status}"


SCENARIOS = [
    Scenario("outcome_kept_out", None, _subscribers,
             REQUEST + " We want segments for retention treatments; churned is our outcome.",
             [(r"segment|treat|how many|k\b|number", "Retention can run 3 to 5 treatments."),
              (r"churn|outcome|profile", "churned is the outcome — don't build segments on it."),
              (r"age|protected|demograph", "Age must not drive who gets an offer."),
              (r"bar|success|judge|threshold|metric", "No numeric bar — they must be stable and differ on churn.")],
             [profiles("churned", "age"), out_of_model("subscriber_id"), k_in_range(3, 5), stable,
              ran("explain_model", "check_readiness")]),
    Scenario("mixed_plans", None, _mixed_plans,
             REQUEST + " We want segments for tariff offers.",
             [(r"segment|offer|how many|k\b|number", "Marketing can run 3 or 4 offers."),
              (r"plan|lens|define|feature|behaviour|behavior|exclude|profile",
               "Build the segments on plan, roaming and calling together — the plan is part of who they are."),
              (r"acknowledge|unstable|stabil|anyway|force|accept",
               "No — don't ship an unstable segmentation; find one that is stable."),
              (r"bar|success|judge|threshold|metric", "No numeric bar — they must be stable.")],
             [out_of_model("subscriber_id"), algorithm_is("kmedoids"), stable, k_in_range(3, 4)]),
    Scenario("monthly_panel", None, _monthly,
             REQUEST + " These are monthly snapshots — we also need to know how subscribers move between "
                       "segments month to month.",
             [(r"segment|how many|k\b|number", "3 to 5 segments."),
              (r"bar|success|judge|threshold|metric", "No bar — stable segments.")],
             [profiles("subscriber_id", "month"), stable, tracked_migration, ran("segment_migration")]),
    Scenario("no_structure", None, _structureless, REQUEST,
             [(r"segment|how many|k\b|number", "Anything from 3 to 8 segments is fine."),
              (r"bar|success|judge|threshold|metric", "No bar.")],
             [admits_no_structure, ran("propose_k")]),
]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], SCENARIOS, agent, m.RUNS_DIR, WORKDIR, REPO))
