"""Agent-level evals for the anomaly agent: does the LLM set the alert share
from what the team can review, keep a field written after the case was
decided out of the detector, drop identifiers, pick a detector by evidence,
and hand over an explained review queue when there are no labels — graded
from the run's artifacts. Runner and shared checks: core/agent_eval.py.

Costs real LLM calls (ANOMALY_AGENT_MODEL via LLM_PROVIDER), so it is not
in CI:

    python eval_agent.py                      # all scenarios
    python eval_agent.py review_capacity      # one
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import agent  # noqa: E402
import mcp_server as m  # noqa: E402
from core.agent_eval import Scenario, asked, main, out_of_model, ran  # noqa: E402

REQUEST = ("Dataset: {path}\nFind the anomalous subscriber lines. Run the full pipeline including the readiness "
           "gates, then log the run to MLflow.")


def _lines(d: Path, n: int = 3000, seed: int = 31, status_leak: bool = False, labelled: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    fraud = (rng.random(n) < 0.01).astype(int)
    df = pd.DataFrame({
        "subscriber_id": [f"SUB{i:06d}" for i in range(n)],
        "minutes": rng.lognormal(3, 1, n).round(1),
        "data_mb": rng.lognormal(5, 1.2, n).round(1),
        "premium_calls": np.where(fraud == 1, rng.poisson(40, n), rng.poisson(0.2, n)),
        "intl_share": np.where(fraud == 1, rng.beta(8, 2, n), rng.beta(1, 20, n)).round(3),
        "plan": np.where(rng.random(n) < 0.02, "satellite", rng.choice(["prepaid", "postpaid"], n)),
    })
    if status_leak:  # filled in by the fraud team AFTER investigating
        df["case_status"] = np.where(fraud == 1, "closed_fraud", np.where(rng.random(n) < 0.03, "closed_ok", "none"))
    if labelled:
        df["is_fraud"] = fraud
    path = d / f"lines_{seed}_{int(status_leak)}_{int(labelled)}.csv"
    df.to_csv(path, index=False)
    return path


def contamination_is(value, tol=0.002):
    def check(meta, questions):
        got = meta.get("contamination")
        return (f"sets the alert share from review capacity ({value})", got is not None and abs(got - value) <= tol,
                f"contamination={got}")
    return check


def queue_precision_at_least(bar):
    def check(meta, questions):
        got = (meta.get("metrics") or {}).get("precision_at_budget")
        return f"ships a queue with precision >= {bar}", got is not None and got >= bar, f"precision_at_budget={got}"
    return check


def stable(meta, questions):
    got = ((meta.get("metrics") or {}).get("stability") or {}).get("top_alert_jaccard_mean")
    return "ships a stable alert list", got is not None and got >= 0.5, f"top-alert Jaccard {got}"


def reviewed(meta, questions):
    queue = meta.get("review_queue") or {}
    return "hands over an explained review queue", bool(queue.get("rows")), f"review_queue={queue}"


SCENARIOS = [
    Scenario("review_capacity", "is_fraud", _lines,
             REQUEST + " is_fraud marks lines our fraud team confirmed.",
             [(r"review|capacity|how many|alerts|volume|per day", "The team can review 30 lines a day; about "
                                                                  "3,000 lines are scored a day."),
              (r"label|is_fraud|confirm|investigat", "Every is_fraud=1 was investigated and confirmed."),
              (r"bar|success|judge|threshold|metric", "At least half of what we review should be fraud.")],
             [contamination_is(0.01), ran("propose_contamination", "compare_detectors", "explain_model"),
              queue_precision_at_least(0.5), out_of_model("subscriber_id")]),
    Scenario("status_leak", "is_fraud", lambda d: _lines(d, seed=32, status_leak=True),
             REQUEST + " is_fraud is the investigated outcome.",
             [(r"review|capacity|how many|alerts|volume|per day", "30 a day out of 3,000."),
              (r"case_status|status|after|investigat", "case_status is filled in by the fraud team when a case "
                                                       "is closed."),
              (r"bar|success|judge|threshold|metric", "No numeric bar.")],
             [out_of_model("case_status", "subscriber_id"), ran("detect_data_leakage")]),
    Scenario("unlabelled_review", None, lambda d: _lines(d, seed=33, labelled=False), REQUEST,
             [(r"review|capacity|how many|alerts|volume|per day", "20 lines a day out of 3,000."),
              (r"label|known|confirm", "We have no labels."),
              (r"bar|success|judge|threshold|metric", "No bar — we will review what you flag.")],
             [asked(r"review|capacity|how many|alerts", "how many alerts the team can review"),
              out_of_model("subscriber_id"), stable, reviewed]),
]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], SCENARIOS, agent, m.RUNS_DIR, WORKDIR, REPO))
