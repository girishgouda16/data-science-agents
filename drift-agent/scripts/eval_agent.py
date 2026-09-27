"""Agent-level evals for the drift agent: does the LLM lead with a broken
feed (a column gone null, a type change) instead of calling it drift, check
the table as a whole when per-column checks see nothing, find WHEN a change
started, and never claim the model is still accurate — graded from the
transcript (the agent is stateless: no run artifacts). Runner:
core/agent_eval.py.

Costs real LLM calls (DRIFT_AGENT_MODEL via LLM_PROVIDER), so it is not in
CI:

    python eval_agent.py                   # all scenarios
    python eval_agent.py broken_feed       # one
"""
import os
import sys
import tempfile
from pathlib import Path

WORKDIR = Path(tempfile.mkdtemp(prefix="agent-eval-"))
os.environ["AGENTIC_ML_DATA_DIR"] = str(WORKDIR / "data")  # inherited by the MCP server subprocess

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import agent  # noqa: E402
from core.agent_eval import Scenario, called, main, never_says, says  # noqa: E402

MODEL_IS_FINE = r"model (is|remains) (still )?(accurate|fine|good|valid|reliable)|safe to keep using the model"


def _pair(d: Path, stem: str, ref: pd.DataFrame, cur: pd.DataFrame) -> Path:
    ref.to_csv(d / f"{stem}_reference.csv", index=False)
    cur.to_csv(d / f"{stem}_current.csv", index=False)
    return d / f"{stem}_current.csv"


def _broken_feed(d: Path) -> Path:
    rng = np.random.default_rng(51)
    ref = pd.DataFrame({"data_mb": rng.lognormal(5, 1, 4000), "arpu": rng.normal(20, 5, 4000).round(2),
                        "calls": rng.poisson(6, 4000)})
    cur = pd.DataFrame({"data_mb": rng.lognormal(5, 1, 4000),
                        "arpu": pd.Series(rng.normal(20, 5, 4000).round(2)).astype(str).str.replace(".", ",", regex=False),
                        "calls": rng.poisson(6, 4000)})
    cur.loc[cur.sample(frac=0.45, random_state=1).index, "data_mb"] = np.nan
    return _pair(d, "feed", ref, cur)


def _joint(d: Path) -> Path:
    rng = np.random.default_rng(52)
    x = rng.normal(0, 1, 6000)
    ref = pd.DataFrame({"voice_min": x[:3000], "data_gb": 0.9 * x[:3000] + rng.normal(0, 0.44, 3000)})
    cur = pd.DataFrame({"voice_min": x[3000:], "data_gb": -0.9 * x[3000:] + rng.normal(0, 0.44, 3000)})
    return _pair(d, "joint", ref, cur)


def _step(d: Path) -> Path:
    rng = np.random.default_rng(53)
    ref = pd.DataFrame({"latency_ms": rng.normal(40, 5, 5000), "drops": rng.poisson(1, 5000)})
    days = pd.date_range("2026-03-02", periods=56, freq="D")
    cur = pd.concat([pd.DataFrame({"day": day, "latency_ms": rng.normal(40 + (12 if day >= days[28] else 0), 5, 300),
                                   "drops": rng.poisson(1, 300)}) for day in days])
    return _pair(d, "step", ref, cur)


WHERE = ("Current (production) data: {path}. The reference (training) data is the same path with _current "
         "replaced by _reference. ")


SCENARIOS = [
    Scenario("broken_feed", None, _broken_feed, WHERE + "Has production data drifted from training?",
             [(r"key|column|model use|profile", "The model uses all three columns."),
              (r"label|target", "No label in these files.")],
             [called("detect_drift"), says(r"null|missing|empty", "a column went missing"),
              says(r"text|type|string|format", "a column changed type"), never_says(MODEL_IS_FINE, "the model is fine")],
             stateless=True),
    Scenario("joint_shift", None, _joint,
             WHERE + "Per-column checks look fine but the model's precision fell. Has the data changed?",
             [(r"key|column|model use|profile", "The model uses both columns.")],
             [called("detect_multivariate_drift"), says(r"together|relationship|joint|correlat|combination",
                                                        "the columns changed how they move together"),
              never_says(MODEL_IS_FINE, "the model is fine")],
             stateless=True),
    Scenario("when_it_started", None, _step,
             WHERE + "Column day is the date. Latency complaints started some weeks ago — when did the data "
             "change?",
             [(r"freq|week|day|period|granular", "Weekly is fine.")],
             [called("drift_over_time"), says(r"2026-03-30|30 mar|march 30|week of (the )?30", "the week it started")],
             stateless=True),
]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], SCENARIOS, agent, WORKDIR / "runs", WORKDIR, REPO))
