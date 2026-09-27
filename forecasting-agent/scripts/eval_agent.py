"""Agent-level evals for the forecasting agent: does the LLM evaluate at the
business horizon, ask how several rows per timestamp combine, choose the
season on evidence, fill gaps as a decision, and beat the seasonal naive —
graded from the run's artifacts. Runner and shared checks: core/agent_eval.py.

Costs real LLM calls (FORECASTING_AGENT_MODEL via LLM_PROVIDER), so it is
not in CI:

    python eval_agent.py                      # all scenarios
    python eval_agent.py region_capacity      # one
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
from core.agent_eval import Scenario, main, ran  # noqa: E402

REQUEST = ("Dataset: {path}\n{target} is the quantity to forecast. Run the full pipeline including the "
           "readiness gates, then log the run to MLflow.")


def _region(d: Path, seed: int = 41) -> Path:
    """Hourly traffic of three cells in one region, eight weeks, one row per
    cell per hour: several rows per timestamp by design."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-05", periods=8 * 168, freq="h")
    base = 50 + 40 * np.clip(np.sin((ts.hour.to_numpy() - 6) / 24 * 2 * np.pi), -0.8, None)
    weekly = np.where(ts.dayofweek.to_numpy() >= 5, 0.6, 1.0)
    rows = pd.concat([pd.DataFrame({"hour": ts, "cell": cell, "gb": (base * weekly * k + rng.normal(0, 3, len(ts)))
                                    .clip(min=0).round(2)}) for cell, k in (("A", 1.0), ("B", 0.7), ("C", 1.4))])
    path = d / "region_traffic.csv"
    rows.to_csv(path, index=False)
    return path


def _calls_with_outage(d: Path, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-10-01", periods=180, freq="D")
    calls = (400 + 80 * (days.dayofweek.to_numpy() < 5) + rng.normal(0, 15, len(days))).round()
    df = pd.DataFrame({"day": days, "calls": calls})
    df = df[~df["day"].isin(days[100:103])]  # three days missing from the feed
    path = d / "contact_centre.csv"
    df.to_csv(path, index=False)
    return path


def _cells(d: Path, seed: int = 43) -> Path:
    """Daily traffic of 25 cells, one weekly shape, very different sizes, a
    few missing reports, one cell decommissioned mid-way."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-05", periods=150, freq="D")
    shape = np.where(dates.dayofweek.to_numpy() >= 5, 0.6, 1.0)
    frames = []
    for c in range(25):
        y = float(rng.lognormal(3.5, 1.0)) * shape * rng.normal(1, 0.08, len(dates))
        frame = pd.DataFrame({"day": dates, "cell_id": f"CELL{c:03d}", "gb": y.clip(min=0).round(3)})
        frames.append(frame.iloc[:90] if c == 0 else frame.drop(frame.index[[33, 34]]) if c % 7 == 3 else frame)
    path = d / "cell_traffic.csv"
    pd.concat(frames).to_csv(path, index=False)
    return path


def meta_is(key, expected, what):
    def check(meta, questions):
        got = meta.get(key)
        return what, got == expected, f"{key}={got}"
    return check


def beats_naive(meta, questions):
    """The baseline_beaten gate: pooled over the backtest origins and the
    held-out period — or the naive seasonal itself shipped."""
    gate = m.compute_readiness(meta["_run_id"])["checks"]["baseline_beaten"]
    return "ships a model that beats the seasonal naive (or the naive)", gate["status"] in ("pass", "not_applicable"), \
        gate["evidence"][:160]


def target_filled(meta, questions):
    how = (meta.get("imputation") or {}).get(meta.get("target"))
    return "fills the missing periods of the target as a decision", how is not None, f"target fill={how}"


SCENARIOS = [
    Scenario("region_capacity", "gb", _region,
             REQUEST + " We plan capacity for the whole region and need next week, hour by hour.",
             [(r"combine|aggregate|sum|cell|series|duplicate", "Sum the three cells — we plan the region."),
              (r"horizon|how far|ahead|next", "One week ahead, hourly: 168 hours."),
              (r"band|percentile|upper|p90|interval", "We size on the upper band."),
              (r"bar|success|judge|threshold|metric", "WAPE under 15%.")],
             [meta_is("aggregate", "sum", "sums the cells (the request asks for the region)"),
              meta_is("horizon", 168, "evaluates at the one-week horizon"),
              meta_is("seasonal_periods", 168, "uses the weekly season the data shows"),
              ran("train_baseline", "compare_models"), beats_naive]),
    Scenario("every_cell", "gb", _cells,
             REQUEST + " We need each cell's daily traffic for the next two weeks — every cell, not the total.",
             [(r"combine|aggregate|sum|cell|series|each|every|one model", "Forecast every cell separately, all "
                                                                           "of them in one run."),
              (r"horizon|how far|ahead|next", "14 days."),
              (r"gap|missing|fill|interpolat", "Reports were lost; traffic happened — interpolate."),
              (r"bar|success|judge|threshold|metric", "No numeric bar.")],
             [meta_is("panel", True, "forecasts every cell in one run"), meta_is("horizon", 14, "at the 14-day horizon"),
              target_filled, ran("train_baseline"), beats_naive]),
    Scenario("contact_centre_gaps", "calls", _calls_with_outage,
             REQUEST + " Forecast daily contact-centre calls for the next 4 weeks for staffing.",
             [(r"horizon|how far|ahead|next", "28 days."),
              (r"gap|missing|outage|fill|interpolat", "The feed was down those days; calls did happen — interpolate."),
              (r"bar|success|judge|threshold|metric", "No numeric bar.")],
             [meta_is("horizon", 28, "evaluates at the 28-day horizon"), target_filled, beats_naive]),
]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], SCENARIOS, agent, m.RUNS_DIR, WORKDIR, REPO))
