"""Agent-level evals for the regression agent: does the LLM make the right
calls — the target's modelling space, the split, target maturity, the
baseline that matters, the loss the business decision implies, intervals when
a range is needed — graded from the run's artifacts. Runner and shared checks:
core/agent_eval.py.

Costs real LLM calls (REGRESSION_AGENT_MODEL via LLM_PROVIDER), so it is not
in CI. Run it whenever the model, a skill or the system prompt changes:

    python eval_agent.py                   # all scenarios
    python eval_agent.py capacity_p90      # one
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
from core.agent_eval import Scenario, main, no_bar, out_of_model, ran  # noqa: E402

REQUEST = (
    "Dataset: {path}\nTrain a regression model for target column {target}. Run the full pipeline including "
    "the readiness gates, then log the run to MLflow."
)


def _write(df: pd.DataFrame, d: Path, stem: str) -> Path:
    path = d / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


# ── datasets ─────────────────────────────────────────────────────────────────

def _skewed_arpu(d: Path) -> Path:
    rng = np.random.default_rng(11)
    n = 1500
    tenure, data_gb = rng.integers(1, 72, n), rng.gamma(2, 3, n)
    plan = rng.choice(["basic", "plus", "max"], n, p=[0.6, 0.3, 0.1])
    log_arpu = 2.5 + 0.01 * tenure + 0.3 * np.log1p(data_gb) + np.where(plan == "max", 1.0, 0) + rng.normal(0, 0.5, n)
    return _write(pd.DataFrame({"tenure_months": tenure, "data_gb": data_gb.round(2), "plan": plan,
                                "arpu": np.expm1(log_arpu).round(2)}), d, "arpu")


def _subscriber_months(d: Path) -> Path:
    """Spend settles two months after the month; the newest two months are
    still accruing (recorded at ~60%) — a trap for a model trained on them."""
    rng = np.random.default_rng(12)
    rows = []
    for s in range(150):
        level = rng.lognormal(3, 0.6)
        for month in range(18):
            usage = level * rng.lognormal(0, 0.25)
            spend = usage * 1.3 + rng.gamma(1, 2)
            rows.append({"msisdn": f"S{s:03d}", "month": month, "usage": round(usage, 2),
                         "spend_next": round(spend * (0.6 if month >= 16 else 1.0), 2)})
    return _write(pd.DataFrame(rows), d, "subscriber_months")


def _plain(d: Path) -> Path:
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, (800, 3))
    return _write(pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2],
                                "delivery_hours": 24 + 5 * x[:, 0] - 3 * x[:, 1] + rng.normal(0, 2, 800)}),
                  d, "delivery")


def _target_copy(d: Path) -> Path:
    rng = np.random.default_rng(14)
    n = 1000
    size, rooms = rng.gamma(4, 20, n), rng.integers(1, 6, n)
    price = 1000 * size + 5000 * rooms + rng.normal(0, 10000, n)
    return _write(pd.DataFrame({"size_m2": size.round(1), "rooms": rooms,
                                # recorded after the sale closed: the price itself, plus fees
                                "final_invoice": (price * 1.02).round(0), "price": price.round(0)}), d, "sales")


def _busy_hour(d: Path) -> Path:
    rng = np.random.default_rng(15)
    rows = []
    for cell in range(40):
        base = rng.gamma(3, 30)
        for hour in range(24 * 14):
            load = base * (1 + 0.8 * np.sin(2 * np.pi * (hour % 24) / 24)) * rng.lognormal(0, 0.35)
            rows.append({"cell_id": f"C{cell:02d}", "hour_of_day": hour % 24, "weekday": (hour // 24) % 7,
                         "base_load": round(base, 1), "peak_erlangs": round(load, 2)})
    return _write(pd.DataFrame(rows), d, "cell_load")


# ── checks specific to regression ───────────────────────────────────────────

def skewed_target_handled(meta, questions):
    space = meta.get("target_transform") or meta.get("objective")
    return ("models the skewed target in log space or with a poisson loss", space in ("log1p", "poisson"),
            f"target_transform={meta.get('target_transform')} objective={meta.get('objective')}")


def split_forward(meta, questions):
    return "splits forward in time", meta.get("split_type") == "temporal", f"split_type={meta.get('split_type')}"


def excludes_unsettled(meta, questions):
    return ("excludes the months still accruing", (meta.get("immature_rows_dropped") or 0) > 0,
            f"immature_after={meta.get('immature_after')} dropped={meta.get('immature_rows_dropped')}")


def persistence_baseline(meta, questions):
    kind = (meta.get("baseline_comparison") or {}).get("kind")
    return "judged against the persistence baseline", kind == "persistence", f"strongest baseline: {kind}"


def leak_dropped(column):
    def check(meta, questions):
        dropped = column in (meta.get("dropped_columns") or [])
        return f"drops the target copy `{column}`", dropped, "dropped" if dropped else "still a feature"
    return check


def quantile_from_costs(meta, questions):
    q = meta.get("quantile")
    ok = meta.get("objective") == "quantile" and q is not None and 0.75 <= q <= 0.9
    return ("prices the 5:1 cost asymmetry as a ~0.83 quantile", ok,
            f"objective={meta.get('objective')} quantile={q}")


SCENARIOS = [
    Scenario("skewed_arpu", "arpu", _skewed_arpu,
             REQUEST + " We use the predictions to set next year's revenue budget per segment.",
             [(r"bar|success|judge|threshold|metric", "MAE under 8 per subscriber-month."),
              (r"transform|log|skew", "Use your recommendation.")],
             [skewed_target_handled, ran("train_baseline", "check_readiness")]),
    Scenario("subscriber_months", "spend_next", _subscriber_months, REQUEST,
             [(r"time|forward|month|temporal|group|msisdn|entity", "Yes — split forward on month and keep "
               "subscribers grouped by msisdn."),
              (r"settle|matur|final|accru|lag|complete", "Spend is final two months after the month closes."),
              (r"bar|success|judge|threshold|metric", "MAE under 4.")],
             [split_forward, excludes_unsettled, persistence_baseline, out_of_model("msisdn")]),
    Scenario("no_bar_given", "delivery_hours", _plain, REQUEST,
             [(r"bar|success|judge|threshold|metric", "I don't have a bar — just report the metrics.")],
             [no_bar, ran("check_readiness")]),
    Scenario("target_copy", "price", _target_copy, REQUEST,
             [(r"final_invoice|leak|acknowledge|keep", "final_invoice is issued after the sale closes."),
              (r"bar|success|judge|threshold|metric", "R^2 above 0.7.")],
             [leak_dropped("final_invoice"), ran("detect_data_leakage")]),
    Scenario("capacity_p90", "peak_erlangs", _busy_hour,
             REQUEST + " We dimension cell capacity from this: under-provisioning a cell costs us five times "
                       "what over-provisioning does.",
             [(r"bar|success|judge|threshold|metric", "No fixed bar — size it sensibly."),
              (r"quantile|percentile|objective|loss|asymmetr", "Use your recommendation."),
              (r"group|cell|entity|split", "Split by cell_id.")],
             [quantile_from_costs, ran("compare_models")]),
    Scenario("range_needed", "delivery_hours", _plain,
             REQUEST + " Operations will quote customers a delivery window, so we need a range, not a point.",
             [(r"bar|success|judge|threshold|metric", "MAE under 3 hours."),
              (r"coverage|interval|range|confidence", "90% of deliveries should land inside the window.")],
             [ran("calibrate_intervals")]),
]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], SCENARIOS, agent, m.RUNS_DIR, WORKDIR, REPO))
