"""Agent-level evals: does the LLM make the RIGHT CALLS, not just do the tools
work. Every other suite here drives the tools directly with no model in the
loop; the thing that actually varies — which model, which prompt — is the
agent's judgment, and nothing measured it.

Each scenario hands the real classification agent (real LLM, in-process, the
same LangGraph host the A2A server uses) a dataset and a request, answers its
questions from a script, and then grades the decisions from the run's
ARTIFACTS — never from what the agent said it did.

Costs real LLM calls (CLASSIFICATION_AGENT_MODEL via LLM_PROVIDER) and a few
minutes per scenario, so it is not in CI. Run it whenever the model, a skill
or the system prompt changes:

    python eval_agent.py                 # all scenarios
    python eval_agent.py label_copy      # one
Writes nothing into the repo's data/ or mlruns/ (both go to a temp dir).
"""

import os
import sys
import tempfile
from pathlib import Path

WORKDIR = Path(tempfile.mkdtemp(prefix="agent-eval-"))
os.environ["AGENTIC_ML_DATA_DIR"] = str(
    WORKDIR / "data"
)  # inherited by the MCP server subprocess
os.environ["MLFLOW_TRACKING_URI"] = f"file:{WORKDIR / 'mlruns'}"

import re  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import agent  # noqa: E402
import mcp_server as m  # noqa: E402
from core import agent_eval  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


Scenario = agent_eval.Scenario  # runner and shared checks: core/agent_eval.py


# ── datasets ─────────────────────────────────────────────────────────────────


def _titanic(d: Path) -> Path:
    return REPO / "data" / "titanic.csv"


def _churn_with_label_copy(d: Path) -> Path:
    rng = np.random.default_rng(1)
    n = 1500
    tenure = rng.integers(1, 72, n)
    calls = rng.poisson(2, n)
    churned = (rng.random(n) < 1 / (1 + np.exp(0.06 * tenure - 0.5 * calls))).astype(
        int
    )
    df = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": rng.normal(70, 20, n).round(2),
            "support_calls": calls,
            "contract": rng.choice(["monthly", "annual", "two_year"], n),
            # Filled in by the retention team AFTER the customer left or stayed.
            "retention_outcome": np.where(churned == 1, "lost", "kept"),
            "churned": churned,
        }
    )
    return _write(df, d, "churn")


def _balanced(d: Path) -> Path:
    rng = np.random.default_rng(2)
    n = 1200
    x = rng.normal(0, 1, (n, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 1, n) > 0).astype(int)
    return _write(
        pd.DataFrame(
            {
                "usage_gb": x[:, 0],
                "roaming_min": x[:, 1],
                "age_months": x[:, 2],
                "upgraded": y,
            }
        ),
        d,
        "upgrade",
    )


def _repeat_entities(d: Path) -> Path:
    rng = np.random.default_rng(3)
    customers = 400
    risk = rng.normal(0, 1, customers)
    rows = []
    for cid in range(customers):
        defaulted = int(risk[cid] + rng.normal(0, 0.7) > 0.8)
        for month in range(4):  # the same customer, four statements — shares a fate
            rows.append(
                {
                    "customer_id": f"C{cid:04d}",
                    "month": month,
                    "balance": 1000 + 400 * risk[cid] + rng.normal(0, 150),
                    "late_payments": max(0, int(risk[cid] + rng.normal(0, 1))),
                    "defaulted": defaulted,
                }
            )
    return _write(pd.DataFrame(rows), d, "credit")


def _imbalanced(d: Path) -> Path:
    rng = np.random.default_rng(4)
    n = 6000
    fraud = rng.random(n) < 0.03
    df = pd.DataFrame(
        {
            "calls_24h": rng.poisson(np.where(fraud, 30, 8)),
            "avg_call_sec": rng.normal(np.where(fraud, 6, 90), 20).clip(1),
            "distinct_callees": rng.poisson(np.where(fraud, 25, 6)),
            "prepaid": rng.choice([0, 1], n),
            "is_wangiri": fraud.astype(int),
        }
    )
    return _write(df, d, "wangiri")


def _caller_days(d: Path) -> Path:
    """Caller x day. A Wangiri day is a burst against the caller's OWN
    baseline: absolute volume overlaps heavily with busy legitimate callers,
    so the signal is only visible to a feature built from the caller's
    history — a lag, a trailing mean, vs_roll_mean."""
    rng = np.random.default_rng(5)
    rows = []
    for c in range(150):
        base = rng.gamma(2, 20)
        for day in range(30):
            burst = rng.random() < 0.06
            rows.append({"caller": f"K{c:03d}", "day": day,
                         "total_calls": int(rng.poisson(base * (6 if burst else 1)) + 1),
                         "distinct_callees": int(rng.poisson(base * (5 if burst else 0.6)) + 1),
                         "is_wangiri": int(burst)})
    return _write(pd.DataFrame(rows), d, "caller_days")


def built_history_features(meta, questions):
    formulas = [v.get("formula", "") for v in (meta.get("engineered_features") or {}).values()]
    history = [f for f in formulas if "earlier rows only" in f or " per " in f]
    return ("builds features from each caller's own history", bool(history),
            f"{len(history)} history feature(s)" if history else f"engineered: {formulas or 'none'}")


def split_forward(meta, questions):
    return ("splits forward in time", meta.get("split_type") == "temporal", f"split_type={meta.get('split_type')}")


def _write(df: pd.DataFrame, d: Path, stem: str) -> Path:
    path = d / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


# ── checks: each reads the run's artifacts, returns (name, passed, detail) ───


def _tools(meta):
    return [e["tool"] for e in meta.get("execution_log") or [] if e.get("ok")]


def drops(*columns):
    """Out of the model: dropped, or a split key (kept for grouped/temporal
    validation, excluded by the pipeline itself)."""
    def check(meta, questions):
        keys = {meta.get("group_column"), meta.get("time_column")}
        kept = [c for c in columns if c not in (meta.get("dropped_columns") or {}) and c not in keys]
        return (
            f"drops {list(columns)}",
            not kept,
            f"still a feature: {kept}" if kept else "dropped",
        )

    return check


def asked(pattern, what):
    def check(meta, questions):
        hit = next((q for q in questions if re.search(pattern, q, re.I)), None)
        return (
            f"asks {what}",
            hit is not None,
            (hit or f"never asked; questions: {questions}")[:160],
        )

    return check


def bar_is(metric, threshold):
    def check(meta, questions):
        got = (meta.get("business_understanding") or {}).get("success_target") or {}
        ok = got.get("metric") == metric and got.get("threshold") == threshold
        return (
            f"records bar {metric} >= {threshold}",
            ok,
            f"recorded {got or 'nothing'}",
        )

    return check


def no_bar(meta, questions):
    got = (meta.get("business_understanding") or {}).get("success_target")
    return (
        "invents no bar",
        not got,
        f"recorded {got}" if got else "none recorded, as the user said",
    )


def threshold_off_test(meta, questions):
    op = meta.get("operating_point") or {}
    basis = (op.get("selected_on") or {}).get("validation", "")
    return (
        "threshold chosen off the test fold",
        "out-of-fold" in basis or "forward" in basis,
        basis or "no threshold tuned",
    )


def ran(*tools):
    def check(meta, questions):
        missing = [t for t in tools if t not in _tools(meta)]
        return (
            f"runs {', '.join(tools)}",
            not missing,
            f"never ran {missing}" if missing else "ran",
        )

    return check


def leak_not_waived(column):
    def check(meta, questions):
        flagged = column in (
            (meta.get("leakage") or {}).get("near_perfect_predictors") or []
        )
        dropped = column in (meta.get("dropped_columns") or {})
        waived = column in (meta.get("acknowledged_identifiers") or {})
        # Removed (whether the screen or the agent's own reading caught it), or
        # flagged and left for the user — never waived by the agent itself.
        ok = dropped or (flagged and not waived)
        return (
            f"keeps `{column}` out, never self-waived",
            ok,
            f"flagged={flagged} dropped={dropped} self-waived={waived}",
        )

    return check


def grouped_by(column):
    def check(meta, questions):
        ok = meta.get("group_column") == column and meta.get("split_type") in (
            "grouped",
            "temporal",
        )
        return (
            f"splits by {column}",
            ok,
            f"split_type={meta.get('split_type')} group={meta.get('group_column')}",
        )

    return check


def cv_metric_is(metric):
    def check(meta, questions):
        got = (meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}).get(
            "cv_metric"
        )
        return f"selects on {metric}", got == metric, f"cv_metric={got}"

    return check


def no_smote(meta, questions):
    return (
        "no SMOTE above the 2% bar",
        not meta.get("use_smote"),
        f"use_smote={meta.get('use_smote')}",
    )


REQUEST = (
    "Dataset: {path}\nTrain a classification model with target column {target}. Run the full pipeline "
    "including the readiness gates, then log the run to MLflow."
)

SCENARIOS = [
    Scenario(
        "entity_time_history",
        "is_wangiri",
        _caller_days,
        REQUEST,
        [
            (r"time|forward|day|temporal", "Yes — split forward on day, and keep callers grouped."),
            (r"caller|group|entity", "Yes, caller is the entity."),
            (r"label|provenance|produced|investigat", "Investigated by the fraud team, confirmed within 2 days."),
            (r"bar|success|judge|threshold|metric", "PR-AUC >= 0.60"),
        ],
        [split_forward, ran("apply_entity_features"), built_history_features, drops("caller")],
    ),
    Scenario(
        "titanic_identifiers",
        "Survived",
        _titanic,
        REQUEST,
        [
            (r"split|group|ticket", "Random per-row stratified split."),
            (r"bar|success|judge|threshold|metric", "ROC-AUC >= 0.80"),
        ],
        [
            drops("PassengerId", "Name", "Ticket"),
            asked(r"bar|success|judge", "for the success bar"),
            bar_is("roc_auc", 0.8),
            threshold_off_test,
            ran("check_readiness", "log_run_to_mlflow"),
        ],
    ),
    Scenario(
        "label_copy",
        "churned",
        _churn_with_label_copy,
        REQUEST,
        [
            (
                r"retention_outcome|leak|acknowledge|keep",
                "I don't know how retention_outcome is filled in.",
            ),
            (r"bar|success|judge|threshold|metric", "PR-AUC >= 0.50"),
        ],
        [leak_not_waived("retention_outcome"), ran("detect_data_leakage")],
    ),
    Scenario(
        "no_bar_given",
        "upgraded",
        _balanced,
        REQUEST,
        [
            (
                r"bar|success|judge|threshold|metric",
                "I don't have a bar — just report the metrics.",
            )
        ],
        [no_bar, ran("check_readiness")],
    ),
    Scenario(
        "repeat_entities",
        "defaulted",
        _repeat_entities,
        REQUEST,
        [
            (r"customer_id|group|split|leak", "Split by customer_id."),
            (r"bar|success|judge|threshold|metric", "Recall on the defaulters >= 0.70"),
        ],
        [
            asked(r"customer_id|group", "about the repeated customers"),
            grouped_by("customer_id"),
            drops("customer_id"),
        ],
    ),
    Scenario(
        "imbalanced_fraud",
        "is_wangiri",
        _imbalanced,
        REQUEST,
        [(r"bar|success|judge|threshold|metric", "PR-AUC >= 0.30")],
        [cv_metric_is("average_precision"), no_smote, threshold_off_test],
    ),
]


# ── runner ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    sys.exit(agent_eval.main(sys.argv[1:], SCENARIOS, agent, m.RUNS_DIR, WORKDIR, REPO))
