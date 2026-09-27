"""What must stay true about explanations.

The failure this agent exists to avoid is not a crash — it is a fluent,
plausible, wrong reason given to someone about their own case. So the
checks here are about the claims, not the plumbing.

Run: python -m pytest test_explain.py   (or: python test_explain.py)
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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp_server import (
    ADDITIVITY_TOLERANCE,
    _decision,
    _final_step_name,
    _narrate,
    _positive_index,
    _task_of,
)


class _Pipe:
    classes_ = ["churn", "no_churn"]

    def __init__(self, steps=(("classifier", None),)):
        self.steps = list(steps)
        self.named_steps = dict(steps)


def test_positive_class_is_the_event_not_the_higher_label():
    assert _positive_index({"positive_label": "churn"}, ["churn", "no_churn"]) == 0
    assert _positive_index({"positive_label": 1}, [0, 1]) == 1
    assert _positive_index({}, ["a", "b"]) == 1  # nothing recorded -> legacy convention


def test_task_and_final_step_are_read_from_the_artifact():
    class Clf(_Pipe):
        def predict_proba(self, x): ...

    assert _final_step_name(_Pipe([("encode", 1), ("regressor", 2)])) == "regressor"
    assert _task_of(Clf(), {}) == "classification"
    assert _task_of(_Pipe([("regressor", 1)]), {}) == "regression"


def test_decision_is_measured_against_the_tuned_threshold():
    """The core claim of this agent: 'why was this flagged' means 'why did
    the score cross THIS model's cutoff', not 0.5. A score of 0.55 is a
    flag under the default and not a flag under a tuned 0.63 — same score,
    opposite answer, and only one of them is the model that shipped."""
    tuned = {
        "task": "classification",
        "pipeline": _Pipe(),
        "bundle": {
            "positive_label": "churn",
            "operating_point": {"threshold": 0.63, "chosen_by": "target_recall>=0.9"},
        },
    }
    default = {
        "task": "classification",
        "pipeline": _Pipe(),
        "bundle": {"positive_label": "churn"},
    }

    assert _decision(tuned, 0.55)["decision"] == "no_churn"
    assert _decision(default, 0.55)["decision"] == "churn"

    d = _decision(tuned, 0.71)
    assert d["threshold"] == 0.63
    assert d["margin_over_threshold"] == round(0.71 - 0.63, 6)
    assert d["positive_class"] == "churn"


def test_missing_threshold_says_so_rather_than_pretending():
    d = _decision(
        {
            "task": "classification",
            "pipeline": _Pipe(),
            "bundle": {"positive_label": "churn"},
        },
        0.9,
    )
    assert "no tuned threshold" in d["threshold_source"]


def test_narration_states_the_threshold_and_never_invents_a_feature():
    d = _decision(
        {
            "task": "classification",
            "pipeline": _Pipe(),
            "bundle": {
                "positive_label": "churn",
                "operating_point": {"threshold": 0.63},
            },
        },
        0.71,
    )
    contributions = [
        {
            "feature": "calls_premium",
            "value": 9,
            "contribution": 0.31,
            "direction": "toward",
        },
        {
            "feature": "tenure",
            "value": 48,
            "contribution": -0.08,
            "direction": "away from",
        },
    ]
    text = _narrate(contributions, d, "churn", "classification")
    assert "0.63 threshold" in text
    assert "calls_premium" in text and "tenure" in text
    # built from the numbers, so it cannot drift from the table above it
    for word in text.replace(",", " ").split():
        assert "premium" not in word or "calls_premium" in word


def test_additivity_tolerance_is_tight_enough_to_catch_a_wrong_explanation():
    assert ADDITIVITY_TOLERANCE <= 1e-3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all explanation checks passed")
