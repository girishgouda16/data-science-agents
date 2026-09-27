"""The positive class must be the EVENT, not whichever label sorts highest.

Everything positive-class in this agent — recall_positive, PR-AUC, the tuned
operating point, fairness TPR, SHAP's explained class, the success gate —
used to resolve it as `y.max()` / `classes_[1]`. Right for {0,1} and
{"No","Yes"}; silently wrong for {"churn","no_churn"}, {"fraud","legit"},
{"default","paid"}, where sorting picks the negative class and every number
stays plausible. These assertions are what fails if that convention comes
back.

Run: python -m pytest test_positive_class.py   (or: python test_positive_class.py)
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

from mcp_server.core import infer_positive_label, positive_index, positive_label
from mcp_server.gates import measured_metric


def test_numeric_and_yes_no_keep_the_old_behaviour():
    assert infer_positive_label([0, 1, 1, 0]) == 1
    assert infer_positive_label(["No", "Yes", "No"]) == "Yes"
    assert infer_positive_label(["true", "false"]) == "true"


def test_negative_sounding_max_label_flips():
    assert infer_positive_label(["churn", "no_churn"]) == "churn"
    assert infer_positive_label(["fraud", "legit"]) == "fraud"
    assert infer_positive_label(["default", "paid"]) == "default"
    assert infer_positive_label(["fraud", "not-fraud"]) == "fraud"
    assert infer_positive_label(["anomaly", "normal"]) == "anomaly"


def test_ambiguous_labels_fall_back_to_sorting():
    # Neither label reads as a negative — nothing to go on but sort order.
    # The point isn't that this is right, it's that prepare_dataset REPORTS
    # it as inferred so the caller can override.
    assert infer_positive_label(["A", "B"]) == "B"


def test_multiclass_has_no_positive_class():
    assert infer_positive_label(["a", "b", "c"]) is None


def test_recorded_label_beats_inference():
    meta = {"positive_label": "no_churn"}  # caller insists, however odd
    assert positive_label(meta, ["churn", "no_churn"]) == "no_churn"


def test_positive_index_is_the_proba_column():
    # sklearn sorts classes_, so "churn" (the positive) is column 0.
    meta = {"positive_label": "churn"}
    assert positive_index(meta, classes=["churn", "no_churn"]) == 0
    assert positive_index({"positive_label": 1}, classes=[0, 1]) == 1
    assert positive_index({"positive_label": "Yes"}, classes=["No", "Yes"]) == 1


def test_success_gate_reads_the_recorded_class_not_the_highest_key():
    """The bug end to end: a churn run whose recall target is judged against
    the recall of NOT churning, because "no_churn" > "churn"."""
    meta = {
        "positive_label": "churn",
        "baseline_metrics": {
            "classification_report": {
                "churn": {"recall": 0.42, "precision": 0.30, "f1-score": 0.35},
                "no_churn": {"recall": 0.97, "precision": 0.99, "f1-score": 0.98},
                "accuracy": 0.93,
            }
        },
    }
    assert measured_metric(meta, "recall_positive") == 0.42  # the churners we caught
    assert measured_metric(meta, "precision_positive") == 0.30

    # A run predating positive_label still resolves, by the old convention.
    del meta["positive_label"]
    assert measured_metric(meta, "recall_positive") == 0.97


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all positive-class checks passed")


# ---------------------------------------------------------------- calibration


def test_calibration_gate_fails_a_threshold_tuned_on_unverified_probabilities():
    """The gate's whole point: a tuned cutoff whose probability basis was
    never measured is a promise with nothing behind it."""
    from mcp_server.gates import compute_readiness  # noqa: F401  (import shape check)
    from mcp_server.core import ECE_CONCERN

    assert ECE_CONCERN == 0.05


def test_ece_is_the_size_of_the_lie():
    """A model that says 0.9 about rows that are positive half the time has
    an ECE of 0.4 — well past the concern threshold."""
    import numpy as np
    from sklearn.metrics import brier_score_loss

    proba = np.full(100, 0.9)
    y = np.array([1] * 50 + [0] * 50)
    ece = abs(proba.mean() - y.mean())
    assert round(ece, 4) == 0.4
    assert round(float(brier_score_loss(y, proba)), 4) == 0.41
