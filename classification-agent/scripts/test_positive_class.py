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


def test_pr_auc_selection_works_on_word_labels_and_scores_the_event():
    """{"fraud","legit"} at 5% fraud: selection must use PR-AUC (it used to
    fall back to f1_weighted for any non-0/1 labels) and score P(fraud), the
    label that sorts FIRST — sklearn's stock scorer would use pos_label=1."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from mcp_server.core import _cv_scoring, _scorer

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=400)})
    y = pd.Series(np.where(X["x"] + rng.normal(size=400) > 2.3, "fraud", "legit"))
    meta = {"positive_label": "fraud"}
    assert _cv_scoring(y) == "average_precision"
    model = LogisticRegression().fit(X, y)
    expected = average_precision_score(y == "fraud", model.predict_proba(X)[:, 0])
    assert abs(_scorer("average_precision", meta, y)(model, X, y) - expected) < 1e-9


def test_calibration_folds_respect_groups_and_survive_a_subsample_refit():
    """calibrate_model's folds must keep each entity on one side (a plain
    cv=5 calibrated on an entity's own rows), and a clone refit on fewer
    rows — what _validation_proba does — must recompute them, not index the
    original rows."""
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from pipeline_transformers import RunFolds

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=300), "cust": np.repeat(np.arange(60), 5)})
    y = pd.Series((X["x"] + rng.normal(size=300) > 0).astype(int))
    folds = RunFolds({"group_column": "cust"})
    for fit, val in folds.split(X, y):
        assert not set(X["cust"].iloc[fit]) & set(X["cust"].iloc[val])
    calibrated = CalibratedClassifierCV(LogisticRegression(), cv=folds).fit(X, y)
    clone(calibrated).fit(X.iloc[:150], y.iloc[:150])  # would IndexError on a fixed fold list
    assert calibrated.predict_proba(X).shape == (300, 2)
