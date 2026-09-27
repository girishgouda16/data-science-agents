"""Integration checks for drift-agent's mcp_server/.

Proves: a genuinely shifted numeric column comes back severe, a stable one
comes back none, a categorical column with a shifted mix (plus a brand-new
category) comes back at least moderate, a near-constant reference column
degrades gracefully instead of crashing, the overall verdict/recommendation
follow the worst column, and every error path returns an error instead of
raising.

Run: python test_mcp_server/
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

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import mcp_server as m

ARTIFACTS_DIR = Path(__file__).parent / "test_artifacts"


def _make_reference(path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 500
    pd.DataFrame(
        {
            "num_shifted": rng.normal(0.0, 1.0, n),
            "num_stable": rng.normal(5.0, 2.0, n),
            "num_const": np.full(n, 7.0),
            "cat_shifted": rng.choice(["a", "b", "c"], n, p=[0.8, 0.15, 0.05]),
            "cat_stable": rng.choice(["x", "y"], n, p=[0.5, 0.5]),
        }
    ).to_csv(path, index=False)


def _make_current(path: Path) -> None:
    rng = np.random.default_rng(23)
    n = 400
    pd.DataFrame(
        {
            "num_shifted": rng.normal(6.0, 1.0, n),  # far from reference -> severe
            "num_stable": rng.normal(5.0, 2.0, n),  # same distribution -> none
            "num_const": np.full(n, 7.0),  # unchanged -> none
            "cat_shifted": rng.choice(
                ["a", "b", "c", "d"], n, p=[0.1, 0.2, 0.3, 0.4]
            ),  # new category + shifted mix
            "cat_stable": rng.choice(["x", "y"], n, p=[0.5, 0.5]),
        }
    ).to_csv(path, index=False)


def main():
    if ARTIFACTS_DIR.exists():
        shutil.rmtree(ARTIFACTS_DIR)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    reference_path = ARTIFACTS_DIR / "reference.csv"
    current_path = ARTIFACTS_DIR / "current.csv"
    _make_reference(reference_path)
    _make_current(current_path)

    eda = json.loads(m.eda(str(reference_path)))
    assert eda["shape"] == [500, 5]

    report = json.loads(m.detect_drift(str(reference_path), str(current_path)))
    assert "error" not in report
    cols = report["columns"]

    assert cols["num_shifted"]["severity"] == "severe", cols["num_shifted"]
    assert cols["num_stable"]["severity"] == "none", cols["num_stable"]
    assert cols["num_const"]["severity"] == "none", cols["num_const"]
    assert cols["cat_shifted"]["severity"] in ("moderate", "severe"), cols[
        "cat_shifted"
    ]
    assert cols["cat_shifted"]["new_categories"] == ["d"]
    assert cols["cat_stable"]["severity"] == "none", cols["cat_stable"]

    assert "num_shifted" in report["drifted_columns"]
    assert "num_stable" not in report["drifted_columns"]
    assert report["overall_severity"] == "severe"
    assert "retrain now" in report["recommendation"]

    # ── columns= subset ─────────────────────────────────────────────────
    subset = json.loads(
        m.detect_drift(
            str(reference_path), str(current_path), columns="num_stable,cat_stable"
        )
    )
    assert set(subset["columns"]) == {"num_stable", "cat_stable"}
    assert subset["overall_severity"] == "none"

    unknown_col = json.loads(
        m.detect_drift(str(reference_path), str(current_path), columns="not_a_column")
    )
    assert "error" in unknown_col

    # ── no shared columns at all ────────────────────────────────────────
    renamed_path = ARTIFACTS_DIR / "renamed.csv"
    pd.read_csv(current_path).rename(columns=lambda c: f"other_{c}").to_csv(
        renamed_path, index=False
    )
    no_shared = json.loads(m.detect_drift(str(reference_path), str(renamed_path)))
    assert "error" in no_shared and "share no column names" in no_shared["error"]

    # ── missing / empty file error paths ────────────────────────────────
    missing_ref = json.loads(
        m.detect_drift(str(ARTIFACTS_DIR / "nope.csv"), str(current_path))
    )
    assert "error" in missing_ref and "no reference file" in missing_ref["error"]

    missing_cur = json.loads(
        m.detect_drift(str(reference_path), str(ARTIFACTS_DIR / "nope.csv"))
    )
    assert "error" in missing_cur and "no current file" in missing_cur["error"]

    empty_path = ARTIFACTS_DIR / "empty.csv"
    pd.DataFrame({"a": []}).to_csv(empty_path, index=False)
    empty_result = json.loads(m.detect_drift(str(empty_path), str(current_path)))
    assert "error" in empty_result and "no rows" in empty_result["error"]

    # ── compare_distributions ───────────────────────────────────────────
    zoom = json.loads(
        m.compare_distributions(str(reference_path), str(current_path), "num_shifted")
    )
    assert zoom["severity"] == "severe"
    assert "mean" in zoom["reference_summary"]

    zoom_cat = json.loads(
        m.compare_distributions(str(reference_path), str(current_path), "cat_shifted")
    )
    assert zoom_cat["dtype"] == "categorical"
    assert "a" in zoom_cat["reference_summary"]

    missing_col = json.loads(
        m.compare_distributions(str(reference_path), str(current_path), "not_a_column")
    )
    assert "error" in missing_col

    print("all checks passed")
    shutil.rmtree(ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
