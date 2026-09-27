"""MCP server exposing tabular-data plotting tools (matplotlib/seaborn).
Headless (Agg backend) — every tool saves a PNG under CHARTS_DIR (ignoring
any directory in the out_path the caller passed, keeping only its filename)
and returns both the local path and an http:// URL — agent.py serves
CHARTS_DIR as static files, so the chat UI can render the image inline
instead of just naming a local file nothing can fetch.

Run: python -m mcp_server.server  (cwd=scripts/; agent.py does this)

Shared kernel of the package: imports, constants, run storage, every
helper and the `mcp` instance. Tools live in one module per skill."""

import json

import os

import sys

import shutil

import uuid

import joblib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from pathlib import Path

from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline as SkPipeline

from sklearn.preprocessing import label_binarize

import shap

# Unused directly here, but joblib.load()-ing classification-agent's
# pipeline.pkl (see _load_run below) needs `pipeline_transformers` importable
# in this process — pickle resolves classes by module name, not by copying
# code. See pipeline_transformers.py's own docstring.
from pipeline_transformers import ColumnDropper, ColumnFiller  # noqa: F401

# "tab10" is matplotlib's built-in port of Tableau's own 10-color categorical
# palette — applies to every hue/multi-line plot below via the shared color
# cycle, no per-plot changes needed.
sns.set_theme(
    style="whitegrid",
    palette="tab10",
    rc={
        # Lighter chrome so the data reads first: no top/right box, faint grid.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1",
        "grid.color": "#eef2f6",
        "axes.titlepad": 10,
    },
)

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for core.*
from core import (
    runtime,
)  # noqa: E402 — whose run this is (a user never charts a colleague's)

mcp = FastMCP("visualization-agent")
from core import toolguard  # noqa: E402

toolguard.install(
    mcp, outputs=False
)  # every tool's paths/models checked before it runs (core/toolguard.py)

# Must match agent.py's CHARTS_DIR/PORT — that's what actually serves these
# files over HTTP (StaticFiles mount at /charts).
CHARTS_DIR = Path(__file__).parent.parent.parent / "charts"

CHARTS_DIR.mkdir(exist_ok=True)

PORT = os.environ.get("VISUALIZATION_AGENT_PORT", "9200")

# Where a browser fetches a chart. Default: same-origin /charts, which the
# gateway serves next to the UI — a remote user's browser cannot reach this
# agent's own port. Set CHARTS_PUBLIC_URL for A2A callers on another origin.
BASE_URL = os.environ.get("CHARTS_PUBLIC_URL", "/charts").rstrip("/")

# Shared with classification-agent/scripts/mcp_server/ — same path, same
# run_id. plot_confusion_matrix/plot_roc_curve read the pipeline.pkl that
# agent wrote (train_model, possibly overwritten by tune_hyperparams)
# instead of fitting their own model, so a chart always reflects the
# classification agent's actual current model for that run, tuned or not —
# not a fresh default-params refit that happens to share a model name.
# AGENTIC_ML_DATA_DIR relocates data/ (runs, artifacts, predictions) for every
# agent at once — the test suites set it so they never write into the real
# data/runs that inspect_runs reports to the orchestrator as "recent runs".
RUNS_DIR = (
    Path(
        os.environ.get("AGENTIC_ML_DATA_DIR")
        or Path(__file__).parent.parent.parent.parent / "data"
    )
    / "runs"
)


def _load_run(run_id: str):
    """Returns (pipeline, meta, test_df) for a run_id, or None if the
    classification agent hasn't fitted a model for it yet.

    Every real run_id is a uuid4 (see classification-agent's
    prepare_dataset) — reject anything else before it reaches
    RUNS_DIR / run_id. Without this, a run_id like "../../etc" would escape
    RUNS_DIR entirely and joblib.load() would deserialize whatever pickle
    sits at the resulting path — path traversal into arbitrary
    deserialization, not just an arbitrary read."""
    try:
        uuid.UUID(run_id)
    except (ValueError, AttributeError, TypeError):
        return None
    pipeline_path = RUNS_DIR / run_id / "pipeline.pkl"
    if not pipeline_path.exists():
        return None
    meta = json.loads((RUNS_DIR / run_id / "meta.json").read_text())
    if runtime.access_error(
        meta.get("owner")
    ):  # another user's run reads as absent — checked before unpickling it
        return None
    pipeline = joblib.load(pipeline_path)
    test = pd.read_csv(RUNS_DIR / run_id / "test.csv")
    return pipeline, meta, test


def _run_meta(run_id: str) -> dict | None:
    """meta.json alone, for charts that need no pickle (feature importance,
    forecasts). Same uuid4 guard as _load_run."""
    try:
        uuid.UUID(run_id)
    except (ValueError, AttributeError, TypeError):
        return None
    path = RUNS_DIR / run_id / "meta.json"
    meta = json.loads(path.read_text()) if path.exists() else None
    return None if meta is None or runtime.access_error(meta.get("owner")) else meta


def _fitted(run_id: str, step: str, tool: str, agent: str):
    """(pipeline, meta, test) when run_id holds `agent`'s fitted pipeline —
    recognised by its final step name (regressor / clusterer / detector) —
    else (None, error-json). Stops a classification run_id handed to a
    regression chart from rendering nonsense instead of failing."""
    loaded = _load_run(run_id)
    if loaded is None:
        return None, json.dumps(
            {
                "error": f"no fitted model for run_id '{run_id}' — ask the {agent} agent to train_model first"
            }
        )
    if step not in getattr(loaded[0], "named_steps", {}):
        return None, json.dumps(
            {
                "error": f"run_id '{run_id}' isn't a {agent} run — {tool} only applies to the {agent} agent's runs"
            }
        )
    return loaded, None


# Mirrors classification-agent/scripts/mcp_server/core.py's positive-class
# resolution. Duplicated rather than imported for the same reason
# pipeline_transformers.py is duplicated: each agent stays independently
# deployable, no cross-agent Python imports. The CONTRACT is meta.json's
# "positive_label", written by prepare_dataset — so the two stay in step
# through the artifact, not through shared code.
#
# This matters more here than anywhere else: a chart is the evidence a human
# actually looks at. A report whose text describes churn and whose ROC curve
# describes no-churn is worse than no chart at all, because it is convincing.
NEGATIVE_LABEL_WORDS = frozenset(
    {
        "0",
        "n",
        "no",
        "none",
        "negative",
        "neg",
        "false",
        "f",
        "absent",
        "legit",
        "legitimate",
        "genuine",
        "normal",
        "benign",
        "clean",
        "good",
        "ok",
        "healthy",
        "safe",
        "active",
        "retained",
        "stayed",
        "stay",
        "paid",
        "current",
        "approved",
        "survived",
        "alive",
        "pass",
    }
)


def _looks_negative(label) -> bool:
    text = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return text in NEGATIVE_LABEL_WORDS or text.startswith(("no_", "non_", "not_"))


def _positive_label(meta: dict, classes):
    """The class this run's positive-class metrics refer to. Prefers the
    label prepare_dataset recorded; falls back to the same inference for a
    run created before it was recorded, so an older run_id still plots the
    class its report describes."""
    recorded = meta.get("positive_label")
    if recorded is not None:
        return recorded
    labels = sorted(classes)
    if len(labels) != 2:
        return None
    return (
        labels[0]
        if _looks_negative(labels[1]) and not _looks_negative(labels[0])
        else labels[1]
    )


def _positive_index(meta: dict, classes) -> int:
    """predict_proba column for the positive class (classes_ is sorted)."""
    labels = sorted(classes)
    pos = _positive_label(meta, labels)
    return (
        0 if len(labels) == 2 and pos is not None and str(labels[0]) == str(pos) else 1
    )


def _classifier_mismatch(run_id: str, pipeline, meta: dict, tool: str) -> dict | None:
    """None if this run's pipeline looks like a classifier (has
    predict_proba) with a "target" key in meta.json. run_id is only ever
    validated as *a* uuid4 by _load_run — it could still belong to a
    regression/clustering/anomaly run (same data/runs/<run_id>/ shape,
    different meta contract, e.g. anomaly-agent uses "label_column" not
    "target"). Without this, plot_confusion_matrix/roc/pr/calibration
    either KeyError on meta["target"] or, for a regression run (which DOES
    have a "target" key), silently render a nonsense chart from continuous
    predictions instead of failing."""
    if hasattr(pipeline, "predict_proba") and "target" in meta:
        return None
    return {
        "error": f"run_id '{run_id}' isn't a classification run (model={meta.get('model', meta.get('algorithm', '?'))}) "
        f"— {tool} only applies to a classification agent's run",
    }


def _save(fig, out_path: str, title: str, run_id: str = "", kind: str = "") -> dict:
    """Saves under CHARTS_DIR (only out_path's filename is kept — the
    directory the caller asked for is ignored, since serving only works out
    of one known folder). Returns {out_path, url, markdown} to merge into
    the tool's JSON result — "markdown" is a ready-made ![]() line so the
    caller can relay it verbatim instead of reformatting a bare path.

    Model-eval charts are prefixed with their run_id. CHARTS_DIR is one flat
    folder with no retention sweep, and "roc.png" is what every caller
    naturally asks for — so without the prefix the second run silently
    overwrites the first, and every report.md already linking that URL starts
    showing a different model's curve. A chart that changes under a report
    that cites it is the same class of failure the classification agent's
    evidence-ordering gate exists to catch."""
    filename = Path(out_path).name or "chart.png"
    if run_id:
        filename = f"{run_id}_{filename}"
    dest = CHARTS_DIR / filename
    fig.savefig(dest, bbox_inches="tight", dpi=200)
    plt.close(fig)
    if run_id and kind:
        # The run's own copy, named for the fit it was drawn from: it travels
        # with the run into the report and MLflow, and the classification
        # agent's `evaluation_charts` gate counts it — a refit (new
        # pipeline_version) leaves it behind as stale instead of passing.
        pv = json.loads((RUNS_DIR / run_id / "meta.json").read_text()).get(
            "pipeline_version", 0
        )
        (RUNS_DIR / run_id / "charts").mkdir(exist_ok=True)
        shutil.copyfile(dest, RUNS_DIR / run_id / "charts" / f"pv{pv}_{kind}.png")
    url = f"{BASE_URL}/{filename}"
    return {"out_path": str(dest), "url": url, "markdown": f"![{title}]({url})"}
