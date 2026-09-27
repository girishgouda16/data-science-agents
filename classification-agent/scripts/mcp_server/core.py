"""Shared kernel: the `mcp` FastMCP instance every other module registers
its tools onto, plus the helpers/constants those tools share (run
storage, the pipeline builder, evaluation, the overfitting gate). No
@mcp.tool() functions live here — this module owns no skill of its own,
it's what eda/data_cleaning/modeling/diagnostics/feat_engineering/
feat_selection/reporting/mlops are all built on top of, so a change to how
a pipeline is assembled or a run is evaluated only has to happen once.

Entry point: server.py (`python -m mcp_server.server`), not this file.
"""

import functools
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pipeline_transformers import ColumnDropper, ColumnFiller, XGBLabelClassifier
from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline as SkPipeline
from mcp.server.fastmcp import FastMCP
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

mcp = FastMCP("classification-agent")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for core.*
from core import runtime  # noqa: E402 — who is asking; run ownership
from core import validation as shared_validation  # noqa: E402
from core import toolguard  # noqa: E402

toolguard.install(
    mcp
)  # every tool's paths/models checked before it runs (core/toolguard.py)


# --- execution ledger -------------------------------------------------------
# Every @mcp.tool() call appends to its run's meta["execution_log"]. This
# exists because a report is only as trustworthy as the claim that the work
# behind it happened: an agent that skipped a step can *say* it ran, or say a
# tool "wasn't available" (a real incident — the reporting/mlops tools were
# registered and callable in the same process the agent was already calling
# train_model in). The ledger makes both claims falsifiable from the artifact
# alone, with no conversation transcript needed.
#
# Wrapping the decorator instead of editing 39 tool bodies is deliberate: a
# per-tool call would be one more thing to forget on tool #40, and forgetting
# it silently re-opens the exact hole this closes.
_register_tool = mcp.tool


def _tool_with_ledger(*d_args, **d_kwargs):
    register = _register_tool(*d_args, **d_kwargs)

    def decorate(fn):
        @functools.wraps(
            fn
        )  # FastMCP builds its JSON schema from the signature — wraps keeps it intact
        def wrapper(*args, **kwargs):
            # Another user's run is refused before the tool touches it.
            run_id = _resolve_run_id(fn, args, kwargs, None)
            if run_id and (_run_dir(run_id) / "meta.json").exists():
                if denied := runtime.access_error(_load_meta(run_id).get("owner")):
                    return json.dumps({"error": denied})
            started = time.time()
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                _record_step(
                    fn, args, kwargs, None, f"{type(e).__name__}: {e}", started
                )
                raise
            _record_step(fn, args, kwargs, result, None, started)
            return result

        return register(wrapper)

    return decorate


mcp.tool = _tool_with_ledger


def _resolve_run_id(fn, args, kwargs, result) -> str | None:
    """A tool either takes run_id (most of them) or creates one and returns it
    (prepare_dataset) — or neither (eda/predict work off a path)."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        if run_id := bound.arguments.get("run_id"):
            return str(run_id)
    except TypeError:
        pass
    if isinstance(result, str):
        try:
            return json.loads(result).get("run_id")
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


def _record_step(fn, args, kwargs, result, error: str | None, started: float) -> None:
    """Never raises — a broken ledger must not break a working tool call, and
    a tool whose ledger write failed is still a tool that ran."""
    try:
        run_id = _resolve_run_id(fn, args, kwargs, result)
        if not run_id or not (_run_dir(run_id) / "meta.json").exists():
            return
        # A tool can return {"error": ...} as a normal value rather than raising
        # — that is still a step that did not do its job, so the ledger says so.
        if error is None and isinstance(result, str):
            try:
                error = json.loads(result).get("error")
            except (json.JSONDecodeError, AttributeError):
                pass
        meta = runtime.stamp_owner(_load_meta(run_id))
        meta.setdefault("execution_log", []).append(
            {
                "tool": fn.__name__,
                "ok": error is None,
                "error": error,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seconds": round(time.time() - started, 3),
                "user": runtime.user(),
            }
        )
        _save_meta(run_id, meta)
    except Exception:
        pass


# Shared with visualization-agent's mcp_server.py (RUNS_DIR there points at
# the same folder) — a run_id's pipeline.pkl/test.csv is how that agent's
# confusion-matrix/ROC tools evaluate the *actual* current model instead of
# blindly refitting a fresh default-params one. File-based contract, no
# Python import between the two agents.
# AGENTIC_ML_DATA_DIR relocates data/ (runs, artifacts, predictions) for every
# agent at once — the test suites set it so they never write into the real
# data/runs that inspect_runs reports to the orchestrator as "recent runs".
DATA_DIR = Path(
    os.environ.get("AGENTIC_ML_DATA_DIR")
    or Path(__file__).parent.parent.parent.parent / "data"
)
RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ponytail: flat TTL swept opportunistically, not a scheduler — add a "pin"
# flag in meta.json if a run ever needs to outlive this on purpose.
RUN_RETENTION_DAYS = float(os.environ.get("RUN_RETENTION_DAYS", 7))


def _cleanup_old_runs(max_age_days: float = RUN_RETENTION_DAYS) -> None:
    """Runs are disposable working state (train/test CSVs + pipeline.pkl),
    not an archive — export_model is what persists a model long-term. Ages
    off meta.json's last write (not the run's creation time), so a run still
    being worked on never gets swept mid-session. Called from
    prepare_dataset, since every new run is a natural checkpoint to do this
    at — no separate cron/scheduler needed."""
    cutoff = time.time() - max_age_days * 86400
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        mtime = (
            meta_path.stat().st_mtime if meta_path.exists() else run_dir.stat().st_mtime
        )
        if mtime >= cutoff:
            continue
        # A run registered in MLflow is pinned: promoting its version to
        # champion exports from this directory, so sweeping it would leave a
        # registry version nobody can ever serve.
        try:
            if json.loads(meta_path.read_text()).get("registry"):
                continue
        except (OSError, ValueError):
            pass
        shutil.rmtree(run_dir, ignore_errors=True)


MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBLabelClassifier,
    # Native missing-value handling and histogram splits: the fast choice on
    # millions of rows, and it needs no imputation to be correct.
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}

# ponytail: small param grids for a fast first pass, widen if search quality matters.
# Keys are "classifier__"-prefixed — these tune the final Pipeline step, not a bare estimator.
PARAM_GRIDS = {
    "logistic_regression": {"classifier__C": [0.01, 0.1, 1, 10, 100]},
    "random_forest": {
        "classifier__n_estimators": [100, 200, 400],
        "classifier__max_depth": [None, 5, 10, 20],
    },
    "xgboost": {
        "classifier__n_estimators": [100, 200, 400],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.3],
    },
    "hist_gradient_boosting": {
        "classifier__learning_rate": [0.03, 0.1, 0.3],
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__l2_regularization": [0.0, 1.0],
    },
}


def _make_model(model: str, y=None):
    """Every model balances the classes. XGBoost has no class_weight=, so a
    binary run gets scale_pos_weight = count(code 0) / count(code 1), where
    code 1 is the sorted-second label — the class XGBoost treats as
    positive. (max/min, the old formula, up-weighted the MAJORITY whenever
    the sorted-second label was the common one.)"""
    if model == "xgboost":
        kwargs = {"random_state": 42, "eval_metric": "logloss"}
        if y is not None and y.nunique() == 2:
            codes = np.unique(np.asarray(y), return_inverse=True)[1]
            kwargs["scale_pos_weight"] = float((codes == 0).sum() / max((codes == 1).sum(), 1))
        return MODELS[model](**kwargs)
    if model == "logistic_regression":
        return MODELS[model](class_weight="balanced", random_state=42, max_iter=2000)
    return MODELS[model](class_weight="balanced", random_state=42)


def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _load_meta(run_id: str) -> dict:
    path = _run_dir(run_id) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown run_id '{run_id}' — call prepare_dataset first."
        )
    return json.loads(path.read_text())


def _save_meta(run_id: str, meta: dict) -> None:
    (_run_dir(run_id) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _load_split(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = _load_meta(run_id)
    train = pd.read_csv(_run_dir(run_id) / "train.csv")
    test = pd.read_csv(_run_dir(run_id) / "test.csv")
    return train, test, meta


# ponytail: flat row cap, not adaptive to compute budget or model type — a
# stratified 200k-row sample gives essentially the same CV estimate as the
# full fold for a well-behaved model, and keeps compare_models/train_model/
# tune_hyperparams's *search* step interactive regardless of dataset size.
# Found necessary against a real 2.8M-row telco file: train_model's 5-fold
# CV alone took 9 minutes; compare_models (3 models) or tune_hyperparams (8
# trials) at that rate would be 30+ minutes to hours. Raise this if more
# compute budget is available and a bigger sample is worth the wait.
CV_SAMPLE_MAX_ROWS = 200_000


def split_keys(meta: dict) -> list[str]:
    """The run's entity id and time column — kept for validation, never a
    feature. Shared with every agent: core/validation.py."""
    return shared_validation.split_keys(meta)


def _cv_sample(
    X_train: pd.DataFrame, y_train: pd.Series, meta: dict | None = None
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Structure-keeping sample for CV/search only (core/validation.py)."""
    return shared_validation.cv_sample(X_train, y_train, meta, stratify=True, max_rows=CV_SAMPLE_MAX_ROWS)


# ---- Decide on validation, report on test ----------------------------------
#
# The test fold is for the final, reported number — scored, never searched.
# Every choice made by looking at held-out predictions (the decision
# threshold, whether and how to calibrate, which features to drop) is made on
# predictions for TRAINING rows the deciding model did not see. Choosing on the
# test fold and then reporting on it measures how well the choice fits those
# rows, not how well the model generalises: the headline number becomes
# optimistic by construction, and more so the smaller the fold.
VALIDATION_FOLDS = 5


def _validation_folds(
    X: pd.DataFrame, y: pd.Series, meta: dict, n_splits: int = VALIDATION_FOLDS, seed: int = 42
) -> tuple[list, str]:
    """Folds that respect the run's split — used by EVERY cross-validation
    here (compare_models, train_model, tune_hyperparams, stability, feature
    selection, threshold/calibration decisions). Stratified by class; the
    implementation is shared with the other agents in core/validation.py."""
    return shared_validation.validation_folds(X, y, meta, n_splits, seed, stratify=True)


def _validation_proba(
    pipeline, X: pd.DataFrame, y: pd.Series, meta: dict
) -> tuple[pd.Series, np.ndarray, str]:
    """(y, P(positive), scheme) for training rows scored by a copy of the
    CURRENT model (same hyperparameters, same calibration wrapper) that was
    fit without them. This is what decisions are made on.

    ponytail: refits clone(pipeline) once per fold — 5x the fit cost, 25x for
    a CalibratedClassifierCV; capped by CV_SAMPLE_MAX_ROWS like every CV."""
    from sklearn.base import clone

    X, y, _ = _cv_sample(X, y, meta)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    pos = positive_index(meta, y=y)
    folds, scheme = _validation_folds(X, y, meta)
    proba = np.full(len(X), np.nan)
    for fit_idx, val_idx in folds:
        model = clone(pipeline).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, pos]
    scored = ~np.isnan(proba)
    return y[scored], proba[scored], scheme


# ---- Uncertainty ---------------------------------------------------------
BOOTSTRAP_RESAMPLES = 500


def _bootstrap_cis(
    y_true, positive_proba, y_pred, pos_label, n_boot: int = BOOTSTRAP_RESAMPLES
) -> dict:
    """95% percentile-bootstrap intervals (test rows resampled with
    replacement) for every binary metric a success bar can name. A verdict on
    a point estimate calls ROC-AUC 0.842 a miss against 0.85 on a 179-row fold
    whose interval is about +-0.03 wide; the interval is what says whether the
    data can tell the two apart at all.

    Each resample is a vector of row counts, so every metric — the AUCs
    included, via scores grouped by distinct value and sorted once — is an
    O(n) weighted sum, never a re-sort. Exactly sklearn's definitions."""
    y = np.asarray(y_true) == pos_label
    p = np.asarray(y_pred) == pos_label
    n, rng = len(y), np.random.default_rng(42)
    if n > 100_000:
        n_boot = min(
            n_boot, 200
        )  # ponytail: ~3s/100 resamples at 700k rows; the interval is narrow there anyway
    if positive_proba is not None:
        _, inverse = np.unique(
            np.asarray(positive_proba, dtype=float), return_inverse=True
        )
        n_distinct = int(inverse.max()) + 1

    def ratio(a, b):
        return a / b if b > 0 else 0.0

    samples: dict[str, list] = {}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        tp, fp = float(w[y & p].sum()), float(w[~y & p].sum())
        fn, tn = float(w[y & ~p].sum()), float(w[~y & ~p].sum())
        if tp + fn == 0 or tn + fp == 0:
            continue  # a resample with one class has no recall/AUC to speak of
        pp, rp, pn, rn = (
            ratio(tp, tp + fp),
            ratio(tp, tp + fn),
            ratio(tn, tn + fn),
            ratio(tn, tn + fp),
        )
        f1p, f1n = ratio(2 * pp * rp, pp + rp), ratio(2 * pn * rn, pn + rn)
        n_pos, n_neg = tp + fn, tn + fp
        values = {
            "accuracy": (tp + tn) / n,
            "precision_positive": pp,
            "recall_positive": rp,
            "f1_positive": f1p,
            "precision_weighted": (n_pos * pp + n_neg * pn) / n,
            "recall_weighted": (n_pos * rp + n_neg * rn) / n,
            "f1_weighted": (n_pos * f1p + n_neg * f1n) / n,
        }
        if positive_proba is not None:
            pos_w = np.bincount(
                inverse, weights=w * y, minlength=n_distinct
            )  # ascending score
            neg_w = np.bincount(inverse, weights=w * ~y, minlength=n_distinct)
            # ROC-AUC: each positive beats the negatives below it, ties count half.
            values["roc_auc"] = float(
                (pos_w * (np.cumsum(neg_w) - 0.5 * neg_w)).sum() / (n_pos * n_neg)
            )
            # PR-AUC (average precision): precision at each distinct threshold,
            # walking down from the top score, weighted by the recall it adds.
            cum_tp, cum_fp = np.cumsum(pos_w[::-1]), np.cumsum(neg_w[::-1])
            precision = np.divide(
                cum_tp,
                cum_tp + cum_fp,
                out=np.zeros(n_distinct),
                where=(cum_tp + cum_fp) > 0,
            )
            values["pr_auc"] = float((pos_w[::-1] / n_pos * precision).sum())
        for key, value in values.items():
            samples.setdefault(key, []).append(value)
    return {
        k: [
            round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4),
        ]
        for k, v in samples.items()
        if v
    }


# ---- Which class is "positive" -------------------------------------------
#
# Every positive-class number in this codebase — recall_positive, the tuned
# operating point, fairness TPR, PR-AUC, SHAP's explained class, the success
# gate — used to resolve the positive class as `y.max()` / `classes_[1]`,
# i.e. the sorted-max label. That is right for {0,1} and for {"No","Yes"},
# and WRONG, silently, for every target labelled with words:
#
#     {"churn", "no_churn"} -> max is "no_churn"
#     {"fraud", "legit"}    -> max is "legit"
#     {"default", "paid"}   -> max is "paid"
#
# In those runs the threshold was tuned to catch non-churners, fairness
# measured TPR on the majority, and the success gate passed or failed against
# the wrong class — all while every reported number stayed entirely plausible.
# Nothing in the run could detect it.
#
# So the positive class is now recorded on the run (meta["positive_label"]),
# inferred once at prepare_dataset and overridable by the caller, and read
# back through positive_label()/positive_index() everywhere. sklearn keeps
# classes_ sorted, so positive_index() is the proba COLUMN for that label.
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
    """True when a label NAMES THE ABSENCE of the event. Prefix forms
    ("no_churn", "non-fraud", "not_default") are the ones that actually sort
    above their positive counterpart, so they matter most."""
    s = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return s in NEGATIVE_LABEL_WORDS or s.startswith(("no_", "non_", "not_"))


def infer_positive_label(y):
    """Best guess at which of a binary target's two labels is the EVENT.

    Default stays the sorted-max label, so {0,1} and {"No","Yes"} behave
    exactly as before. Only flips when the max label reads as an explicit
    negative and the other one does not — a guess that is announced in
    prepare_dataset's result rather than applied quietly, and that the
    caller overrides with prepare_dataset(positive_label=...)."""
    labels = sorted(pd.unique(pd.Series(y).dropna()))
    if len(labels) != 2:
        return None
    low, high = _to_native(labels[0]), _to_native(labels[1])
    if _looks_negative(high) and not _looks_negative(low):
        return low
    return high


def parse_dates(series: pd.Series) -> pd.Series:
    """to_datetime that survives a column mixing formats ("2026-01-01" next
    to "2026-01-01 00:30"): pandas 2 infers ONE format from the first value
    and coerces the rest to NaT. Fast path first; the per-element parse only
    when the fast one lost values."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().sum() > series.isna().sum():
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed


def positive_label(meta: dict, y=None):
    """The class every positive-class metric on this run refers to. Recorded
    on the run by prepare_dataset; falls back to inference for runs created
    before it was recorded, so an old run_id still answers correctly."""
    recorded = meta.get("positive_label")
    if recorded is not None:
        return recorded
    return infer_positive_label(y) if y is not None else None


def positive_index(meta: dict, y=None, classes=None) -> int:
    """The predict_proba COLUMN holding P(positive). sklearn sorts classes_,
    so this is 1 unless the positive class is the lower-sorting label."""
    labels = (
        sorted(classes)
        if classes is not None
        else sorted(pd.unique(pd.Series(y).dropna()))
    )
    if len(labels) != 2:
        return 1
    pos = positive_label(meta, y if y is not None else labels)
    return 0 if pos is not None and _to_native(labels[0]) == _to_native(pos) else 1


def _to_native(value):
    """numpy scalar -> native Python type, so it survives a json.dumps/loads
    round-trip as the same type (a stringified '3.0' fillna-ed into a float
    column is a real bug, not a formatting nitpick)."""
    return value.item() if hasattr(value, "item") else value


def _build_pipeline(meta: dict, model: str, X_train: pd.DataFrame, y_train) -> Pipeline:
    """The one place the full preprocess-then-classify pipeline is assembled
    — train_model and tune_hyperparams both go through this, so a tuned
    refit and a baseline fit are structurally the same shape, never two
    code paths that could silently diverge.

    The run's split keys (entity id, time column) are dropped here, always:
    an entity id one-hot encoded is a memorised-entity lookup, and a raw
    timestamp is either thousands of one-hot columns or a trend a tree cannot
    extrapolate past the training period."""
    steps = []
    dropped = [*(meta.get("dropped_columns") or {}), *split_keys(meta)]
    if dropped:
        steps.append(("drop", ColumnDropper(dropped)))
    if meta.get("imputation"):
        steps.append(("impute", ColumnFiller(meta["imputation"])))

    remaining = [c for c in X_train.columns if c not in dropped]
    cat_cols = [c for c in remaining if not pd.api.types.is_numeric_dtype(X_train[c])]
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"), cat_cols)
    ]
    if model == "logistic_regression":
        # A linear model needs comparable scales to converge and to weigh
        # features fairly, and cannot take the NaN a first lag leaves; trees
        # need neither. Both are fitted inside the pipeline, on training rows only.
        num_cols = [c for c in remaining if c not in cat_cols]
        transformers.append(("num", SkPipeline([("impute", SimpleImputer(strategy="median")),
                                                ("scale", StandardScaler())]), num_cols))
    steps.append(
        ("encode", ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=False))
    )

    if meta.get("use_smote"):
        # A sampler step only ever fires inside .fit()/fit_resample — imblearn's
        # Pipeline skips it during .transform()/.predict(), so it structurally
        # cannot touch the test fold or a future real prediction.
        steps.append(
            (
                "smote",
                SMOTE(
                    sampling_strategy=meta.get("smote_sampling_strategy", 1.0),
                    random_state=42,
                ),
            )
        )

    steps.append(("classifier", _make_model(model, y_train)))
    return Pipeline(steps)


# Results that describe one specific FITTED model rather than the run as a
# whole. Refitting invalidates every one of them, so train_model and
# tune_hyperparams clear them instead of leaving last model's numbers lying
# around under the new model's name.
#
# Without this, the standard remediation loop silently lies: drop the leaking
# column -> retrain -> generate_report, and the report shows the PREVIOUS
# model's SHAP ranking (the one full of `Ticket_*` columns) attributed to the
# new clean model. Clearing them also makes the readiness gate flip those
# checks back to not_run, so a retrain forces re-verification rather than
# inheriting the old run's clean bill of health.
MODEL_DERIVED_KEYS = (
    "explain",
    "error_analysis",
    "stability",
    "fairness",
    "segment_analysis",
    # Calibration describes one fitted model's score
    # distribution. Refit and the old ECE is a measurement
    # of a model that no longer exists.
    "calibration",
    # A threshold is tuned against one fitted pipeline's score
    # distribution. Refit the pipeline and it describes a model
    # that no longer exists — silently, since the number still
    # looks perfectly plausible.
    "operating_point",
    "label_rule_check",
    "backtest",
)


def _invalidate_model_derived(meta: dict) -> list[str]:
    cleared = [k for k in MODEL_DERIVED_KEYS if meta.get(k) is not None]
    for key in cleared:
        # pop, not `= None`: every reader uses .get() so absent and None look
        # the same to them, but setdefault does NOT — it returns an existing
        # None, and `meta.setdefault(k, {})[x] = ...` then crashes on the first
        # analysis after a retrain.
        meta.pop(key)
    return cleared


def _binary_or_macro_auc(y_test, proba, classes, meta: dict | None = None) -> dict:
    """Binary gets a single ROC-AUC/PR-AUC; multiclass gets one-vs-rest macro
    averages. Returns {} rather than raising if a class is missing from the
    test fold (small dataset — AUC isn't computable without both classes
    present), so the rest of the report still comes back.

    PR-AUC is scored against the run's positive class, not against
    classes_[1] — on a {"fraud","legit"} target those are different columns
    and the difference is the whole metric."""
    try:
        if len(classes) == 2:
            pos = positive_index(meta or {}, y=y_test, classes=classes)
            y_pos = (
                pd.Series(y_test).values == _to_native(sorted(classes)[pos])
            ).astype(int)
            return {
                "roc_auc": round(float(roc_auc_score(y_pos, proba[:, pos])), 4),
                "pr_auc": round(
                    float(average_precision_score(y_pos, proba[:, pos])), 4
                ),
            }
        y_bin = pd.get_dummies(y_test)[classes].values
        return {
            "roc_auc_macro": round(
                float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")),
                4,
            ),
            "pr_auc_macro": round(
                float(average_precision_score(y_bin, proba, average="macro")), 4
            ),
        }
    except ValueError:
        return {}


def _evaluate(pipeline: Pipeline, X_test, y_test, meta: dict | None = None) -> dict:
    y_pred = pipeline.predict(X_test)
    result = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    classes = sorted(y_test.unique())
    proba = (
        pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    )
    if proba is not None:
        result["auc"] = _binary_or_macro_auc(y_test, proba, classes, meta)
    if len(classes) == 2:
        # Intervals for every metric a success bar can name (binary only —
        # multiclass verdicts stay point estimates, and say so).
        pos = positive_index(meta or {}, y=y_test, classes=classes)
        result["ci"] = _bootstrap_cis(
            y_test,
            None if proba is None else proba[:, pos],
            y_pred,
            _to_native(classes[pos]),
        )
    return result


# Below this minority:majority ratio, f1_weighted stops being a usable model
# -selection metric: at wangiri's 2.5% positive rate, predicting "not fraud"
# for every row scores ~0.98 weighted F1, so ranking by it picks whichever
# model ignores the minority class most gracefully. That is the opposite of
# what a fraud run wants. Matches check_imbalance's is_imbalanced threshold so
# the two tools can't disagree about whether a dataset is imbalanced.
IMBALANCED_RATIO = 0.2


def _cv_scoring(y) -> str:
    """Which metric CV should optimise and rank by, for THIS target.

    PR-AUC (`average_precision`) on an imbalanced binary target, f1_weighted
    otherwise. `modeling/references/on_demand/domain-notes.md` already
    specifies PR-AUC as the primary metric for fraud/IRSF/Wangiri/churn and
    warns that ROC-AUC flatters a low positive rate — but model SELECTION was
    hardcoded to f1_weighted, so the agent's documented guidance and its
    actual choice of model disagreed on exactly the domains the guidance was
    written for. Selecting on one metric and reporting on another is how a
    run ends up shipping the model that scored best at doing nothing.

    Restricted to 0/1-labelled targets because sklearn's average_precision
    scorer assumes pos_label=1; anything else falls back rather than erroring.
    """
    if y.nunique() != 2 or set(pd.unique(y)) - {0, 1}:
        return "f1_weighted"
    counts = y.value_counts()
    return (
        "average_precision"
        if float(counts.min() / counts.max()) < IMBALANCED_RATIO
        else "f1_weighted"
    )


def _test_score_for(metric: str, test_eval: dict) -> float | None:
    """The held-out-test counterpart of a CV metric, so cv_to_test_gap always
    subtracts like from like. Comparing a CV PR-AUC against a test F1 would
    produce a meaningless number that the overfitting gate then acts on."""
    if metric == "average_precision":
        return (test_eval.get("auc") or {}).get("pr_auc")
    return (
        (test_eval.get("classification_report") or {})
        .get("weighted avg", {})
        .get("f1-score")
    )


# ponytail: a flat threshold on the CV-vs-test gap, not a learned overfitting
# detector — upgrade to a per-dataset-size threshold if this flags too many
# false positives on small test folds.
OVERFIT_GAP_THRESHOLD = 0.15

# Average gap between "the model says 80%" and "80% of those were positive".
# Above this, a threshold tuned on these scores encodes a precision/recall
# tradeoff nobody chose. Lives here, not in diagnostics, so the gate that
# judges calibration doesn't have to import the module that measures it.
ECE_CONCERN = 0.05


def _overfit_gate(
    cv_mean: float, test_eval: dict, metric: str = "f1_weighted"
) -> tuple[float | None, bool]:
    """cv_to_test_gap = train-fold CV mean minus the held-out test score for
    the SAME metric. A large positive gap means the model looked good under CV
    but didn't generalize to the untouched test fold — the gate export_model
    checks before letting a model out the door."""
    test_score = _test_score_for(metric, test_eval)
    if test_score is None:
        return None, False
    gap = round(cv_mean - test_score, 4)
    return gap, gap > OVERFIT_GAP_THRESHOLD
