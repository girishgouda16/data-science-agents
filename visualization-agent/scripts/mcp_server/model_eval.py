"""visualization agent — `model_eval` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import (
    _classifier_mismatch,
    _fitted,
    _load_run,
    _positive_index,
    _positive_label,
    _run_meta,
    _save,
)  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@mcp.tool()
def plot_feature_importance(
    out_path: str, run_id: str = "", importances: str = ""
) -> str:
    """Horizontal bar chart of feature importance.

    Prefer `run_id`: the chart is then built from the importances the
    training agent actually persisted (`meta["explain"]`, written by its
    `explain_model`), read straight off the artifact like every other
    run-scoped chart here.

    `importances` — a `[[feature, value], ...]` JSON list — is the fallback
    for a caller that has numbers but no run (or a run whose explain step
    was never persisted). It plots whatever it is handed: nothing checks
    those numbers against a fitted model, so a transcription slip or an
    invented ranking renders just as convincingly as a real one. The result
    says which path was used in `source`; relay that when it is
    `caller-supplied`.

    Passing both uses run_id and reports the conflict."""
    source, conflict, meta_used = None, None, None
    pairs = None

    if run_id:
        meta_used = _run_meta(run_id)
        if meta_used is None:
            return json.dumps(
                {
                    "error": f"no run '{run_id}' — ask the training agent to train_model first"
                }
            )
        # Where each training agent's explain_model persists its ranking:
        # classification meta["explain"], regression/forecasting
        # meta["explainability"], anomaly meta["feature_importance"].
        explain = meta_used.get("explain") or meta_used.get("explainability") or {}
        pairs = (
            explain.get("feature_importance")
            or explain.get("top_features")
            or meta_used.get("feature_importance")
        )
        if not pairs:
            return json.dumps(
                {
                    "error": f"run_id '{run_id}' has no persisted feature importance — ask the training agent to run "
                    "explain_model first (its result is what this chart reads).",
                }
            )
        source = (
            f"run artifact (method={explain.get('method', 'score_correlation')}, "
            f"model={meta_used.get('model') or meta_used.get('algorithm')})"
        )
        if importances:
            conflict = "both run_id and importances were passed — plotted the run's persisted values and ignored the supplied list"
    elif importances:
        try:
            pairs = json.loads(importances)
        except json.JSONDecodeError as exc:
            return json.dumps(
                {
                    "error": f"importances must be a JSON [[feature, value], ...] list: {exc}"
                }
            )
        source = "caller-supplied — NOT verified against any fitted model"
    else:
        return json.dumps(
            {
                "error": "pass run_id (preferred — reads the training agent's persisted explain_model result) or importances"
            }
        )

    if not isinstance(pairs, list) or not pairs:
        return json.dumps({"error": "no feature-importance pairs to plot"})

    features = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
    tuned = bool((meta_used or {}).get("best_params"))
    model_name = (meta_used or {}).get("model") or (meta_used or {}).get("algorithm")
    title = "Feature importance" + (
        f" ({model_name}{', tuned' if tuned else ''})" if model_name else ""
    )
    sns.barplot(x=values, y=features, ax=ax, orient="h")
    ax.set_title(title)
    saved = _save(fig, out_path, title, run_id, kind="feature_importance")
    return json.dumps(
        {
            **saved,
            "n_features": len(features),
            "source": source,
            **({"conflict": conflict} if conflict else {}),
        }
    )


@mcp.tool()
def plot_confusion_matrix(run_id: str, out_path: str) -> str:
    """Confusion matrix for run_id's CURRENT fitted model — tuned if the
    classification agent's tune_hyperparams ran for this run, its baseline
    otherwise — evaluated on that run's held-out test fold. Reads the same
    pipeline the classification agent fitted; doesn't fit its own."""
    loaded = _load_run(run_id)
    if loaded is None:
        return json.dumps(
            {
                "error": f"no fitted model for run_id '{run_id}' — ask the classification agent to train_model first"
            }
        )
    pipeline, meta, test = loaded
    mismatch = _classifier_mismatch(run_id, pipeline, meta, "plot_confusion_matrix")
    if mismatch:
        return json.dumps(mismatch)
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]
    # Drawn at the threshold the model SHIPS with (export bundles it, serving
    # applies it), not pipeline.predict()'s 0.5 — otherwise the reviewer signs
    # off a confusion matrix of a decision rule nobody deploys.
    op, classes = meta.get("operating_point") or {}, sorted(y_test.unique())
    threshold = None
    if (
        op.get("threshold") is not None
        and op.get("pipeline_version") == meta.get("pipeline_version")
        and len(classes) == 2
        and hasattr(pipeline, "predict_proba")
    ):
        threshold = float(op["threshold"])
        pos_idx = _positive_index(meta, classes)
        proba = pipeline.predict_proba(X_test)[:, pos_idx]
        y_pred = np.where(proba >= threshold, classes[pos_idx], classes[1 - pos_idx])
    else:
        y_pred = pipeline.predict(X_test)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cutoff = f"threshold {threshold:.4g}" if threshold is not None else "default 0.5"
    title = f"Confusion matrix ({meta['model']}{', tuned' if meta.get('best_params') else ''}, {cutoff})"
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.grid(False)  # the whitegrid theme would draw lines across the cells
    ax.set_title(title)
    saved = _save(fig, out_path, title, run_id, kind="confusion_matrix")
    return json.dumps(
        {
            **saved,
            "model": meta["model"],
            "tuned": bool(meta.get("best_params")),
            "n_test": len(y_test),
            "threshold": threshold if threshold is not None else 0.5,
            "threshold_source": (
                "tuned operating point (what serving applies)"
                if threshold is not None
                else "sklearn default — no current tuned operating point"
            ),
        }
    )


@mcp.tool()
def plot_roc_curve(run_id: str, out_path: str) -> str:
    """ROC curve (with AUC) for run_id's CURRENT fitted model on its held-out
    test fold. Binary target: single curve. Multiclass: one-vs-rest curve
    per class. Reads the same pipeline the classification agent fitted;
    doesn't fit its own."""
    loaded = _load_run(run_id)
    if loaded is None:
        return json.dumps(
            {
                "error": f"no fitted model for run_id '{run_id}' — ask the classification agent to train_model first"
            }
        )
    pipeline, meta, test = loaded
    mismatch = _classifier_mismatch(run_id, pipeline, meta, "plot_roc_curve")
    if mismatch:
        return json.dumps(mismatch)
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]

    classes = sorted(y_test.unique())
    proba = pipeline.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))

    if len(classes) == 2:
        pos_label = _positive_label(meta, classes)
        display = RocCurveDisplay.from_predictions(
            y_test,
            proba[:, _positive_index(meta, classes)],
            pos_label=pos_label,
            ax=ax,
            name=f"{meta['model']} (positive={pos_label})",
        )
        aucs = {
            "positive_class": round(float(display.roc_auc), 4),
            "positive_label": str(pos_label),
        }
    else:
        y_bin = label_binarize(y_test, classes=classes)
        aucs = {}
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            roc_auc = auc(fpr, tpr)
            aucs[str(c)] = round(float(roc_auc), 4)
            ax.plot(fpr, tpr, label=f"{c} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

    title = f"ROC curve ({meta['model']}{', tuned' if meta.get('best_params') else ''})"
    ax.set_title(title)
    saved = _save(fig, out_path, title, run_id, kind="roc_curve")
    return json.dumps(
        {
            **saved,
            "model": meta["model"],
            "tuned": bool(meta.get("best_params")),
            "classes": [str(c) for c in classes],
            "auc": aucs,
        }
    )


@mcp.tool()
def plot_pr_curve(run_id: str, out_path: str) -> str:
    """Precision-Recall curve (with average precision) for run_id's CURRENT
    fitted model on its held-out test fold. Binary target: single curve.
    Multiclass: one-vs-rest curve per class. Reads the same pipeline the
    classification agent fitted; doesn't fit its own."""
    loaded = _load_run(run_id)
    if loaded is None:
        return json.dumps(
            {
                "error": f"no fitted model for run_id '{run_id}' — ask the classification agent to train_model first"
            }
        )
    pipeline, meta, test = loaded
    mismatch = _classifier_mismatch(run_id, pipeline, meta, "plot_pr_curve")
    if mismatch:
        return json.dumps(mismatch)
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]

    classes = sorted(y_test.unique())
    proba = pipeline.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))

    if len(classes) == 2:
        pos_label = _positive_label(meta, classes)
        display = PrecisionRecallDisplay.from_predictions(
            y_test,
            proba[:, _positive_index(meta, classes)],
            pos_label=pos_label,
            ax=ax,
            name=f"{meta['model']} (positive={pos_label})",
        )
        aps = {
            "positive_class": round(float(display.average_precision), 4),
            "positive_label": str(pos_label),
        }
    else:
        y_bin = label_binarize(y_test, classes=classes)
        aps = {}
        for i, c in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
            ap = average_precision_score(y_bin[:, i], proba[:, i])
            aps[str(c)] = round(float(ap), 4)
            ax.plot(recall, precision, label=f"{c} (AP={ap:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()

    title = f"PR curve ({meta['model']}{', tuned' if meta.get('best_params') else ''})"
    ax.set_title(title)
    saved = _save(fig, out_path, title, run_id, kind="pr_curve")
    return json.dumps(
        {
            **saved,
            "model": meta["model"],
            "tuned": bool(meta.get("best_params")),
            "classes": [str(c) for c in classes],
            "ap": aps,
        }
    )


@mcp.tool()
def plot_shap_beeswarm(run_id: str, out_path: str, background_size: int = 200) -> str:
    """SHAP beeswarm — per-feature, per-sample impact (not just a mean-|SHAP|
    bar) for run_id's CURRENT fitted model: TreeExplainer for
    random_forest/xgboost, LinearExplainer for logistic_regression. Reads
    the same pipeline.pkl the classification agent fitted; doesn't fit its
    own. background_size is a **stratified** sample of the test fold (same
    logic as classification-agent's explain_model) so a rare positive class
    (e.g. fraud) isn't averaged away — raise it on a heavily imbalanced
    target, or pass a value >= the test fold size to use it in full."""
    loaded = _load_run(run_id)
    if loaded is None:
        return json.dumps(
            {
                "error": f"no fitted model for run_id '{run_id}' — ask the classification agent to train_model first"
            }
        )
    pipeline, meta, test = loaded
    # meta.get(), not meta["model"] — this run_id might belong to a
    # clustering/anomaly run whose meta.json has no "model" key at all
    # (clustering/anomaly use "algorithm" instead), which would otherwise
    # KeyError here before ever reaching the real "wrong model type" check.
    if (
        meta.get("model") not in ("random_forest", "xgboost", "logistic_regression")
        or "target" not in meta
    ):
        return json.dumps(
            {
                "error": f"no SHAP explainer wired up for model '{meta.get('model', meta.get('algorithm', '?'))}'"
            }
        )
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]

    # Same rare-positive floor as classification-agent's explain_model, which
    # this tool's docstring claims parity with. A flat stratified 200 on a
    # 0.5%-fraud target leaves ~1 positive row in the plot, and a beeswarm
    # built from one positive is a picture of the negative class.
    MIN_POSITIVE_BACKGROUND = 10
    classes = sorted(y_test.unique())
    if len(classes) == 2:
        positive_rate = float((y_test == _positive_label(meta, classes)).mean())
        if background_size == 200 and positive_rate > 0:
            needed = int(np.ceil(MIN_POSITIVE_BACKGROUND / positive_rate))
            if needed > background_size:
                background_size = min(needed, len(X_test))

    if 0 < background_size < len(X_test):
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=background_size, stratify=y_test, random_state=42
        )

    # Same "skip the sampler, only encode" step as classification-agent's
    # explain_model — a Pipeline.transform() call breaks on a step (smote)
    # that only has fit_resample, not transform.
    # Same fix as classification-agent's explain_model: calibrate_model wraps the
    # whole fitted pipeline in CalibratedClassifierCV, which has no .steps at all.
    # SHAP is additive in the UNCALIBRATED score anyway, so explain the pipeline
    # it wraps rather than erroring on a missing attribute.
    calibrated = isinstance(pipeline, CalibratedClassifierCV)
    explain_pipeline = pipeline
    if calibrated:
        fitted = getattr(pipeline, "calibrated_classifiers_", None)
        inner = None
        if fitted:
            inner = getattr(fitted[0], "estimator", None) or getattr(
                fitted[0], "base_estimator", None
            )
        if inner is None:
            return json.dumps(
                {
                    "error": "model is calibrated but its inner pipeline could not be recovered for SHAP"
                }
            )
        explain_pipeline = inner

    try:
        encoder_steps = [
            (name, step)
            for name, step in explain_pipeline.steps
            if name not in ("smote", "classifier")
        ]
        X_encoded = SkPipeline(encoder_steps).transform(X_test)
        feature_names = explain_pipeline.named_steps["encode"].get_feature_names_out()
        clf = explain_pipeline.named_steps["classifier"]
    except (KeyError, AttributeError) as e:
        # Belt-and-suspenders past the meta["model"] check above: a
        # same-named model from a different agent (e.g. regression's own
        # "random_forest") could still have a different pipeline shape —
        # missing steps (KeyError) or not even a Pipeline, e.g. a bare
        # estimator with no .steps at all (AttributeError).
        return json.dumps(
            {
                "error": f"pipeline for run_id '{run_id}' doesn't have the expected steps ({e}) — plot_shap_beeswarm "
                "only supports classification-agent's pipeline shape",
            }
        )
    explainer = (
        shap.TreeExplainer(clf)
        if meta["model"] in ("random_forest", "xgboost")
        else shap.LinearExplainer(clf, X_encoded)
    )
    shap_values = explainer.shap_values(X_encoded)
    # Same multi-output normalization as explain_model: collapse to the
    # positive class's (n_samples, n_features) array for a binary target —
    # a beeswarm plots one class's signed contributions, not an |SHAP| average.
    # "Last class" is the sorted-max label, which is the NEGATIVE one on a
    # {"churn","no_churn"} target — every arrow in the plot would then point
    # the wrong way while looking entirely sensible.
    pos_idx = _positive_index(meta, sorted(y_test.unique()))
    if isinstance(shap_values, list):
        shap_values = (
            shap_values[pos_idx] if len(shap_values) > pos_idx else shap_values[-1]
        )
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, pos_idx]

    fig = plt.figure(figsize=(7, max(3, len(feature_names) * 0.35)))
    shap.summary_plot(shap_values, X_encoded, feature_names=feature_names, show=False)
    title = f"SHAP beeswarm ({meta['model']}{', tuned' if meta.get('best_params') else ''}{', pre-calibration' if calibrated else ''})"
    plt.title(title)
    saved = _save(fig, out_path, title, run_id, kind="shap_beeswarm")
    return json.dumps(
        {
            **saved,
            "model": meta["model"],
            "tuned": bool(meta.get("best_params")),
            "n_samples": len(X_test),
            "calibrated": calibrated,
        }
    )


@mcp.tool()
def plot_calibration_curve(run_id: str, out_path: str, n_bins: int = 10) -> str:
    """Reliability curve for run_id's CURRENT fitted model on its held-out
    test fold (binary targets only) — is a 0.8-confidence prediction
    actually right ~80% of the time, or is the model systematically over/
    under-confident? Check this before trusting a hand-picked decision
    threshold (see the classification agent's tune_threshold) — tuning a
    threshold off miscalibrated probabilities picks the wrong cutoff. Reads
    the same pipeline the classification agent fitted; doesn't fit its own."""
    loaded = _load_run(run_id)
    if loaded is None:
        return json.dumps(
            {
                "error": f"no fitted model for run_id '{run_id}' — ask the classification agent to train_model first"
            }
        )
    pipeline, meta, test = loaded
    mismatch = _classifier_mismatch(run_id, pipeline, meta, "plot_calibration_curve")
    if mismatch:
        return json.dumps(mismatch)
    X_test, y_test = test.drop(columns=[meta["target"]]), test[meta["target"]]
    if y_test.nunique() != 2:
        return json.dumps({"error": "calibration curve needs a binary target"})

    classes = sorted(y_test.unique())
    pos_label = _positive_label(meta, classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(
        y_test,
        pipeline.predict_proba(X_test)[:, _positive_index(meta, classes)],
        pos_label=pos_label,
        n_bins=n_bins,
        ax=ax,
        name=f"{meta['model']} (positive={pos_label})",
    )
    title = f"Calibration curve ({meta['model']}{', tuned' if meta.get('best_params') else ''})"
    ax.set_title(title)
    saved = _save(fig, out_path, title, run_id, kind="calibration_curve")
    return json.dumps(
        {
            **saved,
            "model": meta["model"],
            "tuned": bool(meta.get("best_params")),
            "n_test": len(y_test),
        }
    )


# ── Regression / forecasting / clustering / anomaly ─────────────────────────
# Same rule as the classification charts above: read the run's own fitted
# model and held-out fold, never refit, so the chart shows what the training
# agent reported.

MAX_SCATTER_POINTS = 5000  # ponytail: random subsample for scatter legibility/speed; counts/metrics use every row


def _tuned(meta: dict) -> str:
    return ", tuned" if meta.get("best_params") else ""


@mcp.tool()
def plot_regression_diagnostics(run_id: str, out_path: str) -> str:
    """Regression run's CURRENT model on its held-out test fold, three panels:
    predicted vs actual (points on the dashed line are perfect), residuals vs
    predicted (a funnel = error grows with the prediction; a curve = a missed
    non-linearity), and the residual distribution (skew or heavy tails = the
    RMSE understates the bad cases). Reads the regression agent's fitted
    pipeline; doesn't fit its own."""
    loaded, error = _fitted(
        run_id, "regressor", "plot_regression_diagnostics", "regression"
    )
    if error:
        return error
    pipeline, meta, test = loaded
    y = test[meta["target"]].to_numpy(dtype=float)
    pred = np.asarray(
        pipeline.predict(test.drop(columns=[meta["target"]])), dtype=float
    )
    resid = y - pred
    r2, rmse, mae = (
        r2_score(y, pred),
        float(mean_squared_error(y, pred) ** 0.5),
        mean_absolute_error(y, pred),
    )

    idx = np.random.default_rng(42).permutation(len(y))[:MAX_SCATTER_POINTS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(y[idx], pred[idx], s=10, alpha=0.6)
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "--", color="gray")
    axes[0].set(
        xlabel=f"actual {meta['target']}",
        ylabel="predicted",
        title="Predicted vs actual",
    )
    axes[1].scatter(pred[idx], resid[idx], s=10, alpha=0.6)
    axes[1].axhline(0, ls="--", color="gray")
    axes[1].set(
        xlabel="predicted",
        ylabel="residual (actual − predicted)",
        title="Residuals vs predicted",
    )
    sns.histplot(resid, kde=True, ax=axes[2])
    axes[2].set(xlabel="residual", title="Residual distribution")
    title = f"Regression diagnostics ({meta.get('model')}{_tuned(meta)}) — test R² {r2:.3f}, RMSE {rmse:.4g}"
    fig.suptitle(title)
    saved = _save(fig, out_path, title, run_id, kind="regression_diagnostics")
    return json.dumps(
        {
            **saved,
            "model": meta.get("model"),
            "tuned": bool(meta.get("best_params")),
            "n_test": len(y),
            "r2": round(float(r2), 4),
            "rmse": round(rmse, 4),
            "mae": round(float(mae), 4),
            "residual_mean": round(float(resid.mean()), 4),
        }
    )


@mcp.tool()
def plot_forecast(run_id: str, out_path: str, history_points: int = 0) -> str:
    """Forecasting run's CURRENT model: the recent training history, the
    held-out actuals, and the model's forecast for that held-out period — the
    same forecast its test RMSE/MAE/MAPE were scored on (saved by the
    forecasting agent's train_model/tune_hyperparams). history_points: how
    much history to show; 0 = three test-lengths (at least 30)."""
    meta = _run_meta(run_id)
    if meta is None or not meta.get("date_column"):
        return json.dumps(
            {
                "error": f"run_id '{run_id}' isn't a forecasting run — plot_forecast only applies to the forecasting agent's runs"
            }
        )
    run, date, target = RUNS_DIR / run_id, meta["date_column"], meta["target"]
    if not (run / "test_forecast.csv").exists():
        return json.dumps(
            {
                "error": f"run_id '{run_id}' has no saved held-out forecast — ask the forecasting agent to "
                "train_model (a run trained before forecasts were saved must be re-trained)"
            }
        )
    train = pd.read_csv(run / "train.csv", parse_dates=[date])
    test = pd.read_csv(run / "test.csv", parse_dates=[date])
    forecast = pd.read_csv(run / "test_forecast.csv", parse_dates=[date])
    panel = bool(meta.get("panel"))
    if panel:  # many series: draw their total — a per-series band does not add up, so none is drawn
        train, test = (f.groupby(date, as_index=False)[target].sum() for f in (train, test))
        forecast = forecast.groupby(date, as_index=False)["forecast"].sum()
    history = train.tail(history_points or max(3 * len(test), 30))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(history[date], history[target], color="gray", label="history (train)")
    ax.plot(test[date], test[target], color="black", label="actual (held-out)")
    ax.plot(
        forecast[date],
        forecast["forecast"],
        "--",
        color="tab:blue",
        label=f"forecast ({meta.get('model')})",
    )
    if {"lower", "upper"} <= set(forecast.columns):
        band = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
        level = int(round((band.get("interval_level") or 0.8) * 100))
        ax.fill_between(forecast[date], forecast["lower"], forecast["upper"], color="tab:blue", alpha=0.15,
                        label=f"{level}% interval (coverage {band.get('interval_coverage')})")
    ax.axvline(test[date].iloc[0], color="gray", ls=":", lw=1)
    ax.set(xlabel=date, ylabel=f"{target} (total of {meta.get('n_series')} series)" if panel else target)
    ax.legend()
    metrics = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
    title = (f"Forecast vs actual ({meta.get('model')}{_tuned(meta)}) — test RMSE {metrics.get('rmse')}, "
             f"WAPE {metrics.get('wape')}%, MASE {metrics.get('mase')}")
    ax.set_title(title)
    saved = _save(fig, out_path, title, run_id, kind="forecast")
    return json.dumps(
        {
            **saved,
            "model": meta.get("model"),
            "tuned": bool(meta.get("best_params")),
            "horizon": len(test),
            "history_shown": len(history),
            "test_metrics": {k: metrics.get(k) for k in ("rmse", "mae", "mape")},
        }
    )


@mcp.tool()
def plot_clusters(run_id: str, out_path: str) -> str:
    """Clustering run's CURRENT model on its fit fold: the rows projected to
    2-D (PCA of the same scaled/encoded features the clusterer saw), coloured
    by cluster, plus cluster sizes. Well-separated blobs = real structure;
    one smear with colours painted on = the algorithm imposed groups the data
    doesn't have. DBSCAN noise (-1) is grey. Reads the clustering agent's
    fitted pipeline; doesn't fit its own."""
    loaded, error = _fitted(run_id, "clusterer", "plot_clusters", "clustering")
    if error:
        return error
    pipeline, meta, _ = loaded
    fit = pd.read_csv(RUNS_DIR / run_id / "train.csv")
    labels = getattr(pipeline.named_steps["clusterer"], "labels_", None)
    if labels is None:
        return json.dumps({"error": "the current clusterer exposes no fitted labels"})
    preprocess = pipeline.named_steps["preprocess"]
    X = preprocess.transform(fit)
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=float)
    try:
        axis_names = [
            str(n).split("__")[-1] for n in preprocess.get_feature_names_out()
        ] + ["—"]
    except Exception:  # a transformer without feature names
        axis_names = ["feature 1", "feature 2"]
    explained = None
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        coords, explained = (
            pca.fit_transform(X),
            round(float(pca.explained_variance_ratio_.sum()), 4),
        )
    else:
        coords = np.column_stack(
            [X[:, 0], X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))]
        )

    idx = np.random.default_rng(42).permutation(len(labels))[:MAX_SCATTER_POINTS]
    fig, (ax, ax_sizes) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]}
    )
    for label in sorted(set(labels)):
        mask = labels[idx] == label
        ax.scatter(
            coords[idx][mask, 0],
            coords[idx][mask, 1],
            s=10,
            alpha=0.7,
            label="noise" if label == -1 else f"cluster {label}",
            **({"color": "lightgray"} if label == -1 else {}),
        )
    ax.set(
        xlabel="PC1" if explained is not None else f"{axis_names[0]} (scaled)",
        ylabel="PC2" if explained is not None else f"{axis_names[1]} (scaled)",
    )
    ax.legend(markerscale=2)
    sizes = pd.Series(labels).value_counts().sort_index()
    ax_sizes.bar([str(i) for i in sizes.index], sizes.to_numpy())
    ax_sizes.set(xlabel="cluster", ylabel="rows", title="Cluster sizes (fit fold)")
    silhouette = ((meta.get("training_metrics") or {}).get("fit") or {}).get(
        "silhouette_score"
    )
    title = (
        f"Clusters ({meta.get('algorithm')}, k={len([l for l in sizes.index if l != -1])}) — silhouette {silhouette}"
        + (
            f", 2-D view keeps {explained:.0%} of variance"
            if explained is not None
            else ""
        )
    )
    ax.set_title(title)
    saved = _save(fig, out_path, title, run_id, kind="clusters")
    return json.dumps(
        {
            **saved,
            "algorithm": meta.get("algorithm"),
            "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
            "silhouette": silhouette,
            "pca_variance_explained": explained,
            "n_rows": len(labels),
        }
    )


@mcp.tool()
def plot_anomaly_scores(run_id: str, out_path: str) -> str:
    """Anomaly run's CURRENT detector on its held-out fold: the distribution
    of anomaly scores (higher = more anomalous) with the decision threshold
    the detector flags at. With a label column, normal and labelled-anomaly
    rows are drawn separately — overlap around the threshold is where the
    false alarms and misses live. Reads the anomaly agent's fitted pipeline;
    doesn't fit its own."""
    loaded, error = _fitted(run_id, "detector", "plot_anomaly_scores", "anomaly")
    if error:
        return error
    pipeline, meta, test = loaded
    label = meta.get("label_column")
    X = test.drop(columns=[label]) if label and label in test.columns else test
    scores = np.asarray(
        pipeline.decision_function(X), dtype=float
    )  # PyOD: higher = more anomalous
    flagged = int(np.asarray(pipeline.predict(X)).sum())  # PyOD: 1 = anomaly
    threshold = float(pipeline.named_steps["detector"].threshold_)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labelled = bool(meta.get("evaluation_available")) and label in test.columns
    if labelled:
        truth = np.where(
            test[label].astype(str) == str(meta.get("anomaly_label")),
            "labelled anomaly",
            "labelled normal",
        )
        sns.histplot(
            x=scores,
            hue=truth,
            bins=50,
            ax=ax,
            element="step",
            common_norm=False,
            stat="density",
        )
    else:
        sns.histplot(scores, bins=50, ax=ax)
    ax.axvline(threshold, color="red", ls="--")
    ax.text(
        threshold,
        ax.get_ylim()[1] * 0.95,
        f" threshold: {flagged}/{len(scores)} flagged",
        color="red",
        va="top",
    )
    ax.set(xlabel="anomaly score (higher = more anomalous)")
    title = f"Anomaly scores ({meta.get('algorithm')}, contamination {meta.get('contamination')})"
    ax.set_title(title)
    saved = _save(fig, out_path, title, run_id, kind="anomaly_scores")
    return json.dumps(
        {
            **saved,
            "algorithm": meta.get("algorithm"),
            "n_test": len(scores),
            "flagged": flagged,
            "flagged_rate": round(flagged / max(len(scores), 1), 4),
            "threshold": round(threshold, 6),
            "labelled": labelled,
        }
    )
