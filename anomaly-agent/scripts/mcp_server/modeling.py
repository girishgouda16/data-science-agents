"""anomaly agent — `modeling` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .diagnostics import compute_readiness  # noqa: F401
from sklearn.pipeline import Pipeline as SkPipeline
from core import runtime, toolguard
from .core import (
    _build_pipeline,
    _clamp_contamination,
    _current_metrics,
    _evaluate,
    _feature_frame,
    _json_default,
    _label_to_binary,
    _load_meta,
    _load_split,
    _run_dir,
    _save_meta,
    _score_distribution,
    _score_vector,
)  # noqa: F401


@mcp.tool()
def propose_contamination(run_id: str, review_capacity: int = 0, rows_per_period: int = 0) -> str:
    """Recommend the alert share (contamination) — the share of rows that
    will be flagged. It is an OPERATING decision more than a statistical one:
    a queue nobody can work is ignored. review_capacity / rows_per_period
    (e.g. 200 cases a day out of 1,000,000 records a day) is the strongest
    basis; the labelled training prevalence is the fallback, and 5% the last
    resort, flagged as a guess. When the budget is below the prevalence,
    recall is capped at budget / prevalence — say so. Read-only."""
    train, _, meta = _load_split(run_id)
    prevalence = (float(_label_to_binary(train[meta["label_column"]], meta).mean())
                  if meta.get("evaluation_available") else None)
    result = {"run_id": run_id, "evaluation_available": bool(meta.get("evaluation_available"))}
    if prevalence is not None:
        result["derived_from_train_prevalence"] = round(prevalence, 4)
    if review_capacity > 0 and rows_per_period > 0:
        budget = review_capacity / rows_per_period
        result.update(recommended_contamination=round(_clamp_contamination(budget), 6), basis="review_capacity",
                      reasoning=f"the team can review {review_capacity} of every {rows_per_period} rows, so the "
                                "detector flags that share — the threshold is set by what can be worked")
        if prevalence and budget < prevalence:
            result["recall_ceiling"] = round(budget / prevalence, 4)
            result["warning"] = (f"the budget ({budget:.4%}) is below the labelled prevalence ({prevalence:.4%}): "
                                 f"at most {budget / prevalence:.0%} of anomalies can be caught however good the "
                                 "detector — raise capacity or prioritise by value")
    elif prevalence is not None:
        result.update(recommended_contamination=round(_clamp_contamination(prevalence), 4), basis="label_prevalence",
                      reasoning="the training fold's anomaly prevalence — ask how many alerts the team can review; "
                                "the threshold should be what can be worked, not what the labels say")
    else:
        result.update(recommended_contamination=0.05, basis="default_guess",
                      reasoning="no labels and no review capacity — 5% is a guess; ask how many alerts the team "
                                "can review per period and how many rows arrive, then re-propose")
    return json.dumps(result, default=_json_default)


def _fit_detector(meta: dict, train: pd.DataFrame, test: pd.DataFrame, algorithm: str,
                  contamination: float, train_on: str) -> tuple:
    """One detector, fitted and measured. Returns (pipeline, metrics).
    The alert threshold is always the (1 - contamination) quantile of the
    scores on the WHOLE training fold, so 'contamination' means the share of
    real traffic flagged — also when the detector was fitted on known-normal
    rows only, or on a sample."""
    X_train, X_test = _feature_frame(train, meta), _feature_frame(test, meta)
    labelled = bool(meta.get("evaluation_available"))
    y_train = _label_to_binary(train[meta["label_column"]], meta) if labelled else None
    normal_only = train_on == "normal" or (train_on == "auto" and labelled)
    fit_rows = X_train[y_train.to_numpy() == 0] if normal_only else X_train
    sampled = algorithm in QUADRATIC and len(fit_rows) > FIT_SAMPLE_ROWS
    if sampled:
        fit_rows = fit_rows.sample(FIT_SAMPLE_ROWS, random_state=42)

    pipeline = _build_pipeline(meta, algorithm, contamination, X_train)
    pipeline.fit(fit_rows)
    detector = pipeline.named_steps["detector"]
    train_scores = _score_vector(pipeline, X_train)
    detector.threshold_ = float(np.quantile(train_scores, 1 - contamination))

    pre = SkPipeline(pipeline.steps[:-1])
    eval_rows = X_test if len(X_test) >= 50 else X_train
    Xt_eval = np.asarray(pre.transform(eval_rows), dtype=float)
    metrics = {
        "requested_contamination": contamination,
        "trained_on": (f"normal rows only ({int(y_train.sum())} known anomalies left out of the fit)"
                       if normal_only else "all training rows"),
        "fit_rows": int(len(fit_rows)),
        "fit_sampled": sampled,
        "log_scaled_columns": log_scaled_columns(pipeline),
        "holdout_alert_rate": round(float(np.mean(pipeline.predict(X_test))), 4),
        "stability": alert_stability(algorithm, contamination, np.asarray(pre.transform(fit_rows), dtype=float),
                                     Xt_eval, np.asarray(detector.decision_function(Xt_eval))),
    }
    k = max(5, int(round(contamination * len(eval_rows))))
    alerts = eval_rows.iloc[np.argsort(-_score_vector(pipeline, eval_rows), kind="stable")[:k]]
    metrics["model_warnings"] = rare_category_warnings(detector_inputs(X_train, meta), detector_inputs(alerts, meta))
    if labelled:
        metrics.update(_evaluate(pipeline, X_test, _label_to_binary(test[meta["label_column"]], meta), contamination))
    else:
        predicted_rate = float(np.mean(train_scores > detector.threshold_))
        metrics.update(predicted_anomaly_rate=round(predicted_rate, 4),
                       anomaly_rate_gap=round(abs(predicted_rate - contamination), 4),
                       anomaly_rate_match=round(1.0 - abs(predicted_rate - contamination), 4),
                       score_distribution=_score_distribution(train_scores))
    return pipeline, metrics


@mcp.tool()
def train_model(run_id: str, algorithm: str = "iforest", contamination: float = 0.05, train_on: str = "auto") -> str:
    """Fit one detector pipeline (drop -> impute -> log skewed numerics /
    standardise / one-hot -> detector) on the training fold and make it the
    run's current model. Labels never enter .fit(); they measure it.
    algorithm: iforest (default; fast, no distance assumptions), ecod
    (parameter-free, per-feature tails), knn (distance to neighbours), lof
    (density relative to the neighbourhood: 'unusual for its kind'), ocsvm
    (kernel boundary). knn/lof/ocsvm fit on a 20k-row sample above that.
    train_on: "auto" = known-normal rows only when labels exist (novelty
    detection: the detector learns what normal looks like, undiluted by the
    anomalies it must find), else all rows; "all" / "normal" to force.
    Reports, with labels: average precision (PR-AUC), precision / recall /
    lift of the review queue at the budget (the top `contamination` share),
    ROC-AUC. Always: stability — the top-alert overlap across refits on
    resampled data; below 0.5 the alert list is an artefact of the sample."""
    if algorithm not in ALGORITHMS:
        return json.dumps({"error": f"unknown algorithm '{algorithm}', use one of {list(ALGORITHMS)}"})
    if train_on not in ("auto", "all", "normal"):
        return json.dumps({"error": "train_on must be auto, all or normal"})
    contamination = round(_clamp_contamination(contamination), 6)
    train, test, meta = _load_split(run_id)
    if train_on == "normal" and not meta.get("evaluation_available"):
        return json.dumps({"error": "train_on='normal' needs a label column to know which rows are normal"})
    try:
        pipeline, metrics = _fit_detector(meta, train, test, algorithm, contamination, train_on)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    artifact_path = str(_run_dir(run_id) / "pipeline.pkl")
    joblib.dump(pipeline, artifact_path)
    meta["algorithm"] = algorithm
    meta["contamination"] = contamination
    meta["train_on"] = train_on
    meta["artifact_path"] = artifact_path
    meta["metrics"] = metrics
    meta["reason_reference"] = reason_reference(detector_inputs(_feature_frame(train, meta), meta))
    meta.pop("review_queue", None)  # the previous detector's queue
    _save_meta(run_id, meta)

    labelled = bool(meta.get("evaluation_available"))
    save_training_run(
        run_id,
        target_column=meta.get("label_column") or "",
        best_model=algorithm,
        metric="average_precision" if labelled else "top_alert_jaccard",
        best_score=float((metrics.get("average_precision") if labelled
                          else metrics["stability"]["top_alert_jaccard_mean"]) or 0.0),
        test_auc=metrics.get("roc_auc") if labelled else None,
        cv_to_test_gap=None,
        overfitting_warning=metrics["stability"]["top_alert_jaccard_mean"] < STABLE_JACCARD,
        best_params={"contamination": contamination, "train_on": train_on},
        label_classes=[meta["normal_label"], meta["anomaly_label"]] if labelled else None,
        artifact_path=artifact_path,
    )
    return json.dumps({"run_id": run_id, "algorithm": algorithm, "contamination": contamination, **metrics},
                      default=_json_default)


@mcp.tool()
def compare_detectors(run_id: str, algorithms: str = "", contamination: float = 0.0, train_on: str = "auto") -> str:
    """Fit several detectors on the same split and rank them — read-only, the
    run's current model is untouched. With labels: by average precision
    (then precision at the budget). Without: by stability, with each
    detector's agreement with the CONSENSUS (the top alerts of the averaged
    rank of all of them) — alerts several different detectors agree on are
    the ones to trust first. contamination: 0 = the run's last one or 5%."""
    train, test, meta = _load_split(run_id)
    names = [a.strip() for a in algorithms.split(",") if a.strip()] or ["iforest", "ecod", "knn", "lof"]
    unknown = [a for a in names if a not in ALGORITHMS]
    if unknown:
        return json.dumps({"error": f"unknown algorithm(s) {unknown}, use any of {list(ALGORITHMS)}"})
    contamination = round(_clamp_contamination(contamination or meta.get("contamination") or 0.05), 6)
    labelled = bool(meta.get("evaluation_available"))
    X_test = _feature_frame(test, meta)
    rows, ranks = [], []
    for name in names:
        try:
            pipeline, metrics = _fit_detector(meta, train, test, name, contamination, train_on)
        except Exception as exc:
            rows.append({"algorithm": name, "error": str(exc)})
            continue
        scores = _score_vector(pipeline, X_test)
        ranks.append(pd.Series(scores).rank(pct=True).to_numpy())
        rows.append({"algorithm": name, "stability": metrics["stability"]["top_alert_jaccard_mean"],
                     **({k: metrics.get(k) for k in ("average_precision", "precision_at_budget",
                                                      "recall_at_budget", "lift_at_budget", "roc_auc")}
                        if labelled else {"holdout_alert_rate": metrics["holdout_alert_rate"]}),
                     "_scores": scores})
    k = max(1, int(round(contamination * len(X_test))))
    consensus = set(np.argsort(-np.mean(ranks, axis=0), kind="stable")[:k].tolist()) if ranks else set()
    for row in rows:
        scores = row.pop("_scores", None)
        if scores is not None:
            top = set(np.argsort(-scores, kind="stable")[:k].tolist())
            row["agreement_with_consensus"] = round(len(top & consensus) / len(top | consensus), 4)
    scored = [r for r in rows if "error" not in r]
    key = (lambda r: (-(r.get("average_precision") or 0), -(r.get("precision_at_budget") or 0))) if labelled \
        else (lambda r: (-r["stability"], -r["agreement_with_consensus"]))
    ranked = sorted(scored, key=key)
    return json.dumps({"run_id": run_id, "contamination": contamination, "ranked": ranked + [r for r in rows if "error" in r],
                       "recommended": ranked[0]["algorithm"] if ranked else None,
                       "ranking_rule": "average precision on the held-out labels" if labelled
                       else "stability, then agreement with the consensus of all detectors"},
                      default=_json_default)


@mcp.tool()
def explain_model(run_id: str, top_n: int = 20) -> str:
    """What the current detector flags and WHY, on the held-out fold:
      feature_importance  global: each encoded feature's |correlation| with
                          the anomaly score
      review_queue        the top_n highest-scoring held-out rows, each with
                          its reasons — its most extreme columns against the
                          training fold (robust sd from the median; rare or
                          never-seen categories) — and, with labels, whether
                          it is a known anomaly. This is what a reviewer
                          works: without labels, have a person check it
                          before anyone acts on the detector.
    Written to review_queue.csv in the run."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps({"error": "no fitted model for this run yet — call train_model first"})
    pipeline = joblib.load(pipeline_path)
    train, test, meta = _load_split(run_id)
    X_test = _feature_frame(test, meta)
    X_transformed = np.asarray(SkPipeline(pipeline.steps[:-1]).transform(X_test), dtype=float)
    feature_names = pipeline.named_steps["encode"].get_feature_names_out()
    scores = _score_vector(pipeline, X_test)

    importances = []
    for idx, feature in enumerate(feature_names):
        col = X_transformed[:, idx]
        corr = float(np.corrcoef(col, scores)[0, 1]) if np.std(col) and np.std(scores) else 0.0
        importances.append((feature, round(abs(corr), 4) if np.isfinite(corr) else 0.0))
    importances.sort(key=lambda kv: -kv[1])
    top = [[feature, float(value)] for feature, value in importances[:10]]

    order = np.argsort(-scores, kind="stable")[: max(1, top_n)]
    queue_rows = test.iloc[order]
    ref = meta.get("reason_reference") or reason_reference(detector_inputs(_feature_frame(train, meta), meta))
    queue = pd.DataFrame({"row": order, "anomaly_score": np.round(scores[order], 6),
                          "flagged": pipeline.predict(X_test.iloc[order]).astype(int),
                          "reasons": row_reasons(detector_inputs(X_test.iloc[order], meta), ref)})
    if meta.get("evaluation_available"):
        queue["known_anomaly"] = _label_to_binary(queue_rows[meta["label_column"]], meta).to_numpy()
    for key in split_keys(meta):  # who / when, for the reviewer
        queue[key] = queue_rows[key].to_numpy()
    queue.to_csv(_run_dir(run_id) / "review_queue.csv", index=False)
    meta["feature_importance"] = top
    meta["review_queue"] = {"rows": int(len(queue)), "path": str(_run_dir(run_id) / "review_queue.csv"),
                            **({"known_anomalies_in_queue": int(queue["known_anomaly"].sum())}
                               if "known_anomaly" in queue else {})}
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, "algorithm": meta.get("algorithm"), "method": "score_correlation",
                       "feature_importance": top, "review_queue": queue.head(10).to_dict(orient="records"),
                       "review_queue_path": meta["review_queue"]["path"]}, default=_json_default)


@mcp.tool()
def export_model(run_id: str, out_path: str = "", force: bool = False) -> str:
    """Export the current fitted pipeline as one self-contained joblib bundle."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )
    meta = _load_meta(run_id)
    metrics = _current_metrics(meta)
    # A run whose own evidence found a problem does not ship (core/gates.py).
    readiness = compute_readiness(run_id)
    if readiness["overall_status"] == "blocked" and not force:
        return json.dumps(
            {
                "error": "readiness gate BLOCKED this run — exporting it would ship a model with known defects.",
                "failed_gates": {
                    g: readiness["checks"][g]["evidence"]
                    for g in readiness["failed_gates"]
                },
                "remedy": "fix the failed gate(s) and re-run, or retry with force=true if the user explicitly accepts it.",
            }
        )
    if not out_path:
        (RUNS_DIR.parent / "artifacts").mkdir(parents=True, exist_ok=True)
        suggested = str(
            RUNS_DIR.parent / "artifacts" / f"{meta['algorithm']}_{run_id}.pkl"
        )  # data/artifacts: where serving looks
        return json.dumps(
            {
                "prompt": "Where should the .pkl be saved?",
                "suggested_out_path": suggested,
            }
        )
    if (
        meta.get("evaluation_available")
        and metrics.get("roc_auc") is not None
        and metrics["roc_auc"] < EXPORT_MIN_ROC_AUC
        and not force
    ):
        return json.dumps(
            {
                "error": f"roc_auc={metrics['roc_auc']} is below the export gate of {EXPORT_MIN_ROC_AUC} — this detector is close to random on the held-out labeled fold. Ask the user before exporting; retry with force=true if they confirm."
            }
        )
    pipeline = joblib.load(pipeline_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "algorithm": meta.get("algorithm"),
            "contamination": meta.get("contamination"),
            "label_column": meta.get("label_column"),
            "reason_reference": meta.get("reason_reference"),
            "reason_exclude": [*(meta.get("dropped_columns") or []), *split_keys(meta)],
            # Which model this is — serving stamps every score with it.
            "run_id": run_id,
            "readiness_at_export": readiness["overall_status"],
            "registry": meta.get("registry"),
        },
        out_file,
    )
    train, _, _ = _load_split(
        run_id
    )  # the drift baseline travels with the model: the run dir is swept
    profile = core_mlops.write_monitoring_profile(
        str(out_file),
        run_id,
        train,
        meta.get("label_column"),
        meta.get("dropped_columns"),
        meta.get("feature_importance"),
    )
    return json.dumps(
        {
            "out_path": str(out_file),
            "monitoring_profile": profile,
            "algorithm": meta.get("algorithm"),
            "contamination": meta.get("contamination"),
            "evaluation_available": meta.get("evaluation_available"),
        },
        default=_json_default,
    )


@mcp.tool()
def predict(pkl_path: str, data_path: str) -> str:
    """Score new data (any supported format) with an exported detector: an
    anomaly flag, the score, and for every flagged row the reasons — its
    most extreme columns against the training fold."""
    model_file, data_file = Path(pkl_path), Path(data_path)
    if not model_file.exists():
        return json.dumps({"error": f"no exported model at '{pkl_path}' — call export_model first"})
    if not data_file.exists():
        return json.dumps({"error": f"no data file at '{data_path}'"})
    bundle = joblib.load(model_file)
    pipeline = bundle["pipeline"]
    label_column = bundle.get("label_column") or ""
    X = read_table(data_file)
    if label_column and label_column in X.columns:
        X = X.drop(columns=[label_column])
    out = X.copy()
    out["anomaly"] = pipeline.predict(X).astype(int)
    out["anomaly_score"] = np.round(_score_vector(pipeline, X), 6)
    out["anomaly_reasons"] = ""
    flagged = out["anomaly"] == 1
    if bundle.get("reason_reference") and flagged.any():
        keep = [c for c in X.columns if c not in set(bundle.get("reason_exclude") or [])]
        out.loc[flagged, "anomaly_reasons"] = row_reasons(X.loc[flagged, keep], bundle["reason_reference"])
    out_file = data_file.with_name(f"{data_file.name.split('.')[0] or 'data'}_predictions.csv")
    if runtime.user():  # beside the input when allowed, else the user's predictions folder
        try:
            toolguard.check_output(str(out_file), "predict")
        except toolguard.Refused:
            out_file = toolguard.data_dir() / "predictions" / toolguard.user_folder(runtime.user()) / out_file.name
            toolguard.check_output(str(out_file), "predict")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)
    return json.dumps({"algorithm": bundle.get("algorithm"), "n_rows": len(out), "out_path": str(out_file),
                       "anomaly_counts": {str(k): int(v) for k, v in out["anomaly"].value_counts().items()},
                       "preview": out[flagged].head(5).to_dict(orient="records") or out.head(5).to_dict(orient="records")},
                      default=_json_default)


@mcp.tool()
def compare_runs(run_ids: str) -> str:
    """Compare previously trained runs without retraining."""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    rows = []
    for run_id in ids:
        try:
            meta = _load_meta(run_id)
        except FileNotFoundError as exc:
            rows.append({"run_id": run_id, "error": str(exc)})
            continue
        metrics = _current_metrics(meta)
        rows.append(
            {
                "run_id": run_id,
                "algorithm": meta.get("algorithm"),
                "contamination": meta.get("contamination"),
                "evaluation_available": meta.get("evaluation_available"),
                "roc_auc": metrics.get("roc_auc"),
                "average_precision": metrics.get("average_precision"),
                "precision_at_budget": metrics.get("precision_at_budget"),
                "stability": (metrics.get("stability") or {}).get("top_alert_jaccard_mean"),
                "anomaly_rate_gap": metrics.get("anomaly_rate_gap"),
            }
        )
    labeled = sorted(
        [r for r in rows if r.get("roc_auc") is not None],
        key=lambda r: (-(r.get("average_precision") or 0), -r["roc_auc"]),
    )
    unlabeled = sorted(
        [
            r
            for r in rows
            if r.get("roc_auc") is None and r.get("anomaly_rate_gap") is not None
        ],
        key=lambda r: r["anomaly_rate_gap"],
    )
    return json.dumps(
        {
            "runs": rows,
            "ranked": [r["run_id"] for r in labeled + unlabeled],
            "ranking_rule": "average precision (then roc_auc) descending with labels, otherwise anomaly_rate_gap ascending",
        },
        default=_json_default,
    )
