"""serving agent — `serving` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _load_artifact, _predict_forecast, _predict_sklearn  # noqa: F401


@mcp.tool()
def list_exported_models(directory: str = "") -> str:
    """Lists .pkl files available to serve. Defaults to this repo's shared
    data/artifacts directory (where every ML agent's export_model tool
    suggests saving to) — pass directory to look elsewhere. Read-only,
    does not load or validate any file's contents; call inspect_model on a
    specific path for that."""
    target = Path(directory) if directory else ARTIFACTS_DIR
    if not target.exists():
        return json.dumps(
            {"error": f"no directory at '{target}'", "directory": str(target)}
        )
    files = sorted(target.glob("*.pkl"))
    return json.dumps(
        {
            "directory": str(target),
            "models": [
                {
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        f.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
                for f in files
            ],
        }
    )


@mcp.tool()
def inspect_model(pkl_path: str) -> str:
    """Loads an exported .pkl and reports what it is (an sklearn-Pipeline
    bundle from classification/regression/clustering/anomaly-agent, or a
    forecasting-agent ForecastModel) and its metadata — without predicting
    anything. Call this before predict() on a model you didn't export
    yourself this session, to see which arguments it needs."""
    obj, error = _load_artifact(pkl_path)
    if error:
        return json.dumps(error)

    if isinstance(obj, ForecastModel):
        return json.dumps(
            {
                "artifact_kind": "forecast_model",
                "model_type": obj.model_type,
                "date_column": obj.date_column,
                "target": obj.target,
                "exog_columns": obj.exog_columns,
                "freq": obj.freq,
                "seasonal_periods": obj.seasonal_periods,
                "lags": obj.lags,
                "rolling_windows": obj.rolling_windows,
                "history_rows": len(obj.history_target),
                "history_ends_at": str(pd.to_datetime(obj.history_dates.iloc[-1])),
                "call_shape": "predict(pkl_path, horizon=<int>, future_exog_path=<optional csv>)",
            }
        )

    if isinstance(obj, dict) and "pipeline" in obj:
        pipeline = obj["pipeline"]
        feature_columns = list(getattr(pipeline, "feature_names_in_", []))
        return json.dumps(
            {
                "artifact_kind": "sklearn_pipeline",
                "algorithm": obj.get("algorithm") or obj.get("model"),
                "target_column": obj.get("target") or obj.get("label_column"),
                "feature_columns": feature_columns
                or "unknown — send the same raw columns used at training time, minus the target column",
                "supports_confidence": hasattr(pipeline, "predict_proba"),
                "supports_score": hasattr(pipeline, "decision_function"),
                # Without this, a caller cannot discover that the model ships with a
                # tuned decision threshold at all — and an undiscoverable threshold
                # is one nobody can verify got applied.
                "positive_label": obj.get("positive_label"),
                "operating_point": obj.get("operating_point"),
                "decision_rule": (
                    f"predict() will apply the exported threshold {obj['operating_point']['threshold']} to "
                    f"P(positive), not sklearn's 0.5"
                    if (obj.get("operating_point") or {}).get("threshold") is not None
                    else "no tuned operating point — predict() will use sklearn's default 0.5 argmax"
                ),
                "readiness_at_export": obj.get("readiness_at_export"),
                "call_shape": "predict(pkl_path, data_path=<csv>)",
            }
        )

    return json.dumps(
        {
            "error": f"'{pkl_path}' is not a recognized serving-agent artifact shape "
            "(expected an sklearn-Pipeline bundle dict or a forecasting ForecastModel)"
        }
    )


@mcp.tool()
def predict(
    pkl_path: str,
    data_path: str = "",
    horizon: int = 0,
    future_exog_path: str = "",
    out_path: str = "",
    review_threshold: float = 0.0,
    max_rows: int = 0,
) -> str:
    """Serves predictions from ANY agent's exported .pkl — dispatches
    automatically on what kind of artifact it is:
      - sklearn-Pipeline bundle (classification/regression/clustering/
        anomaly-agent): pass data_path, a CSV shaped like that agent's
        original training data (target column optional, dropped if
        present). Writes one row per input row to a CSV next to
        data_path — "prediction" plus "confidence" (predict_proba) or
        "score" (decision_function) when the pipeline supports either.
        If the bundle carries a tuned operating point, predict applies
        THAT threshold to P(positive class) rather than sklearn's 0.5 —
        the shipped model is the one that was reviewed. The rule used is
        always reported back in "decision_rule"; check it.
        Pass review_threshold (0-1) on a classifier for the inference-time
        HITL gate: rows within that band of the decision threshold (in
        probability units, so 0.05 means "within 5 points of the cutoff")
        get needs_review=True and a companion "*_review_queue.csv". On a
        model with no tuned threshold it degrades to max-class confidence
        below the value, which is a weak proxy on an imbalanced target —
        "review_queue.basis" states which rule was used. Requires
        predict_proba; 0 (default) disables it.
      - forecasting-agent ForecastModel: pass horizon (steps beyond
        wherever training ended) and, only if the model has exogenous
        columns, optionally future_exog_path (a CSV with exactly
        `horizon` rows) — omitted exogenous columns carry their last
        known value forward flat.
    max_rows (sklearn bundles): score only the first N rows of data_path
    (0 = all). Numeric predictions come back summarised in
    "prediction_summary" (count/mean/std/min/quartiles/max).
    Call inspect_model first if you're not sure which shape a given
    pkl_path is or what it expects."""
    obj, error = _load_artifact(pkl_path)
    if error:
        return json.dumps(error)

    if isinstance(obj, ForecastModel):
        return _predict_forecast(obj, pkl_path, horizon, future_exog_path, out_path)

    if isinstance(obj, dict) and "pipeline" in obj:
        return _predict_sklearn(
            obj, data_path, out_path, review_threshold, pkl_path, max_rows
        )

    return json.dumps(
        {
            "error": f"'{pkl_path}' is not a recognized serving-agent artifact shape "
            "(expected an sklearn-Pipeline bundle dict or a forecasting ForecastModel)"
        }
    )
