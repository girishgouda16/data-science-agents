---
name: serving
description: Discover, inspect, and predict from any agent's already-exported .pkl model. Load this once you have (or need to find) a path to an exported model and the user wants predictions, forecasts, or cluster/anomaly assignments from it.
---

# Serving

Tools: `list_exported_models(directory="")`, `inspect_model(pkl_path)`,
`predict(pkl_path, data_path="", horizon=0, future_exog_path="", out_path="")`.

All three are read-only with respect to the model itself — none of them
retrain, tune, or overwrite the `.pkl`; `predict` only writes a NEW
predictions/forecast CSV next to the input.

## Steps

1. **Find the model** (skip if the user already gave you a path) —
   `list_exported_models()` defaults to this repo's shared
   `data/artifacts` directory, where every ML agent's `export_model`
   suggests saving to. Pass `directory` for anywhere else.
2. **Check what it is** — `inspect_model(pkl_path)` before your first
   `predict` call on a model you didn't just export yourself this
   conversation. It reports `artifact_kind` (`sklearn_pipeline` or
   `forecast_model`) and the exact `call_shape` to use, so you're not
   guessing which arguments `predict` needs.
3. **Predict**:
   - `sklearn_pipeline` → `predict(pkl_path, data_path=<csv>)`. `data_path`
     should be shaped like the original training data (the target column
     is fine to leave in — it's dropped automatically if present). One
     exception: a clustering agent's DBSCAN export refuses this call
     outright (no valid out-of-sample predict for DBSCAN) — that's an
     error from the tool, not a bug, relay it as-is.
   - `forecast_model` → `predict(pkl_path, horizon=<int>)`. If the model
     has exogenous columns and you want them driven by real future values
     rather than the last known value carried forward flat, also pass
     `future_exog_path` — a CSV with exactly `horizon` rows for those
     columns, in that order.
4. **Report** — `out_path`, row/horizon count, and the tool's own
   `preview` (already a small head-of-file sample) are enough; don't
   re-read the full output CSV yourself.
