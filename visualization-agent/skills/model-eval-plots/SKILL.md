---
name: model-eval-plots
description: Charts about how a trained model behaves — what drives its predictions, where it makes errors, how well it separates classes, how close a forecast lands, how separable the clusters are, where the anomaly threshold falls. Load this for any run_id-based chart (classification, regression, forecasting, clustering, anomaly), instead of exploratory-plots.
---

# Model Evaluation Plots

Tools: `plot_feature_importance`, `plot_confusion_matrix`, `plot_roc_curve`, `plot_pr_curve`, `plot_shap_beeswarm`, `plot_calibration_curve`,
`plot_regression_diagnostics`, `plot_forecast`, `plot_clusters`, `plot_anomaly_scores`.

## Which chart for which agent's run

| run from | evaluation chart | also |
|---|---|---|
| classification | `plot_confusion_matrix`, `plot_roc_curve`, `plot_pr_curve`, `plot_calibration_curve` | `plot_shap_beeswarm`, `plot_feature_importance` |
| regression | `plot_regression_diagnostics` — predicted vs actual, residuals vs predicted, residual distribution | `plot_feature_importance` |
| forecasting | `plot_forecast` — history, held-out actuals, the forecast its test RMSE was scored on | `plot_feature_importance` (xgboost runs) |
| clustering | `plot_clusters` — PCA 2-D map coloured by cluster, cluster sizes | — |
| anomaly | `plot_anomaly_scores` — score distribution, the flagging threshold, normal vs labelled anomalies | `plot_feature_importance` |

Each refuses a run_id from the wrong agent ("isn't a regression run") instead of drawing nonsense — pick by the agent
that produced the run. Relay what the chart shows, not just the image: a funnel in the residuals, a forecast that
lags the turns, clusters that overlap in the 2-D map (say the map keeps only `pca_variance_explained` of the
variance), labelled anomalies scoring below the threshold (misses).

## Picking the right chart

- **What's driving a model's predictions** → `plot_feature_importance(run_id=...)`.
  Pass the `run_id`: the chart is then read from the importances the
  training agent persisted, exactly like every other chart here. Only fall
  back to the `importances=` JSON list when there is no run to read from —
  that path plots whatever numbers it is given, with nothing checking them
  against a fitted model, and the result marks itself
  `source: caller-supplied`. Say so if you relay one.
- **How well a model classifies (errors by class)** → `plot_confusion_matrix(run_id, out_path)`.
- **How well a model separates classes across thresholds** →
  `plot_roc_curve(run_id, out_path)`. Binary target → one curve; more than
  two classes → one-vs-rest curve per class, each with its own AUC.
- **Precision/recall tradeoff across thresholds** → `plot_pr_curve(run_id, out_path)`.
- **Per-sample, per-feature SHAP impact (not just a mean-|SHAP| bar)** →
  `plot_shap_beeswarm(run_id, out_path, background_size=200)` — the global
  ranking you'd get from `plot_feature_importance(explain_model(method="shap"))`
  collapsed to one number per feature; this shows the full spread (direction
  and magnitude per sample) instead. `background_size` is a stratified
  sample of the test fold — raise it on an imbalanced target so the rare
  class stays represented (see the classification agent's `explain_model`
  docstring for the same tradeoff).
- **Are the model's predicted probabilities trustworthy** →
  `plot_calibration_curve(run_id, out_path)` (binary targets only) — check
  this before the classification agent's `tune_threshold` picks a decision
  threshold off those probabilities; a miscalibrated model makes threshold
  tuning misleading. The classification agent has the measured counterpart
  (`check_calibration`: Brier score and expected calibration error, which
  its readiness gate records); this chart shows the *shape* of the
  miscalibration that number summarises — over-confidence at the top of
  the range reads very differently from a flat, uninformative curve.

## Every binary chart plots the run's positive class

`plot_roc_curve`, `plot_pr_curve`, `plot_calibration_curve` and
`plot_shap_beeswarm` resolve the positive class from the run's
`meta["positive_label"]` — the same label the classification agent's
`recall_positive`, operating point and fairness TPR refer to — and name it
in the chart legend and the tool result. They do **not** assume the
higher-sorting label: on a `{"churn","no_churn"}` target that would be
`no_churn`, and the chart would contradict the report that embeds it.

If a chart's stated positive class isn't the event the user cares about,
that's a `prepare_dataset(positive_label=...)` problem in the
classification agent, not something to work around here.

The same holds for the other agents' charts: each needs a `run_id` the matching training agent trained.

`plot_confusion_matrix`/`plot_roc_curve`/`plot_pr_curve`/`plot_shap_beeswarm`/
`plot_calibration_curve` need a `run_id` from the classification agent —
you can't plot these without the classification agent having called
`prepare_dataset` and `train_model` for that dataset first (ask the user to
run classification first, or delegate that step, if they only asked for a
chart). They read the **same fitted pipeline** the classification agent
trained (tuned, if `tune_hyperparams` ran) — they never fit their own
model, so the chart always matches what that agent actually reported, not
a coincidentally-similar fresh fit.

Each tool returns a `markdown` field — paste it verbatim on its own line so
the chart renders inline; see the base SKILL.md's Output format.
