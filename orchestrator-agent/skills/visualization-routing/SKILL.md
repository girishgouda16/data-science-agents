---
name: visualization-routing
description: When and how to delegate to the remote visualization agent — any chart, plot, or "show me" request about a CSV, or turning another agent's numeric output into a chart. Load before the first delegate_to_visualization_agent call in a conversation.
---

# Visualization Routing

Delegate here for ANY chart / plot / "show me" / visualization request —
call `delegate_to_visualization_agent` with the user's request as the
`message` argument. This agent has no human-in-the-loop questions;
generating a chart doesn't destroy anything, so its replies are always final.

A **model-evaluation** chart needs a `run_id` from the agent that trained
the model — classification (confusion matrix, ROC, PR, SHAP beeswarm,
calibration), regression (predicted-vs-actual + residuals), forecasting
(forecast vs actual), clustering (2-D cluster map), anomaly (score
distribution with the threshold). Name the agent in the message so the right
chart is drawn. Each chart needs that agent's `run_id` — it reads that run's already-fitted model rather than fitting its
own. If the classification agent hasn't produced a `run_id` for this
dataset yet in this conversation, delegate there first (EDA →
`prepare_dataset` → `train_model`), then pass the `run_id` it returned
along in the message you send to the visualization agent. An
**exploratory** chart (histogram, boxplot, correlation heatmap, class
distribution, missingness) needs no `run_id` — just the file path.

A **global feature-importance bar chart** takes the `run_id` too — it
plots the importances `explain_model` persisted for that run. (Literal
`[[feature, value], ...]` pairs still work for a model with no run.)

Every `run_id`-keyed chart is also saved into that run's folder, keyed to
the fit it was drawn from: the report embeds it, champion promotion logs it
to MLflow, and the confusion matrix reports the `threshold` it was drawn
at — relay that number with it.
