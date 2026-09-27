"""Business-understanding gate: ambiguities in the business problem —
target definition, prediction unit/horizon, feature meaning, success
criteria — get surfaced and resolved (or explicitly assumed) BEFORE
data-cleaning/modeling start, not discovered after a model is already
trained. See the `business-understanding` skill for when to call ask_user
versus documenting an assumption instead."""

import json

from .core import _load_meta, _save_meta, mcp
from .gates import SUPPORTED_SUCCESS_METRICS


@mcp.tool()
def record_business_context(
    run_id: str,
    business_objective: str,
    target_definition: str,
    success_criteria: str,
    domain: str = "",
    label_provenance: str = "",
    assumptions: str = "[]",
    clarifications: str = "[]",
    success_metric: str = "",
    success_threshold: str = "",
    success_direction: str = ">=",
) -> str:
    """Persists the business-understanding gate's outcome onto this run's
    meta.json, so generate_report can render it without depending on
    conversation history. Call once, right after prepare_dataset, before
    data-cleaning/modeling start.

    business_objective: one or two sentences — what decision or action this
      model's predictions will drive (e.g. "flag likely churners for a
      retention call before their contract renews").
    target_definition: what the target column actually means in business
      terms, and any non-obvious definitional choice made (e.g. "churn = no
      usage in 90 days, not contract cancellation — the raw label conflates
      the two").
    success_criteria: the business-aligned bar for "good enough to ship" —
      not just an ML metric name, but why it matters (e.g. "recall >= 0.8 on
      the churn class — missing a churner costs more than a false alarm").
      Inform this from `modeling/references/on_demand/domain-notes.md` (fetch
      via `load_skill("modeling", reference="domain-notes")`) when `domain`
      matches one of its sections, rather than picking a floor from scratch.
    domain: this run's business domain, e.g. "fraud", "healthcare", "credit",
      "telecom" — match one of domain-notes.md's section names when it
      applies; use a short free-text label ("insurance claims", "retail
      returns", ...) when it doesn't, or leave empty if genuinely unclear.
      THE single declared domain for this run — modeling, feature-engineering,
      and diagnostics all read this back (via meta["business_understanding"]
      ["domain"]) instead of each independently re-guessing it from column
      names, which is how two skills could otherwise land on different
      answers for the same dataset.
    label_provenance: HOW the target column's values were produced, and
      whether the negatives are trustworthy. Three answers change everything
      downstream:
        - "investigated" — a human confirmed each positive. Trust the labels.
        - "rule-generated" — a threshold or heuristic wrote them. Then every
          input to that rule is LEAKAGE, not a feature: if fraud was flagged
          by "more than 500 calls a day", any volume feature reproduces the
          rule and the model scores near-perfectly while learning nothing.
          Name the rule's inputs here so they can be excluded.
        - "proxy" — a downstream event stands in for the real label
          (a chargeback for fraud, no-usage-in-90-days for churn). The label
          lags the event and misses everything the proxy never caught.
      Also state whether the negatives are CONFIRMED negatives or merely
      un-flagged rows. In fraud of every kind they are almost always the
      latter, which makes the problem positive-unlabeled: some share of the
      negative class is undetected positives, so measured recall is an upper
      bound on a number nobody can observe, and a "false positive" in error
      analysis may be the model finding what the labelling process missed.
      Leave empty only if genuinely unknown — and say so in assumptions,
      because "unknown" is itself a material finding about label quality.
    assumptions: JSON list of strings — documented assumptions made instead
      of asking, because the ambiguity was judged non-critical (e.g.
      "assumed 'recent' in the request means the last 30 days of activity").
    clarifications: JSON list of {"question": str, "answer": str, "impact":
      str} — every ask_user round actually used to resolve a business
      ambiguity, and what it changed about the plan. Empty list if nothing
      needed asking.

    success_metric / success_threshold / success_direction: the MACHINE
      -CHECKABLE form of success_criteria, and the only bar the report is
      allowed to judge the model against. success_metric is one of
      roc_auc, pr_auc, accuracy, f1_weighted, precision_weighted,
      recall_weighted, f1_positive, precision_positive, recall_positive;
      success_threshold is a number as a string ("0.80"); success_direction
      is ">=" (default) or "<=".

      Record these whenever a real bar exists — the user stated one, or the
      domain supplies one (see modeling/references/on_demand/domain-notes.md).
      Leave them EMPTY when no bar was actually established. Empty is a
      truthful "no target was set", and the report will say exactly that.
      What must never happen is a threshold appearing for the first time at
      report time: a run once announced "ROC-AUC 0.822 vs the 0.85 target"
      for a 0.85 the user never set, which is a fabricated finding presented
      as a measured one. A number here is a promise the user or the domain
      actually made it.
    """
    meta = _load_meta(run_id)
    try:
        assumptions_list = json.loads(assumptions)
        clarifications_list = json.loads(clarifications)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"assumptions/clarifications must be JSON: {e}"})

    success_target = None
    if success_metric or success_threshold:
        if success_metric not in SUPPORTED_SUCCESS_METRICS:
            return json.dumps(
                {
                    "error": f"success_metric must be one of {list(SUPPORTED_SUCCESS_METRICS)}, got '{success_metric}'"
                }
            )
        try:
            threshold = float(success_threshold)
        except ValueError:
            return json.dumps(
                {
                    "error": f"success_threshold must be a number, got '{success_threshold}'"
                }
            )
        if success_direction not in (">=", "<="):
            return json.dumps(
                {
                    "error": f"success_direction must be '>=' or '<=', got '{success_direction}'"
                }
            )
        success_target = {
            "metric": success_metric,
            "threshold": threshold,
            "direction": success_direction,
            "rationale": success_criteria,
        }

    meta["business_understanding"] = {
        "business_objective": business_objective,
        "target_definition": target_definition,
        "success_criteria": success_criteria,
        "success_target": success_target,
        "domain": domain or None,
        "label_provenance": label_provenance or None,
        "assumptions": assumptions_list,
        "clarifications": clarifications_list,
    }
    _save_meta(run_id, meta)
    return json.dumps({"run_id": run_id, **meta["business_understanding"]})
