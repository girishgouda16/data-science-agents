---
name: business-understanding
description: Identify ambiguity in the business problem before touching data — target definition, prediction unit/horizon, feature meaning, success criteria. Ask the user when the ambiguity is critical; document an assumption and proceed when it isn't. Load this FIRST, before eda, on any new request.
---

# Business Understanding

**MCP module:** `mcp_server/business.py`

Tool: `record_business_context(run_id, business_objective, target_definition, success_criteria, domain, label_provenance, assumptions, clarifications, success_metric, success_threshold, success_direction)`.

**Identify the domain before writing success criteria.** Look at the
column names and the request (fraud/churn/diagnosis/default-risk/etc. are
usually obvious from either). If it's a plausible match for one of
`modeling/references/on_demand/domain-notes.md`'s sections (fraud/AML,
healthcare, credit/fintech, telecom/churn), fetch it now — `load_skill(
"modeling", reference="domain-notes")`. This does NOT load the `modeling`
skill or unlock its tools (fetching a reference is a read, not a load) —
it's safe to call from here, before a `run_id` even exists. Use its
threshold-floor guidance to inform `success_criteria` (a domain's standard
recall floor or explainability requirement IS business-relevant success
criteria, not just a modeling detail) and its fairness-relevant-attributes
list to flag, in `assumptions` or a `clarifications` entry, which column(s)
`diagnostics`'s `assess_fairness` should check later. Pass whatever domain
label you land on — one of domain-notes.md's section names, or a short
free-text label if none matches, or leave it empty if genuinely unclear —
as `record_business_context`'s `domain` argument. **This is the run's one
declared domain**: `modeling`, `feature-engineering`, and `diagnostics` all
read it back from `meta["business_understanding"]["domain"]` instead of
each re-guessing it independently from column shapes, which is how two
skills could otherwise land on different answers for the same dataset.

This is the gate that keeps the agent from being "just a pipeline that runs
on whatever column looks like a target." A model can be technically correct
(good CV score, no leakage, well-calibrated) and still be the wrong model
for the business question, if nobody checked the question was well-defined.

## What to check, before calling `eda`/`prepare_dataset`

Look at the request and the raw columns (`eda(path)`, `inspect_target(path,
target)`) and ask yourself, explicitly, one at a time:

1. **Business objective** — what decision or action will this model's
   predictions actually drive? If the request doesn't say, don't guess a
   generic one ("predict churn" isn't a decision — "flag likely churners
   for a retention call before renewal" is).
2. **Target definition** — does the target column mean what the request
   assumes it means? A column named `churn` might mean "cancelled" or
   "no usage in 90 days" or "didn't renew" — these lead to different
   features and different costs of a false negative. A column named
   `fraud` might be self-reported, investigated-and-confirmed, or a proxy
   (chargeback) — very different label quality.
2b. **Label provenance — how were these labels PRODUCED?** Distinct from
   check 2, which asks what the label means. This asks who or what wrote
   it, and it is the check most likely to invalidate a whole run when it is
   skipped. Three answers, three different consequences:

   - **Investigated** — a human confirmed each positive. Trust the labels
     and move on.
   - **Rule-generated** — a threshold or heuristic wrote them. Then *every
     input to that rule is leakage, not a feature.* If fraud was flagged by
     "more than 500 calls in a day", any volume feature reproduces the rule,
     the model scores near-perfectly, and it has learned the rule rather
     than the fraud. Ask which inputs the rule used and exclude them. This
     is the single most common way a telecom fraud run produces a 0.99 AUC
     that means nothing.
   - **Proxy** — a downstream event stands in for the real thing (a
     chargeback for fraud, no-usage-in-90-days for churn). The label lags
     the event and is blind to everything the proxy never caught.

   Then ask the second half: **are the negatives confirmed, or merely
   un-flagged?** In fraud of every kind they are almost always un-flagged,
   which makes the problem positive-unlabeled. Some share of the negative
   class is undetected positives, so measured recall is an upper bound on a
   number nobody can observe, and a confident "false positive" in
   `error_analysis` may be the model finding what the labelling process
   missed. Say this in `success_criteria` when it applies — a precision
   floor set against contaminated negatives is a floor against the wrong
   number.

   Record the answer in `label_provenance`. If it is genuinely unknown,
   leave it empty and note *that* in `assumptions` — "we don't know how
   these labels were made" is itself a material finding about label
   quality, not a blank to be quietly skipped. This is worth a round-trip
   on the same footing as the success bar.

3. **Prediction unit** — one row = one what? A customer, a transaction, a
   customer-month? This determines whether the run needs a `group_column`
   (see the `eda` skill) and how metrics should be read.
4. **Prediction horizon** — churn in the next 30 days, or ever? Fraud in
   this transaction, or this account over time? If the raw data doesn't
   make this explicit and it changes what a "positive" label means, that's
   a real ambiguity, not a detail.
5. **Feature meaning** — any column whose name is opaque, or whose values
   don't match its name's obvious interpretation (e.g. a `score` column
   that's actually itself a downstream label or a leakage risk — cross-
   check with `detect_data_leakage` once `diagnostics` is loaded).
5b. **Operational capacity** — how many positives can actually be acted on?
   A fraud queue reviews N alerts a day; a retention team makes N calls a
   week. This is business framing, not a modelling detail, and it is the
   input `tune_threshold`'s `max_alert_rate` mode needs. Without it the
   agent falls back to a recall floor, which answers a different question:
   a recall floor asks how much you catch, capacity asks how much work you
   create. A model that flags 23,000 cases for a team that can review 6,000
   has an effective recall of whatever those 6,000 contain, regardless of
   what the report says. Record the number in `assumptions` when the user
   gives one, and ask when the action is obviously capacity-bound.

6. **Success criteria** — what's "good enough to ship" in business terms,
   not just an ML metric name? "High accuracy" is not success criteria;
   "recall >= 0.8 on the fraud class, because missing a fraud case costs
   more than reviewing a false alarm" is. If the domain matches a
   `domain-notes.md` section, start from its threshold floor rather than
   inventing one — e.g. healthcare's `target_recall >= 0.95` default —
   and only deviate from it with a stated reason.

   Record it in BOTH forms. The prose goes in `success_criteria`; the
   machine-checkable bar goes in `success_metric` + `success_threshold`
   (+ `success_direction`), which is the only bar the report will judge the
   model against.

   **A threshold you do not have is not a threshold you may supply later.**
   If the user stated no bar and the domain supplies none, leave
   `success_metric`/`success_threshold` empty. The report will then say
   plainly that no bar was set and let the metrics stand on their own —
   which is the honest outcome. What must never happen is a number
   materialising at report time: one run announced *"Success criteria not
   met: ROC-AUC 0.822 vs the 0.85 target"* for a 0.85 the user had never
   mentioned and no tool had ever recorded. That is a fabricated finding
   wearing the clothes of a measured one, and it is worse than reporting no
   bar at all, because a reviewer cannot tell the difference by reading it.

   Asking for the bar is one of the few questions always worth a round-trip
   (see the top-level HITL rule): nothing downstream can validate a choice
   that defines what "better" means.

## Critical vs non-critical ambiguity

Not every open question needs a human round-trip — that would make the
agent slower without making it more correct.

- **Critical** (call `ask_user`): the ambiguity changes which column is the
  target, what a row represents, or what the pass/fail bar is. Getting
  this wrong invalidates the entire run, not just one step.
- **Non-critical** (document as an assumption, proceed): a reasonable
  default exists and the cost of being wrong is small or easily corrected
  later (e.g. "assumed the default 80/20 test split is fine since the
  request didn't specify one").

When in doubt, treat it as critical once — the cost of an unnecessary
question is one round trip; the cost of an invalidated run is the whole
pipeline.

That tiebreaker applies to **business framing only**, which is what this
skill governs. Do not carry it into the mechanical steps that follow: there,
the rule inverts, because a measurement settles the question better than a
human can. Dropping an identifier-shaped column, adding a standard
row-wise feature, or leaving outliers alone for a tree model are all
decisions the next step evaluates for you — decide them, record the
assumption, and keep moving. See the top-level SKILL.md's HITL rule for the
full test.

## Sequence

1. `eda(path)` + `inspect_target(path, target)` (from the `eda` skill —
   load it too) to see what you're actually working with.
2. Work through the eight checks above. For each critical ambiguity found,
   call `ask_user` with the question, your recommended interpretation, and
   an explicit "go with your recommendation" option. For each non-critical
   one, note the assumption you're making instead of asking.
3. Proceed to `prepare_dataset` (via the `eda` skill) once the target and
   framing are settled.
4. **Immediately after** `prepare_dataset` returns a `run_id`, call
   `record_business_context(run_id, ...)` to persist everything from step 2
   — this is what lets `generate_report` show a human reviewer the business
   framing and every clarification asked, without them needing the chat
   transcript. Do this even when nothing needed asking (empty
   `clarifications`, since the objective/target definition/success
   criteria are still worth recording) — a report with no business-context
   section reads as "nobody thought about this," not "nothing to say."

Then continue to `eda`'s remaining steps and the rest of the pipeline.
