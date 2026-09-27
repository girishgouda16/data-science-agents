---
name: business-understanding
description: Settle what an anomaly IS here and who works the alerts before touching data — the decision an alert triggers, the review capacity that sets the threshold, the kind of anomaly (point, contextual, collective), where any labels came from, and whether this is really a classification problem. Load FIRST on any new anomaly request.
---

# Business Understanding (anomaly)

Tool: `record_business_context(run_id, business_objective,
target_definition, success_criteria, domain, success_metric,
success_threshold, direction, assumptions, clarifications)` — call it right
after `prepare_dataset`; everything settled before goes in.
(`target_definition` here is what counts as an anomaly and what one alert
leads to.)

A detector is judged by the queue it hands a person. Work through these
before `eda`:

1. **What does an alert trigger?** A fraud analyst's review, a line
   barred, a field engineer sent, a ticket. The cost of a false alarm is
   that action wasted; the cost of a miss is the loss it would have
   stopped. Record both, in words.
2. **Review capacity — this sets the threshold.** "How many alerts can the
   team work per day, and how many records arrive per day?" goes straight
   into `propose_contamination(run_id, review_capacity, rows_per_period)`.
   A threshold set from anything else produces a queue nobody works.
3. **Which kind of anomaly?**
   - *point* — one record is extreme on its own (a 4-hour call to a
     premium number): the row as it is may be enough
   - *contextual* — normal elsewhere, abnormal for THIS subscriber, cell
     or hour (3 GB at 3 a.m. on a line that never uses data): needs
     own-baseline and peer features (`feature-engineering`)
   - *collective* — each record is ordinary, the pattern is not (hundreds
     of 1-second calls from one SIM = Wangiri; perfectly regular gaps =
     SIM-box): needs event rows aggregated per entity
   The kind decides the features far more than the algorithm does.
4. **Labels — where did they come from?** Investigated cases, a rule, or
   customer complaints. A rule-generated label makes the rule's inputs
   leakage, not signal; complaint labels only see what customers noticed.
   Unflagged rows are unlabelled, not confirmed normal — measured
   precision is a floor. Ask when unknown; record the answer.
5. **Is this a classification problem?** With a few hundred investigated
   anomalies of the kind that matters and a stable definition, a
   supervised classifier will beat any detector — say so and suggest the
   classification agent. A detector earns its place when anomalies are
   rare, varied, unlabelled or new.
6. **What must not drive an alert?** Protected attributes and their
   proxies when an alert leads to action against a person (a barred line).
7. **Bar** — with labels: precision or recall of the queue at the budget
   (`precision_at_budget`, `recall_at_budget`) or `average_precision`.
   Without labels there is no measurable bar: success is a person's review
   of the queue. Never invent a number.

## Ask, or decide and record

**Ask** what defines a useful alert: the action, the review capacity, the
anomaly kind when unclear, the label source, excluded attributes, the bar.
**Decide and record** everything a measurement settles: imputation,
identifier drops, the algorithm (`compare_detectors`), training on
known-normal rows. Put each in `assumptions` (one per line), each answered
question in `clarifications`; the report shows both.
