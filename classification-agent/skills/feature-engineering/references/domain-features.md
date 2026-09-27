# Deriving features — the method for any domain

A feature worth building encodes *why* the target takes the value it does —
why a row is positive, or why a value is high: the thing the fraudster
cannot avoid doing, the process a device cannot hide, the experience a
customer reacts to, the usage that drives next month's bill. Generic statistics belong in
data-cleaning; correlation mining (`propose_features`) is the last resort.
A domain expert facing a dataset they have never seen does not recall a
formula — they ask a handful of questions about the data. Those questions
are the archetypes below, and every one of them maps to a tool.

## 1. The data's shape picks the tool

| Shape | Example | Tool | When |
|---|---|---|---|
| Event rows; label per entity (or entity × period) | calls → one row per SIM, or per caller-day | `aggregate_events` (default / `period`) | before `prepare_dataset` |
| Event rows; the decision is ONE event inside a wider stream | a SIM change among care contacts and logins | `aggregate_events(at=..., window=...)` | before `prepare_dataset` |
| Every row is a decision; the entity repeats over time | transactions, caller-day, subscriber-month | `apply_entity_features`, `apply_peer_features` | right after `prepare_dataset`, before data-cleaning (so cleaning imputes the first-period gaps) |
| One row per entity, no time axis | a customer snapshot | `apply_peer_features`, `apply_custom_feature` | after `prepare_dataset` |

On top of any shape, `apply_custom_feature` builds ratios and differences
between columns that exist. Changing the grain changes the prediction
unit — business framing, so settle it with the user if it isn't settled.

## 2. Seven archetypes — the questions a feature can ask

| # | Archetype | The question | Build it with |
|---|---|---|---|
| 1 | **Share of a sub-behaviour** | What fraction of the entity's activity is the concerning kind? (a ratio, not a raw count — a big count is not a bad ratio) | `share` + `where`; or an `apply_custom_feature` ratio |
| 2 | **Deviation from its own baseline** | Is this unusual for THIS entity? | `vs_roll_mean` (`window`: "30D", or `n` rows); `pct_change` |
| 3 | **Counterparties — diversity, concentration, novelty** | Who does it deal with: many strangers, one target, someone new? | `nunique`, `entropy`, `top_share`; `prior_count` (0 = first time with this payee, country, device) |
| 4 | **Timing — recency, spacing, rhythm** | How long since the last X? How regular? When in the day? | `since`, `recency_days`, `gap_prev`; `_gap` with mean/min/max/`cv`/`periodicity`; `_hour` `entropy`; `share` where `"_hour < 6"` |
| 5 | **Trend and velocity** | Is it getting worse, and how fast is it happening right now? | two `roll_mean`s (short, long) → ratio; `roll_count` over "1h"/"24h" |
| 6 | **Streak, and whether it just broke** | How long has a pattern held, and did it just change? | `streak`; then `apply_custom_feature` for "broke": `(streak > 0) * (value == 0)` |
| 7 | **Deviation from its peers** | Is this unusual for its KIND — same cell, plan, device model, merchant category? | `apply_peer_features` (`ratio` / `zscore`, fitted on the training fold) |

2 and 7 answer different questions and both matter: a subscriber whose
drop rate doubled (2) on a cell where everyone's did (7) has a network
problem, not a personal one. Use 2 for "something changed", 7 for "this is
abnormal for its type".

## 3. The loop — derive, build, check, revise

1. **Derive.** For the declared domain, fetch its file (section 5). For a
   domain with no file, write the mechanism in one sentence (how does the
   actor get paid, or what process produces the behaviour), then walk the
   seven archetypes against the columns `eda` surfaced.
2. **Build.** State the plan in ONE message — each feature's spec, its
   archetype, its mechanism, its innocent twin; what you pruned and why;
   what the schema lacks (a data request) — then apply it. Don't gate the
   list: the next steps measure whether each feature earned its place.
3. **Check** with `profile_features(run_id)` before modelling:
   `leakage_suspect` → ask what produced the label: a rule over the same
   inputs (the model will learn the rule), or data written after the
   outcome? If labels were investigated and the feature is a mechanism
   feature, it can be this strong — keep it, and record why with
   `acknowledge_identifier_column(run_id, column, justification)`, or the
   leakage gate fails the run;
   `constant` / `mostly_missing` → the column, `where` or window is wrong;
   `weak_alone` → keep it if it discriminates a twin (those are weak alone
   by design), otherwise question the mechanism.
4. **Revise** after the first model: when mechanism features lose to raw
   columns, the mechanism was wrong — change the hypothesis, not the
   formula.

## 4. Rules for every feature

- **Earlier data only.** Every window closes before the decision: `before`
  for whole-history aggregates, `at` for decision events, and
  `apply_entity_features` reads earlier rows by construction. Never lag
  the target — the tools refuse it.
- **Settled labels only.** Fraud, bad debt and default are confirmed
  weeks after the event; rows younger than that are unlabelled, not
  negative. Pass `immature_after` to `prepare_dataset`.
- **Time windows, not row windows,** unless there is exactly one row per
  fixed period: on transactions or active-days-only files, "the last 7
  rows" is not "the last 7 days".
- **Normalise by exposure.** An entity observed for 3 days and one for 30
  are not comparable on counts — divide by `span_days` or active days.
- **Name the innocent twin** — the legitimate population that looks the
  same — and the discriminator that separates them. The discriminator is
  often the better feature. Put mechanism, twin and discriminator in every
  `rationale`; the report prints it.
- **Check the purpose.** Some data may be used for fraud prevention but
  not for marketing. Record which features the use allows.

## 5. Domain files — fetch the ONE that matches

`load_skill("feature-engineering", reference="<name>")`:

- **`telecom`** — procedure and feature vocabulary for xDRs, journeys,
  recharges and network data: voice and SMS fraud, SIM swap, activation
  fraud, IoT/M2M, churn in any market. **`telecom-traces`** — four worked
  examples; read them after drafting your own candidates, not instead.
- **`fraud`** — transaction / AML fraud (banking, payments).
- **`credit`** — lending and underwriting.

No match: apply the archetypes directly. For example, hospital
readmission — visits are events and the discharge is the decision:
`aggregate_events(at="event == 'discharge'", window="365D")` with
emergency-visit `share` (1), admissions `count` in the year (5),
`recency_days` since the previous discharge (4); then
`apply_peer_features` for length of stay against the same diagnosis group
(7).
