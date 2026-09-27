Worked example for credit / lending — application scoring (approve or
decline) and behavioural scoring (an existing account's risk of default).
Fetch via `load_skill("feature-engineering", reference="credit")` once the
domain is credit risk / lending.

## Three traps before any feature

1. **The label exists only for approved applicants.** Declined applicants
   never got a loan, so they have no outcome, and the model learns default
   among the people the *old* policy approved — then scores everyone.
   It will look better on the test fold (also approved-only) than it
   behaves on the full applicant stream. Say this in `assumptions` and in
   the report; the remedies (reject inference, a random-approval sample)
   are business decisions, not something to fake with a model.
2. **Default is defined over a performance window** ("90+ days past due
   within 12 months of booking"). Loans booked less than one window ago
   have not had the chance to default: `prepare_dataset(time_column=
   <booking date>, immature_after=<newest booking date minus the window>)`.
3. **Application scoring may only use what was known at application.**
   Anything written after booking — payments, collections status,
   restructuring, limit changes, account closure — encodes the outcome.
   Behavioural scoring may use account history, but only up to its own
   scoring date.

## Credit / lending features

| Feature | spec | Archetype — mechanism |
|---|---|---|
| debt-to-income, utilisation | `apply_custom_feature` `monthly_debt / (monthly_income + 1)`, `balance / (limit + 1)` | 1 — capacity to absorb a shock, not the raw balance |
| bureau enquiries, last 6 months | enquiry events: `aggregate_events(at="event == 'application'", window="180D")` `{"agg": "count", "where": "event == 'enquiry'"}` | 5 — credit-seeking velocity: shopping for credit under stress |
| time since last delinquency | payment-month rows: `apply_entity_features` `{"op": "since", "column": "days_past_due"}` | 4 — recency of distress |
| on-time payment streak | `{"op": "streak", "column": "paid_on_time"}`, then `(payment_history_streak > 0) * (paid_on_time == 0)` for "just broke" | 6 — a broken streak predicts more than one isolated miss |
| utilisation trend | `roll_mean` of utilisation over "90D" and "365D", then ratio | 5 — rising reliance on credit |
| payment vs own baseline | `{"op": "vs_roll_mean", "column": "payment_amount", "window": "180D"}` | 2 — paying less than they usually do |
| open trade lines | trade records: `aggregate_events` `{"agg": "nunique", "column": "trade_id", "where": "status == 'open'"}` | 3 — thin file vs over-extended |

**Peers (7) with care:** income or utilisation against *region* or
*occupation* peers can proxy a protected characteristic. Credit decisions
fall under anti-discrimination law in most jurisdictions (ECOA in the US,
GDPR Article 22 in the EU) — run `assess_fairness`, and prefer peer keys
that are about the product (same limit band, same loan type) over keys
that are about the person.

## Before using the table

1. **State the mechanism.** For each feature, one sentence on why the
   behaviour it measures precedes default. A feature with no mechanism
   sentence is a correlation you have not tested yet.
2. **Name the innocent twin.** Every one of these features has a
   legitimate population that looks identical — and the discriminator that
   separates them is usually a better feature than the original.

Put both in each spec's `rationale` so they reach the report.

Twins worth knowing here: high utilisation is also a deliberate cash-flow
strategy by a low-risk borrower; a broken payment streak is also one
administrative error; a burst of enquiries is also rate-shopping for one
mortgage (bureaus often de-duplicate those within a short window);
a thin file is youth, not risk. Duration at high utilisation, whether the
miss was cured immediately, and whether the enquiries share one product
type are the usual discriminators.
