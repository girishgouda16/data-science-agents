Worked example for transaction/AML fraud (banking, fintech, payments — not
telecom fraud, see `telecom` for Wangiri/IRSF/SIMbox/SIM swap). Fetch via
`load_skill("feature-engineering", reference="fraud")` once you know the
domain is transaction-based fraud/AML.

## Shape and timing

One row per transaction, label per transaction, the account repeating over
time: `prepare_dataset(group_column="account", time_column="ts",
immature_after=<newest date minus the chargeback window>)`, then
`apply_entity_features` before data-cleaning. Chargebacks and confirmed
fraud arrive 30–120 days late; rows younger than that are unlabelled, not
legitimate. Every window below is a time `window` — transactions are
irregular, so "the last 10 rows" is not a period.

## Fraud / AML (transactions)

| Feature | spec (apply_entity_features) | Archetype — mechanism |
|---|---|---|
| amount vs own baseline | `{"op": "vs_roll_mean", "column": "amount", "window": "30D"}` | 2 — a spike against this account's own norm, not the population's |
| velocity, 1h / 24h | `{"op": "roll_count", "column": "amount", "window": "1h"}`, `"24h"` | 5 — fraud rings and card testers burst |
| spend velocity | `{"op": "roll_sum", "column": "amount", "window": "24h"}` | 5 — draining before the victim notices |
| new payee | `{"op": "prior_count", "column": "payee"}` (0 = first time) | 3 — mule accounts are new counterparties |
| new device / new country | `{"op": "prior_count", "column": "device_id"}`, `"country"` | 3 — account takeover arrives from somewhere new |
| time since previous transaction | `{"op": "gap_prev"}` | 4 — dormant account suddenly active |
| amount vs merchant-category peers | `apply_peer_features` `{"column": "amount", "by": "mcc", "stat": "ratio"}` | 7 — abnormal for this kind of purchase |
| impossible travel | `{"op": "lag", "column": "lat"}` and `"lon"`, then `apply_custom_feature` distance / (`gap_prev` + 1e-6) | 6 — the "consistent location" pattern broken faster than travel allows |

## AML typologies (money laundering, not card fraud)

The launderer is paid for moving money without the movement being
reported, so the constraints are about amounts, speed and counterparties:

| Typology | Build it | Mechanism |
|---|---|---|
| structuring | `apply_custom_feature` `near_limit = (amount >= 9000) * (amount < 10000)` (the jurisdiction's reporting threshold), then `{"op": "roll_sum", "column": "near_limit", "window": "7D"}` | many deposits just under the threshold that triggers a report |
| pass-through | `in_amt = (direction == 'in') * amount`, `out_amt = (direction == 'out') * amount`; `roll_sum` of each over "24h"; then `out_24h / (in_24h + 1)` | money leaves as fast as it arrives — the account is a conduit |
| fan-in / fan-out | `aggregate_events(period="D")` `{"agg": "nunique", "column": "counterparty", "where": "direction == 'in'"}` and `'out'` | many senders into one mule, or one source out to many |
| round amounts | `is_round = (amount % 100 == 0) * 1`, then `{"op": "roll_mean", "column": "is_round", "window": "30D"}` | engineered transfers are round; salaries and bills are not |

Twins: a small business depositing daily takings (structuring), a payroll
account (fan-out), a savings sweep (pass-through). Account type and
counterparty type are the discriminators. AML labels are suspicious
activity reports — investigator decisions, not confirmed crime: record
that in `label_provenance`.

## Before using the table

Two moves from the `telecom` procedure are domain-independent and worth
making here too, because a table cannot make them for you:

1. **State the mechanism.** For each feature, one sentence on why the
   behaviour it measures is something the actor cannot avoid doing to get
   paid. A feature with no mechanism sentence is a correlation you have
   not tested yet.
2. **Name the innocent twin.** Every one of these features has a
   legitimate population that looks identical — and the discriminator
   that separates them is usually a better feature than the original.

Put both in each spec's `rationale` so they reach the report.

Twins worth knowing here: a large transfer relative to the account's own
average is also a house deposit or a tax bill; a burst of transactions is
also a merchant settling a batch; a first-time payee is also every genuine
new relationship; a new country is also a holiday. Recency of account
opening, payee type, whether the counterparty is reused afterwards, and a
travel notice are the usual discriminators.
