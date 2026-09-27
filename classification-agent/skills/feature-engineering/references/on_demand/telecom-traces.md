Worked end-to-end examples for the telecom procedure. Fetch via
`load_skill("feature-engineering", reference="telecom-traces")` AFTER you
have drafted your own candidates from `telecom`: each trace is one schema,
and its features are right for that schema only. What transfers is the
reasoning — mechanism, invariant, view, grain, twin, then specs.

- **A** — Wangiri on a pre-aggregated caller-day export (adversarial;
  most of the work is noticing what the file cannot express)
- **B** — roaming IoT/M2M classification (generative; multiclass; the
  TAC leak)
- **C** — postpaid churn on a subscriber-month file (generative; own
  baseline, lock-in, label maturity)
- **D** — SIM swap scored at the request (adversarial; decision-time
  windows with `at`)

# Worked trace A — Wangiri on a pre-aggregated caller-day file

Real production export (~3.5M rows, caller-per-day grain, ~2.5% positive).

**Step 0/1.** Adversarial. Revenue is the callback, not the bait call.
Constraints: huge outbound volume, bait calls must not connect, callees
must be strangers.

**Step 2.** Invariants: distinct-callee ratio ≈ 1; non-zero-duration share
≈ 0; volume far above this caller's own recent norm.

**Step 3.** Pre-aggregated entity-period. Carries *who* (hashed A-party),
*when* (daily grain plus 3-day and 7-day trailing windows), *how long* only
as a non-zero-duration count. No *where*, no *to where*, no *cost* — so
SIMbox stationarity, IRSF premium targeting and the whole gap family are
**not expressible on this file**; they need the raw CDR — a data request,
stated. Invariants 1 and 2 are already columns (distinct/total and
non-zero/total, also at 3d and 7d).

**Step 4.** Label at caller×day → `prepare_dataset(group_column=<caller>,
time_column=<day>, immature_after=<newest day minus the investigation
lag>)`, then `apply_entity_features` before data-cleaning. Windows close at
end of the prior day. Rows exist only for days a caller called, so every
window is a time `window`, never `n` rows. The caller id repeats across days,
so it reads ~65% unique — caught as the group column, dropped by
data-cleaning after the entity features exist.

**Generate and prune.** ~25 candidates across A-party, A-party×day and
destination grains. Most die at (b) — no *where* / *to where* / events.
Several die at (d) — already computed. What survives:

- `call_burst_ratio_7d` — `apply_custom_feature`:
  `TOTAL_CALLS / (_7_lag_to / 7 + 1e-6)` (the file's own 7d window; a
  column starting with a digit is written with a leading underscore)
- `burst_acceleration` — `apply_custom_feature`:
  `(_3_lag_to / 3) / (_7_lag_to / 7 + 1e-6)`
- `distinct_ratio_vs_own_7d` — `apply_entity_features`
  `{"op": "vs_roll_mean", "column": "<distinct ratio column>", "window": "7D"}`:
  a caller whose callees suddenly become strangers
- `days_since_prev_active` — `{"op": "gap_prev"}`: a Wangiri SIM is
  often fresh or long-dormant; a regular caller is active most days

**Step 5.** Twin for the burst: a genuinely new SIM, whose trailing mean is
also near zero. Discriminator: ramp shape — a real subscriber climbs over
days, a Wangiri burst steps to full volume within one. That is
`burst_acceleration`, which is why it is kept alongside the burst ratio.

**Reflection.** `profile_features` first: the burst ratio should separate
well but not perfectly — `leakage_suspect` on it means the label was
probably written by a rule on daily volume (check `label_provenance`).
Windows close before the labeled day; nothing in the file is written by
case management.

# Worked trace B — roaming IoT/M2M classification

Task: label each roaming SIM as human traveler, mobile IoT, or static IoT,
from data-session records plus mobility signalling.

**Step 0/1.** Generative. The process is a firmware duty cycle versus a
person's day. Forces: periodic low-variance gaps, tiny constant payloads,
no circadian dip, no voice or P2P SMS, and for static IoT a single cell.

**Step 2.** Invariants: gap CV ≈ 0 and gaps concentrated on one interval;
payload variance ≈ 0; hour-of-day entropy ≈ maximum; one APN; distinct
cells = 1 (static) or advancing with no dwell (mobile).

**Step 3.** Data-session records give *when*, *to where* (APN), *how
much*, *how long* — **no B-party**, so the counterparty family is gone,
dropped here, not later. Mobility signalling supplies *where*, but its
periodic location updates are the T3212 twin — compute rhythm on sessions,
not on LU. The device reference table supplies IMEI TAC — **flagged at
prune (c)**: if the TAC-to-M2M mapping built the labels, it is leakage.

**Step 4.** Label at SIM, no time axis → `aggregate_events` with no
`period`, one row per SIM. Features must be comparable across SIMs observed
for different lengths, so normalise by exposure and carry `span_days`.

```json
aggregate_events(path="sessions.csv", entity_column="imsi", time_column="session_start",
  features=[
    {"name": "gap_cv",           "column": "_gap", "agg": "cv"},
    {"name": "gap_periodicity",  "column": "_gap", "agg": "periodicity",
     "rationale": "firmware timer; a polling app on a phone is periodic too — hour_entropy separates them"},
    {"name": "bytes_cv",         "column": "bytes_up", "agg": "cv"},
    {"name": "hour_entropy",     "column": "_hour", "agg": "entropy"},
    {"name": "distinct_cells",   "column": "cell_id", "agg": "nunique"},
    {"name": "top_cell_share",   "column": "cell_id", "agg": "top_share"},
    {"name": "distinct_apns",    "column": "apn", "agg": "nunique"},
    {"name": "active_days",      "column": "_day", "agg": "nunique"},
    {"name": "span_days",        "agg": "span_days"}
  ])
→ prepare_dataset(out_path, "device_class")
→ apply_custom_feature(run_id, "cells_per_active_day", "distinct_cells / (active_days + 1e-6)", ...)
→ apply_peer_features(run_id, [{"name": "bytes_vs_fleet", "column": "bytes_cv", "by": "tac", "stat": "zscore",
     "rationale": "a module behaving unlike its own model's fleet — only if TAC is not how the labels were made"}])
→ profile_features(run_id)
```

Data requests: `sms_to_session_ratio` and voice presence (need the SMS and
voice records joined per IMSI), `home_network_return_flag` (needs visited
PLMN history).

**Step 5.** Twin for `gap_periodicity`: a polling app on a human's phone.
Discriminator: the human phone still shows a circadian dip and still has
voice/SMS, so `hour_entropy` stays in even though it is weak alone.

**Reflection.** Three classes, so imbalance is mild — the trap is the TAC
leak. If accuracy is near-perfect, confirm no TAC-derived,
APN-provisioning or tariff-plan column survived, since all three are
assigned *because* a SIM was already known to be M2M.

# Worked trace C — churn on a subscriber-month file

Columns: `msisdn, month, data_mb, voice_min, bill_amount, dropped_calls,
call_attempts, complaints, plan_allowance_mb, lockin_months_left,
churned_next_month`. Postpaid, voluntary churn, one line per subscriber.
(Market context: on prepaid data the label is inactivity and trace C's
bill features become family 11's recharge features; on multi-line
accounts, aggregate to the account first.)

**Step 0/1.** Generative — an experience, not an actor. Hypotheses: the
network degraded, the bill hurt, the lock-in is running out, and usage
drifts away before the formal exit.

**Step 2.** Invariants: dropped-call rate above the subscriber's own norm;
bill above own norm; usage below own trailing average; a recent complaint;
lock-in close to expiry.

**Step 3.** Subscriber-period snapshot. *who*, *when* (monthly), *how
much*, *what it cost*, *how it ended* (dropped calls). No events inside a
month, so no within-month gaps — the time signal is across months. No
*where* — "coverage got worse at home" is a data request (serving-cell
KPIs).

**Step 4.** Label at subscriber × month, decision point = end of month.
The newest month's label is unknown until the following month ends, so
`prepare_dataset(group_column="msisdn", time_column="month",
immature_after=<second-newest month>)`. One row per month per subscriber,
so `n` rows are periods here; if months can be missing, use `window`
("90D") instead. Then:

```json
apply_custom_feature(run_id, "drop_rate", "dropped_calls / (call_attempts + 1)", ...)
apply_entity_features(run_id, [
  {"name": "data_vs_own_3m",   "op": "vs_roll_mean", "column": "data_mb", "n": 3,
   "rationale": "usage drifting away; a holiday month dips too, but in every service at once and recovers"},
  {"name": "voice_vs_own_3m",  "op": "vs_roll_mean", "column": "voice_min", "n": 3},
  {"name": "bill_vs_own_6m",   "op": "vs_roll_mean", "column": "bill_amount", "n": 6,
   "rationale": "bill shock; an annual device instalment spikes once and recurs yearly"},
  {"name": "drop_rate_vs_own_3m", "op": "vs_roll_mean", "column": "drop_rate", "n": 3},
  {"name": "data_mb_3m",       "op": "roll_mean", "column": "data_mb", "n": 3},
  {"name": "data_mb_12m",      "op": "roll_mean", "column": "data_mb", "n": 12},
  {"name": "days_since_complaint", "op": "since", "column": "complaints"},
  {"name": "days_since_prev", "op": "gap_prev"}
])
apply_custom_feature(run_id, "data_trend", "data_mb_3m / (data_mb_12m + 1e-6)", ...)
apply_custom_feature(run_id, "allowance_pressure", "data_mb / (plan_allowance_mb + 1)", ...)
```

`lockin_months_left` stays as-is — it is already the decision-point
feature.

**Step 5.** Twins: holiday or roaming on a local SIM (usage drops in every
service, then recovers) vs drift (one service decays for months);
one-off bill vs recurring overcharge; a complaint that was resolved vs one
that wasn't (a data request if resolution status exists).

**Reflection.** Check `label_provenance` first: if "churned" means "no
usage for 90 days", `data_vs_own_3m` reproduces the label rule and the
model is learning the definition, not the churn. Retention offers,
port-out requests and final bills are written *after* the decision — prune
at (c). `days_since_prev` well above ~31 means missing months — switch the
`n` specs to time windows. (`gap_prev` and `since` return days on a date
column, and the column's own units on an integer period index.)

# Worked trace D — SIM swap / account takeover, scored at the request

Journey events, one row per event: `msisdn, ts, event_type`
(credential_reset, care_contact, login, sim_change, otp_sms), `channel`,
and `takeover_confirmed` on sim_change rows.

**Step 0/1.** Adversarial. The money is what the SIM unlocks. Constraints:
get the number onto their SIM, then receive passcodes fast, on their
device, wherever they are.

**Market context.** Does the SIM gate money here (mobile money, bank
OTPs)? That sets how much false-positive friction the business accepts.
Is the goal to *block* the swap or to *detect* the takeover afterwards?
Asked, because it moves the decision point (step 4).

**Step 2.** Invariants: a credential reset shortly before a SIM change;
the change through a remote channel; an inbound OTP burst after it; a new
IMEI and a location jump at the same time.

**Step 3.** Journey events give *who*, *when*, *how* (channel). SMS
records give the OTP burst only if short-codes or sender IDs are present —
otherwise a data request. Signalling gives *where*.

**Step 4.** Blocking scores at the SIM-change request, so the decision is
one event inside the stream: `aggregate_events` with `at`. Every window
closes at that request, so the OTP burst and the new IMEI — which come
*after* it — cannot leak in; detecting a takeover would be a different
decision point, a different `at`. Complaints confirm takeovers weeks
later, so the newest weeks are immature.

```json
aggregate_events(path="journey.csv", entity_column="msisdn", time_column="ts",
  at="event_type == 'sim_change'", window="24h",
  features=[
    {"name": "resets_24h",   "agg": "count", "where": "event_type == 'credential_reset'",
     "rationale": "takeovers reset credentials first; a genuine new-phone owner rarely resets and swaps the same day"},
    {"name": "days_since_reset", "agg": "recency_days", "where": "event_type == 'credential_reset'"},
    {"name": "remote_care_24h", "agg": "count", "where": "event_type == 'care_contact' and channel != 'store'"},
    {"name": "logins_24h",   "agg": "count", "where": "event_type == 'login'"},
    {"name": "sim_changes_30d", "agg": "count", "where": "event_type == 'sim_change'", "window": "30D"}
  ])
→ prepare_dataset(out_path, "takeover_confirmed", group_column="msisdn", time_column="ts",
                  immature_after="<newest date minus the complaint lag>")
→ profile_features(run_id)
```

The request's own `channel` is carried through as a column: it is known
the moment the request arrives.

**Step 5.** Twin: a customer who lost their phone. Discriminators: an
in-person change with ID, no credential reset before, the old device going
silent rather than staying active elsewhere.

**Reflection.** Labels come from customer complaints: they lag, and many
takeovers are never reported — the negatives are un-flagged, not
confirmed (`label_provenance`). A false positive stops a genuine
customer's SIM change: report the share of genuine changes that would be
stepped up at the operating point, not just recall.
