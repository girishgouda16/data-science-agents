How to derive telecom features from xDRs and subscriber data — a procedure
plus a vocabulary, not a feature list. Fetch via
`load_skill("feature-engineering", reference="telecom")` once the domain is
telecom. Covers voice fraud (Wangiri / IRSF / SIMbox), SMS analytics
(smishing, A2P bypass, flooding, SMS pumping), SIM swap and account
takeover, activation / subscription fraud, caller-ID spoofing, roaming and
IoT/M2M classification, and churn (postpaid, prepaid, multi-SIM) — and,
because it is a procedure, telecom problems not named here. It is written
for any market: where markets differ, it asks rather than assumes.

The procedure (steps 0–5) decides WHAT must be true in the data; the
feature families say HOW a telco data scientist measures it, each as a
tool spec; the archetypes, data shapes and rules behind them are in
`domain-features` (always loaded with this skill).

**Quick path**

1. **Market context** — ask what the data can't tell you: prepaid or
   postpaid, what locks customers in, number portability, multi-SIM, money
   on the SIM, caller-ID authentication, what purpose the data may serve.
2. **Step 0** — adversarial (someone profits and hides) or generative (a
   process produces the behaviour)?
3. **Steps 1–2** — the mechanism, then the invariants it forces into the
   records. Write them before looking at columns.
4. **Step 3** — which record types you have and which views they carry;
   every missing view is a data request, stated.
5. **Step 4** — the label's grain, the decision point, label maturity →
   which tool.
6. **Step 5** — each feature's innocent twin and the discriminator.
7. **Generate** 20–30 candidates from the families, prune, build,
   `profile_features`, reflect — and loop back to step 1 when the
   mechanism turns out wrong.

Worked end-to-end examples live in `telecom-traces` (Wangiri, IoT/M2M,
churn, SIM swap). Read them after drafting your own candidates: a
different schema needs different features, and copying a trace skips the
reasoning this file exists to teach.

# Step 0 — Which family is this problem?

The first question branches, and getting it wrong wastes the whole
derivation.

- **Adversarial** — someone profits and is actively hiding (Wangiri, IRSF,
  SIMbox, A2P bypass, smishing, subscription fraud). Ask **how do they get
  paid**. Features come from what the economics force them to do anyway.
  Expect the pattern to shift once it is detected, so prefer constraints
  over correlations, and expect extreme class imbalance.
- **Generative** — no adversary; behavior is produced by a process
  (IoT/M2M vs human, device-type classification, roaming traveler vs
  permanent roamer, churn, usage segmentation). Ask **what process
  produces this behavior** — firmware schedules, human circadian rhythm,
  billing events, network experience. Features come from the process's
  signature. Patterns are stable; the risk is confounding, not evasion.

The remaining steps are the same for both. Only the step-1 question
differs.

## Market context — ask, don't assume

Telecom markets differ in ways that change the label and the features, not
just the numbers. Settle these from the user or the data before step 1,
and record each answer — or the assumption you made — in
`record_business_context`:

| Question | Why it changes the work |
|---|---|
| Prepaid, postpaid, or both? | Prepaid has no "cancel" event: churn is inactivity-defined, so the usage-defined-label trap is the default, not an edge case, and recharge behaviour becomes the core feature family |
| What holds a customer in? | Contract end, device instalment balance, handset subsidy, promotional credits forfeited on exit, a fixed+mobile bundle — whichever it is, the approach of its expiry is the churn decision point |
| Is number portability available, and is the port request in the data? | Port-out is then a clean churn label, and the port request is a near-decision event: leakage for churn, a signal for port-out fraud |
| Are multi-SIM phones common? | Churn is often a shift of activity to the second SIM, not an exit — a falling share of the subscriber's usage rather than a disconnect |
| Does the SIM gate money? (mobile money, bank one-time passcodes) | Makes SIM swap / account takeover a first-order problem |
| Is caller-ID authentication available? (e.g. STIR/SHAKEN-style attestation) | Attestation level is then a first-class spoofing feature; without it, number-range validity is the proxy |
| For what purpose may call detail be used? | Most jurisdictions allow network data for fraud prevention and restrict it for marketing. A retention-offer model may not be allowed counterparty or location features — check before generating them |

# Step 1 — Mechanism before columns

Do not open the schema yet.

**Adversarial:** how does the actor get paid, and what must they
physically do that they cannot avoid doing? Those unavoidable actions are
constraints, and constraints survive when correlations decay.

| Behavior | Money path | Constraints the actor cannot avoid |
|---|---|---|
| **Wangiri** | Revenue is the victim's *callback* to a premium number. The bait call earns nothing. | Huge outbound volume; must not pay for it, so bait calls end before answer; must reach strangers — no repeat contact, almost no inbound |
| **IRSF** | Revenue share per *minute* terminated on a premium range. | Calls must be long and answered; must land on specific premium prefixes; volume sustained over hours, usually off-peak to hide |
| **SIMbox** | Arbitrage: international termination bought wholesale, re-originated as domestic. | Physically stationary, so one cell indefinitely; carries traffic it never initiates socially — high outbound, near-zero inbound, no reciprocity; rotates SIMs behind one IMEI as they are blocked |
| **A2P bypass (grey route)** | Enterprise SMS traffic terminated over cheap P2P SIMs instead of the paid A2P interconnect. | Must send enterprise-shaped messages from consumer SIMs: high outbound SMS, near-zero inbound, no voice usage, machine timing, many distinct unrelated recipients, repeated near-identical content |
| **Smishing** | Revenue is the victim's click — credential theft or premium subscription. | Must reach strangers at scale; message carries a link or short-code; sender has no prior relationship with any recipient; campaign bursts then goes silent |
| **SMS flooding / bombing** | Paid to harass, or to mask an OTP among noise. | Extreme message rate at one recipient; concentration rather than diversity — the inverse shape of every other SMS fraud |
| **SIM swap / account takeover** | Revenue is what the SIM unlocks — bank or mobile-money one-time passcodes, the victim's online accounts. | Must get the number onto a SIM they hold (SIM change or port-out), usually after a credential reset, often through a remote channel; must then receive passcode SMS quickly, on a new device, often in a new place; the owner's device loses service at the same moment |
| **Activation / subscription fraud** | Financed devices or services that are never paid for — the device is resold, often abroad. | Must open an account on a stolen or synthetic identity; asks for the highest-value devices; identity attributes are young or reused across applications; the device leaves the network or the country soon after; the first bill is never paid |
| **Artificially inflated traffic (SMS pumping, access stimulation)** | Revenue share on termination — paid per message or minute delivered to ranges the fraudster profits from. | Volume concentrated on number *ranges*, not people; SMS: verification requests to sequential or same-range numbers with near-zero conversion (nobody enters the code); voice: many answered calls to a few terminating ranges |
| **Caller-ID spoofing / robocalls** | The victim answers a call that looks local or trusted. | High volume, short calls; a caller number the originating network cannot vouch for (unauthenticated, or outside ranges it owns); numbers rotate fast |

**Generative:** what process produces this behavior, and what does that
process's schedule or experience force to appear?

| Behavior | Generative process | What the process forces |
|---|---|---|
| **IoT / M2M vs human** | Firmware on a timer, not a person. | Periodic, low-variance inter-event gaps; tiny, near-constant payloads; no circadian dip; no voice or P2P SMS; single APN; long uninterrupted attach |
| **IoT device-type** | The application's duty cycle (meter vs tracker vs POS terminal vs alarm). | Meter: daily/hourly beacon, static cell. Tracker: continuous movement, frequent handovers. POS: business-hours bursts, static. Alarm: near-silent with rare event spikes |
| **Permanent roamer vs traveler** | Device deployed abroad on a home SIM vs a person on a trip. | Traveler: bounded visit, home-network activity before and after, mixed services. Permanent roamer: never returns, single visited network, one service |
| **Subscriber churn** | An experience, not an actor. | Something degraded (dropped calls, throughput), cost something (bill shock, allowance exhaustion), or a decision point arrived (the lock-in expiring, device upgrade eligibility, a port request) — and usage *drifts away* before the formal exit |

**Churn is three problems, not one** — name which before going further:

| Kind | Label | Mechanism | Action it drives |
|---|---|---|---|
| Voluntary (port-out / cancel) | number ported or contract cancelled | a choice: price, network, device, competitor offer | retention offer |
| Involuntary | disconnected for non-payment | ability to pay, not satisfaction | collections, payment plans |
| Prepaid / multi-SIM | inactivity, or activity moving to another SIM | the other SIM got cheaper or better; no formal exit exists | win-back, recharge offer |

A model trained on all three mixed learns an average of mechanisms that
need different features and drive different actions.

If the problem is not in either table, write its row before continuing.
The tables are examples of the step, not the set of things it covers.

# Step 2 — Constraint → invariant

Each constraint implies something that **must be true in the records** if
the mechanism or process is running. Write the invariant before checking
whether the column exists — an invariant with no column is a data request,
which is a legitimate output of this step, not a failure.

- "must not pay for the bait call" → duration below answer threshold;
  termination cause no-answer / cancelled
- "must reach strangers" → distinct-callee count ≈ total; repeat-contact
  rate ≈ 0; inbound ≈ 0
- "autodialler, not a thumb" → gaps between call attempts shorter than a
  human can redial, and to a *different* B-party each time
- "box is stationary" → one serving cell indefinitely, at a volume no
  human sustains from one location
- "SIM rotation behind one box" → many IMSIs per IMEI over time
- "enterprise traffic from a consumer SIM" → SMS-to-voice ratio enormous,
  inbound-SMS ≈ 0, recipients share no social structure with each other
- "campaign then silence" → a burst whose start is a step change and whose
  end is abrupt, not a decay
- "firmware on a timer" → inter-event gap variance near zero and gaps
  clustering on round intervals (300s, 900s, 3600s); no hour-of-day dip
- "deployed abroad permanently" → no home-network record after the first
  visited-network record; one visited PLMN only
- "subscriber experienced degradation" → dropped-call / setup-failure rate
  trending up before the decision point
- "usage drifts away before exit" → this month below the subscriber's
  own trailing average, silences getting longer relative to their usual
  rhythm
- "the second SIM takes the share" → activity falls on this SIM while the
  device keeps attaching; recharges shrink and space out
- "takeover needs the SIM, then the passcode" → a SIM change or port
  request, preceded by a credential reset, followed within hours by an
  inbound passcode-SMS burst on a new device (IMEI) in a new place
- "the identity is new or borrowed" → identity attributes young, or seen
  on other applications; a high-value device requested on a day-zero account
- "paid per delivered message, not per reader" → verification SMS to
  sequential or same-range numbers, delivered but never converted

# Step 3 — Open the schema: record type, then views

Telecom data is not one table. Establish **which record type you have**
first, because it bounds what is derivable:

- **Voice CDR** — A/B party, start, duration, cell, termination cause
- **SMS-DR** — A/B party, timestamp, direction, sometimes length or
  encoding; **usually no content**, so content-similarity invariants
  become a data request, not a feature
- **Data-session record (PDP/PDN)** — APN, up/down bytes, session start
  and duration, RAT type; no B-party at all, so every counterparty and
  graph invariant is unavailable
- **Signalling / EDR / mobility** — attach, location update, handover,
  authentication; this is where mobility and roaming invariants live
- **Subscriber-period snapshot** — one row per subscriber per month (or
  day): usage totals, bill, dropped calls, complaints, plan, lock-in
  remaining. The usual shape for **churn**. No events, so no gaps within
  a period — the time signal is *across* periods (lags, trends, recency)
- **Recharge / top-up records** (prepaid) — amount, channel, time; the
  prepaid equivalent of the bill, and an event stream the gap family
  applies to directly
- **Customer journey** — care contacts, IVR paths, store visits, app
  logins, credential resets, SIM changes, port requests, plan changes.
  Where takeover and churn *decisions* show up before the network sees
  anything — and where most leakage lives (retention offers, cases)
- **Application / activation record** — identity attributes, channel,
  store, requested device and plan, credit-check outcome. Only what was
  known at application time is a feature
- **Network experience per subscriber** — signal quality, drops, session
  failures, voice quality; per-cell outages joined to the subscriber's
  most-used cells. Churn's strongest "something degraded" evidence
- **Pre-aggregated entity-period** — the fraud team's export: one row per
  caller per day with volume / distinct / non-zero counts and trailing
  windows already computed (`_lag_`/`PER_`-style names). Several
  invariants are already columns; what is usually missing is the *ratio
  between* windows, which is where the remaining signal sits
- **Subscriber / device reference** — IMEI TAC (device model and whether
  the TAC is an M2M module), plan, contract dates, APN provisioning

Then map each invariant to a **view**:

| View | Typical fields | What it can express |
|---|---|---|
| **who** | A-party, B-party, IMSI, IMEI/TAC | identity, graph structure, SIM/device pairing, module type |
| **when** | timestamp, gap to previous event | rhythm, burst shape, circadian presence, machine regularity |
| **where** | cell / LAC / TAC, visited PLMN, roaming flag | mobility, stationarity, impossible travel, permanent roaming |
| **how long** | duration, setup time, answer flag, session length | answered vs ring-and-cut, IRSF long-hold, always-on attach |
| **to where** | destination prefix, country, operator, APN, short-code | premium targeting, destination stability, application identity |
| **how it ended** | termination / release cause, delivery receipt | who hung up, whether it connected, SMS delivery failure rate |
| **what it cost / how much** | rate, charge, bytes up/down | the fraud's economics; payload size signature for IoT |

Record which views this schema **lacks**, and say so rather than silently
substituting a weaker proxy. A file with no *where* cannot express SIMbox
stationarity or IoT mobility class at all.

# Step 4 — Fix grain and time; the shape picks the tool

1. **What entity does the label attach to?** MSISDN, IMSI, IMEI, device
   TAC, subscriber account, (entity × day), or a roaming visit? Records
   are events; labels almost never are. Every feature is computed at the
   label's grain. IoT classification is usually labeled at SIM or device
   level with **no time axis at all** — the whole observation window
   collapses into one row. Subscribers nest: account → line → SIM →
   device. Multi-line and business accounts decide churn at the *account*;
   takeover happens on a *line*. Roll line features up to the label's
   level (`aggregate_events` with the account as entity: worst line, share
   of lines, any line).
2. **Where is the decision point, and where does the window close?** Every
   feature is computed strictly from data before it. For churn labelled
   "churned next month", the window closes at the end of this month. For
   fraud, it closes when the case was raised — see "barred SIM" in
   Reflection. State the window in the feature name (`_7d`, `_3m`).
   A problem can have two decision points, and they need different feature
   sets: *blocking* a SIM swap must score at the SIM-change request, before
   any passcode arrives; *detecting* a takeover afterwards may use the
   passcode burst.
3. **Is every label mature?** Bad debt, chargebacks, first-payment default
   and confirmed fraud arrive 30–120 days after the event; "churned next
   month" is unknown until that month ends. On a time-ordered split the
   newest periods hold negatives nobody has investigated yet, so test
   precision looks better than reality. Pass `immature_after` to
   `prepare_dataset` — the newest date minus how long outcomes take to
   settle; ask the user how long that is if the data doesn't say.

The shape then decides the tool (domain-features, section 1). In telecom:

| You have | Tool |
|---|---|
| xDRs, label per SIM / device / caller-day | `aggregate_events` (default, or `period="D"`) |
| Journey events; the decision is one of them (SIM change, port request) | `aggregate_events(at=..., window=...)` |
| Caller-day, subscriber-month, per-call or per-transaction rows | `prepare_dataset(group_column, time_column)`, then `apply_entity_features` / `apply_peer_features` before data-cleaning |

Changing the grain changes the prediction unit, which is business framing
(`business-understanding`'s check 3). If the user already said "predict per
SIM" and the file is events, aggregate to SIM and say so. If they didn't,
that is one question worth asking.

# Step 5 — Name the innocent twin

Every feature has a population that looks identical for a legitimate
reason. If you cannot name it, you have not understood the feature.

| Feature | Innocent twin | Discriminator |
|---|---|---|
| huge outbound voice, no inbound | call centre, dispatcher, alarm line | answer rate, destination stability, working-hours confinement |
| hundreds of distinct callees | delivery driver, survey firm | repeat-contact rate over weeks, callee geographic spread |
| spike from near-zero | newly activated genuine SIM | ramp shape — genuine ramps over days, fraud steps to full volume in hours |
| tiny minimum gap between calls | redial after a dropped call | same B-party (retry) vs a different B-party every time (dialler) |
| one cell forever | home-worker, fixed-wireless subscriber | volume per stationary hour, inbound/outbound ratio |
| long calls to one country | diaspora subscriber calling family | callee count (family = few, IRSF = many), tariff band, repeat structure |
| high outbound SMS, no inbound | a genuine alerting service on a legitimate A2P contract | whether the sender is provisioned as A2P; recipient-set overlap across senders; voice usage present |
| perfectly periodic events | a human with a habit, or a polling app on a human's phone | gap variance (firmware ≈ 0, habit is loose), payload constancy, absence of any circadian dip |
| periodic signalling on every device | the network's periodic location-update timer (T3212) — every handset has it | compute rhythm on *user-initiated* events only, never on periodic LU |
| never returns to home network | emigrant who kept the SIM | service mix — a person keeps using voice and P2P SMS, a deployed device does not |
| zero activity for weeks | dormant prepaid subscriber | whether attach/location-update signalling continues; an IoT device stays attached while idle |
| usage drop this month | holiday, or roaming on a local SIM | drop across *every* service at once and a return to rhythm after, vs one service decaying |
| bill far above own average | annual device instalment, one-off purchase | recurs vs one-off; charge category, if present |
| SIM change, then a burst of passcode SMS | a customer who lost their phone and is logging back into everything | channel (in person with ID vs remote), a credential reset just before, the old device still active elsewhere, a location jump |
| port request | a genuine customer leaving — which IS churn | for churn it is the decision itself (leakage); for port-out fraud, whether the holder started it: channel, reset before, new device |
| verification SMS to many numbers | a real app's sign-up spike after a campaign | destination-range concentration, and conversion — real users enter the code |
| smaller recharges, longer gaps | month-end cash squeeze, seasonal income | recovers next cycle vs keeps declining; activity moving to the device's other SIM |
| brand-new email and address on an application | a young adult or a recent mover | reuse across other applications, device value relative to plan |

**The discriminator is usually a better feature than the original** —
promote it into the candidate set.

# The feature families a telco data scientist reaches for

Each family measures step-2 invariants and instantiates an archetype from
`domain-features` (number in brackets). The specs are what you pass to the
tool — `aggregate_events` on event files, `apply_entity_features` where
every row is a decision. Column names are placeholders: map them to the
schema.

### 1. Time between events — the gap family *(4)*

The most informative and most often forgotten family. The *spacing* of an
entity's events says who or what is producing them, independent of how
many there are. `_gap` = seconds since the entity's previous event (among
the rows `where` kept).

| Feature | spec (aggregate_events) | What it reads |
|---|---|---|
| mean gap | `{"column": "_gap", "agg": "mean"}` | activity rate, independent of window length |
| median gap | `{"column": "_gap", "agg": "median"}` | typical rhythm; mean ≫ median means bursts then long silences |
| min gap | `{"column": "_gap", "agg": "min", "where": "direction == 'out'"}` | burst capability — seconds between outbound attempts is beyond a thumb: autodialler (Wangiri, SIMbox) |
| max gap | `{"column": "_gap", "agg": "max"}` | longest silence — dormancy, SIM swapped out, device offline, pre-churn |
| gap std / CV | `{"column": "_gap", "agg": "cv"}` | regularity: ≈ 0 firmware timer, ≈ 1 random arrivals, > 1 bursty human |
| gap periodicity | `{"column": "_gap", "agg": "periodicity"}` | share of gaps within 5% of the median — the cleanest firmware signature |
| fast-gap share | `{"column": "_gap", "agg": "below", "value": 10}` | share of events within 10s of the previous — dialler burst share |
| recency | `{"agg": "recency_days"}` | time from last event to the cutoff |
| span | `{"agg": "span_days"}` | exposure; divide counts by it so a 3-day SIM and a 30-day SIM compare |

Compose after the split with `apply_custom_feature`: **silence in units of
the entity's own rhythm** — `recency_days * 86400 / (gap_mean + 1)` — is
the churn/dormancy feature that raw recency is not: 5 days silent is
normal for one subscriber and alarming for another. `gap_mean / (gap_median + 1)`
is burstiness. Where every row is a decision, the gaps are between rows:
`{"op": "gap_prev"}`, then `{"op": "roll_std", "column": "<gap_prev name>", "window": "30D"}`
for how irregular the entity's activity is.

### 2. Rhythm — when in the day and week *(4)*

| Feature | spec | What it reads |
|---|---|---|
| hour entropy | `{"column": "_hour", "agg": "entropy"}` | near max (≈ 4.58 bits) = flat, machine; human is peaked with a night dip |
| night share | `{"agg": "share", "where": "_hour < 6"}` | diallers ignore the callee's clock; IRSF hides off-peak |
| weekend share | `{"agg": "share", "where": "_dow >= 5"}` | POS/business devices go quiet; people don't |
| active days | `{"column": "_day", "agg": "nunique"}` | persistence; events per active day = count / this |

### 3. Counterparties — who they talk to, and whether they are new *(3)*

| Feature | spec | What it reads |
|---|---|---|
| distinct B-parties | `{"column": "b_party", "agg": "nunique"}` | reach; divide by count for the strangers ratio (≈ 1 Wangiri, smishing, A2P bypass) |
| top B-party share | `{"column": "b_party", "agg": "top_share"}` | concentration — flooding (≈ 1), family calling, IRSF to few numbers |
| B-party entropy | `{"column": "b_party", "agg": "entropy"}` | diversity with the size of the tail counted |
| inbound share | `{"agg": "share", "where": "direction == 'in'"}` | reciprocity proxy — ≈ 0 for every outbound-only fraud |
| first contact | per-call rows: `{"op": "prior_count", "column": "b_party"}` (0 = never called before) | Wangiri and smishing traffic is almost all first contacts; a person's is mostly repeats |
| first call to a country | per-call rows: `{"op": "prior_count", "column": "dest_country"}` | IRSF's first premium destination from a caller who never called abroad |

Share of first contacts in the last day: `prior_count`, then
`apply_custom_feature` `is_new_b = (b_seen == 0) * 1`, then
`{"op": "roll_mean", "column": "is_new_b", "window": "24h"}`. True
reciprocity (did the B-parties ever call back?) needs the B-parties' own
records — a data request.

### 4. Duration and outcome — how the call ended *(1)*

| Feature | spec | What it reads |
|---|---|---|
| unanswered share | `{"agg": "share", "where": "duration == 0"}` | Wangiri bait must not connect |
| ring-and-cut share | `{"agg": "share", "where": "duration > 0 and duration < 5"}` | calls cut the moment they connect |
| answered mean duration | `{"column": "duration", "agg": "mean", "where": "duration > 0"}` | IRSF must hold the line |
| duration CV | `{"column": "duration", "agg": "cv", "where": "duration > 0"}` | IRSF scripts produce uniform lengths; people don't |
| dominant release cause | `{"column": "release_cause", "agg": "top_share"}` | one cause dominating = one scripted behaviour |

### 5. Mobility — where *(3, applied to places)*

| Feature | spec | What it reads |
|---|---|---|
| distinct cells | `{"column": "cell_id", "agg": "nunique"}` | mobility; / active days for cells per day |
| top-cell share | `{"column": "cell_id", "agg": "top_share"}` | stationarity — SIMbox and static IoT ≈ 1 |
| cell entropy | `{"column": "cell_id", "agg": "entropy"}` | home/work two-cell human vs tracker vs static |

### 6. Destination — to where *(1, 3)*

| Feature | spec | What it reads |
|---|---|---|
| premium-range share | `{"agg": "share", "where": "dest_cc in ['881', '882', '883']"}` | IRSF / Wangiri callback targets (map the real ranges; `read_csv` makes an all-digit prefix an int, so match the dtype) |
| international share | `{"agg": "share", "where": "is_international == 1"}` | SIMbox inbound shape, IRSF |
| destination-country count | `{"column": "dest_country", "agg": "nunique"}` | a person's set is small and stable |

### 7. Payload and service mix — what, and how much *(1, 4)*

| Feature | spec | What it reads |
|---|---|---|
| bytes per session CV | `{"column": "bytes_up", "agg": "cv"}` | constant firmware payload ≈ 0 |
| bytes periodicity | `{"column": "bytes_up", "agg": "periodicity"}` | same payload every time |
| SMS share of events | `{"agg": "share", "where": "service == 'sms'"}` | cross-service mix — the cheapest human-vs-machine separator, and almost always omitted |
| distinct APNs | `{"column": "apn", "agg": "nunique"}` | IoT = one APN |

### 8. Identity pairing — SIM and device *(3)*

Aggregate with the *device* as the entity: `entity_column="imei"`, spec
`{"column": "imsi", "agg": "nunique"}` = SIMs rotated through one box
(SIMbox). Reverse it (`entity_column="imsi"`, column `imei`) for device
swapping. On per-event rows, `{"op": "prior_count", "column": "imei"}` = 0
marks a device never seen on this line — the SIM-swap and stolen-phone
tell (it appears *after* a SIM change, so it is a detection feature, not
a blocking one). Carry a device-grain result to a line-grain label with a
join upstream — a data request when the grains differ.

### 9. Own baseline, trend and velocity *(2, 5)*

Where every row is a decision (`apply_entity_features`), and the archetype
that most often wins: **compare the entity with its own past.** Use a time
`window` unless there is exactly one row per period — a caller-day file
with rows only on active days makes "the last 7 rows" mean anything from 7
days to 7 months.

| Feature | spec (apply_entity_features) | What it reads |
|---|---|---|
| vs own baseline | `{"op": "vs_roll_mean", "column": "total_calls", "window": "7D"}` | today against this caller's last 7 *days* |
| change | `{"op": "pct_change", "column": "data_mb", "n": 1}` | this month vs last (one row per month, so rows are periods) |
| volatility | `{"op": "roll_std", "column": "data_mb", "window": "90D"}` | unstable usage |
| trend (short vs long) | `roll_mean` over "30D" and "180D", then `apply_custom_feature` ratio | decay that one lag misses |
| velocity | per-call rows: `{"op": "roll_count", "column": "b_party", "window": "1h"}` | calls in the last hour — dialler bursts |
| since last event | `{"op": "since", "column": "complaints"}` | recency of friction: last complaint, last dropped-call spike, last top-up |
| time since previous row | `{"op": "gap_prev"}` | on sparse files a feature in its own right — a Wangiri SIM is often fresh or long-dormant |

### 10. Experience, cost and lock-in — churn's own family *(2, 7)*

Degradation (`dropped_calls / call_attempts` and session failures, then
their `vs_roll_mean`), bill shock (`vs_roll_mean` on the bill), allowance
pressure (`data_mb / plan_allowance_mb`, row-wise), and the lock-in
(`lockin_months_left` — contract, instalment balance, promotional credits,
whatever holds the customer in this market; the approach of its expiry is
the feature). **Personal or network?** `apply_peer_features`
`{"column": "drop_rate", "by": "top_cell", "stat": "ratio"}`: ≈ 1 means
everyone on that cell suffers — a network ticket, not a retention offer;
≫ 1 means this subscriber alone — a handset or coverage-at-home problem.
On multi-line accounts roll up to the account: the worst line's
experience, the share of lines whose lock-in has expired.

### 11. Prepaid recharge — the bill the customer chooses *(4, 2, 6)*

| Feature | spec | What it reads |
|---|---|---|
| recharge cadence | recharge events: `{"column": "_gap", "agg": "mean"}`, `"max"` | lengthening gaps = drifting away |
| recharge downgrade | recharge-month rows: `{"op": "vs_roll_mean", "column": "recharge_amount", "n": 3}` | smaller top-ups than their own norm |
| recharge streak | recharge-month rows: `{"op": "streak", "column": "recharge_amount"}` | months in a row with a top-up; `(streak > 0) * (recharge_amount == 0)` = the streak just broke |
| zero-balance share | daily balance rows: `{"agg": "share", "where": "balance <= 0"}` | stranded without credit — when the other SIM takes over |
| silence vs own rhythm | `recency_days`, then `apply_custom_feature` `recency_days * 86400 / (gap_mean + 1)` | days since last recharge in units of their usual cadence |

### 12. Journey and friction — before the network sees anything *(4, 5)*

When the decision is one event in the journey stream — a SIM change, a
port request — build at that event: `aggregate_events(at="event_type ==
'sim_change'", window="24h")`, each window closing at the request.

| Feature | spec (aggregate_events with `at`) | What it reads |
|---|---|---|
| credential resets before the request | `{"agg": "count", "where": "event_type == 'credential_reset'"}` | the takeover precursor |
| days since last reset | `{"agg": "recency_days", "where": "event_type == 'credential_reset'"}` | however long ago — no window |
| remote care contacts | `{"agg": "count", "where": "event_type == 'care_contact' and channel != 'store'"}` | remote channels carry the takeover risk |
| SIM changes, last 30 days | `{"agg": "count", "where": "event_type == 'sim_change'", "window": "30D"}` | takeover — or someone who loses phones |

For churn, where the decision is the month end, the same events become
entity-period counts (`aggregate_events(period="D")`) and then
`{"op": "roll_sum", "column": "care_contacts", "window": "30D"}`.

### 13. Peers — unusual for its kind *(7)*

`apply_peer_features`, fitted on the training fold. Telecom's peer keys:
cell (network vs personal — family 10), plan (usage far above plan peers =
wrong plan, a churn and upsell signal), device TAC (an IoT module whose
payload or cadence differs from its own fleet — faulty, or hijacked), APN,
tariff. Always pair with the entity's own baseline (family 9): peers say
"abnormal for its kind", own baseline says "something changed".

# Build it

The tools, their data shapes and the order they run in are in
`domain-features`; the full parameter contracts are in each tool's
description. Telecom specifics:

- `apply_custom_feature` cannot slice strings: TAC-from-IMEI and
  prefix-from-number must exist as columns before the split. A column
  whose name starts with a digit is written `_7_lag_to`.
- Every spec carries a `rationale` — mechanism, innocent twin,
  discriminator.
- After building, `profile_features(run_id)`; after the first model,
  Reflection below.

**Example — raw voice CDR, Wangiri labelled per caller-day:**

```json
aggregate_events(path="cdr.csv", entity_column="a_party", time_column="start_time", period="D",
  features=[
    {"name": "calls",          "agg": "count"},
    {"name": "unanswered_share","agg": "share", "where": "duration == 0",
     "rationale": "bait calls must not connect; call centres get answered"},
    {"name": "distinct_b",     "column": "b_party", "agg": "nunique"},
    {"name": "min_gap_out",    "column": "_gap", "agg": "min", "where": "direction == 'out'",
     "rationale": "autodialler cadence; a redial after a drop goes to the SAME number (see top_b_share)"},
    {"name": "fast_gap_share", "column": "_gap", "agg": "below", "value": 10},
    {"name": "top_b_share",    "column": "b_party", "agg": "top_share"},
    {"name": "night_share",    "agg": "share", "where": "_hour < 6"},
    {"name": "is_wangiri",     "column": "is_wangiri", "agg": "max"}
  ])
→ prepare_dataset(out_path, "is_wangiri", group_column="a_party", time_column="start_time",
                  immature_after="<newest day minus the fraud team's investigation lag>")
→ apply_entity_features(run_id, [
    {"name": "calls_vs_own_7d", "op": "vs_roll_mean", "column": "calls", "window": "7D",
     "rationale": "burst against this caller's own norm; a new genuine SIM also has no norm — ramp shape separates them"},
    {"name": "days_since_prev_active", "op": "gap_prev"}])
→ apply_custom_feature(run_id, "strangers_ratio", "distinct_b / (calls + 1e-6)", rationale=...)
→ profile_features(run_id)
```

# Generating candidates (ToT) — go wide, then prune

Candidate space is a cross-product. Enumerate across it deliberately
rather than recalling features:

**grain** × **view** × **archetype**

- *grains*: A-party, B-party, (A,B) pair, IMEI/TAC, IMSI, account, cell,
  visited PLMN, destination prefix or short-code, APN, entity×day, roaming
  visit, decision event
- *views*: the seven in step 3
- *archetypes*: the seven in `domain-features` — share, own baseline,
  counterparties (diversity, concentration, novelty), timing (recency,
  spacing, rhythm), trend and velocity, streak, peers — plus cross-service
  share, the cheapest human-vs-machine separator

Produce **20–30 candidates spanning at least three grains**, deliberately
including B-party and (A,B)-pair grains — that is where graph structure
lives, and single-entity thinking always misses it. Generate before
judging; a candidate that dies on computability still tells you which
column you wish you had.

Then prune, in this order:

- **(a) mechanism** — does it express a step-2 invariant, or is it
  decoration? Decoration is what `propose_features` already does.
- **(b) computability** — expressible as a spec on this record type at the
  label's grain? If not, keep it as an explicit data request; don't fake it.
- **(c) leakage** — could it encode the label or anything downstream of
  it? Fraud flags, block or barring status, final bills, case-management
  fields, retention offers, port-out requests, APN or plan codes assigned
  *because* the SIM was classified M2M — and anything that happens after
  the decision point (a new device attaching after the SIM change).
- **(d) redundancy** — near-duplicate of a kept candidate? Keep the one
  whose mechanism sentence is clearer.

State the kept features, the pruned ones with the reason, and the data
requests in one message, then apply. Pruning at (b) is often the most
useful thing you tell the user all session.

# Reflection — three passes, and the loop back

**Before applying.** For each surviving feature state three things: the
mechanism sentence (steps 1–2), the innocent twin (step 5), and the window
boundary (step 4). Missing any one → not ready, however good the formula.

**Before modeling.** Run `profile_features` — it flags leakage suspects
and dead features on the training fold. Then interrogate leakage per
feature, not per batch: was this knowable at the decision point; is any
input written by a downstream process; does the aggregation window cross
the label boundary. Four telco traps:

- **The barred SIM.** Once fraud is detected the SIM is barred and goes
  silent — so `recency_days`, `max gap` and `span_days` computed over the
  whole file encode "was caught". Set `before` to the decision point, or
  build at the decision with `at`.
- **The usage-defined label.** If churn was labelled "no usage for 90
  days", every usage-decline feature reproduces the label rule. Check
  `label_provenance`; the honest target is then the *start* of the
  decline, with features closing before it. In prepaid markets this is
  the normal case, not the exception.
- **The immature label.** Did `prepare_dataset` get `immature_after`? If
  not, the newest period's negatives are un-investigated positives.
- **The wrong purpose.** A feature the market allows for fraud may be
  barred for marketing (Market context). Check before modelling, not at
  the deployment review.

**After the first model.** Three failure signatures, three different
fixes:

- One engineered feature dominates and AUC is implausible (>0.98 on
  fraud) → treat as leakage until proven otherwise; re-run step 5 on it.
- Mechanism-driven features underperform raw columns → the **step-1
  mechanism hypothesis was wrong**. Revise the mechanism, not the formula.
  You likely modeled the wrong fraud type, or the population mixes two
  mechanisms that need separating first.
- Strong on validation, collapses on a later time slice → adversarial
  drift. The pattern you learned was a correlation, not a constraint; go
  back to step 1 and ask which features the actor could cheaply change
  (volume, timing) and which they cannot (unanswered bait, strangers).

Feature engineering loops back to step 1; it is not a one-pass stage. Say
which pass you are on when you report results.
