---
name: domain-notes
description: Domain-specific metric priority, threshold floors, explainability requirements, and fairness-relevant attributes for fraud/AML, healthcare, credit/fintech, and telecom. On-demand — fetch via load_skill("modeling", reference="domain-notes"). This is the single source of domain business-facts in this agent; business-understanding and diagnostics read it too, not just modeling.
---

# Domain Notes

Fetch this as the FIRST thing you do once `modeling` loads (or earlier,
from `business-understanding`, once the domain is known — fetching a
reference doesn't unlock modeling's tools, so this is safe to read before
a run_id even exists). Use it to decide what to optimize for, which
threshold floor to set, how much explainability a given domain expects,
what the run's declared `domain` (persisted by `record_business_context`)
should be set to, and which attributes deserve an `assess_fairness` check.

---

## Fraud / AML / IRSF / Wangiri

**Primary metric:** PR-AUC. ROC-AUC is misleading at < 1% positive rate —
a model predicting all-negative can still score well on ROC. Always headline
with PR-AUC on fraud datasets and flag any large gap between the two scores.

**Threshold:** Set `target_recall ≥ 0.90`. Missing 10%+ of fraud cases is
typically unacceptable operationally and for regulatory reporting. Precision
matters too — false positives generate analyst workload — but the recall
floor is non-negotiable.

**Explainability:** SHAP is often required. Fraud analysts challenge
individual case flags; permutation importance alone is not enough for a
stakeholder-facing artifact. Use `method="permutation"` for internal model
validation, SHAP for anything the analyst or compliance team will see.

**Imbalance handling:**
- At < 0.5% positive rate, raise `background_size` to 2000+ for SHAP —
  the default of 200 yields fewer than one positive expected after
  stratified sampling.
- Consider SMOTE only if `class_weight="balanced"` is not enough after
  tuning. Balanced class weights are free and always on — SMOTE is the
  escalation, not the default.

**Fairness-relevant attributes:** no direct anti-discrimination statute
targets fraud scoring the way ECOA does credit, but geography/nationality-
correlated features (country code, region) are a common proxy-bias vector
in AML/Wangiri models specifically — run `assess_fairness` against any
geography or carrier-network column before treating the model as final.
Internal bias review, not a legal requirement, is still the right bar.

---

## Healthcare / Diagnosis

**Primary metric:** Recall on the positive class (sensitivity). Missing a
true positive (missed diagnosis) has higher cost than a false positive
(unnecessary follow-up) in most screening contexts. PR-AUC is the secondary
metric for overall curve shape.

**Threshold:** Set `target_recall ≥ 0.95` as a starting floor. The right
value is domain and condition specific — confirm with the clinical
stakeholder before tuning. Do not use the 0.5 default.

**Explainability:** SHAP is expected. Individual prediction explanations are
often a regulatory or clinical governance requirement — per-patient reasoning
is not optional in this domain. Always use `method="shap"`, not permutation,
for any model whose output reaches a clinical decision.

**Calibration:** Check `plot_calibration_curve` before threshold tuning.
Clinical risk scores depend on well-calibrated probabilities — a predicted
0.8 should mean approximately 80% likelihood, not just "high risk." Logistic
regression is usually well-calibrated; tree ensembles often need explicit
calibration.

**Fairness-relevant attributes:** age, sex/gender, race/ethnicity, and
disability status are the attributes clinical fairness review typically
starts with (US: Section 1557/ADA context; check local regulation for other
jurisdictions) — run `assess_fairness` against whichever of these are
present as columns (or a close proxy, e.g. zip code for race) before this
model is treated as ready for clinical use. This is not legal advice —
confirm the actual regulatory bar with the clinical/compliance stakeholder,
this is a starting checklist, not a compliance certification.

---

## Credit / Fintech

**Primary metric:**
- Default prediction: PR-AUC.
- Churn / propensity: F1.
- For loan approval / denial: precision matters as much as recall — a false
  positive (wrongly denied credit) has regulatory and reputational cost, not
  just a business cost.

**Threshold:** Negotiable. Driven by the cost ratio of false positives to
false negatives, which varies by product and jurisdiction. Work with the
business stakeholder to set this — do not default to 0.5 or to a generic
recall floor.

**Explainability:** SHAP is required by regulation in many jurisdictions
for adverse action explanations (e.g. ECOA in the US, GDPR Article 22 in
the EU). Default to `method="shap"` for any credit decision model — do
not use permutation importance as the stakeholder-facing artifact.

**Fairness-relevant attributes:** ECOA (US) names race, color, religion,
national origin, sex, marital status, and age as protected classes for
credit decisions — run `assess_fairness` against any of these present in
the data (or a proxy: zip code, first name). This is the one domain here
where a `disparate_impact_concern` flag is a legal exposure, not just a
quality signal — treat it as a hard stop-and-ask, not a footnote.

---

## Telecom / Churn / Spam

**Primary metric:**
- Churn: PR-AUC (typically 5–20% positive rate — moderately imbalanced,
  PR-AUC still preferred over ROC-AUC).
- Spam / content moderation: F1, with the trade-off weighted toward
  precision — a false positive (blocking legitimate content or a legitimate
  call) has higher user-visible cost than a missed spam case.
- IRSF / SIMbox / Wangiri: treat as fraud — see the Fraud section above.

**Threshold:**
- Churn: the precision/recall trade-off is a business decision — cost of a
  retention offer (false positive) vs cost of losing the customer (false
  negative). Work with the commercial team to set the floor.
- Spam: bias toward precision. Do not set a high recall floor without
  confirming that the false-positive cost is acceptable to the product team.

**Explainability:** Permutation importance is usually sufficient for
internal use and model validation. Use SHAP if the model output feeds a
customer-facing decision — for example, a credit limit reduction or service
suspension triggered by a model score.

**Fairness-relevant attributes:** age and region/geography are the usual
starting points for churn/retention-offer fairness (does the model
systematically under-offer retention deals to one age band or region?) —
lighter regulatory bar than credit or healthcare, but still worth an
`assess_fairness` pass before a retention-offer model ships, since the
model's output directly changes what a customer is offered.

### SMS: smishing, A2P bypass, flooding — three problems, not one

Do not treat "SMS fraud" as a single row. The three have different shapes
and opposite error costs:

- **Smishing / spam** — F1, biased toward precision. Blocking a legitimate
  message is user-visible; missing one is not, individually.
- **A2P bypass (grey-route SIM farms)** — this is revenue leakage, not
  harm, and the enforcement action is usually barring a SIM. A false
  positive therefore disconnects a paying subscriber, which is a severe,
  visible, complaint-generating outcome. **Weight precision heavily, and do
  not carry the fraud recall floor over to it** — the ≥0.90 recall bar
  exists because a missed Wangiri victim is unrecoverable, whereas missed
  bypass traffic is lost margin that can be caught next month. Report
  precision at the operating point alongside PR-AUC.
- **SMS flooding / bombing** — the shape is *concentration* at one
  recipient, the inverse of the other two. A model or feature set tuned for
  diversity-based SMS fraud will not find it; check recall on flooding
  cases separately rather than folding them into one spam class.

### Churn — which churn, and what the score will be used for

- **Name the churn type** (voluntary / involuntary / prepaid inactivity —
  see the telecom feature reference). Involuntary churn is a collections
  problem: its cost of a false positive is a payment-plan call, not a
  retention offer, and its threshold is set with a different team.
- **Propensity is not persuadability.** A churn score ranks who will
  leave, not who an offer will keep; offers spent on customers who would
  have stayed, or who leave anyway, are wasted. If the output drives
  retention offers, say so in the report and name the honest next step —
  an uplift model or a hold-out-controlled offer test. Never claim offer
  ROI from PR-AUC.
- **Purpose limits the features.** Most jurisdictions allow call detail
  for fraud prevention and restrict it for marketing. A retention-offer
  model built on counterparty or location features needs that checked;
  record it in `assumptions`.

### SIM swap, activation fraud, inflated traffic, spoofing

- **SIM swap / account takeover** — fraud metrics and the ≥0.90 recall
  floor apply, but a false positive stops a genuine customer's SIM change
  or port. Report the share of genuine requests that would be stepped up
  or blocked at the operating point alongside recall. The negatives are
  usually un-flagged, not confirmed (positive-unlabeled).
- **Activation / subscription fraud** — when the score declines service
  or device financing it is a **credit decision** under most consumer-
  credit law: follow the Credit / Fintech section (SHAP for adverse-action
  reasons, protected-class fairness as a hard stop). Labels such as
  first-payment default mature 30–90 days late — exclude immature rows.
- **Artificially inflated traffic (SMS pumping, access stimulation)** —
  revenue leakage like A2P bypass, so weight precision: blocking
  verification SMS to real users breaks their sign-up. Report precision
  at the operating point.
- **Caller-ID spoofing / robocalls** — precision-biased like spam: a
  blocked genuine call (a clinic, a school) is the visible failure.

### Roaming / IoT / M2M device classification

Generative, not adversarial — nobody is hiding, so the guidance above about
evasion and extreme imbalance does not transfer.

**Primary metric:** macro-F1, with per-class recall reported. This is
typically **multiclass** (human / static IoT / mobile IoT / device
category), so PR-AUC and `target_recall` do not apply at all — they are
binary, positive-class notions, and there is no positive class here. Do not
default to the fraud row's floors because the dataset is telecom.

**Threshold:** none. There is no threshold to tune on a multiclass
classifier; `tune_threshold` is binary-only. Report the confusion matrix —
which classes are confused *with which* is the whole finding, and an
aggregate accuracy hides it entirely.

**Explainability:** permutation is fine for internal fleet classification.
Use SHAP if the classification changes a customer's tariff or provisioning.

**The leakage trap specific to this problem:** IMEI TAC (which resolves to
the device model, including whether the module is an M2M type), APN, and
tariff plan are frequently assigned *because* a SIM was already known to be
M2M. They are then the most predictive columns in the file and are the
label wearing a hat. Check how the labels were produced before trusting
any of them — see `business-understanding`'s label-provenance check.

**Fairness-relevant attributes:** none in the usual sense — devices are not
people. Declare `fairness_assessment_status: not_applicable` rather than
forcing a check.

---

## Quick reference

| Domain              | Lead metric | Threshold default        | Explainability  | Fairness check |
|---------------------|-------------|--------------------------|-----------------|-----------------|
| Fraud / AML         | PR-AUC      | target_recall ≥ 0.90     | SHAP required   | Geography/carrier proxy bias |
| IRSF / Wangiri      | PR-AUC      | target_recall ≥ 0.90     | SHAP required   | Geography/carrier proxy bias |
| Healthcare          | Recall      | target_recall ≥ 0.95     | SHAP required   | Age, sex, race/ethnicity, disability |
| Credit / fintech    | PR-AUC      | Negotiate with business  | SHAP required   | ECOA classes — legal exposure, hard stop |
| Churn / telecom     | PR-AUC      | Negotiate with business  | Permutation OK  | Age, region |
| Spam / smishing     | F1          | Bias toward precision    | Permutation OK  | Usually low — reassess if enforcement is user-facing |
| A2P bypass          | PR-AUC + precision at the operating point | Precision-weighted; do NOT inherit fraud's ≥0.90 recall floor | Permutation OK | Low |
| IoT / M2M class.    | macro-F1    | None — multiclass, no threshold to tune | Permutation OK | Not applicable — devices, not people |
| SIM swap / takeover | PR-AUC + genuine-request friction at the operating point | target_recall ≥ 0.90, friction reported | SHAP required | Geography/channel proxy bias |
| Activation fraud    | PR-AUC      | Negotiate — it is a credit decision | SHAP required (adverse action) | Protected classes — hard stop, as Credit |
| SMS pumping / AIT   | PR-AUC + precision at the operating point | Precision-weighted | Permutation OK | Low |