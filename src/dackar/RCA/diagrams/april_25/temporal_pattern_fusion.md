# temporal-pattern fusion layer

## 1. Purpose
Detect whether the current event reproduces a previously observed temporal sequence, combining:
* signal timing (Step 2c / TSKR)
* recurrence (Step 3)
* signal pattern match (Step 3.5)

Output is supporting evidence, not a gate.

## 2. Inputs
Use only existing artifacts (no new retrieval):
* tskr_patterns → onset, recurrence, novelty
* allen_relation_map → temporal ordering
* signal_evidence → component-linked signals / chains
* kg_context.past_events → historical events
* (optional) similar_event_list → candidate past matches

## 3. Core idea
Represent each event as a temporal signature, then compare: current signature vs historical signatures

## 4. Event representation (Temporal Signature)
For each event (current + past), build:

### a. Ordered precursor sequence

Only signals with causal relations (PRECEDES / OVERLAPS / CONTAINS)
Sorted by start time

Example:

[(Pump_vibration, t=-2h),
 (Bearing_temp, t=-1h),
 (Alarm_X, t=-10m)]

### b. Normalized lags

Time-to-event (relative, not absolute)

### c. Signal type / pattern

anomaly type (drift, spike, oscillation)

### d. Component / failure-mode mapping

via KG

## 5. Matching algorithm
### Step 1 — Candidate historical set
Limit to:
* same component or FM (from Step 3 / 2d)
* OR top-N similar events

### Step 2 — Sequence alignment
Use a constrained alignment (keep it simple and deterministic):
Score components:
* Order match (0–1)
* Do signals appear in same order?
* Lag similarity (0–1)
* Are time gaps similar (within tolerance)?
* Signal type match (0–1)
* Same anomaly types?
* Component/FM match (0–1)

Example:

temporal_similarity =
  0.4 * order_score +
  0.3 * lag_score +
  0.2 * signal_type_score +
  0.1 * component_match
Step 3 — Aggregate over matches

For each candidate FM:
* take top-K matching past events
* compute:
  * temporal_pattern_score (0–1)
  * n_matching_events
  * consistency (variance of scores)

## 6. Outputs
Add new artifact, e.g.:

temporal_pattern_analysis
├── fm_id
├── temporal_pattern_score
├── matched_event_ids[]
├── consistency_score
├── pattern_type   (recurrent | weak | none)

And summary flags:
* has_recurrent_temporal_pattern
* temporal_pattern_strength

## 7. Integration points

### Step 4 (ranking)
add as small weight to temporal score (e.g. +0.05–0.10 max)

### Step 5 (evidence)
treat as structured evidence (like recurrence, but stronger)

### Step 6 (RCA card)
“This event follows a previously observed sequence…”

## 8. Guardrails (critical)
To keep it useful and not misleading:
* Only compare against bounded candidate set
* Require minimum signal count (e.g. ≥2–3 precursors)
* Enforce lag tolerance bands (avoid loose matches)
* Keep transparent scoring (no black box)
* Never override hard gates

## 9. Failure modes
Be explicit about limitations:
* Sparse signals → no pattern
* Poor KG mapping → false mismatch
* Inconsistent timestamps → wrong ordering
* Weak historical labeling → noisy recurrence

# Impact on RCA workflow
Where it fits in the workflow. The cleanest placement is:
* Built after Step 2c / 3 / 3.5 (once you have timing + recurrence + patterns)
* Used in:
  * Step 4 (ranking) → small scoring boost
  * Step 5 (refinement) → structured evidence
  * Step 6 (RCA card) → narrative + confidence
So structurally, the pipeline stays the same.

What actually improves
* 1. Better separation of similar candidates Right now, two failure modes can look similar if:
* both have good telemetry match
* both have recurrence
The new layer adds “does this failure unfold the same way as before?”. That’s often the deciding factor in real RCA.

* 2. Stronger recurrence meaning. Today: recurrence = count of past events
With this: ecurrence = repeatable sequence
That’s a big upgrade: from statistical recurrence → mechanistic recurrence

* 3. More robust temporal reasoning. Currently:
* Step 2c → “before vs after”
* TSKR → “how early / how long”
With the new layer you get structure across signals, not just individual timing

* 4. Improved RCA card quality
You can now say: not just “this happened before” but “this same sequence of degradation → alarm → trip has occurred in prior events”
That’s far more actionable and defensible.

## What does NOT change (important)
* Hard gates stay dominant
  * physical plausibility
  * timeline consistency
  * barrier logic
* KG remains the hypothesis space
* Chroma remains the evidence source
* TSKR still provides base temporal scoring
This new layer should never override these, only reinforce them.

## Risks if done poorly
This is where I’d push back a bit:
* If weighted too heavily → it can overfit to historical patterns
* If matching is loose → you get false “recurrence”
* If data is sparse → it adds noise, not signal
So:
* keep weight low and bounded
* require minimum pattern quality
* expose results transparently (no hidden boosts)

## Net effect on the workflow
* Before: Ranking = structure + timing + evidence + recurrence
* After: Ranking = structure + timing + evidence + recurrence
+ sequence-level temporal consistency