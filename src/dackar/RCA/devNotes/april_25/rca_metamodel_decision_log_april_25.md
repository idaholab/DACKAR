# RCA Metamodel Decision Log (Locked)

**Date:** April 25, 2026  
**Context:** Design interview resolving implementation dependencies for `rca_metamodel.md`  
**Status:** Shared understanding achieved; decisions locked for execution

---

## 1) Completion Contract

An RCA run is complete only if all are satisfied:

1. Scope record exists (equipment, boundary, time window, safety function map).
2. Data adequacy check is explicit (coverage gaps flagged and accepted).
3. Category coverage complete: each `A-L` has at least one scored candidate, or explicit ruled-out/not-applicable rationale.
4. Step 5 hard gates executed with ruled-out audit log.
5. `v2` ranking produced with posture across temporal, logical, documentary, and OE streams.
6. Confidence and review flags evaluated (near-tie, contradiction, sensitivity).
7. RCA card includes proximate, contributing, and root levels, barrier analysis, actions, monitoring plan, unresolved gaps, and analyst sign-off posture.

---

## 2) Candidate Model and Identity

- Canonical identity key:
  - `hash(component_id, failure_mode_id, primary_causal_category, chain_position, event_scope_id)`
- Same key from multiple generators merges into one candidate with aggregated provenance.
- Keep current legacy `candidate_id` values during migration.
- Add `canonical_candidate_key` during compatibility phase.

---

## 3) Migration Strategy (Phased)

### Phase A (non-breaking)

Add optional fields:
- `primary_causal_category` (`A-L`)
- `chain_position` (`initiating|contributing|consequence`)
- `event_scope_id`
- `category_ruleout_reason` (nullable)
- `metamodel_compliance_level` in manifest (`partial|full`)

### Phase B (breaking, after validation gate)

- Make core metamodel fields required.
- Reject candidates without category and chain position.
- Set strict mode default to `metamodel_compliance_level=full`.

---

## 4) Coverage and Applicability Policy

Two-step policy:

1. Category applicability pass per event/category:
   - `applicable | not_applicable | unknown`
2. Coverage enforcement applies to `applicable` and `unknown`:
   - generate and score candidate(s), or explicit ruled-out reason.
   - `not_applicable` still requires rationale but does not count as coverage miss.

---

## 5) Category Assignment Policy

Hybrid with deterministic precedence:

1. Deterministic mapping first (structured signals and rules).
2. LLM fallback only below confidence threshold.
3. Record assignment provenance:
   - `category_assignment_method = deterministic|llm_fallback|analyst_override`
   - `category_assignment_confidence` in `[0,1]`
4. Analyst override is authoritative with rationale.

---

## 6) Chain Position Policy

Deterministic temporal-logic assignment:

- `initiating`: precedes trigger and is necessary.
- `contributing`: increases likelihood/severity but not earliest decisive mechanism.
- `consequence`: follows trigger or is derivative effect.

Rules:
- low-confidence assignment requires analyst review.
- one primary initiator per causal branch unless near-tie is flagged.

---

## 7) Step 5 Gate Governance

### Hard-gate elimination (default binding)

- Physical plausibility
- Timeline consistency
- Barrier logic

### Analyst override policy

Allowed with required fields:
- `override_type` (`physical|timeline|barrier`)
- technical rationale
- evidence refs
- reviewer identity and timestamp

Overridden candidates are marked `reinstated_by_analyst` and flagged for review.

---

## 8) Uncertainty Propagation (Mandatory)

Formula-based confidence degradation:

- Per-stream quality scores `q` in `[0,1]` for temporal, logical, documentary, OE.
- `Q = weighted_mean(q_temporal, q_logical, q_documentary, q_oe)`
- `score_final = score_raw * Q`

Rules:
- Any critical stream below floor (`< 0.30`) sets `data_limited_conclusion` flag.
- Missing data is not contradiction.
- Sensitivity table must identify missing streams that could change ranking.

---

## 9) Near-Tie and Contradiction Policies

### Near-tie

- Default threshold: `near_tie_delta = 0.05`
- If near-tie, do not auto-select single primary.
- Set `decision_status=review_required`.
- Present co-primary alternatives and discriminating evidence needed.

### Contradiction

- Any single-stream contradiction blocks auto-primary status.
- Candidate remains in ranked set as `review_required_contradiction`.
- Analyst may promote only via explicit override.

---

## 10) Scope Revision Triggers (Mandatory)

Require scope revision review if any trigger fires:

1. High-confidence upstream/downstream dependency outside current scope.
2. Similar-event/OE implication outside boundary.
3. `unknown` applicability in high-impact categories (`B`, `F`, `I`, `L`) due to missing boundary data.
4. Near-tie unresolved due to out-of-scope evidence.
5. Barrier logic depends on unmodeled protection logic.

Record:
- reason, boundary delta, expected discriminating value.

---

## 11) Category-Specific Minimum Evidence

Per-category minima are required before posture can be `supported`.  
If minima missing, posture cannot exceed `insufficient`.

---

## 12) OE Provenance Policy (Availability-Aware)

Default weights:
- `plant = 1.0`
- `fleet = 0.7`
- `industry = 0.5`

Rules:
- Fleet/industry OE may be unavailable.
- Unavailable OE is treated as missing stream data (`insufficient`), not contradiction.
- Add explicit flag `external_oe_unavailable` when applicable.
- Primary conclusion is still allowed if non-OE streams are strong and non-contradictory.

OE reinstatement threshold:
- weighted OE support `>= 0.65`
- no hard physical contradiction

---

## 13) Disambiguation Rules

### `G` vs `I`

- Incorrect baseline/change control -> `I`
- Correct baseline but bad execution -> `G`
- If both true, keep both and chain them.

### `H` vs `K`

- Inadequate design/spec despite conforming item -> `H`
- Adequate spec but non-conforming delivered item -> `K`
- If both true, represent both as linked contributors.

---

## 14) Root-Cause Depth Requirement

Category `L` must always be attempted for every event:

- produce at least one `L` candidate, or
- explicit ruled-out/not-applicable rationale with missing evidence notes.

No silent omission of `L`.

---

## 15) Rule-Out Taxonomy (Controlled)

Primary reason code required from:

- `physically_impossible`
- `timeline_inconsistent`
- `barrier_held`
- `no_supporting_data`
- `category_not_applicable`
- `outside_investigation_scope`
- `superseded_by_higher_fidelity_evidence`
- `analyst_excluded`

Optional free-text detail may be attached.

---

## 16) Degraded-Mode Policy

- Allow stage-wise degraded continuation for exploration/ranking.
- Block final `candidate_ready` when critical requirements are unmet.
- Final posture must become `review_required` or `insufficient_evidence` when needed.
- Degraded stages and causes must be logged.

---

## 17) Audit and Replayability

Runs must be replayable:

- Persist effective config and thresholds.
- Persist candidate lifecycle events (generated, gated out, reinstated, rank shifts, overrides).
- Keep audit trail append-only.
- RCA card references artifact IDs/hashes for reconstruction.

---

## 18) Validation Gate for Full Compliance

Before Phase B strict mode, pass:

1. Schema conformance for new fields.
2. A-L coverage enforcement tests.
3. Gate and override/reinstatement tests.
4. Uncertainty propagation tests.
5. Decision posture tests (near-tie, contradiction, degraded input).
6. Replay/audit reconstruction tests.

---

## 19) Default Thresholds (Versioned)

- `near_tie_delta = 0.05`
- `critical_stream_floor = 0.30`
- `oe_reinstatement_threshold = 0.65`
- analyst override minimum: at least one direct evidence reference and rationale

These defaults are config-driven and must be versioned in run manifest artifacts.

---

## 20) Rollout Sequence

1. Wave 1: schema and metadata (non-breaking)
2. Wave 2: reasoning logic and coverage enforcement
3. Wave 3: governance gates and decision posture controls
4. Wave 4: strict mode and full compliance default

---

## Closure

Shared understanding criterion met:

- critical governance decisions locked
- migration defaults set
- no unresolved blocker for Wave 1 execution

