# RCA Causal Metamodel — Design Decisions and Rationale

**Status:** Curated development record. Distilled April–May 2026 from the metamodel
design session and the locked decision log.

This document captures *why* the RCA pipeline reasons over a 12-category causal
taxonomy and *what was decided* about coverage, ranking, gating and governance. It is
a **development-choices** record — the rationale and the locked policy — not a
description of the current code. For how the engine implements this today (categories,
scoring streams, `generate` / `refine_with_evidence`), see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §6; where this document and the code
disagree, the code wins.

---

## 1. Why a taxonomy at all

The taxonomy exists to make RCA coverage **auditable**. For any event the pipeline must
either generate and score at least one candidate in each applicable category, or record
an explicit ruled-out / not-applicable rationale. The categories are classes of causal
*mechanism*; a candidate is a specific instance (a failure mode on a component at a time,
attributed to one primary category). Coverage is enforced at the category level; ranking
and evidence assessment operate at the candidate level.

> Category A = "equipment-internal degradation" (class).
> Candidate = "bearing wear on Pump P-101A, FM-047, consistent with the trip at 14:32"
> (instance). One category yields many candidates; one candidate has exactly one primary
> category.

The driving motivation was that a standard FMEA covers only equipment-internal
degradation (Category A). A regulator-grade nuclear RCA must also reach support-system,
process, human, design, programmatic and organizational causes — most of which an FMEA
structurally cannot produce. The taxonomy names those gaps so the pipeline is forced to
address them rather than silently default to Category A.

## 2. The twelve categories (A–L)

Each category below records what makes it **distinct** and what **data** it needs — the
durable rationale. The concrete failure-mode lists are the working reference.

**A. Equipment-internal** — internal degradation mechanisms. Covered by standard FMEA;
primary source of KG failure-mode nodes. *Material / mechanical / electrical /
instrumentation / control-component degradation.*

**B. Required support not available or degraded** — ancillary support (power, cooling,
lubrication, sealing, instrument air, control signal, communications, thermal
management) is absent or degraded. Needs **connectivity-graph reasoning**: when a
required support S is degraded and equipment E depends on S, generate a candidate linking
the degradation of S to the failure of E. Not derivable from FMEA.

**C. Upstream influence** — inlet process conditions outside design basis (insufficient
inlet flow, poor fluid quality, entrained gas, high inlet temperature, low suction
pressure, wrong feed composition). Needs **flow/energy-path directionality** — topology
proximity alone is insufficient.

**D. Downstream influence** — conditions imposed on the outlet by downstream systems
(backpressure, blocked discharge, unstable demand, downstream isolation, induced
recirculation). Topology expansion finds neighbors but does not reason about flow
direction.

**E. Operating context / mission demand** — operated outside envelope or in a
degradation-accelerating manner (overload, off-design operation, thermal transients,
intermittent cycling, start-stop stress, prolonged standby, runout / low-flow).
Realized in code as an operating-point sub-score (see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §6).

**F. External hazards and disturbances** — conditions from outside the plant process
boundary (thermal environment, flooding, seismic, fire, EMI, foreign-object debris). No
representation in the pipeline data model at design time; flagged as a data gap.

**G. Human and organizational contributors** — actions/omissions that directly caused or
enabled the failure (operator misalignment, maintenance error, calibration error, wrong
or unfollowed procedure, inadequate briefing, delayed response, incorrect setpoint).

**H. Design and specification deficiencies** — the equipment performs *as designed* but
the design is inadequate for the service (undersized, inadequate margin, wrong material
spec, incompatible materials, unaccommodated thermal expansion, underestimated fatigue
life, unanalyzed vibration). FMEA almost never captures this because it assumes design
adequacy. Distinct from A.

**I. Configuration and change control** — the human action was correct but the
configuration baseline was wrong or change control failed (undocumented modification,
drawing/procedure not updated, temporary config not restored, setpoint change without
review, firmware defect, non-equivalent spare substitution).

**J. Inspection and testing program inadequacy** — degradation went undetected because
the surveillance program was not designed to catch this failure mode at its actual
progression rate (interval too long, methodology blind to the mode, acceptance criteria
not conservative, non-representative conditions, insufficient technique sensitivity).

**K. Vendor and supply chain** — spec was correct but the delivered item did not meet it,
or a batch defect affects many installed components (manufacturing defect, material
traceability failure, batch defect, undisclosed vendor deviation, counterfeit part).
Needs supply-chain data (lot numbers, certs, receipt inspection) with no pipeline
representation at design time.

**L. Systemic and latent organizational weaknesses** — the root cause behind the root
cause: the organizational system that let contributing causes G–K persist (ineffective
CAP, OE not incorporated, training gap, resource/staffing constraint, safety-culture
indicator). Requires qualitative evidence across programs, trends and institutional
knowledge; the hardest category to automate.

### Disambiguation rules (locked)

These boundaries were the recurring source of mis-classification and were resolved
explicitly:

- **G vs I** — wrong execution against a correct baseline is **G**; correct execution
  against a wrong baseline is **I**. If both are true, keep both and chain them. *(Example:
  a technician installing the wrong part when the procedure specified the right one is G;
  the same technician correctly installing a part specified incorrectly in an unrevised
  procedure is I.)*
- **H vs K** — an inadequate design or spec despite a conforming item is **H**; an
  adequate spec but a non-conforming delivered item is **K**. If both, represent both as
  linked contributors.

## 3. Causal depth and the AP-913 mapping

The categories carry a natural depth structure, relevant to AP-913 and 10 CFR 50
Appendix B:

| Level | Categories | AP-913 term |
| --- | --- | --- |
| Proximate | A, B, C, D, E, F | Direct cause — immediate physical mechanism |
| Contributing | G, H, I, J, K | Factors that allowed the proximate cause to exist |
| Root | L | Systemic weakness that let contributing causes persist |

At design time the pipeline reasoned almost entirely at the **proximate** level. A
complete nuclear RCA must traverse all three: recommended actions that address only the
proximate cause ("replace the failed bearing") without the contributing cause ("PM
interval inadequate") and the root cause ("AMP not updated to reflect fleet OE") will not
satisfy regulatory expectations. **Category L must always be attempted** for every event
— produce at least one L candidate or an explicit ruled-out/not-applicable rationale with
missing-evidence notes. No silent omission of L.

## 4. Locked design decisions

The decision log of April 25, 2026 resolved the implementation dependencies. The durable
decisions:

**Completion contract.** A run is complete only when all hold: a scope record exists
(equipment, boundary, time window, safety-function map); data adequacy is explicitly
checked (gaps flagged and accepted); every category A–L has a scored candidate or a
ruled-out/N-A rationale; Step 5 hard gates ran with a ruled-out audit log; a v2 ranking
exists with posture across temporal / logical / documentary / OE streams; confidence and
review flags (near-tie, contradiction, sensitivity) are evaluated; and the RCA card
includes proximate / contributing / root levels, barrier analysis, actions, a monitoring
plan, unresolved gaps and an analyst sign-off posture.

**Candidate identity.** Canonical key =
`hash(component_id, failure_mode_id, primary_causal_category, chain_position, event_scope_id)`.
Same key from multiple generators merges into one candidate with aggregated provenance.

**Phased migration.** Phase A (non-breaking) adds optional fields
(`primary_causal_category`, `chain_position`, `event_scope_id`,
`category_ruleout_reason`, `metamodel_compliance_level`). Phase B (breaking, after a
validation gate) makes the core fields required and defaults strict mode to
`metamodel_compliance_level=full`.

**Coverage and applicability.** Two steps: a per-event applicability pass
(`applicable | not_applicable | unknown`), then coverage enforcement over `applicable`
and `unknown` only. `not_applicable` still needs a rationale but is not a coverage miss.

**Category assignment.** Hybrid with deterministic precedence: deterministic mapping
first; LLM fallback only below a confidence threshold; analyst override is authoritative.
Record `category_assignment_method` (`deterministic | llm_fallback | analyst_override`)
and `category_assignment_confidence ∈ [0,1]`.

**Chain position.** Deterministic temporal-logic assignment — `initiating` (precedes the
trigger and is necessary), `contributing` (raises likelihood/severity but is not the
earliest decisive mechanism), `consequence` (follows the trigger). Low-confidence
assignments need analyst review; one primary initiator per branch unless a near-tie is
flagged.

**Step 5 hard gates (default binding).** Physical plausibility, timeline consistency,
barrier logic. Analyst override is allowed only with `override_type`
(`physical | timeline | barrier`), technical rationale, evidence refs, and reviewer
identity + timestamp; overridden candidates are marked `reinstated_by_analyst` and
flagged for review.

**Uncertainty propagation (mandatory).** Per-stream quality scores `q ∈ [0,1]` for
temporal / logical / documentary / OE; `Q = weighted_mean(q...)`;
`score_final = score_raw × Q`. Any critical stream below the floor (`< 0.30`) sets
`data_limited_conclusion`. Missing data is **not** contradiction. The sensitivity table
must identify missing streams that could change the ranking.

**Near-tie and contradiction.** Near-tie (`near_tie_delta = 0.05`) blocks auto-selection
of a single primary → `decision_status = review_required`, with co-primary alternatives
and the discriminating evidence needed. Any single-stream contradiction blocks
auto-primary status; the candidate stays ranked as `review_required_contradiction` and an
analyst may promote it only via explicit override.

**Scope-revision triggers (mandatory).** Require a scope-revision review if: a
high-confidence dependency sits outside scope; a similar-event / OE implication is outside
the boundary; `unknown` applicability occurs in a high-impact category (B, F, I, L) due to
missing boundary data; a near-tie is unresolved for want of out-of-scope evidence; or
barrier logic depends on unmodeled protection logic. Record reason, boundary delta and
expected discriminating value.

**OE provenance (availability-aware).** Default weights `plant = 1.0`, `fleet = 0.7`,
`industry = 0.5`. Unavailable fleet/industry OE is treated as a missing stream
(`insufficient`), not contradiction (set `external_oe_unavailable`); a primary conclusion
is still allowed when non-OE streams are strong and non-contradictory. OE reinstatement of
a ruled-out candidate requires weighted OE support `≥ 0.65` and no hard physical
contradiction.

**Rule-out taxonomy (controlled).** A primary reason code is required from:
`physically_impossible`, `timeline_inconsistent`, `barrier_held`, `no_supporting_data`,
`category_not_applicable`, `outside_investigation_scope`,
`superseded_by_higher_fidelity_evidence`, `analyst_excluded`. Free-text detail may be
attached.

**Degraded mode.** Stage-wise degraded continuation is allowed for exploration/ranking,
but final `candidate_ready` is blocked when critical requirements are unmet (posture
becomes `review_required` or `insufficient_evidence`); degraded stages and causes are
logged.

**Audit and replayability.** Runs must be replayable: persist effective config and
thresholds; persist candidate lifecycle events (generated, gated out, reinstated, rank
shifts, overrides); keep an append-only audit trail; have the RCA card reference artifact
IDs/hashes for reconstruction.

**Versioned default thresholds.** `near_tie_delta = 0.05`, `critical_stream_floor = 0.30`,
`oe_reinstatement_threshold = 0.65`; analyst override minimum is at least one direct
evidence reference plus rationale. These are config-driven and versioned in the run
manifest.

**Rollout sequence.** Wave 1 schema + metadata (non-breaking) → Wave 2 reasoning and
coverage enforcement → Wave 3 governance gates and decision posture → Wave 4 strict mode
and full-compliance default.

## 5. Step 5 — ranking and evidence assessment strategy

Step 5 is elimination-first, then posture classification and ranking on survivors.

**Phase 1 — hard-constraint elimination.** Binary gates applied before any scoring;
eliminated candidates go to the ruled-out log with a reason and are held on standby for
the OE second pass.
- *Gate 1 Physical plausibility* — is the failure mode possible given the operating state
  at event time (power level, flows, pressures, temperatures, mode) against FMEA
  parameters, design-basis envelope and equipment specs?
- *Gate 2 Timeline consistency* — does the mechanism produce the observed sequence? With
  FMEA latency parameters this is a hard gate (observed lag outside the window →
  eliminated); without them (the common case) it degrades to an Allen-relation check
  (anomaly FOLLOWS the event → eliminated as consequence; PRECEDES/OVERLAPS → passes as a
  soft temporal signal). FMEA latency parameters are rarely available, so the
  discrimination burden usually shifts to Phase 2.
- *Gate 3 Barrier logic* — if a barrier held, candidates requiring it to fail are
  eliminated. Requires protection logic modeled in the KG (a significant data
  requirement).

**Phase 2 — evidence posture.** For each surviving candidate, classify
`supported | contradicted | mixed | insufficient` independently across four streams:
temporal (Allen relation map), logical (KG topology), documentary (Chroma retrieval /
lessons learned) and OE (fleet/industry matches).

**Phase 3 — aggregation and ranking.** Contradicted by any single stream → cannot be
primary (analyst review). Supported by all four → strongest conclusion. Within posture
classes, fully-supported > partially > mixed > insufficient, with the number of supporting
streams breaking ties. Near-tie or any contradicted stream on the primary forces analyst
review; the sensitivity check asks whether the ranking would change if a missing source
were available.

## 6. Implementation outcome (dated record)

By end of day **2026-04-25** the Step 0–6 model reached "green" across the board, built in
the wave order above. The notable results, preserved here as record (the live behavior and
line anchors are in [`../ARCHITECTURE.md`](../ARCHITECTURE.md)):

- **Step 0 Scoping** — baseline scope capture plus a versioned iterative scope-revision
  lifecycle in `run_context` (trigger, boundary delta, analyst decision, timestamp, active
  approved version).
- **Step 1 Data management** — an 8-family coverage report with per-artifact quality, the
  SOE ↔ protection-logic paired-data coupling check, and a weighted coverage quality factor
  in scoring.
- **Step 2 KG expansion** — temporal search (2b: precursor-window tagging, per-component
  top-N index) and the Allen relation map (2c), with clock-sync safety. Step 2d
  (three-tier plant/fleet/industry OE lookup) and the architectural search (2a) followed;
  2a scoring forwarding was the deferred piece.
- **Steps 3 / 3.5 Pattern recognition** — `novel_pattern` flagging on TSKR patterns, alarm
  and SOE windows threaded into the temporal scorer, and a signal-lessons-learned artifact.
- **Step 4 Candidate generation** — the canonical 4-tuple, A–L coverage / rule-out
  enforcement, and elimination-first hard gates.
- **Step 5 Ranking** — elimination-first gates, per-stream posture, contradiction blocking,
  near-tie / sensitivity outputs, OE reinstatement with provenance, and a replayability
  signature. The **sensitivity table** projects a composite-score delta per
  missing/degraded source and raises an analyst flag when the ranking could change.
- **Step 6 Conclusion** — a depth-complete RCA card (proximate / contributing / root),
  depth-mapped actions, a `human_performance_assessment` block (H/I/J/K findings,
  performance-mode mapping, AP-913 references), deepened `unresolved_gaps`, and a
  depth-stratified `effectiveness_monitoring_plan`.

Four scoring enhancements from the same push are worth preserving as choices, because they
each tie a category to a concrete score contribution (all realized in
`causality_engine_v32.py`; see [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §6):

- **Finding G** — Allen temporal relations blended into the candidate temporal score
  (`new_temporal = 0.75 × TSKR + 0.25 × allen`, can only raise); a `follows` relation sets a
  temporal-contradiction flag that the timeline gate reads to rule the candidate out.
- **Finding H** — a Category-E operating-point sub-score (7-mode base table) added as a
  capped `op_delta = 0.12 × op_score` to the structural score.
- **Finding I** — `protection_logic_context` read directly in the physical-plausibility and
  barrier-logic gates (barrier `failed`/`degraded` → gate fails with
  `reason_code="barrier_held"`).
- **Category C CCF** — a capped `ccf_delta = 0.10 × common_cause_score` added to the
  structural score only for Category C candidates.
