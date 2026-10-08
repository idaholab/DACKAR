# RCA Pipeline Stages and Scoring — Design Decisions and Results

**Status:** Curated development record. Distilled April–August 2026 from the pipeline-stage
notes, the May 6 workflow reference guide, the Step 5 strategy note, and the development backlog.

This document captures *why* the RCA run is organized as a staged workflow, *how* candidates are
scored, and *what the numbers and the development results were*. It is a **development-choices and
development-results** record, not a description of current code. For the live stage sequence,
collaborator wiring and scoring streams, see [`../ARCHITECTURE.md`](../ARCHITECTURE.md); where this
document and the code disagree, the code wins. The causal taxonomy itself is owned by
[`METAMODEL.md`](METAMODEL.md), the temporal/epistemics routing by [`EPISTEMICS.md`](EPISTEMICS.md),
and the review chronology that touched these mechanics by [`REVIEWS.md`](REVIEWS.md); this doc owns
the staged workflow and the scoring arithmetic.

A caution the notes force on any reader: the numbers below are the development record of what was
chosen and measured at particular dates, and several are known to have drifted between notes (the
contradictions are flagged explicitly in the last section). Trust the code for live values.

---

## 1. The Step 0–6 model and its philosophy

The workflow takes plant-event data and produces two analyst-facing artifacts: the `rca_card`
(ranked hypotheses with evidence citations, corrective actions at three causal depths, barrier
assessment, human-performance findings) and the `run_manifest` (the audit record: which data
sources were present or missing, the sensitivity table, the scope state, the analyst-review
actions). The guiding sentence recurs across the notes: every inference is traceable to the data
that drove it, and the analyst remains accountable for the final safety determination. The pipeline
is a decision-support co-pilot, not a regulatory sign-off.

The realized run proceeds in seven numbered steps:

- **Step 0 — Scoping.** Builds the run context and a versioned scope-revision lifecycle; scope
  version 0 is the open discovery scope.
- **Step 1 — Data management.** Input validation, guards, and the eight-family data-coverage
  summary; KG initialization and expansion, CMMS augmentation, past-event temporal enrichment.
- **Step 2 — KG expansion.** Signal evidence, the Allen relation map, similar-event lookup.
- **Steps 3 and 3.5 — Pattern recognition.** The TSKR temporal patterns and a signal-lessons-learned
  artifact.
- **Step 4 — Candidate generation.** The causality engine's `generate` call producing the candidate
  4-tuples with A-to-L coverage, plus a scope-boundary filter once scope version exceeds 0.
- **Step 5 — Evidence assessment.** Evidence retrieval, supersession, `refine_with_evidence`
  (scoring, the three hard gates, the sensitivity table), and auto-reentry if needed; Ishikawa,
  barrier analysis and cross-pattern run across this step.
- **Step 6 — Conclusion.** Synthesis into the RCA card (causal depth, human performance, monitoring
  plan), manifest finalization, and the Chroma archive.

Three design commitments explain why the workflow is staged this way, and they are the durable part:

1. **Coverage-driven, not fixation-driven.** Every run must either produce at least one candidate
   per causal category or explicitly document why that category is ruled out. This is the structural
   anti-fixation mechanism — it is what stops the pipeline from silently narrowing to the first
   plausible equipment cause.
2. **Elimination before scoring.** Step 5 begins by applying binary hard gates to eliminate
   physically or logically impossible candidates before any scoring. The Step 5 strategy note frames
   these as facts, not evidence: a candidate that violates physics or the timeline is removed, not
   down-weighted. Eliminated candidates go to a ruled-out log with reason codes and are held on a
   standby list for a second pass against operating experience.
3. **Human-in-the-loop checkpoints are functional elements.** Grounded in IAEA TECDOC-1756, the
   near-tie flagging and the sensitivity table mean the pipeline does not silently pick a winner and
   it tells the analyst which missing data sources would change the ranking. The run can be resumed
   from a checkpoint: pre-built artifacts can be injected and only the scoring step re-run with
   updated evidence.

**The closed-world KG hypothesis space.** The pipeline treats the knowledge graph as a closed world:
any failure mode, component or document not loaded into the graph is invisible. The August 20 phase 2
review quantified the bound — the hypothesis universe is the KG neighborhood (part-usage edges up to
a default of two hops, plus exactly one connector hop) intersected with the populated FMEA.
Functional and causal edge types are not traversed for hypothesis generation, and a component with no
FMEA row contributes zero hypotheses, invisibly. The Chroma vector store supplies documentary
evidence on top of this KG-bounded candidate set; it does not expand it.

**The dual stage labeling, stated honestly.** Three incompatible stage-labeling schemes exist in the
notes, and this is worth flagging because it confuses any reader moving between documents. The
authoritative one is the Step 0–6 numeric model in the May 6 reference guide, which every later
review cites. An older April 23 note used a Stage A-through-J lettering (Stage 0 KG init, A run
setup, B KG context, C TSKR, D candidates v1, E evidence, F candidates v2, G Ishikawa, H synthesis,
I persistence, J validation). A third, code-level internal naming (Stage 5A, 5B, 6 through 10) appears
in the TSKR review notes. No note formally retires the A-to-J scheme; the resolution is purely by
recency, and the strongest written authority is the May 6 guide's statement that its cross-walk table
is authoritative for resolving step-order mismatches. The notes are silent on an explicit deprecation,
so a reader should treat Step 0–6 as current because it is newest and universally cited forward, not
because any note says the letters are dead.

## 2. Candidate scoring mechanics

**The candidate identity.** Every candidate is a 4-tuple: a component, a failure mode, a causal
category (A to L), and a chain position (initiating or consequence). This anchors every hypothesis to
a specific physical item in the equipment model. Chain position is a first-class field, though it was
under-used at conclusion time (a recurring review finding, resolved in August — see
[`REVIEWS.md`](REVIEWS.md)). The May 6 guide added diagnostic fields alongside: which scoring profile
was applied and its weights, a temporal-score quality flag (full Allen versus proxy), and a score
confidence interval.

**The five scoring streams** are structural, temporal, telemetry, evidence and governance. (The Step
5 strategy note describes a parallel *posture* layer — temporal, logical, documentary and operating-
experience streams each classified as supported, contradicted, mixed or insufficient — which is a
distinct classification layer sitting on top of the five numeric streams, not a competing weighting.)

**The weights are the second contradiction, and it is a version split.** The older April 23 note used
a single fixed composite:

```
composite = 0.30·structural + 0.20·temporal + 0.20·telemetry + 0.20·evidence + 0.10·governance
```

with an A/B candidate tiering (A-series required composite at least 0.45 and evidence at least 0.35;
B-series at least 0.25; below 0.25 dropped; safety-significant candidates promoted), a top-k of 10,
and severity floors (severity 4 floored the score at 0.45, severity 5 at 0.55).

The newer, authoritative May 6 guide replaced the single vector with **category-specific weight
profiles**, each summing to 1.00:

| Category group | structural | temporal | telemetry | evidence | governance |
|---|---|---|---|---|---|
| A–F equipment origin | 0.30 | 0.20 | 0.20 | 0.20 | 0.10 |
| G human performance | 0.05 | 0.10 | 0.05 | 0.65 | 0.15 |
| H design deficiency | 0.15 | 0.05 | 0.20 | 0.45 | 0.15 |
| I change control | 0.05 | 0.25 | 0.10 | 0.45 | 0.15 |
| J surveillance | 0.05 | 0.05 | 0.05 | 0.55 | 0.30 |
| K vendor / procurement | 0.10 | 0.10 | 0.05 | 0.50 | 0.25 |
| L organizational | 0.05 | 0.05 | 0.05 | 0.60 | 0.25 |

with a flat `minimum_composite_threshold` of 0.30, a `minimum_pre_evidence_threshold` of 0.10, a
`minimum_evidence_threshold` of 0.35, and a top-k of 5 (the engine's internal default of 10 is
overridden at runtime). Category dispatch reads a `causal_category` field on the KG failure-mode node
when it is curated, falling back to keyword inference and recording which was used.

The way to read the two schemes together: the equipment-origin profile (A to F) is numerically
identical to the old fixed vector, so the change is really about the non-equipment categories (G to
L), which get evidence-heavy and governance-heavy profiles because a human, design, surveillance,
vendor or organizational cause is established by documents and program evidence, not by telemetry.
The notes contain no explicit "we migrated from fixed to category-specific weights" statement; the
shift is inferred from the two documents, and the May 6 guide is newer and authoritative.

**Category-specific scoring contributions, as development results.** Two were measured and recorded
in the April 25 backlog. The Category E operating-point sub-score (Finding H) uses a seven-mode base
table — ranging from a power ramp at 0.70 down to shutdown at 0.20 — and adds a capped delta of up to
0.12 into the structural stream. The Category C common-cause contribution adds a capped delta of up
to 0.10 into the structural stream. The per-failure-mode governance weighting carried over from April
23 gave bearing, lubrication and seal modes a 0.20 governance weight, environmental, design and
vendor modes 0.02, and a 0.10 default.

## 3. The TSKR temporal scorer

The temporal scorer (`TSKRTemporalScorerV1`) runs at Step 3 and emits a `patterns[]` list keyed by
failure mode. Its two headline outputs are a confidence and a support score.

The **confidence** is a six-term normalized weighted sum with a contradiction penalty:

```
confidence = clamp01( weighted_sum(
    max(anomaly_score, telemetry_support)  × 0.45,
    onset_score                            × 0.30,
    chain_score                            × 0.10,
    history_score                          × 0.10,
    anomaly_count_score                    × 0.15,
    lag_consistency_score                  × 0.10 )
  − 0.20 if temporal_contradiction )
```

The **support** is a four-term blend with its own penalty:

```
support = clamp01( 0.35·history_score + 0.35·telemetry_support
                 + 0.15·anomaly_count_score + 0.15·lag_consistency_score
                 − 0.15 if temporal_contradiction )
```

Downstream, the causality engine composes the candidate temporal score so that TSKR confidence
enters at 35 percent, latency alignment and relation precedence at 25 percent each, and support at
15 percent, with a further 0.25 subtracted when a temporal contradiction is present:

```
temporal = 0.35·confidence + 0.25·relation_precedence + 0.25·latency_alignment + 0.15·support
         − 0.25 if temporal_contradiction
```

A temporal contradiction is set when the Allen relation is "follows" or the latency violation is
"too fast" or "too slow."

**The Allen relation map** uses a five-relation subset with base scores: overlaps 0.90, contains
0.85, precedes 0.75, during 0.30, follows 0.10, evaluated in the order follows, precedes, contains,
overlaps, during, with a half-hour epsilon. The relation is blended into the refined temporal score
with a weight of 0.25 (`0.75·TSKR + 0.25·allen`). An older April 23 note listed a different relation
set that included a "simultaneous" relation absent from the newer subset; the five-relation set
supersedes it.

**Latency alignment** carries 25 percent of the downstream temporal weight and records a violation
type (none, too fast, too slow, or not available) along with the expected and observed lags. When
FMEA latency parameters are absent, which is the common case, timeline discrimination degrades to the
Allen relation alone — a design choice recorded in [`DATA_MANAGEMENT.md`](DATA_MANAGEMENT.md) (latency
is an optional enrichment, so the scorer abstains rather than penalizes).

**The recurrence profile** maps a recurrence count to a history score (0 maps to 0.0, 1 to 0.35, 2–3
to 0.55, 4–6 to 0.70, more than 6 to 0.80) with bonuses of 0.15 for an increasing trend, 0.10 for
unresolved recurrence, and 0.05 when the most recent occurrence was within 90 days. The
`novel_pattern` flag is the conjunction of two orthogonal flags, documentary-novel and signal-novel.

**Development results — the TSKR fix pass (May 2026).** This is recorded in detail in
[`REVIEWS.md`](REVIEWS.md); the pipeline-relevant summary is that six confirmed bugs and six
integration gaps were resolved across phases 0 through 4, taking the suite from 1571 to 1622 tests
with 51 new tests. The highest-severity bug was recurrence inflation when failure modes shared a
component (fixed by a guarded component fallback). One of these fixes is the source of the third
contradiction below: the recurrence-trend test was changed from a half-and-half interval ratio to an
OLS linear regression on the interval sequence (slope normalized by the mean interval, threshold plus
or minus 0.10).

**The Allen-blend direction — a design-decision result.** The May 23 review flagged the blend as
one-directional: it could only raise the temporal score, never lower it, making it a confirmation
booster rather than a discriminator. The June 6 review verified the fix: the blend is now a true
weighted average that both raises and lowers (`0.75·old + 0.25·allen`), and a "follows" relation sets
the temporal-contradiction flag that trips the timeline gate. One documented nuance from the
epistemics work: only anomaly (affects-class) nodes can raise the causal score through this blend;
alarm and SOE (monitors-class) nodes contribute only to contradiction detection.

## 4. Hard gates and auto-reentry

**The three Step 5 hard gates**, each with a reason code, applied as elimination-first binary checks:

- **Physical plausibility** (reason `physically_impossible`). The *design intent* was to test the
  failure mode against the operating state at event time (power, flow, pressure, temperature, mode)
  using FMEA condition parameters and design-basis envelope limits. The *as-built result*, flagged
  high-severity by the June 6 review, is narrower: the gate fails a candidate only when the structural
  score falls below 0.20, plus an informational note on protection-logic presence. It is effectively
  a minimum-structural-score screen wearing the "physical plausibility" name. The August 20 review
  resolved this by honest labelling (the gate now declares its basis is the minimum structural score
  and discloses that the operating-state envelope is not checked); a real operating-state check
  remains a future enhancement.
- **Timeline consistency** (reason `timeline_inconsistent`). Hard-fails on a too-fast or too-slow
  latency violation or on a temporal contradiction. It degrades to an Allen-only check when FMEA
  latency parameters are absent, recording that it ran in degraded mode. The June 6 review judged this
  gate sound.
- **Barrier logic** (reason `barrier_held`). Fails when a protection-logic state for an affected
  safety function is failed or degraded, or when a prior barrier-held ruleout exists; otherwise it
  passes in degraded mode without protection-logic context. The backlog set the barrier-held signal
  threshold at 0.80.

After the gates, candidates must clear a dual threshold: composite at least 0.30 and the evidence
threshold met.

**The elimination-first contradiction (design versus as-built).** The Step 5 strategy note and the
metamodel both require elimination-first — gates before scoring. The June 6 review found the realized
code does the opposite inside `refine_with_evidence`: candidates are fully composite-scored first and
the gates applied afterward. The output is functionally equivalent, because a gated candidate has its
evidence threshold marked unmet and is filtered out before the synthesizer, but the audit trail then
shows a composite score for a candidate labelled physically impossible. The August 20 review addressed
the auditability concern with an additive gate-disposition block rather than reordering the whole
pipeline. The design note and the as-built review disagree on ordering; the review is the ground-truth
observation.

**The evidence blend** at Step 5 combines a prior, an authority-weighted support term, a context term,
and a contradiction penalty:

```
new_evidence = clip01( 0.30·prior + 0.55·support·authority_weight + 0.15·contextual − 0.45·contradiction )
```

Document-type authority weights run from condition reports at 1.00 and work orders at 0.95 down
through engineering documents to manuals at 0.60 and bulletins at 0.55. A near-tie (a gap of 0.10 or
less) raises a review-alternative flag rather than letting the pipeline auto-select a single primary.

**Auto-reentry (Step 5d).** Enabled by default (and disabled in tests for determinism), with a default
of one attempt. It fires when the top composite is below the confidence threshold *and* the coverage
gap is judged fixable, re-running Steps 1 through 5 with an expanded KG neighborhood. A known risk
recorded in the May 23 review: if the scope accepted on the first run is not serialized and re-injected,
the second run can silently produce an unscoped result.

## 5. Dated development results

The scoring work was tracked by test counts, which are the most concrete development record:

- **April 25, 2026 (backlog).** The suite climbed from 787 to 1049 across the scoring findings, with
  three milestones tied to named work: Finding G (Allen temporal scoring, 25 tests, suite at 958),
  Finding I (protection-logic hard gates, 22 tests, suite at 980), and Finding H (Category E
  operating-point and the Category C common-cause delta, 20 tests, suite at 1023).
- **May 2026 (TSKR fix pass).** Baseline 1571 to 1622 after phase 4, with 51 new tests.
- **May 23, 2026 (architecture review snapshot).** Roughly 70 test files and more than 900 test
  functions over 34 JSON schemas, with v31 and v32 engines coexisting and v32 the production default.
- **June 6, 2026.** Roughly 1000 tests; the Allen-blend direction verified fixed; the physical-gate,
  gate-ordering and chain-position findings open at this date.
- **August 20, 2026 (phase 2 soundness review).** The latest dated numbers: the full suite green at
  1896 plus 7 slow at the start of the remediation, rising to 1953 and then 1957 (with one opt-in
  skip) across the sub-workstreams, with zero golden-card shifts. This pass resolved the physical-gate
  honest-labelling finding, added deterministic ordering to the KG queries for reproducibility, made
  the four previously-swallowed optional phases surface their failures into the manifest warnings, and
  wired the signal-DAG chain-position signals onto the candidate at refine time.

## 6. The contradictions, collected

Preserved as the honest edge of this record, because a reader will otherwise trip on them:

1. **Stage labeling.** Step 0–6 numeric (May 6, newest, authoritative) versus the older Stage A-to-J
   lettering versus the code-level internal Stage 5A/5B/6–10 naming. No note formally retires the
   letters; the resolution is by recency and the May 6 cross-walk table.
2. **Fixed versus category-specific weights.** The April 23 fixed vector with A/B tiering and top-k 10
   versus the May 6 per-category profiles with a flat 0.30 threshold and top-k 5. The equipment-origin
   profile equals the old fixed vector, so the real divergence is categories G to L. No explicit
   migration note exists.
3. **Recurrence-trend method.** The May 6 guide still documents the half-and-half interval ratio; the
   newer TSKR fix pass replaced it with OLS linear regression. The OLS change is the newer statement
   of record and the guide's text is stale on that one point.
4. **Gate ordering.** The Step 5 strategy note and the metamodel require elimination-first; the June 6
   as-built review found the code scores first and gates after. The review is the ground-truth
   observation.
