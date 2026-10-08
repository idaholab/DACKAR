# Step 5 — Ranking and Evidence Assessment Strategy

Step 5 combines hard constraint elimination with evidence posture classification. Applied sequentially — elimination first, then posture classification and ranking on surviving candidates.

---

## Phase 1 — Hard Constraint Elimination

Apply binary gates to eliminate physically or logically impossible candidates before any scoring. These are facts, not evidence. Eliminated candidates go to the ruled-out log with documented reason and held on standby for OE second pass.

**Gate 1 — Physical Plausibility**
Is this failure mode possible given the operating state at event time?
- Operating state record (step 0): power level, flow rates, pressures, temperatures, operating mode
- FMEA failure mode parameters: operating conditions under which each failure mode is possible
- Design basis documents: operating envelope limits per component
- Equipment specifications: rated conditions, material properties

**Gate 2 — Timeline Consistency**
Does the proposed mechanism produce the observed event sequence?
- When FMEA latency parameters are available: hard gate — observed lag outside expected latency window → candidate eliminated
- When FMEA latency parameters are absent (the common case): degrades to Allen relation check only:
  - Anomaly FOLLOWS event → eliminated as consequence not cause
  - Anomaly PRECEDES or OVERLAPS → passes to Phase 2 as soft temporal signal
  - ⚠️ FMEA latency parameters are rarely available — timeline consistency will most often operate in degraded mode; temporal discrimination burden shifts to Phase 2
- Data: Allen relation map (step 2c), SOE log, FMEA latency parameters (when available), confirmed event timeline

**Gate 3 — Barrier Logic**
If a barrier held, candidates requiring that barrier to fail are eliminated.
- Safety function definitions from KG: which barriers were active at event time
- Protection logic: which barriers held and which failed (SOE actuation records)
- Equipment status: barrier availability, test status, operability
- Alarm logs: confirmation of barrier actuation or non-actuation
- ⚠️ Requires protection logic to be modeled in the KG — significant data requirement

→ Surviving candidate set; ruled-out log with gate and reason; standby list for OE second pass

---

## Phase 2 — Evidence Posture Classification

For each surviving candidate, classify evidence posture independently across four streams:

| Stream | Source | Posture basis |
|--------|--------|---------------|
| Temporal | Allen relation map (step 2c) | Strength and consistency of temporal precedence |
| Logical | KG topology (step 2a) | Upstream position, support dependency, proximity |
| Documentary | Chroma retrieval, lessons learned (steps 3, 5) | Supporting, contradicting, contextual snippets |
| OE | Fleet and industry matches (steps 2d, 5) | Prior events with same pattern and known outcome |

Each stream independently returns: `supported | contradicted | mixed | insufficient`

→ Per-candidate evidence posture across four streams

---

## Phase 3 — Posture Aggregation and Ranking

**Aggregation rules**:
- Contradicted by any single stream → cannot be primary; flagged for analyst review
- Supported by all four streams → strongest possible conclusion
- Mixed or insufficient streams → confidence level and analyst review requirements set accordingly

**Ranking**:
- Within posture classes: fully supported > partially supported > mixed > insufficient
- Within each class, number of supporting streams breaks ties
- Strong temporal and logical support but insufficient documentary → ranks below candidate supported across all four streams

**Confidence and review flags**:
- Near-tie between top two candidates → analyst review required
- Primary candidate with any contradicted stream → analyst review required
- Sensitivity check: would ranking change if a missing data source were available?

→ Hypothesis ranking v2; candidate-level posture; confidence level; near-tie flags; sensitivity table; unresolvable gaps flagged
