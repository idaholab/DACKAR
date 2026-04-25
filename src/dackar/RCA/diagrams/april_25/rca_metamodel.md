# RCA Causal Taxonomy Metamodel
**Session**: April 25, 2026
**Purpose**: Define the taxonomy of causal categories that must be covered by an automated RCA pipeline for power plant equipment. This metamodel serves as a required coverage checklist — for a given event, the pipeline must either generate and score at least one candidate in each category, or explicitly document why the category was ruled out.

---

## Key Questions an Automated RCA Workflow Must Answer

### Scoping
- What failed, on which equipment, under what operating conditions?
- Which safety functions were challenged or lost?
- What is the investigation boundary — systems, trains, time window?

### Data and Relations
- What signals were anomalous before the event, and in what order?
- What maintenance, tests, or configuration changes occurred in the precursor window?
- Which anomalies are causes versus consequences?
- Are any relevant data sources missing, degraded, or of insufficient resolution?

### Pattern Recognition
- Is there a degradation trend preceding the event, and over what timescale?
- Is this a first occurrence or a recurrence? If recurrence, did prior corrective actions work?
- Is there evidence of common cause failure across multiple trains or components?
- Does the anomaly signature match a known failure mode?

### Hypothesis Generation and Ranking
- What are the plausible root causes, ranked by evidence strength?
- Which causal categories have been ruled out, and why?
- Are there near-tie hypotheses the evidence cannot discriminate between?
- Is the top-ranked hypothesis physically consistent with the operating conditions at the time?

### Conclusion
- What corrective actions address the proximate, contributing, and root cause levels?
- What barriers failed, and why?
- Was human performance a cause or contributor?
- What gaps in the investigation remain unresolved?
- What should be monitored to verify corrective action effectiveness?

---

## Causal Categories

### A. Equipment-Internal
*Component fails due to internal degradation mechanisms. Covered by standard FMEA. Primary source of KG failure mode nodes.*

- Material degradation
- Mechanical degradation
- Electrical degradation
- Instrumentation degradation
- Control component degradation

### B. Required Support Not Available or Degraded
*Equipment receives insufficient or absent support from ancillary systems. Requires connectivity graph reasoning: "if required support S is degraded AND equipment E depends on S, generate a candidate linking S degradation to E failure." Not derivable from FMEA alone.*

- Electrical power
- Cooling (process)
- Thermal management (heat tracing, HVAC, cooling to instrument rooms)
- Lubrication
- Sealing
- Instrument air
- Control signal
- Communications

### C. Upstream Influence
*Process conditions arriving at the equipment inlet are outside design basis. Requires fluid/energy path directionality — topology proximity alone is insufficient.*

- Insufficient inlet flow
- Poor fluid quality
- Entrained gas
- High inlet temperature
- Low suction pressure
- Wrong feed composition

### D. Downstream Influence
*Conditions imposed on the equipment outlet by downstream systems. Topology expansion captures neighboring components but does not reason about flow path direction.*

- Excessive backpressure
- Blocked discharge path
- Unstable demand
- Downstream isolation
- Induced recirculation

### E. Operating Context / Mission Demand
*Equipment is operated outside its design envelope or in a manner that accelerates degradation. `operational_context.operating_point` is collected by the pipeline but consumed by no scoring stage — this entire category is currently ignored.*

- Overload
- Off-design operation
- Thermal transients (load follow, heatup/cooldown rates)
- Intermittent cycling
- Start-stop stress
- Prolonged standby
- Runout / low-flow operation

### F. External Hazards and Disturbances
*Conditions imposed from outside the plant process boundary. No representation in current pipeline data model.*

- Thermal environment
- Flooding
- Seismic event
- Fire
- EMI / electrical disturbance
- Foreign object debris

### G. Human and Organizational Contributors
*Actions or omissions by individuals that directly caused or enabled the failure. Only hook in current pipeline is Ishikawa `maintenance_human_factors` branch (keyword-driven) and proposed WO-date proximity check (not yet implemented).*

> **Boundary note with Category I**: Category G covers human actions that deviated from a correct configuration baseline (wrong execution). Category I covers cases where the human action was correctly executed but the configuration baseline itself was wrong. A maintenance technician installing the wrong part when the procedure specified the right part is Category G. The same technician correctly installing a part specified incorrectly in an unrevised procedure is Category I.

- Operator misalignment
- Maintenance error
- Calibration error
- Wrong procedure used or procedure not followed
- Inadequate pre-job briefing
- Failure to use procedure
- Delayed response
- Incorrect setpoint

### H. Design and Specification Deficiencies
*Equipment performs as designed — the design itself is inadequate for the service conditions. FMEA almost never captures this because it assumes design adequacy. Distinct from Category A.*

- Undersized for actual duty
- Inadequate design margin
- Material specification wrong for service conditions
- Incompatible materials (galvanic, chemical)
- Thermal expansion not accommodated
- Fatigue life underestimated at design
- Vibration not analyzed at design

### I. Configuration and Change Control
*The human action was correctly executed but the configuration baseline was wrong or the change control process failed. Distinct from Category G (human contributors) — the deviation is from the design basis, not from procedure.*

- Unauthorized or undocumented modification
- Drawing or procedure not updated after change
- Temporary configuration not restored
- Setpoint change without engineering review
- Software or firmware change introduced defect
- Spare part substitution with non-equivalent component

### J. Inspection and Testing Program Inadequacy
*Equipment degraded between inspections not because maintenance was skipped but because the surveillance program was not designed to detect this failure mode at its actual progression rate. Connects to S-13 (prior CA ineffective) and S-11 (AMP gap).*

- Surveillance test interval too long to detect degradation
- Test methodology doesn't reveal actual failure mode
- Acceptance criteria not conservative enough
- Test performed under non-representative conditions
- Inspection technique lacks sensitivity for this degradation mechanism

### K. Vendor and Supply Chain
*Specification was correct but delivered item did not meet it, or a batch-level defect affects multiple installed components. Distinct from design deficiency (H) and maintenance error (G). Requires supply chain data (lot numbers, vendor certifications, receipt inspection records) with no current representation in pipeline.*

- Manufacturing defect in delivered component
- Wrong material certified to correct specification (material traceability failure)
- Batch-level defect affecting multiple installed components
- Vendor procedure deviation not disclosed
- Counterfeit or non-conforming part

### L. Systemic and Latent Organizational Weaknesses
*The root cause behind the root cause — the organizational system that allowed contributing causes (G through K) to persist. Requires qualitative evidence across program documents, trend data, and institutional knowledge. Hardest category to automate.*

- Corrective action program not effective (repeated escapes)
- Operating experience not incorporated
- Training program gap
- Resource or staffing constraint affecting maintenance quality
- Safety culture indicator

---

## Functional Architecture

### Design Principles
- Data and decisions/assessments are the two poles of the framework. Every reasoning step consumes defined data inputs and produces defined output artifacts — either intermediate assessments or final decisions.
- Human-in-the-loop decision points are explicit functional elements, not boundary conditions.
- Both primary output artifacts are traced throughout: the **hypothesis ranking** (scored candidate list) and the **RCA card** (structured conclusion for CAP).

---

### Data Elements

Organized by the reasoning step they primarily support. A data element may serve multiple steps.

#### Structured / Model Data
| Data Element | Primary step | Causal categories |
|---|---|---|
| Equipment hierarchy and topology (MBSE model) | Scoping, Relations | A–D, K |
| Safety function definitions | Scoping, Conclusion | All |
| Protection logic (trips, permissives, interlocks) | Relations, Patterns | A, B, F |
| Configuration change records (ECNs, setpoint changes) | Patterns, Candidates | H, I |

#### Time-Series and Event Data
| Data Element | Primary step | Causal categories |
|---|---|---|
| Telemetry (process historian: pressure, temperature, flow, vibration, electrical) | Patterns | A–F |
| Sequence of Events (SOE) recorder | Relations, Patterns | A, B, F |
| Alarm logs | Relations, Patterns | A–F |
| Environmental monitoring (ambient temperature, humidity, grid disturbances, seismic) | Patterns | F |

> **Note**: interpreting SOE records requires protection logic documentation (trip setpoints, permissive logic diagrams) as context — these two data elements are tightly coupled and must be available together.

#### Maintenance and Operations Data
| Data Element | Primary step | Causal categories |
|---|---|---|
| Work orders (corrective and preventive) | Patterns, Candidates | G, I, J |
| Surveillance and calibration records (as-found / as-left) | Patterns, Candidates | A, J |
| Condition reports (CRs) | Patterns, Candidates | All |
| Operator logs and shift narratives | Relations, Patterns, Candidates | E, G |

#### Document and Institutional Knowledge
| Data Element | Primary step | Causal categories |
|---|---|---|
| FMEAs | Candidates | A–E |
| SOPs and procedures | Candidates, Conclusion | G, I |
| ECAs and RCAs | Candidates, Conclusion | All |
| Industry OE documents (NRC, INPO, EPRI) | Candidates, Conclusion | All |

#### Supply Chain and Vendor Data
| Data Element | Primary step | Causal categories |
|---|---|---|
| Vendor certifications, lot numbers, receipt inspection records | Candidates | K |
| Training records | Candidates, Conclusion | L |

---

### Reasoning Process: Data to Decisions

> **Note on scope revision**: scoping and data management are not strictly sequential — they are iterative. As relations and patterns are identified, the investigation boundary may need to expand (e.g., a discovered upstream cause outside the original scope). An explicit scope revision mechanism is required at each human decision point.

> **Note on uncertainty propagation**: data quality flags from step 1 must propagate and degrade confidence scores systematically through steps 3, 4, and 5. A data-limited flag is not informational only — it has a defined effect on conclusion confidence.

```
0. SCOPING
   Define the investigation boundary before any data is examined
   — Failed equipment, affected systems, trains, safety functions
   — Time window: event onset + precursor horizon
   — Safety function challenge assessment: which functions were lost or degraded
   → investigation scope record: equipment list, system boundary,
     time window, safety function map

        [HD: analyst confirms scope; scope may be revised at any subsequent step]

1. DATA MANAGEMENT
   Establish the two data infrastructure foundations and verify external inputs

   Infrastructure initialization (runs once per plant; verified per event):
   — KG: plant architecture, topology, failure modes, safety functions,
     document metadata — initialized from MBSE models and updated
     with event-specific context (equipment, time window, operating state)
   — Chroma: unstructured document and narrative content — CRs, SOPs,
     ECAs, FMEAs, RCAs, OE documents — indexed for semantic retrieval

   External inputs (provided per event; not generated by the workflow):
   — Pre-processed anomalies: output of plant anomaly detection methods
     applied upstream to historian time-series; received as anomaly records
     (component, timestamp, severity, pattern); the workflow reasons over
     anomalies as facts — it does not perform signal processing
     ⚠️ anomaly detection quality upstream directly limits what the
     workflow can conclude; gaps in anomaly coverage must be flagged
   — SOE logs: millisecond-resolution discrete event sequences
   — Alarm logs: filtered indicators with timestamps and priorities

   Coverage check: verify KG completeness and Chroma corpus coverage
   for the equipment and systems in scope; flag missing or stale data
   → KG initialized and event-scoped; Chroma corpus loaded;
     external inputs received and coverage gaps flagged

        [HD: analyst confirms data adequacy and accepts any coverage
         limitations before analysis proceeds]

2. KG EXPANSION
   Expand the initial KG in four steps; 2a and 2b run in parallel;
   2c and 2d follow after 2a and 2b complete

   2a. ARCHITECTURAL SEARCH [parallel with 2b]
       Expand the MBSE portion of the KG to include related events
       — Identify anomalies and alarm logs associated with components
         and systems in the investigation scope
       — Cross-reference events across types (CR, WO, alarm, anomaly,
         surveillance record, operator log) for the scoped equipment
       → KG expanded with event nodes and cross-references
         across all event types for scoped components

   2b. TEMPORAL SEARCH [parallel with 2a]
       For each component/asset in the KG find top-N past events
       — Query KG and Chroma for historical events by component/asset
       — Retrieve CRs, WOs, surveillance records, operator logs,
         anomalies within the precursor window and beyond
       → per-component past event lists ranked by recency and relevance

   2c. TEMPORAL RELATION IDENTIFICATION [requires 2a and 2b]
       Identify Allen interval relations between events
       — Compute Allen relations between identified anomalies,
         alarm logs, and the triggering event
       — Establish temporal ordering across all event types
       → Allen relation map: ordered event graph with
         temporal relations between all identified events

   2d. SIMILAR EVENT IDENTIFICATION [requires 2b]
       Identify similar events for similar equipment/components
       — At plant level: same equipment type, same failure signature
       — At fleet and industry level: OE databases (INPO, EPRI, NRC)
         when available
       → similar event list with provenance and confidence weight
         reflecting data source distance (plant > fleet > industry)

        [HD: analyst reviews expanded KG, temporal relations,
         and similar event matches before pattern recognition proceeds]

3. PATTERN RECOGNITION — DOCUMENTARY
   Retrieve past lessons learned based on patterns similar to those
   observed in the current event, from documentary evidence
   — Match current CR, WO, and operator shift log patterns against
     historical CRs, WOs, RCAs, and ECAs for similar components
     and failure descriptions
   — Each match is a lesson learned: a prior event with a known
     outcome, corrective action, and effectiveness result
   — No match is explicitly informative: novel pattern with no
     documentary precedent — candidate confidence will be lower
   → documentary lessons learned set: matched historical events
     with outcomes and corrective action effectiveness;
     novel pattern flag when no match found

        [HD: analyst confirms lessons learned matches and novel pattern flags]

3.5 PATTERN RECOGNITION — — SIGNAL
   Retrieve past lessons learned based on patterns similar to those
   observed in the current event, from signal evidence
   — Match current anomaly and alarm log patterns against historical
     anomaly signatures and alarm sequences for similar components
   — Each match is a lesson learned: a prior signal pattern with a
     known causal explanation and resolution
   — No match is explicitly informative: novel signal pattern
     with no historical precedent
   → signal lessons learned set: matched historical signal patterns
     with causal explanations; novel pattern flag when no match found

        [HD: analyst confirms signal lessons learned and novel pattern flags;
         decides whether novel patterns indicate scope expansion]

4. CANDIDATE GENERATION AND INITIAL RANKING
   Generate candidate hypotheses from sequence roots and nodes identified
   in steps 3 and 3.5; apply initial ranking based on local data only

   Candidate definition — each candidate is a 4-tuple:
     (component, failure mode, causal category, chain position)
     where chain position ∈ {initiating, contributing, consequence}

   Generation:
   — For each sequence root and node, ask: which causal category does
     this pattern belong to, and what specific failure mode on this
     specific component best explains the observed pattern?
   — Coverage enforcement: for each causal category A–L, at least one
     candidate must be generated or the category explicitly ruled out
   — Physical plausibility gate (binary, before ranking):
     is this candidate physically possible given the operating state
     recorded in step 0? Implausible candidates are excluded with
     documented rationale — not scored low
   — Ruling out is a named operation: every excluded candidate requires
     a documented reason (physically impossible / no supporting data /
     analyst excluded / out of scope)
     ⚠️ Ruled-out candidates are held on standby — they receive
     a second pass against fleet and industry OE evidence in step 5;
     rationale is retained in output for regulatory defensibility

   Initial ranking (based on local data only — no OE evidence yet):
   — Temporal score: Allen relation strength and chain position
     from step 2c (initiating candidates ranked above contributing)
   — Logical score: topology position and support dependency
     from step 2a (upstream, direct dependency ranked higher)
   — Combined v1 ranking: temporal + logical scores only
   → candidate set as 4-tuples with v1 ranking;
     ruled-out candidate standby list with rationale;
     category coverage report

        [HD: analyst reviews candidate set, v1 ranking, ruled-out list,
         and coverage report; may reinstate or add candidates]

5. RANKING AND EVIDENCE ASSESSMENT
   Refine ranking using documentary and OE evidence; detect unresolvable conflicts

   Documentary evidence (from Chroma):
   — Retrieve supporting, contradicting, and contextual snippets
     per candidate from CRs, WOs, SOPs, ECAs, FMEAs, RCAs
   — Consistency check: does the proposed mechanism produce
     the observed event sequence? Inconsistent candidates excluded
     with documented rationale

   Fleet and industry OE evidence:
   — Apply fleet-wide and industry OE (INPO, EPRI, NRC) as evidence
     to confirm or refute active candidates
   — Second pass on ruled-out standby candidates: a candidate excluded
     on local data may be reinstated if strong fleet precedent exists;
     reinstatement requires documented rationale
   — Each OE match tagged with provenance and confidence weight
     reflecting source distance (plant > fleet > industry)

   Ranking refinement:
   — v2 ranking: composite of v1 (temporal + logical) plus
     documentary and OE evidence scores
   — v1→v2 rank delta is a diagnostic: large movements indicate
     evidence is discriminating between candidates
   — Near-tie detection: flag candidates evidence cannot discriminate
   — Sensitivity check: would ranking change if a missing data
     source were available?
   → hypothesis ranking v2; evidence posture per candidate;
     reinstated candidates with rationale; near-tie flags;
     sensitivity table; unresolvable gaps flagged

        [HD: analyst reviews v2 ranking, evidence posture, reinstated
         candidates; confirms or overrides before conclusion]

6. CONCLUSION
   Draw and document the causal conclusion
   — Primary hypothesis: proximate, contributing, and root cause levels
   — Barrier analysis: which barriers failed, which held, and why
   — Human performance assessment
   — Unresolved gaps: what evidence would change the conclusion
   — Recommended actions per causal level
   — Forward-looking: what indicators should be monitored
     to verify corrective action effectiveness
   → RCA card (CAP-ready): primary conclusion, alternatives,
     evidence linkage, barrier assessment, recommended actions,
     effectiveness monitoring plan, open gaps

        [HD: analyst approves conclusion and signs off on AP-913 checklist]
```

---



### Causal Depth Levels
The categories have a natural causal depth structure relevant to AP-913 and 10 CFR 50 Appendix B:

| Level | Categories | AP-913 term |
|-------|-----------|-------------|
| Proximate cause | A, B, C, D, E, F | Direct cause — immediate physical mechanism |
| Contributing cause | G, H, I, J, K | Factors that allowed the proximate cause to exist |
| Root cause | L | Systemic weakness that allowed contributing causes to persist |

The current pipeline reasons almost entirely at the proximate cause level. A complete nuclear RCA requires traversing all three levels. Recommended actions addressing only the proximate cause (e.g., "replace the failed bearing") without the contributing cause (e.g., "PM interval inadequate") and root cause (e.g., "AMP not updated to reflect fleet OE") will not satisfy regulatory expectations.

### Candidates vs. Causal Categories

**Causal categories (A–L)** are classes of causal mechanisms — they define the *type* of cause and ensure investigation coverage. Coverage enforcement operates at the category level: did the pipeline generate at least one candidate in each applicable category?

**A candidate** is a specific, instance-level hypothesis: a particular failure mode, on a particular component, at a particular time, attributed to a primary causal category. Ranking and evidence assessment operate at the candidate level.

Example:
- Category A = "equipment-internal degradation" (class)
- Candidate = "bearing wear on Pump P-101A, failure mode FM-047, consistent with the trip at 14:32" (instance)

One category can generate multiple candidates. One candidate belongs to exactly one primary category.

### Coverage Requirement
For each event, the pipeline must either:
1. Generate and score at least one candidate in each category, **or**
2. Explicitly document in `rca_card.executive_summary.analyst_attention_flags[]` why the category was ruled out or is not applicable.

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
