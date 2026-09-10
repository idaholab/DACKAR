# RCA Reasoning-Oriented Schema (v0.1)

## Purpose

This document defines a **reasoning-oriented schema** to support a Root Cause Analysis (RCA) engine as described in the pipeline workflow. The intent is not to mirror any specific modeling language (OPM, SysML), but to provide a **causal backbone** that allows the RCA pipeline to move coherently from observed symptoms to defensible root cause hypotheses.

In the RCA pipeline, early stages assemble context (KG, CMMS, telemetry), middle stages generate and score hypotheses, and later stages synthesize and validate conclusions. The schema defined here underpins all of those stages by defining **what entities exist, how they relate, and how causality is represented**. Without this structure, the pipeline would reduce to disconnected heuristics rather than a consistent reasoning system.

Concretely, this schema enables the engine to:

- interpret heterogeneous evidence (sensor data, documents, reports) in a unified way
- traverse causal paths across equipment, functions, and supporting systems
- generate hypotheses that are physically and logically plausible
- evaluate those hypotheses against evidence and engineering knowledge

The schema is therefore not an abstract data model—it is the **reasoning substrate** that ensures the pipeline’s outputs are traceable, explainable, and aligned with engineering logic.

---

## Guiding Principle

The schema is built around a single guiding idea:

> Model what can change, what can fail, and how that change propagates through the system.

Traditional system models emphasize structure (what exists) and function (what it does). For RCA, that is insufficient. The critical requirement is to represent **transformation and propagation**: how an event alters the state of an asset, how that state affects function, and how effects propagate through connections and dependencies.

This leads to a deliberate separation between different concepts that are often conflated:

- assets vs. their states
- causes vs. observed symptoms
- operating conditions vs. intrinsic degradation
- evidence vs. interpretation

By keeping these distinctions explicit, the schema prevents causal ambiguity and allows the engine to reason step-by-step rather than jumping directly from symptom to conclusion.

---

## Core Reasoning Flow

The RCA engine operates by progressively transforming observations into explanations. The schema is designed to support the following canonical reasoning path:

**Observation → Condition/State → Failure Mode → Causal Event → Dependencies → Root Cause**

This flow should be understood as a structured interpretation process rather than a fixed algorithm. An observation (for example, a low-flow signal or a vibration alarm) is first interpreted as an indication of a degraded function or abnormal condition. That condition is then mapped to one or more candidate failure modes—engineering patterns that explain how such a condition could arise.

From there, the reasoning expands outward: each failure mode implies possible causal events (degradation mechanisms, support losses, human actions), and each of those events must be evaluated in the context of system dependencies—upstream equipment, downstream constraints, and supporting systems. Only after traversing these layers does the engine arrive at a set of plausible root causes.

Importantly, this flow is not strictly linear. The RCA pipeline revisits and refines hypotheses as more evidence becomes available (e.g., Stage D → E → F re-ranking). However, every iteration still follows this same conceptual path, and the schema ensures that each step is well-defined and traceable.

---

> Model what can change, what can fail, and how that change propagates — not just what exists.

---

## Core Reasoning Flow

The RCA engine should be able to traverse:

**Observation → Condition/State → Failure Mode → Causal Event → Dependencies → Root Cause**

---

## Schema Layers

1. Asset / Structure Layer
2. Functional Layer
3. Condition / State Layer
4. Causal Event Layer
5. Evidence Layer
6. Knowledge Layer
7. Analysis / Run Layer

---

## Schema Validation Against the RCA Workflow

The current reasoning schema is conceptually aligned with the RCA workflow, but the workflow imposes additional requirements that must be represented explicitly if the schema is to guide implementation. The pipeline is not only a plant knowledge model; it is also a staged reasoning system that creates run-scoped artifacts, candidate hypotheses, temporal interpretations, and synthesized conclusions.

That means the schema must do two jobs at once. First, it must represent relatively stable plant knowledge such as equipment, connections, functions, supporting systems, and reusable failure modes. Second, it must represent the evolving analytical objects that are created during a specific RCA run, such as the investigated event, anomaly evidence, temporal patterns, candidate hypotheses, evidence assessments, and the final synthesized RCA card.

Without that second layer, too much of the workflow logic remains implicit in stage-specific code. In practice, the pipeline already manipulates these objects explicitly: Stage B defines the search space, Stage B.5 produces signal evidence, Stage C produces temporal patterns, Stage D and Stage F create and refine candidates, Stage H synthesizes a conclusion, and Stage J validates the consistency of those outputs. The schema should therefore expose those reasoning objects directly rather than treating them as hidden intermediate artifacts.

The key refinement is to distinguish between **type-level engineering knowledge** and **instance-level runtime analysis**. A `FailureMode` is reusable engineering knowledge. A `CandidateHypothesis` is a run-specific interpretation that a particular failure mode, event, or condition may explain the investigated event. A `SensorObservation` is raw evidence. A `SignalEvidence` object is an interpreted anomaly set and propagation structure derived from that evidence. A `ConditionReport` or `WorkOrder` is documentary evidence, but not itself a causal explanation.

A second refinement is to separate the **investigated event** from other event-like entities. The abnormal plant event that triggers the RCA run is not the same as a causal event such as degradation progression or support loss. Nor is it the same as a documentary event such as a work order creation. Treating these as distinct classes prevents the workflow from conflating symptom, cause, and record.

A third refinement is that the schema must represent **run scope and temporality** explicitly. The pipeline reasons over a bounded search space associated with a particular RCA run, with dynamic evidence, temporal relations, confidence measures, and review flags. These are not static plant facts. They belong to the analysis layer of the schema.

Based on this review, the schema is extended with the following workflow-facing classes:

- `AnalysisRun`
- `InvestigatedEvent`
- `CandidateHypothesis`
- `TemporalPattern`
- `SignalEvidence`
- `EvidenceAssessment`
- `SynthesizedConclusion`

These additions make the schema operational enough to guide workflow development while preserving the original plant-centered causal model.

---

## Reasoning-Oriented Schema Table

| Class | Definition | Role in RCA | Scope | Produced / Updated In | Consumed In | Reasoning Use | Key Relations |
|------|------------|-------------|-------|------------------------|-------------|---------------|---------------|
| AnalysisRun | A single RCA investigation context identified by run_id | Container for run-scoped artifacts, scores, and decisions | Run-scoped | Stage A | All stages | Anchors all reasoning to one investigation | investigates, uses_scope, produces |
| InvestigatedEvent | The abnormal plant event being analyzed | Defines the event to be explained | Run-scoped | Input / Stage A | B, B.5, C, D, H, J | Starting point for causal reasoning | targets_asset, affects_function, evidenced_by |
| Equipment | Physical asset (pump, valve, motor) | Anchor for causality and evidence | Static plant knowledge | Stage 0 | B, 5B, B.5, D, H | Defines where failures, states, and functions exist | performs, connected_to, requires, has_state |
| SupportingSystem | System providing required resources | Enables or disables operation of equipment/functions | Static plant knowledge | Stage 0 | B, D, H | Enables support-loss and hidden-dependency reasoning | supplies, supports, connected_to |
| Connection | Physical or logical linkage between assets | Defines propagation paths and search-space expansion | Static plant knowledge | Stage 0 | B, B.5, D | Constrains plausible causal traversal | connected_to, upstream_of, downstream_of |
| Function | Intended behavior of equipment/system | Defines what performance is degraded or lost | Static plant knowledge | Stage 0 | B, D, H | Translates equipment issues into mission impact | performed_by, affected_by, requires |
| AssetState | State of an equipment item | Intermediate layer between cause and function loss | Mixed: static type + run-time instantiation | Inferred across B, C, D, F | C, D, F, H | Supports reasoning from cause to degraded function | has_state, changed_by, affects |
| OperatingCondition | Boundary or contextual condition experienced by equipment | Represents externally imposed conditions and loads | Run-scoped or historical | B.5, C, 5B | C, D, F, H | Captures upstream/downstream/support effects without collapsing them into asset failure | experienced_by, altered_by, contributes_to |
| CausalEvent | Event that changes state or operating condition | Represents candidate causes such as degradation, power loss, misalignment | Run-scoped or historical | Inferred in D/F, documented in 5B | D, F, H | Root-cause candidate layer | causes, changes_state_of, alters_condition_of |
| FailureMode | Reusable engineering causal pattern | Bridges observed evidence and plausible mechanisms | Static engineering knowledge | Stage 0 | B, C, D, F, H | Supports candidate generation and pattern matching | applies_to, affected_by, indicated_by |
| SensorObservation | Raw measured data from sensors or historian | Primary machine-generated evidence | Run-scoped | Input / B.5 | B.5, C, F | Grounds hypotheses in measured evidence | indicates, observed_on |
| SignalEvidence | Interpreted anomaly evidence, propagation chains, and chain scores | Converts raw observations into structured signal-based reasoning input | Run-scoped | Stage B.5 | C, F, H, J | Supports temporal and propagation reasoning | derived_from, supports, contradicts |
| TemporalPattern | Run-specific temporal interpretation for a candidate failure mode | Encodes lag, Allen relation, recurrence, contradiction, confidence | Run-scoped | Stage C | D, H, J | Supplies temporal score and temporal explanation | derived_from, consistent_with, contradicts |
| ConditionReport | Human-authored operational or engineering report | Provides contextual and qualitative evidence | Historical or run-scoped | 5B | D, E, F, H | Adds narrative evidence not available in signals | documents, suggests, references |
| WorkOrder | Maintenance or corrective-action record | Provides maintenance history and prior action evidence | Historical or run-scoped | 5B | D, E, F, H | Supports recurrence, maintenance context, and prior-action reasoning | addresses, documents, references |
| EvidenceAssessment | Candidate-specific evaluation of supporting and contradicting evidence | Converts retrieved evidence into a scoring input | Run-scoped | Stage E / F | F, H, J | Enables evidence-based re-ranking | supports, contradicts, summarizes |
| CandidateHypothesis | A run-specific causal explanation under evaluation | Central reasoning object scored and ranked by the pipeline | Run-scoped | D, refined in F | E, F, H, J | Carries the current explanation, score breakdown, and review state | proposes, supported_by, contradicted_by, derived_from |
| CausalRule | Encoded causal logic or inference rule | Governs valid reasoning steps | Static knowledge | Knowledge base / design time | C, D, F | Ensures hypothesis generation follows engineering logic | infers, constrains |
| SynthesizedConclusion | Structured RCA conclusion produced for analyst review | Final explanation artifact derived from candidates and evidence | Run-scoped | Stage H | J, analyst review | Projects the reasoning graph into an analyst-facing conclusion | selects, summarizes, cites |

---

## Core Relationship Types

### Structural
- connected_to
- upstream_of
- downstream_of
- supplied_by
- part_of

### Functional
- performs
- requires
- enables

### Condition
- has_state
- experiences_condition

### Causal
- causes
- contributes_to
- degrades
- prevents
- changes_state_of
- alters_condition_of

### Evidence
- indicates
- documents
- addresses
- suggests

---

## Minimal Viable Schema (v1)

### Static plant / knowledge entities
- Equipment
- SupportingSystem
- Connection
- Function
- FailureMode
- CausalRule

### Run-scoped analysis entities
- AnalysisRun
- InvestigatedEvent
- AssetState
- OperatingCondition
- CausalEvent
- SensorObservation
- SignalEvidence
- TemporalPattern
- ConditionReport
- WorkOrder
- EvidenceAssessment
- CandidateHypothesis
- SynthesizedConclusion

### Core relations
- investigates
- targets_asset
- performs
- connected_to
- requires
- has_state
- experiences_condition
- changes_state_of
- alters_condition_of
- indicates
- documents
- addresses
- applies_to
- derived_from
- supports
- contradicts
- causes
- contributes_to
- selects

---

## Reasoning Capabilities Enabled

1. Symptom interpretation (raw evidence → condition/state)
2. Failure mode identification
3. Hypothesis generation across physical, support, and contextual dependencies
4. Temporal interpretation and contradiction detection
5. Propagation reasoning from signal evidence
6. Evidence-based re-ranking of candidate hypotheses
7. Traceable RCA synthesis for analyst review
8. Cross-artifact validation of reasoning outputs

---

## Open Points (to refine)

- Formal definition of FailureMode vs CausalEvent instances
- Representation of uncertainty and confidence
- Temporal modeling (event sequencing)
- Scoring/ranking methodology
- Integration with RCA workflow steps

---

## How the Schema Fits the RCA Workflow

The schema is directly aligned with the RCA pipeline stages. It does not replace the workflow; it provides the structure that each stage operates on.

In **Stage 0 and Stage B**, the schema defines how the knowledge graph is constructed and queried. Equipment, connections, and failure modes correspond directly to the `Equipment`, `Connection`, and `FailureMode` classes. This is where the **causal search space** is established.

In **Stage 5B and Stage B.5**, the schema enables integration of evidence. Sensor observations map to `SensorObservation`, CMMS records to `ConditionReport` and `WorkOrder`, and anomaly propagation chains relate to `OperatingCondition` and `CausalEvent`. These stages populate the graph with **dynamic evidence** tied to the static structure.

In **Stage C**, the schema’s distinction between observations, states, and failure modes becomes critical. Temporal scoring operates on relationships between observed anomalies and expected failure mode behavior, effectively linking `SensorObservation` → `OperatingCondition` → `FailureMode`.

In **Stage D**, the schema supports hypothesis generation. Candidates are built by combining failure modes, equipment context, and evidence signals. This corresponds to traversals such as:
- Equipment → FailureMode
- FailureMode → CausalEvent
- Evidence → supports or contradicts those links

In **Stages E and F**, the schema enables evidence evaluation. Retrieved documents and data are linked back to candidates using explicit relations (e.g., `indicates`, `supports`, `contradicts`). The separation between evidence and cause allows the pipeline to **re-rank hypotheses without conflating observation with explanation**.

In **Stage H**, the schema supports synthesis. The RCA card is effectively a structured projection of the underlying graph: a selected causal path, supported by evidence, with alternatives and uncertainties explicitly represented.

Finally, in **Stage J**, validation relies on the schema to check consistency across artifacts. Cross-artifact validation (e.g., ensuring a candidate referenced in the RCA card exists in the candidate set) is only possible because entities and relationships are explicitly defined.

Overall, the schema acts as the **common language across all stages**. Each stage reads from and writes to this structured representation, ensuring that the pipeline behaves as a coherent reasoning system rather than a sequence of disconnected computations. fileciteturn0file0

---

## Operational Development Table

The table below reframes the schema so it can guide workflow implementation directly. Instead of only describing what each class means, it identifies where it enters the pipeline, what downstream stages depend on it, what kinds of reasoning operations it must support, and what validation concerns should be checked during development.

This table should be read as a bridge between the conceptual schema and the RCA workflow design. It is intended to inform artifact design, stage contracts, graph queries, scoring logic, and validation hooks.

| Class | Stage ownership | Expected inputs | Expected outputs | Typical traversals / queries | Scoring dependencies | Validation checks |
|------|------------------|-----------------|------------------|------------------------------|----------------------|------------------|
| AnalysisRun | A | event metadata, pipeline configuration | run_id, run context, stage references | Retrieve all artifacts and decisions for one run | none directly; anchors all scoring context | run_id uniqueness, artifact completeness, stage traceability |
| InvestigatedEvent | Input / A | event.json, asset_id, timestamps, severity, symptom signature | normalized event context | event -> target asset; event -> affected function; event -> anomaly window alignment | severity gates, symptom matching, temporal alignment | required fields present, event/telemetry asset consistency, event interval validity |
| Equipment | 0 / B | MBSE model, KG topology | neighborhood components, asset references, component type | seed resolution, neighborhood expansion, upstream/downstream traversal, support dependency lookup | structural score, telemetry aggregation by component | asset identity resolution, duplicate alias handling, topology completeness |
| SupportingSystem | 0 / B | MBSE model, support relations, utility definitions | support dependencies in kg_context | equipment -> required support; support system -> affected equipment | structural score, candidate generation for support loss | support relation directionality, shared/common support coverage |
| Connection | 0 / B | topology edges, port/connectivity definitions | component adjacency, propagation paths | upstream_of, downstream_of, path expansion, shared utility reachability | structural score, signal propagation relevance | edge direction validity, cycle handling, hop-bound behavior |
| Function | 0 / B / H | MBSE function definitions, safety functions | function context, affected function statements | equipment -> function; candidate -> degraded function; event -> affected function | symptom interpretation, synthesis prioritization | function-to-equipment mapping, safety-function coverage |
| AssetState | C / D / F | observations, reports, inferred degradation signals | state hypotheses, degraded/failed status | evidence -> state; state -> function degradation; event -> candidate state | temporal score support, candidate plausibility, evidence interpretation | state vocabulary consistency, time-bounded state intervals, confidence assignment |
| OperatingCondition | B.5 / C / D | anomaly context, upstream/downstream conditions, operating point | contextual condition hypotheses | upstream equipment -> condition; support loss -> condition; condition -> FM plausibility | temporal score, structural interpretation, contextual filtering | separation from asset state, operating-point consistency, interval correctness |
| CausalEvent | D / F / H | failure modes, reports, condition changes, known mechanisms | candidate cause instances | FM -> possible causal events; event -> state change; support loss -> function loss | candidate generation, narrative synthesis | distinction from investigated event, causal direction consistency |
| FailureMode | 0 / B / D | FMEA data, failure mode catalog, engineering knowledge | FM candidates, latency windows, pattern expectations | equipment type -> applicable FMs; FM -> expected symptoms; FM -> expected latency | temporal score, symptom match, evidence query planning | vocabulary normalization, component-type mapping, required FM metadata completeness |
| SensorObservation | Input / B.5 | telemetry_summary, historian records, signal metadata | normalized signal observations, anomaly intervals | sensor -> component; observation -> anomaly pattern; observation -> time window | telemetry score, temporal pattern generation | sensor-to-component mapping, timestamp validity, signal quality flags |
| SignalEvidence | B.5 | sensor observations, topology, historian anomalies | augmented anomaly set, propagation chains, chain scores | anomaly -> component -> topology chain; chain -> candidate relevance | temporal score augmentation, post-evidence candidate refinement | merge deduplication, chain acyclicity, source provenance, coverage statistics |
| TemporalPattern | C | event interval, signal evidence, FM latency parameters, recurrence history | per-FM temporal assessment | FM -> aligned anomalies; anomaly -> Allen relation to event; recurrence lookup | temporal score in candidate ranking | latency bounds presence, contradiction logic, recurrence source validity |
| ConditionReport | 5B / E | CMMS/engineering narratives, inspection notes | textual evidence, extracted terms, contextual support | report -> asset; report -> suspected FM; report -> causal clue | evidence score, candidate support/contradiction, synthesis narrative | provenance, text extraction quality, date alignment |
| WorkOrder | 5B / E | CMMS work history, maintenance actions | maintenance evidence, prior action context | WO -> asset; WO -> prior issue; WO -> corrective action history | governance score, recurrence context, evidence posture | asset linkage, action status normalization, temporal ordering |
| EvidenceAssessment | E / F | retrieved snippets, source authority, contradiction/support counts | candidate-level evidence summary | candidate -> supporting evidence; candidate -> contradicting evidence; evidence authority lookup | evidence score, posture classification, rank delta | citation traceability, source-tier presence, support/contradiction consistency |
| CandidateHypothesis | D / F | FMs, temporal patterns, telemetry aggregation, evidence summaries, governance inputs | ranked candidates, review flags, score breakdown | candidate -> FM; candidate -> asset; candidate -> supporting evidence; candidate -> alternatives | composite score, series assignment, review gating | score completeness, score normalization, series logic, rank stability |
| CausalRule | Design-time / C / D / F | encoded engineering logic, rule library | valid inference constraints | observation -> state; condition -> FM; FM -> causal event; support loss -> effect | all reasoning dimensions indirectly | rule versioning, rule conflict detection, traceability to rationale |
| SynthesizedConclusion | H / J | top candidates, evidence assessments, function context, analyst flags | RCA card, primary hypothesis, alternatives, recommended actions | conclusion -> selected candidate; conclusion -> cited evidence; conclusion -> unresolved questions | confidence label, final readiness logic | candidate reference validity, evidence citation completeness, action target consistency |

---

## Workflow-Oriented Design Notes

A few implementation principles follow directly from the operational table.

First, the workflow should treat some classes as **plant knowledge** and others as **analysis products**. `Equipment`, `SupportingSystem`, `Connection`, `Function`, `FailureMode`, and much of the rule layer are relatively stable and should live in the initialized knowledge graph. `SignalEvidence`, `TemporalPattern`, `EvidenceAssessment`, `CandidateHypothesis`, and `SynthesizedConclusion` are run-scoped and should be created per event. This distinction is essential for preventing the pipeline from confusing reusable engineering knowledge with investigation-specific reasoning output.

Second, each stage should have a clearly bounded contract in terms of which classes it is allowed to create, update, or only consume. For example, Stage B should expand `Equipment`, `Connection`, `SupportingSystem`, and `FailureMode` context, but it should not create `CandidateHypothesis`. Stage C should create `TemporalPattern`, but it should not directly change `FailureMode`. Stage F should refine `CandidateHypothesis` using `EvidenceAssessment`, but it should not alter raw `SensorObservation`. These boundaries will help keep the workflow explainable and easier to validate.

Third, the table makes clear that several classes need both **graph semantics** and **artifact semantics**. For example, `CandidateHypothesis` should be a first-class analysis object in the pipeline artifacts, but it should also remain graph-linked to equipment, functions, evidence, and failure modes so that the final RCA card can be traced back to a causal path rather than only to a flat ranking list.

Fourth, validation should be distributed rather than delayed. The schema table shows that each class carries its own likely failure modes: bad timestamps, broken asset links, invalid score normalization, absent authority tags, unresolved action targets. These checks should happen as close as possible to the stage that produces the object, while Stage J focuses on cross-artifact consistency.

---

## Candidate Traversal Templates

The schema should support a small number of canonical reasoning traversals. These traversals can later be implemented as graph queries, rule chains, or orchestration logic.

**Symptom interpretation traversal**

`InvestigatedEvent -> SensorObservation -> AssetState / OperatingCondition -> Function`

This traversal interprets observed signals in the context of degraded behavior.

**Failure mode generation traversal**

`Equipment -> FailureMode -> CausalEvent -> AssetState -> Function`

This traversal generates plausible intrinsic and support-related explanations.

**Dependency traversal**

`Equipment -> Connection / SupportingSystem -> OperatingCondition -> FailureMode`

This traversal supports upstream, downstream, and support-loss reasoning.

**Evidence linkage traversal**

`CandidateHypothesis -> EvidenceAssessment -> ConditionReport / WorkOrder / SignalEvidence`

This traversal attaches supporting and contradicting evidence to a candidate.

**Synthesis traversal**

`CandidateHypothesis -> SynthesizedConclusion -> recommended actions / analyst questions`

This traversal transforms ranked causal reasoning into an analyst-facing RCA artifact.

---

## Alignment to Existing RCA Repo Artifacts

This document should align to the existing RCA implementation rather than define a parallel structure. The current RCA codebase already separates orchestration, schema validation, signal evidence handling, synthesis, PM compliance, CMMS integration, and KG support across dedicated subpackages under `src/dackar/RCA`, including `orchestrators`, `schemas`, `signal_evidence`, `synthesis`, `validation`, `kg`, `cmms_integration`, `pm_compliance`, `storage`, and related support modules. The purpose of this section is therefore to map the reasoning-oriented schema to the existing implementation areas and identify where semantics are already explicit versus where they remain distributed or implicit in stage logic. ([github.com](https://github.com/idaholab/DACKAR/tree/mandd/rca/src/dackar/RCA))

The key design principle is that the reasoning schema should remain a semantic overlay on the existing artifact and code structure. In practice, some concepts already have a direct home in the repo, while others are split across multiple artifacts or encoded primarily in orchestration logic rather than in a single schema file. The table below is intended to make that alignment explicit and to highlight where future workflow work should focus.

| Reasoning class | Closest existing artifact / schema | Primary repo area | Where semantics are explicit today | Where semantics are implicit or split | Current gap / ambiguity |
|------|-------------------------------|-------------------|------------------------------------|---------------------------------------|-------------------------|
| AnalysisRun | `run_context.json`, `run_manifest.json`, `reentry_execution.json` | `orchestrators`, `validation`, `schemas` | Run identity, pipeline configuration, terminal validation and routing | Cross-stage reasoning state distributed across multiple artifacts | Needs clearer semantic role as the container for run-scoped reasoning |
| InvestigatedEvent | `event.json`, referenced by `run_context.json` | `orchestrators`, input layer, `schemas` | Event identity, timestamps, target asset, severity | Symptom semantics and affected-function semantics mostly outside the schema boundary | Needs stronger linkage to function degradation and search scope |
| Equipment | `kg_context.json` | `kg`, `orchestrators`, `schemas` | Asset identity, neighborhood context, component lists | Alias resolution and seed semantics live partly in orchestration logic | Needs tighter mapping between KG entity semantics and workflow use |
| SupportingSystem | `kg_context.json`, indirectly `barrier_analysis.json` | `kg`, `orchestrators`, `schemas` | Support context and safety/barrier relationships partially visible | Often treated as context rather than a first-class causal dependency | Needs stronger role in support-loss reasoning and candidate generation |
| Connection | `kg_context.json`, `signal_evidence.json` | `kg`, `orchestrators`, `signal_evidence`, `schemas` | Structural adjacency, path context, topology-linked signal reasoning | Propagation semantics split between KG expansion and signal evidence logic | Needs clearer contract for causal traversal vs descriptive topology |
| Function | `kg_context.json`, `barrier_analysis.json`, `rca_card.json` | `kg`, `synthesis`, `schemas` | Safety/function references and final RCA summaries | Function degradation often inferred rather than explicitly represented | Needs stronger use in hypothesis interpretation and action prioritization |
| AssetState | No single canonical schema; partly implicit in `tskr_patterns.json`, `causality_candidates.json`, and `rca_card.json` | `orchestrators`, `synthesis` | Sometimes reflected indirectly in candidate and synthesis outputs | Mostly implicit between observations, failure modes, and conclusions | Needs explicit representation to avoid collapsing cause and symptom |
| OperatingCondition | `operational_context.json`, `signal_evidence.json`, partially `tskr_patterns.json` | input layer, `signal_evidence`, `orchestrators`, `schemas` | Event-time context and anomaly context are represented | Split across telemetry, operational context, and temporal scoring | Needs explicit class semantics distinct from asset state |
| CausalEvent | No single dedicated schema; appears through `causality_candidates.json`, `cmms_context.json`, and `rca_card.json` | `orchestrators`, `synthesis`, `cmms_integration` | Appears in hypotheses and summaries | Not clearly separated from investigated event or documentary events | Needs tighter semantic distinction |
| FailureMode | `kg_context.json`, `causality_candidates.json`, `fmea_ingestion_report.json` | `kg`, `schemas`, `orchestrators` | Strongly represented as reusable engineering knowledge | Some workflow semantics are added only during scoring | Needs tighter linkage to candidate and temporal objects |
| SensorObservation | `telemetry_summary.json`, `signal_evidence.json` | input layer, `signal_evidence`, `schemas` | Raw signal/anomaly inputs and merged anomaly evidence | Observation semantics become derived signal objects downstream | Needs explicit split between raw observation and interpreted anomaly evidence |
| SignalEvidence | `signal_evidence.json` | `signal_evidence`, `schemas`, `orchestrators` | Strong artifact-level semantics for anomalies, propagation chains, and gaps | Candidate-specific chain meaning is consumed downstream rather than fully localized | Good alignment already; mostly needs mapping into the overall reasoning model |
| TemporalPattern | `tskr_patterns.json` | `orchestrators`, `schemas` | Strong run-scoped temporal semantics | Recurrence and contradiction meaning still depends on evidence quality upstream | Good alignment already; needs explicit tie to candidate lifecycle |
| ConditionReport | `cmms_context.json`, `document.json`, `processed_text_record.json`, `evidence_bundle.json` | `cmms_integration`, `storage`, `schemas` | Documentary evidence and processed text records are represented | Extracted clues and stance are split across retrieval, processing, and synthesis | Needs cleaner separation between document record and interpreted evidence |
| WorkOrder | `cmms_context.json`, `evidence_bundle.json` | `cmms_integration`, `schemas`, `storage` | Maintenance history and WO-linked evidence are represented | Prior-action semantics and effectiveness reasoning are distributed | Needs stronger role in recurrence and prior corrective-action logic |
| EvidenceAssessment | `evidence_bundle.json`, partly `causality_candidates.json` and `rca_card.json` | `storage`, `orchestrators`, `schemas`, `synthesis` | Retrieval summaries and some evidence metrics are explicit | Assessment semantics are split between retrieval, refinement, and synthesis | Needs explicit unification as a candidate-level evidence object |
| CandidateHypothesis | `causality_candidates.json`, `causality_candidates.v3_2.schema.json` | `orchestrators`, `schemas`, `synthesis`, `validation` | Strong artifact-level presence as ranked candidates | Lifecycle semantics are distributed across D, F, H, and J | Needs explicit recognition as the central run-scoped reasoning object |
| CausalRule | No dedicated schema; encoded in causal/orchestrator logic | `causal`, `orchestrators` | Encoded in scoring and rule logic | Rarely surfaced as first-class artifacts | Needs traceability and versioned reasoning semantics |
| SynthesizedConclusion | `rca_card.json`, `run_manifest.json`, optionally `analyst_override.json` | `synthesis`, `validation`, `schemas`, `cap_integration` | Final RCA output, overrides, and validation status are explicit | Some meaning depends on fallback logic and candidate selection semantics | Good alignment already; needs clearer tie back to candidate/evidence graph |

A few concrete schema files are especially central to the reasoning workflow. The `schemas` directory currently includes `event.json`, `telemetry_summary.json`, `kg_context.json`, `cmms_context.json`, `signal_evidence.json`, `tskr_patterns.json`, `causality_candidates.json`, `causality_candidates.v3_2.schema.json`, `evidence_bundle.json`, `rca_card.json`, `run_context.json`, `run_manifest.json`, `barrier_analysis.json`, `operational_context.json`, `pm_compliance.json`, and supporting files such as `analyst_override.json`, `cap_export_package.json`, `document.json`, `processed_text_record.json`, and `fmea_ingestion_report.json`. The schema README describes these as Draft 7 JSON Schemas used by `RCAArtifactValidator`, and identifies the current core orchestrator artifacts as `event`, `telemetry_summary`, `kg_context`, `causality_candidates`, `evidence_bundle`, `rca_card`, `operational_context`, and `pm_compliance`. ([github.com](https://github.com/idaholab/DACKAR/tree/mandd/rca/src/dackar/RCA/schemas))

### Stage Producer / Consumer Annotation

The table below adds a workflow crosswalk to the alignment view. It also introduces **authority classification**, which clarifies whether each class represents ground truth, observation, or interpretation.

| Reasoning class | Closest schema / artifact | Producer stage(s) | Consumer stage(s) | Authority type | Notes on semantics |
|------|---------------------------|------------------|------------------|----------------|------------------|
| AnalysisRun | run_context.json, run_manifest.json | A, J | All | Derived | Container for all run-scoped reasoning |
| InvestigatedEvent | event.json | External / A | B–J | Authoritative | Ground truth trigger of analysis |
| Equipment | kg_context.json | B (from KG) | 5B–J | Authoritative | Plant model anchor |
| SupportingSystem | kg_context.json | B | D–J | Authoritative | Often underused in causality |
| Connection | kg_context.json, signal_evidence.json | B, B.5 | C–H | Authoritative | Topology vs propagation split |
| Function | kg_context.json, rca_card.json | B, H | D–J | Authoritative | Often implicit until synthesis |
| AssetState | implicit | C–H | D–J | Inferred | Not explicitly modeled → gap |
| OperatingCondition | operational_context.json, signal_evidence.json | External + B.5 + C | D–H | Inferred | Split across artifacts |
| CausalEvent | implicit | D–H | F–J | Inferred | Needs clearer separation |
| FailureMode | kg_context.json | 0, B | C–H | Authoritative | Strong knowledge anchor |
| SensorObservation | telemetry_summary.json | External | B.5–F | Observed | Raw evidence |
| SignalEvidence | signal_evidence.json | B.5 | C–J | Derived | Clean artifact |
| TemporalPattern | tskr_patterns.json | C | D–J | Derived | Clean artifact |
| ConditionReport | cmms_context.json | 5B | C–H | Observed | Documentary evidence |
| WorkOrder | cmms_context.json | 5B | C–H | Observed | Maintenance evidence |
| EvidenceAssessment | evidence_bundle.json | E, F | F–J | Derived | Split across stages |
| CandidateHypothesis | causality_candidates.json | D, F | E–J | Decision | Central reasoning object |
| CausalRule | code | Design-time | C–H | Authoritative | Hidden in logic |
| SynthesizedConclusion | rca_card.json | H | J | Decision | Final output |

This classification enforces a key rule:

- **Authoritative**: ground truth (KG, event)
- **Observed**: raw evidence (signals, reports)
- **Inferred**: interpreted plant condition
- **Derived**: computed reasoning artifacts
- **Decision**: outputs of the RCA engine

Mixing these categories is the primary source of RCA logic errors.

---

## Final Consistency Review

The schema has been reviewed against the RCA workflow with the following conclusions:

The overall logic is consistent with the pipeline architecture. The schema now correctly separates plant knowledge, runtime evidence, and reasoning outputs. It aligns with the staged workflow (A–J) and supports backward reasoning from evidence to cause.

The strongest areas of alignment are:
- SignalEvidence ↔ Stage B.5
- TemporalPattern ↔ Stage C
- CandidateHypothesis ↔ Stages D/F/H
- SynthesizedConclusion ↔ Stage H

These form a coherent reasoning chain.

The remaining controlled gaps are:
- AssetState is not explicitly represented (inferred everywhere)
- OperatingCondition is fragmented across artifacts
- CausalEvent is not cleanly separated from other concepts
- EvidenceAssessment is split across E and F

These do not break the workflow but reduce clarity and traceability.

No logical contradictions remain between schema and workflow.

Completeness is sufficient to:
- guide RCA workflow implementation
- align with existing repo schemas
- support causal reasoning and validation

Further refinement would improve clarity, not correctness.

---

## End-to-End Example: Pump Low-Flow Event

This example shows how the reasoning-oriented schema maps onto a concrete RCA scenario and how the corresponding reasoning objects appear across the workflow.

Assume the investigated event is a **low-flow trip involving a motor-driven pump**. The focal equipment is Pump P-101. The pump receives suction through an upstream valve and depends on a 480V AC supporting power system. During the event window, telemetry indicates low flow and elevated vibration. Historian evidence shows that an upstream valve anomaly preceded the pump anomaly. Maintenance history shows a recent work order involving valve manipulation. The RCA engine must determine whether the most plausible explanation is intrinsic pump degradation, an upstream flow restriction, a support-system issue, or some combination of these.

At the start of the run, the **InvestigatedEvent** is represented by `event.json` and normalized into `run_context.json` during Stage A. It identifies the event, the target asset, the event interval, and severity. In this case, the event points to Pump P-101 as the focal asset.

In Stage B, the workflow queries the KG and produces `kg_context.json`. The **Equipment** context includes Pump P-101, the upstream valve, nearby piping, and relevant support relationships. The **SupportingSystem** context includes the 480V AC supply. The **Connection** context captures that the valve is upstream of the pump in the process path. The **Function** context indicates that the pump performs a flow-delivery function and may also support a safety-relevant function if defined in the KG.

The first important reasoning branch begins with **SensorObservation**. `telemetry_summary.json` provides the original low-flow and vibration signals. In Stage B.5, those raw observations are augmented using historian results and transformed into `signal_evidence.json`. The resulting **SignalEvidence** shows that a valve-related anomaly preceded the pump vibration anomaly and that both preceded the trip. The propagation chain therefore suggests a plausible upstream disturbance rather than an isolated internal pump fault.

In Stage C, the engine produces `tskr_patterns.json`. These **TemporalPattern** objects compare the anomaly timing against known **FailureMode** expectations for pump-related mechanisms. For example, the workflow may evaluate a cavitation-related failure mode, a bearing degradation mode, and a support-loss mode. If the timing is consistent with cavitation induced by low suction conditions, that temporal pattern will score higher than a bearing-only explanation if the observed lag is too short for gradual mechanical degradation.

At this point the workflow is implicitly reasoning over both **OperatingCondition** and **AssetState**, even if those are not yet represented by their own dedicated schema artifacts. The low suction condition induced by the upstream valve is an OperatingCondition. The pump experiencing elevated vibration and degraded hydraulic performance corresponds to an inferred AssetState. This distinction matters. The valve anomaly does not need to damage the pump immediately to explain the event; it may first alter the operating condition, which then drives the pump into an abnormal state and degraded function.

In Stage 5B, the workflow retrieves `cmms_context.json`. This introduces **ConditionReport** and **WorkOrder** evidence. Suppose the recent work order indicates that the upstream valve was manipulated during maintenance, or that a previous condition report mentioned intermittent restriction in that train. These records are documentary evidence, not causes by themselves, but they materially strengthen the plausibility of an upstream-restriction explanation.

In Stage D, the engine generates `causality_candidates.json`. Each **CandidateHypothesis** represents a run-scoped explanation. In this example, likely candidates include:
- cavitation or hydraulic upset on Pump P-101 caused by low suction conditions
- intrinsic pump degradation such as bearing failure
- loss or instability in the 480V AC supporting system
- upstream valve restriction contributing to low-flow conditions

These candidates are scored using structural, temporal, telemetry, evidence, and governance dimensions. Because the topology and signal chain implicate the upstream valve path, and because the work-order history supports recent intervention, the upstream-restriction hypothesis may score strongly even if the event manifested at the pump.

In Stage E, the workflow retrieves evidence into `evidence_bundle.json`. This creates the basis for **EvidenceAssessment**. Supporting snippets may come from a work order, a condition report, or a maintenance note referencing the valve. Contradicting evidence might include the absence of electrical disturbances, which weakens the support-loss candidate. In Stage F, these evidence signals refine the candidates. A pump-bearing hypothesis might drop in rank if there is little supporting evidence and the temporal pattern is weak, while the valve-related hypothesis might rise because both the signal chain and documentary evidence converge on the same explanation.

In Stage H, the workflow synthesizes `rca_card.json`. The **SynthesizedConclusion** selects the highest-ranked candidate as the primary explanation, cites the supporting evidence, and may still include pump degradation as a consequence or alternative rather than the primary cause. In this scenario, the RCA card may conclude that the most plausible root cause is **upstream valve restriction leading to low suction conditions and degraded pump performance**, with pump vibration treated as a downstream symptom of that operating condition rather than the initiating cause.

This example is useful because it shows why the schema must distinguish among several different reasoning roles. The low-flow alarm is not the root cause; it is an observation. The pump is the focal equipment, but not necessarily the origin of the problem. The upstream valve anomaly changes the operating condition, which affects the pump state and function. The work order is evidence, not a cause. The candidate hypothesis is the engine’s interpretation of how these elements fit together. The synthesized RCA conclusion is the final decision layer built on top of that chain.

Summarized as a reasoning path, the example looks like this:

**InvestigatedEvent -> SensorObservation -> SignalEvidence -> OperatingCondition -> TemporalPattern / FailureMode -> CandidateHypothesis -> EvidenceAssessment -> SynthesizedConclusion**

A more plant-centered causal reading of the same case is:

**Upstream valve restriction -> low suction condition -> pump cavitation / degraded hydraulic state -> low flow and vibration -> trip event**

This is the kind of end-to-end reasoning path the schema is intended to support.

---

## Implementation Contracts (Initial Set)

The following section defines **implementation contracts** for the most critical reasoning classes. These contracts are intended to guide development, validation, and artifact design. Each contract specifies required attributes, lifecycle, authority, and validation expectations.

---

### CandidateHypothesis

**Role**: Central reasoning object representing a candidate root cause explanation.

**Required attributes**:
- `candidate_id: string`
- `associated_failure_mode: fm_id`
- `target_equipment: equipment_id`
- `composite_score: float [0,1]`
- `score_breakdown: {structural, temporal, telemetry, evidence, governance}`

**Optional attributes**:
- `evidence_posture`
- `rank`, `rank_delta`
- `review_required`

**Cardinality rules**:
- Must reference exactly 1 primary equipment
- Must reference ≥1 failure mode
- May reference multiple evidence items

**Lifecycle**:
- Created in Stage D
- Updated in Stage F
- Read-only after Stage F

**Storage**:
- Run-scoped artifact (`causality_candidates`)

**Authority type**:
- Decision / inferred

**Validation checks**:
- Score values within [0,1]
- Score breakdown consistent with composite
- Referenced equipment and FM exist in KG

---

### SignalEvidence

**Role**: Structured representation of anomaly evidence and propagation chains.

**Required attributes**:
- `augmented_anomaly_set[]`
- `propagation_chains[]`
- `chain_coverage: float`

**Optional attributes**:
- `per_candidate_chain_score`
- `fetch_gaps[]`

**Cardinality rules**:
- Must contain ≥0 anomalies (empty allowed but flagged)
- Chains derived only from valid anomalies

**Lifecycle**:
- Created in Stage B.5
- Read-only afterward

**Storage**:
- Run-scoped artifact

**Authority type**:
- Derived (from observations)

**Validation checks**:
- No duplicate anomalies
- Chain graph must be acyclic
- All anomalies must map to known or flagged components

---

### TemporalPattern

**Role**: Encodes temporal consistency between anomalies and failure modes.

**Required attributes**:
- `fm_id`
- `dominant_relation`
- `confidence: float`
- `latency_alignment_score`

**Optional attributes**:
- `recurrence_count`
- `temporal_contradiction`

**Cardinality rules**:
- One pattern per failure mode per run

**Lifecycle**:
- Created in Stage C
- Read-only afterward

**Storage**:
- Run-scoped artifact (`tskr_patterns`)

**Authority type**:
- Derived

**Validation checks**:
- Allen relation valid
- Confidence bounded [0,1]
- Latency values consistent with timestamps

---

### EvidenceAssessment

**Role**: Aggregates and evaluates supporting/contradicting evidence for a candidate.

**Required attributes**:
- `candidate_id`
- `supporting_count`
- `contradicting_count`
- `best_support_score`

**Optional attributes**:
- `authority_weight`

**Cardinality rules**:
- One assessment per candidate

**Lifecycle**:
- Created in Stage E
- Updated in Stage F

**Storage**:
- Run-scoped artifact (`evidence_bundle` / derived summary)

**Authority type**:
- Derived

**Validation checks**:
- Evidence counts consistent with retrieved snippets
- Scores within valid bounds

---

### SynthesizedConclusion

**Role**: Final RCA output for analyst review.

**Required attributes**:
- `primary_hypothesis`
- `alternatives[]`
- `evidence[]`
- `recommended_actions[]`

**Optional attributes**:
- `analyst_review`
- `confidence_label`

**Cardinality rules**:
- Exactly 1 primary hypothesis
- ≥0 alternatives

**Lifecycle**:
- Created in Stage H
- Validated in Stage J

**Storage**:
- Run-scoped artifact (`rca_card`)

**Authority type**:
- Decision

**Validation checks**:
- Primary hypothesis must exist in candidates
- All claims must have citations
- Action targets must resolve to valid components

---

## Suggested Next Refinement

- Extend contracts to remaining classes
- Define full attribute schemas (JSON)
- Map contracts directly to pipeline artifacts
- Introduce uncertainty modeling across all classes

