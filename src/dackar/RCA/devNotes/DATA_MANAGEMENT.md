# RCA Data Management — Design Decisions and Rationale

**Status:** Curated development record. Distilled from the April 2026 data-management
strategy, the FMEA-handling spec, the Step-1 hardening plan and the data-elements survey.

This document captures *why* the RCA pipeline ingests plant data the way it does: the
run-scoped retrieval model, how heterogeneous FMEAs are normalized, and the Step-1 data
adequacy / coverage policy. It is a **development-choices** record, not a description of
current code. For the live data-coverage report, the schemas and the validator, see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §5; where this and the code disagree, the code
wins.

---

## 1. The core problem

The pipeline reasons over well-structured JSON artifacts (`event.json`,
`telemetry_summary.json`, `kg_context.json`, and so on). Real plant data arrives from
heterogeneous systems — CMMS, anomaly detection, EDMS, industry OE databases — in formats
that look nothing like those artifacts. Two problems compound: a **translation gap** (raw
source to structured artifact is unimplemented for most sources) and a **volume problem**
(pre-storing everything in a persistent vector index becomes stale, unmanageable and
imprecise).

The decided answer to both is **run-scoped retrieval**: all data is fetched fresh at RCA
invocation time, anchored to the equipment and component IDs resolved by the knowledge
graph, and stored only for the duration of the run.

## 2. Decision — run-scoped Chroma only

There is **no persistent Chroma index**. Every document retrieval happens at invocation
time into a run-scoped collection keyed by `run_id`, which is then archived with the run.
The rationale:

- **No staleness** — the index cannot diverge from plant document state because it is
  built fresh each run.
- **No ingestion infrastructure** — no batch jobs, no continuous EDMS connector.
- **Auditability is free** — the archived run-scoped collection *is* the evidence record,
  which is reconstructable and replayable.
- **Fits the analyst-initiated model** — the pipeline assembles exactly what one event
  needs. (Decision confirmed April 20, 2026: archive the run-scoped collection
  indefinitely as part of the permanent audit record.)

### The KG and equipment register as the retrieval anchor

All retrieval is anchored to equipment/component IDs resolved during KG-context build.
Every plant maintains an equipment register (required under 10 CFR 50.65), and the KG's
`element_usage` nodes represent it. Two query paths come off a resolved `equipment_id`:

- **Instance-level** — query CMMS (CR, WO) and EDMS (SOP, ECA, RCA) directly by equipment
  tag.
- **Class-level** — the KG resolves a `component_type` to the list of all equipment IDs of
  that type at the plant, then CMMS, FMEA and OE queries run against that list.

The class-level path is what retrieves *similar events across similar equipment* (every
bearing failure on centrifugal pumps, not just failures on P-101A). Decided to resolve
class-level queries through a KG-supplied `equipment_id` list rather than native
component-type filters, because an equipment-ID list is universally supported across CMMS
systems whereas component-type filtering is not.

### Evidence source tiers and the timeless-document rule

Each retrieved document carries a `source_tier` used for authority weighting. The durable
decision inside this (implemented Sprint 7, April 21, 2026) is the **timeless vs
operational distinction**:

- **Operational records** (CR, WO, ECR) decay with time; a recency window applies (roughly
  90 days before the event, 7 days after).
- **Engineering-knowledge documents** (ECA, RCA, FMEA, SOP, MANUAL, BULLETIN) are
  **timeless** — retrieved regardless of age, with no recency-proximity bonus.

Rationale: applying recency decay to an FMEA would penalize the most authoritative source
in the corpus in favor of a recent but shallow CR. The validity of an engineering document
is governed by its revision status, not its age.

### Industry OE as a future LLM tier

Industry OE (INPO IRIS, NRC ADAMS) is fleet-wide, not asset-specific, and its role is
plausibility amplification rather than confirmation. Decided architecture: two fine-tuned
LLMs reached by internet API, with a hard contract that both return source citations
(doc_id, title, section, year) so OE output is traceable and can support a primary claim.
No local RAG index. This tier was future-state at the time of writing.

## 3. On-demand input artifacts

Four artifacts are assembled fresh at invocation, before input validation, and are not
stored between runs:

- `event.json` — from a CMMS condition report (header fields mapped through a CMMS
  adapter).
- `telemetry_summary.json` — from the plant anomaly-detection system. **Anomaly detection
  is upstream**: the pipeline reasons over anomalies as facts and does not perform signal
  processing itself.
- `operational_context.json` — a composite of DCS/plant-computer alarm log plus process
  historian operating point at event time.
- `pm_compliance.json` — from CMMS PM records (also auto-built inside the run when not
  supplied).

## 4. FMEA handling

FMEAs are the primary source of KG failure-mode nodes, but they vary enormously (AIAG 4th
and 5th, IEC 60812, MIL-STD-1629A, nuclear-utility templates). Three structural facts drove
the design:

- A small set of fields is **universal** (item/function, failure mode, potential causes,
  local/system/end effect, detection method, corrective actions, severity).
- Several fields are **format-specific** (occurrence, detection rating and RPN exist in
  AIAG/IEC but not MIL-STD, which uses failure rate λ and criticality instead).
- The fields the pipeline most wants — `expected_latency_min/max_hours`,
  `expected_anomaly_pattern`, instance-level `applies_to_component_id` — are **absent from
  every standard FMEA format** and must be added deliberately.

### Decision — a normalization layer with named format profiles

A normalization layer sits between raw FMEA files and KG ingestion, making format
differences explicit rather than silently dropping them. It defines named profiles
(`aiag_4th`, `aiag_5th`, `mil_std_1629a`, `iec_60812`, `nuclear_generic`, and an
auto-detect profile with a confidence score), each declaring its column map, derivation
rules, and required vs optional fields. Derivations compute missing canonical fields from
present ones (RPN from S×O×D; occurrence from λ×mission_time for MIL-STD; severity from a
criticality mapping; cause/effect splitting). Every ingested field is tagged with a
quality status (`present_native`, `derived`, `nlp_inferred`, `missing_critical`,
`missing_optional`, `missing_enrichment`) so downstream governance can react. Multi-level
effects (local / system / safety) are preserved as distinct properties, never flattened,
because the safety-effect field is what the KG safety-function linkage depends on.

This layer was implemented (`doc_parsers/fmea_normalizer.py`, expanded `fmeaParser.py`,
`kg_ingest_fmea_workflow.py`, a `fmea_ingestion_report.json` schema). `failure_mechanism`
is enforced as required at parse time; the column resolver is first-match-wins so one
header cannot map to two canonical fields.

### Decision — latency is an optional enrichment, not a required input

`expected_latency_min/max_hours` is the single most consequential missing field (it is how
Stage C separates two failure modes with the same Allen relation), but it is absent from
all standard FMEAs and, where present, is a rough expert estimate with wide bounds.
Calibrating a scoring function to uncertain bounds of uncertain quality is architecturally
fragile.

The resolution: **treat latency as an optional enrichment and let Stage B.5's
topology-driven anomaly sequencing be the primary temporal discriminator.** An anomaly that
demonstrably preceded the event in plant data and whose sensor is topologically upstream is
stronger temporal evidence than a latency bound asserted in a possibly-never-updated
document. Consequently Stage C's latency alignment must **abstain** (neutral 0.50 score,
`latency_violation_type: "not_available"`) when bounds are absent, rather than apply a floor
penalty. This also makes the pipeline functional on day one of deployment, before any FMEA
enrichment has been done.

### The enrichment workflow

A human-in-the-loop step that runs *before* the pipeline, not inside it, annotating KG
failure-mode nodes with the non-standard fields. Deliberately bounded: it uses a
**neighborhood-first strategy** (enrich only the failure modes in the current event's KG
neighborhood — typically 10–50 FMs, a manageable session) and enrichments accumulate
persistently across runs, so frequently investigated equipment becomes a verification task
over time. Priority order is anomaly-pattern first, safety-function link second, latency
bounds third (deliberately not first, per the argument above), cause/consequence
refinement fourth. Every enrichment is written with full provenance (reviewer, timestamp,
basis, confidence).

## 5. Step 1 data-adequacy policy and its hardening

Step 1 (data management) must verify input adequacy before analysis proceeds, and a
data-limited flag must have a defined effect on conclusion confidence (it is not merely
informational). The April 25 hardening made Step 1 "green" with these durable decisions:

- **An 8-family coverage report**: `kg_context`, `chroma_corpus`,
  `upstream_anomaly_inputs`, `telemetry_detail`, `soe_log`, `alarm_log`,
  `protection_logic_context`, `configuration_change_records` (later extended with the
  Category F/K/L families: environmental monitoring, vendor/supply-chain, training records).
- **Per-artifact quality drives status, not mere presence** — telemetry
  (`missing_fraction`, `flatline_detected`, `outlier_fraction`), SOE (`clock_sync_ok`,
  dropped/duplicate counts) and alarm (`missing_fraction`, `clock_sync_ok`) quality fields
  push a family from `complete` to `partial`.
- **A family that is simply not provided is `not_assessed`, not `missing`** — so an
  optional family's absence does not spuriously drag the overall status to `partial`.
- **Paired-data coupling** — SOE records require protection-logic context to interpret
  (trip setpoints, permissive logic). When SOE is present but protection logic is absent,
  the pairing is flagged (escalated from a warning to `"violated"`, and surfaced into
  `analyst_decisions_required`, not just `degraded_reasons`), because the barrier-logic gate
  would otherwise run degraded and silent.
- **Coverage quality feeds the score** — a weighted coverage factor flows into the
  candidate `quality_multiplier`, prioritizing structural (`kg_context`) and signal
  (`upstream_anomaly_inputs` / `telemetry_detail`) quality over optional-artifact quality.
- **Strict full-mode validation blocks silent degradation** — telemetry is mandatory, the
  paired-data requirement is enforced, and an `overall_status: complete` that contradicts a
  missing required family is a validation error.

## 6. Known limitations (dated record)

Recorded as the honest boundary of the design at the time (April 2026):

- **Closed-world assumption** — the causal search space is bounded by the KG. A failure
  mode not in the KG cannot be generated or retrieved; novel first-of-kind failure modes
  are invisible.
- **Evidence quality depends on EDMS tagging discipline** — poorly tagged documents are
  silently missed and can look like "no evidence found" rather than "retrieval failed."
- **No cross-plant recurrence in v1** — class-level CMMS covers this plant only until the
  OE LLM tier is operational; candidates well-supported by industry OE but sparse in
  plant-specific documentation are under-scored.
- **Embedding latency at invocation** — no cross-run embedding cache by design, so large
  document sets add wall-clock time to a run.
- **KG is not real-time** — a run immediately after a plant modification may use a KG that
  does not yet reflect the new configuration.
- **`EquipmentSimilarityResolver` not wired in** — sister-equipment family retrieval relies
  on KG `component_type` matching, not specification similarity, so two pumps with different
  type labels but identical hydraulics are not matched. The intended integration point is
  the run-scoped fetch stage, not KG-context build.

A design clarification worth preserving: `kg_context.past_events[]` is **not** the source
of recurrence history. It is reserved for accepted RCA conclusions written back to the KG
from closed CAP items (and was empty because write-back was unimplemented). All recurrence
history that feeds temporal scoring and the historical-event candidate pool comes from the
live CMMS at the run-scoped fetch stage. This is intentional: CMMS is the system of record
for plant events; the KG is the system of record for topology and failure-mode taxonomy.
