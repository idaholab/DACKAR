# RCA Data Management Strategy — Real Plant Settings
**Date**: April 20, 2026 · **Revision**: Sprint 7 complete (April 21, 2026)
**Context**: How raw plant data translates into RCA pipeline inputs; how to manage data volume sustainably
**Pipeline baseline**: Orchestrator v3.2, stages A–J as specified in `RCA_workflow_april_2.md`

---

## 1. The Core Problem

The current pipeline consumes well-structured JSON fixtures. In a real plant, data arrives from heterogeneous sources — CMMS, anomaly detection systems, EDMS, and industry databases — in formats that bear no resemblance to `telemetry_summary.json` or `kg_context.json`. Two problems compound each other:

1. **Translation gap**: raw data → structured pipeline artifacts is not implemented for most sources
2. **Volume problem**: if we try to pre-store everything in Chroma, the corpus becomes unmanageable, stale, and imprecise

The answer to both problems is **run-scoped retrieval**: all data is fetched fresh at RCA invocation time, anchored to the equipment and component IDs resolved by the KG, and stored only for the duration of the run.

---

## 2. Where Data Management Fits in the RCA Pipeline

The full pipeline is stages A–J. Data management touches four of them directly and has a silent prerequisite (KG population) that precedes Stage A entirely.

```
[ KG POPULATION — prerequisite, offline ]
  Neo4j loaded with: equipment/component register, system topology,
  failure_mode nodes (from FMEA), document linkages (doc_id → asset node)
        │
        ▼
Stage A — Input validation
  Validates: event.json, telemetry_summary.json,
             operational_context.json, pm_compliance.json
  All four are on-demand artifacts assembled at invocation — see §3
        │
        ▼
Stage B — KG context build
  Queries Neo4j: resolves components, failure_modes, document IDs
                 past_events (KG-resident CAP conclusions only — currently empty)
  Output: kg_context (metadata only — document content not yet fetched)
  Note: historical CR/WO recurrence history comes from Stage 5B (cmms_context), not from kg_context.past_events[]
        │
        ▼
Stage 5B / B+ — Run-scoped data fetch  ← THIS DOCUMENT
  Fetches actual document content and CMMS records from plant systems
  Embeds all content → run-scoped Chroma (keyed by run_id)
        │
        ▼
Stage C — TSKR temporal scoring
  Uses: telemetry_summary.json (from TelemetryAdapter)
Stage D — Candidate generation
  Uses: kg_context, tskr_patterns, operational_context, pm_compliance
Stage E — Evidence retrieval
  Queries: run-scoped Chroma only
Stage F — Candidate refinement
  Uses: evidence_bundle from Stage E
Stages G, H, I, J — Ishikawa, synthesis, persistence, validation
        │
        ▼
Stage I — Artifact persistence
  Archives: run-scoped Chroma → {output_dir}/{run_id}/chroma/
```

---

## 3. On-Demand Input Artifacts (Stage A inputs)

These four artifacts are assembled fresh at RCA invocation before Stage A validation runs. None are stored between runs.

### 3.1 event.json
**Source**: CMMS (new CR filed by plant staff)
**Translation**: CMMS CR header fields → event.json via CMMS adapter field mapping
**Key fields**: `event_id`, `asset_id`, `timestamp_start`, `timestamp_end`, `severity`, `symptom_signature`
**Status**: adapter skeleton exists (`maximo_cmms_adapter.py`, `sap_pm_cmms_adapter.py`); GE SAP adapter missing

### 3.2 telemetry_summary.json
**Source**: Plant anomaly detection system (upstream of RCA pipeline — anomalies are pre-identified)
**Translation**: `TelemetryAdapter` maps plant anomaly records → `telemetry_summary` schema
**Key fields**: `signals[].anomalies[].{timestamp_start, timestamp_end, pattern, severity}`, `signals[].instrument_validity_flag`
**Note**: `instrument_validity_flag` is a new field not yet in the schema; requires calibration status API query
**Status**: `TelemetryAdapter` protocol and implementation missing

### 3.3 operational_context.json
**Source**: Two plant systems queried at invocation:
- DCS / plant computer alarm log → `recent_alarms[]` (alarm_id, priority, system_affected, setpoint, actual_value)
- Process historian API → `operating_point` (power level, temperatures, pressures at event_time)
**Key fields**: `recent_alarms[]`, `operating_point`, `nearby_maintenance[]`
**Status**: no assembly module exists; must be built as a composite adapter querying both systems

### 3.4 pm_compliance.json
**Source**: CMMS PM records
**Translation**: CMMS query by equipment_id, lookback from last PM date → pm_compliance schema
**Key fields**: `pm_checks[]` (check_type, result, date, keywords), `last_pm_date`
**Status**: CMMS adapters partially implemented; pm_compliance assembly not wired

---

## 4. Design Decision — Run-Scoped Chroma Only

There is no persistent Chroma index. All document retrieval is performed at RCA invocation time and stored in a run-scoped Chroma collection keyed by `run_id`. The collection is archived after the run, making every RCA fully self-contained and replayable.

**Why no persistent index:**
- Eliminates staleness risk (no index diverging from plant document state)
- Eliminates ingestion infrastructure (no batch jobs, no continuous EDMS connector)
- Auditability is free — the archived run-scoped collection IS the evidence record
- Fits analyst-initiated model: the pipeline assembles exactly what it needs for one event

**KG + equipment register as the retrieval anchor:**

All document retrieval is anchored to equipment/component IDs resolved in Stage B. Every nuclear plant maintains an equipment/component register (required under 10 CFR 50.65 Maintenance Rule) containing each equipment tag, its component type, system assignment, and main characteristics. The KG's `element_usage` nodes represent this register and are the resolution key for class-level queries.

```
equipment_id (instance)
      │
      ├── Instance-level query (direct)
      │        ├── CMMS  →  CR, WO           (by equipment tag / functional location)
      │        └── EDMS  →  SOP, ECA, RCA    (by equipment ID metadata tag)
      │
      └── Class-level query (via KG equipment register)
               │
               KG: component_type → list of all equipment_ids of that type at this plant
               │
               ├── CMMS  →  CR, WO on similar equipment  (by equipment_id list —
               │                                          universally supported)
               ├── FMEA  →  FMEA document content        (by component_type;
               │                                          metadata already in KG Stage B,
               │                                          content fetched here for embedding)
               └── OE LLM → INPO IRIS + NRC ADAMS        (internet API — see §6)
```

The class-level path retrieves **similar events across similar equipment** — e.g., all bearing failures on centrifugal pumps at this plant, not just failures on pump P-101A. Resolving via an equipment_id list sidesteps native query capability differences across CMMS systems — every CMMS supports querying by equipment ID.

---

## 5. Stage 5B — Run-Scoped Data Fetch

This is the existing Stage 5B in the codebase (`cmms_integration/cmms_context_builder.py`), extended with four new fetch types. It runs between Stage B (KG context) and Stage C (TSKR).

**Important distinction**: Stage B already fetches FMEA *metadata* from the KG (`kg_context.failure_modes[]` — fm_id, component_id, expected_latency, expected_symptoms). Stage 5B fetches FMEA *document content* from the FMEA source system for embedding into Chroma. These are two separate data flows serving different purposes.

```
Stage B output: kg_context
    └── components[].{component_id, component_type}
    └── documents[].doc_id          ← KG-ranked relevant document IDs (metadata only)
    └── failure_modes[].{fm_id, component_type, failure_mode_class}
                         ↑ metadata already in KG — no fetch needed for scoring
    └── seed_context.asset_ids

Stage 5B: Run-Scoped Data Fetch
    │
    ├── 1. Instance-level CMMS fetch  (already implemented)
    │       Query CRs + WOs by equipment_id + KG sister equipment IDs
    │       Lookback window: last_PM_date → event_time
    │
    ├── 2. Class-level CMMS fetch  ← NEW
    │       KG resolves: component_type → list of equipment_ids at this plant
    │       Query CRs + WOs for each equipment_id in that list
    │       Filter by failure_mode_class keywords + date window to limit volume
    │       Ranked by recurrence signal (most recent, same failure mode first)
    │
    ├── 3. EDMS fetch  ← NEW
    │       For each doc_id in kg_context.documents[]:
    │           Fetch document content from EDMS by doc_id
    │       Document types: SOP, ECA, RCA, operating procedures
    │
    ├── 4. FMEA document fetch  ← NEW
    │       For each component_type in kg_context.components[]:
    │           Fetch FMEA document content from FMEA system by component_type
    │           (Distinct from FMEA metadata in KG — this is the source document
    │            for embedding and evidence retrieval in Stage E)
    │
    └── 5. OE LLM query  ← FUTURE
            For each (component_type, failure_mode_class) in kg_context:
                Call IRIS LLM API  (fine-tuned on INPO IRIS, internet API)
                Call ADAMS LLM API (fine-tuned on NRC ADAMS, internet API)
                Both must return source citations (doc_id, title, section, year)

    All fetched content → two-path ingestion (§7) → embed → run-scoped Chroma

Stage E: Evidence retrieval queries run-scoped Chroma only
Stage I: Archive run-scoped Chroma → {output_dir}/{run_id}/chroma/
```

---

## 6. Evidence Source Tiers

Each document in the run-scoped Chroma carries a `source_tier` metadata tag used for authority weighting in evidence scoring. **Note**: `source_tier` is a new field not currently in the `evidence_bundle` schema — it requires a schema update.

| source_tier | document types | query path | authority weight | recency window |
|-------------|---------------|------------|-----------------|----------------|
| `plant_instance` | CR, WO — operational records for this equipment | instance-level CMMS | highest | 90 days before event, 7 days after |
| `plant_instance` | ECA, RCA — analysis documents for this equipment | instance-level EDMS | highest | **None — timeless** (see note below) |
| `plant_procedure` | SOP, operating procedures | EDMS by equipment tag | high | **None — timeless** |
| `plant_fmea` | FMEA document content for this component type | class-level FMEA fetch | high | **None — timeless** |
| `plant_family` | CR, WO for similar equipment at this plant | class-level CMMS | medium-high | Governed by recurrence lookback window |
| `oe_iris` | INPO IRIS events and OE reports | IRIS LLM internet API | medium | **None — timeless** |
| `oe_adams` | NRC ADAMS: INs, GLs, NUREGs, inspection reports | ADAMS LLM internet API | medium | **None — timeless** |

> **Timeless vs operational document types**: CR and WO records are operational records whose relevance decays with time — a CR from 5 years ago is less likely to reflect current equipment condition than one from last month. The ±90-day window in Stage B's Neo4j query applies only to these types. ECA, RCA, FMEA, SOP, MANUAL, and BULLETIN documents are timeless engineering knowledge — their relevance to a failure mode does not decay. Stage B retrieves them regardless of creation date and does not apply a recency proximity bonus to their priority scores. Applying recency decay to FMEA documents would penalise the most authoritative source in the corpus by preferring a recent but shallow CR over a decades-old FMEA that directly addresses the failure mode. (B2 — implemented Sprint 7.)

The `all_claims_cited` validation gate (Stage J) requires a valid `doc_id` for any primary hypothesis citation. Both OE LLM APIs must return source citations — ungrounded LLM output without a traceable `doc_id` is contextual only and cannot support a primary claim.

---

## 7. Document Ingestion — Two-Path Model

All fetched content passes through one of two ingestion paths before embedding into the run-scoped Chroma:

**Path A — Structured records (CMMS database entries)**
CMMS systems expose records as structured entries with well-defined fields. Direct field mapping applies; NLP is needed only for free-text narrative fields.

```
CMMS record (CR / WO / PM) — structured API response
      │
      ├── structured fields → direct JSON field mapping
      │                        (maximo_default.json / sap_pm_default.json / ge_sap.json)
      └── narrative fields  → NER + embed
                               (description, work performed, as-found condition)
```

**Path B — PDF or document format**
Some CMMS records may be returned as PDFs depending on plant digitization maturity. All EDMS documents (SOP, ECA, RCA) and FMEA source documents are typically PDF or Word. Both require the parsing path.

```
CMMS record (PDF)  /  EDMS document (SOP / ECA / RCA)  /  FMEA document
      │
      document type classification
      │
      parser selection (pdfParser / mdParser / fmeaParser)
      │
      section-aware chunking
      │
      NER entity extraction
      │
      embed → run-scoped Chroma
```

**Format detection** must be handled transparently at the adapter level — the `CMMSAdapter` and `EDMSAdapter` detect whether a record is a structured object or a PDF and route accordingly without manual configuration.

---

## 8. OE LLM Tier (Future)

Industry OE data (INPO, NRC) is fundamentally different from plant-specific documents:
- Scope is fleet-wide, not asset-specific
- Authority role is plausibility amplification, not confirmation
- No recency decay — OE from 2005 is as valid as OE from 2025

**Architecture**: two fine-tuned LLMs available via internet API:

| Backend | Corpus | Access |
|---------|--------|--------|
| **IRIS LLM** | INPO IRIS — fleet event reports, SERs, OE reports | Internet API (INPO member access) |
| **ADAMS LLM** | NRC ADAMS — INs, GLs, NUREGs, inspection reports | Internet API (public) |

Both called in Stage 5B with query `(component_type, failure_mode_class)`. Both must return structured responses with source citations (doc_id, title, section, year) — hard API contract requirement for traceability.

The existing `LLMClient` protocol supports both backends without pipeline restructuring. Two new client implementations needed: `IRISLLMClient` and `ADAMSLLMClient`. No local index or RAG infrastructure required.

---

## 9. Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Structured CMMS field mapping (Maximo, SAP PM) | Partial | Field maps defined; adapters not fully implemented |
| GE SAP adapter + field mapping | Missing | New adapter needed |
| PDF parsing (`pdfParser.py`) | Partial | Basic text extraction; limited section role detection |
| Structured doc parsing (`mdParser.py`) | Partial | CR, WO, SOP, ECA, FMEA, OE types covered |
| FMEA parsing (`fmeaParser.py`) | Partial | Spreadsheet; class-level only |
| NER entity extraction (`ner/`) | Partial | Not validated end-to-end on plant text |
| Chunking + embedding pipeline | Missing | No pipeline: ParsedDoc → chunks → embed → Chroma |
| Format detection in CMMS/EDMS adapter | Missing | Structured vs. PDF routing not implemented |
| EDMS connector | Missing | No adapter for fetching documents by doc_id |
| `TelemetryAdapter` | Missing | No adapter translating plant anomaly records |
| `operational_context` assembly | Missing | No composite adapter (DCS alarms + historian) |
| `pm_compliance` assembly | Missing | CMMS adapter partially implemented; assembly not wired |
| Class-level CMMS query (Stage 5B extension) | Missing | Instance-level exists; component_type family query does not |
| `EquipmentSimilarityResolver` integration | Missing | Module exists in `equipment_similarity/`; not wired into Stage 5B |
| `source_tier` field in evidence_bundle schema | Missing | Schema update required |
| `instrument_validity_flag` in telemetry_summary schema | Missing | Schema update required |
| OE LLM backends (`IRISLLMClient`, `ADAMSLLMClient`) | Future | Internet API endpoints not yet confirmed |

---

## 10. KG Population — Silent Prerequisite

The entire pipeline assumes the KG (Neo4j) is populated before the first RCA run. This is a one-time offline process that must be completed per plant. The `kg/` folder contains the ingestion workflows:

| KG content | Source | Ingestion tool |
|------------|--------|---------------|
| Equipment/component register | Plant equipment database or MBSE model | `kg_schema_builder_workflow.py` |
| System topology and component relationships | Plant P&IDs, MBSE model | `kg_ingest_neo4j_workflow.py` |
| Failure mode nodes (FMEA class-level) | FMEA spreadsheets | `kg_ingest_fmea_workflow.py` |
| Document linkages (doc_id → asset node) | EDMS metadata export | Manual or custom loader |
| Accepted RCA conclusions (write-back) | CAP closure, analyst sign-off | `cap_integration/` (future) |

KG governance (update cadence, ECN process, FMEA taxonomy ownership) is a plant process requirement outside the software — see `RCA_workflow_april_2.md` §5.

---

## 11. Integration Gap — Component Family Retrieval

The `EquipmentSimilarityResolver` module (`equipment_similarity/`) resolves sister equipment by specification similarity. It is not wired into the main pipeline.

**Design clarification — where historical event data lives**: `kg_context.past_events[]` is not the source of recurrence history. That field is reserved exclusively for accepted RCA conclusions written back to the KG from closed CAP items; it is currently empty because CAP write-back is not yet implemented. All recurrence history — the historical CR/WO records that feed Stage C (TSKR recurrence profile) and Stage D (historical-event candidate pool) — comes from `cmms_context.cr_records[]`, assembled at Stage 5B from the live CMMS. This is an intentional design: CMMS is the system of record for plant events; the KG is the system of record for equipment topology and failure mode taxonomy.

**Integration point for `EquipmentSimilarityResolver`**: Stage 5B, not Stage B. The resolver expands the set of sister equipment IDs passed to the CMMS query (class-level fetch type 2 in §5). Its output augments `cmms_context.sister_components[]` with Tier-2 (failure-mode overlap) and Tier-3 (spec similarity) results alongside the existing Tier-1 (topological) sisters.

The single wiring change needed:
1. **Stage 5B** (`cmms_context_builder.py`): call `EquipmentSimilarityResolver.by_failure_mode()` and `by_spec_embedding()` to obtain Tier-2 and Tier-3 sister IDs; merge into the `sister_ids` list before the CMMS query is issued.

`kg_context.past_events[]` remains reserved for KG-resident CAP conclusions and requires no change.

Without this, the recurrence model in Stage C/D only sees events on the specific asset and its Tier-1 KG neighbors. A bearing failure on a sister pump in a different system is invisible, even if it is the strongest recurrence signal.

---

## 12. Supported CMMS Systems

| System | Adapter status | Field mapping | Notes |
|--------|---------------|---------------|-------|
| IBM Maximo | Skeleton (`maximo_cmms_adapter.py`) | `maximo_default.json` | Most common in US nuclear plants |
| SAP PM | Skeleton (`sap_pm_cmms_adapter.py`) | `sap_pm_default.json` | Common in newer and GE-affiliated plants |
| GE SAP | Missing | Not yet defined | GE-specific SAP implementation; needs adapter + field mapping delta vs. SAP PM |

All three must support: instance-level query by equipment tag, class-level query by equipment_id list, format detection (structured vs. PDF).

---

## 13. Assumptions

**A1 — Anomaly detection is upstream**
The plant has an anomaly detection system that identifies signal anomalies and returns them via API. The RCA pipeline does not perform anomaly detection. *If violated*: `telemetry_summary.json` must be assembled manually.

**A2 — EDMS documents tagged with equipment IDs**
Relevant documents (SOPs, ECAs, RCAs) in the EDMS carry equipment ID as a queryable metadata field. *If violated*: EDMS fetch returns nothing; evidence bundle contains only CMMS and FMEA records.

**A3 — CMMS queryable by equipment ID**
All three target CMMS systems support querying CR/WO records by equipment ID. No native component_type filter required — class-level queries use a KG-resolved equipment_id list. *If violated*: both instance-level and class-level fetch break.

**A3b — CMMS record format may vary**
CR, WO, PM records may be returned as structured objects or PDF documents. Both are valid inputs; the adapter detects and routes accordingly. *If violated* (format detection fails): records in the undetected format are silently dropped.

**A4 — KG component_type assignments accurate and maintained**
The KG correctly assigns `component_type` to all `element_usage` nodes. FMEA retrieval and class-level queries depend on this. *If violated*: FMEA and family-level queries return wrong or empty results.

**A5 — KG document linkages complete**
Relevant documents are linked to asset nodes in the KG. `kg_context.documents[].doc_id` in Stage B depends on these edges. *If violated*: relevant SOPs, ECAs, RCA reports not fetched from EDMS.

**A6 — Calibration status queryable**
Instrument calibration status accessible via API to populate `instrument_validity_flag`. *If violated*: all signals default to `validity: unknown`; TSKR cannot adjust confidence for out-of-calibration sensors.

**A7 — Analyst-initiated triggering**
RCA is initiated by a human analyst selecting an event from the CMMS. No auto-trigger. *If violated*: data freshness and completeness guarantees harder to enforce.

**A8 — FMEA data current**
FMEA failure mode records in the KG reflect current plant configuration. Expected latency windows and symptom patterns are valid. *If violated*: TSKR temporal scoring produces incorrect latency violation flags.

**A9 — Single-plant scope**
Run-scoped retrieval queries one plant's CMMS and EDMS. Cross-plant fleet data is out of scope except through the OE LLM tier. *If violated*: cross-plant recurrence requires the OE LLM tier to be operational.

---

## 14. Limitations

**L1 — Retrieval bounded by KG completeness**
The causal search space is defined by the KG. A failure mode not in the KG cannot be generated as a candidate or retrieved as evidence. Novel first-of-kind failure modes are invisible. This is the closed-world assumption identified in the April 20 systems engineering review.

**L2 — Evidence quality depends on EDMS tagging discipline**
Poorly tagged documents (missing equipment ID metadata) are silently missed. Evidence gaps may appear as "no evidence found" rather than "retrieval failed."

**L3 — PDF CMMS records lose structured field semantics**
When CMMS records are in PDF format, structured field semantics (equipment tag, priority, as-found condition) are lost unless the PDF parser reconstructs them from layout. Plants with high PDF prevalence will see reduced retrieval precision.

**L4 — Class-level CMMS query volume risk**
Querying all equipment of a common component_type (e.g., gate valves) can return large result sets. Configurable volume caps and failure_mode_class keyword pre-filters are required.

**L5 — EquipmentSimilarityResolver not integrated**
Family retrieval relies on KG `component_type` matching, not specification-based similarity. Two pumps with different component_type labels but identical hydraulic specifications will not be matched.

**L6 — Embedding latency at invocation time**
All documents fetched and embedded fresh per run. No embedding cache between runs by design. For large document sets, this adds wall-clock time to invocation.

**L7 — No cross-plant recurrence in v1**
Class-level CMMS query covers this plant only. Cross-plant recurrence patterns are invisible until the OE LLM tier is operational.

**L8 — OE LLM tier is future state**
No `oe_iris` or `oe_adams` entries in the evidence bundle until `IRISLLMClient` and `ADAMSLLMClient` are implemented. Candidates well-supported by industry OE but with sparse plant-specific documentation will be under-scored.

**L9 — CMMS historical depth is finite**
Records older than the plant's CMMS data retention window are not retrievable. Long-term recurrence patterns beyond that window are invisible.

**L10 — KG not updated in real time**
KG updates on a scheduled basis. An RCA run immediately after a plant modification may use a KG that does not yet reflect the new configuration.

**L11 — Instrument validity flag may be unavailable**
If the calibration management system does not expose a queryable API, `instrument_validity_flag` defaults to `unknown`. TSKR cannot differentiate valid anomalies from instrument-induced ones.

---

## 15. Resolved Design Decisions

**April 20, 2026:**

1. **Fleet data access**: Architecture leaves this open. Fleet-level CR/WO data across sister plants is an optional extension — when available it broadens the class-level CMMS query beyond this plant's boundary; when unavailable it degrades gracefully to plant-only recurrence.

2. **OE LLM backends**: Two fine-tuned LLMs via internet API — one on INPO IRIS data, one on NRC ADAMS data. No local RAG index needed. Hard requirement: both APIs return source citations (doc_id, title, section, year).

3. **Run-scoped Chroma retention**: Archive indefinitely. The archived collection is part of the permanent audit record for that RCA run.

4. **Class-level CMMS query pattern**: Resolved as KG-resolved equipment_id list → per-ID CMMS queries. Universally supported across all three CMMS systems.

**April 21, 2026 (Sprint 7):**

5. **Timeless vs operational document window policy** (B2): The ±90-day document date window applies to operational records only (CR, WO, ECR). Analysis documents — ECA, RCA, FMEA, SOP, MANUAL, BULLETIN — are timeless: Stage B retrieves them regardless of creation date, and the Python enrichment loop applies no recency-proximity bonus to their priority scores. Rationale: an FMEA or ECA written three years ago documenting a known failure mode is equally authoritative today; applying recency decay would penalise the most reliable sources in the corpus in favour of more recent but shallower operational records. The operational window (90 days) is appropriate for CR/WO because it reflects current equipment condition; it is not appropriate for engineering analysis documents whose validity is governed by their revision status, not their age.

---

## 16. Remaining Open Questions

1. **IRIS LLM API contract**: Endpoint, authentication, and citation response schema for the IRIS-trained LLM. Must confirm citation format (doc_id, title, section, year) before `IRISLLMClient` can be implemented.

2. **ADAMS LLM API contract**: Same for the ADAMS-trained LLM.

3. **GE SAP data model**: Which CR/WO/PM fields differ from standard SAP PM? Determines the delta between `sap_pm_default.json` and a new `ge_sap_default.json`.

4. **Fleet data access scope**: If fleet CR/WO data from sister plants becomes available, what data sharing agreements and anonymization requirements apply?
