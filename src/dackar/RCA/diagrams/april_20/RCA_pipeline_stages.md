# RCA Pipeline — Stage-by-Stage Reference
**Date**: April 22, 2026 · **Revision**: Sprint 7 complete + Stage 0 added
**Baseline**: Orchestrator v3.2 · Schema set v3.2
**Companion documents**: `RCA_pipeline_flowchart.md` (architecture) · `RCA_workflow_april_2.md` (formal spec) · `RCA_Data_Management_Strategy.md` (data layer) · `RCA_FMEA_handling_spec.md` (FMEA formats, normalization, enrichment) · `RCA_stage_B5_signal_evidence_spec.md` (topology anomaly fetch, propagation chains)

**Target audience**: Systems engineers and RCA practitioners using or evaluating the pipeline. This document explains what each stage does, why, what data it transforms, and where the current implementation has known limitations.

---

# Part 1 — RCA Workflow Overview

## 1.1 What the Pipeline Does

The RCA pipeline is an automated decision-support system for nuclear plant corrective action program (CAP) root cause analysis. Given an abnormal event — a protective trip, a component failure, a degraded performance condition — the pipeline reasons over plant-specific knowledge, live CMMS records, and historical documentation to produce a ranked list of root cause hypotheses and a structured RCA card. That card shoul not be submitted directly to the CAP system but forwarded to an analyst for review and override.

The pipeline does not replace the analyst. It is a co-pilot: it assembles, scores, and structures evidence at a speed and consistency that manual RCA cannot match, and it surfaces uncertainty and gaps explicitly so the analyst can focus effort where it matters most. Every output is traceable — every score to its inputs, every hypothesis to its evidence, every recommendation to its rationale.

## 1.2 Pipeline Architecture

The pipeline has ten stages designated A through J. Stages run sequentially; each stage produces one or more named artifacts that subsequent stages consume.

```
STAGE 0 INPUTS (run once — not per event)
  mbse_model.json       sensor_component_map.csv
  fmea_data.csv/xlsx    [plant database APIs: CMMS, EDMS]
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 0  │  KG Initialization                  [Neo4j write — offline] │
│           │  Builds plant topology, failure mode catalog, sensor links, │
│           │  and document metadata index in Neo4j from static inputs    │
│           │  → populated Neo4j KG  (prerequisite for Stage B)          │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼  (run per event — pipeline begins here)
INPUT ARTIFACTS
  event.json            telemetry_summary.json
  operational_context.json    pm_compliance.json
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE A  │  Run Setup & Input Validation                               │
│           │  Assigns run_id; checks input artifact consistency          │
│           │  → run_context                                              │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE B  │  KG Context Construction                  [Neo4j live]      │
│           │  Expands equipment neighborhood; retrieves failure modes,   │
│           │  documents, safety functions from knowledge graph           │
│           │  → kg_context                                               │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 5B │  Run-Scoped Data Fetch               [CMMS/EDMS/FMEA live]  │
│           │  Fetches CR/WO history, document content; builds Chroma     │
│           │  evidence corpus; assembles cmms_context                    │
│           │  → cmms_context + run-scoped Chroma collection              │
└─────────────────────────────────────────────────────────────────────────┘
        │                    (Stage 5B and Stage B.5 are independent;
        │                     they can run in parallel after Stage B)
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE B.5│  Topology-Driven Anomaly Fetch       [Historian live]       │
│           │  Queries historian for all neighborhood sensors; merges     │
│           │  with telemetry_summary; constructs propagation chain DAG  │
│           │  → signal_evidence                                          │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE C  │  TSKR Temporal Scoring                                      │
│           │  Computes Allen interval relations between anomaly windows  │
│           │  and event; scores latency alignment, recurrence trend      │
│           │  → tskr_patterns                                            │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE D  │  Candidate Generation  (pre-evidence, v1)                   │
│           │  Scores each failure mode on 5 dimensions (structural,      │
│           │  temporal, telemetry, evidence proxy, governance);          │
│           │  dual hard filter; top-k=10 ranked list                     │
│           │  → causality_candidates v1                                  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE E  │  Evidence Retrieval                   [Chroma run-scoped]   │
│           │  Query plan per candidate; retrieves supporting,            │
│           │  contradicting, contextual snippets from Chroma             │
│           │  → evidence_bundle                                          │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE F  │  Candidate Refinement  (post-evidence, v2)                  │
│           │  Replaces Stage D evidence proxy with Chroma retrieval;     │
│           │  re-ranks; classifies evidence posture; flags near-ties     │
│           │  → causality_candidates v2                                  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE G  │  Ishikawa Structuring  (optional)                           │
│           │  Maps top candidates and evidence to 6-branch fishbone      │
│           │  categories; annotates each branch with source traceability │
│           │  → ishikawa_matrix  (null when skipped)                     │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE H  │  RCA Synthesis                                              │
│           │  Assembles rca_card: executive summary, primary hypothesis, │
│           │  alternatives, evidence, recommended actions, analyst flags │
│           │  → rca_card                                                 │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE I  │  Artifact Persistence                                       │
│           │  Writes all artifacts to run directory; archives Chroma     │
│           │  collection; writes run_status sentinel                     │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE J  │  Validation & Run Manifest                                  │
│           │  JSON schema validation + cross-artifact consistency;       │
│           │  computes review_hooks; produces terminal run_manifest      │
│           │  → run_manifest  →  writeback / analyst_review / remediation│
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT
  rca_card.json         run_manifest.json
  causality_candidates_v2.json   evidence_bundle.json
  (+ 8 more named artifacts in {output_dir}/{run_id}/)
```

## 1.3 Stage-by-Stage Summary

| Stage | Name | Purpose | Primary output |
|-------|------|---------|----------------|
| 0 | KG Initialization | Build plant topology, failure modes, sensor links, doc metadata in Neo4j | Populated Neo4j KG |
| A | Run Setup | Validate inputs; assign run_id | `run_context` |
| B | KG Context | Define causal search space from knowledge graph | `kg_context` |
| 5B | Data Fetch | Fetch live CMMS/EDMS content; build Chroma corpus | `cmms_context` + Chroma |
| B.5 | Topology Anomaly Fetch | Query historian for neighborhood sensors; build propagation chain DAG | `signal_evidence` |
| C | TSKR | Temporal pattern scoring per failure mode | `tskr_patterns` |
| D | Candidates v1 | Score and filter candidate hypotheses (pre-evidence) | `causality_candidates v1` |
| E | Evidence | Retrieve Chroma snippets for each candidate | `evidence_bundle` |
| F | Candidates v2 | Re-score and re-rank using actual evidence | `causality_candidates v2` |
| G | Ishikawa | Map candidates to 6-branch fishbone structure | `ishikawa_matrix` |
| H | Synthesis | Produce analyst-ready RCA card with recommended actions | `rca_card` |
| I | Persistence | Write all artifacts and Chroma archive to disk | run directory |
| J | Validation | Schema + cross-artifact validation; routing decision | `run_manifest` |

## 1.4 Data Source Layers

The pipeline draws from three distinct data layers with different availability characteristics:

**Layer 0 — KG Initialization inputs (static, provided once per plant / per major revision)**
Three input sources that Stage 0 transforms into the Neo4j KG: (1) an MBSE model JSON file encoding the plant's component hierarchy and equipment definitions, (2) a sensor-to-component mapping CSV that links sensor IDs to `element_usage` nodes, and (3) FMEA spreadsheet(s) providing the failure mode catalog per component type. Plant database APIs (CMMS, EDMS) are queried at Stage 0 to pre-populate document metadata references in the KG. These inputs do not change per RCA event; they are updated when the plant model is revised or new FMEA data becomes available. See Stage 0 reference section for full details.

**Layer 1 — Knowledge Graph (offline, pre-populated by Stage 0)**
Neo4j graph database populated by Stage 0 before the pipeline runs. Contains the plant's equipment topology (containment hierarchy and connectivity), the failure mode catalog with TSKR latency parameters, sensor-to-component linkage, document metadata references linked to equipment and failure modes, safety function definitions, and accepted RCA conclusions from closed CAP items. The KG is a closed world: any failure mode, component, or document not loaded into Neo4j is invisible to the pipeline at Stages B, C, and D. KG population quality is the dominant determinant of pipeline quality. Stages B, C, D consume from this layer.

**Layer 2 — Live plant data (run-time, fetched at Stage 5B)**
CMMS corrective action program records (CRs, WOs), Engineering Document Management System (EDMS) procedures and ECAs, and FMEA source documents. Queried fresh per run. Stage 5B assembles this content into the run-scoped Chroma collection and the `cmms_context` artifact. Stages 5B, C, D consume from this layer.

**Layer 3 — Run-scoped Chroma collection (assembled at Stage 5B, queried at Stage E)**
An ephemeral vector database collection keyed by `run_id`, containing embedded text from CR/WO narratives, EDMS documents, FMEA source text, and SOPs. Created at Stage 5B, queried at Stage E, archived to disk at Stage I. No content from this layer is shared across runs — each run builds its own corpus from the live plant systems.

## 1.5 Scoring Architecture

Every candidate hypothesis is scored across five independent dimensions. The dimensions and their weights in the composite score are fixed:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Structural (S) | 0.30 | Topological proximity to the primary asset + symptom match |
| Temporal (T) | 0.20 | Allen interval alignment and latency consistency (TSKR output) |
| Telemetry (Tel) | 0.20 | Number and severity of anomalies on the candidate component |
| Evidence (E) | 0.20 | Document support quality (KG proxy at Stage D; actual Chroma content at Stage F) |
| Governance (G) | 0.10 | PM compliance status and maintenance category weight |

**Composite** = 0.30·S + 0.20·T + 0.20·Tel + 0.20·E + 0.10·G

**Two-pass scoring**: The evidence dimension is scored twice. At Stage D (v1), E is computed from KG document metadata — a proxy that estimates evidence quality from document types and recency without reading content. At Stage F (v2), E is replaced by actual Chroma retrieval results. The v1→v2 rank delta is the primary diagnostic for how well the evidence corpus discriminates between hypotheses: large rank inversions indicate that the KG metadata proxy was not representative of actual document content.

**Candidate tiering**: Stage D tiers candidates into A-series and B-series rather than applying hard filters. A-series requires composite ≥ 0.45 AND evidence ≥ 0.35; B-series requires composite ≥ 0.25 (failing at least one A threshold). Candidates below composite 0.25 are dropped. Safety-significant failure modes are promoted to A-series by policy regardless of score. Both series proceed to Stages E and F; only A-series candidates are eligible for the primary hypothesis and writeback.

**Severity-adjusted floor**: For high-severity events, the run manifest requires the primary candidate's composite score to meet an elevated minimum (`_SEVERITY_SCORE_FLOORS`: severity 4 → 0.45, severity 5 → 0.55). Runs where the primary candidate falls below the floor are flagged `passed_severity_gate: False` and blocked from writeback even if all other conditions pass.

**FM-category governance**: The governance sub-score weight can be elevated for maintenance-preventable failure modes (bearing, lubrication, seal → weight 0.20) or reduced for externally-caused failure modes (environmental, design, vendor → weight 0.02), compared to the default (0.10). This is computed per candidate by `_governance_weight_for_fm` in Stage D.

## 1.6 Analyst Co-Pilot Model

The pipeline does not make final decisions. It routes every run to one of three outcomes:

| Outcome | Condition | Analyst action |
|---------|-----------|----------------|
| `writeback` | All gates pass; no review flags | System submits RCA card to CAP system |
| `analyst_review` | Near-tie, low evidence, or severity gate fail | Analyst reviews rca_card and overrides or confirms |
| `remediation` | Schema validation fails or pipeline health red | Data or pipeline issue must be resolved before analysis can proceed |

The routing decision is computed at Stage J into `run_manifest.review_hooks`. Key signals surfaced to the analyst:

- **`analyst_attention_flags[]`** in `rca_card.executive_summary`: conditions requiring analyst attention — safety function involvement, Ishikawa skipped, evidence posture warnings, recurrence anomalies, rank inversions.
- **`requires_human_review`**: set when confidence is medium or below, when review_required near-tie candidates exist, or when the severity gate is not cleared.
- **`writeback_ready`**: the single boolean a CAP integration can act on — requires all quality gates to pass simultaneously.
- **`pipeline_health`** and **`stage_health`**: per-stage green/yellow/red status with issue lists that let an analyst diagnose degraded runs.

## 1.7 Artifact Inventory

| Artifact | Produced | Schema | Consumed by |
|----------|----------|--------|-------------|
| Populated Neo4j KG | 0 | `mbseSchema.toml` + `fmeaSchema.toml` | B (live queries), Stage A governance check |
| `kg_provenance` node | 0 | (inline in Neo4j) | Stage A `_compute_kg_governance()` |
| `run_context` | A | `schemas/run_context.json` | B, 5B, B.5, C, D, E, F, G, H, I, J |
| `signal_evidence` | B.5 | `schemas/signal_evidence.json` | C, F, J |
| `kg_context` | B | `schemas/kg_context.json` | 5B, C, D, E, F, G, H, J |
| `cmms_context` | 5B | `schemas/cmms_context.json` | C, D, H, J |
| `tskr_patterns` | C | `schemas/tskr_patterns.json` | D, H, J |
| `causality_candidates v1` | D | `schemas/causality_candidates.json` | E, F |
| `evidence_bundle` | E | `schemas/evidence_bundle.json` | F, H, J |
| `causality_candidates v2` | F | `schemas/causality_candidates.json` | G, H, J |
| `ishikawa_matrix` | G (optional) | `schemas/ishikawa_matrix.json` | H, J |
| `rca_card` | H | `schemas/rca_card.json` | I, J |
| `run_manifest` | J | `schemas/run_manifest.json` | CAP integration, analyst review tool |
| `scoring_evolution` | J | (inline) | Analyst review tool |
| `barrier_analysis` | J | `schemas/barrier_analysis.json` | Analyst review tool |
| `ap913_completeness` | J | (inline in run_manifest) | Analyst review tool |

## 1.8 How to Read This Document

**Part 1 (this section)** gives you the complete picture before any stage detail. If you are evaluating whether the pipeline covers a particular RCA scenario, or deciding how to configure it for a specific plant, Part 1 has the information you need.

**Part 2 (per-stage reference)** documents every stage at the implementation level: input/output artifact schemas, core computation logic, hardcoded thresholds, and known limitations. Use this when integrating the pipeline, calibrating it for your plant's data characteristics, or tracing why a specific run produced a specific output.

Cross-references to the April 20/21 SE review use the notation `(April 20: Cn/Hn/Mn)` for findings from that review. Open issues are listed at the end of each stage section and in the consolidated cross-stage table in Part 2, §Open Issues.

---

# Part 2 — Per-Stage Reference

---

## Stage 0 — KG Initialization

**Class / entry-point**: *(not yet implemented — target: `kg/kg_initialization_workflow.py`)*
**Trigger**: manual invocation or CI hook; **not** called per RCA event

### What this stage does

Stage 0 is a prerequisite workflow — not part of the per-event RCA pipeline — that constructs and populates the Neo4j knowledge graph that Stage B depends on. It runs once at initial deployment and is re-run whenever the plant model, FMEA data, or sensor configuration changes significantly. Every per-event RCA run assumes Stage 0 has already been executed successfully and the KG is in a valid, non-empty state.

The stage has four sequential sub-steps. First, it reads the MBSE model JSON and writes the plant topology into Neo4j: `element_definition` nodes (equipment types with `domain_category`), `element_usage` nodes (tagged plant instances with `asset_id`, `tag`, `train`, `division`, `location`), and the structural edges between them (`has_part_usage` containment hierarchy and `owns_port_usage → connects_port → connector` connectivity graph). This graph is what Stage B traverses when expanding the equipment neighborhood.

Second, it reads the sensor-to-component mapping CSV and writes `monitored_variable` nodes linked to their parent `element_usage` nodes. This is what enables Stage B's five-tier seed resolution fallback and the proposed telemetry-driven neighborhood expansion — without this linkage, `neo4j.find_element_usage_by_sensor(sensor_id)` has nothing to match against.

Third, it reads the FMEA spreadsheet(s) and writes `failure_mode` nodes linked to `element_definition` nodes via `APPLIES_TO` edges. Each failure mode carries the parameters the causality engine and TSKR scorer need: `expected_latency_min/max_hours`, `expected_anomaly_pattern`, `severity`, `occurrence`, `detection`, `rpn`. This linkage resolves only if the FMEA `component_type` field matches a `domain_category` value already in the KG from sub-step 1, so the MBSE ingest must precede FMEA ingest. FMEA files arrive in many formats (AIAG, MIL-STD-1629A, IEC 60812, nuclear-utility-specific); sub-step 0.3 must pass all FMEA files through the normalization layer before KG write. Note that `expected_latency_min/max_hours` and `expected_anomaly_pattern` are non-standard enrichment fields absent from all standard FMEA formats — they are populated separately by the FMEA enrichment workflow, not by this ingest step. See `RCA_FMEA_handling_spec.md` for full details.

Fourth, it queries the plant CMMS and EDMS APIs to pre-populate document metadata references in the KG — CR and WO IDs linked to equipment, EDMS procedure doc_ids linked to components and failure modes, FMEA document sources. This is **metadata only** (doc_id, doc_type, component association, date): no document content is stored in the KG. Content retrieval remains a per-run Stage 5B responsibility. The KG document metadata index is what Stage B's document priority scoring queries at run time.

On completion, Stage 0 writes a KG provenance record capturing the MBSE model version, FMEA revision date, sensor map version, and KG population timestamp. Stage A reads this record via `_compute_kg_governance()` to verify KG readiness before allowing a run to proceed.

### Input sources

| Input | Format | Content | When updated |
|-------|--------|---------|--------------|
| `mbse_model.json` | JSON (TOML schema: `mbseSchema.toml`) | Equipment hierarchy: `element_definition` + `element_usage` nodes, ports, connectors, functions, safety functions, plant modes | On plant modification or MBSE model revision |
| `sensor_component_map.csv` | CSV | Columns: `sensor_id`, `asset_id` (`element_usage.asset_id`), `parameter`, `unit`, `normal_min`, `normal_max` | On sensor installation/removal or tag rename |
| `fmea_data.csv` / `.xlsx` | CSV or Excel | Columns: `component_type`, `failure_mode_name`, `failure_mechanism`, `local_effect`, `severity`, `occurrence`, `detection`, `expected_latency_min/max_days`, `expected_anomaly_pattern`, `corrective_actions` | On FMEA revision cycle |
| CMMS API | REST/adapter | CR and WO metadata by `asset_id`: `doc_id`, `doc_type`, `created_date`, `status` | Periodically (nightly batch or on-demand refresh) |
| EDMS API | REST/adapter | Procedure and ECA metadata by component reference: `doc_id`, `doc_type`, `revision_date` | On document revision |

### Sub-step sequence

```
mbse_model.json
      │
      ▼
[0.1] MBSE Ingest ──────────────────────────────────────────────────────────
      Writes: element_definition nodes (domain_category, spec fields)
              element_usage nodes (asset_id, tag, train, division, location)
              has_part_usage edges (containment hierarchy)
              owns_port_usage → connects_port → connector edges (connectivity)
              safety_function nodes + PERFORMS / SUPPORTED_BY / PROVIDES edges
      │
      ▼
sensor_component_map.csv
      │
      ▼
[0.2] Sensor Link Ingest ────────────────────────────────────────────────────
      Writes: monitored_variable nodes (sensor_id, parameter, unit,
                                        normal_min, normal_max)
              HAS_SENSOR edges: element_usage → monitored_variable
      Requires: element_usage nodes already in KG from sub-step 0.1
      │
      ▼
fmea_data.csv
      │
      ▼
[0.3] FMEA Ingest ───────────────────────────────────────────────────────────
      Writes: failure_mode nodes (fm_id, failure_mechanism, expected_latency,
                                  anomaly_pattern, severity, occurrence,
                                  detection, rpn, expected_symptoms)
              APPLIES_TO edges: failure_mode → element_definition
                                (matched by component_type == domain_category)
      Requires: element_definition nodes already in KG from sub-step 0.1
      Warning: component_type values in FMEA that do not match any
               domain_category in KG produce orphaned failure_mode nodes
               (no APPLIES_TO edge) — logged as coverage gaps
      │
      ▼
CMMS API + EDMS API
      │
      ▼
[0.4] Document Metadata Ingest ──────────────────────────────────────────────
      Writes: document reference nodes (doc_id, doc_type, created_date,
                                        revision_date, status)
              REFERENCES edges: element_usage → document (by asset_id)
              COVERS_FM edges: document → failure_mode (where NER linkage
                               available — optional, best-effort)
      Note: metadata only — no document content stored in KG
      │
      ▼
[0.5] KG Provenance Record ──────────────────────────────────────────────────
      Writes: kg_provenance node {
          mbse_model_version, fmea_revision_date, sensor_map_version,
          doc_metadata_refresh_timestamp, kg_population_timestamp,
          component_count, failure_mode_count, sensor_count,
          orphaned_fm_count, coverage_gap_count
      }
      This record is read by Stage A's _compute_kg_governance() to
      determine KG readiness state (green / yellow / red).
```

### KG governance states consumed by Stage A

| State | Condition | Pipeline behavior |
|-------|-----------|-------------------|
| `green` | KG populated; `failure_mode_count ≥ minimum_fm_floor`; provenance record age ≤ `kg_staleness_threshold_days` | Run proceeds normally |
| `yellow` | KG populated but provenance record older than threshold, or `orphaned_fm_count / failure_mode_count > 0.10` | Run proceeds with `kg_governance_warning` flag in `run_context`; analyst is notified |
| `red` | KG empty, provenance record absent, or `failure_mode_count = 0` | Run hard-aborts; `run_manifest.next_step = "remediation"` — re-run Stage 0 |

### Critical path pseudo-code

```python
def run_kg_initialization(mbse_model_path, sensor_map_path, fmea_paths,
                           cmms_adapter, edms_adapter, neo4j_conn):

    kg = Py2NeoWorkflow(neo4j_conn)
    schemas = load_and_merge_schemas([
        "mbseSchema.toml", "documentSchema.toml", "fmeaSchema.toml",
        "conditionReportSchema.toml", "workOrderSchema.toml"
    ])
    kg.ensure_constraints(schemas)

    # 0.1 — MBSE topology
    mbse = load_json(mbse_model_path)
    element_defs   = [build_element_def_node(e)   for e in mbse["element_definitions"]]
    element_usages = [build_element_usage_node(u)  for u in mbse["element_usages"]]
    part_edges     = [build_has_part_edge(e)        for e in mbse["containment_edges"]]
    conn_edges     = [build_connectivity_edges(e)   for e in mbse["connectivity_edges"]]
    sf_nodes       = [build_safety_function_node(s) for s in mbse.get("safety_functions", [])]
    kg.upsert_nodes_batch(element_defs + element_usages + sf_nodes)
    kg.upsert_edges_batch(part_edges + conn_edges)

    # 0.2 — Sensor linkage
    sensor_map = load_csv(sensor_map_path)
    mv_nodes, sensor_edges = [], []
    for row in sensor_map:
        mv_nodes.append(build_monitored_variable_node(row))
        sensor_edges.append(("element_usage", row["asset_id"],
                             "HAS_SENSOR", "monitored_variable", row["sensor_id"]))
    kg.upsert_nodes_batch(mv_nodes)
    kg.upsert_edges_batch(sensor_edges)

    # 0.3 — FMEA failure modes
    orphaned = []
    for fmea_path in fmea_paths:
        records = FMEAParser().parse(fmea_path)
        for fm in records:
            fm_node = build_failure_mode_node(fm)
            kg.upsert_nodes_batch([fm_node])
            matched = kg.find_element_definitions(domain_category=fm["component_type"])
            if matched:
                kg.upsert_edges_batch([("failure_mode", fm["fm_id"],
                                        "APPLIES_TO", "element_definition", d.id)
                                       for d in matched])
            else:
                orphaned.append(fm["fm_id"])
                logger.warning("No element_definition match for component_type=%s",
                               fm["component_type"])

    # 0.4 — Document metadata from plant APIs
    doc_nodes, ref_edges = [], []
    for usage in kg.get_all_element_usages():
        for doc in cmms_adapter.get_document_metadata(asset_id=usage["asset_id"]):
            doc_nodes.append(build_document_node(doc))
            ref_edges.append(("element_usage", usage["asset_id"],
                              "REFERENCES", "document", doc["doc_id"]))
        for doc in edms_adapter.get_document_metadata(component_id=usage["asset_id"]):
            doc_nodes.append(build_document_node(doc))
            ref_edges.append(("element_usage", usage["asset_id"],
                              "REFERENCES", "document", doc["doc_id"]))
    kg.upsert_nodes_batch(doc_nodes)
    kg.upsert_edges_batch(ref_edges)

    # 0.5 — KG provenance record
    stats = kg.compute_stats()
    kg.upsert_nodes_batch([{
        "label": "kg_provenance",
        "id": "kg_provenance_latest",
        "mbse_model_version":           mbse.get("model_version", "unknown"),
        "fmea_revision_date":           max(p.stat().st_mtime for p in fmea_paths),
        "sensor_map_version":           sensor_map_path.stat().st_mtime,
        "doc_metadata_refresh_ts":      utcnow_iso(),
        "kg_population_timestamp":      utcnow_iso(),
        "component_count":              stats["element_usage_count"],
        "failure_mode_count":           stats["failure_mode_count"],
        "sensor_count":                 stats["monitored_variable_count"],
        "orphaned_fm_count":            len(orphaned),
        "coverage_gap_count":           stats["element_usages_with_no_fms"],
    }])

    return KGInitializationResult(stats=stats, orphaned_fms=orphaned)
```

### Key thresholds and configuration

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `minimum_fm_floor` | 5 | KG turns red if fewer than this many failure modes are loaded |
| `kg_staleness_threshold_days` | 180 | KG turns yellow if provenance record is older than this |
| `orphaned_fm_warning_ratio` | 0.10 | KG turns yellow if > 10% of failure modes have no `APPLIES_TO` edge |
| `doc_metadata_lookback_days` | 1825 | How far back to query CMMS for historical document references (default 5 years) |
| `fmea_revision_staleness_days` | 730 | Threshold for `fmea_staleness_warning` in `kg_context` (consumed by Stage B) |

### Known gaps

- **No MBSE ingestor exists**: `kg_ingest_neo4j_workflow.py` ingests NLP-pipeline artifacts (processed JSONL records) but does not read an MBSE model JSON file and produce `element_definition` / `element_usage` nodes. A dedicated ingestor following `mbseSchema.toml` must be written. This is the single largest missing implementation piece in the entire system.
- **No sensor-to-component ingestion**: the five-tier alias resolution in Stage B (`_resolve_via_sensor_aliases`) and the proposed telemetry-driven neighborhood expansion both depend on `element_usage → monitored_variable` edges. No ingestion path for a sensor-to-component CSV currently exists in `kg/`.
- **CMMS/EDMS metadata fetch is a Stage 5B responsibility today**: Stage 5B fetches document content (and implicitly metadata) at run time. Stage 0 should pre-index stable document metadata references into the KG so Stage B's document priority scoring has something to query — currently the KG document references are populated only by the NLP Stage 1–6 pipeline ingest, which requires pre-processed JSONL files.
- **FMEA `component_type` ↔ `domain_category` alignment is manual**: the FMEA parser extracts `component_type` as a free-text field; `domain_category` in the MBSE model is also free text. Mismatches silently produce orphaned failure modes (no `APPLIES_TO` edge). A controlled vocabulary or normalization step is needed. See `RCA_FMEA_handling_spec.md` §3 for the full FMEA ingestion normalization layer design.
- **FMEA format variability not handled**: the current `fmeaParser.py` handles column naming variations but not structural differences between FMEA formats (AIAG, MIL-STD-1629A, IEC 60812, nuclear-utility-specific). Fields critical to the pipeline — `expected_latency_min/max_hours`, `expected_anomaly_pattern` — are absent from all standard FMEA formats and require a dedicated enrichment workflow before they can be used. See `RCA_FMEA_handling_spec.md` for the full analysis, the normalization layer spec (§3), and the human-in-loop enrichment workflow spec (§4).
- **No incremental update support**: the current design is full-rebuild. A partial update workflow (e.g., add one new component, refresh doc metadata for one asset) is not yet specified. For large plants this matters because a full KG rebuild may take tens of minutes.
- **MBSE model format not finalized**: `mbseSchema.toml` defines the KG shape but does not specify an intermediate JSON exchange format. The `mbse_model.json` input format for Stage 0 needs a companion schema definition.
- **Safety function nodes**: the MBSE schema supports safety function definitions, but the Stage 0 ingestor must explicitly map MBSE safety function associations to the `PERFORMS` / `SUPPORTED_BY` / `PROVIDES` edge types that Stage B queries. This mapping logic is not yet written.
- **No CAP write-back integration at Stage 0**: accepted RCA conclusions from closed CAP items are intended to populate `kg_context.past_events[]` via KG write-back. This is architecturally a post-Stage-J workflow (see `RCA_Data_Management_Strategy.md` §10), not a Stage 0 responsibility, but it shares the same KG write infrastructure and should be designed together with Stage 0.

---

## Stage A — Run Setup & Input Validation

**Class / entry-point**: `_stage_a_build_run_context()` (Orchestrator)

### What this stage does

Stage A is the pipeline's entry gate. Before any analysis begins, the orchestrator needs a stable, traceable identity for the run and confidence that the four input artifacts are self-consistent. This stage provides both.

It assigns a unique `run_id` to the invocation so every downstream artifact, every Chroma document, and every log entry can be traced back to a single event-analysis pairing. It then checks each input artifact for required fields and flags cross-artifact inconsistencies — the most important being an `asset_id` mismatch between `event.json` and `telemetry_summary.json`, which would mean the telemetry is from a different piece of equipment than the event being investigated.

If the event artifact is missing its identity keys (`event_id` or `asset_id`), the run aborts immediately. All other validation failures are non-fatal: they are recorded in `input_validation_result` and carried forward so analysts can see exactly what was incomplete when the RCA was produced.

Stage A does not query any external system and does not score anything. Its sole output — `run_context` — is a small bookkeeping artifact consumed only by Stage B.

### Input artifacts

| Artifact | Key fields consumed |
|----------|---------------------|
| `event.json` | `event_id`, `asset_id`, `severity`, `timestamp_start`, `symptom_signature` |
| `telemetry_summary.json` | `event_id`, `asset_id`, `window`, `signals[].sensor_id` |
| `operational_context.json` | `event_id`, `asset_id` |
| `pm_compliance.json` | `asset_id`, `pm_checks[]` |

### Key logic

- Assigns `run_id` (UUID) and stamps `pipeline_version`.
- Validates that `event.asset_id` matches `telemetry_summary.asset_id` (cross-artifact consistency check).
- Checks for required top-level keys in all four input artifacts; records per-artifact pass/fail into `input_validation_result`.
- Aborts the run with `status: "aborted"` if `event_id` or `asset_id` is missing.
- Logs schema version mismatches as warnings (non-fatal) so analysts can detect stale fixtures.

### Output artifact — `run_context`

| Field | Value |
|-------|-------|
| `run_id` | UUID, unique per invocation |
| `pipeline_version` | Hardcoded string (e.g., `"v3.2"`) |
| `event_id` | Echoed from `event.json` |
| `asset_id` | Echoed from `event.json` |
| `input_validation_result` | Per-artifact: `{artifact_name: {valid: bool, missing_keys: [], warnings: []}}` |
| `run_timestamp` | ISO UTC |

### Critical path pseudo-code

```python
def _stage_a_build_run_context(event, telemetry, oc, pm):
    run_id = str(uuid4())
    validation = {}

    for name, artifact, required_keys in [
        ("event",     event,     ["event_id", "asset_id"]),
        ("telemetry", telemetry, ["event_id", "asset_id", "signals"]),
        ("oc",        oc,        ["event_id"]),
        ("pm",        pm,        ["asset_id", "pm_checks"]),
    ]:
        missing = [k for k in required_keys if k not in artifact]
        validation[name] = {"valid": len(missing) == 0, "missing_keys": missing}

    if not validation["event"]["valid"]:
        return {"status": "aborted", "reason": "event artifact missing required keys"}

    if event["asset_id"] != telemetry.get("asset_id"):
        validation["telemetry"]["warnings"] = ["asset_id mismatch with event"]

    return {
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "event_id": event["event_id"],
        "asset_id": event["asset_id"],
        "input_validation_result": validation,
        "run_timestamp": utcnow_iso(),
    }
```

### Key thresholds

_None — Stage A is purely structural validation; no numeric thresholds._

### Known gaps

- **A2 ✅ FIXED (Sprint 6)**: `severity` field is now stored in `run_context.input_refs.event_severity` and consumed at Stage J to enforce `_SEVERITY_SCORE_FLOORS` (severity 4 → composite ≥ 0.45, severity 5 → composite ≥ 0.55). The severity gate blocks `writeback_ready` when the primary candidate's composite score falls below the floor for the event's severity.
- No cross-artifact timestamp consistency check (e.g., `telemetry.window.end` vs `event.timestamp_start`).
- Stage A performs only a key-presence check on the four input artifacts; full JSON schema validation of input artifacts belongs here rather than at Stage J, to catch malformed inputs at the earliest possible detection point.

---

## Stage B — KG Context Construction

**Class / entry-point**: `Neo4jKGContextBuilder.build()`
**File**: `orchestrators/kg_context_builder.py`

### What this stage does

Stage B defines the causal search space for the entire pipeline. Every failure mode the system will ever consider, every document it will rank, and every past event it will reference is determined here. Nothing outside the subgraph built by Stage B can influence the RCA outcome.

The stage starts by locating the equipment under investigation in the Neo4j knowledge graph — a node called `element_usage` that represents a specific installed instance of a component. From that seed node it expands the neighborhood in two directions: upward and downward through the physical containment hierarchy (what contains this component, what does this component contain), and laterally through the connectivity model (what is this component port-connected to). The expansion is bounded at two hops, which in practice covers the immediately relevant system boundary without pulling in the entire plant model.

A third expansion path — not yet implemented — would widen the search space using telemetry evidence. For each sensor in `telemetry_summary` that carries anomalies, Stage B computes a simple Allen interval relation between the anomaly window and the event interval using only timestamps (no scoring needed). Anomalies that **precede or overlap** the event are investigative leads — the corresponding component enters the neighborhood as a potential upstream cause. Anomalies that **follow** the event are likely downstream consequences and are excluded from the causal search space but recorded in `out_of_boundary_anomalies` for analyst review. This mirrors what an RCA engineer does in the first minutes of an investigation: scan all plant indications in the time window and use timing to separate causes from effects before any causal analysis begins. Critically, this temporal pre-filter does not require TSKR — it is a direct timestamp comparison on data already available at Stage B.

Once the neighborhood is established, Stage B retrieves everything the KG knows about those components: the failure modes that apply to them, document references (CRs, WOs, ECAs, SOPs, FMEAs) ranked by type and recency, accepted RCA conclusions written back from closed CAP items, and safety functions that any component in the neighborhood supports or provides.

The result — `kg_context` — is the single most consequential artifact in the pipeline. Its completeness depends entirely on the quality and coverage of the KG population process, which runs offline before the pipeline is invoked. If a failure mode was never loaded into Neo4j, it will not appear as a candidate regardless of how strong the telemetry or evidence signal is. The `out_of_boundary_anomalies` field partially mitigates this by flagging anomalous components that are outside the KG neighborhood — including components not in the KG at all.

### Input artifacts

| Input | Key fields consumed |
|-------|---------------------|
| `run_context` | `asset_id`, `event_id` |
| `event.json` | `timestamp_start`, `timestamp_end` — used for anomaly temporal pre-filter |
| `telemetry_summary` | `signals[].sensor_id` (seed resolution) · `signals[].anomalies[].timestamp_start/end` (temporal pre-filter) |
| Neo4j KG (live) | `element_usage` nodes, edges: `has_part_usage`, `owns_port_usage`, `connects_port`, `APPLIES_TO`, `PERFORMS`, `SUPPORTED_BY` |

> **Note — telemetry role in Stage B**: telemetry serves two purposes here, both lightweight. (1) **Seed node resolution**: if `asset_id` from `event.json` does not match any `element_usage` node directly, Stage B falls back through five sensor-alias fields (`monitored_variable_id → sensor_id → tag_id → mv.ID → aliases`). (2) **Anomaly temporal pre-filter** *(proposed, not yet implemented)*: for each sensor with anomalies, Stage B computes the Allen interval relation between the anomaly window and the event interval using only timestamps — no scoring. PRECEDES/OVERLAPS anomalies drive a third neighborhood expansion; FOLLOWS anomalies are recorded as potential consequences. Neither use involves causal scoring — that is Stage C.

### Key logic

- Resolves the primary seed node from Neo4j: finds the `element_usage` node whose `asset_id` matches the event. Falls back through five sensor-alias fields if direct match fails.
- Expands **two structural** neighborhood patterns up to `max_hops=2`: containment hierarchy (`has_part_usage`) and connectivity (`owns_port_usage → connects_port → connector`).
- *(Proposed)* Expands a **third telemetry-driven** neighborhood path: for each sensor in `telemetry_summary` with anomalies, computes the Allen interval relation between the anomaly window and the event interval using timestamps only. PRECEDES/OVERLAPS → resolve component in KG → add to neighborhood with `seed_match_type = "telemetry_anomaly_precedes"`. FOLLOWS → record in `out_of_boundary_anomalies` as potential consequence. Components with preceding anomalies that are **not found in the KG** are also recorded in `out_of_boundary_anomalies` with `not_in_kg: true` — a direct signal of KG coverage gaps.
- Retrieves all `failure_mode` nodes linked to any component in the expanded neighborhood.
- Retrieves and priority-scores **document references** (CRs, WOs, ECAs, SOPs, FMEAs) from the KG within a ±90/+7-day window using a combined type + recency + asset-match score. **This is metadata only** — doc_id, doc_type, priority_score. Document content is not fetched at Stage B; that happens at Stage 5B (see `RCA_Data_Management_Strategy.md` §5).
- Retrieves **accepted RCA conclusions** stored as event nodes in the KG — these are write-back records from closed CAP items (see `RCA_Data_Management_Strategy.md` §10, KG population). This is currently an empty set (CAP write-back is future). It is **not** a query of historical CR/WO records: CMMS is the system of record for past event history, and that data is fetched at Stage 5B. `kg_context.past_events[]` will remain sparse until CAP write-back is implemented.
- Retrieves safety functions bidirectionally (PERFORMS, SUPPORTED_BY, PROVIDES relations).
- Captures KG snapshot version (`neo4j_version|max_last_modified`) for reproducibility.

### Output artifact — `kg_context`

| Field | Content |
|-------|---------|
| `components[]` | Seed + neighbor components; each has `component_id`, `component_type`, `relation_to_asset`, `seed_match_type` |
| `upstream_paths[]` | Neo4j path objects with edge types and `path_strength` |
| `failure_modes[]` | All FMs applicable to neighborhood components; each has `fm_id`, `expected_latency_min/max_hours`, `pattern` |
| `past_events[]` | Accepted RCA conclusions written back to KG from closed CAP items. Currently empty — CAP write-back is future. Historical CR/WO event data comes from CMMS at Stage 5B, not from this field. |
| `safety_functions[]` | Safety function nodes linked bidirectionally |
| `documents[]` | KG-ranked document references (metadata only): `doc_id`, `doc_type`, `priority_score`, `recency_bonus`. No content — content is fetched at Stage 5B. |
| `out_of_boundary_anomalies[]` | *(proposed)* Sensors with anomalies excluded from causal search space. Each entry: `sensor_id`, `component_id` (if resolved in KG), `anomaly_pattern`, `severity`, `allen_relation` (FOLLOWS = potential consequence), `not_in_kg` (bool — KG coverage gap flag). Surfaced for analyst review in viz app. |
| `seed_context` | `asset_ids`, `monitored_variable_ids`, `component_ids` used in queries |
| `kg_snapshot_version` | Reproducible version marker |

### Critical path pseudo-code

```python
def build(event, telemetry, oc, pm, run_context):
    asset_id   = event["asset_id"]
    event_time = parse(event["timestamp_start"])
    event_end  = parse(event.get("timestamp_end", event["timestamp_start"]))

    # 1. Seed resolution
    seed_node = neo4j.find_element_usage(asset_id)
    if not seed_node:
        seed_node = _resolve_via_sensor_aliases(telemetry)

    # 2. Structural neighborhood expansion (two patterns — existing)
    neighbors_hier = neo4j.expand_has_part_usage(seed_node, hops=2)
    neighbors_conn = neo4j.expand_connectivity(seed_node, hops=2)
    all_components = deduplicate(neighbors_hier + neighbors_conn + [seed_node])

    # 3. Telemetry-driven neighborhood expansion (proposed — not yet implemented)
    out_of_boundary = []
    for signal in telemetry["signals"]:
        for anomaly in signal.get("anomalies", []):
            relation = compute_allen_relation(
                interval_a=(parse(anomaly["timestamp_start"]), parse(anomaly["timestamp_end"])),
                interval_b=(event_time, event_end)
            )
            if relation in ("PRECEDES", "OVERLAPS", "SIMULTANEOUS"):
                node = neo4j.find_element_usage_by_sensor(signal["sensor_id"])
                if node and node not in all_components:
                    node["seed_match_type"] = "telemetry_anomaly_precedes"
                    all_components.append(node)
                elif not node:
                    out_of_boundary.append({
                        "sensor_id": signal["sensor_id"], "component_id": None,
                        "allen_relation": relation, "not_in_kg": True,
                        "severity": anomaly.get("severity"), "anomaly_pattern": anomaly.get("pattern")
                    })
            elif relation == "FOLLOWS":
                node = neo4j.find_element_usage_by_sensor(signal["sensor_id"])
                out_of_boundary.append({
                    "sensor_id": signal["sensor_id"],
                    "component_id": node.id if node else None,
                    "allen_relation": "FOLLOWS", "not_in_kg": node is None,
                    "severity": anomaly.get("severity"), "anomaly_pattern": anomaly.get("pattern")
                })

    # 4. Failure modes
    failure_modes = neo4j.get_failure_modes(component_ids=[c.id for c in all_components])

    # 5. Documents (metadata only — content fetched at Stage 5B)
    #    Window split by doc type (proposed fix — currently uniform 90-day window)
    docs = neo4j.get_documents(
        component_ids=[c.id for c in all_components],
        window_by_type={
            "CR": (event_time - 90d,  event_time + 7d),
            "WO": (event_time - 90d,  event_time + 7d),
            "FMEA": None,   # no time window — analysis docs don't decay
            "RCA":  None,
            "ECA":  None,
            "SOP":  None,
        }
    )
    for doc in docs:
        doc["priority_score"] = (
            DOC_TYPE_PRIORITY[doc["doc_type"]]        # 45–100
            + (10 if doc["asset_id"] == asset_id else 0)
            + (8  if doc["component_id"] in neighbor_ids else 0)
            + recency_bonus(doc, event_time)           # up to +10
        )
    docs.sort(key=lambda d: d["priority_score"], reverse=True)

    # 6. Accepted RCA conclusions from KG (write-back, currently empty)
    past_events = neo4j.get_past_events(asset_id, component_ids, fm_ids)

    # 7. Safety functions
    safety_fns = neo4j.get_safety_functions(component_ids)

    # 8. Path strength
    for path in upstream_paths:
        strength = 1.0
        for edge in path.edges:
            strength *= PATH_MULTIPLIER[edge.type]  # 0.95 / 0.85 / 0.80
        path["path_strength"] = round(strength, 4)

    return build_kg_context_artifact(..., out_of_boundary_anomalies=out_of_boundary)
```

### Key thresholds

| Parameter | Value | Effect |
|-----------|-------|--------|
| `max_hops` | 2 | Bounds neighborhood size |
| `doc_window_days_before` | 90 | Date filter applied to operational records (CR, WO, ECR) only; does not apply to timeless document types (see B2 note below) |
| `doc_window_days_after` | 7 | Small forward window for post-event documentation; applies to operational records only |
| `past_event_window_days` | 3650 | Parameter exists in code but applies to a KG query that is currently empty. Historical event retrieval window is governed by CMMS query at Stage 5B instead. |
| Doc type priorities | CR=100, WO=95, ECA=90, RCA=85, FMEA=70, SOP=60 | Ordering within type; ECA and RCA rank highly because they are authoritative analysis documents |
| Recency proximity bonus | CR, WO, ECR only — up to +10 pts | Timeless doc types (ECA, RCA, FMEA, SOP, MANUAL, BULLETIN) receive no recency bonus; their authority is independent of temporal proximity to the event |
| Path strength multipliers | 0.95 (containment), 0.85 (connectivity), 0.80 (other) | Weaken indirect causal paths |

### Known gaps

- **C1 (April 20 review)**: KG is a closed world — any failure mode not in Neo4j is invisible to the pipeline regardless of evidence. The `out_of_boundary_anomalies` field with `not_in_kg: true` partially surfaces this gap for analysts.
- **H1 (April 20 review)**: Sensor alias resolution has five fallback fields but no logging when fallback is used; analysts cannot tell whether the seed was resolved cleanly or via alias.
- **B2 ✅ FIXED (Sprint 7)**: ECA and RCA document types are now exempt from the ±90-day date window filter. They are retrieved regardless of when they were created, because their relevance to a failure mode does not decay with time. CR, WO, and ECR documents continue to use the 90-day operational window. Additionally, the recency-proximity bonus (+10 pts) is no longer applied to ECA and RCA docs — their ranking is based on type priority and asset/component match, not recency.
- Sensor alias resolution logs are not emitted when fallback aliases are used; production diagnostics should include a `seed_resolution_method` field in `kg_context.seed_context`.

> **Code changes required — Stage B** (`orchestrators/kg_context_builder.py`):
>
> 1. **Telemetry-driven neighborhood expansion** *(new)*: add a third expansion loop over `telemetry["signals"][].anomalies[]`. For each anomaly, compute Allen interval relation against event interval (timestamp comparison only — no scoring). PRECEDES/OVERLAPS: resolve sensor → KG node via `neo4j.find_element_usage_by_sensor(sensor_id)`; if found add to `all_components` with `seed_match_type = "telemetry_anomaly_precedes"`; if not found add to `out_of_boundary_anomalies` with `not_in_kg: true`. FOLLOWS: add to `out_of_boundary_anomalies` regardless.
> 2. **`out_of_boundary_anomalies` field** *(new)*: add to `kg_context` output schema (`schemas/kg_context.json`) and populate in `build_kg_context_artifact()`. Fields: `sensor_id`, `component_id` (nullable), `allen_relation`, `not_in_kg` (bool), `severity`, `anomaly_pattern`.
> 3. **`neo4j.find_element_usage_by_sensor(sensor_id)`** *(new method)*: KG lookup that matches sensor_id against monitored-variable aliases — reuses the same five-tier alias resolution already in `_resolve_via_sensor_aliases()`.
> 4. **Document time window by type** *(fix)*: replace uniform `doc_window_days_before=90` with a per-type window map: 90 days for CR/WO; no window for FMEA, RCA, ECA, SOP. Requires updating `neo4j.get_documents()` signature to accept `window_by_type: dict` instead of a single date range.
> 5. **`past_event_window_days` parameter**: remove or deprecate — past event history now comes from CMMS at Stage 5B, not from KG. The KG query for past events should return only accepted RCA conclusions (no date window needed).

> **Note — document retrieval time window strategy**: the 90-day window applies uniformly to all document types when building the ranked reference list. This is appropriate for operational records (CRs, WOs) where recency reflects current equipment state, but it is too restrictive for analysis documents (FMEAs, RCAs, ECAs) whose relevance does not decay with time — an RCA report from three years ago documenting the same failure mode is equally valid evidence. The time window strategy should be revised to apply a short window (e.g., 90 days) only to CR and WO, and a much longer window — or no window — to FMEA, RCA, and ECA document references. This also affects which doc_ids are passed to Stage 5B for content retrieval, so the fix must be applied here at the metadata selection stage.

---

## Stage 5B — Run-Scoped Data Fetch

**Class / entry-point**: `CMMSContextBuilder.build()` + `ChromaIngestionPipeline.ingest()`
**File**: `cmms_integration/cmms_context_builder.py`

### What this stage does

Stage 5B is where the pipeline makes contact with live plant data systems. Everything before it — Stage A validation, Stage B KG context — works entirely from pre-loaded knowledge and invocation-time input artifacts. Stage 5B is the only stage that reaches out to CMMS, EDMS, and FMEA systems at run time and assembles the evidence corpus that the rest of the pipeline will reason over.

It does two things that must be kept conceptually separate. First, it fetches document content — CR and WO narratives, EDMS procedures, FMEA source documents — and embeds them into a run-scoped Chroma collection keyed by `run_id`. This collection is what Stage E queries for evidence. Second, it fetches the full CR history for the primary equipment and similar equipment, going back as far as CMMS retention allows, and extracts lightweight metadata from it — dates, failure mode keywords, resolution status. This metadata is what Stage C uses to build the recurrence profile and what Stage D uses to generate historical-event candidates. These two purposes require different time windows and different treatment of records: recent records get full-text embedding; older records contribute metadata only.

Stage 5B also resolves the sister equipment scope — the set of components that are operationally similar to the primary asset and whose maintenance history is relevant to the investigation. In the current implementation this is limited to topologically-adjacent equipment (same train, adjacent). The full design extends it to failure-mode overlap and specification similarity via the `EquipmentSimilarityResolver`, broadening the recurrence signal to cover similar equipment across different systems at the plant.

The run-scoped Chroma collection assembled here is ephemeral by design: it exists for the duration of the run, is archived at Stage I as part of the permanent audit record, and is never shared across runs. This keeps the evidence corpus fresh, self-contained, and fully traceable to a single RCA event.

### Input artifacts

| Input | Key fields consumed |
|-------|---------------------|
| `kg_context` | `components[]`, `failure_modes[]`, `past_events[]`, `documents[]` |
| `event.json` | `asset_id`, `timestamp_start` |
| CMMS (live) | CR records, WO records, PM records — by asset_id and sister_ids |
| EDMS (live) | PDF documents by `doc_id` references in `kg_context.documents[]` |
| FMEA source (live) | FMEA documents by `fm_id` references in `kg_context.failure_modes[]` |

### Key logic

Stage 5B is the only point in the pipeline where live plant systems are queried for document content. Everything it fetches is embedded into a run-scoped Chroma collection keyed by `run_id` — assembled fresh per invocation, archived after Stage I, never shared across runs (see `RCA_Data_Management_Strategy.md` §4–5).

The stage performs five fetch types in sequence:

1. **Instance-level CMMS fetch** *(implemented)*: queries CR and WO records by primary `asset_id` plus KG-resolved sister equipment IDs (same_train, adjacent topology). Lookback window anchored to last PM date from `kg_context.past_events`; falls back to `event_time − 90 days`. Records enriched with `days_before_event`, normalized status, `is_sister_equipment` flag. Narrative fields (long_text, short_description) extracted for Chroma embedding.

2. **Class-level CMMS fetch** *(not yet implemented)*: KG resolves `component_type → list of all equipment_ids of that type at this plant`. Queries CR and WO for each ID in that list. Pre-filtered by `failure_mode_class` keywords and date window to limit volume. Retrieves similar failure events across similar equipment — the primary recurrence signal beyond topological neighbors.

3. **EDMS fetch** *(not yet implemented)*: for each `doc_id` in `kg_context.documents[]`, fetches document content from the EDMS. Document types: SOP, ECA, RCA, operating procedures. The doc_id list from Stage B already ranks these by relevance; Stage 5B only fetches content for the IDs it receives.

4. **FMEA document fetch** *(not yet implemented)*: for each `component_type` in `kg_context.components[]`, fetches the FMEA source document from the FMEA system. Distinct from FMEA metadata already in `kg_context.failure_modes[]` — this is the full document text for embedding and evidence retrieval.

5. **OE LLM query** *(future)*: for each `(component_type, failure_mode_class)` pair, calls IRIS LLM API (INPO IRIS corpus) and ADAMS LLM API (NRC ADAMS corpus) via internet API. Both must return structured responses with source citations (doc_id, title, section, year). No local index required.

All fetched content passes through one of two ingestion paths before embedding:
- **Path A (structured)**: CMMS records returned as structured objects → direct field mapping → NER on narrative fields → embed.
- **Path B (document/PDF)**: CMMS records in PDF format, EDMS documents, FMEA documents → parser selection → section-aware chunking → NER → embed.

Each Chroma document carries a `source_tier` metadata tag (plant_instance → plant_procedure → plant_fmea → plant_family → oe_iris → oe_adams) used by Stage E for authority weighting during evidence scoring.

### Output

- **`cmms_context`** artifact (JSON): `lookback_anchor`, `lookback_from/to`, `sister_component_ids`, `cr_records[]`, `wo_records[]`, `recurrence_summary`. This is the authoritative source of historical event data for the pipeline — `cr_records[]` feeds Stage C (TSKR recurrence profile) and Stage D (historical-event candidate pool), replacing the empty `kg_context.past_events[]`.
- **Run-scoped Chroma collection**: documents from CR/WO narratives, EDMS files, FMEA docs, SOPs/ECAs — all keyed by `run_id`

### Critical path pseudo-code

```python
def build(event, kg_context, run_id):
    # 1. Lookback window
    last_pm = find_last_pm_event(kg_context["past_events"])
    lookback_from = last_pm["date"] if last_pm else event_time - timedelta(days=90)

    # 2. Sister components (3-tier)
    sisters_t1 = [c for c in kg_context["components"]
                  if c["relation_to_asset"] in ("same_train", "adjacent")]
    sisters_t2 = similarity_resolver.by_failure_mode(asset_id, failure_modes)  # optional
    sisters_t3 = similarity_resolver.by_spec_embedding(asset_id)                # optional
    sister_ids = deduplicate([s["component_id"] for s in sisters_t1+sisters_t2+sisters_t3])

    # 3. CMMS fetch
    raw_crs, raw_wos = cmms_adapter.fetch(
        primary_id=asset_id,
        sister_ids=sister_ids,
        from_date=lookback_from,
        to_date=event_time
    )

    # 4. Enrich records
    for rec in raw_crs + raw_wos:
        rec["days_before_event"] = (event_time - parse(rec["created_date"])).days
        rec["status"]            = normalize_status(rec.get("status", ""))
        rec["is_sister_equipment"] = rec["asset_id"] != asset_id

    # 5. Extract Chroma documents
    chroma_docs = []
    for rec in raw_crs + raw_wos:
        if text := rec.get("long_text") or rec.get("short_description"):
            chroma_docs.append(ChromaDoc(text, metadata={
                "source": "cmms_live", "record_type": rec["type"],
                "run_id": run_id, "is_sister": rec["is_sister_equipment"]
            }))

    # 6. EDMS + FMEA fetch and embed
    for doc_ref in kg_context["documents"]:
        raw_doc = edms_adapter.fetch(doc_ref["doc_id"])
        chroma_docs += parse_and_chunk(raw_doc)

    chroma_store.add(chroma_docs, collection=f"run_{run_id}")

    return build_cmms_context_artifact(raw_crs, raw_wos, sister_ids, lookback_from)
```

### Key thresholds

Stage 5B serves two distinct retrieval purposes. The two governing parameters — time window and record cap — apply independently to each purpose:

**Time window** controls how far back the CMMS query reaches.
**Record cap** controls how many records are loaded into `cmms_context` and embedded into Chroma.

These are orthogonal concerns: a long time window can still produce a small number of records (infrequent failures), and a short window can produce thousands (high-activity equipment). They must be configured separately per purpose.

| Parameter | Proposed value | Purpose | What it controls |
|-----------|---------------|---------|-----------------|
| `operational_lookback_days` | 90 days (or last PM date, whichever is earlier) | Fetch recent CR/WO records whose **full text** will be embedded into Chroma for Stage E evidence retrieval | Time window for Chroma-bound records |
| `max_cr_records_chroma` | 100 | Cap on how many CR records are embedded into Chroma — limits embedding cost and Chroma collection size | Record cap for Chroma embedding |
| `max_wo_records_chroma` | 100 | Same cap for WO records | Record cap for Chroma embedding |
| `recurrence_history_lookback_days` | Full CMMS retention (plant-specific; typically 5–10 years) | Fetch CR history whose **metadata only** (date, asset_id, failure_mode_keywords, status) is used by Stage C (TSKR recurrence profile) and Stage D (historical-event candidates) | Time window for recurrence metadata |
| `sister_relation_types` | `["same_train", "adjacent"]` | Tier-1 sister scope — topology-based | Scope of sister equipment |

> **Note — two fetch paths, one CMMS query**: in practice both purposes can be served by a single CMMS query with the longer `recurrence_history_lookback_days` window. Records within the operational window get full-text Chroma embedding (capped at `max_cr_records_chroma`); records outside the operational window contribute only lightweight metadata to the recurrence profile (no cap, no embedding). The current code uses a single `fallback_lookback_days = 90` for everything — it must be refactored to implement this split.

### Known gaps

- **Stage 5B is partially implemented**: only Tier-1 (topology) instance-level CMMS fetch is production-ready. Class-level fetch, EDMS, and FMEA document ingestion are stubs.
- **`EquipmentSimilarityResolver` exists** in `equipment_similarity/` but is not wired into `CMMSContextBuilder` by default.
- No `TelemetryAdapter` protocol — `telemetry_summary.json` is assumed to arrive pre-formatted (see Data Management Strategy, gap L3).
- OE LLM tier (`IRISLLMClient`, `ADAMSLLMClient`) is future; calls are not made at Stage 5B.

> **Code changes required — Stage 5B** (`cmms_integration/cmms_context_builder.py`):
>
> 1. **Split CMMS fetch by purpose**: replace the single `fallback_lookback_days = 90` parameter with two: `operational_lookback_days` (90 days, or last PM date) and `recurrence_history_lookback_days` (full CMMS retention). Issue one CMMS query with the longer window. Tag each returned record with `retrieval_purpose: "operational" | "recurrence"` based on whether its date falls within the operational window. Only operational records get full-text Chroma embedding (capped at `max_cr_records_chroma`); recurrence records contribute lightweight metadata only (date, asset_id, failure_mode_keywords, status — no text, no cap).
>
> 2. **`recurrence_metadata[]` field**: add a new field to `cmms_context` containing the lightweight metadata extracted from all historical CR records (both operational and recurrence window). This replaces `kg_context.past_events[]` as the input to Stage C (TSKR recurrence profile) and Stage D (historical-event candidate pool). Schema: `{cr_id, asset_id, created_date, days_before_event, failure_mode_keywords[], status, is_sister_equipment}`.
>
> 3. **Class-level CMMS fetch**: implement fetch type 2 from the key logic section — KG resolves `component_type → equipment_id list`, CMMS queried per ID. Apply same dual-purpose split (operational vs recurrence). Add `failure_mode_class` keyword pre-filter to limit volume (see Data Management Strategy L4).
>
> 4. **EDMS fetch**: implement fetch type 3 — for each `doc_id` in `kg_context.documents[]`, fetch document content from EDMS adapter. Route through two-path ingestion (structured or PDF) before embedding.
>
> 5. **FMEA document fetch**: implement fetch type 4 — for each `component_type` in `kg_context.components[]`, fetch FMEA source document. Embed full text; do not apply operational window (FMEA documents have no recency decay).
>
> 6. **`source_tier` metadata tag**: add to every Chroma document at embed time (`plant_instance`, `plant_procedure`, `plant_fmea`, `plant_family`, `oe_iris`, `oe_adams`). Requires schema update to `evidence_bundle` (see Data Management Strategy §6).
>
> 7. **`EquipmentSimilarityResolver` integration**: wire Tier-2 (failure-mode overlap) and Tier-3 (spec embedding) sister resolution into `CMMSContextBuilder` alongside the existing Tier-1 topology sisters.
>
> 8. **Two-path ingestion routing**: implement format detection in `CMMSAdapter` and `EDMSAdapter` — detect structured object vs PDF and route to Path A (field mapping + NER) or Path B (parser → chunking → NER → embed) transparently.

---

## Stage B.5 — Topology-Driven Anomaly Fetch

**Class / entry-point**: *(not yet implemented — target: `orchestrators/signal_evidence_builder.py`)*
**Runs**: independently of Stage 5B after Stage B; both can execute in parallel

### What this stage does

Stage B.5 closes a structural asymmetry in the pipeline: documents are retrieved intelligently (Stage B queries the KG for relevant doc IDs, Stage 5B fetches their content), but anomalies are supplied externally as a pre-assembled `telemetry_summary.json`. Stage B.5 applies the same topology-driven logic to anomaly retrieval — using the neighborhood sensor IDs already resolved by Stage B to drive a targeted historian query, then reasoning about the temporal ordering of the anomalies across the plant graph to construct propagation chains.

The stage has two conceptually distinct responsibilities that must be kept separate.

**Sub-step 1 — Topology-driven anomaly fetch**: Stage B produced `kg_context.components[]`, each linked to `monitored_variable` nodes via the sensor map (Stage 0). Stage B.5 reads those sensor IDs, defines a query time window anchored to the event, and calls the plant historian API to retrieve pre-flagged anomaly records for each sensor. The returned anomaly set is merged with `telemetry_summary.json` (the external input, which captures the triggering event context) to produce a unified `augmented_anomaly_set`. Stage B.5 does not perform anomaly detection — it assumes the historian provides pre-flagged anomaly records (start timestamp, end timestamp, pattern type, severity). If the historian API is unavailable or returns no additional anomalies, the augmented set equals `telemetry_summary.json` and the stage degrades gracefully.

**Sub-step 2 — Propagation chain construction**: with the augmented anomaly set in hand, Stage B.5 computes pairwise Allen interval relations across all anomaly windows and cross-references each pair against the plant topology graph from the KG. An edge is drawn from anomaly A to anomaly B in a directed propagation graph when two conditions hold simultaneously: (1) A PRECEDES or OVERLAPS B (Allen relation — A is temporally upstream of B), and (2) the component behind sensor A is topologically upstream of the component behind sensor B in the MBSE containment or connectivity graph. Both conditions must hold — temporal precedence without topology alignment, or topology alignment without temporal precedence, does not constitute a directed propagation edge. The resulting graph is a DAG (directed acyclic graph by construction, since Allen PRECEDES/OVERLAPS are antisymmetric with respect to the event endpoint). Longest consistent paths through this DAG — sequences of anomalies that are both temporally ordered and topology-aligned — are the candidate propagation chains. Each chain terminates at the event interval.

The output artifact `signal_evidence` carries the augmented anomaly set, the propagation chains, a per-candidate chain score (which candidate failure mode sits at or near the root of the strongest chain), and coverage statistics for analyst review. `signal_evidence` is a **separate artifact from `evidence_bundle`** by design: document evidence (text snippets, Chroma retrieval) and signal evidence (sensor anomalies, topology chains) have different sources, quality indicators, and failure modes. Keeping them separate preserves clean provenance tracing at Stage J and allows Stage F to degrade gracefully when signal evidence is absent.

If no propagation chains are found (no anomaly pairs satisfy both the Allen and topology conditions), Stage B.5 still emits `signal_evidence` with an empty `propagation_chains[]` and a `chain_coverage: 0` flag. Stage C and Stage F both detect this and fall back to their existing independent-window scoring behavior — the pipeline is structurally unchanged in the no-chain case.

### Input artifacts

| Input | Key fields consumed |
|-------|---------------------|
| `kg_context` | `components[].component_id`, `.monitored_variable_ids`, `.relation_to_seed`; topology edges for upstream/downstream resolution |
| `telemetry_summary.json` | `signals[].sensor_id`, `.anomalies[].timestamp_start/end`, `.pattern`, `.severity` — baseline anomaly set to augment |
| `event.json` | `timestamp_start`, `timestamp_end` — anchor for historian query window |
| `run_context` | `run_id`, `asset_id` |
| Historian API (live) | Pre-flagged anomaly records by `sensor_id` and time window |

### Key logic

**Sub-step 1 — Anomaly fetch and merge**

- Reads `sensor_ids` from `kg_context.components[].monitored_variable_ids` (all neighborhood sensors resolved by Stage B via the sensor map loaded in Stage 0).
- Computes query window: `[event_start − fetch_lookback_hours, event_end + fetch_lookahead_hours]`. Default lookback = 72 h; lookahead = 4 h. Both are configurable.
- Calls `HistorianAdapter.get_anomalies(sensor_ids, window_start, window_end)` — returns pre-flagged anomaly records per sensor.
- Merges historian records with `telemetry_summary.signals[].anomalies[]`: deduplicates by `(sensor_id, timestamp_start)` with a 5-minute tolerance; historian records take precedence when a conflict exists (more precise start timestamp).
- Tags each anomaly record with `source: "historian" | "telemetry_summary"` for provenance.
- Records sensors with no historian response in `fetch_gaps[]` for analyst visibility.

**Sub-step 2 — Propagation chain construction**

- Resolves topology direction for each sensor pair using the KG: `neo4j.is_upstream(component_a, component_b)` — returns True if `component_a` reaches `component_b` via `has_part_usage` or `owns_port_usage → connects_port` traversal with direction preserved from Stage 0 MBSE ingest.
- Computes pairwise Allen relation for every pair of anomaly windows in the augmented set using `allen_relation()` from `temporal_relations.py`.
- Creates a directed edge `(anomaly_i → anomaly_j)` only when **both** hold: Allen relation ∈ {PRECEDES, OVERLAPS} AND `is_upstream(sensor_i.component, sensor_j.component) = True`.
- Builds the propagation DAG from all directed edges.
- Extracts all maximal paths in the DAG (paths that are not subpaths of any longer path) using depth-first traversal.
- Scores each path: `path_score = mean(allen_base_scores) × topology_alignment_factor × lag_consistency_factor`.
  - `topology_alignment_factor`: 1.0 if all edges follow containment hierarchy, 0.85 if mixed containment + connectivity, 0.70 if connectivity only.
  - `lag_consistency_factor`: derived from std/mean of inter-anomaly lags along the path; lower variance → higher score.
- Retains top-N paths (default N=5) ranked by `path_score`.
- Computes `per_candidate_chain_score`: for each failure mode in `kg_context.failure_modes[]`, finds the highest-scoring chain where the FM's component appears, and returns the FM's position score (root = 1.0, intermediate = 0.5 × path_score, absent = 0.0).

### Output artifact — `signal_evidence`

| Field | Content |
|-------|---------|
| `augmented_anomaly_set[]` | Merged anomaly records from historian + telemetry_summary; each has `sensor_id`, `component_id`, `timestamp_start/end`, `pattern`, `severity`, `source` |
| `propagation_chains[]` | Ranked list of chain objects; each has `chain_id`, `path_score`, `topology_alignment_factor`, `lag_consistency_factor`, `nodes[]` (ordered anomaly sequence with `allen_relation_to_next`, `edge_type`) |
| `per_candidate_chain_score` | `{fm_id: {chain_position_score, best_chain_id, position_type}}` — consumed by Stage C and Stage F |
| `chain_coverage` | Float [0,1]: fraction of augmented anomaly set covered by at least one chain |
| `fetch_gaps[]` | Sensors with no historian response: `{sensor_id, component_id, reason}` |
| `augmented_anomaly_count` | Total anomaly count post-merge |
| `historian_anomaly_count` | Anomalies contributed by historian (above telemetry_summary baseline) |

### Critical path pseudo-code

```python
def build_signal_evidence(kg_context, telemetry_summary, event, run_context,
                           historian_adapter, neo4j_conn):

    # --- Sub-step 1: Anomaly fetch and merge ---
    sensor_ids = [mv_id
                  for comp in kg_context["components"]
                  for mv_id in comp.get("monitored_variable_ids", [])]

    window_start = parse(event["timestamp_start"]) - timedelta(hours=FETCH_LOOKBACK_H)
    window_end   = parse(event.get("timestamp_end", event["timestamp_start"])) \
                   + timedelta(hours=FETCH_LOOKAHEAD_H)

    historian_anomalies, fetch_gaps = historian_adapter.get_anomalies(
        sensor_ids, window_start, window_end
    )

    baseline = extract_anomalies(telemetry_summary)   # existing telemetry_summary records
    augmented = merge_anomalies(baseline, historian_anomalies,
                                dedup_tolerance_min=5)  # 5-min tolerance

    # --- Sub-step 2: Propagation chain construction ---
    event_interval = Interval(parse(event["timestamp_start"]),
                               parse(event.get("timestamp_end", event["timestamp_start"])))

    edges = []
    for i, a in enumerate(augmented):
        for j, b in enumerate(augmented):
            if i == j:
                continue
            rel, _ = allen_relation(to_interval(a), to_interval(b))
            if rel not in (PRECEDES, OVERLAPS):
                continue
            if neo4j_conn.is_upstream(a["component_id"], b["component_id"]):
                edges.append((i, j, rel))

    dag = build_dag(nodes=augmented, edges=edges)
    paths = find_maximal_paths(dag)

    scored_chains = []
    for path in paths:
        scored_chains.append(score_chain(path, augmented))
    scored_chains.sort(key=lambda c: -c["path_score"])
    top_chains = scored_chains[:TOP_N_CHAINS]

    # Per-candidate chain scores for Stage C and Stage F
    per_candidate = {}
    for fm in kg_context["failure_modes"]:
        best = find_best_chain_for_fm(fm["fm_id"], fm["applies_to_component_id"],
                                      top_chains, augmented)
        per_candidate[fm["fm_id"]] = best  # {chain_position_score, best_chain_id, position_type}

    return build_signal_evidence_artifact(
        augmented, top_chains, per_candidate, fetch_gaps, run_context["run_id"]
    )
```

### Key thresholds and configuration

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fetch_lookback_hours` | 72 | How far before the event to query the historian |
| `fetch_lookahead_hours` | 4 | Short forward window to capture post-event confirmation signals |
| `dedup_tolerance_minutes` | 5 | Two anomaly records with same sensor_id within this window are considered duplicates |
| `top_n_chains` | 5 | Maximum propagation chains retained in output |
| Chain root position score | 1.0 | FM's component is the first node of the chain |
| Chain intermediate score | 0.5 × path_score | FM's component is a non-root node |
| Chain absent score | 0.0 | FM's component does not appear in any chain |
| `topology_alignment_factor` | 1.0 / 0.85 / 0.70 | Containment-only / mixed / connectivity-only path |

### Known gaps

- **Historian adapter not implemented**: no `HistorianAdapter` class exists in the codebase. The interface must be defined and at minimum one adapter written (OSIsoft PI is the most common in nuclear plants). The adapter must return structured anomaly records — if the historian returns only raw time series, a preprocessing step is needed upstream.
- **`is_upstream()` KG query not implemented**: resolving topology direction between component pairs requires a directed path query in Neo4j. The Stage 0 MBSE ingest must preserve edge direction (parent→child for containment, signal-flow direction for connectivity). The current `kg_ingest_neo4j_workflow.py` does not guarantee directional consistency.
- **Pairwise Allen computation scales quadratically**: for N anomalies in the augmented set, pairwise comparison is O(N²). For a large plant neighborhood with many sensors and a long lookback window, this may produce hundreds of anomalies and tens of thousands of comparisons. A pruning step (e.g., only compare anomalies within a maximum lag window) is needed for production use.
- **Propagation DAG may contain cycles if topology has feedback loops**: the MBSE connectivity model can include recirculation paths (e.g., pump discharge feeding back to suction header). The `is_upstream` query must be cycle-aware — the current proposal assumes a strict DAG. If cycles exist, the longest-path algorithm is undefined; a cycle-detection step and a fallback (e.g., ignore feedback edges) are required.
- **`per_candidate_chain_score` weights are not calibrated**: the root/intermediate/absent scoring (1.0 / 0.5×path_score / 0.0) is a reasonable prior but will require empirical calibration against historical RCA cases. This is explicitly flagged for near-term iteration.
- **No chain evidence for out-of-boundary anomalies**: anomalies from sensors not in the KG neighborhood (`out_of_boundary_anomalies` from Stage B) are not included in the propagation chain analysis. A sensor outside the neighborhood could be the actual root cause — the chain would be incomplete. A partial extension of the DAG to include out-of-boundary nodes (with a lower confidence weight) is a future enhancement.

---

## Stage C — TSKR Temporal Scoring

**Class / entry-point**: `TSKRTemporalScorerV1.score()`
**File**: `orchestrators/tskr_temporal_scorer.py`

### What this stage does

Stage C asks a single question for each failure mode in the KG neighborhood: is the timing of what was observed in the plant consistent with how this failure mode is known to propagate? It does not decide whether a failure mode caused the event — that is Stage D. It characterises the temporal relationship between the anomaly signals and the event, and packages that characterisation as a scored pattern that Stage D can consume.

**How anomalies are connected to failure mode propagation**

The connection works in two steps. First, each sensor in `telemetry_summary.signals[]` is linked to a component in the KG through the monitored-variable relationship (`element_usage → monitors → monitored_variable → sensor`). Stage B already resolved this mapping when it built the neighborhood, so Stage C knows which anomalies belong to components in the causal search space.

Second, each failure mode in the KG carries two temporal parameters loaded from FMEA data during offline KG population: `expected_latency_min_hours` and `expected_latency_max_hours`. These encode domain knowledge about how long that failure mode takes to manifest as an observable event — for example, bearing wear produces a vibration anomaly 2–6 hours before seizure; heat exchanger fouling produces a temperature rise 24–72 hours before a protective trip. Stage C compares the observed lag between the anomaly window and the event against this expected window and classifies the result as `none | too_fast | too_slow | unknown`.

A concrete example: if a vibration anomaly on pump P-101A starts 4 hours before the event, and the bearing wear failure mode has an expected latency of 2–6 hours, Stage C assigns Allen relation PRECEDES, observed lag 4 hours (within window), `latency_violation_type: none`, and returns a high confidence pattern. If the same anomaly appeared only 30 minutes before the event, the lag falls below the minimum — `latency_violation_type: too_fast` — and the confidence drops accordingly.

**What Stage C does not verify**

Stage C scores all anomalies from neighborhood sensors against each failure mode without first checking whether the anomaly pattern physically matches what that failure mode would produce. A pressure anomaly on a downstream valve gets scored against a bearing wear failure mode. The Allen relation and lag check handle implausible combinations naturally — wrong timing produces a low score — but there is no explicit pattern-to-failure-mode filter at this stage. That matching is the role of the symptom match sub-score in Stage D. Stage C and Stage D therefore provide complementary evidence: Stage C answers "is the timing consistent?"; Stage D answers "are the symptoms consistent?". Both must hold for a candidate to score well.

**Recurrence**

Alongside the Allen relation scoring, Stage C builds a recurrence profile from the CR history retrieved at Stage 5B. Recurrence matters because a failure mode that has appeared multiple times on this equipment — especially if prior events were never fully resolved — is a stronger candidate than one appearing for the first time. The trend of the inter-event intervals (accelerating, stable, or improving) is also captured, as an accelerating recurrence is a direct signal of unresolved degradation.

The output of Stage C — `tskr_patterns` — does not rank candidates or make causal judgements. It is a structured temporal evidence package, one pattern per failure mode, that Stage D folds into its 5-dimensional scoring as the temporal sub-score dimension.

### Input artifacts

| Input | Key fields consumed |
|-------|---------------------|
| `kg_context` | `failure_modes[].fm_id`, `.expected_latency_min/max_hours`, `.pattern` |
| `cmms_context` | `cr_records[].{created_date, failure_mode_keywords, asset_id, status}` — source of recurrence history, replacing `kg_context.past_events[]` |
| `signal_evidence` | `augmented_anomaly_set[]` — replaces raw `telemetry_summary` anomalies as the anomaly source; `per_candidate_chain_score` — provides `chain_position_score` sub-score per FM |
| `telemetry_summary` | Fallback anomaly source when `signal_evidence` is absent or `augmented_anomaly_set` is empty |
| `event.json` | `timestamp_start`, `timestamp_end` |
| `operational_context` | Used only to resolve event interval if end timestamp is missing |

### Allen interval relations

The pipeline uses a reduced set of 5 relations from Allen's original 13 (the full set includes MEETS, STARTS, FINISHES and their inverses, which are not discriminating at the time resolution of plant anomaly windows). All relations describe interval A (an anomaly window) relative to interval B (the event interval). An epsilon tolerance of 0.5 h absorbs timestamp noise and near-simultaneous boundary cases.

| Relation | A relative to B | RCA base score | Causal interpretation |
|----------|----------------|----------------|-----------------------|
| `OVERLAPS` | A starts before B onset and is still active when B begins | 0.90 | Strongest causal signal — degradation was actively progressing at the moment of the event |
| `CONTAINS` | A spans the entire event interval (A start < B start and A end > B end) | 0.85 | Long-running latent condition — chronic degradation that predates and outlasts the event |
| `PRECEDES` | A ends before B starts; a measurable lag exists between anomaly resolution and event onset | 0.75 | Classic lead-time pattern — anomaly subsided before the event but may have set conditions |
| `DURING` | A starts at or after B onset (including anomalies that straddle B end) | 0.30 | Likely a consequence or downstream symptom, not a cause; not excluded but scored low |
| `FOLLOWS` | A starts after B ends entirely | 0.10 | Temporal contradiction — anomaly post-dates event resolution; triggers `temporal_contradiction: true` and a −0.20 confidence penalty |

The dominant relation for a failure mode is selected from the full set of anomaly windows using the priority order OVERLAPS > CONTAINS > PRECEDES > DURING > FOLLOWS. When multiple anomalies on the same component show different relations, the highest-priority relation governs. Base scores are refined downstream by latency alignment and severity weighting.

### Key logic

- Uses `signal_evidence.augmented_anomaly_set[]` as the anomaly source when Stage B.5 has run; falls back to `telemetry_summary.signals[].anomalies[]` when `signal_evidence` is absent or empty. All subsequent scoring steps are identical regardless of source.
- Builds a per-failure-mode temporal pattern by scoring each FM's anomaly windows against the event interval using the 5 Allen relations above.
- Derives a dominant Allen relation (priority: OVERLAPS > CONTAINS > PRECEDES > DURING > FOLLOWS) and computes severity-weighted mean/std lag across causal windows.
- Scores latency alignment: compares observed lag to FM's `expected_latency_min/max_hours`; classifies `latency_violation_type` as `none | too_fast | too_slow | not_available`. **When latency bounds are present, this is a strong discriminating signal.** When absent — which is the common case, as `expected_latency_min/max_hours` is not part of any standard FMEA format — the score abstains at 0.50 (neutral, neither helps nor hurts) rather than applying a floor penalty. The primary temporal discriminator when latency bounds are absent is the propagation chain position score from Stage B.5. See `RCA_FMEA_handling_spec.md` §2.1 and §5 for the architectural rationale and the revised `score_latency()` specification.
- Reads `chain_position_score` from `signal_evidence.per_candidate_chain_score[fm_id]` and folds it into the confidence composite as a new sub-score term (see formula below). When `signal_evidence` is absent, `chain_position_score = 0.0` for all FMs and the formula reduces to the existing weights.
- Builds a recurrence profile from `cmms_context.cr_records[]`: count, inter-event trend (increasing/stable/decreasing), unresolved count, recency.
- Detects temporal contradiction: FOLLOWS relation (anomaly after event) → sets `temporal_contradiction: true`, applies −0.20 confidence penalty.
- Combines anomaly, latency, chain position, history, and count sub-scores into `confidence`:

```
confidence = clamp01(
    0.45 × anomaly_score              # reduced from 0.55 to make room for chain term
  + 0.30 × latency_alignment_score   # unchanged — primary discriminator
  + 0.10 × chain_position_score      # new — root of propagation chain boosts score
  + 0.10 × history_score             # reduced from 0.15
  + 0.15 × count_score
  + 0.10 × lag_consistency
  − (0.20 if contradiction else 0.0)
)
```

> **Note**: weights are initial estimates and are explicitly flagged for near-term iteration once calibration data from historical RCA cases is available.

> **Architectural note — what Stage C does not do**: Stage C scores each failure mode against the full pool of neighborhood anomalies but does not reason about the *temporal ordering among anomalies themselves*. It does not reconstruct a propagation chain of the form "sensor A anomaly preceded sensor B anomaly preceded the event, therefore the causal path is A→B→event." Each failure mode is scored independently against each anomaly window; inter-anomaly sequencing is not exploited. This means the pipeline cannot currently distinguish between a root cause that triggered a cascade (multiple anomalies in a temporal chain) and an incidental co-occurrence of unrelated anomalies. Inter-anomaly causal chain reconstruction would require a dedicated propagation-path stage between Stage C and Stage D and is a known architectural gap.

### Output artifact — `tskr_patterns`

| Field | Content |
|-------|---------|
| `patterns[]` | One entry per failure mode |
| `pattern_id` | `"TSKR::{fm_id}"` |
| `confidence` | Weighted composite [0, 1] |
| `support` | Secondary aggregate [0, 1] |
| `relation` | Dominant Allen relation string |
| `mean_lag_hours`, `std_lag_hours` | Severity-weighted statistics |
| `recurrence_count`, `recurrence_trend` | History from `cmms_context.cr_records[]` |
| `unresolved_recurrence_count` | Prior events never resolved |
| `latency_alignment_score` | [0, 1] |
| `latency_violation_type` | `none \| too_fast \| too_slow \| not_available` |
| `temporal_contradiction` | bool |

### Critical path pseudo-code

```python
def score(event, telemetry, kg_context, cmms_context, signal_evidence, oc, run_context):
    event_interval = Interval(event["timestamp_start"], event.get("timestamp_end"))

    # Use augmented anomaly set from Stage B.5 when available; fall back to telemetry_summary
    if signal_evidence and signal_evidence.get("augmented_anomaly_set"):
        anomaly_windows = extract_anomaly_intervals_from_augmented(
            signal_evidence["augmented_anomaly_set"]
        )
    else:
        anomaly_windows = extract_anomaly_intervals(telemetry)

    # Index chain position scores from Stage B.5 (zero when signal_evidence absent)
    chain_score_index = {}
    if signal_evidence:
        for fm_id, entry in signal_evidence.get("per_candidate_chain_score", {}).items():
            chain_score_index[fm_id] = entry.get("chain_position_score", 0.0)

    patterns = []
    for fm in kg_context["failure_modes"]:
        # Allen relation scoring
        allen_scores = []
        for aw in anomaly_windows:
            relation = compute_allen_relation(aw, event_interval, epsilon_h=0.5)
            allen_scores.append((relation, aw.severity, aw.duration))

        dominant_relation = pick_dominant(allen_scores)  # OVERLAPS > CONTAINS > ...
        causal_windows    = [a for a in allen_scores if a.relation in CAUSAL_SET]
        mean_lag = severity_weighted_mean([a.lag for a in causal_windows])
        std_lag  = severity_weighted_std([a.lag for a in causal_windows])

        # Latency alignment — returns (0.50, "not_available") when bounds absent
        lat_score, violation = score_latency(
            mean_lag, fm["expected_latency_min_hours"], fm["expected_latency_max_hours"]
        )

        # Recurrence profile — built from CMMS CR records (not kg_context.past_events)
        profile = build_recurrence_profile(fm["fm_id"], cmms_context["cr_records"])

        # Sub-scores
        anomaly_score   = severity_weighted_mean_of_relation_scores(allen_scores)
        count_score     = score_anomaly_count(len(causal_windows))
        lag_consistency = score_lag_consistency(std_lag)
        history_score   = score_history(profile)
        chain_pos_score = chain_score_index.get(fm["fm_id"], 0.0)

        contradiction = (dominant_relation == "FOLLOWS")

        # Weights: 0.45+0.30+0.10+0.10+0.15+0.10 = 1.20 (non-convex; clamp01 bounds output)
        # flagged for calibration once historical RCA cases are available
        confidence = clamp01(
            0.45 * max(anomaly_score, telemetry_support_floor)
            + 0.30 * lat_score
            + 0.10 * chain_pos_score
            + 0.10 * history_score
            + 0.15 * count_score
            + 0.10 * lag_consistency
            - (0.20 if contradiction else 0.0)
        )
        support = clamp01(
            0.35 * history_score
            + 0.35 * telemetry_support_floor
            + 0.15 * count_score
            + 0.15 * lag_consistency
            - (0.15 if contradiction else 0.0)
        )

        patterns.append(build_tskr_pattern(fm, dominant_relation, confidence, support, ...))

    return {"tskr_patterns": sorted(patterns, key=lambda p: -p["confidence"])}
```

### Key thresholds

| Parameter | Value | Effect |
|-----------|-------|--------|
| `simultaneous_epsilon_hours` | 0.5 h | Tolerance for SIMULTANEOUS relation |
| `fallback_confidence` | 0.25 | Floor when no anomaly windows exist |
| Anomaly weight | 0.45 | Dominant term in confidence composite (reduced from 0.55 to make room for chain term) |
| Latency weight | 0.30 | Second-largest term |
| Chain position weight | 0.10 | Propagation chain root position boost; 0.0 when signal_evidence absent |
| History weight | 0.10 | Recurrence evidence (reduced from 0.15) |
| Count weight | 0.15 | Anomaly count evidence |
| Lag consistency weight | 0.10 | Temporal regularity of inter-anomaly lags |
| `min_confidence_for_support` | 0.35 | Must exceed to be marked "supported" |
| Contradiction penalty | −0.20 (confidence), −0.15 (support) | Discounts FOLLOWS-relation patterns |
| Recurrence trend threshold | ratio < 0.75 = "increasing", > 1.33 = "decreasing" | Three-bucket classification |
| Lag consistency thresholds | ≤0.25 h = 1.0, ≤1 h = 0.8, ≤4 h = 0.55 | Tighter consistency → higher score |

### Known gaps

- **H3 (April 20 review)**: `simultaneous_epsilon_hours = 0.5` is not documented as a user-configurable parameter; inappropriate for slow-developing failures (thermal degradation, leaks) where 0.5 hours may be far too tight.
- Weight coefficients in the confidence composite sum to > 1.0 (0.55 + 0.30 + 0.15 + 0.20 + 0.15 = 1.35), making the formula non-convex. The `clamp01` prevents out-of-range output but the relative contribution of each term depends on the others' magnitude.
- `instrument_validity_flag` is referenced in the telemetry schema but not checked in TSKR; a degraded sensor could produce spurious anomaly windows that inflate confidence.
- **FM-to-CR matching depends on NER quality**: the recurrence profile (`build_recurrence_profile`) matches CR records to failure modes via `failure_mode_keywords[]` extracted by NER at Stage 5B. If NER is partial or imprecise — which the current implementation status confirms — Stage C silently degrades to recurrence-by-asset rather than recurrence-by-failure-mode. A high-maintenance equipment item will show inflated recurrence scores regardless of which specific failure mode is being evaluated. This is a data quality dependency that is not visible in Stage C's output.
- **Allen relation classification requires clean anomaly interval endpoints**: Stage C assumes each anomaly has a reliable `timestamp_end`. In plant data, anomaly detection systems typically provide a precise `timestamp_start` but a fuzzy or missing `timestamp_end`. A missing endpoint forces a point-event fallback that can misclassify OVERLAPS as PRECEDES or SIMULTANEOUS as PRECEDES, producing incorrect Allen relation scores. The same uncertainty applies to the event interval from the CR — `timestamp_end` is often recorded when the condition was restored, not when the causal process ended. Neither case is currently handled.
- **Potential relation inconsistency between Stage B pre-filter and Stage C scoring**: Stage B admits components into the neighborhood using a lightweight Allen relation computed on raw timestamps. Stage C re-scores the same anomalies with the full TSKR machinery including epsilon tolerance and severity weighting. It is possible for a component admitted at Stage B with relation OVERLAPS to be reclassified at Stage C as FOLLOWS, triggering a temporal contradiction flag. This does not break the pipeline — Stage D would score that candidate low and Stage F would refine — but the discrepancy between Stage B's admission decision and Stage C's scoring will be visible to analysts and requires explanation.
- **Inter-anomaly temporal chain reasoning is absent**: Stage C scores each failure mode independently against each anomaly window; it does not analyze the temporal ordering among anomalies themselves. A propagation scenario where anomaly A precedes anomaly B precedes the event — suggesting a causal chain A→B→event — is not detectable at Stage C. All anomalies are treated as independent observations relative to the event interval. Implementing chain reasoning would require computing pairwise Allen relations across all anomaly windows and building a directed temporal graph, which is architecturally not present in the current pipeline.

---

## Stage D — Candidate Generation (pre-evidence)

**Class / entry-point**: `RuleBasedCausalityEngineV31.generate()` (also v32 in flowchart)
**File**: `orchestrators/causality_engine_v31.py`

### What this stage does

Stage D is where the pipeline first produces a ranked list of root cause hypotheses. By this point the search space is fully defined (Stage B), the evidence corpus has been assembled into the run-scoped Chroma collection (Stage 5B) and temporal characterisation of each failure mode is complete (Stage C). Stage D does **not** query Chroma — that happens at Stage E. At this stage the evidence corpus exists but is intentionally not yet read; Stage D uses only KG document metadata (document types and recency) as a proxy for evidence quality.

It generates two candidate pools. The first is built from the failure modes retrieved from the KG: each failure mode becomes a hypothesis of the form "this failure mode, on this component, caused the event." The second is built from the historical CR records in `cmms_context`: each past event involving the same equipment or a failure mode with a similar description becomes a hypothesis of the form "this is a recurrence of a known previous failure." Failure-mode similarity here is determined by NER keyword matching: the `failure_mode_keywords[]` extracted by Stage 5B from CR narrative text are matched against the `name` and `mechanism` fields of KG failure mode records. Without reliable Stage 5B NER output the historical-event pool degrades to asset-level matching only ("previous CRs on this equipment") rather than failure-mode-level matching. The two pools are kept separate so that historical-event candidates do not displace failure-mode candidates in the top-k ranking.

Each candidate is scored across five dimensions: structural (how topologically close is the component to the primary asset, and how well do the symptoms match?), temporal (what does the TSKR pattern say about timing consistency — this is the Stage C output consumed here), telemetry (how many anomalies are present on this component, and how severe?), evidence (how many relevant documents exist in the KG for this failure mode — a proxy score using document metadata only, not Chroma content), and governance (does PM compliance data suggest this maintenance category has been deferred or overdue?). The five scores are combined into a single composite via a weighted average.

Candidates are then tiered into **A-series** and **B-series** rather than filtered with a hard threshold:

- **A-series** (composite ≥ 0.45 AND evidence ≥ 0.35): full-pipeline treatment; eligible to become primary hypothesis; writeback-ready subject to Stage G approval.
- **B-series** (composite ≥ 0.25 AND failing at least one A-series threshold): carried through Stages E and F but always tagged speculative; `requires_human_review` is unconditionally True; never `writeback_ready`; presented in a separate section of the rca_card.
- **Safety-significant failure modes** are promoted to A-series by policy regardless of composite score, so that low-evidence safety candidates are never silently dropped.
- Candidates below composite 0.25 are dropped.

The output — `causality_candidates v1` — is a pre-evidence snapshot. It represents the pipeline's best assessment of plausible root causes before any document content has been read.

### Input artifacts

| Input | Key fields consumed |
|-------|---------------------|
| `kg_context` | `failure_modes[]`, `components[]`, `documents[]` |
| `cmms_context` | `cr_records[]` — source of historical-event candidate pool |
| `tskr_patterns` | Indexed by `fm_id` for temporal sub-scores |
| `telemetry_summary` | `signals[].anomalies[]` |
| `operational_context` | `recent_alarms`, `operating_point` |
| `pm_compliance` | `pm_checks[]`, `last_pm_date` |

### Key logic

- Generates two candidate pools: failure-mode candidates (from `kg_context.failure_modes[]`) and historical-event candidates (from `cmms_context.cr_records[]` — CMMS is the system of record for past events, not the KG).
- Computes a 5-dimensional score for each candidate: structural, temporal, telemetry, evidence, governance.
- Combines via weighted average: 0.30·S + 0.20·T + 0.20·Tel + 0.20·E + 0.10·G.
- Tiers candidates into A-series (composite ≥ 0.45 AND evidence ≥ 0.35) and B-series (composite ≥ 0.25 failing at least one A threshold); candidates below composite 0.25 are dropped.
- Safety-significant failure modes are promoted to A-series by policy, overriding score thresholds.
- Returns top-k=10 candidates ranked by composite score (A-series first, then B-series); this is `causality_candidates v1`.

### Output artifact — `causality_candidates v1`

| Field | Content |
|-------|---------|
| `candidates[]` | Ranked list; each with all 5 sub-scores, composite, series, confidence_label |
| `candidate_id` | `"FM::{fm_id}"` or `"EVT::{event_id}"` |
| `hypothesis_type` | `"failure_mode"` or `"historical_event"` |
| `composite_score` | Weighted sum of 5 sub-scores |
| `series` | `"A"` or `"B"` — tiering result; A = full pipeline; B = speculative |
| `safety_promoted` | bool — True when safety significance overrode score thresholds to assign A-series |
| `confidence_label` | `high ≥ 0.75`, `medium ≥ 0.45`, `low > 0`, `speculative`; B-series is always `speculative` |
| `requires_human_review` | bool — always True for B-series; may be True for A-series with evidence_gap |
| `writeback_eligible_v1` | bool — True for A-series only; **preliminary flag, not the final `writeback_ready`**; finalized at Stage J after all gates pass |
| `meets_evidence_threshold` | bool (evidence ≥ 0.35) |
| `evidence_gap` | bool — set True when zero evidence documents found |
| `recurrence_features` | Count, trend, unresolved count (from TSKR profile) |
| `common_cause_features` | Sister components with same FM (if topology available) |

### Critical path pseudo-code

```python
def generate(event, telemetry, kg_context, tskr_patterns, oc, pm, run_context):
    tskr_index = {p["pattern_id"].split("::")[1]: p for p in tskr_patterns["patterns"]}
    candidates = []

    for fm in kg_context["failure_modes"]:
        comp = resolve_component(fm, kg_context["components"])
        tskr = tskr_index.get(fm["fm_id"], {})

        # 1. Structural score
        topology_base = TOPOLOGY_BASE[comp["seed_match_type"]]  # 0.90/0.85/0.75/0.40
        symptom_delta = 0.40 * (symptom_match(fm, event) - 0.5)  # ±0.20
        S = clamp01(topology_base + symptom_delta)

        # 2. Temporal score (delegates to TSKR)
        T = min(1.0,
            0.35 * tskr.get("confidence", 0)
            + 0.30 * allen_to_score(tskr.get("relation"))
            + 0.20 * latency_score(tskr)
            + 0.15 * tskr.get("support", 0)
        )

        # 3. Telemetry score
        n_anomalies = count_anomalies_for_component(comp, telemetry)
        severity_pts = sum_severity_points(comp, telemetry)
        Tel = min(1.0, 0.35 + 0.12 * n_anomalies + 0.08 * severity_pts
                  + (0.10 if comp["seed_match_type"] == "telemetry" else 0))

        # 4. Evidence score (KG document proxy — pre-Chroma)
        doc_types = get_doc_types_for_fm(fm, kg_context["documents"])
        recency_cr_wo = max_recency_factor(doc_types, ["CR", "WO"])
        E = min(1.0,
            0.30
            + (0.25 * recency_fmea if "FMEA" in doc_types else 0)
            + (0.20 * recency_cr_wo if doc_types & {"CR","WO"} else 0)
            + (0.15 * recency_eca_rca if doc_types & {"ECA","RCA"} else 0)
        )

        # 5. Governance score (PM compliance)
        G = score_governance(fm, pm)

        composite = 0.30*S + 0.20*T + 0.20*Tel + 0.20*E + 0.10*G

        safety_sig = fm.get("safety_significant", False)

        if safety_sig or (composite >= 0.45 and E >= 0.35):
            series = "A"
        elif composite >= 0.25:
            series = "B"
        else:
            continue  # below floor — drop

        c = build_candidate(fm, S, T, Tel, E, G, composite)
        c["series"] = series
        c["safety_promoted"] = safety_sig and series == "A" and not (composite >= 0.45 and E >= 0.35)
        c["requires_human_review"] = (series == "B") or c.get("evidence_gap", False)
        c["writeback_eligible_v1"] = (series == "A")  # preliminary flag only — finalized at Stage J
        # B-series always speculative regardless of raw composite
        if series == "B":
            c["confidence_label"] = "speculative"
        candidates.append(c)

    candidates.sort(key=lambda c: (c["series"] != "A", -c["composite_score"]))
    return {"causality_candidates": candidates[:top_k]}
```

### Key thresholds

| Parameter | Value | Effect |
|-----------|-------|--------|
| Weights | S=0.30, T=0.20, Tel=0.20, E=0.20, G=0.10 | Relative dimension importance |
| `a_series_composite_threshold` | 0.45 | A-series composite floor |
| `a_series_evidence_threshold` | 0.35 | A-series evidence floor |
| `b_series_composite_threshold` | 0.25 | B-series composite floor (below = dropped) |
| Safety promotion | override — A-series regardless of score | Safety-significant FMs never dropped |
| `top_k_candidates` | 10 | Output list cap; A-series ranked before B-series |
| Seed-match topology bases | telemetry=0.90, direct=0.85, neighbor=0.75, none=0.40 | Structural baseline |
| Symptom delta | ±0.20 | Maximum symptom modifier on structural |
| Evidence baseline | 0.30 | Score when no documents exist |
| Recency factor breakpoints | ≤90d=1.0, ≤365d=0.85, ≤730d=0.70, >730d=0.55 | |
| PM governance cap | 0.95 | Governance alone cannot produce perfect score |
| Allen relation score mapping | PRECEDES=1.0, OVERLAPS=0.95, CONTAINS=0.90, … FOLLOWS=0.20 | |

### A-series vs B-series treatment

| Attribute | A-series | B-series |
|-----------|----------|----------|
| `confidence_label` | high / medium / low per composite | always `speculative` |
| Eligible for primary hypothesis | Yes | No — requires analyst override |
| `writeback_eligible_v1` | True (subject to Stage J final gate) | Always False |
| `requires_human_review` | Only when `evidence_gap=True` | Always True |
| rca_card placement | Main candidates section | Separate "speculative candidates" section |
| Carries through Stage E/F | Yes | Yes — evidence still retrieved and scored |

### Known gaps

- **C2 (April 20 review)**: `top_k=10` is a hard cap. Common-cause failures affecting many parallel trains could generate more candidates than the cap allows, silently dropping some.
- **C4 (April 20 review)**: Safety significance is not a scoring dimension. A low-composite candidate on a safety-critical function will rank below a high-composite candidate on a non-safety system.
- Evidence score at Stage D uses only KG document metadata (types and recency). Actual content is not available until Stage E; this makes Stage D evidence scores a proxy, not a real evidence assessment — the v1→v2 delta in Stage F is intended to correct this, but the proxy baseline may be misleading.
- **`evidence_gap` flag** is set when zero documents are found. With A/B-series tiering, zero-evidence candidates with composite ≥ 0.25 now land in B-series (`requires_human_review: True`) rather than being dropped silently. However, B-series candidates appear in a separate rca_card section — an analyst who does not read that section may still miss them.
- **`telemetry_anomaly_precedes` seed_match_type not mapped in structural score**: Stage B's proposed telemetry-driven neighborhood expansion adds components with `seed_match_type = "telemetry_anomaly_precedes"`. This type is not in Stage D's `TOPOLOGY_BASE` mapping and will fall through to the default score of 0.40 — the weakest structural baseline, equivalent to "no topological relationship." A component admitted at Stage B specifically because its anomaly preceded the event would be structurally penalised rather than rewarded. A new entry is needed in the mapping, likely ~0.80, reflecting that admission was evidence-driven but not a direct asset match. **Code change required**: update `TOPOLOGY_BASE` in `causality_engine_v31.py` to include `"telemetry_anomaly_precedes": 0.80`.
- **Evidence baseline (0.30) places undocumented failure modes in B-series**: the evidence score baseline when no documents are linked in the KG is 0.30, which falls below the A-series evidence threshold of 0.35. A failure mode with no KG document links cannot reach A-series on evidence alone, regardless of structural and temporal signal strength. With A/B-series tiering this is no longer a silent drop — the candidate reaches B-series if composite ≥ 0.25 — but it will never be eligible for the primary hypothesis or writeback without analyst override. For a first-occurrence failure mode with strong telemetry and structural signals, this is the correct conservative behavior; the design decision has been made explicit.
- **Historical-event candidate scoring degrades without FM-to-CR mapping**: switching the historical-event pool from `kg_context.past_events[]` (structured, with `fm_ids` and `component_ids`) to `cmms_context.cr_records[]` (free-text narratives + asset_id) breaks the structural overlap scoring (`target_fm_ids intersect matched_fms`). Without the NER keyword→fm_id mapping from Stage 5B, historical-event candidates degrade to asset-level matching only — "there were previous CRs on this equipment" — rather than failure-mode-level matching. The quality of the historical-event pool is directly coupled to Stage 5B NER completeness.
- **Operational context not used in scoring**: `operational_context.operating_point` (power level, temperatures, pressures at event time) is an input to Stage D but contributes to none of the five sub-scores. Operating state can directly discriminate between failure modes — cavitation only occurs below minimum flow, thermal fatigue is more likely during transients, flow-induced vibration scales with power level. Ignoring operating state is a functional gap from a plant engineering standpoint.
- **Two candidate pools, single top-k cap, merge logic unspecified**: Stage D generates failure-mode candidates and historical-event candidates as separate pools, stated as keeping them separate "so that historical events don't displace failure modes." However, the merge and ranking logic before the top-k=10 cut is not defined. If the pools are merged before ranking, failure-mode candidates will dominate and historical-event candidates may not appear at all. If each pool receives an independent sub-cap, the split ratio is not specified. This must be an explicit design decision.

---

## Stage E — Evidence Retrieval

**Class / entry-point**: `ChromaEvidenceRetriever.retrieve()`
**File**: `storage/chroma_evidence_retriever.py`

### What this stage does

Stage E is the first point in the pipeline where actual document content — not metadata, not KG structure, not scoring proxies — is read and linked to specific hypotheses. Stages B through D established what the plausible causes are; Stage E asks whether the plant's own documentation supports or contradicts those hypotheses.

For each candidate in `causality_candidates v1`, Stage E constructs a hypothesis-guided query plan and executes it against the run-scoped Chroma collection assembled at Stage 5B. The query is built from the candidate's cause label, component type, and failure mode description — enough semantic content to retrieve documents that discuss this specific failure mechanism on this type of equipment. Three query variants are issued per candidate, each targeting a different evidence role: supporting (what confirms this hypothesis?), contradicting (what argues against it?), and contextual (what provides relevant background?).

Each retrieved snippet is scored by cosine similarity and by the authority tier of its source document — plant instance records (CRs, WOs on this specific equipment) carry higher authority than fleet-level OE references. The aggregated result per candidate — supporting count, contradicting count, best support score, best contradiction score — feeds directly into Stage F's evidence score update.

It is important to understand what Stage E can and cannot retrieve. Its entire corpus is bounded by what Stage 5B embedded. If Stage 5B did not fetch a document — because EDMS stubs are not implemented, because an FMEA document was not ingested, because the OE LLM tier is future — Stage E will silently return no evidence for those sources. An empty evidence result at Stage E does not mean no evidence exists; it means no evidence was assembled at Stage 5B.

### Input artifacts

| Input | Key fields consumed |
|-------|---------------------|
| `causality_candidates v1` | `candidate_id`, `cause_label`, `fm_id`, `kg_path` |
| `kg_context` | `failure_modes[].description`, `components[].component_type` |
| Run-scoped Chroma | Collection keyed by `run_id`; documents from Stage 5B |

### Key logic

- Constructs a **hypothesis-guided query plan** for each candidate: one query per evidence role (supporting, contradicting, contextual), using cause_label + component_type + FM description.
- Queries the run-scoped Chroma collection; each query retrieves top-k snippets ranked by cosine similarity.
- Assigns support role labels (`supporting | contradicting | contextual`) based on query intent and snippet relevance.
- Scores each snippet by authority tier: `plant_instance` > `plant_procedure` > `plant_fmea` > `plant_family` > `oe_iris` > `oe_adams`.
- Aggregates per-candidate: `supporting_count`, `contradicting_count`, `best_support_score`, `best_contradiction_score`.

### Output artifact — `evidence_bundle`

| Field | Content |
|-------|---------|
| `snippets[]` | Raw evidence items: `snippet_id`, `doc_id`, `score`, `snippet` (text excerpt), `support_role`, `authority_score` |
| `per_candidate_summary` | `{candidate_id: {supporting_count, contradicting_count, best_support_score, best_contradiction_score}}` |
| `bundle_id` | UUID |

### Known gaps

- No `source_tier` field in the evidence_bundle schema (see Data Management Strategy, open issue OQ-1); authority information is computed but not stored as a stable schema field.
- Snippet text is a vector-retrieval excerpt, not verbatim text (see April 20 review, §4.5 linkage break).
- Evidence retrieval is bounded by what Stage 5B embedded; documents not fetched at Stage 5B (e.g., EDMS stubs) produce empty results silently.
- **Support role assignment is query-intent circular**: the current design labels a snippet "supporting" because it was retrieved by a supporting-intent query, and "contradicting" because it was retrieved by a contradicting-intent query. This is circular — a document retrieved via a supporting query may in fact contain contradicting content and vice versa. True support role classification requires reading the snippet in relation to the hypothesis, which is an LLM or NLP task, not a retrieval task. Without this, the `support_role` field reflects retrieval intent, not actual document stance.
- **Contradicting evidence retrieval is semantically ill-defined for vector search**: querying for documents that contradict a hypothesis is fundamentally harder than querying for supporting ones. Negation does not work in vector space — a query encoding "NOT bearing wear" does not retrieve documents arguing against bearing wear; it retrieves documents dissimilar to bearing wear, which is a different and less useful result. Contradicting evidence is more reliably found by retrieving documents about alternative hypotheses (what else could explain this event?) rather than directly querying for negations. Stage F's scoring currently relies on `contradicting_count` from Stage E, which may be unreliable for this reason.
- **No evidence retrieval for `out_of_boundary_anomalies`**: components flagged at Stage B as anomalous but outside the structural neighborhood get no query plan at Stage E. The analyst receives the flag but no supporting material from the Chroma collection — weakening the signal value of the flag.
- **Authority scoring requires `source_tier` to be set at Stage 5B**: the authority weighting (`plant_instance` > `plant_procedure` > `plant_fmea` > `plant_family` > `oe_iris` > `oe_adams`) depends on `source_tier` being embedded as Chroma metadata at Stage 5B. If Stage 5B does not set this tag — which is currently the case (schema update pending) — all snippets receive equal authority weight regardless of whether they are mandatory plant procedures or unverified OE internet results. This collapses a critical quality dimension.
- **No cross-candidate evidence deduplication**: if the same document snippet is relevant to multiple candidates, it is retrieved and counted independently for each. A mandatory procedure that rules out candidate A while supporting candidate B should ideally be shared and compared across both. The current design counts it as separate evidence for each candidate, potentially double-counting the same source.

---

## Stage F — Candidate Refinement (post-evidence)

**Class / entry-point**: `RuleBasedCausalityEngineV31.refine_with_evidence()`
**File**: `orchestrators/causality_engine_v31.py`

### What this stage does

Stage F closes the loop between the candidate hypotheses generated at Stage D and the actual evidence retrieved at Stage E. Stage D scored candidates using a KG document metadata proxy for the evidence dimension — a rough estimate of how well-documented a failure mode is. Stage E retrieved real document content. Stage F replaces the proxy with the real thing and re-ranks accordingly.

For each candidate, Stage F updates the evidence sub-score using the supporting count, contradicting count, and best support score from the `evidence_bundle`. The updated score feeds back into the same 5-dimensional weighted composite, producing a revised ranking — `causality_candidates v2`. Candidates whose rank improves after the evidence update were undervalued by the KG proxy at Stage D; candidates whose rank drops were over-valued. This v1→v2 ranking delta is the primary diagnostic signal of the pipeline: if ranks do not change materially, the evidence corpus assembled at Stage 5B is not discriminating between hypotheses, which itself is a finding worth surfacing to the analyst.

Stage F also classifies each candidate's evidence posture — `supported`, `contested`, `neutral`, or `missing` — and flags near-ties where the composite score gap between adjacent candidates falls within the `review_alternative_gap` threshold. These flags are the pipeline's way of telling the analyst: "the scoring cannot confidently choose between these two hypotheses; human judgement is required."

What Stage F does not do is go back and retrieve more evidence. The pipeline is strictly sequential — Stage E runs once, Stage F refines once. If contradicting evidence drops a previously top-ranked candidate, the pipeline cannot automatically seek additional targeted documentation to resolve the conflict. That follow-up is left to the analyst.

### Input artifacts

| Input | Key fields consumed |
|-------|---------------------|
| `causality_candidates v1` | All candidates with v1 scores |
| `evidence_bundle` | Per-candidate `supporting_count`, `contradicting_count`, `best_support_score`, `best_contradiction_score` |
| `signal_evidence` | `per_candidate_chain_score` — optional; when present, contributes `chain_evidence_score` term to the evidence sub-score update |

### Key logic

- Updates the evidence sub-score for each candidate by combining two independent evidence streams: document evidence from `evidence_bundle` (Chroma retrieval) and chain evidence from `signal_evidence` (propagation chain position). When `signal_evidence` is absent or empty, the chain term is zero and the formula reduces to the document-only behavior from Sprint 7.
- Evidence sub-score formula: `E_new = clamp01(0.70 × E_doc + 0.30 × chain_score)`. Weights are initial estimates flagged for iteration.
- Applies **evidence posture** classification: `supported | contested | neutral | missing`. Posture is derived from document evidence only (supporting/contradicting counts); chain evidence does not contribute to posture classification.
- Flags candidates for analyst review when primary and top-alternative are within `review_alternative_gap = 0.10` of each other.
- Recomputes composite score with updated evidence sub-score; re-ranks the candidate list.
- The **v1 → v2 ranking delta** is the primary diagnostic signal: candidates that move up in rank after evidence retrieval were undervalued by the KG proxy; those that drop were over-valued. Large rank inversions driven primarily by `chain_score` (chain root candidates jumping in rank) indicate that the propagation chain is adding discriminating signal beyond document evidence alone.

### Output artifact — `causality_candidates v2`

Adds to v1 fields:

| New field | Content |
|-----------|---------|
| `evidence_posture` | `supported \| contested \| neutral \| missing` (document evidence only) |
| `v1_rank`, `v2_rank` | Rank position before and after evidence update |
| `rank_delta` | `v1_rank − v2_rank` (positive = moved up) |
| `review_required` | bool — set when rank gap to next candidate ≤ 0.10 |
| `scores.evidence_doc` | Document-only evidence sub-score (E_doc) — for diagnostics |
| `scores.evidence_chain` | Chain-only evidence sub-score (chain_score) — for diagnostics; 0.0 when signal_evidence absent |

### Critical path pseudo-code

```python
def refine_with_evidence(candidates_v1, evidence_bundle, signal_evidence=None):
    ev_index = evidence_bundle["per_candidate_summary"]

    # Build chain score index from signal_evidence (empty dict when absent)
    signal_ev_index = {}
    if signal_evidence:
        for fm_id, entry in signal_evidence.get("per_candidate_chain_score", {}).items():
            signal_ev_index[fm_id] = entry

    refined = []

    for i, cand in enumerate(candidates_v1["candidates"]):
        ev = ev_index.get(cand["candidate_id"], {})

        # Evidence score update formula
        best_support = ev.get("best_support_score", 0.0)
        n_support    = ev.get("supporting_count", 0)
        n_contra     = ev.get("contradicting_count", 0)

        # Document evidence sub-score (from evidence_bundle)
        E_doc = clamp01(
            0.40 * best_support
            + 0.30 * min(1.0, n_support / 3.0)   # saturates at 3+ supporting snippets
            - 0.20 * min(1.0, n_contra / 2.0)     # penalty for contradicting evidence
        )

        # Chain evidence sub-score (from signal_evidence — optional)
        # When signal_evidence is absent, chain_score = 0.0 and E_new == E_doc
        chain_score = (signal_ev_index.get(cand["candidate_id"], {})
                       .get("chain_position_score", 0.0))

        E_new = clamp01(
            0.70 * E_doc          # document evidence (dominant term)
            + 0.30 * chain_score  # propagation chain evidence
            # NOTE: weights (0.70 / 0.30) are initial estimates — flagged for
            # near-term iteration once calibration data is available
        )

        composite_new = (
            0.30 * cand["scores"]["structural"]
            + 0.20 * cand["scores"]["temporal"]
            + 0.20 * cand["scores"]["telemetry"]
            + 0.20 * E_new        # now combines document + chain evidence
            + 0.10 * cand["scores"]["governance"]
        )

        posture = (
            "supported"  if n_support > 0 and n_contra == 0 else
            "contested"  if n_support > 0 and n_contra > 0  else
            "neutral"    if n_support == 0 and n_contra == 0 else
            "missing"
        )

        refined.append({**cand,
            "scores": {**cand["scores"], "evidence": E_new},
            "composite_score": composite_new,
            "evidence_posture": posture,
            "v1_rank": i + 1,
        })

    refined.sort(key=lambda c: -c["composite_score"])
    for j, c in enumerate(refined):
        c["v2_rank"]    = j + 1
        c["rank_delta"] = c["v1_rank"] - c["v2_rank"]

    # Flag near-ties for analyst review
    for j in range(len(refined) - 1):
        gap = refined[j]["composite_score"] - refined[j+1]["composite_score"]
        if gap <= 0.10:
            refined[j]["review_required"]   = True
            refined[j+1]["review_required"] = True

    return {"causality_candidates": refined}
```

### Key thresholds

| Parameter | Value | Effect |
|-----------|-------|--------|
| `review_alternative_gap` | 0.10 | Near-tie threshold that triggers `review_required` |
| Support saturation | 3 snippets | `n_support/3` saturates at 1.0 with ≥3 supporting snippets |
| Contradiction saturation | 2 snippets | `n_contra/2` saturates penalty at 1.0 with ≥2 contra snippets |
| Contradiction penalty weight | 0.20 | Maximum evidence score reduction from contradicting evidence |

### Known gaps

- Evidence score update formula uses `n_support` and `n_contra` counts, but snippet count is a weak proxy — three low-quality snippets outweigh one mandatory procedure citation.
- `review_required` flag does not propagate to `rca_card.analyst_review.questions_to_resolve`; the analyst must check v2 candidates directly.
- The v1→v2 delta is stored per-candidate but not surfaced as a named summary field in the artifact; visualization tools must compute it from `rank_delta`.
- **Evidence update formula ignores authority tier**: the formula uses `best_support_score` (cosine similarity) and `n_support` (raw count) but does not weight by `source_tier`. Three OE internet snippets of medium similarity outweigh one mandatory plant procedure with high similarity. The update formula should incorporate `authority_score` from the evidence_bundle snippets — e.g., weighting `best_support_score` by the authority of its source. This requires `source_tier` to be resolved (see Stage 5B code change 6).
- **Evidence posture classification has a logical gap**: the four-way classification covers `n_support > 0 and n_contra == 0` (supported), `n_support > 0 and n_contra > 0` (contested), `n_support == 0 and n_contra == 0` (neutral), and falls to `missing` for the remaining case. The remaining case is `n_support == 0 and n_contra > 0` — contradicting evidence found with no supporting evidence. Labelling this `missing` is incorrect: evidence was found, it just argues against the hypothesis. This case should be a distinct posture, e.g., `contradicted`, to correctly signal to the analyst that this candidate has active evidence against it.
- **`review_alternative_gap` applied on composite score alone**: the near-tie flag compares composite score differences. Two candidates 0.09 apart could have very different evidence postures — one strongly supported, one contested — yet both trigger `review_required`. Conversely, two candidates with identical evidence postures and scores of 0.51 and 0.40 do not trigger the flag even though neither is well-discriminated. The flag would be more operationally meaningful if it incorporated evidence posture: a near-tie between two `supported` candidates is more critical than a near-tie between two `neutral` ones.
- **No iterative evidence loop**: if Stage F identifies a candidate that drops sharply in rank due to contradicting evidence, there is no mechanism to trigger targeted follow-up retrieval. An RCA engineer in this situation would seek additional documentation to resolve the conflict. The pipeline has no feedback path from Stage F to Stage E — the evidence corpus is fixed after Stage 5B and Stage E.

---

## Stage G — Ishikawa Structuring (optional)

**Class / entry-point**: `HeuristicIshikawaEvaluatorV1.evaluate()`
**File**: `synthesis/ishikawa_evaluator_v1.py`

### What this stage does

Stage G is a structured translation stage. It does not generate new hypotheses or scores — it maps what the pipeline already knows onto a classic Ishikawa (fishbone) causal diagram. The purpose is to ensure that every major causal branch — equipment hardware, process and procedures, measurement and instrumentation, environment, and maintenance/human factors — is at least considered before the RCA card is written. Without this step, Stage H can produce a plausible root cause narrative that inadvertently overlooks an entire causal category simply because the highest-scoring candidates did not happen to cover it.

**Why fishbone categorisation matters for nuclear RCA**

An experienced RCA practitioner reviewing a condenser vacuum loss event would independently ask: could this be equipment hardware (failed seal, worn bearing)? Could it be procedural (wrong valve lineup, missed isolation step)? Could it be instrumentation (false signal driving a false trip)? Could it be an environmental factor (elevated ambient, utility steam quality change)? Could it be a human performance issue on the maintenance crew? The Ishikawa structure ensures that the pipeline asks the same questions systematically, and documents which branches are populated versus empty. An empty branch is as informative as a populated one: it tells the analyst either "no evidence exists for this branch" or "this branch was not investigated."

**How the categorisation works**

Each candidate and evidence snippet from Stage F is assessed against six categories using keyword classification on the cause label, component type, and failure mode description:

- `equipment_hardware`: mechanical failure, material degradation, component wear, corrosion, fatigue
- `process_procedure`: procedure violation, wrong sequence, missed step, inadequate work instruction
- `measurement_instrumentation`: sensor failure, instrument drift, false signal, calibration error
- `environment_operating_context`: elevated temperature/pressure/humidity, utility quality excursion, weather
- `maintenance_human_factors`: PM compliance, maintenance error, incorrect torque, wrong part installed
- (implicit `management_systems`): systemic issues surfaced by multiple `maintenance_human_factors` entries

Each row in a category includes `source_links` — explicit back-references to `candidate_id` or `doc_id` — so every entry in the Ishikawa matrix is traceable to a specific scored hypothesis or retrieved document.

**When Stage G is skipped**

Stage G is optional and is enabled by `OrchestratorConfig.enable_ishikawa`. When it is not run (either because it is disabled or because `ishikawa_evaluator` is not configured), Stage H receives `ishikawa_matrix: null`. As of Sprint 7:

- `run_manifest.pipeline_config.ishikawa_run` is set to `False`
- `run_manifest.pipeline_config.ishikawa_skip_reason` is populated with the reason
- An `analyst_attention_flag` is injected into `rca_card.executive_summary.analyst_attention_flags`: *"Ishikawa structuring was not performed — human performance and organizational factor branches were not systematically evaluated."*

This flag is surfaced to the analyst through the review interface so they know the fishbone structure was absent from the synthesis, and can manually assess the uncovered branches before confirming the RCA card.

### Input artifacts

| Input | Key fields consumed |
|-------|---------------------|
| `kg_context` | `components[]`, `failure_modes[]`, `safety_functions[]` |
| `tskr_patterns` | Top patterns — temporal evidence for categorisation |
| `causality_candidates v2` | All scored candidates with `cause_label`, `fm_id`, `evidence_posture` |
| `evidence_bundle` | Snippets with `support_role`, `authority_score` |
| `operational_context` | `operating_point`, `recent_alarms` — environmental and procedural context |
| `pm_compliance` | `pm_checks[]` — maintenance compliance inputs |

### Key logic

- Translation stage only — no new inference or re-scoring. All scores are fixed at Stage F.
- Six standard categories: `equipment_hardware`, `process_procedure`, `measurement_instrumentation`, `environment_operating_context`, `maintenance_human_factors`, and implicit `management_systems`.
- Categorises each top candidate by keyword matching on `cause_label` + `fm_description` + `component_type`. One candidate can appear in multiple categories if its description spans categories.
- Each populated row includes `source_links[]` back to `candidate_id` or `doc_id` for traceability.
- Empty categories are explicitly represented (empty list `[]`) — the absence of entries is meaningful.
- If Stage G is skipped, Stage H receives `ishikawa_matrix: null` and proceeds with no structural fishbone input; `analyst_attention_flag` is injected by the orchestrator.

### Output artifact — `ishikawa_matrix`

| Field | Content |
|-------|---------|
| `equipment_hardware[]` | Candidates / evidence categorized as hardware causes |
| `process_procedure[]` | Procedure-related candidates or evidence snippets |
| `measurement_instrumentation[]` | Sensor/instrument-related items |
| `environment_operating_context[]` | External / environmental contributors |
| `maintenance_human_factors[]` | PM compliance, human error items |
| Each row: `source_links[]` | `{type: "candidate"|"doc", id: str}` — explicit traceability |
| `populated_categories[]` | List of category names with at least one entry |
| `empty_categories[]` | Categories with no entries — analyst review signal |

### Critical path pseudo-code

```python
def evaluate(event, telemetry, kg_context, tskr_patterns, candidates_v2,
             evidence_bundle, oc, pm, run_context):
    categories = {
        "equipment_hardware": [],
        "process_procedure": [],
        "measurement_instrumentation": [],
        "environment_operating_context": [],
        "maintenance_human_factors": [],
    }
    top_candidates = candidates_v2["candidates"][:5]
    top_snippets   = evidence_bundle["snippets"][:10]

    for cand in top_candidates:
        text = f"{cand.get('cause_label','')} {cand.get('fm_description','')} {cand.get('component_type','')}"
        matched = classify_to_categories(text, CATEGORY_KEYWORD_MAP)
        for cat in matched:
            categories[cat].append({
                "type": "candidate",
                "id": cand["candidate_id"],
                "label": cand["cause_label"],
                "composite_score": cand["composite_score"],
                "evidence_posture": cand.get("evidence_posture"),
                "source_links": [{"type": "candidate", "id": cand["candidate_id"]}],
            })

    for snippet in top_snippets:
        text = snippet.get("snippet", "")
        matched = classify_to_categories(text, CATEGORY_KEYWORD_MAP)
        for cat in matched:
            categories[cat].append({
                "type": "doc",
                "id": snippet["doc_id"],
                "snippet_id": snippet["snippet_id"],
                "support_role": snippet["support_role"],
                "source_links": [{"type": "doc", "id": snippet["doc_id"]}],
            })

    populated = [k for k, v in categories.items() if v]
    empty     = [k for k, v in categories.items() if not v]

    return {**categories,
            "populated_categories": populated,
            "empty_categories": empty,
            "run_id": run_context["run_id"]}
```

### Key thresholds

| Parameter | Value | Effect |
|-----------|-------|--------|
| `enable_ishikawa` | `OrchestratorConfig` flag | Determines whether Stage G runs at all |
| Max candidates to categorise | top-5 by composite | Same selection as Stage H prompt |
| Max snippets to categorise | top-10 by score | Same selection as Stage H prompt |

### Known gaps

- **§9.5 ✅ FIXED (Sprint 7)**: When Stage G is skipped, the run manifest now records `ishikawa_run: False` and `ishikawa_skip_reason`, and an `analyst_attention_flag` is injected into the rca_card. The analyst can no longer receive a completed RCA card without knowing whether the fishbone structure was produced.
- Categorisation is heuristic (keyword classification against cause_label and FM description); there is no validation that keyword-driven assignments are semantically correct. A bearing failure described as "thermal degradation" may not be classified under `equipment_hardware`.
- The six-category Ishikawa schema (4M+E variant) differs from INPO AP-913's root cause taxonomy (latent organizational weaknesses, direct cause, contributing cause, root cause). Ishikawa output cannot be directly compared to an AP-913-formatted RCA report without relabeling.
- Empty categories are recorded but not individually surfaced as `analyst_attention_flags`; an analyst who does not manually check `empty_categories[]` may not realise that the maintenance/human factors branch was completely uninvestigated.
- Stage G does not have access to the `out_of_boundary_anomalies` list from Stage B (proposed feature). Components flagged as anomalous but outside the KG neighborhood could be highly relevant to the human performance or environmental branch but are not represented in the fishbone.

---

## Stage H — RCA Synthesis

**Class / entry-point**: `RuleValidatedRCASynthesizerV31.synthesize()`
**File**: `synthesis/rca_synthesizer_v31.py`

### What this stage does

Stage H is where the pipeline's outputs converge into a single deliverable: the RCA card. Everything produced upstream — the KG neighborhood, the temporal patterns, the scored and ranked candidates, the retrieved evidence, the recurrence history, the Ishikawa structure if available — is assembled into a structured narrative that an analyst can read, validate, challenge, and act on.

The synthesis has two paths. The intended path uses an LLM to write the executive summary, primary hypothesis narrative, alternative hypotheses, evidence linkages, and recommended actions. The LLM receives a structured prompt containing the top-5 candidates and top-10 evidence snippets, along with enough context to reason about recurrence, PM compliance, and operating state. Its output is validated against a JSON schema and a hallucination check — the primary hypothesis must reference a candidate that actually exists in the input. If the LLM output passes, it becomes the RCA card.

The production path, however, is always the deterministic fallback. The LLM synthesis path (`OllamaLLMClient`) exists in the codebase but has not been validated end-to-end. `_fallback_card()` selects the highest-scoring v2 candidate as the primary hypothesis, fills the required fields with deterministic templates, and always caps confidence at "medium." This is the path every real run currently takes.

Stage H is also where the pipeline communicates uncertainty back to the analyst. The `analyst_review` field in the `rca_card` is populated with `decision_required`, `questions_to_resolve`, and `writeback_recommendation`. This is the mechanism by which the pipeline co-pilots rather than automates: it presents its best assessment, flags what it cannot resolve, and defers the final judgement to the human.

### Input artifacts

| Input | Key fields consumed |
|-------|---------------------|
| `kg_context` | Full context (components, safety_functions, documents) |
| `tskr_patterns` | Top patterns with confidence, support, recurrence |
| `causality_candidates v2` | Full scored and ranked list |
| `evidence_bundle` | Snippets with support roles and authority scores |
| `operational_context` | `recent_alarms`, `operating_point`, `nearby_maintenance` |
| `pm_compliance` | `pm_checks[]` |
| `ishikawa_matrix` | Optional; used if produced by Stage G |
| `cmms_context` | Optional; `recurrence_summary`, `cr_records[]` used for recurrence reasoning |

### Key logic

- Selects top-5 candidates and top-10 evidence items to fit within LLM token budget.
- Constructs a structured prompt with JSON schema constraints; the LLM must produce `executive_summary`, `primary_hypothesis`, `alternatives`, `evidence`, `recommended_actions`, and `analyst_review`.
- **Primary validation gate**: `primary_hypothesis.candidate_id` must match an input candidate (hallucination check).
- **All-claims-cited check**: every claim in the narrative must cite a source via `citations[]`.
- If LLM output fails validation and `allow_fallback_template_fill: True`, runs `_fallback_card()` using the highest-scoring v2 candidate — deterministic, confidence always capped at `"medium"`.
- **Production path is always the fallback**: `OllamaLLMClient` path exists but is not validated end-to-end.

### Output artifact — `rca_card`

| Field | Content |
|-------|---------|
| `rca_id` | `"RCA::{event_id}::{run_id}"` |
| `executive_summary` | `decision_status`, `primary_conclusion`, `confidence_label`, `analyst_attention_flags[]` |
| `primary_hypothesis` | `candidate_id`, `cause_label`, `narrative`, `why_primary[]`, `uncertainties[]`, `citations[]`, `composite_score`, `confidence_label` |
| `alternatives[]` | Other candidates with `reason_not_primary`, `supports[]`, `weaknesses[]`, `citations[]` |
| `evidence[]` | Evidence items with `source_type`, `authority_level`, `support_role`, `summary`, `excerpt` |
| `recommended_actions[]` | Each with `action_type`, `priority`, `target_component_id`, `rationale` |
| `analyst_review` | `decision_required`, `questions_to_resolve[]`, `writeback_recommendation` |
| `validation_status` | `schema_valid`, `all_claims_cited`, `passed_minimum_evidence_gate` |

### Critical path pseudo-code

```python
def synthesize(event, telemetry, kg_context, tskr, candidates_v2, evidence,
               oc, pm, ishikawa, run_context, cmms_context=None):
    # 1. Select top inputs for prompt
    top_candidates = sorted(candidates_v2["candidates"],
                            key=lambda c: -c["composite_score"])[:5]
    top_evidence   = sorted(evidence["snippets"],
                            key=lambda s: -s["score"])[:10]

    # 2. Build prompt
    prompt = build_rca_prompt(
        event, top_candidates, top_evidence, kg_context,
        tskr, oc, pm, ishikawa, cmms_context,
        schema=RCA_CARD_JSON_SCHEMA
    )

    # 3. LLM attempt
    try:
        raw = llm_client.generate_json(prompt, temperature=0.1)
        errors = validate_rca_card(raw, valid_candidate_ids={c["candidate_id"]
                                                             for c in top_candidates})
        if errors:
            raise ValidationError(errors)
        rca_card = raw

    except (LLMError, ValidationError):
        if not config.allow_fallback_template_fill:
            raise
        rca_card = _fallback_card(top_candidates[0], top_evidence, event, run_context)
        rca_card["provenance"]["fallback_used"] = True

    # 4. Finalize
    rca_card["rca_id"]   = f"RCA::{event['event_id']}::{run_context['run_id']}"
    rca_card["validation_status"] = {
        "schema_valid":               validate_json_schema(rca_card),
        "all_claims_cited":           check_all_claims_cited(rca_card),
        "passed_minimum_evidence_gate": len(rca_card["evidence"]) >= 2,
    }
    return rca_card
```

### Key thresholds

| Parameter | Value | Effect |
|-----------|-------|--------|
| `max_candidates_in_prompt` | 5 | Token budget cap |
| `max_evidence_in_prompt` | 10 | Token budget cap |
| `temperature` | 0.1 | Near-deterministic LLM output |
| `minimum_primary_score` | 0.35 | Readiness threshold for `decision_status: candidate_ready` |
| Minimum evidence gate | 2 snippets | `passed_minimum_evidence_gate` requires ≥2 evidence items |

### Known gaps

- **C3 (April 20 review)**: Fallback card always caps confidence at `"medium"`. There is no path to `"high"` confidence output in the current production configuration.
- **C5 (April 20 review)**: `kg_context.safety_functions[]` is available in the prompt context but the schema and prompt do not require the LLM to map recommended actions against safety function impact. A recommended action can have `priority: low` on a component that supports a safety function.
- **C2 (April 20 review)**: Single primary hypothesis architecture — Stage H outputs exactly one `primary_hypothesis`. Scenarios with two co-equal root causes (common cause failure, coupled degradation) are not representable.
- **C4 (April 20 review)**: Recommended action `priority` is LLM-inferred from cause severity; it is not cross-referenced to safety significance, corrective action program (CAP) priority codes, or regulatory significance.
- `rca_card.analyst_review.questions_to_resolve[]` is populated by the LLM or fallback, but Stage F's `review_required` flag is not automatically injected into this list.
- **Top-5 candidate truncation ignores `review_required` flags**: candidates are selected for the prompt by composite score descending. If a near-tie pair identified at Stage F straddles the top-5 cut — e.g., rank 5 and rank 6 both have `review_required: true` — the second candidate is dropped from the prompt entirely. The LLM or fallback then reasons over an incomplete hypothesis set, and the analyst never sees the near-tie in the RCA card. Top-5 selection should ensure all `review_required` candidates are included, even if the prompt cap must be raised or a different candidate displaced.
- **Top-10 evidence selected by cosine similarity, not by authority tier**: the prompt selects the 10 highest-similarity snippets globally. A mandatory plant procedure with slightly lower cosine similarity than an OE internet snippet is excluded. For nuclear RCA, authority rank should govern evidence selection first, similarity second. A lower-similarity snippet from a plant FMEA is more authoritative than a higher-similarity snippet from an OE database.
- **Evidence not balanced across candidates**: the top-10 selection is global — all snippets compete regardless of which candidate they support. In practice, the top candidate's evidence may fill all 10 slots, leaving the LLM with nothing about the second or third candidate. The prompt should guarantee a minimum number of evidence snippets per candidate (e.g., at least 2 per top-5 candidate) before filling remaining slots by authority/similarity.
- **`minimum_evidence_gate` counts all snippets including contextual and contradicting**: `passed_minimum_evidence_gate` requires `len(rca_card["evidence"]) >= 2`. A card with 2 contradicting snippets and 0 supporting snippets passes the gate. The check should require at least 2 snippets with `support_role = "supporting"` to be meaningful as an evidence quality gate.
- **`out_of_boundary_anomalies` absent from prompt**: Stage B's proposed `out_of_boundary_anomalies` field — components with preceding anomalies outside the KG neighborhood, including those not in the KG at all — is not included in the Stage H prompt. The analyst receives an RCA card with no visibility of potential causal signals that the pipeline explicitly excluded from its search space. At minimum, the `analyst_attention_flags[]` in `executive_summary` should list any `out_of_boundary_anomalies` with `not_in_kg: true`.

---

## Stage I — Artifact Persistence

**Class / entry-point**: `FileArtifactStore.save()`
**File**: `storage/file_artifact_store.py`

### What this stage does

Stage I is the pipeline's write-out step. By the time it runs, all analytical work is complete — the RCA card exists, the candidates are ranked, the evidence is linked. Stage I's only job is to durably record everything so the run is reproducible, auditable, and accessible to Stage J for validation.

Every artifact produced by the pipeline is written to a dedicated run directory keyed by `run_id`: `{output_dir}/{run_id}/`. Each artifact is a separate JSON file. Alongside each artifact, Stage I writes a validation sidecar (`*__validation.json`) that captures any schema warnings or cross-artifact issues detected during the run. The run-scoped Chroma collection — assembled at Stage 5B and queried at Stage E — is archived to `{output_dir}/{run_id}/chroma/`. After archiving, the Chroma collection is no longer live; it becomes a static record that can be replayed if the run is re-examined.

This archival design means every RCA run is fully self-contained on disk: the JSON artifacts, the Chroma evidence corpus, and the validation sidecars together constitute the complete audit record. No external system needs to be queried to reproduce or review what the pipeline concluded and why. This property is non-negotiable in a nuclear plant RCA context where regulatory traceability and long-term record retention are requirements.

Stage I performs no transformation. If it receives a corrupted or incomplete artifact, it writes what it was given. Detection of incomplete or invalid runs is the responsibility of Stage J.

### Key logic

- Writes all run artifacts to `{output_dir}/{run_id}/` as individual JSON files.
- Also writes per-artifact validation sidecar files (`*__validation.json`).
- Archives the run-scoped Chroma collection to `{output_dir}/{run_id}/chroma/`.
- No transformation — pure persistence.

### Known gaps

- No atomic write (no temp-then-rename); a crash mid-write leaves a partial run directory that Stage J may treat as valid.
- No compression or retention policy for the Chroma archive.
- **No "run complete" marker**: artifacts are written one by one with no sentinel file written as the final act. Stage J cannot distinguish a fully written run from an interrupted one — a partial run directory produces a `run_manifest` that reflects only what was written, silently omitting the rest. A `run_complete.json` or equivalent marker should be the last write of Stage I; Stage J should refuse to validate any run directory that lacks it.
- **Chroma archive failure breaks the audit trail**: JSON artifacts and the Chroma collection are written independently. If Stage I writes all JSON files successfully but then fails during the Chroma archive step — storage quota, network filesystem timeout — the `rca_card` will cite evidence snippets (`doc_id`, `snippet_id`) that no longer exist anywhere on disk. For nuclear plant RCA, where regulatory traceability requires every cited piece of evidence to be recoverable, this is a critical integrity failure. A `chroma_archived: bool` flag should be written to the run manifest so Stage J can detect and reject incomplete runs.
- **Validation sidecar role is ambiguous relative to Stage J**: Stage I writes `*__validation.json` sidecars alongside each artifact; Stage J runs `RCAArtifactValidator.validate_run_bundle()` over the same artifacts. If both use the same validator, validation runs twice with no additional value. If they use different validators, results may be inconsistent. The relationship between sidecar generation and Stage J validation must be explicitly defined — sidecars should either be the input Stage J reads (avoiding re-validation) or be deprecated in favour of a single authoritative validation pass at Stage J.
- **No pre-write completeness check**: Stage I writes what it receives without verifying that all nine expected artifacts are present and non-null. A missing or empty `rca_card` caused by a fallback edge case is written to disk without complaint and forwarded to Stage J as if complete. Stage I should verify artifact presence and non-emptiness before writing, and abort with a clear error rather than producing a silent gap.
- **`output_dir` writability not checked at pipeline entry**: if `output_dir` is non-existent or read-only, Stage I fails after all of Stages A–H have successfully executed and their results are lost. The writability of `output_dir` should be verified at Stage A as a pre-flight check — it is a precondition for the entire run, not just for Stage I.

---

## Stage J — Validation & Run Manifest

**Class / entry-point**: `RCAArtifactValidator.validate_run_bundle()`
**File**: `validation/rca_artifact_validator.py`

### What this stage does

Stage J is the pipeline's terminal quality gate. After Stage I has written all artifacts to disk, Stage J reads the run directory and certifies whether the completed run is fit for purpose — fit to be written back to the CMMS corrective action program, fit to be reviewed by an analyst, or in need of remediation before either can happen.

It does this in two validation layers. Layer 1 is schema validation: every artifact is checked against its JSON schema definition, confirming that required fields are present and correctly typed. Layer 2 is cross-artifact consistency: it checks that references between artifacts resolve — for example, that the `candidate_id` cited in `rca_card.primary_hypothesis` actually exists in `causality_candidates`, and that every `doc_id` in `evidence_bundle` appears in `kg_context.documents`. The result of both layers feeds into a single terminal artifact — the `run_manifest` — which classifies the run as `writeback`, `analyst_review`, or `remediation` and records the per-artifact validation detail.

**The fundamental architectural limitation of Stage J is that all validation is late-binding.** It runs after every other stage has completed. If `kg_context` has a schema error, that error has propagated silently through Stages 5B, C, D, E, F, G, and H before Stage J detects it. Each of those stages has transformed or extended the invalid data, making the original error harder to trace and the downstream artifacts potentially unreliable. This is the wrong validation model for a pipeline that is intended to co-pilot safety-significant decisions.

The correct model is progressive validation: each stage validates its own output schema before passing it to the next. Stage J retains its role for cross-artifact consistency checks — those cannot be moved earlier because both sides of the reference must exist before the check can run. But per-artifact schema validation belongs at the stage that produces the artifact, not at the end of the pipeline. This is consistent with what was already noted for Stage A (input validation should include JSON schema checks, not just key-presence checks).

### Key logic

- **Layer 1 (JSON schema)**: validates every artifact against its schema definition; records per-artifact pass/fail.
- **Layer 2 (cross-artifact)**: checks consistency across artifacts, e.g., `rca_card.primary_hypothesis.candidate_id` ∈ `causality_candidates.candidates[].candidate_id`; all `doc_id` references in `evidence_bundle` exist in `kg_context.documents`.
- Determines `writeback_ready` (all validation passes + no review flags), `requires_human_review` (any review flag set), and `next_step` (`writeback | analyst_review | remediation`).
- Produces `run_manifest` as the terminal artifact.

### Output artifact — `run_manifest`

| Field | Content |
|-------|---------|
| `schema_valid` | bool — all artifacts pass Layer 1 |
| `all_claims_cited` | bool — from `rca_card.validation_status` |
| `writeback_ready` | bool — safe to push to CMMS CAP |
| `requires_human_review` | bool |
| `next_step` | `"writeback" \| "analyst_review" \| "remediation"` |
| `per_artifact_validation` | Dict with pass/fail per artifact |

### Known gaps

- `writeback_ready: true` does not check whether recommended actions already exist as open WOs in CMMS — duplicates would be created on CAP writeback.
- Layer 2 cross-checks do not verify that `rca_card.recommended_actions[].target_component_id` values are valid `element_usage` IDs in the KG.
- All schema validation is late-binding: Layer 1 schema checks run at Stage J, after every upstream stage has already consumed the artifacts being validated. A schema error in `kg_context` is not caught until after Stages 5B, C, D, E, F, G, and H have all executed on invalid data. The correct model is progressive validation — each stage validates its own output schema before passing it downstream; Stage J retains only cross-artifact consistency checks.
- `next_step: writeback | analyst_review | remediation` is computed and written to the manifest but has no downstream routing implementation beyond the analyst manually reading the manifest. A co-pilot deployment requires the `analyst_review` and `writeback` paths to have active handoffs (notification, queue dispatch, CAP integration).
- No re-run capability for `remediation` runs — only full re-runs from Stage A are supported.

**What is already in the run manifest** (as of Sprint 7):
- `review_hooks.writeback_ready`, `requires_human_review`, `next_step`, `degraded_reasons`
- `review_hooks.passed_severity_gate`, `review_hooks.severity_floor` — severity-adjusted floor enforcement
- `pipeline_config.ishikawa_run`, `pipeline_config.ishikawa_skip_reason` — Stage G skip tracking
- `pipeline_config.enable_ishikawa`, `pipeline_config.reentry_execution` — pipeline configuration record
- `stage_health` — per-stage green/yellow/red status with issue lists
- `pipeline_health` — overall health aggregated from stage health + validation
- `kg_governance` — failure mode coverage and KG snapshot age
- `barrier_analysis` — safety function barrier status (degraded/intact)
- `ap913_completeness` — INPO AP-913 checklist booleans
- `artifacts` — presence/count inventory for all major artifacts
- `primary_candidate_summary` — posture, evidence counts, scoring for primary hypothesis

---

## Scoring Weight Summary

| Stage | Dimension | Weight | Baseline | Range |
|-------|-----------|--------|----------|-------|
| D/F | Structural | 0.30 | 0.40–0.90 | by topology |
| D/F | Temporal | 0.20 | 0.0 | by TSKR |
| D/F | Telemetry | 0.20 | 0.35 | by anomaly count |
| D/F | Evidence | 0.20 | 0.30 (v1 proxy) | 0.0–1.0 (v2 actual) |
| D/F | Governance | 0.10 | 0.50 | 0.50–0.95 |
| C | TSKR confidence | — | Anomaly=0.55, Latency=0.30, History=0.15 | — |
| C | TSKR support | — | History=0.35, Telemetry=0.35 | — |

---

## Open Issues — Cross-Stage

| Issue | Stage | Ref | Status |
|-------|-------|-----|--------|
| Safety significance not a scoring dimension | D, F, H | April 20 C4 | ⚠️ Partial — safety function now propagates to rca_card flags and action priority; not a scoring dimension |
| Confidence always capped at "medium" (fallback path) | H | April 20 C3 | Open |
| Single primary hypothesis — no co-equal causes | H | April 20 C2 | Open |
| KG closed-world assumption | B | April 20 C1 | Open — by design |
| `instrument_validity_flag` not consumed | C | April 20 H3 | Open |
| `review_required` not injected into analyst_review | F, H | — | Open |
| Evidence score uses snippet count, not authority | F | — | Open |
| EDMS / FMEA ingestion stubs | 5B | — | Open |
| OE LLM tier | 5B | — | Future |
| Severity-adjusted evidence floor | A, J | April 21 A2 | ✅ Fixed Sprint 6 — `_SEVERITY_SCORE_FLOORS` in synthesizer; `passed_severity_gate` in review_hooks |
| FM-category governance weight undifferentiated | D, F | April 21 NM3 | ✅ Fixed Sprint 6 — `_governance_weight_for_fm`; maintenance-preventable FMs weight 0.20, external 0.02 |
| Stage G skip not surfaced to analyst | G, J | April 21 §9.5 | ✅ Fixed Sprint 7 — `ishikawa_run`/`ishikawa_skip_reason` in manifest; `analyst_attention_flag` injected |
| ECA/RCA doc types subject to ±90-day recency window | B | April 21 B2 | ✅ Fixed Sprint 7 — ECA/RCA exempt from date filter; recency bonus removed for timeless doc types |
| Contributing causes not in rca_card | H | April 21 NC6 | ✅ Fixed Post-Apr21 batch — `contributing_causes[]` in rca_card schema and synthesis |
| AP-913 completeness not tracked | J | April 21 NM15 | ✅ Fixed Post-Apr21 batch — `ap913_completeness` block in run_manifest |

---

# Part 3 — SE Review (April 22, 2026)

**Reviewer role**: Systems engineer responsible for plant RCA
**Scope**: workflow logic, input/output data requirements, stage assumptions, real-scenario stress tests

---

## 3.1 Overall Workflow Logic Assessment

### What works

The stage sequencing is sound. Separating KG population (Stage 0) from per-event execution keeps the KG build cost off the RCA critical path. The two-pass scoring architecture (Stage D pre-evidence → Stage F post-evidence) is a correct design pattern; the v1→v2 rank delta as a diagnostic signal is the right idea. The A/B-series tiering at Stage D properly solves the silent-drop problem without eliminating the ability to bound compute cost. The Stage B.5 propagation chain concept is well-motivated: a sensor A anomaly → sensor B anomaly → event path that is also topology-aligned is strong causal evidence, and keeping chain evidence (`E_chain`) separated from document evidence (`E_doc`) at Stage F preserves clean provenance.

### Internal document inconsistencies

These are errors in the document itself that will cause incorrect implementations:

**IC-1 — Section 1.5 "Hard filters" not updated after Stage D A/B-series change.**
Section 1.5 reads: *"Stage D enforces composite ≥ 0.30 AND evidence ≥ 0.10."* Stage D now uses A-series (composite ≥ 0.45 + evidence ≥ 0.35) and B-series (composite ≥ 0.25). Section 1.5 contradicts the implementation section.

**IC-2 — Stage C pseudo-code uses old weights that contradict the key logic text.**
Key logic text (updated): `0.45×anomaly + 0.30×latency + 0.10×chain + 0.10×history + 0.15×count + 0.10×lag`. Pseudo-code (not updated): `0.55×anomaly + 0.30×latency + 0.15×history + 0.20×count + 0.15×lag`. Neither version sums to 1.0; `clamp01` masks the issue but weights are non-convex in both. An implementer following the pseudo-code will build something different from what the text describes.

**IC-3 — `latency_violation_type` enum mismatch.**
The `tskr_patterns` output artifact table says `none | too_fast | too_slow | unknown`. The key logic text and `RCA_FMEA_handling_spec.md` §5 say it should be `not_available` (replacing `unknown`).

**IC-4 — Stage C pseudo-code signature missing `cmms_context` and `signal_evidence`.**
Function signature: `def score(event, telemetry, kg_context, oc, run_context)`. The key logic section states Stage C consumes `cmms_context.cr_records[]` for recurrence and `signal_evidence.augmented_anomaly_set[]` as the primary anomaly source. Both are missing from the signature; the body references `cmms_context["cr_records"]` which would be a `NameError` at runtime.

**IC-5 — Stage D known gaps reference the old hard-threshold logic.**
Two known-gap bullets still describe the old `E ≥ 0.35` hard filter as if it drops candidates entirely. With A/B-series tiering, zero-evidence candidates with composite ≥ 0.25 land in B-series and are not dropped. These gaps need rewriting.

**IC-6 — Stage F pseudo-code references `signal_ev_index` which is never defined.**
`chain_score = signal_ev_index.get(...)` — `signal_ev_index` does not appear in the function signature or body of `refine_with_evidence()`. The `signal_evidence` input is listed in the input table but never loaded in the pseudo-code.

**IC-7 — `writeback_ready` set at Stage D before evidence exists.**
Stage D pseudo-code sets `c["writeback_ready"] = (series == "A") and not c.get("evidence_gap", False)`. This is set using the KG proxy evidence gap before Chroma retrieval. Stage F re-ranks candidates after evidence update. The `writeback_ready` field in v1 is misleading — it implies readiness for writeback before any evidence has been assessed. This field should only be finalized at Stage J after all gates pass. If it must exist at Stage D it should be named `writeback_eligible_v1` and explicitly flagged as preliminary.

---

## 3.2 Input/Output Data Requirements

### Inputs with no current source

| Required input | Consuming stage | Status |
|----------------|----------------|--------|
| `mbse_model.json` | Stage 0 | No MBSE ingestor exists; exchange format spec incomplete |
| `sensor_component_map.csv` | Stage 0 | No ingestion path exists |
| FMEA `expected_latency_min/max_hours` | Stage 0 / Stage C | Non-standard field; absent from all standard FMEA formats; requires enrichment workflow |
| `HistorianAdapter.get_anomalies()` | Stage B.5 | Interface not defined; no adapter written |
| `is_upstream(comp_a, comp_b)` | Stage B.5 | KG query not implemented; edge directionality not guaranteed by Stage 0 ingest |
| EDMS document content | Stage 5B | Stub only |
| FMEA document text | Stage 5B | Stub only |
| Class-level CMMS records | Stage 5B | Not implemented |
| `source_tier` metadata | Stage 5B → Stage E | Not set; all documents treated as equal authority |
| `OllamaLLMClient` output | Stage H | Exists but not validated end-to-end; production always uses deterministic fallback |

**Critical observation**: the two highest-value pipeline features — Stage B.5 propagation chains and Stage H LLM narrative — both depend entirely on unimplemented infrastructure. The current production pipeline is: static KG context + CMMS metadata + deterministic candidate scoring + template-fill RCA card.

### Missing artifact production definitions

**`telemetry_summary.json` format is unspecified.** This is a primary pipeline input but there is no schema definition, no ingestion path, and no adapter. The document states the pipeline assumes it *"arrives pre-formatted."* In a nuclear plant there is no pre-formatted telemetry — there is a PI historian, an alarm management system, and a plant process computer. The transformation from raw historian/DCS data to `telemetry_summary.json` is outside the pipeline scope with no documented handoff interface.

**`event.json` production is unspecified.** Who creates this and when? When a plant operator enters an abnormal event in the CMMS CAP, that is the trigger — but the schema fields required (`symptom_signature`, `timestamp_end`, `severity`) may not exist in the CAP entry. The transformation from CMMS CAP entry to `event.json` is a critical integration step with no design documentation.

**`pm_compliance.json` has no documented source.** CMMS contains PM work orders with completion dates. The transformation to `pm_compliance.json` with `pm_checks[]` and `last_pm_date` is not described.

**`operational_context.json` has no documented source.** This carries `operating_point` (power level, pressures, temperatures at event time), `recent_alarms`, and `nearby_maintenance` — DCS/SCADA data plus operator log data. No adapter or extraction process is defined.

### Artifact flow synchronization gap

Stage B.5 runs in parallel with Stage 5B after Stage B. Stage C requires both `cmms_context` (from Stage 5B) AND `signal_evidence` (from Stage B.5). The document notes this correctly but does not specify what happens when one finishes before the other, what the orchestrator does when Stage B.5 fails, or whether a timeout exists. The parallel execution model requires an explicit join/synchronization point that is currently absent from both the architecture diagram and the pseudo-code.

---

## 3.3 Stage-Level Assumptions

### Stage 0

| Assumption | Risk |
|------------|------|
| MBSE model accurately represents current plant configuration | Plants modify systems during outages; MBSE models may lag 6–18 months behind as-built state |
| FMEA `component_type` vocabulary aligns with MBSE `domain_category` | Both are free text; mismatches silently produce orphaned failure modes |
| CMMS/EDMS APIs are queryable at Stage 0 time | Offline plants or scheduled maintenance windows break KG pre-population |
| FMEA enrichment is complete before KG initialization | Enrichment is human-in-loop; if not done, the KG has failure modes with no latency bounds or anomaly patterns |

### Stage B

| Assumption | Risk |
|------------|------|
| Two-hop neighborhood captures the causal search space | CCF across multiple trains, or failures in support systems (instrument air, DC power), may require more than 2 hops |
| All relevant failure modes have `APPLIES_TO` edges | First-occurrence failures and vendor-defect failures typically have no KG entry |
| `asset_id` in `event.json` resolves to a KG node | If the CAP entry uses a local tag format differing from the KG asset_id, seed resolution fails and the neighborhood is wrong |

### Stage B.5

| Assumption | Risk |
|------------|------|
| Historian provides pre-flagged anomaly records (start, end, pattern, severity) | Most historians provide raw time-series; anomaly flagging may require a separate upstream process |
| 72-hour lookback covers the relevant degradation period | Thermal fatigue, corrosion, and slow leak buildup can have signatures days or weeks before event onset |
| Plant topology graph is acyclic for propagation chain purposes | Recirculation systems, control loops, and shared utilities create cycles; the DAG assumption breaks |
| `is_upstream()` returns a deterministic answer | For components on multiple parallel paths (redundant trains) "upstream" is ambiguous |

### Stage C

| Assumption | Risk |
|------------|------|
| Each anomaly has a reliable `timestamp_end` | Plant anomaly detection typically provides precise start, fuzzy end; this directly affects Allen relation classification |
| All anomalies in the augmented set are independent observations | A single root cause can produce correlated anomalies on multiple sensors simultaneously, inflating anomaly count and confidence |
| FM-to-CR matching from `failure_mode_keywords[]` is reliable | Without reliable NER, recurrence profile degrades to asset level — a high-maintenance component inflates recurrence for every individual failure mode |

### Stage D

| Assumption | Risk |
|------------|------|
| Structural score (topology proximity) correlates with causal likelihood | A 2-hop component may be the true root cause; a 1-hop component may be a coincidental bystander with many documents |
| KG document metadata density correlates with actual evidence quality | Well-documented but non-causal FMs score higher than the actual cause if it is poorly documented in the KG |
| `safety_significant` flag is populated on FM nodes in the KG | If FMEA ingestion or MBSE ingest does not set this flag, the safety-promotion policy has no effect |

---

## 3.4 Real RCA Scenarios That Challenge the Pipeline

### S-1 — First-Occurrence Failure Mode (No KG Entry)

**Scenario**: A heat exchanger develops stress corrosion cracking (SCC) in a material batch installed fleet-wide two years ago. This is the first occurrence at this plant — no FMEA failure mode node for SCC on this material, no prior CRs on this equipment, no KG document links.

**Pipeline behavior**: Stage B retrieves no failure mode for SCC — the hypothesis is invisible. Stage B.5 may detect a temperature gradient anomaly chain but `per_candidate_chain_score` has no failure mode to map to. Stage D generates no candidate for SCC. The RCA card identifies the nearest documented FM (e.g., "general material degradation") with lower specificity. The real root cause is never surfaced.

**Why it matters**: first-occurrence failures are precisely the events requiring the most careful RCA — they indicate either a new degradation mechanism or a previously undocumented one. The pipeline is weakest exactly here.

### S-2 — Instrument Fault as Root Cause (Self-Referential Anomaly)

**Scenario**: A level transmitter fails high, producing a spurious high-level signal that trips a pump. The "anomaly" in `telemetry_summary.json` is the transmitter reading itself — the same signal that caused the trip.

**Pipeline behavior**: Stage C assigns OVERLAPS or PRECEDES to the transmitter anomaly (it started before the trip), scoring it 0.90. The FM for "instrument spurious high" should score well if it exists in the KG. **But** there is no explicit check asking whether this anomaly *is* the triggering signal rather than an independent precursor. If the FM keyword is "level transmitter malfunction" and the event description says "high level trip on pump suction," keyword matching at Stage D may or may not connect them depending on vocabulary coverage. The pipeline can reach the right answer but relies entirely on FMEA keyword alignment.

### S-3 — Human Performance Root Cause (Maintenance Error)

**Scenario**: A valve is reinstated incorrectly after maintenance — wrong orientation, partial closure. No sensor detects this at reinstallation. Three days later, reduced flow causes a pump trip.

**Pipeline behavior**: `telemetry_summary.json` contains a flow anomaly 30 minutes before the trip. Allen relation: PRECEDES → strong causal score. Stage D generates a strong candidate for "pump flow restriction." **But the actual root cause is the maintenance error, not the pump.** Stage D also generates a historical-event candidate for maintenance-related issues, but only if prior CRs mention valve maintenance errors on this equipment. Stage G's `maintenance_human_factors` Ishikawa branch may catch this — but only if a document snippet or candidate explicitly uses the right keywords. If no document says "wrong valve position," the branch is empty. The recommended action will be "inspect pump for flow restriction" not "review valve reinstallation procedure."

**Why it matters**: INPO AP-913 and 10 CFR 50 Appendix B explicitly require identifying whether human performance is a root cause or contributor. A pipeline that can only identify hardware failure modes when a human error is the root cause is not adequate for nuclear RCA.

### S-4 — Common Cause Failure (CCF) Across Multiple Trains

**Scenario**: Lube oil degradation occurs on all four trains of a safety-significant pump simultaneously, triggered by a contaminated oil batch from a vendor.

**Pipeline behavior**: `event.json` describes the trip on Train A. Stage B expands from Pump A — may or may not include Pumps B/C/D depending on topology. Stage D generates up to 4 candidates for "lube oil degradation" across trains. `common_cause_features` field exists in the output but the pipeline never groups these into a single CCF conclusion. Stage H single-primary-hypothesis architecture outputs "lube oil degradation on Pump A." The CCF nature — that all four trains failed from the same common cause (oil batch) — is not represented. **The recommended action "replace oil on Pump A" misses Pumps B/C/D entirely.**

**Why it matters**: CCF of safety-significant trains is one of the most critical failure patterns for nuclear RCA — it is the scenario that can simultaneously defeat redundancy and is specifically addressed in NUREG-0800 and generic letters.

### S-5 — Long-Latency Degradation Outside Historian Window

**Scenario**: Gradual heat exchanger fouling starts 3 weeks (504 hours) before the event. Subtle temperature differential anomalies appear at t-15 days, moderate at t-7 days, significant at t-24 hours. The protective trip occurs at t=0.

**Pipeline behavior**: Stage B.5 historian lookback = 72 hours → only the last 3 days of anomalies are captured. If `telemetry_summary.json` also uses a 72-hour window, the full 2-week degradation signature is entirely absent. Stage C receives only the short-window anomaly. TSKR confidence is lower than it would be with the complete signature. A competing hypothesis with a strong short-latency signal may score higher. More critically, the RCA card lacks the trending information that an engineer would use to establish this as a latent degradation issue rather than a sudden failure.

### S-6 — Concurrent Unrelated Event Contaminating Telemetry

**Scenario**: During investigation of a condenser vacuum loss, an unrelated cooling water pump trips at t-2 hours due to a separate mechanical issue. Both events appear in the same `telemetry_summary.json`.

**Pipeline behavior**: Stage C assigns PRECEDES (score 0.75) to the cooling water pump anomaly against the condenser event. Stage D generates a candidate for the cooling water pump FM with moderate structural + strong temporal scores. This is structurally correct — cooling water loss would cause vacuum loss — but if it was a truly independent coincidental event, the pipeline has no way to determine this. No access to the plant sequence-of-events log means the pipeline cannot verify whether the two events share a causal link.

### S-7 — Fast Cascade Within Allen Epsilon Tolerance

**Scenario**: An electrical fault causes a bus trip, simultaneously tripping 3 loads within 50 milliseconds. All 3 load anomalies appear within the 0.5h epsilon.

**Pipeline behavior**: epsilon = 0.5h → all three load anomalies are classified as SIMULTANEOUS with the event (mapping to DURING, score 0.30). The pipeline treats them as consequences, not causes. If the upstream electrical bus anomaly (the actual cause) is not in the KG neighborhood and not in `telemetry_summary`, the root cause is missed entirely. Known gap H3 acknowledges epsilon is not configurable; for nuclear plants where fast transients (turbine trips, reactor trips, electrical faults) are common, epsilon = 0.5h is systematically wrong.

### S-8 — No Anomaly Evidence (Design Trip or Spurious Actuation)

**Scenario**: A safety injection actuation occurs due to a design signal (high containment pressure). All sensors were functioning within normal bands. No anomaly exists in `telemetry_summary.json`.

**Pipeline behavior**: Stage B.5 historian query returns nothing. Stage C falls back to `fallback_confidence = 0.25` for all FMs. Telemetry score = 0.35 baseline for all candidates (no differentiation). The pipeline's primary discriminators (temporal, telemetry) are both at their floor values. Scoring is driven entirely by structural topology and KG document density. The highest-ranking candidate will be the FM with the most KG document links on the closest component — irrespective of whether it has any relationship to the actuation. **The RCA card is noise dressed as analysis.** This scenario also applies to spurious trips (false signal, no real process condition) which are among the most common nuclear operational events.

### S-9 — Documentation Density Bias (Regulatory Commitment Advantage)

**Scenario**: One failure mode — seal degradation — is an open NRC commitment item with 18 months of CRs, a dedicated ECA, and an open WO. A different failure mode is the actual cause but has no prior documentation.

**Pipeline behavior**: Stage D evidence sub-score for seal degradation: high (many CRs, ECA, WO linked in KG). Stage D evidence sub-score for actual cause: low (no documents). Seal degradation gets A-series; actual cause gets B-series. Stage E retrieves strong supporting snippets for seal degradation from Chroma (all those CRs and ECAs are embedded). Stage F reaffirms seal degradation as primary. The actual cause is a speculative B-series candidate that an analyst must notice and override.

**The pipeline systematically favors well-documented failure modes over the actual cause when documentation density is not correlated with causal likelihood.** This is a fundamental evidence-quality bias, not an edge case.

---

## 3.5 Cross-Cutting Gaps Not Currently Documented

**G-1 — Operating state absent from all scoring.**
`operational_context.operating_point` (power level, pressures, temperatures) is collected and passed to Stage D but consumed by no scoring stage. Operating condition directly discriminates between failure mode classes: cavitation requires flow below minimum; flow-induced vibration scales with power level; thermal fatigue is more likely during load transients; stress corrosion requires specific temperature/chemistry combinations. This is not a calibration issue — it is a structurally missing scoring dimension.

**G-2 — No representation of physical impossibility.**
At Stage D, every KG neighborhood failure mode is scored. Some may be physically impossible at the time of the event (e.g., cavitation on a pump at shutoff head). There is no mechanism to pre-filter physically impossible hypotheses. A well-documented physically impossible FM can still become a high-scoring A-series candidate.

**G-3 — Analyst override knowledge is not fed back to KG.**
When an analyst overrides the primary hypothesis (the common case for B-series candidates), that override carries RCA knowledge that should enrich the FMEA and KG. Currently this knowledge is lost. The CAP writeback concept covers the RCA conclusion but not the analyst's reasoning about why the pipeline's primary was wrong.

**G-4 — No "no adequate hypothesis" output state.**
The pipeline has no defined output state for the case where the true root cause is outside the KG. The output is always a confidence-capped RCA card with the best available (potentially wrong) hypothesis. There is no `rca_card.conclusion_type = "no_adequate_hypothesis"` state to tell the analyst that the pipeline searched its search space and found nothing above a reliability threshold. This state is needed to distinguish "we found a good hypothesis" from "we found the least-bad hypothesis in an inadequate search space."

**G-5 — Parallel execution failure handling for Stage B.5 not defined.**
If Stage B.5 fails (historian API down, DAG cycle detection fails, adapter timeout), Stage C falls back to `telemetry_summary`. But the orchestrator logic for detecting Stage B.5 failure and performing the fallback is not described. A Stage B.5 timeout that blocks Stage C from starting would halt the entire per-event pipeline.

**G-6 — No defense against embedding quality variation.**
Stage 5B embeds CR narratives and EDMS documents using a default embedding model. General-purpose embedding models produce poor-quality embeddings for domain-specific nuclear terminology (IST, GL 89-10, AOV, safety injection signal, etc.). Stage E's cosine similarity retrieval is only as good as the embedding quality. There is no embedding quality check and no fallback to keyword-based retrieval when semantic similarity is unreliable.

---

## 3.6 Priority Summary

| Priority | Issue | Impact |
|----------|-------|--------|
| P1 | IC-2 Stage C pseudo-code weight inconsistency | Incorrect implementation |
| P1 | IC-4 Stage C pseudo-code missing `cmms_context` / `signal_evidence` parameters | Incorrect implementation |
| P1 | IC-6 Stage F `signal_ev_index` not defined in pseudo-code | Incorrect implementation |
| P1 | IC-1 Section 1.5 not updated after A/B-series change | Reviewer confusion |
| P1 | S-3 Human performance root cause systematically invisible to scoring | Nuclear RCA regulatory adequacy (AP-913, 10 CFR 50 App B) |
| P1 | S-4 CCF not representable in single-primary-hypothesis architecture | Nuclear safety significance |
| P1 | S-8 No-anomaly scenario produces noise-as-analysis output | Operational validity |
| P2 | G-1 Operating state absent from all scoring | Missing discriminating dimension |
| P2 | S-9 Documentation-density bias advantages well-documented FMs over actual cause | Systematic scoring bias |
| P2 | G-4 No "no adequate hypothesis" output state | Pipeline output completeness |
| P2 | §3.2 `event.json` / `telemetry_summary.json` production interface unspecified | Integration prerequisite |
| P3 | S-5 Long-latency degradation outside 72h historian window | Completeness for slow failure modes |
| P3 | G-3 Analyst override not fed back to KG/FMEA | Knowledge capture |
| P3 | IC-5 Stage D known gaps reference obsolete hard-threshold logic | Document maintenance |
| P3 | IC-7 `writeback_ready` set prematurely at Stage D | Semantic clarity |
