# RCA Pipeline — Per-Stage Drill-Down
**Date**: April 21, 2026
**Baseline**: Orchestrator v3.1/v3.2 · Schema set v3.2
**Companion documents**: `RCA_pipeline_flowchart.md` (Layer 1) · `RCA_workflow_april_2.md` (formal spec) · `RCA_Data_Management_Strategy.md` (data layer)

---

## Template (used for every stage)

| Field | Content |
|-------|---------|
| **Input artifacts** | Field-level inputs consumed by this stage |
| **Key logic** | 3–6 bullets describing what the stage does |
| **Output artifact** | New fields produced |
| **Critical path pseudo-code** | 10–20 lines capturing the core computation |
| **Key thresholds & weights** | Hardcoded constants that matter for calibration |
| **Known gaps / open issues** | Cross-referenced to April 20 review where applicable |

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

- **C4 (April 20 review)**: `severity` field is stored but not used to gate minimum evidence requirements or adjust scoring weights. High-severity events get the same scoring floor as low-severity ones.
- No cross-artifact timestamp consistency check (e.g., `telemetry.window.end` vs `event.timestamp_start`).

> **Note — input schema validation**: Stage A currently performs only a key-presence check on the four input artifacts. Full JSON schema validation of all *output* artifacts happens at Stage J. Pulling at least a schema validation of the four input artifacts into Stage A would catch malformed inputs at the earliest possible point and produce clearer error messages than a mid-pipeline failure. This would mirror the same two-layer pattern Stage J already uses (schema + cross-artifact consistency), applied to inputs rather than outputs.

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
| `doc_window_days_before` | 90 | Docs older than 90 days get reduced recency bonus |
| `doc_window_days_after` | 7 | Small forward window for post-event documentation |
| `past_event_window_days` | 3650 | Parameter exists in code but applies to a KG query that is currently empty. Historical event retrieval window should be governed by CMMS query at Stage 5B instead. |
| Doc type priorities | CR=100, WO=95, ECA=90, RCA=85, FMEA=70, SOP=60 | Ordering within type |
| Path strength multipliers | 0.95 (containment), 0.85 (connectivity), 0.80 (other) | Weaken indirect causal paths |

### Known gaps

- **C1 (April 20 review)**: KG is a closed world — any failure mode not in Neo4j is invisible to the pipeline regardless of evidence. The `out_of_boundary_anomalies` field with `not_in_kg: true` partially surfaces this gap for analysts.
- **H1 (April 20 review)**: Sensor alias resolution has five fallback fields but no logging when fallback is used; analysts cannot tell whether the seed was resolved cleanly or via alias.
- `safety_functions` are fetched and stored in `kg_context` but **never propagated** into `rca_card` (see C5).
- Document priority scoring treats CR and WO as higher priority than FMEA, but FMEAs carry stronger causal evidence; this ordering may bury the most relevant documents.

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
| `telemetry_summary` | `signals[].anomalies[].timestamp_start/end`, `.pattern`, `.severity` |
| `event.json` | `timestamp_start`, `timestamp_end` |
| `operational_context` | Used only to resolve event interval if end timestamp is missing |

### Key logic

- Builds a per-failure-mode temporal pattern by scoring each FM's anomaly windows against the event interval using Allen interval algebra (13 primitive relations).
- Derives a dominant Allen relation (priority: OVERLAPS > CONTAINS > PRECEDES > DURING > FOLLOWS) and computes severity-weighted mean/std lag across causal windows.
- Scores latency alignment: compares observed lag to FM's `expected_latency_min/max_hours`; classifies `latency_violation_type` as `none | too_fast | too_slow | unknown`.
- Builds a recurrence profile from `kg_context.past_events`: count, inter-event trend (increasing/stable/decreasing), unresolved count, recency.
- Detects temporal contradiction: FOLLOWS relation (anomaly after event) → sets `temporal_contradiction: true`, applies −0.20 confidence penalty.
- Combines anomaly, latency, history, and count sub-scores into a single `confidence` value per FM pattern.

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
| `latency_violation_type` | `none \| too_fast \| too_slow \| unknown` |
| `temporal_contradiction` | bool |

### Critical path pseudo-code

```python
def score(event, telemetry, kg_context, oc, run_context):
    event_interval = Interval(event["timestamp_start"], event.get("timestamp_end"))
    anomaly_windows = extract_anomaly_intervals(telemetry)  # with severity labels

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

        # Latency alignment
        lat_score, violation = score_latency(
            mean_lag, fm["expected_latency_min_hours"], fm["expected_latency_max_hours"]
        )

        # Recurrence profile — built from CMMS CR records (not kg_context.past_events)
        profile = build_recurrence_profile(fm["fm_id"], cmms_context["cr_records"])

        # Sub-scores
        anomaly_score  = severity_weighted_mean_of_relation_scores(allen_scores)
        count_score    = score_anomaly_count(len(causal_windows))
        lag_consistency = score_lag_consistency(std_lag)
        history_score  = score_history(profile)

        contradiction = (dominant_relation == "FOLLOWS")

        confidence = clamp01(
            0.55 * max(anomaly_score, telemetry_support_floor)
            + 0.30 * lat_score
            + 0.15 * history_score
            + 0.20 * count_score
            + 0.15 * lag_consistency
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
| Anomaly weight | 0.55 | Dominant term in confidence composite |
| Latency weight | 0.30 | Second-largest term |
| History weight | 0.15 | Recurrence evidence |
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

---

## Stage D — Candidate Generation (pre-evidence)

**Class / entry-point**: `RuleBasedCausalityEngineV31.generate()` (also v32 in flowchart)
**File**: `orchestrators/causality_engine_v31.py`

### What this stage does

Stage D is where the pipeline first produces a ranked list of root cause hypotheses. By this point the search space is fully defined (Stage B), the evidence corpus is assembled (Stage 5B), and the temporal characterisation of each failure mode is complete (Stage C). Stage D's job is to synthesise all of that into a scored, filtered, ranked list of candidates — the hypotheses that the rest of the pipeline will investigate, refine, and ultimately present to the analyst.

It generates two candidate pools. The first is built from the failure modes retrieved from the KG: each failure mode becomes a hypothesis of the form "this failure mode, on this component, caused the event." The second is built from the historical CR records in `cmms_context`: each past event involving the same equipment or similar failure mode becomes a hypothesis of the form "this is a recurrence of a known previous failure." The two pools are kept separate so that historical-event candidates do not displace failure-mode candidates in the top-k ranking.

Each candidate is scored across five dimensions: structural (how topologically close is the component to the primary asset, and how well do the symptoms match?), temporal (what does the TSKR pattern say about timing consistency — this is the Stage C output consumed here), telemetry (how many anomalies are present on this component, and how severe?), evidence (how many relevant documents exist in the KG for this failure mode — note this is a proxy score using document metadata only, not Chroma content), and governance (does PM compliance data suggest this maintenance category has been deferred or overdue?). The five scores are combined into a single composite via a weighted average.

Two hard thresholds then filter the ranked list: composite ≥ 0.30 and evidence ≥ 0.35. Candidates failing either are dropped entirely before being returned. This is important to understand: the evidence threshold at Stage D is based on KG document metadata, not on actual retrieved content. A failure mode with strong structural and temporal evidence but no documents linked in the KG will be blocked here, even if the Chroma collection assembled at Stage 5B contains highly relevant material. The v1→v2 ranking delta at Stage F is designed to correct this, but the hard filter means some candidates never reach Stage F at all.

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
- Applies a dual hard filter: composite ≥ 0.30 AND evidence ≥ 0.35; candidates failing either threshold are dropped.
- Returns top-k=10 candidates ranked by composite score; this is `causality_candidates v1`.

### Output artifact — `causality_candidates v1`

| Field | Content |
|-------|---------|
| `candidates[]` | Ranked list; each with all 5 sub-scores, composite, confidence_label |
| `candidate_id` | `"FM::{fm_id}"` or `"EVT::{event_id}"` |
| `hypothesis_type` | `"failure_mode"` or `"historical_event"` |
| `composite_score` | Weighted sum of 5 sub-scores |
| `confidence_label` | `high ≥ 0.75`, `medium ≥ 0.45`, `low > 0`, `speculative` |
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

        if composite >= 0.30 and E >= 0.35:
            candidates.append(build_candidate(fm, S, T, Tel, E, G, composite))

    candidates.sort(key=lambda c: -c["composite_score"])
    return {"causality_candidates": candidates[:top_k]}
```

### Key thresholds

| Parameter | Value | Effect |
|-----------|-------|--------|
| Weights | S=0.30, T=0.20, Tel=0.20, E=0.20, G=0.10 | Relative dimension importance |
| `minimum_composite_threshold` | 0.30 | Hard floor; drops weak candidates |
| `minimum_evidence_threshold` | 0.35 | Hard floor for evidence sub-score |
| `top_k_candidates` | 10 | Output list cap |
| Seed-match topology bases | telemetry=0.90, direct=0.85, neighbor=0.75, none=0.40 | Structural baseline |
| Symptom delta | ±0.20 | Maximum symptom modifier on structural |
| Evidence baseline | 0.30 | Score when no documents exist (just below E threshold) |
| Recency factor breakpoints | ≤90d=1.0, ≤365d=0.85, ≤730d=0.70, >730d=0.55 | |
| PM governance cap | 0.95 | Governance alone cannot produce perfect score |
| Allen relation score mapping | PRECEDES=1.0, OVERLAPS=0.95, CONTAINS=0.90, … FOLLOWS=0.20 | |

### Known gaps

- **C2 (April 20 review)**: `top_k=10` is a hard cap. Common-cause failures affecting many parallel trains could generate more candidates than the cap allows, silently dropping some.
- **C4 (April 20 review)**: Safety significance is not a scoring dimension. A low-composite candidate on a safety-critical function will rank below a high-composite candidate on a non-safety system.
- Evidence score at Stage D uses only KG document metadata (types and recency). Actual content is not available until Stage E; this makes Stage D evidence scores a proxy, not a real evidence assessment — the v1→v2 delta in Stage F is intended to correct this, but the proxy baseline may be misleading.
- **`evidence_gap` flag** is set when zero documents are found, but the hard threshold `E ≥ 0.35` blocks these candidates entirely; an analyst cannot see zero-evidence candidates even to flag for investigation.
- **`telemetry_anomaly_precedes` seed_match_type not mapped in structural score**: Stage B's proposed telemetry-driven neighborhood expansion adds components with `seed_match_type = "telemetry_anomaly_precedes"`. This type is not in Stage D's `TOPOLOGY_BASE` mapping and will fall through to the default score of 0.40 — the weakest structural baseline, equivalent to "no topological relationship." A component admitted at Stage B specifically because its anomaly preceded the event would be structurally penalised rather than rewarded. A new entry is needed in the mapping, likely ~0.80, reflecting that admission was evidence-driven but not a direct asset match. **Code change required**: update `TOPOLOGY_BASE` in `causality_engine_v31.py` to include `"telemetry_anomaly_precedes": 0.80`.
- **Evidence baseline (0.30) below threshold (0.35) silently blocks undocumented failure modes**: the evidence score baseline when no documents are linked in the KG is 0.30, which falls below the hard minimum evidence threshold of 0.35. Any failure mode with no KG document links is blocked before reaching Stage E, even if Stage 5B embedded highly relevant material in Chroma. This creates a dependency on KG document linkage completeness rather than actual evidence quality. A first-occurrence failure mode with strong telemetry and structural signals will be discarded. This should be an explicit design decision: either lower the threshold to 0.25 (allow undocumented candidates into Stage E), or raise the baseline to 0.35 (make the block intentional and documented), or replace the hard filter with a soft penalty that flags low-evidence candidates for analyst review rather than dropping them.
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

### Key logic

- Updates the evidence sub-score for each candidate using actual Chroma retrieval results (replaces the KG-metadata proxy from Stage D).
- Applies a **evidence posture** classification: `supported | contested | neutral | missing`.
- Flags candidates for analyst review when primary and top-alternative are within `review_alternative_gap = 0.10` of each other.
- Recomputes composite score with updated evidence sub-score; re-ranks the candidate list.
- The **v1 → v2 ranking delta** is the primary diagnostic signal: candidates that move up in rank after evidence retrieval were undervalued by the KG proxy; those that drop were over-valued.

### Output artifact — `causality_candidates v2`

Adds to v1 fields:

| New field | Content |
|-----------|---------|
| `evidence_posture` | `supported \| contested \| neutral \| missing` |
| `v1_rank`, `v2_rank` | Rank position before and after evidence update |
| `rank_delta` | `v1_rank − v2_rank` (positive = moved up) |
| `review_required` | bool — set when rank gap to next candidate ≤ 0.10 |

### Critical path pseudo-code

```python
def refine_with_evidence(candidates_v1, evidence_bundle):
    ev_index = evidence_bundle["per_candidate_summary"]
    refined = []

    for i, cand in enumerate(candidates_v1["candidates"]):
        ev = ev_index.get(cand["candidate_id"], {})

        # Evidence score update formula
        best_support = ev.get("best_support_score", 0.0)
        n_support    = ev.get("supporting_count", 0)
        n_contra     = ev.get("contradicting_count", 0)

        E_new = clamp01(
            0.40 * best_support
            + 0.30 * min(1.0, n_support / 3.0)   # saturates at 3+ supporting snippets
            - 0.20 * min(1.0, n_contra / 2.0)     # penalty for contradicting evidence
        )

        composite_new = (
            0.30 * cand["scores"]["structural"]
            + 0.20 * cand["scores"]["temporal"]
            + 0.20 * cand["scores"]["telemetry"]
            + 0.20 * E_new
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

### Input artifacts

All prior artifacts: `kg_context`, `tskr_patterns`, `causality_candidates v2`, `evidence_bundle`, `operational_context`, `pm_compliance`

### Key logic

- Translation stage only — no new inference. Maps the top candidates and their evidence to Ishikawa (fishbone) categories.
- Six standard categories: `equipment_hardware`, `process_procedure`, `measurement_instrumentation`, `environment`, `maintenance_human_factors`, and an implicit `management_systems` catch-all.
- Each populated row includes `source_links` back to `candidate_id` or `doc_id` for traceability.
- If Stage G is skipped, Stage H receives `ishikawa_matrix: null` and proceeds with no structural fishbone input.

### Output artifact — `ishikawa_matrix`

| Field | Content |
|-------|---------|
| `equipment_hardware[]` | Candidates / evidence categorized as hardware causes |
| `process_procedure[]` | Candidates / evidence related to procedures |
| `measurement_instrumentation[]` | Sensor/instrument-related items |
| `environment[]` | External / environmental contributors |
| `maintenance_human_factors[]` | PM compliance, human error items |
| Each row: `source_links[]` | `{type: "candidate"|"doc", id: str}` |

### Known gaps

- Categorization is heuristic (keyword matching against cause_label and FM description); no validation that assignments are correct.
- Six-category Ishikawa is the standard industrial format, but nuclear RCA practices (e.g., INPO AP-913) use a different taxonomy (latent organizational weaknesses, direct cause, contributing cause, root cause). The mapping is not aligned to the nuclear standard.

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

- **H6 (April 20 review)**: `writeback_ready: true` does not check whether recommended actions already exist as open WOs in CMMS (duplicates would be created on CAP writeback).
- Layer 2 cross-checks do not verify that `rca_card.recommended_actions[].target_component_id` values are valid `element_usage` IDs in the KG.
- **All schema validation is late-binding**: Layer 1 schema checks run at Stage J, after every upstream stage has already consumed the artifacts being validated. A schema error in `kg_context` is not caught until after Stages 5B, C, D, E, F, G, and H have all executed on invalid data. **Code change required**: move per-artifact schema validation to the stage that produces each artifact — `kg_context` validated at end of Stage B, `tskr_patterns` at end of Stage C, `causality_candidates v1` at end of Stage D, `evidence_bundle` at end of Stage E, `causality_candidates v2` at end of Stage F, `rca_card` at end of Stage H. Stage J retains only cross-artifact consistency checks (Layer 2), which legitimately require all artifacts to exist before running.
- **Stage A input validation should include JSON schema checks**: as a direct consequence of the above, Stage A's current key-presence check on input artifacts should be upgraded to full JSON schema validation. This is the pipeline's earliest possible detection point for malformed inputs. Catching a schema error in `event.json` at Stage A prevents all downstream stages from executing on bad data.
- **`next_step` classification has no routing implementation**: Stage J determines `next_step: writeback | analyst_review | remediation` but there is no downstream mechanism that acts on this value. The manifest is produced and written to disk; what happens next depends entirely on the analyst reading it. For a co-pilot system, the `analyst_review` path should at minimum trigger a notification or open a review task; the `writeback` path should have a defined handoff to the CAP integration module. Currently all three outcomes are operationally identical — the run ends and the analyst must manually check the manifest.
- **No re-run capability for `remediation` runs**: if Stage J returns `next_step: remediation`, the only recourse is a full re-run from Stage A. There is no mechanism to re-execute specific stages (e.g., re-fetch data at Stage 5B, or re-synthesize at Stage H) after correcting an issue. For a pipeline that may take significant time to execute due to CMMS queries and LLM calls, partial re-runs would be operationally important.
- **`run_complete` marker not checked**: Stage J should verify the presence of a `run_complete` sentinel (proposed in Stage I known gaps) before attempting validation. Without this check, Stage J may produce a `run_manifest` for a partially written run, certifying as valid only the artifacts that were written before the interruption.

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

| Issue | Stage | April 20 ref | Status |
|-------|-------|--------------|--------|
| Safety significance not a scoring dimension | D, F, H | C4 | Open |
| Confidence always capped at "medium" (fallback path) | H | C3 | Open |
| Single primary hypothesis — no co-equal causes | H | C2 | Open |
| Safety functions siloed — not in rca_card | B, H | C5 | Open |
| KG closed-world assumption | B | C1 | Open — by design |
| `instrument_validity_flag` not consumed | C | H3 | Open |
| `review_required` not injected into analyst_review | F, H | — | Open |
| Evidence score uses snippet count, not authority | F | — | Open |
| EDMS / FMEA ingestion stubs | 5B | — | Open |
| OE LLM tier | 5B | — | Future |
