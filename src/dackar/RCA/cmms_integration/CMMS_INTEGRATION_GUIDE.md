# CMMS Live Context Retrieval — Integration Guide

## 1. Problem Statement

### The Gap

The DACKAR RCA pipeline is an evidence-driven system: it reasons over KG topology,
telemetry anomalies, operating experience documents, and historical failure patterns.
None of those sources capture **what happened in the plant maintenance system in the
days leading up to the event**.

Before this module, if a technician filed a Corrective Action Request (CR) three days
ago about bearing vibration on the same pump, and that CR had not been manually
loaded into the KG or Chroma, the RCA engine would not see it. The most time-relevant
operational signal — recent maintenance history on the exact asset — was invisible.

### Operational Consequence

- The synthesizer had no knowledge of open work orders on the failed component.
- Sister-equipment recurrence patterns (common-cause failure signals, critical in
  nuclear and process plant environments) were not captured.
- Engineers reading the RCA card had to manually cross-reference the CMMS before
  trusting or overriding the conclusions.
- Adoption risk: if the tool misses obvious recent maintenance context, engineers
  lose confidence in its conclusions.

### What Is Not Solved Here

This module is **read-only**: it retrieves CMMS records for RCA reasoning.
Writing corrective actions back to the CMMS (the reverse direction) is handled
by the `cap_integration/` module (Gap #15).

---

## 2. Design Decisions

### 2.1 Live Query vs. Pre-Population

Two approaches were considered:

| Approach | Description | Problem |
|---|---|---|
| **Pre-populate** | ETL job syncs CMMS → KG/Chroma on a schedule | Requires ETL to run between event and investigation; creates staleness window; operational burden |
| **Live query at RCA time** | Fetch CRs/WOs at invocation | Always current; no pre-population needed; scoped to the specific event |

**Decision: live query at RCA invocation time.**

The pre-population approach conflates two data types with different latency
requirements: structural knowledge (equipment topology, failure mode libraries)
changes slowly and is right for KG/Chroma. Operational records (recent CRs, open WOs)
change hourly and should be fetched live.

### 2.2 Hybrid Data Flow (Option C)

Three options for what to do with live-fetched data:

- **Option A** — Inject everything into Chroma (embed all fields)
- **Option B** — Structured artifact only, bypass Chroma
- **Option C** — Hybrid: structured fields → `cmms_context` artifact; narrative text → Chroma injection

**Decision: Option C.**

Embedding a CR status code or a date field is a lossy transformation — cosine
similarity over "OPEN" and "2026-01-05" destroys the precision that makes those
fields valuable. The synthesizer needs to reason exactly ("3 open WOs on this
bearing in 60 days") not semantically. Structured fields stay structured.

Narrative text (long_text fields — failure descriptions, technician remarks) has
genuine semantic content and benefits from embedding. These are injected into the
run-scoped Chroma collection so the evidence retriever picks them up alongside
OE documents and procedures.

### 2.3 Lookback Window: Event-Driven, Not Fixed

The lookback window start is derived from the **last PM date on the primary asset**
(sourced from `kg_context.past_events[]`). This is more precise than a fixed
lookback:

- A bearing PM'd two weeks ago is more informative than one PM'd two years ago.
- CRs filed between the last PM and the event are exactly the recurrence signal
  that matters for determining whether a failure was anticipated or whether the
  PM program is adequate.

Fallback: if no PM is found in `past_events[]`, the window defaults to
`event_time − 90 days` (configurable via `CMMSContextBuilderConfig.fallback_lookback_days`).

### 2.4 Sister Equipment Scope

Sister equipment CRs are included in the query. In nuclear and process plant
environments, common-cause failure is a significant risk — if two pumps in the
same train have both accumulated bearing CRs, that is a fundamentally different
situation than an isolated single-unit failure.

Sister equipment is identified from `kg_context.components[]` where
`relation_to_asset` is in `{"same_train", "adjacent"}`. This is configurable via
`CMMSContextBuilderConfig.sister_relation_types`.

Note: fleet-level sister equipment identification (same equipment class across
multiple units/plants) is a future extension (Gap #19-B). The current
implementation scopes to the local KG subgraph.

### 2.5 Prompt Placement

The CMMS context block is placed in the synthesizer prompt **after the evidence
bundle, immediately before the synthesis instruction**. This exploits recency
bias in LLM attention — a compact structured summary of recent maintenance activity
immediately before the generation task is more likely to influence the output than
the same block buried before a long evidence section ("lost in the middle" problem).

---

## 3. Architecture

```
RCA Invocation
    │
    ├── Stage 5A: Neo4jKGContextBuilder.build()   → kg_context artifact (existing)
    │       │
    │       └── kg_context.past_events[]          → last PM date
    │           kg_context.components[]           → sister component IDs
    │                                               (relation_to_asset: same_train, adjacent)
    │
    ├── Stage 5B: CMMSContextBuilder.build()       → cmms_context artifact (NEW)
    │       │
    │       ├── _resolve_lookback(kg_context, event_ts)
    │       │       └── last PM date  OR  event_time − fallback_days
    │       │
    │       ├── _resolve_sister_ids(kg_context)
    │       │       └── component_ids where relation_to_asset in sister_relation_types
    │       │
    │       ├── CMMSContextAdapter.fetch(
    │       │       primary_asset_id,
    │       │       sister_component_ids,
    │       │       lookback_from,          ← ISO timestamp
    │       │       lookback_to,            ← event_time
    │       │       event,
    │       │   )  → {cr_records[], wo_records[]}
    │       │
    │       ├── _enrich_record()
    │       │       ├── days_before_event (derived)
    │       │       └── status normalisation (WAPPR→open, COMP→closed, etc.)
    │       │
    │       └── _build_recurrence_summary()
    │               ├── cr_count_primary / cr_count_sister
    │               ├── open_wo_count / open_cr_count
    │               └── earliest / most_recent CR dates
    │
    ├── Chroma injection (narrative text only)
    │       └── get_chroma_documents(cmms_context)
    │               → [{text: long_text, metadata: {source, run_id, cr_id, ...}}]
    │               → evidence_store.add_documents(docs)
    │
    ├── Stage 6: EvidenceRetriever                → picks up injected CMMS narratives
    │             (no interface change — cmms_live docs retrieved automatically)
    │
    └── Stage 7: RCASynthesizerV31.synthesize()
                ├── compact_cmms_context()        → token-efficient summary (5 CRs, 5 WOs)
                ├── recurrence_summary included   ← structured, no embedding loss
                └── prompt block: after evidence bundle, before synthesis instruction
```

### Data Stores by Role

| Store | Content | Population timing | Changed by this gap |
|---|---|---|---|
| Neo4j KG | Equipment topology, failure modes, past events, PM history | Batch ETL (weekly/monthly) | No |
| Chroma (persistent) | OE docs, procedures, historical investigation reports | Batch ingestion | No |
| Chroma (run-scoped) | CMMS narrative text for this run | At RCA invocation | **Yes — new** |
| `cmms_context` artifact | Structured CR/WO records, recurrence summary | At RCA invocation | **Yes — new** |

---

## 4. File Structure

```
cmms_integration/
    __init__.py
    cmms_adapter.py              Protocol + NoOpCMMSAdapter + MockCMMSAdapter
    cmms_context_builder.py      CMMSContextBuilder + CMMSContextBuilderConfig
    maximo_cmms_adapter.py       MaximoCMMSAdapter skeleton (plant teams implement)
    sap_pm_cmms_adapter.py       SAPPMCMMSAdapter skeleton (plant teams implement)
    CMMS_INTEGRATION_GUIDE.md    (this file)

schemas/
    cmms_context.json            JSON Schema for the cmms_context artifact
```

---

## 5. Key Data Structures

### CMMSContextBuilderConfig

```python
@dataclass
class CMMSContextBuilderConfig:
    fallback_lookback_days: int = 90
    # ^ Days to look back when no PM found in past_events[].
    #   Operational recommendation: align with plant's PM interval for the
    #   most common equipment class. 90 days is a conservative default.

    sister_relation_types: List[str] = ["same_train", "adjacent"]
    # ^ KG relation_to_asset values that qualify a component as sister equipment.
    #   "upstream" and "downstream" are intentionally excluded — they represent
    #   process flow neighbours, not redundant equipment in the same train.

    include_sister_equipment: bool = True

    max_cr_records: int = 100   # cap; most recent kept
    max_wo_records: int = 100   # 0 = no cap
```

### cmms_context artifact (top level)

```json
{
  "cmms_context_id": "CMMSCTX::EVT-001::2026-01-10T12:05:00+00:00",
  "run_id": "...",
  "event_id": "EVT-001",
  "asset_id": "PUMP-01",
  "generated_at": "...",
  "adapter": "MaximoCMMSAdapter",
  "lookback_anchor": "last_pm",
  "lookback_from": "2025-11-15T00:00:00+00:00",
  "lookback_to": "2026-01-10T12:00:00+00:00",
  "sister_component_ids": ["COMP-002", "COMP-003"],
  "cr_records": [...],
  "wo_records": [...],
  "recurrence_summary": {
    "cr_count_primary": 3,
    "cr_count_sister": 1,
    "open_wo_count": 2,
    "open_cr_count": 1,
    "earliest_related_cr_date": "2025-11-20T...",
    "most_recent_cr_date": "2026-01-05T..."
  },
  "provenance": {
    "generated_by": "CMMSContextBuilder",
    "kg_context_id": "SG-...",
    "query_params": {...}
  }
}
```

### CR record (within cr_records[])

```json
{
  "cr_id": "CR-12345",
  "cr_type": "CAL",
  "status": "open",
  "priority": "2",
  "short_description": "Bearing vibration elevated above 3σ",
  "long_text": "Technician remarks: sustained high-freq vibration on ...",
  "functional_location": "PLANT/SYS-A/PUMP-01/BEARING",
  "equipment_id": null,
  "component_id": "COMP-001",
  "created_date": "2026-01-05T08:00:00+00:00",
  "closed_date": null,
  "days_before_event": 5,
  "is_sister_equipment": false
}
```

`days_before_event` is derived by `CMMSContextBuilder._enrich_record()`, not
from the CMMS. Negative values indicate the CR was filed after the event
(data lag from overnight batch jobs is common in Maximo deployments).

---

## 6. Pseudocode

### CMMSContextBuilder.build()

```
build(event, kg_context, run_id):

    event_ts = parse_iso(event.event_time)

    # Step 1: Derive lookback window
    pm_dates = [parse_iso(e.event_date)
                for e in kg_context.past_events
                if "pm" in e.event_type.lower() or "preventive" in e.event_type.lower()]

    if pm_dates:
        lookback_from = max(pm_dates)
        anchor = "last_pm"
    else:
        lookback_from = event_ts - timedelta(days=config.fallback_lookback_days)
        anchor = "event_time_minus_90d"

    lookback_to = event_ts

    # Step 2: Identify sister components
    sister_ids = [c.component_id
                  for c in kg_context.components
                  if c.relation_to_asset in config.sister_relation_types]

    # Step 3: Fetch from CMMS
    raw = adapter.fetch(
        primary_asset_id = event.asset_id,
        sister_component_ids = sister_ids,
        lookback_from = lookback_from.isoformat(),
        lookback_to   = lookback_to.isoformat(),
        event = event,
    )

    # Step 4: Enrich records
    cr_records = [enrich(r, event_ts) for r in raw.cr_records]
    wo_records = [enrich(r, event_ts) for r in raw.wo_records]

    # Step 5: Cap (keep most recent)
    cr_records = sort_by_date(cr_records, desc=True)[:max_cr_records]
    wo_records = sort_by_date(wo_records, desc=True)[:max_wo_records]

    # Step 6: Recurrence summary
    summary = {
        cr_count_primary: count(cr for cr in cr_records if not cr.is_sister),
        cr_count_sister:  count(cr for cr in cr_records if cr.is_sister),
        open_wo_count:    count(wo for wo in wo_records if wo.status == "open"),
        open_cr_count:    count(cr for cr in cr_records if cr.status == "open"),
        earliest_cr_date: min(cr.created_date for cr in cr_records),
        most_recent_cr_date: max(cr.created_date for cr in cr_records),
    }

    return cmms_context_dict(...)
```

### enrich_record()

```
enrich(record, event_ts):
    record.days_before_event = (event_ts - parse_iso(record.created_date)).days

    # Normalise status codes (Maximo / SAP PM / generic)
    raw = record.status.upper()
    if raw in {WAPPR, INPRG, APPR, WMATL, WPCOND, open}:
        record.status = "open"
    elif raw in {COMP, CLOSE, CLSD, TECO, closed}:
        record.status = "closed"
    elif raw in {CAN, DLFL, cancelled}:
        record.status = "cancelled"
    else:
        record.status = raw   # pass through unknown codes

    if is_sister_equipment not in record:
        record.is_sister_equipment = False

    return record
```

### Chroma injection

```
get_chroma_documents(cmms_context):
    docs = []
    for cr in cmms_context.cr_records:
        text = cr.long_text OR cr.short_description
        if not text:
            continue
        docs.append({
            "text": text,
            "metadata": {
                source: "cmms_live",
                record_type: "cr",
                run_id: cmms_context.run_id,
                cr_id: cr.cr_id,
                component_id: cr.component_id,
                is_sister_equipment: cr.is_sister_equipment,
                days_before_event: cr.days_before_event,
                status: cr.status,
            }
        })
    # same for wo_records
    return docs

# Orchestrator calls:
docs = builder.get_chroma_documents(cmms_context)
evidence_store.add_documents(docs)
# EvidenceRetriever picks these up in Stage 6 alongside static docs —
# no retriever interface changes required.
```

---

## 7. Implementing a Live Adapter

All live adapters implement the `CMMSContextAdapter` Protocol:

```python
class CMMSContextAdapter(Protocol):
    def fetch(
        self,
        primary_asset_id: str,
        sister_component_ids: List[str],
        lookback_from: str,   # ISO-8601 UTC
        lookback_to: str,     # ISO-8601 UTC
        event: JsonDict,
    ) -> JsonDict:            # {"cr_records": [...], "wo_records": [...]}
        ...
```

### Minimal implementation checklist

1. **Map `sister_component_ids` to CMMS FLOCs / equipment IDs.**
   The KG stores `maximo_floc` and `sap_equipment_id` as properties on
   `element_usage` nodes. These are available in `kg_context.components[]`
   (see `kg_context.json` schema). Your adapter receives the KG component IDs;
   you need to resolve them to CMMS location codes for the CMMS query.
   The simplest approach: pass the full `kg_context` alongside `sister_component_ids`
   and build a lookup dict at fetch time.

2. **Build the query filter** using `lookback_from`, `lookback_to`, and the
   resolved location codes. See the helper methods in `maximo_cmms_adapter.py`
   (`_build_oslc_where`) and `sap_pm_cmms_adapter.py` (`_build_odata_filter`).

3. **Map CMMS field names to the `cmms_context` schema.**
   Minimum required fields per record:
   - `cr_id` / `wo_id`
   - `status` (raw CMMS code — `_enrich_record` normalises it)
   - `short_description`
   - `created_date` (ISO-8601 string)
   - `is_sister_equipment` (bool)

   Optional but strongly recommended:
   - `long_text` (narrative — embedded into Chroma)
   - `cr_type` / `wo_type`
   - `priority`
   - `functional_location` / `equipment_id`

4. **Set `is_sister_equipment`** based on whether the record's location code
   matched the primary asset or a sister component ID.

5. **Return idempotently.** The same `(primary_asset_id, lookback window)` call
   should return the same records. Do not mutate CMMS state.

### Maximo (skeleton: `maximo_cmms_adapter.py`)

```python
# Endpoint: GET /maximo/oslc/os/SR?oslc.where=<filter>&oslc.select=*
# Auth: apikey header
# Key field map:
#   TICKETID   → cr_id
#   CLASS      → cr_type
#   STATUS     → status  (pass raw; enrich() normalises)
#   PRIORITY   → priority
#   DESCRIPTION → short_description
#   LDTEXT     → long_text
#   LOCATION   → functional_location
#   REPORTDATE → created_date
```

### SAP PM (skeleton: `sap_pm_cmms_adapter.py`)

```python
# Endpoint: GET /sap/opu/odata/sap/PMMAINTNOTIF_ODATA/MaintenanceNotificationSet
#           ?$filter=...&$format=json
# Auth: basic auth (service account, read-only PM authorisations)
# Key field map:
#   NotifNo                   → cr_id
#   MaintNotifType            → cr_type  (M1, M2, M3, M4, Q1, Q3)
#   UserStatus                → status   (pass raw; enrich() normalises)
#   Priority                  → priority
#   ShortText                 → short_description
#   LongText                  → long_text
#   FunctLoc                  → functional_location
#   Equipment                 → equipment_id
#   MaintNotifCreationDate    → created_date
```

---

## 8. Orchestrator Integration

### Configuration

```python
from cmms_integration.cmms_adapter import NoOpCMMSAdapter
from cmms_integration.cmms_context_builder import CMMSContextBuilderConfig

orchestrator = build_dev_orchestrator(
    output_dir="./runs",
    client=neo4j_client,
    # dev/CI: NoOp (default)
    cmms_adapter=NoOpCMMSAdapter(),
    # production:
    # cmms_adapter=MaximoCMMSAdapter(
    #     base_url="https://maximo.plant.corp/maximo",
    #     api_key=os.environ["MAXIMO_API_KEY"],
    #     site_id="PLANT1",
    # ),
    cmms_context_builder_config=CMMSContextBuilderConfig(
        fallback_lookback_days=90,
        sister_relation_types=["same_train", "adjacent"],
        max_cr_records=50,
    ),
)
```

### What happens inside `orchestrator.run()`

```
run(event, telemetry_summary, ...):
    ...
    kg_context = kg_context_builder.build(...)        # Stage 5A
    _validate_and_persist("kg_context", kg_context)

    if cmms_adapter is not None:                       # Stage 5B (NEW)
        cmms_context = build_cmms_context(
            run_id, event, kg_context
        )
        # → fetches records, enriches, builds summary
        # → persists cmms_context artifact
        # → injects narratives into evidence_store

    ...
    evidence_bundle = evidence_retriever.retrieve(...)  # Stage 6
    # (includes injected CMMS narratives automatically)

    rca_card = rca_synthesizer.synthesize(
        ...,
        cmms_context=cmms_context,                     # NEW param
    )
```

### Standalone usage (incremental pipeline)

```python
# If you have a pre-built kg_context from a previous stage:
cmms_context = orchestrator.build_cmms_context(
    run_id="run-001",
    event=event_dict,
    kg_context=kg_context_dict,
)
# cmms_context is already persisted and narratives injected
```

---

## 9. Synthesizer Prompt Integration

`cmms_context` appears in the LLM prompt as a compact structured block
(5 most recent CRs and WOs; `long_text` stripped — it is already in Chroma).

```json
"cmms_context": {
  "lookback_from": "2025-11-15T00:00:00+00:00",
  "lookback_to":   "2026-01-10T12:00:00+00:00",
  "lookback_anchor": "last_pm",
  "recurrence_summary": {
    "cr_count_primary": 3,
    "cr_count_sister": 1,
    "open_wo_count": 2,
    "open_cr_count": 1
  },
  "cr_records": [
    {
      "cr_id": "CR-12345", "cr_type": "CAL", "status": "open",
      "priority": "2",
      "short_description": "Bearing vibration elevated above 3σ",
      "days_before_event": 5, "is_sister_equipment": false
    },
    ...
  ],
  "wo_records": [...]
}
```

The synthesizer is instructed:

> If `cmms_context` is present and non-empty, consider `recurrence_summary`
> when assessing confidence. Open CRs or WOs on the same component strengthen
> `immediate_corrective` actions. Sister equipment CRs are weaker signal —
> note them as contextual, not primary evidence.

A new citation `source_type` value is supported: `"cmms_record"`.

---

## 10. Testing

### Unit tests

`unit_tests/test_cmms_context_builder.py` — 45 tests covering:

| Class | Tests |
|---|---|
| `TestNoOpCMMSAdapter` | Empty returns, required keys |
| `TestMockCMMSAdapter` | Fixture return, copy safety |
| `TestCMMSContextBuilderConfig` | Defaults, custom fallback |
| `TestBuilderPackageStructure` | All required top-level keys, IDs, adapter name |
| `TestLookbackResolution` | Fallback window, last-PM anchor, multiple PMs (latest used), non-PM events ignored |
| `TestSisterComponentResolution` | same_train + adjacent included; primary + downstream excluded; custom types; disabled |
| `TestRecordEnrichment` | `days_before_event` computation, status normalisation (Maximo + SAP codes), `is_sister_equipment` default |
| `TestRecurrenceSummary` | Primary vs. sister counts, open WO/CR counts, earliest/most-recent dates, empty case |
| `TestGetChromaDocuments` | long_text extraction, short_description fallback, empty-text skipping, WO docs, metadata fields |
| `TestRecordCap` | Cap enforcement, most-recent selection, zero-cap passthrough |

### Integration testing with a live CMMS

```python
from cmms_integration.maximo_cmms_adapter import MaximoCMMSAdapter
from cmms_integration.cmms_context_builder import CMMSContextBuilder

adapter = MaximoCMMSAdapter(
    base_url="https://maximo.plant.corp/maximo",
    api_key="...",
    site_id="PLANT1",
)
builder = CMMSContextBuilder(adapter)
ctx = builder.build(event=test_event, kg_context=test_kg_context, run_id="test-001")

# Verify
assert ctx["recurrence_summary"]["cr_count_primary"] >= 0
assert all(r.get("days_before_event") is not None for r in ctx["cr_records"])
assert all(r["status"] in {"open", "closed", "cancelled", "unknown"}
           for r in ctx["cr_records"] + ctx["wo_records"])
```

---

## 11. Key Assumptions and Constraints

| Assumption | Implication |
|---|---|
| CMMS is accessible at RCA invocation time | A network failure during `adapter.fetch()` will produce an empty `cmms_context` (NoOp fallback). The orchestrator catches and logs adapter exceptions; the RCA run continues without CMMS data. |
| `kg_context.past_events[]` contains PM history | If PM events are not loaded into the KG, the lookback anchor falls back to 90 days. Ensure the KG ETL includes PM event records for the equipment in scope. |
| `kg_context.components[].relation_to_asset` is populated | Sister equipment identification depends on this field. Components without a `relation_to_asset` value are silently excluded from the sister scope. |
| CMMS FLOCs / equipment IDs are stored on KG `element_usage` nodes | Live adapters map KG `sister_component_ids` to CMMS location codes using `kg_context.components[].maximo_floc` or `.sap_equipment_id`. If these properties are absent from the KG, sister queries cannot be scoped correctly. See `kg_context.json` schema and Gap #15 notes on KG-augmented FLOC resolution. |
| Chroma `evidence_store` exposes `add_documents()` | The `InMemoryEvidenceStore` and `ChromaEvidenceStore` must implement this method. If the store does not support it, narrative injection is silently skipped (logged as a warning). |
| CMMS data is read-only | This module never writes to the CMMS. All adapters are pure query. |

---

## 12. Future Extensions

- **Fleet-level sister equipment** (Gap #19-B): identify the same equipment
  class across multiple units or plants using a shared equipment taxonomy in
  the KG, not just the local subgraph topology.
- **Full-text CMMS search**: use failure-mode keywords from `kg_context.failure_modes[]`
  to query the CMMS long_text field directly, rather than relying purely on
  asset/FLOC scoping. Maximo supports OSLC full-text search; SAP PM supports
  notification text search.
- **Inspection records**: extend `cmms_context` with an `inspection_records[]`
  array for plants where inspection findings are stored separately from CRs/WOs.
- **Telemetry-driven scope widening**: if a monitored variable matches a known
  failure signature (e.g., bearing frequencies), automatically widen the CMMS
  query to include all components sharing that failure mode in the KG.
