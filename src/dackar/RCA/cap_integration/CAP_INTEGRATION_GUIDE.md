# CAP Integration Guide

## Purpose

This module bridges the DACKAR RCA pipeline to plant Corrective Action Programs (CAP), specifically Maximo and SAP PM. It implements **Option D** from the gap analysis: a payload-mapping layer with a pluggable adapter interface, so the full export pipeline can run and be verified without a live CMMS connection, and the live push adapter can be dropped in later with no other code changes.

The problem being solved: the RCA card's `recommended_actions[]` array contains structured, prioritised actions tied to specific plant components. Without this module, engineers must manually re-enter those findings into Maximo or SAP — a time-consuming step that creates adoption risk and breaks the audit chain from RCA conclusion to corrective action execution.

---

## Architecture

```
rca_card.recommended_actions[]
         │
         │  (+ kg_context.components[] for FLOC resolution)
         ▼
  CAPExportSerializer
  ─────────────────────────────────────────────────
  • Maps action_type   → cr_type / notification_type
  • Maps priority      → priority_code
  • Resolves component → functional_location / equipment_id
    via KG-stored CMMS properties (maximo_floc, sap_equipment_id)
  • Flags unresolved locations as mapping_status: "unresolved"
  • Assembles one CRRecord per recommended_action
         │
         ▼
  CAPExportPackage  (schema: schemas/cap_export_package.json)
         │
         ▼
  CAPAdapter.submit(package) → CAPSubmissionReceipt
  ─────────────────────────────────────────────────
  Concrete implementations:
    FileDropCAPAdapter   — writes JSON to a watched directory (default/dev)
    NoOpCAPAdapter       — silently discards (test environments)
    MaximoCAPAdapter     — calls Maximo REST API   [future]
    SAPPMCAPAdapter      — calls SAP PM OData API  [future]
         │
         ▼
  CAPSubmissionReceipt  (cr_numbers, wo_numbers, status, errors)
  persisted as cap_submission_receipt.json in artifact store
```

The orchestrator wires these together in `RCAReasoningOrchestrator.export_cap()`.

---

## KG-Augmented FLOC Resolution

### The design decision

`recommended_actions[i].target_component_id` contains a KG component ID (e.g. `U2-CND-EXPANSION-JOINT-EXHAUST`). CMMS systems use their own hierarchical location codes: Maximo uses **Functional Location** (FLOC, e.g. `PWR-U2-COND-EXPJT-01`) and SAP PM uses **Equipment ID** (e.g. `EQ-10047821`). These do not match and cannot be mechanically derived.

**Resolution strategy — KG as single source of truth:**

The KG `element_usage` node already stores all authoritative component metadata. CMMS IDs are added as optional node properties:

```cypher
// Maximo
SET c.maximo_floc = 'PWR-U2-COND-EXPJT-01'

// SAP PM
SET c.sap_equipment_id = 'EQ-10047821'
```

`Neo4jKGContextBuilder` returns these properties as part of its existing component queries, so they appear in `kg_context.components[]` without any additional KG call at export time. The serializer reads from the already-persisted `kg_context` artifact — the mapping is reproducible and tied to the exact KG snapshot that produced the RCA.

**Fallback chain** (applied per component, per target system):

```
1. kg_context.components[i].maximo_floc           → use as functional_location
   (or sap_equipment_id for SAP PM)
2. No KG property set                              → functional_location = null,
                                                     mapping_status = "unresolved"
```

Unresolved locations are collected in `cap_export_package.unresolved_locations[]` so the export is still valid and submittable, but the engineer sees exactly which components need KG enrichment.

### How to add CMMS IDs to the KG

This is a one-time data-loading step, performed by the plant's CMMS administrator or KG data steward. A Cypher snippet:

```cypher
MATCH (c:element_usage {id: $component_id})
SET c.maximo_floc      = $floc,
    c.sap_equipment_id = $equipment_id
```

A bulk-load script from a plant's CMMS export CSV is the typical approach. The DACKAR KG ingestion pipeline may also be configured to pick up these fields during plant model loading.

### Impact on `kg_context` schema

`kg_context.json` schema: `components[].items` gains two optional fields:

```json
"maximo_floc":      { "type": ["string", "null"] }
"sap_equipment_id": { "type": ["string", "null"] }
```

`Neo4jKGContextBuilder` seed and neighbor queries are extended with:

```cypher
c.maximo_floc      AS maximo_floc,
c.sap_equipment_id AS sap_equipment_id
```

---

## Field Mapping Design

### `action_type` → CMMS record type

| RCA `action_type`      | Maximo `cr_type` | SAP PM `notification_type` | Notes |
|------------------------|------------------|----------------------------|-------|
| `immediate_corrective` | `CAL`            | `M1`                       | Corrective maintenance |
| `long_term_corrective` | `CAP`            | `M2`                       | Planned corrective |
| `preventive`           | `PM`             | `M3`                       | Preventive maintenance |
| `monitoring`           | `SR`             | `M4`                       | Service request / inspection |
| `procedure_update`     | `TQ`             | `Q3`                       | Technical query / doc change |
| `engineering_evaluation` | `ECR`          | `Q1`                       | Engineering change request |

These defaults are defined in `field_maps/maximo_default.json` and `field_maps/sap_pm_default.json`. They can be overridden per-plant by providing a custom map in `CAPExportConfig`.

### `priority` → CMMS priority code

| RCA `priority` | Maximo `priority` | SAP PM `priority` |
|----------------|-------------------|-------------------|
| `critical`     | `1`               | `1`               |
| `high`         | `2`               | `2`               |
| `medium`       | `3`               | `3`               |
| `low`          | `4`               | `4`               |

### Long-text field composition

Each CMMS record includes a long-text narrative assembled from:

```
[RCA Run: {run_id}]
[Event: {event_id}] [Asset: {asset_id}]
[Primary Cause: {primary_hypothesis.cause_label}]

Action: {action.description}

Rationale: {action.rationale}

Expected observation if true: {action.expected_observation_if_true}

--- Generated by DACKAR RCA v{pipeline_version} ---
```

This ensures the CMMS record carries full traceability back to the RCA run without requiring the engineer to manually link them.

### `owner` mapping

`recommended_actions[i].owner` is carried through as-is to the CMMS `assigned_to` / `reported_by` field. If null, the export record leaves the field blank — the CMMS import job or the analyst fills it in. No mapping is applied.

---

## `CAPExportConfig` Reference

```python
@dataclass
class CAPExportConfig:
    target_system: str = "maximo"
    # "maximo" | "sap_pm" | "generic"
    # Controls which CMMS-specific fields are populated
    # and which field_map file is loaded by default.

    action_type_map: dict = field(default_factory=dict)
    # Overrides for the default action_type → cr_type mapping.
    # Merged over the default field_map at instantiation.
    # Example: {"monitoring": "PM"}  (treat monitoring as PM in this plant)

    priority_map: dict = field(default_factory=dict)
    # Overrides for the priority → priority_code mapping.

    default_work_group: str | None = None
    # If set, stamped on every CRRecord.work_group.
    # Plant-specific; required by some Maximo configurations.

    default_plant_section: str | None = None
    # SAP PM: stamped on every record if set.

    long_text_header: str | None = None
    # Optional custom prefix for the long-text narrative.
    # If None, the standard DACKAR header is used.

    include_rca_run_id_in_description: bool = True
    # If True, prepends "[RCA:{run_id}]" to the short description field,
    # creating a searchable token for reverse-lookup in the CMMS.
```

---

## `CAPAdapter` Protocol and Implementations

### The Protocol

```python
class CAPAdapter(Protocol):
    def submit(
        self,
        package: CAPExportPackage,
    ) -> CAPSubmissionReceipt:
        ...
```

`submit()` must be idempotent with respect to `package.export_id` — if the same package is submitted twice, it should not create duplicate CMMS records. Concrete implementations are responsible for deduplication (e.g. checking `export_id` against a submitted-IDs log or a CMMS query).

### `FileDropCAPAdapter`

Writes `cap_export_package_{export_id}.json` to a configured directory. Returns a receipt with `status: "pending"` and empty `cr_numbers` — the CMMS import job processes the file asynchronously.

This is the default adapter wired in `build_dev_orchestrator()`. It lets the full pipeline run and produce a verifiable export without any CMMS connection.

### `NoOpCAPAdapter`

Discards the package silently. Returns a receipt with `status: "noop"`. Used in unit tests and CI environments where no file I/O is desired.

### Implementing `MaximoCAPAdapter` (future — Option B live push)

```python
class MaximoCAPAdapter:
    def __init__(self, base_url: str, api_key: str, ...):
        ...

    def submit(self, package: CAPExportPackage) -> CAPSubmissionReceipt:
        cr_numbers = []
        errors = []
        for record in package.cr_records:
            payload = _to_maximo_payload(record)
            try:
                response = requests.post(
                    f"{self.base_url}/oslc/os/mxsr",
                    json=payload,
                    headers={"apikey": self.api_key},
                    timeout=30,
                )
                response.raise_for_status()
                cr_numbers.append(response.json()["srnumber"])
            except Exception as exc:
                errors.append({"record": record.export_record_id, "error": str(exc)})
        return CAPSubmissionReceipt(
            ...
            cr_numbers=cr_numbers,
            status="submitted" if not errors else "partial",
            errors=errors,
        )
```

Key points for implementors:
- Maximo REST API endpoint: `POST /oslc/os/mxsr` (Service Requests) or `POST /oslc/os/mxwo` (Work Orders) depending on `cr_type`
- Auth: API key header (`apikey`) or OAuth2 depending on Maximo version
- The `export_id` should be stored in a Maximo custom field (e.g. `EXTERNALREFID`) for deduplication and reverse lookup
- SAP PM equivalent: `POST /sap/opu/odata/sap/PMMAINTNOTIF_ODATA/MaintenanceNotificationSet`

### Implementing `SAPPMCAPAdapter` (future)

Follow the same `CAPAdapter` Protocol. Map `cr_records` to SAP PM `MaintenanceNotification` objects. Store `export_id` in the `LongText` or a custom field.

---

## Orchestrator Integration

### `RCAReasoningOrchestrator.export_cap()`

```python
result = orchestrator.export_cap(
    run_id=run_id,
    rca_card=modified_card,        # post-override card
    kg_context=kg_context,         # for FLOC resolution
    override_record=override_record,  # optional; links override_id in package
)
# result keys: "export_package", "submission_receipt"
```

**Prerequisites:** `export_cap()` should only be called after `apply_override()` has returned a card with `writeback_decision == "accept"`. The orchestrator enforces this by raising `RuntimeError` if `analyst_review.writeback_recommendation != "ready_if_accepted"`.

**Artifacts persisted:**
- `cap_export_package.json` — the full export package
- `cap_submission_receipt.json` — the adapter's receipt (contains `cr_numbers` once live adapter is used)

**Updating `build_dev_orchestrator()`:**

```python
# New optional parameter:
def build_dev_orchestrator(
    ...,
    cap_adapter=None,           # if None, FileDropCAPAdapter(output_dir) is used
    cap_config=None,            # if None, CAPExportConfig() defaults are used
) -> RCAReasoningOrchestrator:
```

### Full post-pipeline call sequence

```python
# 1. Run the RCA pipeline
result = orchestrator.run(event=event, ...)

# 2. Analyst reviews and applies override
modified_card, override_record = orchestrator.apply_override(
    run_id=result["run_id"],
    rca_card=result["rca_card"],
    override_input={
        "override_type": "accept",
        "rationale": "DO elevation unambiguously supports air in-leakage.",
        "writeback_decision": "accept",
        "analyst_id": "jsmith",
        "questions_resolved": [...],
    },
)

# 3. Export to CAP
cap_result = orchestrator.export_cap(
    run_id=result["run_id"],
    rca_card=modified_card,
    kg_context=result["kg_context"],
    override_record=override_record,
)

print(cap_result["submission_receipt"]["cr_numbers"])
# With FileDropCAPAdapter: []  (file written, import pending)
# With MaximoCAPAdapter:   ["SR-2024-04901", "SR-2024-04902"]
```

---

## Testing

### Unit tests (`unit_tests/test_cap_export_serializer.py`)

Cover:
- `action_type` → `cr_type` mapping for all six action types
- `priority` → `priority_code` mapping for all four priorities
- FLOC resolved from `kg_context.components[].maximo_floc` when present
- FLOC null and `mapping_status = "unresolved"` when KG property absent
- `unresolved_locations` list populated correctly
- Long-text narrative contains `run_id`, `event_id`, `action.description`, `action.rationale`
- `include_rca_run_id_in_description = False` omits token from short description
- Custom `action_type_map` override applied correctly
- One `CRRecord` per `recommended_action` (count assertion)
- Export package has correct `export_id` format
- `rca_card_id` and `override_id` propagated to package provenance

### Unit tests (`unit_tests/test_cap_adapter.py`)

Cover:
- `FileDropCAPAdapter` writes a valid JSON file to the configured directory
- `FileDropCAPAdapter` receipt has `status: "pending"` and empty `cr_numbers`
- `NoOpCAPAdapter` returns receipt with `status: "noop"` and makes no file I/O
- Both adapters return a `CAPSubmissionReceipt` with all required fields

### Integration guard (orchestrator level)

- `export_cap()` raises `RuntimeError` if called with a card whose `writeback_recommendation != "ready_if_accepted"`
- Both artifacts (`cap_export_package.json`, `cap_submission_receipt.json`) are persisted when `persist_intermediate_artifacts = True`

### Testing a new adapter without a live CMMS

Use `NoOpCAPAdapter` for all unit tests. For integration testing against a real Maximo/SAP sandbox, set the `DACKAR_MAXIMO_URL` and `DACKAR_MAXIMO_API_KEY` environment variables and instantiate `MaximoCAPAdapter` directly in a dedicated integration test that is excluded from the standard `pytest` run (mark with `@pytest.mark.integration`).

---

## Files in this module

| File | Purpose |
|------|---------|
| `cap_config.py` | `CAPExportConfig` dataclass |
| `cap_export_serializer.py` | `CAPExportSerializer` — maps rca_card + kg_context → `CAPExportPackage` |
| `cap_adapter.py` | `CAPAdapter` Protocol, `FileDropCAPAdapter`, `NoOpCAPAdapter` |
| `field_maps/maximo_default.json` | Default Maximo field mappings |
| `field_maps/sap_pm_default.json` | Default SAP PM field mappings |
| `CAP_INTEGRATION_GUIDE.md` | This file |

Related files outside this module:

| File | Change |
|------|--------|
| `schemas/cap_export_package.json` | New schema (stays in schemas/ for validator) |
| `schemas/kg_context.json` | Add optional `maximo_floc`, `sap_equipment_id` to components items |
| `orchestrators/kg_context_builder.py` | Extend component queries to return CMMS properties |
| `orchestrators/rca_reasoning_orchestrator.py` | Add `cap_adapter`, `cap_config` fields; add `export_cap()` method |
