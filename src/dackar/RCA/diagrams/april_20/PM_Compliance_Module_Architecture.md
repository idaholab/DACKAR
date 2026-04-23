# PM Compliance Verification Module — Architecture Notes
**Date**: April 21, 2026
**Context**: Parenthesis during NC6 sprint planning. The RCA pipeline consumes `pm_compliance.json` at Stage A but currently depends on an externally-supplied artifact. This note designs the module that would produce it.

---

## 1. Purpose and Scope

The PM Compliance Verification Module answers one question for a given RCA event:

> *Was preventive maintenance for the affected equipment performed correctly, on schedule, and with adequate scope to have prevented or detected the failure mode under investigation?*

This is distinct from the governance score in Stage D, which applies a compliance penalty to candidate scoring. This module produces the structured input artifact that makes that scoring meaningful — and surfaces compliance gaps that the analyst needs to see regardless of scoring.

---

## 2. Inputs

| Source | Content | Notes |
|--------|---------|-------|
| CMMS (Maximo / SAP PM) | PM work order history, task lists, completion dates, as-found conditions | Primary data source; queried via API or pre-extracted export |
| Equipment PM schedule | Frequencies, task codes, responsible craft | May live in CMMS or a separate planning system |
| KG / FMEA | Which failure modes each PM task is designed to prevent or detect | Requires explicit PM-to-FM linkage in KG (currently implicit) |
| `event.json` | Event timestamp, affected asset, affected components | Used to scope the time window |
| `kg_context.json` | Component list, failure mode set | Used to scope which PM tasks are relevant |

---

## 3. Module Components

### 3.1 `PMScheduleLoader`
Loads the PM schedule for the asset and each component in scope.

- Queries CMMS for all active PM plans on `asset_id` and each `component_id`
- Returns: task code, description, frequency (days/hours/cycles), last due date, next due date
- Handles multiple frequency types: calendar-based, operating-hour-based, condition-based
- **Key design decision**: operating-hour-based PMs require a runtime hours feed (telemetry or CMMS meter readings); if unavailable, fall back to calendar-based with a `frequency_type_warning`

### 3.2 `PMExecutionVerifier`
Checks actual PM work orders against the schedule for a configurable look-back window.

- For each PM task, finds the most recent closed WO of matching task code before `event.timestamp_start`
- Computes:
  - `last_pm_date` — completion date of most recent execution
  - `overdue_days` — `event.timestamp_start − next_due_date` (negative = on time, positive = overdue)
  - `compliance_status`: `compliant | overdue | missed | not_applicable`
  - `missed_cycles` — how many scheduled cycles were skipped
- **Edge case**: a PM completed the day before the event with an as-found condition of "found degraded" is more significant than a PM that was overdue by 30 days but had no prior degradation. Raw schedule compliance alone is insufficient.

### 3.3 `PMScopeAnalyzer`
Cross-references executed PM tasks against the FMEA failure modes in scope.

- For each candidate failure mode `FM-xxx`, looks up in KG: which PM tasks are tagged as `prevents` or `detects` for that failure mode
- Determines:
  - `scope_covers_failure_modes[]` — FMs for which at least one executed PM task provides coverage
  - `scope_gaps[]` — FMs for which no executed PM task provides coverage
  - `coverage_type` per FM: `preventive` (PM eliminates the mechanism), `detective` (PM detects early degradation), `none`
- **Key dependency**: requires explicit `pm_task_id → failure_mode_id` linkage in the KG. If absent, this component falls back to free-text matching between PM task descriptions and FM labels (brittle but better than nothing).
- **Output flag**: `fmea_pm_linkage_available: bool` — downstream scoring should treat scope analysis as advisory if `False`

### 3.4 `PMEffectivenessAnalyzer`
Evaluates whether past PM executions caught degradation early.

- Looks at `as_found_condition` fields on closed PM WOs for the same component over the past N cycles (configurable, default 3 cycles)
- Signals:
  - `degradation_trend`: `improving | stable | degrading | unknown` — based on as-found condition vocabulary (requires controlled vocabulary in CMMS; otherwise `unknown`)
  - `pm_found_defect_rate`: fraction of PM executions that recorded a defect condition
  - `last_as_found`: verbatim last as-found condition string (passed through to rca_card for analyst)
- **Limitation**: depends entirely on CMMS data quality. Many plants have inconsistent as-found documentation. Module must surface data quality confidence, not silently treat missing data as "no degradation."

### 3.5 `PMCurrencyChecker`
Evaluates whether the current PM frequency is appropriate given operating history.

- Compares PM frequency to mean time between failures for the same failure mode class (from KG recurrence profile if available)
- Flags `pm_frequency_concern` if: `pm_interval_days > 0.5 × mean_failure_interval_days`
- Flags `pm_overdue_at_failure: bool` — was the most recent PM overdue at the time of the event?
- Optional: compares PM frequency to vendor recommendation or fleet benchmark (requires external reference data; advisory only)

### 3.6 `PMComplianceAggregator`
Combines all sub-module outputs into the `pm_compliance.json` artifact.

- Computes `overall_compliance`: `compliant | partial | non_compliant`
  - `compliant`: all tasks on schedule, no scope gaps for in-scope FMs
  - `partial`: some tasks overdue OR scope gaps exist for secondary FMs
  - `non_compliant`: any task missed for a directly-linked FM, OR primary FM has `scope_coverage: none`
- Computes `maintenance_induced_risk`: `low | medium | high`
  - `high` if: primary FM has no PM coverage AND PM was overdue at failure
  - `medium` if: PM was overdue OR scope gap exists for primary FM
  - `low` otherwise
- Writes `pm_compliance.json` (see §5)

---

## 4. Integration with the RCA Pipeline

```
[CMMS API / Export]  [KG / FMEA]  [event.json]
         ↓                ↓              ↓
    ┌────────────────────────────────┐
    │  PM Compliance Verification   │  ← runs as Stage 5A (alongside cmms_context_builder)
    │  Module                       │
    └────────────────┬───────────────┘
                     ↓
           pm_compliance.json
                     ↓
         Stage A — input validation
                     ↓
         Stage D — governance score
         (uses component-level compliance_status and scope_gaps)
                     ↓
         Stage H — synthesis
         (pm gaps surface as recommended_actions of type "pm_corrective")
                     ↓
         rca_card.recommended_actions[].pm_gap_ref
```

**Stage D integration**: the governance score currently applies a flat penalty for non-compliance. With this module, Stage D can use `scope_covers_failure_modes` to apply a targeted penalty only when the PM scope gap is directly linked to the candidate FM — not a blanket compliance hit.

**Stage H integration**: `scope_gaps[]` for the primary hypothesis FM should automatically generate a recommended action of type `pm_corrective` in the rca_card. If `maintenance_induced_risk == "high"`, that action gets `priority: "high"` unconditionally (overrides the default scoring-based priority assignment — directly addresses C4).

---

## 5. Output Schema (`pm_compliance.json`)

```json
{
  "asset_id": "string",
  "assessment_date": "ISO-8601",
  "event_id": "string",
  "look_back_window_days": 730,
  "fmea_pm_linkage_available": true,
  "components": [
    {
      "component_id": "string",
      "pm_tasks": [
        {
          "task_code": "string",
          "description": "string",
          "frequency_days": 180,
          "last_pm_date": "ISO-8601",
          "next_due_date": "ISO-8601",
          "overdue_days": 0,
          "compliance_status": "compliant | overdue | missed | not_applicable",
          "missed_cycles": 0,
          "last_as_found": "string | null",
          "coverage_type": "preventive | detective | none"
        }
      ],
      "scope_covers_failure_modes": ["FM-001"],
      "scope_gaps": ["FM-031"],
      "degradation_trend": "stable | degrading | improving | unknown",
      "pm_overdue_at_failure": false,
      "pm_frequency_concern": false
    }
  ],
  "summary": {
    "overall_compliance": "compliant | partial | non_compliant",
    "maintenance_induced_risk": "low | medium | high",
    "has_scope_gaps_for_primary_fm": false,
    "data_quality_confidence": "high | medium | low"
  },
  "data_quality_notes": []
}
```

---

## 6. Key Design Decisions and Open Questions

| Decision | Options | Recommendation |
|----------|---------|---------------|
| CMMS integration mode | Real-time API vs. pre-extracted export | Export first (simpler, audit-stable); API integration is Phase 2 |
| PM-to-FM linkage source | KG explicit tags vs. free-text matching | Explicit KG tags required; free-text fallback only with `fmea_pm_linkage_available: false` |
| Look-back window | Fixed (e.g., 2 years) vs. N-cycles | N-cycles (default 3) is more meaningful for low-frequency PMs |
| As-found vocabulary | Controlled list vs. free text | Controlled list preferred; module should map CMMS free-text to controlled terms via NER |
| Condition-based PM triggers | Requires sensor feed | Defer to Phase 2; flag as `not_applicable` with note if CBM-triggered PM task detected |
| Multi-unit / fleet scope | Single unit vs. fleet-wide PM history | Single unit for now; fleet scope is a separate module responsibility |

---

## 7. Failure Modes of the Module Itself

- **CMMS data quality**: as-found conditions not documented → `degradation_trend: unknown`, `data_quality_confidence: low`
- **KG linkage absent**: PM task descriptions not linked to FM IDs → scope analysis advisory only; flag with `fmea_pm_linkage_available: false`
- **Mixed frequency types**: calendar PMs mixed with CBM-triggered PMs for same component → treat CBM tasks as `not_applicable` unless runtime hours are available
- **PM performed but not closed in CMMS**: work done, WO left open → appears as `missed`; module cannot detect this without additional data source

---

## 8. Relationship to Existing Pipeline Artifacts

| Artifact | Current State | With This Module |
|----------|--------------|-----------------|
| `pm_compliance.json` | Externally supplied; key-presence-only validation at Stage A | Produced by this module; schema-validated; component-level and FM-level detail |
| Stage D governance score | Asset-level PM compliance flat penalty | Component-level, FM-targeted penalty using `scope_gaps` |
| `rca_card.recommended_actions` | LLM-generated; no PM gap awareness | PM scope gaps auto-generate `pm_corrective` actions; `maintenance_induced_risk: high` forces `priority: high` (closes C4 partially) |
| `run_manifest.ap913_completeness` | Not yet implemented (NM15) | `pm_compliance_verified: true` adds one AP-913 completeness signal |

---

## 9. Implementation (in progress, April 2026)

### 9.1 Code location

| Path | Role |
|------|------|
| `DACKAR/src/dackar/RCA/pm_compliance/` | Python package: loaders, verifiers, scope/effectiveness/currency helpers, **aggregator** |
| `DACKAR/src/dackar/RCA/schemas/pm_compliance.json` | **Canonical** JSON schema for the pipeline (extended with optional fields from §5: `event_id`, `assessment_date`, `look_back_window_days`, `fmea_pm_linkage_available`, `data_quality_notes`, `components[]`, and summary roll-ups; `wo_id` on each `checks[]` row for Stage D traceability) |
| `DACKAR/src/dackar/RCA/unit_tests/test_pm_compliance_aggregator.py` | Schema smoke tests and verifier edge cases (run: `python test_pm_compliance_aggregator.py` from `unit_tests/` with `RCA` on `sys.path`) |

**Public entry point:** `from pm_compliance import build_pm_compliance, PMComplianceConfig` (add `src/dackar/RCA` to `PYTHONPATH` or run from a context that already loads RCA the same way as other unit tests).

### 9.2 Schema alignment (§5 vs. pipeline)

The **original** pipeline and `causality_engine_v32` read **`checks[]`** (with `status` `pass` \| `fail` \| `unknown`, `overdue_by_days`, optional `component_id` / `applicable_fm_ids`). The narrative §5 schema (nested `components[].pm_tasks`) is represented as: **`checks[]` + optional `components[]` summary** so one artifact stays **schema-valid** and **Stage D compatible**. The aggregator fills `components[]` with scope/degradation fields when data exists.

### 9.3 Phase 1 (current) behaviour

- **Input:** `event` (at least `asset_id`, `timestamp_start`), optional `kg_context`, optional `export_rows` (pre-parsed task rows: `check_id`, `check_type`, dates, `applicable_fm_ids`, etc.).
- **PMScheduleLoader** filters rows by `asset_id` and optional `component_id` list from the KG.
- **PMExecutionVerifier** derives `status` / `overdue_by_days` from `next_due_date` + `event` time, or passes through explicit `compliance_status` when the export already computed it.
- **PMScopeAnalyzer** sets `fmea_pm_linkage_available` when the KG exposes PM↔FM fields on failure modes (see `preventing_pm_task_ids` / `detecting_pm_task_ids` / `pm_task_ids`); otherwise scope gap lists are **advisory/empty** until linkage exists.
- **Not yet done:** live CMMS API in this package (reuse / extend `cmms_integration` adapters), orchestrator `run()` hook to call `build_pm_compliance` as “Stage 5A”, NER for as-found vocabulary, Stage H `pm_corrective` action synthesis from `scope_gaps` (orchestrator/synthesizer change).

### 9.3.1 Post-review fixes (Apr 22, 2026)

- **Rollup alignment fix:** `overall_compliance` now marks `non_compliant` when `primary_fm_id` is in scope gaps (per §3.6), without requiring an additional overdue/fail condition.  
- **Advisory gap guard:** primary/scope gap rollups are now applied only when KG linkage is available (or some coverage exists), preserving §3.3 advisory behavior when linkage is absent.  
- **Narrative status fix:** `components[].pm_tasks[].compliance_status` now preserves `not_applicable` (instead of collapsing to `compliant`).  
- **Effectiveness signal fix:** degradation trend now sources from export as-found fields (`as_found_last` / `as_found_condition`) before free-text fallback.  
- **Loader validation fix:** export rows missing required identity/type (`check_id`/`task_code` and `check_type`) are dropped early and surfaced via `data_quality_notes`.

### 9.4 Dependencies still external

- **CMMS export column mapping** — implement a dedicated parser that produces `export_rows` for `build_pm_compliance`.
- **KG PM↔FM tags** — populate Neo4j / `kg_context.failure_modes[]` with the field names the scope analyzer looks for, or expand the mapper when your graph model uses different property names.

### 9.5 Alignment with §3–§7 (code vs. spec)

| Spec item | In code? | Notes |
|-----------|----------|--------|
| **§3.1** PMScheduleLoader, calendar vs. operating hours | **Partial** | Loader filters `export_rows` by `asset_id` / components. Operating-hour / CBM tasks get a **data_quality_note** if `frequency_type=operating_hours` and no `operating_hours_at_event`; `not_applicable` status is supported. |
| **§3.2** Last WO / `last_pm_date`, overdue, `missed_cycles` | **Partial** | Derives from **export** `next_due_date` + event time (or `compliance_status` overrides). Does **not** yet query the most recent **closed** WO from CMMS by task code. |
| **§3.2** As-found “degraded” edge case | **Not yet** | No extra penalty beyond what appears in `details` + effectiveness heuristics. |
| **§3.3** Scope, `fmea_pm_linkage_available` | **Aligned** | `True` only when the KG has explicit PM id lists on failure modes (`preventing_pm_task_ids`, `detecting_pm_task_ids`, `pm_task_ids`). **False** if coverage comes only from export `applicable_fm_ids` (advisory; note added). **Free-text** PM↔FM match is **not** implemented (per §3.3, optional / brittle). |
| **§3.3** `coverage_type` per task | **Partial** | Pass-through on each `pm_tasks[]` row if `export_rows` set `coverage_type`; else `none`. |
| **§3.4** `pm_found_defect_rate`, `last_as_found` | **Partial** | `last_as_found` in `pm_tasks` from `as_found_last` / `as_found_condition` on the export row. Defect **rate** not yet in `summary` (add when controlled vocabulary exists). |
| **§3.5** `pm_frequency_concern` | **Yes** | Uses `PMCurrencyChecker.frequency_concern` with `kg_context.failure_modes[].mean_time_between_events_days` and `export_rows[*].frequency_days` + `applicable_fm_ids`. |
| **§3.5–3.6** `maintenance_induced_risk` / `has_scope_gaps_for_primary_fm` | **Aligned (with `primary_fm_id`)** | Matches §3.6 when **`primary_fm_id=`** is passed to `build_pm_compliance`; primary scope gap now drives `overall_compliance=non_compliant` and risk rollups per spec. Without `primary_fm_id`, heuristics use gap + overdue. |
| **§5** `components[].pm_tasks` + `summary` roll-ups | **Yes** | `pm_tasks` populated from export rows; pipeline **`checks[]`** + optional **`components[]`**; JSON Schema remains canonical for validation. |
| **§4 / Stage H** `pm_corrective` + priority | **Not in module** | Still requires synthesizer / orchestrator to consume `summary` + `components`. |

### 9.6 Unit tests

`unit_tests/test_pm_compliance_aggregator.py` (run as `python test_pm_compliance_aggregator.py` from `RCA/unit_tests/`) includes:

- JSON Schema validation for the built artifact (Draft 7 + `date-time` formats).
- Overdue / fail paths and `overdue_items`.
- **`fmea_pm_linkage_available` is False** when only export `applicable_fm_ids` is present (no KG tags).
- **`not_applicable`** → `checks[].status=pass` (governance-neutral).
- **`not_applicable` preserved** in `components[].pm_tasks[].compliance_status`.
- **`_rollup_risk`** primary + gap + overdue → `high` / `medium`.
- **Primary scope gap** with linkage marks `summary.overall_compliance=non_compliant`.
- **As-found trend source** from export `as_found_last` / `as_found_condition`.
- **Loader drops invalid rows** (missing `check_id`/`task_code` or `check_type`) with `data_quality_notes`.
- **`RuleBasedCausalityEngineV32._governance_details`** accepts the built `pm` dict (Stage D path).

---

*End of architecture notes — April 21, 2026*  
*Implementation status: **Phase 1 package in `RCA/pm_compliance/`**; orchestrator integration pending*  
*Dependencies: KG PM-to-FM linkage (for full scope analysis); CMMS export format spec (for row parser)*
