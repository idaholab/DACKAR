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

*End of architecture notes — April 21, 2026*
*Status: Design only — not yet implemented*
*Dependencies before implementation: KG PM-to-FM linkage tags; CMMS export format specification*
