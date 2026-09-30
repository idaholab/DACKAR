# Step 2c — Allen Relation Map Hardening Plan

**Date:** 2026-04-25  
**Step:** Step 2 — KG Expansion / sub-step 2c: Temporal Relation Identification  
**Source of truth:** `rca_metamodel.md` §2c, `orchestrators/temporal_relations.py`, schemas

---

## Step 2c Definition (from metamodel)

> Identify Allen interval relations between events  
> — Compute Allen relations between identified anomalies, alarm logs, and the triggering event  
> — Establish temporal ordering across all event types  
> → Allen relation map: ordered event graph with temporal relations between all identified events

**Scope decision (agreed):** Allen relations computed between **anomalies, alarms, SOE records** and
the triggering event. CRs, WOs, and KG past events are excluded.

---

## Architecture recap

The Allen algebra engine (`temporal_relations.py`) is fully implemented:
- `Interval(start, end)` — closed time interval
- `allen_relation(a, b, epsilon_hours)` → `(relation_name, base_score)`
- Six relations: `precedes`, `overlaps`, `contains`, `during`, `follows`, `unknown`
- `CAUSAL_PRIORITY`, `RELATION_SCORE` constants already in place

`TSKRTemporalScorerV1` already computes Allen relations between **anomaly windows and the
triggering event** (inside `_score_temporal_posture`), but this is per-FM and embedded in
`tskr_patterns`, not exposed as a standalone ordered graph artifact.

`_stage_g_finalize_manifest` already receives `soe_log`, `alarm_log`, `telemetry_summary`,
and `signal_evidence` (added in Step 1 wiring), so all inputs are available.

---

## Current State Audit

### Already implemented ✅
| Item | Location |
|---|---|
| Full Allen algebra (`allen_relation`, `Interval`, scores, priority) | `orchestrators/temporal_relations.py` |
| Anomaly window → event Allen relation (per-FM, embedded in tskr_patterns) | `tskr_temporal_scorer._score_temporal_posture()` |
| `out_of_boundary_anomalies[].allen_relation` (Stage B signal pre-filter) | `kg_context.json` schema, read by TSKR scorer |
| `soe_log`, `alarm_log`, `telemetry_summary` available in manifest stage | `_stage_g_finalize_manifest` signature |

### Gaps ❌
| Gap | Impact |
|---|---|
| No Allen relation computed for **alarm entries** against the event | Alarm timing (setpoint breach before vs. after event) invisible to ordering |
| No Allen relation computed for **SOE records** against the event | Protection signal sequencing unknown; barrier logic gate uses ordering heuristics only |
| No standalone `allen_relation_map` artifact | No ordered cross-event graph for analyst review or downstream consumption by Steps 4/5 |
| Anomaly window relations buried inside tskr_patterns per-FM — not surfaced as a unified graph | Cannot compare anomaly onset ordering with alarm and SOE sequences |

---

## Input data mapping

### Anomaly windows
- **Source:** `signal_evidence.anomaly_windows` (preferred) or `telemetry_summary.signals[].anomalies[]`
- **Interval:** `[anomaly.timestamp_start, anomaly.timestamp_end]` — use point if no end
- **Node type:** `anomaly`
- **Key fields:** `sensor_id` / `tag_id`, `component_id`, `severity`

### Alarm entries
- **Source:** `alarm_log.alarms[]`
- **Interval:** `[alarm.timestamp, alarm.acknowledged_at]` if `acknowledged_at` present; else point event (`start == end`)
- **Node type:** `alarm`
- **Key fields:** `alarm_id`, `component_id`, `system`, `priority`

### SOE records
- **Source:** `soe_log.records[]`
- **Interval:** always point event (`start == end == record.timestamp`) — SOE records are instantaneous state transitions
- **Node type:** `soe_record`
- **Key fields:** `record_id`, `signal_id`, `component_id`, `transition`, `is_protection_signal`, `priority`

### Current event (anchor)
- **Source:** `event`
- **Interval:** `[event.timestamp_start, event.timestamp_end]`; if no `timestamp_end`, use `timestamp_start` (point)
- **Node type:** `current_event`

---

## Output artifact: `allen_relation_map`

```json
{
  "event_id": "EVT-001",
  "generated_at": "...",
  "event_interval": {
    "start": "2025-04-10T14:23:00Z",
    "end": "2025-04-10T14:23:45Z"
  },
  "summary": {
    "total_nodes": 12,
    "node_type_counts": { "anomaly": 5, "alarm": 4, "soe_record": 3 },
    "causal_nodes": 6,
    "contradiction_nodes": 1,
    "dominant_causal_type": "alarm",
    "timeline_consistent": true
  },
  "nodes": [
    {
      "node_id": "ANOMALY::sensor-T1::0",
      "node_type": "anomaly",
      "source_id": "sensor-T1",
      "component_id": "C1",
      "interval_start": "...",
      "interval_end": "...",
      "is_point_event": false,
      "allen_relation_to_event": "overlaps",
      "allen_base_score": 0.90,
      "causal_candidate": true,
      "severity": "high"
    },
    {
      "node_id": "ALARM::AL-001",
      "node_type": "alarm",
      "source_id": "AL-001",
      "component_id": "C1",
      "interval_start": "...",
      "interval_end": null,
      "is_point_event": true,
      "allen_relation_to_event": "precedes",
      "allen_base_score": 0.75,
      "causal_candidate": true,
      "priority": "critical"
    },
    {
      "node_id": "SOE::SOE-001",
      "node_type": "soe_record",
      "source_id": "SOE-001",
      "component_id": "C2",
      "interval_start": "...",
      "interval_end": "...",
      "is_point_event": true,
      "allen_relation_to_event": "precedes",
      "allen_base_score": 0.75,
      "causal_candidate": true,
      "transition": "trip",
      "is_protection_signal": true
    }
  ]
}
```

**Causal candidate rule:** `allen_relation_to_event` in `{overlaps, contains, precedes}`.  
**Contradiction rule:** `allen_relation_to_event == follows` (entity starts after event resolves).

---

## Workstreams

### WS1 — `_build_allen_relation_map` method

**File:** `orchestrators/rca_reasoning_orchestrator.py`

New static method:

```python
@staticmethod
def _build_allen_relation_map(
    *,
    event: JsonDict,
    telemetry_summary: Optional[JsonDict],
    signal_evidence: Optional[JsonDict],
    alarm_log: Optional[JsonDict],
    soe_log: Optional[JsonDict],
    epsilon_hours: float = 0.5,
) -> JsonDict:
```

Logic:
1. Parse event interval: `event_start = parse_dt(event["timestamp_start"])`,
   `event_end = parse_dt(event.get("timestamp_end")) or event_start`
2. Collect nodes from three sources (see input mapping above)
3. For each node, call `allen_relation(node_interval, event_interval, epsilon_hours)`
4. Mark `causal_candidate = relation in {overlaps, contains, precedes}`
5. Build summary counts
6. Return the `allen_relation_map` dict

**Imports needed:** `from orchestrators.temporal_relations import allen_relation, Interval, RELATION_SCORE`  
(already imported indirectly via TSKR scorer; add explicit import to orchestrator)

### WS2 — Wire into `_stage_g_finalize_manifest`

**File:** `orchestrators/rca_reasoning_orchestrator.py`

In `_stage_g_finalize_manifest`:
```python
allen_relation_map = self._build_allen_relation_map(
    event=...,  # needs to be threaded in (currently not a param)
    telemetry_summary=telemetry_summary,
    signal_evidence=signal_evidence,
    alarm_log=alarm_log,
    soe_log=soe_log,
)
```

Add to `run_manifest`:
- Top-level: `"allen_relation_map": allen_relation_map`
- `artifacts.allen_relation_map`: `{"present": bool, "node_count": N, "causal_node_count": N}`

**Note:** `event` and `telemetry_summary` are not currently parameters of `_stage_g_finalize_manifest`.
They need to be threaded in from `run()`. This is a small signature change.

### WS3 — Thread `event` into `_stage_g_finalize_manifest`

**File:** `orchestrators/rca_reasoning_orchestrator.py`

Add `event: JsonDict` and ensure `telemetry_summary: Optional[JsonDict]` to
`_stage_g_finalize_manifest` signature, and update the call site in `run()`.

### WS4 — New `allen_relation_map` JSON schema

**File:** `schemas/allen_relation_map.json`

Schema for the artifact. Key elements:
- `event_id`, `generated_at`, `event_interval` (required)
- `summary` object: node counts, `timeline_consistent`, `dominant_causal_type`
- `nodes` array: each node with `node_id`, `node_type`, `source_id`, `component_id`,
  `interval_start`, `interval_end`, `is_point_event`, `allen_relation_to_event`,
  `allen_base_score`, `causal_candidate`
- Additional type-specific fields per node_type

### WS5 — Tests

**File:** `unit_tests/test_step2c_allen_relation_map.py`

Test cases:
- Anomaly window that precedes event is `precedes`, `causal_candidate=True`
- Anomaly window that starts before event and overlaps → `overlaps`, causal
- Anomaly window that starts after event resolves → `follows`, `causal_candidate=False`
- Alarm with no `acknowledged_at` treated as point event
- Alarm `precedes` event when `timestamp < event.timestamp_start - epsilon`
- SOE record always `is_point_event=True`
- SOE trip signal that precedes event → `causal_candidate=True`
- Summary `timeline_consistent=False` when any node has `follows` relation
- Manifest contains `allen_relation_map` with correct node count
- Empty inputs (no alarm_log, no soe_log) — only anomaly nodes, no crash
- All missing inputs — empty map returned gracefully

---

## Implementation Sequence

1. WS4: write `allen_relation_map.json` schema first (contracts-first)
2. WS3: thread `event` into `_stage_g_finalize_manifest` signature
3. WS1: implement `_build_allen_relation_map` static method
4. WS2: wire into manifest (top-level + artifacts block)
5. WS5: write tests
6. Run targeted tests → full suite → update backlog + metamodel

---

## Acceptance Criteria

- `allen_relation_map.nodes` contains one entry per anomaly window, alarm, and SOE record
- Each node has a valid `allen_relation_to_event` from the six-relation vocabulary
- `causal_candidate` is `True` iff relation is `overlaps`, `contains`, or `precedes`
- `summary.timeline_consistent` is `False` iff any node has `follows` relation
- `run_manifest.allen_relation_map` is present
- `run_manifest.artifacts.allen_relation_map.present` is set correctly
- Empty/absent optional inputs produce an empty-but-valid map (no crash)
- Full test suite passes

---

## Step 2c Definition of Done

### Required inputs
| Input | Mandatory? | Role in Step 2c |
|---|---|---|
| `event` | Mandatory | Anchor interval for all relations |
| `telemetry_summary` (or `signal_evidence`) | Mandatory | Source of anomaly windows |
| `alarm_log` | Conditional P0 | Source of alarm entry intervals |
| `soe_log` | Conditional P0 | Source of SOE point events |

### Required outputs/artifacts
- `run_manifest.allen_relation_map` — full relation map with nodes
- `run_manifest.artifacts.allen_relation_map` — presence + summary counts

### Residual gaps (intentionally deferred)
- Cross-entity pairwise relations (alarm vs. anomaly, SOE vs. alarm) — O(n²),
  deferred until analyst review shows demand
- Step 2a (Architectural Search)
- Step 2d (fleet/industry OE similar events)

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Alarm has `timestamp` only — no duration → always treated as `precedes` or `during` | Use epsilon window; treat as point event; label `is_point_event=True` in output |
| SOE log has clock-sync issues (flagged in Step 1) | Check `soe_log.quality.clock_sync_ok`; if `False`, set `allen_relation_to_event = "unknown"` and `causal_candidate = False` for all SOE nodes |
| Large SOE logs (hundreds of records) — performance | Cap at `max_soe_nodes` (default 50, configurable); take highest-priority records first |
| `_stage_g_finalize_manifest` already has a long signature | `event` and `telemetry_summary` are additive optional params; existing callers need no change |
