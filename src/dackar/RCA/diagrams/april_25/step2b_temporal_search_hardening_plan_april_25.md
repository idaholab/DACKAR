# Step 2b — Temporal Search Hardening Plan (Revised)

**Date:** 2026-04-25 (revised after architecture review)  
**Step:** Step 2 — KG Expansion / sub-step 2b: Temporal Search  
**Out of scope for this phase:** Step 2a (Architectural Search), Step 2c (Allen relations), Step 2d (similar event / fleet OE)  
**Source of truth:** `rca_metamodel.md` §2b, `kg_context_builder.py`, `schemas/kg_context.json`

---

## Architecture Clarification

Plant databases (CMMS, operator logs, surveillance systems, etc.) are accessed via APIs
**outside the orchestrator** and their records are imported into:

- **Neo4j KG** — structured nodes (`abnormal_event`, `element_usage`, `failure_mode`) linked
  by typed edges. This is the primary store for past events.
- **Chroma** — document text corpus (CRs, WOs, ECAs, RCAs, operator logs as natural language).
  Queried in **Step 3** for documentary pattern recognition, not Step 2b.

The **`Neo4jKGContextBuilder._fetch_past_events()`** method already:
- Issues a single Cypher query across all scoped asset_ids, component_ids, and failure_mode_ids
- Applies a configurable `past_event_window_days` cutoff (default 3650 days / 10 years)
- Computes a `priority_score` per event: asset match (+10), component match (+8), FM match (+9),
  plus a recency term (up to +10, inversely proportional to `days_before_current_event / 30`)
- Sorts by `(−priority_score, time_distance_days)` before returning

The **CMMS adapter** (`cmms_adapter`) is a live supplement only — it fetches CRs/WOs that are
too recent to have been ingested into the KG yet. It is not the primary source for Step 2b.

---

## Current State Audit

### Already implemented ✅
| Item | Location |
|---|---|
| Neo4j KG query for past `abnormal_event` nodes across all scoped assets/components | `kg_context_builder._fetch_past_events()` |
| Surveillance records and operator logs included (if imported as `abnormal_event` nodes) | Cypher query in `_fetch_past_events()` — no event_type filter |
| `priority_score` composite (asset + component + FM match + recency) | `_fetch_past_events()` lines 809–824 |
| `time_distance_days` and `days_before_current_event` fields on each past event | `_fetch_past_events()` → `_compute_time_distance_days()` |
| `past_event_window_days` config parameter (default 3650 days) | `KGContextBuilderConfig` |
| Global top-N cap via `max_past_events` (default 10) | `KGContextBuilderConfig` |
| CMMS live supplement (CRs and WOs not yet in KG) | `_augment_kg_context_with_cmms_past_events()` |

### Gaps ❌
| Gap | Impact |
|---|---|
| **No per-component organization**: `past_events` is a single flat list globally capped at 10; components with many events crowd out components with fewer | Pattern recognition and candidate generation cannot efficiently scope to individual components; sparsely-represented components lose history |
| **No two-tier window tagging**: only one window (`past_event_window_days = 3650`); no distinction between operational precursor window (e.g. 180 days) and extended historical | Steps 3/4 cannot distinguish between high-relevance recent events and deep historical context |
| **No `temporal_search_summary` in manifest**: only raw aggregate counts in `seed_context`; nothing structured surfaces in the run manifest | Auditability and replayability gap for Step 2b; analyst cannot see per-component history coverage at review time |
| **Per-component top-N not enforced**: global cap means the query returns ≤10 events total, not ≤N per component | A single component with 10 matching past events consumes the entire budget |

---

## In Scope

- Add per-component grouping of `past_events` into `kg_context.seed_context.per_component_past_events`
- Add per-component top-N cap (configurable, default 5 per component)
- Tag each past event with `in_precursor_window` (boolean) and `window_tier` (enum)
  using a configurable shorter `precursor_window_days` (default 180 days)
- Build a structured `temporal_search_summary` in `kg_context.seed_context`
- Surface `temporal_search_summary` in `run_manifest.pipeline_config.temporal_search`
- Update `kg_context.json` schema to document the new fields
- Tests for all new behaviors

## Out of Scope

- Changing the Neo4j Cypher query or KG data model
- Chroma historical query per component (correctly belongs to Step 3)
- CMMS adapter changes (surveillance/operator log injection from live CMMS)
- Step 2a (Architectural Search), Step 2c (Allen relations), Step 2d (fleet/industry OE)

---

## Workstreams

### WS1 — Two-tier window tagging on `past_events`

**File:** `orchestrators/rca_reasoning_orchestrator.py`

In `_augment_kg_context_with_cmms_past_events` (and after KG past events are available),
add a post-processing pass over `kg_context["past_events"]`:

```python
precursor_window_days = int(self.config.extra.get("precursor_window_days", 180))
for pe in past_events:
    d = pe.get("days_before_current_event")
    if d is None:
        pe["in_precursor_window"] = None
        pe["window_tier"] = "unknown"
    elif d <= precursor_window_days:
        pe["in_precursor_window"] = True
        pe["window_tier"] = "primary"
    elif d <= precursor_window_days * 2:
        pe["in_precursor_window"] = False
        pe["window_tier"] = "extended"
    else:
        pe["in_precursor_window"] = False
        pe["window_tier"] = "historical"
```

**Schema update (`kg_context.json`):**
- Add `in_precursor_window: boolean | null` to `past_events` item
- Add `window_tier: string` (enum: primary, extended, historical, unknown)

### WS2 — Per-component indexing with per-component top-N

**File:** `orchestrators/rca_reasoning_orchestrator.py`

After the `past_events` list is finalized (KG events + CMMS injection + tier tagging),
build a per-component index:

```python
per_component_top_n = int(self.config.extra.get("per_component_past_event_top_n", 5))
per_component: dict = {}
for pe in sorted(past_events, key=lambda x: -float(x.get("priority_score") or 0)):
    cid = str(pe.get("component_id") or "_no_component")
    bucket = per_component.setdefault(cid, [])
    if len(bucket) < per_component_top_n:
        bucket.append(pe.get("event_id"))
```

Store as `kg_context["seed_context"]["per_component_past_events"]` — dict of
`{component_id: [event_id, ...]}` (IDs only; full records remain in `past_events`).

### WS3 — `temporal_search_summary` in `seed_context`

**File:** `orchestrators/rca_reasoning_orchestrator.py`

After building the index, compute and store:

```json
{
  "component_count_with_history": 3,
  "total_past_event_count": 8,
  "in_window_count": 4,
  "out_of_window_count": 4,
  "unknown_window_count": 0,
  "precursor_window_days_used": 180,
  "source_breakdown": { "kg": 6, "cmms_cr": 1, "cmms_wo": 1 },
  "per_component_top_n_used": 5
}
```

`source_breakdown` is determined by `event_id` prefix convention:
- `"CMMS::CR::"` → `cmms_cr`
- `"CMMS::WO::"` → `cmms_wo`
- everything else → `kg`

### WS4 — `temporal_search` in run manifest

**File:** `orchestrators/rca_reasoning_orchestrator.py`

In `_stage_g_finalize_manifest`, extract `temporal_search_summary` from
`kg_context["seed_context"]` and add to manifest:

```python
temporal_search = (kg_context.get("seed_context") or {}).get("temporal_search_summary") or {}
manifest["pipeline_config"]["temporal_search"] = temporal_search
```

### WS5 — Tests

**File:** `unit_tests/test_step2b_temporal_search.py`

Test cases:
- Events ≤ 180 days tagged `in_precursor_window=True`, `window_tier="primary"`
- Events 181–360 days tagged `in_precursor_window=False`, `window_tier="extended"`
- Events > 360 days tagged `in_precursor_window=False`, `window_tier="historical"`
- Events with no `days_before_current_event` tagged `window_tier="unknown"`
- Per-component index groups correctly; component with 8 events capped to 5
- `temporal_search_summary` counts match the tagged events
- Source breakdown correctly identifies CMMS vs KG events by ID prefix
- Manifest contains `pipeline_config.temporal_search` with correct summary
- Full test suite stays green

---

## Implementation Sequence

1. WS1: tier-tagging pass (non-breaking — new fields added to existing objects)
2. WS2: per-component index in `seed_context`
3. WS3: `temporal_search_summary` in `seed_context`
4. WS4: wire into manifest
5. WS5: schema updates (`kg_context.json`)
6. WS6: tests
7. Run targeted tests → full suite → update backlog + metamodel

---

## Acceptance Criteria

- Every `past_events` entry carries `in_precursor_window` and `window_tier`
- `kg_context.seed_context.per_component_past_events` correctly indexes events by component
- Per-component top-N cap is applied and configurable
- `kg_context.seed_context.temporal_search_summary` present and correctly computed
- `run_manifest.pipeline_config.temporal_search` matches `temporal_search_summary`
- Full test suite passes (no regressions)

---

## Step 2b Definition of Done

### Required inputs
| Input | Schema | Mandatory? | Step 2b role |
|---|---|---|---|
| `kg_context.past_events` | `kg_context.json` | Mandatory | Source of per-component events (already populated by KG builder) |
| `kg_context.components` | `kg_context.json` | Mandatory | Defines component scope for per-component grouping |
| `cmms_context` | `cmms_context.json` | P1 (live supplement) | Injects recent CRs/WOs not yet in KG |
| `event.timestamp_start` | `event.json` | Mandatory | Anchor for precursor window tier calculation |

### Required outputs/artifacts
- `kg_context.past_events[*].in_precursor_window` (bool)
- `kg_context.past_events[*].window_tier` (primary / extended / historical / unknown)
- `kg_context.seed_context.per_component_past_events` (dict: component_id → [event_id])
- `kg_context.seed_context.temporal_search_summary`
- `run_manifest.pipeline_config.temporal_search`

### Residual gaps (intentionally deferred)
- Chroma historical query per component → Step 3 pattern recognition
- Step 2a Architectural Search → future phase
- Step 2c Allen relation computation → future phase
- Step 2d fleet/industry OE similar event identification → future phase

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `days_before_current_event` is null for KG events without timestamps | Tag as `window_tier="unknown"`, `in_precursor_window=None`; count separately in summary |
| Per-component cap of 5 is too low for the primary asset component | Make it configurable via `per_component_past_event_top_n`; primary asset may appear under multiple component_ids |
| Source breakdown by ID prefix is fragile if prefix conventions change | Encapsulate in a single `_classify_past_event_source(event_id)` helper — easy to update |
| Existing tests that assert `kg_context` structure may fail if new fields are unexpected | New fields are additive; existing schema has `additionalProperties: false` — must add to schema first |
