# Step 2d — Similar Event Identification Plan
**Date:** 2026-04-25  
**Status:** Planning  
**Predecessor:** Step 2b (Temporal Search), Phase 3b (Scope-Expansion Hooks), Step 2c (Allen Relation Map)

---

## 1. Problem Statement

The `similar_event_list` artifact is currently a hard-wired stub:
```json
{ "status": "not_implemented", "events": [], "provenance": {...} }
```

This means OE-based reinstatement (`_run_oe_based_reinstatement`) operates without any external corroboration — it can reinstate a ruled-out candidate but cannot cite a comparable past event as evidence. Analysts receive no context from fleet or industry history. The Step 6 unresolved-gap logic cannot flag missing OE lookup as a confidence risk.

### Three data tiers — distinct mechanisms

| Tier | Source | Mechanism | Latency |
|---|---|---|---|
| **Plant** | Plant CMMS / KG `past_events` | Direct query of local database already in memory (`kg_context.past_events` + Step 2b enrichment) | In-process, no I/O |
| **Fleet** | Utility fleet OE database | Pluggable external API backed by an LLM fine-tuned on fleet operating experience | Network call, async-safe |
| **Industry** | INPO SOER, EPRI reports, NRC LERs | Same pluggable API, separate endpoint or query parameter | Network call, async-safe |

---

## 2. Current State

### What exists

- `similar_event_list.json` schema (Step 2d stub): `status`, `events[]`, `provenance`.
- `kg_context.past_events[]` populated by Step 2b with `priority_score`, `in_precursor_window`, `window_tier`, `matched_component_ids`, `matched_failure_mode_ids`.
- `kg_context.seed_context.per_component_past_events` index: `{component_id → [event_id, ...]}` capped to top-5 by priority score.
- OE reinstatement path in causality engine reads `oe_evidence_weight` from signal evidence — currently no fleet/industry anchor.
- `run_manifest.artifacts.similar_event_list.status = "not_implemented"`.

### What is missing

1. **Plant-tier engine:** convert `past_events` into scored `SimilarEvent` records using match dimensions (component, failure mode, event type, actuation type, proximity window).
2. **Fleet/Industry adapter interface:** abstract callable (`SimilarEventAdapter`) that the orchestrator can call with a structured query and receive `events[]`.
3. **LLM-backed adapter implementation:** concrete implementation that calls the fine-tuned LLM API (INPO/EPRI/NRC endpoints), handles timeouts/errors, returns normalised records.
4. **Schema upgrade:** richer event record structure (match dimensions, resolution, contributing categories, root cause label, lessons_learned_ref).
5. **Manifest wiring:** replace stub with live artifact; update `artifacts.similar_event_list` summary.
6. **OE reinstatement linkage:** surface matched similar events as `oe_evidence` in reinstatement rationale.
7. **Unresolved-gap integration:** emit a gap when no plant-level match is found but fleet/industry lookup was skipped/degraded.
8. **Tests:** 20+ tests for plant engine, adapter interface, manifest wiring.

---

## 3. Architecture

### 3.1 Data flow

```
run() ──► _build_similar_event_list(
              event,
              kg_context,
              causality_candidates,   ← top retained candidates drive query terms
              similar_event_adapter   ← optional; None → fleet/industry skipped
          )
         │
         ├─► _query_plant_past_events(...)     [always runs; pure in-memory]
         │     KG past_events × candidate match dimensions → SimilarEvent[]
         │
         ├─► adapter.query_fleet(query)        [only when adapter provided]
         │     returns SimilarEvent[] from fleet LLM API
         │
         └─► adapter.query_industry(query)     [only when adapter provided]
               returns SimilarEvent[] from industry LLM API (INPO/EPRI/NRC)
```

The result is a single merged `similar_event_list` with `status = "partial"` (plant only) or `"complete"` (all three tiers ran without error).

### 3.2 Plant-tier matching dimensions

Scored against each `kg_context.past_events[]` entry:

| Dimension | Match condition | Score contribution |
|---|---|---|
| Component exact match | `component_id ∈ matched_component_ids` | +0.40 |
| Failure mode match | `fm_id ∈ matched_failure_mode_ids` of any top candidate | +0.25 |
| Event type match | `past_event.event_type == current_event.event_type` | +0.15 |
| Actuation type match | `past_event.actuation_type == current_event.actuation_type` | +0.10 |
| In precursor window | `in_precursor_window == True` (primary tier) | +0.10 |

`confidence_weight` = normalised total (0–1), capped at 1.0.  
`source_level = "plant"`, `source_db = "plant_kg"`.  
Top-N = 5 plant events by `confidence_weight` (configurable via `extra.step2d_plant_top_n`, default 5).

### 3.3 Adapter interface

```python
class SimilarEventAdapter(Protocol):
    """Pluggable interface for fleet and industry OE queries."""

    def query(
        self,
        *,
        level: Literal["fleet", "industry"],
        asset_id: str,
        component_ids: List[str],
        failure_mode_ids: List[str],
        event_type: Optional[str],
        actuation_type: Optional[str],
        max_results: int = 5,
        timeout_seconds: float = 10.0,
    ) -> List[JsonDict]:
        """Return list of SimilarEvent dicts (schema-compatible with similar_event_list.json)."""
        ...
```

The orchestrator checks `hasattr(self, "similar_event_adapter")` (or an injected arg). When absent → fleet/industry tiers silently skip; status stays `"partial"`.

### 3.4 LLM-backed adapter (concrete implementation)

Located at `src/dackar/RCA/adapters/llm_oe_adapter.py`.

```
LLMOEAdapter
├── __init__(base_url, api_key, fleet_endpoint, industry_endpoint, timeout)
├── query(level, asset_id, component_ids, failure_mode_ids, ...)
│     ├── Build structured prompt from query terms
│     ├── POST to {fleet_endpoint | industry_endpoint}
│     ├── Parse response → list[SimilarEventRecord]
│     └── Catch (timeout, HTTPError) → return [] with degraded_flag=True
└── _build_query_prompt(level, ...) → str
      Generates a structured retrieval prompt:
      "Retrieve similar events for component {cid} with failure mode {fm}
       event_type={et}, actuation_type={at} from {INPO|EPRI|NRC} databases.
       Return up to {N} results as JSON with fields: event_id, date, summary,
       root_cause_label, lessons_learned_ref, resolution."
```

**Error handling contract:**
- Timeout or HTTP error → returns `[]` with `provenance.degraded_tiers` entry for that level.
- Non-JSON response → same.
- Partial results (fewer than requested) → accepted as-is; status remains `"complete"` if the call succeeded.

### 3.5 Schema upgrade (`similar_event_list.json`)

Additional event record fields:

| Field | Type | Source |
|---|---|---|
| `actuation_type` | string/null | from plant event or LLM response |
| `match_dimensions` | object | scored dimensions and their contributions |
| `root_cause_label` | string/null | LLM response or KG `fm_id` |
| `resolution` | string/null | KG `resolved` field or LLM response |
| `lessons_learned_ref` | string/null | LLM response reference (SOER/LER number) |
| `contributing_categories` | array[string] | categories A–L implied by root cause |
| `window_tier` | string/null | `primary|extended|historical|unknown` (plant only) |

Top-level additions:

| Field | Type | Purpose |
|---|---|---|
| `summary.plant_count` | int | events from plant tier |
| `summary.fleet_count` | int | events from fleet tier |
| `summary.industry_count` | int | events from industry tier |
| `summary.degraded_tiers` | array[string] | tiers that failed/timed out |
| `query_terms` | object | the query dimensions used (for auditability) |

### 3.6 OE reinstatement linkage

`refine_with_evidence` already computes `oe_evidence_weight` from signal evidence.  
When `similar_event_list` is available, the orchestrator (in `_stage_g_finalize_manifest`) injects matched plant/fleet/industry events into the `oe_reinstatement_evidence` block of any candidate where a matching `component_id` or `fm_id` appears in `similar_event_list.events`. The reinstatement rationale then cites the matched event IDs and their `confidence_weight`.

### 3.7 Unresolved-gap integration

`_build_unresolved_gaps` receives `similar_event_list` (new parameter). Emits a gap when:
- `plant_count == 0`: no plant history available for primary candidate component.
- `degraded_tiers` contains `"fleet"` or `"industry"`: OE lookup was attempted but failed.

---

## 4. Workstreams

| WS | Component | Deliverable |
|---|---|---|
| **WS1** | Schema upgrade | `similar_event_list.json` — new fields, `summary` block, `query_terms`, `degraded_tiers` |
| **WS2** | Plant engine | `_query_plant_past_events` static method + `_build_similar_event_list` orchestrator method |
| **WS3** | Adapter interface | `SimilarEventAdapter` Protocol in `adapters/similar_event_adapter.py` |
| **WS4** | LLM adapter | `LLMOEAdapter` in `adapters/llm_oe_adapter.py` with prompt builder + error handling |
| **WS5** | Manifest wiring | Replace stub; update `artifacts.similar_event_list` summary; wire `similar_event_list` into `_build_unresolved_gaps` |
| **WS6** | OE reinstatement linkage | Inject matched events into candidate reinstatement evidence in `_stage_g_finalize_manifest` |
| **WS7** | Tests | `test_step2d_similar_events.py` — 20+ tests |
| **WS8** | Docs | Backlog + metamodel update |

---

## 5. Design Decisions

### D1 — Plant tier always runs; fleet/industry are opt-in
**Rationale:** The plant database (`kg_context.past_events`) is always in memory at the point Step 2d runs. Making it always execute ensures every run produces at least partial OE context at zero latency cost. Fleet/industry require an injected adapter and are async-safe to skip.

### D2 — Adapter is injected, not constructed inside the orchestrator
**Rationale:** The orchestrator should not own network credentials. `LLMOEAdapter` is constructed by the caller and injected via `RCAReasoningOrchestrator.set_similar_event_adapter(adapter)` or as a constructor kwarg. This keeps the orchestrator testable without live LLM calls.

### D3 — LLM adapter uses structured prompt, not free text
**Rationale:** The fine-tuned model is expected to receive a structured JSON-like prompt with clearly labelled query dimensions (`component_id`, `failure_mode_id`, `event_type`, `actuation_type`). Free-text queries produce inconsistent extraction. The adapter owns the prompt template; the orchestrator only passes normalised query terms.

### D4 — `confidence_weight` is tier-discounted
**Rationale:** Plant events are highest-trust (direct history), fleet are mid-trust (same reactor design family, different site), industry are lowest-trust (general LER population). Tier discount multipliers:
- Plant: 1.0 × match score
- Fleet: 0.80 × match score  
- Industry: 0.60 × match score

These multipliers are configurable via `extra.step2d_confidence_multipliers`.

### D5 — LLM adapter timeout and error → silent degraded, not crash
**Rationale:** A network timeout during a nuclear RCA should never block the local pipeline. Degraded tiers are recorded in `provenance.degraded_tiers` for analyst visibility; the local plant result is still emitted.

### D6 — `status` semantics
| Value | Meaning |
|---|---|
| `not_implemented` | No adapter available and plant query not run (should not occur after this implementation) |
| `partial` | Plant tier ran successfully; fleet/industry skipped (no adapter) or all degraded |
| `complete` | All configured tiers ran without error (fleet/industry optional — if no adapter, plant-only counts as complete for plant-only config) |

### D7 — Top candidates drive query terms
**Rationale:** Running a query for every candidate component in scope would generate too many API calls. Only the top-3 retained candidates by composite score contribute `component_ids` and `failure_mode_ids` to the fleet/industry query. Configurable via `extra.step2d_query_top_n_candidates` (default 3).

---

## 6. Schema Changes

### `similar_event_list.json` — additions

```json
"query_terms": {
  "type": "object",
  "properties": {
    "asset_id": { "type": "string" },
    "component_ids": { "type": "array", "items": { "type": "string" } },
    "failure_mode_ids": { "type": "array", "items": { "type": "string" } },
    "event_type": { "type": ["string", "null"] },
    "actuation_type": { "type": ["string", "null"] }
  }
},
"summary": {
  "type": "object",
  "properties": {
    "plant_count": { "type": "integer" },
    "fleet_count": { "type": "integer" },
    "industry_count": { "type": "integer" },
    "total_count": { "type": "integer" },
    "degraded_tiers": { "type": "array", "items": { "type": "string" } },
    "any_plant_match": { "type": "boolean" }
  }
}
```

Event record additions (all optional):
```json
"match_dimensions":      { "type": "object", "additionalProperties": true },
"root_cause_label":      { "type": ["string", "null"] },
"resolution":            { "type": ["string", "null"] },
"lessons_learned_ref":   { "type": ["string", "null"] },
"contributing_categories": { "type": "array", "items": { "type": "string" } },
"window_tier":           { "type": ["string", "null"] },
"actuation_type":        { "type": ["string", "null"] }
```

---

## 7. Test Plan (`test_step2d_similar_events.py` — target: 22 tests)

### Plant engine (WS2) — 8 tests
- `test_plant_engine_component_match_scores_correctly` — component exact match adds 0.40.
- `test_plant_engine_fm_match_adds_score` — failure mode match adds 0.25.
- `test_plant_engine_event_type_match` — event type boost.
- `test_plant_engine_actuation_type_match` — actuation type boost.
- `test_plant_engine_in_precursor_window_boost` — `in_precursor_window=True` adds 0.10.
- `test_plant_engine_top_n_capped` — only top-5 returned.
- `test_plant_engine_no_past_events_returns_empty` — empty `past_events` → empty list, status `partial`.
- `test_plant_engine_tier_discount_applied` — confidence_weight ≤ 1.0 after tier discount.

### Adapter interface (WS3) — 3 tests
- `test_adapter_protocol_satisfied_by_mock` — mock implementing Protocol passes isinstance check.
- `test_no_adapter_skips_fleet_industry` — `similar_event_list` has `fleet_count=0, industry_count=0` without adapter.
- `test_adapter_timeout_returns_partial` — adapter raises timeout → `degraded_tiers` contains tier name.

### LLM adapter (WS4) — 4 tests
- `test_llm_adapter_builds_structured_prompt` — query terms present in generated prompt.
- `test_llm_adapter_parses_valid_response` — valid JSON response → list of SimilarEvent dicts.
- `test_llm_adapter_handles_http_error` — HTTP 500 → returns `[]`, no exception propagated.
- `test_llm_adapter_handles_malformed_json` — non-JSON body → returns `[]`.

### Manifest wiring (WS5) — 4 tests
- `test_manifest_similar_event_list_not_stub` — after WS2, `status != "not_implemented"`.
- `test_manifest_artifacts_summary_fields` — `plant_count`, `fleet_count`, `any_plant_match` in `artifacts.similar_event_list`.
- `test_unresolved_gaps_emits_when_no_plant_match` — `plant_count=0` → gap entry in `unresolved_gaps`.
- `test_unresolved_gaps_emits_when_fleet_degraded` — `degraded_tiers=["fleet"]` → gap entry.

### OE reinstatement linkage (WS6) — 3 tests
- `test_oe_reinstatement_cites_plant_event` — matched plant event appears in reinstatement evidence.
- `test_oe_reinstatement_no_match_no_citation` — no component match → reinstatement evidence unchanged.
- `test_similar_event_confidence_feeds_oe_weight` — high-confidence plant match raises effective OE weight.

---

## 8. Definition of Done

| # | Criterion |
|---|---|
| 1 | `similar_event_list.status` is `"partial"` or `"complete"` on every run (never `"not_implemented"`) |
| 2 | Plant tier always runs; produces scored records from `kg_context.past_events` |
| 3 | Fleet and industry tiers execute when `SimilarEventAdapter` is injected |
| 4 | `LLMOEAdapter` builds structured prompt; handles timeout/HTTP error gracefully |
| 5 | `summary.plant_count`, `fleet_count`, `industry_count`, `degraded_tiers` present |
| 6 | `query_terms` present (component_ids, failure_mode_ids, event_type, actuation_type) |
| 7 | Matched plant events surfaced in OE reinstatement rationale |
| 8 | `unresolved_gaps` emits when plant_count=0 or a tier is degraded |
| 9 | `artifacts.similar_event_list` summary updated in manifest |
| 10 | 22 tests in `test_step2d_similar_events.py` pass |
| 11 | Full suite ≥ 980 tests pass, zero regressions |

---

## 9. Step Readiness Matrix (Target State Post-WS8)

| Dimension | Before | After |
|---|---|---|
| Plant OE data | `past_events` present but not surfaced as similar events | Scored, ranked, top-5 in artifact |
| Fleet OE data | Not implemented | Available via `LLMOEAdapter` (INPO/EPRI fine-tuned endpoint) |
| Industry OE data | Not implemented | Available via `LLMOEAdapter` (NRC LER endpoint) |
| OE reinstatement anchor | No external event citation | Matched events cited in reinstatement rationale |
| Degraded-mode visibility | No signal | `degraded_tiers` list + unresolved-gap entry |
| `similar_event_list` status | `not_implemented` | `partial` (plant only) or `complete` (all tiers) |

---

## 10. Open Questions (to resolve during implementation)

| # | Question | Default assumption |
|---|---|---|
| Q1 | What is the fleet LLM API base URL and auth scheme? | Configurable via `LLMOEAdapter.__init__`; env-var injected in production |
| Q2 | Does the LLM return structured JSON or free text requiring parsing? | Structured JSON; adapter prompt instructs model to return JSON array |
| Q3 | Should fleet and industry queries run sequentially or in parallel? | Sequential for simplicity; parallelism deferred to a follow-up |
| Q4 | Should `contributing_categories` on returned events be LLM-predicted or analyst-entered? | LLM-predicted from summary text; field is informational only (not scored) |
| Q5 | Top-N candidates for query terms: 3 or configurable? | Configurable via `extra.step2d_query_top_n_candidates`, default 3 |
