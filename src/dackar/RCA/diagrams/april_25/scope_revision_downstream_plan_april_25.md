# Scope-Revision Downstream Propagation — Implementation Plan
**Date:** 2026-04-25  
**Author:** RCA Workflow Development  
**Status:** Planned

---

## 1. Problem Statement

The scope revision lifecycle (Step 0) can write decisions but the decisions have no downstream enforcement.

### What works today
- `_stage_a_build_run_context` seeds `scope_snapshot.component_ids` from the event report + CMMS/SOE extras.
- `_detect_scope_expansion_signals` identifies out-of-scope causal components (Allen map, propagation chains, novel TSKR patterns) and stores them as `expansion_suggestions[].analyst_decision = "pending"`.
- `apply_scope_revision` records accepted/deferred/rejected revisions and bumps `active_scope_version`.
- `_build_scope_revision_summary` surfaces the version number in the manifest.

### The gap
`causality_engine.generate()` is called **without consulting the approved scope boundary**. All KG failure modes feed through unconditionally — out-of-scope candidates are scored, ranked, and reported to the analyst as if they were valid hypotheses. The analyst's scope decision has zero mechanical effect on the next run's candidate set.

Additionally, when an analyst accepts an expansion suggestion (e.g., `SEX::ALLEN::valve-001`), there is no bridge method to (a) mark the suggestion as accepted *and* (b) call `apply_scope_revision` with the updated component list in a single atomic step.

---

## 2. Current State

| Mechanism | Status |
|---|---|
| Initial scope snapshot built | ✅ |
| Expansion signals detected from Allen map / chains / TSKR | ✅ |
| Analyst accepts/rejects via `apply_scope_revision` | ✅ |
| Latest approved revision stored in `scope_revisions[]` | ✅ |
| `active_scope_version` bumped on accept | ✅ |
| `generate()` filtered by approved boundary | ❌ Missing |
| Auto-merge `changed_boundary.added_component_ids` into snapshot | ❌ Missing |
| `resolve_expansion_suggestion` convenience method | ❌ Missing |
| Scope filter surfaced in manifest | ❌ Missing |

---

## 3. Design

### 3.1 Activation logic

The scope filter activates **only when `active_scope_version > 0`** (i.e., the analyst has explicitly accepted at least one scope revision beyond the initial intake).

- Version 0 = initial intake — treated as discovery mode; no filter applied.
- Version ≥ 1 = analyst has acted — filter enforced.

Rationale: first-run is a blind discovery pass against the full KG. It would be counterproductive to silently drop candidates the analyst has not yet seen.

### 3.2 Approved boundary extraction

`_resolve_approved_scope_boundary(run_context)` reads `scope_management.scope_revisions[]` backwards to find the latest `analyst_decision == "accepted"` entry and returns its `scope_snapshot.component_ids` as a `frozenset[str]` (lower-cased, stripped). Returns `None` when `active_scope_version == 0` or no accepted revision has a non-empty component list.

### 3.3 Scope boundary filter — soft rule-out

`_apply_scope_boundary_filter(candidates, approved_boundary, scope_version)` moves out-of-scope candidates to `candidates["ruled_out"]` rather than deleting them:

```
candidate.component_id ∉ approved_boundary
    → move to ruled_out[]
    → reason_code: "scope_filtered"
    → reason: "component '{cid}' not in approved scope v{N} — analyst can expand scope to reinstate"
    → hard_gate: false  (not a hard-gate elimination; analyst-controlled)
```

Candidates with no `component_id` are not filtered (defensive).

This preserves full audit trail and allows the analyst to re-expand scope to reinstate candidates.

### 3.4 `apply_scope_revision` enhancement

When `revision_input.changed_boundary.added_component_ids` is non-empty and `analyst_decision == "accepted"`, the method auto-builds the new `scope_snapshot` by:
1. Copying the latest accepted revision's `scope_snapshot`.
2. Merging `added_component_ids` into `scope_snapshot.component_ids` (union, preserving order, deduplicating).
3. Removing any `removed_component_ids` from the set.

This removes the burden from callers of reconstructing the full snapshot from scratch.

### 3.5 `resolve_expansion_suggestion` convenience method

Bridges the expansion-suggestion write path to the scope revision lifecycle:

```python
def resolve_expansion_suggestion(
    self, *, run_id, run_context, signal_id, decision, rationale=None, persist=True
) -> JsonDict:
```

1. Find the suggestion in `expansion_suggestions[]` by `signal_id`.
2. Set `analyst_decision = decision` (accepted / deferred / rejected) and `resolution_timestamp`.
3. If `decision == "accepted"`: call `apply_scope_revision` with:
   - `trigger = "expansion_suggestion_accepted"`
   - `analyst_decision = "accepted"`
   - `changed_boundary.added_component_ids = suggestion.suggested_component_ids`
4. If `decision == "rejected"`: no scope change; just mark the suggestion.
5. Return the updated `run_context`.

### 3.6 Manifest surfacing

Add a `scope_filter` sub-block to `run_manifest.artifacts`:

```json
"scope_filter": {
  "applied": true,
  "approved_scope_version": 1,
  "approved_boundary_size": 8,
  "filtered_count": 3,
  "filtered_component_ids": ["valve-007", "pump-012", "sensor-099"]
}
```

When `applied == false`: `filtered_count = 0`, `filtered_component_ids = []`.

---

## 4. Data Flow

```
run_context (from prior run or initial intake)
    ↓
_resolve_approved_scope_boundary(run_context)
    → None (version 0): skip filter
    → frozenset[str] (version ≥ 1): proceed to filter

causality_engine.generate(...)
    → causality_candidates (all KG failure modes)

_apply_scope_boundary_filter(candidates, boundary, scope_version)
    → out-of-scope candidates → ruled_out[reason_code="scope_filtered"]
    → in-scope candidates → remain in candidates[]

refine_with_evidence(...)
    (only in-scope candidates refined)

_stage_g_finalize_manifest(...)
    → scope_filter block in artifacts
```

---

## 5. Workstreams

### WS1 — Helpers in `rca_reasoning_orchestrator.py`

**`_resolve_approved_scope_boundary(run_context) → Optional[FrozenSet[str]]`**
- Static method
- Walk `scope_revisions[]` backwards; find latest `analyst_decision == "accepted"`
- Return `frozenset(c.strip().lower() for c in snapshot.component_ids)` — or `None` if version 0 or empty

**`_apply_scope_boundary_filter(candidates, approved_boundary, scope_version) → JsonDict`**
- Static method
- Iterate `candidates["candidates"]`; collect out-of-scope by `component_id`
- Build rule-out entry with `reason_code = "scope_filtered"`, `scope_version`, `rationale`
- Move to `candidates.setdefault("ruled_out", [])` (append)
- Rebuild `candidates["candidates"]` without filtered entries
- Set `candidates["scope_filter_applied"] = True` and `candidates["scope_filter_version"] = scope_version`

### WS2 — `apply_scope_revision` enhancement

- After `analyst_decision == "accepted"` validation:
  - If caller provides no explicit `scope_snapshot` in `revision_input`:
    - Copy latest accepted snapshot
    - Union in `changed_boundary.get("added_component_ids", [])`
    - Subtract `changed_boundary.get("removed_component_ids", [])`
    - Store as `revision_row["scope_snapshot"]`

### WS3 — New `resolve_expansion_suggestion` method

- Instance method (needs `self.artifact_store`)
- Finds suggestion by `signal_id`; raises `ValueError` if not found
- Applies `analyst_decision`; if accepted → delegates to `apply_scope_revision`
- `persist=True` by default (saves updated `run_context`)
- Returns updated `run_context`

### WS4 — Wire filter into `run()` + manifest

In `run()`, after `causality_engine.generate(...)` and before `refine_with_evidence`:

```python
approved_boundary = self._resolve_approved_scope_boundary(run_context)
if approved_boundary is not None:
    scope_version = run_context["scope_management"]["active_scope_version"]
    causality_candidates = self._apply_scope_boundary_filter(
        causality_candidates, approved_boundary, scope_version
    )
```

In `_stage_g_finalize_manifest`, add `scope_filter` block to `artifacts`.

### WS5 — Tests: `test_scope_revision_downstream.py` (~20 tests)

| Test | What it verifies |
|---|---|
| `test_resolve_boundary_version_zero_returns_none` | version 0 → None |
| `test_resolve_boundary_version_one_returns_frozenset` | version 1 → frozenset of CIDs |
| `test_resolve_boundary_lower_cases_ids` | component IDs normalised |
| `test_resolve_boundary_empty_snapshot_returns_none` | accepted revision with empty list → None |
| `test_resolve_boundary_latest_accepted_wins` | multiple revisions, returns last accepted |
| `test_apply_filter_moves_out_of_scope_to_ruled_out` | candidate outside boundary → ruled_out |
| `test_apply_filter_keeps_in_scope_candidates` | candidate inside boundary → candidates[] |
| `test_apply_filter_no_component_id_not_filtered` | no component_id → kept |
| `test_apply_filter_ruled_out_reason_code` | reason_code = "scope_filtered" |
| `test_apply_filter_empty_boundary_none_filtered` | boundary=None → no filter applied |
| `test_apply_scope_revision_merges_added_component_ids` | `changed_boundary.added_component_ids` merged into snapshot |
| `test_apply_scope_revision_removes_component_ids` | `removed_component_ids` subtracted |
| `test_apply_scope_revision_no_snapshot_builds_from_prior` | caller omits scope_snapshot → auto-built |
| `test_resolve_expansion_suggestion_accepted_updates_scope` | acceptance bumps scope version |
| `test_resolve_expansion_suggestion_rejected_no_scope_change` | rejection = no version change |
| `test_resolve_expansion_suggestion_deferred_no_scope_change` | defer = no version change |
| `test_resolve_expansion_suggestion_marks_suggestion` | `analyst_decision` field updated on suggestion |
| `test_resolve_expansion_suggestion_unknown_signal_id_raises` | ValueError on unknown signal_id |
| `test_run_version_zero_no_filter_applied` | full pipeline, version 0 → no candidates removed |
| `test_run_version_one_filter_removes_out_of_scope` | full pipeline, version 1 → out-of-scope in ruled_out |

### WS6 — Schema + Docs

- `run_context.json`: no change (expansion_suggestions already has `analyst_decision` enum with "accepted")
- `causality_candidates.json`: add `"scope_filtered"` to `ruled_out[].reason_code` enum description
- Backlog + metamodel update

---

## 6. Design Decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Soft vs hard filter | **Soft** (move to ruled_out, never delete) | Preserves audit trail; analyst can widen scope and reinstate |
| 2 | Activation threshold | `active_scope_version > 0` | Version 0 is discovery mode; analyst must act first |
| 3 | Case sensitivity | Lowercase-normalise both boundary and candidate CIDs | KG IDs are inconsistently cased |
| 4 | Missing CID on candidate | Not filtered | Defensive; do not silently discard candidates lacking CID |
| 5 | `apply_scope_revision` snapshot build | Auto-merge when caller omits snapshot | Reduces caller burden for the common expansion-suggestion workflow |
| 6 | `resolve_expansion_suggestion` scope | Instance method with persist flag | Needs artifact store; `persist=True` default for analyst workflow |

---

## 7. Definition of Done

| # | Criterion |
|---|---|
| 1 | `_resolve_approved_scope_boundary` returns `None` for version 0, frozenset for version ≥ 1 |
| 2 | `_apply_scope_boundary_filter` moves out-of-scope candidates to `ruled_out[]` with `reason_code="scope_filtered"` |
| 3 | `apply_scope_revision` auto-builds scope snapshot from `changed_boundary.added/removed_component_ids` |
| 4 | `resolve_expansion_suggestion` atomically marks suggestion + updates scope |
| 5 | Filter wired into `run()` between `generate()` and `refine_with_evidence` |
| 6 | `scope_filter` block present in `run_manifest.artifacts` |
| 7 | 20 targeted tests pass |
| 8 | Full suite passes, zero regressions |

---

## 8. Step Readiness Matrix (Before vs After)

| Dimension | Before | After |
|---|---|---|
| Scope decision enforceability | Documentation only | Filters generate() output on re-run |
| Expansion suggestion lifecycle | Pending only → never resolved | accept/defer/reject → scope version bumped |
| Out-of-scope candidates | Scored and surfaced to analyst | Moved to ruled_out with `scope_filtered` reason |
| `apply_scope_revision` usability | Caller must reconstruct full snapshot | Auto-merged from `changed_boundary` |
| Manifest traceability | Scope version only | Full `scope_filter` block with filtered count and CIDs |
