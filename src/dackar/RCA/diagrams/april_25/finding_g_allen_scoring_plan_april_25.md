# Finding G — Wire `allen_base_score` into Causality Scoring

**Date:** 2026-04-25  
**Status:** Planning  
**Author:** RCA workflow systems engineering session

---

## 1. Problem Statement

The `allen_relation_map` (Step 2c artifact) attaches an `allen_base_score` (0.0–1.0) and an `allen_relation_to_event` (precedes / overlaps / contains / during / follows / unknown) to every anomaly, alarm, and SOE node relative to the current event interval.

These scores are **computed but never consumed** by the causality engine.  
The current temporal scoring path for each candidate uses:

| Source | Role |
| --- | --- |
| TSKR pattern match | Primary temporal signal (recurrence, latency) |
| Telemetry anomaly signals | Corroborating signal (component alignment) |
| spaCy back-fill in `refine_with_evidence` | Lag and relation from documentary evidence |

Allen scores would add a **direct interval-algebra confirmation layer** grounded in the actual event timestamps.  
When TSKR is weak or absent, Allen provides an independent temporal floor.  
When an Allen node has `allen_relation_to_event = "follows"`, it is a **direct timeline contradiction** that should be surfaced.

---

## 2. Current Architecture Snapshot

```
run()
  │
  ├─ generate(...)                          ← composite_score includes temporal (w=0.20)
  │     └─ _combine_scores(structural=0.30, temporal=0.20, telemetry=0.20,
  │                         evidence=0.20, governance=0.10)
  │     └─ _apply_uncertainty_propagation   ← quality_multiplier; composite_raw stored
  │
  ├─ refine_with_evidence(...)              ← updates evidence scores; no temporal change
  │
  └─ _stage_g_finalize_manifest(...)        ← _build_allen_relation_map() called HERE
        └─ allen_relation_map written to manifest (top-level artifact)
```

**Root problem:** `allen_relation_map` is built **inside** `_stage_g_finalize_manifest`,  
which is called **after** `refine_with_evidence`. Allen scores are therefore never available  
when composite scores are being updated.

---

## 3. Design Decisions

### 3.1 Blending formula

For each candidate whose `component_id` matches ≥1 causal Allen node (`causal_candidate=True`):

```
allen_score  = max(allen_base_score for matching causal nodes)
               (SOE nodes discounted ×0.80 when soe_clock_sync_ok=False)

new_temporal = 0.75 × old_temporal + 0.25 × allen_score
               iff allen_score > 0 and causal match found
             = old_temporal                                 (no match → no change)
```

Allen can raise but not lower the temporal score — it is an *additive confirmation*.  
α = 0.25 keeps Allen as a secondary signal and prevents it dominating TSKR.

### 3.2 Contradiction flag

If the candidate's `component_id` has any Allen node with `allen_relation_to_event = "follows"`:

```
temporal_evidence["temporal_contradiction"] = True
```

`_temporal_posture` already checks this field (line 1154 in engine) → candidate is  
automatically flagged `temporal_posture = "contradicted"` and `primary_eligibility = "blocked"`.

### 3.3 Composite score update

`composite_raw` was frozen at the end of `_apply_uncertainty_propagation`.  
`composite_score = composite_raw × quality_multiplier`.

After blending temporal:

```
temporal_delta   = new_temporal - old_temporal
raw_delta        = config.weights["temporal"] * temporal_delta / sum(weights)   # ≈ 0.20 * delta
new_composite_raw = clip(composite_raw + raw_delta, 0.0, 1.0)
new_composite     = clip(new_composite_raw × quality_multiplier, 0.0, 1.0)
```

This is a linear additive update — no full re-score needed.

### 3.4 Scope of match

Only nodes with a **non-null `component_id`** contribute to the index.  
Match is `candidate["component_id"] == node["component_id"]`.  
Node types in scope: `anomaly`, `alarm`, `soe_record` (all three Step 2c types).  
Past-event candidates (`hypothesis_type = "past_event"`) also matched on their `component_id`.

### 3.5 No-match behavior

When no Allen node matches the candidate's component: blend is **skipped entirely**.  
`scores["allen_temporal_score"]` is set to `null`; `scores["allen_blend_applied"]` = `false`.  
This is safe: candidates whose components don't appear in anomalies/alarms/SOE are unaffected.

---

## 4. Workstreams

### WS1 — Extract Allen map computation to before `refine_with_evidence` (orchestrator)

**File:** `rca_reasoning_orchestrator.py`

**Change 1 — Call `_build_allen_relation_map` early in `run()`**

After tskr_patterns is available and before the `refine_with_evidence` block (around line 394):

```python
# Step 2c — pre-compute Allen map for use in refine_with_evidence
pre_refine_allen_map = self._build_allen_relation_map(
    event=event,
    telemetry_summary=telemetry_summary,
    alarm_log=alarm_log,
    soe_log=soe_log,
)
```

**Change 2 — Thread into `refine_kwargs`**

```python
refine_kwargs["allen_relation_map"] = pre_refine_allen_map
```
(only when `inspect.signature` confirms `allen_relation_map` is an accepted parameter — same guard already in place for `coverage_summary`)

**Change 3 — Pass to `_stage_g_finalize_manifest`**

Add parameter `pre_computed_allen_map: Optional[JsonDict] = None` to `_stage_g_finalize_manifest`.  
When provided, skip the rebuild: `allen_relation_map = pre_computed_allen_map`.

---

### WS2 — Update `refine_with_evidence` in causality engine

**File:** `causality_engine_v32.py`

**New parameter:**
```python
def refine_with_evidence(
    self,
    causality_candidates: JsonDict,
    evidence_bundle: JsonDict,
    kg_context: Optional[JsonDict] = None,
    signal_evidence: Optional[JsonDict] = None,
    entity_normalizer_cfg: Optional[Dict[str, Any]] = None,
    coverage_summary: Optional[JsonDict] = None,
    allen_relation_map: Optional[JsonDict] = None,   # NEW
) -> JsonDict:
```

**New helper — `_build_allen_component_index`:**

```python
@staticmethod
def _build_allen_component_index(
    allen_relation_map: Optional[JsonDict],
) -> Tuple[Dict[str, float], Dict[str, str], Set[str]]:
    """
    Returns:
        causal_scores   {component_id → best allen_base_score (causal nodes only)}
        causal_relation {component_id → allen_relation_to_event of best node}
        follow_ids      set of component_ids that have any "follows" node
    """
```

Steps:
1. Extract `nodes` from `allen_relation_map`
2. Apply soe_clock_sync discount (×0.80) to soe_record nodes when `quality_flags.soe_clock_sync_ok = False`
3. Build causal index: component_id → max discounted score where `causal_candidate=True`
4. Build follows set: component_ids where any node has `allen_relation_to_event = "follows"`

**New helper — `_apply_allen_temporal_blend`:**

```python
@staticmethod
def _apply_allen_temporal_blend(
    candidate: JsonDict,
    causal_scores: Dict[str, float],
    causal_relations: Dict[str, str],
    follow_ids: Set[str],
    weights: Dict[str, float],
) -> None:
    """Mutates candidate in-place. Blends Allen score into temporal; flags contradictions."""
```

Steps:
1. Look up `component_id` in `causal_scores` and `follow_ids`
2. Set `scores["allen_temporal_score"] = allen_score` (or `None`)
3. Set `scores["allen_relation"] = causal_relations.get(cid)` (or `None`)
4. If `follow_ids` match: set `temporal_evidence["temporal_contradiction"] = True`
5. If causal match: compute blend, update `scores["temporal"]`, `scores["composite_raw"]`, `composite_score`, `confidence_label`
6. Set `scores["allen_blend_applied"] = True/False`

**Integration in the candidate loop of `refine_with_evidence`:**

At the end of each candidate's update block, just before `_apply_category_minimum_evidence_gate`:
```python
self._apply_allen_temporal_blend(
    candidate,
    causal_scores=allen_causal_scores,
    causal_relations=allen_causal_relations,
    follow_ids=allen_follow_ids,
    weights=dict(self.config.weights),
)
```

---

### WS3 — Update `temporal_posture` backfill path in `refine_with_evidence`

**No change needed.** `_apply_timeline_consistency_gate` (line 1122) reads  
`temporal_evidence["temporal_contradiction"]` directly. If Allen sets it to `True`,  
the gate will block the candidate on the next pass (lines 1154-1159 in current code).

The only check needed: ensure `_apply_allen_temporal_blend` is called **before** the  
gates block at lines 1121-1123. Currently gates run after the evidence-score loop — this  
is already correct.

---

### WS4 — Update `_update_score_rationale_for_refinement`

Add `allen_blend_applied` and `allen_temporal_score` to the score rationale comment
for traceability in the manifest.

---

### WS5 — Tests (`test_finding_g_allen_scoring.py`, ~22 tests)

| Test | What it checks |
| --- | --- |
| `test_allen_causal_match_boosts_low_tskr_temporal` | `scores["temporal"]` increases when Allen has high causal score and TSKR temporal is low |
| `test_allen_causal_match_raises_composite_proportional_to_weight` | delta ≈ 0.20 × temporal_delta |
| `test_allen_no_match_leaves_temporal_unchanged` | no component_id overlap → no change |
| `test_allen_follows_sets_temporal_contradiction` | node with follows → `temporal_evidence["temporal_contradiction"] = True` |
| `test_allen_follows_triggers_timeline_gate` | gate blocks the candidate |
| `test_allen_soe_clock_sync_false_applies_discount` | SOE node score × 0.80 |
| `test_allen_soe_clock_degraded_still_blends_if_causal` | discounted score still blends if > 0 |
| `test_allen_blend_applied_flag_is_true_on_match` | `scores["allen_blend_applied"] = True` |
| `test_allen_blend_applied_flag_is_false_on_no_match` | `scores["allen_blend_applied"] = False` |
| `test_allen_score_field_stored` | `scores["allen_temporal_score"]` correct value |
| `test_allen_relation_field_stored` | `scores["allen_relation"]` correct string |
| `test_allen_temporal_field_null_when_no_match` | `scores["allen_temporal_score"]` is null |
| `test_allen_scores_past_event_candidates_by_component` | past_event hypothesis also matched |
| `test_allen_multiple_nodes_same_component_takes_max` | max of multiple matching nodes |
| `test_allen_alarm_node_contributes` | alarm node type included in index |
| `test_allen_soe_node_contributes_without_clock_degradation` | SOE without degradation gets full score |
| `test_allen_blend_cannot_lower_temporal` | blended score ≥ old temporal always |
| `test_allen_blend_caps_at_one` | composite_raw never exceeds 1.0 |
| `test_allen_map_none_is_safe` | `allen_relation_map=None` → no change, no error |
| `test_allen_empty_nodes_is_safe` | empty `nodes` list → no change |
| `test_allen_orchestrator_passes_map_to_refine` | orchestrator `refine_kwargs` contains `allen_relation_map` |
| `test_allen_stage_g_receives_precomputed_map` | `_stage_g_finalize_manifest` skips rebuild when pre-computed |

---

### WS6 — Documentation

- Update `rca_workflow_development_backlog_april_25.md` with Finding G section, DoD, and Readiness Matrix.
- Update `rca_metamodel.md` — Phase 5 Finding G entry in Execution Sequencing.
- Update Step 2c / Step 5 metamodel status notes to reflect Allen → scoring integration.

---

## 5. Schema Changes

No new schema file needed.  
`causality_candidates.json` schema (if it validates the `scores` dict) may need:
- `allen_temporal_score`: number | null
- `allen_relation`: string | null
- `allen_blend_applied`: boolean

These are additive extensions to the `scores` object — no existing validations broken.

---

## 6. Definition of Done

| # | Criterion | Target |
| --- | --- | --- |
| 1 | `allen_relation_map` built before `refine_with_evidence` in `run()` | Yes |
| 2 | `_stage_g_finalize_manifest` uses pre-computed map (no rebuild) | Yes |
| 3 | `refine_with_evidence` accepts and uses `allen_relation_map` | Yes |
| 4 | `_build_allen_component_index` static helper implemented | Yes |
| 5 | `_apply_allen_temporal_blend` static helper implemented | Yes |
| 6 | Allen blend raises `scores["temporal"]` for matched candidates | Yes |
| 7 | Allen `follows` sets `temporal_evidence["temporal_contradiction"]` | Yes |
| 8 | SOE clock-sync discount applied | Yes |
| 9 | New score fields stored: `allen_temporal_score`, `allen_relation`, `allen_blend_applied` | Yes |
| 10 | `allen_relation_map = None` is fully backward-compatible | Yes |
| 11 | 22 targeted tests pass | Yes |
| 12 | Full suite 933+ tests pass, zero regressions | Yes |

---

## 7. Step Readiness Matrix (pre-implementation)

| Dimension | Current state | After this plan |
| --- | --- | --- |
| Allen map built | After refine (manifest only) | Before refine (used in scoring) |
| Temporal score source | TSKR only | TSKR primary + Allen blend |
| Contradiction detection | TSKR/spaCy | TSKR/spaCy + Allen `follows` |
| Score traceability | No Allen fields | `allen_temporal_score`, `allen_relation`, `allen_blend_applied` |
| Sequencing risk | Allen never reaches engine | Pre-refine computation; manifest reuses result |
| Backward compatibility | N/A | `Optional[JsonDict] = None`; no change when absent |
