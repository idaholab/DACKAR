# Issue: Missing Feedback Loop for Past-Event Failure-Mode Linkage

**Date:** 2026-05-07
**Status:** Partially implemented — Phase 3 complete; Phase 1 Risk 1 Tier 1 + Risk 2 complete; Phase 1 FM enrichment + Phase 2 pending

### Implementation Status

| Phase / Item | Status | Commit / Notes |
|---|---|---|
| Phase 3 — KG governance FM link coverage (`_compute_kg_governance`) | ✅ Done | Adds `past_event_count`, `past_events_with_fm`, `fm_link_coverage`, `fm_link_gap` to governance dict; raises `"yellow"` when gap detected |
| Phase 3 — Distinguishing attention flags (`_apply_kg_governance_attention_flags`) | ✅ Done | "No prior KG events" vs "N of M carry no FM link" messages; fm_link_gap issue skips generic prefix |
| Phase 1 Risk 2 — `exact_doc_ids` wiring at Step 6 cross-pattern query | ✅ Done | `_build_cross_pattern_evidence()` now accepts `kg_context`; builds exclusion set from CMMS event_ids; passes `None` when no CMMS events present |
| Phase 1 Risk 1 Tier 1 — CMMS-vs-CMMS dedup guard | ✅ Done | `_augment_kg_context_with_cmms_past_events()` builds `existing_cmms_doc_ids` set; same doc_id from separate CMMS paths is suppressed |
| Unit tests (23 new) | ✅ Done | `unit_tests/test_fm_linkage_pipeline.py` — covers governance metrics, attention flag messages, dedup guard, exclusion set |
| Phase 1 — `_enrich_cmms_event_fm()` (CMMS event FM population) | ⏳ Pending | Needs `DocExtractionStore` ID-based lookup interface review before coding |
| Phase 2 — `write_rca_findings_to_kg()` + `orchestrator.commit()` | ⏳ Pending | Requires `abnormal_event` TOML schema review and `run_manifest` writeback_ready structure |

---

## Problem Statement

The RCA pipeline's recurrence detection and pattern-recognition (Steps 2–3) depend on
`abnormal_event` nodes in the KG carrying `CONFIRMED_CAUSE` or `MAY_CAUSE` edges to
`failure_mode` nodes. Without those edges, a past event can only match the current event
on **asset or component**, not on **failure mode** — making recurrence detection nearly
useless for FM-level analysis.

In practice, confirmed FM links are absent for the vast majority of past events because:

1. **CMMS injection is thin.** `_cmms_record_to_past_event()` in
   `orchestrators/rca_reasoning_orchestrator.py:1556` converts CR and WO records into
   past-event dicts with `fm_id: None` and `matched_failure_mode_ids: []`. No FM link
   is ever produced by this path.

2. **No automated writeback from completed RCAs.** When an RCA run concludes and
   identifies a confirmed root cause, nothing writes a `CONFIRMED_CAUSE` edge back onto
   the originating `abnormal_event` node. `build_graph_from_workflow_artifacts()` in
   `kg/kg_schema_builder_workflow.py` can write `rca_case` nodes and
   `identifies_causal_factor` edges, but it does not update the source event node with
   the confirmed FM.

3. **Manual entry is the only working path.** The reference guide (§ Model Governance)
   states that "plant event records staff enter past events and their causal links after
   each RCA or CR closure" — but this is a manual governance step with no enforcement
   and no tooling.

## Consequences

- `recurrence_count` in `tskr_patterns` is systematically understated: the FM-match
  branch of `_fetch_past_events()` returns zero results even when true recurrences
  exist as closed CRs or completed RCAs in the CMMS.
- The `documentary_novel` flag (`history_score < 0.20`) fires incorrectly for events
  that are genuine recurrences, misleading the analyst toward investigating a known
  failure mode as if it were novel.
- `similar_event_list` FM-dimension scoring (one of five similarity dimensions) is
  always zero for CMMS-sourced past events.
- The KG governance check cannot distinguish "no prior events" from "prior events
  present but FM links missing" — both produce an amber/red state for the same reason.

## What Was Considered and Ruled Out

**Operator shift logs as a surrogate.**
Shift logs were considered as a richer source of historical observations than CRs/WOs.
They are earlier (written per shift, before a CR is filed) and contain component
references and symptom descriptions. However:
- They record *observations*, not confirmed causes — epistemic class
  `monitors_performance`, not `analyzes_past_degradation`.
- They cover the current event window and feed `operational_context` (Categories E, G),
  not the historical past-event pool.
- Indexing them in `DocExtractionStore` would improve cross-pattern evidence (Step 6)
  but would not produce `CONFIRMED_CAUSE` links in the KG.
- The `operational_context.json` schema has no `shift_narrative` field; shift logs are
  not yet a first-class input (reference guide §2.1 note).

Shift logs remain useful for the **current-event** timeline but are not a substitute
for the KG writeback gap.

## Double-Counting Risks

Two independent double-counting risks exist in the current code. Phase 1 would
exacerbate Risk 1 if not addressed alongside it.

### Risk 1 — CMMS event vs KG-native event (no guard)

If a plant has both:
- a KG-native `abnormal_event` node (e.g. `EVT-456`) created from an ECA or RCA
  finding, and
- the originating CR (`CR-12345`) fetched via CMMS injection,

both appear in `past_events` under different `event_id` values (`EVT-456` vs
`CMMS::CR::CR-12345`). The deduplication in `_augment_kg_context_with_cmms_past_events()`
(line 1776) is by `event_id` only. Because KG-native events use `e.id` values (e.g.
`EVT-456`) while CMMS-injected events use `CMMS::CR::CR-12345`, two event_ids will
never collide even when they refer to the same physical event. Phase 1 would worsen
this: the enriched CMMS event would gain an `fm_id` and then match on FM-dimension
scoring alongside the KG-native event that already carries a `CONFIRMED_CAUSE` link
to the same FM. The only way to detect this cross-type duplication is `source_doc_refs`
on KG-native nodes (added in Phase 2).

### Risk 2 — DocExtractionStore semantic retrieval (guard designed, never wired)

`DocExtractionStore.query()` accepts an `exact_doc_ids` parameter described in its
docstring as: *"Set of doc_ids already counted via exact-match recurrence —
excluded to prevent double-counting."* However, no call site in the orchestrator
passes this parameter — confirmed by grep. CRs already present in `past_events`
therefore appear again in semantic retrieval results (Step 5 / cross-pattern linkage),
inflating evidence scores for candidates that already have exact-match recurrence
support.

## Relevant Code Locations

| Location | Relevance |
|----------|-----------|
| `orchestrators/rca_reasoning_orchestrator.py:1556` | `_cmms_record_to_past_event()` — where `fm_id` is hard-coded to `None` |
| `orchestrators/kg_context_builder.py:737` | `_fetch_past_events()` — Neo4j query that filters on `CONFIRMED_CAUSE` / `MAY_CAUSE` |
| `kg/kg_schema_builder_workflow.py:828` | Past-event writeback in `build_graph_from_workflow_artifacts()` — writes `root_cause_was` edge but does not update source event node |
| `orchestrators/rca_reasoning_orchestrator.py:1676` | `_enrich_past_events_temporal_metadata()` — temporal tagging applied to whatever past events arrive; downstream of the gap |
| `doc_extraction/schema.py:111` | `DocExtractionRecord` — the extraction schema that could bridge documents to FM candidates if wired into KG writeback |

---

## Plan

Four phases in priority order. Each phase is independently deployable.

The two double-counting risks (described above) are first-class deliverables, not
optional guards. Risk 2 is addressed entirely in Phase 1 (standalone wiring fix).
Risk 1 is addressed in two tiers: a fallback check in Phase 1 that works immediately,
and a complete check in Phase 2 once `source_doc_refs` is populated by the writeback.

---

### Phase 1 — CMMS Event FM Enrichment + Double-Counting Guards

**Goal:** (a) Populate `fm_id` on CMMS-injected past events using FM candidates already
resolved by `DocExtractionStore`; (b) close Risk 2 entirely; (c) apply a fallback guard
for Risk 1.  No KG schema changes required in this phase.

**Why now:** The plumbing is almost complete. `_source_doc_id_from_event_id()` already
extracts the source CR/WO id from a CMMS event's `event_id`. `DocExtractionStore`
already holds extracted causal chains from CRs and WOs with resolved `fm_id_candidate`
fields. These two are used together for *scoring* in `_query_plant_past_events()` (line
3243) but are never used to *populate* `fm_id` on the injected event itself.
Risk 2 (`exact_doc_ids`) requires only passing an existing parameter — it has no
dependency on the KG schema or the writeback pipeline.

**Approach — FM enrichment:**

1. In `_augment_kg_context_with_cmms_past_events()`, after `_cmms_record_to_past_event()`
   produces a thin event dict, call a new helper `_enrich_cmms_event_fm()`.

2. `_enrich_cmms_event_fm()` derives the source doc id via `_source_doc_id_from_event_id()`,
   queries `DocExtractionStore` for records where `source_cr_id` or `source_wo_id` matches,
   and picks the best `auto_resolved` FM candidate by `fm_resolution_score`.

3. If a candidate is found:
   - Set `fm_id` on the event dict.
   - Set `fm_id_confidence = "inferred"` (new field) to distinguish from KG-confirmed links.
   - Append the fm_id to `matched_failure_mode_ids`.
   - Log the enrichment for the run manifest.

4. If the best candidate is `ambiguous` (not `auto_resolved`), set `fm_id = None` and
   add an `fm_id_ambiguous = True` flag — do not promote uncertain candidates silently.

**Quality ceiling:** These enriched events carry `MAY_CAUSE` quality at best (the source
documents are CRs/WOs, epistemic class `monitors_performance`). They must not be treated
as `CONFIRMED_CAUSE` by downstream scoring. The `fm_id_confidence = "inferred"` field
enforces this distinction.

**Approach — Risk 2 fix (exact_doc_ids wiring):**

Wire `exact_doc_ids` at the **cross-pattern Step 6** `DocExtractionStore.query()` call
site only (orchestrator line ~3697). Build the exclusion set from
`_source_doc_id_from_event_id()` applied to all events in `past_events` before that
query runs. This activates the guard that already exists in `DocExtractionStore.query()`
but has never been passed a non-None argument. Effect: CRs/WOs already counted as past
events will not reappear in cross-pattern semantic retrieval and inflate Step 6 evidence
scores.

**Do NOT wire this at** `_build_doc_id_semantic_scores()` (line ~1659). That call builds
the `doc_id → similarity` map used to compute `dim_semantic` scores for past events in
`_query_plant_past_events()`. Excluding those docs from that query would zero out the
semantic dimension for past events whose source doc is already in `past_events` — the
opposite of the intent.

**Approach — Risk 1 fallback guard (Tier 1):**

Before injecting a CMMS event, extract its CR/WO id via `_source_doc_id_from_event_id()`
and check whether that id already appears in any other CMMS event_id already present in
`past_events`. Concretely: build a set of doc ids by applying `_source_doc_id_from_event_id()`
to all event_ids in `existing` before the injection loop; skip any new CMMS event whose
doc id is already in that set.

**Scope limit:** KG-native events returned by `_fetch_past_events()` carry no
`source_doc_id` equivalent — that field is only added in `_query_plant_past_events()`
to CMMS result dicts, not to KG-native event dicts. Tier 1 therefore prevents only
CMMS-vs-CMMS duplication (e.g. the same CR surfacing twice via separate code paths),
not CMMS-vs-KG-native duplication. Detecting that a CMMS-injected CR is the same
physical event as a KG-native `abnormal_event` node requires `source_doc_refs` on the
KG node — that is Phase 2 only.

**Files changed:**
- ✅ `orchestrators/rca_reasoning_orchestrator.py` — Risk 1 Tier 1 deduplication guard added to `_augment_kg_context_with_cmms_past_events()`; Risk 2 `exact_doc_ids` wiring added to `_build_cross_pattern_evidence()` (new `kg_context` parameter).
- ⏳ `orchestrators/rca_reasoning_orchestrator.py` — `_enrich_cmms_event_fm()` and call from `_augment_kg_context_with_cmms_past_events()`; `_cmms_record_to_past_event()` update for `fm_id_confidence` / `fm_id_ambiguous` fields — pending `DocExtractionStore` ID lookup review.

**Config flag:** `enable_cmms_fm_enrichment: bool = True` in orchestrator config, so
sites without a populated `DocExtractionStore` can disable without side effects.

---

### Phase 2 — RCA Writeback Pipeline + Risk 1 Complete Guard

**Goal:** (a) After each completed RCA run, write a `CONFIRMED_CAUSE` edge from the
originating `abnormal_event` node to the confirmed `failure_mode` node, closing the
feedback loop for future runs; (b) populate `source_doc_refs` on written nodes so the
Risk 1 guard becomes fully effective for all KG records.

**Why this matters:** This is the only path to `CONFIRMED_CAUSE`-quality FM links at
scale. Every confirmed RCA improves recurrence detection for all future runs on the same
or similar equipment. Without it, the KG's past-event pool degrades over time relative
to the document record. `source_doc_refs` is also the enabler for the Risk 1 Tier 2
guard: once nodes carry this field, the Phase 1 deduplication check gains complete
coverage and no longer needs to rely on the `source_doc_id` fallback.

**Approach:**

1. Add a new function `write_rca_findings_to_kg(rca_card, event, neo4j_client)` in
   `kg/kg_schema_builder_workflow.py`. It performs two writes:
   - **Update `abnormal_event` node:** set `fm_id`, `resolved = True`,
     `rca_run_id` (for audit traceability), `source_doc_refs` (originating CR/WO ids).
   - **Create `CONFIRMED_CAUSE` edge:** `(abnormal_event)-[:CONFIRMED_CAUSE]->(failure_mode)`
     with properties `confirmed_at`, `rca_run_id`, `confidence = "confirmed"`.

2. The function resolves the `abnormal_event` node by `event.event_id`. If the node does
   not exist in Neo4j (e.g. the event was CMMS-injected and never written to the KG as a
   native node), it creates it first using existing `abnormal_event` schema attributes,
   then writes the edges. This handles both native KG events and CMMS-sourced events.

3. Expose via a new `orchestrator.commit(run_id)` method — a **separate, explicit call**
   that the analyst makes after reviewing the `rca_card`. It must never run automatically
   at the end of `run()`. The reference guide already states "The pipeline does not write
   to the KG during a run" (§3.1); `commit()` is the post-run complement that satisfies
   that invariant.

4. The writeback is idempotent: re-running after a correction replaces the existing
   `CONFIRMED_CAUSE` edge properties rather than duplicating it.

**Two-gate model (both required before any Neo4j write):**

`commit()` checks two independent conditions and raises if either is false:

| Gate | Source | What it checks |
|------|--------|----------------|
| **Site gate** | `enable_kg_writeback: bool` (orchestrator config, default `False`) | This site has a writable KG and a populated `DocExtractionStore`. Opt-in — not default, because sites need to validate their KG schema before enabling writeback. |
| **Run gate** | `run_manifest.review_hooks.writeback_ready` (per-run) | All schema, citation, evidence, and severity checks passed; no degraded-run conditions; analyst decisions resolved. Computed by the existing `run_manifest` mechanism at Step 6. |

This maps directly onto the reference guide's existing workflow: after `orchestrator.run()`,
the analyst checks `run_manifest.review_hooks.next_step`. When it reads `"writeback"` and
`writeback_ready = True`, calling `orchestrator.commit(run_id)` is the technical action that
fulfills the writeback step — automating what the guide's §3.1 currently assigns to *plant
event records staff* as a manual governance responsibility.

**Risk 1 Tier 2 — complete guard enabled by this phase:**
The `source_doc_refs` field written onto each `abnormal_event` node allows the Phase 1
deduplication check to detect CMMS-vs-KG-native duplication: before injecting a CMMS
event, check its doc id against the `source_doc_refs` list of every KG-native event in
`past_events`. This is the only path to detecting that a CMMS-injected CR and an
existing KG-native `abnormal_event` node describe the same physical event, because
KG-native event dicts from `_fetch_past_events()` carry no source document reference
by default. The Tier 1 fallback remains active as a safety net for CMMS-vs-CMMS cases.

**Name note:** `source_doc_refs` (not `linked_doc_ids`) is used deliberately. The name
`linked_doc_ids` already exists on `HistoricalSignalEpisode` objects in the orchestrator
(lines ~3684, ~3833, ~3887, ~6325) with different semantics (supporting documents for a
pattern episode). Using a distinct name avoids confusion and potential bugs if generic
code iterates over these fields.

**Files to change:**
- `kg/kg_schema_builder_workflow.py` — add `write_rca_findings_to_kg()`.
- `orchestrators/rca_reasoning_orchestrator.py` — add `commit()` method and
  `enable_kg_writeback` config flag.
- `schemas/` — add `source_doc_refs`, `rca_run_id`, and `fm_id_confidence` properties
  to `abnormal_event` TOML schema.

---

### Phase 3 — KG Governance Distinction (FM Link Coverage)

**Goal:** Make the `kg_governance` check distinguish "no prior events exist" from "prior
events exist but carry no FM links" — currently both produce the same amber/red signal,
giving the analyst no actionable information.

**Why this matters:** An analyst seeing an amber governance flag today cannot tell whether
the recurrence pool is empty (equipment is new, no history) or corrupted (history exists
in CMMS but FM links were never written). The corrective action is different in each case.

**Approach:**

1. In `_compute_kg_governance()` (`rca_reasoning_orchestrator.py:5125`), compute from
   `kg_context["past_events"]`:
   - `past_event_count`: total events in the list.
   - `past_events_with_fm`: count of events where `fm_id` is not None.
   - `fm_link_coverage`: `past_events_with_fm / past_event_count` (0.0 if count is 0).

   Note: `_compute_kg_governance()` is called at line 402, **before** CMMS augmentation
   (line 435). This is intentional — the check measures KG-native event quality only,
   which is the correct signal for diagnosing the data gap. CMMS-enriched events are not
   yet in `past_events` at that point.

2. Add a new issue `"past_events_fm_link_gap"` when `past_event_count > 0` and
   `fm_link_coverage < threshold` (default 0.5). Status escalates to `"yellow"` (not
   `"red"` — data quality warning, not a hard blocker), consistent with the existing
   `stale_fm_ids` and `missing_revision_count` patterns in the same method.

3. Surface in `rca_card.executive_summary.analyst_attention_flags` via the existing
   `_apply_kg_governance_attention_flags()` mechanism, with a message that distinguishes:
   - `past_event_count == 0`: "No prior KG events found for this asset — cannot assess
     recurrence."
   - `fm_link_coverage < threshold`: "Prior KG events found but N of M carry no failure
     mode link — recurrence detection is partial. Run Phase 1 enrichment or review KG
     data entry."

**Files changed:**
- ✅ `orchestrators/rca_reasoning_orchestrator.py` — `_compute_kg_governance()` extended with FM link coverage; `_apply_kg_governance_attention_flags()` extended with distinguishing messages. No changes to `kg_context_builder.py` required.

**Config:** `kg_governance_fm_link_coverage_threshold: float = 0.5` in orchestrator
config `extra` dict (consistent with how `kg_min_failure_modes_default` and
`fmea_staleness_threshold_days` are already configured).

---

### Out of Scope for This Plan

**Operator shift logs** are a separate improvement tracked independently. They address
current-event evidence coverage (Categories E, G) rather than the historical recurrence
gap. The relevant schema change (`shift_narrative` in `operational_context.json`) and
a `ShiftLogParser` feeding `DocExtractionStore` should be planned as a distinct issue.
