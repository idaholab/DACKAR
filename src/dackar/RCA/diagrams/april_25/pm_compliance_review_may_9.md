# PM Compliance Module Review
**Date**: May 9, 2026  
**Reviewer**: Claude (session review)  
**Module**: `src/dackar/RCA/pm_compliance/`  
**Reference**: `diagrams/april_20/PM_Compliance_Module_Architecture.md`

---

## Overall Assessment

The module is well-architected — clean schema, thoughtful three-tier governance matching (structural → FM-level → keyword fallback), and graceful degradation throughout. The main gaps are spec items acknowledged as "not yet done" in the architecture doc plus a handful of correctness issues.

---

## 1. Confirmed Bugs

### 1.1 `assessment_date` set to event timestamp, not build time
**File**: `aggregator.py:299`

```python
out["assessment_date"] = event_ts or utcnow_iso()   # sets event time, not now
```

The `input_guards.py` staleness check computes `days = (event_ts.date() - ad.date()).days`. When `build_pm_compliance` generates the artifact, `assessment_date == event_ts`, so `days == 0` — the staleness guard is **permanently inactive** for any auto-built artifact.

**Fix**: set `assessment_date` to `utcnow_iso()` at build time. Store the event reference time separately as `as_of_event_timestamp` if needed for traceability.

---

### 1.2 `analyze_degradation` keyword heuristic is too narrow
**File**: `effectiveness_analyzer.py:18-21`

The current heuristic uses 5 hardcoded stems for degrading (`"degrad"`, `"worse"`, `"increas"`, `"severity"`) and 4 for improving (`"improv"`, `"normal"`, `"no defect"`, `"acceptable"`). Several common as-found patterns are silently misclassified:

- `"bearing wear observed"` → `"stable"` (should be `"degrading"`: `"wear"` excluded from improving but not included in degrading)
- `"leak found at seal"` → `"stable"` (no leak stem)
- `"corrosion on casing"` → `"stable"` (no corrosion stem)
- `"pump shaft cracked"` → `"stable"` (no crack/fracture stem)
- `"vibration anomalous"` → `"stable"` (no vibration stem)

**Existing vocabulary resource**: the project already maintains curated, domain-relevant keyword lists in `/Users/mandd/projects/DACKAR/data/`:

| File | Terms | Relevant mapping |
|------|-------|-----------------|
| `health_status_keywords_negative.csv` | 118 terms | → `"degrading"` signal |
| `health_status_keywords_positive.csv` | 43 terms | → `"improving"` signal |
| `health_status_keywords_neutral.csv` | 52 terms | → `"stable"` signal |

The negative vocabulary already includes: Breakdown, Collapse, Deterioration, Decay, Rupture, Fracture, Crack, Corrosion, Erosion, Vibration, Degradation, Disintegration, Worsening, Degeneration, Weakening, Leaky, Malfunctioning, Damaged, Cracked, Broken, Shaky, Brittle, inoperable, neglected — covering all the gaps above.

**Recommended fix**: replace the hardcoded stems in `analyze_degradation` with a loader that reads these CSV files (Nouns + Adjectives columns are most relevant for as-found text; Verbs add coverage for WO narrative sentences). Match lowercased full-word tokens rather than substrings for precision. The `PMComplianceConfig` could expose the path so tests can point to a minimal fixture without depending on the data directory.

**Benefit beyond bug fix**: this also aligns the PM compliance degradation signal with the same vocabulary used by the rest of the DACKAR NLP pipeline, making the two systems consistent.

---

### 1.3 Timezone comparison risk in `_derive_status`
**File**: `execution_verifier.py:119`

`parse_dt` returns tz-aware datetimes only when the source string contains timezone info. If a CMMS export row supplies `next_due_date` without timezone (e.g., `"2024-05-01T00:00:00"`), comparing it against the tz-aware `event_dt` raises `TypeError: can't compare offset-naive and offset-aware datetimes`.

**Fix**: normalize both `event_dt` and `next_due` to UTC before comparison using `.replace(tzinfo=timezone.utc)` when `tzinfo` is None.

---

### 1.4 `"unknown"` check status mapped to `"compliant"` in narrative
**File**: `aggregator.py:116` (`_compliance_status_md`)

When `st == "unknown"` (no schedule data), the function returns `"compliant"`. An analyst reading `components[].pm_tasks[].compliance_status` sees `"compliant"` for tasks the module had no data on. This is documented as governance-neutral, but is misleading in the narrative view.

**Fix**: return a dedicated `"undetermined"` label (requires schema extension), or ensure the corresponding `data_quality_note` references the specific `task_code` so the analyst can correlate.

---

## 2. Spec Gaps (architecture says yes, code says no)

### 2.1 Stage H `pm_corrective` action auto-generation — not implemented
**Reference**: Architecture §4, §9.5

The architecture explicitly states:
> *scope_gaps[] for the primary hypothesis FM should automatically generate a recommended action of type `pm_corrective` in the rca_card. If `maintenance_induced_risk == "high"`, that action gets `priority: "high"` unconditionally (closes C4 partially).*

The synthesizer currently passes `pm_compliance` as raw JSON to the LLM prompt and relies on the LLM to notice and surface PM gaps. There is no structured extraction of `scope_gaps` → `pm_corrective` recommended_action anywhere in the synthesizer or orchestrator.

**This is the highest-value missing piece.** It also closes C4 per the architecture doc.

---

### 2.2 `pm_found_defect_rate` missing from summary
**Reference**: Architecture §3.4

The architecture specifies this metric: fraction of PM executions that recorded a defect condition. `last_as_found` is passed through on each `pm_tasks[]` row, but no aggregate rate is computed in `summary`. Without it, the LLM prompt and the analyst have no quick signal for "how often was PM finding a problem before the failure."

**Fix**: count rows where `as_found_last` / `as_found_condition` maps to a non-normal vocabulary term; express as `pm_found_defect_rate: float` in `summary`. Requires controlled vocabulary (acknowledged in arch doc §3.4) or the existing `analyze_degradation` heuristic as a proxy.

---

### 2.3 `coverage_type` not derived from KG relationship type
**Reference**: Architecture §3.3

The architecture defines `coverage_type: preventive | detective | none` based on *which* KG property linked the PM task: `preventing_pm_task_ids` → `preventive`; `detecting_pm_task_ids` → `detective`. The scope analyzer iterates all three KG fields (`scope_analyzer.py:55-66`) but does not record which field produced the match, so `coverage_type` is always passed through from the raw export row (defaulting to `"none"`).

**Fix**: in `scope_analyzer.py`, track which KG field produced the coverage match and set `coverage_type` accordingly on the component view.

---

### 2.4 Silent risk underestimation when `primary_fm_id` provided but KG linkage absent
**File**: `aggregator.py` (`_rollup_risk` inputs)

When a caller passes `primary_fm_id="FM-XYZ"` but the KG has no explicit PM↔FM tags, `all_gap` is empty and `primary_in_gap = False`, so `has_scope_gaps_for_primary_fm = False` and `maintenance_induced_risk` may be `"low"`. The caller gets a confident-looking low-risk assessment when in reality scope analysis was **skipped** for the primary FM.

**Fix**: append a `data_quality_note` when `primary_fm_id` is provided but `fmea_pm_linkage_available` is False:
```
"primary_fm_id provided but KG linkage absent — scope gap for <FM-XYZ> not evaluable; maintenance_induced_risk may be underestimated"
```

---

## 3. Dead / Unused Code

### 3.1 `effectiveness_lookback_cycles` config parameter never used
**File**: `config.py:15`

Documented as "How many past PM cycles to read for as-found / degradation trend" but `analyze_degradation` and `collect_as_found_from_rows` process all export rows unconditionally. The config value is never passed to either function.

**Options**: implement cycle-count limiting in `collect_as_found_from_rows` (slice the last N rows by completed date), or remove the field to avoid confusion.

---

## 4. Improvements Worth Considering

| # | Item | Value | Effort | Notes |
|---|------|-------|--------|-------|
| A | Stage H `pm_corrective` synthesis (§2.1) | **High** — closes C4, reduces LLM guess-work | Medium | Requires synthesizer + orchestrator change |
| B | Fix `assessment_date` semantics (§1.1) | **High** — activates the existing staleness guard | Low | One-line fix in aggregator |
| C | DQ note for primary FM gap when no linkage (§2.4) | Medium — prevents silent risk underestimation | Low | One condition in aggregator |
| D | `pm_found_defect_rate` in summary (§2.2) | Medium — quantifies PM effectiveness | Low | Count non-normal as-found rows |
| E | `coverage_type` from KG field type (§2.3) | Medium — enables detective vs preventive reasoning | Low | Track field name in scope_analyzer |
| F | Timezone normalization in `_derive_status` (§1.3) | Medium — prevents runtime TypeError | Low | Normalize to UTC before compare |
| G | `"wear"` keyword in degradation detection (§1.2) | Low-Medium — corrects a quiet misclassification | Low | Add to degrading keyword list |
| H | Free-text PM↔FM fallback via token Jaccard | Low-Medium — helps when KG linkage absent | Medium | Arch doc notes it as "brittle" |
| I | CMMS adapter / export parser (Maximo, SAP PM) | High for production | High | External dependency; Phase 2 per arch doc |

**Priority order for next sprint**: B → C → F → G (all low-effort correctness fixes), then A (highest-value feature gap).

---

## 5. What Works Well

- **Three-tier governance matching** in `_governance_details` (structural → FM-level → keyword) is well-designed and avoids false penalization of unrelated failures.
- **Schema** is strict (`additionalProperties: false`), Draft 7 validated, and the `checks[]` + optional `components[]` dual-view is a clean way to serve both the pipeline (checks) and the analyst (component narrative).
- **`not_applicable` handling** is correct throughout: governance-neutral pass, preserved in narrative, DQ note emitted.
- **`primary_fm_id` mode in `_rollup_risk`** correctly separates "primary FM gap + overdue" (high) from "something else is overdue" (medium/low).
- **`data_quality_confidence` rollup** surface data limitations automatically — avoids silent high-confidence scores when data is sparse.
- **Unit test coverage** hits the key governance paths: schema validation, overdue detection, rollup logic, fmea_pm_linkage_available semantics, staleness path.

---

---

## 6. RCA Workflow Context — Critical Observations

Before staging the implementation, it is worth mapping each gap to the pipeline stage it actually damages. The pm_compliance artifact touches four distinct touchpoints in `rca_reasoning_orchestrator.run()`:

| Touchpoint | Pipeline location | What pm_compliance feeds |
|---|---|---|
| **Build** | pre-Stage A (`_build_pm_compliance_if_needed`) | artifact produced here; all bugs live here |
| **Staleness guard** | Stage A input_guards | `assessment_date` checked against `event.timestamp_start` |
| **Governance scoring** | Stage D (`_governance_details` in causality engine) | failed checks → score boost; `coverage_type` shapes reasoning |
| **Synthesis** | Stage H (`rca_synthesizer.synthesize`) | raw JSON to LLM; `scope_gaps` and `maintenance_induced_risk` not extracted |

**Key observation**: bug §1.1 (`assessment_date`) silently disables the Stage A staleness guard for every auto-built artifact. This means the pipeline never flags a stale PM assessment on the auto-build path — the guard only fires for externally-supplied artifacts that happen to carry a different date. This is the highest-risk correctness issue relative to the RCA workflow.

**Key observation on §2.1**: the Stage H `pm_corrective` gap is not just a missing feature — it means the LLM is being asked to infer a deterministic fact (scope gap for primary FM → corrective action) from a large JSON blob. LLM inference of structured PM gaps is unreliable and untestable. The orchestrator already has a clean pattern for deterministic post-synthesis injection (`_apply_rank_inversion_attention_flag`, `_apply_kg_governance_attention_flags`, etc.) — `pm_corrective` actions belong there, not in the LLM prompt.

**Key observation on §1.2**: the vocabulary gap is not just a misclassification — the degradation trend signal feeds `components[].degradation_trend`, which is embedded in the LLM prompt and influences the synthesizer's evidence narrative. Misclassifying "bearing wear observed" as `"stable"` directly weakens the causal chain the LLM builds for maintenance-induced hypotheses.

---

## 7. Implementation Strategy

Changes are grouped into four waves. Each wave leaves the pipeline in a fully working state with a clear test gate before the next begins.

---

### Wave 1 — Correctness fixes (no schema changes, no new features)

**Scope**: bugs §1.1, §1.3, §2.4, and dead code §3.1.  
**Files touched**: `aggregator.py`, `execution_verifier.py`, `config.py`, `effectiveness_analyzer.py`.  
**Risk**: low — all changes are confined to the pm_compliance package; no orchestrator or schema changes.

| Fix | Change |
|---|---|
| §1.1 `assessment_date` | `aggregator.py:299` → `assessment_date = utcnow_iso()`; add `as_of_event_timestamp = event_ts` as a separate field for traceability |
| §1.3 timezone | `execution_verifier.py` `_derive_status` → normalize `next_due` and `event_dt` to UTC before comparison using `.replace(tzinfo=timezone.utc)` when `tzinfo is None` |
| §2.4 silent DQ note | `aggregator.py` end of `build_pm_compliance` → if `primary_fm_id` is given and `fmea_pm_linkage_available` is False, append a DQ note naming the FM |
| §3.1 dead config | Option A: wire `effectiveness_lookback_cycles` to slice `collect_as_found_from_rows` by most-recent-N rows (sort by `completed_date`). Option B: remove from `PMComplianceConfig` and mark as reserved for Phase 2 |

**Test gate**: all 1527 existing unit tests pass; add one test confirming the staleness guard now fires on an auto-built artifact where `assessment_date` differs from the event timestamp (previously impossible to trigger on this path).

---

### Wave 2 — Vocabulary upgrade + `pm_found_defect_rate`

**Scope**: §1.2 (keyword heuristic) and §2.2 (`pm_found_defect_rate`).  
**Files touched**: `effectiveness_analyzer.py`, `config.py`, `schemas/pm_compliance.json`.  
**Risk**: medium — requires a schema extension for `pm_found_defect_rate`; vocabulary loader introduces a file-system dependency that must be made testable.

**Design**:

1. Add `data_dir: Optional[Path] = None` to `PMComplianceConfig`. When `None`, the loader falls back to the current hardcoded stems so existing callers without the data path are unaffected.

2. Introduce `PMVocabularyLoader` (new file or top of `effectiveness_analyzer.py`) that reads `health_status_keywords_negative.csv`, `health_status_keywords_positive.csv`, and `health_status_keywords_neutral.csv` from `data_dir`. Load Nouns + Adjectives columns (most relevant for as-found text); optionally Verbs for WO narrative sentences. Cache on first load.

3. Rewrite `analyze_degradation` to use token-level matching (split blob on whitespace/punctuation, match lowercased tokens against vocabulary sets) rather than substring matching. This prevents "increase" from matching "increased flow rate is acceptable" in the degrading path.

4. Add `pm_found_defect_rate: float` to `summary` in `aggregator.py`: count export rows where `as_found_last` / `as_found_condition` classifies as `"degrading"` via `analyze_degradation`, divided by total rows with as-found data. Schema extension required in `schemas/pm_compliance.json` (optional field under `summary`).

**Test gate**: new unit tests verify "bearing wear", "leak at seal", "corrosion on casing", "shaft cracked", "anomalous vibration" all return `"degrading"`. Tests use a minimal in-memory fixture vocabulary (3-row CSV), not the real data path. All existing tests still pass via fallback.

---

### Wave 3 — Scope analysis enhancements

**Scope**: §2.3 (`coverage_type` from KG field) and §1.4 (unknown→narrative fix).  
**Files touched**: `scope_analyzer.py`, `aggregator.py` (`_compliance_status_md`), `schemas/pm_compliance.json`.  
**Risk**: low-medium — `scope_analyzer` changes are localized; the narrative fix requires a design decision on the schema.

**`coverage_type` derivation** (`scope_analyzer.py`):

In the KG linkage loop (lines 55-66), track which field produced the match:
- `preventing_pm_task_ids` → `coverage_type = "preventive"`
- `detecting_pm_task_ids` → `coverage_type = "detective"`
- `pm_task_ids` → `coverage_type = "preventive"` (default for ambiguous linkage)

Return the per-FM coverage_type from `analyze_scope` and propagate it to `components[].pm_tasks[].coverage_type` in the aggregator. This replaces the current pass-through default of `"none"`.

Stage D impact: `_governance_details` can distinguish a detective PM failure (the PM existed but didn't catch the degradation — surveillance gap) from a preventive PM failure (the PM that would have prevented the mechanism was missed — direct causal link). The governance score boost may differ between these two cases.

**Narrative fix** (`_compliance_status_md`):

Preferred approach (no schema change): keep `"compliant"` in the governance path but append a task-specific DQ note in `aggregator.py` when `st == "unknown"` — e.g., `"Task PM-XYZ: status unknown (no schedule dates); shown as compliant in narrative per §6 but excluded from overdue count"`. This makes the gap visible to the analyst without requiring a schema enum extension.

**Test gate**: new unit tests for `coverage_type` derivation from `preventing_pm_task_ids` vs `detecting_pm_task_ids`; DQ note test for unknown-status tasks.

---

### Wave 4 — Stage H `pm_corrective` action injection

**Scope**: §2.1 — deterministic `pm_corrective` recommended action generation from `scope_gaps[]`.  
**Files touched**: `rca_reasoning_orchestrator.py` (new method `_apply_pm_corrective_actions`), `schemas/rca_card.json` (if `pm_corrective` type needs to be schema-enumerated).  
**Risk**: medium — touches the orchestrator and the rca_card schema; but implementation follows the established `_apply_*_attention_flags` pattern and requires no LLM changes.

**Design**: add `_apply_pm_corrective_actions` to the orchestrator, called after synthesis alongside the existing attention-flag methods (lines 704–734):

```python
def _apply_pm_corrective_actions(self, rca_card, pm_compliance, causality_candidates):
    if not pm_compliance:
        return
    # 1. Identify primary FM from rca_card.primary_hypothesis
    primary_fm_id = (rca_card.get("primary_hypothesis") or {}).get("fm_id")
    risk = (pm_compliance.get("summary") or {}).get("maintenance_induced_risk", "low")
    priority = "high" if risk == "high" else "medium"

    # 2. Collect scope gaps for primary FM across components
    for comp in pm_compliance.get("components") or []:
        gaps = comp.get("scope_gaps") or []
        if primary_fm_id and primary_fm_id not in gaps:
            continue
        if not gaps:
            continue
        action = {
            "type": "pm_corrective",
            "description": f"Establish PM coverage for {', '.join(gaps)} on component {comp['component_id']}",
            "priority": priority,
            "source_artifact": "pm_compliance",
            "scope_gaps": gaps,
            "component_id": comp["component_id"],
        }
        rca_card.setdefault("recommended_actions", []).append(action)
```

**Why here, not in the synthesizer**: the `maintenance_induced_risk == "high"` → `priority: "high"` override is a deterministic rule, not an LLM judgment. Injecting it post-synthesis (like the existing attention flags) makes it testable, auditable, and immune to LLM hallucination. The LLM prompt already receives `pm_compliance` as context so its narrative will be coherent; the structured action is a guaranteed addition on top.

**Test gate**: extend TC-6 (maintenance execution error scenario) to include a PM scope gap for the primary FM. Assert that `rca_card.recommended_actions` contains a `pm_corrective` entry with `priority: "high"` when `maintenance_induced_risk == "high"`. All existing TC-4 through TC-7 assertions still pass.

---

### Wave summary

| Wave | Items | Files | Schema change | Test additions | Dependency |
|---|---|---|---|---|---|
| 1 | §1.1, §1.3, §2.4, §3.1 | aggregator, verifier, config | None | +1 staleness guard test | None |
| 2 | §1.2, §2.2 | effectiveness_analyzer, config, schema | `summary.pm_found_defect_rate` | +5 vocabulary classification tests | Wave 1 (uses `data_dir` config) |
| 3 | §2.3, §1.4 | scope_analyzer, aggregator | Optional (`coverage_type` on pm_tasks) | +2 coverage_type derivation tests | Wave 2 (vocabulary used for as-found classification) |
| 4 | §2.1 | orchestrator | rca_card `pm_corrective` type | Extend TC-6 | Wave 3 (coverage_type informs action description) |

*[User comments below]*
