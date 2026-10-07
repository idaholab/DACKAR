# TC-7 — Scope Expansion in a Degraded-Data Environment

## Scenario Description

**Reactor type:** PWR Unit 1 (Northbrook Nuclear Station)  
**Event ID:** `E2026-04-20-001`  
**Asset ID:** `U1-RCP-C`  
**Operational mode:** Full power (100% rated)  
**CR Number:** `CR-2026-04-1182`

During routine parameter surveillance at 100% power, an operator identifies that RCP-C No. 1 seal leakoff flow (`U1-FT-RCP-C-SEAL1`) is 2.4 gpm — above the administrative action level of 2.0 gpm. The plant is stable; no automatic actuation. The process computer historian is in a scheduled backup maintenance window, so `soe_log` and `alarm_log` are **unavailable**. TSKR patterns are available from the independent TSKR subsystem.

A TSKR pattern identifies a 4.5°F `gradual_drift` anomaly in seal water HX inlet temperature (`U1-TE-SW-HX-4C-IN`) — onset 6 hours before the seal flow exceedance. The HX (`U1-SWP-SEAL-WATER-HX-4C`) is **outside the initial scope boundary** (which covers only `U1-RCP-C` and its seal package). The pipeline generates a scope expansion suggestion.

**Competing hypotheses (post-expansion):**

| Hypothesis | FM ID | Category | Chain Position |
|---|---|---|---|
| SW-HX-4C fouling — reduced seal cooling | `FM-SWHX4C-FOULING` | A | root |
| RCP-C seal face wear — intrinsic | `FM-RCPC-SEAL-WEAR` | A | proximate |
| Seal control valve drift | `FM-RCPC-SEAL-CV-DRIFT` | D | root |

---

## Data Elements Used

| Fixture | Required | Contents |
|---|---|---|
| `event.json` | Yes | Seal leakoff elevated, data availability note |
| `telemetry_summary.json` | Yes | Seal leakoff (sustained exceedance) + HX inlet temp (gradual drift, outside scope) |
| `tskr_patterns.json` | No | HX gradual drift precedes seal leakoff by 6.1 hours — scope expansion trigger |
| `kg_context.json` | Yes | Three failure modes across two scopes; prior event EVT-U1-2024-0308 |
| `operational_context.json` | No | Steady state 100% power; no nearby maintenance |
| `evidence_bundle.json` | No | SOP leakoff action levels; prior CR confirming HX fouling mechanism; PM gap for HX-4C |
| `soe_log.json` | — | **Not present** (data historian unavailable — deliberate data gap) |
| `alarm_log.json` | — | **Not present** (data historian unavailable — deliberate data gap) |

---

## Analysis Performed in Notebook

**Two-run demonstration:**

### Run 1 — Discovery Mode (`active_scope_version=0`)
- Scope covers only `U1-RCP-C` and `U1-RCP-C-SEAL1-PKG`
- `scope_filter.applied = False`
- `soe_log` and `alarm_log` show `"not_assessed"` in data coverage
- Sensitivity table includes both sources as missing
- Scope expansion suggestion generated for `U1-SW-HX-4C` (temporal predecessor, 6h lead)

### Analyst Action (between runs)
```python
orchestrator.resolve_expansion_suggestion(
    signal_id="U1-SW-HX-4C",
    decision="accepted",
    analyst_rationale="HX inlet temperature precursor is plausible thermal path to seal degradation"
)
```

### Run 2 — Scope Active (`active_scope_version=1`)
- `scope_filter.applied = True`, version = 1
- `FM-SWHX4C-FOULING` enters candidates
- HX fouling ranked above intrinsic wear (6h precursor signal discriminates)
- Similar event match: `EVT-U1-2024-0308` (same plant, same mechanism, confirmed root cause)
- Causal chain: HX fouling (root) → reduced seal cooling → seal face wear → leakoff increase

### Show-stopper
Run 1 says: "we found something outside your original scope — here it is." The analyst approves it in one line of code. Run 2 includes it and it becomes the primary hypothesis. The before/after comparison in a live notebook is impossible to replicate with a static checklist tool.

---

## Expected Outputs (Assertions)

| ID | Claim | Path / Condition |
|---|---|---|
| A7-1 | Run 1: no filter applied | `r1 scope_filter applied == False` |
| A7-2 | Run 1: expansion suggestion generated | `r1 scope_management expansion_suggestions len >= 1` |
| A7-3 | Run 1: SOE not assessed | `r1 data_coverage soe_log status == "not_assessed"` |
| A7-4 | Run 1: alarm not assessed | `r1 data_coverage alarm_log status == "not_assessed"` |
| A7-5 | Run 1: SOE in sensitivity check | `soe_log` in `r1 sensitivity_table missing_sources_checked` |
| A7-6 | Run 2: scope filter active | `r2 scope_filter applied == True` and `version == 1` |
| A7-7 | Run 2: HX candidate present | Candidate with `U1-SWP-SEAL-WATER-HX-4C` or `U1-SW-HX-4C` in `r2` |
| A7-8 | Run 2: similar event match | `r2 similar_event_list any_plant_match == True` |
| A7-9 | Run 2: seal wear deprioritised | Wear candidate composite < HX fouling candidate composite |
