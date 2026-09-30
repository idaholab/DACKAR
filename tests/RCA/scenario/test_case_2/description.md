# TC-2 — PWR Secondary Side Condenser Vacuum Loss

## Scenario Description

**Reactor type:** PWR Unit 1 (Northbrook Nuclear Station)  
**Event ID:** `E2026-04-15-001`  
**Asset ID:** `CONDENSER_TRAIN_A`  
**Operational mode:** Full power (~96% rated)  
**CR Number:** `CR-2026-04821`

Unit 1 is at full power. Starting at 13:22 the air removal system (Train B) differential
pressure rises above its setpoint. Two minutes later, condenser backpressure (`U1-PT-CND-4201`)
begins climbing and triggers two successive alarm tiers. The operator initiates a 5% load
reduction per AOP-10 at 14:00, arresting the vacuum degradation before the turbine trip
setpoint (3.50 inHgA) is reached. The primary root cause is air in-leakage into the condenser
secondary side (`FM-CND-AIR-INLEAK`).

Competing hypotheses: tube fouling, hotwell level control degradation, feedwater flow
instrumentation bias.

---

## Data Elements Used

| Fixture File | Required | Contents |
|---|---|---|
| `event.json` | Yes | Event descriptor, symptom signature |
| `telemetry_summary.json` | Yes | COND_VAC_A, TBP_A, HOTWELL_LVL_A, FW_FLOW_A, AIR_EJECTOR_DP_A |
| `kg_context.json` | Yes | Failure modes, prior events, documents |
| `operational_context.json` | No | Steady state, 96% rated power |
| `pm_compliance.json` | No | Air ejector PM compliance |
| `tskr_patterns.json` | No | Pre-built Allen temporal patterns (bypasses live scoring) |
| `evidence_bundle.json` | No | Pre-built evidence snippets from CRs and SOPs |
| `soe_log.json` | No | 6 SOE records: air removal DP → condenser BP alarm × 2 → hotwell level → FW deviation → operator action |
| `alarm_log.json` | No | 5 alarms with setpoint values and priority tiers |
| `protection_logic_context.json` | No | Turbine trip and runback logic sets; barrier state = `held` (trip setpoint not reached) |

---

## Analysis Performed in Notebook

1. **Fixture loading and sanity checks** — `event_id` / `asset_id` consistency across all fixtures.
2. **Evidence store construction** — Chroma vector store built from `processed_records.jsonl`.
3. **v31 vs v32 engine comparison** — both engines run against identical inputs; scoring and
   filtering differences are compared side by side.
4. **Candidate ranking** — top candidates sorted by `composite_score`; v32 filters 3 of 5
   generated candidates below threshold.
5. **Data coverage summary** — shows `soe_log`, `alarm_log`, and `protection_logic_context`
   as `present` when all three new fixtures are loaded.
6. **BONUS DEMO — SOE/PLC pairing validation:**
   - Run without PLC → `soe_plc_pairing = "violated"` in coverage summary.
   - Inspect `analyst_decisions_required` entries.
   - Run with PLC restored → pairing flag is gone; pipeline can interpret SOE in barrier context.

### Show-stopper demonstration

The SOE records the air removal DP anomaly at 13:22 and the condenser backpressure alarm at
13:24 — **two minutes apart across two separate systems**. The Allen `precedes` relation is
built automatically during the TSKR temporal stage and surfaces in the candidate score
rationale. A human analyst would need to manually correlate these across multiple alarm
response procedures; the pipeline makes the temporal link explicit and uses it to score air
in-leakage above competing hypotheses.

---

## Expected Outputs

| Assertion | Path | Expected Value |
|---|---|---|
| A2-1 | `rca_card.primary_hypothesis.cause_label` | `"FM-CND-AIR-INLEAK"` or `"FM::FM_AIR_INLEAK"` |
| A2-2 | `rca_card.decision_status` | `"candidate_ready"` or `"review_required"` |
| A2-3 | `causality_candidates.candidates` length | ≥ 1 |
| A2-4 | v32 vs v31 primary candidate ID | Same (both agree) |
| A2-5 | `run_manifest.artifacts.data_coverage_summary["soe_log"].status` | `"present"` |
| A2-6 | `run_manifest.artifacts.data_coverage_summary["alarm_log"].status` | `"present"` |
| A2-7 | `run_manifest.artifacts.data_coverage_summary["protection_logic_context"].status` | `"present"` |
| A2-DEMO | Without PLC: `data_coverage_summary["soe_log"].soe_plc_pairing` | `"violated"` |

---

## Accepted Gaps

- **Chroma / live evidence retrieval** is required for this test case (unlike TC-4 through TC-7
  which use the `evidence_bundle` fixture directly). If Chroma / Ollama are not available,
  the evidence bundle fallback can be used.
- The v31 comparison is retained for historical continuity but is not the focus of the new
  Phase B enrichment.
