# TC-5 — ECCS Train Degradation: Common-Cause Failure

## Scenario Description

**Reactor type:** BWR Unit 3 (Northbrook Nuclear Station)  
**Event ID:** `E2026-02-14-001`  
**Asset ID:** `U3-HPCI-SYSTEM`  
**Operational mode:** Cold shutdown (maintenance outage, mode 4)  
**CR Numbers:** `CR-2026-02847`, `CR-2026-02848`

During a refueling outage, both HPCI trains A and B were overhauled by the same crew over a 5-day window. Pump coupling assemblies were replaced using parts from vendor lot `VEN-2026-Q1-HC7`. Post-maintenance surveillance (SURV-U3-HPCI-001) shows both trains fail to reach required injection flow (5000 gpm) — identical failure mode on both: coupling slippage under load from insufficient torque.

A vendor advisory (VA-2025-HC7-001) for this lot was received 3 months prior but was not dispositioned. The station surveillance procedure does not include the vendor's torque acceptance criterion.

**Hypotheses:**

| Hypothesis | FM ID | Category |
|---|---|---|
| CCF: common vendor lot, same crew, undispositioned advisory | `FM-HPCI-CCF-COUPLING` | C |
| Independent wear — Train A | `FM-HPCI-A-WEAR` | A |
| Independent wear — Train B | `FM-HPCI-B-WEAR` | A |

---

## Data Elements Used

| Fixture | Required | Contents |
|---|---|---|
| `event.json` | Yes | Dual surveillance failure descriptor |
| `telemetry_summary.json` | Yes | Both HPCI injection flow signals |
| `kg_context.json` | Yes | CCF failure mode with coupling mechanism, 2 wear modes, prior event |
| `operational_context.json` | No | Cold shutdown, two maintenance WOs nearby |
| `pm_compliance.json` | No | Train A PM overdue; Train B current — asymmetry contradicts independent wear |
| `alarm_log.json` | No | Identical alarm sequences on both trains — CCF alarm signature |
| `protection_logic_context.json` | No | HPCI actuation logic; both barriers `failed` |
| `configuration_change_records.json` | No | Both coupling replacements — same crew, same lot, same procedure gap |
| `vendor_supply_chain_records.json` | No | Lot VEN-2026-Q1-HC7; advisory VA-2025-HC7-001 not dispositioned |
| `evidence_bundle.json` | No | CR CCF evaluation, vendor advisory text, surveillance procedure gap |

---

## Analysis Performed in Notebook

1. **Fixture loading** — load all 10 fixture files.
2. **Orchestrator** — `build_fixture_orchestrator()` from shared helpers.
3. **Full pipeline run** — `run_rca()` with all fixtures.
4. **CCF scoring** — show that `FM-HPCI-CCF-COUPLING` has `ccf_score > 0` from structural delta.
5. **Vendor records coverage** — verify `vendor_supply_chain` status = `present`.
6. **Sensitivity table** — show that if the vendor advisory had been dispositioned, CCF confidence increases further (pointer to programmatic gap).
7. **Barrier analysis** — show both HPCI barriers recorded as `failed`.
8. **Assertions** — A5-1 through A5-9.

### Show-stopper demonstration

The CCF structural delta (Category C scoring) gives `FM-HPCI-CCF-COUPLING` a scoring boost from `common_cause_score` even before evidence refinement. The sensitivity table then shows that if the vendor supply chain records were richer (the advisory had a disposition WO), CCF confidence would increase further — a direct pointer to a programmatic gap with board-level significance.

---

## Expected Outputs (Assertions)

| ID | Claim | Path / Condition |
|---|---|---|
| A5-1 | CCF candidate ranked top | `FM-HPCI-CCF-COUPLING` composite ≥ wear candidates |
| A5-2 | CCF delta applied | `cand["scores"]["ccf_score"] > 0` for CCF candidate |
| A5-3 | Vendor records in coverage | `data_coverage_summary["vendor_supply_chain"]["status"] == "present"` |
| A5-4 | Barrier analysis present | `barrier_analysis` artifact is not empty |
| A5-5 | Both barriers failed | `barrier_analysis.summary.degraded_barrier_count >= 2` |
| A5-6 | PLC consulted | `cand["hard_gates"]["barrier_logic"]["plc_consulted"] == True` for CCF candidate |
| A5-7 | Wear candidates below CCF | Both wear candidate composites < CCF candidate composite |
| A5-8 | AP-913 completeness present | `run_manifest["ap913_completeness"]` is not None |
