# TC-4 — Reactor Trip: PLC Gates and Timeline Gate

## Scenario Description

**Reactor type:** PWR Unit 1 (Northbrook Nuclear Station)  
**Event ID:** `E2026-03-10-001`  
**Asset ID:** `U1-RPS-CORE`  
**Operational mode:** Full power (99% rated)  
**CR Number:** `CR-2026-03021`

At 14:22:03 UTC, RPS channel 4 (`U1-NI-RPS-4A`) initiates an automatic reactor trip on high neutron flux (112.3% rated vs 105% trip setpoint). All four CRD banks insert within 2.4 seconds. The unit is secured; post-trip review must determine the initiating cause.

**Three hypotheses:**

| Hypothesis | FM ID | Category |
|---|---|---|
| Spurious NI instrument signal (EMI + calibration offset) | `FM-NI-SPURIOUS` | D (instrumentation) |
| Real power excursion from FW transient | `FM-FW-TRANSIENT` | A (process) |
| CRD mechanism fault | `FM-CRD-MECHANICAL` | A (mechanical) |

---

## Data Elements Used

| Fixture | Required | Contents |
|---|---|---|
| `event.json` | Yes | SCRAM descriptor, symptom signature |
| `telemetry_summary.json` | Yes | NI channels 1-4, FCV-FW-3301, CRD-2214 signals |
| `kg_context.json` | Yes | 3 failure modes, 1 prior event, 4 documents |
| `operational_context.json` | No | Full power, post-calibration WO nearby |
| `soe_log.json` | No | 6 records: FCV T-45s → NI4 trip T+0 → Bank A T+0.8s → Banks B-D T+2.4s → CRD-2214 T+10s → operator T+90s |
| `alarm_log.json` | No | NI-4 high-high flux alarm + FCV deviation alarm |
| `protection_logic_context.json` | No | RPS trip logic set; all barriers `held` |
| `environmental_monitoring.json` | No | Containment background radiation + EMI record in NI cabinet area |
| `evidence_bundle.json` | No | WO calibration note, SOP guidance, CR EMI finding, post-trip CR |

---

## Analysis Performed in Notebook

1. **Fixture loading** — load all 9 fixture files; sanity-check event_id/asset_id consistency.
2. **Orchestrator construction** — `build_fixture_orchestrator()` from shared helpers.
3. **Full pipeline run** — single `run_rca()` call with all fixtures.
4. **Timeline gate demo** — show that `FM-CRD-MECHANICAL` is eliminated because CRD-2214 SOE record is T+10s *after* the trip initiator; the Allen `follows` relation is identified automatically.
5. **PLC gate demo** — show that NI candidate has `hard_gates.barrier_logic.plc_consulted = True` since RPS logic set is in the PLC fixture.
6. **Multi-channel discrimination** — NI channels 1, 2, 3 show no anomaly; genuine flux excursion would require correlated response.
7. **Coverage summary** — verify environmental_monitoring and all new fixtures are reflected as `present`.
8. **Assertions** — run A4-1 through A4-10.

### Show-stopper demonstration

Hypothesis C (`FM-CRD-MECHANICAL`) is eliminated automatically by the timeline consistency gate because the SOE records the CRD anomaly 10 seconds **after** the trip initiator. No analyst action required. The Allen `follows` relation is identified from the SOE timestamps and propagated to the hard gate decision.

---

## Expected Outputs (Assertions)

| ID | Claim | Path / Condition |
|---|---|---|
| A4-1 | PLC consulted for NI candidate | `cand["hard_gates"]["barrier_logic"]["plc_consulted"] == True` for NI-4 candidate |
| A4-2 | CRD ruled out by timeline | `cand["hard_gates"]["timeline_consistency"]["passed"] == False` for CRD candidate |
| A4-3 | NI spurious is primary hypothesis | `rca_card["primary_hypothesis"]["cause_label"]` contains `"FM-NI-SPURIOUS"` |
| A4-4 | FW transient scores below NI spurious | `FM-FW-TRANSIENT` composite < `FM-NI-SPURIOUS` composite |
| A4-5 | Environmental monitoring consumed | `data_coverage_summary["environmental_monitoring"]["status"] == "present"` |
| A4-6 | Barriers held noted | `barrier_logic` rationale references `"held"` barrier state |
| A4-7 | Decision trail present | `result["decision_trail"]` is not empty |

---

## Accepted Gaps

- No TSKR patterns provided; TSKR scorer runs live (no-op with no historical patterns loaded).
- Signal lessons learned may be empty if no matching pattern library entries are loaded.
