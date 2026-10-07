# TC-RPS-1 — Feedwater Heater Drain Valve Failure (RCA Pattern Search Demo)

## Scenario Description

**Incident ID:** `INC-2024-09-15-FWH3`
**Asset ID:** `FWH3` (Feedwater Heater No. 3, high-pressure extraction stage)
**Plant type:** PWR (Pressurized Water Reactor), 4-loop, ~1000 MWe
**Operational mode:** Steady-state full-power operation (100% RTP)
**Incident date:** 2024-09-15, ~10:00 plant time

### What Happened

Feedwater Heater No. 3 drain control valve **DCV-3A** began sticking partially closed
due to stem packing wear accumulated over the previous operating cycle. The reduced drain
flow allowed condensate to accumulate inside the FWH-3 shell, gradually inundating the
lower tube rows and throttling extraction steam condensation.

The cascade of consequences:

| Step | Effect |
|---|---|
| DCV-3A sticks | FWH-3 shell level rises above normal band |
| FWH-3 tube bundle partially flooded | Feedwater temperature rise across FWH-3 reduced |
| Feedwater delivered to steam generators is cooler | Condenser must absorb proportionally more heat |
| Condenser backpressure rises | Turbine exhaust pressure exceeds setpoint |
| Turbine governor reduces output | Reactor load follow response → power limit alarm |

A drain flow alarm (`FWH3_DRAIN_FLOW_ALM`) cycles repeatedly (6 times) as operators
attempt to troubleshoot the drain circuit — an alarm-flood pattern characteristic of
drain valve faults.  A turbine bearing temperature alarm (`LUBE_OIL_TEMP_HI`) activates
due to the load transient but is unrelated to the root cause.

### Alarm / SOE / Anomaly Sequence (Query Incident)

| Time (T+min) | Source | Event |
|---|---|---|
| T+0 | Alarm | `FWH3_LEVEL_HIGH` — shell level > 60% |
| T+1 | Alarm | `FWH3_DRAIN_FLOW_ALM` × 6 — drain flow cycling (flood alarm, filtered by freq_threshold) |
| T+2 | Alarm | `FW_TEMP_LOW` — feedwater temperature < design setpoint |
| T+5 | Alarm | `COND_BP_HIGH` — condenser backpressure > 3.5 inHgA |
| T+7 | SOE | `FWH3_LVL_CTRL::trip` — level controller trips to manual |
| T+8 | Alarm | `TURB_EFF_LOW` — turbine heat-rate deviation alarm |
| T+9 | Alarm | `LUBE_OIL_TEMP_HI` — turbine bearing temp (load-transient noise, not root cause) |
| T+11 | SOE | `COND_BP_CTRL::actuate` — condenser bypass valve opens |
| T+12 | Alarm | `RX_POWER_HI_LIMIT` — approaching 100% RTP limit |
| T+15 | SOE | `TURB_LOAD_LIMIT::actuate` — turbine load limiter engages |
| T+28 | SOE | `TURB_LOAD_LIMIT::reset` — load limiter resets after operator action |
| T+2 | Anomaly | `FWH3_LEVEL::drift` (sustained level rise) |
| T+3 | Anomaly | `FW_SUPPLY_TEMP::step_down` (temperature step decrease) |
| T+5 | Anomaly | `COND_BACKPRESS::spike` (backpressure excursion) |

---

## Historical Database

Six historical episodes are indexed, spanning the previous 15 months:

| Episode | Date | System affected | RCA | Similarity to query |
|---|---|---|---|---|
| EP1 | 2024-06-10 | FWH-3 drain circuit | FWH3 drain valve seat erosion | **High** — same subsystem, same cascade, same flood pattern |
| EP6 | 2023-07-08 | FWH-3 drain circuit | FWH3 drain valve actuator solenoid failure | **High (near-tie)** — same subsystem, partial cascade, no backpressure anomaly |
| EP5 | 2024-01-20 | FWH-3 level controls | FWH3 level instrument fault | **Moderate** — same subsystem, but reversed ordering (controller trips *before* level alarms) |
| EP2 | 2024-04-22 | Condenser | Condenser tube fouling | **Low-moderate** — condenser alarms shared, different initiating cause |
| EP3 | 2023-12-15 | Circulating water | Circ-water pump impeller wear | **Low (borderline Jaccard)** — only backpressure alarm shared |
| EP4 | 2023-09-05 | Main steam | MSIV spurious closure | **None** — filtered by Jaccard gate |

---

## Retrieval Challenges

This test case is specifically designed to stress the three-metric pipeline:

| Challenge | Which episode | Which metric reveals it |
|---|---|---|
| **Near-tie** between two FWH-3 events | EP1 vs EP6 | Jaccard + EMD: EP1 wins because it has more complete cascade (COND_BACKPRESS::spike, RX_POWER_HI_LIMIT) and matching flood pattern (both have FWH3_DRAIN_FLOW_ALM >> freq_threshold) |
| **Reordered cascade** (same types, different order) | EP5 | NLCS: EP5's NLCS is notably lower than EP1/EP6 because its controller trip precedes the level alarm |
| **Alarm flooding** (FWH3_DRAIN_FLOW_ALM × 6/8) | Query & EP1 | freq_threshold=4 filters it from Jaccard/NLCS but retains it in EMD freq_vec — the flood pattern is captured only by EMD |
| **Distractor alarm** (LUBE_OIL_TEMP_HI in query, not in EP1) | EP1 | Jaccard is slightly reduced but EP1 still ranks first |
| **Jaccard gate** filtering a clearly unrelated episode | EP4 | EP4 shares only RX_POWER_HI_LIMIT with query; Jaccard ≈ 0.04 < min_jaccard=0.15 → never scored on NLCS or EMD |
| **Borderline Jaccard pass** (single shared alarm) | EP3 | EP3 shares COND_BP_HIGH, COND_BP_CTRL::actuate, COND_BACKPRESS::spike; passes gate but ranks last |

---

## Purpose

This test case demonstrates the end-to-end `rca_pattern_search` module on a realistic
nuclear plant scenario with deliberately constructed retrieval challenges:

1. **Stage 1** — Build a historical episode index from a 15-month event log using
   KDE-based density detection.
2. **Stage 2** — Extract a query fingerprint from the current incident's alarm, SOE,
   and anomaly records (including flood alarm and distractor).
3. **Retrieval** — Run coarse-to-fine similarity search and verify that:
   - EP1 (FWH drain valve erosion) ranks first
   - EP6 (FWH actuator solenoid) ranks second — the near-tie case
   - EP5 (FWH instrument fault) ranks third but with notably lower NLCS
   - EP4 (MSIV closure) does not appear in results (Jaccard-filtered)
4. **Diagnostics** — Bandwidth scan, alarm flood filtering, EMD normalization modes.

## Files

| File | Contents |
|---|---|
| `data/historical_events.csv` | Flat event log: 15 months, all sources, ~220 events |
| `data/alarm_log_query.json` | Alarm log for the 2024-09-15 query incident |
| `data/soe_log_query.json` | SOE log for the 2024-09-15 query incident |
| `data/anomaly_log_query.json` | Telemetry summary for the 2024-09-15 query incident |
| `data/episode_rca_labels.json` | Known RCA labels for historical episodes (by window date) |
| `rca_pattern_search_demo.ipynb` | Notebook: full pipeline + challenges + plots |
