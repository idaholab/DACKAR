# TC-8 — PWR Service Water Check Valve: Repeated Failures → Systemic Organizational Root Cause

**Test case type:** Full-depth multi-layer causal reasoning  
**Plant:** Unit 1, Facility B (PWR)  
**System:** Service Water (SW) — continuously operating, safety-grade  
**Event ID:** `EVT-U1B-2025-0312`  
**Asset ID:** `CHK-SW-HX-07A` (model `GWB-250-SS-316`, EDG HX-07A Train A supply header)

---

## Scenario Description

On 2025-03-12, check valve `CHK-SW-HX-07A` is found leaking through at 0.42 gpm during the
quarterly SW flow balance surveillance (WO-2025-SW-0147). Acceptance criterion is < 0.10 gpm.
This is the **third leakthrough failure of valve model `GWB-250-SS-316`** across SW Train A
and Train B in 18 months:

- **CR-2023-11-847** (November 2023): Train B valve `CHK-SW-HX-07B` — closed at proximate cause
- **CR-2024-04-219** (April 2024): Train A valve `CHK-SW-HX-07A` — closed at proximate cause
- **EVT-U1B-2025-0312** (March 2025): Train A valve `CHK-SW-HX-07A` — current event

The SW system operates continuously at 800–1,200 cycles/year. No real-time alarm or continuous
monitoring anomaly preceded the surveillance finding — the degradation was not visible to the
process historian.

The shift supervisor's initial working hypothesis:
> "Valve wear consistent with normal service life under high-cycle conditions. Replace with
> like-for-like. No further action required."

**This is the wrong answer.** DACKAR's structured investigation reveals five causal contributors
spanning four categories across three causal depths — and explicitly preserves two items as
open questions requiring analyst determination.

### The Investigative Journey

| Layer | Candidate | Category | Depth | Final Score |
|---|---|---|---|---|
| Proximate | `FM-CHK-SEAT-EROSION` | A (Equipment-Internal) | Proximate | 0.81 |
| Contributing | `FM-PM-FREQ-NONCONF` | J (Inspection/Testing Inadequacy) | Contributing | 0.82 |
| Contributing | `FM-VENDOR-BATCH-TRACEABILITY` | K (Vendor/Supply Chain) | Contributing | 0.69 |
| Contributing | `FM-PM-CONFIG-CONTROL-GAP` | I (Configuration/Change Control) | Contributing | 0.76 |
| Root cause | `FM-OE-SCREENING-MISS` | L (Systemic/Organizational) | Root cause | 0.71 |

**Layer 1 (Category A):** Poppet seat erosion confirmed by teardown — consistent with
high-frequency cycling without adequate inspection frequency.

**Layer 2 (Category J):** The valve model requires a 12-month inspection per vendor specification
`VND-SPEC-GWB-250`. A 2021 PM program restructuring (`WO-2021-PMREV-04`) set the interval to
18 months without reviewing the vendor manual. The governance scorer detects the mismatch from
the PM task node in the KG, which carries both `plant_interval_months=18` and
`vendor_spec_interval_months=12`. **If `vendor_spec_interval_months` is absent from the KG PM
task node, this candidate is never generated** — this is the KG coverage bound illustrated in
the LWRS report.

**Layer 3 (Category K):** INPO IRIS Operating Experience report `IRIS-OE-2023-SW-0047` identifies
failures of valve model `GWB-250-SS-316` at three other units traced to production lots
`GWB-2020-L05` through `GWB-2020-L09`. Teardown confirms the current valve is lot `GWB-2020-L07`.
The teardown initially appears to **contradict** the vendor batch hypothesis (seat dimensions
within spec) — but NER cross-reference resolves this: the fleet OE specifies that the
dimensional nonconformance manifests as fatigue failure after 12–24 months of cycling, not
as an out-of-spec dimension at initial inspection. Final score 0.69 intentionally reflects
residual uncertainty — the teardown role label stays "contradicting" with modulated weight.

**Layer 4 (Category I):** The 2021 PM frequency change was processed as a minor procedure
revision without engineering evaluation of the vendor specification impact. No 50.59 screening
was triggered. The configuration baseline was changed without documented technical basis.

**Layer 5 (Category L):** `IRIS-OE-2023-SW-0047` was issued in September 2023 and classified
non-applicable by the plant's OE screener. The screener associated the OE with RCS boundary
service valves (cavitation mechanism) rather than SW system check valves. The non-applicable
classification is an opinion — hedge_fraction = 0.38. Whether this reflects a systemic screening
program gap or an individual judgment call is explicitly left as **analyst determination required**.

---

## Key Design Decisions and Constraints

1. **Telemetry pattern:** "No anomalous telemetry" — the leakthrough was surveillance-detected,
   not visible in the continuous historian. `Tel = 0.60` for Category A (direct failure measurement),
   `Tel = 0.40` for all other categories (no process variable signature expected for programmatic
   or organizational contributors). This deliberately contrasts with TC-2, which uses rich
   alarm-sequence temporal scoring.

2. **Temporal reasoning source:** Recurrence episode detection across plant historical CAP entries
   (`CR-2023-11-847`, `CR-2024-04-219`), not intra-event SOE alarm logic. TSKR patterns use
   `operator_family = "episode_recurrence"`.

3. **Contradiction resolution:** `TD-REPORT-2025-0312` has evidence role "contradicting" for
   `FM-VENDOR-BATCH-TRACEABILITY`. After lot-number cross-reference by the CompatibilityEngine,
   the contradiction weight is modulated (not zeroed). The role label stays "contradicting" per
   confirmed three-role taxonomy (supporting / contradicting / contextual only).

4. **Intentional residual ambiguity (by design):**
   - `FM-VENDOR-BATCH-TRACEABILITY`: score capped at 0.69 — "probable contributing, traceability
     action required." Pipeline cannot confirm batch-defect vs. accelerated-wear without
     metallurgical analysis.
   - `FM-OE-SCREENING-MISS`: score 0.71 — "probable, analyst determination required." Pipeline
     cannot determine whether screening miss was a systemic program gap or individual judgment call.

5. **Category J > Category A post-refine:** `FM-PM-FREQ-NONCONF` (0.82) scores higher than
   `FM-CHK-SEAT-EROSION` (0.81) because Category J's weight profile (w_E=0.55, w_G=0.30) gives
   dominant weight to documentary and governance evidence, which is strong. Scores are
   **intra-category composites under different weight profiles** and should not be interpreted
   as comparative likelihoods. Causal depth (proximate vs. contributing vs. root) is determined
   by category, not score rank.

6. **causality_candidates NOT pre-supplied:** Per show-and-tell test suite convention, the
   causality engine generates candidates at runtime. This ensures scoring logic is exercised.

---

## Weight Profiles Used (from analysis_steps.tex / causality_engine_v32.py)

| Category | w_S | w_T | w_Tel | w_E | w_G |
|---|---|---|---|---|---|
| A | 0.30 | 0.20 | 0.20 | 0.20 | 0.10 |
| I | 0.05 | 0.25 | 0.10 | 0.45 | 0.15 |
| J | 0.05 | 0.05 | 0.05 | 0.55 | 0.30 |
| K | 0.10 | 0.10 | 0.05 | 0.50 | 0.25 |
| L | 0.05 | 0.05 | 0.05 | 0.60 | 0.25 |

---

## Pre- vs Post-Refinement Score Table

| Candidate | Cat | S | T | Tel | E (pre) | G | **Composite (pre)** | E (post) | **Composite (post)** | Depth |
|---|---|---|---|---|---|---|---|---|---|---|
| FM-CHK-SEAT-EROSION | A | 0.90 | 0.82 | 0.60 | 0.75 | 0.65 | **0.77** | 0.92 | **0.81** | Proximate |
| FM-PM-FREQ-NONCONF | J | 0.65 | 0.60 | 0.40 | 0.75 | 0.88 | **0.77** | 0.86 | **0.82** | Contributing |
| FM-VENDOR-BATCH-TRACEABILITY | K | 0.62 | 0.65 | 0.40 | 0.61 | 0.72 | **0.63** | 0.70 | **0.69** | Contributing |
| FM-PM-CONFIG-CONTROL-GAP | I | 0.55 | 0.70 | 0.40 | 0.75 | 0.75 | **0.72** | 0.80 | **0.76** | Contributing |
| FM-OE-SCREENING-MISS | L | 0.45 | 0.40 | 0.40 | 0.75 | 0.68 | **0.68** | 0.76 | **0.71** | Root cause |

---

## Data Elements Used

| Fixture File | Required | Contents |
|---|---|---|
| `event.json` | Yes | Event descriptor, recurrence context (prior event IDs, valve model) |
| `telemetry_summary.json` | Yes | 2 SW signals, no continuous anomalies, 1 surveillance finding |
| `kg_context.json` | Yes | 7 components, 6 failure modes, 2 past events, 5 documents, 1 PM task node with interval mismatch |
| `soe_log.json` | Optional | 4 sparse entries — surveillance sequence only, no alarms |
| `tskr_patterns.json` | Optional | 5 recurrence-episode patterns (episode_recurrence operator family) |
| `pm_compliance.json` | Optional | 3 checks: 2 fail (interval nonconformance), 1 pass |
| `evidence_bundle.json` | Optional | 7 snippets: 5 supporting, 1 contradicting (TD-REPORT), 1 supporting (VND-SPEC) |
| `operational_context.json` | Optional | Full-power steady-state, SW Train A in service |

---

## Expected Outputs

| Field | Expected Value |
|---|---|
| `input_validation.ok` | True |
| `output_validation.ok` | True |
| Candidates generated | 5 (FM-CHK-SEAT-EROSION, FM-PM-FREQ-NONCONF, FM-VENDOR-BATCH-TRACEABILITY, FM-PM-CONFIG-CONTROL-GAP, FM-OE-SCREENING-MISS) |
| Category span | A, I, J, K, L |
| Primary by score (post-refine) | FM-PM-FREQ-NONCONF (0.82) — Category J |
| Proximate cause in rca_card | FM-CHK-SEAT-EROSION (0.81) |
| Root cause in rca_card | FM-OE-SCREENING-MISS (0.71) — with "analyst determination required" flag |
| `rca_card.executive_summary.causal_depth_summary.depth_complete` | True (all three depths represented) |
| Open items in rca_card | ≥ 2 (vendor batch traceability, OE screening determination) |
| Contradicting evidence present | TD-REPORT-2025-0312 role = "contradicting" for FM-VENDOR-BATCH-TRACEABILITY |
| PM compliance failure | `interval_nonconformance` for PM-SW-CHK-ANNUAL |

---

## Accepted Gaps

- No `alarm_log.json` — surveillance-detected failure, no annunciator alarms
- No `protection_logic_context.json` — no automatic protection function actuated
- No `configuration_change_records.json` — WO-2021-PMREV-04 is captured in evidence_bundle; the PM revision is not a configuration change in the MCA/licensing basis sense
- Category I and L weight profiles are less well-exercised in prior test cases; this test case is the primary exercise of both

---

## Pipeline Capabilities Demonstrated

| Feature | Where It Appears |
|---|---|
| Multi-category generation (A, I, J, K, L) | All five candidates across all three causal depths |
| Recurrence detection (3 events in 18 months) | TSKR episode_recurrence patterns; past_events in kg_context |
| Fleet OE integration | IRIS-OE-2023-SW-0047 retrieved and scored as Category K evidence |
| PM frequency nonconformance detection | Governance scorer detects plant_interval ≠ vendor_spec_interval in KG PM task node |
| KG coverage bound | Category J candidate generated ONLY if vendor_spec_interval_months is present on the PM task node |
| Conflicting evidence handling | TD-REPORT-2025-0312 initially contradicting FM-VENDOR-BATCH-TRACEABILITY; resolved by NER lot-number cross-reference |
| Contradiction modulation (not zeroing) | Role label stays "contradicting"; CompatibilityEngine reduces contradiction weight |
| Intentional residual ambiguity | Two candidates with non-definitive confidence labels at card issuance |
| Causal depth mapping | rca_card assigns proximate / contributing / root depth by category, not score rank |
| AP-913 multi-level closure | rca_card generates actions at each causal depth matching AP-913 levels 1–3 |
| Hedge fraction | OE screening determination has hedge_fraction=0.38 (opinion, not measurement) |
| Category L as root cause | First test case in the suite to have a Category L candidate as the root-cause-depth finding |

---

## Show-Stopper

The pipeline prevents the CAP entry from closing at proximate cause. Without structured
multi-category RCA:
- The PM interval deviation would remain undetected (no process variable signature)
- The vendor lot would not be reviewed (teardown appears to clear the hypothesis)
- The IRIS OE report would remain classified non-applicable
- The recurrence would continue

DACKAR did not make the investigation easier. It made it harder to stop too soon.
