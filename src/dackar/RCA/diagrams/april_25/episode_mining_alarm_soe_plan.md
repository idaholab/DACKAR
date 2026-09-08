# Episode Mining for Alarm / SOE Data — Design & Implementation Plan

**Date:** 2026-05-09  
**Status:** Draft — pre-implementation planning  
**Scope:** A new DACKAR module that applies frequent episode mining (FEM) to alarm logs and SOE sequences to discover recurring temporal patterns and map them to failure modes for RCA evidence.

---

## 1. Motivation

The current RCA pipeline treats each alarm/signal anomaly as an **independent flag**. It does not exploit the **ordering and co-occurrence structure** of the alarm cascade — the sequence in which alarms fire relative to each other and to the triggering event. Episode mining fills this gap by discovering recurring subsequences (episodes) in alarm/SOE streams that reliably precede or accompany specific failure modes.

**Key paper:** Ouarem, Nouioua & Fournier-Viger, *A Survey of Episode Mining*, WIREs DMKD 2024.  
**Most relevant applications from the paper:** ISD sensor network (§6.2), telecom alarm analysis (§1), episode rules for prediction (§5).

---

## 2. Assumptions

### 2.1 Data assumptions

| # | Assumption | Rationale |
|---|---|---|
| A1 | Alarm/SOE data arrives as an ordered list of `(alarm_id, timestamp, component_id, priority, system)` tuples | Matches existing `alarm_log.json` schema |
| A2 | Timestamps are UTC ISO-8601 strings with at least second resolution | Both TC-3 and TC-5 fixtures confirm this |
| A3 | The alarm window covers a **pre-event look-back** (minutes to days) plus the event itself | TC-3 window: 14 days; TC-5 window: 2 hours |
| A4 | An alarm sequence for a **single event** is short (5–50 alarms); a **fleet-level historical** corpus may be hundreds of events × 5–50 alarms each | Single-event mode and fleet mode require different freq thresholds |
| A5 | The KG provides a mapping from `component_id` and `system` tags to `fm_id` | Needed to label mined episodes with failure mode hypotheses |
| A6 | SOE data (if available) can be encoded as the same event-type schema as alarms — one event per SOE entry | Allows a unified mining pipeline |
| A7 | Clock synchronisation is reliable across alarms (confirmed by `quality.clock_sync_ok` field) | Required for inter-event interval correctness |
| A8 | `alarm_id` is the event type (not the specific alarm instance); repeated firings of the same alarm are multiple occurrences of the same type | Standard FEM encoding |

### 2.2 Algorithmic assumptions

| # | Assumption | Rationale |
|---|---|---|
| A9 | **Non-overlapping occurrence-based frequency** (NONEPI style) is preferred over window-based | Avoids double-counting; anti-monotonic; better for sparse sequences |
| A10 | A **span constraint** (max time between first and last event of an episode) is mandatory | Without it, an episode spanning weeks is not causal |
| A11 | Minimum 3 prior inter-event intervals required for OLS trend — same threshold as TSKR | Consistency across modules |
| A12 | The module operates in two modes: (a) **single-event mode** — pattern matching against a library; (b) **fleet mode** — pattern discovery from a corpus | Single-event mode is the primary RCA use case |
| A13 | Episode rules take the form `precursor_episode → terminal_alarm` where the terminal alarm is the event trigger | Causal direction is preserved by temporal ordering |

---

## 3. Method Design

### 3.1 Module architecture

```
AlarmSOESequence          KG failure mode map
      │                          │
      ▼                          ▼
┌─────────────────────────────────────────┐
│  AlarmEpisodeEncoder                    │
│  - Encode alarm_id as event type        │
│  - Apply span constraint filter         │
│  - Tag each event with component_id     │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────────┐
│  FLEET MODE │  │  SINGLE-EVENT    │
│  FEM miner  │  │  PATTERN MATCHER │
│  (offline)  │  │  (online, RCA)   │
└──────┬──────┘  └───────┬──────────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────────────────┐
│  EpisodeLibrary                         │
│  - {episode_pattern → fm_id(s),         │
│      frequency, span_stats,             │
│      episode_type (serial/parallel)}    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  AlarmEpisodeScorerV1                   │
│  - Match current alarm sequence         │
│    against EpisodeLibrary               │
│  - Score each FM: episode_match_score   │
│  - Detect novel sequences               │
│  - Flag parallel episodes (CCF signal)  │
└──────────────┬──────────────────────────┘
               │
               ▼ (feeds into TSKR or standalone)
        RCA evidence dict
```

### 3.2 Event encoding

Each alarm entry maps to an **event tuple**:

```python
EventTuple = (
    event_type: str,        # alarm_id (e.g. "ALM-U2-CNDSR-BP-TREND")
    timestamp:  datetime,   # UTC
    component:  str | None, # component_id from alarm log
    priority:   str,        # low | medium | high | critical
    system:     str,        # system tag
)
```

Alternative encoding strategies (to be tested):
- **By alarm_id** (finest grain — default)
- **By system tag** (coarser — groups related alarms)
- **By component_id** (maps to KG components)
- **By priority tier** (coarsest — suitable for pattern-class analysis)

### 3.3 Episode types targeted

| Type | Description | RCA use |
|---|---|---|
| **Serial** | `ALM-A → ALM-B → ALM-C` (strict order, within span) | Causal alarm cascade (TC-3 archetype) |
| **Parallel** | `{ALM-A, ALM-B}` (co-occur within window, no order) | CCF simultaneous failure (TC-5 archetype) |
| **Serial with gap constraint** | Serial + max inter-event gap | Tightens causality: excludes coincident but unrelated alarms |

### 3.4 Frequency definition

Use **non-overlapping occurrence-based frequency** (consistent with NONEPI / POERM from the survey):

- Each distinct non-overlapping occurrence of the episode in the corpus counts once
- Anti-monotonic → supports efficient pruning
- Suitable for sparse sequences (no double-counting across windows)

Span constraint: configurable, default **72 hours** for process plant cascades, **60 seconds** for SOE-level electrical/protection sequences.

### 3.5 Episode scoring (single-event mode)

For a given RCA event, the alarm sequence is matched against the episode library:

```
episode_match_score(fm) = 
    base_score
    × frequency_weight       # how often this episode precedes fm
    × span_alignment_weight  # how well the observed span matches library stats
    × completeness           # fraction of episode events present (partial match)
```

Novel sequences (no library match) are flagged separately — same concept as TSKR `signal_novel`.

### 3.6 Parallel episode detector (CCF flag)

A parallel episode detector checks whether two or more alarms on **different trains/divisions** fire with the same type and similar magnitude within a configurable window. This is a lightweight but high-value CCF signal.

```
ccf_score = 1.0  if parallel episode detected on redundant components
            0.0  otherwise
```

---

## 4. Integration with existing RCA pipeline

| Integration point | How |
|---|---|
| **Upstream of TSKR** | `AlarmEpisodeScorerV1.score()` returns per-FM episode evidence; TSKR uses it alongside telemetry anomaly matching |
| **`alarm_log.json` fixture** | New required fixture for any test case that uses episode scoring |
| **RCA card** | Add `alarm_episode_patterns` field to evidence section; feed novel/parallel flags into `analyst_attention_flags` |
| **Fleet mode** | Offline — run against historical CR corpus alarm logs to build/update `EpisodeLibrary` |

---

## 5. Unit Tests

### 5.1 Encoder tests

| ID | Test | Expected |
|---|---|---|
| E1 | Encode TC-3 alarm log → event sequence | 5 tuples, timestamps ascending, alarm_ids preserved |
| E2 | Encode with `encoding=system` → event types are system tags | `secondary-side-condenser`, `turbine-building-hvac`, etc. |
| E3 | Span filter drops alarms outside window | Only alarms within [window.start, window.end] retained |
| E4 | Clock sync flag False → encoder raises `DataQualityWarning` | Warning raised, processing continues |

### 5.2 Serial episode detection

| ID | Test | Expected |
|---|---|---|
| S1 | TC-3 cascade: `BP-TREND → HVAC-VIB-H → BP-H → BP-HH` is a 4-event serial episode | Detected with span ≈ 14 days |
| S2 | Same episode with span constraint 1 day → episode not detected | `[]` returned |
| S3 | Subsequence `BP-H → BP-HH → RUNBACK` (3-event) detected independently | Count = 1 for this window |
| S4 | Reversed order `BP-HH → BP-H` is NOT a valid serial episode | Not returned |
| S5 | Episode missing one intermediate alarm (partial match) → partial score returned | `completeness = 0.75` for 3/4 events |

### 5.3 Parallel episode detection (CCF)

| ID | Test | Expected |
|---|---|---|
| P1 | TC-5: `{ALM-U3-HPCI-A-FLOW-LO, ALM-U3-HPCI-B-FLOW-LO}` within 75-min window → parallel episode detected | `ccf_score = 1.0` |
| P2 | Same alarms but 10-hour gap → outside parallel window (default 2h) → not detected | `ccf_score = 0.0` |
| P3 | TC-5 motor current alarms precede flow alarms on both trains → serial sub-episode within each train detected | Two independent serial episodes, one per train |
| P4 | Parallel episode on same component (not redundant trains) → no CCF flag | `ccf_flag = False` |

### 5.4 Episode library (fleet mode)

| ID | Test | Expected |
|---|---|---|
| L1 | Build library from 3 identical alarm sequences → frequency = 3 | `episode.frequency = 3` |
| L2 | Anti-monotonicity: sub-episode frequency ≥ super-episode frequency | `freq(A→B) ≥ freq(A→B→C)` for all tested cases |
| L3 | Episode with frequency < minsup not returned | Absent from library |
| L4 | Library serialises to/from JSON without loss | Round-trip equality |

### 5.5 Scorer tests

| ID | Test | Expected |
|---|---|---|
| SC1 | TC-3 alarm sequence matched against library containing the cascade episode → FM-AIR-INLEAKAGE gets highest score | Score > 0 for correct FM |
| SC2 | TC-3 alarm sequence against library with only tube-fouling episode → no match | `episode_match_score = 0` for fouling FM |
| SC3 | Novel sequence (no library match) → `signal_novel = True` in output | Flag set |
| SC4 | TC-5 parallel episode → `ccf_flag = True` in output dict | Flag set |
| SC5 | Empty alarm log → scorer returns zero scores, no exception | Graceful empty output |
| SC6 | Partial match (3 of 4 alarms present) → score between 0 and full match score | `0 < partial_score < full_score` |

### 5.6 Integration tests

| ID | Test | Expected |
|---|---|---|
| I1 | Full pipeline TC-3: alarm_log → encoder → matcher → scorer → evidence dict | Evidence dict has `alarm_episode_patterns` key with ≥ 1 entry |
| I2 | Full pipeline TC-5: parallel episode detected → `analyst_attention_flags` includes CCF note | Flag in RCA card |
| I3 | TSKR + episode scorer combined: air in-leakage FM gets additive boost from both modules | Combined confidence > TSKR alone |

---

## 6. Demo Test Cases

### 6.1 TC-3 — Condenser vacuum loss (serial cascade)

**Goal:** Show that the backpressure alarm cascade is a serial episode, demonstrate how the episode maps to the air in-leakage failure mode, and contrast with the absence of any matching episode for tube fouling.

**Key episode to highlight:**
```
ALM-U2-CNDSR-BP-TREND (Day 0)
  → ALM-U2-HVAC-VIB-H (Day +4)
  → ALM-U2-CNDSR-BP-H (Day +12)
  → ALM-U2-CNDSR-BP-HH + ALM-U2-TRB-RUNBACK (Day +14)
```

**Narrative:** The episode has a 14-day span — a slow-developing cascade. The HVAC vibration alarm on Day +4 is the key discriminating alarm: it appears *between* two backpressure alarms, linking the HVAC contributing factor to the pressure boundary degradation.

**Demo outputs:**
- Event timeline (text table)
- Serial episode table (alarm chain, timestamps, inter-event gaps)
- FM mapping: episode → air in-leakage hypothesis
- Contrast: no episode match for tube fouling hypothesis

### 6.2 TC-5 — HPCI CCF (parallel episode)

**Goal:** Show that the simultaneous motor-current + flow alarms on both trains constitute a parallel episode, that the inter-train gap (75 min) is within the CCF detection window, and that the alarm signatures are structurally identical.

**Key episode to highlight:**
```
Train A: ALM-U3-HPCI-A-MOTOR-CURRENT → ALM-U3-HPCI-A-FLOW-LO  (Δt = 15s)
Train B: ALM-U3-HPCI-B-MOTOR-CURRENT → ALM-U3-HPCI-B-FLOW-LO  (Δt = 15s)
Parallel: {Train A sequence ‖ Train B sequence}  (inter-train gap = 75 min)
```

**Narrative:** Both trains produce the same 2-alarm serial sub-episode (motor current anomaly → flow low) with the same 15-second internal gap. The parallel structure across redundant trains is the CCF fingerprint.

**Demo outputs:**
- Dual-train alarm timeline
- Serial sub-episode table per train
- Parallel episode detection result + CCF flag
- Comparison of alarm signatures (identical magnitude, same pattern)

### 6.3 TC-NEW — SOE-level sequence (future)

**Goal:** Demonstrate episode mining on a millisecond-resolution SOE sequence for a reactor trip scenario — where the distinction between serial and composite (partially ordered) episodes matters.

**Status:** Fixture not yet created. Requires SOE data schema definition.

---

## 7. Open Questions

| # | Question | Impact |
|---|---|---|
| Q1 | What is the canonical SOE schema in DACKAR? Does it differ from `alarm_log.json`? | Determines whether a unified encoder handles both |
| Q2 | Is there a fleet-level historical alarm corpus available for library building, or does the library start empty and grow? | Determines initial mode of operation (match vs. discover) |
| Q3 | Should parallel episode detection use a fixed time window or a plant-specific redundancy window (e.g., from KG)? | KG-driven window is more defensible but requires KG linkage |
| Q4 | How does the episode scorer output integrate with the TSKR confidence formula — additive boost, multiplicative, or separate evidence channel? | Affects score calibration |
| Q5 | Should `alarm_id` or `system` tag be the primary event type for mining? `alarm_id` is more specific; `system` generalises across plants | Cross-plant generalisability vs. precision |

---

## 8. Implementation Phases

| Phase | Scope | Target |
|---|---|---|
| **1 — Encoder** | `AlarmEpisodeEncoder`: ingest `alarm_log.json`, output typed event sequence; span filter; encoding strategies | Unit tests E1–E4 |
| **2 — Serial detector** | `SerialEpisodeMatcher`: match a serial episode pattern against a sequence; span + gap constraints; partial match scoring | Unit tests S1–S5 |
| **3 — Parallel detector** | `ParallelEpisodeMatcher`: detect co-occurring alarm pairs on redundant components; CCF flag | Unit tests P1–P4 |
| **4 — Episode library** | `EpisodeLibrary`: JSON-backed store; frequency tracking; serialise/deserialise | Unit tests L1–L4 |
| **5 — Scorer** | `AlarmEpisodeScorerV1`: orchestrate encoder + matchers + library lookup; produce per-FM evidence dict | Unit tests SC1–SC6, I1–I3 |
| **6 — Demos** | Extend TC-3 and TC-5 notebooks with episode analysis section | TC-3 and TC-5 demo notebooks |

---

*This is a planning document — no code has been written yet. Review open questions before starting Phase 1.*
