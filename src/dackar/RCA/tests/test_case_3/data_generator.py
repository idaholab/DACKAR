#!/usr/bin/env python3
"""
RCA Pipeline Test Case
======================
Scenario : PWR Unit 2 — Condenser Vacuum Loss / Turbine Load Runback
Event    : EVT-U2-2024-0847  (2024-07-14 03:22 UTC)
Asset    : U2-CONDENSER-MAIN (main condenser, secondary side)

TRUE ROOT CAUSE    : Air in-leakage through turbine exhaust duct expansion joint
CONTRIBUTING CAUSE : HVAC fan bearing failure → elevated pit ambient temperature
                     → accelerated expansion joint seal thermal fatigue
RED HERRING        : Condenser waterbox tube cleaning 21 days prior (found acceptable)
RECURRENCE TRAP    : Most recent similar event (18 months ago) had FOULING as root
                     cause — the recurrence scorer must weight all historical analogs,
                     not just the most recent one.  Air in-leakage has 2 prior events
                     vs fouling's 1 — recurrence should favour air in-leakage.
KEY DISCRIMINATOR  : Hotwell dissolved oxygen = 142 ppb (normal < 10 ppb).
                     DO elevation is diagnostic for air in-leakage and is NOT produced
                     by tube fouling.  The condenser tube inspection WO explicitly shows
                     "within normal limits" — contradicting evidence for fouling.

Pipeline assertion targets
--------------------------
A1  Primary hypothesis maps to FM-CND-AIR-INLEAK (air in-leakage)
A2  FM-CND-TUBE-FOUL appears in alternatives, not as primary
A3  FM-CND-TUBE-LEAK filtered out or flagged temporal_contradiction
       (expected latency 2-48 h vs 336-h gradual drift)
A4  WO-2024-11847 classified as contradicting evidence for fouling candidate
A5  Score gap between #1 and #2 candidates >= 0.05 after evidence refinement
A6  Air in-leakage recurrence score >= fouling recurrence score
       (2 prior air-inleak events vs 1 fouling event)
A7  FM-CW-TEMP-RISE not primary and ranked position >= 2
A8  FM-HVAC-DEGRAD present in Ishikawa matrix
A9  Analyst review questions mention expansion joint / inspection / PM deferral
A10 RCA card: schema_valid=True AND all_claims_cited=True

Usage
-----
    # 1. Add your repo src/ to PYTHONPATH
    export PYTHONPATH=/path/to/repo/src:$PYTHONPATH

    # 2. Run
    python test_case_condenser_vacuum_loss.py

    # 3. Fixtures are written to ./test_fixtures/
    #    Pipeline output is written to ./test_output/<run_id>/

Requirements (pip install ...)
    neo4j          (imported by rca_reasoning_orchestrator even if not called)
    jsonschema
    langchain-core langchain-community langchain-chroma
    rank-bm25

    The test uses InMemoryEvidenceStore and FixtureKGContextBuilder so
    no live Neo4j or Chroma instance is required.

Notes
-----
  stop_on_validation_error is set to False for this test because the pre-built
  KG context fixture includes extra fields (asset_id, subgraph_id, generated_at,
  hop_limit) beyond the strict v2 schema that are required by the causality
  engine.  Switch to the original kg_context.json schema (not v2) and set
  stop_on_validation_error=True for full schema-compliant runs.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrators.causality_engine_v32 import (
    CausalityEngineConfigV32,
    RuleBasedCausalityEngineV32,
)
from orchestrators.evidence_retriever import (
    ChromaEvidenceRetriever,
    EvidenceRetrieverConfig,
    InMemoryEvidenceStore,
)
from orchestrators.rca_reasoning_orchestrator import (
    FileArtifactStore,
    HeuristicIshikawaEvaluatorV1,
    NoOpSchemaValidator,
    OrchestratorConfig,
    RCAReasoningOrchestrator,
)
from orchestrators.tskr_temporal_scorer import TSKRTemporalScorerV1
from synthesis.rca_synthesizer_v31 import (
    DummyLLMClient,
    RCASynthesizerConfig,
    RuleValidatedRCASynthesizerV31,
)

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: EVENT
# ─────────────────────────────────────────────────────────────────────────────

EVENT: Dict[str, Any] = {
    "event_id": "EVT-U2-2024-0847",
    "timestamp_start": "2024-07-14T03:22:00Z",
    "timestamp_end": "2024-07-14T03:45:00Z",
    "asset_id": "U2-CONDENSER-MAIN",
    "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
    "severity": "HIGH",
    "event_type": "DEGRADATION",
    "trigger_source": "alarm",
    "symptom_signature": {
        "description": (
            "Condenser backpressure reached automatic turbine load runback setpoint of 3.0 inHg "
            "following 14-day monotonic rise from 1.8 inHg baseline. Turbine load reduced from "
            "97 percent to 85 percent on automatic actuation. Hotwell dissolved oxygen elevated "
            "to 142 ppb (normal < 10 ppb) — indicative of air in-leakage."
        ),
        "symptom_types": ["pressure", "temperature"],
        "anomaly_pattern": "gradual_drift",
        "affected_parameters": [
            {
                "parameter": "condenser_backpressure",
                "sensor_id": "U2-PT-1847A",
                "observed_value": 3.02,
                "unit": "inHg",
                "normal_range": {"min": 1.5, "max": 2.8},
            },
            {
                "parameter": "hotwell_dissolved_oxygen",
                "sensor_id": "U2-AIT-0341",
                "observed_value": 142.0,
                "unit": "ppb",
                "normal_range": {"min": 0.0, "max": 10.0},
            },
            {
                "parameter": "turbine_exhaust_temperature",
                "sensor_id": "U2-TE-2201",
                "observed_value": 134.2,
                "unit": "degF",
                "normal_range": {"min": 100.0, "max": 130.0},
            },
        ],
    },
    "related_cr_ids": ["CR-2024-04821", "CR-2024-04799"],
    "related_alarm_ids": ["ALM-U2-CNDSR-BP-HH", "ALM-U2-TRB-RUNBACK"],
    "reported_by": "U2-OPS-SHIFT-D",
    "reported_at": "2024-07-14T03:30:00Z",
    "initial_assessment": (
        "Turbine automatic runback on high condenser backpressure. Trending upward for "
        "approximately 14 days. Hotwell dissolved oxygen elevated at 142 ppb. Engineering "
        "evaluation required to discriminate air in-leakage from tube fouling."
    ),
    "rca_required": True,
}

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: TELEMETRY SUMMARY
# Analysis window: 14-day precursor + event (2024-06-30 to 2024-07-14)
# Baseline:        prior 29-day stable period (2024-06-01 to 2024-06-29)
#
# Signal design:
#   ANOMALY   — U2-PT-1847A  backpressure       gradual_drift + sustained_exceedance
#   ANOMALY   — U2-AIT-0341  hotwell DO          gradual_drift + sustained_exceedance (KEY)
#   ANOMALY   — U2-TE-2201   exhaust temp        gradual_drift
#   ANOMALY   — U2-VIB-7701  HVAC fan vibration  step_change (Day 4)
#   ANOMALY   — U2-TE-5501   pit ambient temp    gradual_drift (onset Day 4, after HVAC trip)
#   NO ANOMALY — U2-TE-4401  CW inlet temp       seasonal rise, within bounds
#   NO ANOMALY — U2-FT-3301A condensate flow     slight reduction, not diagnostic
#   NO ANOMALY — U2-TE-7801  tube outlet temp    within limits  → CONTRADICTS fouling
#   NO ANOMALY — U2-AIT-0342 condensate conduct. within limits  → CONTRADICTS tube leakage
# ─────────────────────────────────────────────────────────────────────────────

TELEMETRY_SUMMARY: Dict[str, Any] = {
    "asset_id": "U2-CONDENSER-MAIN",
    "event_id": "EVT-U2-2024-0847",
    "generated_at": "2024-07-14T06:00:00Z",
    "window": {
        "start": "2024-06-30T00:00:00Z",
        "end": "2024-07-14T04:00:00Z",
        "baseline_start": "2024-06-01T00:00:00Z",
        "baseline_end": "2024-06-29T23:59:59Z",
    },
    "analysis_methods": [
        "descriptive_stats",
        "baseline_comparison",
        "changepoint_detection",
        "threshold_check",
    ],
    "signals": [
        # ── 1. Condenser backpressure — primary alarm signal ─────────────────
        {
            "sensor_id": "U2-PT-1847A",
            "parameter": "condenser_backpressure",
            "unit": "inHg",
            "stats": {
                "mean": 2.41, "std": 0.31, "min": 1.83, "max": 3.02,
                "p25": 1.95, "p75": 2.72, "p95": 2.94, "sample_count": 20160,
            },
            "baseline_comparison": {
                "baseline_mean": 1.82, "baseline_std": 0.04,
                "delta_mean": 0.59, "delta_std": 0.27,
                "percent_change": 32.4, "deviation_sigma": 14.75,
            },
            "anomalies": [
                {
                    "anomaly_id": "ANOM-BP-001",
                    "detection_method": "changepoint_detection",
                    "pattern": "gradual_drift",
                    "timestamp_start": "2024-06-30T12:00:00Z",
                    "timestamp_end": "2024-07-12T14:00:00Z",
                    "severity": "low",
                    "score": 0.35,
                    "peak_value": 2.51,
                    "promoted_to_kg_event": False,
                },
                {
                    "anomaly_id": "ANOM-BP-002",
                    "detection_method": "threshold_exceedance",
                    "pattern": "sustained_exceedance",
                    "timestamp_start": "2024-07-12T14:15:00Z",
                    "timestamp_end": "2024-07-14T03:45:00Z",
                    "severity": "high",
                    "score": 0.88,
                    "peak_value": 3.02,
                    "promoted_to_kg_event": True,
                    "kg_event_id": "EVT-U2-2024-0847",
                    "promotion_rationale": (
                        "Backpressure exceeded high-high setpoint triggering automatic "
                        "turbine runback — constitutes an abnormal plant event."
                    ),
                },
            ],
            "changepoints": [
                {
                    "changepoint_id": "CP-BP-001",
                    "timestamp": "2024-06-30T08:00:00Z",
                    "confidence": 0.91,
                    "before_mean": 1.82,
                    "after_mean": 1.95,
                    "detection_method": "changepoint_detection",
                },
            ],
            "within_normal_limits": False,
            "data_quality": {
                "missing_fraction": 0.001, "flatline_detected": False,
                "outlier_fraction": 0.002, "quality_flags": [],
            },
        },
        # ── 2. Hotwell dissolved oxygen — KEY DISCRIMINATING SIGNAL ──────────
        # DO elevation: 142 ppb vs 6.2 ppb baseline = 23× increase.
        # Caused exclusively by air in-leakage, not tube fouling.
        # Onset Day 2 of precursor — earlier than backpressure alarm.
        {
            "sensor_id": "U2-AIT-0341",
            "parameter": "hotwell_dissolved_oxygen",
            "unit": "ppb",
            "stats": {
                "mean": 89.4, "std": 41.2, "min": 6.8, "max": 142.0,
                "p25": 18.3, "p75": 128.4, "p95": 139.7, "sample_count": 2016,
            },
            "baseline_comparison": {
                "baseline_mean": 6.2, "baseline_std": 0.8,
                "delta_mean": 83.2, "delta_std": 40.4,
                "percent_change": 1341.9, "deviation_sigma": 104.0,
            },
            "anomalies": [
                {
                    "anomaly_id": "ANOM-DO-001",
                    "detection_method": "baseline_deviation",
                    "pattern": "gradual_drift",
                    "timestamp_start": "2024-07-02T06:00:00Z",
                    "timestamp_end": "2024-07-09T23:00:00Z",
                    "severity": "medium",
                    "score": 0.72,
                    "peak_value": 85.0,
                    "promoted_to_kg_event": False,
                },
                {
                    "anomaly_id": "ANOM-DO-002",
                    "detection_method": "threshold_exceedance",
                    "pattern": "sustained_exceedance",
                    "timestamp_start": "2024-07-10T00:00:00Z",
                    "timestamp_end": "2024-07-14T03:45:00Z",
                    "severity": "high",
                    "score": 0.94,
                    "peak_value": 142.0,
                    "promoted_to_kg_event": False,
                },
            ],
            "changepoints": [
                {
                    "changepoint_id": "CP-DO-001",
                    "timestamp": "2024-07-02T04:00:00Z",
                    "confidence": 0.87,
                    "before_mean": 6.4,
                    "after_mean": 28.3,
                    "detection_method": "changepoint_detection",
                },
            ],
            "within_normal_limits": False,
            "data_quality": {
                "missing_fraction": 0.003, "flatline_detected": False,
                "outlier_fraction": 0.001, "quality_flags": [],
            },
        },
        # ── 3. Turbine exhaust temperature — correlated with backpressure ────
        {
            "sensor_id": "U2-TE-2201",
            "parameter": "turbine_exhaust_temperature",
            "unit": "degF",
            "stats": {
                "mean": 121.8, "std": 6.3, "min": 113.1, "max": 134.2,
                "p25": 115.4, "p75": 128.1, "p95": 132.9, "sample_count": 20160,
            },
            "baseline_comparison": {
                "baseline_mean": 112.4, "baseline_std": 2.1,
                "delta_mean": 9.4, "delta_std": 4.2,
                "percent_change": 8.4, "deviation_sigma": 4.48,
            },
            "anomalies": [
                {
                    "anomaly_id": "ANOM-TE-001",
                    "detection_method": "baseline_deviation",
                    "pattern": "gradual_drift",
                    "timestamp_start": "2024-07-01T00:00:00Z",
                    "timestamp_end": "2024-07-14T03:45:00Z",
                    "severity": "medium",
                    "score": 0.61,
                    "peak_value": 134.2,
                    "promoted_to_kg_event": False,
                },
            ],
            "changepoints": [],
            "within_normal_limits": False,
            "data_quality": {
                "missing_fraction": 0.0, "flatline_detected": False,
                "outlier_fraction": 0.001, "quality_flags": [],
            },
        },
        # ── 4. HVAC fan motor bearing vibration — step_change Day 4 ─────────
        # Precedes pit ambient temperature rise.  Links HVAC degradation to
        # the expansion joint thermal fatigue causal chain.
        {
            "sensor_id": "U2-VIB-7701",
            "parameter": "hvac_fan_motor_bearing_vibration",
            "unit": "in_s_peak",
            "stats": {
                "mean": 0.31, "std": 0.18, "min": 0.07, "max": 0.89,
                "p25": 0.09, "p75": 0.51, "p95": 0.82, "sample_count": 20160,
            },
            "baseline_comparison": {
                "baseline_mean": 0.08, "baseline_std": 0.01,
                "delta_mean": 0.23, "delta_std": 0.17,
                "percent_change": 287.5, "deviation_sigma": 23.0,
            },
            "anomalies": [
                {
                    "anomaly_id": "ANOM-VIB-001",
                    "detection_method": "threshold_exceedance",
                    "pattern": "step_change",
                    "timestamp_start": "2024-07-04T09:00:00Z",
                    "timestamp_end": "2024-07-10T11:22:00Z",
                    "severity": "high",
                    "score": 0.82,
                    "peak_value": 0.89,
                    "promoted_to_kg_event": False,
                },
            ],
            "changepoints": [
                {
                    "changepoint_id": "CP-VIB-001",
                    "timestamp": "2024-07-04T09:10:00Z",
                    "confidence": 0.95,
                    "before_mean": 0.08,
                    "after_mean": 0.52,
                    "detection_method": "changepoint_detection",
                },
            ],
            "within_normal_limits": False,
            "data_quality": {
                "missing_fraction": 0.0, "flatline_detected": False,
                "outlier_fraction": 0.003, "quality_flags": [],
            },
        },
        # ── 5. Condenser pit ambient temperature — onset coincides with HVAC ─
        {
            "sensor_id": "U2-TE-5501",
            "parameter": "condenser_pit_ambient_temperature",
            "unit": "degF",
            "stats": {
                "mean": 89.3, "std": 4.2, "min": 82.3, "max": 97.6,
                "p25": 84.1, "p75": 93.8, "p95": 96.7, "sample_count": 20160,
            },
            "baseline_comparison": {
                "baseline_mean": 82.1, "baseline_std": 1.8,
                "delta_mean": 7.2, "delta_std": 2.4,
                "percent_change": 8.8, "deviation_sigma": 4.0,
            },
            "anomalies": [
                {
                    "anomaly_id": "ANOM-AMB-001",
                    "detection_method": "baseline_deviation",
                    "pattern": "gradual_drift",
                    "timestamp_start": "2024-07-04T09:00:00Z",
                    "timestamp_end": "2024-07-14T03:45:00Z",
                    "severity": "medium",
                    "score": 0.55,
                    "peak_value": 97.6,
                    "promoted_to_kg_event": False,
                },
            ],
            "changepoints": [],
            "within_normal_limits": False,
            "data_quality": {
                "missing_fraction": 0.001, "flatline_detected": False,
                "outlier_fraction": 0.0, "quality_flags": [],
            },
        },
        # ── 6. CW inlet temperature — NO ANOMALY (seasonal rise only) ────────
        # 4.2 degF seasonal rise is within expected summer range.
        # Insufficient alone to explain 1.2 inHg backpressure increase.
        # Operator increasing CW flow had minimal effect — excludes CW temperature
        # as root cause.
        {
            "sensor_id": "U2-TE-4401",
            "parameter": "circulating_water_inlet_temperature",
            "unit": "degF",
            "stats": {
                "mean": 75.4, "std": 0.9, "min": 73.8, "max": 77.2,
                "p25": 74.7, "p75": 76.1, "p95": 76.9, "sample_count": 20160,
            },
            "baseline_comparison": {
                "baseline_mean": 71.2, "baseline_std": 1.4,
                "delta_mean": 4.2, "delta_std": -0.5,
                "percent_change": 5.9, "deviation_sigma": 3.0,
            },
            "anomalies": [],
            "changepoints": [],
            "within_normal_limits": True,
            "data_quality": {
                "missing_fraction": 0.0, "flatline_detected": False,
                "outlier_fraction": 0.0, "quality_flags": [],
            },
        },
        # ── 7. Condensate flow — NO ANOMALY ──────────────────────────────────
        {
            "sensor_id": "U2-FT-3301A",
            "parameter": "condensate_flow_train_a",
            "unit": "gpm",
            "stats": {
                "mean": 18210, "std": 385, "min": 15100, "max": 18490,
                "p25": 18050, "p75": 18400, "p95": 18460, "sample_count": 20160,
            },
            "baseline_comparison": {
                "baseline_mean": 18450, "baseline_std": 120,
                "delta_mean": -240.0, "delta_std": 265.0,
                "percent_change": -1.3, "deviation_sigma": -2.0,
            },
            "anomalies": [],
            "changepoints": [],
            "within_normal_limits": True,
            "data_quality": {
                "missing_fraction": 0.0, "flatline_detected": False,
                "outlier_fraction": 0.0, "quality_flags": [],
            },
        },
        # ── 8. Condenser tube outlet temperature — NO ANOMALY ─────────────────
        # Normal tube outlet temps CONTRADICT tube fouling hypothesis.
        # Fouling reduces heat transfer → elevated outlet temps expected.
        {
            "sensor_id": "U2-TE-7801",
            "parameter": "condenser_tube_outlet_temperature_waterbox_a",
            "unit": "degF",
            "stats": {
                "mean": 104.7, "std": 0.9, "min": 103.6, "max": 106.1,
                "p25": 104.0, "p75": 105.4, "p95": 105.9, "sample_count": 20160,
            },
            "baseline_comparison": {
                "baseline_mean": 104.3, "baseline_std": 0.8,
                "delta_mean": 0.4, "delta_std": 0.1,
                "percent_change": 0.4, "deviation_sigma": 0.5,
            },
            "anomalies": [],
            "changepoints": [],
            "within_normal_limits": True,
            "data_quality": {
                "missing_fraction": 0.0, "flatline_detected": False,
                "outlier_fraction": 0.0, "quality_flags": [],
            },
        },
        # ── 9. Condensate specific conductivity — NO ANOMALY ──────────────────
        # Normal conductivity CONTRADICTS tube leakage hypothesis.
        # Tube leakage would introduce CW chemistry → elevated conductivity.
        {
            "sensor_id": "U2-AIT-0342",
            "parameter": "condensate_specific_conductivity",
            "unit": "uS_cm",
            "stats": {
                "mean": 0.13, "std": 0.01, "min": 0.11, "max": 0.15,
                "p25": 0.12, "p75": 0.14, "p95": 0.14, "sample_count": 2016,
            },
            "baseline_comparison": {
                "baseline_mean": 0.12, "baseline_std": 0.01,
                "delta_mean": 0.01, "delta_std": 0.0,
                "percent_change": 8.3, "deviation_sigma": 1.0,
            },
            "anomalies": [],
            "changepoints": [],
            "within_normal_limits": True,
            "data_quality": {
                "missing_fraction": 0.0, "flatline_detected": False,
                "outlier_fraction": 0.0, "quality_flags": [],
            },
        },
    ],
    "overall_assessment": {
        "any_anomaly_detected": True,
        "anomaly_count": 8,
        "most_anomalous_signal": "U2-AIT-0341",
        "earliest_anomaly_time": "2024-06-30T12:00:00Z",
        "pattern_summary": (
            "Five of nine signals show anomalies across three timescales. "
            "Condenser backpressure gradual drift (onset Day 0) is the primary alarm driver. "
            "Hotwell dissolved oxygen elevated 23x above baseline (onset Day 2, severity_score=0.94): "
            "most diagnostically significant — DO elevation is caused by air in-leakage and is not "
            "consistent with tube fouling. HVAC fan motor bearing vibration step change (onset Day 4) "
            "preceded pit ambient temperature rise by less than 1 hour, establishing causal sequence. "
            "Condenser tube outlet temperatures, condensate conductivity, and CW flow all within "
            "normal limits — inconsistent with tube fouling or tube leakage hypotheses. "
            "CW inlet temperature seasonal rise of 4.2 degF is within expected summer range and "
            "insufficient alone to explain the observed 1.2 inHg backpressure increase. "
            "Operator action to increase CW pump speed had minimal effect (0.04 inHg reduction), "
            "further contradicting CW temperature as the dominant cause."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: KG CONTEXT
# Uses the v2 schema format augmented with fields required by the causality
# engine (asset_id, subgraph_id, generated_at, hop_limit).
#
# Components: 9 (expansion joint seed + 8 neighbors)
# Failure modes: 5 (air inleak, tube foul, tube leak, CW temp, HVAC degrad)
# Past events: 3 (recurrence trap embedded — see comments)
# Documents: 9
# ─────────────────────────────────────────────────────────────────────────────

KG_CONTEXT: Dict[str, Any] = {
    # Fields beyond strict v2 schema — needed by causality engine & orchestrator
    "event_id": "EVT-U2-2024-0847",
    "asset_id": "U2-CONDENSER-MAIN",
    "subgraph_id": "KGCTX::EVT-U2-2024-0847::U2-CONDENSER-MAIN",
    "generated_at": "2024-07-14T06:00:00Z",
    "hop_limit": 2,

    "components": [
        # seed — primary event component
        {
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "name": "Turbine Exhaust Duct Expansion Joint",
            "type": "expansion_joint",
            "role": "pressure_boundary",
            "asset_id": "U2-CONDENSER-MAIN",
            "seed_match_type": "seed",
        },
        # telemetry seed — hotwell DO sensor maps here
        {
            "component_id": "U2-CND-HOTWELL",
            "name": "Condenser Hotwell",
            "type": "hotwell",
            "role": "condensate_collection",
            "asset_id": "U2-CONDENSER-MAIN",
            "seed_match_type": "telemetry",
        },
        {
            "component_id": "U2-CND-WATERBOX-A",
            "name": "Condenser Waterbox A",
            "type": "waterbox",
            "role": "heat_exchange",
            "asset_id": "U2-CONDENSER-MAIN",
            "seed_match_type": "neighbor",
        },
        {
            "component_id": "U2-CND-WATERBOX-B",
            "name": "Condenser Waterbox B",
            "type": "waterbox",
            "role": "heat_exchange",
            "asset_id": "U2-CONDENSER-MAIN",
            "seed_match_type": "neighbor",
        },
        {
            "component_id": "U2-CND-TUBE-BUNDLE-A",
            "name": "Condenser Tube Bundle A",
            "type": "heat_exchanger_tubes",
            "role": "heat_transfer_surface",
            "asset_id": "U2-CONDENSER-MAIN",
            "seed_match_type": "neighbor",
        },
        {
            "component_id": "U2-CND-TUBE-BUNDLE-B",
            "name": "Condenser Tube Bundle B",
            "type": "heat_exchanger_tubes",
            "role": "heat_transfer_surface",
            "asset_id": "U2-CONDENSER-MAIN",
            "seed_match_type": "neighbor",
        },
        {
            "component_id": "U2-AIR-EJECTOR-A",
            "name": "Air Ejector Train A",
            "type": "air_removal",
            "role": "non_condensable_gas_removal",
            "asset_id": "U2-CONDENSER-MAIN",
            "seed_match_type": "neighbor",
        },
        {
            "component_id": "U2-AIR-EJECTOR-B",
            "name": "Air Ejector Train B",
            "type": "air_removal",
            "role": "non_condensable_gas_removal",
            "asset_id": "U2-CONDENSER-MAIN",
            "seed_match_type": "neighbor",
        },
        # ops_context — HVAC fan linked via nearby_maintenance and VIB telemetry
        {
            "component_id": "U2-HVAC-TURBINE-BAY-FAN-A",
            "name": "Turbine Bay HVAC Fan Motor A",
            "type": "hvac_fan",
            "role": "turbine_building_ventilation",
            "asset_id": "U2-HVAC-TURBINE-BAY",
            "seed_match_type": "ops_context",
        },
    ],

    "upstream_paths": [
        {
            "from": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "to": "U2-CND-HOTWELL",
            "path": ["U2-CND-EXPANSION-JOINT-EXHAUST", "U2-CND-HOTWELL"],
            "edge_types": ["UPSTREAM_OF"],
            "path_strength": 0.90,
        },
        {
            "from": "U2-CND-TUBE-BUNDLE-A",
            "to": "U2-CND-HOTWELL",
            "path": ["U2-CND-TUBE-BUNDLE-A", "U2-CND-HOTWELL"],
            "edge_types": ["UPSTREAM_OF"],
            "path_strength": 0.85,
        },
        {
            "from": "U2-CND-TUBE-BUNDLE-B",
            "to": "U2-CND-HOTWELL",
            "path": ["U2-CND-TUBE-BUNDLE-B", "U2-CND-HOTWELL"],
            "edge_types": ["UPSTREAM_OF"],
            "path_strength": 0.85,
        },
        # HVAC → expansion joint: thermal path (reduced ventilation → elevated pit
        # ambient → accelerated expansion joint seal degradation)
        {
            "from": "U2-HVAC-TURBINE-BAY-FAN-A",
            "to": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "path": ["U2-HVAC-TURBINE-BAY-FAN-A", "U2-CND-EXPANSION-JOINT-EXHAUST"],
            "edge_types": ["CONNECTED_TO"],
            "path_strength": 0.60,
        },
        {
            "from": "U2-AIR-EJECTOR-A",
            "to": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "path": ["U2-AIR-EJECTOR-A", "U2-CND-EXPANSION-JOINT-EXHAUST"],
            "edge_types": ["UPSTREAM_OF"],
            "path_strength": 0.75,
        },
    ],

    "failure_modes": [
        # FM-1: TRUE ROOT CAUSE — latency 48-336 h covers the 336-h gradual drift
        {
            "fm_id": "FM-CND-AIR-INLEAK",
            "name": "Air in-leakage through boundary",
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "component_name": "Turbine Exhaust Duct Expansion Joint",
            "superclass": "pressure_boundary_failure",
            "expected_latency_min_hours": 48.0,
            "expected_latency_max_hours": 336.0,
        },
        # FM-2: RED HERRING — tube inspection passed; DO not elevated in fouling
        {
            "fm_id": "FM-CND-TUBE-FOUL",
            "name": "Condenser tube fouling",
            "component_id": "U2-CND-TUBE-BUNDLE-A",
            "component_name": "Condenser Tube Bundle A",
            "superclass": "heat_transfer_degradation",
            "expected_latency_min_hours": 168.0,
            "expected_latency_max_hours": 720.0,
        },
        # FM-3: SHOULD BE FILTERED — expected latency 2-48 h vs 336-h drift
        #        temporal_contradiction expected: latency_violation_type=too_slow
        {
            "fm_id": "FM-CND-TUBE-LEAK",
            "name": "Condenser tube leakage",
            "component_id": "U2-CND-TUBE-BUNDLE-A",
            "component_name": "Condenser Tube Bundle A",
            "superclass": "pressure_boundary_failure",
            "expected_latency_min_hours": 2.0,
            "expected_latency_max_hours": 48.0,
        },
        # FM-4: CONTRIBUTING FACTOR ONLY — seasonal rise insufficient alone;
        #        operator increased CW flow with minimal effect (0.04 inHg)
        {
            "fm_id": "FM-CW-TEMP-RISE",
            "name": "Circulating water inlet temperature elevation",
            "component_id": "U2-CND-WATERBOX-A",
            "component_name": "Condenser Waterbox A",
            "superclass": "thermal_performance_degradation",
            "expected_latency_min_hours": 0.0,
            "expected_latency_max_hours": 720.0,
        },
        # FM-5: CONTRIBUTING CAUSE — HVAC PM overdue 60 days; fan trip Day 4
        {
            "fm_id": "FM-HVAC-DEGRAD",
            "name": "HVAC cooling capacity reduction",
            "component_id": "U2-HVAC-TURBINE-BAY-FAN-A",
            "component_name": "Turbine Bay HVAC Fan Motor A",
            "superclass": "auxiliary_system_degradation",
            "expected_latency_min_hours": 24.0,
            "expected_latency_max_hours": 120.0,
        },
    ],

    "past_events": [
        # ── RECURRENCE TRAP: most recent event was FOULING ───────────────────
        # A system over-weighting recency will incorrectly elevate FM-CND-TUBE-FOUL.
        # Correct behavior: FM-CND-AIR-INLEAK has 2 prior events vs fouling's 1.
        {
            "event_id": "EVT-U2-2022-1103",
            "asset_id": "U2-CONDENSER-MAIN",
            "component_id": "U2-CND-TUBE-BUNDLE-A",
            "timestamp_start": "2023-01-21T00:00:00Z",
            "timestamp_end": "2023-01-21T08:00:00Z",
            "severity": "MEDIUM",
            "event_type": "DEGRADATION",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-TUBE-BUNDLE-A", "U2-CND-TUBE-BUNDLE-B"],
            "matched_failure_mode_ids": ["FM-CND-TUBE-FOUL"],
            "priority_score": 18.5,
            "time_distance_days": 540,
        },
        # ── Prior air in-leakage event 1 (36 months ago) ─────────────────────
        {
            "event_id": "EVT-U2-2021-0612",
            "asset_id": "U2-CONDENSER-MAIN",
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "timestamp_start": "2021-07-14T00:00:00Z",
            "timestamp_end": "2021-07-14T06:00:00Z",
            "severity": "HIGH",
            "event_type": "DEGRADATION",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-EXPANSION-JOINT-EXHAUST"],
            "matched_failure_mode_ids": ["FM-CND-AIR-INLEAK"],
            "priority_score": 22.0,
            "time_distance_days": 1095,
        },
        # ── Prior air in-leakage event 2 (60 months ago) ─────────────────────
        {
            "event_id": "EVT-U2-2019-1847",
            "asset_id": "U2-CONDENSER-MAIN",
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "timestamp_start": "2019-07-14T00:00:00Z",
            "timestamp_end": "2019-07-14T12:00:00Z",
            "severity": "HIGH",
            "event_type": "DEGRADATION",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-EXPANSION-JOINT-EXHAUST"],
            "matched_failure_mode_ids": ["FM-CND-AIR-INLEAK"],
            "priority_score": 21.0,
            "time_distance_days": 1826,
        },
    ],

    "documents": [
        {
            "doc_id": "CR-2024-04821",
            "doc_type": "CR",
            "title": "Post-event CR: Turbine runback on high condenser backpressure with elevated hotwell DO",
            "created_at": "2024-07-14T04:30:00Z",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-EXPANSION-JOINT-EXHAUST", "U2-CND-HOTWELL"],
            "priority_score": 125.0,
            "time_distance_days": 0,
        },
        {
            "doc_id": "CR-2024-04799",
            "doc_type": "CR",
            "title": "Trend CR: Condenser backpressure rise over 9 days — engineering evaluation requested",
            "created_at": "2024-07-09T10:00:00Z",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-EXPANSION-JOINT-EXHAUST"],
            "priority_score": 115.0,
            "time_distance_days": 5,
        },
        {
            "doc_id": "WO-2024-11847",
            "doc_type": "WO",
            "title": "Condenser Waterbox A tube cleaning and inspection — PM",
            "created_at": "2024-06-23T16:00:00Z",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-WATERBOX-A", "U2-CND-TUBE-BUNDLE-A"],
            "priority_score": 108.0,
            "time_distance_days": 21,
        },
        {
            "doc_id": "WO-2024-12001",
            "doc_type": "WO",
            "title": "Helium leak test of condenser pressure boundaries — post-event",
            "created_at": "2024-07-15T08:00:00Z",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-EXPANSION-JOINT-EXHAUST"],
            "priority_score": 112.0,
            "time_distance_days": 1,
        },
        {
            "doc_id": "WO-2024-11901",
            "doc_type": "WO",
            "title": "HVAC turbine bay fan A motor bearing replacement — corrective maintenance",
            "created_at": "2024-07-10T11:30:00Z",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-HVAC-TURBINE-BAY-FAN-A"],
            "priority_score": 97.0,
            "time_distance_days": 4,
        },
        {
            "doc_id": "SOP-U2-CND-001",
            "doc_type": "SOP",
            "title": "Condenser Performance Monitoring and Backpressure Diagnosis Procedure",
            "created_at": "2022-03-15T00:00:00Z",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-EXPANSION-JOINT-EXHAUST", "U2-CND-HOTWELL"],
            "priority_score": 70.0,
            "time_distance_days": None,
        },
        {
            "doc_id": "SOP-U2-CHE-041",
            "doc_type": "SOP",
            "title": "Secondary Chemistry Surveillance — Condensate and Feedwater Monitoring",
            "created_at": "2021-11-01T00:00:00Z",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-HOTWELL"],
            "priority_score": 65.0,
            "time_distance_days": None,
        },
        {
            "doc_id": "ECA-2022-1103",
            "doc_type": "ECA",
            "title": "Engineering Cause Analysis: Condenser backpressure rise Jan 2023 — confirmed tube fouling",
            "created_at": "2023-02-15T00:00:00Z",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-TUBE-BUNDLE-A", "U2-CND-TUBE-BUNDLE-B"],
            "priority_score": 98.0,
            "time_distance_days": 175,
        },
        {
            "doc_id": "OE-INPO-2023-CND-047",
            "doc_type": "BULLETIN",
            "title": "Industry OE: Condenser backpressure rise — air in-leakage misidentified as CW temperature",
            "created_at": "2023-06-15T00:00:00Z",
            "matched_asset_ids": ["U2-CONDENSER-MAIN"],
            "matched_component_ids": ["U2-CND-EXPANSION-JOINT-EXHAUST"],
            "priority_score": 55.0,
            "time_distance_days": 394,
        },
    ],

    "seed_context": {
        "asset_ids": ["U2-CONDENSER-MAIN"],
        "seed_component_ids": [
            "U2-CND-EXPANSION-JOINT-EXHAUST",
            "U2-CND-HOTWELL",
        ],
        "monitored_variables": [
            {
                "monitored_variable_id": "MV-U2-PT-1847A",
                "variable": "condenser_backpressure",
                "sensor_id": "U2-PT-1847A",
                "tag_id": "U2-PT-1847A",
                "source_system": "PI_HISTORIAN",
                "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
                "component_name": "Turbine Exhaust Duct Expansion Joint",
                "relation_type": "MONITORS",
                "matched_on": "sensor_id",
                "match_confidence": 1.0,
            },
            {
                "monitored_variable_id": "MV-U2-AIT-0341",
                "variable": "hotwell_dissolved_oxygen",
                "sensor_id": "U2-AIT-0341",
                "tag_id": "U2-AIT-0341",
                "source_system": "PI_HISTORIAN",
                "component_id": "U2-CND-HOTWELL",
                "component_name": "Condenser Hotwell",
                "relation_type": "MONITORS",
                "matched_on": "sensor_id",
                "match_confidence": 1.0,
            },
            {
                "monitored_variable_id": "MV-U2-TE-2201",
                "variable": "turbine_exhaust_temperature",
                "sensor_id": "U2-TE-2201",
                "tag_id": "U2-TE-2201",
                "source_system": "PI_HISTORIAN",
                "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
                "component_name": "Turbine Exhaust Duct Expansion Joint",
                "relation_type": "MONITORS",
                "matched_on": "sensor_id",
                "match_confidence": 1.0,
            },
            {
                "monitored_variable_id": "MV-U2-VIB-7701",
                "variable": "hvac_fan_motor_bearing_vibration",
                "sensor_id": "U2-VIB-7701",
                "tag_id": "U2-VIB-7701",
                "source_system": "VIBRATION_MON",
                "component_id": "U2-HVAC-TURBINE-BAY-FAN-A",
                "component_name": "Turbine Bay HVAC Fan Motor A",
                "relation_type": "MONITORS",
                "matched_on": "sensor_id",
                "match_confidence": 1.0,
            },
            {
                "monitored_variable_id": "MV-U2-TE-5501",
                "variable": "condenser_pit_ambient_temperature",
                "sensor_id": "U2-TE-5501",
                "tag_id": "U2-TE-5501",
                "source_system": "PI_HISTORIAN",
                "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
                "component_name": "Turbine Exhaust Duct Expansion Joint",
                "relation_type": "MONITORS",
                "matched_on": "sensor_id",
                "match_confidence": 0.85,
            },
        ],
    },
    "provenance": {
        "builder": "FixtureKGContextBuilder",
        "run_id": "TEST-FIXTURE-CONDENSER-VACUUM-001",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: OPERATIONAL CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONAL_CONTEXT: Dict[str, Any] = {
    "asset_id": "U2-CONDENSER-MAIN",
    "window": {
        "start": "2024-06-30T00:00:00Z",
        "end": "2024-07-14T04:00:00Z",
    },
    "mode": "steady",
    "operating_mode": "steady_state",
    "percent_rated_power": 97.0,
    "power_mw": 1145.0,
    "train_configuration": {
        "train_id": "NA",
        "in_service": True,
    },
    "recent_alarms": [
        {
            "alarm_id": "ALM-U2-HVAC-VIB-H",
            "timestamp": "2024-07-04T09:33:00Z",
            "description": "Turbine bay HVAC fan motor high vibration — 0.52 in/s peak exceeds 0.35 alarm setpoint",
            "priority": "medium",
            "actual_value": 0.52,
            "unit": "in_s_peak",
            "acknowledged_at": "2024-07-04T09:45:00Z",
            "acknowledged_by": "U2-MAINT-SHIFT",
            "system_affected": "U2-HVAC-TURBINE-BAY",
        },
        {
            "alarm_id": "ALM-U2-CNDSR-BP-H",
            "timestamp": "2024-07-12T14:15:00Z",
            "description": "Condenser backpressure high — 2.51 inHg exceeds 2.50 inHg alarm setpoint",
            "priority": "medium",
            "setpoint": 2.50,
            "actual_value": 2.51,
            "unit": "inHg",
            "acknowledged_at": "2024-07-12T14:22:00Z",
            "acknowledged_by": "U2-OPS-SHIFT-C",
            "system_affected": "U2-CONDENSER-MAIN",
        },
        {
            "alarm_id": "ALM-U2-CNDSR-BP-HH",
            "timestamp": "2024-07-14T03:22:00Z",
            "description": "Condenser backpressure high-high — automatic turbine runback setpoint 3.0 inHg actuated",
            "priority": "critical",
            "setpoint": 3.00,
            "actual_value": 3.02,
            "unit": "inHg",
            "acknowledged_at": "2024-07-14T03:25:00Z",
            "acknowledged_by": "U2-OPS-SHIFT-D",
            "system_affected": "U2-CONDENSER-MAIN",
        },
    ],
    "recent_operations": [
        {
            "timestamp": "2024-07-04T09:45:00Z",
            "action_type": "operator_action",
            "description": (
                "Engineering notified of HVAC fan A high vibration. Fan A tripped on "
                "automatic high-vibration protection at 0.89 in/s. Fan B placed in service. "
                "Condenser pit ambient temperature monitoring frequency increased."
            ),
            "system_affected": "U2-HVAC-TURBINE-BAY",
        },
        {
            "timestamp": "2024-07-10T11:22:00Z",
            "action_type": "state_change",
            "description": (
                "HVAC turbine bay fan A tripped on high vibration 0.89 in/s. "
                "Fan B confirmed in service. WO-2024-11901 issued for fan A bearing "
                "replacement — parts on order, no firm completion date."
            ),
            "system_affected": "U2-HVAC-TURBINE-BAY",
        },
        {
            "timestamp": "2024-07-12T16:00:00Z",
            "action_type": "operator_action",
            "description": (
                "Increased CW pump speed from 85 percent to 100 percent in response "
                "to backpressure trend per shift supervisor direction. Effect: backpressure "
                "reduced by only 0.04 inHg — minimal improvement. Engineering evaluation "
                "requested. Operator notes: CW temperature increase alone cannot explain "
                "the magnitude of backpressure rise observed."
            ),
            "system_affected": "U2-CW-SYSTEM",
        },
        {
            "timestamp": "2024-07-14T03:25:00Z",
            "action_type": "auto_actuation",
            "description": (
                "Turbine load runback automatically actuated at 3.0 inHg condenser "
                "backpressure. Reactor power reduced from 97 percent to 85 percent via "
                "automatic load follow. Operators entered EOPs and notified shift manager."
            ),
            "system_affected": "U2-CONDENSER-MAIN",
            "procedure_ref": "SOP-U2-CND-001",
        },
    ],
    "nearby_maintenance": [
        {
            "wo_id": "WO-2024-11847",
            "maintenance_type": "preventive",
            "component_id": "U2-CND-WATERBOX-A",
            "proximity": "same_asset",
            "completed_at": "2024-06-23T16:00:00Z",
            "notes": (
                "Condenser waterbox A tube cleaning and inspection per PM schedule. "
                "847 tubes inspected, 0 tubes plugged, as-left acceptable. "
                "No abnormality found."
            ),
        },
        {
            "wo_id": "WO-2024-11901",
            "maintenance_type": "corrective",
            "component_id": "U2-HVAC-TURBINE-BAY-FAN-A",
            "proximity": "adjacent_system",
            "completed_at": None,
            "notes": (
                "Fan A motor bearing replacement. Parts on order. Fan B providing "
                "full turbine bay ventilation coverage in interim."
            ),
        },
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: PM COMPLIANCE
# Key design: failed PM items are specific to air in-leakage precursors.
# The tube inspection PASSED — evidence against fouling.
# A uniform governance scorer will fail A6; a candidate-specific scorer will pass.
# ─────────────────────────────────────────────────────────────────────────────

PM_COMPLIANCE: Dict[str, Any] = {
    "asset_id": "U2-CONDENSER-MAIN",
    "window": {
        "start": "2024-04-01T00:00:00Z",
        "end": "2024-07-14T03:22:00Z",
    },
    "checks": [
        {
            "check_id": "PM-U2-CND-EXPJOINT-INSP-A",
            "check_type": "inspection",
            "source_ref": "SOP-U2-CND-001 §4.3",
            "status": "fail",
            "scheduled_date": "2024-04-01T00:00:00Z",
            "overdue_by_days": 104.0,
            "details": (
                "Annual condenser expansion joint visual inspection overdue 104 days. "
                "Deferred during last outage window due to scaffold unavailability. "
                "Not rescheduled. Expansion joint condition unknown at time of event. "
                "Directly relevant to air in-leakage hypothesis."
            ),
            "evidence_refs": [],
        },
        {
            "check_id": "PM-U2-AIR-EJECTOR-SURV-Q",
            "check_type": "surveillance_test",
            "source_ref": "TECH-SPEC-3.7.2",
            "status": "fail",
            "scheduled_date": "2024-07-02T00:00:00Z",
            "overdue_by_days": 12.0,
            "details": (
                "Quarterly air removal system performance surveillance overdue 12 days. "
                "Test not completed due to outage scheduling conflict. Air ejector "
                "performance unconfirmed — cannot verify adequate non-condensable gas removal. "
                "Directly relevant to air in-leakage and backpressure hypothesis."
            ),
            "evidence_refs": [],
        },
        {
            "check_id": "PM-U2-CND-TUBE-INSP-A",
            "check_type": "inspection",
            "source_ref": "SOP-U2-CND-001 §4.1",
            "status": "pass",
            "scheduled_date": "2024-06-23T00:00:00Z",
            "completed_date": "2024-06-23T16:00:00Z",
            "overdue_by_days": 0.0,
            "details": (
                "Condenser tube inspection per WO-2024-11847. 847 tubes inspected. "
                "0 tubes plugged. Tube sheet acceptable. No fouling, biological growth, "
                "or debris accumulation. Heat transfer surface in good condition. "
                "This passed PM item contradicts the tube fouling hypothesis."
            ),
            "evidence_refs": ["WO-2024-11847"],
        },
        {
            "check_id": "PM-U2-HVAC-PM-A",
            "check_type": "scheduled_pm",
            "source_ref": "PM-TASK-HVAC-BAY-MTR",
            "status": "fail",
            "scheduled_date": "2024-05-15T00:00:00Z",
            "overdue_by_days": 60.0,
            "details": (
                "HVAC fan motor bearing lubrication and vibration check overdue 60 days. "
                "Fan motor bearing failure on 2024-07-10 resulted in trip on high vibration. "
                "PM non-compliance is a latent contributor to HVAC degradation and the "
                "subsequent pit ambient temperature rise."
            ),
            "evidence_refs": ["WO-2024-11901"],
        },
    ],
    "overdue_items": [
        {
            "check_id": "PM-U2-CND-EXPJOINT-INSP-A",
            "check_type": "inspection",
            "scheduled_date": "2024-04-01T00:00:00Z",
            "overdue_by_days": 104.0,
            "source_ref": "SOP-U2-CND-001 §4.3",
        },
        {
            "check_id": "PM-U2-AIR-EJECTOR-SURV-Q",
            "check_type": "surveillance_test",
            "scheduled_date": "2024-07-02T00:00:00Z",
            "overdue_by_days": 12.0,
            "source_ref": "TECH-SPEC-3.7.2",
        },
        {
            "check_id": "PM-U2-HVAC-PM-A",
            "check_type": "scheduled_pm",
            "scheduled_date": "2024-05-15T00:00:00Z",
            "overdue_by_days": 60.0,
            "source_ref": "PM-TASK-HVAC-BAY-MTR",
        },
    ],
    "overdue_tasks": [
        {
            "check_id": "PM-U2-CND-EXPJOINT-INSP-A",
            "check_type": "inspection",
            "scheduled_date": "2024-04-01T00:00:00Z",
            "overdue_by_days": 104.0,
            "source_ref": "SOP-U2-CND-001 §4.3",
        },
        {
            "check_id": "PM-U2-AIR-EJECTOR-SURV-Q",
            "check_type": "surveillance_test",
            "scheduled_date": "2024-07-02T00:00:00Z",
            "overdue_by_days": 12.0,
            "source_ref": "TECH-SPEC-3.7.2",
        },
        {
            "check_id": "PM-U2-HVAC-PM-A",
            "check_type": "scheduled_pm",
            "scheduled_date": "2024-05-15T00:00:00Z",
            "overdue_by_days": 60.0,
            "source_ref": "PM-TASK-HVAC-BAY-MTR",
        },
    ],
    "summary": {
        "total_checks": 4,
        "passed": 1,
        "failed": 3,
        "unknown": 0,
        "overdue_count": 3,
        "last_pm_date": "2024-06-23T16:00:00Z",
        "next_pm_date": "2024-10-01T00:00:00Z",
        "compliance_rate": 0.25,
        "notes": (
            "Three of four checks failed. All failed items are associated with "
            "pressure boundary monitoring (expansion joint inspection), non-condensable "
            "gas removal (air ejector surveillance), and HVAC system maintenance — "
            "each directly linked to the air in-leakage causal chain. "
            "The tube inspection passed, providing evidence against the fouling hypothesis."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: EVIDENCE STORE ROWS
# These simulate what Chroma / the InMemoryEvidenceStore returns.
#
# Snippet design for the keyword classifier in ChromaEvidenceRetriever:
#   support_cues    : "caused by", "due to", "resulted in", "degraded",
#                     "failed", "wear", "leak", "fouling", "drift", "damage"
#   contradiction_cues: "no evidence of", "not observed", "within normal limits",
#                       "acceptable", "as left acceptable", "no abnormality",
#                       "failed to reproduce", "normal condition"
#
#  CR-2024-04821     → "caused by air in-leakage"           SUPPORTING for FM-CND-AIR-INLEAK
#  WO-2024-11847     → "within normal limits", "acceptable" CONTRADICTING for FM-CND-TUBE-FOUL
#  WO-2024-12001     → "caused by", "air in-leakage"        SUPPORTING for FM-CND-AIR-INLEAK
#  SOP-U2-CND-001    → diagnostic rules                     CONTEXTUAL/SUPPORTING
#  SOP-U2-CHE-041    → DO acceptance criteria               CONTEXTUAL
#  ECA-2022-1103     → prior fouling analysis (trap doc)     CONTEXTUAL (not contradicting)
#  OE-INPO-2023      → fleet OE on air in-leakage           SUPPORTING
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE_STORE_ROWS: List[Dict[str, Any]] = [
    # ── CR-2024-04821: Post-event condition report ────────────────────────────
    {
        "snippet_id": "CR-2024-04821::cause_statement",
        "doc_id": "CR-2024-04821",
        "section": "cause_statement",
        "snippet": (
            "Elevated hotwell dissolved oxygen at 142 ppb is caused by air in-leakage "
            "through a degraded expansion joint weld at the turbine exhaust duct. "
            "Air in-leakage resulted in degraded condenser vacuum and sustained backpressure rise. "
            "Tube fouling is contradicted by the recent tube inspection — no evidence of fouling, "
            "biological growth, or debris found. All tubes within normal limits. "
            "Dissolved oxygen elevation is not consistent with tube fouling mechanism."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "CR",
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "authority_level": "mandatory",
            "extraction_quality": 0.97,
        },
    },
    {
        "snippet_id": "CR-2024-04821::symptom_description",
        "doc_id": "CR-2024-04821",
        "section": "symptom_description",
        "snippet": (
            "Condenser backpressure gradual drift from 1.82 inHg baseline to 3.02 inHg "
            "over 14 days, resulting in automatic turbine load runback at 03:22 UTC. "
            "Hotwell dissolved oxygen elevated to 142 ppb at time of event — "
            "23 times above the normal limit of 10 ppb. Chemistry notified. "
            "Helium leak test work order issued for condenser boundary inspection per "
            "SOP-U2-CND-001 Step 4.2 diagnostic guidance."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "CR",
            "component_id": "U2-CND-HOTWELL",
            "authority_level": "mandatory",
            "extraction_quality": 0.97,
        },
    },
    {
        "snippet_id": "CR-2024-04821::corrective_actions",
        "doc_id": "CR-2024-04821",
        "section": "corrective_actions",
        "snippet": (
            "Immediate corrective action: reduce reactor power to relieve backpressure "
            "loading on expansion joint seal. Issue WO-2024-12001 for helium leak test "
            "of condenser expansion joint and all accessible pressure boundaries. "
            "Evaluate expansion joint condition — replacement anticipated if leak confirmed. "
            "Review and reschedule PM-U2-CND-EXPJOINT-INSP-A, currently overdue 104 days. "
            "Evaluate programmatic significance of PM deferral. Address air ejector "
            "surveillance PM-U2-AIR-EJECTOR-SURV-Q overdue 12 days."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "CR",
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "authority_level": "mandatory",
            "extraction_quality": 0.95,
        },
    },
    # ── CR-2024-04799: Pre-event trend CR (predates DO data) ─────────────────
    {
        "snippet_id": "CR-2024-04799::initial_assessment",
        "doc_id": "CR-2024-04799",
        "section": "initial_assessment",
        "snippet": (
            "Backpressure trend observed over past 9 days. Initial evaluation suggests "
            "possible circulating water temperature effect or condenser fouling. "
            "Recommend condenser performance test per SOP-U2-CND-001 to discriminate cause. "
            "Note: no chemistry anomalies noted at time of this writing — DO data not yet reviewed. "
            "Operator increased CW pump speed from 85 to 100 percent — minimal effect observed, "
            "backpressure reduction of only 0.04 inHg. CW temperature alone may be insufficient "
            "to explain the magnitude of observed backpressure rise."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "CR",
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "authority_level": "mandatory",
            "extraction_quality": 0.94,
        },
    },
    # ── WO-2024-11847: Tube cleaning WO — key contradicting evidence ──────────
    {
        "snippet_id": "WO-2024-11847::as_left_condition",
        "doc_id": "WO-2024-11847",
        "section": "as_left_condition",
        "snippet": (
            "As left acceptable. All 847 condenser tubes within normal limits. "
            "Zero tubes plugged. Tube sheet in acceptable condition. "
            "No evidence of fouling, biological growth, or debris accumulation observed. "
            "Heat transfer surface within acceptance criteria. Normal condition confirmed. "
            "No abnormality found during inspection. Tube outlet temperatures normal "
            "throughout the work scope. Failed to reproduce any indication of fouling."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "WO",
            "component_id": "U2-CND-TUBE-BUNDLE-A",
            "authority_level": "mandatory",
            "extraction_quality": 0.96,
        },
    },
    {
        "snippet_id": "WO-2024-11847::work_scope",
        "doc_id": "WO-2024-11847",
        "section": "work_scope",
        "snippet": (
            "Condenser tube cleaning and inspection per scheduled PM. Waterbox A opened, "
            "tubes hydroblasted and inspected. Tube outlet temperatures within normal limits "
            "before and after cleaning. As-found condition acceptable — no fouling noted. "
            "Eddy current testing of 10 percent tube sample showed no degradation. "
            "Work completed within scope, no additional corrective work required. "
            "Tube cleanliness score 0.94 out of 1.0 — within acceptance criteria."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "WO",
            "component_id": "U2-CND-WATERBOX-A",
            "authority_level": "mandatory",
            "extraction_quality": 0.95,
        },
    },
    # ── WO-2024-12001: Helium leak test — confirmation WO ─────────────────────
    {
        "snippet_id": "WO-2024-12001::findings",
        "doc_id": "WO-2024-12001",
        "section": "findings",
        "snippet": (
            "Helium leak test performed on all accessible condenser boundary connections. "
            "Helium detected at expansion joint weld, north face, adjacent to turbine exhaust flange. "
            "Active air in-leakage pathway confirmed. Estimated in-leakage rate 150 SCFM at operating vacuum. "
            "Weld damage consistent with thermal fatigue from elevated pit ambient temperature. "
            "Air in-leakage caused by degraded expansion joint seal due to thermal cycling. "
            "Tube bundles A and B tested — failed to reproduce any indication of tube leakage or fouling. "
            "Condenser tubes not the source of vacuum degradation."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "WO",
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "authority_level": "mandatory",
            "extraction_quality": 0.96,
        },
    },
    # ── SOP-U2-CND-001: Condenser monitoring SOP ─────────────────────────────
    {
        "snippet_id": "SOP-U2-CND-001::diagnosis_guidance",
        "doc_id": "SOP-U2-CND-001",
        "section": "backpressure_diagnosis_step_4_2",
        "snippet": (
            "Step 4.2 — Discriminate air in-leakage from tube fouling. "
            "Check hotwell dissolved oxygen sensor U2-AIT-0341. "
            "DO above 20 ppb is indicative of air in-leakage through condenser boundaries. "
            "Tube fouling does not cause dissolved oxygen elevation — DO within normal limits "
            "suggests fouling or thermal degradation rather than air in-leakage. "
            "If DO is elevated, initiate helium leak test per Attachment A before concluding fouling. "
            "Do not attribute backpressure rise to fouling if DO is elevated — these mechanisms "
            "are mutually exclusive with respect to DO response."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "SOP",
            "component_id": "U2-CND-HOTWELL",
            "authority_level": "mandatory",
            "extraction_quality": 0.98,
        },
    },
    {
        "snippet_id": "SOP-U2-CND-001::backpressure_limits",
        "doc_id": "SOP-U2-CND-001",
        "section": "operational_limits",
        "snippet": (
            "Condenser backpressure operational limits and required actions: "
            "High alarm setpoint 2.5 inHg — notify shift supervisor and initiate engineering evaluation. "
            "High-high setpoint 3.0 inHg — automatic turbine load runback actuates. "
            "Backpressure trending above 2.5 inHg requires engineering evaluation within 24 hours. "
            "Gradual drift above 2.0 inHg sustained more than 72 hours requires Condition Report. "
            "Annual expansion joint inspection required per §4.3 — overdue inspections shall be "
            "elevated to immediate corrective action when backpressure trends are observed."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "SOP",
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "authority_level": "mandatory",
            "extraction_quality": 0.98,
        },
    },
    # ── SOP-U2-CHE-041: Secondary chemistry SOP ──────────────────────────────
    {
        "snippet_id": "SOP-U2-CHE-041::do_acceptance_criteria",
        "doc_id": "SOP-U2-CHE-041",
        "section": "acceptance_criteria_table_2",
        "snippet": (
            "Hotwell dissolved oxygen acceptance criterion: less than 10 ppb "
            "during normal power operation above 50 percent rated power. "
            "Values above 20 ppb require immediate notification of chemistry supervisor "
            "and engineering evaluation for air in-leakage through condenser boundaries. "
            "Values above 100 ppb require prompt engineering and operations joint action. "
            "Elevated dissolved oxygen is caused by air infiltration through condenser "
            "boundary — this mechanism is not consistent with tube fouling or tube leakage. "
            "Do not attribute elevated DO to condensate chemistry variations."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "SOP",
            "component_id": "U2-CND-HOTWELL",
            "authority_level": "mandatory",
            "extraction_quality": 0.98,
        },
    },
    # ── ECA-2022-1103: Prior event — fouling confirmed (RECURRENCE TRAP DOC) ──
    # This document will be retrieved and could mislead if not interpreted carefully.
    # Key: the prior event had NORMAL DO (7.2 ppb) — distinguishing it from current.
    {
        "snippet_id": "ECA-2022-1103::root_cause_determination",
        "doc_id": "ECA-2022-1103",
        "section": "root_cause_determination",
        "snippet": (
            "Root cause of January 2023 condenser backpressure degradation confirmed as "
            "tube fouling due to biological growth in circulating water system. "
            "Hotwell dissolved oxygen was within normal limits during that event at 7.2 ppb — "
            "confirming tube fouling rather than air in-leakage as the mechanism. "
            "Corrective actions: biocide treatment, tube cleaning, enhanced CW monitoring. "
            "Analyst note: DO was normal in the 2023 event, which distinguishes it from any "
            "future event showing DO elevation. The presence of elevated DO in a current event "
            "changes the causal hypothesis from fouling to air in-leakage."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "ECA",
            "component_id": "U2-CND-TUBE-BUNDLE-A",
            "authority_level": "guidance",
            "extraction_quality": 0.95,
        },
    },
    # ── WO-2024-11901: HVAC bearing WO ───────────────────────────────────────
    {
        "snippet_id": "WO-2024-11901::maintenance_scope",
        "doc_id": "WO-2024-11901",
        "section": "maintenance_scope_and_findings",
        "snippet": (
            "HVAC turbine bay fan A motor bearing replacement — corrective maintenance. "
            "Fan motor bearing failed due to inadequate lubrication — PM overdue 60 days. "
            "Degraded HVAC performance resulted in elevated condenser pit ambient temperature. "
            "Pit ambient rose from 82 degF baseline to 97.6 degF peak over 10 days. "
            "Elevated ambient temperature accelerated thermal fatigue of expansion joint seal material. "
            "Damage to expansion joint weld caused by thermal cycling due to HVAC degradation. "
            "HVAC PM deferral identified as contributing maintenance factor."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "WO",
            "component_id": "U2-HVAC-TURBINE-BAY-FAN-A",
            "authority_level": "mandatory",
            "extraction_quality": 0.94,
        },
    },
    # ── OE-INPO-2023-CND-047: Industry OE ────────────────────────────────────
    {
        "snippet_id": "OE-INPO-2023-CND-047::mechanism_description",
        "doc_id": "OE-INPO-2023-CND-047",
        "section": "mechanism_description_and_lessons",
        "snippet": (
            "Fleet operating experience: condenser backpressure rise was initially attributed "
            "to circulating water temperature elevation but was confirmed as air in-leakage "
            "through expansion joint degradation. Circulating water temperature was a minor "
            "contributing factor, not the dominant cause. "
            "Elevated hotwell dissolved oxygen was the most reliable discriminator between "
            "thermal performance degradation and air in-leakage mechanisms. "
            "Air in-leakage causes dissolved oxygen elevation; tube fouling does not."
        ),
        "metadata": {
            "asset_id": "U2-CONDENSER-MAIN",
            "doc_type": "BULLETIN",
            "component_id": "U2-CND-EXPANSION-JOINT-EXHAUST",
            "authority_level": "guidance",
            "extraction_quality": 0.90,
        },
    },
]



class FixtureKGContextBuilder:
    """Protocol-compatible KG builder that returns the static fixture."""

    def build(
        self,
        event: Dict[str, Any],
        telemetry_summary: Dict[str, Any],
        operational_context: Optional[Dict[str, Any]],
        pm_compliance: Optional[Dict[str, Any]],
        run_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = copy.deepcopy(KG_CONTEXT)
        out["event_id"] = event["event_id"]
        out["asset_id"] = event["asset_id"]
        out["subgraph_id"] = f"KGCTX::{event['event_id']}::{event['asset_id']}"
        out["provenance"] = {
            "builder": "FixtureKGContextBuilder",
            "run_id": run_context.get("run_id"),
        }
        return out


def build_evidence_store() -> InMemoryEvidenceStore:
    store = InMemoryEvidenceStore()
    for row in EVIDENCE_STORE_ROWS:
        store.add(copy.deepcopy(row))
    return store


def build_orchestrator(output_dir: Path) -> RCAReasoningOrchestrator:
    return RCAReasoningOrchestrator(
        validator=NoOpSchemaValidator(),
        artifact_store=FileArtifactStore(output_dir),
        kg_context_builder=FixtureKGContextBuilder(),
        tskr_temporal_scorer=TSKRTemporalScorerV1(),
        causality_engine=RuleBasedCausalityEngineV32(
            config=CausalityEngineConfigV32(top_k_candidates=5)
        ),
        evidence_retriever=ChromaEvidenceRetriever(
            store=build_evidence_store(),
            config=EvidenceRetrieverConfig(
                top_k_total=12,
                top_k_per_query=6,
                score_threshold=0.0,
            ),
        ),
        rca_synthesizer=RuleValidatedRCASynthesizerV31(
            llm_client=DummyLLMClient(),
            config=RCASynthesizerConfig(
                max_candidates_in_prompt=5,
                max_evidence_in_prompt=12,
                minimum_primary_score=0.35,
            ),
        ),
        ishikawa_evaluator=HeuristicIshikawaEvaluatorV1(),
        config=OrchestratorConfig(
            enable_ishikawa=True,
            persist_intermediate_artifacts=True,
            stop_on_validation_error=False,
            run_label="fixture-condenser-vacuum-loss",
            top_k_candidates=5,
            top_k_evidence=12,
            extra={"causality_engine_version": "v32"},
        ),
    )


def score_gap(candidates: Dict[str, Any]) -> float:
    rows = candidates.get("candidates", []) or []
    if len(rows) < 2:
        return float(rows[0].get("composite_score", 0.0)) if rows else 0.0
    return float(rows[0].get("composite_score", 0.0)) - float(rows[1].get("composite_score", 0.0))


def find_candidate(candidates: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    for row in candidates.get("candidates", []) or []:
        if row.get("candidate_id") == candidate_id:
            return row
    for row in candidates.get("filtered_out_candidates", []) or []:
        if row.get("candidate_id") == candidate_id:
            return row
    return None


def find_evidence_summary(evidence_bundle: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    for row in evidence_bundle.get("candidate_evidence_summary", []) or []:
        if row.get("candidate_id") == candidate_id:
            return row
    return None


def flatten_ishikawa_rows(ishikawa_matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cat in ishikawa_matrix.get("categories", []) or []:
        rows.extend(cat.get("rows", []) or [])
    return rows


def dump_fixture_files(fixture_dir: Path) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "event.json": EVENT,
        "telemetry_summary.json": TELEMETRY_SUMMARY,
        "kg_context.json": KG_CONTEXT,
        "operational_context.json": OPERATIONAL_CONTEXT,
        "pm_compliance.json": PM_COMPLIANCE,
        "evidence_store_rows.json": EVIDENCE_STORE_ROWS,
    }
    for name, payload in fixtures.items():
        (fixture_dir / name).write_text(json.dumps(payload, indent=2))


def run_assertions(
    pre_candidates: Dict[str, Any],
    final_bundle: Dict[str, Any],
) -> None:
    post_candidates = final_bundle["causality_candidates"]
    evidence_bundle = final_bundle["evidence_bundle"]
    ishikawa_matrix = final_bundle["ishikawa_matrix"] or {}
    rca_card = final_bundle["rca_card"]

    primary = rca_card["primary_hypothesis"]
    alternatives = rca_card.get("alternatives", []) or []
    review_questions = " ".join(rca_card.get("analyst_review", {}).get("questions_to_resolve", [])).lower()

    assert primary.get("candidate_id") == "FM::FM-CND-AIR-INLEAK", primary
    assert any(a.get("candidate_id") == "FM::FM-CND-TUBE-FOUL" for a in alternatives), alternatives

    tube_leak = find_candidate(post_candidates, "FM::FM-CND-TUBE-LEAK")
    assert tube_leak is not None, "FM-CND-TUBE-LEAK missing"
    leak_temporal = (tube_leak.get("temporal_evidence") or {})
    assert (
        bool(leak_temporal.get("temporal_contradiction"))
        or tube_leak.get("filter_reason") is not None
    ), tube_leak

    foul_summary = find_evidence_summary(evidence_bundle, "FM::FM-CND-TUBE-FOUL")
    assert foul_summary is not None, "fouling evidence summary missing"
    assert foul_summary.get("contradicting_count", 0) >= 1, foul_summary

    assert score_gap(post_candidates) >= 0.05, post_candidates.get("candidates", [])

    air = find_candidate(post_candidates, "FM::FM-CND-AIR-INLEAK")
    foul = find_candidate(post_candidates, "FM::FM-CND-TUBE-FOUL")
    assert air is not None and foul is not None
    air_rec = float((air.get("recurrence") or {}).get("recurrence_score", 0.0))
    foul_rec = float((foul.get("recurrence") or {}).get("recurrence_score", 0.0))
    assert air_rec >= foul_rec, {"air": air_rec, "foul": foul_rec}

    cw_temp = find_candidate(post_candidates, "FM::FM-CW-TEMP-RISE")
    assert cw_temp is not None
    ranked_ids = [c.get("candidate_id") for c in post_candidates.get("candidates", []) or []]
    assert primary.get("candidate_id") != "FM::FM-CW-TEMP-RISE"
    if "FM::FM-CW-TEMP-RISE" in ranked_ids:
        assert ranked_ids.index("FM::FM-CW-TEMP-RISE") >= 1, ranked_ids

    ish_rows = flatten_ishikawa_rows(ishikawa_matrix)
    assert any("FM::FM-HVAC-DEGRAD" in (row.get("linked_candidate_ids") or []) for row in ish_rows), ish_rows

    assert any(tok in review_questions for tok in ["expansion joint", "inspection", "pm deferral"]), review_questions

    validation = rca_card.get("validation_status", {})
    assert validation.get("schema_valid") is True, validation
    assert validation.get("all_claims_cited") is True, validation

    pre_gap = score_gap(pre_candidates)
    post_gap = score_gap(post_candidates)
    print(f"[ok] score gap pre={pre_gap:.4f} post={post_gap:.4f}")


def main() -> int:
    root = Path.cwd()
    fixture_dir = root / "test_fixtures"
    output_dir = root / "test_output"
    dump_fixture_files(fixture_dir)

    orchestrator = build_orchestrator(output_dir)

    kg_context = copy.deepcopy(KG_CONTEXT)
    tskr_patterns = orchestrator.tskr_temporal_scorer.score(
        event=EVENT,
        telemetry_summary=TELEMETRY_SUMMARY,
        kg_context=kg_context,
        operational_context=OPERATIONAL_CONTEXT,
        run_context={"run_id": "fixture-precompute"},
    )
    pre_candidates = orchestrator.causality_engine.generate(
        event=EVENT,
        telemetry_summary=TELEMETRY_SUMMARY,
        kg_context=kg_context,
        tskr_patterns=tskr_patterns,
        operational_context=OPERATIONAL_CONTEXT,
        pm_compliance=PM_COMPLIANCE,
        run_context={"run_id": "fixture-precompute"},
    )
    evidence_bundle = orchestrator.evidence_retriever.retrieve(
        event=EVENT,
        kg_context=kg_context,
        causality_candidates=pre_candidates,
        operational_context=OPERATIONAL_CONTEXT,
        run_context={"run_id": "fixture-precompute"},
    )

    final_bundle = orchestrator.run(
        event=copy.deepcopy(EVENT),
        telemetry_summary=copy.deepcopy(TELEMETRY_SUMMARY),
        operational_context=copy.deepcopy(OPERATIONAL_CONTEXT),
        pm_compliance=copy.deepcopy(PM_COMPLIANCE),
        kg_context=copy.deepcopy(kg_context),
        tskr_patterns=copy.deepcopy(tskr_patterns),
        causality_candidates=copy.deepcopy(pre_candidates),
        evidence_bundle=copy.deepcopy(evidence_bundle),
    )

    (fixture_dir / "tskr_patterns.json").write_text(json.dumps(tskr_patterns, indent=2))
    (fixture_dir / "pre_refinement_candidates.json").write_text(json.dumps(pre_candidates, indent=2))
    (fixture_dir / "evidence_bundle.json").write_text(json.dumps(evidence_bundle, indent=2))

    run_id = final_bundle["run_context"]["run_id"]
    print(f"[info] run_id={run_id}")
    print(f"[info] output_dir={output_dir / run_id}")

    run_assertions(pre_candidates=pre_candidates, final_bundle=final_bundle)
    print("[ok] all scenario assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())