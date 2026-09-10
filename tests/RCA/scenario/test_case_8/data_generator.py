"""
TC-8 fixture generator — SW check valve systemic RCA.

Reproduces all fixtures in test_case_8/fixtures/ deterministically.
Run from the test_case_8/ directory:
    python data_generator.py
"""

import json
import pathlib

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


def write(name: str, data: dict) -> None:
    path = FIXTURES_DIR / name
    path.write_text(json.dumps(data, indent=2))
    print(f"  wrote {path.relative_to(pathlib.Path(__file__).parent)}")


def build_event() -> dict:
    return {
        "event_id": "EVT-U1B-2025-0312",
        "asset_id": "CHK-SW-HX-07A",
        "timestamp_start": "2025-03-12T08:14:00Z",
        "timestamp_end": "2025-03-12T08:44:00Z",
        "event_type": "DEGRADATION",
        "severity": "HIGH",
        "trigger_source": "surveillance_test",
        "symptom_signature": {
            "description": (
                "Service water check valve CHK-SW-HX-07A (model GWB-250-SS-316, "
                "EDG HX-07A Train A supply) found leaking through at 0.42 gpm during "
                "quarterly flow balance surveillance (WO-2025-SW-0147). Acceptance criterion "
                "is < 0.10 gpm. This is the third leakthrough failure of the same valve model "
                "across SW Train A and Train B in 18 months. No real-time alarm or process "
                "anomaly preceded the surveillance finding; degradation was not detectable by "
                "continuous monitoring."
            ),
            "anomaly_pattern": "sustained_exceedance",
            "symptom_types": ["leakage", "flow"],
            "affected_parameters": [
                {
                    "parameter": "sw_leakthrough_rate",
                    "sensor_id": "U1B-FT-2204",
                    "observed_value": 0.42,
                    "unit": "gpm",
                    "normal_range": {"min": 0.0, "max": 0.10},
                }
            ],
        },
        "related_cr_ids": ["CR-2023-11-847", "CR-2024-04-219"],
        "initial_assessment": (
            "Third leakthrough failure of valve model GWB-250-SS-316 in 18 months. "
            "Prior CAP entries closed at proximate cause only. Recurrence pattern suggests "
            "deeper causal contributors beyond normal wear."
        ),
        "rca_required": True,
    }


def build_operational_context() -> dict:
    return {
        "asset_id": "CHK-SW-HX-07A",
        "window": {
            "start": "2025-03-12T08:00:00Z",
            "end": "2025-03-12T09:00:00Z",
        },
        "mode": "steady",
        "percent_rated_power": 100,
        "train_configuration": {
            "train_id": "Train-A",
            "in_service": True,
        },
        "recent_operations": [
            {
                "timestamp": "2025-03-12T08:00:00Z",
                "action_type": "surveillance_test",
                "description": (
                    "Quarterly SW flow balance surveillance initiated per WO-2025-SW-0147. "
                    "Full-power steady-state operation. No power reduction or load change "
                    "during surveillance window."
                ),
                "system_affected": "Service Water Train A",
                "procedure_ref": "WO-2025-SW-0147",
            }
        ],
        "nearby_maintenance": [
            {
                "wo_id": "WO-2025-SW-0147",
                "maintenance_type": "surveillance",
                "component_id": "CHK-SW-HX-07A",
                "proximity": "same_component",
                "completed_at": "2025-03-12T08:44:00Z",
            }
        ],
    }


def build_telemetry_summary() -> dict:
    return {
        "event_id": "EVT-U1B-2025-0312",
        "asset_id": "CHK-SW-HX-07A",
        "generated_at": "2025-03-12T09:00:00Z",
        "window": {
            "start": "2025-03-12T06:00:00Z",
            "end": "2025-03-12T09:00:00Z",
        },
        "analysis_methods": ["threshold_check", "descriptive_stats"],
        "signals": [
            {
                "sensor_id": "U1B-PT-2201",
                "parameter": "sw_pump_discharge_pressure",
                "unit": "psig",
                "stats": {
                    "sample_count": 1080,
                    "mean": 82.4,
                    "std": 0.6,
                    "min": 81.1,
                    "max": 83.7,
                },
                "anomalies": [],
                "within_normal_limits": True,
            },
            {
                "sensor_id": "U1B-FT-2204",
                "parameter": "sw_flow_to_edg_hx07a",
                "unit": "gpm",
                "stats": {
                    "sample_count": 1080,
                    "mean": 312.7,
                    "std": 2.1,
                    "min": 307.4,
                    "max": 318.2,
                },
                "anomalies": [
                    {
                        "anomaly_id": "ANOM-SURV-001",
                        "detection_method": "other",
                        "pattern": "sustained_exceedance",
                        "timestamp_start": "2025-03-12T08:31:00Z",
                        "timestamp_end": "2025-03-12T08:44:00Z",
                        "severity_score": 0.60,
                        "peak_value": 0.42,
                        "promoted_to_kg_event": False,
                        "promotion_rationale": (
                            "Surveillance-detected leakthrough; recorded in CAP as "
                            "EVT-U1B-2025-0312. No continuous historian anomaly."
                        ),
                    }
                ],
                "within_normal_limits": False,
            },
        ],
        "overall_assessment": {
            "any_anomaly_detected": True,
            "anomaly_count": 1,
            "most_anomalous_signal": "U1B-FT-2204",
            "earliest_anomaly_time": "2025-03-12T08:31:00Z",
            "pattern_summary": (
                "SW pump discharge pressure nominal throughout window; no continuous "
                "historian anomaly. Single surveillance-detected leakthrough on flow path "
                "signal U1B-FT-2204 during controlled flow balance test step. Leakthrough "
                "rate 0.42 gpm exceeds acceptance criterion of 0.10 gpm. Telemetry score: "
                "0.60 for Category A (direct measurement of failure condition), 0.40 for "
                "programmatic/organizational categories."
            ),
        },
    }


def build_soe_log() -> dict:
    return {
        "soe_id": "SOE-EVT-U1B-2025-0312",
        "event_id": "EVT-U1B-2025-0312",
        "asset_id": "CHK-SW-HX-07A",
        "generated_at": "2025-03-12T09:15:00Z",
        "window": {
            "start": "2025-03-12T08:10:00Z",
            "end": "2025-03-12T08:50:00Z",
        },
        "source": {
            "recorder_id": "SOE-LOGGER-U1B-SW",
            "source_system": "U1B-OPS-LOG",
            "timezone": "UTC",
        },
        "soe_context": "sparse_surveillance_sequence",
        "records": [
            {
                "record_id": "SOE-001",
                "sequence_index": 1,
                "timestamp": "2025-03-12T08:14:00Z",
                "signal_id": "WO-2025-SW-0147",
                "signal_label": "SW Quarterly Flow Balance Surveillance",
                "component_id": "SW-TRAIN-A",
                "transition": "state_change",
                "old_state": "NOT_STARTED",
                "new_state": "IN_PROGRESS",
                "priority": "informational",
                "is_protection_signal": False,
                "protection_logic_ref": None,
                "remarks": "Quarterly SW flow balance surveillance initiated per WO-2025-SW-0147.",
            },
            {
                "record_id": "SOE-002",
                "sequence_index": 2,
                "timestamp": "2025-03-12T08:31:00Z",
                "signal_id": "U1B-FT-2204",
                "signal_label": "SW Flow to EDG HX-07A — Isolation Test Step",
                "component_id": "CHK-SW-HX-07A",
                "transition": "assert",
                "old_state": "ISOLATED",
                "new_state": "LEAKTHROUGH_DETECTED",
                "priority": "high",
                "is_protection_signal": False,
                "protection_logic_ref": None,
                "remarks": "Leakthrough detected on CHK-SW-HX-07A. Measured: 0.42 gpm vs. criterion < 0.10 gpm.",
            },
            {
                "record_id": "SOE-003",
                "sequence_index": 3,
                "timestamp": "2025-03-12T08:36:00Z",
                "signal_id": "OPERATOR-ACTION",
                "signal_label": "Operations Supervisor Notified",
                "component_id": None,
                "transition": "state_change",
                "old_state": None,
                "new_state": "LOGGED",
                "priority": "informational",
                "is_protection_signal": False,
                "protection_logic_ref": None,
                "remarks": "Operations supervisor notified. Third leakthrough of GWB-250-SS-316 in 18 months.",
            },
            {
                "record_id": "SOE-004",
                "sequence_index": 4,
                "timestamp": "2025-03-12T08:44:00Z",
                "signal_id": "CAP-ENTRY",
                "signal_label": "Corrective Action Program Entry",
                "component_id": "CHK-SW-HX-07A",
                "transition": "state_change",
                "old_state": None,
                "new_state": "CAP_INITIATED",
                "priority": "high",
                "is_protection_signal": False,
                "protection_logic_ref": None,
                "remarks": "CAP initiated. Event ID EVT-U1B-2025-0312 assigned. Significance: SIGNIFICANT.",
            },
        ],
        "quality": {
            "clock_sync_ok": True,
            "dropped_record_count": 0,
            "duplicate_record_count": 0,
            "quality_flags": ["sparse_surveillance_log"],
        },
        "provenance": {
            "generated_by": "tc8-fixture-generator",
            "query_params": {"test_case": "TC-8"},
        },
    }


def build_kg_context() -> dict:
    return {
        "event_id": "EVT-U1B-2025-0312",
        "asset_id": "CHK-SW-HX-07A",
        "subgraph_id": "KGCTX::EVT-U1B-2025-0312::CHK-SW-HX-07A",
        "generated_at": "2025-03-12T09:00:00Z",
        "hop_limit": 3,
        "components": [
            {"component_id": "CHK-SW-HX-07A", "seed_match_type": "seed", "model": "GWB-250-SS-316", "train": "A"},
            {"component_id": "SW-PUMP-1A-01", "seed_match_type": "topology", "train": "A"},
            {"component_id": "SW-PUMP-1A-02", "seed_match_type": "topology", "train": "A"},
            {"component_id": "SW-STRAINER-1A", "seed_match_type": "topology", "train": "A"},
            {"component_id": "SW-HX-EDG-07A", "seed_match_type": "topology", "train": "A"},
            {"component_id": "EDG-07A", "seed_match_type": "safety_function_beneficiary", "train": "A"},
            {
                "component_id": "CHK-SW-HX-07B",
                "seed_match_type": "cross_train_ccf_candidate",
                "model": "GWB-250-SS-316",
                "train": "B",
                "ccf_note": "Same valve model. Prior failure CR-2024-04-219 on Train B.",
            },
        ],
        "upstream_paths": [
            {
                "path_id": "PATH-001",
                "nodes": ["SW-PUMP-1A-01", "SW-STRAINER-1A", "CHK-SW-HX-07A"],
                "edges": [
                    {"from_node": "SW-PUMP-1A-01", "to_node": "SW-STRAINER-1A", "edge_type": "process_flow"},
                    {"from_node": "SW-STRAINER-1A", "to_node": "CHK-SW-HX-07A", "edge_type": "process_flow"},
                ],
            },
            {
                "path_id": "PATH-002",
                "nodes": ["CHK-SW-HX-07A", "SW-HX-EDG-07A"],
                "edges": [{"from_node": "CHK-SW-HX-07A", "to_node": "SW-HX-EDG-07A", "edge_type": "provides_cooling"}],
            },
            {
                "path_id": "PATH-003",
                "nodes": ["SW-HX-EDG-07A", "EDG-07A"],
                "edges": [{"from_node": "SW-HX-EDG-07A", "to_node": "EDG-07A", "edge_type": "supports_safety_function"}],
            },
        ],
        "failure_modes": [
            {"fm_id": "FM-CHK-SEAT-EROSION", "component_id": "CHK-SW-HX-07A", "name": "Check valve poppet seat erosion — high-cycle wear", "causal_category": "A", "causal_category_source": "curated", "causal_depth": "proximate"},
            {"fm_id": "FM-CHK-DISC-DAMAGE", "component_id": "CHK-SW-HX-07A", "name": "Check valve disc damage — impact from high-frequency slam", "causal_category": "A", "causal_category_source": "inferred", "causal_depth": "proximate"},
            {"fm_id": "FM-PM-FREQ-NONCONF", "component_id": "CHK-SW-HX-07A", "name": "Inspection/testing program inadequacy — PM frequency nonconformance with vendor specification", "causal_category": "J", "causal_category_source": "governance_derived", "causal_depth": "contributing"},
            {"fm_id": "FM-VENDOR-BATCH-TRACEABILITY", "component_id": "CHK-SW-HX-07A", "name": "Vendor supply chain — lot GWB-2020-L07 in affected range with dimensional nonconformance", "causal_category": "K", "causal_category_source": "evidence_derived", "causal_depth": "contributing"},
            {"fm_id": "FM-PM-CONFIG-CONTROL-GAP", "component_id": "CHK-SW-HX-07A", "name": "Configuration/change control gap — 2021 PM revision without engineering evaluation of vendor spec deviation", "causal_category": "I", "causal_category_source": "evidence_derived", "causal_depth": "contributing"},
            {"fm_id": "FM-OE-SCREENING-MISS", "component_id": "CHK-SW-HX-07A", "name": "Systemic organizational weakness — IRIS-OE-2023-SW-0047 classified non-applicable, OE not incorporated", "causal_category": "L", "causal_category_source": "evidence_derived", "causal_depth": "root_cause"},
        ],
        "past_events": [
            {"event_id": "CR-2023-11-847", "asset_id": "CHK-SW-HX-07B", "component_id": "CHK-SW-HX-07B", "timestamp_start": "2023-11-12T00:00:00Z", "failure_mode_ref": "FM-CHK-SEAT-EROSION", "lag_hours_prior_to_current": 11520},
            {"event_id": "CR-2024-04-219", "asset_id": "CHK-SW-HX-07A", "component_id": "CHK-SW-HX-07A", "timestamp_start": "2024-04-07T00:00:00Z", "failure_mode_ref": "FM-CHK-SEAT-EROSION", "lag_hours_prior_to_current": 7920},
        ],
        "maintenance_tasks": [
            {
                "task_id": "PM-SW-CHK-ANNUAL",
                "component_id": "CHK-SW-HX-07A",
                "task_name": "Service Water Check Valve Annual Inspection (Lapping and Seat Check)",
                "plant_interval_months": 18,
                "vendor_spec_interval_months": 12,
                "vendor_spec_ref": "VND-SPEC-GWB-250",
                "interval_nonconformance": True,
                "interval_deviation_pct": 50.0,
                "pm_revision_ref": "WO-2021-PMREV-04",
                "last_performed": "2023-09-15T00:00:00Z",
                "next_due_plant_schedule": "2025-03-15T00:00:00Z",
            }
        ],
        "documents": [
            {"doc_id": "CR-2023-11-847", "doc_type": "CR"},
            {"doc_id": "CR-2024-04-219", "doc_type": "CR"},
            {"doc_id": "WO-2021-PMREV-04", "doc_type": "WO"},
            {"doc_id": "IRIS-OE-2023-SW-0047", "doc_type": "OE_REPORT"},
            {"doc_id": "TD-REPORT-2025-0312", "doc_type": "TEARDOWN_REPORT"},
            {"doc_id": "OE-SCREEN-LOG-2023", "doc_type": "OE_SCREENING_LOG"},
            {"doc_id": "VND-SPEC-GWB-250", "doc_type": "VENDOR_SPEC"},
        ],
        "seed_context": {
            "asset_ids": ["CHK-SW-HX-07A"],
            "seed_component_ids": ["CHK-SW-HX-07A"],
            "monitored_variables": ["sw_pump_discharge_pressure", "sw_flow_to_edg_hx07a"],
            "valve_model": "GWB-250-SS-316",
            "cycle_rate_per_year": 1000,
        },
    }


def build_pm_compliance() -> dict:
    return {
        "asset_id": "CHK-SW-HX-07A",
        "event_id": "EVT-U1B-2025-0312",
        "assessment_date": "2025-03-12T09:00:00Z",
        "look_back_window_days": 548,
        "fmea_pm_linkage_available": True,
        "window": {"start": "2023-09-01T00:00:00Z", "end": "2025-03-12T09:00:00Z"},
        "checks": [
            {
                "check_id": "PM-CHK-SW-INTERVAL-NONCONFORMANCE",
                "check_type": "inspection",
                "status": "fail",
                "component_id": "CHK-SW-HX-07A",
                "source_ref": "VND-SPEC-GWB-250",
                "wo_id": "WO-2021-PMREV-04",
                "overdue_by_days": 178,
                "applicable_fm_ids": ["FM-PM-FREQ-NONCONF", "FM-PM-CONFIG-CONTROL-GAP"],
                "details": (
                    "PM task PM-SW-CHK-ANNUAL (check valve lapping and seat inspection) is set "
                    "to an 18-month interval per plant PM program revision WO-2021-PMREV-04. "
                    "Vendor specification VND-SPEC-GWB-250 (Section 6.4) requires a 12-month "
                    "inspection interval for continuous high-cycle service exceeding 500 cycles "
                    "per year. Plant interval (18 months) exceeds vendor specification (12 months) "
                    "by 50%. Under the vendor-required schedule, the last inspection (September 2023) "
                    "should have been followed by another inspection in September 2024, which was not "
                    "performed. As of event date (2025-03-12), the vendor-spec-based inspection is "
                    "178 days overdue. The 2021 PM program revision (WO-2021-PMREV-04) was processed "
                    "as a minor procedure revision without engineering evaluation of the vendor "
                    "specification impact and without 50.59 screening."
                ),
            },
            {
                "check_id": "PM-CHK-SW-LAST-EXECUTION",
                "check_type": "scheduled_pm",
                "status": "pass",
                "component_id": "CHK-SW-HX-07A",
                "source_ref": "WO-2021-PMREV-04",
                "scheduled_date": "2023-09-15T00:00:00Z",
                "completed_date": "2023-09-15T00:00:00Z",
                "overdue_by_days": 0,
                "details": (
                    "Last PM execution (September 2023) was completed on schedule per the plant "
                    "18-month interval. Inspection as-found: seat condition acceptable at time of "
                    "inspection. As-left: acceptable. This pass reflects compliance with the plant "
                    "nonconforming schedule, not compliance with the vendor specification requirement "
                    "of 12-month maximum interval."
                ),
            },
            {
                "check_id": "PM-CHK-SW-07B-CROSS-TRAIN",
                "check_type": "inspection",
                "status": "fail",
                "component_id": "CHK-SW-HX-07B",
                "source_ref": "VND-SPEC-GWB-250",
                "overdue_by_days": 178,
                "applicable_fm_ids": ["FM-PM-FREQ-NONCONF"],
                "details": (
                    "Same PM frequency nonconformance applies to Train B cross-train valve "
                    "CHK-SW-HX-07B (model GWB-250-SS-316). The 18-month plant interval versus "
                    "12-month vendor specification applies fleet-wide for this valve model at this "
                    "plant. Both train A and train B valves are subject to the same governance "
                    "nonconformance."
                ),
            },
        ],
        "overdue_items": [
            {
                "check_id": "PM-CHK-SW-INTERVAL-NONCONFORMANCE",
                "check_type": "inspection",
                "scheduled_date": "2024-09-15T00:00:00Z",
                "overdue_by_days": 178,
                "source_ref": "VND-SPEC-GWB-250",
            }
        ],
        "data_quality_notes": [
            "KG FMEA/PM task linkage available (fmea_pm_linkage_available=true). PM task node PM-SW-CHK-ANNUAL carries both plant_interval_months=18 and vendor_spec_interval_months=12. Governance scorer detects the interval nonconformance and generates FM-PM-FREQ-NONCONF (Category J) candidate.",
            "If vendor_spec_interval_months were absent from the KG PM task node, the Category J candidate would not be generated. This is the KG coverage bound demonstrated in this test case.",
        ],
        "summary": {
            "total_checks": 3,
            "passed": 1,
            "failed": 2,
            "unknown": 0,
            "overdue_count": 1,
            "compliance_rate": 0.333,
            "overall_compliance": "partial",
            "maintenance_induced_risk": "high",
            "has_scope_gaps_for_primary_fm": True,
            "data_quality_confidence": "high",
        },
    }


def build_tskr_patterns() -> dict:
    base = {
        "event_id": "EVT-U1B-2025-0312",
        "asset_id": "CHK-SW-HX-07A",
        "prior_event_ids": ["CR-2023-11-847", "CR-2024-04-219"],
        "mean_lag_hours": 9720.0,
        "std_lag_hours": 1800.0,
        "anomaly_count": 1,
        "operator_family": "episode_recurrence",
        "episode_count": 3,
        "source": "tc8_episode_recurrence_fixture",
    }
    patterns = [
        {**base, "pattern_id": "TSKR::FM-CHK-SEAT-EROSION", "target_id": "FM-CHK-SEAT-EROSION", "component_id": "CHK-SW-HX-07A", "relation": "follows", "support": 0.82, "confidence": 0.85, "matching_signal_ids": ["U1B-FT-2204", "CR-2023-11-847", "CR-2024-04-219"], "lag_consistency": 0.78, "mean_recurrence_interval_hours": 4860.0},
        {**base, "pattern_id": "TSKR::FM-PM-FREQ-NONCONF", "target_id": "FM-PM-FREQ-NONCONF", "component_id": "CHK-SW-HX-07A", "relation": "follows", "support": 0.65, "confidence": 0.60, "matching_signal_ids": ["U1B-FT-2204", "CR-2023-11-847", "CR-2024-04-219"], "lag_consistency": 0.65, "mean_recurrence_interval_hours": 4860.0},
        {**base, "pattern_id": "TSKR::FM-VENDOR-BATCH-TRACEABILITY", "target_id": "FM-VENDOR-BATCH-TRACEABILITY", "component_id": "CHK-SW-HX-07A", "relation": "follows", "support": 0.68, "confidence": 0.65, "matching_signal_ids": ["U1B-FT-2204", "CR-2023-11-847", "CR-2024-04-219"], "lag_consistency": 0.70, "mean_recurrence_interval_hours": 4860.0},
        {**base, "pattern_id": "TSKR::FM-PM-CONFIG-CONTROL-GAP", "target_id": "FM-PM-CONFIG-CONTROL-GAP", "component_id": "CHK-SW-HX-07A", "relation": "precedes", "support": 0.58, "confidence": 0.70, "matching_signal_ids": ["WO-2021-PMREV-04", "CR-2023-11-847", "CR-2024-04-219"], "lag_consistency": 0.75, "mean_lag_hours": 33120.0, "prior_event_ids": ["WO-2021-PMREV-04"], "episode_count": 2, "mean_recurrence_interval_hours": None},
        {**base, "pattern_id": "TSKR::FM-OE-SCREENING-MISS", "target_id": "FM-OE-SCREENING-MISS", "component_id": "CHK-SW-HX-07A", "relation": "precedes", "support": 0.42, "confidence": 0.40, "matching_signal_ids": ["OE-SCREEN-LOG-2023", "CR-2024-04-219"], "lag_consistency": 0.60, "mean_lag_hours": 12960.0, "prior_event_ids": ["OE-SCREEN-LOG-2023"], "episode_count": 1, "mean_recurrence_interval_hours": None},
    ]
    return {
        "event_id": "EVT-U1B-2025-0312",
        "asset_id": "CHK-SW-HX-07A",
        "tskr_mode": "episode_recurrence",
        "patterns": patterns,
        "summary": {
            "has_temporal_support": True,
            "mode": "tc8_episode_recurrence",
            "n_patterns": 5,
            "n_supported_patterns": 5,
            "operator_family": "episode_recurrence",
            "anomaly_point_count": 1,
            "signal_count": 2,
            "avg_confidence": 0.64,
            "top_supported_targets": ["FM-CHK-SEAT-EROSION", "FM-VENDOR-BATCH-TRACEABILITY", "FM-PM-CONFIG-CONTROL-GAP"],
        },
        "provenance": {"generated_by": "tc8-fixture-generator", "generated_at": "2025-03-12T09:00:00Z"},
    }


def build_evidence_bundle() -> dict:
    return {
        "event_id": "EVT-U1B-2025-0312",
        "bundle_id": "BUNDLE-EVT-U1B-2025-0312",
        "generated_at": "2025-03-12T09:05:00Z",
        "query": "check valve leakthrough seat erosion service water PM frequency vendor batch GWB-250-SS-316 OE screening",
        "score_metric": "hybrid_fixture",
        "score_threshold": 0.0,
        "retrieval_scope": {
            "event_id": "EVT-U1B-2025-0312",
            "asset_id": "CHK-SW-HX-07A",
            "doc_ids": [
                "CR-2023-11-847",
                "CR-2024-04-219",
                "WO-2021-PMREV-04",
                "IRIS-OE-2023-SW-0047",
                "TD-REPORT-2025-0312",
                "OE-SCREEN-LOG-2023",
                "VND-SPEC-GWB-250",
            ],
            "component_ids": ["CHK-SW-HX-07A", "CHK-SW-HX-07B"],
        },
        "results": [
            {
                "snippet_id": "SNIP-001",
                "doc_id": "CR-2023-11-847",
                "score": 0.91,
                "snippet": (
                    "SW Train B check valve CHK-SW-HX-07B (model GWB-250-SS-316) found leaking "
                    "through during quarterly flow balance surveillance. Measured leakthrough rate "
                    "0.55 gpm. Teardown inspection revealed poppet seat erosion consistent with "
                    "high-cycle service. Valve replaced with like-for-like. CAP closed at proximate "
                    "cause — valve wear, normal service life."
                ),
                "metadata": {
                    "doc_type": "CR",
                    "component_id": "CHK-SW-HX-07B",
                    "authority_level": "plant",
                    "authority_weight": 1.0,
                    "linked_candidate_id": "FM::FM-CHK-SEAT-EROSION",
                    "support_role": "supporting",
                    "extraction_quality": 0.92,
                },
            },
            {
                "snippet_id": "SNIP-002",
                "doc_id": "CR-2024-04-219",
                "score": 0.93,
                "snippet": (
                    "SW Train A check valve CHK-SW-HX-07A (model GWB-250-SS-316) found leaking "
                    "through during routine surveillance. Second occurrence of this failure mode on "
                    "this valve model in six months. Poppet seat erosion confirmed. Valve replaced. "
                    "Shift supervisor noted pattern but no formal recurrence evaluation initiated. "
                    "CAP closed at proximate cause."
                ),
                "metadata": {
                    "doc_type": "CR",
                    "component_id": "CHK-SW-HX-07A",
                    "authority_level": "plant",
                    "authority_weight": 1.0,
                    "linked_candidate_id": "FM::FM-CHK-SEAT-EROSION",
                    "support_role": "supporting",
                    "extraction_quality": 0.94,
                },
            },
            {
                "snippet_id": "SNIP-003",
                "doc_id": "WO-2021-PMREV-04",
                "score": 0.88,
                "snippet": (
                    "PM program revision — SW system check valve inspection task interval changed "
                    "from 12 months to 18 months as part of 2021 PM program restructuring. Basis "
                    "for change: resource optimization and alignment with outage scheduling. Vendor "
                    "manual PM-GWB-250-REV-4 not reviewed during revision. Change processed as minor "
                    "procedure revision (no engineering evaluation required per applicable threshold "
                    "criteria). No 50.59 screening performed."
                ),
                "metadata": {
                    "doc_type": "WO",
                    "component_id": "CHK-SW-HX-07A",
                    "authority_level": "plant",
                    "authority_weight": 1.0,
                    "linked_candidate_id": "FM::FM-PM-FREQ-NONCONF",
                    "support_role": "supporting",
                    "extraction_quality": 0.90,
                },
            },
            {
                "snippet_id": "SNIP-004",
                "doc_id": "IRIS-OE-2023-SW-0047",
                "score": 0.87,
                "snippet": (
                    "INPO IRIS Operating Experience Report: Four failures of valve model "
                    "GWB-250-SS-316 identified at three separate PWR units. Root cause attributed "
                    "to production lot range GWB-2020-L05 through GWB-2020-L09 — undersized seat "
                    "inserts with dimensional nonconformance. Nonconformance manifests as fatigue "
                    "seat failure after 12-24 months of continuous cycling service, not at initial "
                    "acceptance inspection. Units with high-cycle service applications (>500 "
                    "cycles/year) are at elevated risk. Recommendation: verify lot numbers for "
                    "installed GWB-250-SS-316 valves."
                ),
                "metadata": {
                    "doc_type": "OE",
                    "component_id": "CHK-SW-HX-07A",
                    "authority_level": "fleet",
                    "authority_weight": 0.85,
                    "linked_candidate_id": "FM::FM-VENDOR-BATCH-TRACEABILITY",
                    "support_role": "supporting",
                    "extraction_quality": 0.88,
                },
            },
            {
                "snippet_id": "SNIP-005",
                "doc_id": "TD-REPORT-2025-0312",
                "score": 0.82,
                "snippet": (
                    "Teardown inspection of valve CHK-SW-HX-07A (removed 2025-03-12). Lot number: "
                    "GWB-2020-L07. Seat insert OD: 2.247 in — within acceptance criteria "
                    "2.240-2.260 in. Finding: poppet seat shows wear pattern consistent with "
                    "high-frequency cycling; no dimensional nonconformance observed at time of "
                    "inspection. Conclusion: seat erosion is wear-related, not due to initial "
                    "dimensional defect."
                ),
                "metadata": {
                    "doc_type": "TEARDOWN_REPORT",
                    "component_id": "CHK-SW-HX-07A",
                    "authority_level": "plant",
                    "authority_weight": 1.0,
                    "linked_candidate_id": "FM::FM-VENDOR-BATCH-TRACEABILITY",
                    "support_role": "contradicting",
                    "evidence_role": "contradicting",
                    "extraction_quality": 0.95,
                    "hedge_fraction": 0.10,
                    "contradiction_resolution_note": (
                        "NER lot-number cross-reference: lot GWB-2020-L07 falls within affected "
                        "range GWB-2020-L05 through L09 per IRIS-OE-2023-SW-0047. IRIS-OE specifies "
                        "that the dimensional nonconformance manifests as fatigue failure after 12-24 "
                        "months of cycling, not at initial inspection. The 'contradicting' role is "
                        "retained per the confirmed three-role taxonomy; contradiction weight is "
                        "modulated (not zeroed) by the CompatibilityEngine based on the lot-number "
                        "cross-reference."
                    ),
                },
            },
            {
                "snippet_id": "SNIP-006",
                "doc_id": "OE-SCREEN-LOG-2023",
                "score": 0.79,
                "snippet": (
                    "OE Screening Decision — IRIS-OE-2023-SW-0047: Classified NON-APPLICABLE. "
                    "Screener rationale: IRIS OE associated with RCS boundary service valves "
                    "experiencing cavitation-induced seat degradation; SW system check valves "
                    "operate at lower pressure differential and are not subject to the same "
                    "cavitation mechanism. GWB-250-SS-316 valve model not specifically identified "
                    "in SW system context by screener. No corrective action initiated."
                ),
                "metadata": {
                    "doc_type": "OE_SCREENING_LOG",
                    "component_id": "CHK-SW-HX-07A",
                    "authority_level": "plant",
                    "authority_weight": 1.0,
                    "linked_candidate_id": "FM::FM-OE-SCREENING-MISS",
                    "support_role": "supporting",
                    "extraction_quality": 0.86,
                    "hedge_fraction": 0.38,
                },
            },
            {
                "snippet_id": "SNIP-007",
                "doc_id": "VND-SPEC-GWB-250",
                "score": 0.84,
                "snippet": (
                    "Grinnell Technical Specification GWB-250-SS-316 Rev. 3. Section 6.4 — "
                    "Maintenance Requirements: For continuous cycling service applications exceeding "
                    "500 cycles per year, the following PM tasks are required: (1) annual lapping "
                    "inspection and seat reconditioning (12-month maximum interval); (2) disc and "
                    "hinge pin inspection at 24-month interval. Failure to maintain the 12-month "
                    "inspection interval in high-cycle service will accelerate seat wear progression "
                    "and may result in leakthrough failure prior to the next scheduled inspection."
                ),
                "metadata": {
                    "doc_type": "SPEC",
                    "component_id": "CHK-SW-HX-07A",
                    "authority_level": "vendor",
                    "authority_weight": 0.75,
                    "linked_candidate_id": "FM::FM-PM-FREQ-NONCONF",
                    "support_role": "supporting",
                    "extraction_quality": 0.91,
                },
            },
        ],
        "candidate_evidence_summary": [
            {
                "candidate_id": "FM::FM-CHK-SEAT-EROSION",
                "hit_count": 3,
                "best_support_score": 0.92,
                "best_contradiction_score": 0.0,
                "best_context_score": 0.60,
                "has_affects_class_hit": True,
                "has_analyzes_class_hit": True,
                "mean_conjecture_fraction": 0.05,
                "best_source_tier": "plant_instance",
                "supporting_snippet_ids": ["SNIP-001", "SNIP-002", "SNIP-005"],
                "contradicting_snippet_ids": [],
                "contextual_snippet_ids": ["SNIP-004"],
            },
            {
                "candidate_id": "FM::FM-PM-FREQ-NONCONF",
                "hit_count": 2,
                "best_support_score": 0.98,
                "best_contradiction_score": 0.0,
                "best_context_score": 0.92,
                "has_affects_class_hit": True,
                "has_analyzes_class_hit": True,
                "mean_conjecture_fraction": 0.0,
                "best_source_tier": "plant_instance",
                "supporting_snippet_ids": ["SNIP-003", "SNIP-007"],
                "contradicting_snippet_ids": [],
                "contextual_snippet_ids": [],
            },
            {
                "candidate_id": "FM::FM-VENDOR-BATCH-TRACEABILITY",
                "hit_count": 2,
                "best_support_score": 0.80,
                "best_contradiction_score": 0.20,
                "best_context_score": 0.50,
                "has_affects_class_hit": True,
                "has_analyzes_class_hit": True,
                "mean_conjecture_fraction": 0.12,
                "best_source_tier": "fleet",
                "supporting_snippet_ids": ["SNIP-004"],
                "contradicting_snippet_ids": ["SNIP-005"],
                "contextual_snippet_ids": ["SNIP-001"],
            },
            {
                "candidate_id": "FM::FM-PM-CONFIG-CONTROL-GAP",
                "hit_count": 2,
                "best_support_score": 0.80,
                "best_contradiction_score": 0.0,
                "best_context_score": 0.65,
                "has_affects_class_hit": True,
                "has_analyzes_class_hit": True,
                "mean_conjecture_fraction": 0.08,
                "best_source_tier": "plant_instance",
                "supporting_snippet_ids": ["SNIP-003", "SNIP-007"],
                "contradicting_snippet_ids": [],
                "contextual_snippet_ids": [],
            },
            {
                "candidate_id": "FM::FM-OE-SCREENING-MISS",
                "hit_count": 2,
                "best_support_score": 0.78,
                "best_contradiction_score": 0.0,
                "best_context_score": 0.55,
                "has_affects_class_hit": True,
                "has_analyzes_class_hit": True,
                "mean_conjecture_fraction": 0.38,
                "best_source_tier": "plant_instance",
                "supporting_snippet_ids": ["SNIP-006", "SNIP-004"],
                "contradicting_snippet_ids": [],
                "contextual_snippet_ids": [],
            },
            {
                "candidate_id": "FM::FM-CHK-DISC-DAMAGE",
                "hit_count": 1,
                "best_support_score": 0.45,
                "best_contradiction_score": 0.0,
                "best_context_score": 0.30,
                "has_affects_class_hit": True,
                "has_analyzes_class_hit": False,
                "mean_conjecture_fraction": 0.05,
                "best_source_tier": "plant_instance",
                "supporting_snippet_ids": ["SNIP-005"],
                "contradicting_snippet_ids": [],
                "contextual_snippet_ids": [],
            },
        ],
        "provenance": {
            "retriever": "tc8-fixture-generator",
            "run_id": None,
            "query_count": 7,
        },
    }


if __name__ == "__main__":
    print("Generating TC-8 fixtures...")
    write("event.json", build_event())
    write("operational_context.json", build_operational_context())
    write("telemetry_summary.json", build_telemetry_summary())
    write("soe_log.json", build_soe_log())
    write("kg_context.json", build_kg_context())
    write("pm_compliance.json", build_pm_compliance())
    write("tskr_patterns.json", build_tskr_patterns())
    write("evidence_bundle.json", build_evidence_bundle())
    print("Done.")
