"""
test_step2b_temporal_search.py — Step 2b Temporal Search hardening tests

Covers:
- Two-tier window tagging (in_precursor_window, window_tier)
- Per-component indexing and top-N cap
- temporal_search_summary content and counts
- Source breakdown (kg / cmms_cr / cmms_wo)
- Manifest pipeline_config.temporal_search populated
- Edge cases: no past events, missing timestamps, CMMS-only events

Run:  pytest test_step2b_temporal_search.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator, OrchestratorConfig


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _orchestrator(precursor_window_days=180, per_component_top_n=5):
    return RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        config=OrchestratorConfig(extra={
            "precursor_window_days": precursor_window_days,
            "per_component_past_event_top_n": per_component_top_n,
        }),
    )


def _kg_with_past_events(past_events):
    return {
        "subgraph_id": "KG::S2B",
        "event_id": "EVT-1",
        "asset_id": "ASSET-1",
        "components": [{"component_id": "C1"}, {"component_id": "C2"}],
        "failure_modes": [{"fm_id": "FM-1"}],
        "past_events": past_events,
        "seed_context": {},
    }


def _pe(event_id, component_id, days_before, priority_score=5.0, source="kg"):
    return {
        "event_id": event_id,
        "asset_id": "ASSET-1",
        "component_id": component_id,
        "timestamp_start": "2025-01-01T00:00:00Z",
        "days_before_current_event": days_before,
        "time_distance_days": days_before,
        "priority_score": priority_score,
        "resolved": True,
        "fm_id": None,
        "matched_asset_ids": [],
        "matched_component_ids": [component_id] if component_id else [],
        "matched_failure_mode_ids": [],
    }


def _cmms_pe(record_type, rid, component_id, days_before):
    pe = _pe(
        event_id=f"CMMS::{record_type.upper()}::{rid}",
        component_id=component_id,
        days_before=days_before,
    )
    pe["event_type"] = f"cmms_{record_type}"
    return pe


# ─────────────────────────────────────────────────────────────────────────────
# 1. Window tier tagging
# ─────────────────────────────────────────────────────────────────────────────

def test_event_within_precursor_window_tagged_primary():
    o = _orchestrator(precursor_window_days=180)
    kg = _kg_with_past_events([_pe("E1", "C1", days_before=90)])
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={"asset_id": "ASSET-1"})
    pe = result["past_events"][0]
    assert pe["in_precursor_window"] is True
    assert pe["window_tier"] == "primary"


def test_event_at_window_boundary_tagged_primary():
    o = _orchestrator(precursor_window_days=180)
    kg = _kg_with_past_events([_pe("E1", "C1", days_before=180)])
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    assert result["past_events"][0]["window_tier"] == "primary"
    assert result["past_events"][0]["in_precursor_window"] is True


def test_event_in_extended_window_tagged_extended():
    o = _orchestrator(precursor_window_days=180)
    kg = _kg_with_past_events([_pe("E1", "C1", days_before=270)])  # 180 < 270 <= 360
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    pe = result["past_events"][0]
    assert pe["in_precursor_window"] is False
    assert pe["window_tier"] == "extended"


def test_event_beyond_double_window_tagged_historical():
    o = _orchestrator(precursor_window_days=180)
    kg = _kg_with_past_events([_pe("E1", "C1", days_before=400)])  # > 360
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    pe = result["past_events"][0]
    assert pe["in_precursor_window"] is False
    assert pe["window_tier"] == "historical"


def test_event_with_no_days_before_tagged_unknown():
    o = _orchestrator()
    pe_data = _pe("E1", "C1", days_before=None)
    pe_data["days_before_current_event"] = None
    pe_data["time_distance_days"] = None
    kg = _kg_with_past_events([pe_data])
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    pe = result["past_events"][0]
    assert pe["in_precursor_window"] is None
    assert pe["window_tier"] == "unknown"


def test_mixed_tiers_all_tagged_correctly():
    o = _orchestrator(precursor_window_days=180)
    past_events = [
        _pe("E1", "C1", 60),    # primary
        _pe("E2", "C1", 200),   # extended
        _pe("E3", "C2", 500),   # historical
    ]
    past_events[2]["days_before_current_event"] = None  # unknown override for 4th
    past_events_with_unknown = past_events + [_pe("E4", "C2", None)]
    past_events_with_unknown[3]["days_before_current_event"] = None
    kg = _kg_with_past_events(past_events_with_unknown)
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    tiers = {pe["event_id"]: pe["window_tier"] for pe in result["past_events"]}
    assert tiers["E1"] == "primary"
    assert tiers["E2"] == "extended"
    assert tiers["E3"] == "unknown"
    assert tiers["E4"] == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-component indexing and top-N cap
# ─────────────────────────────────────────────────────────────────────────────

def test_per_component_index_groups_by_component_id():
    o = _orchestrator(per_component_top_n=10)
    past_events = [
        _pe("E1", "C1", 30, priority_score=9.0),
        _pe("E2", "C1", 60, priority_score=8.0),
        _pe("E3", "C2", 45, priority_score=7.0),
    ]
    kg = _kg_with_past_events(past_events)
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    index = result["seed_context"]["per_component_past_events"]
    assert "C1" in index
    assert "C2" in index
    assert set(index["C1"]) == {"E1", "E2"}
    assert index["C2"] == ["E3"]


def test_per_component_top_n_cap_applied():
    o = _orchestrator(per_component_top_n=3)
    past_events = [_pe(f"E{i}", "C1", i * 10, priority_score=10.0 - i) for i in range(1, 9)]
    kg = _kg_with_past_events(past_events)
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    index = result["seed_context"]["per_component_past_events"]
    assert len(index["C1"]) == 3


def test_per_component_index_uses_priority_score_order():
    """Top-N should be the highest-scored events."""
    o = _orchestrator(per_component_top_n=2)
    past_events = [
        _pe("LOW", "C1", 30, priority_score=1.0),
        _pe("HIGH", "C1", 300, priority_score=20.0),
        _pe("MID", "C1", 150, priority_score=10.0),
    ]
    kg = _kg_with_past_events(past_events)
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    index = result["seed_context"]["per_component_past_events"]
    assert "HIGH" in index["C1"]
    assert "MID" in index["C1"]
    assert "LOW" not in index["C1"]


def test_events_without_component_id_go_to_no_component_bucket():
    o = _orchestrator()
    pe = _pe("E1", None, 60)
    pe["component_id"] = None
    kg = _kg_with_past_events([pe])
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    index = result["seed_context"]["per_component_past_events"]
    assert "_no_component" in index


# ─────────────────────────────────────────────────────────────────────────────
# 3. temporal_search_summary
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_search_summary_counts_match_tiers():
    o = _orchestrator(precursor_window_days=180)
    past_events = [
        _pe("E1", "C1", 60),   # primary
        _pe("E2", "C1", 200),  # extended
        _pe("E3", "C2", 500),  # historical
    ]
    kg = _kg_with_past_events(past_events)
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    summary = result["seed_context"]["temporal_search_summary"]
    assert summary["total_past_event_count"] == 3
    assert summary["in_window_count"] == 1
    assert summary["out_of_window_count"] == 2
    assert summary["unknown_window_count"] == 0
    assert summary["precursor_window_days_used"] == 180


def test_temporal_search_summary_component_count():
    o = _orchestrator()
    past_events = [
        _pe("E1", "C1", 60),
        _pe("E2", "C2", 90),
        _pe("E3", "C3", 120),
    ]
    kg = _kg_with_past_events(past_events)
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    summary = result["seed_context"]["temporal_search_summary"]
    assert summary["component_count_with_history"] == 3


def test_temporal_search_summary_source_breakdown_kg():
    o = _orchestrator()
    past_events = [_pe("KG-E1", "C1", 60), _pe("KG-E2", "C2", 90)]
    kg = _kg_with_past_events(past_events)
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    bd = result["seed_context"]["temporal_search_summary"]["source_breakdown"]
    assert bd["kg"] == 2
    assert bd["cmms_cr"] == 0
    assert bd["cmms_wo"] == 0


def test_temporal_search_summary_source_breakdown_mixed():
    o = _orchestrator()
    past_events = [
        _pe("KG-E1", "C1", 60),
        _cmms_pe("cr", "CR-001", "C1", 30),
        _cmms_pe("wo", "WO-001", "C2", 20),
    ]
    kg = _kg_with_past_events(past_events)
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    bd = result["seed_context"]["temporal_search_summary"]["source_breakdown"]
    assert bd["kg"] == 1
    assert bd["cmms_cr"] == 1
    assert bd["cmms_wo"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_no_past_events_returns_kg_context_unchanged():
    o = _orchestrator()
    kg = _kg_with_past_events([])
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    assert result is kg  # unchanged reference


def test_enrichment_does_not_drop_existing_seed_context_keys():
    o = _orchestrator()
    kg = _kg_with_past_events([_pe("E1", "C1", 50)])
    kg["seed_context"]["cmms_past_events_injected"] = 1
    kg["seed_context"]["historical_support_channels"] = {"same_component_count": 1}
    result = o._enrich_past_events_temporal_metadata(kg_context=kg, event={})
    seed = result["seed_context"]
    assert seed["cmms_past_events_injected"] == 1
    assert seed["historical_support_channels"]["same_component_count"] == 1
    assert "per_component_past_events" in seed
    assert "temporal_search_summary" in seed


# ─────────────────────────────────────────────────────────────────────────────
# 5. _classify_past_event_source helper
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_source_kg():
    assert RCAReasoningOrchestrator._classify_past_event_source("KG::EVT-123") == "kg"
    assert RCAReasoningOrchestrator._classify_past_event_source("EVT-PLAIN") == "kg"
    assert RCAReasoningOrchestrator._classify_past_event_source(None) == "kg"


def test_classify_source_cmms_cr():
    assert RCAReasoningOrchestrator._classify_past_event_source("CMMS::CR::WO-001") == "cmms_cr"


def test_classify_source_cmms_wo():
    assert RCAReasoningOrchestrator._classify_past_event_source("CMMS::WO::WO-001") == "cmms_wo"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Manifest pipeline_config.temporal_search
# ─────────────────────────────────────────────────────────────────────────────

def test_manifest_contains_temporal_search_summary():
    o = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )
    temporal_summary = {
        "component_count_with_history": 2,
        "total_past_event_count": 4,
        "in_window_count": 2,
        "out_of_window_count": 2,
        "unknown_window_count": 0,
        "precursor_window_days_used": 180,
        "per_component_top_n_used": 5,
        "source_breakdown": {"kg": 3, "cmms_cr": 1, "cmms_wo": 0},
    }
    kg_context = {
        "subgraph_id": "KG::MAN",
        "components": [{"component_id": "C1"}],
        "failure_modes": [{"fm_id": "FM-1"}],
        "past_events": [],
        "seed_context": {
            "temporal_search_summary": temporal_summary,
            "historical_support_channels": {},
        },
    }
    manifest = o._stage_g_finalize_manifest(
        run_context={"run_id": "RUN-S2B", "input_refs": {"event_id": "EVT-S2B", "asset_id": "A1"}},
        kg_context=kg_context,
        tskr_patterns={"patterns": []},
        causality_candidates={
            "candidates": [],
            "provenance": {},
            "category_coverage": {},
            "uncertainty_summary": {},
            "applicability_summary": {},
            "decision_posture": {},
        },
        causality_candidates_pre_refine=None,
        evidence_bundle={"results": [], "pipeline_health": {"issues": []}},
        ishikawa_matrix=None,
        cmms_context=None,
        rca_card={
            "validation_status": {"schema_valid": True, "all_claims_cited": True,
                                  "passed_minimum_evidence_gate": True, "fallback_used": False},
            "analyst_review": {"decision_required": False, "writeback_recommendation": "ready_if_accepted"},
            "executive_summary": {"decision_status": "candidate_ready"},
            "primary_hypothesis": {"candidate_id": "NONE", "confidence_label": "low"},
            "recommended_actions": [],
            "contributing_causes": [],
        },
        input_validation={"ok": True},
        output_validation={"ok": True},
        optional_artifact_failures=[],
        kg_governance={"status": "green", "issues": [], "failure_mode_count": 1, "min_failure_modes_required": 0},
        barrier_analysis={"barriers": [], "summary": {"overall_status": "green", "barrier_count": 0,
                                                       "degraded_barrier_count": 0}},
    )
    temporal_in_manifest = (manifest.get("pipeline_config") or {}).get("temporal_search") or {}
    assert temporal_in_manifest.get("total_past_event_count") == 4
    assert temporal_in_manifest.get("component_count_with_history") == 2
    assert temporal_in_manifest.get("precursor_window_days_used") == 180
    assert temporal_in_manifest.get("source_breakdown", {}).get("kg") == 3


ALL_TESTS = [
    test_event_within_precursor_window_tagged_primary,
    test_event_at_window_boundary_tagged_primary,
    test_event_in_extended_window_tagged_extended,
    test_event_beyond_double_window_tagged_historical,
    test_event_with_no_days_before_tagged_unknown,
    test_mixed_tiers_all_tagged_correctly,
    test_per_component_index_groups_by_component_id,
    test_per_component_top_n_cap_applied,
    test_per_component_index_uses_priority_score_order,
    test_events_without_component_id_go_to_no_component_bucket,
    test_temporal_search_summary_counts_match_tiers,
    test_temporal_search_summary_component_count,
    test_temporal_search_summary_source_breakdown_kg,
    test_temporal_search_summary_source_breakdown_mixed,
    test_no_past_events_returns_kg_context_unchanged,
    test_enrichment_does_not_drop_existing_seed_context_keys,
    test_classify_source_kg,
    test_classify_source_cmms_cr,
    test_classify_source_cmms_wo,
    test_manifest_contains_temporal_search_summary,
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {test_fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
