"""
test_step2d_similar_events.py — Step 2d: Similar Event Identification

Covers:
- _query_plant_past_events: scoring dimensions, top-N cap, empty input
- _build_similar_event_list: plant-only (no adapter), adapter injected,
  degraded adapter, status transitions
- _annotate_candidates_with_oe_evidence: matching, threshold, no match
- _build_unresolved_gaps: OE gap entries
- LLMOEAdapter: prompt building, response parsing, HTTP error, malformed JSON
- SimilarEventAdapter Protocol: mock satisfaction
- Manifest artifacts summary

Run:  pytest test_step2d_similar_events.py -v
"""
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator
from adapters.similar_event_adapter import SimilarEventAdapter, TIER_CONFIDENCE_MULTIPLIERS
from adapters.llm_oe_adapter import LLMOEAdapter
from synthesis.rca_synthesizer_v31 import RuleValidatedRCASynthesizerV31

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _past_event(
    event_id: str,
    component_id: str = "PUMP-1",
    matched_component_ids: Optional[List[str]] = None,
    matched_failure_mode_ids: Optional[List[str]] = None,
    event_type: str = "reactor_trip",
    actuation_type: str = "reactor_protection_system_actuation",
    in_precursor_window: bool = True,
    window_tier: str = "primary",
) -> dict:
    return {
        "event_id": event_id,
        "asset_id": "PLANT-A",
        "component_id": component_id,
        "event_type": event_type,
        "actuation_type": actuation_type,
        "timestamp_start": "2023-06-01T10:00:00Z",
        "timestamp_end": None,
        "matched_component_ids": matched_component_ids or [component_id],
        "matched_failure_mode_ids": matched_failure_mode_ids or [],
        "in_precursor_window": in_precursor_window,
        "window_tier": window_tier,
        "priority_score": 0.80,
        "resolved": True,
        "fm_id": None,
    }


def _event(
    event_type: str = "reactor_trip",
    actuation_type: str = "reactor_protection_system_actuation",
) -> dict:
    return {
        "event_id": "EVT-001",
        "asset_id": "PLANT-A",
        "component_id": "PUMP-1",
        "event_type": event_type,
        "actuation_type": actuation_type,
        "timestamp_start": "2024-01-15T08:00:00Z",
        "severity": "significant",
        "symptom_signature": "unexpected_trip",
    }


def _kg_context(past_events: Optional[list] = None) -> dict:
    return {
        "subgraph_id": "SG-001",
        "event_id": "EVT-001",
        "asset_id": "PLANT-A",
        "components": [],
        "failure_modes": [],
        "past_events": past_events or [],
        "seed_context": {},
    }


def _candidate(component_id: str, fm_id: str = "FM-001") -> dict:
    return {
        "candidate_id": f"FM::{component_id}",
        "component_id": component_id,
        "failure_mode_id": fm_id,
        "composite_score": 0.70,
        "canonical_tuple": {"component": component_id, "failure_mode": fm_id},
    }


def _candidates_payload(cands: list) -> dict:
    return {"candidates": cands}


# ===========================================================================
# _query_plant_past_events
# ===========================================================================

def test_plant_engine_component_match_scores_correctly():
    """Component exact match contributes 0.40 to raw score."""
    past = [_past_event("EVT-PAST-1", component_id="PUMP-1", in_precursor_window=False)]
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context(past),
        causality_candidates=_candidates_payload([_candidate("PUMP-1")]),
    )
    assert len(result) == 1
    dims = result[0]["match_dimensions"]
    assert dims["component_match"] == 0.40


def test_plant_engine_fm_match_adds_score():
    """Failure mode ID match adds 0.25."""
    past = [_past_event("EVT-FM", component_id="PUMP-1",
                         matched_failure_mode_ids=["FM-001"],
                         in_precursor_window=False)]
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context(past),
        causality_candidates=_candidates_payload([_candidate("PUMP-1", fm_id="FM-001")]),
    )
    assert result[0]["match_dimensions"]["fm_match"] == 0.25


def test_plant_engine_event_type_match():
    """Event type match adds 0.15."""
    past = [_past_event("EVT-ET", event_type="reactor_trip", in_precursor_window=False)]
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(event_type="reactor_trip"),
        kg_context=_kg_context(past),
        causality_candidates=_candidates_payload([]),
    )
    assert result[0]["match_dimensions"]["event_type_match"] == 0.15


def test_plant_engine_actuation_type_match():
    """Actuation type match adds 0.10."""
    past = [_past_event("EVT-AT",
                         actuation_type="reactor_protection_system_actuation",
                         in_precursor_window=False)]
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(actuation_type="reactor_protection_system_actuation"),
        kg_context=_kg_context(past),
        causality_candidates=_candidates_payload([]),
    )
    assert result[0]["match_dimensions"]["actuation_match"] == 0.10


def test_plant_engine_in_precursor_window_boost():
    """in_precursor_window=True adds 0.10 window boost."""
    past = [_past_event("EVT-WIN", in_precursor_window=True)]
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context(past),
        causality_candidates=_candidates_payload([]),
    )
    assert result[0]["match_dimensions"]["window_boost"] == 0.10


def test_plant_engine_top_n_capped():
    """Only top-5 results are returned (default top_n=5)."""
    past = [_past_event(f"EVT-{i}") for i in range(10)]
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context(past),
        causality_candidates=_candidates_payload([]),
        top_n=5,
    )
    assert len(result) <= 5


def test_plant_engine_no_past_events_returns_empty():
    """Empty past_events list → empty result."""
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([]),
        causality_candidates=_candidates_payload([]),
    )
    assert result == []


def test_plant_engine_tier_discount_applied():
    """Plant confidence_weight uses tier multiplier 1.0 (no discount)."""
    past = [_past_event("EVT-DISC", in_precursor_window=True)]
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context(past),
        causality_candidates=_candidates_payload([_candidate("PUMP-1")]),
    )
    cw = result[0]["confidence_weight"]
    raw = result[0]["match_dimensions"]["raw_score"]
    assert abs(cw - min(1.0, raw * TIER_CONFIDENCE_MULTIPLIERS["plant"])) < 1e-5


# ===========================================================================
# SimilarEventAdapter Protocol
# ===========================================================================

def test_adapter_protocol_satisfied_by_mock():
    """A class implementing query() and degraded satisfies the Protocol."""
    class MockAdapter:
        degraded = False
        def query(self, *, level, asset_id, component_ids,
                  failure_mode_ids, event_type=None, actuation_type=None,
                  max_results=5, timeout_seconds=10.0):
            return []
    assert isinstance(MockAdapter(), SimilarEventAdapter)


def test_no_adapter_gives_plant_only_partial_status():
    """Without adapter: status=partial, fleet_count=0, industry_count=0."""

    class _Orch(RCAReasoningOrchestrator):
        pass

    orch = object.__new__(_Orch)
    orch.config = type("C", (), {"extra": {}})()
    orch.similar_event_adapter = None

    result = orch._build_similar_event_list(
        event=_event(),
        kg_context=_kg_context([_past_event("EVT-P1")]),
        causality_candidates=_candidates_payload([_candidate("PUMP-1")]),
    )
    assert result["status"] == "partial"
    assert result["summary"]["fleet_count"] == 0
    assert result["summary"]["industry_count"] == 0


def test_adapter_timeout_records_degraded_tier():
    """Adapter that raises on query → tier recorded in degraded_tiers."""

    class _FailingAdapter:
        degraded = True
        def query(self, **kwargs):
            raise TimeoutError("network timeout")

    class _Orch(RCAReasoningOrchestrator):
        pass

    orch = object.__new__(_Orch)
    orch.config = type("C", (), {"extra": {}})()
    orch.similar_event_adapter = _FailingAdapter()

    result = orch._build_similar_event_list(
        event=_event(),
        kg_context=_kg_context([]),
        causality_candidates=_candidates_payload([]),
    )
    assert "fleet" in result["summary"]["degraded_tiers"]
    assert "industry" in result["summary"]["degraded_tiers"]
    assert result["status"] == "partial"


# ===========================================================================
# LLMOEAdapter
# ===========================================================================

def test_llm_adapter_builds_structured_prompt():
    """Query terms appear in generated prompt."""
    adapter = LLMOEAdapter(fleet_url="http://x", industry_url="http://y")
    prompt = adapter._build_query_prompt(
        level="fleet",
        asset_id="PLANT-A",
        component_ids=["PUMP-1", "VALVE-2"],
        failure_mode_ids=["FM-001"],
        event_type="reactor_trip",
        actuation_type="RPS",
        max_results=5,
    )
    assert "PUMP-1" in prompt
    assert "FM-001" in prompt
    assert "reactor_trip" in prompt
    assert "fleet" in prompt.lower()


def test_llm_adapter_parses_valid_response():
    """Valid JSON array response → list of event dicts."""
    adapter = LLMOEAdapter()
    raw = [
        {"event_id": "FLEET-001", "confidence_weight": 0.80,
         "summary": "Pump seal failure", "root_cause_label": "wear",
         "date": "2022-03-01"},
    ]
    records = LLMOEAdapter._parse_response(raw, level="fleet")
    assert len(records) == 1
    assert records[0]["event_id"] == "FLEET-001"
    assert records[0]["source_level"] == "fleet"
    assert records[0]["source_db"] == "fleet_oe"


def test_llm_adapter_handles_http_error():
    """HTTP error → returns [], sets degraded=True, no exception raised."""
    adapter = LLMOEAdapter(fleet_url="http://test-fleet", api_key="key")

    class _FakeResp:
        def raise_for_status(self):
            from requests.exceptions import HTTPError
            raise HTTPError("500 Server Error")
        def json(self):
            return []

    with patch("requests.post", return_value=_FakeResp()):
        result = adapter.query(
            level="fleet",
            asset_id="A",
            component_ids=[],
            failure_mode_ids=[],
        )
    assert result == []
    assert adapter.degraded is True


def test_llm_adapter_handles_malformed_json():
    """Non-JSON response → returns []."""
    adapter = LLMOEAdapter(fleet_url="http://test-fleet")

    class _BadResp:
        def raise_for_status(self): pass
        def json(self):
            raise ValueError("No JSON")

    with patch("requests.post", return_value=_BadResp()):
        result = adapter.query(
            level="fleet",
            asset_id="A",
            component_ids=[],
            failure_mode_ids=[],
        )
    assert result == []
    assert adapter.degraded is True


# ===========================================================================
# _annotate_candidates_with_oe_evidence
# ===========================================================================

def test_oe_annotation_matching_component_cited():
    """A plant event matching candidate component_id is injected into candidate."""
    cand = _candidate("PUMP-1")
    cands_payload = _candidates_payload([deepcopy(cand)])
    sel = {
        "status": "partial",
        "events": [
            {
                "event_id": "EVT-MATCH",
                "source_level": "plant",
                "confidence_weight": 0.55,
                "component_id": "PUMP-1",
                "failure_signature": None,
                "root_cause_label": None,
                "source_db": "plant_kg",
                "date": "2023-06-01",
                "summary": "Seal degradation",
                "lessons_learned_ref": None,
            }
        ],
        "summary": {},
        "provenance": {},
    }
    RCAReasoningOrchestrator._annotate_candidates_with_oe_evidence(
        causality_candidates=cands_payload,
        similar_event_list=sel,
    )
    oe = cands_payload["candidates"][0].get("oe_reinstatement_evidence") or []
    assert any(e.get("event_id") == "EVT-MATCH" for e in oe)


def test_oe_annotation_below_threshold_not_cited():
    """Event with confidence_weight < 0.30 is not injected."""
    cand = _candidate("PUMP-1")
    cands_payload = _candidates_payload([deepcopy(cand)])
    sel = {
        "events": [
            {"event_id": "EVT-LOW", "source_level": "plant",
             "confidence_weight": 0.15, "component_id": "PUMP-1",
             "failure_signature": None, "root_cause_label": None},
        ],
    }
    RCAReasoningOrchestrator._annotate_candidates_with_oe_evidence(
        causality_candidates=cands_payload,
        similar_event_list=sel,
    )
    oe = cands_payload["candidates"][0].get("oe_reinstatement_evidence") or []
    assert not any(e.get("event_id") == "EVT-LOW" for e in oe)


def test_oe_annotation_no_match_no_injection():
    """Event with different component → no annotation on candidate."""
    cand = _candidate("PUMP-1")
    cands_payload = _candidates_payload([deepcopy(cand)])
    sel = {
        "events": [
            {"event_id": "EVT-OTHER", "source_level": "plant",
             "confidence_weight": 0.70, "component_id": "VALVE-99",
             "failure_signature": None, "root_cause_label": None},
        ],
    }
    RCAReasoningOrchestrator._annotate_candidates_with_oe_evidence(
        causality_candidates=cands_payload,
        similar_event_list=sel,
    )
    oe = cands_payload["candidates"][0].get("oe_reinstatement_evidence") or []
    assert oe == []


# ===========================================================================
# _build_unresolved_gaps — OE gap entries
# ===========================================================================

def _make_synth() -> RuleValidatedRCASynthesizerV31:
    from synthesis.rca_synthesizer_v31 import RCASynthesizerConfig
    cfg = RCASynthesizerConfig(allow_fallback_template_fill=True)
    return RuleValidatedRCASynthesizerV31(config=cfg, llm_client=MagicMock())


def test_unresolved_gaps_no_plant_match_emits_gap():
    """plant_count=0 → gap entry about no plant history."""
    synth = _make_synth()
    sel = {"summary": {"plant_count": 0, "degraded_tiers": []}}
    gaps = synth._build_unresolved_gaps(
        primary_candidate={},
        evidence_summary={"supporting": 1, "contradicting": 0},
        pattern_posture={},
        analyst_attention_flags=[],
        similar_event_list=sel,
    )
    assert any("plant" in g.lower() or "precedent" in g.lower() for g in gaps)


def test_unresolved_gaps_degraded_fleet_emits_gap():
    """degraded_tiers=['fleet'] → gap entry about fleet tier failure."""
    synth = _make_synth()
    sel = {"summary": {"plant_count": 2, "degraded_tiers": ["fleet"]}}
    gaps = synth._build_unresolved_gaps(
        primary_candidate={},
        evidence_summary={"supporting": 1, "contradicting": 0},
        pattern_posture={},
        analyst_attention_flags=[],
        similar_event_list=sel,
    )
    assert any("fleet" in g.lower() for g in gaps)


def test_unresolved_gaps_no_sel_no_oe_gap():
    """similar_event_list=None → no OE gap emitted."""
    synth = _make_synth()
    gaps = synth._build_unresolved_gaps(
        primary_candidate={},
        evidence_summary={"supporting": 1, "contradicting": 0},
        pattern_posture={},
        analyst_attention_flags=[],
        similar_event_list=None,
    )
    assert not any("plant" in g.lower() and "precedent" in g.lower() for g in gaps)


# ===========================================================================
# Manifest artifacts summary
# ===========================================================================

def test_build_similar_event_list_status_not_stub():
    """After WS2, status must not be 'not_implemented'."""

    class _Orch(RCAReasoningOrchestrator):
        pass

    orch = object.__new__(_Orch)
    orch.config = type("C", (), {"extra": {}})()
    orch.similar_event_adapter = None

    result = orch._build_similar_event_list(
        event=_event(),
        kg_context=_kg_context([_past_event("EVT-X")]),
        causality_candidates=_candidates_payload([]),
    )
    assert result["status"] != "not_implemented"
    assert "summary" in result
    assert "query_terms" in result


def test_build_similar_event_list_query_terms_present():
    """query_terms carries asset_id, component_ids, failure_mode_ids."""

    class _Orch(RCAReasoningOrchestrator):
        pass

    orch = object.__new__(_Orch)
    orch.config = type("C", (), {"extra": {}})()
    orch.similar_event_adapter = None

    result = orch._build_similar_event_list(
        event=_event(),
        kg_context=_kg_context([]),
        causality_candidates=_candidates_payload([_candidate("PUMP-1", fm_id="FM-001")]),
    )
    qt = result["query_terms"]
    assert qt["asset_id"] == "PLANT-A"
    assert "PUMP-1" in qt["component_ids"]
    assert "FM-001" in qt["failure_mode_ids"]
