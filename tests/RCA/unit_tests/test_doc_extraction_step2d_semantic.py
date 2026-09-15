"""
test_doc_extraction_step2d_semantic.py — Phase 3b unit tests for semantic
similarity scoring in the Step 2d plant-tier similar event list.

Coverage:
  - _source_doc_id_from_event_id: CMMS::CR prefix, CMMS::WO prefix, KG-native (None)
  - _query_plant_past_events with doc_id_semantic_scores=None → original weights (backward compat)
  - _query_plant_past_events with doc_id_semantic_scores={} → renormalized weights, zero semantic dim
  - _query_plant_past_events with semantic score present → dim_semantic added, confidence raised
  - _query_plant_past_events: KG-native event → sem_sim=0.0, source_doc_id=None
  - _query_plant_past_events: CMMS event not in scores dict → sem_sim=0.0
  - _query_plant_past_events: output fields (semantic_similarity_score, source_doc_id, match_dimensions.semantic_match)
  - _query_plant_past_events: purely semantic hit (no structural match) surfaces via semantic alone
  - _query_plant_past_events: renormalized weights sum to 1.0 when all dims fire + semantic=1.0
  - _build_doc_id_semantic_scores: returns None when store absent
  - _build_doc_id_semantic_scores: returns None when semantic disabled
  - _build_doc_id_semantic_scores: returns {} when no candidates
  - _build_doc_id_semantic_scores: queries store per candidate, builds max-sim map
  - _build_doc_id_semantic_scores: uses fm name + symptoms from kg_context.failure_modes
  - _build_doc_id_semantic_scores: handles store exception gracefully
  - _build_doc_id_semantic_scores: same doc_id in two FM queries → keeps maximum
  - _build_similar_event_list: passes semantic scores to plant tier when enabled
  - _build_similar_event_list: semantic_scoring_applied in provenance
  - _build_similar_event_list: semantic disabled → provenance.semantic_scoring_applied=False
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in (
    "neo4j", "py2neo", "chromadb",
    "langchain_chroma", "langchain_community",
    "langchain_community.vectorstores", "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import (
    RCAReasoningOrchestrator,
    OrchestratorConfig,
)


# ---------------------------------------------------------------------------
# Minimal SemanticMatch stand-in (avoids importing DocExtractionStore deps)
# ---------------------------------------------------------------------------

@dataclass
class _FakeMatch:
    doc_id: str
    similarity_score: float = 0.85


def _fake_store(
    *,
    matches: Optional[List[_FakeMatch]] = None,
    near_matches: Optional[List[_FakeMatch]] = None,
    raise_exc: Optional[Exception] = None,
) -> MagicMock:
    store = MagicMock()
    if raise_exc is not None:
        store.query.side_effect = raise_exc
    else:
        store.query.return_value = (matches or [], near_matches or [])
    return store


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _past_event(
    event_id: str = "CMMS::CR::CR-2024-001",
    component_id: str = "PUMP-1",
    matched_component_ids: Optional[List[str]] = None,
    matched_failure_mode_ids: Optional[List[str]] = None,
    event_type: str = "reactor_trip",
    actuation_type: str = "rps_actuation",
    in_precursor_window: bool = False,
) -> dict:
    return {
        "event_id": event_id,
        "asset_id": "PLANT-A",
        "component_id": component_id,
        "event_type": event_type,
        "actuation_type": actuation_type,
        "timestamp_start": "2023-06-01T10:00:00Z",
        "matched_component_ids": matched_component_ids or [component_id],
        "matched_failure_mode_ids": matched_failure_mode_ids or [],
        "in_precursor_window": in_precursor_window,
        "resolved": True,
        "fm_id": None,
    }


def _event() -> dict:
    return {
        "event_id": "EVT-001",
        "asset_id": "PLANT-A",
        "event_type": "reactor_trip",
        "actuation_type": "rps_actuation",
    }


def _kg_context(
    past_events: Optional[list] = None,
    failure_modes: Optional[list] = None,
) -> dict:
    return {
        "asset_id": "PLANT-A",
        "past_events": past_events or [],
        "failure_modes": failure_modes or [],
    }


def _candidate(fm_id: str = "FM-001", component_id: str = "PUMP-1") -> dict:
    return {
        "candidate_id": f"FM::{component_id}",
        "component_id": component_id,
        "failure_mode_id": fm_id,
        "canonical_tuple": {"component": component_id, "failure_mode": fm_id},
    }


def _cands(*fm_ids: str) -> dict:
    return {"candidates": [_candidate(fm_id=fid) for fid in fm_ids]}


def _fm(fm_id: str, name: str = "", symptoms: str = "") -> dict:
    return {"fm_id": fm_id, "name": name, "expected_symptoms": symptoms}


def _make_orchestrator(
    *,
    store: Optional[Any] = None,
    enable_semantic: bool = True,
) -> RCAReasoningOrchestrator:
    orch = RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        doc_extraction_store=store,
        config=OrchestratorConfig(
            enable_semantic_recurrence=enable_semantic,
            semantic_similarity_threshold=0.75,
            near_match_window=0.10,
            top_k_semantic=5,
        ),
    )
    return orch


# ═══════════════════════════════════════════════════════════════════════════════
# _source_doc_id_from_event_id
# ═══════════════════════════════════════════════════════════════════════════════

def test_source_doc_id_cmms_cr():
    result = RCAReasoningOrchestrator._source_doc_id_from_event_id("CMMS::CR::CR-2024-001")
    assert result == "CR-2024-001"


def test_source_doc_id_cmms_wo():
    result = RCAReasoningOrchestrator._source_doc_id_from_event_id("CMMS::WO::WO-2024-042")
    assert result == "WO-2024-042"


def test_source_doc_id_kg_native_returns_none():
    result = RCAReasoningOrchestrator._source_doc_id_from_event_id("EVT-PAST-001")
    assert result is None


def test_source_doc_id_empty_returns_none():
    result = RCAReasoningOrchestrator._source_doc_id_from_event_id("")
    assert result is None


def test_source_doc_id_partial_prefix_returns_none():
    result = RCAReasoningOrchestrator._source_doc_id_from_event_id("CMMS::CR::")
    # Empty suffix — no doc_id
    assert result == ""  # edge case: prefix matches but value is empty string


# ═══════════════════════════════════════════════════════════════════════════════
# _query_plant_past_events — backward compatibility (doc_id_semantic_scores=None)
# ═══════════════════════════════════════════════════════════════════════════════

def test_plant_query_no_semantic_uses_original_weights():
    """When doc_id_semantic_scores=None original weights are used (backward compat)."""
    pe = _past_event("CMMS::CR::CR-001", in_precursor_window=False)
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands(),
    )
    assert len(result) == 1
    dims = result[0]["match_dimensions"]
    assert dims["component_match"] == 0.40


def test_plant_query_no_semantic_output_has_semantic_fields_zero():
    """Even without semantic scores, output records carry the new fields at zero."""
    pe = _past_event("CMMS::CR::CR-001")
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands(),
    )
    assert result[0]["semantic_similarity_score"] == 0.0
    assert result[0]["match_dimensions"]["semantic_match"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# _query_plant_past_events — semantic enabled, scores provided
# ═══════════════════════════════════════════════════════════════════════════════

def test_plant_query_renormalized_component_weight():
    """When semantic scores dict is provided component weight becomes 0.36."""
    pe = _past_event("CMMS::CR::CR-001", in_precursor_window=False)
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands(),
        doc_id_semantic_scores={},
    )
    dims = result[0]["match_dimensions"]
    assert abs(dims["component_match"] - 0.36) < 1e-9


def test_plant_query_renormalized_fm_weight():
    pe = _past_event("CMMS::CR::CR-001", matched_failure_mode_ids=["FM-001"])
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands("FM-001"),
        doc_id_semantic_scores={},
    )
    dims = result[0]["match_dimensions"]
    assert abs(dims["fm_match"] - 0.225) < 1e-9


def test_plant_query_all_dims_renormalized_sum_to_one():
    """All five structural dims fire + semantic=1.0 → raw_score=1.0, confidence=1.0."""
    pe = _past_event(
        "CMMS::CR::CR-001",
        matched_failure_mode_ids=["FM-001"],
        in_precursor_window=True,
    )
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands("FM-001"),
        doc_id_semantic_scores={"CR-001": 1.0},
    )
    raw = result[0]["match_dimensions"]["raw_score"]
    # 0.36 + 0.225 + 0.135 + 0.09 + 0.09 + 0.10*1.0 = 1.00
    assert abs(raw - 1.0) < 1e-9
    assert result[0]["confidence_weight"] == 1.0


def test_plant_query_semantic_score_fractional():
    """Semantic sim=0.80 → dim_semantic = 0.10 * 0.80 = 0.08."""
    pe = _past_event("CMMS::CR::CR-001", in_precursor_window=False)
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands(),
        doc_id_semantic_scores={"CR-001": 0.80},
    )
    dims = result[0]["match_dimensions"]
    assert abs(dims["semantic_match"] - 0.08) < 1e-6
    assert abs(result[0]["semantic_similarity_score"] - 0.80) < 1e-6


def test_plant_query_cmms_event_not_in_scores_gets_zero_semantic():
    """CMMS event whose doc_id is not in scores dict → semantic_sim=0.0."""
    pe = _past_event("CMMS::CR::CR-999")
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands(),
        doc_id_semantic_scores={"CR-001": 0.90},  # different doc
    )
    assert result[0]["semantic_similarity_score"] == 0.0
    assert result[0]["match_dimensions"]["semantic_match"] == 0.0


def test_plant_query_kg_native_event_gets_zero_semantic():
    """KG-native event (no CMMS:: prefix) → source_doc_id=None, semantic=0.0."""
    pe = _past_event("EVT-PAST-KG-001")
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands(),
        doc_id_semantic_scores={"EVT-PAST-KG-001": 0.95},  # key doesn't match prefix logic
    )
    assert result[0]["semantic_similarity_score"] == 0.0
    assert result[0]["source_doc_id"] is None


def test_plant_query_source_doc_id_in_output():
    """source_doc_id field is populated correctly for CMMS events."""
    pe = _past_event("CMMS::WO::WO-2024-007")
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands(),
        doc_id_semantic_scores={},
    )
    assert result[0]["source_doc_id"] == "WO-2024-007"


def test_plant_query_purely_semantic_event_surfaces():
    """An event with zero structural matches still surfaces if semantic score is high."""
    # Build manually — avoid fixture's `or [component_id]` default for matched_component_ids
    pe = {
        "event_id": "CMMS::CR::CR-SEM",
        "asset_id": "PLANT-A",
        "component_id": "VALVE-X",
        "event_type": "maintenance",        # different from current "reactor_trip"
        "actuation_type": "manual",         # different from current "rps_actuation"
        "timestamp_start": "2023-06-01T10:00:00Z",
        "matched_component_ids": [],        # explicitly empty — no component match
        "matched_failure_mode_ids": [],
        "in_precursor_window": False,
        "resolved": True,
        "fm_id": None,
    }
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe]),
        causality_candidates=_cands(),
        doc_id_semantic_scores={"CR-SEM": 0.92},
        top_n=5,
    )
    assert len(result) == 1
    # raw_score = 0 + 0 + 0 + 0 + 0 + 0.10*0.92 = 0.092
    assert abs(result[0]["match_dimensions"]["raw_score"] - 0.092) < 1e-6
    assert result[0]["confidence_weight"] > 0.0


def test_plant_query_semantic_boosts_ranking():
    """Event with semantic match ranks above a structurally equivalent event without one."""
    pe_sem = _past_event("CMMS::CR::CR-SEM", in_precursor_window=False)
    pe_no  = _past_event("CMMS::CR::CR-NON", in_precursor_window=False)
    result = RCAReasoningOrchestrator._query_plant_past_events(
        event=_event(),
        kg_context=_kg_context([pe_sem, pe_no]),
        causality_candidates=_cands(),
        doc_id_semantic_scores={"CR-SEM": 0.88},
        top_n=5,
    )
    ids = [r["event_id"] for r in result]
    assert ids[0] == "CMMS::CR::CR-SEM"


# ═══════════════════════════════════════════════════════════════════════════════
# _build_doc_id_semantic_scores
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_semantic_scores_no_store_returns_none():
    orch = _make_orchestrator(store=None, enable_semantic=True)
    result = orch._build_doc_id_semantic_scores(
        kg_context=_kg_context(), causality_candidates=_cands("FM-1"), query_top_n=3
    )
    assert result is None


def test_build_semantic_scores_disabled_returns_none():
    orch = _make_orchestrator(store=_fake_store(), enable_semantic=False)
    result = orch._build_doc_id_semantic_scores(
        kg_context=_kg_context(), causality_candidates=_cands("FM-1"), query_top_n=3
    )
    assert result is None


def test_build_semantic_scores_no_candidates_returns_empty():
    orch = _make_orchestrator(store=_fake_store())
    result = orch._build_doc_id_semantic_scores(
        kg_context=_kg_context(), causality_candidates={"candidates": []}, query_top_n=3
    )
    assert result == {}


def test_build_semantic_scores_queries_store_per_candidate():
    """One store.query call per candidate that has a resolvable FM name."""
    store = _fake_store(matches=[_FakeMatch("CR-001", 0.88)])
    orch = _make_orchestrator(store=store)
    kg = _kg_context(failure_modes=[
        _fm("FM-001", name="bearing wear", symptoms="vibration noise"),
        _fm("FM-002", name="pump cavitation", symptoms="pressure drop"),
    ])
    orch._build_doc_id_semantic_scores(
        kg_context=kg,
        causality_candidates=_cands("FM-001", "FM-002"),
        query_top_n=3,
    )
    assert store.query.call_count == 2


def test_build_semantic_scores_uses_fm_name_and_symptoms():
    """Query text contains FM name and expected_symptoms joined by ' | '."""
    store = _fake_store()
    orch = _make_orchestrator(store=store)
    kg = _kg_context(failure_modes=[
        _fm("FM-001", name="bearing wear", symptoms="high vibration"),
    ])
    orch._build_doc_id_semantic_scores(
        kg_context=kg,
        causality_candidates=_cands("FM-001"),
        query_top_n=3,
    )
    query_text = store.query.call_args[0][0]
    assert "bearing wear" in query_text
    assert "high vibration" in query_text


def test_build_semantic_scores_skips_candidate_without_fm_in_kg():
    """Candidate whose fm_id is not in kg_context.failure_modes → no query text → skipped."""
    store = _fake_store()
    orch = _make_orchestrator(store=store)
    orch._build_doc_id_semantic_scores(
        kg_context=_kg_context(failure_modes=[]),  # FM-001 not in kg
        causality_candidates=_cands("FM-001"),
        query_top_n=3,
    )
    store.query.assert_not_called()


def test_build_semantic_scores_builds_max_sim_map():
    """Returns doc_id → max similarity across both matches and near_matches."""
    matches = [_FakeMatch("CR-001", 0.90), _FakeMatch("CR-002", 0.82)]
    near    = [_FakeMatch("CR-003", 0.70)]
    store = _fake_store(matches=matches, near_matches=near)
    orch = _make_orchestrator(store=store)
    kg = _kg_context(failure_modes=[_fm("FM-001", name="bearing wear")])
    result = orch._build_doc_id_semantic_scores(
        kg_context=kg, causality_candidates=_cands("FM-001"), query_top_n=3
    )
    assert result["CR-001"] == 0.90
    assert result["CR-002"] == 0.82
    assert result["CR-003"] == 0.70


def test_build_semantic_scores_keeps_maximum_across_fm_queries():
    """Same doc_id returned by two different FM queries → keeps higher score."""
    store = MagicMock()
    # First FM query: CR-001 at 0.80
    # Second FM query: CR-001 at 0.92
    store.query.side_effect = [
        ([_FakeMatch("CR-001", 0.80)], []),
        ([_FakeMatch("CR-001", 0.92)], []),
    ]
    orch = _make_orchestrator(store=store)
    kg = _kg_context(failure_modes=[
        _fm("FM-001", name="bearing wear"),
        _fm("FM-002", name="corrosion"),
    ])
    result = orch._build_doc_id_semantic_scores(
        kg_context=kg, causality_candidates=_cands("FM-001", "FM-002"), query_top_n=3
    )
    assert abs(result["CR-001"] - 0.92) < 1e-9


def test_build_semantic_scores_store_exception_graceful():
    """Store query exception does not propagate — result is partial (may be empty)."""
    store = _fake_store(raise_exc=RuntimeError("Chroma offline"))
    orch = _make_orchestrator(store=store)
    kg = _kg_context(failure_modes=[_fm("FM-001", name="bearing wear")])
    result = orch._build_doc_id_semantic_scores(
        kg_context=kg, causality_candidates=_cands("FM-001"), query_top_n=3
    )
    assert isinstance(result, dict)
    assert result == {}


# ═══════════════════════════════════════════════════════════════════════════════
# _build_similar_event_list — semantic integration
# ═══════════════════════════════════════════════════════════════════════════════

def _minimal_orchestrator_for_list(
    store: Optional[Any] = None,
    enable_semantic: bool = True,
) -> RCAReasoningOrchestrator:
    orch = _make_orchestrator(store=store, enable_semantic=enable_semantic)
    orch.similar_event_adapter = None
    return orch


def test_similar_event_list_provenance_semantic_applied():
    """When semantic enabled + store present, provenance.semantic_scoring_applied=True."""
    store = _fake_store(matches=[_FakeMatch("CR-001", 0.88)])
    orch = _minimal_orchestrator_for_list(store=store)
    pe = _past_event("CMMS::CR::CR-001")
    kg = _kg_context(
        past_events=[pe],
        failure_modes=[_fm("FM-001", name="bearing wear")],
    )
    result = orch._build_similar_event_list(
        event=_event(),
        kg_context=kg,
        causality_candidates=_cands("FM-001"),
    )
    assert result["provenance"]["semantic_scoring_applied"] is True
    assert result["provenance"]["semantic_doc_count"] >= 1


def test_similar_event_list_provenance_semantic_not_applied_when_disabled():
    """When semantic is disabled, provenance.semantic_scoring_applied=False."""
    store = _fake_store()
    orch = _minimal_orchestrator_for_list(store=store, enable_semantic=False)
    pe = _past_event("CMMS::CR::CR-001")
    kg = _kg_context(past_events=[pe])
    result = orch._build_similar_event_list(
        event=_event(),
        kg_context=kg,
        causality_candidates=_cands("FM-001"),
    )
    assert result["provenance"]["semantic_scoring_applied"] is False
    assert result["provenance"]["semantic_doc_count"] == 0


def test_similar_event_list_semantic_match_raises_confidence():
    """A CMMS past event whose doc matched semantically gets higher confidence_weight."""
    store = _fake_store(matches=[_FakeMatch("CR-MATCH", 0.90)])
    orch = _minimal_orchestrator_for_list(store=store)

    pe_semantic = _past_event("CMMS::CR::CR-MATCH", matched_component_ids=[])
    pe_baseline = _past_event("CMMS::CR::CR-OTHER", matched_component_ids=[])
    kg = _kg_context(
        past_events=[pe_semantic, pe_baseline],
        failure_modes=[_fm("FM-001", name="bearing wear")],
    )
    result = orch._build_similar_event_list(
        event=_event(),
        kg_context=kg,
        causality_candidates=_cands("FM-001"),
    )
    events = result["events"]
    assert len(events) >= 1
    sem_event = next((e for e in events if e["event_id"] == "CMMS::CR::CR-MATCH"), None)
    other_event = next((e for e in events if e["event_id"] == "CMMS::CR::CR-OTHER"), None)
    assert sem_event is not None
    if other_event is not None:
        assert sem_event["confidence_weight"] > other_event["confidence_weight"]


def test_similar_event_list_no_store_no_semantic_fields_broken():
    """Without store, plant events are scored normally (semantic_similarity_score=0.0)."""
    orch = _minimal_orchestrator_for_list(store=None, enable_semantic=False)
    pe = _past_event("EVT-KG-001")
    result = orch._build_similar_event_list(
        event=_event(),
        kg_context=_kg_context(past_events=[pe]),
        causality_candidates=_cands(),
    )
    events = result["events"]
    assert all(e.get("semantic_similarity_score", 0.0) == 0.0 for e in events)
