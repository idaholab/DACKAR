"""
test_doc_extraction_pipeline_integration.py — Phase 3 unit tests for semantic
recurrence integration into TSKRTemporalScorerV1 and OrchestratorConfig.

Coverage:
  - TSKRTemporalScorerConfig: new semantic fields present with correct defaults
  - TSKRTemporalScorerV1: accepts doc_extraction_store parameter
  - _score_from_effective_count: float-count thresholds, profile bonuses
  - _score_failure_mode_pattern: semantic disabled → baseline unchanged
  - _score_failure_mode_pattern: semantic enabled, store returns matches →
      effective_recurrence_count updated, history_score re-scored, near_match_pattern=False
  - _score_failure_mode_pattern: semantic enabled, only near-matches →
      near_match_pattern=True, effective_recurrence_count unchanged
  - _score_failure_mode_pattern: semantic enabled, store raises → graceful fallback
  - _score_failure_mode_pattern: empty query text (no FM name) → store not called
  - novel_pattern uses effective_recurrence_count not raw count
  - Output dict fields: effective_recurrence_count, semantic_match_count, near_match_count
  - OrchestratorConfig: new semantic fields, correct defaults
  - RCAReasoningOrchestrator: set_doc_extraction_store setter
  - _apply_tskr_runtime_overrides: propagates semantic config + store into scorer
  - _apply_near_match_pattern_attention_flags: flag added when near_match_pattern=True
  - _apply_near_match_pattern_attention_flags: no flag when no near_match_pattern
  - _build_semantic_recurrence_provenance: summarises across patterns correctly
"""
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

# Stub heavy optional dependencies unavailable in the unit-test environment
for _mod in (
    "neo4j", "py2neo", "chromadb",
    "langchain_chroma", "langchain_community",
    "langchain_community.vectorstores", "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.tskr_temporal_scorer import (
    TSKRTemporalScorerV1,
    TSKRTemporalScorerConfig,
    RecurrenceProfile,
)
from orchestrators.rca_reasoning_orchestrator import (
    OrchestratorConfig,
    RCAReasoningOrchestrator,
)


# ---------------------------------------------------------------------------
# Minimal SemanticMatch stand-in (avoids importing DocExtractionStore deps)
# ---------------------------------------------------------------------------

@dataclass
class _FakeSemanticMatch:
    doc_id: str
    chain_index: int = 0
    similarity_score: float = 0.85
    confidence_weight: float = 1.0  # HIGH
    cause_is_symptom_factor: float = 1.0

    @property
    def semantic_contribution(self) -> float:
        return self.similarity_score * self.confidence_weight * self.cause_is_symptom_factor


def _fake_store(
    matches: Optional[List[_FakeSemanticMatch]] = None,
    near_matches: Optional[List[_FakeSemanticMatch]] = None,
    raise_exc: Optional[Exception] = None,
) -> MagicMock:
    store = MagicMock()
    if raise_exc is not None:
        store.query.side_effect = raise_exc
    else:
        store.query.return_value = (matches or [], near_matches or [])
    return store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _null_profile() -> RecurrenceProfile:
    return RecurrenceProfile(
        fm_id="FM-001",
        component_id="C-1",
        count=0,
        mean_inter_event_days=None,
        trend="insufficient_data",
        unresolved_count=0,
        most_recent_days_ago=None,
    )


def _profile_with_count(count: int) -> RecurrenceProfile:
    return RecurrenceProfile(
        fm_id="FM-001",
        component_id="C-1",
        count=count,
        mean_inter_event_days=30.0 if count > 1 else None,
        trend="stable",
        unresolved_count=0,
        most_recent_days_ago=None,
    )


def _minimal_scorer(
    cfg: Optional[TSKRTemporalScorerConfig] = None,
    store: Optional[Any] = None,
) -> TSKRTemporalScorerV1:
    return TSKRTemporalScorerV1(config=cfg, doc_extraction_store=store)


def _run_pattern(
    scorer: TSKRTemporalScorerV1,
    fm: Optional[Dict] = None,
    past_events: Optional[List] = None,
) -> Dict:
    """Call _score_failure_mode_pattern with minimal valid inputs."""
    fm = fm or {"fm_id": "FM-001", "component_id": "C-1", "name": "bearing wear"}
    return scorer._score_failure_mode_pattern(
        event_id="EVT-1",
        asset_id="ASSET-1",
        event_start=BASE,
        event_end=BASE,
        anomaly_windows=[],
        anomaly_window_summary={"window_start": None, "window_end": None, "duration_hours": None},
        signal_ids=[],
        telemetry_support=0.0,
        operator_family=None,
        fm=fm,
        past_events=past_events or [],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TSKRTemporalScorerConfig — new semantic fields
# ═══════════════════════════════════════════════════════════════════════════════

def test_config_semantic_recurrence_default_off():
    cfg = TSKRTemporalScorerConfig()
    assert cfg.enable_semantic_recurrence is False


def test_config_semantic_threshold_default():
    cfg = TSKRTemporalScorerConfig()
    assert cfg.semantic_similarity_threshold == 0.75


def test_config_near_match_window_default():
    cfg = TSKRTemporalScorerConfig()
    assert cfg.near_match_window == 0.10


def test_config_top_k_semantic_default():
    cfg = TSKRTemporalScorerConfig()
    assert cfg.top_k_semantic == 5


def test_config_override_semantic_fields():
    cfg = TSKRTemporalScorerConfig(
        enable_semantic_recurrence=True,
        semantic_similarity_threshold=0.80,
        near_match_window=0.05,
        top_k_semantic=3,
    )
    assert cfg.enable_semantic_recurrence is True
    assert cfg.semantic_similarity_threshold == 0.80
    assert cfg.near_match_window == 0.05
    assert cfg.top_k_semantic == 3


# ═══════════════════════════════════════════════════════════════════════════════
# TSKRTemporalScorerV1 construction
# ═══════════════════════════════════════════════════════════════════════════════

def test_scorer_default_no_store():
    scorer = TSKRTemporalScorerV1()
    assert scorer.doc_extraction_store is None


def test_scorer_accepts_store():
    store = _fake_store()
    scorer = TSKRTemporalScorerV1(doc_extraction_store=store)
    assert scorer.doc_extraction_store is store


# ═══════════════════════════════════════════════════════════════════════════════
# _score_from_effective_count
# ═══════════════════════════════════════════════════════════════════════════════

def test_effective_count_zero_returns_zero():
    scorer = _minimal_scorer()
    profile = _null_profile()
    assert scorer._score_from_effective_count(0.0, profile) == 0.0


def test_effective_count_less_than_one_floor_zero():
    scorer = _minimal_scorer()
    profile = _null_profile()
    # 0.4 semantic contribution → floor=0 → base=0.0
    assert scorer._score_from_effective_count(0.4, profile) == 0.0


def test_effective_count_one():
    scorer = _minimal_scorer()
    profile = _profile_with_count(0)
    score = scorer._score_from_effective_count(1.0, profile)
    assert abs(score - 0.35) < 1e-6


def test_effective_count_two():
    scorer = _minimal_scorer()
    profile = _profile_with_count(0)
    score = scorer._score_from_effective_count(2.0, profile)
    assert abs(score - 0.55) < 1e-6


def test_effective_count_four():
    scorer = _minimal_scorer()
    profile = _profile_with_count(0)
    score = scorer._score_from_effective_count(4.0, profile)
    assert abs(score - 0.70) < 1e-6


def test_effective_count_seven():
    scorer = _minimal_scorer()
    profile = _profile_with_count(0)
    score = scorer._score_from_effective_count(7.0, profile)
    assert abs(score - 0.80) < 1e-6


def test_effective_count_applies_trend_bonus():
    scorer = _minimal_scorer()
    profile = RecurrenceProfile(
        fm_id="FM-001", component_id="C-1", count=2,
        mean_inter_event_days=10.0, trend="increasing",
        unresolved_count=0, most_recent_days_ago=None,
    )
    score = scorer._score_from_effective_count(2.0, profile)
    assert abs(score - min(1.0, 0.55 + 0.15)) < 1e-6


def test_effective_count_applies_unresolved_bonus():
    scorer = _minimal_scorer()
    profile = RecurrenceProfile(
        fm_id="FM-001", component_id="C-1", count=1,
        mean_inter_event_days=None, trend="insufficient_data",
        unresolved_count=1, most_recent_days_ago=None,
    )
    score = scorer._score_from_effective_count(1.0, profile)
    assert abs(score - 0.45) < 1e-6  # 0.35 + 0.10


def test_effective_count_clamped_at_one():
    scorer = _minimal_scorer()
    profile = RecurrenceProfile(
        fm_id="FM-001", component_id="C-1", count=10,
        mean_inter_event_days=5.0, trend="increasing",
        unresolved_count=1, most_recent_days_ago=30,
    )
    score = scorer._score_from_effective_count(10.0, profile)
    assert score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# _score_failure_mode_pattern — semantic disabled (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def test_pattern_semantic_disabled_no_store():
    """When semantic is disabled and no store, baseline fields present."""
    scorer = _minimal_scorer()
    pat = _run_pattern(scorer)
    assert pat["recurrence_count"] == 0
    assert pat["effective_recurrence_count"] == 0.0
    assert pat["semantic_match_count"] == 0
    assert pat["near_match_count"] == 0
    assert pat["near_match_pattern"] is False


def test_pattern_semantic_disabled_with_store_not_queried():
    """Store present but semantic disabled → store.query never called."""
    store = _fake_store()
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=False)
    scorer = _minimal_scorer(cfg=cfg, store=store)
    _run_pattern(scorer)
    store.query.assert_not_called()


def test_pattern_baseline_novel_pattern_no_recurrence():
    """novel_pattern=True when no exact recurrence and no semantic matches."""
    scorer = _minimal_scorer()
    pat = _run_pattern(scorer)
    assert pat["novel_pattern"] is True


def test_pattern_baseline_novel_pattern_with_past_event():
    """novel_pattern=False when there is at least one exact past event."""
    scorer = _minimal_scorer()
    past = [{"matched_failure_mode_ids": ["FM-001"], "timestamp_start": "2024-01-01T00:00:00"}]
    pat = _run_pattern(scorer, past_events=past)
    assert pat["recurrence_count"] == 1
    assert pat["novel_pattern"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# _score_failure_mode_pattern — semantic enabled, store returns matches
# ═══════════════════════════════════════════════════════════════════════════════

def test_pattern_semantic_match_updates_effective_count():
    """A HIGH-confidence semantic match (sim=0.85) adds to effective_recurrence_count."""
    match = _FakeSemanticMatch(doc_id="CR-001", similarity_score=0.85)
    store = _fake_store(matches=[match])
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    pat = _run_pattern(scorer)
    # effective = 0 (exact) + 0.85*1.0*1.0 = 0.85
    assert abs(pat["effective_recurrence_count"] - 0.85) < 1e-4
    assert pat["semantic_match_count"] == 1
    assert pat["near_match_count"] == 0
    assert pat["near_match_pattern"] is False


def test_pattern_semantic_match_near_match_pattern_false_when_matches_present():
    """near_match_pattern stays False when there are above-threshold matches."""
    match = _FakeSemanticMatch(doc_id="CR-001", similarity_score=0.90)
    near = _FakeSemanticMatch(doc_id="CR-002", similarity_score=0.72)
    store = _fake_store(matches=[match], near_matches=[near])
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    pat = _run_pattern(scorer)
    assert pat["near_match_pattern"] is False
    assert pat["near_match_count"] == 1


def test_pattern_two_semantic_matches_contribution():
    """Two matches: uncapped sum would be 1.46, but tier cap clamps to 0.99 when exact count==0."""
    m1 = _FakeSemanticMatch(doc_id="CR-001", similarity_score=0.90)
    m2 = _FakeSemanticMatch(doc_id="CR-002", similarity_score=0.80, confidence_weight=0.7)
    store = _fake_store(matches=[m1, m2])
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    pat = _run_pattern(scorer)
    # Phase 0 tier cap: when recurrence_profile.count==0, effective_recurrence_count is capped at 0.99
    # to prevent semantic-only contributions from reaching the first exact-match tier.
    assert abs(pat["effective_recurrence_count"] - 0.99) < 1e-9
    assert pat["semantic_recurrence_capped"] is True


def test_pattern_semantic_match_rescores_history():
    """Semantic contributions are visible but capped below tier-1 when exact count==0."""
    m1 = _FakeSemanticMatch(doc_id="CR-001", similarity_score=0.90)
    m2 = _FakeSemanticMatch(doc_id="CR-002", similarity_score=0.85)
    store = _fake_store(matches=[m1, m2])
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    pat = _run_pattern(scorer)
    # Phase 0 tier cap: uncapped sum ~1.75 is clamped to 0.99; semantic_recurrence_capped=True
    assert abs(pat["effective_recurrence_count"] - 0.99) < 1e-9
    assert pat["semantic_recurrence_capped"] is True
    # novel_pattern must be False because effective_count > 0
    assert pat["novel_pattern"] is False


def test_pattern_store_query_receives_correct_args():
    """Store.query is called with FM name and configured params."""
    store = _fake_store()
    cfg = TSKRTemporalScorerConfig(
        enable_semantic_recurrence=True,
        semantic_similarity_threshold=0.80,
        near_match_window=0.05,
        top_k_semantic=3,
    )
    scorer = _minimal_scorer(cfg=cfg, store=store)
    fm = {"fm_id": "FM-001", "component_id": "C-1", "name": "pump cavitation", "expected_symptoms": "noise vibration"}
    _run_pattern(scorer, fm=fm)

    store.query.assert_called_once()
    call_kwargs = store.query.call_args
    query_text = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("query_text", "")
    # Query text should include FM name and symptoms
    assert "pump cavitation" in query_text
    assert "noise vibration" in query_text
    # Threshold and params forwarded
    assert call_kwargs[1].get("top_k") == 3 or (call_kwargs[0] and len(call_kwargs[0]) > 1)


# ═══════════════════════════════════════════════════════════════════════════════
# _score_failure_mode_pattern — only near-matches
# ═══════════════════════════════════════════════════════════════════════════════

def test_pattern_near_match_only_sets_flag():
    """No above-threshold matches + near-matches + zero exact recurrence → near_match_pattern=True."""
    near = _FakeSemanticMatch(doc_id="CR-003", similarity_score=0.72)
    store = _fake_store(matches=[], near_matches=[near])
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    pat = _run_pattern(scorer)
    assert pat["near_match_pattern"] is True
    assert pat["near_match_count"] == 1
    assert pat["semantic_match_count"] == 0
    # effective_recurrence_count unchanged when no above-threshold matches
    assert pat["effective_recurrence_count"] == 0.0


def test_pattern_near_match_not_set_when_exact_recurrence_exists():
    """near_match_pattern is False when there is at least one exact past event."""
    near = _FakeSemanticMatch(doc_id="CR-003", similarity_score=0.72)
    store = _fake_store(matches=[], near_matches=[near])
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    past = [{"matched_failure_mode_ids": ["FM-001"], "timestamp_start": "2024-01-01T00:00:00"}]
    pat = _run_pattern(scorer, past_events=past)
    # recurrence_count=1 → near_match_pattern condition requires count==0
    assert pat["near_match_pattern"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# _score_failure_mode_pattern — graceful fallback on store error
# ═══════════════════════════════════════════════════════════════════════════════

def test_pattern_store_exception_graceful_fallback():
    """If store.query raises, semantic fields default to zero and pattern proceeds."""
    store = _fake_store(raise_exc=RuntimeError("Chroma unavailable"))
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    pat = _run_pattern(scorer)
    assert pat["effective_recurrence_count"] == 0.0
    assert pat["semantic_match_count"] == 0
    assert pat["near_match_count"] == 0
    assert pat["near_match_pattern"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# _score_failure_mode_pattern — empty FM name
# ═══════════════════════════════════════════════════════════════════════════════

def test_pattern_empty_fm_name_skips_query():
    """FM with no name and no expected_symptoms → store.query not called."""
    store = _fake_store()
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    fm = {"fm_id": "FM-001", "component_id": "C-1"}  # no name or expected_symptoms
    _run_pattern(scorer, fm=fm)
    store.query.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# novel_pattern uses effective_recurrence_count
# ═══════════════════════════════════════════════════════════════════════════════

def test_novel_pattern_false_when_semantic_match_present():
    """Semantic match raises effective_recurrence_count above zero → novel_pattern=False."""
    match = _FakeSemanticMatch(doc_id="CR-001", similarity_score=0.90)
    store = _fake_store(matches=[match])
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    pat = _run_pattern(scorer)
    assert pat["effective_recurrence_count"] > 0
    assert pat["novel_pattern"] is False


def test_novel_pattern_true_near_match_only():
    """Near-match only: effective_count=0 → novel_pattern still True (no confirmed recurrence)."""
    near = _FakeSemanticMatch(doc_id="CR-003", similarity_score=0.72)
    store = _fake_store(matches=[], near_matches=[near])
    cfg = TSKRTemporalScorerConfig(enable_semantic_recurrence=True)
    scorer = _minimal_scorer(cfg=cfg, store=store)

    pat = _run_pattern(scorer)
    assert pat["effective_recurrence_count"] == 0.0
    assert pat["novel_pattern"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# OrchestratorConfig semantic fields
# ═══════════════════════════════════════════════════════════════════════════════

def test_orchestrator_config_semantic_defaults():
    cfg = OrchestratorConfig()
    assert cfg.enable_semantic_recurrence is False
    assert cfg.semantic_similarity_threshold == 0.75
    assert cfg.near_match_window == 0.10
    assert cfg.fm_id_resolution_threshold == 0.88
    assert cfg.top_k_semantic == 5


def test_orchestrator_config_override_semantic():
    cfg = OrchestratorConfig(
        enable_semantic_recurrence=True,
        semantic_similarity_threshold=0.80,
        near_match_window=0.05,
        fm_id_resolution_threshold=0.85,
        top_k_semantic=10,
    )
    assert cfg.enable_semantic_recurrence is True
    assert cfg.semantic_similarity_threshold == 0.80
    assert cfg.near_match_window == 0.05
    assert cfg.fm_id_resolution_threshold == 0.85
    assert cfg.top_k_semantic == 10


# ═══════════════════════════════════════════════════════════════════════════════
# RCAReasoningOrchestrator.set_doc_extraction_store
# ═══════════════════════════════════════════════════════════════════════════════

def _make_minimal_orchestrator() -> RCAReasoningOrchestrator:
    return RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )


def test_orchestrator_has_doc_extraction_store_field():
    orch = _make_minimal_orchestrator()
    assert hasattr(orch, "doc_extraction_store")
    assert orch.doc_extraction_store is None


def test_orchestrator_set_doc_extraction_store():
    orch = _make_minimal_orchestrator()
    store = MagicMock()
    orch.set_doc_extraction_store(store)
    assert orch.doc_extraction_store is store


# ═══════════════════════════════════════════════════════════════════════════════
# _apply_tskr_runtime_overrides propagates semantic config
# ═══════════════════════════════════════════════════════════════════════════════

def test_runtime_overrides_propagate_semantic_config():
    """Orchestrator propagates enable_semantic_recurrence and related params into scorer config."""
    scorer_cfg = TSKRTemporalScorerConfig()
    scorer = TSKRTemporalScorerV1(config=scorer_cfg)
    store = MagicMock()

    orch = _make_minimal_orchestrator()
    orch.tskr_temporal_scorer = scorer
    orch.doc_extraction_store = store
    orch.config = OrchestratorConfig(
        enable_semantic_recurrence=True,
        semantic_similarity_threshold=0.82,
        near_match_window=0.08,
        top_k_semantic=7,
    )

    orch._apply_tskr_runtime_overrides()

    assert scorer.config.enable_semantic_recurrence is True
    assert abs(scorer.config.semantic_similarity_threshold - 0.82) < 1e-9
    assert abs(scorer.config.near_match_window - 0.08) < 1e-9
    assert scorer.config.top_k_semantic == 7
    assert scorer.doc_extraction_store is store


def test_runtime_overrides_no_scorer_no_crash():
    orch = _make_minimal_orchestrator()
    orch.tskr_temporal_scorer = None
    orch._apply_tskr_runtime_overrides()  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# _apply_near_match_pattern_attention_flags
# ═══════════════════════════════════════════════════════════════════════════════

def _tskr_with_patterns(patterns: list) -> dict:
    return {"patterns": patterns}


def _rca_card() -> dict:
    return {"executive_summary": {"analyst_attention_flags": []}}


def test_near_match_attention_flag_added():
    orch = _make_minimal_orchestrator()
    patterns = [{"target_id": "FM-001", "near_match_pattern": True}]
    card = _rca_card()
    orch._apply_near_match_pattern_attention_flags(card, _tskr_with_patterns(patterns))
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("near-match" in f.lower() or "Near-match" in f for f in flags)


def test_near_match_attention_flag_not_added_when_none():
    orch = _make_minimal_orchestrator()
    patterns = [{"target_id": "FM-001", "near_match_pattern": False}]
    card = _rca_card()
    orch._apply_near_match_pattern_attention_flags(card, _tskr_with_patterns(patterns))
    assert card["executive_summary"]["analyst_attention_flags"] == []


def test_near_match_attention_flag_not_duplicated():
    orch = _make_minimal_orchestrator()
    patterns = [{"target_id": "FM-001", "near_match_pattern": True}]
    card = _rca_card()
    orch._apply_near_match_pattern_attention_flags(card, _tskr_with_patterns(patterns))
    orch._apply_near_match_pattern_attention_flags(card, _tskr_with_patterns(patterns))
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert len(flags) == 1  # idempotent


def test_near_match_attention_flag_no_patterns():
    orch = _make_minimal_orchestrator()
    card = _rca_card()
    orch._apply_near_match_pattern_attention_flags(card, None)
    assert card["executive_summary"]["analyst_attention_flags"] == []


def test_near_match_attention_multiple_fm_ids():
    orch = _make_minimal_orchestrator()
    patterns = [
        {"target_id": "FM-001", "near_match_pattern": True},
        {"target_id": "FM-002", "near_match_pattern": True},
        {"target_id": "FM-003", "near_match_pattern": False},
    ]
    card = _rca_card()
    orch._apply_near_match_pattern_attention_flags(card, _tskr_with_patterns(patterns))
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert len(flags) == 1
    assert "FM-001" in flags[0] and "FM-002" in flags[0]


# ═══════════════════════════════════════════════════════════════════════════════
# _build_semantic_recurrence_provenance
# ═══════════════════════════════════════════════════════════════════════════════

def test_semantic_provenance_disabled_store_none():
    orch = _make_minimal_orchestrator()
    orch.config = OrchestratorConfig(enable_semantic_recurrence=False)
    prov = orch._build_semantic_recurrence_provenance(None)
    assert prov["semantic_recurrence_used"] is False
    assert prov["store_present"] is False
    assert prov["semantic_match_count"] == 0
    assert prov["near_match_count"] == 0


def test_semantic_provenance_enabled_with_store():
    orch = _make_minimal_orchestrator()
    orch.doc_extraction_store = MagicMock()
    orch.config = OrchestratorConfig(
        enable_semantic_recurrence=True,
        semantic_similarity_threshold=0.80,
        near_match_window=0.05,
        top_k_semantic=3,
    )
    patterns = [
        {"target_id": "FM-001", "semantic_match_count": 2, "near_match_count": 1, "near_match_pattern": True},
        {"target_id": "FM-002", "semantic_match_count": 0, "near_match_count": 0, "near_match_pattern": False},
    ]
    prov = orch._build_semantic_recurrence_provenance({"patterns": patterns})
    assert prov["semantic_recurrence_used"] is True
    assert prov["store_present"] is True
    assert prov["semantic_match_count"] == 2
    assert prov["near_match_count"] == 1
    assert "FM-001" in prov["near_match_fm_ids"]
    assert prov["similarity_threshold"] == 0.80
    assert prov["near_match_window"] == 0.05
    assert prov["top_k_semantic"] == 3


def test_semantic_provenance_empty_patterns():
    orch = _make_minimal_orchestrator()
    orch.doc_extraction_store = MagicMock()
    orch.config = OrchestratorConfig(enable_semantic_recurrence=True)
    prov = orch._build_semantic_recurrence_provenance({"patterns": []})
    assert prov["semantic_match_count"] == 0
    assert prov["near_match_count"] == 0
    assert prov["near_match_fm_ids"] == []
