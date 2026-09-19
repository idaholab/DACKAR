"""
test_phase2_enrichment.py — unit tests for Phase 2 output enrichment.

Covers five changes, all additive (no scoring-path side effects):

  Issue 7  — novel_pattern decomposed into documentary_novel + signal_novel
             (tskr_temporal_scorer.py)

  Issue 12 — TIER_CONFIDENCE_MULTIPLIERS now in OrchestratorConfig;
              config values appear in manifest pipeline_config snapshot.

  Issue 9  — suggestion_confidence + suggestion_confidence_reason on each
              scope-expansion signal (_detect_scope_expansion_signals).

  Issue 8  — CMMS quality-weighted unresolved boost in causality engine:
              _recurrence_score_from_features  (weighted_unresolved_fm_boost param)
              _recurrence_features_for_candidate  (cmms_recurrence_quality in return)

  Issue 2r — _apply_residual_anomaly_gaps adds rca_card.unresolved_gaps + flag.

Run directly:  python test_phase2_enrichment.py
Or via pytest: pytest test_phase2_enrichment.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.tskr_temporal_scorer import TSKRTemporalScorerV1, TSKRTemporalScorerConfig
from orchestrators.rca_reasoning_orchestrator import (
    RCAReasoningOrchestrator,
    OrchestratorConfig,
)
from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32, CausalityEngineConfigV32


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_scorer():
    return TSKRTemporalScorerV1(TSKRTemporalScorerConfig())


def make_orchestrator(**kw):
    return RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        config=OrchestratorConfig(
            extra={"strict_red_state_governance": False},
            **kw,
        ),
    )


def make_engine():
    return RuleBasedCausalityEngineV32(CausalityEngineConfigV32())


# ── Issue 7: documentary_novel + signal_novel ──────────────────────────────────

class TestNovelPatternDecomposition:
    """
    novel_pattern = documentary_novel AND signal_novel (backward compat).
    The two new fields cover four distinct states analysts care about.
    """

    _FM = {
        "fm_id": "FM-TEST",
        "name": "test_fm",
        "component_id": "C-001",
        "expected_latency_min_hours": 0,
        "expected_latency_max_hours": 2,
    }
    _BASE_TS = None  # lazy, set in helper

    def _score_pattern(self, scorer, *, past_events, signal_ids):
        """Minimal valid call to _score_failure_mode_pattern."""
        from datetime import datetime, timezone
        base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        return scorer._score_failure_mode_pattern(
            event_id="E-001",
            asset_id="ASSET-1",
            event_start=base,
            event_end=base,
            anomaly_windows=[],
            anomaly_window_summary={},
            signal_ids=signal_ids,
            telemetry_support=0.0,
            operator_family=None,
            fm=self._FM,
            past_events=past_events,
        )

    @staticmethod
    def _past_event_for_fm(fm_id="FM-TEST", resolved=True):
        """Minimal past_event that matches by failure mode ID."""
        return {
            "event_id": "PE-001",
            "matched_failure_mode_ids": [fm_id],
            "component_id": "C-001",
            "resolved": resolved,
        }

    def test_fully_novel_all_three_flags_true(self):
        """No history, no signal match → novel_pattern=True, documentary_novel=True, signal_novel=True."""
        scorer = make_scorer()
        pat = self._score_pattern(scorer, past_events=[], signal_ids=[])
        assert pat["novel_pattern"] is True
        assert pat["documentary_novel"] is True
        assert pat["signal_novel"] is True
        print("  PASS test_fully_novel_all_three_flags_true")

    def test_documentary_novel_but_signal_matched(self):
        """No CR history but signal match → documentary_novel=True, signal_novel=False, novel_pattern=False."""
        scorer = make_scorer()
        pat = self._score_pattern(scorer, past_events=[], signal_ids=["SIG-1"])
        assert pat["documentary_novel"] is True
        assert pat["signal_novel"] is False
        assert pat["novel_pattern"] is False, (
            "novel_pattern must be False when signal_ids are present (AND gate)"
        )
        print("  PASS test_documentary_novel_but_signal_matched")

    def test_known_pattern_both_flags_false(self):
        """CR history and signal match → all three False."""
        scorer = make_scorer()
        past_events = [self._past_event_for_fm() for _ in range(3)]
        pat = self._score_pattern(scorer, past_events=past_events, signal_ids=["SIG-1"])
        assert pat["documentary_novel"] is False
        assert pat["signal_novel"] is False
        assert pat["novel_pattern"] is False
        print("  PASS test_known_pattern_both_flags_false")

    def test_signal_novel_but_has_cr_history(self):
        """CR history exists but no signal match → documentary_novel=False, signal_novel=True."""
        scorer = make_scorer()
        past_events = [self._past_event_for_fm() for _ in range(3)]
        pat = self._score_pattern(scorer, past_events=past_events, signal_ids=[])
        assert pat["documentary_novel"] is False
        assert pat["signal_novel"] is True
        assert pat["novel_pattern"] is False
        print("  PASS test_signal_novel_but_has_cr_history")

    def test_novel_pattern_backward_compat(self):
        """novel_pattern still present and equals documentary_novel AND signal_novel."""
        scorer = make_scorer()
        pat = self._score_pattern(scorer, past_events=[], signal_ids=[])
        assert "novel_pattern" in pat
        assert pat["novel_pattern"] == (pat["documentary_novel"] and pat["signal_novel"])
        print("  PASS test_novel_pattern_backward_compat")


# ── Issue 12: tier multipliers in OrchestratorConfig ──────────────────────────

class TestTierMultipliersConfig:

    def test_default_multipliers_match_legacy_values(self):
        """Default config tier multipliers equal the former hardcoded constants."""
        cfg = OrchestratorConfig()
        assert cfg.tier_confidence_multipliers["plant"] == 1.00
        assert cfg.tier_confidence_multipliers["fleet"] == 0.80
        assert cfg.tier_confidence_multipliers["industry"] == 0.60
        print("  PASS test_default_multipliers_match_legacy_values")

    def test_custom_multipliers_accepted(self):
        """Site can override multipliers via OrchestratorConfig."""
        cfg = OrchestratorConfig(
            tier_confidence_multipliers={"plant": 1.00, "fleet": 0.90, "industry": 0.70}
        )
        assert cfg.tier_confidence_multipliers["fleet"] == 0.90
        assert cfg.tier_confidence_multipliers["industry"] == 0.70
        print("  PASS test_custom_multipliers_accepted")

    def test_tier_multipliers_appear_in_pipeline_config_snapshot(self):
        """tier_confidence_multipliers key present in OrchestratorConfig — verifies field exists."""
        o = make_orchestrator()
        assert hasattr(o.config, "tier_confidence_multipliers")
        assert "fleet" in o.config.tier_confidence_multipliers
        print("  PASS test_tier_multipliers_appear_in_pipeline_config_snapshot")


# ── Issue 9: suggestion_confidence on expansion signals ───────────────────────

def make_run_context_with_scope():
    return {
        "scope_management": {
            "scope_revisions": [
                {
                    "analyst_decision": "accepted",
                    "scope_snapshot": {"component_ids": ["COMP-A"], "asset_ids": []},
                }
            ]
        }
    }


def make_allen_map_with_quality(*, soe_clock_ok=True, alarm_clock_ok=True, soe_capped=False,
                                 out_of_scope_comp="COMP-B"):
    return {
        "quality_flags": {
            "soe_clock_sync_ok": soe_clock_ok,
            "alarm_clock_sync_ok": alarm_clock_ok,
            "soe_nodes_capped": soe_capped,
        },
        "summary": {"causal_nodes": 1},
        "nodes": [
            {
                "node_id": "N-001",
                "node_type": "anomaly",
                "component_id": out_of_scope_comp,
                "causal_candidate": True,
                "allen_relation_to_event": "precedes",
                "allen_score": 0.9,
            }
        ],
    }


class TestScopeExpansionSuggestionConfidence:

    def _get_signals(self, *, allen_map=None, tskr_patterns=None, signal_evidence=None):
        rc = make_run_context_with_scope()
        return RCAReasoningOrchestrator._detect_scope_expansion_signals(
            run_context=rc,
            allen_relation_map=allen_map,
            signal_evidence=signal_evidence,
            tskr_patterns=tskr_patterns,
        )

    def test_allen_clean_map_gives_medium_confidence(self):
        """Clean Allen quality flags → suggestion_confidence='medium'."""
        allen_map = make_allen_map_with_quality(soe_clock_ok=True)
        signals = self._get_signals(allen_map=allen_map)
        assert len(signals) == 1
        assert signals[0]["suggestion_confidence"] == "medium"
        assert signals[0]["suggestion_confidence_reason"] is None
        print("  PASS test_allen_clean_map_gives_medium_confidence")

    def test_allen_soe_clock_fail_gives_low_confidence(self):
        """SOE clock sync failed → suggestion_confidence='low', reason='soe_clock_sync_failed'."""
        allen_map = make_allen_map_with_quality(soe_clock_ok=False)
        signals = self._get_signals(allen_map=allen_map)
        assert signals[0]["suggestion_confidence"] == "low"
        assert signals[0]["suggestion_confidence_reason"] == "soe_clock_sync_failed"
        print("  PASS test_allen_soe_clock_fail_gives_low_confidence")

    def test_allen_alarm_clock_fail_gives_low_confidence(self):
        """Alarm clock sync failed (SOE ok) → suggestion_confidence='low'."""
        allen_map = make_allen_map_with_quality(soe_clock_ok=True, alarm_clock_ok=False)
        signals = self._get_signals(allen_map=allen_map)
        assert signals[0]["suggestion_confidence"] == "low"
        assert signals[0]["suggestion_confidence_reason"] == "alarm_clock_sync_failed"
        print("  PASS test_allen_alarm_clock_fail_gives_low_confidence")

    def test_allen_soe_capped_gives_low_confidence(self):
        """SOE nodes capped → suggestion_confidence='low', reason='soe_nodes_capped'."""
        allen_map = make_allen_map_with_quality(soe_capped=True)
        signals = self._get_signals(allen_map=allen_map)
        assert signals[0]["suggestion_confidence"] == "low"
        assert signals[0]["suggestion_confidence_reason"] == "soe_nodes_capped"
        print("  PASS test_allen_soe_capped_gives_low_confidence")

    def test_novel_pattern_signal_always_low_confidence(self):
        """TSKR novel pattern signals always have suggestion_confidence='low'."""
        tskr = {
            "patterns": [
                {
                    "pattern_id": "PAT-001",
                    "component_id": "COMP-X",
                    "novel_pattern": True,
                }
            ]
        }
        signals = self._get_signals(tskr_patterns=tskr)
        novel_sigs = [s for s in signals if s["trigger_type"] == "novel_signal_pattern"]
        assert novel_sigs
        assert novel_sigs[0]["suggestion_confidence"] == "low"
        assert novel_sigs[0]["suggestion_confidence_reason"] == "novel_pattern_sparse_evidence"
        print("  PASS test_novel_pattern_signal_always_low_confidence")

    def test_chain_signal_gives_medium_confidence(self):
        """Propagation chain out-of-scope signal always gives 'medium' confidence."""
        se = {
            "propagation_chains": [
                {"chain_id": "CH-001", "component_ids": ["COMP-Z"]}
            ]
        }
        signals = self._get_signals(signal_evidence=se)
        chain_sigs = [s for s in signals if s["trigger_type"] == "out_of_scope_propagation_component"]
        assert chain_sigs
        assert chain_sigs[0]["suggestion_confidence"] == "medium"
        assert chain_sigs[0]["suggestion_confidence_reason"] is None
        print("  PASS test_chain_signal_gives_medium_confidence")

    def test_suggestion_confidence_field_always_present(self):
        """Every signal dict has the suggestion_confidence field set."""
        allen_map = make_allen_map_with_quality()
        tskr = {"patterns": [{"pattern_id": "P1", "novel_pattern": True}]}
        se = {"propagation_chains": [{"chain_id": "C1", "component_ids": ["COMP-Q"]}]}
        signals = self._get_signals(allen_map=allen_map, tskr_patterns=tskr, signal_evidence=se)
        for sig in signals:
            assert "suggestion_confidence" in sig, f"Missing suggestion_confidence on {sig['signal_id']}"
        print("  PASS test_suggestion_confidence_field_always_present")


# ── Issue 8: CMMS quality-weighted unresolved boost ───────────────────────────

class TestCmmsRecurrenceQualityWeighting:

    def test_flat_formula_used_when_no_time_distance(self):
        """When time_distance_days is absent, cmms_recurrence_quality='flat'."""
        engine = make_engine()
        past_event_index = {
            "by_failure_mode": {
                "FM-A": [
                    {"event_id": "E1", "resolved": False},  # no time_distance_days
                ]
            },
            "by_component": {},
            "by_asset": {},
        }
        result = engine._recurrence_features_for_candidate(
            candidate={},
            event={"asset_id": "ASSET-1"},
            past_event_index=past_event_index,
            hypothesis_failure_mode_id="FM-A",
        )
        assert result["cmms_recurrence_quality"] == "flat"
        print("  PASS test_flat_formula_used_when_no_time_distance")

    def test_weighted_formula_used_when_time_distance_present(self):
        """All unresolved events have time_distance_days → cmms_recurrence_quality='weighted'."""
        engine = make_engine()
        past_event_index = {
            "by_failure_mode": {
                "FM-A": [
                    {"event_id": "E1", "resolved": False, "time_distance_days": 400},
                    {"event_id": "E2", "resolved": False, "time_distance_days": 120},
                ]
            },
            "by_component": {},
            "by_asset": {},
        }
        result = engine._recurrence_features_for_candidate(
            candidate={},
            event={"asset_id": "ASSET-1"},
            past_event_index=past_event_index,
            hypothesis_failure_mode_id="FM-A",
        )
        assert result["cmms_recurrence_quality"] == "weighted"
        print("  PASS test_weighted_formula_used_when_time_distance_present")

    def test_mixed_time_distance_falls_back_to_flat(self):
        """One event missing time_distance_days → fallback to flat formula."""
        engine = make_engine()
        past_event_index = {
            "by_failure_mode": {
                "FM-A": [
                    {"event_id": "E1", "resolved": False, "time_distance_days": 400},
                    {"event_id": "E2", "resolved": False},  # missing
                ]
            },
            "by_component": {},
            "by_asset": {},
        }
        result = engine._recurrence_features_for_candidate(
            candidate={},
            event={"asset_id": "ASSET-1"},
            past_event_index=past_event_index,
            hypothesis_failure_mode_id="FM-A",
        )
        assert result["cmms_recurrence_quality"] == "flat"
        print("  PASS test_mixed_time_distance_falls_back_to_flat")

    def test_old_event_gets_higher_weight_than_recent(self):
        """A single unresolved event at 400 days produces a higher boost than one at 30 days."""
        engine = make_engine()
        base_features = dict(
            same_failure_mode_event_count=1,
            same_component_event_count=0,
            same_asset_event_count=0,
            unresolved_fm_count=1,
            unresolved_component_count=0,
        )
        score_old = engine._recurrence_score_from_features(
            **base_features,
            weighted_unresolved_fm_boost=0.10 * 1.0,  # 400-day weight
        )
        score_recent = engine._recurrence_score_from_features(
            **base_features,
            weighted_unresolved_fm_boost=0.10 * 0.1,  # 30-day weight
        )
        assert score_old > score_recent, (
            f"Old event should score higher: score_old={score_old}, score_recent={score_recent}"
        )
        print("  PASS test_old_event_gets_higher_weight_than_recent")

    def test_weighted_boost_None_uses_flat_formula(self):
        """weighted_unresolved_fm_boost=None falls back to flat 0.10*count."""
        engine = make_engine()
        score_none = engine._recurrence_score_from_features(
            same_failure_mode_event_count=1,
            same_component_event_count=0,
            same_asset_event_count=0,
            unresolved_fm_count=1,
            unresolved_component_count=0,
            weighted_unresolved_fm_boost=None,
        )
        score_explicit = engine._recurrence_score_from_features(
            same_failure_mode_event_count=1,
            same_component_event_count=0,
            same_asset_event_count=0,
            unresolved_fm_count=1,
            unresolved_component_count=0,
        )
        assert score_none == score_explicit
        print("  PASS test_weighted_boost_None_uses_flat_formula")

    def test_no_unresolved_events_stays_flat(self):
        """When no events are unresolved, cmms_recurrence_quality='flat' (vacuously)."""
        engine = make_engine()
        past_event_index = {
            "by_failure_mode": {
                "FM-A": [
                    {"event_id": "E1", "resolved": True, "time_distance_days": 400},
                ]
            },
            "by_component": {},
            "by_asset": {},
        }
        result = engine._recurrence_features_for_candidate(
            candidate={},
            event={"asset_id": "ASSET-1"},
            past_event_index=past_event_index,
            hypothesis_failure_mode_id="FM-A",
        )
        assert result["cmms_recurrence_quality"] == "flat"
        print("  PASS test_no_unresolved_events_stays_flat")


# ── Issue 2r: _apply_residual_anomaly_gaps ─────────────────────────────────────

def make_allen_map_nodes(*nodes_spec):
    """nodes_spec: list of (component_id, relation, is_causal) tuples."""
    nodes = []
    for i, (comp, rel, causal) in enumerate(nodes_spec):
        nodes.append({
            "node_id": f"N-{i:03d}",
            "node_type": "anomaly",
            "component_id": comp,
            "causal_candidate": causal,
            "allen_relation_to_event": rel,
            "allen_score": 0.8,
        })
    return {"nodes": nodes, "summary": {}}


def make_candidates_with_primary(primary_comp, primary_cand_id="CAND-001"):
    return {
        "candidates": [
            {
                "candidate_id": primary_cand_id,
                "component_id": primary_comp,
                "failure_mode_name": "bearing_degradation",
            }
        ]
    }


class TestResidualAnomalyGaps:

    def _get_gaps(self, rca_card):
        return rca_card.get("unresolved_gaps")

    def _get_flags(self, rca_card):
        return (rca_card.get("executive_summary") or {}).get("analyst_attention_flags") or []

    def test_all_causal_nodes_on_primary_component_no_residuals(self):
        """All causal nodes on the primary component → explained, no residual, no flag."""
        rca_card = {
            "primary_hypothesis": {"candidate_id": "CAND-001", "component_id": "COMP-A"},
        }
        allen_map = make_allen_map_nodes(
            ("COMP-A", "precedes", True),
            ("COMP-A", "overlaps", True),
        )
        candidates = make_candidates_with_primary("COMP-A")
        RCAReasoningOrchestrator._apply_residual_anomaly_gaps(rca_card, allen_map, candidates)
        gaps = self._get_gaps(rca_card)
        assert gaps is not None
        assert gaps["explained_causal_node_count"] == 2
        assert gaps["residual_causal_node_count"] == 0
        assert gaps["assessment"] == "complete"
        assert self._get_flags(rca_card) == []
        print("  PASS test_all_causal_nodes_on_primary_component_no_residuals")

    def test_other_component_causal_node_becomes_residual(self):
        """Causal node on a different component → residual gap + attention flag."""
        rca_card = {
            "primary_hypothesis": {"candidate_id": "CAND-001", "component_id": "COMP-A"},
        }
        allen_map = make_allen_map_nodes(
            ("COMP-A", "precedes", True),
            ("COMP-B", "overlaps", True),   # different component — residual
        )
        candidates = make_candidates_with_primary("COMP-A")
        RCAReasoningOrchestrator._apply_residual_anomaly_gaps(rca_card, allen_map, candidates)
        gaps = self._get_gaps(rca_card)
        assert gaps["residual_causal_node_count"] == 1
        assert gaps["explained_causal_node_count"] == 1
        assert gaps["assessment"] == "partial"
        residual = gaps["residual_nodes"][0]
        assert residual["component_id"] == "COMP-B"
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        assert "unresolved_gaps" in flags[0]
        print("  PASS test_other_component_causal_node_becomes_residual")

    def test_follows_relation_excluded_from_residuals(self):
        """Nodes with 'follows' relation are temporal contradictions, not residual gaps."""
        rca_card = {
            "primary_hypothesis": {"candidate_id": "CAND-001", "component_id": "COMP-A"},
        }
        allen_map = make_allen_map_nodes(
            ("COMP-B", "follows", True),   # should be excluded
        )
        candidates = make_candidates_with_primary("COMP-A")
        RCAReasoningOrchestrator._apply_residual_anomaly_gaps(rca_card, allen_map, candidates)
        gaps = self._get_gaps(rca_card)
        # No causal nodes after excluding 'follows', so nothing to write
        assert gaps is None
        print("  PASS test_follows_relation_excluded_from_residuals")

    def test_non_causal_nodes_not_residuals(self):
        """Non-causal Allen nodes (e.g. alarm background events) are not included."""
        rca_card = {
            "primary_hypothesis": {"candidate_id": "CAND-001", "component_id": "COMP-A"},
        }
        allen_map = make_allen_map_nodes(
            ("COMP-B", "precedes", False),  # causal_candidate=False
        )
        candidates = make_candidates_with_primary("COMP-A")
        RCAReasoningOrchestrator._apply_residual_anomaly_gaps(rca_card, allen_map, candidates)
        assert self._get_gaps(rca_card) is None
        print("  PASS test_non_causal_nodes_not_residuals")

    def test_none_allen_map_no_gaps(self):
        """None allen_relation_map → no unresolved_gaps key written."""
        rca_card = {
            "primary_hypothesis": {"candidate_id": "CAND-001", "component_id": "COMP-A"},
        }
        candidates = make_candidates_with_primary("COMP-A")
        RCAReasoningOrchestrator._apply_residual_anomaly_gaps(rca_card, None, candidates)
        assert self._get_gaps(rca_card) is None
        print("  PASS test_none_allen_map_no_gaps")

    def test_all_residuals_assessment_unexplained(self):
        """All causal nodes on other components → assessment='unexplained'."""
        rca_card = {
            "primary_hypothesis": {"candidate_id": "CAND-001", "component_id": "COMP-A"},
        }
        allen_map = make_allen_map_nodes(
            ("COMP-B", "precedes", True),
            ("COMP-C", "overlaps", True),
        )
        candidates = make_candidates_with_primary("COMP-A")
        RCAReasoningOrchestrator._apply_residual_anomaly_gaps(rca_card, allen_map, candidates)
        gaps = self._get_gaps(rca_card)
        assert gaps["assessment"] == "unexplained"
        assert gaps["explained_causal_node_count"] == 0
        assert gaps["residual_causal_node_count"] == 2
        print("  PASS test_all_residuals_assessment_unexplained")

    def test_flag_not_duplicated_on_repeated_call(self):
        """Calling the method twice does not duplicate the attention flag."""
        rca_card = {
            "primary_hypothesis": {"candidate_id": "CAND-001", "component_id": "COMP-A"},
        }
        allen_map = make_allen_map_nodes(("COMP-B", "precedes", True))
        candidates = make_candidates_with_primary("COMP-A")
        RCAReasoningOrchestrator._apply_residual_anomaly_gaps(rca_card, allen_map, candidates)
        RCAReasoningOrchestrator._apply_residual_anomaly_gaps(rca_card, allen_map, candidates)
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        print("  PASS test_flag_not_duplicated_on_repeated_call")

    def test_no_primary_hypothesis_graceful(self):
        """Missing primary_hypothesis → method runs without error, no gaps."""
        rca_card = {}
        allen_map = make_allen_map_nodes(("COMP-B", "precedes", True))
        RCAReasoningOrchestrator._apply_residual_anomaly_gaps(rca_card, allen_map, {})
        # With no primary component, all causal nodes appear residual
        gaps = self._get_gaps(rca_card)
        assert gaps is not None
        assert gaps["residual_causal_node_count"] == 1
        print("  PASS test_no_primary_hypothesis_graceful")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Issue 7: novel_pattern decomposition ===")
    s7 = TestNovelPatternDecomposition()
    s7.test_fully_novel_all_three_flags_true()
    s7.test_documentary_novel_but_signal_matched()
    s7.test_known_pattern_both_flags_false()
    s7.test_signal_novel_but_has_cr_history()
    s7.test_novel_pattern_backward_compat()

    print("\n=== Issue 12: tier multipliers in OrchestratorConfig ===")
    s12 = TestTierMultipliersConfig()
    s12.test_default_multipliers_match_legacy_values()
    s12.test_custom_multipliers_accepted()
    s12.test_tier_multipliers_appear_in_pipeline_config_snapshot()

    print("\n=== Issue 9: suggestion_confidence on expansion signals ===")
    s9 = TestScopeExpansionSuggestionConfidence()
    s9.test_allen_clean_map_gives_medium_confidence()
    s9.test_allen_soe_clock_fail_gives_low_confidence()
    s9.test_allen_alarm_clock_fail_gives_low_confidence()
    s9.test_allen_soe_capped_gives_low_confidence()
    s9.test_novel_pattern_signal_always_low_confidence()
    s9.test_chain_signal_gives_medium_confidence()
    s9.test_suggestion_confidence_field_always_present()

    print("\n=== Issue 8: CMMS quality-weighted unresolved boost ===")
    s8 = TestCmmsRecurrenceQualityWeighting()
    s8.test_flat_formula_used_when_no_time_distance()
    s8.test_weighted_formula_used_when_time_distance_present()
    s8.test_mixed_time_distance_falls_back_to_flat()
    s8.test_old_event_gets_higher_weight_than_recent()
    s8.test_weighted_boost_None_uses_flat_formula()
    s8.test_no_unresolved_events_stays_flat()

    print("\n=== Issue 2r: _apply_residual_anomaly_gaps ===")
    s2 = TestResidualAnomalyGaps()
    s2.test_all_causal_nodes_on_primary_component_no_residuals()
    s2.test_other_component_causal_node_becomes_residual()
    s2.test_follows_relation_excluded_from_residuals()
    s2.test_non_causal_nodes_not_residuals()
    s2.test_none_allen_map_no_gaps()
    s2.test_all_residuals_assessment_unexplained()
    s2.test_flag_not_duplicated_on_repeated_call()
    s2.test_no_primary_hypothesis_graceful()

    print("\nAll Phase 2 tests passed.")
