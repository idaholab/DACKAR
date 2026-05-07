"""
test_phase1_hardening.py — unit tests for Phase 1 safety-critical hardening.

Covers three new behaviours added to rca_reasoning_orchestrator.py:

  Issue 3 — barrier_gate_degraded_acknowledged
      _compute_review_hooks: SOE/PLC pairing violation blocks writeback unless
      analyst_review.barrier_gate_degraded_acknowledged == True.

  Issue 5 — _apply_fast_transient_attention_flags
      Fires an analyst attention flag when event_type is a configured fast-transient
      type AND the Allen map contains at least one causal node.

  Issue 11 — _apply_category_l_floor_attention_flags
      Fires an analyst attention flag when no Category L candidate clears the
      score floor AND a recurrence signal exists (open CRs or candidate recurrence).

Run directly:  python test_phase1_hardening.py
Or via pytest: pytest test_phase1_hardening.py
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

from orchestrators.rca_reasoning_orchestrator import (
    RCAReasoningOrchestrator,
    OrchestratorConfig,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def make_orchestrator(**extra_config):
    return RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        config=OrchestratorConfig(
            extra={"strict_red_state_governance": False, "hard_abort_on_kg_red_state": False},
            **extra_config,
        ),
    )


def make_clean_card(extra_analyst_review=None):
    card = {
        "validation_status": {
            "schema_valid": True,
            "all_claims_cited": True,
            "passed_minimum_evidence_gate": True,
            "fallback_used": False,
        },
        "analyst_review": {
            "decision_required": False,
            "writeback_recommendation": "ready_if_accepted",
        },
        "executive_summary": {
            "decision_status": "candidate_ready",
        },
    }
    if extra_analyst_review:
        card["analyst_review"].update(extra_analyst_review)
    return card


def make_ok_output():
    return {"ok": True}


def make_coverage_with_soe_plc_violated():
    return {
        "overall_status": "complete",
        "paired_data_checks": {"soe_protection_logic_pairing": "violated"},
    }


def make_coverage_with_soe_plc_warning():
    return {
        "overall_status": "complete",
        "paired_data_checks": {"soe_protection_logic_pairing": "warning"},
    }


def make_coverage_clean():
    return {
        "overall_status": "complete",
        "paired_data_checks": {"soe_protection_logic_pairing": "ok"},
    }


# ── Issue 3: barrier_gate_degraded_acknowledged ────────────────────────────────

class TestBarrierGateDegradedAcknowledged:

    def test_soe_plc_violated_blocks_writeback_by_default(self):
        """SOE present + PLC absent → writeback blocked when analyst has not acknowledged."""
        o = make_orchestrator()
        result = o._compute_review_hooks(
            make_clean_card(),
            make_ok_output(),
            coverage_summary=make_coverage_with_soe_plc_violated(),
        )
        assert result["writeback_ready"] is False
        degraded = result.get("degraded_reasons") or []
        assert any("Barrier logic gate" in r or "protection logic" in r.lower() for r in degraded), (
            f"Expected barrier degradation reason in: {degraded}"
        )
        print("  PASS test_soe_plc_violated_blocks_writeback_by_default")

    def test_soe_plc_warning_blocks_writeback_by_default(self):
        """Warning-level SOE/PLC pairing also blocks writeback without acknowledgement."""
        o = make_orchestrator()
        result = o._compute_review_hooks(
            make_clean_card(),
            make_ok_output(),
            coverage_summary=make_coverage_with_soe_plc_warning(),
        )
        assert result["writeback_ready"] is False
        print("  PASS test_soe_plc_warning_blocks_writeback_by_default")

    def test_barrier_gate_ack_clears_soe_plc_block(self):
        """analyst_review.barrier_gate_degraded_acknowledged=True clears the SOE/PLC degraded_reason."""
        o = make_orchestrator()
        card = make_clean_card(extra_analyst_review={"barrier_gate_degraded_acknowledged": True})
        result = o._compute_review_hooks(
            card,
            make_ok_output(),
            coverage_summary=make_coverage_with_soe_plc_violated(),
        )
        assert result["writeback_ready"] is True, (
            f"Expected writeback_ready=True after barrier ack. degraded_reasons={result.get('degraded_reasons')}"
        )
        degraded = result.get("degraded_reasons") or []
        assert not any("Barrier logic gate" in r or "protection logic" in r.lower() for r in degraded), (
            f"Barrier reason still present after ack: {degraded}"
        )
        print("  PASS test_barrier_gate_ack_clears_soe_plc_block")

    def test_barrier_gate_ack_also_clears_analyst_decisions_required(self):
        """After ack, no barrier entry appears in analyst_decisions_required either."""
        o = make_orchestrator()
        card = make_clean_card(extra_analyst_review={"barrier_gate_degraded_acknowledged": True})
        result = o._compute_review_hooks(
            card,
            make_ok_output(),
            coverage_summary=make_coverage_with_soe_plc_violated(),
        )
        adr = result.get("analyst_decisions_required") or []
        assert not any("protection_logic" in r or "barrier" in r.lower() for r in adr), (
            f"Barrier entry still in analyst_decisions_required after ack: {adr}"
        )
        print("  PASS test_barrier_gate_ack_also_clears_analyst_decisions_required")

    def test_no_soe_plc_issue_still_writeback_ready(self):
        """Clean SOE/PLC pairing → not blocked by barrier gate logic."""
        o = make_orchestrator()
        result = o._compute_review_hooks(
            make_clean_card(),
            make_ok_output(),
            coverage_summary=make_coverage_clean(),
        )
        assert result["writeback_ready"] is True
        print("  PASS test_no_soe_plc_issue_still_writeback_ready")

    def test_barrier_gate_ack_false_still_blocks(self):
        """Explicit False acknowledgement still blocks — default is safe."""
        o = make_orchestrator()
        card = make_clean_card(extra_analyst_review={"barrier_gate_degraded_acknowledged": False})
        result = o._compute_review_hooks(
            card,
            make_ok_output(),
            coverage_summary=make_coverage_with_soe_plc_violated(),
        )
        assert result["writeback_ready"] is False
        print("  PASS test_barrier_gate_ack_false_still_blocks")


# ── Issue 5: _apply_fast_transient_attention_flags ────────────────────────────

def make_allen_map_with_causal_nodes(n_causal=2):
    return {
        "summary": {
            "total_nodes": n_causal + 1,
            "causal_nodes": n_causal,
            "contradiction_nodes": 0,
            "timeline_consistent": True,
        },
        "nodes": [],
    }


def make_allen_map_no_causal():
    return {
        "summary": {
            "total_nodes": 3,
            "causal_nodes": 0,
            "contradiction_nodes": 0,
        },
        "nodes": [],
    }


class TestFastTransientAttentionFlags:

    def _get_flags(self, rca_card):
        return (rca_card.get("executive_summary") or {}).get("analyst_attention_flags") or []

    def test_reactor_trip_with_causal_nodes_fires_flag(self):
        """reactor_trip event + causal Allen nodes → epsilon unreliability flag fires."""
        rca_card = {}
        event = {"event_type": "reactor_trip", "event_id": "E-001"}
        allen_map = make_allen_map_with_causal_nodes(n_causal=3)
        event_types = {"reactor_trip", "eccs_actuation", "turbine_trip", "loss_of_feedwater"}
        RCAReasoningOrchestrator._apply_fast_transient_attention_flags(
            rca_card, event, allen_map, event_types
        )
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        assert "reactor_trip" in flags[0]
        assert "epsilon" in flags[0].lower() or "temporal" in flags[0].lower()
        print("  PASS test_reactor_trip_with_causal_nodes_fires_flag")

    def test_eccs_actuation_fires_flag(self):
        """eccs_actuation event type also triggers the flag."""
        rca_card = {}
        event = {"event_type": "eccs_actuation"}
        allen_map = make_allen_map_with_causal_nodes(n_causal=1)
        event_types = {"reactor_trip", "eccs_actuation", "turbine_trip", "loss_of_feedwater"}
        RCAReasoningOrchestrator._apply_fast_transient_attention_flags(
            rca_card, event, allen_map, event_types
        )
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        assert "eccs_actuation" in flags[0]
        print("  PASS test_eccs_actuation_fires_flag")

    def test_non_fast_transient_event_no_flag(self):
        """A routine maintenance event does not trigger the flag."""
        rca_card = {}
        event = {"event_type": "equipment_failure"}
        allen_map = make_allen_map_with_causal_nodes(n_causal=5)
        event_types = {"reactor_trip", "eccs_actuation", "turbine_trip", "loss_of_feedwater"}
        RCAReasoningOrchestrator._apply_fast_transient_attention_flags(
            rca_card, event, allen_map, event_types
        )
        flags = self._get_flags(rca_card)
        assert flags == []
        print("  PASS test_non_fast_transient_event_no_flag")

    def test_fast_transient_no_causal_nodes_no_flag(self):
        """reactor_trip with zero causal Allen nodes → no flag (nothing temporal to warn about)."""
        rca_card = {}
        event = {"event_type": "reactor_trip"}
        allen_map = make_allen_map_no_causal()
        event_types = {"reactor_trip", "eccs_actuation"}
        RCAReasoningOrchestrator._apply_fast_transient_attention_flags(
            rca_card, event, allen_map, event_types
        )
        flags = self._get_flags(rca_card)
        assert flags == []
        print("  PASS test_fast_transient_no_causal_nodes_no_flag")

    def test_fast_transient_none_allen_map_no_flag(self):
        """reactor_trip with None Allen map → no flag (map not built)."""
        rca_card = {}
        event = {"event_type": "reactor_trip"}
        event_types = {"reactor_trip"}
        RCAReasoningOrchestrator._apply_fast_transient_attention_flags(
            rca_card, event, None, event_types
        )
        flags = self._get_flags(rca_card)
        assert flags == []
        print("  PASS test_fast_transient_none_allen_map_no_flag")

    def test_flag_not_duplicated_on_repeated_call(self):
        """Calling the method twice does not append the flag twice."""
        rca_card = {}
        event = {"event_type": "turbine_trip"}
        allen_map = make_allen_map_with_causal_nodes(n_causal=2)
        event_types = {"turbine_trip"}
        RCAReasoningOrchestrator._apply_fast_transient_attention_flags(
            rca_card, event, allen_map, event_types
        )
        RCAReasoningOrchestrator._apply_fast_transient_attention_flags(
            rca_card, event, allen_map, event_types
        )
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        print("  PASS test_flag_not_duplicated_on_repeated_call")

    def test_empty_event_type_no_flag(self):
        """Missing event_type → no flag."""
        rca_card = {}
        event = {}
        allen_map = make_allen_map_with_causal_nodes(n_causal=1)
        event_types = {"reactor_trip"}
        RCAReasoningOrchestrator._apply_fast_transient_attention_flags(
            rca_card, event, allen_map, event_types
        )
        flags = self._get_flags(rca_card)
        assert flags == []
        print("  PASS test_empty_event_type_no_flag")

    def test_custom_event_types_config(self):
        """A site-configured custom event type triggers the flag."""
        rca_card = {}
        event = {"event_type": "steam_generator_tube_rupture"}
        allen_map = make_allen_map_with_causal_nodes(n_causal=4)
        event_types = {"steam_generator_tube_rupture"}
        RCAReasoningOrchestrator._apply_fast_transient_attention_flags(
            rca_card, event, allen_map, event_types
        )
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        assert "steam_generator_tube_rupture" in flags[0]
        print("  PASS test_custom_event_types_config")


# ── Issue 11: _apply_category_l_floor_attention_flags ─────────────────────────

def make_candidates_with_l(l_score=0.10):
    """Return a causality_candidates dict with one L-category candidate at the given score."""
    return {
        "candidates": [
            {"candidate_id": "CAND-001", "primary_category": "A", "composite_score": 0.72},
            {"candidate_id": "CAND-002", "primary_category": "L", "composite_score": l_score},
        ],
        "category_coverage": {
            "L": {"status": "low_confidence"},
        },
        "recurrence_summary": {"candidate_count_with_recurrence": 1},
    }


def make_candidates_no_l():
    """No L-category candidate at all."""
    return {
        "candidates": [
            {"candidate_id": "CAND-001", "primary_category": "A", "composite_score": 0.72},
        ],
        "category_coverage": {
            "L": {"status": "no_supporting_data"},
        },
        "recurrence_summary": {"candidate_count_with_recurrence": 2},
    }


def make_candidates_no_recurrence():
    """No L candidate and no recurrence signal."""
    return {
        "candidates": [
            {"candidate_id": "CAND-001", "primary_category": "A", "composite_score": 0.72},
        ],
        "category_coverage": {
            "L": {"status": "no_supporting_data"},
        },
        "recurrence_summary": {"candidate_count_with_recurrence": 0},
    }


def make_cmms_with_open_crs(count=3):
    return {"recurrence_summary": {"open_cr_count": count}}


def make_cmms_no_open_crs():
    return {"recurrence_summary": {"open_cr_count": 0}}


class TestCategoryLFloorAttentionFlags:

    def _get_flags(self, rca_card):
        return (rca_card.get("executive_summary") or {}).get("analyst_attention_flags") or []

    def test_l_below_floor_with_recurrence_fires_flag(self):
        """L candidate below floor + candidate recurrence → flag fires."""
        rca_card = {}
        candidates = make_candidates_with_l(l_score=0.10)
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, candidates, make_cmms_no_open_crs(), 0.20
        )
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        assert "Category L" in flags[0]
        assert "0.20" in flags[0]
        print("  PASS test_l_below_floor_with_recurrence_fires_flag")

    def test_l_above_floor_no_flag(self):
        """L candidate above the floor → no flag regardless of recurrence."""
        rca_card = {}
        candidates = make_candidates_with_l(l_score=0.35)
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, candidates, make_cmms_with_open_crs(5), 0.20
        )
        flags = self._get_flags(rca_card)
        assert flags == []
        print("  PASS test_l_above_floor_no_flag")

    def test_l_at_exact_floor_no_flag(self):
        """L candidate at exactly the floor score → not below floor, no flag."""
        rca_card = {}
        candidates = make_candidates_with_l(l_score=0.20)
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, candidates, make_cmms_no_open_crs(), 0.20
        )
        flags = self._get_flags(rca_card)
        assert flags == []
        print("  PASS test_l_at_exact_floor_no_flag")

    def test_no_l_candidate_with_open_crs_fires_flag(self):
        """No L candidate at all + open CRs in CMMS → flag fires."""
        rca_card = {}
        candidates = make_candidates_no_l()
        candidates["recurrence_summary"]["candidate_count_with_recurrence"] = 0
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, candidates, make_cmms_with_open_crs(3), 0.20
        )
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        assert "Category L" in flags[0]
        assert "open CR" in flags[0] or "open_cr" in flags[0].lower()
        print("  PASS test_no_l_candidate_with_open_crs_fires_flag")

    def test_no_recurrence_no_flag(self):
        """No L candidate but also no recurrence signal → flag does not fire."""
        rca_card = {}
        candidates = make_candidates_no_recurrence()
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, candidates, make_cmms_no_open_crs(), 0.20
        )
        flags = self._get_flags(rca_card)
        assert flags == []
        print("  PASS test_no_recurrence_no_flag")

    def test_none_cmms_context_uses_candidate_recurrence_only(self):
        """cmms_context=None → flag still fires if candidate recurrence is present."""
        rca_card = {}
        candidates = make_candidates_no_l()
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, candidates, None, 0.20
        )
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        print("  PASS test_none_cmms_context_uses_candidate_recurrence_only")

    def test_none_candidates_no_recurrence_no_flag(self):
        """No candidates dict, no open CRs, no recurrence → no flag (graceful handling)."""
        rca_card = {}
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, None, make_cmms_no_open_crs(), 0.20
        )
        flags = self._get_flags(rca_card)
        assert flags == []
        print("  PASS test_none_candidates_no_recurrence_no_flag")

    def test_none_candidates_with_open_crs_fires_flag(self):
        """No candidates but open CRs present → flag fires (no L candidate despite recurrence)."""
        rca_card = {}
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, None, make_cmms_with_open_crs(5), 0.20
        )
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        assert "Category L" in flags[0]
        print("  PASS test_none_candidates_with_open_crs_fires_flag")

    def test_flag_not_duplicated_on_repeated_call(self):
        """Calling the method twice does not duplicate the flag."""
        rca_card = {}
        candidates = make_candidates_no_l()
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, candidates, make_cmms_with_open_crs(2), 0.20
        )
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, candidates, make_cmms_with_open_crs(2), 0.20
        )
        flags = self._get_flags(rca_card)
        assert len(flags) == 1
        print("  PASS test_flag_not_duplicated_on_repeated_call")

    def test_configurable_floor_respected(self):
        """A lower floor threshold (0.10) suppresses the flag when L score is 0.12."""
        rca_card = {}
        candidates = make_candidates_with_l(l_score=0.12)
        RCAReasoningOrchestrator._apply_category_l_floor_attention_flags(
            rca_card, candidates, make_cmms_with_open_crs(1), 0.10
        )
        flags = self._get_flags(rca_card)
        assert flags == []
        print("  PASS test_configurable_floor_respected")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Issue 3: barrier_gate_degraded_acknowledged ===")
    suite3 = TestBarrierGateDegradedAcknowledged()
    suite3.test_soe_plc_violated_blocks_writeback_by_default()
    suite3.test_soe_plc_warning_blocks_writeback_by_default()
    suite3.test_barrier_gate_ack_clears_soe_plc_block()
    suite3.test_barrier_gate_ack_also_clears_analyst_decisions_required()
    suite3.test_no_soe_plc_issue_still_writeback_ready()
    suite3.test_barrier_gate_ack_false_still_blocks()

    print("\n=== Issue 5: _apply_fast_transient_attention_flags ===")
    suite5 = TestFastTransientAttentionFlags()
    suite5.test_reactor_trip_with_causal_nodes_fires_flag()
    suite5.test_eccs_actuation_fires_flag()
    suite5.test_non_fast_transient_event_no_flag()
    suite5.test_fast_transient_no_causal_nodes_no_flag()
    suite5.test_fast_transient_none_allen_map_no_flag()
    suite5.test_flag_not_duplicated_on_repeated_call()
    suite5.test_empty_event_type_no_flag()
    suite5.test_custom_event_types_config()

    print("\n=== Issue 11: _apply_category_l_floor_attention_flags ===")
    suite11 = TestCategoryLFloorAttentionFlags()
    suite11.test_l_below_floor_with_recurrence_fires_flag()
    suite11.test_l_above_floor_no_flag()
    suite11.test_l_at_exact_floor_no_flag()
    suite11.test_no_l_candidate_with_open_crs_fires_flag()
    suite11.test_no_recurrence_no_flag()
    suite11.test_none_cmms_context_uses_candidate_recurrence_only()
    suite11.test_none_candidates_no_recurrence_no_flag()
    suite11.test_none_candidates_with_open_crs_fires_flag()
    suite11.test_flag_not_duplicated_on_repeated_call()
    suite11.test_configurable_floor_respected()

    print("\nAll Phase 1 tests passed.")
