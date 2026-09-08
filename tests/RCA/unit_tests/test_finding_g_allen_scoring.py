"""
test_finding_g_allen_scoring.py — Finding G: Allen base score blending into causality scoring

Covers:
- _build_allen_component_index: causal index, follows set, SOE clock-sync discount, multi-node max
- _apply_allen_temporal_blend: boost, no-change on no-match, contradiction flag,
  composite delta proportional to weight, fields stored, caps
- refine_with_evidence: passes allen_relation_map through to blend
- Orchestrator: pre_refine_allen_map threaded into refine_kwargs and stage_g

Run:  pytest test_finding_g_allen_scoring.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
from typing import Optional

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32
from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator

ENGINE = RuleBasedCausalityEngineV32

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(
    node_type: str,
    component_id: Optional[str],
    relation: str,
    base_score: float,
    *,
    causal_candidate: bool = True,
) -> dict:
    return {
        "node_id": f"{node_type}::{component_id}",
        "node_type": node_type,
        "source_id": f"src-{component_id}",
        "component_id": component_id,
        "interval_start": "2024-01-01T10:00:00Z",
        "is_point_event": False,
        "allen_relation_to_event": relation,
        "allen_base_score": base_score,
        "causal_candidate": causal_candidate,
    }


def _allen_map(nodes: list, *, soe_clock_ok: bool = True) -> dict:
    return {
        "event_id": "EVT-001",
        "generated_at": "2024-01-01T12:00:00Z",
        "event_interval": {"start": "2024-01-01T12:00:00Z"},
        "quality_flags": {
            "soe_clock_sync_ok": soe_clock_ok,
            "alarm_clock_sync_ok": True,
        },
        "summary": {
            "total_nodes": len(nodes),
            "node_type_counts": {},
            "causal_nodes": sum(1 for n in nodes if n.get("causal_candidate")),
            "contradiction_nodes": 0,
            "timeline_consistent": True,
        },
        "nodes": nodes,
    }


def _candidate(
    component_id: str,
    temporal: float = 0.20,
    composite: float = 0.50,
    quality_multiplier: float = 1.0,
) -> dict:
    """Build a minimal candidate dict with composite_raw pre-set."""
    return {
        "candidate_id": f"FM::{component_id}",
        "component_id": component_id,
        "hypothesis_type": "failure_mode",
        "composite_score": composite,
        "quality_multiplier": quality_multiplier,
        "scores": {
            "temporal": temporal,
            "structural": 0.60,
            "evidence": 0.40,
            "telemetry": 0.30,
            "governance": 0.20,
            "composite_raw": composite / quality_multiplier if quality_multiplier else composite,
        },
        "temporal_evidence": {},
    }


_DEFAULT_WEIGHTS = {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20, "evidence": 0.20, "governance": 0.10}


# ===========================================================================
# _build_allen_component_index
# ===========================================================================

def test_index_none_map_returns_empty():
    scores, relation, follows = ENGINE._build_allen_component_index(None)
    assert scores == {} and relation == {} and len(follows) == 0


def test_index_empty_nodes_returns_empty():
    scores, relation, follows = ENGINE._build_allen_component_index(_allen_map([]))
    assert scores == {} and relation == {} and len(follows) == 0


def test_index_causal_node_indexed():
    nodes = [_node("anomaly", "COMP-A", "precedes", 0.75)]
    scores, relation, follows = ENGINE._build_allen_component_index(_allen_map(nodes))
    assert "COMP-A" in scores
    assert abs(scores["COMP-A"] - 0.75) < 1e-6
    assert relation["COMP-A"] == "precedes"


def test_index_follows_node_added_to_follow_ids():
    nodes = [_node("alarm", "COMP-B", "follows", 0.10, causal_candidate=False)]
    scores, relation, follows = ENGINE._build_allen_component_index(_allen_map(nodes))
    assert "COMP-B" in follows
    assert "COMP-B" not in scores  # follows nodes are NOT causal


def test_index_multiple_nodes_same_component_takes_max():
    """Phase C: only anomaly contributes — alarm score ignored; anomaly wins."""
    nodes = [
        _node("anomaly", "COMP-C", "precedes", 0.60),
        _node("alarm",   "COMP-C", "overlaps", 0.85),
    ]
    scores, relation, follows = ENGINE._build_allen_component_index(_allen_map(nodes))
    assert abs(scores["COMP-C"] - 0.60) < 1e-6   # anomaly only; alarm excluded
    assert relation["COMP-C"] == "precedes"


def test_index_soe_node_excluded_from_causal_scores():
    """Phase C: SOE nodes do not contribute to causal_scores regardless of clock sync."""
    nodes = [_node("soe_record", "COMP-D", "precedes", 1.00)]
    scores, _, _ = ENGINE._build_allen_component_index(_allen_map(nodes, soe_clock_ok=False))
    assert "COMP-D" not in scores


def test_index_alarm_node_excluded_from_causal_scores():
    """Phase C: alarm nodes do not contribute to causal_scores."""
    nodes = [_node("alarm", "COMP-E", "precedes", 1.00)]
    scores, _, _ = ENGINE._build_allen_component_index(_allen_map(nodes, soe_clock_ok=False))
    assert "COMP-E" not in scores


def test_index_null_component_id_skipped():
    nodes = [_node("anomaly", None, "precedes", 0.90)]
    scores, _, _ = ENGINE._build_allen_component_index(_allen_map(nodes))
    assert scores == {}


def test_index_non_causal_node_not_in_causal_scores():
    nodes = [_node("anomaly", "COMP-F", "during", 0.30, causal_candidate=False)]
    scores, _, _ = ENGINE._build_allen_component_index(_allen_map(nodes))
    assert "COMP-F" not in scores


# ===========================================================================
# _apply_allen_temporal_blend
# ===========================================================================

def test_blend_raises_low_tskr_temporal():
    c = _candidate("COMP-A", temporal=0.20, composite=0.50)
    # composite_raw = 0.50 with qm=1.0
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.80}, {"COMP-A": "precedes"}, set(), _DEFAULT_WEIGHTS
    )
    assert c["scores"]["allen_blend_applied"] is True
    new_temporal = c["scores"]["temporal"]
    # 0.75*0.20 + 0.25*0.80 = 0.15 + 0.20 = 0.35
    assert abs(new_temporal - 0.35) < 1e-5


def test_blend_composite_delta_proportional_to_weight():
    c = _candidate("COMP-A", temporal=0.20, composite=0.50)
    old_raw = c["scores"]["composite_raw"]
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.80}, {"COMP-A": "precedes"}, set(), _DEFAULT_WEIGHTS
    )
    new_raw = c["scores"]["composite_raw"]
    temporal_delta = 0.35 - 0.20  # new - old temporal
    expected_raw_delta = _DEFAULT_WEIGHTS["temporal"] * temporal_delta  # 0.20 * 0.15 = 0.03
    assert abs((new_raw - old_raw) - expected_raw_delta) < 1e-5


def test_blend_no_match_leaves_temporal_unchanged():
    c = _candidate("COMP-X", temporal=0.50, composite=0.60)
    orig_temporal = c["scores"]["temporal"]
    orig_composite = c["composite_score"]
    ENGINE._apply_allen_temporal_blend(
        c, {}, {}, set(), _DEFAULT_WEIGHTS
    )
    assert c["scores"]["allen_blend_applied"] is False
    assert c["scores"]["temporal"] == orig_temporal
    assert c["composite_score"] == orig_composite


def test_blend_allen_can_lower_temporal():
    """Allen score below current temporal should lower temporal (true weighted blend).

    Updated 2026-05-23: the old max(old, blend) clamp was removed so that weak
    Allen relations (low allen_base_score) correctly reduce the temporal score,
    enabling discrimination between OVERLAPS and PRECEDES candidates.
    """
    c = _candidate("COMP-A", temporal=0.90, composite=0.80)
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.40}, {"COMP-A": "precedes"}, set(), _DEFAULT_WEIGHTS
    )
    # 0.75*0.90 + 0.25*0.40 = 0.675 + 0.10 = 0.775 — blend lowers temporal
    expected = round(0.75 * 0.90 + 0.25 * 0.40, 6)
    assert abs(c["scores"]["temporal"] - expected) < 1e-6, (
        f"temporal={c['scores']['temporal']:.6f}, expected {expected:.6f}. "
        "Allen blend should produce a true weighted average, not be clamped."
    )
    assert c["scores"]["allen_blend_applied"] is True


def test_blend_follows_sets_temporal_contradiction():
    c = _candidate("COMP-B", temporal=0.50, composite=0.60)
    ENGINE._apply_allen_temporal_blend(
        c, {}, {}, {"COMP-B"}, _DEFAULT_WEIGHTS
    )
    assert c["temporal_evidence"]["temporal_contradiction"] is True
    assert c["scores"]["allen_relation"] == "follows"
    assert c["scores"]["allen_blend_applied"] is False


def test_blend_composite_score_updates_with_quality_multiplier():
    c = _candidate("COMP-A", temporal=0.20, composite=0.40, quality_multiplier=0.80)
    # composite_raw = 0.40 / 0.80 = 0.50
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.80}, {"COMP-A": "overlaps"}, set(), _DEFAULT_WEIGHTS
    )
    new_raw = c["scores"]["composite_raw"]
    assert c["composite_score"] == round(min(1.0, max(0.0, new_raw * 0.80)), 6)


def test_blend_allen_temporal_score_field_stored():
    c = _candidate("COMP-A", temporal=0.20)
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.75}, {"COMP-A": "contains"}, set(), _DEFAULT_WEIGHTS
    )
    assert abs(c["scores"]["allen_temporal_score"] - 0.75) < 1e-6


def test_blend_allen_relation_field_stored():
    c = _candidate("COMP-A", temporal=0.20)
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.75}, {"COMP-A": "contains"}, set(), _DEFAULT_WEIGHTS
    )
    assert c["scores"]["allen_relation"] == "contains"


def test_blend_null_fields_on_no_match():
    c = _candidate("COMP-X", temporal=0.50)
    ENGINE._apply_allen_temporal_blend(c, {}, {}, set(), _DEFAULT_WEIGHTS)
    assert c["scores"]["allen_temporal_score"] is None
    assert c["scores"]["allen_relation"] is None


def test_blend_composite_raw_does_not_exceed_one():
    c = _candidate("COMP-A", temporal=0.99, composite=0.99)
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 1.00}, {"COMP-A": "precedes"}, set(), _DEFAULT_WEIGHTS
    )
    assert c["scores"]["composite_raw"] <= 1.0
    assert c["composite_score"] <= 1.0


def test_blend_past_event_candidate_matched_by_component():
    c = _candidate("COMP-A", temporal=0.25, composite=0.45)
    c["hypothesis_type"] = "past_event"
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.80}, {"COMP-A": "precedes"}, set(), _DEFAULT_WEIGHTS
    )
    assert c["scores"]["allen_blend_applied"] is True


def test_blend_alarm_node_excluded_from_causal_scores():
    """Phase C: alarm node does not populate causal_scores; no Allen blend applied."""
    nodes = [_node("alarm", "COMP-G", "overlaps", 0.90)]
    scores, _, _ = ENGINE._build_allen_component_index(_allen_map(nodes))
    assert "COMP-G" not in scores


def test_blend_soe_node_excluded_no_blend_applied():
    """Phase C: SOE node absent from causal_scores; _apply_allen_temporal_blend skips blend."""
    nodes = [_node("soe_record", "COMP-H", "precedes", 0.90)]
    scores, rel, _ = ENGINE._build_allen_component_index(_allen_map(nodes, soe_clock_ok=False))
    assert "COMP-H" not in scores
    c = _candidate("COMP-H", temporal=0.20)
    ENGINE._apply_allen_temporal_blend(c, scores, rel, set(), _DEFAULT_WEIGHTS)
    assert c["scores"]["allen_blend_applied"] is False
    assert abs(c["scores"]["temporal"] - 0.20) < 1e-6  # unchanged


# ===========================================================================
# refine_with_evidence integration
# ===========================================================================

def _make_candidates(component_id: str, temporal: float = 0.20, composite: float = 0.50) -> dict:
    """Minimal causality_candidates payload for refine_with_evidence.

    Uses structural=0.80 so that after _apply_uncertainty_propagation (quality floor 0.70)
    the composite stays above minimum_composite_threshold=0.30, ensuring the candidate is
    retained (not compacted) and its scores dict stays intact for Allen field assertions.
    """
    cand = _candidate(component_id, temporal=temporal, composite=composite)
    # Override structural high enough to survive the quality-multiplier floor
    cand["scores"]["structural"] = 0.80
    cand["scores"]["evidence"] = 0.40   # >= minimum_evidence_threshold (0.35)
    cand["primary_causal_category"] = "A"
    cand["chain_position"] = "proximate"
    cand["canonical_tuple"] = {}
    cand["canonical_candidate_key"] = f"A::proximate::{component_id}"
    return {
        "event_id": "EVT-001",
        "candidates": [cand],
        "filtered_out_candidates": [],
        "event_analogs": [],
        "summary": {},
        "category_coverage": {},
        "applicability_assessment": {},
    }


def _make_evidence_bundle() -> dict:
    return {
        "per_candidate_summary": [],
        "retrieval_run_id": "RUN-001",
    }


def test_refine_applies_allen_blend_when_map_provided():
    from orchestrators.causality_engine_v32 import CausalityEngineConfigV32
    # Use a low evidence threshold so the candidate survives the refine pass
    # (with no retrieved docs, refined evidence = 0.30 × prior ≈ 0.12).
    engine = RuleBasedCausalityEngineV32(CausalityEngineConfigV32(
        minimum_evidence_threshold=0.05, minimum_composite_threshold=0.20))
    nodes = [_node("anomaly", "COMP-A", "precedes", 0.85)]
    allen_map = _allen_map(nodes)
    candidates = _make_candidates("COMP-A", temporal=0.20, composite=0.50)
    result = engine.refine_with_evidence(
        causality_candidates=candidates,
        evidence_bundle=_make_evidence_bundle(),
        allen_relation_map=allen_map,
    )
    all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
    assert len(all_cands) > 0
    scores = all_cands[0].get("scores") or {}
    assert scores.get("allen_blend_applied") is True
    assert scores.get("allen_temporal_score") is not None


def test_refine_no_allen_map_leaves_blend_fields_false():
    from orchestrators.causality_engine_v32 import CausalityEngineConfigV32
    engine = RuleBasedCausalityEngineV32(CausalityEngineConfigV32(
        minimum_evidence_threshold=0.05, minimum_composite_threshold=0.20))
    candidates = _make_candidates("COMP-A", temporal=0.50)
    result = engine.refine_with_evidence(
        causality_candidates=candidates,
        evidence_bundle=_make_evidence_bundle(),
        allen_relation_map=None,
    )
    all_cands = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
    scores = all_cands[0].get("scores") or {}
    assert scores.get("allen_blend_applied") is False


def test_refine_allen_follows_blocks_candidate_via_timeline_gate():
    """Allen 'follows' relation sets temporal_contradiction, which the timeline gate
    detects and records in hard_gates and ruleout (preserved in compacted filtered candidate)."""
    from orchestrators.causality_engine_v32 import CausalityEngineConfigV32
    engine = RuleBasedCausalityEngineV32(CausalityEngineConfigV32(
        minimum_evidence_threshold=0.05, minimum_composite_threshold=0.20))
    nodes = [_node("anomaly", "COMP-B", "follows", 0.10, causal_candidate=False)]
    allen_map = _allen_map(nodes)
    candidates = _make_candidates("COMP-B", temporal=0.60, composite=0.65)
    result = engine.refine_with_evidence(
        causality_candidates=candidates,
        evidence_bundle=_make_evidence_bundle(),
        allen_relation_map=allen_map,
    )
    all_candidates = (result.get("candidates") or []) + (result.get("filtered_out_candidates") or [])
    comp_b = next((c for c in all_candidates if c.get("component_id") == "COMP-B"), None)
    assert comp_b is not None
    # The timeline gate detects temporal_contradiction and marks the candidate as ruled-out.
    # The compacted candidate preserves hard_gates and ruleout fields.
    hard_gates = comp_b.get("hard_gates") or {}
    tl_gate = hard_gates.get("timeline_consistency") or {}
    ruleout = comp_b.get("ruleout") or {}
    gate_failed = (tl_gate.get("passed") is False) or (ruleout.get("reason_code") == "timeline_inconsistent")
    assert gate_failed, (
        f"Expected timeline gate to fail for COMP-B (Allen 'follows'). "
        f"hard_gates={hard_gates}, ruleout={ruleout}"
    )


# ===========================================================================
# Phase 4a — temporal_score_quality (Issue 1 / D1)
# ===========================================================================

def test_tsq_full_allen_when_component_matched():
    """Causal Allen node found for component → temporal_score_quality = 'full_allen'."""
    c = _candidate("COMP-A", temporal=0.20)
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.80}, {"COMP-A": "precedes"}, set(), _DEFAULT_WEIGHTS
    )
    assert c["scores"]["temporal_score_quality"] == "full_allen"


def test_tsq_proxy_when_no_component_match():
    """No Allen node for component → temporal_score_quality = 'proxy'."""
    c = _candidate("COMP-Z", temporal=0.50)
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.80}, {"COMP-A": "precedes"}, set(), _DEFAULT_WEIGHTS
    )
    assert c["scores"]["temporal_score_quality"] == "proxy"


def test_tsq_proxy_when_contradiction_follows():
    """'follows' contradiction → blend not applied → temporal_score_quality = 'proxy'."""
    c = _candidate("COMP-B", temporal=0.30)
    ENGINE._apply_allen_temporal_blend(
        c, {}, {}, {"COMP-B"}, _DEFAULT_WEIGHTS
    )
    assert c["scores"]["allen_blend_applied"] is False
    assert c["scores"]["temporal_score_quality"] == "proxy"


def test_tsq_proxy_when_no_allen_map():
    """Empty causal_scores (no Allen map provided) → temporal_score_quality = 'proxy'."""
    c = _candidate("COMP-A", temporal=0.40)
    ENGINE._apply_allen_temporal_blend(c, {}, {}, set(), _DEFAULT_WEIGHTS)
    assert c["scores"]["temporal_score_quality"] == "proxy"


def test_tsq_full_allen_even_when_blend_clamped():
    """Allen score equals existing temporal → score unchanged but quality = 'full_allen'."""
    # Blend formula: 0.75*0.80 + 0.25*0.80 = 0.80 — no delta, but component was found.
    c = _candidate("COMP-A", temporal=0.80)
    ENGINE._apply_allen_temporal_blend(
        c, {"COMP-A": 0.80}, {"COMP-A": "precedes"}, set(), _DEFAULT_WEIGHTS
    )
    assert c["scores"]["allen_blend_applied"] is True
    assert c["scores"]["temporal_score_quality"] == "full_allen"


def test_tsq_per_component_not_per_run():
    """Two candidates in same run: one matches Allen index, one does not.
    quality field is independent per candidate."""
    c_match = _candidate("COMP-A", temporal=0.20)
    c_miss = _candidate("COMP-X", temporal=0.20)
    causal_scores = {"COMP-A": 0.80}
    causal_rel = {"COMP-A": "precedes"}
    ENGINE._apply_allen_temporal_blend(c_match, causal_scores, causal_rel, set(), _DEFAULT_WEIGHTS)
    ENGINE._apply_allen_temporal_blend(c_miss, causal_scores, causal_rel, set(), _DEFAULT_WEIGHTS)
    assert c_match["scores"]["temporal_score_quality"] == "full_allen"
    assert c_miss["scores"]["temporal_score_quality"] == "proxy"


def test_tsq_proxy_when_soe_node_excluded_from_index():
    """SOE node is excluded from causal_scores by _build_allen_component_index.
    The candidate for that component therefore gets temporal_score_quality = 'proxy'."""
    nodes = [_node("soe_record", "COMP-H", "precedes", 0.90)]
    scores, rel, _ = ENGINE._build_allen_component_index(_allen_map(nodes))
    assert "COMP-H" not in scores
    c = _candidate("COMP-H", temporal=0.20)
    ENGINE._apply_allen_temporal_blend(c, scores, rel, set(), _DEFAULT_WEIGHTS)
    assert c["scores"]["temporal_score_quality"] == "proxy"
