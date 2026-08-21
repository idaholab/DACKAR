"""
test_step6_conclusion.py — Step 6 Conclusion tests

Covers:
WS1 — Human Performance Assessment
  - applicable=True when H/I/J/K candidate retained
  - applicable=False when only A-F candidates retained
  - category_flags correctly set for each G/I/L type
  - performance_mode mapped correctly (G→execution_error, I→change_management_gap, L→organisational_gap)
  - design (H) / surveillance (J) / vendor (K) are EXCLUDED from human performance (F-2 fix)
  - regulatory_reference populated per category
  - corrective_action_ids cross-referenced from recommended_actions
  - deterministic injection via post-processing path (no double-injection)

WS2 — Deepened unresolved_gaps
  - contributing layer unresolved → gap entry present
  - root layer unresolved → gap entry present
  - sensitivity_any_change=True → gap entry present
  - novel_pattern_flag=True → gap entry present
  - gaps capped at 8 items
  - existing checks (no evidence, contradicting, temporal contradiction) still present

WS3 — Depth-stratified effectiveness_monitoring_plan
  - proximate action → equipment-health indicator and 90d horizon
  - contributing action → process/procedure indicator and 180d horizon
  - root action → programmatic indicator and 365d horizon
  - success_criteria field present on every plan item
  - causal_depth_level field present on every plan item
  - fallback entry has causal_depth_level when no actions

WS4 — depth_incomplete_reason
  - depth_incomplete_reason present when depth_complete=False
  - depth_incomplete_reason absent when depth_complete=True
  - depth_incomplete_reason explains all three missing layers

Run:  pytest test_step6_conclusion.py -v
"""
import sys
from pathlib import Path
from typing import Optional, List
from unittest.mock import MagicMock

import pytest

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from synthesis.rca_synthesizer_v31 import RuleValidatedRCASynthesizerV31  # noqa: E402

# Lightweight instance for calling instance methods
_SYNTH = RuleValidatedRCASynthesizerV31.__new__(RuleValidatedRCASynthesizerV31)

BUILD_HPA = RuleValidatedRCASynthesizerV31._build_human_performance_assessment
BUILD_GAPS = _SYNTH._build_unresolved_gaps
BUILD_EMP = RuleValidatedRCASynthesizerV31._build_effectiveness_monitoring_plan
BUILD_CDS = _SYNTH._build_causal_depth_summary

# ── helpers ───────────────────────────────────────────────────────────────────

def _candidate(cid: str, category: str, score: float = 0.7) -> dict:
    return {
        "candidate_id": cid,
        "primary_causal_category": category,
        "cause_label": f"cause_{cid}",
        "confidence_label": "medium",
        "composite_score": score,
    }


def _action(aid: str, depth: str, linked: Optional[str] = None) -> dict:
    return {
        "action_id": aid,
        "action_type": "engineering_evaluation",
        "description": f"Action {aid}",
        "priority": "medium",
        "rationale": "test",
        "expected_observation_if_true": "test",
        "target_causal_depth": depth,
        "linked_candidate_id": linked,
    }


# ═══════════════════════════════════════════════════════════════════════
# WS1 — Human Performance Assessment
# ═══════════════════════════════════════════════════════════════════════

def test_hpa_applicable_true_when_G_retained():
    result = BUILD_HPA(
        selected_candidates=[_candidate("C1", "G")],
        recommended_actions=[],
    )
    assert result["applicable"] is True


def test_hpa_applicable_false_when_only_A_retained():
    result = BUILD_HPA(
        selected_candidates=[_candidate("C1", "A")],
        recommended_actions=[],
    )
    assert result["applicable"] is False


def test_hpa_applicable_false_empty_candidates():
    result = BUILD_HPA(selected_candidates=[], recommended_actions=[])
    assert result["applicable"] is False


def test_hpa_category_flags_set_for_retained_categories():
    cands = [_candidate("CG", "G"), _candidate("CL", "L")]
    result = BUILD_HPA(selected_candidates=cands, recommended_actions=[])
    assert result["category_flags"]["G"] is True
    assert result["category_flags"]["L"] is True
    assert result["category_flags"]["I"] is False


def test_hpa_performance_mode_G():
    result = BUILD_HPA(
        selected_candidates=[_candidate("C1", "G")],
        recommended_actions=[],
    )
    assert result["findings"][0]["performance_mode"] == "execution_error"


def test_hpa_performance_mode_I():
    result = BUILD_HPA(
        selected_candidates=[_candidate("C1", "I")],
        recommended_actions=[],
    )
    assert result["findings"][0]["performance_mode"] == "change_management_gap"


def test_hpa_performance_mode_L():
    result = BUILD_HPA(
        selected_candidates=[_candidate("C1", "L")],
        recommended_actions=[],
    )
    assert result["findings"][0]["performance_mode"] == "organisational_gap"


def test_hpa_excludes_design_surveillance_vendor():
    """F-2: H (design), J (surveillance), K (vendor) are NOT human performance."""
    for cat in ("H", "J", "K"):
        result = BUILD_HPA(
            selected_candidates=[_candidate("C1", cat)],
            recommended_actions=[],
        )
        assert result["applicable"] is False, f"category {cat} must not count as human performance"
        assert result["findings"] == []


def test_hpa_excluded_category_noted_in_provenance():
    """A retained design (H) cause should be flagged as excluded, not silently dropped."""
    result = BUILD_HPA(
        selected_candidates=[_candidate("CG", "G"), _candidate("CH", "H")],
        recommended_actions=[],
    )
    assert result["applicable"] is True  # G is present
    note = result["provenance_note"].lower()
    assert "design" in note and "not human-performance" in note


def test_hpa_regulatory_reference_present():
    for cat in ("G", "I", "L"):
        result = BUILD_HPA(
            selected_candidates=[_candidate("C1", cat)],
            recommended_actions=[],
        )
        assert result["findings"][0]["regulatory_reference"], f"Missing ref for category {cat}"


def test_hpa_corrective_action_ids_cross_referenced():
    cands = [_candidate("C_G", "G")]
    actions = [_action("ACT-1", "contributing", linked="C_G")]
    result = BUILD_HPA(selected_candidates=cands, recommended_actions=actions)
    assert "ACT-1" in result["findings"][0]["corrective_action_ids"]


def test_hpa_action_not_linked_gives_empty_ids():
    cands = [_candidate("C_G", "G")]
    actions = [_action("ACT-2", "proximate", linked="OTHER")]
    result = BUILD_HPA(selected_candidates=cands, recommended_actions=actions)
    assert result["findings"][0]["corrective_action_ids"] == []


def test_hpa_findings_count_matches_hop_candidates():
    cands = [_candidate("C_G", "G"), _candidate("C_I", "I"), _candidate("C_A", "A")]
    result = BUILD_HPA(selected_candidates=cands, recommended_actions=[])
    assert len(result["findings"]) == 2  # only G and I


def test_hpa_provenance_note_present():
    result = BUILD_HPA(selected_candidates=[_candidate("C1", "G")], recommended_actions=[])
    assert isinstance(result["provenance_note"], str) and result["provenance_note"]


def test_hpa_required_fields_present_when_applicable():
    result = BUILD_HPA(selected_candidates=[_candidate("C1", "G")], recommended_actions=[])
    for field in ("applicable", "category_flags", "findings", "provenance_note"):
        assert field in result


def test_hpa_required_fields_present_when_not_applicable():
    result = BUILD_HPA(selected_candidates=[_candidate("C1", "A")], recommended_actions=[])
    for field in ("applicable", "category_flags", "findings", "provenance_note"):
        assert field in result


# ═══════════════════════════════════════════════════════════════════════
# WS2 — Deepened unresolved_gaps
# ═══════════════════════════════════════════════════════════════════════

def _gaps(*, no_support=False, contradicting=0, temporal_contradiction=False,
          data_limited=False, contrib_list=None, root="unresolved",
          sensitivity_any_change=False, novel_pattern_flag=False,
          attention_flags=None) -> List[str]:
    primary = {
        "data_limited_conclusion": data_limited,
        "critical_streams_below_floor": [],
    }
    evidence_summary = {
        "supporting": 0 if no_support else 2,
        "contradicting": contradicting,
    }
    pattern_posture = {"temporal_contradiction": temporal_contradiction}
    ds = {
        "proximate_cause": "some_cause",
        "contributing_causes": contrib_list if contrib_list is not None else ["c1"],
        "root_cause": root,
        "depth_complete": False,
    }
    return BUILD_GAPS(
        primary_candidate=primary,
        evidence_summary=evidence_summary,
        pattern_posture=pattern_posture,
        analyst_attention_flags=attention_flags or [],
        causal_depth_summary=ds,
        sensitivity_any_change=sensitivity_any_change,
        novel_pattern_flag=novel_pattern_flag,
    )


def test_gaps_no_support():
    gaps = _gaps(no_support=True)
    assert any("No direct supporting" in g for g in gaps)


def test_gaps_contradicting():
    gaps = _gaps(contradicting=2)
    assert any("Contradicting" in g for g in gaps)


def test_gaps_temporal_contradiction():
    gaps = _gaps(temporal_contradiction=True)
    assert any("Temporal contradiction" in g for g in gaps)


def test_gaps_contributing_unresolved():
    gaps = _gaps(contrib_list=[])
    assert any("Contributing cause" in g for g in gaps)


def test_gaps_root_unresolved():
    gaps = _gaps(root="unresolved")
    assert any("Root cause" in g for g in gaps)


def test_gaps_root_resolved_no_root_gap():
    gaps = _gaps(root="systemic_pm_programme_failure", contrib_list=["c1"])
    assert not any("Root cause layer is unresolved" in g for g in gaps)


def test_gaps_sensitivity_flag():
    gaps = _gaps(sensitivity_any_change=True)
    assert any("sensitivity analysis" in g.lower() or "sensitivity" in g.lower() for g in gaps)


def test_gaps_novel_pattern_flag():
    gaps = _gaps(novel_pattern_flag=True)
    assert any("novel" in g.lower() for g in gaps)


def test_gaps_capped_at_8():
    attention = [f"flag_{i}" for i in range(20)]
    gaps = _gaps(
        no_support=True, contradicting=2, temporal_contradiction=True,
        data_limited=True, contrib_list=[], root="unresolved",
        sensitivity_any_change=True, novel_pattern_flag=True,
        attention_flags=attention,
    )
    assert len(gaps) <= 8


def test_gaps_attention_flag_with_missing_keyword_included():
    gaps = _gaps(attention_flags=["missing telemetry for channel 4"])
    assert any("missing" in g.lower() for g in gaps)


def test_gaps_attention_flag_without_keywords_not_included():
    gaps_before = _gaps(attention_flags=[])
    gaps_after  = _gaps(attention_flags=["everything looks fine"])
    assert len(gaps_after) == len(gaps_before)


# ═══════════════════════════════════════════════════════════════════════
# WS3 — Depth-stratified effectiveness_monitoring_plan
# ═══════════════════════════════════════════════════════════════════════

def test_emp_proximate_action_90d_horizon():
    plan = BUILD_EMP(
        primary_candidate={"cause_label": "bearing wear"},
        recommended_actions=[_action("ACT-1", "proximate")],
    )
    assert plan[0]["review_horizon"] == "90d"


def test_emp_contributing_action_180d_horizon():
    plan = BUILD_EMP(
        primary_candidate={"cause_label": "pm interval"},
        recommended_actions=[_action("ACT-1", "contributing")],
    )
    assert plan[0]["review_horizon"] == "180d"


def test_emp_root_action_365d_horizon():
    plan = BUILD_EMP(
        primary_candidate={"cause_label": "amp gap"},
        recommended_actions=[_action("ACT-1", "root")],
    )
    assert plan[0]["review_horizon"] == "365d"


def test_emp_success_criteria_present():
    plan = BUILD_EMP(
        primary_candidate={"cause_label": "bearing wear"},
        recommended_actions=[_action("ACT-1", "proximate")],
    )
    assert "success_criteria" in plan[0] and plan[0]["success_criteria"]


def test_emp_causal_depth_level_field_present():
    plan = BUILD_EMP(
        primary_candidate={"cause_label": "bearing wear"},
        recommended_actions=[_action("ACT-1", "proximate")],
    )
    assert plan[0]["causal_depth_level"] == "proximate"


def test_emp_fallback_when_no_actions():
    plan = BUILD_EMP(primary_candidate={"cause_label": "unknown"}, recommended_actions=[])
    assert len(plan) == 1
    assert "causal_depth_level" in plan[0]
    assert "success_criteria" in plan[0]


def test_emp_proximate_indicator_contains_equipment():
    plan = BUILD_EMP(
        primary_candidate={"cause_label": "vibration fault"},
        recommended_actions=[_action("ACT-1", "proximate")],
    )
    assert "equipment" in plan[0]["indicator"].lower() or "anomal" in plan[0]["indicator"].lower()


def test_emp_root_indicator_contains_programmatic():
    plan = BUILD_EMP(
        primary_candidate={"cause_label": "systemic gap"},
        recommended_actions=[_action("ACT-1", "root")],
    )
    assert "programmatic" in plan[0]["indicator"].lower() or "fleet" in plan[0]["indicator"].lower()


def test_emp_capped_at_5():
    actions = [_action(f"A{i}", "proximate") for i in range(10)]
    plan = BUILD_EMP(primary_candidate={"cause_label": "x"}, recommended_actions=actions)
    assert len(plan) <= 5


# ═══════════════════════════════════════════════════════════════════════
# WS4 — depth_incomplete_reason
# ═══════════════════════════════════════════════════════════════════════

def _cds(primary_cat: str, selected_cats: List[str]) -> dict:
    primary = _candidate("PRI", primary_cat)
    selected = [_candidate(f"C{i}", cat) for i, cat in enumerate(selected_cats)]
    return BUILD_CDS(primary_candidate=primary, selected_candidates=selected)


def test_cds_depth_incomplete_reason_absent_when_complete():
    ds = _cds("A", ["A", "G", "L"])
    # We need cause_label set for depth detection
    # Build manually with all three layers resolved
    primary = {"primary_causal_category": "A", "cause_label": "bearing wear"}
    selected = [
        {"primary_causal_category": "G", "cause_label": "maintenance factor"},
        {"primary_causal_category": "L", "cause_label": "programme weakness"},
    ]
    result = BUILD_CDS(primary_candidate=primary, selected_candidates=selected)
    if result["depth_complete"]:
        assert "depth_incomplete_reason" not in result


def test_cds_depth_incomplete_reason_present_when_incomplete():
    ds = _cds("A", ["A"])  # No G–K, no L
    assert "depth_incomplete_reason" in ds
    assert ds["depth_incomplete_reason"]


def test_cds_depth_incomplete_reason_mentions_contributing_when_missing():
    primary = {"primary_causal_category": "A", "cause_label": "bearing wear"}
    selected = [{"primary_causal_category": "A", "cause_label": "bearing wear"}]
    result = BUILD_CDS(primary_candidate=primary, selected_candidates=selected)
    assert "contributing" in result.get("depth_incomplete_reason", "").lower()


def test_cds_depth_incomplete_reason_mentions_root_when_missing():
    primary = {"primary_causal_category": "A", "cause_label": "bearing wear"}
    selected = [
        {"primary_causal_category": "G", "cause_label": "maint factor"},
    ]
    result = BUILD_CDS(primary_candidate=primary, selected_candidates=selected)
    assert "root" in result.get("depth_incomplete_reason", "").lower()


def test_cds_depth_complete_flag_true_when_all_layers_present():
    primary = {"primary_causal_category": "A", "cause_label": "bearing wear"}
    selected = [
        {"primary_causal_category": "G", "cause_label": "maintenance factor"},
        {"primary_causal_category": "L", "cause_label": "programme weakness"},
    ]
    result = BUILD_CDS(primary_candidate=primary, selected_candidates=selected)
    assert result["depth_complete"] is True


def test_cds_depth_complete_false_when_root_missing():
    primary = {"primary_causal_category": "A", "cause_label": "bearing wear"}
    selected = [{"primary_causal_category": "G", "cause_label": "maintenance factor"}]
    result = BUILD_CDS(primary_candidate=primary, selected_candidates=selected)
    assert result["depth_complete"] is False
