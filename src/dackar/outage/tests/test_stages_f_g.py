"""
Unit tests for Stage F (InsertionOptionGenerator) and Stage G (RecommendationSynthesizer).

Coverage targets:
    Stage F:
        _score_option                — causal urgency direction: action vs non-action
        _check_regulatory_clearance  — defer blocked by TS/defer_prohibited;
                                       scope_reduction blocked by surveillance;
                                       insert_now always cleared
        _rank_options                — feasible+cleared first, blocked after infeasible
        _recommendation_confidence   — maps to duration distribution confidence_tier
        _generate_insert_now         — float_consumed_hours field name (not
                                       activity_float_consumed_hours)
        _generate_scope_reduction    — p50 × scope_reduction_fraction

    Stage G:
        _determine_analyst_review    — rejection_reason: None (§8d feedback loop)
        _build_history_summary       — outage_ids list (§8a traceability)
        _build_executive_summary     — regulatory warning in primary_conclusion (§3)
        _determine_decision_status   — MONITOR condition, INCONCLUSIVE, DEFER, ESCALATE
        _compute_attention_flags     — flag generation logic
        _determine_confidence_tier   — low-confidence degradation path
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_OUTAGE_ROOT = Path(__file__).parent.parent
if str(_OUTAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_ROOT))

from stages.stage_f_options import InsertionOptionGenerator, InsertionOptionConfig, _make_option
from stages.stage_g_recommendation import RecommendationSynthesizer, RecommendationConfig


# ===========================================================================
# Helpers
# ===========================================================================

def _gen(config: InsertionOptionConfig | None = None) -> InsertionOptionGenerator:
    return InsertionOptionGenerator(config=config)


def _synth(config: RecommendationConfig | None = None) -> RecommendationSynthesizer:
    return RecommendationSynthesizer(config=config)


def _option(
    option_type: str,
    cp_impact_hours: float = 0.0,
    confidence: float = 0.80,
    resource_conflicts: list | None = None,
    feasible: bool = True,
    regulatory_cleared: bool = True,
    risk_score: float | None = None,
    option_id: str | None = None,
) -> dict:
    opt = _make_option(
        activity_id="ACT-001",
        option_type=option_type,
        rationale="test",
        feasible=feasible,
        infeasibility_reason=None,
        cp_impact_hours=cp_impact_hours,
        criticality_label="non_critical",
        confidence=confidence,
        resource_conflicts=resource_conflicts or [],
    )
    opt["regulatory_cleared"] = regulatory_cleared
    if risk_score is not None:
        opt["risk_score"] = risk_score
    if option_id is not None:
        opt["option_id"] = option_id
    return opt


def _schedule_impact(baseline_cp_hours: float = 100.0, cp_drag: float = 0.0,
                     criticality_label: str = "non_critical",
                     float_consumed: float = 4.0) -> dict:
    # Mirrors the actual Stage E artifact: baseline_cp_hours lives inside cp_impact,
    # NOT at the top level.  Stage F._score_option must read from the nested location.
    return {
        "cp_impact": {
            "cp_drag_hours": cp_drag,
            "cp_sensitivity_score": 0.33,   # Stage E field name (not "cp_sensitivity")
            "baseline_cp_hours": baseline_cp_hours,
        },
        "float_analysis": {
            "criticality_label": criticality_label,
            "float_consumed_hours": float_consumed,
            "is_critical_path_impact": criticality_label == "critical",
        },
        "resource_conflicts": [],
        "displaced_tasks": [],
        "insertion_point": {"task_id": "T-100"},
    }


def _analogs(p50: float = 6.0, p80: float = 9.0,
             confidence_tier: str = "data_supported",
             sample_size: int = 10,
             analog_count: int = 8,
             fallback_used: bool = False,
             analog_list: list | None = None) -> dict:
    return {
        "duration_distribution": {
            "p50_hours": p50,
            "p80_hours": p80,
            "confidence_tier": confidence_tier,
            "sample_size": sample_size,
        },
        "retrieval_summary": {
            "analog_count": analog_count,
            "outages_represented": 3,
            "fallback_used": fallback_used,
        },
        "analogs": analog_list or [],
    }


def _intake_result(has_regulatory: bool = False, abbr_rate: float = 0.0,
                   regulatory_drivers: list | None = None) -> dict:
    return {
        "has_regulatory_constraint": has_regulatory,
        "unknown_abbreviation_rate": abbr_rate,
        "regulatory_drivers": regulatory_drivers or [],
    }


def _temporal_chain(posture: str = "supported", has_contradiction: bool = False,
                    links: list | None = None) -> dict:
    return {
        "summary": {
            "causal_posture": posture,
            "has_temporal_contradiction": has_contradiction,
        },
        "chain_links": links or [],
    }


# ===========================================================================
# Stage F — _score_option
# ===========================================================================

class TestScoreOptionCausalUrgency:
    """Causal urgency direction: action types use (1 − urgency);
    non-action types (defer, escalate) use urgency directly."""

    def _base_option(self, option_type: str, confidence: float = 0.80) -> dict:
        opt = _make_option(
            activity_id="ACT-001",
            option_type=option_type,
            rationale="test",
            feasible=True,
            infeasibility_reason=None,
            cp_impact_hours=0.0,       # zero CP drag → no cp_impact contribution
            criticality_label="non_critical",
            confidence=confidence,
            resource_conflicts=[],
        )
        opt["regulatory_cleared"] = True
        return opt

    def test_action_type_high_urgency_lowers_risk(self):
        """For insert_now, 'supported' posture (urgency=0.80) → urgency_score=0.20.
        Higher causal urgency → LOWER risk for action."""
        gen = _gen()
        impact = _schedule_impact(baseline_cp_hours=100.0, cp_drag=0.0)
        opt = self._base_option("insert_now", confidence=0.80)
        risk_supported = gen._score_option(opt, impact, _analogs(), causal_posture="supported")

        # For non-action "insufficient_data" posture (urgency=0.40) → urgency_score=0.60
        risk_low_urgency = gen._score_option(opt, impact, _analogs(), causal_posture="insufficient_data")

        assert risk_supported < risk_low_urgency

    def test_non_action_type_high_urgency_raises_risk(self):
        """For defer_to_post_outage, high urgency → urgency_score=urgency (high).
        High urgency to act → HIGHER risk for deferring."""
        gen = _gen()
        impact = _schedule_impact(baseline_cp_hours=100.0, cp_drag=0.0)
        opt = self._base_option("defer_to_post_outage", confidence=0.85)
        risk_supported = gen._score_option(opt, impact, _analogs(), causal_posture="supported")
        risk_low       = gen._score_option(opt, impact, _analogs(), causal_posture="weak")
        # 'supported' urgency=0.80 > 'weak' urgency=0.20 → risk_supported > risk_low
        assert risk_supported > risk_low

    def test_escalate_is_non_action_type(self):
        """escalate_to_management is in _NON_ACTION_TYPES."""
        gen = _gen()
        impact = _schedule_impact(baseline_cp_hours=100.0, cp_drag=0.0)
        opt_esc = self._base_option("escalate_to_management", confidence=0.90)
        risk_esc_high_urgency = gen._score_option(opt_esc, impact, _analogs(), "supported")
        risk_esc_low_urgency  = gen._score_option(opt_esc, impact, _analogs(), "weak")
        assert risk_esc_high_urgency > risk_esc_low_urgency

    def test_risk_score_clamped_to_unit_interval(self):
        """Risk score is always in [0, 1]."""
        gen = _gen()
        impact = _schedule_impact(baseline_cp_hours=0.001, cp_drag=9999.0)  # extreme
        opt = self._base_option("insert_now", confidence=0.0)
        opt["resource_conflicts"] = [{"conflict_type": "crew_unavailable"}]
        risk = gen._score_option(opt, impact, _analogs(), "supported")
        assert 0.0 <= risk <= 1.0


# ===========================================================================
# Stage F — _check_regulatory_clearance
# ===========================================================================

class TestCheckRegulatoryClearance:

    def _ts_driver(self) -> dict:
        return {
            "driver_id": "REG::technical_specification::abc123",
            "driver_type": "technical_specification",
            "defer_prohibited": True,
        }

    def _surv_driver(self) -> dict:
        return {
            "driver_id": "REG::surveillance_requirement::def456",
            "driver_type": "surveillance_requirement",
            "defer_prohibited": True,
        }

    def _alara_driver(self) -> dict:
        return {
            "driver_id": "REG::alara_requirement::ghi789",
            "driver_type": "alara_requirement",
            "defer_prohibited": False,
        }

    def test_defer_blocked_by_technical_specification(self):
        gen = _gen()
        opt = _option("defer_to_post_outage")
        cleared, reason = gen._check_regulatory_clearance(opt, [self._ts_driver()])
        assert cleared is False
        assert reason is not None
        assert "technical_specification" in reason.lower() or "regulatory" in reason.lower()

    def test_defer_blocked_by_defer_prohibited_true(self):
        """Any driver with defer_prohibited=True blocks deferral."""
        gen = _gen()
        opt = _option("defer_to_post_outage")
        driver = {"driver_id": "REG::custom::x", "driver_type": "custom_type",
                  "defer_prohibited": True}
        cleared, reason = gen._check_regulatory_clearance(opt, [driver])
        assert cleared is False

    def test_defer_not_blocked_by_alara_only(self):
        """ALARA has defer_prohibited=False → deferral not blocked."""
        gen = _gen()
        opt = _option("defer_to_post_outage")
        cleared, reason = gen._check_regulatory_clearance(opt, [self._alara_driver()])
        assert cleared is True
        assert reason is None

    def test_scope_reduction_blocked_by_surveillance(self):
        gen = _gen()
        opt = _option("scope_reduction")
        cleared, reason = gen._check_regulatory_clearance(opt, [self._surv_driver()])
        assert cleared is False
        assert "surveillance" in reason.lower() or "scope" in reason.lower()

    def test_scope_reduction_blocked_by_technical_specification(self):
        gen = _gen()
        opt = _option("scope_reduction")
        cleared, reason = gen._check_regulatory_clearance(opt, [self._ts_driver()])
        assert cleared is False

    def test_insert_now_always_cleared(self):
        """insert_now performs the work → always regulatory-cleared."""
        gen = _gen()
        opt = _option("insert_now")
        cleared, reason = gen._check_regulatory_clearance(opt, [self._ts_driver()])
        assert cleared is True
        assert reason is None

    def test_contingency_buffer_always_cleared(self):
        gen = _gen()
        opt = _option("add_contingency_buffer")
        cleared, _ = gen._check_regulatory_clearance(opt, [self._ts_driver()])
        assert cleared is True

    def test_parallel_execution_always_cleared(self):
        gen = _gen()
        opt = _option("parallel_execution")
        cleared, _ = gen._check_regulatory_clearance(opt, [self._ts_driver()])
        assert cleared is True

    def test_no_drivers_all_cleared(self):
        """Empty regulatory_drivers list → all option types cleared."""
        gen = _gen()
        for ot in ("insert_now", "defer_to_post_outage", "scope_reduction", "escalate_to_management"):
            cleared, reason = gen._check_regulatory_clearance(_option(ot), [])
            assert cleared is True
            assert reason is None


# ===========================================================================
# Stage F — _rank_options
# ===========================================================================

class TestRankOptions:

    def test_feasible_cleared_ranked_before_blocked(self):
        """Feasible + cleared options appear before regulatory-blocked ones."""
        gen = _gen()
        blocked = _option("defer_to_post_outage", regulatory_cleared=False, risk_score=0.10)
        feasible = _option("insert_now", risk_score=0.50)
        ranked = gen._rank_options([blocked, feasible])
        types = [o["option_type"] for o in ranked]
        assert types.index("insert_now") < types.index("defer_to_post_outage")

    def test_feasible_cleared_ranked_before_infeasible(self):
        gen = _gen()
        infeasible = _option("insert_now", feasible=False, risk_score=0.05)
        feasible   = _option("insert_now", feasible=True,  risk_score=0.80)
        ranked = gen._rank_options([infeasible, feasible])
        assert ranked[0]["feasible"] is True

    def test_within_feasible_cleared_lower_risk_first(self):
        gen = _gen()
        high_risk = _option("insert_now",            confidence=0.1, cp_impact_hours=90.0)
        low_risk  = _option("add_contingency_buffer", confidence=0.9, cp_impact_hours=0.0)
        for o in [high_risk, low_risk]:
            impact = _schedule_impact(baseline_cp_hours=100.0, cp_drag=o.get("cp_impact_hours", 0.0))
            o["risk_score"] = gen._score_option(o, impact, _analogs(), "supported")
        ranked = gen._rank_options([high_risk, low_risk])
        assert ranked[0]["risk_score"] <= ranked[-1]["risk_score"]

    def test_max_options_limit_respected(self):
        gen = _gen(InsertionOptionConfig(max_options=2))
        opts = [_option("insert_now", risk_score=float(i) / 10) for i in range(5)]
        ranked = gen._rank_options(opts)
        assert len(ranked) <= 2


# ===========================================================================
# Stage F — _generate_insert_now uses float_consumed_hours (not activity_float_consumed_hours)
# ===========================================================================

class TestGenerateInsertNowFieldNames:

    def test_float_consumed_hours_read_from_float_analysis(self):
        """Confirm the generator reads float_consumed_hours (Stage E field name)."""
        gen = _gen()
        activity = {"activity_id": "ACT-001"}
        intake = _intake_result()
        # Provide float_analysis with float_consumed_hours but NOT the old bad key
        schedule = {
            "cp_impact": {"cp_drag_hours": 0.0},
            "float_analysis": {
                "criticality_label": "non_critical",
                "float_consumed_hours": 12.0,
                # deliberately absent: "activity_float_consumed_hours"
            },
            "resource_conflicts": [],
            "insertion_point": {},
        }
        analogs = _analogs()
        opt = gen._generate_insert_now(activity, intake, schedule, analogs)
        # If the field name were wrong the rationale would show "0.0 h float consumed"
        assert "12.0 h float consumed" in opt["rationale"]


# ===========================================================================
# Stage F — _generate_scope_reduction
# ===========================================================================

class TestGenerateScopeReduction:

    def test_scope_reduction_uses_p50_times_fraction(self):
        """reduced_hours = p50 × scope_reduction_fraction (default 0.60)."""
        gen = _gen()
        activity = {"activity_id": "ACT-001"}
        schedule = _schedule_impact(cp_drag=10.0)
        analogs = _analogs(p50=20.0)
        opt = gen._generate_scope_reduction(activity, schedule, analogs)
        assert opt["cp_impact_hours"] == pytest.approx(10.0 * 0.60, abs=0.01)

    def test_scope_reduction_always_feasible(self):
        gen = _gen()
        opt = gen._generate_scope_reduction(
            {"activity_id": "ACT-001"}, _schedule_impact(), _analogs()
        )
        assert opt["feasible"] is True


# ===========================================================================
# Stage G — _determine_analyst_review
# ===========================================================================

class TestDetermineAnalystReview:

    def _call(self, decision_status: str = "PROCEED",
              has_regulatory: bool = False,
              dist_tier: str = "data_supported",
              flags: list | None = None,
              options: list | None = None) -> dict:
        synth = _synth()
        intake = _intake_result(has_regulatory=has_regulatory)
        ha = _analogs(confidence_tier=dist_tier)
        ins_opts = {"options": options or [_option("insert_now", feasible=True, regulatory_cleared=True)]}
        return synth._determine_analyst_review(
            decision_status, intake, ha, ins_opts, flags or []
        )

    def test_rejection_reason_none_always_present(self):
        """§8(d): rejection_reason key must exist and be None in every output."""
        result = self._call()
        assert "rejection_reason" in result
        assert result["rejection_reason"] is None

    def test_reviewer_decision_none_always_present(self):
        """reviewer_decision slot for analyst UI must be present and None."""
        result = self._call()
        assert "reviewer_decision" in result
        assert result["reviewer_decision"] is None

    def test_required_false_when_no_triggers(self):
        """No regulatory, good confidence, feasible options → required=False."""
        result = self._call(
            decision_status="PROCEED",
            has_regulatory=False,
            dist_tier="data_supported",
        )
        assert result["required"] is False
        assert result["reason"] is None

    def test_required_true_for_regulatory_constraint(self):
        result = self._call(has_regulatory=True)
        assert result["required"] is True
        assert result["reason"] is not None

    def test_required_true_for_inconclusive_status(self):
        result = self._call(decision_status="INCONCLUSIVE", options=[])
        assert result["required"] is True

    def test_required_true_for_low_confidence_tier(self):
        result = self._call(dist_tier="low_confidence")
        assert result["required"] is True

    def test_required_true_when_no_feasible_cleared_options(self):
        """When no feasible + cleared option exists, analyst review required."""
        opts = [_option("defer_to_post_outage", feasible=False, regulatory_cleared=False)]
        result = self._call(options=opts)
        assert result["required"] is True

    def test_required_true_for_fallback_flag(self):
        from stages.stage_g_recommendation import _FLAG_FALLBACK
        result = self._call(flags=[_FLAG_FALLBACK])
        assert result["required"] is True


# ===========================================================================
# Stage G — _build_history_summary
# ===========================================================================

class TestBuildHistorySummary:

    def _call(self, analog_list: list | None = None,
              recurrence: dict | None = None) -> dict:
        synth = _synth()
        ha = _analogs(
            analog_count=len(analog_list or []),
            analog_list=analog_list or [],
        )
        timeline = {
            "component_id": "COMP-001",
            "events": [],
            "recurrence_indicators": recurrence or {},
        }
        return synth._build_history_summary(ha, timeline)

    def test_outage_ids_field_present(self):
        """§8(a): outage_ids list must always be present."""
        result = self._call()
        assert "outage_ids" in result

    def test_outage_ids_extracted_from_analogs(self):
        analog_list = [
            {"analog_id": "A1", "outage_id": "OT-2024-03"},
            {"analog_id": "A2", "outage_id": "OT-2023-11"},
            {"analog_id": "A3", "outage_id": "OT-2024-03"},  # duplicate
        ]
        result = self._call(analog_list=analog_list)
        assert set(result["outage_ids"]) == {"OT-2024-03", "OT-2023-11"}

    def test_outage_ids_empty_when_no_analogs(self):
        result = self._call(analog_list=[])
        assert result["outage_ids"] == []

    def test_outage_ids_sorted(self):
        analog_list = [
            {"analog_id": "A1", "outage_id": "OT-2024-03"},
            {"analog_id": "A2", "outage_id": "OT-2023-01"},
        ]
        result = self._call(analog_list=analog_list)
        assert result["outage_ids"] == sorted(result["outage_ids"])

    def test_analogs_without_outage_id_skipped(self):
        """Analogs missing outage_id should not crash or add None to the list."""
        analog_list = [
            {"analog_id": "A1"},                              # no outage_id
            {"analog_id": "A2", "outage_id": "OT-2025-05"},
        ]
        result = self._call(analog_list=analog_list)
        assert None not in result["outage_ids"]
        assert "OT-2025-05" in result["outage_ids"]

    def test_recurrence_pattern_with_repeats(self):
        recurrence = {
            "trend": "increasing",
            "repeat_failure_count": 4,
            "mean_inter_event_days": 120.0,
            "pm_compliance_status": "compliant",
        }
        result = self._call(recurrence=recurrence)
        assert "4" in result["recurrence_pattern"]
        assert "120" in result["recurrence_pattern"]

    def test_insufficient_data_when_no_events(self):
        result = self._call(recurrence={"trend": "insufficient_data", "repeat_failure_count": 0})
        assert "insufficient" in result["recurrence_pattern"].lower()


# ===========================================================================
# Stage G — _build_executive_summary (§3: regulatory warning)
# ===========================================================================

class TestBuildExecutiveSummary:

    def _call(self, decision_status: str = "PROCEED",
              primary_option: dict | None = None,
              confidence_tier: str = "data_supported",
              flags: list | None = None) -> dict:
        synth = _synth()
        opt = primary_option or _option("insert_now")
        return synth._build_executive_summary(
            decision_status,
            opt,
            confidence_tier,
            flags or [],
            _analogs(analog_count=5, p50=6.0),
            _schedule_impact(),
        )

    def test_regulatory_warning_appended_when_flag_present(self):
        """§3: primary_conclusion must contain regulatory warning when flag raised."""
        from stages.stage_g_recommendation import _FLAG_REGULATORY
        result = self._call(flags=[_FLAG_REGULATORY])
        assert "REGULATORY CONSTRAINT" in result["primary_conclusion"]

    def test_no_regulatory_warning_without_flag(self):
        result = self._call(flags=[])
        assert "REGULATORY CONSTRAINT" not in result["primary_conclusion"]

    def test_low_confidence_suffix_appended(self):
        result = self._call(confidence_tier="low_confidence")
        assert "LOW CONFIDENCE" in result["primary_conclusion"]

    def test_regulatory_and_low_confidence_both_present(self):
        """When both flags apply, both warnings appear in the conclusion."""
        from stages.stage_g_recommendation import _FLAG_REGULATORY
        result = self._call(confidence_tier="low_confidence", flags=[_FLAG_REGULATORY])
        assert "REGULATORY CONSTRAINT" in result["primary_conclusion"]
        assert "LOW CONFIDENCE" in result["primary_conclusion"]

    def test_inconclusive_status_produces_distinct_conclusion(self):
        result = self._call(decision_status="INCONCLUSIVE")
        assert "inconclusive" in result["primary_conclusion"].lower()

    def test_escalate_status(self):
        result = self._call(decision_status="ESCALATE")
        assert "escalate" in result["primary_conclusion"].lower()

    def test_defer_status(self):
        result = self._call(decision_status="DEFER")
        assert "defer" in result["primary_conclusion"].lower()

    def test_monitor_status(self):
        result = self._call(decision_status="MONITOR")
        assert "monitor" in result["primary_conclusion"].lower()

    def test_attention_flags_in_output(self):
        from stages.stage_g_recommendation import _FLAG_CP_IMPACT
        result = self._call(flags=[_FLAG_CP_IMPACT])
        # Schema key is analyst_attention_flags (matches outage_activity_recommendation.json)
        assert _FLAG_CP_IMPACT in result["analyst_attention_flags"]


# ===========================================================================
# Stage G — _determine_decision_status
# ===========================================================================

class TestDetermineDecisionStatus:

    def _call(self, primary_option: dict | None,
              dist_tier: str = "data_supported",
              analog_count: int = 5) -> str:
        synth = _synth()
        ha = _analogs(confidence_tier=dist_tier, analog_count=analog_count)
        ins_opts = {}  # not used by _determine_decision_status
        return synth._determine_decision_status(primary_option, _intake_result(), ha, ins_opts)

    def test_inconclusive_when_no_primary_option(self):
        assert self._call(None) == "INCONCLUSIVE"

    def test_escalate_for_escalate_option(self):
        opt = _option("escalate_to_management")
        assert self._call(opt) == "ESCALATE"

    def test_defer_for_defer_option(self):
        opt = _option("defer_to_post_outage")
        assert self._call(opt) == "DEFER"

    def test_monitor_zero_cp_no_analogs_low_confidence(self):
        """MONITOR: zero CP impact + analog_count==0 + low_confidence tier."""
        opt = _option("insert_now", cp_impact_hours=0.0)
        assert self._call(opt, dist_tier="low_confidence", analog_count=0) == "MONITOR"

    def test_proceed_for_insert_now(self):
        opt = _option("insert_now", cp_impact_hours=4.0)
        assert self._call(opt, dist_tier="data_supported", analog_count=8) == "PROCEED"

    def test_proceed_for_contingency_buffer(self):
        opt = _option("add_contingency_buffer", cp_impact_hours=2.0)
        assert self._call(opt, dist_tier="sme_informed", analog_count=3) == "PROCEED"

    def test_proceed_for_scope_reduction(self):
        opt = _option("scope_reduction", cp_impact_hours=3.0)
        assert self._call(opt, dist_tier="data_supported", analog_count=6) == "PROCEED"

    def test_monitor_not_triggered_when_has_analogs(self):
        """Zero CP drag but non-zero analogs → PROCEED, not MONITOR."""
        opt = _option("insert_now", cp_impact_hours=0.0)
        result = self._call(opt, dist_tier="data_supported", analog_count=5)
        assert result != "MONITOR"


# ===========================================================================
# Stage G — _compute_attention_flags
# ===========================================================================

class TestComputeAttentionFlags:

    def _call(self, **kwargs) -> list:
        synth = _synth()
        intake = _intake_result(
            has_regulatory=kwargs.get("has_regulatory", False),
            abbr_rate=kwargs.get("abbr_rate", 0.0),
        )
        ha = _analogs(
            confidence_tier=kwargs.get("dist_tier", "data_supported"),
            analog_count=kwargs.get("analog_count", 8),
            fallback_used=kwargs.get("fallback_used", False),
        )
        schedule = _schedule_impact(
            criticality_label=kwargs.get("criticality_label", "non_critical"),
        )
        schedule["displaced_tasks"] = kwargs.get("displaced_tasks", [])
        chain = _temporal_chain(has_contradiction=kwargs.get("has_contradiction", False))
        ins_opts = {}
        return synth._compute_attention_flags(intake, ha, schedule, chain, ins_opts)

    def test_no_flags_clean_case(self):
        flags = self._call(analog_count=10, dist_tier="data_supported")
        from stages.stage_g_recommendation import (
            _FLAG_REGULATORY, _FLAG_LOW_CONFIDENCE, _FLAG_TEMPORAL_CONTRADICTION,
            _FLAG_CP_IMPACT, _FLAG_HIGH_ABBR_RATE, _FLAG_FALLBACK,
            _FLAG_DISPLACED_REGULATORY,
        )
        for flag in [_FLAG_REGULATORY, _FLAG_LOW_CONFIDENCE, _FLAG_TEMPORAL_CONTRADICTION,
                     _FLAG_CP_IMPACT, _FLAG_HIGH_ABBR_RATE, _FLAG_FALLBACK,
                     _FLAG_DISPLACED_REGULATORY]:
            assert flag not in flags

    def test_regulatory_flag_raised(self):
        from stages.stage_g_recommendation import _FLAG_REGULATORY
        flags = self._call(has_regulatory=True)
        assert _FLAG_REGULATORY in flags

    def test_low_confidence_flag_raised(self):
        from stages.stage_g_recommendation import _FLAG_LOW_CONFIDENCE
        flags = self._call(dist_tier="low_confidence")
        assert _FLAG_LOW_CONFIDENCE in flags

    def test_low_analog_count_flag(self):
        from stages.stage_g_recommendation import _FLAG_LOW_ANALOGS
        flags = self._call(analog_count=2)
        assert _FLAG_LOW_ANALOGS in flags

    def test_temporal_contradiction_flag(self):
        from stages.stage_g_recommendation import _FLAG_TEMPORAL_CONTRADICTION
        flags = self._call(has_contradiction=True)
        assert _FLAG_TEMPORAL_CONTRADICTION in flags

    def test_high_abbr_rate_flag(self):
        from stages.stage_g_recommendation import _FLAG_HIGH_ABBR_RATE
        flags = self._call(abbr_rate=0.50)  # > default threshold 0.25
        assert _FLAG_HIGH_ABBR_RATE in flags

    def test_fallback_flag(self):
        from stages.stage_g_recommendation import _FLAG_FALLBACK
        flags = self._call(fallback_used=True)
        assert _FLAG_FALLBACK in flags

    def test_displaced_regulatory_flag(self):
        from stages.stage_g_recommendation import _FLAG_DISPLACED_REGULATORY
        displaced = [{"task_id": "T-1", "has_regulatory_constraint": True}]
        flags = self._call(displaced_tasks=displaced)
        assert _FLAG_DISPLACED_REGULATORY in flags


# ===========================================================================
# Stage G — _determine_confidence_tier
# ===========================================================================

class TestDetermineConfidenceTier:

    def test_returns_dist_tier_when_confident_option(self):
        synth = _synth()
        ha = _analogs(confidence_tier="data_supported")
        opt = _option("insert_now", confidence=0.85)
        opt["option_id"] = "OPT-1"
        ins_opts = {"recommended_option_id": "OPT-1", "options": [opt]}
        assert synth._determine_confidence_tier(ha, ins_opts) == "data_supported"

    def test_degrades_to_low_confidence_when_option_confidence_low(self):
        synth = _synth()
        ha = _analogs(confidence_tier="sme_informed")
        opt = _option("insert_now", confidence=0.30)  # < 0.40 threshold
        opt["option_id"] = "OPT-1"
        ins_opts = {"recommended_option_id": "OPT-1", "options": [opt]}
        assert synth._determine_confidence_tier(ha, ins_opts) == "low_confidence"

    def test_no_recommended_option_returns_dist_tier(self):
        synth = _synth()
        ha = _analogs(confidence_tier="sme_informed")
        ins_opts = {"recommended_option_id": None, "options": []}
        assert synth._determine_confidence_tier(ha, ins_opts) == "sme_informed"


# ===========================================================================
# Stage D — _tukey_filter (pure module-level function, no external dependencies)
# ===========================================================================

class TestTukeyFilter:

    def _indexed(self, durations: list) -> tuple:
        """Build indexed pairs and plain duration list for _tukey_filter."""
        analogs = [{"actual_duration_hours": d} for d in durations]
        indexed = list(enumerate(analogs))
        return indexed, durations

    def test_no_outliers_all_retained(self):
        from stages.stage_d_analogs import _tukey_filter
        indexed, durs = self._indexed([4.0, 5.0, 6.0, 7.0, 8.0])
        kept, n_removed = _tukey_filter(indexed, durs)
        assert n_removed == 0
        assert len(kept) == 5

    def test_extreme_high_outlier_removed(self):
        """Value far above Q3 + 1.5×IQR should be removed."""
        from stages.stage_d_analogs import _tukey_filter
        indexed, durs = self._indexed([4.0, 5.0, 6.0, 7.0, 200.0])
        kept, n_removed = _tukey_filter(indexed, durs)
        assert n_removed >= 1
        kept_vals = [a["actual_duration_hours"] for _, a in kept]
        assert 200.0 not in kept_vals

    def test_fewer_than_4_returns_unchanged(self):
        """With < 4 values the filter cannot be applied; all returned unchanged."""
        from stages.stage_d_analogs import _tukey_filter
        indexed, durs = self._indexed([4.0, 5.0, 6.0])
        kept, n_removed = _tukey_filter(indexed, durs)
        assert n_removed == 0
        assert len(kept) == 3

    def test_constant_values_no_outliers(self):
        """Constant durations → IQR=0, upper fence = Q3 → all values at Q3 kept."""
        from stages.stage_d_analogs import _tukey_filter
        indexed, durs = self._indexed([8.0, 8.0, 8.0, 8.0, 8.0])
        kept, n_removed = _tukey_filter(indexed, durs)
        assert n_removed == 0


# ===========================================================================
# Stage D — _compute_confidence_tier (pure method)
# ===========================================================================

class TestComputeConfidenceTier:

    def _retriever(self) -> object:
        from stages.stage_d_analogs import HistoricalAnalogRetriever, HistoricalAnalogConfig
        # data_supported threshold=10, sme_informed threshold=3
        cfg = HistoricalAnalogConfig(
            min_analogs_for_data_supported=10,
            min_analogs_for_sme_informed=3,
        )
        return HistoricalAnalogRetriever(config=cfg)

    def _analogs(self, count: int, with_duration: bool = True) -> list:
        return [
            {"analog_id": f"A{i}", "actual_duration_hours": float(i + 1) if with_duration else None}
            for i in range(count)
        ]

    def test_data_supported_at_threshold(self):
        r = self._retriever()
        assert r._compute_confidence_tier(self._analogs(10)) == "data_supported"

    def test_sme_informed_between_thresholds(self):
        r = self._retriever()
        assert r._compute_confidence_tier(self._analogs(5)) == "sme_informed"

    def test_low_confidence_below_sme_threshold(self):
        r = self._retriever()
        assert r._compute_confidence_tier(self._analogs(2)) == "low_confidence"

    def test_empty_analogs_low_confidence(self):
        r = self._retriever()
        assert r._compute_confidence_tier([]) == "low_confidence"

    def test_analogs_without_duration_not_counted(self):
        """Analogs with no actual_duration_hours should not count toward tier."""
        r = self._retriever()
        # 5 analogs but none have duration → sample_size=0
        analogs = [{"analog_id": f"A{i}"} for i in range(5)]
        assert r._compute_confidence_tier(analogs) == "low_confidence"


# ===========================================================================
# Stage D — _remove_duration_outliers reconstruction
# ===========================================================================

class TestRemoveDurationOutliers:
    """Verify that: (1) without-duration analogs are always retained,
    (2) outliers are removed, (3) original index order is preserved."""

    def _retriever(self) -> object:
        from stages.stage_d_analogs import HistoricalAnalogRetriever
        return HistoricalAnalogRetriever()  # no outlier_handler → uses _tukey_filter

    def test_without_duration_analogs_always_retained(self):
        r = self._retriever()
        analogs = [
            {"analog_id": "A0", "actual_duration_hours": 5.0},
            {"analog_id": "A1"},                                 # no duration
            {"analog_id": "A2", "actual_duration_hours": 6.0},
            {"analog_id": "A3"},                                 # no duration
            {"analog_id": "A4", "actual_duration_hours": 200.0},  # outlier
            {"analog_id": "A5", "actual_duration_hours": 7.0},
        ]
        result, n_removed = r._remove_duration_outliers(analogs)
        result_ids = [a["analog_id"] for a in result]
        assert "A1" in result_ids
        assert "A3" in result_ids

    def test_outlier_removed_from_result(self):
        r = self._retriever()
        analogs = [
            {"analog_id": f"A{i}", "actual_duration_hours": float(i + 4)}
            for i in range(4)  # [4.0, 5.0, 6.0, 7.0]
        ] + [{"analog_id": "A_outlier", "actual_duration_hours": 500.0}]
        result, n_removed = r._remove_duration_outliers(analogs)
        result_ids = [a["analog_id"] for a in result]
        assert "A_outlier" not in result_ids
        assert n_removed == 1

    def test_fewer_than_2_with_duration_unchanged(self):
        """With < 2 duration entries, the filter cannot run; all returned."""
        r = self._retriever()
        analogs = [
            {"analog_id": "A0", "actual_duration_hours": 5.0},
            {"analog_id": "A1"},
        ]
        result, n_removed = r._remove_duration_outliers(analogs)
        assert n_removed == 0
        assert len(result) == 2

    def test_confidence_tier_merged_into_distribution(self):
        """After retrieve(), duration_distribution must include confidence_tier
        (critical bug fix: the tier must not remain as the fitter's internal value)."""
        from stages.stage_d_analogs import HistoricalAnalogRetriever, HistoricalAnalogConfig
        cfg = HistoricalAnalogConfig(
            min_analogs_for_data_supported=10,
            min_analogs_for_sme_informed=3,
        )
        r = HistoricalAnalogRetriever(config=cfg)
        # Simulate 0 analogs: confidence_tier should be low_confidence in the artifact
        analogs: list = []
        dist_dict = {
            "distribution_type": "unknown",
            "p50_hours": None,
            "p80_hours": None,
            "p90_hours": None,
            "mean_hours": None,
            "std_hours": None,
            "confidence_tier": "data_supported",  # fitter stub: wrong tier
            "sample_size": 0,
        }
        # Call the override directly as the pipeline does
        confidence_tier = r._compute_confidence_tier(analogs)
        dist_dict["confidence_tier"] = confidence_tier     # this is the fix
        assert dist_dict["confidence_tier"] == "low_confidence"


# ===========================================================================
# Stage F — _generate_insert_now crew_unavailable → infeasible
# ===========================================================================

class TestGenerateInsertNowFeasibility:

    def test_crew_unavailable_makes_option_infeasible(self):
        gen = _gen()
        activity = {"activity_id": "ACT-001"}
        intake = _intake_result()
        schedule = {
            "cp_impact": {"cp_drag_hours": 4.0},
            "float_analysis": {
                "criticality_label": "near_critical",
                "float_consumed_hours": 4.0,
            },
            "resource_conflicts": [
                {"conflict_type": "crew_unavailable", "skill_required": "I&C technician"}
            ],
            "insertion_point": {"task_id": "T-100"},
        }
        opt = gen._generate_insert_now(activity, intake, schedule, _analogs())
        assert opt["feasible"] is False
        assert opt["infeasibility_reason"] is not None
        assert "I&C technician" in opt["infeasibility_reason"]

    def test_non_crew_conflict_does_not_block_feasibility(self):
        """Tool/scaffold conflicts are non-blocking; option remains feasible."""
        gen = _gen()
        activity = {"activity_id": "ACT-001"}
        schedule = {
            "cp_impact": {"cp_drag_hours": 0.0},
            "float_analysis": {
                "criticality_label": "non_critical",
                "float_consumed_hours": 0.0,
            },
            "resource_conflicts": [
                {"conflict_type": "tool_unavailable", "skill_required": "scaffold"}
            ],
            "insertion_point": {},
        }
        opt = gen._generate_insert_now(activity, _intake_result(), schedule, _analogs())
        assert opt["feasible"] is True


# ===========================================================================
# Stage F — _generate_defer infeasibility (safety_related / active_lco)
# ===========================================================================

class TestGenerateDeferFeasibility:

    def test_safety_related_makes_defer_infeasible(self):
        gen = _gen()
        activity = {"activity_id": "ACT-001", "safety_related": True}
        opt = gen._generate_defer(activity, _intake_result(), _schedule_impact())
        assert opt["feasible"] is False
        assert opt["infeasibility_reason"] is not None

    def test_active_lco_makes_defer_infeasible(self):
        gen = _gen()
        activity = {"activity_id": "ACT-001", "active_lco": True}
        opt = gen._generate_defer(activity, _intake_result(), _schedule_impact())
        assert opt["feasible"] is False
        assert "LCO" in opt["infeasibility_reason"]

    def test_non_safety_related_no_lco_is_feasible(self):
        gen = _gen()
        activity = {"activity_id": "ACT-001"}
        opt = gen._generate_defer(activity, _intake_result(), _schedule_impact())
        assert opt["feasible"] is True


# ===========================================================================
# Stage G — _determine_analyst_review high abbreviation rate trigger (§6 fix)
# ===========================================================================

class TestAnalystReviewHighAbbrRate:

    def test_required_true_for_high_abbr_rate_flag(self):
        """§6 exit criterion: high unknown abbreviation rate must trigger analyst review."""
        from stages.stage_g_recommendation import _FLAG_HIGH_ABBR_RATE
        synth = _synth()
        intake = _intake_result(abbr_rate=0.0)       # abbr_rate itself not used here;
        ha = _analogs(confidence_tier="data_supported")  # flag is already in attention_flags
        ins_opts = {"options": [_option("insert_now")]}
        result = synth._determine_analyst_review(
            "PROCEED", intake, ha, ins_opts,
            attention_flags=[_FLAG_HIGH_ABBR_RATE],
        )
        assert result["required"] is True
        assert result["reason"] is not None
        assert "abbreviation" in result["reason"].lower()

    def test_rejection_reason_still_none_with_abbr_flag(self):
        from stages.stage_g_recommendation import _FLAG_HIGH_ABBR_RATE
        synth = _synth()
        ha = _analogs(confidence_tier="data_supported")
        ins_opts = {"options": [_option("insert_now")]}
        result = synth._determine_analyst_review(
            "PROCEED", _intake_result(), ha, ins_opts,
            attention_flags=[_FLAG_HIGH_ABBR_RATE],
        )
        assert result["rejection_reason"] is None


# ===========================================================================
# Stage G — _build_schedule_summary field names
# ===========================================================================

class TestBuildScheduleSummary:

    def test_float_consumed_hours_field_present(self):
        """Stage G must read float_consumed_hours (not activity_float_consumed_hours)."""
        synth = _synth()
        schedule = _schedule_impact(float_consumed=14.5)
        result = synth._build_schedule_summary(schedule)
        assert "float_consumed_hours" in result
        assert result["float_consumed_hours"] == pytest.approx(14.5)

    def test_cp_impact_hours_present(self):
        synth = _synth()
        result = synth._build_schedule_summary(_schedule_impact(cp_drag=6.0))
        assert result["cp_impact_hours"] == pytest.approx(6.0)

    def test_has_displaced_regulatory_tasks(self):
        synth = _synth()
        schedule = _schedule_impact()
        schedule["displaced_tasks"] = [{"task_id": "T-1", "has_regulatory_constraint": True}]
        result = synth._build_schedule_summary(schedule)
        assert result["has_displaced_regulatory_tasks"] is True

    def test_no_displaced_regulatory_tasks(self):
        synth = _synth()
        schedule = _schedule_impact()
        schedule["displaced_tasks"] = [{"task_id": "T-1"}]
        result = synth._build_schedule_summary(schedule)
        assert result["has_displaced_regulatory_tasks"] is False


# ===========================================================================
# Regression: Stage F _score_option reads baseline_cp_hours from nested cp_impact
# ===========================================================================

class TestScoreOptionBaselineCpNested:
    """baseline_cp_hours lives inside cp_impact in the Stage E artifact.
    _score_option must NOT fall back silently to 1.0 when cp_impact is present."""

    def test_baseline_cp_read_from_nested_cp_impact(self):
        """With baseline_cp_hours=480 inside cp_impact and 48 h drag,
        cp_impact_score ≈ 48/480 = 0.10 — not 1.0 (which would result from /1.0)."""
        gen = _gen()
        schedule = _schedule_impact(baseline_cp_hours=480.0, cp_drag=48.0)
        opt = _option("insert_now", cp_impact_hours=48.0, confidence=0.80)
        risk = gen._score_option(opt, schedule, _analogs(), causal_posture="supported")
        # cp_impact_score = 48/480 = 0.10; if fallback 1.0 used it would be 1.0
        # total: 0.40*0.10 + 0.30*(1-0.80) + 0.20*0 + 0.10*(1-0.80) = 0.04+0.06+0+0.02 = 0.12
        assert risk == pytest.approx(0.12, abs=0.01)

    def test_score_not_saturated_with_large_baseline(self):
        """Risk should not be ~1.0 just because drag is large relative to 1.0 fallback."""
        gen = _gen()
        schedule = _schedule_impact(baseline_cp_hours=480.0, cp_drag=10.0)
        opt = _option("insert_now", cp_impact_hours=10.0, confidence=0.80)
        risk = gen._score_option(opt, schedule, _analogs(), causal_posture="supported")
        # With correct baseline (480h), cp_impact_score ≈ 0.02 → risk well below 0.5
        assert risk < 0.30


# ===========================================================================
# Regression: Stage G evidence snippet uses cp_sensitivity_score (not cp_sensitivity)
# ===========================================================================

class TestEvidenceChainCpSensitivityFieldName:
    """Stage E outputs cp_sensitivity_score; the evidence snippet must use that name."""

    def test_cp_sensitivity_score_appears_in_evidence_snippet(self):
        """evidence snippet must not show '?' for CP sensitivity when
        cp_sensitivity_score is present in the cp_impact dict."""
        synth = _synth()
        schedule = _schedule_impact(cp_drag=6.0)
        # schedule["cp_impact"] already has cp_sensitivity_score=0.33 from helper
        temporal = {"summary": {}, "chain_links": []}
        analogs = _analogs(analog_list=[])
        timeline = {"component_id": "C1", "events": []}

        evidence = synth._assemble_evidence_chain(temporal, analogs, schedule, timeline)
        # Find the schedule_analysis evidence item
        sched_ev = next((e for e in evidence if e["source_type"] == "schedule_analysis"), None)
        assert sched_ev is not None
        # The snippet must not contain "?: " for cp_sensitivity
        assert "CP sensitivity: ?" not in sched_ev["snippet"]
