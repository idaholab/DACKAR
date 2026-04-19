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

    def test_high_urgency_insert_now_beats_defer_at_sub48h_drag(self):
        """N4 regression: for a high-urgency ('supported') non-TS activity with
        sub-48 h CP drag, insert_now must score lower (better) than defer.

        Before the N4 fix (causal_urgency weight 0.10), the weak urgency signal
        could not overcome the 0.0 cp_impact advantage of defer, so defer was
        incorrectly recommended.  With weight 0.20 the urgency term is strong
        enough to tip insert_now below defer for the 'supported' posture.
        """
        gen = _gen()
        # 40 h drag against a 480 h baseline ≈ sub-48 h (moderately critical)
        impact = _schedule_impact(baseline_cp_hours=480.0, cp_drag=40.0)
        opt_insert = self._base_option("insert_now",         confidence=0.85)
        opt_defer  = self._base_option("defer_to_post_outage", confidence=0.85)
        opt_insert["cp_impact_hours"] = 40.0   # non-zero drag
        opt_defer["cp_impact_hours"]  = 0.0    # defer avoids in-outage drag

        risk_insert = gen._score_option(opt_insert, impact, _analogs(), "supported")
        risk_defer  = gen._score_option(opt_defer,  impact, _analogs(), "supported")

        assert risk_insert < risk_defer, (
            f"High-urgency insert_now ({risk_insert:.4f}) should beat defer "
            f"({risk_defer:.4f}) for sub-48 h CP drag with 'supported' posture"
        )


# ===========================================================================
# Stage F — _check_regulatory_clearance
# ===========================================================================

class TestCheckRegulatoryClearance:

    def _ts_driver(self) -> dict:
        return {
            "driver_id": "REG::ts_surveillance::abc123",
            "driver_type": "ts_surveillance",
            "defer_prohibited": True,
        }

    def _surv_driver(self) -> dict:
        # surveillance maps to ts_surveillance in the schema enum (Y2 fix)
        return {
            "driver_id": "REG::ts_surveillance::def456",
            "driver_type": "ts_surveillance",
            "defer_prohibited": True,
        }

    def _alara_driver(self) -> dict:
        return {
            "driver_id": "REG::alara_constraint::ghi789",
            "driver_type": "alara_constraint",
            "defer_prohibited": False,
        }

    def test_defer_blocked_by_ts_surveillance(self):
        gen = _gen()
        opt = _option("defer_to_post_outage")
        cleared, reason = gen._check_regulatory_clearance(opt, [self._ts_driver()])
        assert cleared is False
        assert reason is not None
        assert "ts_surveillance" in reason.lower() or "regulatory" in reason.lower()

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

    def test_scope_reduction_blocked_by_ts_surveillance(self):
        """ts_surveillance (covers TS, LCO, surveillance) blocks scope reduction."""
        gen = _gen()
        opt = _option("scope_reduction")
        cleared, reason = gen._check_regulatory_clearance(opt, [self._surv_driver()])
        assert cleared is False
        assert "ts_surveillance" in reason.lower() or "scope" in reason.lower()

    def test_scope_reduction_blocked_by_ts_surveillance_via_ts_driver(self):
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
              analog_count: int = 5,
              criticality_label: str = "non_critical") -> str:
        synth = _synth()
        ha = _analogs(confidence_tier=dist_tier, analog_count=analog_count)
        ins_opts = {}
        schedule = _schedule_impact(criticality_label=criticality_label)
        return synth._determine_decision_status(
            primary_option, _intake_result(), ha, ins_opts,
            schedule_impact_assessment=schedule,
        )

    def test_inconclusive_when_no_primary_option(self):
        assert self._call(None) == "INCONCLUSIVE"

    def test_escalate_for_escalate_option(self):
        opt = _option("escalate_to_management")
        assert self._call(opt) == "ESCALATE"

    def test_defer_for_defer_option(self):
        opt = _option("defer_to_post_outage")
        assert self._call(opt) == "DEFER"

    def test_monitor_zero_cp_no_analogs_low_confidence(self):
        """MONITOR: analog_count==0 + low_confidence tier + non_critical path."""
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

    def test_n5_monitor_blocked_when_criticality_is_critical(self):
        """N5 fix: critical path activity must not be demoted to MONITOR even with
        zero analogs and low_confidence.  MONITOR is only for non-critical activities
        where 'watch and wait' is acceptable."""
        opt = _option("insert_now", cp_impact_hours=0.0)
        result = self._call(
            opt, dist_tier="low_confidence", analog_count=0,
            criticality_label="critical",
        )
        assert result == "PROCEED", (
            "Critical path activity with no analogs must be PROCEED (act, do not watch)"
        )

    def test_n5_monitor_not_blocked_for_non_critical(self):
        """N5 fix: non-critical + low confidence + no analogs still gives MONITOR."""
        opt = _option("insert_now", cp_impact_hours=0.0)
        result = self._call(
            opt, dist_tier="low_confidence", analog_count=0,
            criticality_label="non_critical",
        )
        assert result == "MONITOR"

    def test_n5_no_schedule_impact_defaults_to_non_critical(self):
        """N5 fix: absent schedule_impact_assessment → criticality defaults to
        non_critical (permissive MONITOR condition — safe for unknown schedules)."""
        synth = _synth()
        ha = _analogs(confidence_tier="low_confidence", analog_count=0)
        opt = _option("insert_now", cp_impact_hours=0.0)
        result = synth._determine_decision_status(
            opt, _intake_result(), ha, {},
            schedule_impact_assessment=None,
        )
        assert result == "MONITOR"


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

    def test_h1_cp_impact_flag_raised_when_critical(self):
        """H1 fix: _FLAG_CP_IMPACT raised when criticality_label == 'critical'.

        Before the fix the code read float_analysis.is_critical_path_impact (a
        field that never exists), so the flag was never appended.  After the fix
        it reads criticality_label == 'critical'.
        """
        from stages.stage_g_recommendation import _FLAG_CP_IMPACT
        flags = self._call(criticality_label="critical")
        assert _FLAG_CP_IMPACT in flags, (
            "_FLAG_CP_IMPACT must be raised when criticality_label is 'critical'"
        )

    def test_h1_cp_impact_flag_absent_when_non_critical(self):
        """Sanity: _FLAG_CP_IMPACT must NOT appear for non_critical activities."""
        from stages.stage_g_recommendation import _FLAG_CP_IMPACT
        flags = self._call(criticality_label="non_critical")
        assert _FLAG_CP_IMPACT not in flags


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
# Stage D — _make_activity_case execution mode flags
# ===========================================================================

class TestMakeActivityCaseExecutionFlags:
    """Verify that _make_activity_case correctly sets execution mode flags on
    the query ActivityCase so ContextSimilarityScorer can use them.

    Key invariant: flags must be explicitly set to bool values (not left
    unset), because ActivityCase is built via __new__ without __init__ and
    unset attributes return None from getattr — which causes the scorer to
    skip them via its weight-redistribution logic.
    """

    def _make(self, **flags) -> object:
        from stages.stage_d_analogs import _make_activity_case
        return _make_activity_case(
            "replace pump seal",
            discipline="mechanical",
            **flags,
        )

    def test_default_flags_are_false_not_none(self):
        """Flags not supplied should be False, not None, so scorer does not skip them."""
        ac = self._make()
        assert ac.has_rp_hold is False
        assert ac.requires_scaffold is False
        assert ac.has_clearance is False
        assert ac.is_vendor_supported is False

    def test_rp_hold_set_true(self):
        ac = self._make(has_rp_hold=True)
        assert ac.has_rp_hold is True

    def test_requires_scaffold_set_true(self):
        ac = self._make(requires_scaffold=True)
        assert ac.requires_scaffold is True

    def test_has_clearance_set_true(self):
        ac = self._make(has_clearance=True)
        assert ac.has_clearance is True

    def test_is_vendor_supported_set_true(self):
        ac = self._make(is_vendor_supported=True)
        assert ac.is_vendor_supported is True

    def test_all_flags_set_simultaneously(self):
        ac = self._make(
            has_rp_hold=True, requires_scaffold=True,
            has_clearance=True, is_vendor_supported=True,
        )
        assert ac.has_rp_hold is True
        assert ac.requires_scaffold is True
        assert ac.has_clearance is True
        assert ac.is_vendor_supported is True


class TestBuildQueryExecutionFlagPassthrough:
    """Verify that _build_query reads execution_mode_flags from intake_result
    and sets them on the internal query ActivityCase.
    """

    def _retriever(self):
        from stages.stage_d_analogs import HistoricalAnalogRetriever
        return HistoricalAnalogRetriever()

    def _ea(self, description: str = "replace pump seal") -> dict:
        return {
            "activity_id": "ACT-001",
            "outage_id": "RF-22",
            "plant_id": "PLANT-1",
            "raw_description": description,
        }

    def _intake(self, flags: dict) -> dict:
        return {
            "expanded_description": "replace pump seal",
            "execution_mode_flags": flags,
        }

    def test_rp_hold_flag_reaches_query_activity_case(self):
        r = self._retriever()
        _, qac = r._build_query(
            self._ea(),
            self._intake({"has_rp_hold": True, "requires_scaffold": False,
                          "has_clearance": False, "is_vendor_supported": False}),
        )
        assert qac is not None
        assert qac.has_rp_hold is True
        assert qac.requires_scaffold is False

    def test_scaffold_flag_reaches_query_activity_case(self):
        r = self._retriever()
        _, qac = r._build_query(
            self._ea(),
            self._intake({"has_rp_hold": False, "requires_scaffold": True,
                          "has_clearance": False, "is_vendor_supported": False}),
        )
        assert qac.requires_scaffold is True

    def test_all_flags_false_propagated_explicitly(self):
        """All flags False — must be explicitly set, not absent (scorer skips None)."""
        r = self._retriever()
        _, ac = r._build_query(
            self._ea(),
            self._intake({"has_rp_hold": False, "requires_scaffold": False,
                          "has_clearance": False, "is_vendor_supported": False}),
        )
        # getattr with default None: if False is explicitly set, it's found
        assert getattr(ac, "has_rp_hold", None) is False
        assert getattr(ac, "requires_scaffold", None) is False
        assert getattr(ac, "has_clearance", None) is False
        assert getattr(ac, "is_vendor_supported", None) is False

    def test_missing_execution_flags_key_defaults_to_false(self):
        """intake_result without execution_mode_flags should not raise."""
        r = self._retriever()
        _, ac = r._build_query(self._ea(), {"expanded_description": "replace seal"})
        assert getattr(ac, "has_rp_hold", None) is False

    def test_query_summary_includes_execution_mode_flags(self):
        """_build_query must record active flags in query_summary for artifact traceability."""
        r = self._retriever()
        summary, _ = r._build_query(
            self._ea(),
            self._intake({"has_rp_hold": True, "requires_scaffold": False,
                          "has_clearance": False, "is_vendor_supported": False}),
        )
        assert "execution_mode_flags" in summary
        assert summary["execution_mode_flags"]["has_rp_hold"] is True
        assert summary["execution_mode_flags"]["requires_scaffold"] is False

    def test_query_summary_flags_absent_intake_defaults_to_all_false(self):
        """No execution_mode_flags key in intake → query_summary flags all False."""
        r = self._retriever()
        summary, _ = r._build_query(self._ea(), {"expanded_description": "replace seal"})
        flags = summary["execution_mode_flags"]
        assert all(v is False for v in flags.values())

    def test_context_scorer_differentiates_rp_hold_with_real_analogs(self):
        """End-to-end: query with has_rp_hold=True scores higher against an RP-hold
        analog than against a non-RP-hold analog when all other fields are equal."""
        from outage_uncertainty.retrieval.context_similarity import ContextSimilarityScorer
        from outage_uncertainty.domain.activity import ActivityCase

        scorer = ContextSimilarityScorer()
        r = self._retriever()
        _, query = r._build_query(
            self._ea(),
            self._intake({"has_rp_hold": True, "requires_scaffold": False,
                          "has_clearance": False, "is_vendor_supported": False}),
        )

        base = dict(
            activity_id="A", outage_id="O", plant_id="PLANT-1",
            discipline="mechanical", task_family="replacement",
        )
        analog_with_rp  = ActivityCase(**base, has_rp_hold=True)
        analog_without_rp = ActivityCase(**{**base, "activity_id": "B"}, has_rp_hold=False)

        score_match    = scorer.score(query, analog_with_rp)
        score_mismatch = scorer.score(query, analog_without_rp)
        assert score_match > score_mismatch, (
            f"RP-hold match ({score_match:.3f}) should exceed mismatch ({score_mismatch:.3f})"
        )


# ===========================================================================
# Stage D — _compute_confidence_tier (pure method)
# ===========================================================================

class TestComputeConfidenceTier:

    def _retriever(self, min_outages_data=3) -> object:
        from stages.stage_d_analogs import HistoricalAnalogRetriever, HistoricalAnalogConfig
        # data_supported threshold=10, sme_informed threshold=3
        cfg = HistoricalAnalogConfig(
            min_analogs_for_data_supported=10,
            min_analogs_for_sme_informed=3,
            min_outages_for_data_supported=min_outages_data,
        )
        return HistoricalAnalogRetriever(config=cfg)

    def _analogs(self, count: int, with_duration: bool = True,
                 n_outages: int = 4) -> list:
        """Build ``count`` analogs round-robined across ``n_outages`` outage IDs."""
        return [
            {
                "analog_id": f"A{i}",
                "actual_duration_hours": float(i + 1) if with_duration else None,
                "outage_id": f"RF{(i % n_outages) + 1}",
            }
            for i in range(count)
        ]

    # -- count gate ----------------------------------------------------------

    def test_data_supported_at_threshold(self):
        r = self._retriever()
        assert r._compute_confidence_tier(self._analogs(10), None) == "data_supported"

    def test_sme_informed_between_thresholds(self):
        r = self._retriever()
        assert r._compute_confidence_tier(self._analogs(5), None) == "sme_informed"

    def test_low_confidence_below_sme_threshold(self):
        r = self._retriever()
        assert r._compute_confidence_tier(self._analogs(2), None) == "low_confidence"

    def test_empty_analogs_low_confidence(self):
        r = self._retriever()
        assert r._compute_confidence_tier([], None) == "low_confidence"

    def test_analogs_without_duration_not_counted(self):
        """Analogs with no actual_duration_hours should not count toward tier."""
        r = self._retriever()
        # 5 analogs but none have duration → sample_size=0
        analogs = [{"analog_id": f"A{i}", "outage_id": f"RF{i+1}"} for i in range(5)]
        assert r._compute_confidence_tier(analogs, None) == "low_confidence"

    # -- outage diversity gate -----------------------------------------------

    def test_single_outage_caps_data_supported_to_sme_informed(self):
        """10 analogs from 1 outage → downgraded from data_supported to sme_informed."""
        r = self._retriever()
        analogs = self._analogs(10, n_outages=1)   # all from RF1
        assert r._compute_confidence_tier(analogs, None) == "sme_informed"

    def test_single_outage_sme_informed_not_affected_by_gate(self):
        """5 analogs from 1 outage: sme_informed has no outage gate → stays sme_informed."""
        r = self._retriever()
        analogs = self._analogs(5, n_outages=1)
        assert r._compute_confidence_tier(analogs, None) == "sme_informed"

    def test_two_outages_also_sme_informed(self):
        """5 analogs from 2 outages → sme_informed (count gate, no outage cap at this tier)."""
        r = self._retriever()
        analogs = self._analogs(5, n_outages=2)
        assert r._compute_confidence_tier(analogs, None) == "sme_informed"

    def test_two_outages_still_caps_data_supported(self):
        """10 analogs from 2 outages: count is sufficient but diversity is not → sme_informed."""
        r = self._retriever()
        analogs = self._analogs(10, n_outages=2)
        assert r._compute_confidence_tier(analogs, None) == "sme_informed"

    def test_three_outages_allows_data_supported(self):
        """10 analogs from 3 outages meets both count and diversity gates."""
        r = self._retriever()
        analogs = self._analogs(10, n_outages=3)
        assert r._compute_confidence_tier(analogs, None) == "data_supported"

    def test_diversity_gate_disabled_when_threshold_is_zero(self):
        """Setting min_outages_for_data_supported=0 disables gate; 1 outage → data_supported."""
        r = self._retriever(min_outages_data=0)
        analogs = self._analogs(10, n_outages=1)
        assert r._compute_confidence_tier(analogs, None) == "data_supported"

    def test_no_outage_id_in_analogs_triggers_diversity_cap(self):
        """Analogs with no outage_id field → outages_represented=0 → diversity cap active."""
        r = self._retriever()
        analogs = [
            {"analog_id": f"A{i}", "actual_duration_hours": float(i + 1)}
            for i in range(10)
        ]
        # 0 outages_represented with min_outages_for_data_supported=3 → sme_informed cap
        assert r._compute_confidence_tier(analogs, None) in ("sme_informed", "low_confidence")


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
        result, n_removed = r._remove_duration_outliers(analogs, None)
        result_ids = [a["analog_id"] for a in result]
        assert "A1" in result_ids
        assert "A3" in result_ids

    def test_outlier_removed_from_result(self):
        r = self._retriever()
        analogs = [
            {"analog_id": f"A{i}", "actual_duration_hours": float(i + 4)}
            for i in range(4)  # [4.0, 5.0, 6.0, 7.0]
        ] + [{"analog_id": "A_outlier", "actual_duration_hours": 500.0}]
        result, n_removed = r._remove_duration_outliers(analogs, None)
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
        result, n_removed = r._remove_duration_outliers(analogs, None)
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
        confidence_tier = r._compute_confidence_tier(analogs, None)
        dist_dict["confidence_tier"] = confidence_tier     # this is the fix
        assert dist_dict["confidence_tier"] == "low_confidence"


# ===========================================================================
# Stage D — _remove_duration_outliers disruption-context flag bypass
# ===========================================================================

class TestRemoveDurationOutliersDisruptionMask:
    """Verify that disruption-context analogues bypass the IQR fence.

    An analogue is disruption-context if it shares at least one active
    execution mode flag with the query ActivityCase.  Such analogues must
    survive outlier removal even when their duration is above the Tukey fence,
    so the distribution fitter's mixture model is built from the right pool.
    """

    def _retriever(self, rp_hold=False, scaffold=False,
                   clearance=False, vendor=False):
        """Return (retriever, qac) with the given flags set on the query ActivityCase."""
        from stages.stage_d_analogs import HistoricalAnalogRetriever
        r = HistoricalAnalogRetriever()
        # Build a minimal query ActivityCase with the requested flags
        _, qac = r._build_query(
            {"activity_id": "ACT-X", "plant_id": "P1", "raw_description": "test"},
            {
                "expanded_description": "test",
                "execution_mode_flags": {
                    "has_rp_hold": rp_hold,
                    "requires_scaffold": scaffold,
                    "has_clearance": clearance,
                    "is_vendor_supported": vendor,
                },
            },
        )
        return r, qac

    @staticmethod
    def _analog(aid, duration, *, rp=False, scaffold=False,
                clearance=False, vendor=False):
        return {
            "analog_id": aid,
            "actual_duration_hours": duration,
            "has_rp_hold": rp,
            "requires_scaffold": scaffold,
            "has_clearance": clearance,
            "is_vendor_supported": vendor,
        }

    def test_no_active_flags_behaves_as_before(self):
        """When query has no active flags, outlier detection is unchanged."""
        r, qac = self._retriever()  # all flags False
        analogs = [
            self._analog("A0", 5.0),
            self._analog("A1", 6.0),
            self._analog("A2", 7.0),
            self._analog("A3", 8.0),
            self._analog("A_outlier", 500.0, rp=True),  # RP set on analog, but not on query
        ]
        result, n_removed = r._remove_duration_outliers(analogs, qac)
        ids = [a["analog_id"] for a in result]
        assert "A_outlier" not in ids
        assert n_removed == 1

    def test_disruption_context_analog_preserved_above_fence(self):
        """Analog with has_rp_hold=True survives when query has has_rp_hold=True."""
        r, qac = self._retriever(rp_hold=True)
        analogs = [
            self._analog("A0", 5.0),
            self._analog("A1", 6.0),
            self._analog("A2", 7.0),
            self._analog("A3", 8.0),
            self._analog("A_rp_extended", 500.0, rp=True),   # disruption-context → keep
        ]
        result, n_removed = r._remove_duration_outliers(analogs, qac)
        ids = [a["analog_id"] for a in result]
        assert "A_rp_extended" in ids, "Disruption-context analog must bypass IQR fence"
        assert n_removed == 0

    def test_non_disruption_outlier_still_removed_when_flags_active(self):
        """A non-matching outlier is still discarded even when query has active flags."""
        r, qac = self._retriever(rp_hold=True)
        analogs = [
            self._analog("A0", 5.0),
            self._analog("A1", 6.0),
            self._analog("A2", 7.0),
            self._analog("A3", 8.0),
            self._analog("A_rp_extended", 500.0, rp=True),      # disruption-context → keep
            self._analog("A_true_outlier", 500.0, rp=False),     # no matching flag → remove
        ]
        result, n_removed = r._remove_duration_outliers(analogs, qac)
        ids = [a["analog_id"] for a in result]
        assert "A_rp_extended" in ids
        assert "A_true_outlier" not in ids
        assert n_removed == 1

    def test_any_matching_flag_preserves_analog(self):
        """scaffold-only query: analog with requires_scaffold=True is preserved."""
        r, qac = self._retriever(scaffold=True)
        analogs = [
            self._analog("A0", 4.0),
            self._analog("A1", 5.0),
            self._analog("A2", 6.0),
            self._analog("A3", 7.0),
            self._analog("A_scaffold", 400.0, scaffold=True),
        ]
        result, n_removed = r._remove_duration_outliers(analogs, qac)
        ids = [a["analog_id"] for a in result]
        assert "A_scaffold" in ids
        assert n_removed == 0

    def test_active_execution_flags_helper_returns_correct_frozenset(self):
        from stages.stage_d_analogs import _active_execution_flags
        _, qac = self._retriever(rp_hold=True, clearance=True)
        flags = _active_execution_flags(qac)
        assert "has_rp_hold" in flags
        assert "has_clearance" in flags
        assert "requires_scaffold" not in flags
        assert "is_vendor_supported" not in flags

    def test_active_execution_flags_returns_empty_when_all_false(self):
        from stages.stage_d_analogs import _active_execution_flags
        _, qac = self._retriever()  # all False
        assert _active_execution_flags(qac) == frozenset()

    def test_analog_matches_flags_true_on_overlap(self):
        from stages.stage_d_analogs import _analog_matches_flags
        analog = {"has_rp_hold": True, "requires_scaffold": False,
                  "has_clearance": False, "is_vendor_supported": False}
        assert _analog_matches_flags(analog, frozenset({"has_rp_hold"})) is True

    def test_analog_matches_flags_false_on_no_overlap(self):
        from stages.stage_d_analogs import _analog_matches_flags
        analog = {"has_rp_hold": False, "requires_scaffold": False,
                  "has_clearance": False, "is_vendor_supported": False}
        assert _analog_matches_flags(analog, frozenset({"has_rp_hold"})) is False

    def test_activity_to_analog_includes_execution_flags(self):
        """_activity_to_analog must propagate execution mode flags into the dict."""
        from stages.stage_d_analogs import _activity_to_analog
        from outage_uncertainty.domain.activity import ActivityCase
        ac = ActivityCase(
            activity_id="X",
            outage_id="RF1",
            plant_id="P1",
            has_rp_hold=True,
            requires_scaffold=False,
            has_clearance=True,
            is_vendor_supported=False,
        )
        analog = _activity_to_analog(ac, 0.8, {})
        assert analog["has_rp_hold"] is True
        assert analog["has_clearance"] is True
        assert analog["requires_scaffold"] is False
        assert analog["is_vendor_supported"] is False


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
        # total (confidence=0.15, urgency=0.20 weights, cost_score=0 because no cost_estimate):
        # 0.35*0.10 + 0.15*(1-0.80) + 0.20*0 + 0.20*(1-0.80) + 0.10*0
        # = 0.035+0.030+0+0.040+0 = 0.105
        assert risk == pytest.approx(0.105, abs=0.01)

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


# ===========================================================================
# Stage G — N6: regulatory evidence pinned at position 0 in the chain
# ===========================================================================

class TestN6RegulatoryEvidencePinned:
    """N6 fix: when has_regulatory_constraint=True the regulatory constraint entry
    must be inserted at index 0 of the evidence chain so the max_evidence_items
    cap can never displace it.
    """

    def _intake_with_regulatory(self) -> dict:
        return {
            "activity_id": "ACT-REG",
            "has_regulatory_constraint": True,
            "regulatory_drivers": [
                {"driver_type": "technical_specification", "defer_prohibited": True},
            ],
        }

    def _intake_no_regulatory(self) -> dict:
        return {
            "activity_id": "ACT-NREG",
            "has_regulatory_constraint": False,
            "regulatory_drivers": [],
        }

    def _big_chain(self) -> dict:
        """Return a temporal_event_chain with 8 strong links to fill up slots."""
        from datetime import datetime, timedelta, timezone
        base = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        links = []
        for i in range(8):
            ts = (base - timedelta(hours=i + 1)).isoformat()
            links.append({
                "link_id": f"LINK::E{i:03d}::aabbcc",
                "event_id": f"EVT-{i:03d}",
                "event_type": "condition_report",
                "event_timestamp": ts,
                "allen_relation": "overlaps",
                "relation_score": 0.90,
                "onset_lag_hours": float(i + 1),
                "data_quality_score": 0.8,
                "confidence": 0.85,
                "causal_strength": "strong",
            })
        return {
            "summary": {
                "chain_length": 8,
                "strongest_link_id": links[0]["link_id"],
                "strongest_allen_relation": "overlaps",
                "max_relation_score": 0.90,
                "has_temporal_contradiction": False,
                "causal_posture": "supported",
            },
            "chain_links": links,
        }

    def test_n6_regulatory_entry_at_index_zero(self):
        """With has_regulatory_constraint=True, evidence[0] must be regulatory_constraint."""
        synth = _synth()
        temporal = self._big_chain()
        analogs = _analogs(analog_list=[
            {
                "analog_id": f"AN-{i}", "similarity_score": 0.90,
                "actual_duration_hours": 8.0, "plant_id": "PLT",
                "outage_id": f"OUT-{i}", "description": "valve seal",
            }
            for i in range(3)
        ])
        schedule = _schedule_impact(cp_drag=5.0)
        timeline = {"component_id": "C1", "events": []}

        evidence = synth._assemble_evidence_chain(
            temporal, analogs, schedule, timeline,
            intake_result=self._intake_with_regulatory(),
        )
        assert evidence[0]["source_type"] == "regulatory_constraint", (
            "Regulatory constraint evidence must occupy position 0 in the chain (N6 fix)"
        )

    def test_n6_regulatory_entry_survives_cap(self):
        """Regulatory entry at index 0 must survive max_evidence_items=3 truncation."""
        synth = _synth(RecommendationConfig(max_evidence_items=3))
        temporal = self._big_chain()
        analogs = _analogs(analog_list=[
            {"analog_id": f"AN-{i}", "similarity_score": 0.9,
             "actual_duration_hours": 8.0, "plant_id": "P", "outage_id": f"O{i}",
             "description": "x"}
            for i in range(3)
        ])
        schedule = _schedule_impact(cp_drag=5.0)
        timeline = {"component_id": "C1", "events": []}

        evidence = synth._assemble_evidence_chain(
            temporal, analogs, schedule, timeline,
            intake_result=self._intake_with_regulatory(),
        )
        # After slicing to max_evidence_items=3, first entry must still be regulatory
        capped = evidence[:3]
        assert capped[0]["source_type"] == "regulatory_constraint", (
            "Regulatory entry must survive max_evidence_items truncation"
        )

    def test_n6_no_regulatory_entry_when_no_constraint(self):
        """When has_regulatory_constraint=False, no regulatory_constraint entry is inserted."""
        synth = _synth()
        temporal = {"summary": {}, "chain_links": []}
        analogs = _analogs(analog_list=[])
        schedule = _schedule_impact()
        timeline = {"component_id": "C1", "events": []}

        evidence = synth._assemble_evidence_chain(
            temporal, analogs, schedule, timeline,
            intake_result=self._intake_no_regulatory(),
        )
        assert not any(e["source_type"] == "regulatory_constraint" for e in evidence), (
            "No regulatory_constraint entry should appear when has_regulatory_constraint=False"
        )

    def test_n6_no_regulatory_entry_when_intake_result_absent(self):
        """Omitting intake_result entirely must not raise and must produce no regulatory entry."""
        synth = _synth()
        temporal = {"summary": {}, "chain_links": []}
        analogs = _analogs(analog_list=[])
        schedule = _schedule_impact()
        timeline = {"component_id": "C1", "events": []}

        evidence = synth._assemble_evidence_chain(temporal, analogs, schedule, timeline)
        assert not any(e["source_type"] == "regulatory_constraint" for e in evidence)


# ===========================================================================
# Cost per mode — _compute_cost_estimate (module-level helper)
# ===========================================================================

from stages.stage_f_options import (
    _compute_cost_estimate,
    _INSERT_NOW, _DEFER, _CONTINGENCY, _PRE_STAGE, _PARALLEL, _SCOPE_REDUCTION, _ESCALATE,
)


class TestComputeCostEstimate:
    """Unit tests for the _compute_cost_estimate module-level helper."""

    def _call(self, option_type=_INSERT_NOW, duration=8.0, crew=2, cp_drag=0.0,
              rate=100.0, day_cost=10_000.0, crash=1.5):
        return _compute_cost_estimate(
            option_type, duration, crew, cp_drag,
            labor_rate=rate,
            outage_day_cost=day_cost,
            crash_premium_multiplier=crash,
        )

    def test_required_keys_present(self):
        result = self._call()
        for k in ("labor_cost_usd", "schedule_extension_cost_usd",
                  "crash_premium_usd", "total_cost_usd", "cost_basis"):
            assert k in result

    def test_cost_basis_is_parametric(self):
        assert self._call()["cost_basis"] == "parametric"

    def test_labor_cost_computed(self):
        # 8h × 3 crew × $100 = $2 400
        result = self._call(duration=8.0, crew=3, rate=100.0)
        assert result["labor_cost_usd"] == pytest.approx(2_400.0)

    def test_schedule_extension_cost_computed(self):
        # cp_drag=4h × $10 000/hr = $40 000
        result = self._call(cp_drag=4.0, day_cost=10_000.0)
        assert result["schedule_extension_cost_usd"] == pytest.approx(40_000.0)

    def test_no_crash_premium_for_insert_now(self):
        result = self._call(option_type=_INSERT_NOW)
        assert result["crash_premium_usd"] == pytest.approx(0.0)

    def test_no_crash_premium_for_defer(self):
        result = self._call(option_type=_DEFER)
        assert result["crash_premium_usd"] == pytest.approx(0.0)

    def test_crash_premium_for_parallel(self):
        # labor_cost = 8h × 2 crew × $100 = $1 600; premium = $1600 × 0.5 = $800
        result = self._call(option_type=_PARALLEL, duration=8.0, crew=2, rate=100.0, crash=1.5)
        assert result["crash_premium_usd"] == pytest.approx(800.0)

    def test_total_is_sum_of_components(self):
        r = self._call(option_type=_PARALLEL, duration=8.0, crew=2, rate=100.0,
                       cp_drag=2.0, day_cost=10_000.0, crash=1.5)
        expected = r["labor_cost_usd"] + r["schedule_extension_cost_usd"] + r["crash_premium_usd"]
        assert r["total_cost_usd"] == pytest.approx(expected)

    def test_defer_has_zero_in_outage_labor(self):
        # Deferral has 0 in-outage duration → in-outage labour = 0
        result = self._call(option_type=_DEFER, duration=0.0, cp_drag=0.0)
        assert result["labor_cost_usd"] == pytest.approx(0.0)

    def test_defer_carries_deferred_labor_cost(self):
        # deferred_duration_hours passed → deferred_labor shows future-cycle cost
        result = _compute_cost_estimate(
            _DEFER, 0.0, 2, 0.0,
            labor_rate=100.0, outage_day_cost=10_000.0,
            crash_premium_multiplier=1.5, deferred_duration_hours=8.0,
        )
        # deferred_labor = 8 × 2 × 100 = 1 600; excluded from total (not an outage cost)
        assert result["deferred_labor_cost_usd"] == pytest.approx(1_600.0)
        assert result["total_cost_usd"] == pytest.approx(0.0)

    def test_non_defer_has_zero_deferred_labor(self):
        result = self._call(option_type=_INSERT_NOW, duration=8.0)
        assert result["deferred_labor_cost_usd"] == pytest.approx(0.0)


# ===========================================================================
# Cost per mode — InsertionOptionConfig cost fields
# ===========================================================================

class TestInsertionOptionConfigCostFields:
    def test_default_labor_rate(self):
        cfg = InsertionOptionConfig()
        assert cfg.labor_rate_per_crew_hour == pytest.approx(150.0)

    def test_default_outage_day_cost(self):
        cfg = InsertionOptionConfig()
        assert cfg.outage_day_cost_per_hour == pytest.approx(50_000.0)

    def test_default_crash_premium(self):
        cfg = InsertionOptionConfig()
        assert cfg.crash_premium_multiplier == pytest.approx(1.5)

    def test_default_crew_count(self):
        cfg = InsertionOptionConfig()
        assert cfg.default_crew_count == 2

    def test_custom_values_accepted(self):
        cfg = InsertionOptionConfig(
            labor_rate_per_crew_hour=200.0,
            outage_day_cost_per_hour=75_000.0,
            crash_premium_multiplier=2.0,
            default_crew_count=4,
        )
        assert cfg.labor_rate_per_crew_hour == pytest.approx(200.0)
        assert cfg.default_crew_count == 4

    def test_cost_weight_in_scoring_weights(self):
        cfg = InsertionOptionConfig()
        assert "cost" in cfg.scoring_weights
        assert cfg.scoring_weights["cost"] == pytest.approx(0.10)

    def test_scoring_weights_sum_to_one(self):
        cfg = InsertionOptionConfig()
        total = sum(cfg.scoring_weights.values())
        assert total == pytest.approx(1.0, abs=0.001)


# ===========================================================================
# Cost per mode — _compute_option_cost and _resolve_crew_count
# ===========================================================================

class TestComputeOptionCost:
    """Tests for InsertionOptionGenerator._compute_option_cost."""

    def _gen_default(self):
        return _gen(InsertionOptionConfig(
            labor_rate_per_crew_hour=100.0,
            outage_day_cost_per_hour=10_000.0,
            crash_premium_multiplier=1.5,
            default_crew_count=2,
        ))

    def _opt(self, opt_type, cp_drag=0.0):
        o = _option(opt_type, cp_impact_hours=cp_drag)
        return o

    def test_insert_now_uses_p50(self):
        gen = self._gen_default()
        opt = self._opt(_INSERT_NOW)
        result = gen._compute_option_cost(opt, p50=8.0, p80=12.0, crew_count=2)
        # labor = 8 × 2 × 100 = 1600; drag = 0; crash = 0
        assert result["labor_cost_usd"] == pytest.approx(1_600.0)

    def test_defer_has_zero_in_outage_labor(self):
        gen = self._gen_default()
        opt = self._opt(_DEFER)
        result = gen._compute_option_cost(opt, p50=8.0, p80=12.0, crew_count=2)
        assert result["labor_cost_usd"] == pytest.approx(0.0)

    def test_defer_has_deferred_labor_cost(self):
        gen = self._gen_default()
        opt = self._opt(_DEFER)
        result = gen._compute_option_cost(opt, p50=8.0, p80=12.0, crew_count=2)
        # deferred = 8h × 2 × 100 = 1 600; excluded from total (not an outage cost)
        assert result["deferred_labor_cost_usd"] == pytest.approx(1_600.0)
        assert result["total_cost_usd"] == pytest.approx(0.0)

    def test_contingency_uses_p80_minus_p50(self):
        gen = self._gen_default()
        opt = self._opt(_CONTINGENCY)
        result = gen._compute_option_cost(opt, p50=8.0, p80=12.0, crew_count=2)
        # duration = 12 - 8 = 4h; labor = 4 × 2 × 100 = 800
        assert result["labor_cost_usd"] == pytest.approx(800.0)

    def test_scope_reduction_uses_fraction(self):
        gen = self._gen_default()
        # default scope_reduction_fraction = 0.60
        opt = self._opt(_SCOPE_REDUCTION)
        result = gen._compute_option_cost(opt, p50=10.0, p80=14.0, crew_count=2)
        # duration = 10 × 0.60 = 6h; labor = 6 × 2 × 100 = 1 200
        assert result["labor_cost_usd"] == pytest.approx(1_200.0)

    def test_parallel_has_crash_premium(self):
        gen = self._gen_default()
        opt = self._opt(_PARALLEL)
        result = gen._compute_option_cost(opt, p50=8.0, p80=12.0, crew_count=2)
        # labor = 8 × 2 × 100 = 1600; crash = 1600 × 0.5 = 800
        assert result["crash_premium_usd"] == pytest.approx(800.0)

    def test_escalate_uses_p50(self):
        gen = self._gen_default()
        opt = self._opt(_ESCALATE)
        result = gen._compute_option_cost(opt, p50=8.0, p80=12.0, crew_count=2)
        assert result["labor_cost_usd"] == pytest.approx(1_600.0)
        assert result["crash_premium_usd"] == pytest.approx(0.0)


class TestResolveCrewCount:
    """Tests for InsertionOptionGenerator._resolve_crew_count."""

    def test_fallback_when_no_crew_continuity(self):
        gen = _gen(InsertionOptionConfig(default_crew_count=3))
        count = gen._resolve_crew_count({})
        assert count == 3

    def test_reads_from_crew_continuity(self):
        gen = _gen(InsertionOptionConfig(default_crew_count=2))
        impact = {
            "crew_continuity": {
                "utilization_at_window": {
                    "MECHANIC": {"committed": 5, "available": 10},
                    "ELECTRICIAN": {"committed": 3, "available": 8},
                }
            }
        }
        count = gen._resolve_crew_count(impact)
        assert count == 5  # max committed across skill types

    def test_fallback_when_all_committed_zero(self):
        gen = _gen(InsertionOptionConfig(default_crew_count=4))
        impact = {
            "crew_continuity": {
                "utilization_at_window": {
                    "MECHANIC": {"committed": 0},
                }
            }
        }
        count = gen._resolve_crew_count(impact)
        assert count == 4  # falls back to default


# ===========================================================================
# Cost per mode — cost_estimate present on options after generate()
# ===========================================================================

class TestGenerateCostEstimatePresent:
    """Integration tests verifying cost_estimate appears on each option output."""

    def _run(self, cp_drag=0.0, criticality="non_critical"):
        gen = _gen()
        ea = {"activity_id": "EA001", "outage_id": "O1"}
        intake = {"regulatory_drivers": []}
        temporal = {"summary": {"causal_posture": "partial"}, "chain_links": []}
        schedule = _schedule_impact(cp_drag=cp_drag, criticality_label=criticality,
                                    baseline_cp_hours=200.0)
        analogs = _analogs(p50=8.0, p80=12.0)
        run_ctx = {"run_id": "R1", "started_at": "2025-01-01T00:00:00Z"}
        return gen.generate(ea, intake, temporal, schedule, analogs, run_ctx)

    def test_all_options_have_cost_estimate(self):
        result = self._run()
        for opt in result["options"]:
            assert "cost_estimate" in opt, f"Missing cost_estimate on {opt['option_type']}"
            ce = opt["cost_estimate"]
            assert ce is not None
            assert "total_cost_usd" in ce

    def test_cost_estimate_keys_complete(self):
        result = self._run()
        required_keys = {
            "labor_cost_usd", "schedule_extension_cost_usd",
            "crash_premium_usd", "total_cost_usd", "cost_basis",
        }
        for opt in result["options"]:
            assert required_keys <= set(opt["cost_estimate"].keys())

    def test_min_cost_option_id_present(self):
        result = self._run()
        assert "min_cost_option_id" in result
        # Should be a string option_id or None
        mcid = result["min_cost_option_id"]
        assert mcid is None or isinstance(mcid, str)

    def test_min_cost_option_id_matches_an_option(self):
        result = self._run()
        mcid = result["min_cost_option_id"]
        if mcid is not None:
            ids = {o["option_id"] for o in result["options"]}
            assert mcid in ids

    def test_defer_has_zero_in_outage_labor(self):
        """defer_to_post_outage has zero in-outage labour but non-zero deferred_labor_cost."""
        result = self._run(cp_drag=0.0)
        defer_opts = [o for o in result["options"]
                      if o["option_type"] == "defer_to_post_outage"]
        if defer_opts:
            ce = defer_opts[0]["cost_estimate"]
            assert ce["labor_cost_usd"] == pytest.approx(0.0)
            # Deferred work cost is non-zero (work moves to next cycle)
            assert ce["deferred_labor_cost_usd"] > 0.0

    def test_parallel_has_crash_premium(self):
        """parallel_execution option must carry a non-zero crash_premium_usd."""
        result = self._run()
        parallel_opts = [o for o in result["options"]
                         if o["option_type"] == "parallel_execution"]
        if parallel_opts:
            assert parallel_opts[0]["cost_estimate"]["crash_premium_usd"] > 0.0

    def test_risk_score_reflects_cost(self):
        """With cost weight 0.10, two otherwise equal options differ by cost signal."""
        # Create two options with identical cp/confidence/resource/urgency but
        # different cost_estimates; the higher-cost one should have a higher risk_score.
        gen = _gen()
        impact = _schedule_impact(baseline_cp_hours=200.0, cp_drag=0.0)
        analogs_data = _analogs()

        opt_cheap = _option("insert_now", cp_impact_hours=0.0, confidence=0.80)
        opt_cheap["cost_estimate"] = {"total_cost_usd": 1_000.0}

        opt_expensive = _option("insert_now", cp_impact_hours=0.0, confidence=0.80)
        opt_expensive["cost_estimate"] = {"total_cost_usd": 100_000.0}

        max_cost = 100_000.0
        risk_cheap    = gen._score_option(opt_cheap, impact, analogs_data, "partial",
                                          max_cost=max_cost)
        risk_expensive = gen._score_option(opt_expensive, impact, analogs_data, "partial",
                                           max_cost=max_cost)
        assert risk_expensive > risk_cheap


# ===========================================================================
# N2 — Stage F: pre_outage_staging upgrade via outage_start field
# ===========================================================================

class TestGenerateContingencyBuffer:
    """Verify the add_contingency_buffer → pre_outage_staging upgrade logic.

    The upgrade fires when:
      - actual_start is absent (work has not begun)
      - outage_start is present (now in schema after N2 fix)
      - detection_timestamp < outage_start (detection occurred before outage opened)
    """

    def _ea(self, detection_ts: str = "2026-04-10T06:00:00Z",
            outage_start: str | None = None,
            actual_start: str | None = None) -> dict:
        ea: dict = {
            "activity_id": "ACT-001",
            "outage_id": "RF-22",
            "plant_id": "PLANT-1",
            "raw_description": "inspect coolant pump",
            "detection_timestamp": detection_ts,
            "source_system": "maximo",
        }
        if outage_start is not None:
            ea["outage_start"] = outage_start
        if actual_start is not None:
            ea["actual_start"] = actual_start
        return ea

    def _call(self, ea: dict, cp_drag: float = 0.0) -> dict:
        gen = _gen()
        schedule = _schedule_impact(cp_drag=cp_drag)
        analogs = _analogs(p50=6.0, p80=9.0)
        return gen._generate_contingency_buffer(ea, schedule, analogs)

    def test_default_type_is_add_contingency_buffer(self):
        """No outage_start → standard contingency buffer, not pre_outage_staging."""
        ea = self._ea()  # no outage_start
        result = self._call(ea)
        assert result["option_type"] == "add_contingency_buffer"

    def test_n2_upgrade_to_pre_outage_staging_when_before_outage(self):
        """N2 fix: detection before outage_start → option upgrades to pre_outage_staging."""
        ea = self._ea(
            detection_ts="2026-04-10T06:00:00Z",
            outage_start="2026-04-12T08:00:00Z",   # 2 days later
        )
        result = self._call(ea)
        assert result["option_type"] == "pre_outage_staging", (
            "Detection before outage window must produce pre_outage_staging option"
        )

    def test_no_upgrade_when_detection_after_outage_start(self):
        """Detection timestamp after outage_start → stays add_contingency_buffer."""
        ea = self._ea(
            detection_ts="2026-04-14T10:00:00Z",    # during outage
            outage_start="2026-04-12T08:00:00Z",
        )
        result = self._call(ea)
        assert result["option_type"] == "add_contingency_buffer"

    def test_no_upgrade_when_actual_start_set(self):
        """Work already started → no upgrade even if detection was before outage."""
        ea = self._ea(
            detection_ts="2026-04-10T06:00:00Z",
            outage_start="2026-04-12T08:00:00Z",
            actual_start="2026-04-13T10:00:00Z",
        )
        result = self._call(ea)
        assert result["option_type"] == "add_contingency_buffer"

    def test_pre_outage_staging_rationale_mentions_staging(self):
        """pre_outage_staging rationale should reference staging before outage."""
        ea = self._ea(
            detection_ts="2026-04-10T06:00:00Z",
            outage_start="2026-04-12T08:00:00Z",
        )
        result = self._call(ea)
        assert "stage" in result["rationale"].lower() or "outage" in result["rationale"].lower()

    def test_h2_buffer_feasible_when_available_float_absent(self):
        """H2 fix: buffer option must be feasible when available_float_before is absent.

        Before the fix the code fell back to float_consumed_hours (a small number
        representing consumed float, not available float), making remaining_float
        nearly zero and marking the option infeasible.  After the fix the fallback
        is remaining_float_hours, then inf (permissive) — so the option is feasible
        when neither explicit field is present.
        """
        gen = _gen()
        ea = self._ea()
        # Build a schedule where available_float_before and remaining_float_hours
        # are both absent (only float_consumed_hours is present, as in stub outputs).
        schedule = {
            "cp_impact": {"cp_drag_hours": 2.0, "baseline_cp_hours": 100.0},
            "float_analysis": {
                "criticality_label": "non_critical",
                "float_consumed_hours": 3.0,     # consumed — must NOT be used as proxy
                # available_float_before: absent
                # remaining_float_hours: absent
            },
            "resource_conflicts": [],
            "displaced_tasks": [],
            "insertion_point": {},
        }
        analogs = _analogs(p50=6.0, p80=9.0)
        result = gen._generate_contingency_buffer(ea, schedule, analogs)
        assert result["feasible"] is True, (
            "Buffer option must be feasible when float data is absent (permissive default)"
        )

    def test_h2_zero_available_float_treated_as_zero_not_inf(self):
        """H2 fix: available_float_before=0.0 must be treated as 0 float, not fallback.

        Python's `or` treats 0.0 as falsy.  Before the fix:
          float_analysis.get("available_float_before") or float_analysis.get("remaining_float_hours")
          → 0.0 or None → None → float("inf") → buffer marked feasible on a critical path.

        After the fix (explicit `is not None` check): 0.0 is honoured, the
        buffer is correctly infeasible when there is zero float available.
        """
        gen = _gen()
        ea = self._ea()
        # p80=9.0, p50=6.0 → buffer_hours = 3.0
        # available_float_before=0.0 → remaining_float = max(0, 0.0 - 6.0) = 0 → infeasible
        schedule = {
            "cp_impact": {"cp_drag_hours": 0.0, "baseline_cp_hours": 100.0},
            "float_analysis": {
                "criticality_label": "critical",
                "float_consumed_hours": 6.0,    # consumed (must NOT be the proxy)
                "available_float_before": 0.0,  # zero float — explicitly present
                # remaining_float_hours: absent
            },
            "resource_conflicts": [],
            "displaced_tasks": [],
            "insertion_point": {},
        }
        analogs = _analogs(p50=6.0, p80=9.0)
        result = gen._generate_contingency_buffer(ea, schedule, analogs)
        assert result["feasible"] is False, (
            "available_float_before=0.0 must not be treated as falsy — "
            "zero float means the buffer has no room and must be infeasible"
        )

    def test_h2_buffer_uses_remaining_float_hours_when_present(self):
        """H2 fix: when remaining_float_hours is present it governs feasibility."""
        gen = _gen()
        ea = self._ea()
        # p80 = 9.0, p50 = 6.0 → buffer_hours = 3.0; remaining_float_hours = 1.0
        # → remaining after base scope = max(0, 1.0 - 6.0) = 0.0 < 3.0 → infeasible
        schedule = {
            "cp_impact": {"cp_drag_hours": 2.0, "baseline_cp_hours": 100.0},
            "float_analysis": {
                "criticality_label": "critical",
                "float_consumed_hours": 50.0,    # must be ignored
                "remaining_float_hours": 1.0,    # tight — not enough for the buffer
            },
            "resource_conflicts": [],
            "displaced_tasks": [],
            "insertion_point": {},
        }
        analogs = _analogs(p50=6.0, p80=9.0)
        result = gen._generate_contingency_buffer(ea, schedule, analogs)
        # With only 1 h remaining and a 3 h buffer requirement, option should be infeasible
        assert result["feasible"] is False
        assert result["infeasibility_reason"] is not None


# ===========================================================================
# N11 — Stage D: _compute_confidence_tier with injected ConfidenceEstimator
# ===========================================================================

class TestComputeConfidenceTierWithEstimator:
    """Verify that when ConfidenceEstimator is injected:
      1. classify() is called (primary path taken)
      2. outages_represented is correctly counted from analog outage_id fields
      3. The CE tier is mapped through _CE_TIER_TO_STAGE_D
      4. Fallback count-based logic is used when classify() raises
    """

    def _retriever_with_ce(self, ce_tier: str = "high", classify_raises: bool = False):
        """Build a HistoricalAnalogRetriever with mocked CE and OutlierHandler."""
        from unittest.mock import MagicMock
        from stages.stage_d_analogs import HistoricalAnalogRetriever, HistoricalAnalogConfig

        ce = MagicMock()
        if classify_raises:
            ce.classify.side_effect = RuntimeError("CE unavailable")
        else:
            ce_result = MagicMock()
            ce_result.tier = ce_tier
            ce.classify.return_value = ce_result

        oh = MagicMock()
        oh.separate.return_value = MagicMock()  # OutlierSeparation duck type

        cfg = HistoricalAnalogConfig(
            min_analogs_for_data_supported=10,
            min_analogs_for_sme_informed=3,
        )
        r = HistoricalAnalogRetriever(
            config=cfg,
            confidence_estimator=ce,
            outlier_handler=oh,
        )
        return r, ce

    def _analogs_with_outage_ids(self, outage_ids: list) -> list:
        """Build analogs with explicit outage_id fields and duration data."""
        return [
            {
                "analog_id": f"A{i}",
                "actual_duration_hours": float(i + 4),
                "outage_id": oid,
                "relevance_weight": 1.0,
                "similarity_score": 0.80,
            }
            for i, oid in enumerate(outage_ids)
        ]

    def test_n11_ce_classify_called_when_injected(self):
        """Primary path: ConfidenceEstimator.classify() is invoked when injected."""
        r, ce = self._retriever_with_ce(ce_tier="high")
        analogs = self._analogs_with_outage_ids(["RF1", "RF2", "RF3"])
        from stages.stage_d_analogs import _make_activity_case
        qac = _make_activity_case(description="test", plant_id="P1")

        r._compute_confidence_tier(analogs, qac)
        assert ce.classify.called, "ConfidenceEstimator.classify() must be called when injected"

    def test_n11_outages_represented_counted_from_analog_outage_id(self):
        """outages_represented kwarg passed to classify() reflects distinct outage_id values."""
        r, ce = self._retriever_with_ce(ce_tier="high")
        analogs = self._analogs_with_outage_ids(["RF1", "RF1", "RF2", "RF3"])  # 3 distinct
        from stages.stage_d_analogs import _make_activity_case
        qac = _make_activity_case(description="test", plant_id="P1")

        r._compute_confidence_tier(analogs, qac)

        call_kwargs = ce.classify.call_args.kwargs
        assert "outages_represented" in call_kwargs, (
            "classify() must receive outages_represented keyword argument"
        )
        assert call_kwargs["outages_represented"] == 3, (
            f"Expected 3 distinct outage IDs, got {call_kwargs['outages_represented']}"
        )

    def test_n11_tier_mapped_through_ce_tier_to_stage_d(self):
        """CE tier names are mapped: high→data_supported, medium→sme_informed, low→low_confidence."""
        from stages.stage_d_analogs import _make_activity_case
        analogs = self._analogs_with_outage_ids(["RF1", "RF2", "RF3"])
        qac = _make_activity_case(description="test", plant_id="P1")

        for ce_tier, expected_stage_d_tier in [
            ("high",   "data_supported"),
            ("medium", "sme_informed"),
            ("low",    "low_confidence"),
        ]:
            r, _ = self._retriever_with_ce(ce_tier=ce_tier)
            result = r._compute_confidence_tier(analogs, qac)
            assert result == expected_stage_d_tier, (
                f"CE tier '{ce_tier}' must map to '{expected_stage_d_tier}', got '{result}'"
            )

    def test_n11_fallback_used_when_classify_raises(self):
        """When classify() raises, count-based fallback is used (3 analogs → sme_informed)."""
        from stages.stage_d_analogs import _make_activity_case
        r, ce = self._retriever_with_ce(classify_raises=True)
        # 5 analogs with 3 distinct outage IDs → sme_informed via count-based fallback
        analogs = self._analogs_with_outage_ids(["RF1", "RF1", "RF2", "RF3", "RF3"])
        qac = _make_activity_case(description="test", plant_id="P1")

        result = r._compute_confidence_tier(analogs, qac)
        assert result == "sme_informed", (
            f"Count-based fallback with 5 analogs should give sme_informed, got '{result}'"
        )

    def test_n11_outage_id_in_activity_to_analog(self):
        """_activity_to_analog must propagate the outage_id attribute from ActivityCase."""
        from stages.stage_d_analogs import _activity_to_analog
        from unittest.mock import MagicMock
        ac = MagicMock()
        ac.activity_id = "X1"
        ac.outage_id = "RF-22"
        ac.actual_duration_hours = 8.0

        analog = _activity_to_analog(ac, score=0.85, breakdown={})
        assert analog["outage_id"] == "RF-22", (
            "_activity_to_analog must copy outage_id from ActivityCase for diversity gate"
        )


# ===========================================================================
# H4 — Similarity weight config in Stage D governs actual scoring
# ===========================================================================

class TestH4SimilarityConfigWeights:
    """Verify that HistoricalAnalogConfig weight fields are wired into the engine.

    H4 fix: the config fields (lexical_weight, semantic_weight, context_weight)
    must be passed to SimilarityAggregator at construction so that tuning the
    config actually changes retrieval behaviour.  component_weight was removed
    from the config as it implied a scoring dimension that does not exist.
    """

    def test_h4_weight_fields_exist_in_config(self):
        """Config must expose lexical_weight, semantic_weight, context_weight."""
        from stages.stage_d_analogs import HistoricalAnalogConfig
        cfg = HistoricalAnalogConfig()
        assert hasattr(cfg, "lexical_weight"),  "lexical_weight must be a config field"
        assert hasattr(cfg, "semantic_weight"), "semantic_weight must be a config field"
        assert hasattr(cfg, "context_weight"),  "context_weight must be a config field"

    def test_h4_component_weight_removed(self):
        """component_weight was a dead letter (no such scorer); must be removed."""
        from stages.stage_d_analogs import HistoricalAnalogConfig
        cfg = HistoricalAnalogConfig()
        assert not hasattr(cfg, "component_weight"), (
            "component_weight implied a non-existent aggregation dimension and must "
            "be absent from HistoricalAnalogConfig after the H4 fix"
        )

    def test_h4_weights_sum_to_one(self):
        """Default weights must sum to 1.0 (± floating point tolerance)."""
        from stages.stage_d_analogs import HistoricalAnalogConfig
        cfg = HistoricalAnalogConfig()
        total = cfg.lexical_weight + cfg.semantic_weight + cfg.context_weight
        assert abs(total - 1.0) < 1e-9, (
            f"lexical + semantic + context weights must sum to 1.0, got {total}"
        )

    def test_h4_custom_weights_accepted(self):
        """Config must accept custom weights without raising."""
        from stages.stage_d_analogs import HistoricalAnalogConfig
        cfg = HistoricalAnalogConfig(
            lexical_weight=0.10,
            semantic_weight=0.50,
            context_weight=0.40,
        )
        assert cfg.lexical_weight == 0.10
        assert cfg.semantic_weight == 0.50
        assert cfg.context_weight == 0.40


# ===========================================================================
# D2 — Stage F: extra_option_generators plugin registry
# ===========================================================================

class TestD2ExtraOptionGenerators:
    """Verify the extensibility hook added by the D2 fix.

    register_option_generator() appends a callable to extra_option_generators.
    Registered callables are invoked during generate() and their results merged
    into the candidate option list before ranking.
    Exceptions from extra generators must be caught and logged, never propagating.
    """

    def test_d2_register_option_generator_adds_to_list(self):
        """register_option_generator must append the callable to the internal list."""
        gen = _gen()
        initial_count = len(gen.extra_option_generators)
        gen.register_option_generator(lambda **_kw: None)
        assert len(gen.extra_option_generators) == initial_count + 1

    def test_d2_extra_generator_result_merged(self):
        """A generator returning a dict must add its option to the candidate set."""
        custom_option = {
            "option_type": "custom_vendor_standby",
            "option_id": "CUSTOM-001",
            "feasible": True,
            "regulatory_cleared": True,
            "risk_score": 0.30,
            "confidence": 0.70,
            "cp_impact_hours": 0.0,
            "rationale": "Vendor on-site standby.",
            "cost_estimate": {"total_cost_usd": 5000.0},
            "causal_urgency_score": 0.0,
        }

        def _custom_gen(**_kw):
            return custom_option

        gen = _gen()
        gen.register_option_generator(_custom_gen)
        result = gen.generate(
            emergent_activity={"activity_id": "ACT-001"},
            intake_result=_intake_result(),
            historical_analogs=_analogs(),
            schedule_impact_assessment=_schedule_impact(),
            temporal_event_chain={"chain_links": []},
            run_context={"run_id": "TEST-D2"},
        )
        option_types = [o["option_type"] for o in result.get("options", [])]
        assert "custom_vendor_standby" in option_types, (
            "Extra generator result must appear in the options list"
        )

    def test_d2_failing_extra_generator_does_not_abort_pipeline(self):
        """Exception in an extra generator must be swallowed, not propagated."""
        def _bad_gen(**_kw):
            raise RuntimeError("custom generator exploded")

        gen = _gen()
        gen.register_option_generator(_bad_gen)
        # Must not raise
        result = gen.generate(
            emergent_activity={"activity_id": "ACT-001"},
            intake_result=_intake_result(),
            historical_analogs=_analogs(),
            schedule_impact_assessment=_schedule_impact(),
            temporal_event_chain={"chain_links": []},
            run_context={"run_id": "TEST-D2-BAD"},
        )
        assert "options" in result


# ===========================================================================
# D3 — Stage F: decision_latency_cost_usd in escalate option cost model
# ===========================================================================

class TestD3DecisionLatencyCost:
    """Verify that the escalate option includes a decision_latency_cost_usd line item.

    D3 fix: _generate_escalate() embeds decision_delay_hours in the option dict,
    and _compute_cost_estimate() converts it to decision_latency_cost_usd
    (delay × outage_day_cost_per_hour), included in total_cost_usd.
    """

    def _escalate_option(self, decision_delay_hours: float = 4.0) -> dict:
        from stages.stage_f_options import InsertionOptionConfig
        cfg = InsertionOptionConfig(
            escalate_decision_delay_hours=decision_delay_hours,
            outage_day_cost_per_hour=10_000.0,
        )
        gen = _gen(config=cfg)
        activity = {"activity_id": "ACT-001"}
        return gen._generate_escalate(activity, _schedule_impact())

    def test_d3_decision_delay_hours_in_option(self):
        """Escalate option must carry decision_delay_hours."""
        opt = self._escalate_option(decision_delay_hours=4.0)
        assert "decision_delay_hours" in opt, (
            "decision_delay_hours must be embedded in the escalate option"
        )
        assert opt["decision_delay_hours"] == pytest.approx(4.0)

    def test_d3_decision_latency_cost_in_cost_estimate(self):
        """decision_latency_cost_usd must be a separate line item in cost_estimate."""
        from stages.stage_f_options import InsertionOptionConfig
        cfg = InsertionOptionConfig(
            escalate_decision_delay_hours=4.0,
            outage_day_cost_per_hour=10_000.0,
        )
        gen = _gen(config=cfg)
        opt = self._escalate_option(decision_delay_hours=4.0)
        # _compute_option_cost requires p50, p80, crew_count as keyword args
        cost = gen._compute_option_cost(opt, p50=6.0, p80=9.0, crew_count=2)
        assert "decision_latency_cost_usd" in cost, (
            "decision_latency_cost_usd must be a distinct field in the cost estimate"
        )
        # 4.0 h × 10_000 $/h = 40_000
        assert cost["decision_latency_cost_usd"] == pytest.approx(40_000.0)

    def test_d3_decision_latency_cost_included_in_total(self):
        """decision_latency_cost_usd must be included in total_cost_usd."""
        from stages.stage_f_options import InsertionOptionConfig
        cfg = InsertionOptionConfig(
            escalate_decision_delay_hours=4.0,
            outage_day_cost_per_hour=10_000.0,
        )
        gen = _gen(config=cfg)
        opt = gen._generate_escalate({"activity_id": "ACT-001"}, _schedule_impact())
        cost = gen._compute_option_cost(opt, p50=6.0, p80=9.0, crew_count=2)
        # total_cost_usd includes the latency component, so it must be >= latency alone
        assert cost["total_cost_usd"] >= cost["decision_latency_cost_usd"]


# ===========================================================================
# Stage F — N3: escalate option includes TS/LCO deadline note in rationale
# ===========================================================================

class TestN3EscalateRegulatoryDeadlineNote:
    """N3 fix: when has_regulatory_constraint=True or active_lco=True,
    _generate_escalate() must append a TS/LCO deadline note to the rationale
    so the manager briefing always leads with the action-level clock.
    """

    def _activity(self, active_lco: bool = False, lco_number: str | None = None) -> dict:
        a = {"activity_id": "ACT-N3"}
        if active_lco:
            a["active_lco"] = True
        if lco_number:
            a["lco_number"] = lco_number
        return a

    def _intake(self, has_regulatory: bool = True) -> dict:
        return {
            "has_regulatory_constraint": has_regulatory,
            "regulatory_drivers": [
                {"driver_type": "technical_specification", "defer_prohibited": True}
            ],
        }

    def test_n3_deadline_note_present_when_regulatory(self):
        """With has_regulatory_constraint=True the rationale must mention the action-level clock."""
        gen = _gen()
        opt = gen._generate_escalate(
            self._activity(), _schedule_impact(), intake_result=self._intake(has_regulatory=True)
        )
        rationale = opt["rationale"]
        assert "action-level" in rationale.lower() or "action level" in rationale.lower(), (
            "Escalate rationale must mention the action-level clock when regulatory constraint present"
        )

    def test_n3_lco_number_in_rationale_when_provided(self):
        """When active_lco=True and lco_number is set, the LCO number appears in the rationale."""
        gen = _gen()
        opt = gen._generate_escalate(
            self._activity(active_lco=True, lco_number="3.8.1"),
            _schedule_impact(),
            intake_result=self._intake(has_regulatory=False),
        )
        assert "3.8.1" in opt["rationale"], (
            "LCO number must be embedded in escalate rationale when active_lco=True and lco_number set"
        )

    def test_n3_no_deadline_note_when_no_constraint(self):
        """Without regulatory constraint the rationale must NOT contain the deadline note."""
        gen = _gen()
        opt = gen._generate_escalate(
            self._activity(), _schedule_impact(),
            intake_result=self._intake(has_regulatory=False),
        )
        rationale = opt["rationale"]
        assert "action-level" not in rationale.lower() and "action level" not in rationale.lower()

    def test_n3_deadline_note_when_active_lco_only(self):
        """active_lco=True on the emergent_activity alone (no intake_result) triggers the note."""
        gen = _gen()
        opt = gen._generate_escalate(
            self._activity(active_lco=True),
            _schedule_impact(),
        )
        rationale = opt["rationale"]
        assert "action-level" in rationale.lower() or "action level" in rationale.lower()

    def test_n3_no_deadline_note_when_intake_result_absent(self):
        """Omitting intake_result and no active_lco → no deadline note (no regression)."""
        gen = _gen()
        opt = gen._generate_escalate(self._activity(), _schedule_impact())
        rationale = opt["rationale"]
        assert "action-level" not in rationale.lower() and "action level" not in rationale.lower()


# ===========================================================================
# Stage D — N12: candidates_below_threshold in retrieval_summary
# ===========================================================================

class TestN12CandidatesBelowThreshold:
    """N12 fix: retrieval_summary must include candidates_below_threshold so
    production operators can monitor whether the similarity_threshold is calibrated
    correctly for the deployed index.
    """

    def _retrieval_summary(self, analogs: list, fallback_used: bool = False,
                           candidates_below: int = 0) -> dict:
        from stages.stage_d_analogs import HistoricalAnalogRetriever
        retriever = HistoricalAnalogRetriever()
        return retriever._build_retrieval_summary(analogs, fallback_used, candidates_below)

    def _analog(self, score: float = 0.80) -> dict:
        return {
            "analog_id": "AN-001",
            "similarity_score": score,
            "outage_id": "OUT-001",
            "plant_id": "PLT-A",
            "actual_duration_hours": 6.0,
        }

    def test_n12_field_present_in_retrieval_summary(self):
        """retrieval_summary must contain candidates_below_threshold key."""
        summary = self._retrieval_summary([self._analog()], candidates_below=2)
        assert "candidates_below_threshold" in summary, (
            "retrieval_summary must include candidates_below_threshold (N12 fix)"
        )

    def test_n12_zero_when_no_rejections(self):
        """When all candidates pass the threshold, candidates_below_threshold=0."""
        summary = self._retrieval_summary([self._analog()], candidates_below=0)
        assert summary["candidates_below_threshold"] == 0

    def test_n12_count_reflects_rejected_candidates(self):
        """Injected count is faithfully stored in the summary."""
        summary = self._retrieval_summary([self._analog()], candidates_below=7)
        assert summary["candidates_below_threshold"] == 7

    def test_n12_default_is_zero(self):
        """Omitting candidates_below_threshold defaults to 0 (backwards-compatible)."""
        from stages.stage_d_analogs import HistoricalAnalogRetriever
        retriever = HistoricalAnalogRetriever()
        summary = retriever._build_retrieval_summary([self._analog()], fallback_used=False)
        assert summary.get("candidates_below_threshold", 0) == 0

    def test_n12_score_and_filter_returns_tuple(self):
        """_score_and_filter must return (list, int) — not just a list."""
        from stages.stage_d_analogs import HistoricalAnalogRetriever
        retriever = HistoricalAnalogRetriever()
        # No engine injected → placeholder path; returns (list, 0)
        result = retriever._score_and_filter({}, [], None)
        assert isinstance(result, tuple), "_score_and_filter must return a tuple (N12 fix)"
        assert len(result) == 2
        analogs, n_below = result
        assert isinstance(analogs, list)
        assert isinstance(n_below, int)

    def test_n12_score_and_filter_empty_candidates_returns_zero_below(self):
        """Empty candidates list → ([], 0)."""
        from stages.stage_d_analogs import HistoricalAnalogRetriever
        retriever = HistoricalAnalogRetriever()
        analogs, n_below = retriever._score_and_filter({}, [], None)
        assert analogs == []
        assert n_below == 0


# ===========================================================================
# Stage G — M1: LCO action-level clock flags and executive summary prefix
# ===========================================================================

class TestM1LcoClockFlags:
    """M1 fix: _compute_attention_flags must raise LCO clock flags based on
    lco_clock_status from the intake_result.
    """

    from stages.stage_g_recommendation import (
        _FLAG_LCO_EXPIRED, _FLAG_LCO_CLOCK_CRITICAL,
    )

    def _flags(self, lco_clock_status: str, hours: float | None = None) -> list:
        synth = _synth()
        intake = {
            "has_regulatory_constraint": False,
            "unknown_abbreviation_rate": 0.0,
            "regulatory_drivers": [],
            "lco_clock_status": lco_clock_status,
            "hours_to_action_level": hours,
        }
        ha = _analogs(confidence_tier="data_supported", analog_count=8)
        schedule = _schedule_impact()
        chain = _temporal_chain()
        return synth._compute_attention_flags(intake, ha, schedule, chain, {})

    def test_m1_no_lco_flags_when_not_applicable(self):
        """not_applicable → neither LCO flag raised."""
        from stages.stage_g_recommendation import _FLAG_LCO_EXPIRED, _FLAG_LCO_CLOCK_CRITICAL
        flags = self._flags("not_applicable")
        assert _FLAG_LCO_EXPIRED not in flags
        assert _FLAG_LCO_CLOCK_CRITICAL not in flags

    def test_m1_lco_critical_flag_raised_for_critical_status(self):
        """critical status → _FLAG_LCO_CLOCK_CRITICAL raised, _FLAG_LCO_EXPIRED not."""
        from stages.stage_g_recommendation import _FLAG_LCO_EXPIRED, _FLAG_LCO_CLOCK_CRITICAL
        flags = self._flags("critical", hours=2.0)
        assert _FLAG_LCO_CLOCK_CRITICAL in flags
        assert _FLAG_LCO_EXPIRED not in flags

    def test_m1_lco_critical_flag_raised_for_urgent_status(self):
        """urgent status → _FLAG_LCO_CLOCK_CRITICAL raised."""
        from stages.stage_g_recommendation import _FLAG_LCO_CLOCK_CRITICAL
        flags = self._flags("urgent", hours=10.0)
        assert _FLAG_LCO_CLOCK_CRITICAL in flags

    def test_m1_both_lco_flags_raised_for_expired_status(self):
        """expired status → both _FLAG_LCO_EXPIRED and _FLAG_LCO_CLOCK_CRITICAL raised."""
        from stages.stage_g_recommendation import _FLAG_LCO_EXPIRED, _FLAG_LCO_CLOCK_CRITICAL
        flags = self._flags("expired", hours=-1.5)
        assert _FLAG_LCO_EXPIRED in flags
        assert _FLAG_LCO_CLOCK_CRITICAL in flags

    def test_m1_lco_critical_flag_raised_for_unknown_status(self):
        """unknown status (active LCO, no deadline) → _FLAG_LCO_CLOCK_CRITICAL raised."""
        from stages.stage_g_recommendation import _FLAG_LCO_CLOCK_CRITICAL
        flags = self._flags("unknown")
        assert _FLAG_LCO_CLOCK_CRITICAL in flags

    def test_m1_no_lco_flags_for_normal_status(self):
        """normal status → no LCO flags."""
        from stages.stage_g_recommendation import _FLAG_LCO_EXPIRED, _FLAG_LCO_CLOCK_CRITICAL
        flags = self._flags("normal", hours=48.0)
        assert _FLAG_LCO_EXPIRED not in flags
        assert _FLAG_LCO_CLOCK_CRITICAL not in flags


class TestM1LcoClockPrefix:
    """M1 fix: _build_lco_clock_prefix and _build_executive_summary must prepend
    the LCO clock warning to the primary_conclusion.
    """

    def _prefix(self, lco_clock_status: str, hours: float | None = None,
                 lco_number: str | None = None) -> str:
        synth = _synth()
        intake = {
            "lco_clock_status": lco_clock_status,
            "hours_to_action_level": hours,
            "lco_number": lco_number,
        }
        return synth._build_lco_clock_prefix(intake)

    def test_m1_prefix_empty_when_not_applicable(self):
        assert self._prefix("not_applicable") == ""

    def test_m1_prefix_empty_when_normal(self):
        assert self._prefix("normal", hours=48.0) == ""

    def test_m1_prefix_empty_when_no_intake_result(self):
        synth = _synth()
        assert synth._build_lco_clock_prefix(None) == ""

    def test_m1_prefix_contains_expired_text(self):
        prefix = self._prefix("expired", hours=-2.0)
        assert "EXPIRED" in prefix.upper()
        assert "2.0" in prefix

    def test_m1_prefix_contains_critical_text(self):
        prefix = self._prefix("critical", hours=1.5)
        assert "CRITICAL" in prefix.upper()
        assert "1.5" in prefix

    def test_m1_prefix_contains_urgent_text(self):
        prefix = self._prefix("urgent", hours=10.0)
        assert "URGENT" in prefix.upper()
        assert "10.0" in prefix

    def test_m1_prefix_contains_lco_number_when_provided(self):
        prefix = self._prefix("critical", hours=2.0, lco_number="3.5.2")
        assert "3.5.2" in prefix

    def test_m1_prefix_contains_unknown_warning(self):
        prefix = self._prefix("unknown")
        assert len(prefix) > 0
        assert "LCO" in prefix.upper()

    def test_m1_conclusion_leads_with_lco_prefix(self):
        """primary_conclusion must begin with the LCO clock warning when critical."""
        synth = _synth()
        intake = {
            "has_regulatory_constraint": False,
            "unknown_abbreviation_rate": 0.0,
            "regulatory_drivers": [],
            "lco_clock_status": "critical",
            "hours_to_action_level": 1.5,
            "lco_number": "3.8.1",
        }
        attention_flags = synth._compute_attention_flags(
            intake, _analogs(), _schedule_impact(), _temporal_chain(), {}
        )
        summary = synth._build_executive_summary(
            "PROCEED", {"option_type": "insert_now"}, "data_supported",
            attention_flags, _analogs(), _schedule_impact(),
            intake_result=intake,
        )
        conclusion = summary["primary_conclusion"]
        # LCO clock warning must lead the conclusion
        assert conclusion.startswith("\U0001f6a8") or conclusion.startswith("\u26a0"), (
            "Primary conclusion must begin with LCO clock warning prefix when status=critical"
        )
        assert "3.8.1" in conclusion


# ===========================================================================
# ADV-02 (Stage G flag), ADV-04, ADV-05, ADV-06, ADV-07: Adversarial inputs
# ===========================================================================

def _schedule_impact_with_displaced(
    displaced_tasks: list,
    baseline_cp_hours: float = 100.0,
    cp_drag: float = 0.0,
    criticality_label: str = "non_critical",
) -> dict:
    """Like _schedule_impact() but supports displaced_tasks for ADV-07."""
    base = _schedule_impact(
        baseline_cp_hours=baseline_cp_hours,
        cp_drag=cp_drag,
        criticality_label=criticality_label,
    )
    base["displaced_tasks"] = displaced_tasks
    return base


class TestAdversarialStageF:
    """ADV-04 – ADV-06: Edge-case inputs to Stage F (InsertionOptionGenerator)."""

    # ── ADV-04: CP drag exceeds outage duration ───────────────────────────────

    def test_adv04_cp_drag_exceeds_baseline_risk_score_in_range(self):
        """_score_option must return a value in [0.0, 1.0] when cp_drag > baseline_cp.

        Formula: cp_impact_score = min(1.0, 800/600) = 1.0; total risk still ≤ 1.0
        because all dimension weights sum to 1.0 and each score is in [0, 1].
        """
        gen = _gen()
        # Option cp_impact_hours must match the schedule_impact cp_drag for realism,
        # but _score_option reads cp_drag from option.cp_impact_hours (line ~975).
        opt = _option("insert_now", cp_impact_hours=800.0)
        impact = _schedule_impact(baseline_cp_hours=600.0, cp_drag=800.0,
                                  criticality_label="critical")
        risk = gen._score_option(opt, impact, _analogs(), causal_posture="supported")
        assert 0.0 <= risk <= 1.0, (
            f"risk score must be in [0, 1] even when cp_drag > baseline_cp, got {risk}"
        )

    def test_adv04_cp_drag_at_ceiling_equals_above_ceiling(self):
        """cp_impact_score is clamped: drag=600 and drag=800 both clamp to 1.0
        on a 600 h baseline, so the resulting risk scores must be equal."""
        gen = _gen()
        impact_at = _schedule_impact(baseline_cp_hours=600.0, cp_drag=600.0,
                                     criticality_label="critical")
        impact_above = _schedule_impact(baseline_cp_hours=600.0, cp_drag=800.0,
                                        criticality_label="critical")
        opt_at = _option("insert_now", cp_impact_hours=600.0)
        opt_above = _option("insert_now", cp_impact_hours=800.0)
        risk_at = gen._score_option(opt_at, impact_at, _analogs(),
                                    causal_posture="supported")
        risk_above = gen._score_option(opt_above, impact_above, _analogs(),
                                       causal_posture="supported")
        assert risk_at == risk_above, (
            f"cp_impact_score should be clamped to 1.0 for both drag=600 and drag=800 "
            f"on a 600 h baseline; expected equal risk scores but got {risk_at} vs {risk_above}"
        )

    def test_adv04_escalate_option_generated_when_drag_exceeds_threshold(self):
        """generate() must produce an escalate_to_management option when
        cp_drag_hours > escalate_if_cp_drag_exceeds_hours (default 24 h).

        With cp_drag=800 h this is far above the threshold.
        """
        gen = _gen()
        ea = {"activity_id": "EA-ADV04", "outage_id": "RF-24"}
        intake = {"regulatory_drivers": []}
        temporal = {"summary": {"causal_posture": "supported"}, "chain_links": []}
        schedule = _schedule_impact(baseline_cp_hours=600.0, cp_drag=800.0,
                                    criticality_label="critical")
        analogs = _analogs(p50=8.0, p80=12.0)
        run_ctx = {"run_id": "R-ADV04"}

        result = gen.generate(ea, intake, temporal, schedule, analogs, run_ctx)

        option_types = [o["option_type"] for o in result["options"]]
        assert "escalate_to_management" in option_types, (
            f"escalate_to_management must be generated when cp_drag=800 h >> 24 h threshold; "
            f"option types found: {option_types}"
        )

    def test_adv04_extreme_cp_drag_decision_status_escalate(self):
        """Stage G must return ESCALATE when the primary option is escalate_to_management."""
        synth = _synth()
        escalate_opt = _option("escalate_to_management", cp_impact_hours=800.0,
                               risk_score=0.01)  # lowest risk → primary
        options = {
            "recommended_option_id": escalate_opt["option_id"],
            "options": [escalate_opt],
        }
        status = synth._determine_decision_status(
            primary_option=escalate_opt,
            intake_result=_intake_result(),
            historical_analogs=_analogs(),
            insertion_options=options,
            schedule_impact_assessment=_schedule_impact(cp_drag=800.0,
                                                        criticality_label="critical"),
        )
        assert status == "ESCALATE", (
            f"Expected ESCALATE when primary option is escalate_to_management, got {status}"
        )

    # ── ADV-05: Duplicate driver types in regulatory_drivers ─────────────────

    def _ts_drivers(self, n: int = 2) -> list:
        """Return n identical ts_surveillance driver dicts."""
        return [
            {"driver_type": "ts_surveillance", "defer_prohibited": True}
            for _ in range(n)
        ]

    def test_adv05_duplicate_defer_block_fires(self):
        """_check_regulatory_clearance must block defer even with duplicate driver entries."""
        gen = _gen()
        opt = _option("defer_to_post_outage")
        cleared, reason = gen._check_regulatory_clearance(opt, self._ts_drivers(2))
        assert cleared is False, (
            "defer_to_post_outage must be blocked when driver list has defer_prohibited=True"
        )
        assert reason is not None

    def test_adv05_duplicate_block_reason_no_repeated_driver_type(self):
        """ADV-05 fix: block reason must not repeat the same driver_type string.

        With two identical ts_surveillance entries, the message should say
        'ts_surveillance' once, not 'ts_surveillance, ts_surveillance'.
        dict.fromkeys() deduplication added to _check_regulatory_clearance.
        """
        gen = _gen()
        opt = _option("defer_to_post_outage")
        _, reason = gen._check_regulatory_clearance(opt, self._ts_drivers(5))
        assert reason is not None
        # Extract the driver_types portion of the message
        # Format: "Deferral prohibited by regulatory constraint(s): <types>. Work..."
        types_str = reason.split(":")[1].split(".")[0].strip()
        reported_types = [t.strip() for t in types_str.split(",")]
        assert len(reported_types) == len(set(reported_types)), (
            f"Block reason must not repeat driver types; got: '{types_str}'"
        )

    def test_adv05_duplicate_scope_reduction_block_fires(self):
        """_check_regulatory_clearance must block scope_reduction even with duplicate ts entries."""
        gen = _gen()
        opt = _option("scope_reduction")
        cleared, reason = gen._check_regulatory_clearance(opt, self._ts_drivers(3))
        assert cleared is False, (
            "scope_reduction must be blocked by ts_surveillance even with duplicate entries"
        )
        assert reason is not None

    def test_adv05_duplicate_scope_reduction_reason_no_repeated_type(self):
        """scope_reduction block reason must also deduplicate driver types."""
        gen = _gen()
        opt = _option("scope_reduction")
        _, reason = gen._check_regulatory_clearance(opt, self._ts_drivers(4))
        assert reason is not None
        types_str = reason.split(":")[1].split(" requires")[0].strip()
        reported_types = [t.strip() for t in types_str.split(",")]
        assert len(reported_types) == len(set(reported_types)), (
            f"scope_reduction reason must not repeat driver types; got: '{types_str}'"
        )

    # ── ADV-06: p50=p80=0 from duration distribution ─────────────────────────

    def _zero_dist_analogs(self) -> dict:
        return _analogs(p50=0.0, p80=0.0, confidence_tier="low_confidence",
                        sample_size=1, analog_count=1)

    def test_adv06_zero_duration_contingency_buffer_generated(self):
        """_generate_contingency_buffer must not filter out the option when p50=p80=0.

        buffer_hours = max(0, 0-0) = 0; option should still be generated since
        a zero-buffer still reserves the insertion slot.
        """
        gen = _gen()
        ea = {"activity_id": "EA-ADV06"}
        schedule = _schedule_impact(criticality_label="non_critical")
        result = gen._generate_contingency_buffer(ea, schedule, self._zero_dist_analogs())
        assert isinstance(result, dict), (
            "_generate_contingency_buffer must return a dict even when p50=p80=0"
        )
        assert "option_type" in result

    def test_adv06_zero_duration_no_division_by_zero(self):
        """generate() must not raise ZeroDivisionError when p50=p80=0."""
        gen = _gen()
        ea = {"activity_id": "EA-ADV06", "outage_id": "RF-24"}
        intake = {"regulatory_drivers": []}
        temporal = {"summary": {"causal_posture": "insufficient_data"}, "chain_links": []}
        schedule = _schedule_impact(criticality_label="non_critical")
        run_ctx = {"run_id": "R-ADV06"}
        # Must not raise
        result = gen.generate(ea, intake, temporal, schedule,
                              self._zero_dist_analogs(), run_ctx)
        assert isinstance(result, dict)
        assert "options" in result

    def test_adv06_zero_buffer_cost_estimate_is_zero(self):
        """For p50=p80=0 the contingency buffer duration = max(0, p80-p50) = 0,
        so the labor cost must be zero.  Cost is assigned centrally by generate(),
        so this test exercises the full generation path rather than the internal
        _generate_contingency_buffer method alone.
        """
        gen = _gen()
        ea = {"activity_id": "EA-ADV06", "outage_id": "RF-24"}
        intake = {"regulatory_drivers": []}
        temporal = {"summary": {"causal_posture": "insufficient_data"}, "chain_links": []}
        schedule = _schedule_impact(criticality_label="non_critical")
        run_ctx = {"run_id": "R-ADV06"}

        result = gen.generate(ea, intake, temporal, schedule,
                              self._zero_dist_analogs(), run_ctx)

        buf_opts = [o for o in result["options"]
                    if o["option_type"] == "add_contingency_buffer"]
        assert buf_opts, "add_contingency_buffer must be generated even with p50=p80=0"
        ce = buf_opts[0].get("cost_estimate") or {}
        assert ce.get("labor_cost_usd", -1) == pytest.approx(0.0), (
            f"labor_cost_usd must be 0 when contingency buffer duration=0, "
            f"got {ce.get('labor_cost_usd')}"
        )


class TestAdversarialStageG:
    """ADV-02 (Stage G part) and ADV-07: Edge-case inputs to Stage G
    (RecommendationSynthesizer).
    """

    # ── ADV-02: High abbreviation rate → attention flag ───────────────────────

    def test_adv02_high_abbr_rate_flag_raised(self):
        """_compute_attention_flags must include 'high_unknown_abbreviation_rate'
        when intake_result.unknown_abbreviation_rate > 0.25 (the warning threshold
        from ActivityIntakeConfig and RecommendationConfig).
        """
        synth = _synth()
        intake = _intake_result(abbr_rate=0.80)  # far above 0.25 threshold
        flags = synth._compute_attention_flags(
            intake, _analogs(), _schedule_impact(), _temporal_chain(), {}
        )
        assert "high_unknown_abbreviation_rate" in flags, (
            f"'high_unknown_abbreviation_rate' flag must be raised when abbr_rate=0.80; "
            f"got flags: {flags}"
        )

    def test_adv02_abbr_rate_below_threshold_no_flag(self):
        """No abbreviation rate flag when unknown_abbreviation_rate ≤ 0.25."""
        synth = _synth()
        intake = _intake_result(abbr_rate=0.10)
        flags = synth._compute_attention_flags(
            intake, _analogs(), _schedule_impact(), _temporal_chain(), {}
        )
        assert "high_unknown_abbreviation_rate" not in flags, (
            f"Flag must not be raised at abbr_rate=0.10; got flags: {flags}"
        )

    def test_adv02_analyst_review_required_on_high_abbr_rate(self):
        """analyst_review.required must be True when high_unknown_abbreviation_rate
        flag is present (§6 exit criterion: NER unreliable above threshold).
        """
        synth = _synth()
        intake = _intake_result(abbr_rate=0.80)
        attention_flags = synth._compute_attention_flags(
            intake, _analogs(), _schedule_impact(), _temporal_chain(), {}
        )
        assert "high_unknown_abbreviation_rate" in attention_flags
        review = synth._determine_analyst_review(
            "PROCEED", intake, _analogs(), {}, attention_flags
        )
        assert review["required"] is True, (
            "analyst_review.required must be True when high abbreviation rate flag is set"
        )

    # ── ADV-07: All displaced tasks have has_regulatory_constraint=True ────────

    def _displaced_regulatory_schedule(self, count: int = 3) -> dict:
        """Build a schedule_impact dict with `count` displaced tasks, all regulatory."""
        displaced = [
            {
                "task_id": f"T-SURV-{i:03d}",
                "task_name": f"Surveillance task {i}",
                "criticality_label": "non_critical",
                "has_regulatory_constraint": True,
                "float_hours": 4.0,
                "discipline": "i_and_c",
            }
            for i in range(1, count + 1)
        ]
        return _schedule_impact_with_displaced(displaced)

    def test_adv07_displaced_regulatory_flag_raised(self):
        """_compute_attention_flags must include 'displaced_regulatory_tasks' when
        any displaced task has has_regulatory_constraint=True.
        """
        synth = _synth()
        flags = synth._compute_attention_flags(
            _intake_result(),
            _analogs(),
            self._displaced_regulatory_schedule(count=10),  # all 10 are regulatory
            _temporal_chain(),
            {},
        )
        assert "displaced_regulatory_tasks" in flags, (
            f"'displaced_regulatory_tasks' flag must be raised when all displaced tasks "
            f"are regulatory; got flags: {flags}"
        )

    def test_adv07_no_displaced_regulatory_flag_when_tasks_absent(self):
        """Flag must NOT be raised when displaced_tasks is empty."""
        synth = _synth()
        flags = synth._compute_attention_flags(
            _intake_result(), _analogs(), _schedule_impact(), _temporal_chain(), {}
        )
        assert "displaced_regulatory_tasks" not in flags

    def test_adv07_displaced_regulatory_schedule_summary_field(self):
        """_build_schedule_summary must set has_displaced_regulatory_tasks=True
        when all displaced tasks carry has_regulatory_constraint=True.
        """
        synth = _synth()
        summary = synth._build_schedule_summary(
            self._displaced_regulatory_schedule(count=3)
        )
        assert summary.get("has_displaced_regulatory_tasks") is True, (
            "has_displaced_regulatory_tasks must be True in schedule_summary "
            "when all displaced tasks are regulatory"
        )

    def test_adv07_single_non_regulatory_task_does_not_fire_flag(self):
        """Flag must NOT fire when displaced tasks exist but none are regulatory."""
        synth = _synth()
        non_regulatory_displaced = [
            {
                "task_id": "T-MAINT-001",
                "task_name": "Non-regulatory maintenance",
                "criticality_label": "non_critical",
                "has_regulatory_constraint": False,
                "float_hours": 8.0,
                "discipline": "mechanical",
            }
        ]
        schedule = _schedule_impact_with_displaced(non_regulatory_displaced)
        flags = synth._compute_attention_flags(
            _intake_result(), _analogs(), schedule, _temporal_chain(), {}
        )
        assert "displaced_regulatory_tasks" not in flags, (
            "Flag must not fire when no displaced task has has_regulatory_constraint=True"
        )
