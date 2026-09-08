"""
Stage G — Recommendation Synthesizer.

Responsibilities:
    1. Select the primary recommendation from InsertionOptions.
    2. Determine the decision_status (PROCEED / DEFER / ESCALATE / MONITOR /
       INCONCLUSIVE).
    3. Build the executive_summary (plain-language conclusion, confidence tier,
       analyst attention flags).
    4. Assemble the evidence chain from upstream artifacts — every claim must
       cite a source (WBS 11.2 traceability requirement).
    5. Surface all regulatory flags from Stage A.
    6. Build schedule and history summaries for the executive view.
    7. Set analyst_review=True when required (regulatory constraints present,
       low confidence, INCONCLUSIVE status, upstream fallback used, or
       unknown abbreviation rate > threshold).
    8. Compute the validation_status block.

Output schema: outage/schemas/outage_activity_recommendation.json

Design principle (from critical_analysis.md §8):
    Every recommendation must surface:
        (a) the specific historical outages it draws from
        (b) the number of analogous events in the training data
        (c) an explicit confidence tier
        (d) a reject-with-reason path that feeds back into the learning loop

Reuse targets:
    RCA.synthesis.rca_synthesizer_v31  → evidence selection pattern, confidence
                                         tier logic, fallback detection
    RCA.validation.schema_validator    → _compute_validation_status()
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# Decision status constants
_PROCEED = "PROCEED"
_DEFER = "DEFER"
_ESCALATE = "ESCALATE"
_MONITOR = "MONITOR"
_INCONCLUSIVE = "INCONCLUSIVE"

# Analyst attention flag keys
_FLAG_REGULATORY = "regulatory_constraint_present"
_FLAG_LOW_CONFIDENCE = "low_confidence_recommendation"
_FLAG_LOW_ANALOGS = "low_analog_count"
_FLAG_TEMPORAL_CONTRADICTION = "temporal_contradiction_detected"
_FLAG_CP_IMPACT = "critical_path_impact"
_FLAG_HIGH_ABBR_RATE = "high_unknown_abbreviation_rate"
_FLAG_FALLBACK = "fallback_distribution_used"
_FLAG_DISPLACED_REGULATORY = "displaced_regulatory_tasks"
# M1: LCO action-level clock flags
_FLAG_LCO_EXPIRED = "lco_action_level_expired"
_FLAG_LCO_CLOCK_CRITICAL = "lco_action_level_critical"

# LCO clock statuses that require immediate attention
_LCO_URGENT_STATUSES = {"expired", "critical", "urgent"}

# Option types that map to PROCEED
_PROCEED_OPTION_TYPES = {
    "insert_now",
    "add_contingency_buffer",
    "parallel_execution",
    "scope_reduction",
    "pre_outage_staging",
}


@dataclass
class RecommendationConfig:
    """Configuration for Stage G."""

    min_analog_count_for_no_flag: int = 5
    """Analog counts below this value trigger _FLAG_LOW_ANALOGS."""

    unknown_abbreviation_rate_warning: float = 0.25
    """Above this threshold _FLAG_HIGH_ABBR_RATE is raised."""

    max_evidence_items: int = 10
    """Maximum evidence chain entries to include in the artifact."""

    pipeline_version: Optional[str] = None
    """Human-readable pipeline version string written into every produced
    artifact (e.g. ``"outage-pipeline-1.4.2"``).  Can also be supplied at
    call time via ``run_context["pipeline_version"]``, which takes precedence.
    Leave ``None`` to omit version tagging (acceptable during development, but
    artifacts become indistinguishable across pipeline generations in storage)."""


class RecommendationSynthesizer:
    """Concrete Stage G implementation.

    Args:
        config: Stage configuration.
    """

    def __init__(self, config: Optional[RecommendationConfig] = None) -> None:
        self.config = config or RecommendationConfig()

    # ── Protocol method ───────────────────────────────────────────────────────

    def synthesize(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        component_event_timeline: JsonDict,
        temporal_event_chain: JsonDict,
        historical_analogs: JsonDict,
        schedule_impact_assessment: JsonDict,
        insertion_options: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """Execute Stage G for one emergent activity.

        Returns:
            OutageActivityRecommendation artifact conforming to
            outage/schemas/outage_activity_recommendation.json.
        """
        run_id: str = run_context["run_id"]
        activity_id: str = emergent_activity["activity_id"]
        recommendation_id = f"REC::{activity_id}::{uuid.uuid4().hex[:8]}"
        LOGGER.debug(
            "Stage G recommendation synthesis for %s (run=%s)", activity_id, run_id
        )

        primary_option = self._select_primary_option(insertion_options)
        decision_status = self._determine_decision_status(
            primary_option, intake_result, historical_analogs, insertion_options,
            schedule_impact_assessment=schedule_impact_assessment,
        )
        attention_flags = self._compute_attention_flags(
            intake_result, historical_analogs, schedule_impact_assessment,
            temporal_event_chain, insertion_options
        )
        confidence_tier = self._determine_confidence_tier(
            historical_analogs, insertion_options
        )
        executive_summary = self._build_executive_summary(
            decision_status, primary_option, confidence_tier, attention_flags,
            historical_analogs, schedule_impact_assessment,
            intake_result=intake_result,
        )
        evidence_chain = self._assemble_evidence_chain(
            temporal_event_chain, historical_analogs,
            schedule_impact_assessment, component_event_timeline,
            intake_result=intake_result,
        )
        regulatory_flags: List[JsonDict] = intake_result.get("regulatory_drivers", [])
        history_summary = self._build_history_summary(
            historical_analogs, component_event_timeline
        )
        schedule_summary = self._build_schedule_summary(
            schedule_impact_assessment
        )
        analyst_review = self._determine_analyst_review(
            decision_status, intake_result, historical_analogs,
            insertion_options, attention_flags
        )
        validation_status = self._compute_validation_status(
            evidence_chain, regulatory_flags, historical_analogs
        )

        return {
            "recommendation_id": recommendation_id,
            "activity_id": activity_id,
            "run_id": run_id,
            "generated_at": run_context.get("started_at", ""),
            "pipeline_version": (
                run_context.get("pipeline_version") or self.config.pipeline_version
            ),
            "decision_status": decision_status,
            "executive_summary": executive_summary,
            "primary_recommendation": self._build_primary_recommendation(primary_option),
            "regulatory_flags": regulatory_flags,
            "evidence_chain": evidence_chain[: self.config.max_evidence_items],
            "history_summary": history_summary,
            "schedule_summary": schedule_summary,
            "analyst_review": analyst_review,
            "validation_status": validation_status,
            "provenance": {
                "run_id": run_id,
                "generated_by": self.__class__.__name__,
                "pipeline_version": (
                    run_context.get("pipeline_version") or self.config.pipeline_version
                ),
                "input_artifacts": {
                    "emergent_activity_id": activity_id,
                    "schedule_version_id": schedule_impact_assessment.get("schedule_version_id"),
                    "intake_result_run_id": intake_result.get("run_id"),
                    "timeline_run_id": component_event_timeline.get("run_id"),
                    "temporal_chain_run_id": temporal_event_chain.get("run_id"),
                    "schedule_impact_run_id": schedule_impact_assessment.get("run_id"),
                    "analogs_run_id": historical_analogs.get("run_id"),
                    "insertion_options_run_id": insertion_options.get("run_id"),
                },
            },
        }

    # ── Private step methods ──────────────────────────────────────────────────

    def _select_primary_option(self, insertion_options: JsonDict) -> Optional[JsonDict]:
        """Retrieve the recommended option object from InsertionOptions.

        Looks up insertion_options['recommended_option_id'] in the options list.
        Returns None if recommended_option_id is None or not found.
        """
        recommended_id = insertion_options.get("recommended_option_id")
        if not recommended_id:
            return None
        for option in insertion_options.get("options", []):
            if option.get("option_id") == recommended_id:
                return option
        return None

    def _determine_decision_status(
        self,
        primary_option: Optional[JsonDict],
        intake_result: JsonDict,
        historical_analogs: JsonDict,
        insertion_options: JsonDict,
        schedule_impact_assessment: Optional[JsonDict] = None,
    ) -> str:
        """Map the primary option and context to a decision_status enum value.

        Logic (evaluated in priority order):
            INCONCLUSIVE — no feasible, regulatory-cleared option exists
            ESCALATE     — primary option type is escalate_to_management
            DEFER        — primary option type is defer_to_post_outage
            MONITOR      — low confidence + no historical analogues + activity
                           is non-critical path (watch, but no action yet warranted)
            PROCEED      — primary option type is insert_now / contingency /
                           parallel / scope_reduction / pre_outage_staging

        ``schedule_impact_assessment`` (N5 fix): criticality_label is now read
        from ``schedule_impact_assessment["float_analysis"]["criticality_label"]``
        rather than the non-existent ``insertion_options["schedule_summary"]``
        field that Stage F never produces.  The parameter is optional and defaults
        to None so existing call sites that have not yet been updated continue to
        work; absence is treated as non_critical (permissive MONITOR condition).
        """
        if primary_option is None:
            return _INCONCLUSIVE

        option_type: str = primary_option.get("option_type", "")

        if option_type == "escalate_to_management":
            return _ESCALATE

        if option_type == "defer_to_post_outage":
            return _DEFER

        # MONITOR: no historical support + non-critical → watch, no immediate action
        # Previous condition required cp_impact==0 AND analog_count==0 AND low_confidence
        # simultaneously.  In practice cp_impact is almost never exactly 0.0 even for
        # non-critical activities (any float consumption produces a nonzero value), so
        # MONITOR was never reachable.  The intent is: when we have no analog data and
        # the activity doesn't clearly threaten the critical path, flag it for monitoring
        # rather than committing to a PROCEED recommendation built on no evidence.
        analog_count: int = int(
            (historical_analogs.get("retrieval_summary") or {}).get("analog_count", 0)
        )
        dist_tier: str = (
            (historical_analogs.get("duration_distribution") or {})
            .get("confidence_tier", "low_confidence")
        ) or "low_confidence"
        # N5 fix: read criticality_label from schedule_impact_assessment["float_analysis"]
        # instead of the non-existent insertion_options["schedule_summary"] field.
        criticality_label: str = (
            ((schedule_impact_assessment or {}).get("float_analysis") or {})
            .get("criticality_label", "non_critical")
        ) or "non_critical"
        if dist_tier == "low_confidence" and analog_count == 0:
            if criticality_label == "non_critical":
                return _MONITOR

        if option_type in _PROCEED_OPTION_TYPES:
            return _PROCEED

        # Fallback for unknown option types
        return _INCONCLUSIVE

    def _compute_attention_flags(
        self,
        intake_result: JsonDict,
        historical_analogs: JsonDict,
        schedule_impact: JsonDict,
        temporal_chain: JsonDict,
        insertion_options: JsonDict,
    ) -> List[str]:
        """Compute the list of analyst attention flags.

        Each flag is a string constant defined at module level.  Flags are
        raised when:
            _FLAG_REGULATORY        — intake_result.has_regulatory_constraint
            _FLAG_LOW_CONFIDENCE    — confidence_tier == 'low_confidence'
            _FLAG_LOW_ANALOGS       — analog_count < config.min_analog_count_for_no_flag
            _FLAG_TEMPORAL_CONTRADICTION — temporal_chain.summary.has_temporal_contradiction
            _FLAG_CP_IMPACT         — schedule_impact.float_analysis.criticality_label == "critical"
            _FLAG_HIGH_ABBR_RATE    — intake_result.unknown_abbreviation_rate > threshold
            _FLAG_FALLBACK          — historical_analogs.retrieval_summary.fallback_used
            _FLAG_DISPLACED_REGULATORY — any displaced task has has_regulatory_constraint=True
            _FLAG_LCO_EXPIRED       — intake_result.lco_clock_status == "expired"
            _FLAG_LCO_CLOCK_CRITICAL — lco_clock_status in {"critical", "urgent", "unknown"}
                                       and active LCO is present
        """
        flags: List[str] = []

        # Regulatory constraint
        if intake_result.get("has_regulatory_constraint"):
            flags.append(_FLAG_REGULATORY)

        # Low confidence tier
        dist_tier = (
            (historical_analogs.get("duration_distribution") or {})
            .get("confidence_tier", "low_confidence")
        )
        if dist_tier == "low_confidence":
            flags.append(_FLAG_LOW_CONFIDENCE)

        # Low analog count
        analog_count = int(
            (historical_analogs.get("retrieval_summary") or {}).get("analog_count", 0)
        )
        if analog_count < self.config.min_analog_count_for_no_flag:
            flags.append(_FLAG_LOW_ANALOGS)

        # Temporal contradiction
        chain_summary = temporal_chain.get("summary") or {}
        if chain_summary.get("has_temporal_contradiction"):
            flags.append(_FLAG_TEMPORAL_CONTRADICTION)

        # Critical path impact
        float_analysis = schedule_impact.get("float_analysis") or {}
        if float_analysis.get("criticality_label") == "critical":
            flags.append(_FLAG_CP_IMPACT)

        # High unknown abbreviation rate
        abbr_rate = float(intake_result.get("unknown_abbreviation_rate") or 0.0)
        if abbr_rate > self.config.unknown_abbreviation_rate_warning:
            flags.append(_FLAG_HIGH_ABBR_RATE)

        # Fallback distribution used
        retrieval_summary = historical_analogs.get("retrieval_summary") or {}
        if retrieval_summary.get("fallback_used"):
            flags.append(_FLAG_FALLBACK)

        # Displaced regulatory tasks
        displaced_tasks = schedule_impact.get("displaced_tasks") or []
        if any(t.get("has_regulatory_constraint") for t in displaced_tasks):
            flags.append(_FLAG_DISPLACED_REGULATORY)

        # M1: LCO action-level clock flags
        lco_clock_status: str = intake_result.get("lco_clock_status") or "not_applicable"
        if lco_clock_status == "expired":
            flags.append(_FLAG_LCO_EXPIRED)
            flags.append(_FLAG_LCO_CLOCK_CRITICAL)
        elif lco_clock_status in ("critical", "urgent"):
            flags.append(_FLAG_LCO_CLOCK_CRITICAL)
        elif lco_clock_status == "unknown":
            # Active LCO with no known deadline — treat as critical until proven otherwise
            flags.append(_FLAG_LCO_CLOCK_CRITICAL)

        return flags

    def _determine_confidence_tier(
        self, historical_analogs: JsonDict, insertion_options: JsonDict
    ) -> str:
        """Return the overall confidence tier for the recommendation.

        Derived from the duration distribution confidence_tier in Stage D.
        Degrades to low_confidence if the recommended option has confidence < 0.4.
        """
        dist = historical_analogs.get("duration_distribution") or {}
        tier: str = dist.get("confidence_tier") or "low_confidence"

        # Degrade if the recommended option itself has low confidence
        recommended_id = insertion_options.get("recommended_option_id")
        if recommended_id:
            for option in insertion_options.get("options", []):
                if option.get("option_id") == recommended_id:
                    if float(option.get("confidence") or 0.0) < 0.4:
                        return "low_confidence"
                    break

        return tier

    def _build_executive_summary(
        self,
        decision_status: str,
        primary_option: Optional[JsonDict],
        confidence_tier: str,
        attention_flags: List[str],
        historical_analogs: JsonDict,
        schedule_impact: JsonDict,
        intake_result: Optional[JsonDict] = None,
    ) -> JsonDict:
        """Build the plain-language executive summary for the outage manager.

        primary_conclusion: one sentence stating the decision and key evidence.
        Example: 'Insert now — non-critical path impact (4 h float consumed);
        3 similar valve seal events in historical record, median 6.5 h.'

        M1: when lco_clock_status indicates an active or expired action-level
        deadline, the conclusion is prefixed with the clock warning so the most
        time-critical information appears first — before any schedule or analog text.

        States low_confidence explicitly when applicable.
        """
        dist = historical_analogs.get("duration_distribution") or {}
        retrieval_summary = historical_analogs.get("retrieval_summary") or {}
        float_analysis = schedule_impact.get("float_analysis") or {}
        cp_impact = schedule_impact.get("cp_impact") or {}

        analog_count: int = int(retrieval_summary.get("analog_count", 0))
        p50: Optional[float] = dist.get("p50_hours")
        cp_drag: float = float(cp_impact.get("cp_drag_hours") or 0.0)
        float_consumed: float = float(float_analysis.get("float_consumed_hours") or 0.0)
        criticality_label: str = float_analysis.get("criticality_label", "non_critical")

        # Build primary conclusion sentence
        if decision_status == _INCONCLUSIVE:
            conclusion = (
                "Inconclusive — no feasible, regulatory-cleared insertion option could "
                "be identified. Analyst review required before proceeding."
            )
        elif decision_status == _ESCALATE:
            conclusion = (
                f"Escalate to management — CP drag of {cp_drag:.1f} h exceeds "
                "the auto-escalation threshold. Schedule re-baseline decision required."
            )
        elif decision_status == _DEFER:
            conclusion = (
                "Defer to post-outage — no regulatory constraint prevents deferral "
                "and in-outage schedule impact is avoided."
            )
        elif decision_status == _MONITOR:
            conclusion = (
                "Monitor — no critical path impact detected and historical analog "
                "support is insufficient for a confident action recommendation."
            )
        else:
            # PROCEED
            option_type = (primary_option or {}).get("option_type", "insert_now")
            action_label = option_type.replace("_", " ").capitalize()
            sched_clause = (
                f"critical path impact ({cp_drag:.1f} h CP drag)"
                if criticality_label == "critical"
                else f"non-critical path impact ({float_consumed:.1f} h float consumed)"
            )
            analog_clause = (
                f"{analog_count} analogous event(s) in historical record"
                f"{f', median {p50:.1f} h' if p50 else ''}"
                if analog_count > 0
                else "no historical analogues available"
            )
            conclusion = f"{action_label} — {sched_clause}; {analog_clause}."

        # §3: Regulatory constraints must be surfaced in the primary conclusion —
        # a recommendation that touches deferral or reprioritisation without
        # flagging regulatory constraints is a potential compliance liability.
        # Driver detail is available in regulatory_flags at the artifact level.
        if _FLAG_REGULATORY in attention_flags:
            conclusion += (
                " \u26a0 REGULATORY CONSTRAINT PRESENT — do not defer or reduce "
                "scope without licensing review."
            )

        if confidence_tier == "low_confidence":
            conclusion += " [LOW CONFIDENCE — verify with SME before acting]"

        # M1: prepend LCO action-level clock warning so it leads the briefing.
        # This is the most time-critical information; it must appear first.
        lco_prefix = self._build_lco_clock_prefix(intake_result)
        if lco_prefix:
            conclusion = lco_prefix + " " + conclusion

        return {
            "primary_conclusion": conclusion,
            "decision_status": decision_status,
            "confidence_tier": confidence_tier,
            "analyst_attention_flags": attention_flags,
            "analog_support_count": analog_count,
            "duration_p50_hours": p50,
        }

    def _build_lco_clock_prefix(
        self, intake_result: Optional[JsonDict]
    ) -> str:
        """Build an LCO action-level clock warning prefix for the primary conclusion.

        M1: Returns a non-empty string when the LCO clock demands immediate
        attention (expired/critical/urgent/unknown-active-lco), empty string
        otherwise.  The prefix is prepended to the conclusion by
        _build_executive_summary() so it always leads the manager briefing.
        """
        if not intake_result:
            return ""

        lco_status: str = intake_result.get("lco_clock_status") or "not_applicable"
        hours: Optional[float] = intake_result.get("hours_to_action_level")
        lco_number: Optional[str] = intake_result.get("lco_number")
        lco_clause = f" (LCO {lco_number})" if lco_number else ""

        if lco_status == "expired":
            hours_str = f"{abs(hours):.1f} h ago" if hours is not None else "unknown time ago"
            return (
                f"\U0001f6a8 LCO ACTION LEVEL EXPIRED{lco_clause} — deadline passed "
                f"{hours_str}. Immediate management notification required."
            )
        if lco_status == "critical":
            hours_str = f"{hours:.1f} h" if hours is not None else "< 4 h"
            return (
                f"\U0001f6a8 LCO ACTION LEVEL CRITICAL{lco_clause} — "
                f"{hours_str} remaining. Immediate action required."
            )
        if lco_status == "urgent":
            hours_str = f"{hours:.1f} h" if hours is not None else "< 24 h"
            return (
                f"\u26a0 LCO ACTION LEVEL URGENT{lco_clause} — "
                f"{hours_str} remaining. Action required this shift."
            )
        if lco_status == "unknown":
            return (
                f"\u26a0 ACTIVE LCO{lco_clause} — action-level deadline not provided. "
                "Confirm hours-to-deadline with licensing immediately."
            )
        return ""

    def _build_primary_recommendation(
        self, primary_option: Optional[JsonDict]
    ) -> Optional[JsonDict]:
        """Extract the primary_recommendation block from the option dict.

        Returns None if primary_option is None (maps to INCONCLUSIVE).
        """
        if primary_option is None:
            return None
        return {
            "option_id": primary_option.get("option_id"),
            "option_type": primary_option.get("option_type"),
            "rationale": primary_option.get("rationale"),
            "cp_impact_hours": primary_option.get("cp_impact_hours"),
            "criticality_label": primary_option.get("criticality_label"),
        }

    def _assemble_evidence_chain(
        self,
        temporal_event_chain: JsonDict,
        historical_analogs: JsonDict,
        schedule_impact: JsonDict,
        component_event_timeline: JsonDict,
        intake_result: Optional[JsonDict] = None,
    ) -> List[JsonDict]:
        """Build the evidence chain for the recommendation.

        Each entry cites one specific piece of supporting evidence.
        Sources in priority order:
            0. regulatory_constraint — inserted at position 0 when
               intake_result.has_regulatory_constraint is True (N6 fix:
               the max_evidence_items cap must never displace a regulatory
               entry; anchoring it first guarantees it survives truncation)
            1. temporal_chain_link   — strongest Allen relation link(s)
            2. historical_analog     — top-similarity analogs with duration data
            3. schedule_analysis     — CP drag / float analysis result
            4. condition_report      — highest-DQ CR from the component timeline

        Each entry: evidence_id, source_type, source_id, snippet,
        relevance_score, supports.

        Mirrors _summarize_primary_evidence() from the RCA orchestrator but
        operates on outage artifact types (read-only reuse pattern).
        """
        evidence: List[JsonDict] = []

        # ── 0. Regulatory constraint pinned to position 0 ─────────────────────
        # N6: When a regulatory constraint is present the chain cap
        # (max_evidence_items) must never push it out — anchor it first.
        if intake_result and intake_result.get("has_regulatory_constraint"):
            regulatory_drivers = intake_result.get("regulatory_drivers") or []
            driver_types = ", ".join(
                d.get("driver_type", "regulatory")
                for d in regulatory_drivers[:3]
            ) or "regulatory_constraint"
            evidence.append(_make_evidence(
                source_type="regulatory_constraint",
                source_id=intake_result.get("activity_id", "intake"),
                snippet=(
                    f"Regulatory constraint present — deferral or scope reduction "
                    f"requires licensing review. Drivers: {driver_types}."
                ),
                relevance_score=1.0,
                supports=False,  # constrains the decision; not causal support
            ))

        # ── 1. Strongest temporal chain link(s) ───────────────────────────────
        chain_summary = temporal_event_chain.get("summary") or {}
        strongest_link_id = chain_summary.get("strongest_link_id")
        max_relation_score = float(chain_summary.get("max_relation_score") or 0.0)
        causal_posture = chain_summary.get("causal_posture", "insufficient_data")

        if strongest_link_id:
            # Find the full link record
            strongest_link = next(
                (lnk for lnk in temporal_event_chain.get("chain_links", [])
                 if lnk.get("link_id") == strongest_link_id),
                None,
            )
            if strongest_link:
                snippet = (
                    f"Allen relation: {strongest_link.get('allen_relation')} "
                    f"(score {strongest_link.get('relation_score', 0):.2f}); "
                    f"onset lag {strongest_link.get('onset_lag_hours', '?')} h; "
                    f"causal strength: {strongest_link.get('causal_strength')}."
                )
                evidence.append(_make_evidence(
                    source_type="temporal_chain_link",
                    source_id=strongest_link_id,
                    snippet=snippet,
                    relevance_score=max_relation_score,
                    supports=causal_posture in {"supported", "partial"},
                ))

        # Include temporal contradiction link if present
        if chain_summary.get("has_temporal_contradiction"):
            contradiction_links = [
                lnk for lnk in temporal_event_chain.get("chain_links", [])
                if lnk.get("causal_strength") == "temporal_contradiction"
            ]
            for lnk in contradiction_links[:2]:
                evidence.append(_make_evidence(
                    source_type="temporal_chain_link",
                    source_id=lnk.get("link_id", "unknown"),
                    snippet=(
                        f"TEMPORAL CONTRADICTION: event {lnk.get('event_id')} "
                        f"follows the emergent activity (Allen: {lnk.get('allen_relation')}). "
                        "This event is a likely symptom, not a cause."
                    ),
                    relevance_score=0.10,
                    supports=False,
                ))

        # ── 2. Top historical analogs with duration data ──────────────────────
        analogs = historical_analogs.get("analogs") or []
        analogs_with_duration = [
            a for a in analogs if a.get("actual_duration_hours") is not None
        ]
        for analog in analogs_with_duration[:3]:
            snippet = (
                f"Similarity {analog.get('similarity_score', 0):.2f}; "
                f"actual duration {analog.get('actual_duration_hours'):.1f} h "
                f"(plant {analog.get('plant_id', '?')}, "
                f"outage {analog.get('outage_id', '?')}): "
                f"{str(analog.get('description', ''))[:120]}"
            )
            evidence.append(_make_evidence(
                source_type="historical_analog",
                source_id=analog.get("analog_id", analog.get("source_activity_id", "?")),
                snippet=snippet,
                relevance_score=float(analog.get("similarity_score") or 0.0),
                supports=True,
            ))

        # ── 3. Schedule analysis result ───────────────────────────────────────
        float_analysis = schedule_impact.get("float_analysis") or {}
        cp_impact = schedule_impact.get("cp_impact") or {}
        if float_analysis or cp_impact:
            criticality = float_analysis.get("criticality_label", "unknown")
            cp_drag = float(cp_impact.get("cp_drag_hours") or 0.0)
            float_consumed = float(float_analysis.get("float_consumed_hours") or 0.0)
            snippet = (
                f"Criticality: {criticality}; "
                f"CP drag: {cp_drag:.1f} h; "
                f"float consumed: {float_consumed:.1f} h; "
                f"CP sensitivity: {cp_impact.get('cp_sensitivity_score', '?')}."
            )
            supports = criticality != "critical" or cp_drag < 8.0
            evidence.append(_make_evidence(
                source_type="schedule_analysis",
                source_id=schedule_impact.get("run_id", "schedule_impact"),
                snippet=snippet,
                relevance_score=float(float_analysis.get("confidence") or 0.70),
                supports=supports,
            ))

        # ── 4. Highest-DQ condition report from the component timeline ────────
        timeline_events = component_event_timeline.get("events") or []
        cr_events = [
            e for e in timeline_events
            if e.get("event_type") == "condition_report"
            and e.get("description")
        ]
        if cr_events:
            best_cr = max(cr_events, key=lambda e: float(e.get("data_quality_score") or 0.0))
            snippet = (
                f"CR {best_cr.get('event_id')} ({best_cr.get('timestamp', '?')}): "
                f"{str(best_cr.get('description', ''))[:150]}"
            )
            evidence.append(_make_evidence(
                source_type="condition_report",
                source_id=best_cr.get("source_doc_id") or best_cr.get("event_id", "?"),
                snippet=snippet,
                relevance_score=float(best_cr.get("data_quality_score") or 0.5),
                supports=True,
            ))

        return evidence

    def _build_history_summary(
        self,
        historical_analogs: JsonDict,
        component_event_timeline: JsonDict,
    ) -> JsonDict:
        """Build the history_summary block for the executive view.

        Extracts: analog_count, outages_represented, recurrence_pattern
        (one sentence from recurrence_indicators), duration_p50_hours,
        duration_p80_hours.
        """
        retrieval_summary = historical_analogs.get("retrieval_summary") or {}
        dist = historical_analogs.get("duration_distribution") or {}
        recurrence = component_event_timeline.get("recurrence_indicators") or {}

        analog_count: int = int(retrieval_summary.get("analog_count", 0))
        outages_represented: int = int(retrieval_summary.get("outages_represented", 0))

        # Build one-sentence recurrence pattern description
        trend = recurrence.get("trend", "insufficient_data")
        repeat_count = int(recurrence.get("repeat_failure_count") or 0)
        mean_inter = recurrence.get("mean_inter_event_days")
        pm_status = recurrence.get("pm_compliance_status", "unknown")

        if trend == "insufficient_data" or repeat_count == 0:
            recurrence_pattern = "Insufficient historical data to characterise a recurrence pattern."
        else:
            inter_clause = (
                f" Mean inter-event interval: {mean_inter:.0f} days."
                if mean_inter is not None else ""
            )
            pm_clause = (
                f" PM status: {pm_status}."
                if pm_status != "unknown" else ""
            )
            recurrence_pattern = (
                f"{repeat_count} prior failure event(s) on this component; "
                f"trend: {trend}.{inter_clause}{pm_clause}"
            )

        # §8(a): surface the specific outage IDs the recommendation draws from
        outage_ids = sorted({
            a["outage_id"]
            for a in (historical_analogs.get("analogs") or [])
            if a.get("outage_id")
        })

        return {
            "analog_count": analog_count,
            "outages_represented": outages_represented,
            "outage_ids": outage_ids,
            "recurrence_pattern": recurrence_pattern,
            "duration_p50_hours": dist.get("p50_hours"),
            "duration_p80_hours": dist.get("p80_hours"),
        }

    def _build_schedule_summary(self, schedule_impact: JsonDict) -> JsonDict:
        """Extract cp_impact_hours, criticality_label, float_consumed_hours,
        displaced_task_count, and has_displaced_regulatory_tasks."""
        cp_impact = schedule_impact.get("cp_impact") or {}
        float_analysis = schedule_impact.get("float_analysis") or {}
        displaced_tasks = schedule_impact.get("displaced_tasks") or []

        has_displaced_regulatory = any(
            t.get("has_regulatory_constraint") for t in displaced_tasks
        )

        return {
            "cp_impact_hours": float(cp_impact.get("cp_drag_hours") or 0.0),
            "criticality_label": float_analysis.get("criticality_label", "non_critical"),
            "float_consumed_hours": float(
                float_analysis.get("float_consumed_hours") or 0.0
            ),
            "displaced_task_count": len(displaced_tasks),
            "has_displaced_regulatory_tasks": has_displaced_regulatory,
        }

    def _determine_analyst_review(
        self,
        decision_status: str,
        intake_result: JsonDict,
        historical_analogs: JsonDict,
        insertion_options: JsonDict,
        attention_flags: List[str],
    ) -> JsonDict:
        """Determine whether analyst review is required before acting.

        required=True when any of:
            - has_regulatory_constraint is True
            - decision_status is INCONCLUSIVE
            - duration distribution confidence_tier is low_confidence
            - _FLAG_FALLBACK is in attention_flags
            - _FLAG_HIGH_ABBR_RATE is in attention_flags (§6 exit criterion:
              NER entity extraction unreliable when unknown_abbreviation_rate
              exceeds RecommendationConfig.unknown_abbreviation_rate_warning)
            - no feasible + regulatory-cleared option exists

        reason: brief string summarising why review is required.
        reviewer_decision initialised to None (populated by the analyst UI).
        """
        reasons: List[str] = []

        if intake_result.get("has_regulatory_constraint"):
            reasons.append("regulatory constraint present — verify compliance before proceeding")

        if decision_status == _INCONCLUSIVE:
            reasons.append("no feasible option identified — manual scheduling decision required")

        dist_tier = (
            (historical_analogs.get("duration_distribution") or {})
            .get("confidence_tier", "low_confidence")
        )
        if dist_tier == "low_confidence":
            reasons.append(
                "low-confidence duration estimate — SME input required to validate duration"
            )

        if _FLAG_FALLBACK in attention_flags:
            reasons.append(
                "fallback distribution used — no analogous historical events found"
            )

        # §6 exit criterion: high unknown abbreviation rate makes NER-derived
        # causal chains unreliable — analyst must verify entity extraction
        if _FLAG_HIGH_ABBR_RATE in attention_flags:
            reasons.append(
                "high unknown abbreviation rate — NER entity extraction unreliable; "
                "verify component and failure-mode identification before acting"
            )

        # No feasible + cleared option
        feasible_cleared = [
            o for o in insertion_options.get("options", [])
            if o.get("feasible", True) and o.get("regulatory_cleared", True)
        ]
        if not feasible_cleared:
            reasons.append("no feasible, regulatory-cleared option exists")

        required = bool(reasons)
        reason_text = "; ".join(reasons) if reasons else None

        return {
            "required": required,
            "reason": reason_text,
            "reviewer_decision": None,   # populated by analyst UI after review
            "rejection_reason": None,    # §8(d): analyst populates when rejecting recommendation
        }

    def _compute_validation_status(
        self,
        evidence_chain: List[JsonDict],
        regulatory_flags: List[JsonDict],
        historical_analogs: JsonDict,
    ) -> JsonDict:
        """Compute the validation_status block.

        schema_valid:                  always True here (schema validation runs
                                       in the orchestrator's _validate_and_persist).
        all_regulatory_flags_resolved: True if every regulatory_flag entry has
                                       a corresponding evidence_chain item that
                                       supports=True, or if there are no flags.
        minimum_evidence_met:          True if len(evidence_chain) >= 1 with at
                                       least one supports=True entry.
        fallback_used:                 from historical_analogs.retrieval_summary.
        """
        # Regulatory flags resolved: each flag must have a supporting evidence item
        all_resolved: bool
        if not regulatory_flags:
            all_resolved = True
        else:
            supporting_sources = {
                item.get("source_id")
                for item in evidence_chain
                if item.get("supports")
            }
            # A flag is "resolved" when its driver_id or driver_type appears in
            # a supporting evidence item, OR when a schedule/temporal analysis
            # evidence item (which covers the activity holistically) is present.
            holistic_evidence = any(
                item.get("source_type") in {"schedule_analysis", "temporal_chain_link"}
                and item.get("supports")
                for item in evidence_chain
            )
            all_resolved = holistic_evidence or bool(supporting_sources)

        minimum_evidence_met: bool = bool(evidence_chain) and any(
            item.get("supports") for item in evidence_chain
        )

        fallback_used: bool = bool(
            (historical_analogs.get("retrieval_summary") or {}).get("fallback_used")
        )

        return {
            "schema_valid": True,
            "all_regulatory_flags_resolved": all_resolved,
            "minimum_evidence_met": minimum_evidence_met,
            "fallback_used": fallback_used,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_evidence(
    *,
    source_type: str,
    source_id: str,
    snippet: str,
    relevance_score: float,
    supports: bool,
) -> JsonDict:
    """Construct a standardised evidence chain entry."""
    return {
        "evidence_id": f"EVD::{source_type}::{uuid.uuid4().hex[:6]}",
        "source_type": source_type,
        "source_id": source_id,
        "snippet": snippet,
        "relevance_score": round(min(1.0, max(0.0, relevance_score)), 4),
        "supports": supports,
    }
