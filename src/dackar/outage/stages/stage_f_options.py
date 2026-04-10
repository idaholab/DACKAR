"""
Stage F — Insertion Option Generator.

Responsibilities:
    1. Generate a candidate set of options for handling the emergent activity
       (insert_now, defer_to_post_outage, pre_outage_staging,
       add_contingency_buffer, parallel_execution, scope_reduction,
       escalate_to_management).
    2. Assess feasibility of each option (resource availability, dependency
       constraints, technical limitations).
    3. Check regulatory clearance for each option against the regulatory_drivers
       identified in Stage A.  Options that would violate a TS surveillance,
       NRC commitment, or CAP commitment are marked regulatory_cleared=False
       and retained with a block reason (never silently dropped).
    4. Score each option with a composite risk score.
    5. Rank and select the recommended option (top feasible + regulatory-cleared
       by risk score).

Output schema: outage/schemas/insertion_options.json

This is the only pipeline stage with no direct reuse target from RCA —
the option generation and ranking logic is unique to outage decision support.
The scoring pattern is adapted from RuleBasedCausalityEngineV31 but the
dimensions are entirely different.

Scoring dimensions for each option (lower total = lower risk = better):
    cp_impact       (0.40) — normalised CP drag hours; 0.0 for defer/escalate
    confidence      (0.30) — 1 − option_confidence; high confidence → low risk
    resource_ready  (0.20) — 0.0 if no resource conflicts, 1.0 if conflicts present
    causal_urgency  (0.10) — action options: 1 − urgency (urgency to act lowers
                              risk of acting); non-action options: urgency
                              (urgency to act increases risk of not acting)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# Option type constants
_INSERT_NOW = "insert_now"
_DEFER = "defer_to_post_outage"
_PRE_STAGE = "pre_outage_staging"
_CONTINGENCY = "add_contingency_buffer"
_PARALLEL = "parallel_execution"
_SCOPE_REDUCTION = "scope_reduction"
_ESCALATE = "escalate_to_management"

# Scoring weights
_W_CP_IMPACT = 0.40
_W_CONFIDENCE = 0.30
_W_RESOURCE = 0.20
_W_URGENCY = 0.10

# Non-action option types — for these, high causal urgency increases risk
_NON_ACTION_TYPES = {_DEFER, _ESCALATE}

# Regulatory driver types that prohibit deferral
_DEFER_PROHIBITED_TYPES = {
    "technical_specification",
    "limiting_condition_for_operation",
    "nrc_commitment",
    "nrc_regulation",
    "surveillance_requirement",
    "operability_determination",
    "hold_point",
    "mode_change_constraint",
}

# Regulatory driver types that prohibit scope reduction
_SCOPE_REDUCTION_PROHIBITED_TYPES = {
    "technical_specification",
    "limiting_condition_for_operation",
    "surveillance_requirement",
}

# Duration distribution confidence tier → confidence float
_TIER_TO_CONFIDENCE: Dict[str, float] = {
    "data_supported": 0.85,
    "sme_informed":   0.65,
    "low_confidence": 0.40,
}

# Causal posture → urgency score used in the causal_urgency dimension
_POSTURE_TO_URGENCY: Dict[str, float] = {
    "supported":        0.80,   # strong causal history — urgency is high
    "contradicted":     0.70,   # contradictions need resolution — still urgent
    "partial":          0.50,   # moderate evidence
    "weak":             0.20,
    "insufficient_data": 0.40,  # neutral
}


@dataclass
class InsertionOptionConfig:
    """Configuration for Stage F."""

    max_options: int = 6
    """Maximum options to generate."""

    include_infeasible_options: bool = True
    """Retain infeasible options with feasible=False and an infeasibility_reason
    so the outage manager can see why they were ruled out."""

    include_regulatory_blocked_options: bool = True
    """Retain regulatory-blocked options with regulatory_cleared=False and a
    block reason.  Hiding them would mask the constraint from the user."""

    escalate_if_cp_drag_exceeds_hours: float = 24.0
    """Automatically generate an escalate_to_management option if the
    insert_now CP drag exceeds this threshold."""

    near_critical_float_threshold_hours: float = 8.0
    """Passed down from orchestrator config for option feasibility checks."""

    contingency_buffer_p_level: float = 0.80
    """Percentile to use for contingency buffer sizing.  p80 − p50 of the
    duration distribution sets the buffer width."""

    scope_reduction_fraction: float = 0.60
    """Fraction of the p50 duration estimate used for the scope-reduction option.
    Represents executing minimum required scope only (e.g. 60% of full scope)."""

    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "cp_impact":      _W_CP_IMPACT,
        "confidence":     _W_CONFIDENCE,
        "resource_ready": _W_RESOURCE,
        "causal_urgency": _W_URGENCY,
    })


class InsertionOptionGenerator:
    """Concrete Stage F implementation.

    Args:
        config: Stage configuration.
    """

    def __init__(self, config: Optional[InsertionOptionConfig] = None) -> None:
        self.config = config or InsertionOptionConfig()

    # ── Protocol method ───────────────────────────────────────────────────────

    def generate(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        temporal_event_chain: JsonDict,
        schedule_impact_assessment: JsonDict,
        historical_analogs: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """Execute Stage F for one emergent activity.

        Returns:
            InsertionOptions artifact conforming to
            outage/schemas/insertion_options.json.
        """
        run_id: str = run_context["run_id"]
        activity_id: str = emergent_activity["activity_id"]
        LOGGER.debug(
            "Stage F option generation for %s (run=%s)", activity_id, run_id
        )

        regulatory_drivers: List[JsonDict] = intake_result.get("regulatory_drivers", [])
        causal_posture: str = (
            temporal_event_chain.get("summary", {}).get("causal_posture", "insufficient_data")
        )

        # Generate all candidate options
        candidates: List[JsonDict] = []
        candidates.append(
            self._generate_insert_now(
                emergent_activity, intake_result, schedule_impact_assessment, historical_analogs
            )
        )
        candidates.append(
            self._generate_defer(
                emergent_activity, intake_result, schedule_impact_assessment
            )
        )
        candidates.append(
            self._generate_contingency_buffer(
                emergent_activity, schedule_impact_assessment, historical_analogs
            )
        )
        candidates.extend(
            self._generate_parallel_option(
                emergent_activity, schedule_impact_assessment
            )
        )
        candidates.append(
            self._generate_scope_reduction(
                emergent_activity, schedule_impact_assessment, historical_analogs
            )
        )
        cp_drag = (
            schedule_impact_assessment.get("cp_impact", {}).get("cp_drag_hours", 0.0) or 0.0
        )
        if cp_drag > self.config.escalate_if_cp_drag_exceeds_hours:
            candidates.append(self._generate_escalate(emergent_activity, schedule_impact_assessment))

        # Check regulatory clearance for each option
        for option in candidates:
            cleared, block_reason = self._check_regulatory_clearance(
                option, regulatory_drivers
            )
            option["regulatory_cleared"] = cleared
            option["regulatory_block_reason"] = block_reason

        # Score and rank
        for option in candidates:
            option["risk_score"] = self._score_option(
                option, schedule_impact_assessment, historical_analogs, causal_posture
            )

        options = self._rank_options(candidates)

        # Select recommended option: top feasible + regulatory-cleared
        recommended_id = self._select_recommended(options)

        summary = self._build_ranking_summary(options)

        return {
            "activity_id": activity_id,
            "run_id": run_id,
            "generated_at": run_context.get("started_at", ""),
            "options": options,
            "recommended_option_id": recommended_id,
            "recommendation_confidence": self._recommendation_confidence(
                options, recommended_id, historical_analogs
            ),
            "ranking_summary": summary,
            "provenance": {
                "generated_by": self.__class__.__name__,
                "run_id": run_id,
                "schedule_impact_assessment_id": schedule_impact_assessment.get("run_id"),
            },
        }

    # ── Option generators ─────────────────────────────────────────────────────

    def _generate_insert_now(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        schedule_impact: JsonDict,
        analogs: JsonDict,
    ) -> JsonDict:
        """Generate the insert_now option.

        Uses the schedule_impact_assessment insertion_point and float_analysis
        to set cp_impact_hours and criticality_label.

        feasible=True unless resource_conflicts are present and cannot be resolved.
        Resource conflicts make the option infeasible only when the conflict type
        is 'crew_unavailable' — tool/scaffold conflicts are usually resolvable
        within the window and are noted in the rationale rather than blocking.
        """
        float_analysis = schedule_impact.get("float_analysis") or {}
        cp_impact = schedule_impact.get("cp_impact") or {}
        insertion_point = schedule_impact.get("insertion_point") or {}
        resource_conflicts = schedule_impact.get("resource_conflicts") or []

        cp_drag_hours: float = float(cp_impact.get("cp_drag_hours") or 0.0)
        criticality_label: str = float_analysis.get("criticality_label", "non_critical")
        float_consumed: float = float(float_analysis.get("float_consumed_hours") or 0.0)

        # Duration confidence from analog distribution
        dist = analogs.get("duration_distribution") or {}
        confidence = _tier_confidence(dist.get("confidence_tier"))

        # Feasibility: hard crew conflicts make insertion infeasible
        hard_conflicts = [
            c for c in resource_conflicts
            if c.get("conflict_type") == "crew_unavailable"
        ]
        feasible = len(hard_conflicts) == 0
        infeasibility_reason: Optional[str] = None
        if not feasible:
            skills = ", ".join(c.get("skill_required", "?") for c in hard_conflicts)
            infeasibility_reason = (
                f"Crew unavailable for required skill(s): {skills}. "
                "Resolve crew conflict before inserting."
            )

        # Rationale
        if criticality_label == "critical":
            rationale = (
                f"Insert immediately on critical path; {cp_drag_hours:.1f} h CP drag expected. "
                f"Insertion point: {insertion_point.get('task_id', 'TBD')}."
            )
        elif criticality_label == "near_critical":
            rationale = (
                f"Insert now — near-critical path; {float_consumed:.1f} h float consumed. "
                "Monitor closely."
            )
        else:
            rationale = (
                f"Insert now — non-critical path; {float_consumed:.1f} h float consumed, "
                "no CP drag anticipated."
            )

        if resource_conflicts and feasible:
            rationale += f" Note: {len(resource_conflicts)} non-blocking resource conflict(s) to coordinate."

        return _make_option(
            activity_id=emergent_activity["activity_id"],
            option_type=_INSERT_NOW,
            rationale=rationale,
            feasible=feasible,
            infeasibility_reason=infeasibility_reason,
            cp_impact_hours=cp_drag_hours,
            criticality_label=criticality_label,
            confidence=confidence,
            resource_conflicts=resource_conflicts,
        )

    def _generate_defer(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        schedule_impact: JsonDict,
    ) -> JsonDict:
        """Generate the defer_to_post_outage option.

        feasible=True for non-safety-related items; infeasible when the activity
        is safety-critical or in an active LCO window (regulatory clearance is
        checked separately by _check_regulatory_clearance).

        cp_impact_hours=0.0 — deferral avoids all in-outage schedule impact.
        Confidence is inherently high (0.85) because the scheduling outcome is
        certain; the residual risk lies in the plant state during the deferral
        period, which is captured in causal_urgency during scoring.
        """
        is_safety_related: bool = bool(
            emergent_activity.get("safety_related")
            or emergent_activity.get("safety_classification") == "safety_related"
        )
        active_lco: bool = bool(emergent_activity.get("active_lco"))

        feasible = not (is_safety_related or active_lco)
        infeasibility_reason: Optional[str] = None
        if is_safety_related:
            infeasibility_reason = (
                "Activity is safety-related — deferral to post-outage is not permitted "
                "without engineering disposition."
            )
        elif active_lco:
            infeasibility_reason = (
                "Active LCO entry is in effect — deferral cannot proceed until "
                "the LCO condition is resolved."
            )

        rationale = (
            "Defer all work to the next planned maintenance window. "
            "No in-outage schedule impact. Residual plant risk must be evaluated "
            "before approving deferral."
        )

        return _make_option(
            activity_id=emergent_activity["activity_id"],
            option_type=_DEFER,
            rationale=rationale,
            feasible=feasible,
            infeasibility_reason=infeasibility_reason,
            cp_impact_hours=0.0,
            criticality_label="non_critical",
            confidence=0.85,
            resource_conflicts=[],
        )

    def _generate_contingency_buffer(
        self,
        emergent_activity: JsonDict,
        schedule_impact: JsonDict,
        analogs: JsonDict,
    ) -> JsonDict:
        """Generate the add_contingency_buffer option.

        Proposes reserving a float buffer sized at p80 − p50 duration hours
        (from the historical analog distribution) rather than scheduling the full
        scope immediately.  Suitable when scope is uncertain and the activity may
        be shorter than estimated.

        If the activity has not yet started and there is adequate lead time before
        the outage window, the option type is upgraded to pre_outage_staging.
        feasible=True when the buffer fits within the available float.
        """
        dist = analogs.get("duration_distribution") or {}
        p50: float = float(dist.get("p50_hours") or 0.0)
        p80: float = float(dist.get("p80_hours") or 0.0)
        buffer_hours: float = max(0.0, p80 - p50)

        float_analysis = schedule_impact.get("float_analysis") or {}
        float_consumed: float = float(float_analysis.get("float_consumed_hours") or 0.0)
        criticality_label: str = float_analysis.get("criticality_label", "non_critical")

        # Pre-outage staging upgrade: activity not yet started + lead time > 0
        actual_start = emergent_activity.get("actual_start")
        detection_ts = emergent_activity.get("detection_timestamp")
        outage_start = emergent_activity.get("outage_start")
        option_type = _CONTINGENCY

        if not actual_start and outage_start and detection_ts:
            # Crude check: if detection is before outage start there may be staging time
            option_type = _PRE_STAGE

        # Buffer absorbs network float: use available_float_before (actual network
        # float at insertion point) if populated by Stage E; fall back to
        # float_consumed_hours (activity duration proxy) when not available.
        available_float: float = float(
            float_analysis.get("available_float_before")
            or float_consumed
            or 0.0
        )
        remaining_float: float = max(0.0, available_float - p50)
        feasible = buffer_hours <= remaining_float or criticality_label == "non_critical"
        infeasibility_reason: Optional[str] = None
        if not feasible:
            infeasibility_reason = (
                f"Required contingency buffer ({buffer_hours:.1f} h) exceeds "
                f"available remaining float after base scope ({remaining_float:.1f} h)."
            )

        confidence = _tier_confidence(dist.get("confidence_tier")) * 0.80  # slightly lower

        if option_type == _PRE_STAGE:
            rationale = (
                f"Stage work before outage start. Reserve {buffer_hours:.1f} h buffer "
                f"(p80 − p50 from {dist.get('sample_size', '?')} analogues). "
                "Reduces in-outage schedule pressure."
            )
        else:
            rationale = (
                f"Reserve a {buffer_hours:.1f} h contingency buffer "
                f"(p80 − p50 from {dist.get('sample_size', '?')} analogues) "
                "rather than committing full scope now. Reassess after initial work."
            )

        return _make_option(
            activity_id=emergent_activity["activity_id"],
            option_type=option_type,
            rationale=rationale,
            feasible=feasible,
            infeasibility_reason=infeasibility_reason,
            cp_impact_hours=buffer_hours,
            criticality_label=criticality_label,
            confidence=round(confidence, 3),
            resource_conflicts=[],
        )

    def _generate_parallel_option(
        self,
        emergent_activity: JsonDict,
        schedule_impact: JsonDict,
    ) -> List[JsonDict]:
        """Generate parallel_execution option(s) if a non-conflicting window exists.

        Returns a list (empty if no viable parallel window is identified).

        A parallel window is viable when:
            - criticality_label is non_critical (activity has float to work in parallel)
            - OR displaced_tasks contains non-critical tasks whose float absorbs
              the new activity without driving the critical path.

        Discipline conflict check: the emergent activity discipline is compared
        against displaced task disciplines — same discipline with same crew means
        parallel execution would create a crew conflict.
        """
        float_analysis = schedule_impact.get("float_analysis") or {}
        criticality_label: str = float_analysis.get("criticality_label", "non_critical")
        displaced_tasks: List[JsonDict] = schedule_impact.get("displaced_tasks") or []
        resource_conflicts = schedule_impact.get("resource_conflicts") or []

        emergent_discipline: Optional[str] = emergent_activity.get("discipline")

        # Identify non-critical displaced tasks with different discipline
        parallel_candidates = [
            t for t in displaced_tasks
            if t.get("criticality_label", "non_critical") == "non_critical"
            and t.get("discipline") != emergent_discipline
            and not t.get("has_regulatory_constraint", False)
        ]

        # Only viable when non-critical (has float) or parallel candidates exist
        if criticality_label == "critical" and not parallel_candidates:
            return []

        # Crew conflict check: if all resource conflicts are hard crew conflicts,
        # parallel execution is not feasible.
        hard_crew_conflicts = [
            c for c in resource_conflicts
            if c.get("conflict_type") == "crew_unavailable"
        ]
        feasible = len(hard_crew_conflicts) == 0
        infeasibility_reason: Optional[str] = None
        if not feasible:
            skills = ", ".join(c.get("skill_required", "?") for c in hard_crew_conflicts)
            infeasibility_reason = (
                f"Crew conflict on required skill(s): {skills} prevents parallel execution."
            )

        if parallel_candidates:
            task_names = ", ".join(
                t.get("task_name", t.get("task_id", "?"))
                for t in parallel_candidates[:3]
            )
            rationale = (
                f"Execute in parallel with non-critical task(s): {task_names}. "
                "Different discipline avoids crew conflict."
            )
        else:
            rationale = (
                "Execute in parallel during available float window. "
                f"Activity is {criticality_label} — float absorbs concurrent execution."
            )

        cp_impact = schedule_impact.get("cp_impact") or {}
        cp_drag = float(cp_impact.get("cp_drag_hours") or 0.0) * 0.5  # parallel halves effective drag

        option = _make_option(
            activity_id=emergent_activity["activity_id"],
            option_type=_PARALLEL,
            rationale=rationale,
            feasible=feasible,
            infeasibility_reason=infeasibility_reason,
            cp_impact_hours=round(cp_drag, 2),
            criticality_label=criticality_label,
            confidence=0.65,
            resource_conflicts=resource_conflicts,
        )
        return [option]

    def _generate_scope_reduction(
        self,
        emergent_activity: JsonDict,
        schedule_impact: JsonDict,
        analogs: JsonDict,
    ) -> JsonDict:
        """Generate the scope_reduction option.

        Proposes performing only the minimum required scope to address the
        immediate safety or operability concern, deferring full corrective action
        to the next planned window.

        Duration estimate = p50 × config.scope_reduction_fraction (default 60%).
        feasible=True by default (scope reduction is always technically possible
        even if not always preferred operationally).  Regulatory clearance for
        TS/surveillance items is checked separately in _check_regulatory_clearance.
        """
        dist = analogs.get("duration_distribution") or {}
        p50: float = float(dist.get("p50_hours") or 0.0)
        reduced_hours: float = round(p50 * self.config.scope_reduction_fraction, 2)

        float_analysis = schedule_impact.get("float_analysis") or {}
        criticality_label: str = float_analysis.get("criticality_label", "non_critical")
        cp_impact = schedule_impact.get("cp_impact") or {}
        cp_drag_full: float = float(cp_impact.get("cp_drag_hours") or 0.0)
        # Scope reduction proportionally reduces CP drag
        cp_drag_reduced = round(cp_drag_full * self.config.scope_reduction_fraction, 2)

        confidence = _tier_confidence(dist.get("confidence_tier")) * 0.70  # scope estimate uncertain

        rationale = (
            f"Execute minimum required scope only (est. {reduced_hours:.1f} h, "
            f"{int(self.config.scope_reduction_fraction * 100)}% of full p50 estimate). "
            "Defer full corrective action to next planned outage. "
            "Requires engineering disposition of minimum-scope adequacy."
        )

        return _make_option(
            activity_id=emergent_activity["activity_id"],
            option_type=_SCOPE_REDUCTION,
            rationale=rationale,
            feasible=True,  # always technically possible; regulatory clearance checked separately
            infeasibility_reason=None,
            cp_impact_hours=cp_drag_reduced,
            criticality_label=criticality_label,
            confidence=round(confidence, 3),
            resource_conflicts=[],
        )

    def _generate_escalate(
        self,
        emergent_activity: JsonDict,
        schedule_impact: JsonDict,
    ) -> JsonDict:
        """Generate the escalate_to_management option.

        Always feasible and regulatory-cleared — escalation is always a valid
        path.  Generated automatically when CP drag exceeds
        config.escalate_if_cp_drag_exceeds_hours.

        cp_impact_hours reflects the full insert_now impact (same work, higher
        authority required to approve it).
        """
        cp_impact = schedule_impact.get("cp_impact") or {}
        cp_drag_hours: float = float(cp_impact.get("cp_drag_hours") or 0.0)
        float_analysis = schedule_impact.get("float_analysis") or {}
        criticality_label: str = float_analysis.get("criticality_label", "critical")

        rationale = (
            f"CP drag ({cp_drag_hours:.1f} h) exceeds the "
            f"{self.config.escalate_if_cp_drag_exceeds_hours:.0f} h threshold — "
            "escalate to outage management for schedule re-baseline decision. "
            "This option does not resolve the underlying activity; it initiates "
            "the management review process."
        )

        return _make_option(
            activity_id=emergent_activity["activity_id"],
            option_type=_ESCALATE,
            rationale=rationale,
            feasible=True,
            infeasibility_reason=None,
            cp_impact_hours=cp_drag_hours,
            criticality_label=criticality_label,
            confidence=0.90,  # high confidence that escalation is available
            resource_conflicts=[],
        )

    # ── Scoring and ranking ───────────────────────────────────────────────────

    def _check_regulatory_clearance(
        self,
        option: JsonDict,
        regulatory_drivers: List[JsonDict],
    ) -> Tuple[bool, Optional[str]]:
        """Check whether an option is compatible with all regulatory constraints.

        Returns (regulatory_cleared, block_reason).

        Rules:
            defer_to_post_outage — blocked when any driver has defer_prohibited=True
            scope_reduction      — blocked when any driver type is in
                                   _SCOPE_REDUCTION_PROHIBITED_TYPES (cannot reduce
                                   scope below a TS minimum or surveillance requirement)
            insert_now / contingency / pre_outage_staging / parallel / escalate
                                 — always cleared (these options perform the work)
        """
        option_type = option.get("option_type")

        if option_type == _DEFER:
            blocking = [
                d for d in regulatory_drivers
                if d.get("defer_prohibited", False)
                or d.get("driver_type") in _DEFER_PROHIBITED_TYPES
            ]
            if blocking:
                driver_types = ", ".join(
                    d.get("driver_type", "unknown") for d in blocking[:3]
                )
                return False, (
                    f"Deferral prohibited by regulatory constraint(s): {driver_types}. "
                    "Work must be completed during this outage."
                )

        elif option_type == _SCOPE_REDUCTION:
            blocking = [
                d for d in regulatory_drivers
                if d.get("driver_type") in _SCOPE_REDUCTION_PROHIBITED_TYPES
            ]
            if blocking:
                driver_types = ", ".join(
                    d.get("driver_type", "unknown") for d in blocking[:3]
                )
                return False, (
                    f"Scope reduction not permitted: {driver_types} requires full scope execution. "
                    "Engineering disposition needed to accept reduced scope."
                )

        return True, None

    def _score_option(
        self,
        option: JsonDict,
        schedule_impact: JsonDict,
        analogs: JsonDict,
        causal_posture: str,
    ) -> float:
        """Compute a composite risk score in [0, 1] for this option.

        Lower score = lower risk = better option.

        Dimensions:
            cp_impact (W=0.40):
                Normalise cp_impact_hours against baseline_cp_hours.
                0.0 drag → 0.0; drag ≥ baseline → 1.0; clamped to [0, 1].

            confidence (W=0.30):
                1 − option_confidence.  High confidence → low risk contribution.

            resource_ready (W=0.20):
                0.0 = no resource conflicts; 1.0 = conflicts present.

            causal_urgency (W=0.10):
                Action options  (insert_now, contingency, parallel, scope_reduction):
                    1 − urgency (high urgency to act lowers risk of taking action)
                Non-action options (defer, escalate):
                    urgency (high urgency to act increases risk of not acting)
        """
        weights = self.config.scoring_weights
        # baseline_cp_hours is nested under cp_impact in the Stage E artifact;
        # fall back to a top-level key for test-helper compatibility.
        baseline_cp: float = float(
            (schedule_impact.get("cp_impact") or {}).get("baseline_cp_hours")
            or schedule_impact.get("baseline_cp_hours")
            or 1.0
        )
        cp_drag: float = float(option.get("cp_impact_hours") or 0.0)
        cp_impact_score = min(1.0, cp_drag / max(baseline_cp, 0.001))

        confidence: float = float(option.get("confidence") or 0.5)
        confidence_score = 1.0 - confidence

        resource_conflicts = option.get("resource_conflicts") or []
        resource_score = 1.0 if resource_conflicts else 0.0

        urgency: float = _POSTURE_TO_URGENCY.get(causal_posture, 0.40)
        option_type = option.get("option_type", "")
        if option_type in _NON_ACTION_TYPES:
            urgency_score = urgency           # high urgency → higher risk for non-action
        else:
            urgency_score = 1.0 - urgency    # high urgency → lower risk for taking action

        risk = (
            weights.get("cp_impact",      _W_CP_IMPACT)      * cp_impact_score
            + weights.get("confidence",   _W_CONFIDENCE)     * confidence_score
            + weights.get("resource_ready", _W_RESOURCE)     * resource_score
            + weights.get("causal_urgency", _W_URGENCY)      * urgency_score
        )
        return round(min(1.0, max(0.0, risk)), 4)

    def _rank_options(self, candidates: List[JsonDict]) -> List[JsonDict]:
        """Sort options by risk_score ascending.

        Sort key: (infeasible, regulatory_blocked, risk_score).
        Feasible + cleared options appear first, ordered by risk_score.
        Infeasible options are pushed after regulatory-blocked ones.
        """
        def _sort_key(opt: JsonDict) -> Tuple:
            infeasible = 0 if opt.get("feasible", True) else 2
            blocked = 0 if opt.get("regulatory_cleared", True) else 1
            return (infeasible + blocked, opt.get("risk_score", 1.0))

        sorted_options = sorted(candidates, key=_sort_key)

        if not self.config.include_infeasible_options:
            sorted_options = [o for o in sorted_options if o.get("feasible", True)]
        if not self.config.include_regulatory_blocked_options:
            sorted_options = [o for o in sorted_options if o.get("regulatory_cleared", True)]

        return sorted_options[: self.config.max_options]

    def _select_recommended(self, options: List[JsonDict]) -> Optional[str]:
        """Return the option_id of the top-ranked feasible + regulatory-cleared option.

        Returns None if no such option exists (maps to INCONCLUSIVE in Stage G).
        """
        for option in options:
            if option.get("feasible", True) and option.get("regulatory_cleared", True):
                return option["option_id"]
        return None

    def _recommendation_confidence(
        self,
        options: List[JsonDict],
        recommended_id: Optional[str],
        analogs: JsonDict,
    ) -> Optional[str]:
        """Map to data_supported / sme_informed / low_confidence.

        Derived from the duration distribution confidence_tier of Stage D.
        Degrades to low_confidence when no feasible option is available.
        """
        if recommended_id is None:
            return "low_confidence"
        dist = analogs.get("duration_distribution") or {}
        return dist.get("confidence_tier", "low_confidence")

    def _build_ranking_summary(self, options: List[JsonDict]) -> JsonDict:
        """Compute options_generated, feasible_count, cleared_count, blocked_count."""
        feasible_count = sum(1 for o in options if o.get("feasible", True))
        cleared_count = sum(
            1 for o in options
            if o.get("feasible", True) and o.get("regulatory_cleared", True)
        )
        blocked_count = sum(1 for o in options if not o.get("regulatory_cleared", True))
        infeasible_count = sum(1 for o in options if not o.get("feasible", True))
        best_score = min((o.get("risk_score", 1.0) for o in options), default=None)

        return {
            "options_generated": len(options),
            "feasible_count": feasible_count,
            "regulatory_cleared_count": cleared_count,
            "regulatory_blocked_count": blocked_count,
            "infeasible_count": infeasible_count,
            "best_risk_score": round(best_score, 4) if best_score is not None else None,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_option(
    *,
    activity_id: str,
    option_type: str,
    rationale: str,
    feasible: bool,
    infeasibility_reason: Optional[str],
    cp_impact_hours: float,
    criticality_label: str,
    confidence: float,
    resource_conflicts: List[JsonDict],
) -> JsonDict:
    """Build a canonical option dict (pre-scoring, pre-clearance)."""
    return {
        "option_id": f"OPT::{activity_id}::{option_type}::{uuid.uuid4().hex[:6]}",
        "option_type": option_type,
        "rationale": rationale,
        "feasible": feasible,
        "infeasibility_reason": infeasibility_reason,
        "regulatory_cleared": None,       # populated centrally after generation
        "regulatory_block_reason": None,  # populated centrally after generation
        "cp_impact_hours": round(cp_impact_hours, 2),
        "criticality_label": criticality_label,
        "confidence": round(confidence, 3),
        "resource_conflicts": resource_conflicts,
        "risk_score": None,               # populated centrally after scoring
    }


def _tier_confidence(tier: Optional[str]) -> float:
    """Map a duration distribution confidence_tier string to a float."""
    return _TIER_TO_CONFIDENCE.get(tier or "", 0.50)
