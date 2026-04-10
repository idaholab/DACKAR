"""
OutageActivityOrchestrator — stage-based pipeline for unexpected outage activities.

Executes stages A → B → C → D → E → F → G in sequence, validating and
optionally persisting each artifact at stage boundaries.  Any stage can be
skipped by passing a pre-computed artifact as a keyword argument to ``run()``;
this supports replay, partial re-runs, and unit testing of individual stages.

Stage execution order:
    A  _stage_a_intake            emergent_activity → intake_result
    B  _stage_b_kg_timeline       intake_result → component_event_timeline
    C  _stage_c_temporal_chain    component_event_timeline → temporal_event_chain
    D  _stage_d_analogs           intake_result → historical_analogs
    E  _stage_e_schedule          historical_analogs → schedule_impact_assessment
    F  _stage_f_options           all prior → insertion_options
    G  _stage_g_recommendation    all prior → outage_activity_recommendation
                                  + run_manifest

Design notes:
  - Protocol implementations are injected at construction time; no concrete
    backend is imported here.
  - _validate_and_persist() mirrors the RCA orchestrator pattern: required
    artifacts raise on validation failure (when stop_on_validation_error=True);
    optional artifacts log a warning and continue.
  - run() returns a dict of all artifacts keyed by artifact name, so callers
    can inspect any intermediate result without reaching into the file store.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .protocols import (
    ActivityIntakeProcessor,
    ArtifactStore,
    FileArtifactStore,
    HistoricalAnalogRetriever,
    InsertionOptionGenerator,
    JsonDict,
    KGTimelineBuilder,
    NoOpSchemaValidator,
    OutageOrchestratorConfig,
    RecommendationSynthesizer,
    ScheduleImpactAssessor,
    SchemaValidator,
    TemporalChainScorer,
    new_run_id,
    utcnow_iso,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class OutageActivityOrchestrator:
    """Pipeline orchestrator for unexpected outage activity analysis.

    All stage implementations are injected via their Protocols, keeping this
    class free of any backend dependency.

    Minimal wiring example (all no-ops)::

        from outage.orchestrators.protocols import (
            NoOpSchemaValidator, FileArtifactStore, OutageOrchestratorConfig,
        )

        class _StubStage:
            def process(self, *a, **kw): return {}
            def build(self, *a, **kw): return {}
            def score(self, *a, **kw): return {}
            def retrieve(self, *a, **kw): return {}
            def assess(self, *a, **kw): return {}
            def generate(self, *a, **kw): return {}
            def synthesize(self, *a, **kw): return {}

        stub = _StubStage()
        orch = OutageActivityOrchestrator(
            validator=NoOpSchemaValidator(),
            artifact_store=FileArtifactStore("/tmp/outage_runs"),
            intake_processor=stub,
            kg_timeline_builder=stub,
            temporal_chain_scorer=stub,
            analog_retriever=stub,
            schedule_impact_assessor=stub,
            option_generator=stub,
            recommendation_synthesizer=stub,
        )
        result = orch.run(emergent_activity={...})
    """

    # ── Infrastructure ────────────────────────────────────────────────────────
    validator: SchemaValidator
    artifact_store: ArtifactStore

    # ── Stage implementations (injected) ─────────────────────────────────────
    intake_processor: ActivityIntakeProcessor
    kg_timeline_builder: KGTimelineBuilder
    temporal_chain_scorer: TemporalChainScorer
    analog_retriever: HistoricalAnalogRetriever
    schedule_impact_assessor: ScheduleImpactAssessor
    option_generator: InsertionOptionGenerator
    recommendation_synthesizer: RecommendationSynthesizer

    # ── Configuration ─────────────────────────────────────────────────────────
    config: OutageOrchestratorConfig = field(
        default_factory=OutageOrchestratorConfig
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        emergent_activity: JsonDict,
        *,
        # Pre-computed artifacts — pass any to skip the corresponding stage.
        intake_result: Optional[JsonDict] = None,
        component_event_timeline: Optional[JsonDict] = None,
        temporal_event_chain: Optional[JsonDict] = None,
        historical_analogs: Optional[JsonDict] = None,
        schedule_impact_assessment: Optional[JsonDict] = None,
        insertion_options: Optional[JsonDict] = None,
    ) -> JsonDict:
        """Execute the full A → G pipeline for one emergent activity.

        Args:
            emergent_activity: Validated EmergentActivity artifact (required).
            intake_result: Pre-computed Stage A output — skips Stage A if provided.
            component_event_timeline: Pre-computed Stage B output.
            temporal_event_chain: Pre-computed Stage C output.
            historical_analogs: Pre-computed Stage D output.
            schedule_impact_assessment: Pre-computed Stage E output.
            insertion_options: Pre-computed Stage F output.

        Returns:
            Dict containing all artifact keys:
                run_context, intake_result, component_event_timeline,
                temporal_event_chain, historical_analogs,
                schedule_impact_assessment, insertion_options,
                outage_activity_recommendation, run_manifest.
        """
        run_id = new_run_id()
        optional_failures: List[JsonDict] = []

        # ── Run context (always built; not a skippable stage) ─────────────────
        run_context = self._build_run_context(
            run_id=run_id,
            emergent_activity=emergent_activity,
        )

        # ── Stage A — Activity Intake ─────────────────────────────────────────
        intake_result = self._stage_a_intake(
            run_id=run_id,
            emergent_activity=emergent_activity,
            run_context=run_context,
            precomputed=intake_result,
        )

        # ── Stage B — KG Timeline ─────────────────────────────────────────────
        component_event_timeline = self._stage_b_kg_timeline(
            run_id=run_id,
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            run_context=run_context,
            precomputed=component_event_timeline,
        )

        # ── Stage C — Temporal Event Chain ────────────────────────────────────
        temporal_event_chain = self._stage_c_temporal_chain(
            run_id=run_id,
            emergent_activity=emergent_activity,
            component_event_timeline=component_event_timeline,
            run_context=run_context,
            precomputed=temporal_event_chain,
        )

        # ── Stage D — Historical Analogs (before E: provides duration dist.) ──
        historical_analogs = self._stage_d_analogs(
            run_id=run_id,
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            run_context=run_context,
            precomputed=historical_analogs,
        )

        # ── Stage E — Schedule Impact Assessment ──────────────────────────────
        schedule_impact_assessment = self._stage_e_schedule(
            run_id=run_id,
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            historical_analogs=historical_analogs,
            run_context=run_context,
            precomputed=schedule_impact_assessment,
        )

        # ── Stage F — Insertion Option Generation ────────────────────────────
        insertion_options = self._stage_f_options(
            run_id=run_id,
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            temporal_event_chain=temporal_event_chain,
            schedule_impact_assessment=schedule_impact_assessment,
            historical_analogs=historical_analogs,
            run_context=run_context,
            precomputed=insertion_options,
        )

        # ── Stage G — Recommendation + Manifest ──────────────────────────────
        outage_activity_recommendation = self._stage_g_recommendation(
            run_id=run_id,
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            component_event_timeline=component_event_timeline,
            temporal_event_chain=temporal_event_chain,
            historical_analogs=historical_analogs,
            schedule_impact_assessment=schedule_impact_assessment,
            insertion_options=insertion_options,
            run_context=run_context,
            optional_failures=optional_failures,
        )

        run_manifest = self._finalize_manifest(
            run_context=run_context,
            intake_result=intake_result,
            component_event_timeline=component_event_timeline,
            temporal_event_chain=temporal_event_chain,
            historical_analogs=historical_analogs,
            schedule_impact_assessment=schedule_impact_assessment,
            insertion_options=insertion_options,
            outage_activity_recommendation=outage_activity_recommendation,
            optional_failures=optional_failures,
        )
        self.artifact_store.save(run_id, "run_manifest", run_manifest)

        return {
            "run_context": run_context,
            "intake_result": intake_result,
            "component_event_timeline": component_event_timeline,
            "temporal_event_chain": temporal_event_chain,
            "historical_analogs": historical_analogs,
            "schedule_impact_assessment": schedule_impact_assessment,
            "insertion_options": insertion_options,
            "outage_activity_recommendation": outage_activity_recommendation,
            "run_manifest": run_manifest,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Run context (not a skippable stage)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_run_context(
        self,
        run_id: str,
        emergent_activity: JsonDict,
    ) -> JsonDict:
        """Build and persist the run metadata block.

        This is always executed; it cannot be skipped via a pre-computed artifact.
        Mirrors _stage_a_build_run_context in the RCA orchestrator.
        """
        run_context: JsonDict = {
            "run_id": run_id,
            "run_label": self.config.run_label,
            "started_at": utcnow_iso(),
            "config": {
                "persist_intermediate_artifacts": self.config.persist_intermediate_artifacts,
                "stop_on_validation_error": self.config.stop_on_validation_error,
                "unknown_abbreviation_rate_warning": self.config.unknown_abbreviation_rate_warning,
                "near_critical_float_threshold_hours": self.config.near_critical_float_threshold_hours,
                "monte_carlo_runs": self.config.monte_carlo_runs,
                **self.config.extra,
            },
            "input_refs": {
                "activity_id": emergent_activity.get("activity_id"),
                "outage_id": emergent_activity.get("outage_id"),
                "plant_id": emergent_activity.get("plant_id"),
                "unit_id": emergent_activity.get("unit_id"),
                "source_system": emergent_activity.get("source_system"),
                "detection_timestamp": emergent_activity.get("detection_timestamp"),
            },
        }
        self.artifact_store.save(run_id, "run_context", run_context)
        LOGGER.info("Run %s started for activity %s", run_id, emergent_activity.get("activity_id"))
        return run_context

    # ─────────────────────────────────────────────────────────────────────────
    # Stage methods
    # ─────────────────────────────────────────────────────────────────────────

    def _stage_a_intake(
        self,
        run_id: str,
        emergent_activity: JsonDict,
        run_context: JsonDict,
        precomputed: Optional[JsonDict],
    ) -> JsonDict:
        """Stage A — Activity Intake.

        NER, abbreviation expansion, emergence type classification, regulatory
        flag detection, component/WO/CR reference resolution, DQ scoring.

        Exits with analyst_review=True if unknown_abbreviation_rate exceeds
        the configured warning threshold.
        """
        if precomputed is not None:
            LOGGER.debug("Stage A skipped — using pre-computed intake_result")
            self._validate_and_persist(run_id, "intake_result", precomputed)
            return precomputed

        LOGGER.info("Stage A — intake processing (run=%s)", run_id)
        intake_result = self.intake_processor.process(
            emergent_activity=emergent_activity,
            run_context=run_context,
        )

        # Exit criterion: high unknown abbreviation rate forces analyst review.
        abbr_rate = intake_result.get("unknown_abbreviation_rate", 0.0)
        if abbr_rate > self.config.unknown_abbreviation_rate_warning:
            LOGGER.warning(
                "Stage A (run=%s): unknown_abbreviation_rate=%.2f exceeds threshold %.2f "
                "— downstream semantic extraction may be unreliable.",
                run_id, abbr_rate, self.config.unknown_abbreviation_rate_warning,
            )

        self._validate_and_persist(run_id, "intake_result", intake_result)
        return intake_result

    def _stage_b_kg_timeline(
        self,
        run_id: str,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        run_context: JsonDict,
        precomputed: Optional[JsonDict],
    ) -> JsonDict:
        """Stage B — KG Timeline Builder.

        Queries the knowledge graph for the resolved component(s) and assembles
        a time-ordered ComponentEventTimeline of CRs, WOs, maintenance events,
        inspections, and prior emergent activities.
        """
        if precomputed is not None:
            LOGGER.debug("Stage B skipped — using pre-computed component_event_timeline")
            self._validate_and_persist(run_id, "component_event_timeline", precomputed)
            return precomputed

        LOGGER.info("Stage B — KG timeline query (run=%s)", run_id)
        component_event_timeline = self.kg_timeline_builder.build(
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            run_context=run_context,
        )

        self._validate_and_persist(run_id, "component_event_timeline", component_event_timeline)
        return component_event_timeline

    def _stage_c_temporal_chain(
        self,
        run_id: str,
        emergent_activity: JsonDict,
        component_event_timeline: JsonDict,
        run_context: JsonDict,
        precomputed: Optional[JsonDict],
    ) -> JsonDict:
        """Stage C — Temporal Event Chain Scorer.

        Applies Allen interval algebra to the ComponentEventTimeline.
        Classifies each prior event as PRECEDES, OVERLAPS, CONTAINS, DURING,
        FOLLOWS, or SIMULTANEOUS relative to the emergent activity interval.
        """
        if precomputed is not None:
            LOGGER.debug("Stage C skipped — using pre-computed temporal_event_chain")
            self._validate_and_persist(run_id, "temporal_event_chain", precomputed)
            return precomputed

        LOGGER.info("Stage C — temporal chain scoring (run=%s)", run_id)
        temporal_event_chain = self.temporal_chain_scorer.score(
            emergent_activity=emergent_activity,
            component_event_timeline=component_event_timeline,
            run_context=run_context,
        )

        self._validate_and_persist(run_id, "temporal_event_chain", temporal_event_chain)
        return temporal_event_chain

    def _stage_d_analogs(
        self,
        run_id: str,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        run_context: JsonDict,
        precomputed: Optional[JsonDict],
    ) -> JsonDict:
        """Stage D — Historical Analog Retriever.

        Retrieves the most similar past emergent activities from the indexed
        historical record and fits a duration distribution.  The duration
        distribution is consumed by Stage E — this is why D runs before E.
        """
        if precomputed is not None:
            LOGGER.debug("Stage D skipped — using pre-computed historical_analogs")
            self._validate_and_persist(run_id, "historical_analogs", precomputed)
            return precomputed

        LOGGER.info("Stage D — historical analog retrieval (run=%s)", run_id)
        historical_analogs = self.analog_retriever.retrieve(
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            run_context=run_context,
        )

        self._validate_and_persist(run_id, "historical_analogs", historical_analogs)
        return historical_analogs

    def _stage_e_schedule(
        self,
        run_id: str,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        historical_analogs: JsonDict,
        run_context: JsonDict,
        precomputed: Optional[JsonDict],
    ) -> JsonDict:
        """Stage E — Schedule Impact Assessor.

        Inserts the emergent activity into the current schedule network and
        computes critical path impact using the duration distribution from
        Stage D for Monte Carlo simulation.
        """
        if precomputed is not None:
            LOGGER.debug("Stage E skipped — using pre-computed schedule_impact_assessment")
            self._validate_and_persist(run_id, "schedule_impact_assessment", precomputed)
            return precomputed

        LOGGER.info("Stage E — schedule impact assessment (run=%s)", run_id)
        schedule_impact_assessment = self.schedule_impact_assessor.assess(
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            historical_analogs=historical_analogs,
            run_context=run_context,
        )

        self._validate_and_persist(run_id, "schedule_impact_assessment", schedule_impact_assessment)
        return schedule_impact_assessment

    def _stage_f_options(
        self,
        run_id: str,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        temporal_event_chain: JsonDict,
        schedule_impact_assessment: JsonDict,
        historical_analogs: JsonDict,
        run_context: JsonDict,
        precomputed: Optional[JsonDict],
    ) -> JsonDict:
        """Stage F — Insertion Option Generator.

        Generates a ranked set of options for handling the emergent activity.
        Filters options that violate regulatory constraints from Stage A.
        Ranks feasible options by composite risk score.
        """
        if precomputed is not None:
            LOGGER.debug("Stage F skipped — using pre-computed insertion_options")
            self._validate_and_persist(run_id, "insertion_options", precomputed)
            return precomputed

        LOGGER.info("Stage F — insertion option generation (run=%s)", run_id)
        insertion_options = self.option_generator.generate(
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            temporal_event_chain=temporal_event_chain,
            schedule_impact_assessment=schedule_impact_assessment,
            historical_analogs=historical_analogs,
            run_context=run_context,
        )

        self._validate_and_persist(run_id, "insertion_options", insertion_options)
        return insertion_options

    def _stage_g_recommendation(
        self,
        run_id: str,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        component_event_timeline: JsonDict,
        temporal_event_chain: JsonDict,
        historical_analogs: JsonDict,
        schedule_impact_assessment: JsonDict,
        insertion_options: JsonDict,
        run_context: JsonDict,
        optional_failures: List[JsonDict],
    ) -> JsonDict:
        """Stage G — Recommendation Synthesizer.

        Synthesizes all upstream artifacts into a single
        OutageActivityRecommendation.  Sets analyst_review=True when:
          - has_regulatory_constraint is True
          - confidence_tier is low_confidence
          - decision_status is INCONCLUSIVE
          - any upstream stage produced an optional failure
          - unknown_abbreviation_rate exceeded the warning threshold in Stage A
        """
        LOGGER.info("Stage G — recommendation synthesis (run=%s)", run_id)
        outage_activity_recommendation = self.recommendation_synthesizer.synthesize(
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            component_event_timeline=component_event_timeline,
            temporal_event_chain=temporal_event_chain,
            historical_analogs=historical_analogs,
            schedule_impact_assessment=schedule_impact_assessment,
            insertion_options=insertion_options,
            run_context=run_context,
        )

        self._validate_and_persist(
            run_id, "outage_activity_recommendation", outage_activity_recommendation
        )
        return outage_activity_recommendation

    # ─────────────────────────────────────────────────────────────────────────
    # Run manifest
    # ─────────────────────────────────────────────────────────────────────────

    def _finalize_manifest(
        self,
        run_context: JsonDict,
        intake_result: JsonDict,
        component_event_timeline: JsonDict,
        temporal_event_chain: JsonDict,
        historical_analogs: JsonDict,
        schedule_impact_assessment: JsonDict,
        insertion_options: JsonDict,
        outage_activity_recommendation: JsonDict,
        optional_failures: List[JsonDict],
    ) -> JsonDict:
        """Build the run manifest: a single summary of what the pipeline produced.

        Mirrors _stage_g_finalize_manifest in the RCA orchestrator.
        Used for monitoring, audit, and the learning loop.
        """
        review_hooks = self._compute_review_hooks(
            intake_result=intake_result,
            outage_activity_recommendation=outage_activity_recommendation,
            optional_failures=optional_failures,
        )

        rec = outage_activity_recommendation
        exec_summary = rec.get("executive_summary") or {}
        sched = rec.get("schedule_summary") or {}
        hist = rec.get("history_summary") or {}
        val = rec.get("validation_status") or {}
        analogs_summary = historical_analogs.get("retrieval_summary") or {}
        chain_summary = temporal_event_chain.get("summary") or {}
        options_summary = insertion_options.get("ranking_summary") or {}

        return {
            "run_id": run_context["run_id"],
            "completed_at": utcnow_iso(),
            "input_refs": run_context["input_refs"],
            "pipeline_config": run_context["config"],
            "artifacts": {
                "intake_result": {
                    "present": True,
                    "emergence_type": intake_result.get("emergence_type"),
                    "has_regulatory_constraint": intake_result.get("has_regulatory_constraint"),
                    "unknown_abbreviation_rate": intake_result.get("unknown_abbreviation_rate"),
                    "data_quality_score": intake_result.get("data_quality_score"),
                    "resolved_component_count": len(intake_result.get("resolved_component_ids") or []),
                },
                "component_event_timeline": {
                    "present": True,
                    "total_events": (component_event_timeline.get("data_coverage") or {}).get("total_events"),
                    "repeat_failure_count": (component_event_timeline.get("recurrence_indicators") or {}).get("repeat_failure_count"),
                    "pm_compliance_status": (component_event_timeline.get("recurrence_indicators") or {}).get("pm_compliance_status"),
                },
                "temporal_event_chain": {
                    "present": True,
                    "chain_length": chain_summary.get("chain_length"),
                    "causal_posture": chain_summary.get("causal_posture"),
                    "has_temporal_contradiction": chain_summary.get("has_temporal_contradiction"),
                },
                "historical_analogs": {
                    "present": True,
                    "analog_count": analogs_summary.get("analog_count"),
                    "outages_represented": analogs_summary.get("outages_represented"),
                    "fallback_used": analogs_summary.get("fallback_used"),
                    "duration_p50_hours": (historical_analogs.get("duration_distribution") or {}).get("p50_hours"),
                    "confidence_tier": (historical_analogs.get("duration_distribution") or {}).get("confidence_tier"),
                },
                "schedule_impact_assessment": {
                    "present": True,
                    "criticality_label": (schedule_impact_assessment.get("float_analysis") or {}).get("criticality_label"),
                    "cp_drag_hours": (schedule_impact_assessment.get("cp_impact") or {}).get("cp_drag_hours"),
                    "float_consumed_hours": (schedule_impact_assessment.get("float_analysis") or {}).get("float_consumed_hours"),
                    "displaced_task_count": len(schedule_impact_assessment.get("displaced_tasks") or []),
                },
                "insertion_options": {
                    "present": True,
                    "options_generated": options_summary.get("options_generated"),
                    "options_feasible": options_summary.get("options_feasible"),
                    "options_regulatory_blocked": options_summary.get("options_regulatory_blocked"),
                    "recommended_option_id": insertion_options.get("recommended_option_id"),
                },
                "outage_activity_recommendation": {
                    "present": True,
                    "decision_status": rec.get("decision_status"),
                    "confidence_tier": exec_summary.get("confidence_tier"),
                    "cp_impact_hours": sched.get("cp_impact_hours"),
                    "criticality_label": sched.get("criticality_label"),
                    "analog_count": hist.get("analog_count"),
                    "analyst_review_required": (rec.get("analyst_review") or {}).get("required"),
                    "fallback_used": val.get("fallback_used"),
                    "attention_flags": exec_summary.get("analyst_attention_flags") or [],
                },
            },
            "validation": {
                "optional_artifact_failures": optional_failures,
                "optional_artifacts_degraded": bool(optional_failures),
            },
            "review_hooks": review_hooks,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Review hooks
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_review_hooks(
        self,
        intake_result: JsonDict,
        outage_activity_recommendation: JsonDict,
        optional_failures: List[JsonDict],
    ) -> JsonDict:
        """Compute attention flags for the run manifest.

        These hooks drive monitoring dashboards and the low-confidence watchlist
        (WBS 11.3). Mirrors _compute_review_hooks in the RCA orchestrator.
        """
        rec = outage_activity_recommendation
        val = rec.get("validation_status") or {}
        exec_summary = rec.get("executive_summary") or {}
        analyst_review = rec.get("analyst_review") or {}

        has_regulatory_constraint = bool(intake_result.get("has_regulatory_constraint", False))
        abbr_rate = float(intake_result.get("unknown_abbreviation_rate") or 0.0)
        abbr_rate_high = abbr_rate > self.config.unknown_abbreviation_rate_warning

        decision_status = rec.get("decision_status")
        confidence_tier = exec_summary.get("confidence_tier")
        fallback_used = bool(val.get("fallback_used", False))
        analyst_review_required = bool(analyst_review.get("required", True))
        schema_valid = bool(val.get("schema_valid", False))
        min_evidence_met = bool(val.get("minimum_evidence_met", False))
        optional_artifacts_degraded = bool(optional_failures)

        # Determine the recommended next step for the analyst dashboard.
        if not schema_valid:
            next_step = "validation_remediation"
        elif decision_status == "INCONCLUSIVE":
            next_step = "analyst_review"
        elif has_regulatory_constraint:
            next_step = "licensing_review"
        elif analyst_review_required:
            next_step = "analyst_review"
        else:
            next_step = "ready_to_act"

        return {
            "requires_human_review": analyst_review_required,
            "next_step": next_step,
            "schema_valid": schema_valid,
            "minimum_evidence_met": min_evidence_met,
            "fallback_used": fallback_used,
            "has_regulatory_constraint": has_regulatory_constraint,
            "high_unknown_abbreviation_rate": abbr_rate_high,
            "unknown_abbreviation_rate": abbr_rate,
            "confidence_tier": confidence_tier,
            "decision_status": decision_status,
            "optional_artifacts_degraded": optional_artifacts_degraded,
            "optional_failure_count": len(optional_failures),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_and_persist(
        self,
        run_id: str,
        artifact_name: str,
        payload: JsonDict,
        *,
        optional: bool = False,
        optional_failures: Optional[List[JsonDict]] = None,
    ) -> None:
        """Validate an artifact and optionally persist it to the ArtifactStore.

        For required artifacts (optional=False): a validation failure raises
        immediately when stop_on_validation_error=True; otherwise it is logged
        as an error and execution continues.

        For optional artifacts: a validation failure is logged as a warning and
        appended to optional_failures (if provided).
        """
        try:
            validation = self.validator.validate_artifact(artifact_name, payload)
        except Exception as exc:
            if optional:
                LOGGER.warning(
                    "Optional artifact '%s' failed validation (run=%s): %s",
                    artifact_name, run_id, exc,
                )
                failure = {"artifact": artifact_name, "error": str(exc), "optional": True}
                if optional_failures is not None:
                    optional_failures.append(failure)
                if self.config.persist_intermediate_artifacts:
                    self.artifact_store.save(run_id, artifact_name, payload)
                return
            else:
                LOGGER.error(
                    "Artifact '%s' failed validation (run=%s): %s",
                    artifact_name, run_id, exc,
                )
                if self.config.stop_on_validation_error:
                    raise
                validation = {"ok": False, "issues": [str(exc)]}
        else:
            is_ok = validation.get("ok", True) if isinstance(validation, dict) else True
            if not is_ok:
                issues = validation.get("issues", []) if isinstance(validation, dict) else []
                LOGGER.error(
                    "Artifact '%s' validation issues (run=%s): %s",
                    artifact_name, run_id, issues,
                )
                if self.config.stop_on_validation_error and not optional:
                    raise ValueError(
                        f"Artifact '{artifact_name}' failed validation: {issues}"
                    )

        if self.config.persist_intermediate_artifacts:
            self.artifact_store.save(run_id, artifact_name, payload)
            if isinstance(validation, dict):
                self.artifact_store.save(
                    run_id, f"{artifact_name}__validation", validation
                )
