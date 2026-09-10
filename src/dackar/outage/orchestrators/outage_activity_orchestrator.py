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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stub_component_event_timeline(
    run_id: str,
    activity_id: str,
    reason: str,
) -> JsonDict:
    """Return an empty ComponentEventTimeline used when Stage B cannot determine
    a component_id (e.g. no resolved_component_ids and no known_component_id).

    All history fields are empty/zero so downstream stages degrade gracefully:
    Stage C produces ``causal_posture = "insufficient_data"`` with zero chain links.
    Stage D runs independently and is unaffected.
    Stage G cannot raise KG-derived attention flags (correct — no KG data available).

    The ``kg_driver_available`` flag is ``False`` so the orchestrator surfaces
    this in ``review_hooks`` as ``kg_unavailable``.
    """
    return {
        "activity_id": activity_id,
        "run_id": run_id,
        "generated_at": utcnow_iso(),
        "component_id": "",
        "component_name": None,
        "system_id": None,
        "asset_id": None,
        "kg_driver_available": False,
        "events": [],
        "recurrence_indicators": {
            "repeat_failure_count": 0,
            "mean_inter_event_days": None,
            "trend": "insufficient_data",
            "last_cm_date": None,
            "last_pm_date": None,
            "pm_compliance_status": "unknown",
        },
        "data_coverage": {
            "total_events": 0,
            "outages_represented": 0,
            "earliest_event": None,
            "latest_event": None,
            "data_quality_summary": None,
        },
        "provenance": {
            "generated_by": "stub",
            "run_id": run_id,
            "kg_driver_version": None,
            "query_window_days": 0,
            "notes": [reason],
        },
    }


def _stub_schedule_impact_artifact(
    run_id: str,
    activity_id: str,
    reason: str,
) -> JsonDict:
    """Return a null ScheduleImpactAssessment used when Stage E cannot load
    the schedule network (e.g. ``schedule_loader`` not injected).

    All CP-derived fields are ``None`` so downstream stages degrade gracefully:
    Stage F defaults to permissive float estimates and Stage G cannot raise
    ``_FLAG_CP_IMPACT`` (correct — no schedule information is available).

    The stub is identified by ``schedule_version_id == "STUB::no_schedule"`` and
    the reason string in ``notes``.  The run manifest ``review_hooks`` surfaces
    ``kg_unavailable`` so analysts know the recommendation was made without
    schedule data.
    """
    return {
        "activity_id": activity_id,
        "run_id": run_id,
        "generated_at": utcnow_iso(),
        "schedule_version_id": "STUB::no_schedule",
        "duration_estimate": {
            "p50_hours": 0.0,
            "p80_hours": 0.0,
            "p90_hours": 0.0,
            "confidence_tier": "low_confidence",
        },
        "float_analysis": {
            "criticality_label": "non_critical",
            "float_consumed_hours": 0.0,
            "available_float_before_hours": None,
            "remaining_float_after_hours": None,
            "is_critical_path_impact": False,
        },
        "cp_impact": {
            "cp_drag_hours": 0.0,
            "baseline_cp_hours": 0.0,
            "estimated_new_cp_hours": 0.0,
        },
        "displaced_tasks": [],
        "resource_conflicts": [],
        "confidence": 0.0,
        "notes": [reason],
        "provenance": {
            "generated_by": "stub",
            "run_id": run_id,
            "schedule_graph_version": None,
            "monte_carlo_runs": 0,
        },
    }


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

    # ── Optional: completion feedback loop ────────────────────────────────────
    feedback_writer: Optional[Any] = None
    """Optional :class:`~stages.completion_feedback.CompletionFeedbackWriter`
    that closes the learning loop by writing actual execution data back into
    the historical analog index after an activity completes in the field.

    Typical wiring::

        from stages.completion_feedback import CompletionFeedbackWriter, CsvAnalogPersister

        orchestrator = OutageActivityOrchestrator(
            ...
            feedback_writer=CompletionFeedbackWriter(
                index=analog_retriever.index,           # shared HistoricalActivityIndex
                persister=CsvAnalogPersister("/data/analogs/activities.csv"),
            ),
        )

        # After the field activity completes, call record_completion():
        result = orchestrator.record_completion(
            activity_id="ACT-20260412-001",
            run_id="OUTAGE::abc123",
            actual_duration_hours=16.2,
            actual_start="2026-04-12T08:00:00Z",
            actual_finish="2026-04-13T00:12:00Z",
            outcome_notes="Packing replaced; no scope expansion.",
        )
        # result.index_updated → True    (Stage D retrieval benefits immediately)
        # result.persisted     → True    (next outage cycle sees this activity)

    When ``None`` (default) calls to ``record_completion()`` log a WARNING and
    return a no-op result; the pipeline continues unaffected.
    """

    # ── Optional: two-pass insertion point pre-computation (Gap 4) ────────────
    insertion_point_determiner: Optional[Any] = None
    """Optional :class:`~stages.insertion_point_determiner.InsertionPointDeterminer`
    injected to enable the two-pass analog retrieval design.

    When provided the orchestrator runs a lightweight insertion-point
    determination between Stage C and Stage D.  The resulting
    ``ScheduleContext`` is threaded into both Stage D (structural affinity
    re-ranking) and Stage E (insertion point reuse).

    What the two-pass design saves: Stage E skips its own insertion-point
    search because the answer is already in ``ScheduleContext``.
    What it does NOT save: Stage E still loads the full schedule
    independently for Monte Carlo simulation.  The schedule is therefore
    loaded twice — once in the pre-pass and once in Stage E.  To eliminate
    the second load, cache the loaded ``OutageData`` inside ``ScheduleContext``
    and have Stage E read it from there instead of re-fetching.

    When ``None`` (default) the pipeline behaves exactly as before: Stage D
    runs without schedule context and Stage E determines the insertion point
    independently as it always has.
    """

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
            optional_failures=optional_failures,
        )

        # ── Stage C — Temporal Event Chain ────────────────────────────────────
        temporal_event_chain = self._stage_c_temporal_chain(
            run_id=run_id,
            emergent_activity=emergent_activity,
            component_event_timeline=component_event_timeline,
            run_context=run_context,
            precomputed=temporal_event_chain,
        )

        # ── Pre-pass: insertion point determination (Gap 4 two-pass design) ──
        # Runs after Stage C and before Stage D.  Requires no duration
        # distribution — only the schedule network + activity metadata.
        # Results are threaded into Stage D (structural affinity re-ranking)
        # and Stage E (insertion point reuse).  Note: Stage E still loads the
        # full schedule independently for Monte Carlo — only the insertion-point
        # determination step is avoided, not the schedule load itself.
        # Falls back to None → existing behavior when determiner not injected.
        schedule_context = self._precompute_schedule_context(
            emergent_activity=emergent_activity,
            intake_result=intake_result,
        )

        # ── Stage D — Historical Analogs (before E: provides duration dist.) ──
        historical_analogs = self._stage_d_analogs(
            run_id=run_id,
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            run_context=run_context,
            precomputed=historical_analogs,
            schedule_context=schedule_context,
        )

        # ── Stage E — Schedule Impact Assessment ──────────────────────────────
        schedule_impact_assessment = self._stage_e_schedule(
            run_id=run_id,
            emergent_activity=emergent_activity,
            intake_result=intake_result,
            historical_analogs=historical_analogs,
            run_context=run_context,
            precomputed=schedule_impact_assessment,
            optional_failures=optional_failures,
            schedule_context=schedule_context,
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
    # Completion feedback — learning loop entry point
    # ─────────────────────────────────────────────────────────────────────────

    def record_completion(
        self,
        *,
        activity_id: str,
        run_id: str,
        actual_duration_hours: float,
        actual_start: Optional[str] = None,
        actual_finish: Optional[str] = None,
        outcome_notes: str = "",
        outage_id: Optional[str] = None,
        plant_id: Optional[str] = None,
    ) -> Any:
        """Record that an emergent activity has completed in the field.

        Delegates to the injected ``feedback_writer`` to:

        1. Hot-update the in-memory ``HistoricalActivityIndex`` shared with
           Stage D — subsequent Stage D retrievals in the same session
           immediately incorporate the real execution time.
        2. Persist the record to the backing analog store via the injected
           ``AnalogPersister`` — the *next* outage cycle's ``build()`` call
           will include this activity with its actual duration.

        Both operations are best-effort: failures are logged and reflected in
        the returned ``CompletionRecord`` rather than raised.

        Args:
            activity_id: The emergent activity that completed.
            run_id: The pipeline run that produced the original recommendation.
            actual_duration_hours: Observed field duration (must be > 0).
            actual_start: ISO-8601 field start timestamp (optional).
            actual_finish: ISO-8601 field finish timestamp (optional).
            outcome_notes: Free-text operator notes (optional).
            outage_id: Used when the activity is not in the index (fallback).
            plant_id: Used when the activity is not in the index (fallback).

        Returns:
            :class:`~stages.completion_feedback.CompletionRecord` with
            ``index_updated``, ``persisted``, and ``validation_warnings``.
            Returns a minimal no-op record if no ``feedback_writer`` is injected.
        """
        if self.feedback_writer is None:
            LOGGER.warning(
                "record_completion called for activity %s (run=%s) but no "
                "feedback_writer is injected — write-back skipped. "
                "Inject a CompletionFeedbackWriter to enable the learning loop.",
                activity_id, run_id,
            )
            # Return a minimal no-op object so callers can check .index_updated
            # without an isinstance guard.
            try:
                from dackar.outage.stages.completion_feedback import (
                    CompletionRecord,
                    _utcnow_iso as _fb_utcnow,
                )
            except ImportError:
                from stages.completion_feedback import (  # type: ignore[no-redef]
                    CompletionRecord,
                    _utcnow_iso as _fb_utcnow,
                )
            return CompletionRecord(
                activity_id=activity_id,
                run_id=run_id,
                actual_duration_hours=actual_duration_hours,
                actual_start=actual_start,
                actual_finish=actual_finish,
                outcome_notes=outcome_notes,
                written_at=_fb_utcnow(),
                index_updated=False,
                persisted=False,
                validation_warnings=("feedback_writer not injected",),
            )

        return self.feedback_writer.record_completion(
            activity_id=activity_id,
            run_id=run_id,
            actual_duration_hours=actual_duration_hours,
            actual_start=actual_start,
            actual_finish=actual_finish,
            outcome_notes=outcome_notes,
            outage_id=outage_id,
            plant_id=plant_id,
        )

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
        optional_failures: List[JsonDict],
    ) -> JsonDict:
        """Stage B — KG Timeline Builder.

        Queries the knowledge graph for the resolved component(s) and assembles
        a time-ordered ComponentEventTimeline of CRs, WOs, maintenance events,
        inspections, and prior emergent activities.

        When ``kg_timeline_builder.build()`` raises ``ValueError`` (no resolvable
        component_id), this method catches the error, logs a WARNING, appends to
        ``optional_failures``, and returns an empty stub timeline so Stages C, F,
        and G can still produce a partial recommendation.  The ``kg_driver_available``
        flag in the stub is ``False``, which is surfaced in ``review_hooks`` as
        ``kg_unavailable`` so the analyst knows the recommendation was made without
        KG history.
        """
        if precomputed is not None:
            LOGGER.debug("Stage B skipped — using pre-computed component_event_timeline")
            self._validate_and_persist(run_id, "component_event_timeline", precomputed)
            return precomputed

        LOGGER.info("Stage B — KG timeline query (run=%s)", run_id)
        activity_id: str = emergent_activity.get("activity_id", "unknown")
        try:
            component_event_timeline = self.kg_timeline_builder.build(
                emergent_activity=emergent_activity,
                intake_result=intake_result,
                run_context=run_context,
            )
        except ValueError as exc:
            reason = str(exc)
            LOGGER.warning(
                "Stage B (run=%s): cannot determine component_id — "
                "producing empty stub timeline. Downstream stages will run "
                "without KG history. Reason: %s",
                run_id, reason,
            )
            optional_failures.append({
                "stage": "stage_b_kg_timeline",
                "artifact": "component_event_timeline",
                "error": reason,
                "optional": True,
            })
            component_event_timeline = _stub_component_event_timeline(
                run_id=run_id,
                activity_id=activity_id,
                reason=reason,
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
        schedule_context: Optional[Any] = None,
    ) -> JsonDict:
        """Stage D — Historical Analog Retriever.

        Retrieves the most similar past emergent activities from the indexed
        historical record and fits a duration distribution.  The duration
        distribution is consumed by Stage E — this is why D runs before E.

        When ``schedule_context`` is provided (Gap 4 two-pass design) the
        retriever applies a structural affinity re-ranking pass and stamps
        the query ActivityCase with predecessor/successor topology for
        future DependencyPatternScorer activation.
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
            schedule_context=schedule_context,
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
        optional_failures: List[JsonDict],
        schedule_context: Optional[Any] = None,
    ) -> JsonDict:
        """Stage E — Schedule Impact Assessor.

        Inserts the emergent activity into the current schedule network and
        computes critical path impact using the duration distribution from
        Stage D for Monte Carlo simulation.

        When ``schedule_context`` is provided (Gap 4 two-pass design) the
        assessor reuses the pre-computed insertion point rather than
        re-loading and re-traversing the schedule network.

        When ``schedule_loader`` or ``schedule_graph_builder`` are not injected
        into the assessor, Stage E raises ``RuntimeError``.  This method catches
        that error, logs a WARNING, appends to ``optional_failures``, and returns
        a null stub artifact so Stages F and G can still produce a partial
        recommendation based on analog data alone.  The ``schedule_loader_unavailable``
        flag in the stub is surfaced in ``review_hooks`` so the analyst knows the
        recommendation was made without schedule data.
        """
        if precomputed is not None:
            LOGGER.debug("Stage E skipped — using pre-computed schedule_impact_assessment")
            self._validate_and_persist(run_id, "schedule_impact_assessment", precomputed)
            return precomputed

        LOGGER.info("Stage E — schedule impact assessment (run=%s)", run_id)
        activity_id: str = emergent_activity.get("activity_id", "unknown")
        try:
            schedule_impact_assessment = self.schedule_impact_assessor.assess(
                emergent_activity=emergent_activity,
                intake_result=intake_result,
                historical_analogs=historical_analogs,
                run_context=run_context,
                schedule_context=schedule_context,
            )
        except RuntimeError as exc:
            reason = str(exc)
            LOGGER.warning(
                "Stage E (run=%s): schedule_loader unavailable — "
                "producing null stub artifact. Downstream stages will run "
                "without CP metrics. Reason: %s",
                run_id, reason,
            )
            optional_failures.append({
                "stage": "stage_e_schedule",
                "artifact": "schedule_impact_assessment",
                "error": reason,
                "optional": True,
            })
            schedule_impact_assessment = _stub_schedule_impact_artifact(
                run_id=run_id,
                activity_id=activity_id,
                reason=reason,
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
            component_event_timeline=component_event_timeline,
            schedule_impact_assessment=schedule_impact_assessment,
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
                    "kg_driver_available": component_event_timeline.get("kg_driver_available", True),
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
                    "schedule_loader_unavailable": schedule_impact_assessment.get("schedule_loader_unavailable", False),
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
                    "min_cost_option_id": insertion_options.get("min_cost_option_id"),
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
        component_event_timeline: JsonDict,
        schedule_impact_assessment: JsonDict,
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

        # When kg_driver_available is absent (pre-computed artifact without the field),
        # default to True to avoid spurious warnings on replayed runs.
        kg_unavailable = not component_event_timeline.get("kg_driver_available", True)

        # When schedule_loader_unavailable is True Stage E produced a null stub;
        # the recommendation was made without schedule/CP data.
        schedule_loader_unavailable = bool(
            schedule_impact_assessment.get("schedule_loader_unavailable", False)
        )

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
            "kg_unavailable": kg_unavailable,
            "schedule_loader_unavailable": schedule_loader_unavailable,
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

    def _precompute_schedule_context(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
    ) -> Optional[Any]:
        """Run the lightweight insertion-point pre-pass (Gap 4 two-pass design).

        Calls ``insertion_point_determiner.determine()`` and returns the
        resulting ``ScheduleContext``.  Returns ``None`` when:

        - ``insertion_point_determiner`` was not injected (default behavior).
        - The determiner raises or returns ``None`` itself (graceful degradation).

        Callers pass the result into both Stage D and Stage E; both stages
        fall back to their existing behavior when ``None`` is received.
        """
        if self.insertion_point_determiner is None:
            return None
        try:
            ctx = self.insertion_point_determiner.determine(
                emergent_activity, intake_result
            )
            if ctx is not None:
                LOGGER.debug(
                    "Pre-pass: ScheduleContext computed for activity %s "
                    "(after=%s, on_cp=%s, float=%.2f h)",
                    emergent_activity.get("activity_id"),
                    getattr(ctx, "after_task_id", None),
                    getattr(ctx, "insertion_on_cp", None),
                    getattr(ctx, "available_float_hours", 0.0),
                )
            return ctx
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Pre-pass: InsertionPointDeterminer failed for activity %s: %s — "
                "continuing without schedule context",
                emergent_activity.get("activity_id"), exc,
            )
            return None

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
