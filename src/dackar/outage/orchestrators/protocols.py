"""
Outage Activity Workflow — Protocol definitions and configuration dataclasses.

Each Protocol defines the contract for one pipeline stage. Concrete
implementations are injected into OutageActivityOrchestrator at construction
time, keeping the orchestration logic decoupled from any specific NLP backend,
KG driver, or scheduling library.

Stage execution order (and data dependencies):
    A  ActivityIntakeProcessor      emergent_activity → intake_result
    B  KGTimelineBuilder            intake_result → component_event_timeline
    C  TemporalChainScorer          component_event_timeline → temporal_event_chain
    D  HistoricalAnalogRetriever    intake_result → historical_analogs
    E  ScheduleImpactAssessor       historical_analogs → schedule_impact_assessment
    F  InsertionOptionGenerator     temporal_event_chain + schedule_impact_assessment
                                    + historical_analogs → insertion_options
    G  RecommendationSynthesizer    all upstream artifacts → outage_activity_recommendation

Note: Stage D runs before Stage E because the duration distribution produced by
analog retrieval is a required input to schedule impact assessment.

Shared infrastructure:
    SchemaValidator     validate_artifact / validate_run_bundle
    ArtifactStore       save / save_list

Concrete no-op implementations (NoOpSchemaValidator, FileArtifactStore) are
provided here for bootstrapping and testing.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "OUTAGE") -> str:
    """Generate a stable run identifier with an optional prefix."""
    return f"{prefix}::{uuid.uuid4().hex[:12]}"


# ── Stage A — Activity Intake ─────────────────────────────────────────────────

@runtime_checkable
class ActivityIntakeProcessor(Protocol):
    """Stage A.

    Consumes a raw EmergentActivity record and produces an ActivityIntakeResult:
    cleaned and expanded description, NER entities, emergence type classification,
    resolved component/WO/CR references, regulatory constraint flags, and data
    quality scores.

    Reuse targets from RCA:
        - HybridNERPipeline (entity extraction)
        - SpacyAnnotator (temporal refs, measurements)
        - EntityNormalizer (component ID resolution)
        - outage_uncertainty.preprocessing (cleaners, abbreviation expander)
    """

    def process(
        self,
        emergent_activity: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """
        Args:
            emergent_activity: Validated EmergentActivity artifact.
            run_context: Run metadata produced by the orchestrator (run_id,
                         timestamps, config snapshot).

        Returns:
            Validated ActivityIntakeResult artifact.
        """
        ...


# ── Stage B — KG Timeline Builder ────────────────────────────────────────────

@runtime_checkable
class KGTimelineBuilder(Protocol):
    """Stage B.

    Queries the knowledge graph for the component(s) resolved in Stage A and
    assembles a time-ordered ComponentEventTimeline: condition reports, work
    orders, preventive and corrective maintenance records, inspection results,
    and prior emergent activities. Also computes recurrence indicators and
    PM compliance status.

    Reuse targets from RCA:
        - Py2Neo (KG driver)
        - ProcessedRecordStore (indexed CR/WO text retrieval)
    """

    def build(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """
        Args:
            emergent_activity: Validated EmergentActivity artifact.
            intake_result: Validated ActivityIntakeResult from Stage A.
            run_context: Run metadata.

        Returns:
            Validated ComponentEventTimeline artifact.
        """
        ...


# ── Stage C — Temporal Chain Scorer ──────────────────────────────────────────

@runtime_checkable
class TemporalChainScorer(Protocol):
    """Stage C.

    Applies Allen interval algebra to the ComponentEventTimeline to classify
    the temporal relationship between each prior event and the emergent activity:
    PRECEDES, OVERLAPS, CONTAINS, DURING, FOLLOWS, SIMULTANEOUS.

    Produces a TemporalEventChain with per-link causal strength scores and an
    overall causal posture assessment.

    Reuse targets from RCA:
        - TSKRTemporalScorerV1 (Allen relation logic, confidence scoring)
          Adaptation: operates on CR/WO event intervals rather than telemetry
          anomaly windows. The interval algebra itself is unchanged.
    """

    def score(
        self,
        emergent_activity: JsonDict,
        component_event_timeline: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """
        Args:
            emergent_activity: Validated EmergentActivity artifact.
            component_event_timeline: Validated ComponentEventTimeline from Stage B.
            run_context: Run metadata.

        Returns:
            Validated TemporalEventChain artifact.
        """
        ...


# ── Stage D — Historical Analog Retriever ────────────────────────────────────

@runtime_checkable
class HistoricalAnalogRetriever(Protocol):
    """Stage D.

    Retrieves the most similar past emergent activities from the indexed
    historical record using lexical, semantic, component, and context
    similarity. Fits a duration distribution from the analog set. The
    resulting HistoricalAnalogs artifact is consumed by Stage E to size
    the duration estimate used in schedule impact assessment.

    Reuse targets from RCA:
        - ChromaEvidenceRetriever (semantic + keyword search with KG filters)
        - ProcessedRecordStore (activity text index)
        - outage_uncertainty.retrieval (lexical_similarity, semantic_similarity,
          dependency_similarity, context_similarity, similarity_engine)
        - outage_uncertainty.uncertainty (distribution_fitter, outlier_handler,
          confidence, fallback_policy)
    """

    def retrieve(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """
        Args:
            emergent_activity: Validated EmergentActivity artifact.
            intake_result: Validated ActivityIntakeResult from Stage A.
            run_context: Run metadata.

        Returns:
            Validated HistoricalAnalogs artifact including the fitted
            duration distribution.
        """
        ...


# ── Stage E — Schedule Impact Assessor ───────────────────────────────────────

@runtime_checkable
class ScheduleImpactAssessor(Protocol):
    """Stage E.

    Inserts the emergent activity into the current schedule network and
    computes critical path impact: float consumed, CP drag, criticality label,
    displaced tasks, and resource conflicts. Uses the duration distribution
    from Stage D for probabilistic impact estimation (Monte Carlo).

    Reuse targets from RCA: none directly.
    Reuse targets from outage_uncertainty:
        - schedule_risk.schedule_graph (ScheduleGraph)
        - schedule_risk.cp_analyzer (CriticalPathRiskAnalyzer)
        - schedule_risk.monte_carlo (MonteCarloSimulator)
        - schedule_risk.scenario_runner (ScenarioRunner)
        - P6_adapter.outage_model (OutageDataset — schedule version access)
    """

    def assess(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        historical_analogs: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """
        Args:
            emergent_activity: Validated EmergentActivity artifact.
            intake_result: Validated ActivityIntakeResult from Stage A.
            historical_analogs: Validated HistoricalAnalogs from Stage D.
                                Duration distribution is consumed here.
            run_context: Run metadata.

        Returns:
            Validated ScheduleImpactAssessment artifact.
        """
        ...


# ── Stage F — Insertion Option Generator ─────────────────────────────────────

@runtime_checkable
class InsertionOptionGenerator(Protocol):
    """Stage F.

    Generates a ranked set of options for handling the emergent activity:
    insert now, defer, pre-stage, add contingency buffer, parallel execution,
    scope reduction, or escalate. Filters options that violate regulatory
    constraints identified in Stage A. Ranks remaining feasible options by
    composite risk score (CP impact, confidence, resource availability).

    This is the only stage with no direct reuse target from RCA or
    outage_uncertainty — it is new logic specific to outage decision support.

    Reuse targets (partial):
        - RuleBasedCausalityEngineV31 scoring pattern (adapt for option ranking)
        - outage_uncertainty.services.schedule_service
    """

    def generate(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        temporal_event_chain: JsonDict,
        schedule_impact_assessment: JsonDict,
        historical_analogs: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """
        Args:
            emergent_activity: Validated EmergentActivity artifact.
            intake_result: Validated ActivityIntakeResult from Stage A.
            temporal_event_chain: Validated TemporalEventChain from Stage C.
            schedule_impact_assessment: Validated ScheduleImpactAssessment
                                        from Stage E.
            historical_analogs: Validated HistoricalAnalogs from Stage D.
            run_context: Run metadata.

        Returns:
            Validated InsertionOptions artifact with ranked options and a
            recommended_option_id.
        """
        ...


# ── Stage G — Recommendation Synthesizer ─────────────────────────────────────

@runtime_checkable
class RecommendationSynthesizer(Protocol):
    """Stage G.

    Synthesizes all upstream artifacts into a single OutageActivityRecommendation.
    Selects the top-ranked feasible option, assembles the evidence chain,
    surfaces regulatory flags, computes the executive summary, and sets the
    analyst_review flag when required (regulatory constraints present, low
    confidence, or INCONCLUSIVE status).

    This is the artifact the outage manager reads. It must be fully traceable:
    every claim cites a source, every flag is visible, and the analyst can
    reject with a reason that feeds back into the learning loop.

    Reuse targets from RCA:
        - RCASynthesizer pattern (evidence selection, confidence tier logic)
        - RCAArtifactValidator (schema + semantic validation)
        - ArtifactStore (persistence)
    """

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
        """
        Args:
            All validated upstream artifacts from Stages A–F.
            run_context: Run metadata.

        Returns:
            Validated OutageActivityRecommendation artifact.
        """
        ...


# ── Shared infrastructure Protocols ──────────────────────────────────────────

class SchemaValidator(Protocol):
    """Validates pipeline artifacts against their JSON schemas.

    Compatible with RCAArtifactValidator from dackar.RCA.validation — that
    class can be wrapped with outage-specific schema paths and reused here
    without modification.
    """

    def validate_artifact(self, artifact_name: str, payload: JsonDict) -> Any:
        """Validate a single artifact. Returns a ValidationReport or raises."""
        ...

    def validate_run_bundle(self, **kwargs: Any) -> Any:
        """Cross-artifact semantic consistency check across a full run."""
        ...


class ArtifactStore(Protocol):
    """Persists pipeline artifacts keyed by run_id and artifact name.

    Compatible with FileArtifactStore from dackar.RCA.orchestrators — that
    class implements this protocol and can be reused directly.
    """

    def save(self, run_id: str, artifact_name: str, payload: JsonDict) -> str:
        """Persist a single artifact. Returns the storage path or key."""
        ...

    def save_list(
        self, run_id: str, artifact_name: str, payload: List[JsonDict]
    ) -> str:
        """Persist a list of artifacts (e.g., JSONL). Returns path or key."""
        ...


# ── Analog persistence — completion feedback loop ────────────────────────────

@runtime_checkable
class AnalogPersister(Protocol):
    """Write completed ActivityCase records to the backing analog store.

    Called by ``CompletionFeedbackWriter`` after a successful
    ``record_completion()`` call.  Concrete implementations live in
    ``stages/completion_feedback.py`` (``CsvAnalogPersister``).

    The contract is intentionally narrow — a single ``append`` method —
    to keep the persistence surface minimal, testable, and easy to swap.
    Implementations must be idempotent: appending the same activity twice
    (e.g. from a duplicate callback) must not corrupt the store.
    """

    def append(self, activity: Any) -> None:
        """Persist one completed ActivityCase to the backing store."""
        ...


class NoOpAnalogPersister:
    """Test stub and safe default — accepts writes silently.

    ``appended`` accumulates all received activities and is inspectable in
    tests to verify what was written without touching the filesystem.
    """

    def __init__(self) -> None:
        self.appended: List[Any] = []

    def append(self, activity: Any) -> None:  # noqa: D102
        self.appended.append(activity)


# ── Configuration dataclasses ─────────────────────────────────────────────────

@dataclass
class OutageOrchestratorConfig:
    """Top-level orchestrator configuration.

    Controls which intermediate artifacts are persisted, whether validation
    errors halt the run, and key thresholds that apply across multiple stages.
    """

    persist_intermediate_artifacts: bool = True
    """Write each stage output to the ArtifactStore before passing downstream."""

    stop_on_validation_error: bool = True
    """Halt the run if any stage output fails schema validation. If False,
    the error is logged and the run continues with the invalid artifact."""

    run_label: Optional[str] = None
    """Optional human-readable label appended to the run_id for traceability."""

    unknown_abbreviation_rate_warning: float = 0.25
    """Stage A exit criterion. If unknown_abbreviation_rate exceeds this value
    the intake result is flagged and analyst_review is forced to True."""

    near_critical_float_threshold_hours: float = 8.0
    """Stage E threshold. Activities with remaining float below this value after
    insertion are labelled near_critical."""

    monte_carlo_runs: int = 1000
    """Number of Monte Carlo samples for Stage E schedule impact simulation."""

    extra: JsonDict = field(default_factory=dict)
    """Reserved for future configuration without breaking the dataclass interface."""


@dataclass
class KGTimelineBuilderConfig:
    """Configuration for the Stage B KG timeline query."""

    timeline_window_days: int = 1825
    """How far back (in days) to query the KG for component events. Default 5 years."""

    max_events: int = 100
    """Maximum number of events to include in the timeline."""

    include_work_orders: bool = True
    include_condition_reports: bool = True
    include_preventive_maintenance: bool = True
    include_corrective_maintenance: bool = True
    include_prior_emergent_activities: bool = True
    include_inspections: bool = True


@dataclass
class TemporalChainScorerConfig:
    """Configuration for the Stage C Allen relation scorer."""

    epsilon_hours: float = 0.5
    """Tolerance for Allen relation boundary comparisons. Events within
    epsilon_hours of a boundary are treated as simultaneous at that boundary."""

    include_follows_relations: bool = True
    """If True, FOLLOWS links are included in the chain with causal_strength
    'temporal_contradiction'. If False, they are silently dropped."""

    min_relation_score_threshold: float = 0.0
    """Links with relation_score below this value are excluded. Default 0.0
    (all links included) — tighten for high-volume component histories."""


@dataclass
class HistoricalAnalogRetrieverConfig:
    """Configuration for the Stage D analog retrieval and duration fitting."""

    top_k: int = 20
    """Maximum number of analogs to retrieve before similarity filtering."""

    similarity_threshold: float = 0.60
    """Minimum overall similarity_score for an analog to be retained."""

    min_analogs_for_data_supported: int = 5
    """Minimum analog count with actual durations required to assign the
    data_supported confidence tier."""

    min_analogs_for_sme_informed: int = 1
    """Minimum analog count for the sme_informed tier. Below this, the
    distribution falls back to the population-level fallback."""

    outlier_iqr_factor: float = 1.5
    """IQR multiplier for outlier removal before duration fitting."""


@dataclass
class ScheduleImpactAssessorConfig:
    """Configuration for the Stage E schedule impact computation."""

    use_p80_for_float_analysis: bool = False
    """If True, use the p80 duration estimate for float analysis (conservative).
    Default False uses p50 (expected-value basis)."""

    max_displaced_tasks_reported: int = 20
    """Cap on the number of displaced tasks surfaced in the artifact."""

    check_resource_conflicts: bool = True
    """Whether to check for crew and vendor resource conflicts at the
    insertion point."""


@dataclass
class InsertionOptionGeneratorConfig:
    """Configuration for the Stage F option generator."""

    max_options: int = 6
    """Maximum number of options to generate and surface."""

    include_infeasible_options: bool = True
    """If True, infeasible options are included with feasible=False and an
    infeasibility_reason. Outage managers benefit from seeing why an option
    was ruled out."""

    include_regulatory_blocked_options: bool = True
    """If True, regulatory-blocked options (regulatory_cleared=False) are
    included with the block reason. This makes the regulatory constraint
    visible rather than silently hiding the option."""


# ── Concrete no-op / file-backed implementations ─────────────────────────────

class NoOpSchemaValidator:
    """Pass-through validator for bootstrapping and unit tests.

    Every artifact passes validation. Logs a debug message per call.
    """

    def validate_artifact(self, artifact_name: str, payload: JsonDict) -> JsonDict:
        LOGGER.debug("NoOpSchemaValidator: skipping validation for %s", artifact_name)
        return {"ok": True, "issues": []}

    def validate_run_bundle(self, **kwargs: Any) -> JsonDict:
        LOGGER.debug("NoOpSchemaValidator: skipping run-bundle validation")
        return {"ok": True, "issues": []}


class FileArtifactStore:
    """Persists artifacts as JSON files under {root_dir}/{run_id}/{artifact_name}.json.

    Drop-in replacement for FileArtifactStore from dackar.RCA.orchestrators —
    identical interface, no dependency on the RCA package.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def save(self, run_id: str, artifact_name: str, payload: JsonDict) -> str:
        out_dir = self.root_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{artifact_name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        LOGGER.debug("Artifact saved: %s", path)
        return str(path)

    def save_list(
        self, run_id: str, artifact_name: str, payload: List[JsonDict]
    ) -> str:
        out_dir = self.root_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{artifact_name}.jsonl"
        with path.open("w") as fh:
            for record in payload:
                fh.write(json.dumps(record, default=str) + "\n")
        LOGGER.debug("Artifact list saved: %s (%d records)", path, len(payload))
        return str(path)
