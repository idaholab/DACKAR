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
import re
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
        schedule_context: Optional[Any] = None,
    ) -> JsonDict:
        """
        Args:
            emergent_activity: Validated EmergentActivity artifact.
            intake_result: Validated ActivityIntakeResult from Stage A.
            run_context: Run metadata.
            schedule_context: Optional pre-computed schedule context (Gap 4
                two-pass design); the orchestrator always passes this.

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
        - P6_adapter.outage_model (OutageDataset — schedule version access)

    Implemented by the Stage E schedule stage, which builds the critical-path
    network and runs the Monte Carlo impact estimate directly on the LOGOS
    ``Pert`` engine.
    """

    def assess(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        historical_analogs: JsonDict,
        run_context: JsonDict,
        schedule_context: Optional[Any] = None,
    ) -> JsonDict:
        """
        Args:
            emergent_activity: Validated EmergentActivity artifact.
            intake_result: Validated ActivityIntakeResult from Stage A.
            historical_analogs: Validated HistoricalAnalogs from Stage D.
                                Duration distribution is consumed here.
            run_context: Run metadata.
            schedule_context: Optional pre-computed schedule context (Gap 4
                two-pass design); the orchestrator always passes this.

        Returns:
            Validated ScheduleImpactAssessment artifact.
        """
        ...


# ── Stage F — Insertion Option Generator ─────────────────────────────────────

@runtime_checkable
class InsertionOptionGeneratorProtocol(Protocol):
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
class RecommendationSynthesizerProtocol(Protocol):
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
    """Orchestrator-level analyst-review gate. If a stage reports an
    unknown_abbreviation_rate above this value, the orchestrator forces
    analyst_review. (Stage G keeps its own copy on RecommendationConfig for its
    internal flagging; this one drives the orchestrator's review hook.)"""

    # NOTE: per-stage thresholds such as near_critical_float_threshold_hours
    # (insertion_point_determiner) and monte_carlo_runs (ScheduleImpactConfig)
    # are NOT configured here — setting them on the orchestrator had no effect
    # because each stage reads its own config, yet the run manifest reported
    # them as the pipeline config.  Configure them on the relevant stage.

    extra: JsonDict = field(default_factory=dict)
    """Reserved for future configuration without breaking the dataclass interface."""


# NOTE: Per-stage configuration lives with each stage (RecommendationConfig,
# ScheduleImpactConfig, HistoricalAnalogConfig, …) and is what the stages
# actually read.  Earlier revisions duplicated those here
# (KGTimelineBuilderConfig, TemporalChainScorerConfig,
# HistoricalAnalogRetrieverConfig, ScheduleImpactAssessorConfig,
# InsertionOptionGeneratorConfig); those copies were never consumed and their
# defaults had already drifted from the real per-stage configs, so they were
# removed to avoid implying the orchestrator configures the stages.


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

    def _run_dir(self, run_id: str) -> Path:
        """Return the per-run directory, with the run_id made a filesystem-safe
        path segment.

        run_ids are formatted ``PREFIX::hex`` (see new_run_id), and ``:`` is a
        reserved character on Windows (drive separator), so using the run_id
        verbatim as a directory name raises WinError 123 there.  Characters that
        are illegal on Windows (``<>:"/\\|?*``) are mapped to ``_`` so the same
        run writes to the same portable directory on every OS.  The run_id value
        stored *inside* the artifacts is unchanged.
        """
        safe = re.sub(r'[<>:"/\\|?*]', "_", run_id)
        return self.root_dir / safe

    def save(self, run_id: str, artifact_name: str, payload: JsonDict) -> str:
        out_dir = self._run_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{artifact_name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        LOGGER.debug("Artifact saved: %s", path)
        return str(path)

    def save_list(
        self, run_id: str, artifact_name: str, payload: List[JsonDict]
    ) -> str:
        out_dir = self._run_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{artifact_name}.jsonl"
        with path.open("w") as fh:
            for record in payload:
                fh.write(json.dumps(record, default=str) + "\n")
        LOGGER.debug("Artifact list saved: %s (%d records)", path, len(payload))
        return str(path)
