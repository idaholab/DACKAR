"""
Stage D — Historical Analog Retriever.

Responsibilities:
    1. Build a retrieval query from the intake result (normalized description,
       component family, task family, system, discipline).
    2. Search the indexed historical activity store using a combination of
       lexical (BM25), semantic (embedding cosine), component-exact, and
       context similarity signals.
    3. Filter retrieved candidates by similarity threshold.
    4. Remove duration outliers from the analog set.
    5. Fit a duration distribution (lognormal / normal / empirical) from
       analog actual_duration_hours values.
    6. Fall back to population-level statistics when analog count is too low.
    7. Assign a confidence tier: data_supported / sme_informed / low_confidence.

Output schema: outage/schemas/historical_analogs.json

Note on stage ordering: Stage D runs before Stage E because the duration
distribution produced here is a required input to schedule impact assessment
(Stage E uses it to size the Monte Carlo simulation).

Reuse targets (all read-only imports, no modifications):
    outage_uncertainty.retrieval.similarity_engine.SimilarityEngine
        compare(query: ActivityCase, candidate: ActivityCase) → SimilarityMatch
    outage_uncertainty.retrieval.neighbor_selector.NeighborSelector
        select(matches: list[SimilarityMatch]) → list[SimilarityMatch]
    outage_uncertainty.retrieval.retrieval_index.HistoricalActivityIndex
        search(query: ActivityCase, top_k) → list[activity_id]
        get(activity_id) → ActivityCase
    outage_uncertainty.uncertainty.outlier_handler.OutlierHandler
        separate(durations, weights) → OutlierSeparation
    outage_uncertainty.uncertainty.distribution_fitter.DistributionFitter
        fit_from_separation(separation) → DurationDistribution
    outage_uncertainty.uncertainty.fallback_policy.HierarchicalFallbackPolicy
        estimate(query: ActivityCase, historical) → ActivityEstimate
    RCA.storage.chroma_store.ChromaRecordStore   (injected, semantic search backend)
    RCA.storage.processed_record_store.ProcessedRecordStore  (injected, text retrieval)

ActivityCase bridging:
    All retrieval scorers operate on ActivityCase objects.  Stage D receives
    JsonDict inputs, so _build_query() constructs a query ActivityCase from
    intake_result and returns it alongside the query_summary dict.  The
    ActivityCase is passed explicitly through the private call chain
    (_retrieve_candidates, _score_and_filter, _remove_duration_outliers,
    _compute_confidence_tier, _fit_duration_distribution) so that concurrent
    calls to retrieve() on the same instance cannot overwrite each other's state.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# Confidence tier labels
_TIER_DATA_SUPPORTED = "data_supported"
_TIER_SME_INFORMED = "sme_informed"
_TIER_LOW_CONFIDENCE = "low_confidence"

# Maps ConfidenceEstimator tier names ("high"/"medium"/"low") to Stage D tier names.
# Used when confidence_estimator is injected so the richer similarity-aware score
# drives the tier instead of the count-only fallback.
_CE_TIER_TO_STAGE_D: Dict[str, str] = {
    "high":   _TIER_DATA_SUPPORTED,
    "medium": _TIER_SME_INFORMED,
    "low":    _TIER_LOW_CONFIDENCE,
}


@dataclass
class _MatchProxy:
    """Minimal duck-typed SimilarityMatch for ConfidenceEstimator delegation.

    ConfidenceEstimator only accesses ``total_score`` and ``relevance_weight``
    on each match, so a full SimilarityMatch import is not required.
    """
    total_score: float
    relevance_weight: float

# ---------------------------------------------------------------------------
# ActivityCase bridge helpers
# ---------------------------------------------------------------------------

def _make_activity_case(
    description: str,
    *,
    component_family: Optional[str] = None,
    task_family: Optional[str] = None,
    discipline: Optional[str] = None,
    component_id: Optional[str] = None,
    system_id: Optional[str] = None,
    plant_id: Optional[str] = None,
    outage_phase: Optional[str] = None,
    is_emergent: bool = True,
    planned_duration_hours: Optional[float] = None,
    # Gap 3 execution mode flags — extracted by Stage A, passed through here
    # so ContextSimilarityScorer can weight analogues with matching conditions
    # higher.  Must be explicitly set (not left unset) because ActivityCase is
    # built via __new__ without __init__, so unset fields return None rather
    # than the dataclass default of False, causing the scorer to skip them.
    has_rp_hold: bool = False,
    requires_scaffold: bool = False,
    has_clearance: bool = False,
    is_vendor_supported: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construct an ActivityCase from free-form inputs.

    ActivityCase is imported from outage_uncertainty.domain.activity.  If the
    import fails, returns a lightweight _DictActivityCase stand-in that exposes
    the same attribute interface so all downstream scorers still work.

    Args:
        description: Cleaned/expanded text description.
        has_rp_hold: Radiation protection hold detected in description (Stage A).
        requires_scaffold: Scaffolding required (Stage A).
        has_clearance: Electrical/mechanical clearance required (Stage A).
        is_vendor_supported: Vendor/OEM involvement detected (Stage A).
        extra_fields: Any additional ActivityCase fields to set (e.g. metadata).

    Returns:
        ActivityCase (or _DictActivityCase fallback) ready for similarity scoring.
    """
    fields = {
        "raw_description": description,
        "cleaned_description": description,
        "component_family": component_family,
        "task_family": task_family,
        "discipline": discipline,
        "component_id": component_id,
        "system_id": system_id,
        "plant_id": plant_id,
        "outage_phase": outage_phase,
        "is_emergent": is_emergent,
        "planned_duration_hours": planned_duration_hours,
        "has_rp_hold": has_rp_hold,
        "requires_scaffold": requires_scaffold,
        "has_clearance": has_clearance,
        "is_vendor_supported": is_vendor_supported,
        **(extra_fields or {}),
    }

    try:
        from dackar.outage.outage_uncertainty.domain.activity import ActivityCase
        activity = ActivityCase.__new__(ActivityCase)
        for k, v in fields.items():
            try:
                object.__setattr__(activity, k, v)
            except (AttributeError, TypeError):
                pass
        return activity
    except ImportError:
        return _DictActivityCase(fields)


class _DictActivityCase:
    """Minimal stand-in for ActivityCase when the domain module is unavailable.

    Used when ``outage_uncertainty.domain.activity.ActivityCase`` cannot be
    imported.  All attribute access returns ``None`` for missing fields so
    scorers degrade gracefully instead of raising ``AttributeError``.

    .. note::
        If a scorer receives ``None`` for a field it expects, it will silently
        apply missing-field redistribution logic.  Check ``repr(qac)`` to
        confirm whether you are working with a real ``ActivityCase`` or this
        fallback — the repr includes ``"_DictActivityCase(fallback, ...)"``
        to make this immediately visible in logs and debugger output.
    """

    def __init__(self, fields: Dict[str, Any]) -> None:
        self.__dict__.update(fields)
        # Similarity scorers may access metadata["features"] for embeddings
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {"features": {}}

    def __getattr__(self, name: str) -> Any:
        return None

    def __repr__(self) -> str:
        populated = [k for k, v in self.__dict__.items() if k != "metadata" and v is not None]
        return f"_DictActivityCase(fallback, populated={populated})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HistoricalAnalogConfig:
    """Configuration for Stage D."""

    top_k: int = 20
    """Maximum number of candidates to retrieve before threshold filtering."""

    similarity_threshold: float = 0.60
    """Minimum overall similarity_score for an analog to be retained."""

    min_analogs_for_data_supported: int = 5
    """Minimum analogs with actual durations required for data_supported tier."""

    min_analogs_for_sme_informed: int = 1
    """Minimum analogs for sme_informed tier.  Below this, fallback is used."""

    min_outages_for_data_supported: int = 3
    """Minimum distinct outage IDs required in the analogue pool for the
    data_supported tier.  Prevents claiming cross-cycle validity from a single
    outage's worth of data.  When the count gate passes but the outage gate
    does not, the tier is capped at sme_informed (never lower than sme_informed
    solely due to this gate).  Set to 0 to disable."""

    outlier_iqr_factor: float = 1.5
    """IQR multiplier for duration outlier removal (Tukey fence).
    Wired into the injected OutlierHandler.iqr_multiplier at construction time.
    Standard Tukey value is 1.5; increase to 3.0 for a more permissive fence
    (fewer outliers removed) on small or heavily right-skewed pools."""

    lexical_weight: float = 0.20
    """Weight for the lexical (BM25 / token-overlap) similarity component.
    Passed to SimilarityAggregator at retriever construction time."""

    semantic_weight: float = 0.40
    """Weight for the semantic (embedding cosine / WordNet) similarity component.
    Passed to SimilarityAggregator at retriever construction time."""

    context_weight: float = 0.40
    """Weight for the context (structured metadata) similarity component.
    component_family affinity is embedded inside the context scorer, so this
    weight covers both execution-context fields and component-family matching.
    Passed to SimilarityAggregator at retriever construction time."""

    neighbor_selector_top_k: int = 20
    """Top-k passed to NeighborSelector (should equal or exceed config.top_k)."""

    prescorer_top_k_multiplier: int = 5
    """HistoricalActivityIndex.search() top_k = top_k × this multiplier
    (wider recall before expensive full scoring)."""

    schedule_context_rerank_weight: float = 0.10
    """Blend weight for the structural affinity boost applied when a
    ScheduleContext is available (Gap 4 two-pass design).

    Final score = (1 − w) × similarity_score + w × structural_affinity_score.

    Conservative default (0.10): the primary signal is still text / semantic /
    context similarity.  The structural component adds a tiebreaker based on
    CP membership, float tightness, and fan-out complexity at the insertion site.
    Set to 0.0 to disable re-ranking even when a ScheduleContext is provided.

    Keep this value low (≤ 0.15) until validated against multi-outage data."""


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class HistoricalAnalogRetriever:
    """Concrete Stage D implementation.

    Args:
        config: Stage configuration.
        similarity_engine: SimilarityEngine from outage_uncertainty.retrieval.
            compare(query: ActivityCase, candidate: ActivityCase) → SimilarityMatch.
        retrieval_index: HistoricalActivityIndex holding indexed historical activities.
            search(query: ActivityCase, top_k) → list[activity_id].
            get(activity_id) → ActivityCase.
        distribution_fitter: DistributionFitter from outage_uncertainty.uncertainty.
            fit_from_separation(OutlierSeparation) → DurationDistribution.
        outlier_handler: OutlierHandler from outage_uncertainty.uncertainty.
            separate(durations, weights) → OutlierSeparation.
        fallback_policy: HierarchicalFallbackPolicy from outage_uncertainty.uncertainty.
            estimate(query: ActivityCase, historical) → ActivityEstimate.
    """

    def __init__(
        self,
        config: Optional[HistoricalAnalogConfig] = None,
        *,
        similarity_engine=None,
        retrieval_index=None,
        distribution_fitter=None,
        outlier_handler=None,
        fallback_policy=None,
        confidence_estimator=None,
    ) -> None:
        self.config = config or HistoricalAnalogConfig()
        self.similarity_engine = similarity_engine
        # Wire config weights into the aggregator so lexical_weight /
        # semantic_weight / context_weight in HistoricalAnalogConfig actually
        # govern scoring.  Without this the injected SimilarityAggregator uses
        # its own construction-time defaults regardless of operator config.
        if similarity_engine is not None and hasattr(similarity_engine, "aggregator"):
            try:
                from dackar.outage.outage_uncertainty.retrieval.similarity_engine import (
                    SimilarityAggregator,
                )
                cfg = self.config
                similarity_engine.aggregator = SimilarityAggregator(
                    weights={
                        "lexical": cfg.lexical_weight,
                        "semantic": cfg.semantic_weight,
                        "context": cfg.context_weight,
                        "dependency": 0.0,  # dependency scorer controlled separately
                    }
                )
            except ImportError:
                LOGGER.debug(
                    "Stage D: could not import SimilarityAggregator — "
                    "aggregator weights not updated from config"
                )
        self.retrieval_index = retrieval_index
        self.distribution_fitter = distribution_fitter
        self.outlier_handler = outlier_handler
        # Wire config IQR multiplier so outlier_iqr_factor actually governs the fence.
        if outlier_handler is not None and hasattr(outlier_handler, "iqr_multiplier"):
            outlier_handler.iqr_multiplier = self.config.outlier_iqr_factor
        self.fallback_policy = fallback_policy
        self.confidence_estimator = confidence_estimator

        # (no per-call mutable state — qac is passed explicitly through the call chain)

    # ── Protocol method ───────────────────────────────────────────────────────

    def retrieve(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        run_context: JsonDict,
        *,
        schedule_context: Any = None,
    ) -> JsonDict:
        """Execute Stage D for one emergent activity.

        Args:
            emergent_activity: EmergentActivity artifact.
            intake_result: Stage A output.
            run_context: Run metadata block.
            schedule_context: Optional :class:`~stages.insertion_point_determiner.ScheduleContext`
                produced by the pre-pass ``InsertionPointDeterminer``.  When
                provided it enables two additional behaviours:

                1. **Topology stamp**: ``predecessor_ids`` and ``successor_ids``
                   are set on the query ``ActivityCase`` from the insertion
                   point neighbourhood.  This makes ``DependencyPatternScorer``
                   active as soon as historical topology data enters the index
                   (Gap 4 Layer 2 — deferred).

                2. **Structural affinity re-ranking**: ``_rerank_by_schedule_context()``
                   applies a soft boost to analogs that match the insertion
                   site's structural risk signals (CP membership, tight float,
                   fan-out complexity).  Blend weight controlled by
                   ``config.schedule_context_rerank_weight``.

        Returns:
            HistoricalAnalogs artifact conforming to
            outage/schemas/historical_analogs.json.
        """
        run_id: str = run_context["run_id"]
        activity_id: str = emergent_activity["activity_id"]
        LOGGER.debug("Stage D analog retrieval for %s (run=%s)", activity_id, run_id)

        query, qac = self._build_query(emergent_activity, intake_result)

        # ── Layer 2 prep: stamp insertion-site topology onto query ActivityCase ─
        # When historical ActivityCase objects in the index gain predecessor_ids /
        # successor_ids, DependencyPatternScorer will activate automatically.
        if schedule_context is not None and qac is not None:
            _stamp_topology(qac, schedule_context)

        candidates = self._retrieve_candidates(query, qac)
        analogs, candidates_below_threshold = self._score_and_filter(query, candidates, qac)

        # ── Layer 1: structural affinity re-ranking ──────────────────────────
        if schedule_context is not None and self.config.schedule_context_rerank_weight > 0:
            analogs = self._rerank_by_schedule_context(analogs, schedule_context)

        # ── Two-pass outlier design ───────────────────────────────────────────
        # Pass 1 (_remove_duration_outliers): hard-removes noise outliers from
        # the non-disruption sub-pool using the IQR fence.  Disruption-context
        # analogs (sharing an active execution-mode flag with the query) bypass
        # this fence — their elevated durations are signal, not noise.
        #
        # Pass 2 (_fit_duration_distribution → _fit_from_data): runs
        # OutlierHandler.separate() again on the cleaned pool (routine analogs +
        # disruption-context analogs mixed together).  Its purpose is different
        # from Pass 1: it classifies the disruption-context durations into the
        # "extended" group so DistributionFitter.fit_from_separation() can build
        # a two-component mixture model (routine mode + disruption mode).
        #
        # The two passes are intentionally sequential and serve distinct roles:
        # Pass 1 cleans noise; Pass 2 structures the remaining signal.
        analogs, outliers_removed = self._remove_duration_outliers(analogs, qac)
        distribution, fallback_used = self._fit_duration_distribution(analogs, qac)
        confidence_tier = self._compute_confidence_tier(analogs, qac)
        # Override the fitter's internal tier with the authoritative value from
        # _compute_confidence_tier(), which uses ConfidenceEstimator (similarity-
        # aware: CV, disruption fraction, outage diversity) when injected, or
        # count-based two-gate logic otherwise.  Either way, Stage E and G see
        # the same tier that drove the confidence assessment.
        distribution["confidence_tier"] = confidence_tier
        retrieval_summary = self._build_retrieval_summary(
            analogs, fallback_used, candidates_below_threshold
        )

        return {
            "activity_id": activity_id,
            "run_id": run_id,
            "generated_at": run_context.get("started_at", ""),
            "query_summary": query,
            "analogs": analogs,
            "duration_distribution": {**distribution, "outliers_removed": outliers_removed},
            "retrieval_summary": retrieval_summary,
            "provenance": {
                "generated_by": self.__class__.__name__,
                "run_id": run_id,
                "retrieval_index_version": None,
                "embedding_model": None,
            },
        }

    # ── Private step methods ──────────────────────────────────────────────────

    def _build_query(
        self, emergent_activity: JsonDict, intake_result: JsonDict
    ) -> JsonDict:
        """Assemble the retrieval query from intake result fields.

        Constructs an ActivityCase for the similarity scorers and returns it
        alongside the query_summary dict so callers can pass it explicitly
        through the private call chain without touching instance state.

        Returns (query_summary_dict, query_activity_case).

        Reuse: _make_activity_case() bridge — constructs ActivityCase from
        JsonDict fields without modifying outage_uncertainty domain classes.
        """
        description = (
            intake_result.get("expanded_description")
            or intake_result.get("normalized_description")
            or emergent_activity.get("raw_description", "")
        )
        component_ids = intake_result.get("resolved_component_ids") or []
        system_ids = intake_result.get("resolved_system_ids") or []
        primary_component_id = component_ids[0] if component_ids else None
        primary_system_id = system_ids[0] if system_ids else None

        planned_hours: Optional[float] = None
        try:
            raw = emergent_activity.get("planned_duration_hours")
            if raw is not None:
                planned_hours = float(raw)
        except (ValueError, TypeError):
            pass

        # Read execution mode flags extracted by Stage A.
        # Fall back to all-False when absent (e.g. pre-P2 intake artifacts or
        # test fixtures that don't include the key).
        _flags = intake_result.get("execution_mode_flags") or {}

        # Build ActivityCase for similarity scorers
        qac = _make_activity_case(
            description=description,
            component_family=intake_result.get("component_family"),
            task_family=intake_result.get("task_family"),
            discipline=intake_result.get("discipline"),
            component_id=primary_component_id,
            system_id=primary_system_id,
            plant_id=emergent_activity.get("plant_id"),
            outage_phase=emergent_activity.get("outage_phase"),
            is_emergent=True,
            planned_duration_hours=planned_hours,
            has_rp_hold=bool(_flags.get("has_rp_hold", False)),
            requires_scaffold=bool(_flags.get("requires_scaffold", False)),
            has_clearance=bool(_flags.get("has_clearance", False)),
            is_vendor_supported=bool(_flags.get("is_vendor_supported", False)),
        )

        return {
            "normalized_description": description,
            "component_id": primary_component_id,
            "system_id": primary_system_id,
            "component_family": intake_result.get("component_family"),
            "task_family": intake_result.get("task_family"),
            "discipline": intake_result.get("discipline"),
            "plant_id": emergent_activity.get("plant_id"),
            "outage_phase": emergent_activity.get("outage_phase"),
            "planned_duration_hours": planned_hours,
            # Execution mode flags recorded for artifact traceability — an analyst
            # reviewing the output can see which disruption conditions were active
            # for this retrieval run (affects similarity scoring and outlier routing).
            "execution_mode_flags": {
                "has_rp_hold": bool(_flags.get("has_rp_hold", False)),
                "requires_scaffold": bool(_flags.get("requires_scaffold", False)),
                "has_clearance": bool(_flags.get("has_clearance", False)),
                "is_vendor_supported": bool(_flags.get("is_vendor_supported", False)),
            },
        }, qac

    def _retrieve_candidates(self, query: JsonDict, qac: Any) -> List[Any]:
        """Search the retrieval index for the top-k most similar past activities.

        Two-step retrieval:
            1. HistoricalActivityIndex.search() — cheap lexical+context pre-filter
               returning candidate IDs (wide recall set).
            2. HistoricalActivityIndex.get() — hydrate each candidate ActivityCase
               for full SimilarityEngine scoring in _score_and_filter().

        Returns list of ActivityCase objects.  Returns [] when retrieval_index
        is not injected (no historical data available).

        Reuse: HistoricalActivityIndex from outage_uncertainty.retrieval.retrieval_index
        (read-only import, no modification).
        """
        if self.retrieval_index is None:
            LOGGER.warning("Stage D: retrieval_index not injected — returning empty analog set")
            return []

        query_activity = qac
        if query_activity is None:
            return []

        prescore_k = self.config.top_k * self.config.prescorer_top_k_multiplier
        try:
            candidate_ids: List[str] = self.retrieval_index.search(
                query_activity, top_k=prescore_k
            )
        except Exception:  # noqa: BLE001
            LOGGER.warning("Stage D: retrieval_index.search() failed", exc_info=True)
            return []

        candidates = []
        for cid in candidate_ids:
            try:
                activity = self.retrieval_index.get(cid)
                if activity is not None:
                    candidates.append(activity)
            except Exception:  # noqa: BLE001
                pass

        LOGGER.debug("Stage D: %d candidates from pre-filter", len(candidates))
        return candidates

    def _score_and_filter(
        self, query: JsonDict, candidates: List[Any], qac: Any
    ) -> Tuple[List[JsonDict], int]:
        """Score each candidate with the weighted composite similarity and filter.

        Composite score (3 components):
            lexical  × config.lexical_weight   (BM25 / token-overlap)
            semantic × config.semantic_weight  (embedding cosine / WordNet)
            context  × config.context_weight   (structured metadata; includes
                                                component_family affinity)

        Weights are wired into the injected SimilarityAggregator at construction
        time (see __init__).  config.lexical_weight / semantic_weight /
        context_weight are the authoritative sources for scoring behaviour.

        Steps:
            1. SimilarityEngine.compare(query_activity, candidate) → SimilarityMatch
            2. NeighborSelector.select(matches) — top-k + relevance weighting
            3. Filter matches below config.similarity_threshold
            4. Convert SimilarityMatch + ActivityCase → analog dict

        Returns:
            (analogs, candidates_below_threshold) where candidates_below_threshold
            is the count of scored candidates whose similarity score was below
            config.similarity_threshold (N12: exposed in retrieval_summary for
            production monitoring of retrieval quality).

        Reuse:
            SimilarityEngine from outage_uncertainty.retrieval.similarity_engine
            NeighborSelector from outage_uncertainty.retrieval.neighbor_selector
            (read-only imports, no modification).
        """
        if not candidates:
            return [], 0

        query_activity = qac

        # ── Scoring ───────────────────────────────────────────────────────────
        matches: List[Any] = []
        candidate_map: Dict[int, Any] = {}  # id(match) → ActivityCase

        if self.similarity_engine is not None and query_activity is not None:
            for candidate in candidates:
                try:
                    match = self.similarity_engine.compare(query_activity, candidate)
                    matches.append(match)
                    candidate_map[id(match)] = candidate
                except Exception:  # noqa: BLE001
                    LOGGER.debug("SimilarityEngine.compare() failed for a candidate")
        else:
            # No engine injected — assign uniform score 0.5 as placeholder.
            # candidates_below_threshold = 0 in this path: scores are synthetic,
            # not real threshold rejections.
            LOGGER.warning("Stage D: similarity_engine not injected — using placeholder scores")
            return [
                _activity_to_analog(c, 0.50, {}, relevance_weight=1.0)
                for c in candidates[: self.config.top_k]
            ], 0

        if not matches:
            return [], 0

        # N12: count candidates below threshold before any selection step.
        n_below_threshold: int = sum(
            1 for m in matches if _get_score(m) < self.config.similarity_threshold
        )

        # ── Neighbor selection (top-k + relevance weighting) ─────────────────
        try:
            from dackar.outage.outage_uncertainty.retrieval.neighbor_selector import (
                NeighborSelector,
            )
            selector = NeighborSelector(
                top_k=self.config.neighbor_selector_top_k,
                min_score=self.config.similarity_threshold,
            )
            matches = selector.select(matches)
        except ImportError:
            # Manual fallback: sort descending by score, cap, filter
            matches = sorted(
                matches,
                key=lambda m: _get_score(m),
                reverse=True,
            )[: self.config.top_k]
            matches = [m for m in matches if _get_score(m) >= self.config.similarity_threshold]

        # ── Convert to analog dicts ───────────────────────────────────────────
        analogs: List[JsonDict] = []
        for match in matches:
            score = _get_score(match)
            if score < self.config.similarity_threshold:
                continue
            candidate = candidate_map.get(id(match))
            if candidate is None:
                continue
            breakdown = _get_breakdown(match)
            rw = getattr(match, "relevance_weight", 1.0)
            analog = _activity_to_analog(candidate, score, breakdown, relevance_weight=rw)
            analogs.append(analog)

        LOGGER.debug("Stage D: %d analogs after scoring and filtering", len(analogs))
        return analogs, n_below_threshold

    def _rerank_by_schedule_context(
        self,
        analogs: List[JsonDict],
        schedule_context: Any,
    ) -> List[JsonDict]:
        """Apply a structural affinity boost to analog similarity scores.

        Blends the existing ``similarity_score`` with a ``structural_affinity_score``
        derived from the insertion site's schedule-risk signals.  The blend weight
        is ``config.schedule_context_rerank_weight`` (default 0.10 — conservative).

        Three structural signals are evaluated:

        CP membership boost (0–0.5):
            When the insertion site is on the critical path (``insertion_on_cp``),
            analogs whose actual duration exceeded their planned duration by ≥10%
            are boosted.  CP-overrun histories indicate the activity type is
            prone to extending the project finish, making them the most relevant
            comparators for a CP insertion.

        Tight-float precision boost (0–0.5):
            When ``available_float_hours < 8 h`` (near-critical insertion),
            analogs whose actual duration was within 15% of their planned
            duration are boosted.  Under tight float the outage manager needs
            predictable estimates — analogs with low relative variance provide
            tighter confidence bounds.

        Fan-out coordination boost (0–0.3):
            When the insertion site has high fan-out (``insertion_out_degree ≥ 3``,
            blocking 3+ downstream tasks simultaneously), ``is_vendor_supported``
            analogs are boosted.  Vendor activities carry inherent coordination
            complexity that mirrors the multi-successor blockage scenario.

        Boosts are capped at 1.0; each analog also gets a
        ``structural_affinity_score`` field for artifact traceability.
        Analogs are re-sorted by the blended score before returning.

        This is Layer 1 of the Gap 4 design.  Layer 2 (activating
        ``DependencyPatternScorer`` once historical topology enters the index)
        is enabled automatically when ``predecessor_ids`` / ``successor_ids``
        are populated on both query and candidate ``ActivityCase`` objects.
        """
        w = self.config.schedule_context_rerank_weight
        if w <= 0.0 or not analogs:
            return analogs

        on_cp: bool = bool(getattr(schedule_context, "insertion_on_cp", False))
        available_float: float = float(
            getattr(schedule_context, "available_float_hours", float("inf"))
        )
        out_degree: int = int(getattr(schedule_context, "insertion_out_degree", 0))

        tight_float = available_float < 8.0
        burst_site = out_degree >= 3

        result: List[JsonDict] = []
        for analog in analogs:
            actual = analog.get("actual_duration_hours")
            planned = analog.get("planned_duration_hours")
            is_vendor = bool(analog.get("is_vendor_supported"))

            affinity: float = 0.0

            # Signal 1 — CP overrun: overrun analogs are most relevant for CP insertions
            if on_cp and actual is not None and planned and planned > 0:
                if actual / planned > 1.10:
                    affinity += 0.5

            # Signal 2 — Tight float: low-variance analogs preferred near-critical path
            if tight_float and actual is not None and planned and planned > 0:
                if abs(actual - planned) / planned < 0.15:
                    affinity += 0.5

            # Signal 3 — Fan-out: vendor activities suit burst-node insertions
            if burst_site and is_vendor:
                affinity += 0.3

            affinity = min(1.0, affinity)
            orig = analog["similarity_score"]
            blended = round((1.0 - w) * orig + w * affinity, 4)

            updated = dict(analog)
            updated["similarity_score"] = blended
            updated["structural_affinity_score"] = round(affinity, 4)
            result.append(updated)

        result.sort(key=lambda a: a["similarity_score"], reverse=True)
        LOGGER.debug(
            "Stage D: structural affinity re-ranking applied "
            "(on_cp=%s, tight_float=%s, burst_site=%s, w=%.2f)",
            on_cp, tight_float, burst_site, w,
        )
        return result

    def _remove_duration_outliers(
        self, analogs: List[JsonDict], qac: Any
    ) -> Tuple[List[JsonDict], int]:
        """Pass 1 of the two-pass outlier design: hard-remove noise outliers.

        Only non-disruption analogs — those that do *not* share an active
        execution-mode flag with the query (has_rp_hold, requires_scaffold,
        has_clearance, is_vendor_supported) — are subject to the IQR fence.
        Their outliers are genuine noise (e.g. data-entry errors, mismatched
        activities) that would inflate the distribution tail.

        Disruption-context analogs bypass the fence entirely.  Their elevated
        durations reflect real execution under constrained conditions and must
        be preserved so that Pass 2 (``_fit_duration_distribution``) can
        classify them into the mixture model's extended component.

        Only analogs with non-null actual_duration_hours participate in
        outlier detection; analogs without duration data are always retained.

        Returns (cleaned_analogs, outliers_removed_count).

        Reuse: OutlierHandler from outage_uncertainty.uncertainty.outlier_handler
        (read-only import, no modification).
        """
        with_duration = [(i, a) for i, a in enumerate(analogs)
                         if a.get("actual_duration_hours") is not None]
        without_duration = [(i, a) for i, a in enumerate(analogs)
                            if a.get("actual_duration_hours") is None]

        if len(with_duration) < 2:
            return analogs, 0

        # Partition with-duration analogs into disruption-context and non-disruption.
        # Disruption-context analogs share at least one active execution mode flag
        # with the query; they bypass IQR so their extended durations feed the
        # mixture model.  Non-disruption analogs go through the normal IQR fence.
        active_flags = _active_execution_flags(qac)
        disruption_local_indices: set = set()
        if active_flags:
            for local_idx, (_, a) in enumerate(with_duration):
                if _analog_matches_flags(a, active_flags):
                    disruption_local_indices.add(local_idx)

        if active_flags and disruption_local_indices:
            LOGGER.debug(
                "Stage D: %d disruption-context analog(s) bypassing IQR fence "
                "(query active flags: %s)",
                len(disruption_local_indices),
                sorted(active_flags),
            )

        nd_pairs = [(i, a) for local_idx, (i, a) in enumerate(with_duration)
                    if local_idx not in disruption_local_indices]
        dc_pairs = [(i, a) for local_idx, (i, a) in enumerate(with_duration)
                    if local_idx in disruption_local_indices]

        # Apply IQR fence only to non-disruption analogs
        outliers_removed = 0
        if len(nd_pairs) >= 2:
            nd_durations = [a["actual_duration_hours"] for _, a in nd_pairs]
            nd_weights = [a.get("relevance_weight", 1.0) for _, a in nd_pairs]

            if self.outlier_handler is not None:
                try:
                    separation = self.outlier_handler.separate(nd_durations, nd_weights)
                    # Use list.remove() rather than set-membership so that
                    # duplicate duration values are consumed one-at-a-time.
                    # A set lookup would keep *all* analogs whose duration
                    # matches any routine value, incorrectly retaining
                    # duplicates that were classified as extended.
                    routine_remaining = list(separation.routine)
                    kept_nd = []
                    for (i, a), d in zip(nd_pairs, nd_durations):
                        try:
                            routine_remaining.remove(d)
                            kept_nd.append((i, a))
                        except ValueError:
                            pass  # d was classified as extended — drop it
                    outliers_removed = len(nd_pairs) - len(kept_nd)
                    # Integrity check: kept count must equal len(separation.routine).
                    # If it doesn't, OutlierHandler.separate() returned modified
                    # float values (e.g. rounded) that no longer match nd_durations
                    # exactly, causing silent data loss.  Fall back to keep-all so
                    # that the failure is loud and safe rather than quiet and lossy.
                    if len(kept_nd) != len(separation.routine):
                        LOGGER.warning(
                            "Stage D: outlier removal kept %d analogs but expected %d "
                            "(separation.routine size) — OutlierHandler may be "
                            "returning modified float values.  Falling back to "
                            "keep-all non-disruption analogs.",
                            len(kept_nd),
                            len(separation.routine),
                        )
                        kept_nd = nd_pairs
                        outliers_removed = 0
                except Exception:  # noqa: BLE001
                    LOGGER.warning("Stage D: outlier_handler.separate() failed; keeping all analogs")
                    kept_nd = nd_pairs
            else:
                kept_nd, outliers_removed = _tukey_filter(nd_pairs, nd_durations)
        else:
            kept_nd = nd_pairs

        # Reconstruct in original index order: routine non-disruption +
        # all disruption-context (always kept) + all without-duration analogs.
        kept = kept_nd + dc_pairs
        kept_indices = {i for i, _ in kept}
        with_duration_indices = {i for i, _ in with_duration}
        all_pairs = sorted(kept + without_duration, key=lambda x: x[0])
        result = [
            a for i, a in all_pairs
            if i in kept_indices or i not in with_duration_indices
        ]
        return result, outliers_removed

    def _fit_duration_distribution(
        self, analogs: List[JsonDict], qac: Any
    ) -> Tuple[JsonDict, bool]:
        """Fit a duration distribution from analog actual_duration_hours values.

        Returns (distribution_dict, fallback_used).

        distribution_dict keys: distribution_type, p50_hours, p80_hours,
        p90_hours, mean_hours, std_hours, confidence_tier, sample_size.

        Logic:
            sample_size = count of analogs with non-null actual_duration_hours.
            If sample_size >= max(2, config.min_analogs_for_sme_informed):
                Fit via DistributionFitter.fit_from_separation() (empirical,
                power-weight–adjusted percentiles).  fallback_used = False.
            Elif sample_size == 1:
                Single-analog prior: p50 = analog duration, p80/p90 inflated
                by fixed factors, distribution_type = "single_analog_prior",
                confidence_tier = low_confidence.  fallback_used = True.
                Prevents degenerate p50 == p80 == p90 that the empirical fitter
                would produce for n = 1, and avoids the false-precision of
                presenting a single data point as sme_informed (0.65 confidence).
            Else (sample_size == 0):
                Use HierarchicalFallbackPolicy.estimate() if injected, or
                compute a trivial planned-duration prior.  fallback_used = True.

        Reuse:
            DistributionFitter  from outage_uncertainty.uncertainty.distribution_fitter
            HierarchicalFallbackPolicy from outage_uncertainty.uncertainty.fallback_policy
            OutlierHandler for the OutlierSeparation needed by fit_from_separation()
            (all read-only imports, no modification).
        """
        durations = [
            a["actual_duration_hours"]
            for a in analogs
            if a.get("actual_duration_hours") is not None
        ]
        weights = [
            a.get("relevance_weight", 1.0)
            for a in analogs
            if a.get("actual_duration_hours") is not None
        ]
        sample_size = len(durations)

        if sample_size >= max(2, self.config.min_analogs_for_sme_informed):
            dist_dict = self._fit_from_data(durations, weights, sample_size)
            return dist_dict, False

        # ── Single-analog prior ───────────────────────────────────────────────
        if sample_size == 1:
            # One data point carries no variance information; fitting would
            # produce a degenerate distribution (p50 == p80 == p90).  Instead,
            # use the analog as a p50 anchor and apply conservative inflation
            # factors for higher percentiles.  Marking as low_confidence and
            # fallback_used=True ensures Stage G raises _FLAG_FALLBACK and the
            # outage manager knows the estimate rests on a single observation.
            single_dur = durations[0]
            return {
                "distribution_type": "single_analog_prior",
                "p50_hours": round(single_dur, 2),
                "p80_hours": round(single_dur * 1.30, 2),
                "p90_hours": round(single_dur * 1.50, 2),
                "mean_hours": round(single_dur, 2),
                "std_hours": None,
                "confidence_tier": _TIER_LOW_CONFIDENCE,
                "sample_size": 1,
            }, True

        # ── Fallback path (zero analogs) ──────────────────────────────────────
        fallback_dict = self._fit_from_fallback(durations, qac)
        return fallback_dict, True

    def _compute_confidence_tier(self, analogs: List[JsonDict], qac: Any) -> str:
        """Assign a confidence tier to the duration estimate.

        When a ``confidence_estimator`` is injected (preferred), delegates to
        :class:`~outage_uncertainty.uncertainty.confidence.ConfidenceEstimator`
        which uses similarity scores, coefficient of variation, and disruption
        fraction in addition to sample count and outage diversity.  The result
        tier ("high"/"medium"/"low") is mapped to Stage D names via
        ``_CE_TIER_TO_STAGE_D``.

        When no estimator is injected (or the call fails), falls back to the
        count-based two-gate logic:

        1. Sample-size gate:
           data_supported : sample_size >= config.min_analogs_for_data_supported
           sme_informed   : sample_size >= config.min_analogs_for_sme_informed
           low_confidence : otherwise

        2. Outage diversity gate (applied only at data_supported tier):
           Caps data_supported → sme_informed when the pool spans fewer than
           config.min_outages_for_data_supported distinct outage cycles.
           Disabled when the config threshold is 0.
        """
        # ── Primary path: delegate to ConfidenceEstimator ────────────────────
        # Requires outlier_handler to build the OutlierSeparation that
        # ConfidenceEstimator needs for CV and disruption-fraction signals.
        if (self.confidence_estimator is not None
                and qac is not None
                and self.outlier_handler is not None):
            try:
                with_dur = [a for a in analogs if a.get("actual_duration_hours") is not None]
                durations = [a["actual_duration_hours"] for a in with_dur]
                weights = [a.get("relevance_weight", 1.0) for a in with_dur]
                separation = self.outlier_handler.separate(durations, weights)
                matches = [
                    _MatchProxy(
                        total_score=a.get("similarity_score", 0.0),
                        relevance_weight=a.get("relevance_weight", 1.0),
                    )
                    for a in with_dur
                ]
                outages_represented = len(
                    {a["outage_id"] for a in analogs if a.get("outage_id")}
                )
                result = self.confidence_estimator.classify(
                    qac,
                    matches,
                    separation,
                    outages_represented=outages_represented,
                )
                return _CE_TIER_TO_STAGE_D.get(result.tier, _TIER_LOW_CONFIDENCE)
            except Exception:  # noqa: BLE001
                LOGGER.debug(
                    "Stage D: ConfidenceEstimator.classify() failed; "
                    "using count-based confidence tier"
                )

        # ── Fallback: count-based two-gate logic ─────────────────────────────
        sample_size = sum(
            1 for a in analogs if a.get("actual_duration_hours") is not None
        )

        # Count distinct outages in the analogue pool
        outages_represented = len(
            {a["outage_id"] for a in analogs if a.get("outage_id")}
        )

        # Sample-size gate
        if sample_size >= self.config.min_analogs_for_data_supported:
            count_tier = _TIER_DATA_SUPPORTED
        elif sample_size >= self.config.min_analogs_for_sme_informed:
            count_tier = _TIER_SME_INFORMED
        else:
            return _TIER_LOW_CONFIDENCE   # no analogs — diversity gate irrelevant

        # Outage diversity gate (applied only at data_supported tier).
        # The gate prevents over-claiming cross-cycle validity.  When the count
        # gate says data_supported but the outage gate fails, the tier is capped
        # at sme_informed — never lower.  sme_informed itself carries no outage
        # gate because it is already a "use with caution" tier.
        min_out_data = self.config.min_outages_for_data_supported

        if count_tier == _TIER_DATA_SUPPORTED:
            if min_out_data > 0 and outages_represented < min_out_data:
                LOGGER.debug(
                    "Stage D: outage diversity cap applied "
                    "(outages_represented=%d < min_outages_for_data_supported=%d); "
                    "capping data_supported → sme_informed.",
                    outages_represented, min_out_data,
                )
                return _TIER_SME_INFORMED
            return _TIER_DATA_SUPPORTED

        return _TIER_SME_INFORMED

    def _build_retrieval_summary(
        self,
        analogs: List[JsonDict],
        fallback_used: bool,
        candidates_below_threshold: int = 0,
    ) -> JsonDict:
        """Compute analog_count, outages_represented, plants_represented.

        N12 fix: ``candidates_below_threshold`` records how many candidates were
        scored by the SimilarityEngine but rejected because their score was below
        ``config.similarity_threshold``.  A high value relative to total candidates
        signals that the threshold may be too strict for the current retrieval index
        and should be reviewed before production deployment.
        """
        analog_count = len(analogs)
        outage_ids = {a["outage_id"] for a in analogs if a.get("outage_id")}
        plant_ids = {a["plant_id"] for a in analogs if a.get("plant_id")}

        best_score: Optional[float] = None
        if analogs:
            best_score = round(max(a.get("similarity_score", 0.0) for a in analogs), 4)

        return {
            "analog_count": analog_count,
            "outages_represented": len(outage_ids),
            "plants_represented": len(plant_ids),
            "best_similarity_score": best_score,
            "fallback_used": fallback_used,
            "candidates_below_threshold": candidates_below_threshold,
        }

    # ── Distribution fitting helpers ──────────────────────────────────────────

    def _fit_from_data(
        self,
        durations: List[float],
        weights: List[float],
        sample_size: int,
    ) -> JsonDict:
        """Pass 2 of the two-pass outlier design: fit the mixture distribution.

        Receives the noise-cleaned analog pool from Pass 1
        (``_remove_duration_outliers``), which contains both routine analogs
        and any disruption-context analogs that bypassed the IQR fence.

        Calls ``OutlierHandler.separate()`` a second time — intentionally.
        Here the separation's role is different from Pass 1: it classifies the
        disruption-context durations into ``separation.extended`` so that
        ``DistributionFitter.fit_from_separation()`` can build a two-component
        mixture model.  Pass 1 removed noise; this pass structures the signal.

        Falls back to ``_manual_distribution`` (weighted percentiles) if the
        fitter or handler is not injected or raises.

        Reuse: DistributionFitter + OutlierHandler (read-only imports).
        """
        if self.distribution_fitter is not None and self.outlier_handler is not None:
            try:
                separation = self.outlier_handler.separate(durations, weights)
                dist = self.distribution_fitter.fit_from_separation(separation)
                return _distribution_to_dict(dist, sample_size)
            except Exception:  # noqa: BLE001
                LOGGER.warning("Stage D: DistributionFitter failed; using manual percentiles")

        # Manual fallback: weighted percentiles
        return _manual_distribution(durations, weights, sample_size)

    def _fit_from_fallback(self, durations: List[float], qac: Any) -> JsonDict:
        """Build a fallback distribution when too few analogs are available.

        Uses HierarchicalFallbackPolicy (if injected) with the query ActivityCase.
        When that also fails, builds a trivial distribution from the planned
        duration stored in the query summary.

        Reuse: HierarchicalFallbackPolicy from outage_uncertainty.uncertainty.fallback_policy
        (read-only import, no modification).
        """
        if self.fallback_policy is not None and qac is not None:
            try:
                estimate = self.fallback_policy.estimate(qac)
                return _estimate_to_dict(estimate, len(durations))
            except Exception:  # noqa: BLE001
                LOGGER.warning("Stage D: fallback_policy.estimate() failed; using prior only")

        # Last resort: planned_duration_hours as a point estimate
        planned = getattr(qac, "planned_duration_hours", None)
        if planned:
            return {
                "distribution_type": "point_prior",
                "p50_hours": float(planned),
                "p80_hours": round(float(planned) * 1.30, 2),
                "p90_hours": round(float(planned) * 1.50, 2),
                "mean_hours": float(planned),
                "std_hours": None,
                "confidence_tier": _TIER_LOW_CONFIDENCE,
                "sample_size": len(durations),
            }

        return {
            "distribution_type": "unknown",
            "p50_hours": None,
            "p80_hours": None,
            "p90_hours": None,
            "mean_hours": None,
            "std_hours": None,
            "confidence_tier": _TIER_LOW_CONFIDENCE,
            "sample_size": len(durations),
        }


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _get_score(match: Any) -> float:
    """Extract the total similarity score from a SimilarityMatch."""
    for attr in ("score", "total_score", "similarity_score"):
        val = getattr(match, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def _get_breakdown(match: Any) -> Dict[str, float]:
    """Extract per-dimension scores from a SimilarityMatch."""
    breakdown: Dict[str, float] = {}
    for dim in ("lexical", "semantic", "context", "dependency"):
        val = getattr(match, dim, None)
        if val is not None:
            try:
                breakdown[dim] = round(float(val), 4)
            except (TypeError, ValueError):
                pass
    return breakdown


def _activity_to_analog(
    activity: Any,
    score: float,
    breakdown: Dict[str, float],
    *,
    relevance_weight: float = 1.0,
) -> JsonDict:
    """Convert an ActivityCase + SimilarityMatch score into an analog dict."""
    def _g(attr, default=None):
        val = getattr(activity, attr, None)
        return val if val is not None else default

    return {
        "analog_id": f"ANALOG::{_g('activity_id', uuid.uuid4().hex[:8])}",
        "source_activity_id": _g("activity_id"),
        "similarity_score": round(score, 4),
        "component_id": _g("component_id"),
        "component_family": _g("component_family"),
        "task_family": _g("task_family"),
        "discipline": _g("discipline"),
        "outage_id": _g("outage_id"),
        "plant_id": _g("plant_id"),
        "actual_duration_hours": _g("actual_duration_hours"),
        "planned_duration_hours": _g("planned_duration_hours"),
        "description": _g("cleaned_description") or _g("raw_description", ""),
        "similarity_breakdown": breakdown,
        "relevance_weight": round(relevance_weight, 4),
        # Execution mode flags — propagated for disruption-context outlier routing
        # and downstream traceability.  False when absent on the source ActivityCase.
        "has_rp_hold": bool(_g("has_rp_hold") or False),
        "requires_scaffold": bool(_g("requires_scaffold") or False),
        "has_clearance": bool(_g("has_clearance") or False),
        "is_vendor_supported": bool(_g("is_vendor_supported") or False),
    }


def _stamp_topology(query_activity_case: Any, schedule_context: Any) -> None:
    """Stamp predecessor_ids / successor_ids onto the query ActivityCase.

    Uses the insertion site neighbourhood from ``ScheduleContext`` to give
    the query a topological role.  This is a no-op when the fields cannot
    be set (e.g. ``_DictActivityCase`` fallback).

    Purpose: Gap 4 Layer 2 preparation.  Once historical ``ActivityCase``
    objects in the retrieval index gain populated predecessor_ids /
    successor_ids, ``DependencyPatternScorer`` will begin contributing a
    non-zero score automatically — no additional code changes required.
    """
    after_id = getattr(schedule_context, "after_task_id", None)
    before_id = getattr(schedule_context, "before_task_id", None)
    try:
        if after_id:
            query_activity_case.predecessor_ids = [after_id]
        if before_id:
            query_activity_case.successor_ids = [before_id]
    except (AttributeError, TypeError):
        pass  # _DictActivityCase or frozen dataclass — skip gracefully


def _tukey_filter(
    indexed: List[Tuple[int, JsonDict]],
    durations: List[float],
) -> Tuple[List[Tuple[int, JsonDict]], int]:
    """Fallback Tukey IQR fence when OutlierHandler is not injected.

    Uses the same linear-interpolation quartile method as
    ``OutlierHandler._interpolated_percentile`` so that the fence value is
    consistent with the primary path.  The previous nearest-rank approach
    (``sorted_d[n // 4]`` / ``sorted_d[(3 * n) // 4]``) set q3 = max(data)
    for n == 4, making the fence unreachable and silently skipping all
    outlier removal at that sample size.

    Note: quartiles are computed on unweighted sorted durations (weights are
    not available at this call site — they are embedded in each analog dict).
    This matches the unweighted Q1/Q3 in ``OutlierHandler._iqr``.
    """
    if len(durations) < 4:
        return indexed, 0
    sorted_d = sorted(durations)
    n = len(sorted_d)

    def _interp(q: float) -> float:
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_d[lo] * (1.0 - frac) + sorted_d[hi] * frac

    q1 = _interp(0.25)
    q3 = _interp(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    kept = [(i, a) for (i, a), d in zip(indexed, durations) if d <= upper]
    return kept, len(indexed) - len(kept)


# ---------------------------------------------------------------------------
# Execution-mode flag helpers for disruption-context outlier routing
# ---------------------------------------------------------------------------

_EXECUTION_MODE_FLAGS: Tuple[str, ...] = (
    "has_rp_hold",
    "requires_scaffold",
    "has_clearance",
    "is_vendor_supported",
)


def _active_execution_flags(query_activity: Any) -> frozenset:
    """Return the set of execution mode flag names that are True on the query.

    A frozenset is returned so callers can cheaply check intersection.  An
    empty frozenset means no flags are active (gate is disabled).
    """
    if query_activity is None:
        return frozenset()
    return frozenset(
        f for f in _EXECUTION_MODE_FLAGS
        if bool(getattr(query_activity, f, False))
    )


def _analog_matches_flags(analog: JsonDict, active_flags: frozenset) -> bool:
    """Return True if *analog* has any flag in *active_flags* set to True.

    Used to identify disruption-context analogues that should bypass the
    IQR fence in ``_remove_duration_outliers``.
    """
    return any(bool(analog.get(f, False)) for f in active_flags)


def _distribution_to_dict(dist: Any, sample_size: int) -> JsonDict:
    """Convert a DurationDistribution object to a Stage D schema dict."""
    def _g(attr):
        v = getattr(dist, attr, None)
        return round(float(v), 2) if v is not None else None

    return {
        "distribution_type": getattr(dist, "distribution_type", "empirical"),
        "p50_hours": _g("p50"),
        "p80_hours": _g("p80"),
        "p90_hours": _g("p90"),
        "mean_hours": round(dist.mean(), 2),
        "std_hours": (
            round(dist.variance() ** 0.5, 2)
            if len(getattr(dist, "samples", None) or []) >= 2
            else None
        ),
        "confidence_tier": getattr(dist, "confidence_tier", None),
        "sample_size": sample_size,
    }


def _estimate_to_dict(estimate: Any, sample_size: int) -> JsonDict:
    """Convert an ActivityEstimate (fallback policy output) to a distribution dict."""
    # ActivityEstimate nests the distribution under `estimated_distribution`.
    dist = getattr(estimate, "estimated_distribution", None) or estimate
    mean_h = round(dist.mean(), 2) if callable(getattr(dist, "mean", None)) else _safe_float(getattr(dist, "p50", None))
    return {
        "distribution_type": getattr(dist, "distribution_type", "fallback"),
        "p50_hours": _safe_float(getattr(dist, "p50", None)),
        "p80_hours": _safe_float(getattr(dist, "p80", None)),
        "p90_hours": _safe_float(getattr(dist, "p90", None)),
        "mean_hours": mean_h,
        "std_hours": _safe_float(getattr(dist, "std", None)),
        "confidence_tier": _TIER_LOW_CONFIDENCE,
        "sample_size": sample_size,
    }


def _manual_distribution(
    durations: List[float],
    weights: List[float],
    sample_size: int,
) -> JsonDict:
    """Compute weighted percentile distribution without DistributionFitter.

    Uses the same midpoint-CDF + linear-interpolation algorithm as
    ``_weighted_percentile`` in distribution_fitter.py so that this fallback
    path produces consistent percentile estimates.
    """
    if not durations:
        return {
            "distribution_type": "empirical",
            "p50_hours": None, "p80_hours": None, "p90_hours": None,
            "mean_hours": None, "std_hours": None,
            "confidence_tier": None, "sample_size": 0,
        }

    try:
        from dackar.outage.outage_uncertainty.uncertainty.distribution_fitter import (
            _weighted_percentile,
        )
    except ImportError:
        _weighted_percentile = _weighted_percentile_fallback  # type: ignore[assignment]

    def _wp(q: float) -> float:
        return _weighted_percentile(durations, weights, q)

    total_w = sum(weights) or 1.0
    norm_w = [w / total_w for w in weights]
    mean_h = sum(d * w for d, w in zip(durations, norm_w))
    variance = sum(w * (d - mean_h) ** 2 for d, w in zip(durations, norm_w))
    std_h = math.sqrt(variance) if variance > 0 else 0.0

    return {
        "distribution_type": "empirical",
        "p50_hours": round(_wp(0.50), 2),
        "p80_hours": round(_wp(0.80), 2),
        "p90_hours": round(_wp(0.90), 2),
        "mean_hours": round(mean_h, 2),
        "std_hours": round(std_h, 2),
        "confidence_tier": None,
        "sample_size": sample_size,
    }


def _weighted_percentile_fallback(
    values: List[float],
    weights: List[float],
    q: float,
) -> float:
    """Minimal midpoint-CDF weighted percentile used when distribution_fitter
    cannot be imported (pure-stdlib emergency fallback, same algorithm)."""
    if not values:
        return 0.0
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    sorted_vals = [p[0] for p in pairs]
    sorted_w = [p[1] for p in pairs]
    total = sum(sorted_w)
    if total <= 0.0:
        sorted_w = [1.0] * len(sorted_vals)
        total = float(len(sorted_vals))
    cdf: List[float] = []
    cumulative = 0.0
    for w in sorted_w:
        cdf.append((cumulative + 0.5 * w) / total)
        cumulative += w
    if q <= cdf[0]:
        return sorted_vals[0]
    if q >= cdf[-1]:
        return sorted_vals[-1]
    for i in range(1, len(sorted_vals)):
        if cdf[i] >= q:
            span = cdf[i] - cdf[i - 1]
            if span <= 0.0:
                return sorted_vals[i]
            frac = (q - cdf[i - 1]) / span
            return sorted_vals[i - 1] + frac * (sorted_vals[i] - sorted_vals[i - 1])
    return sorted_vals[-1]


def _safe_float(val: Any) -> Optional[float]:
    """Convert to rounded float or None."""
    if val is None:
        return None
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None
