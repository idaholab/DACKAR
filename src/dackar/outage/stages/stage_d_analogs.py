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
    intake_result and stores it as self._query_activity_case (temporary
    per-call attribute) without polluting the query_summary output dict.
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
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construct an ActivityCase from free-form inputs.

    ActivityCase is imported from outage_uncertainty.domain.activity.  If the
    import fails, returns a lightweight _DictActivityCase stand-in that exposes
    the same attribute interface so all downstream scorers still work.

    Args:
        description: Cleaned/expanded text description.
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
    """Minimal stand-in for ActivityCase when the domain module is unavailable."""

    def __init__(self, fields: Dict[str, Any]) -> None:
        self.__dict__.update(fields)
        # Similarity scorers may access metadata["features"] for embeddings
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {"features": {}}

    def __getattr__(self, name: str) -> Any:
        return None


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

    outlier_iqr_factor: float = 1.5
    """IQR multiplier for duration outlier removal (Tukey fence).
    Passed to OutlierHandler(strategy='iqr').  The IQR factor is not directly
    configurable on OutlierHandler; Tukey 1.5 IQR is the default."""

    lexical_weight: float = 0.30
    """Weight for BM25 / token-overlap similarity component."""

    semantic_weight: float = 0.40
    """Weight for embedding cosine similarity component."""

    component_weight: float = 0.20
    """Weight for component-exact / component-family match component."""

    context_weight: float = 0.10
    """Weight for execution-context similarity component
    (phase, discipline, execution mode flags)."""

    neighbor_selector_top_k: int = 20
    """Top-k passed to NeighborSelector (should equal or exceed config.top_k)."""

    prescorer_top_k_multiplier: int = 5
    """HistoricalActivityIndex.search() top_k = top_k × this multiplier
    (wider recall before expensive full scoring)."""


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
    ) -> None:
        self.config = config or HistoricalAnalogConfig()
        self.similarity_engine = similarity_engine
        self.retrieval_index = retrieval_index
        self.distribution_fitter = distribution_fitter
        self.outlier_handler = outlier_handler
        self.fallback_policy = fallback_policy

        # Per-call temporary state — set in _build_query, consumed by
        # _retrieve_candidates and _score_and_filter.  Cleared after retrieve().
        self._query_activity_case: Any = None

    # ── Protocol method ───────────────────────────────────────────────────────

    def retrieve(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """Execute Stage D for one emergent activity.

        Returns:
            HistoricalAnalogs artifact conforming to
            outage/schemas/historical_analogs.json.
        """
        run_id: str = run_context["run_id"]
        activity_id: str = emergent_activity["activity_id"]
        LOGGER.debug("Stage D analog retrieval for %s (run=%s)", activity_id, run_id)

        query = self._build_query(emergent_activity, intake_result)
        candidates = self._retrieve_candidates(query)
        analogs = self._score_and_filter(query, candidates)
        analogs, outliers_removed = self._remove_duration_outliers(analogs)
        distribution, fallback_used = self._fit_duration_distribution(analogs)
        confidence_tier = self._compute_confidence_tier(analogs)
        # Authoritative tier is count-based (not the fitter's internal tier),
        # so we override here to ensure Stage E and G see a consistent value.
        distribution["confidence_tier"] = confidence_tier
        retrieval_summary = self._build_retrieval_summary(analogs, fallback_used)

        # Clear per-call state
        self._query_activity_case = None

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

        Constructs an ActivityCase for the similarity scorers and stores it in
        self._query_activity_case (not included in the returned dict).

        Returns a clean query_summary dict included verbatim in the artifact.

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

        # Build ActivityCase for similarity scorers — stored as temp attribute
        self._query_activity_case = _make_activity_case(
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
        }

    def _retrieve_candidates(self, query: JsonDict) -> List[Any]:
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

        query_activity = self._query_activity_case
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
        self, query: JsonDict, candidates: List[Any]
    ) -> List[JsonDict]:
        """Score each candidate with the weighted composite similarity and filter.

        Composite score = (
            lexical  × config.lexical_weight  +
            semantic × config.semantic_weight +
            component_match × config.component_weight +
            context  × config.context_weight
        )

        Steps:
            1. SimilarityEngine.compare(query_activity, candidate) → SimilarityMatch
            2. NeighborSelector.select(matches) — top-k + relevance weighting
            3. Filter matches below config.similarity_threshold
            4. Convert SimilarityMatch + ActivityCase → analog dict

        Reuse:
            SimilarityEngine from outage_uncertainty.retrieval.similarity_engine
            NeighborSelector from outage_uncertainty.retrieval.neighbor_selector
            (read-only imports, no modification).
        """
        if not candidates:
            return []

        query_activity = self._query_activity_case

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
            # No engine injected — assign uniform score 0.5 as placeholder
            LOGGER.warning("Stage D: similarity_engine not injected — using placeholder scores")
            return [
                _activity_to_analog(c, 0.50, {}, relevance_weight=1.0)
                for c in candidates[: self.config.top_k]
            ]

        if not matches:
            return []

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
        return analogs

    def _remove_duration_outliers(
        self, analogs: List[JsonDict]
    ) -> Tuple[List[JsonDict], int]:
        """Remove duration outliers using OutlierHandler (IQR strategy).

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

        durations = [a["actual_duration_hours"] for _, a in with_duration]
        weights = [a.get("relevance_weight", 1.0) for _, a in with_duration]

        if self.outlier_handler is not None:
            try:
                separation = self.outlier_handler.separate(durations, weights)
                routine_set = set(separation.routine)
                kept = [(i, a) for (i, a), d in zip(with_duration, durations)
                        if d in routine_set]
                outliers_removed = len(with_duration) - len(kept)
            except Exception:  # noqa: BLE001
                LOGGER.warning("Stage D: outlier_handler.separate() failed; keeping all analogs")
                kept = with_duration
                outliers_removed = 0
        else:
            # Fallback: manual Tukey IQR fence
            kept, outliers_removed = _tukey_filter(with_duration, durations)

        # Reconstruct in original index order: retained with-duration + all without-duration
        kept_indices = {i for i, _ in kept}
        all_pairs = sorted(kept + without_duration, key=lambda x: x[0])
        result = [a for i, a in all_pairs if i in kept_indices or i not in {j for j, _ in with_duration}]
        return result, outliers_removed

    def _fit_duration_distribution(
        self, analogs: List[JsonDict]
    ) -> Tuple[JsonDict, bool]:
        """Fit a duration distribution from analog actual_duration_hours values.

        Returns (distribution_dict, fallback_used).

        distribution_dict keys: distribution_type, p50_hours, p80_hours,
        p90_hours, mean_hours, std_hours, confidence_tier, sample_size.

        Logic:
            sample_size = count of analogs with non-null actual_duration_hours.
            If sample_size >= config.min_analogs_for_sme_informed:
                Fit via DistributionFitter.fit_from_separation() (empirical,
                power-weight–adjusted percentiles).  fallback_used = False.
            Else:
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

        if sample_size >= self.config.min_analogs_for_sme_informed:
            dist_dict = self._fit_from_data(durations, weights, sample_size)
            return dist_dict, False

        # ── Fallback path ─────────────────────────────────────────────────────
        fallback_dict = self._fit_from_fallback(durations)
        return fallback_dict, True

    def _compute_confidence_tier(self, analogs: List[JsonDict]) -> str:
        """Map analog count to a confidence tier.

        data_supported : sample_size >= config.min_analogs_for_data_supported
        sme_informed   : sample_size >= config.min_analogs_for_sme_informed
        low_confidence : otherwise (fallback distribution in use)
        """
        sample_size = sum(
            1 for a in analogs if a.get("actual_duration_hours") is not None
        )
        if sample_size >= self.config.min_analogs_for_data_supported:
            return _TIER_DATA_SUPPORTED
        if sample_size >= self.config.min_analogs_for_sme_informed:
            return _TIER_SME_INFORMED
        return _TIER_LOW_CONFIDENCE

    def _build_retrieval_summary(
        self, analogs: List[JsonDict], fallback_used: bool
    ) -> JsonDict:
        """Compute analog_count, outages_represented, plants_represented."""
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
        }

    # ── Distribution fitting helpers ──────────────────────────────────────────

    def _fit_from_data(
        self,
        durations: List[float],
        weights: List[float],
        sample_size: int,
    ) -> JsonDict:
        """Fit empirical distribution from analog duration data.

        Delegates to DistributionFitter.fit_from_separation() when the fitter
        is injected; falls back to direct weighted-percentile computation.

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

    def _fit_from_fallback(self, durations: List[float]) -> JsonDict:
        """Build a fallback distribution when too few analogs are available.

        Uses HierarchicalFallbackPolicy (if injected) with the query ActivityCase.
        When that also fails, builds a trivial distribution from the planned
        duration stored in the query summary.

        Reuse: HierarchicalFallbackPolicy from outage_uncertainty.uncertainty.fallback_policy
        (read-only import, no modification).
        """
        if self.fallback_policy is not None and self._query_activity_case is not None:
            try:
                estimate = self.fallback_policy.estimate(self._query_activity_case)
                return _estimate_to_dict(estimate, len(durations))
            except Exception:  # noqa: BLE001
                LOGGER.warning("Stage D: fallback_policy.estimate() failed; using prior only")

        # Last resort: planned_duration_hours as a point estimate
        planned = getattr(self._query_activity_case, "planned_duration_hours", None)
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
    }


def _tukey_filter(
    indexed: List[Tuple[int, JsonDict]],
    durations: List[float],
) -> Tuple[List[Tuple[int, JsonDict]], int]:
    """Fallback Tukey IQR fence when OutlierHandler is not injected."""
    if len(durations) < 4:
        return indexed, 0
    sorted_d = sorted(durations)
    n = len(sorted_d)
    q1 = sorted_d[n // 4]
    q3 = sorted_d[(3 * n) // 4]
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    kept = [(i, a) for (i, a), d in zip(indexed, durations) if d <= upper]
    return kept, len(indexed) - len(kept)


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
        "mean_hours": _g("mean") or _g("p50"),
        "std_hours": _g("std"),
        "confidence_tier": getattr(dist, "confidence_tier", None),
        "sample_size": sample_size,
    }


def _estimate_to_dict(estimate: Any, sample_size: int) -> JsonDict:
    """Convert an ActivityEstimate (fallback policy output) to a distribution dict."""
    # ActivityEstimate may have a nested DurationDistribution or direct percentile attrs
    dist = getattr(estimate, "duration_distribution", None) or estimate
    return {
        "distribution_type": getattr(dist, "distribution_type", "fallback"),
        "p50_hours": _safe_float(getattr(dist, "p50", None)),
        "p80_hours": _safe_float(getattr(dist, "p80", None)),
        "p90_hours": _safe_float(getattr(dist, "p90", None)),
        "mean_hours": _safe_float(getattr(dist, "mean", None)),
        "std_hours": _safe_float(getattr(dist, "std", None)),
        "confidence_tier": _TIER_LOW_CONFIDENCE,
        "sample_size": sample_size,
    }


def _manual_distribution(
    durations: List[float],
    weights: List[float],
    sample_size: int,
) -> JsonDict:
    """Compute weighted percentile distribution without DistributionFitter."""
    if not durations:
        return {
            "distribution_type": "empirical",
            "p50_hours": None, "p80_hours": None, "p90_hours": None,
            "mean_hours": None, "std_hours": None,
            "confidence_tier": None, "sample_size": 0,
        }
    total_w = sum(weights) or 1.0
    norm_w = [w / total_w for w in weights]
    pairs = sorted(zip(durations, norm_w), key=lambda x: x[0])
    sorted_d = [p[0] for p in pairs]
    sorted_w = [p[1] for p in pairs]
    cumulative = []
    acc = 0.0
    for w in sorted_w:
        acc += w
        cumulative.append(acc)

    def _wp(q: float) -> float:
        for i, c in enumerate(cumulative):
            if c >= q:
                return sorted_d[i]
        return sorted_d[-1]

    mean_h = sum(d * w for d, w in zip(sorted_d, sorted_w))
    variance = sum(w * (d - mean_h) ** 2 for d, w in zip(sorted_d, sorted_w))
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


def _safe_float(val: Any) -> Optional[float]:
    """Convert to rounded float or None."""
    if val is None:
        return None
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None
