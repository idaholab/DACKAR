from __future__ import annotations

from dackar.similarity.scoreAggregator import WeightedScoreAggregator

from dackar.outage.outage_uncertainty.domain.activity import ActivityCase
from dackar.outage.outage_uncertainty.domain.result_types import SimilarityMatch


class SimilarityAggregator(WeightedScoreAggregator):
    """Combine the outage similarity dimensions into a single total score.

    A thin outage adapter over
    :class:`dackar.similarity.scoreAggregator.WeightedScoreAggregator` that
    pins the four retrieval dimensions (lexical / semantic / context /
    dependency) and keeps the keyword-style :meth:`combine` signature the
    engine calls.

    Args:
        weights: ``{dimension: weight}`` dict.  The default ships with
            ``"dependency": 0.0`` so the aggregator is backwards compatible
            when no dependency scorer is wired.  Pass a non-zero
            ``"dependency"`` weight together with a
            :class:`~outage_uncertainty.retrieval.dependency_similarity.DependencyPatternScorer`
            to activate the schedule-neighbourhood component (Gap 4).

    Default weights (dependency disabled)::

        lexical: 0.20,  semantic: 0.40,  context: 0.40,  dependency: 0.00
    """

    def __init__(self, weights: dict[str, float] | None = None):
        super().__init__(weights or {
            "lexical":     0.2,
            "semantic":    0.4,
            "context":     0.4,
            "dependency":  0.0,   # disabled by default
        })

    def combine(
        self,
        lexical: float,
        semantic: float,
        context: float,
        *,
        dependency: float = 0.0,
    ) -> float:
        """Return the weighted combination of per-dimension scores, clamped to [0, 1].

        The ``dependency`` parameter is keyword-only so existing call sites
        that pass only the three positional-style keyword arguments continue
        to work without modification.

        Clamping to ``[0, 1]`` (inherited from the base aggregator) matters
        here: without it, unnormalized weight dicts (weights summing to > 1)
        would produce total scores above 1.0, which would inflate
        ``NeighborSelector`` power weights quadratically and let any moderate
        match trivially pass ``ConfidenceEstimator``'s best-match threshold.
        """
        return super().combine({
            "lexical":    lexical,
            "semantic":   semantic,
            "context":    context,
            "dependency": dependency,
        })


class SimilarityEngine:
    """Compute a composite similarity score between two activities.

    Args:
        lexical_scorer: Scores token-overlap between cleaned descriptions.
        semantic_scorer: Scores WordNet / embedding-based text similarity.
        context_scorer: Scores structured metadata similarity.
        aggregator: Combines the per-dimension scores.
        dependency_scorer: Optional; scores schedule-structural similarity
            (Gap 4).  When ``None`` the dependency dimension contributes 0.
    """

    def __init__(
        self,
        lexical_scorer,
        semantic_scorer,
        context_scorer,
        aggregator,
        dependency_scorer=None,
    ):
        self.lexical_scorer = lexical_scorer
        self.semantic_scorer = semantic_scorer
        self.context_scorer = context_scorer
        self.aggregator = aggregator
        self.dependency_scorer = dependency_scorer

    def compare(self, query: ActivityCase, candidate: ActivityCase) -> SimilarityMatch:
        lexical    = self.lexical_scorer.score(query, candidate)
        semantic   = self.semantic_scorer.score(query, candidate)
        context    = self.context_scorer.score(query, candidate)
        dependency = (
            self.dependency_scorer.score(query, candidate)
            if self.dependency_scorer is not None
            else 0.0
        )
        total = self.aggregator.combine(
            lexical=lexical,
            semantic=semantic,
            context=context,
            dependency=dependency,
        )

        return SimilarityMatch(
            query_activity_id=query.activity_id,
            candidate_activity_id=candidate.activity_id,
            total_score=total,
            lexical_score=lexical,
            semantic_score=semantic,
            context_score=context,
            dependency_score=dependency,
            candidate_duration_hours=candidate.actual_duration_hours,
            explanation={
                "query_text": query.cleaned_description or query.raw_description,
                "candidate_text": candidate.cleaned_description or candidate.raw_description,
            },
        )
