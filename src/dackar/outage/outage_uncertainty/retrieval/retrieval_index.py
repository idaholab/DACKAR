"""
Historical activity index with two-stage retrieval.

Retrieval pipeline
------------------
1. ``LexicalContextPrescorer``  (cheap — O(N × fast set/field ops))
   Narrows the full historical corpus to a candidate set of ~200 activities
   using token Jaccard on the cleaned description plus exact-match context
   fields with weight redistribution for missing (None) values.

2. ``SimilarityEngine.compare()``  (expensive — called only on the ~200
   candidates; may invoke WordNet via ``SemanticSimilarityScorer``)

3. ``NeighborSelector.select()``  (cheap — final soft top-k from the ~200)

Separating the cheap pre-filter from the expensive scorer ensures the
system scales to corpora of tens of thousands of activities without
running WordNet on every pair.
"""
from __future__ import annotations

import logging

from outage_uncertainty.domain.activity import ActivityCase

logger = logging.getLogger(__name__)

# Context fields considered by the prescorer (must be valid attributes of
# ActivityCase).  Order does not matter; each gets an equal share of the
# context half-weight after redistribution.
_DEFAULT_CONTEXT_FIELDS: list[str] = [
    "discipline",
    "task_family",
    "component_family",
    "plant_id",
    "outage_phase",
]


class LexicalContextPrescorer:
    """Cheap pre-filter that combines token Jaccard + exact-match context.

    Designed to rapidly eliminate clearly dissimilar activities before the
    expensive semantic scorer runs.  Intentionally simpler than
    ``ContextSimilarityScorer``:

    - No partial credit (exact match only)
    - Equal per-field weighting (not configurable per field)
    - Weight redistribution: ``None`` fields are skipped and the remaining
      field scores are re-normalised — unknown metadata does not penalise

    Score formula::

        score = text_weight × token_jaccard(a, b)
                + (1 − text_weight) × context_match(a, b)

    where ``context_match`` is the fraction of *available* (non-None) fields
    that match exactly, averaged over those fields only.

    Args:
        text_weight: Share of the score attributed to text similarity.
            The rest goes to context.  Default 0.5.
        context_fields: Activity attributes to compare.  Defaults to
            ``['discipline', 'task_family', 'component_family',
            'plant_id', 'outage_phase']``.
    """

    def __init__(
        self,
        text_weight: float = 0.5,
        context_fields: list[str] | None = None,
    ) -> None:
        if not 0.0 <= text_weight <= 1.0:
            raise ValueError(f"text_weight must be in [0, 1]; got {text_weight}")
        self.text_weight = text_weight
        self.context_weight = 1.0 - text_weight
        self.context_fields = context_fields or list(_DEFAULT_CONTEXT_FIELDS)

    def score(self, query: ActivityCase, candidate: ActivityCase) -> float:
        """Return a pre-filter score in [0, 1]."""
        text = self._token_jaccard(query, candidate)
        context = self._context_match(query, candidate)
        return self.text_weight * text + self.context_weight * context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _token_jaccard(a: ActivityCase, b: ActivityCase) -> float:
        a_tok = set((a.cleaned_description or a.raw_description or "").lower().split())
        b_tok = set((b.cleaned_description or b.raw_description or "").lower().split())
        if not a_tok and not b_tok:
            return 1.0
        if not a_tok or not b_tok:
            return 0.0
        return len(a_tok & b_tok) / len(a_tok | b_tok)

    def _context_match(self, a: ActivityCase, b: ActivityCase) -> float:
        """Fraction of available field pairs that are identical.

        Pairs where either side is ``None`` are skipped entirely so that
        missing taxonomy labels do not penalise the score.
        """
        hits = 0
        available = 0
        for field in self.context_fields:
            val_a = getattr(a, field, None)
            val_b = getattr(b, field, None)
            if val_a is None or val_b is None:
                continue
            available += 1
            if val_a == val_b:
                hits += 1
        return hits / available if available > 0 else 0.0


class HistoricalActivityIndex:
    """In-memory index of historical activities with prescored candidate search.

    ``build()`` stores all activities.  ``search()`` uses
    :class:`LexicalContextPrescorer` to cheaply rank the corpus and returns
    the top-``top_k`` candidate IDs for downstream full scoring.

    Args:
        prescorer: Pre-filter scorer instance.  Defaults to
            ``LexicalContextPrescorer()`` with equal text/context weighting.
    """

    def __init__(self, prescorer: LexicalContextPrescorer | None = None) -> None:
        self._activities: dict[str, ActivityCase] = {}
        self._prescorer = prescorer or LexicalContextPrescorer()

    def build(self, activities: list[ActivityCase]) -> None:
        """Index a list of historical activities (replaces any prior index)."""
        self._activities = {a.activity_id: a for a in activities}
        logger.debug("HistoricalActivityIndex: indexed %d activities", len(self._activities))

    def search(self, query: ActivityCase, top_k: int = 200) -> list[str]:
        """Return up to *top_k* candidate IDs ranked by pre-filter score.

        All activities in the index are scored cheaply; only the top-*top_k*
        proceed to the expensive full ``SimilarityEngine`` stage.
        """
        if not self._activities:
            return []

        scored: list[tuple[str, float]] = [
            (activity_id, self._prescorer.score(query, activity))
            for activity_id, activity in self._activities.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [activity_id for activity_id, _ in scored[:top_k]]

    def get(self, activity_id: str) -> ActivityCase:
        """Retrieve a single activity by ID."""
        return self._activities[activity_id]

    def __len__(self) -> int:
        return len(self._activities)
