from __future__ import annotations

from dackar.similarity.stringSimilarity import token_jaccard

from dackar.outage.outage_uncertainty.domain.activity import ActivityCase


class LexicalSimilarityScorer:
    """Token-overlap (Jaccard) similarity between two activities' descriptions.

    A thin outage adapter: it extracts the cleaned (or raw) description from
    each :class:`ActivityCase` and defers to
    :func:`dackar.similarity.stringSimilarity.token_jaccard`.
    """

    def score(self, a: ActivityCase, b: ActivityCase) -> float:
        return token_jaccard(
            a.cleaned_description or a.raw_description,
            b.cleaned_description or b.raw_description,
        )
