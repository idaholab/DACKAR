from __future__ import annotations

from outage_uncertainty.domain.activity import ActivityCase


class LexicalSimilarityScorer:
    def score(self, a: ActivityCase, b: ActivityCase) -> float:
        a_tokens = set((a.cleaned_description or a.raw_description).lower().split())
        b_tokens = set((b.cleaned_description or b.raw_description).lower().split())
        if not a_tokens and not b_tokens:
            return 1.0
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
