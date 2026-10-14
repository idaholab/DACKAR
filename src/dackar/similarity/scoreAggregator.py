# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Weighted combination of per-dimension similarity scores.

A small, domain-agnostic helper: given a ``{dimension: weight}`` mapping, it
combines a ``{dimension: score}`` mapping into a single scalar, clamped to
``[0, 1]``.  Dimensions are arbitrary string labels, so the same aggregator
serves any multi-dimensional scoring scheme.

Only the standard library is required.
"""
from __future__ import annotations

from typing import Mapping


class WeightedScoreAggregator:
    """Combine per-dimension scores into one total via a weights mapping.

    Args:
        weights: ``{dimension: weight}``.  Any dimension absent from a given
            ``scores`` mapping contributes 0; any score whose dimension is
            absent from ``weights`` is ignored.

    The output of :meth:`combine` is clamped to ``[0, 1]`` regardless of the
    weight configuration.  Without clamping, un-normalized weights (summing to
    more than 1) would yield totals above 1.0.
    """

    def __init__(self, weights: Mapping[str, float]):
        self.weights = dict(weights)

    def combine(self, scores: Mapping[str, float]) -> float:
        """Return the weighted sum of *scores*, clamped to ``[0, 1]``.

        Only dimensions present in ``weights`` contribute; a weighted
        dimension missing from ``scores`` contributes 0.
        """
        raw = sum(
            weight * scores.get(dimension, 0.0)
            for dimension, weight in self.weights.items()
        )
        return max(0.0, min(1.0, raw))
