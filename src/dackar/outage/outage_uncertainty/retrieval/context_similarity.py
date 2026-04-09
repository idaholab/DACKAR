"""
Context similarity scorer for outage activities.

Scores how similar two activities are based on structured metadata — the
"case" dimensions beyond the free-text description.  This implements the
case-based similarity philosophy from the PDF: task descriptions alone are
insufficient; discipline, component family, plant context, and execution
mode all carry independent similarity signal.

Key design choices
------------------
Weight redistribution for missing fields
    When a field is ``None`` on either activity, it is skipped and the
    remaining applicable weights are re-normalised to sum to 1.0.  This
    means unknown taxonomy labels do not penalise the score — the result is
    "similarity based on available evidence."

    Contrast with the original stub where ``_same(None, None)`` → 0.0,
    which incorrectly treated "both unlabelled" as "confirmed different."

Partial credit for related values
    Exact string equality gives 1.0.  A configurable ``partial_credit`` dict
    allows domain-specific partial scores for values that are related but not
    identical (e.g. ``("task_family", "valve_inspection", "valve_replacement")
    → 0.3``).  Both orderings of the pair are checked automatically.

    This is intentionally left sparse by default — the Phase 4 taxonomy work
    will populate it with outage-domain knowledge.
"""
from __future__ import annotations

import logging

from outage_uncertainty.domain.activity import ActivityCase

logger = logging.getLogger(__name__)

# Default weights.  Must be non-negative; they are *not* required to sum to
# 1.0 because redistribution re-normalises on the fly.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "discipline": 0.25,          # mechanical / I&C / electrical / civil …
    "task_family": 0.30,         # inspection / replacement / calibration …
    "component_family": 0.20,    # valve / transmitter / pump …
    "plant_id": 0.10,            # same plant → stronger analogy
    "is_emergent": 0.10,         # emergent work has structurally different durations
    "outage_phase": 0.05,        # pre-outage / forced / planned maintenance
    # Gap 3: execution mode flags — strong duration-variance predictors
    "has_rp_hold": 0.08,         # RP hold points add waiting time regardless of task type
    "requires_scaffold": 0.07,   # scaffold erection/removal drives setup time
    "has_clearance": 0.05,       # electrical/mechanical clearances add coordination overhead
    "is_vendor_supported": 0.05, # vendor mobilisation and schedule dependency
}

# Partial-credit pairs shipped with the package as a starter set.
# Format: (field_name, value_a, value_b) → score in (0, 1).
# Both orderings are checked, so only one direction is needed here.
_DEFAULT_PARTIAL_CREDIT: dict[tuple, float] = {
    # Within task_family: same broad action, different scope
    ("task_family", "inspection", "surveillance"):        0.4,
    ("task_family", "replacement", "refurbishment"):      0.4,
    ("task_family", "calibration", "testing"):            0.3,
    ("task_family", "disassembly", "replacement"):        0.3,
    ("task_family", "restoration", "replacement"):        0.25,
    # Within component_family: same equipment class, different sub-type
    ("component_family", "valve", "actuator"):            0.35,
    ("component_family", "transmitter", "sensor"):        0.35,
    ("component_family", "pump", "motor"):                0.30,
    ("component_family", "breaker", "switchgear"):        0.30,
    ("component_family", "heat_exchanger", "condenser"):  0.25,
}


class ContextSimilarityScorer:
    """Score contextual (metadata-level) similarity between two activities.

    Args:
        weights: Per-field weight dict.  Missing-field redistribution means
            the absolute values matter only relative to each other, not their
            sum.  Defaults to :data:`_DEFAULT_WEIGHTS`.
        partial_credit: ``{(field, val_a, val_b): score}`` dict for related
            but non-identical values.  Merged on top of the built-in starter
            set unless ``replace_defaults=True``.
        replace_defaults: When ``True``, the supplied ``partial_credit`` dict
            completely replaces the built-in one instead of extending it.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        partial_credit: dict[tuple, float] | None = None,
        replace_defaults: bool = False,
    ) -> None:
        self.weights: dict[str, float] = weights or dict(_DEFAULT_WEIGHTS)

        if replace_defaults:
            self._partial: dict[tuple, float] = dict(partial_credit or {})
        else:
            self._partial = dict(_DEFAULT_PARTIAL_CREDIT)
            if partial_credit:
                self._partial.update(partial_credit)

    # ------------------------------------------------------------------
    # Public interface expected by SimilarityEngine
    # ------------------------------------------------------------------

    def score(self, a: ActivityCase, b: ActivityCase) -> float:
        """Return a context similarity score in [0, 1].

        Only fields where *both* activities carry a non-None value contribute
        to the score.  The contributing weights are re-normalised so the
        result always reflects the full available evidence.
        """
        applicable: dict[str, float] = {}   # field → applicable weight
        raw: dict[str, float] = {}           # field → raw similarity

        for field, weight in self.weights.items():
            val_a = getattr(a, field, None)
            val_b = getattr(b, field, None)

            if val_a is None or val_b is None:
                # One or both sides unknown: skip, redistribute weight
                continue

            applicable[field] = weight
            raw[field] = self._compare(field, val_a, val_b)

        if not applicable:
            # No field has data on both sides — can't say anything
            return 0.0

        total_weight = sum(applicable.values())
        return sum(applicable[f] / total_weight * raw[f] for f in applicable)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compare(self, field: str, val_a, val_b) -> float:
        """Return similarity for a single field pair.

        Returns 1.0 for exact match, a partial-credit value for related pairs,
        or 0.0 for unrelated values.
        """
        if val_a == val_b:
            return 1.0

        # Normalise strings for comparison (strip, lowercase) to avoid
        # penalising trivial formatting differences
        norm_a = val_a.strip().lower() if isinstance(val_a, str) else val_a
        norm_b = val_b.strip().lower() if isinstance(val_b, str) else val_b

        if norm_a == norm_b:
            return 1.0

        # Look up partial credit (try both orderings)
        pc = self._partial.get((field, norm_a, norm_b))
        if pc is None:
            pc = self._partial.get((field, norm_b, norm_a))
        return float(pc) if pc is not None else 0.0
