"""
Dependency pattern similarity scorer for outage activities.

Compares activities based on their structural position in the schedule
network (predecessor/successor topology) rather than content.  The PDF
allocates 20% of composite similarity weight to schedule neighbourhood /
dependency pattern similarity.

Two activities that both serve as merge points (multiple predecessors
converging before a downstream milestone) share execution characteristics —
coordination overhead, waiting time, resource contention — regardless of
whether their text descriptions match.

Structural features
-------------------
Topological role
    Derived from in-degree (``len(predecessor_ids)``) and out-degree
    (``len(successor_ids)``):

    =========  =========  ==========  ===================================
    in-degree  out-degree  role        typical outage example
    =========  =========  ==========  ===================================
    0          0           isolated    standalone, no network links
    0          >0          source      outage start / kickoff activity
    >0         0           sink        outage end / final completion
    1          1           chain       sequential work step
    >1         ≤1          merge       pre-milestone convergence point
    ≤1         >1          burst       work-package split / parallel start
    >1         >1          internal    complex node (fan-in and fan-out)
    =========  =========  ==========  ===================================

Degree similarity
    Exponential decay on ``|deg_a − deg_b|`` so activities with similar
    fan-in / fan-out patterns score higher:
    ``deg_sim(a, b) = exp(−|a − b| / degree_scale)``

Score formula
    ``score = role_weight × role_sim + (1 − role_weight) × degree_sim``
    where ``degree_sim = 0.5 × in_deg_sim + 0.5 × out_deg_sim``.

When both activities have empty adjacency lists they are both ``isolated``,
giving ``role_sim = 1.0`` and ``degree_sim = 1.0``.  Activities without
dependency information are therefore treated as structurally equivalent —
they are not penalised for missing data.
"""
from __future__ import annotations

import math

from outage_uncertainty.domain.activity import ActivityCase

# Partial credit for structurally related but non-identical roles.
# Both orderings are checked automatically; only one direction is listed here.
_PARTIAL_CREDIT: dict[tuple[str, str], float] = {
    ("source",   "chain"):    0.5,   # source is a chain without predecessors
    ("sink",     "chain"):    0.5,   # sink is a chain without successors
    ("isolated", "source"):   0.4,
    ("isolated", "sink"):     0.4,
    ("isolated", "chain"):    0.3,
    ("merge",    "chain"):    0.3,
    ("burst",    "chain"):    0.3,
    ("merge",    "internal"): 0.4,   # merge is internal with out-degree ≤ 1
    ("burst",    "internal"): 0.4,   # burst is internal with in-degree ≤ 1
    ("source",   "burst"):    0.3,   # both have multiple successors or near start
    ("sink",     "merge"):    0.3,   # both are convergence-style endpoints
}


def topological_role(activity: ActivityCase) -> str:
    """Classify an activity's structural role from its adjacency lists.

    Returns one of: ``"isolated"``, ``"source"``, ``"sink"``, ``"chain"``,
    ``"merge"``, ``"burst"``, ``"internal"``.
    """
    n_in  = len(activity.predecessor_ids)
    n_out = len(activity.successor_ids)

    if n_in == 0 and n_out == 0:
        return "isolated"
    if n_in == 0:
        return "source"
    if n_out == 0:
        return "sink"
    if n_in == 1 and n_out == 1:
        return "chain"
    if n_in > 1 and n_out <= 1:
        return "merge"
    if n_in <= 1 and n_out > 1:
        return "burst"
    return "internal"


class DependencyPatternScorer:
    """Score schedule-structural similarity between two activities.

    Args:
        role_weight: Weight given to topological role comparison.  The
            complementary weight goes to degree similarity.  Default 0.5.
        degree_scale: Exponential-decay scale for degree difference.  At
            scale = 2.0, a difference of 2 predecessors gives
            ``exp(−1) ≈ 0.37``.  Default 2.0.
    """

    def __init__(
        self,
        role_weight: float = 0.5,
        degree_scale: float = 2.0,
    ) -> None:
        self.role_weight = role_weight
        self.degree_scale = degree_scale

    def score(self, a: ActivityCase, b: ActivityCase) -> float:
        """Return a dependency-pattern similarity score in ``[0, 1]``."""
        role_sim = _role_similarity(topological_role(a), topological_role(b))
        degree_sim = 0.5 * (
            self._degree_sim(len(a.predecessor_ids), len(b.predecessor_ids))
            + self._degree_sim(len(a.successor_ids), len(b.successor_ids))
        )
        return self.role_weight * role_sim + (1.0 - self.role_weight) * degree_sim

    def _degree_sim(self, deg_a: int, deg_b: int) -> float:
        return math.exp(-abs(deg_a - deg_b) / self.degree_scale)


def _role_similarity(role_a: str, role_b: str) -> float:
    if role_a == role_b:
        return 1.0
    pc = _PARTIAL_CREDIT.get((role_a, role_b))
    if pc is None:
        pc = _PARTIAL_CREDIT.get((role_b, role_a))
    return float(pc) if pc is not None else 0.0
