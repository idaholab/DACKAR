from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

# ---------------------------------------------------------------------------
# The 6 RCA-relevant Allen relations (A = anomaly/maintenance interval,
# B = event interval).
#
# Semantics (A relative to B):
#   PRECEDES   — A ends before B starts; lag = A-onset to B-onset
#   OVERLAPS   — A started before B, still active at B onset (strongest causal signal)
#   CONTAINS   — A encompasses B; long-running latent degradation
#   DURING     — A started at or after B onset; likely consequential, not a cause
#   FOLLOWS    — A starts after B ends; temporal contradiction
# ---------------------------------------------------------------------------

PRECEDES  = "precedes"
OVERLAPS  = "overlaps"
CONTAINS  = "contains"
DURING    = "during"
FOLLOWS   = "follows"

# Causal priority order used by consumers to pick the dominant relation.
# Higher priority = stronger causal candidacy.
CAUSAL_PRIORITY: Tuple[str, ...] = (OVERLAPS, CONTAINS, PRECEDES, DURING, FOLLOWS)

# Base RCA relevance scores.  Refined downstream by latency alignment
# and severity weighting; these express the prior relevance of each relation.
RELATION_SCORE: dict[str, float] = {
    OVERLAPS: 0.90,   # degradation active at event onset — very strong
    CONTAINS: 0.85,   # long-running latent condition
    PRECEDES: 0.75,   # classic causal lead-time
    DURING:   0.30,   # anomaly appeared after event onset — likely a symptom
    FOLLOWS:  0.10,   # anomaly after event resolution — contradiction
}


@dataclass(frozen=True)
class Interval:
    """A closed time interval [start, end].  For point events set end == start."""
    start: datetime
    end: datetime


def allen_relation(
    a: Interval,
    b: Interval,
    epsilon_hours: float = 0.5,
    interval_type: str = "closed",
) -> Tuple[str, float]:
    """Classify the temporal relation of interval A relative to reference interval B.

    Returns ``(relation_name, base_rca_relevance_score)``.

    ``b`` is the event interval; ``a`` is an anomaly or maintenance window.
    ``epsilon_hours`` absorbs timestamp noise and near-simultaneous boundary
    cases — boundaries within epsilon are treated as touching.
    ``interval_type`` controls whether anomaly endpoints are interpreted as
    closed/open when evaluating boundary-touching cases.

    Decision logic (evaluated in order):

    1. FOLLOWS   — a starts meaningfully after b ends
    2. PRECEDES  — a ends meaningfully before b starts
    3. CONTAINS  — a started before b AND ends after b  (encompasses event)
    4. OVERLAPS  — a started before b, ends within b    (degradation active at onset)
    5. DURING    — everything else: a started at or after b onset, including
                   "started inside b and ended after b" (consequential anomaly)
    """
    eps = epsilon_hours * 3600.0
    a_s_raw = a.start.timestamp()
    a_e_raw = a.end.timestamp()
    b_s = b.start.timestamp()
    b_e = b.end.timestamp()

    norm_interval_type = str(interval_type or "closed").strip().lower()
    if norm_interval_type not in {"closed", "open", "half_open_start", "half_open_end"}:
        norm_interval_type = "closed"

    # Use a tiny endpoint shift to model open boundaries. This keeps relation
    # semantics deterministic while preserving existing epsilon-based tolerance.
    endpoint_shift_s = 1e-6
    start_shift = endpoint_shift_s if norm_interval_type in {"open", "half_open_start"} else 0.0
    end_shift = endpoint_shift_s if norm_interval_type in {"open", "half_open_end"} else 0.0
    a_s = a_s_raw + start_shift
    a_e = a_e_raw - end_shift
    if a_e < a_s:
        midpoint = (a_s_raw + a_e_raw) / 2.0
        a_s = midpoint
        a_e = midpoint

    if a_s > b_e + eps:
        rel = FOLLOWS
    elif a_e < b_s - eps:
        rel = PRECEDES
    elif a_s < b_s - eps and a_e > b_e + eps:
        rel = CONTAINS
    elif a_s < b_s - eps:          # and a_e <= b_e + eps
        rel = OVERLAPS
    else:                           # a_s >= b_s - eps
        rel = DURING

    return rel, RELATION_SCORE[rel]


def onset_lag_hours(a: Interval, b: Interval) -> float:
    """Signed lag in hours from A onset to B onset (b.start − a.start).

    Positive  → A predates B onset (causal candidate).
    Negative  → A postdates B onset (symptom candidate).
    """
    return (b.start - a.start).total_seconds() / 3600.0
