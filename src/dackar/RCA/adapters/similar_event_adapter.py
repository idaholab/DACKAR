"""
similar_event_adapter.py — Protocol definition for Step 2d similar-event adapters.

The orchestrator accepts any object satisfying SimilarEventAdapter to query
fleet and industry OE databases.  The concrete production implementation is
LLMOEAdapter (llm_oe_adapter.py).  Tests may supply any mock that satisfies
the protocol.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Literal, Optional

# runtime_checkable requires Python ≥ 3.8
if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable

JsonDict = Dict[str, object]

TIER_CONFIDENCE_MULTIPLIERS: Dict[str, float] = {
    "plant": 1.00,
    "fleet": 0.80,
    "industry": 0.60,
}


@runtime_checkable
class SimilarEventAdapter(Protocol):
    """Pluggable interface for fleet and industry OE similar-event queries.

    The orchestrator calls ``query()`` once for fleet and once for industry
    when an adapter is injected.  Each call must return a list of dicts
    compatible with the ``similar_event_list.json`` event item schema.

    Contract:
    - Never raise on network/timeout/parse errors; return ``[]`` instead.
    - Set ``degraded`` to ``True`` on the adapter instance when a call fails
      so the orchestrator can record the degraded tier.
    - ``confidence_weight`` on returned records should be the *raw* match
      score (0–1) before tier discount; the orchestrator applies the
      multiplier defined in ``TIER_CONFIDENCE_MULTIPLIERS``.
    """

    #: Set to True by the adapter implementation if the last query call failed.
    degraded: bool

    def query(
        self,
        *,
        level: Literal["fleet", "industry"],
        asset_id: Optional[str],
        component_ids: List[str],
        failure_mode_ids: List[str],
        event_type: Optional[str] = None,
        actuation_type: Optional[str] = None,
        max_results: int = 5,
        timeout_seconds: float = 10.0,
    ) -> List[JsonDict]:
        """Return a list of similar-event dicts for the given tier."""
        ...
