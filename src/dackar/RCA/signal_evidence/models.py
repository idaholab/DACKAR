from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from orchestrators.temporal_relations import Interval


@dataclass
class AnomalyRecord:
    """A single flagged anomaly on one sensor over a time interval.

    Fields carry the sensor and (resolved) component ids, the anomaly window,
    a ``pattern`` label, a ``severity`` in ``[0, 1]``, the ``source`` that
    produced it (``"telemetry_summary"`` or ``"historian"``), and optional raw
    values / units.
    """

    sensor_id: str
    component_id: Optional[str]
    timestamp_start: datetime
    timestamp_end: datetime
    pattern: str
    severity: float
    source: str
    raw_value_start: Optional[float] = None
    raw_value_peak: Optional[float] = None
    units: Optional[str] = None

    def to_interval(self) -> Interval:
        """Return this record's ``[start, end]`` as a temporal ``Interval``."""
        return Interval(start=self.timestamp_start, end=self.timestamp_end)


@dataclass
class PropagationEdge:
    """A directed propagation edge between two anomalies (by list index).

    ``allen_rel`` captures the temporal relation between the two anomaly
    intervals and ``allen_score`` its raw event-calibrated relevance prior
    (kept for provenance). ``prop_score`` is the *propagation*-calibrated
    weight actually used to rank chains: it rewards a demonstrated causal lead
    at least as much as a co-temporal overlap and discounts co-temporal edges,
    so a clean ``precedes`` lead is never outranked by an ``overlaps`` edge the
    way the raw event score would (MR#49 review). ``edge_type`` is the KG
    relation (``containment`` / ``connectivity`` / ``mixed``), and
    ``onset_lag_h`` the onset lead in hours from source to target.
    """

    from_idx: int
    to_idx: int
    allen_rel: str
    allen_score: float
    prop_score: float
    edge_type: str
    onset_lag_h: float


@dataclass
class NodeTopology:
    """In/out degree and derived pattern (linear/divergence/convergence/hub/isolated) for one anomaly node."""

    anomaly_idx: int
    in_degree: int
    out_degree: int
    pattern_type: str


@dataclass
class ScoredChain:
    """A scored propagation path plus the factors that produced its ``path_score``.

    ``path_score`` is driven by ``mean_propagation_score`` (the lead-time-aware
    edge weight); ``mean_allen_score`` is retained alongside as the mean of the
    raw event-calibrated relation priors, for provenance.
    """

    chain_id: str
    path: list[int]
    path_score: float
    topology_alignment_factor: float
    lag_consistency_factor: float
    mean_allen_score: float
    mean_propagation_score: float
    hub_boost: float
    root_pattern_type: str
    nodes: list[dict]
