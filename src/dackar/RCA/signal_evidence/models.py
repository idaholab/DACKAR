from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from orchestrators.temporal_relations import Interval


@dataclass
class AnomalyRecord:
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
        return Interval(start=self.timestamp_start, end=self.timestamp_end)


@dataclass
class PropagationEdge:
    from_idx: int
    to_idx: int
    allen_rel: str
    allen_score: float
    edge_type: str
    onset_lag_h: float


@dataclass
class NodeTopology:
    anomaly_idx: int
    in_degree: int
    out_degree: int
    pattern_type: str


@dataclass
class ScoredChain:
    chain_id: str
    path: list[int]
    path_score: float
    topology_alignment_factor: float
    lag_consistency_factor: float
    mean_allen_score: float
    hub_boost: float
    root_pattern_type: str
    nodes: list[dict]
