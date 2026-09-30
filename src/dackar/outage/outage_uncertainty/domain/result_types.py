from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .duration import DurationDistribution


@dataclass
class SimilarityMatch:
    query_activity_id: str
    candidate_activity_id: str
    total_score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    context_score: float = 0.0
    dependency_score: float = 0.0   # Gap 4: schedule-structural similarity
    label_score: float = 0.0
    candidate_duration_hours: float | None = None
    # Populated by NeighborSelector: normalised relevance weight in [0, 1]
    # such that sum(relevance_weight) == 1.0 across the selected neighbor set.
    # Used by DurationEstimator for weighted distribution fitting (Phase 3).
    relevance_weight: float = 1.0
    explanation: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivityEstimate:
    activity_id: str
    estimated_distribution: DurationDistribution
    confidence_score: float
    # Three-tier evidence classification (Phase 3).
    # "high"   → many strong analogues; estimate is reliable
    # "medium" → moderate analogue coverage; use with caution
    # "low"    → weak or no analogues; expert review recommended
    confidence_tier: str = "low"
    support_count: int = 0
    matched_cases: list[SimilarityMatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Gap 2: epistemic vs aleatory uncertainty labelling
    # "epistemic" → weak data; action = better work package / SME review
    # "aleatory"  → well-characterised natural variability; action = schedule float
    # "mixed"     → both signals present; action = contingency + awareness
    # "unknown"   → not yet classified
    uncertainty_type: str = "unknown"
    recommended_action: str = ""


@dataclass
class SimulationResult:
    cp_times: list[float]
    cp_paths: list[list[str]]
    activity_criticality: dict[str, int]
