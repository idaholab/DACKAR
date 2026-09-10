from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrossPatternConfig:
    temporal_compatibility_max_gap_days: float = 180.0
    temporal_compatibility_mode: str = "gate"   # "gate" | "formula"
    link_confidence_threshold: float = 0.25
    fm_alignment_score_threshold: float = 0.60
    signal_similarity_floor: float = 0.20
    stale_index_confidence_cap: float = 0.70
