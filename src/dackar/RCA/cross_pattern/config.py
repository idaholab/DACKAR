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
    # When True, the linker ALSO links a doc that concerns the same asset as the
    # episode but a DIFFERENT failure mode than the candidate, so the conflict
    # path in classify_support_posture (and the conflict wording/flags in
    # summary.py) can surface it for analyst review. Default False preserves the
    # FM-only linking behavior — mirrors the "present but not yet wired" posture
    # of the wider cross-pattern layer (cf. orchestrator enable_cross_pattern_linkage).
    enable_conflict_detection: bool = False
