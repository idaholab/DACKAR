from __future__ import annotations

from dataclasses import dataclass, field

from .duration import DurationDistribution


@dataclass
class ScheduleActivity:
    activity_id: str
    name: str
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    baseline_duration_hours: float = 0.0
    duration_distribution: DurationDistribution | None = None
    # Advance-notice period required before this activity can start once its
    # last predecessor has finished.  Equivalent to LOGOS
    # ``Activity.mobilization_lead_hours`` — vendor call-out time, crew
    # staging delay, radiation survey before entry, etc.
    # Applied in BOTH the forward pass (shifts ES right) and the backward pass
    # (tightens LF of each predecessor).  Defaults to 0.0.
    mobilization_lead_hours: float = 0.0
