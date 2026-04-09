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
