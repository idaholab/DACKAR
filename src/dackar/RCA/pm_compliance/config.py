"""Configuration for the PM Compliance Verification module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PMComplianceConfig:
    """Tuning parameters for schedule vs. execution analysis and reporting."""

    look_back_window_days: int = 730
    """Window [event - N days, event] for CMMS PM / WO history (export / API phase)."""

    effectiveness_lookback_cycles: int = 3
    """How many past PM cycles to read for as-found / degradation trend."""

    pm_frequency_concern_ratio: float = 0.5
    """Flag PM frequency concern when pm_interval > ratio * mean_failure_interval (advisory)."""

    data_dir: Optional[Path] = field(default=None, compare=False, hash=False)
    """Path to the DACKAR data directory containing health-status keyword CSVs.

    When set, ``analyze_degradation`` and ``compute_pm_found_defect_rate`` use the
    curated vocabulary files (``health_status_keywords_negative/positive.csv``).
    When ``None``, the original hardcoded stem fallback is used.
    """

