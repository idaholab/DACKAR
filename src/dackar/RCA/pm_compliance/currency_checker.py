"""PMCurrencyChecker — PM interval vs. recurrence (advisory)."""

from __future__ import annotations

from typing import Optional

from .types import JsonDict


def frequency_concern(
    pm_interval_days: Optional[float],
    mean_failure_interval_days: Optional[float],
    ratio: float = 0.5,
) -> bool:
    """True when *pm_interval_days* is long relative to mean failure spacing."""
    if not pm_interval_days or not mean_failure_interval_days:
        return False
    if mean_failure_interval_days <= 0:
        return False
    return pm_interval_days > ratio * mean_failure_interval_days


def mean_interval_from_tskr(kg_context: Optional[JsonDict], fm_id: str) -> Optional[float]:
    """Best-effort: use recurrence or FM metadata; otherwise None."""
    if not kg_context or not fm_id:
        return None
    for fm in kg_context.get("failure_modes") or []:
        if str(fm.get("fm_id")) == str(fm_id):
            v = fm.get("mean_time_between_events_days")
            if v is not None:
                return float(v)
    return None
