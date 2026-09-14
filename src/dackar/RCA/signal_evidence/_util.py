"""Shared helpers for the signal-evidence package.

Centralizes the small utilities that were previously duplicated between
``builder.py`` and ``historian_adapter.py`` (MR#49 review), so the
UTC-normalization rule below lives in exactly one place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp, normalized to a UTC-aware ``datetime``.

    A naive input (no offset, e.g. ``"2024-01-01T00:00:00"``) is assumed to
    be UTC and stamped with ``timezone.utc``, while offset-aware inputs
    (``...+00:00`` / ``...Z``) are preserved. This guarantees every downstream
    comparison and subtraction operates on offset-aware datetimes, avoiding
    ``TypeError: can't compare offset-naive and offset-aware datetimes``.

    Args:
        value: An ISO-8601 timestamp string, or ``None``/empty.

    Returns:
        A UTC-aware ``datetime``, or ``None`` when *value* is empty or cannot
        be parsed.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clamp01(x: float) -> float:
    """Clamp *x* to the closed unit interval ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(x)))
