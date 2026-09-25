"""Shared date/time helpers for the RCA package.

Consolidates the ``parse_dt`` / ``utcnow_iso`` helpers that were previously
defined only in ``orchestrators.causality_engine_v32`` and reached for by
importing that module directly.  Centralizing them here lets leaf packages
(e.g. ``pm_compliance``) stay independently importable rather than depending on
the orchestrator module that lands later in the pipeline (MR#56 review B1/A1).

``parse_dt`` normalizes naive inputs to UTC-aware datetimes so downstream
comparisons and ``min()`` / ``max()`` never mix offset-naive and offset-aware
values.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into a UTC-aware ``datetime``.

    A naive input (no offset, e.g. ``"2024-01-01T00:00:00"``) is assumed to be
    UTC and stamped with ``timezone.utc``; offset-aware inputs (``...+00:00`` /
    ``...Z``) are preserved.  Returns ``None`` when *value* is empty or cannot
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
