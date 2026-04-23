"""PMEffectivenessAnalyzer — as-found trend signals (advisory, data-quality limited)."""

from __future__ import annotations

from typing import List

from .types import JsonDict


def analyze_degradation(as_found_texts: List[str]) -> str:
    """Return *degradation_trend* label when controlled vocabulary is absent.

    Heuristic: search for 'degrad', 'worse', 'increas' vs 'improv', 'stable'.
    """
    if not as_found_texts or not any(t and str(t).strip() for t in as_found_texts):
        return "unknown"
    blob = " ".join(str(t).lower() for t in as_found_texts if t)
    if any(s in blob for s in ("improv", "normal", "no defect", "acceptable")) and not any(
        s in blob for s in ("degrad", "worse", "wear", "increas")
    ):
        return "improving"
    if any(s in blob for s in ("degrad", "worse", "increas", "severity")):
        return "degrading"
    return "stable"


def collect_as_found_from_checks(checks: List[JsonDict]) -> List[str]:
    """Use ``details`` (as-found) from checks when available."""
    out: List[str] = []
    for c in checks:
        d = c.get("details")
        if d:
            out.append(str(d))
    return out


def collect_as_found_from_rows(rows: List[JsonDict]) -> List[str]:
    """Prefer structured as-found columns from export rows."""
    out: List[str] = []
    for row in rows:
        for key in ("as_found_last", "as_found_condition", "details"):
            val = row.get(key)
            if val is not None and str(val).strip():
                out.append(str(val))
                break
    return out
