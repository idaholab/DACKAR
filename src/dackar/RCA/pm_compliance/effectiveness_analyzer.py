"""PMEffectivenessAnalyzer — as-found trend signals (advisory, data-quality limited)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from orchestrators.causality_engine_v32 import parse_dt
from .types import JsonDict
from .vocabulary_loader import PMVocabularyLoader, matches_any


def analyze_degradation(
    as_found_texts: List[str],
    data_dir: Optional[Path] = None,
) -> str:
    """Return *degradation_trend* label based on as-found text.

    When *data_dir* points to the DACKAR data directory, uses the curated
    ``health_status_keywords_negative/positive.csv`` vocabulary.  Falls back to
    hardcoded stems when *data_dir* is ``None`` or the files are absent.

    Classification logic (in priority order):
    1. No non-empty texts → ``"unknown"``
    2. Improving terms match AND no degrading terms match → ``"improving"``
    3. Degrading terms match → ``"degrading"``
    4. Otherwise → ``"stable"``
    """
    if not as_found_texts or not any(t and str(t).strip() for t in as_found_texts):
        return "unknown"

    blob = " ".join(str(t).lower() for t in as_found_texts if t)
    vocab = PMVocabularyLoader.load(data_dir)

    has_degrading = matches_any(blob, vocab["degrading"])
    has_improving = matches_any(blob, vocab["improving"])

    if has_improving and not has_degrading:
        return "improving"
    if has_degrading:
        return "degrading"
    return "stable"


def compute_pm_found_defect_rate(
    rows: List[JsonDict],
    data_dir: Optional[Path] = None,
) -> Optional[float]:
    """Return fraction of PM executions with as-found data that recorded a defect.

    Iterates *all* export rows (not cycle-limited) to give a statistical signal
    over the full lookback history.  Returns ``None`` when no rows carry as-found
    data (avoids a misleading 0.0 rate).

    A row is counted as a defect when its as-found text classifies as
    ``"degrading"`` per ``analyze_degradation``.
    """
    total_with_asf = 0
    defect_count = 0
    for row in rows:
        asf_text: Optional[str] = None
        for key in ("as_found_last", "as_found_condition", "details"):
            val = row.get(key)
            if val is not None and str(val).strip():
                asf_text = str(val)
                break
        if asf_text is None:
            continue
        total_with_asf += 1
        if analyze_degradation([asf_text], data_dir=data_dir) == "degrading":
            defect_count += 1

    if total_with_asf == 0:
        return None
    return round(defect_count / total_with_asf, 6)


def collect_as_found_from_checks(checks: List[JsonDict]) -> List[str]:
    """Use ``details`` (as-found) from checks when available."""
    out: List[str] = []
    for c in checks:
        d = c.get("details")
        if d:
            out.append(str(d))
    return out


def collect_as_found_from_rows(rows: List[JsonDict], max_cycles: Optional[int] = None) -> List[str]:
    """Prefer structured as-found columns from export rows.

    When *max_cycles* is set, only the most-recent N rows (by ``completed_date``)
    are considered — honouring ``PMComplianceConfig.effectiveness_lookback_cycles``.
    """
    if max_cycles is not None and max_cycles > 0:
        def _completed_key(r: JsonDict) -> str:
            dt = parse_dt(r.get("completed_date") or r.get("last_pm_date") or "")
            return dt.isoformat() if dt else ""
        rows = sorted(rows, key=_completed_key, reverse=True)[:max_cycles]

    out: List[str] = []
    for row in rows:
        for key in ("as_found_last", "as_found_condition", "details"):
            val = row.get(key)
            if val is not None and str(val).strip():
                out.append(str(val))
                break
    return out
