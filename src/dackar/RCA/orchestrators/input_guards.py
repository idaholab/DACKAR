"""
Stage A cross-artifact checks (RCA SE Review §3.4–3.5, §6.1 A1 / A2).

Non-blocking warnings—runs continue; issues surface in ``run_context.input_guards``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

JsonDict = Dict[str, Any]


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def assert_output_dir_writable(root: Path) -> None:
    """A3: fail fast if artifact root cannot be created or written (SE review §6.1)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(str(root), os.W_OK):
        raise ValueError(
            f"output_dir is not writable: {root} — fix permissions before running the pipeline."
        )
    t = root / ".__dackar_w_ok"
    try:
        t.write_text("1", encoding="utf-8")
        try:
            t.unlink()
        except FileNotFoundError:
            pass
    except OSError as exc:
        raise ValueError(
            f"output_dir is not writable (probe write failed): {root} — {exc}"
        ) from exc


def build_input_guards(
    event: JsonDict,
    telemetry_summary: Optional[JsonDict],
    operational_context: Optional[JsonDict],
    pm_compliance: Optional[JsonDict],
    *,
    pm_staleness_threshold_days: int = 30,
    oc_staleness_threshold_hours: int = 48,
) -> JsonDict:
    """
    Return structured warnings for *temporal consistency* and *event scoping* (soft signals).

    Does not block the run; the orchestrator stores this on ``run_context`` for analysts / viz.
    """
    flags: List[str] = []
    notes: List[str] = []
    event_ts = _parse(
        (event or {}).get("timestamp_start")
        or (event or {}).get("timestamp")
    )
    eid = str((event or {}).get("event_id") or (event or {}).get("id") or "")

    if event_ts and telemetry_summary:
        win = (telemetry_summary.get("window") or {}) if isinstance(telemetry_summary, dict) else {}
        wend = _parse(win.get("end"))
        wstart = _parse(win.get("start"))
        if wend and event_ts and wend < event_ts:
            flags.append("telemetry_window_end_before_event")
            notes.append(
                "telemetry_summary.window.end is before event.timestamp_start — summary may not cover the event (SE §3.4 / A1)."
            )
        if wstart and event_ts and wstart > event_ts:
            flags.append("telemetry_window_starts_after_event")
            notes.append(
                "telemetry_summary.window.start is after event.timestamp_start — time window is inconsistent (SE §3.4 / A1)."
            )

    if event_ts and pm_compliance and isinstance(pm_compliance, dict):
        ad = _parse(pm_compliance.get("assessment_date"))
        if ad and event_ts:
            days = (event_ts.date() - ad.date()).days
            if days > pm_staleness_threshold_days:
                flags.append("pm_compliance_possibly_stale")
                notes.append(
                    f"pm_compliance.assessment_date is {days} d before the event (threshold {pm_staleness_threshold_days} d) (SE §3.4 / NM1)."
                )
            if days < -1:
                flags.append("pm_compliance_assessment_after_event")
                notes.append(
                    "pm_compliance.assessment_date is after the event time — check assessment_date semantics."
                )

    if event_ts and operational_context and isinstance(operational_context, dict):
        as_of = _parse(operational_context.get("as_of_timestamp"))
        if as_of and event_ts:
            delta = abs((event_ts - as_of).total_seconds() / 3600.0)
            if delta > float(oc_staleness_threshold_hours):
                flags.append("operational_context_as_of_may_be_stale")
                notes.append(
                    f"operational_context.as_of_timestamp is {delta:.1f} h from the event (threshold {oc_staleness_threshold_hours} h) (SE §3.4 / NM1)."
                )
        if eid:
            for alarm in (operational_context.get("recent_alarms") or []):
                if isinstance(alarm, dict):
                    oth = str(
                        alarm.get("related_event_id")
                        or alarm.get("correlated_event_id")
                        or alarm.get("parent_event_id")
                        or ""
                    )
                    if oth and oth != eid:
                        flags.append("possible_multi_event_overlap")
                        notes.append(
                            "recent_alarms references a different event id than the current analysis (SE §3.5 / NM2)."
                        )
                        break
                    txt = " ".join(
                        str(x)
                        for x in (alarm.get("message"), alarm.get("description"), alarm.get("text"))
                        if x
                    )
                    if eid in txt and any(
                        k in txt.lower() for k in ("related event", "linked event", "repeat", "prior event")
                    ):
                        flags.append("possible_multi_event_overlap")
                        notes.append(
                            "recent_alarms text may indicate multi-event / correlated sequence (SE §3.5 / NM2)."
                        )
                        break
                elif isinstance(alarm, str) and eid in alarm:
                    flags.append("possible_multi_event_overlap")
                    notes.append(
                        f"Event id {eid!r} appears in a recent_alarms string (SE §3.5 / NM2)."
                    )
                    break

    return {
        "flags": list(dict.fromkeys(flags)),
        "notes": list(dict.fromkeys(notes)),
    }
