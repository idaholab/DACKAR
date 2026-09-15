"""PMComplianceAggregator — assemble the ``pm_compliance`` artifact (JSON-serializable dict)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from orchestrators.causality_engine_v32 import parse_dt, utcnow_iso
from .config import PMComplianceConfig
from .currency_checker import frequency_concern, mean_interval_from_tskr
from .effectiveness_analyzer import (
    analyze_degradation,
    collect_as_found_from_rows,
    compute_pm_found_defect_rate,
)
from .execution_verifier import PMExecutionVerifier
from .scope_analyzer import analyze_scope
from .types import JsonDict
from .schedule_loader import PMScheduleLoader


def _window_for_event(event_ts: str, lookback_days: int) -> tuple[str, str]:
    end = parse_dt(event_ts)
    if not end:
        end = datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=lookback_days)
    return start.isoformat(), end.isoformat()


def _summary_metrics(checks: List[JsonDict]) -> Dict[str, Any]:
    total = len(checks)
    passed = sum(1 for c in checks if c.get("status") == "pass")
    failed = sum(1 for c in checks if c.get("status") == "fail")
    unknown = sum(1 for c in checks if c.get("status") == "unknown")
    overdue_count = sum(1 for c in checks if (c.get("overdue_by_days") or 0) > 0.0 and c.get("status") == "fail")
    compliance_rate = (passed / total) if total else 1.0
    m: Dict[str, Any] = {
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "overdue_count": overdue_count,
        "compliance_rate": round(compliance_rate, 6),
    }
    comp_dates: List[datetime] = []
    for c in checks:
        d = parse_dt(c.get("completed_date"))
        if d:
            comp_dates.append(d)
    if comp_dates:
        m["last_pm_date"] = max(comp_dates).isoformat()
    nexts: List[datetime] = []
    for c in checks:
        d = parse_dt(c.get("scheduled_date"))
        if d:
            nexts.append(d)
    if nexts:
        m["next_pm_date"] = min(nexts).isoformat()
    return m


def _rollup_risk(
    primary_fm_id: Optional[str],
    all_gap: Set[str],
    has_overdue: bool,
    has_fail: bool,
) -> Tuple[str, str, bool]:
    """Architecture §3.6 — *maintenance_induced_risk* and *overall_compliance* roll-ups.

    When *primary_fm_id* is supplied (e.g. after synthesis), risk matches the spec:
    high = primary in scope gap and PM was overdue; medium = primary gap or overdue; else low.
    Without *primary_fm_id*, risk falls back to gap + overdue heuristics.
    """
    primary_in_gap = bool(primary_fm_id) and (primary_fm_id in all_gap)
    has_scope_primary = primary_in_gap

    if primary_fm_id:
        if primary_in_gap and has_overdue:
            risk = "high"
        elif primary_in_gap or has_overdue:
            risk = "medium"
        else:
            risk = "low"
    else:
        if all_gap and has_overdue and has_fail:
            risk = "high"
        elif has_overdue or bool(all_gap):
            risk = "medium"
        else:
            risk = "low"

    if not has_fail and not all_gap and not has_overdue:
        overall: str = "compliant"
    elif primary_fm_id and primary_in_gap:
        overall = "non_compliant"
    elif has_fail or all_gap or has_overdue:
        overall = "partial"
    else:
        overall = "compliant"

    return overall, risk, has_scope_primary


def _compliance_status_md(
    overdue_days: float,
    st: str,
    missed_cycles: int,
    raw_compliance_status: Optional[str] = None,
) -> str:
    """Narrative labels for ``pm_tasks[].compliance_status`` (architecture §5).

    ``"unknown"`` check status (no schedule dates in the export) returns
    ``"undetermined"`` so analysts can distinguish it from a genuine pass.
    The governance-scoring path (``checks[].status``) is unaffected — it still
    carries ``"unknown"``; only the component narrative label changes.
    """
    raw = str(raw_compliance_status or "").strip().lower()
    if raw in {"not_applicable", "n_a"}:
        return "not_applicable"
    if st == "pass":
        return "compliant"
    if st == "unknown":
        return "undetermined"  # §1.4 fix: was "compliant", which was misleading
    if st == "fail":
        if missed_cycles > 0:
            return "missed"
        if overdue_days > 0:
            return "overdue"
        return "missed"
    return "compliant"


def _build_pm_tasks_per_component(
    raw_rows: List[JsonDict],
    checks: List[JsonDict],
    check_to_coverage_type: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Architecture §5 — ``components[].pm_tasks`` (narrative) alongside pipeline ``checks``.

    *check_to_coverage_type* (from ``analyze_scope``) maps check_id →
    ``"preventive"`` | ``"detective"``.  When present, it overrides the export
    row's ``coverage_type`` field, which defaults to ``"none"``.
    """
    by: Dict[str, List[Dict[str, Any]]] = {}
    check_by_id = {c.get("check_id"): c for c in checks if c.get("check_id")}
    cov_map = check_to_coverage_type or {}
    for r in raw_rows:
        cid = str(r.get("component_id") or "_asset")
        task_code = str(r.get("task_code") or r.get("check_id") or "")
        ck = check_by_id.get(r.get("check_id") or r.get("task_code") or "")
        overdue = float((ck or {}).get("overdue_by_days") or 0.0)
        st = (ck or {}).get("status") or "unknown"
        mcy = int(r.get("missed_cycles") or 0)
        # §2.3: KG-derived coverage_type takes precedence over export row value
        coverage_type = cov_map.get(task_code) or r.get("coverage_type") or "none"
        task: Dict[str, Any] = {
            "task_code": task_code,
            "description": str(r.get("description") or r.get("task_description") or ""),
            "frequency_days": r.get("frequency_days"),
            "last_pm_date": r.get("last_pm_date") or r.get("completed_date"),
            "next_due_date": r.get("next_due_date") or r.get("next_due"),
            "overdue_days": overdue,
            "compliance_status": _compliance_status_md(
                overdue,
                st,
                mcy,
                raw_compliance_status=r.get("compliance_status"),
            ),
            "missed_cycles": mcy,
            "last_as_found": r.get("as_found_last") or r.get("as_found_condition"),
            "coverage_type": coverage_type,
        }
        by.setdefault(cid, []).append(task)
    return by


def _apply_frequency_flags(
    comp_views: List[Dict[str, Any]],
    raw_rows: List[JsonDict],
    kg_context: Optional[JsonDict],
    ratio: float,
) -> None:
    """Set ``pm_frequency_concern`` on each component (architecture §3.5)."""
    for view in comp_views:
        cid = str(view.get("component_id") or "_asset")
        concern = False
        for r in raw_rows:
            r_cid = str(r.get("component_id") or "_asset")
            if r_cid != cid:
                continue
            f_days = r.get("frequency_days")
            if f_days is None:
                continue
            fm_list = r.get("applicable_fm_ids") or []
            for fm_id in fm_list:
                mean_inter = mean_interval_from_tskr(kg_context, str(fm_id)) if kg_context else None
                if frequency_concern(float(f_days), mean_inter, ratio=ratio):
                    concern = True
                    break
            if concern:
                break
        view["pm_frequency_concern"] = concern


def build_pm_compliance(
    event: JsonDict,
    kg_context: Optional[JsonDict] = None,
    export_rows: Optional[Sequence[JsonDict]] = None,
    config: Optional[PMComplianceConfig] = None,
    primary_fm_id: Optional[str] = None,
) -> JsonDict:
    """Assemble a ``pm_compliance`` object validated by ``schemas/pm_compliance.json``.

    Parameters
    ----------
    event
        Must include ``asset_id`` and ``timestamp_start`` (ISO). ``event_id`` is optional.
    kg_context
        Optional; supplies failure modes for scope analysis.
    export_rows
        Pre-parsed PM tasks from a CMMS export or adapter (see *PMExportTaskRow*).
    config
        Lookback and thresholds.
    primary_fm_id
        Optional failure mode id (e.g. from the eventual primary hypothesis) to evaluate
        ``has_scope_gaps_for_primary_fm`` and *maintenance_induced_risk* per
        ``PM_Compliance_Module_Architecture.md`` §3.5–3.6.
    """
    cfg = config or PMComplianceConfig()
    asset = str(event.get("asset_id") or "")
    if not asset:
        raise ValueError("event.json must include asset_id for PM compliance build")

    event_ts = str(event.get("timestamp_start") or event.get("timestamp") or "")
    comp_ids: Optional[List[str]] = None
    if kg_context:
        comp_ids = [str(c.get("component_id")) for c in (kg_context.get("components") or []) if c.get("component_id")]

    loader = PMScheduleLoader(asset, component_ids=comp_ids)
    raw_rows, loader_notes = loader.load_from_export_rows_with_notes(export_rows or ())
    extra_notes: List[str] = []
    for r in raw_rows:
        ft = (r.get("frequency_type") or "").lower()
        if ft in ("operating_hours", "operating", "hours"):
            if r.get("operating_hours_at_event") is None and r.get("compliance_status") not in (
                "not_applicable",
                "n_a",
            ):
                extra_notes.append(
                    f"Task {r.get('check_id')!r}: operating-hour-based PM without runtime hours — "
                    f"using calendar fallback per architecture §3.1"
                )

    verifier = PMExecutionVerifier(event_timestamp_iso=event_ts)
    checks, dq_notes = verifier.verify_rows(raw_rows)
    dq_notes = list(loader_notes) + list(dq_notes) + extra_notes

    comp_views, fmea_kg, covered_fms, all_fms, check_to_coverage_type = analyze_scope(kg_context, checks)
    if raw_rows and kg_context and not fmea_kg:
        if any(c.get("applicable_fm_ids") for c in checks):
            dq_notes.append(
                "PM-to-FM coverage from export `applicable_fm_ids` only; no KG FMEA/PM task linkage (advisory, §3.3)"
            )

    asf = collect_as_found_from_rows(raw_rows, max_cycles=cfg.effectiveness_lookback_cycles)
    dqtrend = analyze_degradation(asf, data_dir=cfg.data_dir) if asf else "unknown"
    for view in comp_views:
        view.setdefault("degradation_trend", dqtrend)
        cid = view.get("component_id")
        view["pm_overdue_at_failure"] = any(
            (c.get("overdue_by_days") or 0) > 0
            and c.get("status") == "fail"
            and (c.get("component_id") or "_asset") == cid
            for c in checks
        )
        view.setdefault("pm_frequency_concern", False)

    pm_by_comp = _build_pm_tasks_per_component(raw_rows, checks, check_to_coverage_type=check_to_coverage_type)
    for v in comp_views:
        key = str(v.get("component_id") or "_asset")
        v["pm_tasks"] = pm_by_comp.get(key, [])

    if kg_context:
        _apply_frequency_flags(comp_views, raw_rows, kg_context, cfg.pm_frequency_concern_ratio)
    w_start, w_end = _window_for_event(event_ts, cfg.look_back_window_days)

    m = _summary_metrics(checks)
    has_overdue = any((c.get("overdue_by_days") or 0) > 0.0 for c in checks)
    has_fail = any(c.get("status") == "fail" for c in checks)
    all_gap: Set[str] = set()
    if all_fms and (fmea_kg or covered_fms):
        all_gap = set(all_fms) - set(covered_fms)

    overall, risk, has_scope_gaps = _rollup_risk(
        primary_fm_id, all_gap, has_overdue, has_fail
    )
    m["overall_compliance"] = overall
    m["maintenance_induced_risk"] = risk
    m["has_scope_gaps_for_primary_fm"] = has_scope_gaps
    defect_rate = compute_pm_found_defect_rate(raw_rows, data_dir=cfg.data_dir)
    if defect_rate is not None:
        m["pm_found_defect_rate"] = defect_rate
    dq_conf = "high" if not dq_notes and checks else "medium" if checks else "low"
    if any(c.get("status") == "unknown" for c in checks):
        dq_conf = "medium" if dq_conf == "high" else "low"
    if not asf and checks:
        dq_conf = "medium" if dq_conf == "high" else dq_conf
    m["data_quality_confidence"] = dq_conf

    out: JsonDict = {
        "asset_id": asset,
        "window": {"start": w_start, "end": w_end},
        "checks": checks,
        "summary": m,
    }

    eid = event.get("event_id")
    if eid is not None:
        out["event_id"] = str(eid)
    # assessment_date records when this artifact was built, not the event time.
    # The event reference time is already captured in window.end.
    out["assessment_date"] = utcnow_iso()
    out["look_back_window_days"] = cfg.look_back_window_days
    out["fmea_pm_linkage_available"] = bool(fmea_kg)
    if primary_fm_id and not bool(fmea_kg):
        dq_notes.append(
            f"primary_fm_id '{primary_fm_id}' provided but KG PM↔FM linkage absent — "
            f"scope gap for this FM is not evaluable; maintenance_induced_risk may be underestimated (architecture §3.3)"
        )
    out["data_quality_notes"] = dq_notes
    if comp_views:
        out["components"] = comp_views

    overdues: List[JsonDict] = []
    for c in checks:
        o = c.get("overdue_by_days") or 0.0
        if c.get("status") == "fail" and o > 0.0:
            overdues.append(
                {
                    "check_id": c.get("check_id"),
                    "check_type": c.get("check_type", "other"),
                    "scheduled_date": c.get("scheduled_date", w_start),
                    "overdue_by_days": float(o),
                    "source_ref": c.get("source_ref", ""),
                }
            )
    if overdues:
        out["overdue_items"] = overdues

    return out
