"""PMScopeAnalyzer — FMEA/KG ↔ PM task coverage and scope gaps."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .types import JsonDict

# Priority order when a check appears in multiple KG fields:
# "preventive" outranks "detective" — a task that prevents a failure mode is the
# stronger claim; we never downgrade a "preventive" assignment to "detective".
_FIELD_COVERAGE_TYPE: Dict[str, str] = {
    "detecting_pm_task_ids":  "detective",   # processed first (lowest priority)
    "pm_task_ids":            "preventive",  # generic linkage; treated as preventive
    "preventing_pm_task_ids": "preventive",  # highest priority
    # MR#56 review I3: prevents_pm_tasks also advertises PM↔FM linkage (recognized
    # by _explicit_pm_fm_links), so it must contribute coverage too — otherwise an
    # FM linked only via this field flips linkage on but shows every FM as a gap.
    "prevents_pm_tasks":      "preventive",
}


def _fm_ids_from_kg(kg_context: JsonDict) -> Set[str]:
    fms: List[JsonDict] = kg_context.get("failure_modes") or []
    return {str(fm.get("fm_id")) for fm in fms if fm.get("fm_id")}


def _explicit_pm_fm_links(kg_context: JsonDict) -> bool:
    """Return True if any failure mode advertises PM linkage fields."""
    fms: List[JsonDict] = kg_context.get("failure_modes") or []
    for fm in fms:
        for key in (
            "preventing_pm_task_ids",
            "detecting_pm_task_ids",
            "pm_task_ids",
            "prevents_pm_tasks",
        ):
            if fm.get(key):
                return True
    return False


def analyze_scope(
    kg_context: Optional[JsonDict],
    checks: List[JsonDict],
) -> Tuple[List[Dict[str, Any]], bool, Set[str], Set[str], Dict[str, str]]:
    """Build per-component scope view.

    Returns
    -------
    components_out
        Per-component list with ``scope_covers_failure_modes``, ``scope_gaps``.
    linkage_flag
        True when the KG carries explicit PM↔FM tags.
    covered_fms
        FM IDs covered by at least one passing check.
    all_fms
        All FM IDs present in the KG.
    check_to_coverage_type
        Maps check_id → ``"preventive"`` | ``"detective"`` for every check whose
        task ID appears in a KG FM linkage field.  Empty when linkage is absent.
        Used by the aggregator to set ``pm_tasks[].coverage_type``.
    """
    if not kg_context:
        return [], False, set(), set(), {}

    all_fms = _fm_ids_from_kg(kg_context)
    explicit = _explicit_pm_fm_links(kg_context)

    covered: Set[str] = set()
    for c in checks:
        if c.get("status") != "pass":
            continue
        for fid in c.get("applicable_fm_ids") or []:
            if not all_fms or fid in all_fms:
                covered.add(str(fid))

    # Build check_to_coverage_type while also updating covered.
    # Iterate fields in ascending priority order so "preventive" always wins.
    check_to_coverage_type: Dict[str, str] = {}
    # fm_id -> set of KG-linked PM task IDs, reused for per-component coverage below.
    fm_task_ids: Dict[str, Set[str]] = {}
    if explicit and all_fms:
        for fm in kg_context.get("failure_modes") or []:
            fm_id = str(fm.get("fm_id") or "")
            if not fm_id:
                continue
            for k, cov_type in _FIELD_COVERAGE_TYPE.items():
                for task_id in (fm.get(k) or []):
                    tid = str(task_id)
                    if not tid:
                        continue
                    # Assign coverage_type; "preventive" is never downgraded to "detective"
                    if check_to_coverage_type.get(tid) != "preventive":
                        check_to_coverage_type[tid] = cov_type
            # Mark FM as covered if any passing check matches a linked task ID
            all_task_ids: Set[str] = set()
            for k in _FIELD_COVERAGE_TYPE:
                all_task_ids |= {str(x) for x in (fm.get(k) or []) if x}
            fm_task_ids[fm_id] = all_task_ids
            for c in checks:
                if c.get("status") != "pass":
                    continue
                if str(c.get("check_id") or "") in all_task_ids:
                    covered.add(fm_id)

    scope_gaps: Set[str] = set()
    if all_fms:
        if explicit or covered:
            scope_gaps = all_fms - covered

    by_comp: Dict[str, List[JsonDict]] = {}
    for c in checks:
        cid = c.get("component_id") or "_asset"
        by_comp.setdefault(cid, []).append(c)

    comp_out: List[Dict[str, Any]] = []
    for cid, ch in by_comp.items():
        local_cov: Set[str] = set()
        passing_check_ids: Set[str] = set()
        for c in ch:
            if c.get("status") != "pass":
                continue
            passing_check_ids.add(str(c.get("check_id") or ""))
            for fid in c.get("applicable_fm_ids") or []:
                local_cov.add(str(fid))
        # MR#56 review I4: honor KG PM↔FM task-ID linkage per component too, so a
        # component's scope_gaps stay consistent with the global covered set (which
        # already credits KG linkage) instead of relying on export applicable_fm_ids only.
        for fm_id, tids in fm_task_ids.items():
            if passing_check_ids & tids:
                local_cov.add(fm_id)
        if explicit and all_fms:
            sgap = sorted(fm for fm in all_fms if fm not in local_cov)
        else:
            sgap = []
        comp_out.append(
            {
                "component_id": cid,
                "scope_covers_failure_modes": sorted(local_cov) if local_cov else [],
                "scope_gaps": sgap,
            }
        )

    # Per architecture §3.3: only *KG* PM↔FM tags count as "FMEA/PM linkage available".
    fmea_kg_linkage_available = bool(explicit)
    return comp_out, fmea_kg_linkage_available, covered, all_fms, check_to_coverage_type
