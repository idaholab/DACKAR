"""
Shared regulatory / LCO decision policy for the outage pipeline.

Stage F (option feasibility) and Stage G (decision status) both reason about the
LCO action-level clock and regulatory constraints.  Previously each stage kept
its own copy of the relevant LCO-status sets, and the two drifted — the cause of
the ``unknown`` → DEFER gap (an undetermined clock was treated as safe in Stage
F but unsafe in Stage G).  This module is the single source of truth for that
policy so the two stages cannot disagree.

LCO clock status vocabulary (produced by Stage A ``lco_clock_status``):
    not_applicable  no LCO in effect.
    normal          clock running, deadline beyond the current shift (known).
    urgent          clock running, action needed this shift.
    critical        clock running, immediate action required.
    expired         action-level deadline already passed.
    unknown         clock running but the deadline was NOT supplied.

Two distinct policy questions, two distinct sets:

* LCO_ACTIVE_STATUSES — the clock is actively running, so deferring the work out
  of this outage is NOT feasible.  Includes ``normal`` (a known future deadline
  still forbids deferral) and ``unknown`` (running clock, treat as unsafe).

* LCO_FAIL_CLOSED_STATUSES — the decision must fail closed: no PROCEED or
  MONITOR, escalate for review instead.  Excludes ``normal`` (a known,
  non-imminent deadline does not by itself force escalation) but includes
  ``unknown`` (an undetermined deadline is treated as unsafe).

``unknown`` is deliberately in BOTH: a running clock with no supplied deadline
is the conservative/unsafe case for either question.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

# Clock is running → deferral out of the outage is not feasible.
LCO_ACTIVE_STATUSES = frozenset({"expired", "critical", "urgent", "normal", "unknown"})

# Decision must fail closed (no PROCEED/MONITOR; escalate) — excludes "normal".
LCO_FAIL_CLOSED_STATUSES = frozenset({"expired", "critical", "urgent", "unknown"})


def lco_status(intake_result: Mapping[str, Any]) -> str:
    """Return the normalised (lower-case) lco_clock_status, or '' when absent."""
    return str(intake_result.get("lco_clock_status") or "").lower()


def is_lco_active(intake_result: Mapping[str, Any]) -> bool:
    """True when the LCO clock is running (deferral not feasible)."""
    return lco_status(intake_result) in LCO_ACTIVE_STATUSES


def is_fail_closed(
    intake_result: Mapping[str, Any],
    schedule_impact: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when the decision must fail closed (escalate, never PROCEED/MONITOR).

    Fail-closed conditions:
      * the LCO action-level clock is in a fail-closed status
        (expired/critical/urgent/unknown), OR
      * any regulatory constraint is present, OR
      * critical-path impact could not be computed (Stage E set
        ``cp_impact.cp_impact_unavailable`` because the duration was unknown) —
        "couldn't compute CP impact" must not read as "no CP impact".
    """
    if lco_status(intake_result) in LCO_FAIL_CLOSED_STATUSES:
        return True
    if intake_result.get("has_regulatory_constraint"):
        return True
    cp = (schedule_impact or {}).get("cp_impact") or {}
    if cp.get("cp_impact_unavailable"):
        return True
    return False


def escalation_reason(
    intake_result: Mapping[str, Any],
    schedule_impact: Optional[Mapping[str, Any]] = None,
    *,
    option_type: Optional[str] = None,
    cp_drag_hours: Optional[float] = None,
    cp_escalate_threshold_hours: Optional[float] = None,
    no_options: bool = False,
) -> str:
    """Return a machine code naming WHY a decision escalated.

    Checked in the same priority order the decision uses, so the executive
    summary can state the actual cause instead of always blaming CP drag:
        no_options | regulatory_constraint | lco_clock_<status> |
        cp_impact_unavailable | cp_drag_threshold | escalate_option | unknown
    """
    if no_options:
        return "no_options"
    if intake_result.get("has_regulatory_constraint"):
        return "regulatory_constraint"
    status = lco_status(intake_result)
    if status in LCO_FAIL_CLOSED_STATUSES:
        return f"lco_clock_{status}"
    cp = (schedule_impact or {}).get("cp_impact") or {}
    if cp.get("cp_impact_unavailable"):
        return "cp_impact_unavailable"
    if option_type == "escalate_to_management":
        return "escalate_option"
    if (
        cp_drag_hours is not None
        and cp_escalate_threshold_hours is not None
        and cp_drag_hours > cp_escalate_threshold_hours
    ):
        return "cp_drag_threshold"
    return "unknown"


# Human-readable escalation-reason sentences for the executive summary.
ESCALATION_REASON_TEXT: Mapping[str, str] = {
    "no_options": (
        "no feasible, regulatory-cleared insertion option could be identified"
    ),
    "regulatory_constraint": (
        "an active regulatory constraint requires licensing review before action"
    ),
    "cp_impact_unavailable": (
        "critical-path impact could not be computed (unknown duration); schedule "
        "risk is unquantified"
    ),
    "cp_drag_threshold": "critical-path drag exceeds the auto-escalation threshold",
    "escalate_option": "the recommended option is an explicit escalation",
    "unknown": "management review is required",
}


def escalation_reason_text(reason: str, lco_status_value: str = "") -> str:
    """Map an escalation_reason code to a human-readable clause."""
    if reason.startswith("lco_clock_"):
        status = reason[len("lco_clock_"):]
        return f"the LCO action-level clock is '{status}' and requires licensing review"
    return ESCALATION_REASON_TEXT.get(reason, ESCALATION_REASON_TEXT["unknown"])
