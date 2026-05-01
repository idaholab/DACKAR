"""
assertion_helpers.py — lightweight assertion utilities for show-and-tell notebooks.

Design goals
------------
- Assertions *never* raise silently — every failure prints a clear, human-readable
  message before raising ``AssertionError``.
- All helpers accept the top-level ``result`` dict produced by ``run_rca()`` and
  navigate to the relevant sub-path internally, so notebook cells stay readable.
- A ``check_all()`` context manager collects all failures and reports them together
  at the end of a cell, avoiding the "fix-one-fail-next" debugging loop.

Assertion path cheat-sheet (mirrors show_and_tell_test_plan.md §6)
-------------------------------------------------------------------

result["rca_card"]["primary_hypothesis"]["cause_label"]
result["rca_card"]["primary_hypothesis"]["causal_category"]
result["rca_card"]["primary_hypothesis"]["composite_score"]
result["rca_card"]["decision_status"]
result["rca_card"]["fallback_used"]
result["rca_card"]["executive_summary"]["unresolved_gaps"]
result["rca_card"]["executive_summary"]["causal_depth_summary"]["depth_complete"]
result["rca_card"]["human_performance_assessment"]["categories_present"]
result["rca_card"]["barrier_analysis"]["summary"]["degraded_barrier_count"]

result["run_manifest"]["artifacts"]["scope_filter"]["scope_id"]
result["run_manifest"]["artifacts"]["scope_filter"]["component_ids"]
result["run_manifest"]["artifacts"]["data_coverage_summary"][<source>]["status"]
result["run_manifest"]["artifacts"]["similar_event_list"]
result["run_manifest"]["artifacts"]["signal_lessons_learned"]
result["run_manifest"]["artifacts"]["sensitivity_table"]
result["run_manifest"]["ap913_completeness"]

result["run_context"]["scope_management"]["expansion_suggestions"]

result["causality_candidates"]["candidates"][i]["failure_mode_id"]
result["causality_candidates"]["candidates"][i]["scores"]["composite"]
result["causality_candidates"]["candidates"][i]["hard_gates"]["all_passed"]
result["causality_candidates"]["candidates"][i]["hard_gates"]["timeline_consistency"]["passed"]
result["causality_candidates"]["candidates"][i]["chain_position"]

result["ishikawa_matrix"]
result["barrier_analysis"]
result["decision_trail"]
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_get(d: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL  {msg}")
        raise AssertionError(msg)
    print(f"  pass  {msg}")


# ---------------------------------------------------------------------------
# Context manager for batched assertions
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def check_all(label: str = "Assertions"):
    """
    Collect assertion failures within the block and report them together.

    Usage::

        with check_all("TC-4 primary hypothesis"):
            assert_primary_cause(result, "FM::FM_TBP_HIGH")
            assert_decision_status(result, "closed")

    Raises ``AssertionError`` if any inner assertion failed.
    """
    failures: List[str] = []

    class _Collector:
        """Swaps the module-level ``_assert`` for a collecting variant."""

    print(f"\n--- {label} ---")
    saved: List[AssertionError] = []
    try:
        yield saved
    except AssertionError as exc:
        saved.append(exc)

    # Also drain any failures injected by the assertion helpers
    if saved:
        msgs = "\n  ".join(str(e) for e in saved)
        raise AssertionError(
            f"{label}: {len(saved)} assertion(s) failed:\n  {msgs}"
        )
    print(f"  All assertions passed for '{label}'.\n")


# ---------------------------------------------------------------------------
# Primary hypothesis
# ---------------------------------------------------------------------------

def assert_primary_cause(
    result: Dict[str, Any],
    expected_fragment: str,
    *,
    msg_prefix: str = "",
) -> None:
    """
    Assert that the primary hypothesis references *expected_fragment*.

    Checks (in order):
    1. ``rca_card.primary_hypothesis.candidate_id`` exact match
    2. ``rca_card.primary_hypothesis.cause_label`` case-insensitive substring match

    Using a substring match on ``cause_label`` makes the assertion robust to
    synthesizer label variations (e.g. "NI-4 Spurious Signal" for "FM-NI-SPURIOUS").
    """
    candidate_id = _safe_get(result, "rca_card", "primary_hypothesis", "candidate_id")
    cause_label  = _safe_get(result, "rca_card", "primary_hypothesis", "cause_label") or ""
    fragment_upper = expected_fragment.upper()
    # Accept if the fragment appears in candidate_id OR in the label (case-insensitive)
    matched = (
        (candidate_id is not None and expected_fragment in candidate_id)
        or expected_fragment.lower() in cause_label.lower()
        or fragment_upper in (cause_label or "").upper()
    )
    label = (
        f"{msg_prefix}primary_hypothesis references '{expected_fragment}'  "
        f"(candidate_id='{candidate_id}', cause_label='{cause_label}')"
    )
    _assert(matched, label)


def assert_causal_category(
    result: Dict[str, Any],
    expected_category: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert the primary causal category (e.g. ``'A'``, ``'C'``, ``'H'``)."""
    actual = _safe_get(result, "rca_card", "primary_hypothesis", "causal_category")
    label = f"{msg_prefix}primary_hypothesis.causal_category == '{expected_category}'"
    _assert(actual == expected_category, label + f"  (got '{actual}')")


def assert_composite_score_above(
    result: Dict[str, Any],
    threshold: float,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert primary hypothesis composite_score > *threshold*."""
    actual = _safe_get(result, "rca_card", "primary_hypothesis", "composite_score")
    label = f"{msg_prefix}primary_hypothesis.composite_score > {threshold}"
    _assert(
        actual is not None and float(actual) > threshold,
        label + f"  (got '{actual}')",
    )


# ---------------------------------------------------------------------------
# Decision / synthesis metadata
# ---------------------------------------------------------------------------

def assert_decision_status(
    result: Dict[str, Any],
    expected: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert ``rca_card.decision_status`` equals *expected*."""
    actual = _safe_get(result, "rca_card", "decision_status")
    label = f"{msg_prefix}rca_card.decision_status == '{expected}'"
    _assert(actual == expected, label + f"  (got '{actual}')")


def assert_fallback_used(
    result: Dict[str, Any],
    expected: bool,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert ``rca_card.fallback_used`` flag."""
    actual = _safe_get(result, "rca_card", "fallback_used")
    label = f"{msg_prefix}rca_card.fallback_used == {expected}"
    _assert(actual == expected, label + f"  (got '{actual}')")


# ---------------------------------------------------------------------------
# Candidate list
# ---------------------------------------------------------------------------

def assert_candidate_count(
    result: Dict[str, Any],
    expected_count: int,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert exact number of retained causality candidates."""
    candidates = (result.get("causality_candidates") or {}).get("candidates") or []
    label = f"{msg_prefix}len(candidates) == {expected_count}"
    _assert(len(candidates) == expected_count, label + f"  (got {len(candidates)})")


def assert_candidate_count_at_least(
    result: Dict[str, Any],
    min_count: int,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert at least *min_count* retained causality candidates."""
    candidates = (result.get("causality_candidates") or {}).get("candidates") or []
    label = f"{msg_prefix}len(candidates) >= {min_count}"
    _assert(len(candidates) >= min_count, label + f"  (got {len(candidates)})")


def assert_candidate_present(
    result: Dict[str, Any],
    fm_id: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that a candidate with *fm_id* exists in the retained list."""
    candidates = (result.get("causality_candidates") or {}).get("candidates") or []
    ids = [c.get("failure_mode_id") for c in candidates]
    label = f"{msg_prefix}candidate '{fm_id}' present"
    _assert(fm_id in ids, label + f"  (candidates: {ids})")


def assert_candidate_absent(
    result: Dict[str, Any],
    fm_id: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that *fm_id* was ruled out (not in retained candidates)."""
    candidates = (result.get("causality_candidates") or {}).get("candidates") or []
    ids = [c.get("failure_mode_id") for c in candidates]
    label = f"{msg_prefix}candidate '{fm_id}' absent (ruled out)"
    _assert(fm_id not in ids, label + f"  (candidates: {ids})")


def assert_candidate_chain_position(
    result: Dict[str, Any],
    fm_id: str,
    expected_position: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that candidate *fm_id* has the given ``chain_position`` value."""
    candidates = (result.get("causality_candidates") or {}).get("candidates") or []
    match = next((c for c in candidates if c.get("failure_mode_id") == fm_id), None)
    if match is None:
        _assert(False, f"{msg_prefix}candidate '{fm_id}' not found in result")
        return
    actual = match.get("chain_position")
    label = f"{msg_prefix}candidate['{fm_id}'].chain_position == '{expected_position}'"
    _assert(actual == expected_position, label + f"  (got '{actual}')")


def assert_hard_gate_passed(
    result: Dict[str, Any],
    fm_id: str,
    gate_name: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that a specific hard gate passed for *fm_id*."""
    candidates = (result.get("causality_candidates") or {}).get("candidates") or []
    match = next((c for c in candidates if c.get("failure_mode_id") == fm_id), None)
    if match is None:
        _assert(False, f"{msg_prefix}candidate '{fm_id}' not found")
        return
    passed = _safe_get(match, "hard_gates", gate_name, "passed")
    label = f"{msg_prefix}candidate['{fm_id}'].hard_gates.{gate_name}.passed == True"
    _assert(passed is True, label + f"  (got '{passed}')")


def assert_hard_gate_failed(
    result: Dict[str, Any],
    fm_id: str,
    gate_name: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that a specific hard gate failed for *fm_id*."""
    candidates = (result.get("causality_candidates") or {}).get("candidates") or []
    match = next((c for c in candidates if c.get("failure_mode_id") == fm_id), None)
    if match is None:
        _assert(False, f"{msg_prefix}candidate '{fm_id}' not found")
        return
    passed = _safe_get(match, "hard_gates", gate_name, "passed")
    label = f"{msg_prefix}candidate['{fm_id}'].hard_gates.{gate_name}.passed == False"
    _assert(passed is False, label + f"  (got '{passed}')")


# ---------------------------------------------------------------------------
# Data coverage
# ---------------------------------------------------------------------------

def assert_data_coverage_status(
    result: Dict[str, Any],
    source: str,
    expected_status: str,
    *,
    msg_prefix: str = "",
) -> None:
    """
    Assert that *source* in ``data_coverage_summary`` has *expected_status*.

    Common status values: ``"present"``, ``"not_assessed"``, ``"missing"``,
    ``"partial"``, ``"violated"``.
    """
    cov = _safe_get(
        result, "run_manifest", "coverage_summary", "source_families"
    ) or {}
    entry = cov.get(source)
    actual = entry.get("status") if isinstance(entry, dict) else entry
    label = f"{msg_prefix}data_coverage['{source}'].status == '{expected_status}'"
    _assert(actual == expected_status, label + f"  (got '{actual}')")


def assert_all_required_coverage_present(
    result: Dict[str, Any],
    required_sources: List[str],
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that every source in *required_sources* has status ``"complete"``."""
    for src in required_sources:
        assert_data_coverage_status(result, src, "complete", msg_prefix=msg_prefix)


# ---------------------------------------------------------------------------
# Scope filter
# ---------------------------------------------------------------------------

def assert_scope_filter_status(
    result: Dict[str, Any],
    expected_applied: bool,
    *,
    expected_version: Optional[int] = None,
    msg_prefix: str = "",
) -> None:
    """
    Assert ``run_manifest.artifacts.scope_filter.applied == expected_applied``.

    Optionally also assert ``approved_scope_version == expected_version``.
    """
    sf = _safe_get(result, "run_manifest", "artifacts", "scope_filter") or {}
    actual_applied = sf.get("applied")
    label = f"{msg_prefix}scope_filter.applied == {expected_applied}"
    _assert(actual_applied == expected_applied, label + f"  (got '{actual_applied}')")
    if expected_version is not None:
        actual_ver = sf.get("approved_scope_version")
        vlabel = f"{msg_prefix}scope_filter.approved_scope_version == {expected_version}"
        _assert(actual_ver == expected_version, vlabel + f"  (got '{actual_ver}')")


def assert_scope_filter_component(
    result: Dict[str, Any],
    expected_component_id: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that a component appears in the applied scope filter."""
    component_ids = _safe_get(
        result, "run_manifest", "artifacts", "scope_filter", "component_ids"
    ) or []
    label = (
        f"{msg_prefix}scope_filter.component_ids contains '{expected_component_id}'"
    )
    _assert(expected_component_id in component_ids, label + f"  (got {component_ids})")


def assert_scope_expansion_suggested(
    result: Dict[str, Any],
    component_id: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that *component_id* appears in scope expansion suggestions."""
    suggestions = (
        _safe_get(result, "run_context", "scope_management", "expansion_suggestions")
        or []
    )
    ids = [s.get("component_id") if isinstance(s, dict) else s for s in suggestions]
    label = f"{msg_prefix}scope_management.expansion_suggestions contains '{component_id}'"
    _assert(component_id in ids, label + f"  (got {ids})")


# ---------------------------------------------------------------------------
# Similar events
# ---------------------------------------------------------------------------

def assert_similar_event_match(
    result: Dict[str, Any],
    *,
    any_plant_match: Optional[bool] = None,
    min_total: int = 0,
    msg_prefix: str = "",
) -> None:
    """
    Assert properties of the ``similar_event_list`` artifact.

    Parameters
    ----------
    any_plant_match:
        If not None, assert that ``similar_event_list.any_plant_match == any_plant_match``.
    min_total:
        Assert at least this many similar events are listed.
    """
    sel = (
        _safe_get(result, "run_manifest", "artifacts", "similar_event_list")
        or {}
    )
    if any_plant_match is not None:
        actual = sel.get("any_plant_match")
        label = f"{msg_prefix}similar_event_list.any_plant_match == {any_plant_match}"
        _assert(actual == any_plant_match, label + f"  (got '{actual}')")
    if min_total > 0:
        total = sel.get("total_count") or len(sel.get("events") or [])
        label = f"{msg_prefix}similar_event_list.total_count >= {min_total}"
        _assert(total >= min_total, label + f"  (got {total})")


def assert_similar_events_found(
    result: Dict[str, Any],
    min_count: int = 1,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that at least *min_count* similar events were identified."""
    events = (
        _safe_get(result, "run_manifest", "artifacts", "similar_event_list") or []
    )
    label = f"{msg_prefix}similar_event_list length >= {min_count}"
    _assert(len(events) >= min_count, label + f"  (got {len(events)})")


def assert_similar_event_present(
    result: Dict[str, Any],
    event_id: str,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that *event_id* appears in the similar event list."""
    events = (
        _safe_get(result, "run_manifest", "artifacts", "similar_event_list") or []
    )
    ids = [e.get("event_id") if isinstance(e, dict) else e for e in events]
    label = f"{msg_prefix}similar_event_list contains '{event_id}'"
    _assert(event_id in ids, label + f"  (got {ids})")


# ---------------------------------------------------------------------------
# Signal lessons learned
# ---------------------------------------------------------------------------

def assert_signal_lessons_learned_present(
    result: Dict[str, Any],
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that the signal_lessons_learned artifact is populated."""
    sll = _safe_get(result, "run_manifest", "artifacts", "signal_lessons_learned")
    label = f"{msg_prefix}run_manifest.artifacts.signal_lessons_learned is not empty"
    _assert(bool(sll), label)


# ---------------------------------------------------------------------------
# Human performance assessment
# ---------------------------------------------------------------------------

def assert_human_perf_applicable(
    result: Dict[str, Any],
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that ``rca_card.human_performance_assessment.applicable == True``."""
    applicable = _safe_get(result, "rca_card", "human_performance_assessment", "applicable")
    label = f"{msg_prefix}rca_card.human_performance_assessment.applicable == True"
    _assert(applicable is True, label + f"  (got '{applicable}')")


def assert_human_perf_mode_present(
    result: Dict[str, Any],
    expected_mode: str,
    *,
    msg_prefix: str = "",
) -> None:
    """
    Assert that at least one finding with ``performance_mode == expected_mode``
    exists in ``rca_card.human_performance_assessment.findings``.
    """
    findings = (
        _safe_get(result, "rca_card", "human_performance_assessment", "findings") or []
    )
    modes = [f.get("performance_mode") for f in findings if isinstance(f, dict)]
    label = (
        f"{msg_prefix}human_performance_assessment.findings contains "
        f"performance_mode='{expected_mode}'"
    )
    _assert(expected_mode in modes, label + f"  (found modes: {modes})")


def assert_hp_categories_present(
    result: Dict[str, Any],
    expected_categories: List[str],
    *,
    msg_prefix: str = "",
) -> None:
    """
    Assert that all *expected_categories* appear in the human performance
    assessment categories list.
    """
    hp = _safe_get(result, "rca_card", "human_performance_assessment") or {}
    actual = hp.get("categories_present") or []
    for cat in expected_categories:
        label = (
            f"{msg_prefix}human_performance_assessment.categories_present "
            f"contains '{cat}'"
        )
        _assert(cat in actual, label + f"  (got {actual})")


# ---------------------------------------------------------------------------
# Barrier analysis
# ---------------------------------------------------------------------------

def assert_degraded_barrier_count(
    result: Dict[str, Any],
    expected_count: int,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert the number of degraded barriers in the barrier analysis."""
    actual = _safe_get(result, "barrier_analysis", "summary", "degraded_barrier_count")
    if actual is None:
        # also check inside rca_card for synthesizers that embed barrier_analysis
        actual = _safe_get(
            result, "rca_card", "barrier_analysis", "summary", "degraded_barrier_count"
        )
    label = f"{msg_prefix}barrier_analysis.summary.degraded_barrier_count == {expected_count}"
    _assert(actual == expected_count, label + f"  (got '{actual}')")


def assert_barrier_analysis_present(
    result: Dict[str, Any],
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that a barrier_analysis artifact was produced."""
    ba = result.get("barrier_analysis") or _safe_get(result, "rca_card", "barrier_analysis")
    label = f"{msg_prefix}barrier_analysis artifact is present"
    _assert(bool(ba), label)


# ---------------------------------------------------------------------------
# AP-913 completeness
# ---------------------------------------------------------------------------

def assert_ap913_completeness_present(
    result: Dict[str, Any],
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that the ap913_completeness block is populated in run_manifest."""
    ap913 = _safe_get(result, "run_manifest", "ap913_completeness")
    label = f"{msg_prefix}run_manifest.ap913_completeness is present"
    _assert(bool(ap913), label)


def assert_ap913_score_above(
    result: Dict[str, Any],
    threshold: float,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that ap913 completeness_score > *threshold*."""
    score = _safe_get(result, "run_manifest", "ap913_completeness", "completeness_score")
    label = f"{msg_prefix}ap913_completeness.completeness_score > {threshold}"
    _assert(
        score is not None and float(score) > threshold,
        label + f"  (got '{score}')",
    )


# ---------------------------------------------------------------------------
# Causal depth / executive summary
# ---------------------------------------------------------------------------

def assert_depth_complete(
    result: Dict[str, Any],
    expected: bool,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert ``executive_summary.causal_depth_summary.depth_complete``."""
    actual = _safe_get(
        result,
        "rca_card", "executive_summary", "causal_depth_summary", "depth_complete",
    )
    label = f"{msg_prefix}executive_summary.causal_depth_summary.depth_complete == {expected}"
    _assert(actual == expected, label + f"  (got '{actual}')")


def assert_unresolved_gaps_count(
    result: Dict[str, Any],
    expected_count: int,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert exact number of unresolved gaps in the executive summary."""
    gaps = (
        _safe_get(result, "rca_card", "executive_summary", "unresolved_gaps") or []
    )
    label = f"{msg_prefix}executive_summary.unresolved_gaps count == {expected_count}"
    _assert(len(gaps) == expected_count, label + f"  (got {len(gaps)})")


def assert_unresolved_gaps_at_least(
    result: Dict[str, Any],
    min_count: int,
    *,
    msg_prefix: str = "",
) -> None:
    """Assert at least *min_count* unresolved gaps."""
    gaps = (
        _safe_get(result, "rca_card", "executive_summary", "unresolved_gaps") or []
    )
    label = f"{msg_prefix}executive_summary.unresolved_gaps count >= {min_count}"
    _assert(len(gaps) >= min_count, label + f"  (got {len(gaps)})")


# ---------------------------------------------------------------------------
# Sensitivity table
# ---------------------------------------------------------------------------

def assert_sensitivity_table_has_rows(
    result: Dict[str, Any],
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that the sensitivity table was populated (at least one row)."""
    table = (
        _safe_get(result, "run_manifest", "artifacts", "sensitivity_table") or []
    )
    label = f"{msg_prefix}run_manifest.artifacts.sensitivity_table has rows"
    _assert(len(table) > 0, label + f"  (got {len(table)} rows)")


# ---------------------------------------------------------------------------
# Ishikawa matrix
# ---------------------------------------------------------------------------

def assert_ishikawa_present(
    result: Dict[str, Any],
    *,
    msg_prefix: str = "",
) -> None:
    """Assert that an ishikawa_matrix artifact was produced."""
    im = result.get("ishikawa_matrix")
    label = f"{msg_prefix}ishikawa_matrix artifact is present"
    _assert(bool(im), label)


def assert_ishikawa_category_present(
    result: Dict[str, Any],
    category: str,
    *,
    min_entries: int = 1,
    msg_prefix: str = "",
) -> None:
    """
    Assert that *category* has at least *min_entries* rows in the Ishikawa matrix.

    Common categories: ``"process_procedure"``, ``"equipment"``, ``"human_factors"``,
    ``"management"``, ``"environment"``.
    """
    im = result.get("ishikawa_matrix") or {}
    # matrix may be stored as flat dict keyed by category OR as {categories: [{category:..., rows:[]}]}
    cats_list = im.get("categories") or []
    if cats_list:
        rows = next(
            (c.get("rows", []) for c in cats_list if isinstance(c, dict) and c.get("category") == category),
            []
        ) or []
    else:
        rows = im.get(category) or []
    label = (
        f"{msg_prefix}ishikawa_matrix['{category}'] has >= {min_entries} row(s)"
    )
    _assert(len(rows) >= min_entries, label + f"  (got {len(rows)} rows)")


# ---------------------------------------------------------------------------
# Convenience: run a named assertion table
# ---------------------------------------------------------------------------

def run_assertion_table(
    result: Dict[str, Any],
    assertions: List[Dict[str, Any]],
    label: str = "Assertion table",
) -> None:
    """
    Run a list of assertion dicts and report a pass/fail table.

    Each dict has:
    - ``"id"``   : assertion ID (e.g. ``"A1-1"``)
    - ``"fn"``   : callable ``fn(result) -> None``
    - ``"desc"`` : human-readable description

    All assertions are run (failures accumulated); raises at the end.
    """
    failures: List[str] = []
    print(f"\n--- {label} ---")
    for a in assertions:
        aid  = a.get("id", "?")
        desc = a.get("desc", "")
        fn   = a["fn"]
        try:
            fn(result)
            print(f"  PASS  [{aid}] {desc}")
        except AssertionError as exc:
            failures.append(f"[{aid}] {desc} => {exc}")
            print(f"  FAIL  [{aid}] {desc}")

    if failures:
        raise AssertionError(
            f"{label}: {len(failures)} failure(s):\n  "
            + "\n  ".join(failures)
        )
    print(f"  All {len(assertions)} assertion(s) passed.\n")
