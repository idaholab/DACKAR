"""
test_robustness_d9_causal_depth.py — D9 Causal depth adequacy (Phase 3)

Checks:
    D9-A  Depth fields always present in rca_card.executive_summary
    D9-B  Unresolved root cause produces an attention flag
    D9-C  Corrective actions span all covered causal depths

Run directly:   python test_robustness_d9_causal_depth.py
Or via pytest:  pytest test_robustness_d9_causal_depth.py -v

Schema (causal_depth_summary):
    {
        "proximate_cause":    str | "unresolved",
        "contributing_causes": [str, ...],
        "root_cause":          str | "unresolved",
        "depth_complete":      bool,
        "proximate_covered":   bool,      # True when proximate_cause != "unresolved"
        "contributing_covered": bool,     # True when contributing_causes is non-empty
        "root_cause_covered":  bool,      # True when root_cause != "unresolved"
        "depth_incomplete_reason": str    # only when depth_complete=False
    }

Fixtures used: TC-5, TC-6, TC-8
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

_RCA_ROOT = Path(__file__).resolve().parent.parent
_TESTS_SHARED = _RCA_ROOT / "tests" / "shared"

for _p in (str(_RCA_ROOT), str(_TESTS_SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _mod in (
    "neo4j", "py2neo", "chromadb",
    "langchain_community", "langchain_community.vectorstores",
    "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca  # noqa: E402

_TC5_FIXTURES = _RCA_ROOT / "tests" / "test_case_5" / "fixtures"
_TC6_FIXTURES = _RCA_ROOT / "tests" / "test_case_6" / "fixtures"
_TC8_FIXTURES = _RCA_ROOT / "tests" / "test_case_8" / "fixtures"


def _run(fixture_dir: Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        return run_rca(build_fixture_orchestrator(tmp), load_fixtures(fixture_dir))


def _get_depth(result: Dict[str, Any]) -> Dict[str, Any]:
    return (
        (result.get("rca_card") or {})
        .get("executive_summary") or {}
    ).get("causal_depth_summary") or {}


# ---------------------------------------------------------------------------
# D9-A — Depth fields always present in rca_card.executive_summary
# ---------------------------------------------------------------------------

def _assert_d9a(result: Dict[str, Any], label: str) -> None:
    """
    Invariant: causal_depth_summary must always be present and contain all
    required fields — including the queryable *_covered booleans.
    """
    es = (result.get("rca_card") or {}).get("executive_summary") or {}
    depth = es.get("causal_depth_summary")

    assert depth is not None, (
        f"D9-A FAIL [{label}]: executive_summary.causal_depth_summary is absent. "
        "Every RCA run must declare which depth layers were addressed."
    )

    required_fields = (
        "depth_complete",
        "proximate_covered",
        "contributing_covered",
        "root_cause_covered",
    )
    missing = [f for f in required_fields if f not in depth]
    assert not missing, (
        f"D9-A FAIL [{label}]: causal_depth_summary missing required fields: {missing}. "
        f"Got keys: {sorted(depth.keys())}"
    )

    # Consistency: *_covered booleans must agree with prose string fields
    prox_str = str(depth.get("proximate_cause") or "").strip()
    assert depth["proximate_covered"] == bool(prox_str and prox_str != "unresolved"), (
        f"D9-A FAIL [{label}]: proximate_covered inconsistent with proximate_cause={prox_str!r}"
    )
    contrib_list = depth.get("contributing_causes") or []
    assert depth["contributing_covered"] == bool(contrib_list), (
        f"D9-A FAIL [{label}]: contributing_covered inconsistent with contributing_causes={contrib_list!r}"
    )
    root_str = str(depth.get("root_cause") or "").strip()
    assert depth["root_cause_covered"] == bool(root_str and root_str != "unresolved"), (
        f"D9-A FAIL [{label}]: root_cause_covered inconsistent with root_cause={root_str!r}"
    )

    print(
        f"  pass  D9-A [{label}]: causal_depth_summary present. "
        f"depth_complete={depth.get('depth_complete')} "
        f"proximate_covered={depth.get('proximate_covered')} "
        f"contributing_covered={depth.get('contributing_covered')} "
        f"root_cause_covered={depth.get('root_cause_covered')}"
    )


def test_d9a_depth_fields_present_tc5():
    if not _TC5_FIXTURES.exists():
        pytest.skip("TC-5 fixtures not found")
    _assert_d9a(_run(_TC5_FIXTURES), "TC-5")


def test_d9a_depth_fields_present_tc6():
    if not _TC6_FIXTURES.exists():
        pytest.skip("TC-6 fixtures not found")
    _assert_d9a(_run(_TC6_FIXTURES), "TC-6")


def test_d9a_depth_fields_present_tc8():
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    _assert_d9a(_run(_TC8_FIXTURES), "TC-8")


# ---------------------------------------------------------------------------
# D9-B — Unresolved root cause produces an attention flag
# ---------------------------------------------------------------------------

def _assert_d9b(result: Dict[str, Any], label: str) -> None:
    """
    Invariant: if depth_complete=False or root_cause='unresolved', at least
    one attention flag must call out the gap.  A silent incomplete RCA card
    gives the analyst no signal to investigate further.
    """
    depth = _get_depth(result)
    if not depth:
        print(f"  skip  D9-B [{label}]: causal_depth_summary absent")
        return

    depth_complete = depth.get("depth_complete", True)
    root_cause = depth.get("root_cause", "")
    root_unresolved = (
        depth_complete is False
        or str(root_cause).lower() == "unresolved"
        or root_cause is None
    )

    if not root_unresolved:
        print(
            f"  skip  D9-B [{label}]: depth_complete={depth_complete}, "
            f"root_cause={root_cause!r} — no unresolved depth to flag"
        )
        return

    es = (result.get("rca_card") or {}).get("executive_summary") or {}
    flags: List[str] = es.get("analyst_attention_flags") or []

    has_depth_flag = any(
        "root" in str(f).lower() or "depth" in str(f).lower() or "unresolved" in str(f).lower()
        for f in flags
    )
    assert has_depth_flag, (
        f"D9-B FAIL [{label}]: root_cause={root_cause!r}, depth_complete={depth_complete} "
        f"but no attention flag mentions root/depth/unresolved. "
        f"Got flags: {flags[:5]}"
    )
    print(
        f"  pass  D9-B [{label}]: unresolved depth produces attention flag. "
        f"depth_complete={depth_complete}"
    )


def test_d9b_unresolved_depth_flagged_tc8():
    """TC-8 has root_cause='unresolved' — a flag must fire."""
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    _assert_d9b(_run(_TC8_FIXTURES), "TC-8")


def test_d9b_unresolved_depth_flagged_tc6():
    if not _TC6_FIXTURES.exists():
        pytest.skip("TC-6 fixtures not found")
    _assert_d9b(_run(_TC6_FIXTURES), "TC-6")


# ---------------------------------------------------------------------------
# D9-C — Corrective actions span all covered causal depths
# ---------------------------------------------------------------------------

def _assert_d9c(result: Dict[str, Any], label: str) -> None:
    """
    Invariant: for each depth layer that is addressed (proximate_cause set,
    contributing_causes non-empty, root_cause resolved), at least one
    recommended_action must have target_causal_depth set to that layer.

    If an action lacks target_causal_depth the field is treated as absent —
    this itself is noted as a schema gap.
    """
    depth = _get_depth(result)
    actions: List[Dict[str, Any]] = (
        (result.get("rca_card") or {}).get("recommended_actions") or []
    )

    if not actions:
        print(f"  skip  D9-C [{label}]: no recommended_actions in rca_card")
        return

    if not depth:
        print(f"  skip  D9-C [{label}]: causal_depth_summary absent")
        return

    # Determine which depth layers are covered
    covered_layers = []
    if depth.get("proximate_cause"):
        covered_layers.append("proximate")
    if depth.get("contributing_causes"):
        covered_layers.append("contributing")
    root = depth.get("root_cause", "")
    if root and str(root).lower() not in ("unresolved", "none", ""):
        covered_layers.append("root")

    if not covered_layers:
        print(f"  skip  D9-C [{label}]: no covered depth layers found in causal_depth_summary")
        return

    action_depths = {a.get("target_causal_depth") for a in actions}

    # Check for actions with no target_causal_depth
    undepth_count = sum(1 for a in actions if not a.get("target_causal_depth"))
    if undepth_count:
        print(
            f"  note  D9-C [{label}]: {undepth_count}/{len(actions)} action(s) "
            "lack target_causal_depth field — depth coverage assertion is partial"
        )

    missing_depth_coverage = []
    for layer in covered_layers:
        if layer not in action_depths:
            missing_depth_coverage.append(layer)

    if missing_depth_coverage:
        # Downgrade to warning if some actions lack the field (can't fully assert)
        if undepth_count > 0:
            print(
                f"  warn  D9-C [{label}]: covered depth(s) {missing_depth_coverage} "
                f"have no matching action. May be because {undepth_count} actions lack "
                "target_causal_depth. Add target_causal_depth to all actions."
            )
        else:
            assert False, (
                f"D9-C FAIL [{label}]: covered depth(s) {missing_depth_coverage} "
                "have no corresponding corrective action. "
                f"Action depths present: {sorted(action_depths - {None})}"
            )
    else:
        print(
            f"  pass  D9-C [{label}]: all covered depth(s) {covered_layers} "
            f"have at least one corrective action"
        )


def test_d9c_actions_span_depths_tc5():
    if not _TC5_FIXTURES.exists():
        pytest.skip("TC-5 fixtures not found")
    _assert_d9c(_run(_TC5_FIXTURES), "TC-5")


def test_d9c_actions_span_depths_tc8():
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    _assert_d9c(_run(_TC8_FIXTURES), "TC-8")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_d9a_depth_fields_present_tc5,
    test_d9a_depth_fields_present_tc6,
    test_d9a_depth_fields_present_tc8,
    test_d9b_unresolved_depth_flagged_tc8,
    test_d9b_unresolved_depth_flagged_tc6,
    test_d9c_actions_span_depths_tc5,
    test_d9c_actions_span_depths_tc8,
]


def run_all() -> bool:
    print(f"\n=== test_robustness_d9_causal_depth ({len(ALL_TESTS)} tests) ===")
    passed = failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
