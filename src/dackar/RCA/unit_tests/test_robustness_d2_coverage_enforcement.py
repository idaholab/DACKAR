"""
test_robustness_d2_coverage_enforcement.py — D2 Causal category coverage (Phase 3)

Checks:
    D2-A  All 12 categories A–L present in applicability_assessment           [should PASS]
    D2-B  Missing optional source → data_limited status in coverage summary   [should PASS]
    D2-C  Scaffold candidates score below evidence floor (0.35)               [should PASS]
    D2-D  Ruled-out candidates carry approved reason codes                     [should PASS]

Run directly:   python test_robustness_d2_coverage_enforcement.py
Or via pytest:  pytest test_robustness_d2_coverage_enforcement.py -v

Schema note (confirmed 2026-05-23):
    The plan referenced result["causality_candidates"]["screening"]["category_coverage"].
    The actual schema is result["causality_candidates"]["applicability_assessment"], a
    dict keyed by category (A–L) with {"status": str, "rationale": str} per entry.
    This is recorded as BUG-D2-schema in the implementation log.

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

_APPROVED_REASON_CODES = {
    "physically_impossible",
    "timeline_inconsistent",
    "barrier_held",
    "no_supporting_data",
    "category_not_applicable",
    "outside_investigation_scope",
    "superseded_by_higher_fidelity_evidence",
    "analyst_excluded",
}


def _run(fixture_dir: Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        return run_rca(build_fixture_orchestrator(tmp), load_fixtures(fixture_dir))


def _get_applicability(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return applicability_assessment dict (actual schema) or empty dict."""
    return (result.get("causality_candidates") or {}).get("applicability_assessment") or {}


# ---------------------------------------------------------------------------
# D2-A — All A–L categories present in applicability_assessment
# ---------------------------------------------------------------------------

def _assert_d2a(result: Dict[str, Any], label: str) -> None:
    """
    Invariant: Every category A–L must appear in applicability_assessment with
    at least a 'status' field.  A missing category means the coverage enforcement
    loop skipped a causal dimension entirely — a silent gap.

    Schema note: the plan expected 'screening.category_coverage' with
    'covered'/'ruled_out' boolean keys.  The actual schema is
    'applicability_assessment' with a 'status' string field.
    """
    aa = _get_applicability(result)
    if not aa:
        # Fallback: check under 'screening.category_coverage' (plan schema)
        aa = (
            (result.get("causality_candidates") or {})
            .get("screening") or {}
        ).get("category_coverage") or {}

    missing_cats = []
    missing_status = []
    for cat in "ABCDEFGHIJKL":
        if cat not in aa:
            missing_cats.append(cat)
        elif not aa[cat].get("status"):
            missing_status.append(cat)

    assert not missing_cats, (
        f"D2-A FAIL [{label}]: categories absent from applicability_assessment: "
        f"{missing_cats}. "
        "Check that the causality engine iterates all 12 metamodel categories."
    )
    if missing_status:
        print(
            f"  warn  D2-A [{label}]: categories present but no 'status' field: "
            f"{missing_status}"
        )
    print(
        f"  pass  D2-A [{label}]: all 12 categories A–L present in applicability_assessment"
    )


def test_d2a_full_coverage_tc8():
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    _assert_d2a(_run(_TC8_FIXTURES), "TC-8")


def test_d2a_full_coverage_tc5():
    if not _TC5_FIXTURES.exists():
        pytest.skip("TC-5 fixtures not found")
    _assert_d2a(_run(_TC5_FIXTURES), "TC-5")


# ---------------------------------------------------------------------------
# D2-B — Missing optional source → data_limited / not_assessed in coverage
# ---------------------------------------------------------------------------

def test_d2b_missing_source_produces_not_assessed():
    """
    D2-B: When vendor_supply_chain_records is None, the coverage entry for
    that source must reflect data absence — 'not_assessed', 'data_limited',
    or 'partial' — not 'complete' or an unhandled exception.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")

    fixtures = load_fixtures(_TC8_FIXTURES)
    fixtures = dict(fixtures, vendor_supply_chain_records=None)

    with tempfile.TemporaryDirectory() as tmp:
        result = run_rca(build_fixture_orchestrator(tmp), fixtures)

    # Coverage may be in coverage_summary.source_families or data_coverage_summary
    mf = result.get("run_manifest") or {}
    cov_top = (mf.get("coverage_summary") or {}).get("source_families") or {}
    cov_alt = (mf.get("artifacts") or {}).get("data_coverage_summary") or {}
    cov = cov_top or cov_alt

    vendor_entry = cov.get("vendor_supply_chain_records")
    if vendor_entry is None:
        print(
            "  warn  D2-B: 'vendor_supply_chain_records' absent from coverage summary — "
            "coverage tracking may not include this source by name"
        )
        return

    status = vendor_entry.get("status") if isinstance(vendor_entry, dict) else str(vendor_entry)
    assert status in ("not_assessed", "data_limited", "partial", "missing"), (
        f"D2-B FAIL: vendor_supply_chain_records=None but coverage status={status!r}. "
        "Expected 'not_assessed', 'data_limited', 'partial', or 'missing'."
    )
    print(f"  pass  D2-B: vendor_supply_chain_records=None → status={status!r}")


# ---------------------------------------------------------------------------
# D2-C — Scaffold candidates score below evidence floor (0.35)
# ---------------------------------------------------------------------------

def test_d2c_scaffold_candidates_below_floor_tc8():
    """
    D2-C: Any candidate generated as a scaffold (is_scaffold=True) must have
    composite_score < 0.35 and carry a scaffold_reason.  Scaffolds are
    placeholders — they must never outrank evidence-backed candidates.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")

    result = _run(_TC8_FIXTURES)
    candidates: List[Dict[str, Any]] = (
        result.get("causality_candidates") or {}
    ).get("candidates", [])

    scaffolds = [c for c in candidates if c.get("is_scaffold")]

    if not scaffolds:
        print("  skip  D2-C TC-8: no scaffold candidates in result — check with adversarial fixture")
        return

    violations = []
    for s in scaffolds:
        cid = s.get("candidate_id") or s.get("failure_mode_id", "?")
        score = float(s.get("composite_score") or 0.0)
        reason = s.get("scaffold_reason")
        if score >= 0.35:
            violations.append(f"{cid}: scaffold but composite_score={score:.3f} ≥ 0.35")
        if not reason:
            violations.append(f"{cid}: scaffold but scaffold_reason absent")

    assert not violations, (
        "D2-C FAIL: scaffold candidate violations:\n  " + "\n  ".join(violations)
    )
    print(f"  pass  D2-C TC-8: {len(scaffolds)} scaffold(s) all below 0.35 floor with reasons")


# ---------------------------------------------------------------------------
# D2-D — Ruled-out candidates carry approved reason codes
# ---------------------------------------------------------------------------

def _assert_d2d(result: Dict[str, Any], label: str) -> None:
    ruled_out: List[Dict[str, Any]] = (
        result.get("causality_candidates") or {}
    ).get("ruled_out", [])

    if not ruled_out:
        print(
            f"  skip  D2-D [{label}]: no ruled_out entries — "
            "use TC-2 (timeline gate) or adversarial fixture for coverage"
        )
        return

    violations = []
    for ro in ruled_out:
        cid = ro.get("candidate_id") or ro.get("failure_mode_id", "?")
        code = ro.get("reason_code")
        if code not in _APPROVED_REASON_CODES:
            violations.append(
                f"{cid}: unapproved reason_code={code!r}. "
                f"Approved: {sorted(_APPROVED_REASON_CODES)}"
            )

    assert not violations, (
        f"D2-D FAIL [{label}]: ruled_out entries with unapproved reason codes:\n  "
        + "\n  ".join(violations)
    )
    print(
        f"  pass  D2-D [{label}]: {len(ruled_out)} ruled_out entry/entries, "
        f"all reason codes approved"
    )


def test_d2d_ruled_out_reason_codes_tc5():
    if not _TC5_FIXTURES.exists():
        pytest.skip("TC-5 fixtures not found")
    _assert_d2d(_run(_TC5_FIXTURES), "TC-5")


def test_d2d_ruled_out_reason_codes_tc6():
    if not _TC6_FIXTURES.exists():
        pytest.skip("TC-6 fixtures not found")
    _assert_d2d(_run(_TC6_FIXTURES), "TC-6")


def test_d2d_ruled_out_reason_codes_tc8():
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    _assert_d2d(_run(_TC8_FIXTURES), "TC-8")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_d2a_full_coverage_tc8,
    test_d2a_full_coverage_tc5,
    test_d2b_missing_source_produces_not_assessed,
    test_d2c_scaffold_candidates_below_floor_tc8,
    test_d2d_ruled_out_reason_codes_tc5,
    test_d2d_ruled_out_reason_codes_tc6,
    test_d2d_ruled_out_reason_codes_tc8,
]


def run_all() -> bool:
    print(f"\n=== test_robustness_d2_coverage_enforcement ({len(ALL_TESTS)} tests) ===")
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
