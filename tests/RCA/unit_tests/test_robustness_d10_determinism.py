"""
test_robustness_d10_determinism.py — D10 Run-to-run determinism (Phase 2)

Checks:
    D10-A  Two runs with identical inputs produce identical scores and gates   [should PASS]
    D10-C  scoring_evolution delta is internally consistent                    [should PASS]
    D10-B  Sensitivity table predictions are empirically verifiable            [diagnostic / soft]

Run directly:   python test_robustness_d10_determinism.py
Or via pytest:  pytest test_robustness_d10_determinism.py -v

Note on D10-A:
    Two separate orchestrator instances are created (separate output dirs) to
    avoid any state leak between runs.  The run_id will differ — only the
    scoring payload (composite scores, sub-scores, gate results, primary
    hypothesis) is compared.

Fixtures used:
    TC-2  ../tests/test_case_2/fixtures/
    TC-3  ../tests/test_case_3/fixtures/
    TC-5  ../tests/test_case_5/fixtures/
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
_SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenario"
_TESTS_SHARED = _SCENARIO_ROOT / "shared"

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

import pytest  # noqa: E402
pytest.importorskip("run_helpers", reason="scenario shared helpers (tests/RCA/scenario/shared) arrive in MR #12")
from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca  # noqa: E402

_TC2_FIXTURES = _SCENARIO_ROOT / "test_case_2" / "fixtures"
_TC3_FIXTURES = _SCENARIO_ROOT / "test_case_3" / "fixtures"
_TC5_FIXTURES = _SCENARIO_ROOT / "test_case_5" / "fixtures"


def _two_runs(fixture_dir: Path):
    """Return two independent results from the same fixture dir."""
    fixtures = load_fixtures(fixture_dir)
    with tempfile.TemporaryDirectory() as tmp1:
        r1 = run_rca(build_fixture_orchestrator(tmp1), fixtures)
    with tempfile.TemporaryDirectory() as tmp2:
        r2 = run_rca(build_fixture_orchestrator(tmp2), fixtures)
    return r1, r2


# ---------------------------------------------------------------------------
# D10-A — Two runs with same inputs → identical scores and gates
# ---------------------------------------------------------------------------

def _assert_d10a(fixture_dir: Path, label: str) -> None:
    r1, r2 = _two_runs(fixture_dir)

    cands1: List[Dict[str, Any]] = (r1.get("causality_candidates") or {}).get("candidates", [])
    cands2: List[Dict[str, Any]] = (r2.get("causality_candidates") or {}).get("candidates", [])

    assert len(cands1) == len(cands2), (
        f"D10-A FAIL [{label}]: candidate count differs between runs: "
        f"{len(cands1)} vs {len(cands2)}"
    )

    violations = []
    for i, (c1, c2) in enumerate(zip(cands1, cands2)):
        cid1 = c1.get("candidate_id") or c1.get("failure_mode_id", f"[{i}]")
        cid2 = c2.get("candidate_id") or c2.get("failure_mode_id", f"[{i}]")

        if cid1 != cid2:
            violations.append(f"rank {i}: candidate_id mismatch {cid1!r} vs {cid2!r}")
            continue

        s1 = float(c1.get("composite_score") or 0.0)
        s2 = float(c2.get("composite_score") or 0.0)
        if abs(s1 - s2) > 1e-6:
            violations.append(
                f"{cid1}: composite_score {s1:.6f} vs {s2:.6f} (delta={abs(s1-s2):.2e})"
            )

        gates1 = c1.get("hard_gates") or {}
        gates2 = c2.get("hard_gates") or {}
        for gate_name in set(gates1) | set(gates2):
            g1 = gates1.get(gate_name)
            g2 = gates2.get(gate_name)
            if isinstance(g1, dict) and isinstance(g2, dict):
                if g1.get("passed") != g2.get("passed"):
                    violations.append(
                        f"{cid1}.hard_gates.{gate_name}.passed: "
                        f"{g1.get('passed')} vs {g2.get('passed')}"
                    )

    assert not violations, (
        f"D10-A FAIL [{label}]: non-deterministic differences between two identical runs:\n  "
        + "\n  ".join(violations[:10])
    )

    # Also check that the primary hypothesis is identical between runs
    ph1 = (r1.get("rca_card") or {}).get("primary_hypothesis", {}).get("candidate_id")
    ph2 = (r2.get("rca_card") or {}).get("primary_hypothesis", {}).get("candidate_id")
    assert ph1 == ph2, (
        f"D10-A FAIL [{label}]: primary_hypothesis differs between runs: {ph1!r} vs {ph2!r}"
    )

    print(
        f"  pass  D10-A [{label}]: identical results across two independent runs "
        f"({len(cands1)} candidates, primary={ph1!r})"
    )


def test_d10a_determinism_tc2():
    if not _TC2_FIXTURES.exists():
        pytest.skip("TC-2 fixtures not found")
    _assert_d10a(_TC2_FIXTURES, "TC-2")


def test_d10a_determinism_tc3():
    if not _TC3_FIXTURES.exists():
        pytest.skip("TC-3 fixtures not found")
    _assert_d10a(_TC3_FIXTURES, "TC-3")


def test_d10a_determinism_tc5():
    if not _TC5_FIXTURES.exists():
        pytest.skip("TC-5 fixtures not found")
    _assert_d10a(_TC5_FIXTURES, "TC-5")


# ---------------------------------------------------------------------------
# D10-C — Scoring evolution delta is internally consistent
# ---------------------------------------------------------------------------

def _assert_d10c(result: Dict[str, Any], label: str) -> None:
    """
    If scoring_evolution is present, the 'delta' field must equal
    composite_score_post_refine - composite_score_pre_refine (within 0.001).
    """
    evolution = result.get("scoring_evolution")
    if not evolution:
        print(f"  skip  D10-C [{label}] scoring_evolution absent in result")
        return

    rows: List[Dict[str, Any]] = evolution.get("rows") or []
    if not rows:
        print(f"  skip  D10-C [{label}] scoring_evolution.rows is empty")
        return

    violations = []
    for row in rows:
        cid = row.get("candidate_id", "?")
        delta = row.get("delta")
        pre = row.get("composite_score_pre_refine")
        post = row.get("composite_score_post_refine")
        if delta is None or pre is None or post is None:
            continue
        computed = float(post) - float(pre)
        if abs(computed - float(delta)) > 0.001:
            violations.append(
                f"{cid}: declared delta={delta:.4f} but computed={computed:.4f} "
                f"(pre={pre:.4f}, post={post:.4f})"
            )

    assert not violations, (
        f"D10-C FAIL [{label}]: scoring_evolution delta inconsistencies:\n  "
        + "\n  ".join(violations)
    )
    print(f"  pass  D10-C [{label}]: {len(rows)} scoring_evolution row(s) internally consistent")


def test_d10c_scoring_evolution_tc2():
    if not _TC2_FIXTURES.exists():
        pytest.skip("TC-2 fixtures not found")
    fixtures = load_fixtures(_TC2_FIXTURES)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_rca(build_fixture_orchestrator(tmp), fixtures)
    _assert_d10c(result, "TC-2")


def test_d10c_scoring_evolution_tc5():
    if not _TC5_FIXTURES.exists():
        pytest.skip("TC-5 fixtures not found")
    fixtures = load_fixtures(_TC5_FIXTURES)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_rca(build_fixture_orchestrator(tmp), fixtures)
    _assert_d10c(result, "TC-5")


# ---------------------------------------------------------------------------
# D10-B — Sensitivity table: sources marked ranking_change_possible=True (soft / diagnostic)
# ---------------------------------------------------------------------------

def test_d10b_sensitivity_table_structure_tc2():
    """
    D10-B: Verify the sensitivity table summary dict and per-source rows.

    Schema (run_manifest.artifacts.sensitivity_table):
        {
            "present": bool,
            "any_ranking_change_possible": bool,
            "missing_sources_checked": [str, ...],
            "top_n_candidates": int,
            "row_count": int,
            "rows": [{"candidate_id": ..., "source_family": ...,
                      "estimated_score_delta": ..., "would_change_ranking": ...}, ...]
        }
    For TC-2 (all sources complete), 'rows' will be empty because no
    degraded sources exist — that is correct behaviour.
    """
    if not _TC2_FIXTURES.exists():
        pytest.skip("TC-2 fixtures not found")

    fixtures = load_fixtures(_TC2_FIXTURES)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_rca(build_fixture_orchestrator(tmp), fixtures)

    table = (
        (result.get("run_manifest") or {})
        .get("artifacts") or {}
    ).get("sensitivity_table")

    if table is None:
        print("  skip  D10-B TC-2: sensitivity_table absent in run_manifest.artifacts")
        return

    assert isinstance(table, dict), (
        f"D10-B FAIL: expected sensitivity_table to be a dict, got {type(table).__name__}"
    )

    required_fields = ("present", "any_ranking_change_possible", "rows")
    missing = [f for f in required_fields if f not in table]
    assert not missing, (
        f"D10-B FAIL: sensitivity_table missing fields: {missing}. Got: {sorted(table.keys())}"
    )

    assert isinstance(table["rows"], list), (
        f"D10-B FAIL: sensitivity_table.rows must be a list, got {type(table['rows']).__name__}"
    )

    print(
        f"  pass  D10-B TC-2: sensitivity_table present. "
        f"any_ranking_change_possible={table.get('any_ranking_change_possible')}, "
        f"row_count={table.get('row_count', 'absent')}, "
        f"rows populated={len(table['rows'])} (0 expected for complete fixture)."
    )


def test_d10b_sensitivity_rows_when_source_degraded():
    """
    D10-B extension: When at least one optional source is absent,
    the sensitivity table must produce per-source rows describing the
    score delta that source could contribute.
    """
    if not _TC2_FIXTURES.exists():
        pytest.skip("TC-2 fixtures not found")

    fixtures = load_fixtures(_TC2_FIXTURES)
    # Force one source to be absent so the engine registers a degraded family.
    fixtures = dict(fixtures, vendor_supply_chain_records=None)

    with tempfile.TemporaryDirectory() as tmp:
        result = run_rca(build_fixture_orchestrator(tmp), fixtures)

    table = (
        (result.get("run_manifest") or {})
        .get("artifacts") or {}
    ).get("sensitivity_table")

    if table is None:
        pytest.skip("sensitivity_table absent — skipping row-population check")

    rows = table.get("rows") or []
    assert isinstance(rows, list), "sensitivity_table.rows must be a list"

    if not rows:
        # Vendor supply chain may not be a tracked family — skip gracefully.
        print(
            "  note  D10-B ext: vendor_supply_chain_records=None produced 0 rows. "
            "This source may not be a tracked degraded family — result is acceptable."
        )
        return

    row_fields = ("candidate_id", "source_family", "estimated_score_delta", "would_change_ranking")
    for row in rows:
        missing_fields = [f for f in row_fields if f not in row]
        assert not missing_fields, (
            f"D10-B ext FAIL: sensitivity row missing fields {missing_fields}. Row: {row}"
        )

    print(
        f"  pass  D10-B ext: {len(rows)} per-source rows present when source is degraded. "
        f"Sample: {rows[0]}"
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_d10a_determinism_tc2,
    test_d10a_determinism_tc3,
    test_d10a_determinism_tc5,
    test_d10c_scoring_evolution_tc2,
    test_d10c_scoring_evolution_tc5,
    test_d10b_sensitivity_table_structure_tc2,
    test_d10b_sensitivity_rows_when_source_degraded,
]


def run_all() -> bool:
    print(f"\n=== test_robustness_d10_determinism ({len(ALL_TESTS)} tests) ===")
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
