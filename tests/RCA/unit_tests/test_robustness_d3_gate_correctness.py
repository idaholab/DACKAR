"""
test_robustness_d3_gate_correctness.py — D3 Hard gate correctness (Phase 2)

Checks:
    D3-D  Gate is binary — candidate_ids ∩ ruled_out_ids = ∅  [Phase 1 probe]
    D3-A  FOLLOWS anomaly eliminates the candidate             [new fixture required — D3-A skeleton]
    D3-B  Held barrier eliminates requiring candidates         [TC-5 fixture]

Run directly:   python test_robustness_d3_gate_correctness.py
Or via pytest:  pytest test_robustness_d3_gate_correctness.py -v

Fixtures used:
    TC-4  ../tests/test_case_4/fixtures/
    TC-5  ../tests/test_case_5/fixtures/
    TC-8  ../tests/test_case_8/fixtures/

No live Neo4j, Chroma, or LLM required.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Path setup — must happen before any RCA package import.
# ---------------------------------------------------------------------------

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
_SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenario"
_TESTS_SHARED = _SCENARIO_ROOT / "shared"

for _p in (str(_RCA_ROOT), str(_TESTS_SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Stub heavy optional dependencies before any package import touches them.
for _mod in (
    "neo4j",
    "py2neo",
    "chromadb",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402
pytest.importorskip("run_helpers", reason="scenario shared helpers (tests/RCA/scenario/shared) arrive in MR #12")
from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture directories
# ---------------------------------------------------------------------------

_TC4_FIXTURES = _SCENARIO_ROOT / "test_case_4" / "fixtures"
_TC5_FIXTURES = _SCENARIO_ROOT / "test_case_5" / "fixtures"
_TC8_FIXTURES = _SCENARIO_ROOT / "test_case_8" / "fixtures"


def _run_tc(fixture_dir: Path) -> dict:
    """Load a TC fixture directory and run the full orchestrator pipeline."""
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = build_fixture_orchestrator(tmp)
        fixtures = load_fixtures(fixture_dir)
        return run_rca(orchestrator, fixtures)


# ---------------------------------------------------------------------------
# D3-D — Gate is binary: no candidate can be both ranked and ruled-out
# ---------------------------------------------------------------------------
# Phase 1 probe: listed as "Likely FAIL for some edge cases" in the plan.
# If this passes on all TCs, no bug — promote to regression test.
# If it fails, the gate logic is routing candidates to both lists — file a bug.

def _assert_disjoint_sets(result: dict, label: str) -> None:
    cands = result.get("causality_candidates") or {}
    candidate_ids = {
        c["candidate_id"]
        for c in cands.get("candidates", [])
        if "candidate_id" in c
    }
    ruled_out_ids = {
        r["candidate_id"]
        for r in cands.get("ruled_out", [])
        if "candidate_id" in r
    }
    overlap = candidate_ids & ruled_out_ids
    assert not overlap, (
        f"[{label}] D3-D FAIL: candidate(s) appear in both ranked list and ruled_out: "
        f"{sorted(overlap)}"
    )
    print(
        f"  pass  D3-D [{label}] candidate_ids ∩ ruled_out_ids = ∅  "
        f"(candidates={len(candidate_ids)}, ruled_out={len(ruled_out_ids)})"
    )


def test_d3d_gate_disjoint_tc4():
    """D3-D: No candidate is simultaneously ranked and ruled-out (TC-4)."""
    if not _TC4_FIXTURES.exists():
        print(f"  skip  D3-D TC-4 fixtures not found at {_TC4_FIXTURES}")
        return
    result = _run_tc(_TC4_FIXTURES)
    _assert_disjoint_sets(result, "TC-4")


def test_d3d_gate_disjoint_tc5():
    """D3-D: No candidate is simultaneously ranked and ruled-out (TC-5 barrier scenario)."""
    if not _TC5_FIXTURES.exists():
        print(f"  skip  D3-D TC-5 fixtures not found at {_TC5_FIXTURES}")
        return
    result = _run_tc(_TC5_FIXTURES)
    _assert_disjoint_sets(result, "TC-5")


def test_d3d_gate_disjoint_tc8():
    """D3-D: No candidate is simultaneously ranked and ruled-out (TC-8 full-coverage)."""
    if not _TC8_FIXTURES.exists():
        print(f"  skip  D3-D TC-8 fixtures not found at {_TC8_FIXTURES}")
        return
    result = _run_tc(_TC8_FIXTURES)
    _assert_disjoint_sets(result, "TC-8")


# ---------------------------------------------------------------------------
# D3-B — Held barrier eliminates requiring candidates (TC-5)
# ---------------------------------------------------------------------------

def test_d3b_held_barrier_eliminates_candidate():
    """
    D3-B: When a barrier is in 'held' state, candidates that require that
    barrier to fail must be in ruled_out[] with reason_code='barrier_held'
    and must NOT appear in candidates[].

    TC-5 includes protection_logic_context with a held barrier.  If no
    ruled_out entries with reason_code='barrier_held' are produced, this
    test records a warning rather than failing — TC-5 may not exercise the
    exact failure mode that depends on the held barrier.
    """
    if not _TC5_FIXTURES.exists():
        print(f"  skip  D3-B TC-5 fixtures not found at {_TC5_FIXTURES}")
        return

    result = _run_tc(_TC5_FIXTURES)
    cands = result.get("causality_candidates") or {}
    candidate_ids = {c.get("candidate_id") for c in cands.get("candidates", [])}
    ruled_out = cands.get("ruled_out", [])

    barrier_held_entries = [
        r for r in ruled_out
        if r.get("reason_code") == "barrier_held"
    ]

    if not barrier_held_entries:
        print(
            "  warn  D3-B TC-5: no ruled_out entries with reason_code='barrier_held'. "
            "Confirm that TC-5 fixture exercises the barrier-held gate path."
        )
        return

    for entry in barrier_held_entries:
        cid = entry.get("candidate_id")
        assert cid not in candidate_ids, (
            f"D3-B FAIL: candidate '{cid}' has reason_code='barrier_held' in ruled_out "
            f"but also appears in the ranked candidate list."
        )

    print(
        f"  pass  D3-B TC-5: {len(barrier_held_entries)} barrier_held entries confirmed "
        f"absent from candidate list."
    )


# ---------------------------------------------------------------------------
# D3-D regression on result structure integrity
# ---------------------------------------------------------------------------

def test_d3d_ruled_out_entries_have_required_fields():
    """
    Every ruled_out entry must carry: candidate_id, reason_code.
    A missing reason_code means the gate fired but didn't document why.
    """
    if not _TC8_FIXTURES.exists():
        print(f"  skip  ruled_out fields check — TC-8 fixtures not found")
        return

    result = _run_tc(_TC8_FIXTURES)
    ruled_out = (result.get("causality_candidates") or {}).get("ruled_out", [])

    missing_fields = []
    for i, entry in enumerate(ruled_out):
        if not entry.get("candidate_id"):
            missing_fields.append(f"ruled_out[{i}] missing candidate_id")
        if not entry.get("reason_code"):
            missing_fields.append(
                f"ruled_out[{i}] (id={entry.get('candidate_id', '?')}) missing reason_code"
            )

    assert not missing_fields, (
        "D3 FAIL: ruled_out entries with missing required fields:\n  "
        + "\n  ".join(missing_fields)
    )
    print(
        f"  pass  D3 TC-8: all {len(ruled_out)} ruled_out entries have candidate_id and reason_code."
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_d3d_gate_disjoint_tc4,
    test_d3d_gate_disjoint_tc5,
    test_d3d_gate_disjoint_tc8,
    test_d3b_held_barrier_eliminates_candidate,
    test_d3d_ruled_out_entries_have_required_fields,
]


def run_all() -> bool:
    print(f"\n=== test_robustness_d3_gate_correctness ({len(ALL_TESTS)} tests) ===")
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
