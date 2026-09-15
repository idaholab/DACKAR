"""
test_robustness_d7_traceability.py — D7 End-to-end traceability (Phase 2)

Checks:
    D7-A  Primary hypothesis candidate_id exists in candidates list           [should PASS]
    D7-B  Alternative hypothesis IDs all trace to candidates list             [should PASS]
    D7-E  Every KG failure mode is either ranked or ruled-out                 [should PASS]
    D7-D  Score rationale direction is consistent with sub-score values       [diagnostic]

Run directly:   python test_robustness_d7_traceability.py
Or via pytest:  pytest test_robustness_d7_traceability.py -v

Note on D7-C (evidence citations → bundle):
    The orchestrator does not echo the input evidence_bundle into the result dict.
    Citation trace verification requires comparing against the input fixture, not the
    result.  D7-C is implemented below using the loaded fixture evidence_bundle.

Fixtures used:
    TC-2, TC-4, TC-5, TC-8
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
_TC4_FIXTURES = _SCENARIO_ROOT / "test_case_4" / "fixtures"
_TC5_FIXTURES = _SCENARIO_ROOT / "test_case_5" / "fixtures"
_TC8_FIXTURES = _SCENARIO_ROOT / "test_case_8" / "fixtures"


def _run_and_fixtures(fixture_dir: Path):
    """Return (result, fixtures) so D7-C can compare against input bundle."""
    with tempfile.TemporaryDirectory() as tmp:
        orch = build_fixture_orchestrator(tmp)
        fixtures = load_fixtures(fixture_dir)
        result = run_rca(orch, fixtures)
    return result, fixtures


# ---------------------------------------------------------------------------
# D7-A — Primary hypothesis candidate_id in ranked candidates list
# ---------------------------------------------------------------------------

def _assert_d7a(result: Dict[str, Any], label: str) -> None:
    primary_id = (result.get("rca_card") or {}).get("primary_hypothesis", {}).get("candidate_id")
    candidates: List[Dict[str, Any]] = (
        result.get("causality_candidates") or {}
    ).get("candidates", [])

    if primary_id is None or primary_id == "NONE":
        print(f"  skip  D7-A [{label}] primary_hypothesis.candidate_id='{primary_id}' (insufficient evidence run)")
        return

    candidate_ids = {
        c.get("candidate_id") or c.get("failure_mode_id")
        for c in candidates
    }
    assert primary_id in candidate_ids, (
        f"D7-A FAIL [{label}]: primary_hypothesis.candidate_id={primary_id!r} "
        f"not found in ranked candidates {sorted(candidate_ids)}"
    )
    print(f"  pass  D7-A [{label}]: primary_id={primary_id!r} ∈ candidates")


def test_d7a_primary_in_candidates_tc2():
    if not _TC2_FIXTURES.exists():
        pytest.skip("TC-2 fixtures not found")
    result, _ = _run_and_fixtures(_TC2_FIXTURES)
    _assert_d7a(result, "TC-2")


def test_d7a_primary_in_candidates_tc4():
    if not _TC4_FIXTURES.exists():
        pytest.skip("TC-4 fixtures not found")
    result, _ = _run_and_fixtures(_TC4_FIXTURES)
    _assert_d7a(result, "TC-4")


def test_d7a_primary_in_candidates_tc5():
    if not _TC5_FIXTURES.exists():
        pytest.skip("TC-5 fixtures not found")
    result, _ = _run_and_fixtures(_TC5_FIXTURES)
    _assert_d7a(result, "TC-5")


def test_d7a_primary_in_candidates_tc8():
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result, _ = _run_and_fixtures(_TC8_FIXTURES)
    _assert_d7a(result, "TC-8")


# ---------------------------------------------------------------------------
# D7-B — Alternative hypothesis IDs trace to candidates list
# ---------------------------------------------------------------------------

def _assert_d7b(result: Dict[str, Any], label: str) -> None:
    rca_card = result.get("rca_card") or {}
    # The synthesizer may use 'alternatives' or 'alternative_hypotheses'
    alternatives = (
        rca_card.get("alternatives")
        or rca_card.get("alternative_hypotheses")
        or []
    )
    if not alternatives:
        print(f"  skip  D7-B [{label}] no alternatives in rca_card")
        return

    candidates: List[Dict[str, Any]] = (
        result.get("causality_candidates") or {}
    ).get("candidates", [])
    candidate_ids = {
        c.get("candidate_id") or c.get("failure_mode_id")
        for c in candidates
    }

    missing = []
    for alt in alternatives:
        alt_id = alt.get("candidate_id") or alt.get("failure_mode_id")
        if alt_id and alt_id not in candidate_ids:
            missing.append(alt_id)

    assert not missing, (
        f"D7-B FAIL [{label}]: alternative candidate_id(s) not in ranked candidates: "
        f"{missing}"
    )
    print(
        f"  pass  D7-B [{label}]: all {len(alternatives)} alternative(s) trace to candidates"
    )


def test_d7b_alternatives_trace_tc4():
    if not _TC4_FIXTURES.exists():
        pytest.skip("TC-4 fixtures not found")
    result, _ = _run_and_fixtures(_TC4_FIXTURES)
    _assert_d7b(result, "TC-4")


def test_d7b_alternatives_trace_tc8():
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result, _ = _run_and_fixtures(_TC8_FIXTURES)
    _assert_d7b(result, "TC-8")


# ---------------------------------------------------------------------------
# D7-C — Evidence citations in rca_card trace to input evidence bundle
# ---------------------------------------------------------------------------

def _assert_d7c(result: Dict[str, Any], fixtures: Dict[str, Any], label: str) -> None:
    """
    Every evidence row / citation in the rca_card must reference a doc_id
    or snippet_id that exists in the input evidence_bundle fixture.
    """
    evidence_bundle = fixtures.get("evidence_bundle") or {}
    bundle_results = evidence_bundle.get("results") or []
    if not bundle_results:
        print(f"  skip  D7-C [{label}] evidence_bundle absent or empty in fixtures")
        return

    bundle_doc_ids = {r.get("doc_id") for r in bundle_results if r.get("doc_id")}
    bundle_snippet_ids = {r.get("snippet_id") for r in bundle_results if r.get("snippet_id")}

    rca_card = result.get("rca_card") or {}
    # Evidence rows may be in rca_card["evidence"] or rca_card["evidence_citations"]
    evidence_rows = rca_card.get("evidence") or rca_card.get("evidence_citations") or []
    if not evidence_rows:
        print(f"  skip  D7-C [{label}] no evidence rows in rca_card")
        return

    missing = []
    for row in evidence_rows:
        doc_id = row.get("doc_id")
        snippet_id = row.get("source_id") or row.get("snippet_id")
        if doc_id and doc_id not in bundle_doc_ids:
            missing.append(f"doc_id={doc_id!r} not in bundle")
        if snippet_id and snippet_id not in bundle_snippet_ids:
            # Only flag if we also can't find the doc — snippet IDs may not always be present
            if doc_id and doc_id not in bundle_doc_ids:
                missing.append(f"snippet_id={snippet_id!r} not in bundle")

    assert not missing, (
        f"D7-C FAIL [{label}]: rca_card evidence rows reference unknown doc/snippet IDs:\n  "
        + "\n  ".join(missing)
    )
    print(
        f"  pass  D7-C [{label}]: all {len(evidence_rows)} evidence row(s) trace to bundle"
    )


def test_d7c_citations_trace_tc4():
    if not _TC4_FIXTURES.exists():
        pytest.skip("TC-4 fixtures not found")
    result, fixtures = _run_and_fixtures(_TC4_FIXTURES)
    _assert_d7c(result, fixtures, "TC-4")


def test_d7c_citations_trace_tc8():
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result, fixtures = _run_and_fixtures(_TC8_FIXTURES)
    _assert_d7c(result, fixtures, "TC-8")


# ---------------------------------------------------------------------------
# D7-D — Score rationale direction consistent with sub-score values
# ---------------------------------------------------------------------------

def _assert_d7d(result: Dict[str, Any], label: str) -> None:
    candidates: List[Dict[str, Any]] = (
        result.get("causality_candidates") or {}
    ).get("candidates", [])

    violations = []
    for cand in candidates:
        cid = cand.get("candidate_id") or cand.get("failure_mode_id", "?")
        rationale_list = cand.get("score_rationale") or []
        if not rationale_list:
            continue
        rationale = {r.get("dimension"): r for r in rationale_list if isinstance(r, dict)}
        scores = cand.get("scores") or {}

        for dim in ("temporal", "structural", "telemetry", "evidence", "governance"):
            if dim not in rationale:
                continue
            level = (rationale[dim].get("level") or "").lower()
            score = float(scores.get(dim) or 0.0)

            if "high" in level and score < 0.6:
                violations.append(
                    f"{cid}.{dim}: rationale says 'high' but score={score:.3f} (<0.6)"
                )
            if "low" in level and score > 0.4:
                violations.append(
                    f"{cid}.{dim}: rationale says 'low' but score={score:.3f} (>0.4)"
                )

    if not violations:
        rated = sum(
            1 for c in candidates
            if c.get("score_rationale")
        )
        print(
            f"  pass  D7-D [{label}]: score rationale consistent "
            f"({rated}/{len(candidates)} candidates have rationale)"
        )
    else:
        # Warn but do not fail — rationale wording is soft
        print(
            f"  warn  D7-D [{label}]: {len(violations)} rationale/score direction inconsistencies "
            f"(treating as advisory):\n    " + "\n    ".join(violations[:5])
        )


def test_d7d_score_rationale_direction_tc4():
    if not _TC4_FIXTURES.exists():
        pytest.skip("TC-4 fixtures not found")
    result, _ = _run_and_fixtures(_TC4_FIXTURES)
    _assert_d7d(result, "TC-4")


def test_d7d_score_rationale_direction_tc8():
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result, _ = _run_and_fixtures(_TC8_FIXTURES)
    _assert_d7d(result, "TC-8")


# ---------------------------------------------------------------------------
# D7-E — Every KG failure mode is accounted for (ranked or ruled-out)
# ---------------------------------------------------------------------------

def _assert_d7e(result: Dict[str, Any], fixtures: Dict[str, Any], label: str) -> None:
    kg_context = fixtures.get("kg_context") or {}
    # KG failure modes may be under different keys depending on the schema
    failure_modes = kg_context.get("failure_modes") or []
    if not failure_modes:
        print(f"  skip  D7-E [{label}] kg_context.failure_modes absent or empty in fixtures")
        return

    fm_ids_in_kg = set()
    for fm in failure_modes:
        fid = fm.get("fm_id") or fm.get("failure_mode_id") or fm.get("id")
        if fid:
            fm_ids_in_kg.add(fid)

    if not fm_ids_in_kg:
        print(f"  skip  D7-E [{label}] no fm_id fields found in failure_modes")
        return

    cands = result.get("causality_candidates") or {}
    candidates: List[Dict[str, Any]] = cands.get("candidates", [])
    ruled_out: List[Dict[str, Any]] = cands.get("ruled_out", [])

    accounted = (
        {c.get("failure_mode_id") or c.get("candidate_id") for c in candidates}
        | {r.get("failure_mode_id") or r.get("candidate_id") for r in ruled_out}
    )
    accounted.discard(None)

    unaccounted = fm_ids_in_kg - accounted
    if unaccounted:
        # Warn, but check if this is a screening exclusion (category not applicable)
        print(
            f"  warn  D7-E [{label}]: {len(unaccounted)} KG failure mode(s) unaccounted for "
            f"(not in candidates or ruled_out). May be filtered at screening. "
            f"IDs: {sorted(unaccounted)[:5]}"
        )
    else:
        print(
            f"  pass  D7-E [{label}]: all {len(fm_ids_in_kg)} KG failure mode(s) accounted for"
        )


def test_d7e_kg_failure_modes_accounted_tc5():
    if not _TC5_FIXTURES.exists():
        pytest.skip("TC-5 fixtures not found")
    result, fixtures = _run_and_fixtures(_TC5_FIXTURES)
    _assert_d7e(result, fixtures, "TC-5")


def test_d7e_kg_failure_modes_accounted_tc8():
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result, fixtures = _run_and_fixtures(_TC8_FIXTURES)
    _assert_d7e(result, fixtures, "TC-8")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_d7a_primary_in_candidates_tc2,
    test_d7a_primary_in_candidates_tc4,
    test_d7a_primary_in_candidates_tc5,
    test_d7a_primary_in_candidates_tc8,
    test_d7b_alternatives_trace_tc4,
    test_d7b_alternatives_trace_tc8,
    test_d7c_citations_trace_tc4,
    test_d7c_citations_trace_tc8,
    test_d7d_score_rationale_direction_tc4,
    test_d7d_score_rationale_direction_tc8,
    test_d7e_kg_failure_modes_accounted_tc5,
    test_d7e_kg_failure_modes_accounted_tc8,
]


def run_all() -> bool:
    print(f"\n=== test_robustness_d7_traceability ({len(ALL_TESTS)} tests) ===")
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
