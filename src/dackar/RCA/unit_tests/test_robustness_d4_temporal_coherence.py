"""
test_robustness_d4_temporal_coherence.py — D4 Temporal coherence (Phase 2)

Checks:
    D4-A   Allen relation ↔ chain_position consistency
    D4-C   novel_pattern ↔ recurrence_count consistency in TSKR patterns
    D4-D   Earliest causal onset precedes event timestamp_start
    D4-B   Allen blend discriminates OVERLAPS vs PRECEDES (formula fix verified)
    D4-B2  Full-pipeline Allen blend discrimination (requires P7 fixtures)

Run directly:   python test_robustness_d4_temporal_coherence.py
Or via pytest:  pytest test_robustness_d4_temporal_coherence.py -v

Fixtures used:
    TC-2  ../tests/test_case_2/fixtures/   (temporal timeline scenario)
    TC-3  ../tests/test_case_3/fixtures/   (ranking inversion)
    TC-4  ../tests/test_case_4/fixtures/   (SOE-rich, most Allen-resolved candidates)
    P7    ../tests/fixtures_robustness/allen_overlaps_fixture/  (pending)
          ../tests/fixtures_robustness/allen_precedes_fixture/  (pending)
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

sys.path.insert(0, str(_RCA_ROOT / "orchestrators"))
from causality_engine_v32 import RuleBasedCausalityEngineV32  # noqa: E402

_TC2_FIXTURES = _RCA_ROOT / "tests" / "test_case_2" / "fixtures"
_TC3_FIXTURES = _RCA_ROOT / "tests" / "test_case_3" / "fixtures"
_TC4_FIXTURES = _RCA_ROOT / "tests" / "test_case_4" / "fixtures"
_ALLEN_OVERLAPS = _RCA_ROOT / "tests" / "fixtures_robustness" / "allen_overlaps_fixture"
_ALLEN_PRECEDES = _RCA_ROOT / "tests" / "fixtures_robustness" / "allen_precedes_fixture"


def _run(fixture_dir: Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        orch = build_fixture_orchestrator(tmp)
        return run_rca(orch, load_fixtures(fixture_dir))


# ---------------------------------------------------------------------------
# D4-A — Allen relation ↔ chain_position consistency
# ---------------------------------------------------------------------------

_CAUSAL_RELATIONS = {"OVERLAPS", "CONTAINS", "PRECEDES"}


def _assert_d4a(result: Dict[str, Any], label: str) -> None:
    """
    Invariant:
        FOLLOWS  → chain_position must be "consequence"
        initiating → allen_relation must be causal (OVERLAPS/CONTAINS/PRECEDES) or None
    """
    candidates: List[Dict[str, Any]] = (
        result.get("causality_candidates") or {}
    ).get("candidates", [])

    violations = []
    for cand in candidates:
        cid = cand.get("candidate_id") or cand.get("failure_mode_id", "?")
        allen = (cand.get("scores") or {}).get("allen_relation")
        chain_pos = cand.get("chain_position")

        if allen == "FOLLOWS" and chain_pos != "consequence":
            violations.append(
                f"{cid}: allen_relation=FOLLOWS but chain_position={chain_pos!r} "
                f"(expected 'consequence')"
            )
        if chain_pos == "initiating" and allen is not None and allen not in _CAUSAL_RELATIONS:
            violations.append(
                f"{cid}: chain_position=initiating but allen_relation={allen!r} "
                f"(must be causal or None)"
            )

    assert not violations, (
        f"[{label}] D4-A violations:\n  " + "\n  ".join(violations)
    )
    allen_count = sum(
        1 for c in candidates
        if (c.get("scores") or {}).get("allen_relation") is not None
    )
    print(
        f"  pass  D4-A [{label}] Allen↔chain_position consistent "
        f"({len(candidates)} candidates, {allen_count} with Allen relation)"
    )


def test_d4a_allen_chain_position_tc2():
    if not _TC2_FIXTURES.exists():
        pytest.skip(f"TC-2 fixtures not found")
    _assert_d4a(_run(_TC2_FIXTURES), "TC-2")


def test_d4a_allen_chain_position_tc4():
    if not _TC4_FIXTURES.exists():
        pytest.skip(f"TC-4 fixtures not found")
    _assert_d4a(_run(_TC4_FIXTURES), "TC-4")


# ---------------------------------------------------------------------------
# D4-C — novel_pattern ↔ recurrence_count consistency
# ---------------------------------------------------------------------------

def _assert_d4c(result: Dict[str, Any], label: str) -> None:
    """
    Invariant:
        novel_pattern=True  → recurrence_count == 0
        recurrence_count > 0 → novel_pattern is False/absent
    """
    tskr = result.get("tskr_patterns") or {}
    patterns: List[Dict[str, Any]] = tskr.get("patterns", [])

    if not patterns:
        print(f"  skip  D4-C [{label}] no tskr_patterns.patterns to check")
        return

    violations = []
    for pat in patterns:
        pid = pat.get("signal_id") or pat.get("component_id", "?")
        novel = pat.get("novel_pattern", False)
        recurrence = pat.get("recurrence_count", 0) or 0

        if novel and recurrence > 0:
            violations.append(
                f"{pid}: novel_pattern=True but recurrence_count={recurrence} "
                f"(a novel pattern cannot recur)"
            )
        if recurrence > 0 and novel:
            violations.append(
                f"{pid}: recurrence_count={recurrence} but novel_pattern=True "
                f"(a recurrent pattern is not novel)"
            )

    assert not violations, (
        f"[{label}] D4-C violations:\n  " + "\n  ".join(violations)
    )
    print(
        f"  pass  D4-C [{label}] novel_pattern↔recurrence_count consistent "
        f"({len(patterns)} patterns)"
    )


def test_d4c_novel_pattern_consistency_tc2():
    if not _TC2_FIXTURES.exists():
        pytest.skip(f"TC-2 fixtures not found")
    _assert_d4c(_run(_TC2_FIXTURES), "TC-2")


def test_d4c_novel_pattern_consistency_tc4():
    if not _TC4_FIXTURES.exists():
        pytest.skip(f"TC-4 fixtures not found")
    _assert_d4c(_run(_TC4_FIXTURES), "TC-4")


# ---------------------------------------------------------------------------
# D4-D — Earliest causal onset precedes event timestamp_start
# ---------------------------------------------------------------------------

def test_d4d_earliest_onset_precedes_event_tc4():
    """
    D4-D: If the Allen relation map records an earliest_causal_onset, that
    onset must be ≤ event.timestamp_start.  An onset *after* the event
    would indicate a consequence, not a cause — a contradiction.
    """
    if not _TC4_FIXTURES.exists():
        pytest.skip(f"TC-4 fixtures not found")

    result = _run(_TC4_FIXTURES)
    allen_map = (
        (result.get("run_manifest") or {})
        .get("artifacts") or {}
    ).get("allen_relation_map") or {}
    earliest = allen_map.get("earliest_causal_onset")

    if earliest is None:
        print("  skip  D4-D TC-4: allen_relation_map.earliest_causal_onset absent — skip")
        return

    event = result.get("run_context", {}).get("event") or {}
    event_start = event.get("timestamp_start")

    if event_start is None:
        print("  skip  D4-D TC-4: event.timestamp_start absent in run_context — skip")
        return

    assert earliest <= event_start, (
        f"D4-D FAIL: earliest_causal_onset={earliest!r} > event.timestamp_start={event_start!r}. "
        "A causal onset cannot be later than the event it causes."
    )
    print(f"  pass  D4-D TC-4: earliest_causal_onset={earliest!r} ≤ event_start={event_start!r}")


# ---------------------------------------------------------------------------
# D4-B — Allen blend discriminates OVERLAPS vs PRECEDES  [KNOWN-FAIL]
# ---------------------------------------------------------------------------

def test_d4b_allen_blend_formula_direct():
    """
    D4-B (direct unit test): _apply_allen_temporal_blend() must produce
    distinct temporal scores for OVERLAPS (high allen_base_score) vs.
    PRECEDES (low allen_base_score) when starting from the same TSKR baseline.

    This directly validates the formula fix: the old max(old, blend) clamp
    prevented discrimination when both blends fell below the TSKR floor.
    The corrected formula is a true weighted average with no floor clamp.
    """
    TSKR_BASELINE = 0.85  # high TSKR-derived temporal score (common floor)
    ALLEN_HIGH = 0.92     # strong Allen signal (e.g. tight OVERLAPS window)
    ALLEN_LOW  = 0.55     # weak Allen signal (e.g. loose PRECEDES with gap)
    ALPHA = 0.25
    weights = {"temporal": 0.25, "structural": 0.25, "telemetry": 0.25, "evidence": 0.25}

    def _make_candidate(cid: str, allen_score: float, relation: str) -> dict:
        return {
            "component_id": cid,
            "composite_score": 0.70,
            "quality_multiplier": 1.0,
            "scores": {
                "temporal": TSKR_BASELINE,
                "composite_raw": 0.70,
            },
        }, {cid: allen_score}, {cid: relation}

    cand_high, scores_high, rel_high = _make_candidate("CMP-HIGH", ALLEN_HIGH, "overlaps")
    cand_low,  scores_low,  rel_low  = _make_candidate("CMP-LOW",  ALLEN_LOW,  "precedes")

    RuleBasedCausalityEngineV32._apply_allen_temporal_blend(
        cand_high, causal_scores=scores_high, causal_relation=rel_high,
        follow_ids=set(), weights=weights,
    )
    RuleBasedCausalityEngineV32._apply_allen_temporal_blend(
        cand_low, causal_scores=scores_low, causal_relation=rel_low,
        follow_ids=set(), weights=weights,
    )

    t_high = cand_high["scores"]["temporal"]
    t_low  = cand_low["scores"]["temporal"]

    # Expected with true blend (no max() clamp):
    expected_high = min(1.0, (1 - ALPHA) * TSKR_BASELINE + ALPHA * ALLEN_HIGH)
    expected_low  = min(1.0, (1 - ALPHA) * TSKR_BASELINE + ALPHA * ALLEN_LOW)

    assert abs(t_high - expected_high) < 1e-6, (
        f"D4-B FAIL: high-Allen candidate temporal={t_high:.6f}, "
        f"expected {expected_high:.6f}"
    )
    assert abs(t_low - expected_low) < 1e-6, (
        f"D4-B FAIL: low-Allen candidate temporal={t_low:.6f}, "
        f"expected {expected_low:.6f}"
    )
    assert t_high > t_low, (
        f"D4-B FAIL: high-Allen temporal={t_high:.4f} should exceed "
        f"low-Allen temporal={t_low:.4f}. Blend formula is not discriminating."
    )
    print(
        f"  pass  D4-B (direct): high-Allen temporal={t_high:.4f} > "
        f"low-Allen temporal={t_low:.4f}. Blend formula discriminates correctly."
    )


def test_d4b_allen_blend_discriminates():
    """
    D4-B (full pipeline): OVERLAPS candidate must score strictly higher temporally
    than PRECEDES candidate when all other inputs are identical.
    Requires P7 fixtures — skipped until allen_overlaps_fixture/ and
    allen_precedes_fixture/ are created.
    """
    if not _ALLEN_OVERLAPS.exists() or not _ALLEN_PRECEDES.exists():
        pytest.skip(
            "P7 fixtures not found — create allen_overlaps_fixture/ and "
            "allen_precedes_fixture/ before running full-pipeline D4-B."
        )

    result_overlaps = _run(_ALLEN_OVERLAPS)
    result_precedes = _run(_ALLEN_PRECEDES)

    cands_o = result_overlaps["causality_candidates"]["candidates"]
    cands_p = result_precedes["causality_candidates"]["candidates"]

    assert cands_o, "allen_overlaps_fixture produced no candidates"
    assert cands_p, "allen_precedes_fixture produced no candidates"

    temporal_o = (cands_o[0].get("scores") or {}).get("temporal", 0.0)
    temporal_p = (cands_p[0].get("scores") or {}).get("temporal", 0.0)

    assert temporal_o > temporal_p, (
        f"D4-B FAIL: OVERLAPS temporal={temporal_o:.4f} should exceed "
        f"PRECEDES temporal={temporal_p:.4f}. "
        "If equal, the Allen blend is not discriminating — check _apply_allen_temporal_blend()."
    )
    print(
        f"  pass  D4-B: OVERLAPS temporal={temporal_o:.4f} > PRECEDES temporal={temporal_p:.4f}"
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_d4a_allen_chain_position_tc2,
    test_d4a_allen_chain_position_tc4,
    test_d4c_novel_pattern_consistency_tc2,
    test_d4c_novel_pattern_consistency_tc4,
    test_d4d_earliest_onset_precedes_event_tc4,
    test_d4b_allen_blend_formula_direct,
    test_d4b_allen_blend_discriminates,
]


def run_all() -> bool:
    print(f"\n=== test_robustness_d4_temporal_coherence ({len(ALL_TESTS)} tests) ===")
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
