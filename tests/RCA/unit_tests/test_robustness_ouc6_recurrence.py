"""
test_robustness_ouc6_recurrence.py — Tier 2 Scenario correctness: Recurrence + Ineffective CA

OUC-6  Recurring failure with ineffective corrective actions

Scenario
--------
Third occurrence of RCP pump seal failure within 7 months. Two prior events exist,
both resolved=False (seals replaced but root cause never addressed). The corrective
action (seal replacement per OEM spec) was implemented after each event but the failure
recurred, indicating the CA was ineffective at the system level.

A concurrent shaft vibration anomaly exists but has no recurrence history.

  Candidate              FM                       Recurrence events  Unresolved  Score
  ---------------------  -----------------------  -----------------  ----------  -----
  RCP seal (primary)     FM-OUC6-SEAL-WEAR        2 (both unresol)   2 FM-lvl    1.0
  RCP shaft vibration    FM-OUC6-SHAFT-VIBRATION  0                  0           0.07

Ground truth
------------
  - Seal-wear FM is retained with recurrence_score = 1.0 (high confidence)
  - Seal-wear FM matched_past_event_ids contains both prior event IDs
  - Seal-wear FM unresolved_fm_count = 2 (both past CAs classified as unresolved)
  - Seal-wear FM ranks first in retained list
  - recurrence_summary correctly identifies FM-OUC6-SEAL-WEAR as the top mechanism

Design notes
------------
  The recurrence_score formula is:
    base = 0.55 * min(1, fm_count/2) + 0.35 * min(1, comp_count/2) + 0.10 * min(1, asset_count/3)
    + unresolved_boost (min 0.20)
  With fm_count=2, comp_count=2, asset_count=2 and 2 unresolved FM events:
    base = 0.55 + 0.35 + 0.067 = 0.967
    unresolved_boost = min(0.20, weighted) — the fixture supplies time_distance_days on both
    prior events, so CMMS time-weighted quality applies.
  Result: recurrence_score = 1.0 (capped), recurrence_confidence = "high".

  The recurrence boost lifts composite_score by +0.03 * recurrence_score, which is small
  but meaningful. The main discriminator is the much stronger base composite from TSKR
  and evidence on the seal FM vs. shaft vibration.

  The recurrence_summary.top_recurrent_mechanism_candidate_id resolves to FM::FM-OUC6-SEAL-WEAR,
  confirming the engine correctly surfaces the recurring mechanism rather than the historical
  event analog as the primary recommendation.

Fixture: tests/fixtures_robustness/ouc6_recurrence_ineffective_ca/
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
_FIXTURE_DIR = _SCENARIO_ROOT / "fixtures_robustness" / "ouc6_recurrence_ineffective_ca"

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run() -> Dict[str, Any]:
    fixtures = load_fixtures(_FIXTURE_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        orch = build_fixture_orchestrator(tmp)
        return run_rca(orch, fixtures)


def _retained(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["causality_candidates"].get("candidates") or []


def _seal_cand(result: Dict[str, Any]) -> Dict[str, Any]:
    retained = _retained(result)
    for c in retained:
        if c.get("failure_mode_id") == "FM-OUC6-SEAL-WEAR":
            return c
    return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ouc6_seal_fm_retained():
    """OUC-6: The recurring seal-wear FM must be retained by the screening."""
    result = _run()
    assert _seal_cand(result), (
        "OUC-6: FM-OUC6-SEAL-WEAR not found in retained candidates. "
        "Ensure evidence_bundle.candidate_evidence_summary covers FM::FM-OUC6-SEAL-WEAR "
        "with best_support_score >= minimum_evidence_threshold."
    )


def test_ouc6_seal_ranked_first():
    """OUC-6: Seal-wear FM must rank first — recurrence boost over shaft vibration."""
    result = _run()
    retained = _retained(result)
    assert retained, "OUC-6: No candidates retained"
    assert retained[0].get("failure_mode_id") == "FM-OUC6-SEAL-WEAR", (
        f"OUC-6: Top candidate is {retained[0].get('failure_mode_id')!r}; "
        "expected FM-OUC6-SEAL-WEAR. "
        "The recurring seal FM should rank first given high TSKR support, strong evidence, "
        "and a recurrence_score=1.0 (high confidence)."
    )


def test_ouc6_recurrence_score_high():
    """OUC-6: Seal-wear FM recurrence_score must reach 1.0 (two matched unresolved events)."""
    result = _run()
    cand = _seal_cand(result)
    assert cand, "OUC-6: Seal FM not retained"
    rec = cand.get("recurrence") or {}
    score = rec.get("recurrence_score", 0.0)
    assert score >= 0.95, (
        f"OUC-6: recurrence_score={score:.4f} < 0.95. "
        "Two matched unresolved FM-level events should push score to 1.0 (capped)."
    )


def test_ouc6_recurrence_confidence_high():
    """OUC-6: Seal-wear FM recurrence_confidence must be 'high' (score >= 0.75)."""
    result = _run()
    cand = _seal_cand(result)
    assert cand, "OUC-6: Seal FM not retained"
    rec = cand.get("recurrence") or {}
    confidence = rec.get("recurrence_confidence", "none")
    assert confidence == "high", (
        f"OUC-6: recurrence_confidence={confidence!r}; expected 'high'. "
        "Two unresolved FM-level recurrences with time_distance_days present should yield "
        "a score above the 0.75 threshold for 'high' confidence."
    )


def test_ouc6_matched_past_events():
    """OUC-6: Seal-wear FM must match both prior event IDs in matched_past_event_ids."""
    result = _run()
    cand = _seal_cand(result)
    assert cand, "OUC-6: Seal FM not retained"
    rec = cand.get("recurrence") or {}
    matched = set(rec.get("matched_past_event_ids") or [])
    for expected_id in {"EVT-OUC6-SEAL-FAIL-001", "EVT-OUC6-SEAL-FAIL-002"}:
        assert expected_id in matched, (
            f"OUC-6: Expected {expected_id!r} in matched_past_event_ids={sorted(matched)}. "
            "Both prior events must be indexed by the failure-mode recurrence lookup."
        )


def test_ouc6_unresolved_fm_count():
    """OUC-6: unresolved_fm_count must equal 2 (both prior CAs explicitly unresolved)."""
    result = _run()
    cand = _seal_cand(result)
    assert cand, "OUC-6: Seal FM not retained"
    rec = cand.get("recurrence") or {}
    unresolved = rec.get("unresolved_fm_count", 0)
    assert unresolved == 2, (
        f"OUC-6: unresolved_fm_count={unresolved}; expected 2. "
        "Both past events have resolved=False and matched_failure_mode_ids=['FM-OUC6-SEAL-WEAR']."
    )


def test_ouc6_recurrence_summary_mechanism():
    """OUC-6: recurrence_summary must identify FM-OUC6-SEAL-WEAR as top mechanism."""
    result = _run()
    summary = (result["causality_candidates"].get("recurrence_summary") or {})
    top_mech = summary.get("top_recurrent_mechanism_candidate_id", "")
    assert "FM-OUC6-SEAL-WEAR" in str(top_mech), (
        f"OUC-6: top_recurrent_mechanism_candidate_id={top_mech!r}; "
        "expected it to reference FM-OUC6-SEAL-WEAR. "
        "The engine must surface the recurring FM (not just the historical event analog) "
        "as the top mechanism candidate."
    )


def test_ouc6_high_recurrence_in_summary():
    """OUC-6: FM-OUC6-SEAL-WEAR must appear in recurrence_summary.high_recurrence_candidate_ids."""
    result = _run()
    summary = (result["causality_candidates"].get("recurrence_summary") or {})
    high_ids = summary.get("high_recurrence_candidate_ids") or []
    seal_present = any("FM-OUC6-SEAL-WEAR" in str(cid) for cid in high_ids)
    assert seal_present, (
        f"OUC-6: FM-OUC6-SEAL-WEAR not in high_recurrence_candidate_ids={high_ids}. "
        "A recurrence_score=1.0 candidate must be included in the high-recurrence list."
    )


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_ouc6_seal_fm_retained,
    test_ouc6_seal_ranked_first,
    test_ouc6_recurrence_score_high,
    test_ouc6_recurrence_confidence_high,
    test_ouc6_matched_past_events,
    test_ouc6_unresolved_fm_count,
    test_ouc6_recurrence_summary_mechanism,
    test_ouc6_high_recurrence_in_summary,
]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {e}")
