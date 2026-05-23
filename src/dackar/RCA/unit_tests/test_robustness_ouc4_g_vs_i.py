"""
test_robustness_ouc4_g_vs_i.py — Tier 2 Scenario correctness: G vs. I evidence discrimination

OUC-4  Human execution error vs. configuration/change-control failure (Category G vs. I)

Two variants, identical KG / telemetry / TSKR — only the evidence_bundle differs.

Scenario
--------
Motor-operated valve MOV-001 fails to open on demand. Work order WO-2024-1201 was
executed 48 hours prior (actuator stem packing replacement). The KG carries two FMs
on the same component:

  FM-OUC4-HUMAN-ERROR    (Category G) — technician applied wrong torque
  FM-OUC4-CHANGE-CONTROL (Category I) — SOP specified wrong torque (ECN not applied)

The telemetry and TSKR patterns are IDENTICAL in both variants (same actuator torque
signal, same OVERLAPS relation, same support/confidence). Only the WO text — encoded
as `candidate_evidence_summary` support/contradiction scores — differs.

Variant 4a — "WO says technician skipped step 7"
  G evidence: best_support_score=0.92,  best_contradiction_score=0.00
  I evidence: best_support_score=0.18,  best_contradiction_score=0.60
  Expected: G retained and ranked first; I filtered out.

Variant 4b — "Technician followed procedure; SOP wrong per ECN-2024-0441"
  G evidence: best_support_score=0.18,  best_contradiction_score=0.65
  I evidence: best_support_score=0.91,  best_contradiction_score=0.00
  Expected: I retained and ranked first; G filtered out.

Key observation
---------------
Category G has evidence weight = 0.65 (dominant), Category I has evidence weight = 0.45.
Both candidates receive identical structural (0.85) and temporal (0.70) sub-scores.
A high contradiction_score collapses the refined evidence score to ~0, pushing the
losing candidate below both the minimum_evidence_threshold and minimum_composite_threshold,
causing it to be FILTERED OUT rather than merely ranked lower.

This is correct and desirable behavior: when documentary evidence actively contradicts a
hypothesis, the engine should eliminate it from the retained set, not weakly de-rank it.

Evidence weight table (causal_category_weights):
  Category G: structural=0.05, temporal=0.10, telemetry=0.05, evidence=0.65, governance=0.15
  Category I: structural=0.05, temporal=0.25, telemetry=0.10, evidence=0.45, governance=0.15

Fixtures:
  tests/fixtures_robustness/ouc4a_human_error/
  tests/fixtures_robustness/ouc4b_change_control/
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

_RCA_ROOT = Path(__file__).resolve().parent.parent
_TESTS_SHARED = _RCA_ROOT / "tests" / "shared"
_FIX_4A = _RCA_ROOT / "tests" / "fixtures_robustness" / "ouc4a_human_error"
_FIX_4B = _RCA_ROOT / "tests" / "fixtures_robustness" / "ouc4b_change_control"

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(fixture_dir: Path) -> Dict[str, Any]:
    fixtures = load_fixtures(fixture_dir)
    with tempfile.TemporaryDirectory() as tmp:
        orch = build_fixture_orchestrator(tmp)
        return run_rca(orch, fixtures)


def _retained(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["causality_candidates"].get("candidates") or []


def _filtered(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["causality_candidates"].get("filtered_out_candidates") or []


def _best_by_cat(cands: List[Dict[str, Any]], cat: str) -> Optional[Dict[str, Any]]:
    matching = [c for c in cands if c.get("primary_causal_category") == cat]
    if not matching:
        return None
    return max(matching, key=lambda c: c.get("composite_score", 0.0))


# ---------------------------------------------------------------------------
# OUC-4a: Category G (human error) should win
# ---------------------------------------------------------------------------

def test_ouc4a_g_retained():
    """OUC-4a: Category G candidate must be retained when WO text documents technician error."""
    result = _run(_FIX_4A)
    cats = {c["primary_causal_category"] for c in _retained(result)}
    assert "G" in cats, (
        "OUC-4a: Category G not retained. "
        "evidence_bundle for variant 4a must have best_support_score=0.92 on FM-OUC4-HUMAN-ERROR "
        "to clear the evidence threshold."
    )


def test_ouc4a_i_not_retained():
    """OUC-4a: Category I candidate must be filtered out — WO text contradicts it."""
    result = _run(_FIX_4A)
    cats = {c["primary_causal_category"] for c in _retained(result)}
    assert "I" not in cats, (
        f"OUC-4a: Category I unexpectedly retained in variant 4a. "
        "WO text attributes the failure to technician error, not procedure inadequacy. "
        "High contradiction_score on FM-OUC4-CHANGE-CONTROL should collapse its refined "
        "evidence score to ~0 and push it below the screening threshold."
    )


def test_ouc4a_g_ranked_first():
    """OUC-4a: Category G must be the top-ranked retained candidate."""
    result = _run(_FIX_4A)
    retained = _retained(result)
    assert retained, "OUC-4a: No candidates retained"
    assert retained[0]["primary_causal_category"] == "G", (
        f"OUC-4a: Top candidate is Category {retained[0]['primary_causal_category']!r}; "
        "expected Category G (human execution error)."
    )


def test_ouc4a_g_has_strong_evidence():
    """OUC-4a: Category G retained candidate must have evidence sub-score >= 0.45."""
    result = _run(_FIX_4A)
    g_cand = _best_by_cat(_retained(result), "G")
    assert g_cand is not None, "OUC-4a: No Category G retained"
    ev = (g_cand.get("scores") or {}).get("evidence", 0.0)
    assert ev >= 0.45, (
        f"OUC-4a: Category G evidence sub-score={ev:.4f} < 0.45. "
        "WO snippet with best_support_score=0.92 should produce a high refined evidence score."
    )


# ---------------------------------------------------------------------------
# OUC-4b: Category I (change control failure) should win
# ---------------------------------------------------------------------------

def test_ouc4b_i_retained():
    """OUC-4b: Category I candidate must be retained when WO text documents ECN/SOP failure."""
    result = _run(_FIX_4B)
    cats = {c["primary_causal_category"] for c in _retained(result)}
    assert "I" in cats, (
        "OUC-4b: Category I not retained. "
        "evidence_bundle for variant 4b must have best_support_score=0.91 on FM-OUC4-CHANGE-CONTROL "
        "to clear the evidence threshold."
    )


def test_ouc4b_g_not_retained():
    """OUC-4b: Category G candidate must be filtered out — WO text says technician was compliant."""
    result = _run(_FIX_4B)
    cats = {c["primary_causal_category"] for c in _retained(result)}
    assert "G" not in cats, (
        f"OUC-4b: Category G unexpectedly retained in variant 4b. "
        "WO text explicitly states the technician followed the procedure correctly. "
        "High contradiction_score on FM-OUC4-HUMAN-ERROR should collapse its refined "
        "evidence score to ~0 and push it below the screening threshold."
    )


def test_ouc4b_i_ranked_first():
    """OUC-4b: Category I must be the top-ranked retained candidate."""
    result = _run(_FIX_4B)
    retained = _retained(result)
    assert retained, "OUC-4b: No candidates retained"
    assert retained[0]["primary_causal_category"] == "I", (
        f"OUC-4b: Top candidate is Category {retained[0]['primary_causal_category']!r}; "
        "expected Category I (change control / SOP specification failure)."
    )


def test_ouc4b_i_has_strong_evidence():
    """OUC-4b: Category I retained candidate must have evidence sub-score >= 0.45."""
    result = _run(_FIX_4B)
    i_cand = _best_by_cat(_retained(result), "I")
    assert i_cand is not None, "OUC-4b: No Category I retained"
    ev = (i_cand.get("scores") or {}).get("evidence", 0.0)
    assert ev >= 0.45, (
        f"OUC-4b: Category I evidence sub-score={ev:.4f} < 0.45. "
        "WO snippet with best_support_score=0.91 should produce a high refined evidence score."
    )


# ---------------------------------------------------------------------------
# Cross-variant: structural and temporal must be equal (fixture sanity)
# ---------------------------------------------------------------------------

def test_ouc4_structural_temporal_equal_across_variants():
    """OUC-4 sanity: the retained winner in each variant must have equal structural and temporal scores.

    Both G (winner in 4a) and I (winner in 4b) are on the same component with identical
    TSKR patterns and telemetry, so their structural and temporal sub-scores must match.
    Only the evidence dimension should differ.
    """
    res_a = _run(_FIX_4A)
    res_b = _run(_FIX_4B)

    winner_a = (_retained(res_a) or [{}])[0]
    winner_b = (_retained(res_b) or [{}])[0]

    s_a = winner_a.get("scores") or {}
    s_b = winner_b.get("scores") or {}

    struct_a = round(s_a.get("structural", 0.0), 3)
    struct_b = round(s_b.get("structural", 0.0), 3)
    temp_a = round(s_a.get("temporal", 0.0), 3)
    temp_b = round(s_b.get("temporal", 0.0), 3)

    assert struct_a == struct_b, (
        f"OUC-4 sanity: structural score of retained winner differs between variants: "
        f"4a (cat={winner_a.get('primary_causal_category')})={struct_a}, "
        f"4b (cat={winner_b.get('primary_causal_category')})={struct_b}. "
        "Both FMs are on the same component with identical KG — structural must match."
    )
    assert temp_a == temp_b, (
        f"OUC-4 sanity: temporal score of retained winner differs between variants: "
        f"4a={temp_a}, 4b={temp_b}. "
        "Both FMs use identical TSKR patterns and telemetry — temporal must match."
    )


def test_ouc4_inversion_confirmed():
    """OUC-4 key property: top candidate category inverts between 4a and 4b."""
    res_a = _run(_FIX_4A)
    res_b = _run(_FIX_4B)
    top_a = (_retained(res_a) or [{}])[0].get("primary_causal_category")
    top_b = (_retained(res_b) or [{}])[0].get("primary_causal_category")
    assert top_a == "G", f"OUC-4a top={top_a!r}, expected G"
    assert top_b == "I", f"OUC-4b top={top_b!r}, expected I"
    assert top_a != top_b, (
        "OUC-4: Top category must invert between variants 4a (G) and 4b (I). "
        "If the ranking does not invert, the evidence retriever is not discriminating "
        "between human execution error and change-control failure text."
    )


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_ouc4a_g_retained,
    test_ouc4a_i_not_retained,
    test_ouc4a_g_ranked_first,
    test_ouc4a_g_has_strong_evidence,
    test_ouc4b_i_retained,
    test_ouc4b_g_not_retained,
    test_ouc4b_i_ranked_first,
    test_ouc4b_i_has_strong_evidence,
    test_ouc4_structural_temporal_equal_across_variants,
    test_ouc4_inversion_confirmed,
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
