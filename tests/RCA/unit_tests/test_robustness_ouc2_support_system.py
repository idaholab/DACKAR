"""
test_robustness_ouc2_support_system.py — Tier 2 Scenario correctness: Category B vs A

OUC-2  Support system failure vs. component failure (Category B vs. A)

Scenario
--------
Pump P101A fails. Two anomalies are present:

  Component            Category  Timing                 Allen     Severity
  -------------------  --------  ---------------------  --------  --------
  CWS supply line      B         starts 2h before trip  OVERLAPS  0.70
  Pump bearing temp    A         starts 35min before    OVERLAPS  0.75

The KG contains an explicit support-dependency edge:
  U1-CWS-SUPPLY-LINE --[provides_cooling_to]--> U1-PUMP-P101A-BEARING

Both signals have the same Allen relation (OVERLAPS). The discrimination is driven by:
  - TSKR: CWS has higher support (0.88/0.92) than bearing (0.68/0.72)
  - Evidence: CWS gets stronger support from the work order snippet

Ground truth
------------
  - Category B candidate ranks above Category A (composite B > composite A)
  - Both candidates are retained (neither is purely a consequence)
  - Category B candidate has allen_relation = "overlaps" (causal lead time confirmed)

Design notes
------------
  - The structural score for Category A (bearing, seed component) is HIGHER than Category B
    (0.85 vs 0.75). Category B wins on temporal and evidence scores — the support edge
    improves TSKR match quality (longer, more consistent lead time) rather than giving
    a direct structural delta. This is the correct discriminator: the pipeline should
    identify the upstream support failure even when the downstream component has stronger
    KG seed-based structural score.

Fixture: tests/fixtures_robustness/ouc2_b_vs_a_topology/
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
_FIXTURE_DIR = _SCENARIO_ROOT / "fixtures_robustness" / "ouc2_b_vs_a_topology"

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


def _cands(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["causality_candidates"].get("candidates") or []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ouc2_both_categories_retained():
    """OUC-2: Both Category B (CWS) and Category A (bearing) must be in the retained list."""
    result = _run()
    cands = _cands(result)
    categories = {c["primary_causal_category"] for c in cands}
    assert "B" in categories, (
        "OUC-2: No Category B candidate retained. "
        "Verify that the KG fixture includes a failure mode with causal_category='B' "
        "and that the CWS component evidence clears the screening threshold."
    )
    assert "A" in categories, (
        "OUC-2: No Category A candidate retained — fixture may need adjustment."
    )


def test_ouc2_b_ranks_above_a():
    """OUC-2: Category B composite score must exceed Category A composite score."""
    result = _run()
    cands = _cands(result)
    b_scores = [c["composite_score"] for c in cands if c["primary_causal_category"] == "B"]
    a_scores = [c["composite_score"] for c in cands if c["primary_causal_category"] == "A"]
    assert b_scores, "OUC-2: No Category B candidates found in retained list"
    assert a_scores, "OUC-2: No Category A candidates found in retained list"
    assert max(b_scores) > max(a_scores), (
        f"OUC-2: Category B best composite={max(b_scores):.4f} should exceed "
        f"Category A best composite={max(a_scores):.4f}. "
        "Support system failure (B) must outrank internal component failure (A) "
        "when the upstream support degradation has stronger TSKR pattern and evidence."
    )


def test_ouc2_b_ranked_first():
    """OUC-2: Category B candidate must occupy the top rank."""
    result = _run()
    cands = _cands(result)
    assert cands, "OUC-2: No candidates retained"
    assert cands[0]["primary_causal_category"] == "B", (
        f"OUC-2: Top candidate is Category {cands[0]['primary_causal_category']!r} "
        f"({cands[0]['component_id']}), expected Category B (CWS supply line). "
        "Support system failure must rank first when it precedes and causes the "
        "downstream component failure."
    )


def test_ouc2_b_has_causal_allen():
    """OUC-2: Category B candidate must have an OVERLAPS Allen relation."""
    result = _run()
    cands = _cands(result)
    b_cand = next((c for c in cands if c["primary_causal_category"] == "B"), None)
    assert b_cand is not None, "OUC-2: No Category B candidate"
    allen_rel = (b_cand.get("scores") or {}).get("allen_relation")
    assert allen_rel in {"overlaps", "contains", "precedes"}, (
        f"OUC-2: Category B candidate allen_relation={allen_rel!r}; "
        "expected a causal relation (overlaps/contains/precedes). "
        "CWS anomaly started 2h before event — must be classified as OVERLAPS."
    )


def test_ouc2_b_has_higher_temporal_than_a():
    """OUC-2: Category B temporal sub-score exceeds Category A — longer, more consistent lead time."""
    result = _run()
    cands = _cands(result)
    b_cand = next((c for c in cands if c["primary_causal_category"] == "B"), None)
    a_cand = next((c for c in cands if c["primary_causal_category"] == "A"), None)
    assert b_cand and a_cand, "OUC-2: Both categories must be retained"
    b_temporal = (b_cand.get("scores") or {}).get("temporal", 0.0)
    a_temporal = (a_cand.get("scores") or {}).get("temporal", 0.0)
    assert b_temporal > a_temporal, (
        f"OUC-2: Category B temporal={b_temporal:.4f} should exceed "
        f"Category A temporal={a_temporal:.4f}. "
        "CWS has higher TSKR support/confidence and longer lag consistency than bearing."
    )


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_ouc2_both_categories_retained,
    test_ouc2_b_ranks_above_a,
    test_ouc2_b_ranked_first,
    test_ouc2_b_has_causal_allen,
    test_ouc2_b_has_higher_temporal_than_a,
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
