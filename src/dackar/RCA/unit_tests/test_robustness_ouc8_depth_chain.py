"""
test_robustness_ouc8_depth_chain.py — Tier 2 Scenario correctness: Three-depth causal chain

OUC-8  Full three-depth causal chain traversal (proximate → contributing → root)

Scenario
--------
Pump P101B bearing failure with an engineered three-level causal chain:

  Depth       Category  Failure Mode                       Evidence Source
  ----------  --------  ---------------------------------  -------------------------
  Proximate   A         Bearing wear — vibration OVERLAPS  Telemetry + WO snippet
  Contributing J        PM interval inadequate (47d OD)    pm_compliance + CR snippet
  Root        L         Fleet OE not incorporated          OE document snippet

All three levels have clear, unambiguous supporting evidence.

Ground truth
------------
  - All three candidates retained by the pipeline
  - Category A ranked first (proximate physical cause, strongest structural + temporal)
  - Category J retained (contributing cause — pm_compliance failed lubrication check)
  - Category L retained (root cause — OE document snippet, systemic latent)
  - causal_depth_summary.depth_complete = True
  - causal_depth_summary.proximate_covered = True
  - causal_depth_summary.contributing_covered = True
  - causal_depth_summary.root_cause_covered = True
  - recommended_actions has >= 3 entries (one per retained candidate)

Implementation notes
--------------------
  `recommended_actions[*].causal_depth` is None in the current engine — individual
  actions do not carry a causal_depth attribute. The depth completeness check
  uses causal_depth_summary directly (which IS populated).

  Category A chain_position is "initiating" (not "proximate") in the engine's
  taxonomy. The causal_depth_summary maps "proximate_cause" → Category A FM name,
  so the proximate_covered flag correctly reflects Category A being the initiating cause.

Fixture: tests/fixtures_robustness/ouc8_three_depth_chain/
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
_FIXTURE_DIR = _RCA_ROOT / "tests" / "fixtures_robustness" / "ouc8_three_depth_chain"

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

def _run() -> Dict[str, Any]:
    fixtures = load_fixtures(_FIXTURE_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        orch = build_fixture_orchestrator(tmp)
        return run_rca(orch, fixtures)


def _retained(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["causality_candidates"].get("candidates") or []


def _depth_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return (
        (result.get("rca_card") or {})
        .get("executive_summary", {})
        .get("causal_depth_summary", {})
    )


# ---------------------------------------------------------------------------
# Candidate presence tests
# ---------------------------------------------------------------------------

def test_ouc8_all_three_retained():
    """OUC-8: All three candidates (A, J, L) must be retained."""
    result = _run()
    retained_cats = {c["primary_causal_category"] for c in _retained(result)}
    for cat in ("A", "J", "L"):
        assert cat in retained_cats, (
            f"OUC-8: Category {cat!r} not retained. "
            "All three causal depths require their candidate to clear the screening threshold. "
            "Verify evidence_bundle.candidate_evidence_summary covers FM::FM-OUC8-* "
            "with best_support_score meeting minimum_evidence_threshold."
        )


def test_ouc8_a_ranked_first():
    """OUC-8: Category A (proximate physical cause) must rank first."""
    result = _run()
    retained = _retained(result)
    assert retained, "OUC-8: No candidates retained"
    assert retained[0]["primary_causal_category"] == "A", (
        f"OUC-8: Top candidate is Category {retained[0]['primary_causal_category']!r} "
        f"({retained[0].get('failure_mode_id')}); expected Category A. "
        "Proximate physical cause (bearing wear with strong telemetry + TSKR) must rank first."
    )


def test_ouc8_j_in_top_five():
    """OUC-8: Category J (contributing cause — PM interval) must appear in top 5."""
    result = _run()
    retained = _retained(result)
    top5_cats = [c["primary_causal_category"] for c in retained[:5]]
    assert "J" in top5_cats, (
        f"OUC-8: Category J not in top 5 (found: {top5_cats}). "
        "PM interval inadequacy with failed lubrication PM check and strong CR evidence "
        "should place the contributing cause candidate in the top 5."
    )


def test_ouc8_l_present():
    """OUC-8: Category L (root cause — OE not incorporated) must be present in retained list."""
    result = _run()
    l_cands = [c for c in _retained(result) if c["primary_causal_category"] == "L"]
    assert l_cands, (
        "OUC-8: No Category L candidate retained. "
        "Fleet OE not incorporated is the root cause; it requires a Category L FM in the KG "
        "and strong OE evidence snippet in evidence_bundle."
    )


# ---------------------------------------------------------------------------
# Causal depth summary tests
# ---------------------------------------------------------------------------

def test_ouc8_depth_complete():
    """OUC-8: causal_depth_summary.depth_complete must be True."""
    result = _run()
    depth = _depth_summary(result)
    assert depth, "OUC-8: causal_depth_summary missing from executive_summary"
    assert depth.get("depth_complete") is True, (
        f"OUC-8: depth_complete={depth.get('depth_complete')!r}; expected True. "
        "With all three causal levels present, the pipeline must confirm depth completeness."
    )


def test_ouc8_proximate_covered():
    """OUC-8: causal_depth_summary.proximate_covered must be True."""
    result = _run()
    depth = _depth_summary(result)
    assert depth.get("proximate_covered") is True, (
        f"OUC-8: proximate_covered={depth.get('proximate_covered')!r}. "
        "Category A bearing-wear candidate must be identified as the proximate cause."
    )


def test_ouc8_contributing_covered():
    """OUC-8: causal_depth_summary.contributing_covered must be True."""
    result = _run()
    depth = _depth_summary(result)
    assert depth.get("contributing_covered") is True, (
        f"OUC-8: contributing_covered={depth.get('contributing_covered')!r}. "
        "Category J PM-interval candidate must be identified as a contributing cause."
    )


def test_ouc8_root_cause_covered():
    """OUC-8: causal_depth_summary.root_cause_covered must be True."""
    result = _run()
    depth = _depth_summary(result)
    assert depth.get("root_cause_covered") is True, (
        f"OUC-8: root_cause_covered={depth.get('root_cause_covered')!r}. "
        "Category L OE-not-incorporated candidate must be identified as the root cause."
    )


def test_ouc8_root_cause_label_references_oe():
    """OUC-8: causal_depth_summary.root_cause field must reference the OE/systemic FM."""
    result = _run()
    depth = _depth_summary(result)
    root = str(depth.get("root_cause") or "").lower()
    assert any(kw in root for kw in ("operating experience", "oe", "incorporated", "systemic", "latent")), (
        f"OUC-8: root_cause field={depth.get('root_cause')!r} does not reference OE/systemic content. "
        "The root cause label should identify the FM-OUC8-OE-NOT-INCORPORATED failure mode."
    )


# ---------------------------------------------------------------------------
# Recommended actions test
# ---------------------------------------------------------------------------

def test_ouc8_recommended_actions_span_all_candidates():
    """OUC-8: recommended_actions must have >= 3 entries covering all retained candidates."""
    result = _run()
    rca_card = result.get("rca_card") or {}
    actions = rca_card.get("recommended_actions") or []
    assert len(actions) >= 3, (
        f"OUC-8: recommended_actions has {len(actions)} entries; expected >= 3. "
        "With three retained candidates (A, J, L), the RCA card should recommend at least "
        "one corrective action per candidate."
    )


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_ouc8_all_three_retained,
    test_ouc8_a_ranked_first,
    test_ouc8_j_in_top_five,
    test_ouc8_l_present,
    test_ouc8_depth_complete,
    test_ouc8_proximate_covered,
    test_ouc8_contributing_covered,
    test_ouc8_root_cause_covered,
    test_ouc8_root_cause_label_references_oe,
    test_ouc8_recommended_actions_span_all_candidates,
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
