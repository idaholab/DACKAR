"""TC-8 integration test — multi-category causal span with contradiction modulation.

Promotes the assertions in ``run_test_case_8.ipynb`` to a pytest-collected
end-to-end test driven by the full orchestrator in fixture-only mode. The
local check helpers below are ported verbatim from the notebook so the
assertion contract is identical.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca
from assertion_helpers import (
    assert_candidate_count,
    assert_depth_complete,
    assert_unresolved_gaps_at_least,
    run_assertion_table,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _get_by_fm(cands, fm_id):
    return next((c for c in cands if c.get("failure_mode_id") == fm_id), None)


def _categories_present(result, expected):
    cands = (result.get("causality_candidates") or {}).get("candidates") or []
    found = {c.get("primary_causal_category") for c in cands}
    missing = [cat for cat in expected if cat not in found]
    assert not missing, f"Missing causal categories: {missing} (found: {found})"


def _check_highest_composite(result, fm_id):
    cands = (result.get("causality_candidates") or {}).get("candidates") or []
    target = _get_by_fm(cands, fm_id)
    assert target is not None, f"{fm_id} not found in candidates"
    max_score = max(float(c.get("composite_score", 0) or 0) for c in cands)
    my_score = float(target.get("composite_score", 0) or 0)
    assert my_score >= max_score - 1e-6, (
        f"{fm_id} composite={my_score:.3f} is not the highest (max={max_score:.3f})"
    )


def _check_category(result, fm_id, expected_cat):
    cands = (result.get("causality_candidates") or {}).get("candidates") or []
    target = _get_by_fm(cands, fm_id)
    assert target is not None, f"{fm_id} not found in candidates"
    actual = target.get("primary_causal_category")
    assert actual == expected_cat, f"{fm_id} category={actual}, expected {expected_cat}"


def _check_contradiction_modulated(result, fm_id):
    """Assert fm_id is retained post-refine despite contradicting evidence,
    and its contradiction score in the evidence bundle is modulated (< 0.5)."""
    post_cands = (result.get("causality_candidates") or {}).get("candidates") or []
    assert any(c.get("failure_mode_id") == fm_id for c in post_cands), (
        f"{fm_id} was filtered out despite contradiction modulation"
    )
    eb = result.get("evidence_bundle") or {}
    ces = eb.get("candidate_evidence_summary") or []
    entry = next((e for e in ces if e.get("candidate_id") == f"FM::{fm_id}"), None)
    assert entry is not None, f"No evidence summary entry for FM::{fm_id}"
    contra = float(entry.get("best_contradiction_score", 1.0) or 1.0)
    assert contra < 0.5, (
        f"{fm_id} best_contradiction_score={contra:.2f} is not modulated — expected < 0.5 "
        f"(lot-number cross-reference should reduce contradiction weight)"
    )


def _check_pm_compliance_failed(result):
    """Assert pm_compliance fixture was received and records at least one failed check."""
    pm = result.get("pm_compliance") or {}
    assert pm, "pm_compliance artifact is absent from result"
    failed = (pm.get("summary") or {}).get("failed", 0)
    assert failed >= 1, (
        f"pm_compliance.summary.failed={failed} — expected >= 1 "
        f"(interval nonconformance vs vendor spec)"
    )


def _check_contradicting_doc(result, doc_id):
    eb = result.get("evidence_bundle") or {}
    for snip in eb.get("results", []):
        meta = snip.get("metadata") or {}
        if snip.get("doc_id") == doc_id and meta.get("support_role") == "contradicting":
            return
    raise AssertionError(
        f"No snippet with doc_id='{doc_id}' and metadata.support_role='contradicting' found"
    )


@pytest.mark.integration
def test_case_8_multi_category_contradiction(tmp_path: Path) -> None:
    fixtures = load_fixtures(_FIXTURE_DIR)
    orchestrator = build_fixture_orchestrator(tmp_path, top_k_candidates=6, enable_ishikawa=True)
    result = run_rca(orchestrator, fixtures)

    assertions = [
        {"id": "A8-1", "desc": "Five candidates retained in post-refine",
         "fn": lambda r: assert_candidate_count(r, 5)},
        {"id": "A8-2", "desc": "Category span A/I/J/K/L all present",
         "fn": lambda r: _categories_present(r, ["A", "I", "J", "K", "L"])},
        {"id": "A8-3", "desc": "FM-PM-FREQ-NONCONF has highest post-refine composite",
         "fn": lambda r: _check_highest_composite(r, "FM-PM-FREQ-NONCONF")},
        {"id": "A8-4", "desc": "FM-CHK-SEAT-EROSION is Category A (proximate)",
         "fn": lambda r: _check_category(r, "FM-CHK-SEAT-EROSION", "A")},
        {"id": "A8-5", "desc": "FM-OE-SCREENING-MISS is Category L (root cause)",
         "fn": lambda r: _check_category(r, "FM-OE-SCREENING-MISS", "L")},
        {"id": "A8-6", "desc": "FM-VENDOR-BATCH-TRACEABILITY retained despite contradicting evidence",
         "fn": lambda r: _check_contradiction_modulated(r, "FM-VENDOR-BATCH-TRACEABILITY")},
        {"id": "A8-7", "desc": "TD-REPORT-2025-0312 has contradicting role in evidence bundle",
         "fn": lambda r: _check_contradicting_doc(r, "TD-REPORT-2025-0312")},
        {"id": "A8-8", "desc": "PM compliance present with >= 1 failed check",
         "fn": lambda r: _check_pm_compliance_failed(r)},
        {"id": "A8-9", "desc": "Causal depth complete (proximate + contributing + root)",
         "fn": lambda r: assert_depth_complete(r, True)},
        {"id": "A8-10", "desc": "rca_card has >= 2 open items (intentional residual ambiguity)",
         "fn": lambda r: assert_unresolved_gaps_at_least(r, 2)},
    ]
    run_assertion_table(result, assertions, label="TC-8 Assertions")
