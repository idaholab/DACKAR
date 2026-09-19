"""
test_evidence_dedup.py — standalone unit tests for
ChromaEvidenceRetriever._dedupe_and_rank

Run directly:   python test_evidence_dedup.py
Or via pytest:  pytest test_evidence_dedup.py

Key invariants tested:
  1. Same snippet + different candidates → both hits survive
  2. Same snippet + same candidate → only highest-score hit survives
  3. Context hits for a claimed snippet_id → dropped
  4. Context hits for unclaimed snippet_id → kept
  5. Output sorted by descending score
  6. Empty input → empty output
  7. All-context input → all unique snippets included
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.evidence_retriever import (
    ChromaEvidenceRetriever,
    InMemoryEvidenceStore,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_retriever():
    return ChromaEvidenceRetriever(store=InMemoryEvidenceStore([]), annotator=None)


def make_hit(snippet_id, score, candidate_id=None, doc_id="DOC-001", section="test"):
    return {
        "snippet_id": snippet_id,
        "doc_id": doc_id,
        "section": section,
        "score": score,
        "snippet": "placeholder",
        "metadata": {"linked_candidate_id": candidate_id} if candidate_id else {},
    }


# ── Test functions ────────────────────────────────────────────────────────────

def test_same_snippet_different_candidates_both_kept():
    """
    SNIP-001 linked to CAND-A and CAND-B → both survive because
    the dedup key is snippet_id::candidate_id, not snippet_id alone.
    """
    r = make_retriever()
    hits = [
        make_hit("SNIP-001", score=0.8, candidate_id="CAND-A"),
        make_hit("SNIP-001", score=0.7, candidate_id="CAND-B"),
    ]
    result = r._dedupe_and_rank(hits)
    keys = {(h["snippet_id"], h["metadata"].get("linked_candidate_id")) for h in result}
    assert ("SNIP-001", "CAND-A") in keys
    assert ("SNIP-001", "CAND-B") in keys
    assert len(result) == 2
    print("  PASS test_same_snippet_different_candidates_both_kept")


def test_same_snippet_same_candidate_keeps_highest_score():
    """Two hits for SNIP-001/CAND-A → only score=0.9 survives."""
    r = make_retriever()
    hits = [
        make_hit("SNIP-001", score=0.6, candidate_id="CAND-A"),
        make_hit("SNIP-001", score=0.9, candidate_id="CAND-A"),
    ]
    result = r._dedupe_and_rank(hits)
    assert len(result) == 1
    assert abs(result[0]["score"] - 0.9) < 1e-6
    print("  PASS test_same_snippet_same_candidate_keeps_highest_score")


def test_context_hit_excluded_when_snippet_claimed_by_candidate():
    """
    SNIP-001 claimed by CAND-A; context hit for same SNIP-001 → dropped.
    """
    r = make_retriever()
    hits = [
        make_hit("SNIP-001", score=0.8, candidate_id="CAND-A"),
        make_hit("SNIP-001", score=0.5, candidate_id=None),  # context duplicate
    ]
    result = r._dedupe_and_rank(hits)
    assert len(result) == 1
    assert result[0]["metadata"].get("linked_candidate_id") == "CAND-A"
    print("  PASS test_context_hit_excluded_when_snippet_claimed_by_candidate")


def test_context_hit_included_for_unclaimed_snippet():
    """SNIP-002 is only a context hit → kept."""
    r = make_retriever()
    hits = [
        make_hit("SNIP-001", score=0.8, candidate_id="CAND-A"),
        make_hit("SNIP-002", score=0.5, candidate_id=None),
    ]
    result = r._dedupe_and_rank(hits)
    ids = {h["snippet_id"] for h in result}
    assert "SNIP-001" in ids
    assert "SNIP-002" in ids
    assert len(result) == 2
    print("  PASS test_context_hit_included_for_unclaimed_snippet")


def test_output_sorted_by_descending_score():
    r = make_retriever()
    hits = [
        make_hit("SNIP-A", score=0.3, candidate_id="CAND-1"),
        make_hit("SNIP-B", score=0.9, candidate_id="CAND-1"),
        make_hit("SNIP-C", score=0.6, candidate_id="CAND-1"),
    ]
    result = r._dedupe_and_rank(hits)
    scores = [h["score"] for h in result]
    assert scores == sorted(scores, reverse=True)
    print("  PASS test_output_sorted_by_descending_score")


def test_empty_input_returns_empty_list():
    r = make_retriever()
    assert r._dedupe_and_rank([]) == []
    print("  PASS test_empty_input_returns_empty_list")


def test_all_context_hits_unique_snippets_all_included():
    """When no hits have candidate links, all unique snippets survive."""
    r = make_retriever()
    hits = [
        make_hit("SNIP-X", score=0.4, candidate_id=None),
        make_hit("SNIP-Y", score=0.7, candidate_id=None),
        make_hit("SNIP-Z", score=0.2, candidate_id=None),
    ]
    result = r._dedupe_and_rank(hits)
    assert len(result) == 3
    print("  PASS test_all_context_hits_unique_snippets_all_included")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_same_snippet_different_candidates_both_kept,
    test_same_snippet_same_candidate_keeps_highest_score,
    test_context_hit_excluded_when_snippet_claimed_by_candidate,
    test_context_hit_included_for_unclaimed_snippet,
    test_output_sorted_by_descending_score,
    test_empty_input_returns_empty_list,
    test_all_context_hits_unique_snippets_all_included,
]


def run_all():
    print(f"\n=== test_evidence_dedup ({len(ALL_TESTS)} tests) ===")
    passed, failed = 0, 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
