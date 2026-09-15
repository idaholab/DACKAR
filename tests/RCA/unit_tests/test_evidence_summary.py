"""
test_evidence_summary.py — standalone unit tests for
ChromaEvidenceRetriever._build_candidate_evidence_summary

Run directly:   python test_evidence_summary.py
Or via pytest:  pytest test_evidence_summary.py

Key invariants tested:
  1. Hits without linked_candidate_id are ignored
  2. supporting / contradicting / contextual counts and best_scores accumulate correctly
  3. dominant_temporal_relation picks the most-voted relation
  4. best_lag_hours picks from the highest-support snippet
  5. aggregated_mechanisms and aggregated_outcomes deduplicated (case-insensitive)
  6. Output sorted by best_support - 0.5*best_contradiction
  7. Empty input → empty list
  8. mean_conjecture_fraction averaged across hits
  9. best_source_tier tracks the strongest supporting snippet tier
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


def make_hit(
    snippet_id,
    candidate_id,
    role,  # "supporting" | "contradicting" | "contextual"
    support_score=0.0,
    contradiction_score=0.0,
    context_score=0.0,
    temporal_relation=None,
    lag_hours=None,
    conjecture_fraction=None,
    mechanisms=None,
    outcomes=None,
    source_tier=None,
):
    meta = {
        "linked_candidate_id": candidate_id,
        "support_role": role,
        "spacy_temporal_relation": temporal_relation,
        "spacy_lag_hours": lag_hours,
        "spacy_conjecture_fraction": conjecture_fraction,
        "mechanisms": mechanisms or [],
        "outcomes": outcomes or [],
        "source_tier": source_tier,
    }
    return {
        "snippet_id": snippet_id,
        "doc_id": "DOC-001",
        "section": "test",
        "score": 0.8,
        "support_score": support_score,
        "contradiction_score": contradiction_score,
        "context_score": context_score,
        "metadata": meta,
    }


def find_group(summary, candidate_id):
    for g in summary:
        if g["candidate_id"] == candidate_id:
            return g
    return None


# ── Test functions ────────────────────────────────────────────────────────────

def test_hits_without_candidate_id_ignored():
    r = make_retriever()
    hits = [
        {"snippet_id": "X", "doc_id": "D", "section": "s", "score": 0.5, "metadata": {}},
    ]
    result = r._build_candidate_evidence_summary(hits)
    assert result == []
    print("  PASS test_hits_without_candidate_id_ignored")


def test_empty_input_returns_empty_list():
    r = make_retriever()
    assert r._build_candidate_evidence_summary([]) == []
    print("  PASS test_empty_input_returns_empty_list")


def test_counts_and_best_scores_accumulate():
    """3 supporting + 1 contradicting + 1 contextual for CAND-A."""
    r = make_retriever()
    hits = [
        make_hit("S1", "CAND-A", "supporting",    support_score=0.70),
        make_hit("S2", "CAND-A", "supporting",    support_score=0.85),
        make_hit("S3", "CAND-A", "supporting",    support_score=0.60),
        make_hit("S4", "CAND-A", "contradicting", contradiction_score=0.55),
        make_hit("S5", "CAND-A", "contextual",    context_score=0.30),
    ]
    result = r._build_candidate_evidence_summary(hits)
    g = find_group(result, "CAND-A")
    assert g is not None
    assert g["hit_count"] == 5
    assert g["supporting_count"] == 3
    assert g["contradicting_count"] == 1
    assert g["contextual_count"] == 1
    assert abs(g["best_support_score"] - 0.85) < 1e-6
    assert abs(g["best_contradiction_score"] - 0.55) < 1e-6
    assert abs(g["best_context_score"] - 0.30) < 1e-6
    assert set(g["supporting_snippet_ids"]) == {"S1", "S2", "S3"}
    print("  PASS test_counts_and_best_scores_accumulate")


def test_two_candidates_independent():
    """Hits for two candidates are grouped separately."""
    r = make_retriever()
    hits = [
        make_hit("S1", "CAND-A", "supporting", support_score=0.80),
        make_hit("S2", "CAND-B", "supporting", support_score=0.60),
        make_hit("S3", "CAND-B", "contradicting", contradiction_score=0.50),
    ]
    result = r._build_candidate_evidence_summary(hits)
    assert len(result) == 2
    a = find_group(result, "CAND-A")
    b = find_group(result, "CAND-B")
    assert a["supporting_count"] == 1
    assert b["supporting_count"] == 1
    assert b["contradicting_count"] == 1
    print("  PASS test_two_candidates_independent")


def test_dominant_temporal_relation_majority_vote():
    """dominant_temporal_relation is the most-voted relation string."""
    r = make_retriever()
    hits = [
        make_hit("S1", "CAND-A", "supporting", temporal_relation="precedes"),
        make_hit("S2", "CAND-A", "supporting", temporal_relation="precedes"),
        make_hit("S3", "CAND-A", "contextual", temporal_relation="follows"),
    ]
    result = r._build_candidate_evidence_summary(hits)
    g = find_group(result, "CAND-A")
    assert g["dominant_temporal_relation"] == "precedes"
    print("  PASS test_dominant_temporal_relation_majority_vote")


def test_best_lag_hours_from_highest_support_hit():
    """best_lag_hours selected from the hit with highest support_score."""
    r = make_retriever()
    hits = [
        make_hit("S1", "CAND-A", "supporting", support_score=0.50, lag_hours=48.0),
        make_hit("S2", "CAND-A", "supporting", support_score=0.90, lag_hours=12.0),
        make_hit("S3", "CAND-A", "contextual", support_score=0.10, lag_hours=200.0),
    ]
    result = r._build_candidate_evidence_summary(hits)
    g = find_group(result, "CAND-A")
    assert abs(g["best_lag_hours"] - 12.0) < 1e-6, (
        f"Expected best_lag_hours=12.0 (from highest-support hit), got {g['best_lag_hours']}"
    )
    print("  PASS test_best_lag_hours_from_highest_support_hit")


def test_aggregated_mechanisms_deduped_case_insensitive():
    """Same mechanism in different cases → appears once."""
    r = make_retriever()
    hits = [
        make_hit("S1", "CAND-A", "supporting", mechanisms=["Thermal Fatigue", "corrosion"]),
        make_hit("S2", "CAND-A", "supporting", mechanisms=["thermal fatigue", "Wear"]),
    ]
    result = r._build_candidate_evidence_summary(hits)
    g = find_group(result, "CAND-A")
    mechs_lower = [m.lower() for m in g["aggregated_mechanisms"]]
    assert mechs_lower.count("thermal fatigue") == 1
    assert "corrosion" in mechs_lower
    assert "wear" in mechs_lower
    print("  PASS test_aggregated_mechanisms_deduped_case_insensitive")


def test_mean_conjecture_fraction_averaged():
    r = make_retriever()
    hits = [
        make_hit("S1", "CAND-A", "supporting", conjecture_fraction=0.2),
        make_hit("S2", "CAND-A", "supporting", conjecture_fraction=0.4),
    ]
    result = r._build_candidate_evidence_summary(hits)
    g = find_group(result, "CAND-A")
    assert abs(g["mean_conjecture_fraction"] - 0.30) < 0.01
    print("  PASS test_mean_conjecture_fraction_averaged")


def test_output_sorted_by_net_support_score():
    """
    Sort key: -(best_support - 0.5 * best_contradiction).
    CAND-B has higher net score → should appear first.
    """
    r = make_retriever()
    hits = [
        make_hit("S1", "CAND-A", "supporting",    support_score=0.60),
        make_hit("S2", "CAND-A", "contradicting", contradiction_score=0.40),  # net: 0.60 - 0.20 = 0.40
        make_hit("S3", "CAND-B", "supporting",    support_score=0.90),         # net: 0.90
    ]
    result = r._build_candidate_evidence_summary(hits)
    assert result[0]["candidate_id"] == "CAND-B"
    assert result[1]["candidate_id"] == "CAND-A"
    print("  PASS test_output_sorted_by_net_support_score")


def test_best_source_tier_from_strongest_supporting_hit():
    r = make_retriever()
    hits = [
        make_hit("S1", "CAND-A", "supporting", support_score=0.61, source_tier="plant_family"),
        make_hit("S2", "CAND-A", "supporting", support_score=0.82, source_tier="plant_instance"),
        make_hit("S3", "CAND-A", "contextual", context_score=0.4, source_tier="oe_iris"),
    ]
    result = r._build_candidate_evidence_summary(hits)
    g = find_group(result, "CAND-A")
    assert g["best_source_tier"] == "plant_instance"
    print("  PASS test_best_source_tier_from_strongest_supporting_hit")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_hits_without_candidate_id_ignored,
    test_empty_input_returns_empty_list,
    test_counts_and_best_scores_accumulate,
    test_two_candidates_independent,
    test_dominant_temporal_relation_majority_vote,
    test_best_lag_hours_from_highest_support_hit,
    test_aggregated_mechanisms_deduped_case_insensitive,
    test_mean_conjecture_fraction_averaged,
    test_output_sorted_by_net_support_score,
    test_best_source_tier_from_strongest_supporting_hit,
]


def run_all():
    print(f"\n=== test_evidence_summary ({len(ALL_TESTS)} tests) ===")
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
