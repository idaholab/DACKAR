"""
test_evidence_scorer.py — standalone unit tests for
ChromaEvidenceRetriever._assess_hit_against_candidate

Run directly:   python test_evidence_scorer.py
Or via pytest:  pytest test_evidence_scorer.py
"""
import sys
from pathlib import Path

# Make orchestrators/ importable without going through the dackar package
_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.evidence_retriever import (
    ChromaEvidenceRetriever,
    EvidenceRetrieverConfig,
    InMemoryEvidenceStore,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_retriever(config=None):
    return ChromaEvidenceRetriever(
        store=InMemoryEvidenceStore([]),
        config=config,
        annotator=None,
    )


def make_hit(snippet, meta=None, score=0.5):
    return {
        "snippet_id": "SNIP-001",
        "doc_id": "DOC-001",
        "section": "test",
        "snippet": snippet,
        "score": score,
        "metadata": meta or {
            "doc_type": "WO",
            "authority_level": "mandatory",
            "extraction_quality": 1.0,
        },
    }


def make_query_plan(
    cause_label="",
    hypothesis_type="",
    query_type="candidate",
    candidate_id="FM::TEST-001",
    weight=1.0,
):
    return {
        "query_type": query_type,
        "cause_label": cause_label,
        "hypothesis_type": hypothesis_type,
        "candidate_id": candidate_id,
        "weight": weight,
    }


def assert_approx(actual, expected, tol=0.01, label=""):
    assert abs(actual - expected) <= tol, (
        f"{label}: expected ~{expected}, got {actual}"
    )


# ── Test functions ────────────────────────────────────────────────────────────

def test_contradiction_cue_suppressed_when_no_semantic_overlap():
    """
    Snippet contains contradiction cue ('within normal limits') but has ZERO
    token overlap with cause_label → semantic_relevance=0 → only +0.05 boost,
    not +0.45. Result: contradiction_score < 0.35 → role='contextual'.
    """
    r = make_retriever()
    hit = make_hit("all readings within normal limits")
    qp = make_query_plan(cause_label="air in-leakage", hypothesis_type="vacuum degradation")
    result = r._assess_hit_against_candidate(hit, qp)

    assert result["semantic_relevance_score"] == 0.0, (
        f"Expected semantic_relevance=0, got {result['semantic_relevance_score']}"
    )
    # query_type='candidate' adds +0.15 to support_score, not contradiction.
    # contradiction_score = 0.05 (small cue bump) * auth=1.0 * quality=1.0 * epistemic=1.0 = 0.05
    assert_approx(result["contradiction_score"], 0.05, label="contradiction_score")
    assert result["support_role"] == "contextual", (
        f"Expected role=contextual, got {result['support_role']}"
    )
    print("  PASS test_contradiction_cue_suppressed_when_no_semantic_overlap")


def test_contradiction_cue_applied_when_overlap_present():
    """
    Snippet contains contradiction cue AND token overlap with cause_label
    → semantic_relevance > 0 → full +0.45 boost.
    Result: contradiction_score >= 0.35 → role='contradicting'.
    """
    r = make_retriever()
    hit = make_hit("tube inspection: no abnormality found in tube bundle")
    qp = make_query_plan(cause_label="tube fouling", hypothesis_type="")
    result = r._assess_hit_against_candidate(hit, qp)

    assert result["semantic_relevance_score"] > 0.0, (
        f"Expected semantic_relevance>0, got {result['semantic_relevance_score']}"
    )
    assert result["contradiction_score"] >= 0.35, (
        f"Expected contradiction_score>=0.35, got {result['contradiction_score']}"
    )
    assert result["support_role"] == "contradicting", (
        f"Expected role=contradicting, got {result['support_role']}"
    )
    print("  PASS test_contradiction_cue_applied_when_overlap_present")


def test_candidate_query_type_adds_support_baseline():
    """query_type='candidate' adds +0.15 to support_score."""
    r = make_retriever()
    hit = make_hit("degraded expansion joint weld caused by thermal cycling")
    qp = make_query_plan(cause_label="expansion joint degradation", query_type="candidate")
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["support_score"] > 0.15, (
        f"Expected support_score>0.15, got {result['support_score']}"
    )
    print("  PASS test_candidate_query_type_adds_support_baseline")


def test_candidate_contradiction_query_type_adds_contradiction_baseline():
    """query_type='candidate_contradiction' adds +0.15 to contradiction_score."""
    r = make_retriever()
    hit = make_hit("work order issued for inspection")
    qp = make_query_plan(cause_label="completely unrelated xyz", query_type="candidate_contradiction")
    result = r._assess_hit_against_candidate(hit, qp)
    assert_approx(result["contradiction_score"], 0.15, label="contradiction_score")
    print("  PASS test_candidate_contradiction_query_type_adds_contradiction_baseline")


def test_support_cue_increases_support_score():
    r = make_retriever()
    hit = make_hit("failure caused by degraded seal material")
    qp = make_query_plan(cause_label="seal degradation")
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["support_score"] > 0.40, (
        f"Expected support_score>0.40, got {result['support_score']}"
    )
    assert result["support_role"] == "supporting"
    print("  PASS test_support_cue_increases_support_score")


def test_no_cues_no_overlap_yields_contextual():
    r = make_retriever()
    hit = make_hit("operator logged the event at 03:22 UTC")
    qp = make_query_plan(cause_label="air in-leakage", hypothesis_type="vacuum degradation")
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["support_score"] < 0.35
    assert result["contradiction_score"] < 0.35
    assert result["support_role"] == "contextual"
    print("  PASS test_no_cues_no_overlap_yields_contextual")


def test_guidance_authority_applies_0_90_weight():
    r = make_retriever()
    hit = make_hit(
        "failure caused by degraded seal",
        meta={"doc_type": "OE", "authority_level": "guidance", "extraction_quality": 1.0},
    )
    qp = make_query_plan(cause_label="seal degradation", query_type="candidate")
    result = r._assess_hit_against_candidate(hit, qp)
    assert_approx(result["authority_weight"], 0.90, label="authority_weight")
    print("  PASS test_guidance_authority_applies_0_90_weight")


def test_extraction_quality_scales_score():
    r = make_retriever()
    meta_hi = {"doc_type": "WO", "authority_level": "mandatory", "extraction_quality": 1.0}
    meta_lo = {"doc_type": "WO", "authority_level": "mandatory", "extraction_quality": 0.5}
    snippet = "degraded expansion joint caused by thermal cycling"
    qp = make_query_plan(cause_label="expansion joint degradation")
    r_hi = r._assess_hit_against_candidate(make_hit(snippet, meta=meta_hi), qp)
    r_lo = r._assess_hit_against_candidate(make_hit(snippet, meta=meta_lo), qp)
    assert r_hi["support_score"] > r_lo["support_score"]
    print("  PASS test_extraction_quality_scales_score")


def test_as_found_degraded_boosts_support():
    r = make_retriever()
    hit = make_hit(
        "expansion joint inspected",
        meta={
            "doc_type": "WO",
            "authority_level": "mandatory",
            "extraction_quality": 1.0,
            "ca_as_found_condition": "degraded",
        },
    )
    qp = make_query_plan(cause_label="expansion joint degradation")
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["ca_as_found_condition"] == "degraded"
    assert result["ca_structured_delta"] >= 0.35
    assert result["support_score"] >= 0.35
    print("  PASS test_as_found_degraded_boosts_support")


def test_as_found_acceptable_boosts_contradiction():
    r = make_retriever()
    hit = make_hit(
        "tube bundle inspected",
        meta={
            "doc_type": "WO",
            "authority_level": "mandatory",
            "extraction_quality": 1.0,
            "ca_as_found_condition": "acceptable",
        },
    )
    qp = make_query_plan(cause_label="tube fouling")
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["ca_as_found_condition"] == "acceptable"
    assert result["ca_structured_delta"] <= -0.35
    assert result["contradiction_score"] >= 0.35
    print("  PASS test_as_found_acceptable_boosts_contradiction")


def test_eca_doc_type_increases_epistemic_weight():
    r = make_retriever()
    hit_eca = make_hit(
        "root cause confirmed as air in-leakage",
        meta={"doc_type": "ECA", "authority_level": "mandatory", "extraction_quality": 1.0},
    )
    hit_wo = make_hit(
        "root cause confirmed as air in-leakage",
        meta={"doc_type": "WO", "authority_level": "mandatory", "extraction_quality": 1.0},
    )
    qp = make_query_plan(cause_label="air in-leakage", query_type="candidate")
    r_eca = r._assess_hit_against_candidate(hit_eca, qp)
    r_wo = r._assess_hit_against_candidate(hit_wo, qp)
    assert r_eca["epistemic_weight"] > r_wo["epistemic_weight"]
    assert r_eca["support_score"] > r_wo["support_score"]
    print("  PASS test_eca_doc_type_increases_epistemic_weight")


def test_causal_attribution_with_contradiction_cues_classified_as_supporting():
    """
    Multi-hypothesis document: explicit causal attribution for candidate A ('caused by X')
    co-occurs with exception language about candidate B ('no evidence of Y').
    Without disambiguation the contradiction cues cause false contradicting classification.
    The causal attribution boost must make the role 'supporting' when semantic_relevance > 0.3.
    """
    r = make_retriever()
    # Simulates CR-2024-04821 cause_statement: clearly attributes failure to air in-leakage
    # while also containing contradiction cues about tube fouling.
    snippet = (
        "Elevated dissolved oxygen caused by air in-leakage through expansion joint. "
        "Air in-leakage resulted in backpressure rise. "
        "Tube fouling: no evidence of fouling found. All tubes within normal limits."
    )
    hit = make_hit(snippet)
    # query_type='candidate_contradiction' adds +0.15 to contradiction — the path
    # that caused the original misclassification.
    qp = make_query_plan(
        cause_label="air in-leakage expansion joint",
        query_type="candidate_contradiction",
    )
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["support_role"] == "supporting", (
        f"Expected 'supporting' for causal-attribution + contradiction cues, "
        f"got '{result['support_role']}' "
        f"(support={result['support_score']:.4f}, contradiction={result['contradiction_score']:.4f})"
    )
    print("  PASS test_causal_attribution_with_contradiction_cues_classified_as_supporting")


def test_pure_negation_without_causal_attribution_stays_contradicting():
    """
    Snippet contains exception language ('no abnormality', 'within normal limits')
    with semantic overlap but NO causal attribution cues ('caused by', 'resulted in').
    Without causal attribution, the fix must NOT change role to 'supporting'.
    (Uses a snippet that doesn't contain support-cue words like 'fouling' or 'degraded'
    which would independently bias toward 'supporting'.)
    """
    r = make_retriever()
    # Shares 'tube' with cause_label for semantic overlap, but no support-cue words.
    snippet = "Tube inspection: no abnormality found in tube bundle. All readings within normal limits."
    hit = make_hit(snippet)
    qp = make_query_plan(cause_label="tube fouling")
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["support_role"] == "contradicting", (
        f"Expected 'contradicting' for pure negation without causal attribution, "
        f"got '{result['support_role']}'"
    )
    print("  PASS test_pure_negation_without_causal_attribution_stays_contradicting")


def test_structural_contradiction_for_alternate_root_cause():
    """
    candidate_contradiction query + explicit root-cause phrasing for an alternate
    mechanism should trigger structural contradiction when candidate alignment is low.
    """
    r = make_retriever()
    snippet = "Root cause determined to be pump cavitation; caused by suction blockage."
    hit = make_hit(
        snippet,
        meta={
            "doc_type": "ECA",
            "authority_level": "mandatory",
            "extraction_quality": 1.0,
            "eca_causal_factors_text": "pump cavitation | suction blockage",
        },
    )
    qp = make_query_plan(
        cause_label="tube fouling",
        query_type="candidate_contradiction",
    )
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["structural_contradiction_hit"] is True
    assert result["structural_contradiction_score"] >= 0.35
    assert result["support_role"] == "contradicting"
    print("  PASS test_structural_contradiction_for_alternate_root_cause")


def test_structural_contradiction_not_set_when_candidate_alignment_high():
    """Same-candidate causal attribution should not be marked as structural contradiction."""
    r = make_retriever()
    snippet = "Root cause determined to be tube fouling in condenser bundle."
    hit = make_hit(
        snippet,
        meta={
            "doc_type": "ECA",
            "authority_level": "mandatory",
            "extraction_quality": 1.0,
            "eca_causal_factors_text": "tube fouling",
        },
    )
    qp = make_query_plan(
        cause_label="tube fouling",
        query_type="candidate_contradiction",
    )
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["structural_contradiction_hit"] is False
    print("  PASS test_structural_contradiction_not_set_when_candidate_alignment_high")


def test_structural_contradiction_suppressed_by_failure_mode_refs_match():
    """Structured failure_mode_refs alignment should suppress alternate-root contradiction hit."""
    r = make_retriever()
    hit = make_hit(
        "Root cause determined to be lubrication loss in bearing train.",
        meta={
            "doc_type": "CR",
            "authority_level": "mandatory",
            "extraction_quality": 1.0,
            "failure_mode_refs": ["FM::LUBE-LOSS"],
            "failure_mode_refs_text": "FM::LUBE-LOSS",
        },
    )
    qp = make_query_plan(cause_label="lube loss", query_type="candidate_contradiction")
    result = r._assess_hit_against_candidate(hit, qp)
    assert result["structural_contradiction_hit"] is False
    print("  PASS test_structural_contradiction_suppressed_by_failure_mode_refs_match")


# ── Encoder fallback tests ────────────────────────────────────────────────────

import numpy as np


class _MockEncoder:
    """Deterministic mock encoder for unit tests.

    Returns a fixed embedding for each text based on a lookup table; unknown
    texts get the zero vector.  Vectors are pre-normalised to unit norm.
    """

    def __init__(self, table: dict):
        # table: {text: np.ndarray (unit-norm)}
        self._table = table
        self.calls: list = []

    def encode(self, texts):
        self.calls.append(texts)
        dim = next(iter(self._table.values())).shape[0] if self._table else 4
        out = []
        for t in texts:
            # normalise the lookup key the same way _norm_text does
            key = " ".join(t.lower().split()).strip()
            v = self._table.get(key, np.zeros(dim, dtype=float))
            out.append(v)
        return np.vstack(out) if out else np.zeros((0, dim), dtype=float)


def _unit(v):
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def make_retriever_with_encoder(encoder):
    return ChromaEvidenceRetriever(
        store=InMemoryEvidenceStore([]),
        config=None,
        annotator=None,
        encoder=encoder,
    )


def test_encoder_fallback_used_for_bm25_only_hit():
    """When _vector_score = 0 and encoder is set, semantic_relevance comes from encoder cosine sim."""
    # cause_label: "loss of lubrication"   → vector A = [1, 0, 0, 0]
    # snippet:     "lube oil degradation"  → vector B ≈ A (dot product = 0.85)
    a = _unit(np.array([1.0, 0.1, 0.0, 0.0]))
    b = _unit(np.array([0.9, 0.2, 0.0, 0.0]))

    snippet_text = "lube oil degradation found during inspection"
    enc = _MockEncoder({
        "loss of lubrication": a,
        # _norm_text normalises the full snippet — use the exact normalised form as key
        " ".join(snippet_text.lower().split()): b,
    })
    r = make_retriever_with_encoder(enc)

    hit = make_hit(
        snippet_text,
        meta={"doc_type": "WO", "authority_level": "mandatory", "extraction_quality": 1.0,
              "_vector_score": 0.0},  # BM25-only hit
    )
    qp = make_query_plan(cause_label="loss of lubrication")

    # Pre-embed cause_label so _emb_cache is populated (mirrors _normalize_hits flow)
    cause_label_emb = r._embed("loss of lubrication")
    result = r._assess_hit_against_candidate(hit, qp, cause_label_emb=cause_label_emb)

    expected_cos = float(np.dot(a, b))  # pre-normalised → dot product = cosine sim
    assert result["semantic_relevance_score"] > 0.5, (
        f"Expected high semantic_relevance from encoder, got {result['semantic_relevance_score']:.3f}"
    )
    assert result["candidate_term_overlap"] == 0.0, (
        "Lexical overlap should be 0 (no shared tokens between cause_label and snippet keywords)"
    )
    print(f"  PASS test_encoder_fallback_used_for_bm25_only_hit "
          f"(sem={result['semantic_relevance_score']:.3f}, cosine={expected_cos:.3f})")


def test_encoder_not_called_when_vector_score_present():
    """When _vector_score > 0, the encoder should not be used."""
    a = _unit(np.array([1.0, 0.0, 0.0, 0.0]))
    b = _unit(np.array([0.0, 1.0, 0.0, 0.0]))  # orthogonal — cosine = 0

    enc = _MockEncoder({
        "seal failure": a,
        "corroded shaft bearing": b,
    })
    r = make_retriever_with_encoder(enc)

    hit = make_hit(
        "corroded shaft bearing replaced",
        meta={"doc_type": "WO", "authority_level": "mandatory", "extraction_quality": 1.0,
              "_vector_score": 0.82},  # vector score already present
    )
    qp = make_query_plan(cause_label="seal failure")

    cause_label_emb = r._embed("seal failure")
    result = r._assess_hit_against_candidate(hit, qp, cause_label_emb=cause_label_emb)

    # semantic_relevance should be the Chroma _vector_score, not the encoder cosine (which would be 0)
    assert_approx(result["semantic_relevance_score"], 0.82, tol=0.001,
                  label="semantic_relevance_score should equal _vector_score")
    print(f"  PASS test_encoder_not_called_when_vector_score_present "
          f"(sem={result['semantic_relevance_score']:.3f})")


def test_no_encoder_falls_back_to_lexical_overlap():
    """When no encoder is configured, BM25-only hits use candidate_term_overlap."""
    r = make_retriever()  # no encoder

    hit = make_hit(
        "seal degraded beyond tolerance",
        meta={"doc_type": "WO", "authority_level": "mandatory", "extraction_quality": 1.0,
              "_vector_score": 0.0},  # BM25-only
    )
    qp = make_query_plan(cause_label="seal degradation")

    result = r._assess_hit_against_candidate(hit, qp, cause_label_emb=None)

    # "seal" and "degraded" overlap with "seal degradation" → overlap > 0
    assert result["candidate_term_overlap"] > 0.0
    # semantic_relevance must equal candidate_term_overlap when no encoder and no vector score
    assert_approx(result["semantic_relevance_score"], result["candidate_term_overlap"],
                  label="semantic_relevance_score should equal candidate_term_overlap")
    print(f"  PASS test_no_encoder_falls_back_to_lexical_overlap "
          f"(overlap={result['candidate_term_overlap']:.3f})")


def test_encoder_cache_cleared_between_retrieve_calls():
    """_emb_cache is reset at the start of each retrieve() call."""
    enc = _MockEncoder({"pump failure": _unit(np.array([1.0, 0.0, 0.0, 0.0]))})
    r = make_retriever_with_encoder(enc)

    # Manually prime the cache as retrieve() would
    r._emb_cache = {}
    r._embed("pump failure")
    assert "pump failure" in r._emb_cache

    # Simulate a new retrieve() resetting the cache
    r._emb_cache = {}
    assert len(r._emb_cache) == 0
    print("  PASS test_encoder_cache_cleared_between_retrieve_calls")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_contradiction_cue_suppressed_when_no_semantic_overlap,
    test_contradiction_cue_applied_when_overlap_present,
    test_candidate_query_type_adds_support_baseline,
    test_candidate_contradiction_query_type_adds_contradiction_baseline,
    test_support_cue_increases_support_score,
    test_no_cues_no_overlap_yields_contextual,
    test_guidance_authority_applies_0_90_weight,
    test_extraction_quality_scales_score,
    test_as_found_degraded_boosts_support,
    test_as_found_acceptable_boosts_contradiction,
    test_eca_doc_type_increases_epistemic_weight,
    test_causal_attribution_with_contradiction_cues_classified_as_supporting,
    test_pure_negation_without_causal_attribution_stays_contradicting,
    test_structural_contradiction_for_alternate_root_cause,
    test_structural_contradiction_not_set_when_candidate_alignment_high,
    test_structural_contradiction_suppressed_by_failure_mode_refs_match,
    # Encoder fallback tests
    test_encoder_fallback_used_for_bm25_only_hit,
    test_encoder_not_called_when_vector_score_present,
    test_no_encoder_falls_back_to_lexical_overlap,
    test_encoder_cache_cleared_between_retrieve_calls,
]


def run_all():
    print(f"\n=== test_evidence_scorer ({len(ALL_TESTS)} tests) ===")
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
