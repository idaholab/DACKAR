"""
Unit tests for dackar.RCA.ner.hybrid_ner.generators.description_embed_generator.

Drives the generator with an INJECTED deterministic embedder (no SBERT, no
network) and a tiny label-description JSON, per the Plan D Tier-2 pattern. Golden
coverage of generate()/generate_for_candidate() — the per-candidate embedding
loop that Plan B will batch — so the refactor can assert identical output.
"""
from __future__ import annotations

import json

import numpy as np

from dackar.RCA.ner.hybrid_ner.generators.description_embed_generator import (
    DescriptionEmbedGenerator,
)
from dackar.RCA.ner.hybrid_ner.models import CandidateSpan, Document
from dackar.RCA.ner.hybrid_ner.schema import SchemaIndex


def _labels_json(tmp_path):
    p = tmp_path / "labels.json"
    json.dump(
        {"deg_mech": {
            "description": "degradation mechanism wear corrosion pitting",
            "canonical_examples": ["corrosion"],
            "keywords": ["corrosion"],
        }},
        p.open("w"),
    )
    return str(p)


def _fake_embedder(texts):
    # 2-D space: corrosion-ish text -> axis [1,0]; everything else -> [0,1].
    return np.array([[1.0, 0.0] if "corrosion" in t.lower() else [0.0, 1.0]
                     for t in texts])


def _schema():
    return SchemaIndex(
        label_to_group={"deg_mech": "G4_MECHANISM_PROCESS"},
        group_to_labels={"G4_MECHANISM_PROCESS": {"deg_mech"}},
        pair_decision={}, conditional_rules=[],
    )


def _fitted(tmp_path, embedder=_fake_embedder, **kw):
    g = DescriptionEmbedGenerator(
        label_json_path=_labels_json(tmp_path), embedder=embedder,
        score_threshold=0.5, **kw,
    )
    g.fit(_schema())
    return g


def _span(text, start, end):
    return CandidateSpan(span_id=f"c-{start}", doc_id="D1", start=start, end=end, text=text)


# ---------------------------------------------------------------------------
# generate_for_candidate — a matching mechanism span
# ---------------------------------------------------------------------------

def test_matching_span_gets_label(tmp_path):
    g = _fitted(tmp_path)
    doc = Document(doc_id="D1", text="observed corrosion damage")
    out = g.generate_for_candidate(doc, _span("corrosion", 9, 18))
    assert [h.label for h in out] == ["deg_mech"]
    assert out[0].group == "G4_MECHANISM_PROCESS"


def test_non_matching_span_gets_nothing(tmp_path):
    g = _fitted(tmp_path)
    doc = Document(doc_id="D1", text="the pump assembly")
    out = g.generate_for_candidate(doc, _span("pump", 4, 8))
    assert out == []  # embeds to [0,1] -> low sim -> below threshold


# ---------------------------------------------------------------------------
# Rule 1 gate: deg_mech requires a mechanism cue in span or context
# ---------------------------------------------------------------------------

def test_deg_mech_requires_mechanism_cue(tmp_path):
    # Fake embedder forces high similarity even for a non-mechanism span, so only
    # Rule 1 can block it. Span "widget" has no mechanism cue in span or context.
    g = _fitted(tmp_path, embedder=lambda texts: np.array([[1.0, 0.0]] * len(texts)))
    doc = Document(doc_id="D1", text="the widget here")
    out = g.generate_for_candidate(doc, _span("widget", 4, 10))
    assert out == []  # high sim, but no mechanism cue -> Rule 1 rejects


# ---------------------------------------------------------------------------
# generate() over several candidates — the loop Plan B will batch
# ---------------------------------------------------------------------------

def test_generate_returns_per_candidate_tuples(tmp_path):
    g = _fitted(tmp_path)
    doc = Document(doc_id="D1", text="corrosion and a pump")
    cands = [_span("corrosion", 0, 9), _span("pump", 16, 20)]
    out = g.generate(doc, cands)
    # shape: list of (cand, [hyps]) in candidate order
    assert [c.span_id for c, _ in out] == ["c-0", "c-16"]
    by_id = {c.span_id: [h.label for h in lh] for c, lh in out}
    assert by_id["c-0"] == ["deg_mech"]   # corrosion matches
    assert by_id["c-16"] == []            # pump does not


def test_generate_empty_candidates(tmp_path):
    g = _fitted(tmp_path)
    assert g.generate(Document(doc_id="D1", text="x"), []) == []


# ---------------------------------------------------------------------------
# Batching invariance: scoring one-at-a-time vs together gives the same labels
# (locks the behavior Plan B's batch refactor must preserve)
# ---------------------------------------------------------------------------

def test_scoring_is_independent_of_candidate_grouping(tmp_path):
    g = _fitted(tmp_path)
    doc = Document(doc_id="D1", text="corrosion and a pump near a bearing")
    cands = [_span("corrosion", 0, 9), _span("pump", 16, 20), _span("bearing", 28, 35)]
    # together
    together = {c.span_id: [h.label for h in lh] for c, lh in g.generate(doc, cands)}
    # one at a time
    single = {}
    for c in cands:
        single[c.span_id] = [h.label for h in g.generate_for_candidate(doc, c)]
    assert together == single


def test_batching_invariance_on_tfidf_path(tmp_path, monkeypatch):
    """Batching invariance must also hold on the REAL TF-IDF fallback embedder
    (no injected embedder, SBERT forced off) — not just the fake embedder."""
    import dackar.RCA.ner.hybrid_ner.generators.description_embed_generator as mod
    monkeypatch.setattr(mod, "_HAS_SBERT", False)

    g = DescriptionEmbedGenerator(
        label_json_path=_labels_json(tmp_path),
        embedder=None, use_tfidf_fallback=True, score_threshold=0.3,
    )
    g.fit(_schema())
    doc = Document(doc_id="D1", text="corrosion and a pump near a bearing")
    cands = [_span("corrosion", 0, 9), _span("pump", 16, 20), _span("bearing", 28, 35)]

    together = {c.span_id: [h.label for h in lh] for c, lh in g.generate(doc, cands)}
    single = {c.span_id: [h.label for h in g.generate_for_candidate(doc, c)] for c in cands}
    assert together == single
