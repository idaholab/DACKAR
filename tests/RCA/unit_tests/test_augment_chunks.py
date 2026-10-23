"""
Tests for dackar.RCA.ner.augment_chunks serialization — specifically that a
chunk with no page metadata (page_start/page_end absent in the source JSONL)
serializes null pages instead of crashing on int(None).

Regression for the four bare ``int(ctx.page_*)`` sites: ``build_chunk_metadata``
and ``build_processed_text_record`` here, and ``_trusted_citations`` in the
reliability summarizer (covered in test_reliability_summarizer.py).
"""
from __future__ import annotations

from dackar.RCA.ner.augment_chunks import (
    build_chunk_metadata,
    build_processed_text_record,
)
from dackar.RCA.summarizers.reliability_summarizer import (
    ChunkContext,
    NERSeed,
    empty_rca_frame,
    empty_retrieval_summary,
)


def _ctx(**kw) -> ChunkContext:
    base = dict(
        doc_id="D1", doc_type="CR", chunk_id="C1",
        section_path="7 Procedure > 7.2 Stroke Time",
        page_start=3, page_end=4,
    )
    base.update(kw)
    return ChunkContext(**base)


def _seed() -> NERSeed:
    return NERSeed(
        systems=[], equipment_ids=[], components=[], mechanisms=[], outcomes=[],
        surveillance_actions=[], maintenance_actions=[], properties=[], tools=[],
    )


def _chunk() -> dict:
    return {"chunk_id": "C1", "doc_id": "D1", "type": "paragraph", "granularity": "paragraph"}


# ---------------------------------------------------------------------------
# build_chunk_metadata  (the :694-695 int(ctx.page_*) site)
# ---------------------------------------------------------------------------

def test_metadata_null_pages_do_not_crash():
    ctx = _ctx(page_start=None, page_end=None)
    meta = build_chunk_metadata(
        _chunk(), ctx, _seed(),
        empty_retrieval_summary(ctx), empty_rca_frame(ctx),
    )
    assert meta["page_start"] is None
    assert meta["page_end"] is None


def test_metadata_int_pages_preserved():
    ctx = _ctx(page_start=3, page_end=4)
    meta = build_chunk_metadata(
        _chunk(), ctx, _seed(),
        empty_retrieval_summary(ctx), empty_rca_frame(ctx),
    )
    assert meta["page_start"] == 3
    assert meta["page_end"] == 4


# ---------------------------------------------------------------------------
# build_processed_text_record  (the :795-796 provenance int(ctx.page_*) site)
# ---------------------------------------------------------------------------

def test_provenance_null_pages_do_not_crash():
    ctx = _ctx(page_start=None, page_end=None)
    rs = empty_retrieval_summary(ctx)
    rf = empty_rca_frame(ctx)
    meta = build_chunk_metadata(_chunk(), ctx, _seed(), rs, rf)
    record = build_processed_text_record(
        doc_id="D1", doc_type="CR", chunk_index=0, chunk=_chunk(),
        ctx=ctx, ner_seed=_seed(), retrieval_summary=rs, rca_frame=rf,
        metadata=meta, embedding_text="", stage5_payload={},
    )
    prov = record["provenance"]
    assert prov["page_start"] is None
    assert prov["page_end"] is None
