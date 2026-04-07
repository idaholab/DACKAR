"""
augment_chunks.py
=================

Second-pass augmentation of mdParser chunks with structured JSON summaries.

Goal
----
Read the mdParser output `*_chunks.jsonl`, and for each chunk that is intended for
vector indexing, generate two structured JSON views using Ollama:

1) retrieval_summary_json
2) rca_frame_json

Then write an enriched JSONL file: `*_chunks_enriched.jsonl`

This module is designed for Option A:
- Do NOT modify your existing pdfParser.py or mdParser.py pipelines.
- Run this as a separate step from a notebook or script.

Inputs
------
- chunks_jsonl_path: path to `<doc_id>_chunks.jsonl` produced by mdParser.py
- document_index: dict loaded from `index/document_index.json` produced by pdfParser.py
- structured_output (optional): dict loaded from `<doc_id>_structured_output.json`
- ner_provider (optional): callable that returns NERSeed per chunk

Outputs
-------
- `<doc_id>_chunks_enriched.jsonl` file (same directory as input chunks file)
- returns output path and summary stats
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import json
import os
import re
from datetime import datetime, timezone

from causal_condition_adapter import extract_stage5_causal_condition

from ..summarizers.reliability_summarizer import (
    detect_doc_type,
    section_role_from_title,
    ChunkContext,
    NERSeed,
    summarize_with_retry,
    flatten_retrieval_summary_for_embedding,
    flatten_rca_frame_for_embedding,
)


# ---------------------------
# Utilities
# ---------------------------

_EQUIP_TAG_DEFAULT_RE = re.compile(r"\b([A-Z]{1,6}-\d{1,6}[A-Z]{0,2})\b")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _safe_optional_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None

def _extract_equipment_ids_quick(text: str, limit: int = 50) -> List[str]:
    """
    Minimal equipment tag extraction for seeding. (You can swap in your full helper.)

    Input: free text
    Output: list of tags like P-101A, MOV-204A, PT-1102...
    """
    if not text:
        return []
    seen = set()
    out: List[str] = []
    for m in _EQUIP_TAG_DEFAULT_RE.finditer(text.upper()):
        tag = m.group(1).strip(".,;:()[]{}<>\"'")
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
        if len(out) >= limit:
            break
    return out


# ---------------------------
# Default NER seed builder
# ---------------------------

def default_ner_seed_from_chunk(chunk: Dict[str, Any]) -> NERSeed:
    """
    Build an NERSeed using only fields already present in mdParser output.

    This is intentionally conservative and works even if you haven't wired your
    full nuclear NER pipeline into mdParser chunks yet.

    Inputs
    ------
    chunk: dict from chunks.jsonl. Typical keys from mdParser include:
      - text: str
      - keywords: list[str]
      - mentions_component_ids: list[str] (if present)
      - standards_refs: dict (if present)
      - type, granularity, etc.

    Output
    ------
    NERSeed with:
      - equipment_ids: derived via regex
      - components: from mentions_component_ids if present
      - everything else empty (you can fill later using your NER outputs)
    """
    txt = chunk.get("text") or ""
    mentions = chunk.get("mentions_component_ids") or []
    if not isinstance(mentions, list):
        mentions = []

    return NERSeed(
        systems=[],
        equipment_ids=_extract_equipment_ids_quick(txt),
        components=[str(x) for x in mentions if x],
        mechanisms=[],
        outcomes=[],
        surveillance_actions=[],
        maintenance_actions=[],
        properties=[],
        tools=[],
    )


# ---------------------------
# Main entrypoint
# ---------------------------

@dataclass
class AugmentStats:
    """
    Statistics returned by augmentation.
    """
    total_chunks: int
    eligible_chunks: int
    summarized_chunks: int
    skipped_already_present: int
    failed_chunks: int
    output_path: str


def augment_chunks_with_structured_summaries(
    chunks_jsonl_path: str | Path,
    *,
    model: Optional[str] = None,
    timeout: int = 90,
    max_tries: int = 3,
    output_suffix: str = "_enriched",
    overwrite: bool = False,
    summarize_granularities: Tuple[str, ...] = ("section", "paragraph"),
    only_indexable: bool = True,
    doc_type_override: Optional[str] = None,
    authority_override: Optional[str] = None,
    ner_seed_provider: Optional[Callable[[Dict[str, Any]], NERSeed]] = None,
    stage5_nlp: Any = None,
    stage5_llm_cfg: Optional[Dict[str, Any]] = None,
) -> AugmentStats:
    """
    Enrich an mdParser chunks.jsonl file with structured summaries (JSON) using Ollama.

    Inputs
    ------
    chunks_jsonl_path:
        Path to `<doc_id>_chunks.jsonl` produced by mdParser.py.
    model:
        Ollama model name. If None, reliability_summarizer uses env OLLAMA_MODEL or default.
    timeout:
        Request timeout seconds per summary call.
    max_tries:
        Retry ladder attempts per summary.
    output_suffix:
        Output file suffix. If input is `abc_chunks.jsonl`, output becomes `abc_chunks_enriched.jsonl`.
    overwrite:
        If False and output exists, raises an error.
    summarize_granularities:
        Which chunk granularities to summarize. mdParser uses "section" and "paragraph" for TextChunk.
    only_indexable:
        If True, only summarize chunks where `index_in_vector_store == True`.
    doc_type_override:
        Force doc_type for all chunks ("SOP", "CR", "WO", "ECA", "OTHER"). If None, auto-detect using early chunk text.
    authority_override:
        Force authority_level: "mandatory","guidance","informational","unknown". If None, SOP->mandatory else informational.
    ner_seed_provider:
        Optional function chunk->NERSeed. If not provided, uses default_ner_seed_from_chunk().
    stage5_nlp:
        Optional initialized NLP pipeline to pass explicitly into Stage 5 causal/condition extraction.

    Output
    ------
    AugmentStats including output_path. Writes a new JSONL alongside input.

    Output record format (per chunk)
    --------------------------------
    Adds (when summarized):
      - retrieval_summary_json: dict
      - rca_frame_json: dict
      - retrieval_summary_text: str (flattened for embeddings)
      - rca_frame_text: str (flattened for embeddings)
      - augmentation: { "status": "ok"|"error", "error": str|None }

    Chunks that are not summarized are written unchanged (plus minimal augmentation status if desired).
    """
    chunks_path = Path(chunks_jsonl_path)
    if not chunks_path.exists():
        raise FileNotFoundError(str(chunks_path))

    out_path = chunks_path.with_name(chunks_path.stem + output_suffix + chunks_path.suffix)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out_path} (set overwrite=True to replace)")

    # Read all chunks
    chunks = list(_iter_jsonl(chunks_path))
    total = len(chunks)

    # Auto-detect doc_type using first eligible text chunk (early_text)
    early_text = ""
    for c in chunks:
        if (c.get("type") == "TextChunk") and (c.get("text")):
            early_text = (c.get("text") or "")[:5000]
            break

    doc_name = chunks_path.name
    inferred_doc_type = detect_doc_type(doc_name=doc_name, source_path=str(chunks_path), early_text=early_text)
    doc_type = (doc_type_override or inferred_doc_type)

    # Default authority
    if authority_override:
        authority = authority_override
    else:
        authority = "mandatory" if doc_type == "SOP" else "informational"

    seed_fn = ner_seed_provider or default_ner_seed_from_chunk

    eligible = 0
    summarized = 0
    skipped_present = 0
    failed = 0

    enriched_records: List[Dict[str, Any]] = []

    for chunk_index, c in enumerate(chunks):
        # Decide eligibility
        if c.get("type") != "TextChunk":
            enriched_records.append(c)
            continue

        gran = c.get("granularity")
        if summarize_granularities and gran not in summarize_granularities:
            enriched_records.append(c)
            continue

        if only_indexable and not c.get("index_in_vector_store", False):
            enriched_records.append(c)
            continue

        eligible += 1

        # If already has summaries and not overwriting per record, skip
        if (not overwrite) and ("retrieval_summary_json" in c or "rca_frame_json" in c):
            c.setdefault("augmentation", {"status": "skipped", "reason": "already_present"})
            skipped_present += 1
            enriched_records.append(c)
            continue

        chunk_text = (c.get("text") or "").strip()
        if not chunk_text:
            c.setdefault("augmentation", {"status": "skipped", "reason": "empty_text"})
            enriched_records.append(c)
            continue

        # Build section role from section title if present
        # mdParser puts `section_title` on paragraph subchunks, and title on section chunks
        section_title = c.get("section_title") or c.get("title") or ""
        role = section_role_from_title(doc_type, str(section_title))

        # Build ChunkContext
        ctx = ChunkContext(
            doc_id=str(c.get("doc_id") or "unknown_doc"),
            doc_type=doc_type,  # use inferred or override
            chunk_id=str(c.get("chunk_id") or "unknown_chunk"),
            section_path=str(section_title or c.get("belongs_to_section") or "unknown_section"),
            page_start=_safe_optional_int(c.get("page_start")),
            page_end=_safe_optional_int(c.get("page_end")),
            authority_level=authority,
            section_role=role,
        )

        # Build NER seed
        seed = seed_fn(c)

        try:
            rs = summarize_with_retry(
                doc_type=doc_type,
                view_type="retrieval_summary",
                ctx=ctx,
                ner_seed=seed,
                chunk_text=chunk_text,
                model=model,
                timeout=timeout,
                max_tries=max_tries,
            )
            rf = summarize_with_retry(
                doc_type=doc_type,
                view_type="rca_frame",
                ctx=ctx,
                ner_seed=seed,
                chunk_text=chunk_text,
                model=model,
                timeout=timeout,
                max_tries=max_tries,
            )

            doc_id = str(c.get("doc_id") or ctx.doc_id)

            stage5_payload = extract_stage5_causal_condition(
                doc_id=doc_id,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                doc_type=doc_type,
                section_role=ctx.section_role,
                nlp=stage5_nlp,
                llm_cfg=stage5_llm_cfg,
            )

            metadata = build_chunk_metadata(
                chunk=c,
                ctx=ctx,
                ner_seed=seed,
                retrieval_summary=rs,
                rca_frame=rf,
                stage5_payload=stage5_payload,
            )

            embedding_text = build_embedding_text(
                chunk_text=chunk_text,
                ner_seed=seed,
                retrieval_summary=rs,
                rca_frame=rf,
                stage5_payload=stage5_payload,
            )

            processed_record = build_processed_text_record(
                doc_id=doc_id,
                doc_type=doc_type,
                chunk_index=chunk_index,
                chunk=c,
                ctx=ctx,
                ner_seed=seed,
                retrieval_summary=rs,
                rca_frame=rf,
                metadata=metadata,
                embedding_text=embedding_text,
                stage5_payload=stage5_payload,
            )

            validation_errors = validate_processed_text_record(processed_record)
            if validation_errors:
                raise ValueError(
                    f"Invalid processed_text_record for chunk_index={chunk_index}: {validation_errors}"
                )

            c["retrieval_summary_json"] = rs
            c["rca_frame_json"] = rf
            c["embedding_text"] = embedding_text
            c["metadata"] = processed_record["metadata"]
            c["processed_text_record"] = processed_record
            c["stage5_causal_condition"] = stage5_payload

            # optional temporary debug fields
            c["retrieval_summary_text"] = flatten_retrieval_summary_for_embedding(rs)
            c["rca_frame_text"] = flatten_rca_frame_for_embedding(rf)

            c["augmentation"] = {
                "status": "ok",
                "doc_type": doc_type,
                "authority_level": authority,
                "stage5_nlp_provided": bool(stage5_nlp is not None),
                "stage5_llm_provided": bool(stage5_llm_cfg is not None),
                "stage5_extractor_used": stage5_payload.get("extractor", {}).get("used", ""),
            }
            summarized += 1

        except Exception as e:
            failed += 1
            c["augmentation"] = {"status": "error", "error": str(e), "doc_type": doc_type}
            # Keep chunk even if it failed, so you can inspect errors later.

        enriched_records.append(c)

    _write_jsonl(out_path, enriched_records)

    return AugmentStats(
        total_chunks=total,
        eligible_chunks=eligible,
        summarized_chunks=summarized,
        skipped_already_present=skipped_present,
        failed_chunks=failed,
        output_path=str(out_path),
    )


def _validate_stage5_alias_consistency(record: Dict[str, Any]) -> List[str]:
    """
    Ensure compatibility aliases remain aligned with enrichment payload.
    """
    errors: List[str] = []
    enrichment = record.get("enrichment") or {}
    stage5 = enrichment.get("stage5_causal_condition") or {}
    cond_alias = record.get("condition_assessment")
    cond_stage5 = stage5.get("condition_state")

    if cond_alias is not None and cond_stage5 is not None and cond_alias != cond_stage5:
        errors.append("condition_assessment_mismatch_with_stage5")

    if stage5:
        if not isinstance(stage5.get("summary_flags", {}), dict):
            errors.append("stage5_summary_flags_not_object")
        if not isinstance(stage5.get("extracted_causal_statements", []), list):
            errors.append("stage5_extracted_causal_statements_not_list")
        if not isinstance(stage5.get("condition_state", {}), dict):
            errors.append("stage5_condition_state_not_object")
        if not isinstance(stage5.get("procedural_deviation", {}), dict):
            errors.append("stage5_procedural_deviation_not_object")

    return errors

def _extract_causal_spans(
    stage5_payload: Optional[Dict[str, Any]],
    min_confidence: float = 0.35,
) -> Tuple[List[str], List[str]]:
    """Return (cause_texts, effect_texts) from Stage 5 statements at or above min_confidence.

    cause_texts  — causal precursor spans; routed to mechanisms in NERSeed backfill.
    effect_texts — failure/outcome spans; routed to outcomes in NERSeed backfill.

    min_confidence=0.35 corresponds to at least one filled field (connector OR a cause/effect
    span) in _score_causal_statement, filtering out the emptiest extractions.
    """
    cause_texts: List[str] = []
    effect_texts: List[str] = []
    for stmt in ((stage5_payload or {}).get("extracted_causal_statements") or []):
        if not isinstance(stmt, dict):
            continue
        if float(stmt.get("confidence", 0.0)) < min_confidence:
            continue
        c = str(stmt.get("cause_text") or "").strip()
        e = str(stmt.get("effect_text") or "").strip()
        if c:
            cause_texts.append(c)
        if e:
            effect_texts.append(e)
    return cause_texts, effect_texts


def build_embedding_text(
    chunk_text: str,
    ner_seed: NERSeed,
    retrieval_summary: Dict[str, Any],
    rca_frame: Dict[str, Any],
    stage5_payload: Optional[Dict[str, Any]] = None,
    max_chars: int = 3500,
) -> str:
    parts: List[str] = []

    # Canonical labels first, per processed-record design
    causals = ((stage5_payload or {}).get("extracted_causal_statements") or [])
    measurements = (retrieval_summary.get("numbers_limits") or [])

    # Backfill causal spans into FM_LABELS so the embedding picks up event spans
    # that the gazetteer may have missed (Fix B: causal-slot backfill).
    causal_cause_spans, causal_effect_spans = _extract_causal_spans(stage5_payload)
    fm_labels = _uniq(
        list(ner_seed.mechanisms or [])
        + list(ner_seed.outcomes or [])
        + causal_cause_spans
        + causal_effect_spans
    )
    if fm_labels:
        parts.append("FM_LABELS: " + ", ".join(fm_labels))

    if causals:
        causal_lines = []
        for row in causals[:10]:
            c = str(row.get("cause_text") or "").strip()
            e = str(row.get("effect_text") or "").strip()
            k = str(row.get("connector") or "").strip()
            if c or e:
                causal_lines.append(f"{c} {k} {e}".strip())
        if causal_lines:
            parts.append("CAUSAL_STATEMENTS: " + " | ".join(causal_lines))

    if measurements:
        parts.append("MEASUREMENTS: " + ", ".join(str(x).strip() for x in measurements[:15] if str(x).strip()))

    # spaCy-extracted signals: structured measurements, temporal refs, doc cross-refs
    spacy_measurements = getattr(ner_seed, "measurements", None) or []
    if spacy_measurements:
        meas_parts = [
            f"{m.get('value', '')} {m.get('unit', '')}".strip()
            for m in spacy_measurements[:10]
            if isinstance(m, dict) and m.get("value") is not None
        ]
        if meas_parts:
            parts.append("UNIT_MEASUREMENTS: " + ", ".join(meas_parts))

    temporal_refs = getattr(ner_seed, "temporal_refs", None) or []
    if temporal_refs:
        parts.append("TEMPORAL_REFS: " + ", ".join(temporal_refs[:10]))

    doc_refs = getattr(ner_seed, "doc_refs", None) or []
    if doc_refs:
        parts.append("DOC_REFS: " + ", ".join(doc_refs[:15]))

    # Seed-derived normalization
    if ner_seed.systems:
        parts.append("SYSTEMS_SEED: " + ", ".join(ner_seed.systems))
    if ner_seed.equipment_ids:
        parts.append("EQUIPMENT_IDS_SEED: " + ", ".join(ner_seed.equipment_ids))
    if ner_seed.components:
        parts.append("COMPONENTS_SEED: " + ", ".join(ner_seed.components))
    if ner_seed.mechanisms:
        parts.append("MECHANISMS_SEED: " + ", ".join(ner_seed.mechanisms))
    if ner_seed.outcomes:
        parts.append("OUTCOMES_SEED: " + ", ".join(ner_seed.outcomes))
    if ner_seed.maintenance_actions:
        parts.append("MAINT_ACTIONS_SEED: " + ", ".join(ner_seed.maintenance_actions))
    if ner_seed.surveillance_actions:
        parts.append("SURV_ACTIONS_SEED: " + ", ".join(ner_seed.surveillance_actions))
    if ner_seed.properties:
        parts.append("PROPERTIES_SEED: " + ", ".join(ner_seed.properties))
    if ner_seed.tools:
        parts.append("TOOLS_SEED: " + ", ".join(ner_seed.tools))

    # Summaries after canonical extracted content
    rs_text = flatten_retrieval_summary_for_embedding(retrieval_summary)
    rf_text = flatten_rca_frame_for_embedding(rca_frame)
    if rs_text.strip():
        parts.append(rs_text)
    if rf_text.strip():
        parts.append(rf_text)

    # Raw text fallback / anchor
    snippet = (chunk_text or "").strip()
    if snippet:
        if len(snippet) > 750:
            snippet = snippet[:750]
        parts.append("TEXT_SNIPPET: " + snippet)

    text = "\n".join(p for p in parts if p.strip())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text

def build_chunk_metadata(
    chunk: Dict[str, Any],
    ctx: ChunkContext,
    ner_seed: NERSeed,
    retrieval_summary: Dict[str, Any],
    rca_frame: Dict[str, Any],
    stage5_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entities = retrieval_summary.get("entities") or {}

    symptoms = retrieval_summary.get("symptoms_outcomes") or []
    mechanisms = retrieval_summary.get("mechanisms") or []
    diagnostics = retrieval_summary.get("diagnostics") or []
    corrective_actions = retrieval_summary.get("corrective_actions") or []
    numbers_limits = retrieval_summary.get("numbers_limits") or []
    hypotheses = rca_frame.get("hypotheses") or []
    tests_to_confirm = rca_frame.get("tests_to_confirm") or []
    candidate_actions = rca_frame.get("candidate_actions") or []
    constraints = rca_frame.get("constraints") or []

    # Causal-slot backfill: cause_texts → mechanisms, effect_texts → outcomes.
    # Runs after stage5 so the dep-tree causal spans enrich metadata even when
    # the gazetteer didn't produce a match for those surface forms (Fix B).
    causal_cause_spans, causal_effect_spans = _extract_causal_spans(stage5_payload)

    # NER / deterministic extraction is authoritative; summaries and causal slots are additive.
    systems = _uniq((ner_seed.systems or []) + (entities.get("systems") or []))
    equipment_ids = _uniq(ner_seed.equipment_ids or [])
    components = _uniq((ner_seed.components or []) + (entities.get("components") or []))
    mechanisms_all = _uniq((ner_seed.mechanisms or []) + mechanisms + causal_cause_spans)
    outcomes_all = _uniq((ner_seed.outcomes or []) + symptoms + causal_effect_spans)
    maintenance_actions = _uniq((ner_seed.maintenance_actions or []) + corrective_actions + candidate_actions)
    surveillance_actions = _uniq((ner_seed.surveillance_actions or []) + diagnostics + tests_to_confirm)
    properties = _uniq((ner_seed.properties or []) + numbers_limits)
    tools = _uniq(ner_seed.tools or [])

    stage5_flags = (stage5_payload or {}).get("summary_flags", {})

    return {
        "doc_type": ctx.doc_type,
        "authority_level": ctx.authority_level,
        "section_role": ctx.section_role,
        "page_start": int(ctx.page_start),
        "page_end": int(ctx.page_end),

        # retrieval/filter fields
        "equipment_ids": equipment_ids,
        "system_names": systems,
        "component_names": components,
        "mechanisms": mechanisms_all,
        "failure_outcomes": outcomes_all,
        "maintenance_actions": maintenance_actions,
        "surveillance_actions": surveillance_actions,
        "tools_methods": tools,

        # keep properties only if compact
        "properties_or_limits": properties[:25],

        # document and alarm cross-references from NER
        "doc_refs": _uniq(list(getattr(ner_seed, "doc_refs", None) or [])),
        "alarm_ids": _uniq(list(getattr(ner_seed, "alarm_ids", None) or [])),

        # boolean retrieval flags
        "has_causal_language": bool(
            hypotheses
            or constraints
            or stage5_flags.get("has_explicit_causal_statement", False)
        ),
        "has_diagnostics": bool(diagnostics or tests_to_confirm),
        "has_maintenance_action": bool(maintenance_actions),
        "has_surveillance_action": bool(surveillance_actions),
        "has_failure_signal": bool(outcomes_all or mechanisms_all),
        "has_temporal_signal": bool(
            getattr(ner_seed, "temporal_refs", None)
            or getattr(ner_seed, "temporal_relations", None)
        ),
        "has_conjecture": bool(getattr(ner_seed, "conjectures", None)),
        "has_location_signal": bool(getattr(ner_seed, "locations", None)),
    }

def _uniq(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        x = str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def build_processed_text_record(
    *,
    doc_id: str,
    doc_type: str,
    chunk_index: int,
    chunk: Dict[str, Any],
    ctx: ChunkContext,
    ner_seed: NERSeed,
    retrieval_summary: Dict[str, Any],
    rca_frame: Dict[str, Any],
    metadata: Dict[str, Any],
    embedding_text: str,
    stage5_payload: Dict[str, Any],
) -> Dict[str, Any]:
    chunk_id = str(chunk.get("chunk_id") or f"{doc_id}::{chunk_index}")
    source_hash = (
        chunk.get("content_hash")
        or chunk.get("source_hash")
        or chunk.get("doc_hash")
        or chunk.get("hash")
    )

    # Causal-slot backfill (Fix B): pull cause/effect spans from Stage 5 dep-tree
    # extraction so syntactic failure events reach extracted_entities even when the
    # gazetteer has no exact-phrase match for those surface forms.
    _causal_cause_spans, _causal_effect_spans = _extract_causal_spans(stage5_payload)

    return {
        "record_id": f"{doc_id}::{chunk_index}",
        "doc_id": doc_id,
        "doc_type": doc_type,
        "chunk_index": int(chunk_index),
        "embedding_text": embedding_text,
        "metadata": {
            **metadata,
            "has_explicit_causal_statement": bool(
                stage5_payload.get("summary_flags", {}).get("has_explicit_causal_statement", False)
            ),
            "has_condition_state": bool(
                stage5_payload.get("summary_flags", {}).get("has_condition_state", False)
            ),
            "has_procedural_deviation": bool(
                stage5_payload.get("summary_flags", {}).get("has_procedural_deviation", False)
            ),
        },

        "provenance": {
            "chunk_id": chunk_id,
            "section_path": ctx.section_path,
            "page_start": int(ctx.page_start),
            "page_end": int(ctx.page_end),
            "authority_level": ctx.authority_level,
            "section_role": ctx.section_role,
            "source_chunk_type": chunk.get("type"),
            "source_granularity": chunk.get("granularity"),
            "index_in_vector_store": bool(chunk.get("index_in_vector_store", False)),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "nlp_pipeline_version": "augment_chunks_v2",
            "embedding_model": os.getenv("OLLAMA_MODEL", "") or "unknown",
            "source_hash": source_hash,
            "entity_resolution_version": "ner_seed_v2",
         },
        "section_path": ctx.section_path,
        "condition_assessment": stage5_payload.get("condition_state"),

        "enrichment": {
            "extracted_entities": {
                # Core entity buckets (hybrid NER pipeline + causal-slot backfill).
                # cause_texts from Stage 5 dep-tree extraction are merged into mechanisms;
                # effect_texts are merged into outcomes.  This ensures that failure events
                # described syntactically (e.g. "pump failed to start") are captured even
                # when the gazetteer has no exact-phrase match for the surface form (Fix B).
                "systems": _uniq(list(ner_seed.systems or [])),
                "equipment_ids": _uniq(list(ner_seed.equipment_ids or [])),
                "components":  _uniq(list(ner_seed.components or [])),
                "mechanisms":  _uniq(list(ner_seed.mechanisms or []) + _causal_cause_spans),
                "outcomes":    _uniq(list(ner_seed.outcomes or []) + _causal_effect_spans),
                "surveillance_actions": _uniq(list(ner_seed.surveillance_actions or [])),
                "maintenance_actions": _uniq(list(ner_seed.maintenance_actions or [])),
                "properties": _uniq(list(ner_seed.properties or [])),
                "tools": _uniq(list(ner_seed.tools or [])),
                # Document and alarm cross-references (regex extractors)
                "doc_refs": _uniq(list(getattr(ner_seed, "doc_refs", None) or [])),
                "alarm_ids": _uniq(list(getattr(ner_seed, "alarm_ids", None) or [])),
                # spaCy-extracted signals (Tier 1 SpacyAnnotator)
                "measurements": list(getattr(ner_seed, "measurements", None) or []),
                "temporal_refs": list(getattr(ner_seed, "temporal_refs", None) or []),
                "temporal_relations": list(getattr(ner_seed, "temporal_relations", None) or []),
                "temporal_qualifiers": list(getattr(ner_seed, "temporal_qualifiers", None) or []),
                "locations": list(getattr(ner_seed, "locations", None) or []),
                "conjectures": list(getattr(ner_seed, "conjectures", None) or []),
            },

            "semantic_signals": {
                "systems": metadata.get("system_names", []),
                "equipment_ids": metadata.get("equipment_ids", []),
                "components": metadata.get("component_names", []),
                "mechanisms": metadata.get("mechanisms", []),
                "failure_outcomes": metadata.get("failure_outcomes", []),
                "maintenance_actions": metadata.get("maintenance_actions", []),
                "surveillance_actions": metadata.get("surveillance_actions", []),
                "tools_methods": metadata.get("tools_methods", []),
                "has_causal_language": metadata.get("has_causal_language", False),
                "has_diagnostics": metadata.get("has_diagnostics", False),
                "has_failure_signal": metadata.get("has_failure_signal", False),
            },

            "stage5_causal_condition": stage5_payload,
            "retrieval_summary_json": retrieval_summary,
            "rca_frame_json": rca_frame,
            "raw_text": chunk.get("raw_text", chunk.get("text", "")),
        },
    }

def validate_processed_text_record(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    required_keys = [
        "record_id",
        "doc_id",
        "doc_type",
        "chunk_index",
        "embedding_text",
        "metadata",
        "provenance",
    ]
    for key in required_keys:
        if key not in record:
            errors.append(f"missing_{key}")

    if not isinstance(record.get("chunk_index"), int):
        errors.append("chunk_index_not_int")

    embedding_text = record.get("embedding_text")
    if not isinstance(embedding_text, str) or not embedding_text.strip():
        errors.append("embedding_text_empty")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata_not_object")

    enrichment = record.get("enrichment")
    if enrichment is not None and not isinstance(enrichment, dict):
        errors.append("enrichment_not_object")

    if isinstance(metadata, dict):
        allowed_doc_types = {"CR", "WO", "SOP", "ECA", "OTHER"}
        for key in [
            "doc_type",
            "authority_level",
            "section_role",
            "page_start",
            "page_end",
            "equipment_ids",
            "system_names",
            "component_names",
            "mechanisms",
            "failure_outcomes",
        ]:
            if key not in metadata:
                errors.append(f"metadata_missing_{key}")

        for key in [
            "equipment_ids",
            "system_names",
            "component_names",
            "mechanisms",
            "failure_outcomes",
            "maintenance_actions",
            "surveillance_actions",
            "tools_methods",
            "properties_or_limits",
        ]:
            if key in metadata and not isinstance(metadata[key], list):
                errors.append(f"metadata_{key}_not_list")

        if metadata.get("doc_type") != record.get("doc_type"):
            errors.append("metadata_doc_type_mismatch")
        if metadata.get("page_start") is not None and metadata.get("page_end") is not None:
            if int(metadata["page_start"]) > int(metadata["page_end"]):
                errors.append("metadata_page_range_invalid")
        if record.get("doc_type") not in allowed_doc_types:
            errors.append("doc_type_invalid")

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance_not_object")

    if isinstance(provenance, dict):
        for key in [
            "chunk_id",
            "section_path",
            "page_start",
            "page_end",
            "authority_level",
            "section_role",
            "processed_at",
            "nlp_pipeline_version",
            "embedding_model",
            "entity_resolution_version",
        ]:
            if key not in provenance:
                errors.append(f"provenance_missing_{key}")
        if provenance.get("page_start") is not None and provenance.get("page_end") is not None:
            if int(provenance["page_start"]) > int(provenance["page_end"]):
                errors.append("provenance_page_range_invalid")

    if isinstance(enrichment, dict):
        extracted_entities = enrichment.get("extracted_entities")
        if extracted_entities is not None and not isinstance(extracted_entities, dict):
            errors.append("enrichment_extracted_entities_not_object")

        semantic_signals = enrichment.get("semantic_signals")
        if semantic_signals is not None and not isinstance(semantic_signals, dict):
            errors.append("enrichment_semantic_signals_not_object")

        retrieval_summary_json = enrichment.get("retrieval_summary_json")
        if retrieval_summary_json is not None and not isinstance(retrieval_summary_json, dict):
            errors.append("enrichment_retrieval_summary_json_not_object")

        rca_frame_json = enrichment.get("rca_frame_json")
        if rca_frame_json is not None and not isinstance(rca_frame_json, dict):
            errors.append("enrichment_rca_frame_json_not_object")

        stage5_causal_condition = enrichment.get("stage5_causal_condition")
        if stage5_causal_condition is not None and not isinstance(stage5_causal_condition, dict):
            errors.append("enrichment_stage5_causal_condition_not_object")

    errors.extend(_validate_stage5_alias_consistency(record))

    if not isinstance(record.get("record_id"), str) or not record["record_id"].strip():
        errors.append("record_id_empty")
    elif record["record_id"] != f"{record.get('doc_id')}::{record.get('chunk_index')}":
        errors.append("record_id_format_invalid")
    if not isinstance(record.get("doc_id"), str) or not record["doc_id"].strip():
        errors.append("doc_id_empty")

    if not isinstance(record.get("doc_type"), str) or not record["doc_type"].strip():
        errors.append("doc_type_empty")

    return errors

def to_chroma_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(record.get("metadata") or {})
    meta.setdefault("record_id", record.get("record_id"))
    meta.setdefault("doc_id", record.get("doc_id"))
    meta.setdefault("doc_type", record.get("doc_type"))
    meta.setdefault("chunk_index", record.get("chunk_index"))
    return {
        "id": record["record_id"],
        "document": record["embedding_text"],
        "metadata": meta,
    }