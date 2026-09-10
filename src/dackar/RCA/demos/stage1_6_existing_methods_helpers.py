from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import pdfplumber
import spacy
from functools import lru_cache
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

PROJECT_ROOT = Path(project_root)

from ner.ner_adapter import build_ner_pipeline, ner_seed_provider_from_pipeline
from ner.spacy_annotator import build_spacy_annotator

from doc_parsers.pdfParser import pdfParser
from doc_parsers.mdParser import md_parser
from ner.equipment_ID_extractor import extract_equipment_ids 
from storage.chroma_store import ChromaRecordStore
from ner.causal_condition_adapter import extract_stage5_causal_condition
from summarizers.reliability_summarizer import NERSeed, ChunkContext, empty_retrieval_summary, empty_rca_frame, summarize_with_retry, flatten_retrieval_summary_for_embedding, flatten_rca_frame_for_embedding, section_role_from_title

# ---------- generic helpers ----------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _first_text_chunk(chunks: List[Dict[str, Any]]) -> str:
    for c in chunks:
        if c.get('type') == 'TextChunk' and c.get('text'):
            return str(c.get('text'))[:5000]
    return ''


def _uniq(xs: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def _safe_optional_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None

def _summary_safe_ctx(ctx: ChunkContext) -> ChunkContext:
    """
    reliability_summarizer.empty_retrieval_summary() / empty_rca_frame()
    still assume page_start/page_end are int-castable.
    Keep the pipeline ctx nullable, but provide a summary-safe clone here.
    """
    return ChunkContext(
        doc_id=ctx.doc_id,
        doc_type=ctx.doc_type,
        chunk_id=ctx.chunk_id,
        section_path=ctx.section_path,
        page_start=-1 if ctx.page_start is None else int(ctx.page_start),
        page_end=-1 if ctx.page_end is None else int(ctx.page_end),
        authority_level=ctx.authority_level,
        section_role=ctx.section_role,
    )

def _ollama_reachable(base_url: Optional[str] = None) -> bool:
    base_url = base_url or os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    try:
        host_port = base_url.split('//', 1)[-1]
        host_port = host_port.split('/', 1)[0]
        host, port = host_port.split(':', 1)
        with socket.create_connection((host, int(port)), timeout=1.5):
            return True
    except Exception:
        return False


# ---------- Stage 1 ----------

def _fallback_pdf_parser(pdf_path: str, destination_folder: str) -> Dict[str, Any]:
    pdf_path = str(pdf_path)
    dest = Path(destination_folder)
    _ensure_dir(dest)
    text_dir = dest / 'text'
    tables_dir = dest / 'tables'
    index_dir = dest / 'index'
    _ensure_dir(text_dir)
    _ensure_dir(tables_dir)
    _ensure_dir(index_dir)

    md_lines: List[str] = []
    tables: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ''
            md_lines.append(f'## Page {page_num}')
            md_lines.append(page_text)
            for tbl in page.extract_tables() or []:
                if not tbl or len(tbl) < 1:
                    continue
                columns = [str(x).strip() if x is not None else '' for x in tbl[0]]
                rows = [[str(x).strip() if x is not None else '' for x in row] for row in tbl[1:]]
                tables.append({'columns': columns, 'rows': rows, 'caption': None, 'page': page_num})

    text_md_path = text_dir / 'text.md'
    tables_path = tables_dir / 'tables.json'
    with text_md_path.open('w', encoding='utf-8') as f:
        f.write('\n\n'.join(md_lines))
    with tables_path.open('w', encoding='utf-8') as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)

    doc_name = Path(pdf_path).name
    doc_id = Path(pdf_path).stem
    index = {
        'doc_id': doc_id,
        'doc_name': doc_name,
        'source_path': pdf_path,
        'classification': 'internal',
        'text_md_path': str(text_md_path),
        'tables_paths': str(tables_path),
        'figures': [],
        'tables': [{'path': str(tables_path)}],
        'ingest_id': None,
        'content_hash': None,
        'doc_version': None,
    }
    with (index_dir / 'document_index.json').open('w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return index


def stage1_ingest(pdf_path: str, output_root: str) -> Dict[str, Any]:
    pdf_path = str(pdf_path)
    doc_key = Path(pdf_path).stem
    dest = Path(output_root) / doc_key / 'stage1'
    _ensure_dir(dest)
    try:
        return pdfParser(
            home_folder=None,
            red_filepath=pdf_path,
            destination_folder=str(dest),
            text2markdown='marker',
            tableParser='pdfplumber',
            classification='internal',
            source_path=pdf_path,
        )
    except Exception:
        return _fallback_pdf_parser(pdf_path, str(dest))


# ---------- Stage 2 ----------

def stage2_structure_parse(document_index: Dict[str, Any], output_root: str) -> Dict[str, Any]:
    doc_key = Path(document_index.get('doc_name') or document_index.get('doc_id') or 'doc').stem
    dest = Path(output_root) / doc_key / 'stage2'
    _ensure_dir(dest)
    return md_parser(document_index=document_index, destination_folder=str(dest), mbse_entities=None)


# ---------- Stage 3-5 helpers ----------

def build_ner_seed_fallback(chunk: Dict[str, Any]):
    txt = str(chunk.get('text') or '')
    mentions = chunk.get('mentions_component_ids') or []
    if not isinstance(mentions, list):
        mentions = []
    return NERSeed(
        systems=[],
        equipment_ids=extract_equipment_ids(txt),
        components=[str(x) for x in mentions if x],
        mechanisms=[],
        outcomes=[],
        surveillance_actions=[],
        maintenance_actions=[],
        properties=[],
        tools=[],
    )

def build_ner_seed_from_chunk(chunk: Dict[str, Any], syntactic_nlp: Optional[Dict[str, Any]] = None):
    provider = get_hybrid_ner_seed_provider()
    if provider is not None:
        try:
            return provider(chunk)
        except Exception:
            pass
    return build_ner_seed_fallback(chunk)

def _fallback_retrieval_summary(ctx, ner_seed, chunk_text, chunk=None):
    obj = empty_retrieval_summary(_summary_safe_ctx(ctx))
    low = (chunk_text or "").lower()

    obj["scope"] = (chunk_text or "")[:400].strip()
    obj["entities"]["systems"] = list(ner_seed.systems or [])
    obj["entities"]["equipment_ids"] = list(ner_seed.equipment_ids or [])
    obj["entities"]["components"] = list(ner_seed.components or [])
    obj["mechanisms"] = list(ner_seed.mechanisms or [])
    obj["symptoms_outcomes"] = list(ner_seed.outcomes or [])
    obj["corrective_actions"] = list(ner_seed.maintenance_actions or [])
    obj["diagnostics"] = [t for t in ["verify", "check", "measure", "monitor", "test"] if t in low]
    obj["numbers_limits"] = re.findall(
        r"\b\d+(?:\.\d+)?\s?(?:gpm|psi|psig|degc|°c|rpm|mils|in|mm|amps|a|volts|v|%)\b",
        low,
        flags=re.I,
    )

    keywords = []
    if isinstance(chunk, dict):
        raw_keywords = chunk.get("keywords") or []
        if isinstance(raw_keywords, list):
            keywords = [str(x).strip() for x in raw_keywords if str(x).strip()]

    obj["keywords_synonyms"] = _uniq(
        keywords + obj["mechanisms"] + obj["symptoms_outcomes"]
    )

    return obj

def _fallback_rca_frame(ctx, ner_seed, chunk_text):
    obj = empty_rca_frame(_summary_safe_ctx(ctx))
    low = (chunk_text or "").lower()

    obj["observed"] = _coerce_list_of_str(list(ner_seed.outcomes or []))
    obj["hypotheses"] = _coerce_list_of_str(list(ner_seed.mechanisms or []))
    obj["tests_to_confirm"] = [t for t in ["verify", "check", "measure", "monitor", "test"] if t in low]
    obj["candidate_actions"] = _coerce_list_of_str(list(ner_seed.maintenance_actions or []))
    obj["constraints"] = [t for t in ["warning", "caution", "hold", "limitation"] if t in low]

    return obj


def summarize_chunk(doc_type: str, ctx, ner_seed, chunk_text: str, view_type: str, chunk: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    sctx = _summary_safe_ctx(ctx)
    if _ollama_reachable():
        try:
            return summarize_with_retry(
                doc_type=doc_type,
                view_type=view_type,
                ctx=sctx,
                ner_seed=ner_seed,
                chunk_text=chunk_text,
                model=model,
                timeout=90,
                max_tries=2,
            )
        except Exception:
            pass

    if view_type == 'retrieval_summary':
        return _fallback_retrieval_summary(sctx, ner_seed, chunk_text, chunk=chunk)
    return _fallback_rca_frame(sctx, ner_seed, chunk_text)


def _normalize_stage5_empty() -> Dict[str, Any]:
    return {
        'stage': 'stage5_causal_condition_extraction',
        'status': 'empty',
        'extractor': {
            'primary': 'CausalSentence',
            'fallback': 'CausalSimple',
            'used': '',
            'version': 'v1',
        },
        'summary_flags': {
            'has_explicit_causal_statement': False,
            'has_condition_state': False,
            'has_as_found': False,
            'has_as_left': False,
            'has_procedural_deviation': False,
            'has_negation': False,
            'has_conjecture': False,
        },
        'extracted_causal_statements': [],
        'condition_state': {
            'as_found': None,
            'as_left': None,
            'status_mentions': [],
            'evidence': [],
        },
        'procedural_deviation': {
            'detected': False,
            'evidence': [],
            'confidence': 0.0,
        },
        'errors': [],
    }


def _stage5_has_signal(payload: Dict[str, Any]) -> bool:
    flags = payload.get('summary_flags', {})
    return any([
        flags.get('has_explicit_causal_statement', False),
        flags.get('has_condition_state', False),
        flags.get('has_procedural_deviation', False),
    ])


def _normalize_health_state(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    if any(x in s for x in ['fail', 'inoperable', 'unavailable', 'trip']):
        return 'failed'
    if any(x in s for x in ['degrad', 'wear', 'leak', 'high', 'low', 'out of spec', 'abnormal', 'misalignment']):
        return 'degraded'
    if any(x in s for x in ['acceptable', 'normal', 'satisfactory', 'within spec', 'within tolerance', 'operable']):
        return 'acceptable'
    return 'unknown'


def _infer_condition_from_text(text: str) -> Optional[str]:
    low = (text or '').lower()
    if any(t in low for t in ['failed', 'failure', 'inoperable', 'unavailable', 'tripped']):
        return 'failed'
    if any(t in low for t in ['degraded', 'worn', 'damaged', 'leaking', 'misalignment', 'high vibration', 'high temperature', 'out of spec']):
        return 'degraded'
    if any(t in low for t in ['acceptable', 'normal', 'satisfactory', 'within spec', 'within tolerance', 'operable']):
        return 'acceptable'
    return None


def _detect_procedural_deviation(chunk_text: str, doc_type: str, section_role: str) -> Dict[str, Any]:
    low = (chunk_text or '').lower()
    evidence = []
    score = 0.0
    patterns = [
        'deviation',
        'did not follow',
        'not performed per procedure',
        'contrary to procedure',
        'step skipped',
        'omitted',
        'not completed as written',
        'bypassed',
        'deferred',
    ]
    for p in patterns:
        if p in low:
            evidence.append(p)
            score += 0.25
    if doc_type == 'SOP' or section_role in {'steps', 'constraints'}:
        if evidence:
            score += 0.05
    score = max(0.0, min(1.0, round(score, 3)))
    return {
        'detected': score >= 0.25,
        'evidence': evidence,
        'confidence': score,
    }

def _stage5_pick_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ''):
            return d[k]
    return default


def _stage5_extract_rows(obj: Any, attr_names: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for attr in attr_names:
        value = getattr(obj, attr, None)
        if value is None:
            continue

        if hasattr(value, 'to_dict'):
            try:
                recs = value.to_dict(orient='records')
                for r in recs:
                    if isinstance(r, dict):
                        rows.append(r)
                continue
            except Exception:
                pass

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.append(item)
                elif isinstance(item, (list, tuple)):
                    # Legacy tuple layouts are inconsistent; do not force-map
                    # simplistic cause/effect/connector positions here.
                    continue
    return rows

def _normalize_stage5_from_extractor(
    extractor_name: str,
    extractor_obj: Any,
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    doc_type: str,
    section_role: str,
) -> Dict[str, Any]:
    out = _normalize_stage5_empty()
    out['extractor']['used'] = extractor_name

    causal_rows = _stage5_extract_rows(
        extractor_obj,
        ['_causalRelation', '_rawCausalList', '_extractedCausals']
    )
    status_rows = _stage5_extract_rows(
        extractor_obj,
        ['_entHS', '_entStatus']
    )

    causals = []
    for i, row in enumerate(causal_rows):
        cause_text = str(_stage5_pick_first(row, ['cause_text', 'cause', 'subj', 'subject'], '')).strip()
        effect_text = str(_stage5_pick_first(row, ['effect_text', 'effect', 'obj', 'object'], '')).strip()
        connector = str(_stage5_pick_first(row, ['connector', 'keyword', 'causal_keyword'], '')).strip()
        sentence_text = str(_stage5_pick_first(row, ['sentence', 'sent', 'sentence_text', 'text'], '')).strip()
        negated = bool(_stage5_pick_first(row, ['negated', 'negation'], False))
        conjectural = bool(_stage5_pick_first(row, ['conjectural', 'conjecture'], False))

        confidence = 0.0
        if connector:
            confidence += 0.35
        if cause_text:
            confidence += 0.25
        if effect_text:
            confidence += 0.25
        if cause_text and effect_text:
            confidence += 0.10
        if negated:
            confidence -= 0.10
        if conjectural:
            confidence -= 0.05
        confidence = max(0.0, min(1.0, round(confidence, 3)))

        causals.append({
            'statement_id': f'{doc_id}::{chunk_index}::cause::{i}',
            'sentence_text': sentence_text,
            'connector': connector,
            'cause_text': cause_text,
            'effect_text': effect_text,
            'cause_entity': None,
            'effect_entity': None,
            'negated': negated,
            'conjectural': conjectural,
            'confidence': confidence,
            'source': extractor_name,
        })

    status_mentions = []
    for row in status_rows:
        entity = str(_stage5_pick_first(row, ['entity', 'ent', 'subject', 'obj', 'node'], '')).strip()
        status = str(_stage5_pick_first(row, ['status', 'health_status', 'hs', 'condition'], '')).strip()
        sentence_text = str(_stage5_pick_first(row, ['sentence', 'sent', 'text'], '')).strip()
        negated = bool(_stage5_pick_first(row, ['negated', 'negation'], False))
        conjectural = bool(_stage5_pick_first(row, ['conjectural', 'conjecture'], False))

        status_mentions.append({
            'entity': entity,
            'status': status,
            'health_state': _normalize_health_state(status),
            'negated': negated,
            'conjectural': conjectural,
            'sentence_text': sentence_text,
            'source': extractor_name,
        })

    low = (chunk_text or '').lower()
    as_found = None
    as_left = None
    if 'as found' in low:
        as_found = _infer_condition_from_text(low) or 'unknown'
    if 'as left' in low:
        as_left = _infer_condition_from_text(low) or 'unknown'

    if as_found is None and section_role == 'as_found':
        hs = [m.get('health_state') for m in status_mentions if m.get('health_state')]
        as_found = hs[0] if hs else None

    if as_left is None and section_role == 'as_left':
        hs = [m.get('health_state') for m in status_mentions if m.get('health_state')]
        as_left = hs[0] if hs else None

    proc_dev = _detect_procedural_deviation(chunk_text, doc_type, section_role)

    out['status'] = 'ok'
    out['extracted_causal_statements'] = causals
    out['condition_state'] = {
        'as_found': as_found,
        'as_left': as_left,
        'status_mentions': status_mentions,
        'evidence': [],
    }
    out['procedural_deviation'] = proc_dev
    out['summary_flags'] = {
        'has_explicit_causal_statement': bool(causals),
        'has_condition_state': bool(status_mentions or as_found or as_left),
        'has_as_found': as_found is not None,
        'has_as_left': as_left is not None,
        'has_procedural_deviation': bool(proc_dev.get('detected')),
        'has_negation': any(c.get('negated') for c in causals) or any(m.get('negated') for m in status_mentions),
        'has_conjecture': any(c.get('conjectural') for c in causals) or any(m.get('conjectural') for m in status_mentions),
    }
    return out

def extract_stage5_payload(doc_id, chunk_index, chunk_text, doc_type, section_role, ner_seed=None):
    return extract_stage5_causal_condition(
        doc_id=doc_id,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        doc_type=doc_type,
        section_role=section_role,
        nlp=_get_stage5_nlp(),
    )

# ---------- Stage 6 ----------

def build_embedding_text(
    chunk_text: str,
    retrieval_summary: Dict[str, Any],
    rca_frame: Dict[str, Any],
    ner_seed,
    stage5_payload: Optional[Dict[str, Any]] = None,
) -> str:
    retrieval_summary = _normalize_retrieval_summary(retrieval_summary)
    rca_frame = _normalize_rca_frame(rca_frame)

    parts = []
    stage5_payload = stage5_payload or {}
    causals = stage5_payload.get("extracted_causal_statements") or []
    if causals:
        causal_lines = []
        for row in causals[:10]:
            cause = str(row.get("cause_text") or "").strip()
            conn = str(row.get("connector") or "").strip()
            effect = str(row.get("effect_text") or "").strip()
            text = " ".join(x for x in [cause, conn, effect] if x).strip()
            if text:
                causal_lines.append(text)
        if causal_lines:
            parts.append("CAUSAL_STATEMENTS: " + " | ".join(causal_lines))

    parts.extend([
        flatten_retrieval_summary_for_embedding(retrieval_summary),
        flatten_rca_frame_for_embedding(rca_frame),
    ])

    if ner_seed.equipment_ids:
        parts.append('EQUIPMENT_IDS_SEED: ' + ', '.join(ner_seed.equipment_ids))
    if ner_seed.components:
        parts.append('COMPONENTS_SEED: ' + ', '.join(ner_seed.components))
    txt = ' '.join((chunk_text or '').split())[:700]
    if txt:
        parts.append('TEXT_SNIPPET: ' + txt)
    return '\n'.join(p for p in parts if p and p.strip())


def build_metadata(doc_type: str, authority_level: str, section_role: str, retrieval_summary: Dict[str, Any], rca_frame: Dict[str, Any], ner_seed, stage5_payload: Dict[str, Any], ctx) -> Dict[str, Any]:
    ent = retrieval_summary.get('entities') or {}
    return {
        'doc_type': doc_type,
        'authority_level': authority_level,
        'section_role': section_role,
        'page_start': ctx.page_start,
        'page_end': ctx.page_end,
        'equipment_ids': _uniq(ner_seed.equipment_ids),
        'system_names': _uniq((ent.get('systems') or []) + list(ner_seed.systems or [])),
        'component_names': _uniq((ent.get('components') or []) + list(ner_seed.components or [])),
        'mechanisms': _uniq((retrieval_summary.get('mechanisms') or []) + list(ner_seed.mechanisms or [])),
        'failure_outcomes': _uniq((retrieval_summary.get('symptoms_outcomes') or []) + list(ner_seed.outcomes or [])),
        'maintenance_actions': _uniq((retrieval_summary.get('corrective_actions') or []) + list(ner_seed.maintenance_actions or [])),
        'surveillance_actions': _uniq((retrieval_summary.get('diagnostics') or []) + list(ner_seed.surveillance_actions or [])),
        'tools_methods': _uniq(list(ner_seed.tools or [])),
        'properties_or_limits': _uniq(list(ner_seed.properties or []) + list(retrieval_summary.get('numbers_limits') or []))[:25],
        'has_causal_language': bool(
            rca_frame.get('hypotheses')
            or rca_frame.get('constraints')
            or stage5_payload.get('summary_flags', {}).get('has_explicit_causal_statement', False)
        ),
        'has_diagnostics': bool(retrieval_summary.get('diagnostics') or rca_frame.get('tests_to_confirm')),
        'has_maintenance_action': bool(retrieval_summary.get('corrective_actions') or ner_seed.maintenance_actions),
        'has_surveillance_action': bool(retrieval_summary.get('diagnostics') or ner_seed.surveillance_actions),
        'has_failure_signal': bool(retrieval_summary.get('symptoms_outcomes') or retrieval_summary.get('mechanisms')),
        'has_explicit_causal_statement': bool(stage5_payload.get('summary_flags', {}).get('has_explicit_causal_statement', False)),
        'has_condition_state': bool(stage5_payload.get('summary_flags', {}).get('has_condition_state', False)),
        'has_procedural_deviation': bool(stage5_payload.get('summary_flags', {}).get('has_procedural_deviation', False)),
    }


def build_processed_text_record(
    doc_id: str,
    doc_type: str,
    chunk_index: int,
    chunk: Dict[str, Any],
    ctx,
    embedding_text: str,
    metadata: Dict[str, Any],
    retrieval_summary: Dict[str, Any],
    rca_frame: Dict[str, Any],
    stage5_payload: Dict[str, Any],
    ner_seed,
    syntactic_nlp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    
    chunk_id = str(chunk.get('chunk_id') or f'{doc_id}::{chunk_index}')
    return {
        'record_id': f'{doc_id}::{chunk_index}',
        'doc_id': doc_id,
        'doc_type': doc_type,
        'chunk_index': int(chunk_index),
        'embedding_text': embedding_text,
        'metadata': metadata,
        'provenance': {
            'chunk_id': chunk_id,
            'section_path': ctx.section_path,
            'page_start': ctx.page_start,
            'page_end': ctx.page_end,
            'authority_level': ctx.authority_level,
            'section_role': ctx.section_role,
            'source_chunk_type': chunk.get('type'),
            'source_granularity': chunk.get('granularity'),
            'index_in_vector_store': bool(chunk.get('index_in_vector_store', False)),
        },
        'enrichment': {
            'stage3_syntactic_nlp': syntactic_nlp or {},
            'extracted_entities': {
                'systems': _uniq(list(ner_seed.systems or [])),
                'equipment_ids': _uniq(list(ner_seed.equipment_ids or [])),
                'components': _uniq(list(ner_seed.components or [])),
                'mechanisms': _uniq(list(ner_seed.mechanisms or [])),
                'outcomes': _uniq(list(ner_seed.outcomes or [])),
                'surveillance_actions': _uniq(list(ner_seed.surveillance_actions or [])),
                'maintenance_actions': _uniq(list(ner_seed.maintenance_actions or [])),
                'properties': _uniq(list(ner_seed.properties or [])),
                'tools': _uniq(list(ner_seed.tools or [])),
            },
            'retrieval_summary_json': retrieval_summary,
            'rca_frame_json': rca_frame,
            'stage5_causal_condition': stage5_payload,
            'raw_text': chunk.get('raw_text', chunk.get('text', '')),
        },
    }


def validate_processed_text_record(record: Dict[str, Any]) -> List[str]:
    req = ['record_id', 'doc_id', 'doc_type', 'chunk_index', 'embedding_text', 'metadata', 'provenance']
    errs = [f'missing_{k}' for k in req if k not in record]
    if not isinstance(record.get('chunk_index'), int):
        errs.append('chunk_index_not_int')
    if not str(record.get('embedding_text') or '').strip():
        errs.append('embedding_text_empty')

    enrichment = record.get('enrichment')
    if enrichment is not None and not isinstance(enrichment, dict):
        errs.append('enrichment_not_object')

    if isinstance(enrichment, dict):
        if 'stage5_causal_condition' in enrichment and not isinstance(enrichment['stage5_causal_condition'], dict):
            errs.append('enrichment_stage5_causal_condition_not_object')
        if 'stage3_syntactic_nlp' in enrichment and not isinstance(enrichment['stage3_syntactic_nlp'], dict):
            errs.append('enrichment_stage3_syntactic_nlp_not_object')

    return errs


@dataclass
class PipelineRunResult:
    pdf_path: str
    doc_type: str
    stage1_index: Dict[str, Any]
    stage2_structured: Dict[str, Any]
    enriched_jsonl_path: str
    processed_records: List[Dict[str, Any]]


def stage3_to_6_process(structured_output: Dict[str, Any], output_root: str, model: Optional[str] = None) -> PipelineRunResult:
    chunks_jsonl_path = (
        structured_output.get("chunks_jsonl_path")
        or structured_output.get("chunks_path")
        or structured_output.get("chunks_jsonl")
    )

    if chunks_jsonl_path:
        chunks_path = Path(chunks_jsonl_path)
    else:
        # mdParser writes:
        #   <base_doc_folder>/index/<doc_id>_chunks.jsonl
        doc_id = structured_output.get("doc_id")
        source_path = structured_output.get("source_path")
        if not doc_id or not source_path:
            raise ValueError(
                "structured_output is missing both chunks path keys and the doc_id/source_path "
                "needed to reconstruct the mdParser chunks file path."
            )

        # Reconstruct base_doc_folder from source_path convention used in the demo
        # Example:
        #   <output_root>/<pdf_stem>/stage1/<...>
        # and mdParser writes under:
        #   <base_doc_folder>/index/<doc_id>_chunks.jsonl
        source_path = Path(source_path)
        base_doc_folder = source_path.parent.parent
        chunks_path = base_doc_folder / "index" / f"{doc_id}_chunks.jsonl"

    if not chunks_path.exists():
        candidate = Path(structured_output.get("output_dir", ".")) / "chunks.jsonl"
        if candidate.exists():
            chunks_path = candidate
        else:
            raise FileNotFoundError(f"Could not locate chunks JSONL: {chunks_path}")

    chunks = list(_iter_jsonl(chunks_path))
    doc_type = (
    structured_output.get("doc_type")
    or next((c.get("doc_type") for c in chunks if c.get("doc_type")), "OTHER"))
    authority = 'mandatory' if doc_type == 'SOP' else 'informational'

    enriched_rows: List[Dict[str, Any]] = []
    processed_records: List[Dict[str, Any]] = []

    for chunk_index, c in enumerate(chunks):
        chunk_type = c.get("type")
        if chunk_type == "Table" and c.get("index_in_vector_store", False):
            table_record = enrich_table_chunk(c, doc_id=str(c.get("doc_id") or "unknown_doc"),
                                            doc_type=doc_type, chunk_index=chunk_index)
            c["processed_text_record"] = table_record
            c["metadata"] = table_record["metadata"]
            enriched_rows.append(c)
            processed_records.append(table_record)
            continue

        if chunk_type != "TextChunk" or not c.get("index_in_vector_store", False):
            enriched_rows.append(c)
            continue
        chunk_text = str(c.get('text') or '').strip()
        if not chunk_text:
            enriched_rows.append(c)
            continue
        section_title = c.get('section_title') or c.get('title') or ''
        role = c.get('section_role') or section_role_from_title(doc_type, str(section_title))
        ctx = ChunkContext(
            doc_id=str(c.get('doc_id') or 'unknown_doc'),
            doc_type=doc_type,
            chunk_id=str(c.get('chunk_id') or 'unknown_chunk'),
            section_path=str(c.get('section_path') or section_title or c.get('belongs_to_section') or 'unknown_section'),
            page_start=_safe_optional_int(c.get('page_start')),
            page_end=_safe_optional_int(c.get('page_end')),
            authority_level=authority,
            section_role=role,
        )
        syntactic_nlp = run_stage3_syntactic_nlp(chunk_text)
        seed = build_ner_seed_from_chunk(c, syntactic_nlp=syntactic_nlp)
        
        rs = _normalize_retrieval_summary(summarize_chunk(doc_type, ctx, seed, chunk_text, 'retrieval_summary', c, model=model))
        rf = _normalize_rca_frame(summarize_chunk(doc_type, ctx, seed, chunk_text, 'rca_frame', c, model=model))
        
        stage5 = extract_stage5_payload(str(c.get('doc_id') or 'unknown_doc'), chunk_index, chunk_text, doc_type, role, ner_seed=seed)
        embedding_text = build_embedding_text(chunk_text, rs, rf, seed, stage5)
        metadata = build_metadata(doc_type, authority, role, rs, rf, seed, stage5, ctx)
        record = build_processed_text_record(
            str(c.get('doc_id') or 'unknown_doc'),
            doc_type,
            chunk_index,
            c,
            ctx,
            embedding_text,
            metadata,
            rs,
            rf,
            stage5,
            seed,
            syntactic_nlp=syntactic_nlp,
        )
        errs = validate_processed_text_record(record)
        if errs:
            raise ValueError(f'Invalid processed_text_record for chunk {chunk_index}: {errs}')
        c['processed_text_record'] = record
        c['retrieval_summary_json'] = rs
        c['rca_frame_json'] = rf
        c['stage5_causal_condition'] = stage5
        c['metadata'] = record['metadata']
        c["stage3_syntactic_nlp"] = syntactic_nlp
        c['stage4_ner_seed'] = {
            'systems': list(seed.systems or []),
            'equipment_ids': list(seed.equipment_ids or []),
            'components': list(seed.components or []),
            'mechanisms': list(seed.mechanisms or []),
            'outcomes': list(seed.outcomes or []),
            'surveillance_actions': list(seed.surveillance_actions or []),
            'maintenance_actions': list(seed.maintenance_actions or []),
            'properties': list(seed.properties or []),
            'tools': list(seed.tools or []),
        }
        enriched_rows.append(c)
        processed_records.append(record)

    out_dir = Path(output_root)
    _ensure_dir(out_dir)
    enriched_jsonl_path = out_dir / f'{chunks_path.stem}_enriched.jsonl'
    _write_jsonl(enriched_jsonl_path, enriched_rows)

    return PipelineRunResult(
        pdf_path='',
        doc_type=doc_type,
        stage1_index={},
        stage2_structured=structured_output,
        enriched_jsonl_path=str(enriched_jsonl_path),
        processed_records=processed_records,
    )


def run_stage1_to_6(
    pdf_path: str,
    output_root: str,
    model: Optional[str] = None,
    upsert_chroma: bool = False,
    chroma_persist_directory: Optional[str] = None,
) -> PipelineRunResult:
    
    stage1 = stage1_ingest(pdf_path, output_root)
    stage2 = stage2_structure_parse(stage1, output_root)
    stage34 = stage3_to_6_process(stage2, str(Path(output_root) / Path(pdf_path).stem / 'stage3_6'), model=model)

    if upsert_chroma:
        persist_dir = chroma_persist_directory or str(Path(output_root) / 'chroma_store')
        try:
            store = ChromaRecordStore(persist_directory=persist_dir)
            store.upsert_jsonl(stage34.enriched_jsonl_path)
        except Exception as e:
            print(f"[WARN] Chroma upsert failed for {pdf_path}: {e}")

    stage34.pdf_path = pdf_path
    stage34.stage1_index = stage1
    return stage34


def run_many(
    pdf_paths: List[str],
    output_root: str,
    model: Optional[str] = None,
    upsert_chroma: bool = False,
    chroma_persist_directory: Optional[str] = None) -> List[PipelineRunResult]:
    
    results = []
    for p in pdf_paths:
        results.append(
            run_stage1_to_6(
                p,
                output_root=output_root,
                model=model,
                upsert_chroma=upsert_chroma,
                chroma_persist_directory=chroma_persist_directory,
            )
        )
    return results


def enrich_table_chunk(
    chunk: dict,
    doc_id: str,
    doc_type: str,
    chunk_index: int,
) -> dict:
    """
    Lightweight enrichment for Table chunks.
    Extracts measurements, equipment IDs, and condition signals from
    structured column/row data — no LLM or spaCy required.
    """
    headers = chunk.get("columns") or []
    rows = chunk.get("rows") or []
    chunk_text = chunk.get("text") or ""   # textified representation from mdParser

    # ── Measurement extraction from table cells ──────────────────────────────
    MEAS_RE = re.compile(
        r"(?P<value>-?\d+(?:\.\d+)?)\s*"
        r"(?P<unit>degC|°C|°F|psi|psig|gpm|rpm|mils|in|mm|amps?|A|volts?|V|%|kPa|MPa|kW|MW)",
        re.I,
    )
    measurements = []
    for row in rows:
        for cell in row:
            for m in MEAS_RE.finditer(str(cell)):
                measurements.append({
                    "value": float(m.group("value")),
                    "unit": m.group("unit"),
                    "parameter": None,   # resolved below if header available
                    "context": str(cell)[:120],
                })

    # Attempt to pair measurements with column headers
    if headers and rows:
        for row in rows:
            for col_idx, cell in enumerate(row):
                for m in MEAS_RE.finditer(str(cell)):
                    param = headers[col_idx] if col_idx < len(headers) else None
                    measurements.append({
                        "value": float(m.group("value")),
                        "unit": m.group("unit"),
                        "parameter": param,
                        "context": str(cell)[:120],
                    })

    # Deduplicate by (value, unit, parameter)
    seen, deduped = set(), []
    for meas in measurements:
        key = (meas["value"], meas["unit"], meas["parameter"])
        if key not in seen:
            seen.add(key)
            deduped.append(meas)
    measurements = deduped

    # ── Equipment ID extraction from table text ───────────────────────────────
    equipment_ids = extract_equipment_ids(chunk_text)

    # ── Condition signal detection ────────────────────────────────────────────
    low = chunk_text.lower()
    as_found = _infer_condition_from_text(low) if "as found" in low or "as-found" in low else None
    as_left  = _infer_condition_from_text(low) if "as left"  in low or "as-left"  in low else None

    # ── Assemble enrichment ───────────────────────────────────────────────────
    section_role = (
        chunk.get("section_role")
        or section_role_from_title(doc_type, str(chunk.get("belongs_to_section") or ""))
        or "table"
    )

    metadata = {
        "doc_type": doc_type,
        "authority_level": chunk.get("authority_level", "informational"),
        "section_role": section_role,
        "page_start": _safe_optional_int(chunk.get("page")),
        "page_end": _safe_optional_int(chunk.get("page")),
        "equipment_ids": _uniq(equipment_ids),
        "system_names": [],
        "component_names": [],
        "mechanisms": [],
        "failure_outcomes": [],
        "maintenance_actions": [],
        "surveillance_actions": [],
        "tools_methods": [],
        "properties_or_limits": [
            f"{m['value']} {m['unit']}" for m in measurements
        ][:25],
        "has_causal_language": False,
        "has_diagnostics": False,
        "has_maintenance_action": False,
        "has_surveillance_action": False,
        "has_failure_signal": bool(as_found and as_found != "acceptable"),
        "has_explicit_causal_statement": False,
        "has_condition_state": bool(as_found or as_left),
        "has_procedural_deviation": False,
    }

    # Build a compact embedding text from column headers + measurements
    parts = []
    if headers:
        parts.append("COLUMNS: " + " | ".join(str(h) for h in headers if h))
    if measurements:
        parts.append("MEASUREMENTS: " + "; ".join(
            f"{m['parameter'] or 'value'}: {m['value']} {m['unit']}"
            for m in measurements[:10]
        ))
    if equipment_ids:
        parts.append("EQUIPMENT: " + ", ".join(equipment_ids))
    if as_found:
        parts.append(f"AS_FOUND: {as_found}")
    if as_left:
        parts.append(f"AS_LEFT: {as_left}")
    parts.append(f"DOC_TYPE: {doc_type}")
    embedding_text = "\n".join(parts) or chunk_text[:512]

    record_id = f"{doc_id}::{chunk_index}"
    return {
        "record_id": record_id,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "chunk_index": chunk_index,
        "embedding_text": embedding_text,
        "metadata": metadata,
        "provenance": {
            "chunk_id": chunk.get("chunk_id", record_id),
            "section_path": chunk.get("section_path") or chunk.get("belongs_to_section") or "table",
            "page_start": _safe_optional_int(chunk.get("page")),
            "page_end": _safe_optional_int(chunk.get("page")),
            "authority_level": "informational",
            "section_role": section_role,
            "source_chunk_type": "Table",
            "source_granularity": "table",
            "index_in_vector_store": chunk.get("index_in_vector_store", True),
        },
        "enrichment": {
            "stage3_syntactic_nlp": {},
            "extracted_entities": {
                "systems": [],
                "equipment_ids": _uniq(equipment_ids),
                "components": [],
                "mechanisms": [],
                "outcomes": [],
                "surveillance_actions": [],
                "maintenance_actions": [],
                "measurements": measurements,
            },
            "retrieval_summary_json": {},
            "rca_frame_json": {},
            "stage5_causal_condition": {
                **_normalize_stage5_empty(),
                "status": "table_chunk",
                "condition_state": {
                    "as_found": as_found,
                    "as_left": as_left,
                    "status_mentions": [],
                    "evidence": [],
                },
                "summary_flags": {
                    **_normalize_stage5_empty()["summary_flags"],
                    "has_condition_state": bool(as_found or as_left),
                },
            },
            "raw_text": chunk.get("raw_text", chunk.get("text", "")),
        },
    }


def maybe_upsert_with_chroma(enriched_jsonl_paths: List[str], persist_directory: str):
    try:
        store = ChromaRecordStore(persist_directory=persist_directory)
    except Exception as e:
        return {'status': 'skipped', 'reason': f'Could not initialize ChromaRecordStore: {e}'}
    counts = {}
    for path in enriched_jsonl_paths:
        try:
            counts[path] = store.upsert_jsonl(path)
        except Exception as e:
            counts[path] = {'error': str(e)}
    return {'status': 'ok', 'counts': counts}

def _coerce_list_of_str(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif isinstance(item, dict):
                # prefer a text-like field if present
                for k in ["text", "value", "label", "name", "content", "sentence"]:
                    if k in item and item[k] is not None:
                        s = str(item[k]).strip()
                        if s:
                            out.append(s)
                            break
                else:
                    s = str(item).strip()
                    if s:
                        out.append(s)
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out

    # scalar fallback
    s = str(value).strip()
    return [s] if s else []

def _normalize_rca_frame(rca: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(rca, dict):
        return {
            "observed": [],
            "hypotheses": [],
            "tests_to_confirm": [],
            "candidate_actions": [],
            "constraints": [],
        }

    out = dict(rca)
    for key in ["observed", "hypotheses", "tests_to_confirm", "candidate_actions", "constraints"]:
        out[key] = _coerce_list_of_str(out.get(key))
    return out

def _normalize_retrieval_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {
            "scope": "",
            "entities": {"systems": [], "equipment_ids": [], "components": []},
            "symptoms_outcomes": [],
            "mechanisms": [],
            "diagnostics": [],
            "corrective_actions": [],
            "numbers_limits": [],
            "keywords_synonyms": [],
            "unknowns": [],
        }

    out = dict(summary)
    entities = out.get("entities")
    if not isinstance(entities, dict):
        entities = {}

    out["entities"] = {
        "systems": _coerce_list_of_str(entities.get("systems")),
        "equipment_ids": _coerce_list_of_str(entities.get("equipment_ids")),
        "components": _coerce_list_of_str(entities.get("components")),
    }

    for key in [
        "symptoms_outcomes",
        "mechanisms",
        "diagnostics",
        "corrective_actions",
        "numbers_limits",
        "keywords_synonyms",
        "unknowns",
    ]:
        out[key] = _coerce_list_of_str(out.get(key))

    scope = out.get("scope")
    out["scope"] = "" if scope is None else str(scope).strip()
    return out

@lru_cache(maxsize=1)
def get_spacy_nlp():
    """
    Stage 3 syntactic NLP pipeline.
    Prefer a transformer/sm model if available, fall back gracefully.
    """
    model_candidates = [
        "en_core_web_trf",
        "en_core_web_lg",
        "en_core_web_sm",
    ]
    last_err = None
    for name in model_candidates:
        try:
            return spacy.load(name)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any spaCy English model: {last_err}")

def run_stage3_syntactic_nlp(chunk_text: str) -> Dict[str, Any]:
    nlp = get_spacy_nlp()
    doc = nlp(chunk_text or "")

    return {
        "sentences": [sent.text.strip() for sent in doc.sents if sent.text.strip()],
        "tokens": [
            {
                "text": tok.text,
                "lemma": tok.lemma_,
                "pos": tok.pos_,
                "tag": tok.tag_,
                "dep": tok.dep_,
                "head": tok.head.text,
                "is_stop": tok.is_stop,
            }
            for tok in doc
        ],
        "noun_chunks": [nc.text.strip() for nc in getattr(doc, "noun_chunks", []) if nc.text.strip()],
    }

@lru_cache(maxsize=1)
def get_hybrid_ner_seed_provider():
    """
    Build and cache the hybrid NER-backed seed provider (pipeline + SpacyAnnotator).
    Falls back to the regex-only seed builder if initialization fails.

    Paths resolve to the canonical NER data files under ner/data/.
    PROJECT_ROOT is set to DACKAR/src/dackar/RCA when run from demos/.
    """
    try:
        ner_data = PROJECT_ROOT / "ner" / "data"
        schema_json_path = str(ner_data / "group-schema.json")
        gazetteer_xl_path = str(ner_data / "tag_keywords_lists.xlsx")
        # label_json uses the same group-schema file as the schema
        label_json_path = schema_json_path

        pipeline = build_ner_pipeline(
            schema_json_path=schema_json_path,
            gazetteer_xl=gazetteer_xl_path,
            label_json=label_json_path,
            llm_cfg=None,
            generator_mode="anchored_np",
        )
        annotator = build_spacy_annotator()
        return ner_seed_provider_from_pipeline(pipeline, NERSeed, annotator=annotator)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Hybrid NER provider init failed: %s", e)
        return None
    
def _regex_stage5_fallback(
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    doc_type: str,
    section_role: str,
) -> Dict[str, Any]:
    out = _normalize_stage5_empty()
    low = (chunk_text or "").lower()

    connectors = [
        "due to",
        "caused by",
        "resulted in",
        "led to",
        "because of",
        "following",
        "attributed to",
    ]

    causals = []
    for i, conn in enumerate(connectors):
        if conn in low:
            causals.append({
                "statement_id": f"{doc_id}::{chunk_index}::cause::{i}",
                "sentence_text": chunk_text[:500],
                "connector": conn,
                "cause_text": "",
                "effect_text": "",
                "cause_entity": None,
                "effect_entity": None,
                "negated": False,
                "conjectural": False,
                "confidence": 0.35,
                "source": "regex_fallback",
            })

    as_found = None
    as_left = None
    if "as found" in low:
        as_found = _infer_condition_from_text(low) or "unknown"
    if "as left" in low:
        as_left = _infer_condition_from_text(low) or "unknown"

    proc_dev = _detect_procedural_deviation(chunk_text, doc_type, section_role)

    out["status"] = "fallback"
    out["extractor"]["used"] = "regex_fallback"
    out["extracted_causal_statements"] = causals
    out["condition_state"] = {
        "as_found": as_found,
        "as_left": as_left,
        "status_mentions": [],
        "evidence": [],
    }
    out["procedural_deviation"] = proc_dev
    out["summary_flags"] = {
        "has_explicit_causal_statement": bool(causals),
        "has_condition_state": bool(as_found or as_left),
        "has_as_found": as_found is not None,
        "has_as_left": as_left is not None,
        "has_procedural_deviation": bool(proc_dev.get("detected")),
        "has_negation": False,
        "has_conjecture": False,
    }
    return out

# ADD to helpers.py — never cache this, always produce a fresh nlp for Stage 5
def _get_stage5_nlp():
    """
    Return a fresh spaCy nlp instance for Stage 5 extractors.
    Must NOT be cached — CausalBase.__init__ mutates the nlp object
    by removing 'ner' and adding 8 custom pipeline components.
    A cached instance would be permanently modified after first use.
    """
    model_candidates = ["en_core_web_trf", "en_core_web_lg", "en_core_web_sm"]
    for name in model_candidates:
        try:
            return spacy.load(name)   # fresh load each time
        except Exception:
            continue
    raise RuntimeError("Could not load any spaCy English model for Stage 5.")

def _run_causal_extractor(extractor_cls, nlp, chunk_text: str, ner_seed) -> Any:
    """
    Instantiate extractor, register NER seed entities as patterns, then run.
    Without registered entity patterns, CausalBase finds nothing to extract.
    """
    extractor = extractor_cls(nlp)

    # Build entity patterns from NER seed — equipment IDs and components
    patterns = []
    for eid in (ner_seed.equipment_ids or []):
        patterns.append({"label": "SSC", "pattern": eid})
    for comp in (ner_seed.components or []):
        patterns.append({"label": "SSC", "pattern": comp})

    if patterns:
        extractor.addEntityPattern("SSC", patterns)

    extractor(chunk_text, extract=True)
    return extractor