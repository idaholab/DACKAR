
# mdParser.py
# -------------------------------------------------------------------------
# Markdown -> hierarchical sections/chunks + Table/Figure mapping
# Enrich with:
#  - NER for MBSE entities (dictionary-based) & standards references (regex)
#  - Summarization via Ollama (with healthcheck, /api/chat first, /api/generate fallback)
#  - Keyword extraction for RAG
#  - Provenance + confidence scoring
# Emits:
#  - structured_output.json
#  - chunks.jsonl (TextChunk, Table, Figure nodes)
# -------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import time
import logging
import datetime
import requests
import unicodedata
import string
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from pathlib import Path

# ------------------------------
# Logging
# ------------------------------
LOGGER = logging.getLogger("mdParser")
if not LOGGER.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    LOGGER.addHandler(ch)
LOGGER.setLevel(logging.INFO)


# Canonical section roles per doc type.
# Keys are lowercase regex patterns; values are canonical role names.
# Extend per plant by adding entries — no YAML dependency needed at this stage.
FIELD_LABEL_MAP: dict[str, dict[str, str]] = {
    "CR": {
        r"description|condition\s+found|condition\s+description|event\s+description": "description",
        r"immediate\s+action|initial\s+action|corrective\s+action\s+taken\s+immediately": "immediate_actions",
        r"cause|probable\s+cause|root\s+cause|cause\s+statement": "cause_statement",
        r"corrective\s+action|recommended\s+action|action\s+item": "corrective_actions",
        r"scope|header|title|identification": "header",
    },
    "WO": {
        r"scope|work\s+scope|description\s+of\s+work|job\s+plan": "work_scope",
        r"as.?found|equipment\s+condition\s+found|condition\s+as\s+found": "as_found",
        r"work\s+performed|task\s+performed|maintenance\s+performed|work\s+done": "work_performed",
        r"as.?left|equipment\s+condition\s+as\s+left|condition\s+as\s+left": "as_left",
        r"parts?\s+replaced|material|parts?\s+used": "parts_replaced",
        r"scope|header|title|identification": "header",
    },
    "SOP": {
        r"purpose|objective|scope": "purpose",
        r"precondition|initial\s+condition|prerequisites?": "preconditions",
        r"safety|caution|warning|hold": "safety_holds",
        r"step|procedure\s+step|action": "steps",
        r"reference|related\s+document": "references",
    },
    "ECA": {
        r"event|event\s+description|background": "event",
        r"causal\s+factor|cause|contributing\s+cause": "causal_factors",
        r"evidence|supporting\s+evidence|data": "evidence_items",
        r"rationale|analysis|finding": "rationale",
        r"recommendation|corrective\s+action|follow.?up": "recommended_followups",
    },
    "OTHER": {},
}

# Roles that carry high causal/RCA signal — used to set index_in_vector_store priority
HIGH_SIGNAL_ROLES = {
    "cause_statement", "as_found", "as_left", "causal_factors",
    "rationale", "corrective_actions", "immediate_actions",
}

def canonicalize_section_role(title: str, doc_type: str) -> str:
    """
    Map a raw section heading to a canonical role name using FIELD_LABEL_MAP.
    Falls back to 'body' if no pattern matches.
    """
    label_map = FIELD_LABEL_MAP.get(doc_type, {})
    title_low = title.lower().strip()
    for pattern, role in label_map.items():
        if re.search(pattern, title_low):
            return role
    return "body"

# ------------------------------
# Utilities
# ------------------------------
STOPWORDS = {
    "the","and","of","to","in","a","for","is","on","with","by","at","from",
    "as","that","this","it","be","are","or","an","was","were","has","have","had",
    "which","not","can","may","will","within","per","performs","function","system"
}

def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

# ------------------------------
# Markdown parsing
# ------------------------------
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
IMAGE_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>[^)]+)\)")

def parse_markdown_sections(md_text: str) -> List[Dict[str, Any]]:
    """
    Parse headings and build a hierarchical section list.
    Each element: {"title", "level", "text", "figures": [...], "tables": [...]}
    """
    lines = md_text.splitlines()
    sections: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, Any]] = None

    def start_section(title: str, level: int):
        return {"title": title, "level": level, "text": "", "figures": [], "tables": []}

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if current_section and (current_section["title"] or current_section["text"].strip()):
                sections.append(current_section)
            level = len(m.group("hashes"))
            title = m.group("title").strip()
            current_section = start_section(title, level)
            continue

        im = IMAGE_RE.search(line)
        if im:
            current_section = current_section or start_section("Untitled", 3)
            current_section["figures"].append({
                "alt": im.group("alt").strip(),
                "path": im.group("path").strip()
            })
            current_section["text"] += (line + "\n")
            continue

        current_section = current_section or start_section("Untitled", 3)
        current_section["text"] += (line + "\n")

    if current_section and (current_section["title"] or current_section["text"].strip()):
        sections.append(current_section)

    return sections

def build_section_paths(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds a hierarchical section_path based on heading nesting.
    """
    stack: List[Tuple[int, str]] = []
    out: List[Dict[str, Any]] = []
    for s in sections:
        level = int(s.get("level", 3))
        title = str(s.get("title") or "Untitled").strip()
        m = re.match(r"^(\d+)\.\s+", title)
        top_num = int(m.group(1)) if m else None

        # Force headings like "1. ...", "2. ..." to be top-level siblings.
        # This is more robust for CR/WO/ECA forms where heading levels often drift.
        if top_num is not None:
            stack = []
            eff_level = 1
            stack.append((eff_level, title))
        else:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        ss = dict(s)
        ss["section_path"] = " > ".join(t for _, t in stack)
        out.append(ss)
    return out

# ------------------------------
# Table ingestion (JSON normalized)
# ------------------------------
def load_tables_from_json(paths: List[str]) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    for p in paths or []:
        if not p or not os.path.exists(p):
            LOGGER.warning("Table JSON path missing or not found: %s", p)
            continue
        try:
            data = json.loads(read_text(p))
            if isinstance(data, list):
                tables.extend([clean_table_object(t) for t in data])
            else:
                LOGGER.warning("Unexpected table JSON format in: %s", p)
        except Exception as e:
            LOGGER.exception("Failed to read table JSON %s: %s", p, e)
    return tables

def _clean_table_cell(x: Any) -> str:
    text = "" if x is None else str(x)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u200b\u200c\u200d\u00ad\u2060]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # fix obvious punctuation-inserted word breaks from wrapped extraction
    text = re.sub(r'([A-Za-z])\.([A-Za-z])', r'\1\2', text)
    # strip stray leading punctuation from wrapped cells
    text = re.sub(r'^[\.\,\;\:\-]+\s*', '', text)
    # repair common owner-field corruption
    if re.fullmatch(r'.*maintenance', text, flags=re.IGNORECASE):
        text = "Maintenance"
    elif re.fullmatch(r'.*engineering', text, flags=re.IGNORECASE):
        text = "Engineering"
    return text


def clean_table_object(t: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(t)
    out["columns"] = [_clean_table_cell(c) for c in (t.get("columns") or [])]
    rows = [[_clean_table_cell(c) for c in row] for row in (t.get("rows") or [])]
    out["rows"] = _repair_wrapped_action_rows(out["columns"], rows)
    if "caption" in out and out["caption"] is not None:
        out["caption"] = _clean_table_cell(out["caption"])
    return out

def _repair_wrapped_action_rows(columns: List[str], rows: List[List[str]]) -> List[List[str]]:
    """
    Light repair for common corrective-action tables where wrapped lines corrupt
    owner/status columns. Only applies when columns look action-like.
    """
    cols_low = [c.lower() for c in columns]
    if not any("owner" in c for c in cols_low):
        return rows

    repaired: List[List[str]] = []
    owner_idx = next((i for i, c in enumerate(cols_low) if "owner" in c), None)
    if owner_idx is None:
        return rows

    for row in rows:
        rr = list(row)
        while len(rr) < len(columns):
            rr.append("")
        owner = rr[owner_idx].strip()
        if re.fullmatch(r'.*maintenance', owner, flags=re.IGNORECASE):
            rr[owner_idx] = "Maintenance"
        elif re.fullmatch(r'.*engineering', owner, flags=re.IGNORECASE):
            rr[owner_idx] = "Engineering"
        repaired.append(rr)
    return repaired

# ------------------------------
# NER & references
# ------------------------------

def detect_mbse_mentions(
    text: str, mbse_entities: Optional[List[Dict[str, Any]]]
) -> Tuple[List[str], float]:
    fallback_tags = detect_equipment_tags(text)
    if not mbse_entities:
        return fallback_tags, (0.7 if fallback_tags else 0.0)

    found_ids: List[str] = []
    id_matched: List[bool] = []   # True = matched by explicit ID, False = label-only
    text_low = text.lower()

    for ent in mbse_entities:
        ent_id = str(ent.get("id", "")).strip()
        label  = str(ent.get("label", "")).strip()

        if ent_id and re.search(rf"\b{re.escape(ent_id.lower())}\b", text_low):
            found_ids.append(ent_id)
            id_matched.append(True)       # explicit ID hit → high confidence
            continue
        if label and re.search(rf"\b{re.escape(label.lower())}\b", text_low):
            found_ids.append(ent_id or label)
            id_matched.append(False)      # label-only hit → lower confidence

    found_ids = list(dict.fromkeys(found_ids))  # preserve order, deduplicate

    found_ids = list(dict.fromkeys(found_ids + fallback_tags))
    if not found_ids:
        return [], 0.0

    # 0.9 when at least one explicit-ID match; 0.6 for label-only matches
    conf = 0.9 if any(id_matched) else (0.7 if fallback_tags else 0.6)
    return found_ids, conf

STANDARD_PATTERNS = [
    r"\bASME\b.*\bBPVC\b.*\b(Section|Sec\.?|III|NCA|NC|NB|NH)\b",
    r"\bIEEE\b.*\b603\b(?:\-\d{4})?",
    r"\bANSI\/ANS\-?15\.\d+\b",
    r"\bNFPA\b\s?\d+\b",
    r"\bASTM\b\s?[A-Z]?\d+\b",
]

TAG_PATTERNS = [
    r"\b[A-Z]{1,4}-\d{2,5}[A-Z]?\b",       # P-101A, FT-1102
    r"\b[A-Z]{2,6}\d{2,5}[A-Z]?\b",        # more compact variants
]

DOC_REF_PREFIXES = {"CR", "WO", "ECA", "SOP", "MR", "PM"}

def detect_equipment_tags(text: str) -> List[str]:
    found: List[str] = []
    for pat in TAG_PATTERNS:
        found.extend(m.group(0) for m in re.finditer(pat, text))
    clean: List[str] = []
    for tag in found:
        prefix = tag.split("-", 1)[0] if "-" in tag else re.match(r"^[A-Z]+", tag).group(0) if re.match(r"^[A-Z]+", tag) else ""
        if prefix in DOC_REF_PREFIXES:
            continue
        clean.append(tag)
    return list(dict.fromkeys(clean))


def detect_document_refs(text: str) -> List[str]:
    refs: List[str] = []
    for prefix in DOC_REF_PREFIXES:
        refs.extend(
            m.group(0)
            for m in re.finditer(rf"\b{prefix}-\d{{2,5}}(?:-\d{{2,5}})?[A-Z]?\b", text)
        )
    return list(dict.fromkeys(refs))

def detect_standard_refs(text: str) -> Tuple[List[str], float]:
    refs: List[str] = []
    for pat in STANDARD_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            refs.append(m.group(0).strip())
    refs = list(dict.fromkeys(refs))
    conf = 0.9 if refs else 0.0
    return refs, conf

# ------------------------------
# Summarization (Ollama) + fallback + healthcheck
# ------------------------------
def _extractive_summary(text: str, max_chars: int = 600) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:3])
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
    return summary

def _ollama_healthcheck(base: str, timeout: int = 5) -> bool:
    try:
        r = requests.get(f"{base}/api/version", timeout=timeout)
        r.raise_for_status()
        return True
    except Exception:
        return False


def summarize_text_ollama(text: str, model: Optional[str] = None, timeout: int = 60) -> str:
    """
    Summarize using Ollama REST API, preferring /api/chat with streaming.
    Falls back to /api/generate streaming. We also truncate input to fit context.
    """

    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = model or os.environ.get("OLLAMA_MODEL", "mistral:latest")
    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

    # Truncate section text to fit into context safely
    text_safe = _truncate_for_context(text, num_ctx, safety_ratio=0.7)

    user_prompt = (
        "You are preparing a concise evidence summary for a nuclear research reactor "
        "licensing document. Summarize the following content in 3-5 sentences, focusing "
        "on technical facts, parameters, and compliance-relevant details:\n\n" + text_safe
    )

    # Try /api/chat with streaming first
    try:
        with requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": user_prompt}],
                "options": {"num_ctx": num_ctx},
                "stream": True  # stream chunks
            },
            timeout=timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            # Accumulate streamed chunks
            chunks = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # On /api/chat stream, each chunk often has {"message":{"content":"..."}}
                    msg = data.get("message") or {}
                    piece = (msg.get("content") or "").strip()
                    if piece:
                        chunks.append(piece)
                except Exception:
                    # Sometimes Ollama emits plain text chunks; accept them
                    chunks.append(line)
            final = " ".join(chunks).strip()
            return final or _extractive_summary(text_safe)
    except Exception as e_chat:
        LOGGER.warning("Ollama /api/chat (stream) summarization failed (%s); trying /api/generate.", e_chat)

    # Fallback: /api/generate with streaming
    try:
        with requests.post(
            f"{base}/api/generate",
            json={
                "model": model,
                "prompt": user_prompt,
                "options": {"num_ctx": num_ctx},
                "stream": True
            },
            timeout=timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            chunks = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # On /api/generate stream, chunks typically have {"response":"..."}
                    piece = (data.get("response") or "").strip()
                    if piece:
                        chunks.append(piece)
                except Exception:
                    chunks.append(line)
            final = " ".join(chunks).strip()
            return final or _extractive_summary(text_safe)
    except Exception as e_gen:
        LOGGER.warning("Ollama /api/generate (stream) summarization failed (%s); using extractive fallback.", e_gen)
        return _extractive_summary(text_safe)


# ------------------------------
# Keyword extraction
# ------------------------------
def extract_keywords(text: str, top_k: int = 12) -> List[str]:
    # Keywords should be computed from cleaned text without markup noise.
    text = strip_markup_noise(text or "")
    tokens = re.findall(r"[A-Za-z0-9\-]+", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    freq = Counter(tokens)
    bigrams = Counter([" ".join([tokens[i], tokens[i+1]]) for i in range(len(tokens)-1)])
    combined = [(w, c) for w, c in freq.items()]
    combined += [(bg, c) for bg, c in bigrams.items() if all(tok not in STOPWORDS for tok in bg.split())]
    combined.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in combined[:top_k]]

MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MD_LINK_RE  = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MD_BOLD_ITALIC_RE = re.compile(r"(\*\*|\*|__|_)(.*?)\1")

def strip_markdown_noise(text: str) -> str:
    if not text:
        return text
    text = MD_IMAGE_RE.sub(" ", text)
    text = MD_LINK_RE.sub(r"\1", text)         # keep link text, drop URL
    text = MD_BOLD_ITALIC_RE.sub(r"\2", text)  # drop emphasis markers
    return re.sub(r"\s+", " ", text).strip()

# ------------------------------
# Table/Figure → Section assignment
# ------------------------------
def assign_tables_to_sections(tables: List[Dict[str, Any]], sections: List[Dict[str, Any]], doc_index: Dict[str, Any]) -> List[Dict[str, Any]]:
    section_texts = [(i, s["title"], s["text"]) for i, s in enumerate(sections)]
    enriched_tables = []
    for idx, t in enumerate(tables):
        text_probe = " ".join((t.get("columns") or [])) + " " + " ".join([" ".join(r) for r in t.get("rows", [])])
        best_score, best_sec_idx = 0.0, None
        t_tokens = set(extract_keywords(text_probe, top_k=20))
        for si, stitle, stext in section_texts:
            s_tokens = set(extract_keywords(stext, top_k=20))
            overlap = len(t_tokens & s_tokens)
            score = overlap / (len(t_tokens) + 1e-6)
            if score > best_score:
                best_score, best_sec_idx = score, si
        if best_sec_idx is None and sections:
            best_sec_idx = min(idx, len(sections) - 1)
            best_score = 0.5
        enriched = dict(t)
        enriched["belongs_to_section"] = sections[best_sec_idx]["title"] if best_sec_idx is not None else None
        enriched["link_confidence"] = round(min(0.95, 0.5 + best_score), 2)
        enriched_tables.append(enriched)
    return enriched_tables

def assign_figures_to_sections(
    figures_in_text: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    doc_figures: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build a figure list with section assignments.

    Sources (merged, deduplicated by path):
      1. Inline Markdown image references parsed per section (s["figures"]).
      2. `doc_figures`: figures list from document_index (Marker path-indexed).
    """
    fig_records: List[Dict[str, Any]] = []
    seen_paths: set = set()

    # Source 1: inline markdown figures already attached to sections
    for s in sections:
        for fig in s.get("figures", []):
            path = fig.get("path", "")
            seen_paths.add(path)
            fig_records.append({
                "path": path,
                "alt": fig.get("alt"),
                "belongs_to_section": s["title"],
                "link_confidence": 0.9,
            })

    # Source 2: Marker-indexed figures not already captured above
    for fig in (doc_figures or []):
        path = fig.get("path", "")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        # No inline section reference: mark as unassigned (graph query can resolve later)
        fig_records.append({
            "path": path,
            "alt": fig.get("name"),
            "belongs_to_section": None,
            "link_confidence": 0.5,   # lower — heuristically unassigned
        })

    return fig_records

# ------------------------------
# Main API
# ------------------------------


def _section_to_chunk(
    s: dict, idx: int, doc_id: str, doc_type: str, doc_name: str,
    source_path: str, ingest_id, classification: str, content_hash
) -> dict:
    """Build a single section-level TextChunk with canonical role."""
    role = canonicalize_section_role(s["title"], doc_type)
    return {
        "type": "TextChunk",
        "granularity": "section",
        "chunk_id": f"chunk:{doc_id}:sec{idx}",
        "doc_id": doc_id,
        "doc_type": doc_type,
        "doc_key": Path(doc_name).stem,
        "section_title": s["title"],
        "section_path": s.get("section_path"),
        "section_role": role,                       # canonical role — feeds stage3_to_6
        "heading_level": s["level"],
        "text": s["text"],
        "raw_text": s.get("raw_text", ""),
        "keywords": s.get("keywords", []),
        "mentions_component_ids": s.get("mentions_component_ids", []),
        "document_refs": s.get("document_refs", []),
        "references_standard_refs": s.get("references_standard_refs", []),
        "nureg_section_ids": s.get("nureg_section_ids", []),
        "page_start": s.get("page_start"),
        "page_end": s.get("page_end"),
        "provenance": {
            "source_doc_id": doc_id,
            "extraction_method": "nlp+regex (raw-first)",
            "timestamp": _now_iso(),
            "confidence": max(s.get("mentions_confidence", 0.0),
                              s.get("references_confidence", 0.0)),
        },
        "classification": classification,
        "index_in_vector_store": True,
        "index_in_graph": True,
        "source_path": source_path,
        "doc_name": doc_name,
        "ingest_id": ingest_id,
        "content_hash": content_hash,
    }


def _paragraph_subchunks(
    s: dict, sec_idx: int, doc_id: str, doc_type: str, doc_name: str,
    source_path: str, ingest_id, classification: str, content_hash,
    mbse_entities
) -> list[dict]:
    """Paragraph sub-chunks for a section. Only produced for high-signal roles."""
    role = canonicalize_section_role(s["title"], doc_type)
    # Skip sub-chunking for header/body/reference sections — they add noise
    if role not in HIGH_SIGNAL_ROLES and doc_type not in ("SOP",):
        return []
    paragraphs = split_into_paragraphs(s.get("text", ""))
    subtexts = chunk_paragraphs(paragraphs, max_chars=1400, overlap_chars=200)
    chunks = []
    for j, subtext in enumerate(subtexts, start=1):
        chunks.append({
            "type": "TextChunk",
            "granularity": "paragraph",
            "chunk_id": f"chunk:{doc_id}:sec{sec_idx}:p{j:03d}",
            "doc_id": doc_id,
            "doc_type": doc_type,
            "section_title": s["title"],
            "section_path": s.get("section_path"),
            "section_role": role,
            "heading_level": s["level"],
            "section_index": sec_idx,
            "paragraph_index": j,
            "text": subtext,
            "raw_text": "",
            "keywords": extract_keywords(subtext),
            "mentions_component_ids": detect_mbse_mentions(subtext, mbse_entities)[0],
            "references_standard_refs": detect_standard_refs(subtext)[0],
            "nureg_section_ids": s.get("nureg_section_ids", []),
            "page_start": s.get("page_start"),
            "page_end": s.get("page_end"),
            "truncated": j < len(subtexts),
            "provenance": {
                "source_doc_id": doc_id,
                "extraction_method": "nlp+regex (raw-first)",
                "timestamp": _now_iso(),
                "confidence": 0.8,
            },
            "classification": classification,
            "index_in_vector_store": True,
            "index_in_graph": False,
            "source_path": source_path,
            "doc_name": doc_name,
            "ingest_id": ingest_id,
            "content_hash": content_hash,
        })
    return chunks


def build_chunks_for_doc_type(
    sections: list[dict],
    doc_id: str,
    doc_type: str,
    doc_name: str,
    source_path: str,
    ingest_id,
    classification: str,
    content_hash,
    mbse_entities,
) -> list[dict]:
    """
    Dispatch to the correct chunking strategy based on doc_type.
    All strategies produce section-level chunks with canonical roles.
    SOP additionally produces one chunk per step group.
    """
    chunks = []

    if doc_type == "SOP":
        # SOP: each section is a step/step-group; sub-chunk every section
        for idx, s in enumerate(sections, start=1):
            chunks.append(_section_to_chunk(
                s, idx, doc_id, doc_type, doc_name,
                source_path, ingest_id, classification, content_hash))
            chunks.extend(_paragraph_subchunks(
                s, idx, doc_id, doc_type, doc_name,
                source_path, ingest_id, classification, content_hash, mbse_entities))

    elif doc_type in ("CR", "WO", "ECA"):
        # Field-boundary chunking: one chunk per canonical role.
        # Sections that map to the same role are merged before chunking.
        role_buckets: dict[str, list[dict]] = {}
        for s in sections:
            role = canonicalize_section_role(s["title"], doc_type)
            role_buckets.setdefault(role, []).append(s)

        idx = 0
        for role, role_sections in role_buckets.items():
            idx += 1
            # Merge all sections sharing a role into one logical chunk
            merged_text = "\n\n".join(s["text"] for s in role_sections).strip()
            merged_raw = "\n\n".join(s.get("raw_text", "") for s in role_sections).strip()
            merged_keywords = list(dict.fromkeys(
                kw for s in role_sections for kw in s.get("keywords", [])))
            merged_mentions = list(dict.fromkeys(
                m for s in role_sections for m in s.get("mentions_component_ids", [])))
            merged_refs = list(dict.fromkeys(
                r for s in role_sections for r in s.get("references_standard_refs", [])))
            merged_doc_refs = list(dict.fromkeys(
                r for s in role_sections for r in s.get("document_refs", [])))

            synthetic_section = {
                "title": role_sections[0]["title"],
                "level": role_sections[0]["level"],
                "section_path": role_sections[0].get("section_path"),
                "text": merged_text,
                "raw_text": merged_raw,
                "keywords": merged_keywords,
                "mentions_component_ids": merged_mentions,
                "references_standard_refs": merged_refs,
                "document_refs": merged_doc_refs,
                "mentions_confidence": max(
                    s.get("mentions_confidence", 0.0) for s in role_sections),
                "references_confidence": max(
                    s.get("references_confidence", 0.0) for s in role_sections),
                "nureg_section_ids": role_sections[0].get("nureg_section_ids", []),
                "page_start": role_sections[0].get("page_start"),
                "page_end": role_sections[-1].get("page_end"),
            }
            chunk = _section_to_chunk(
                synthetic_section, idx, doc_id, doc_type, doc_name,
                source_path, ingest_id, classification, content_hash)
            chunk["section_role"] = role          # already set but be explicit
            chunks.append(chunk)

            # Sub-chunk high-signal roles only
            if role in HIGH_SIGNAL_ROLES and len(merged_text) > 600:
                chunks.extend(_paragraph_subchunks(
                    synthetic_section, idx, doc_id, doc_type, doc_name,
                    source_path, ingest_id, classification, content_hash, mbse_entities))
    else:
        # OTHER: original heading-based strategy
        for idx, s in enumerate(sections, start=1):
            chunks.append(_section_to_chunk(
                s, idx, doc_id, doc_type, doc_name,
                source_path, ingest_id, classification, content_hash))

    return chunks

def md_parser(
    document_index: Dict[str, Any],
    destination_folder: Optional[str],
    mbse_entities: Optional[List[Dict[str, Any]]] = None,
    nureg_section_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    
    if not document_index or not isinstance(document_index, dict):
        raise ValueError("document_index must be a dict produced by pdfParser.")
    text_md_path = document_index.get("text_md_path")
    if not text_md_path or not os.path.exists(text_md_path):
        raise FileNotFoundError(f"text_md_path not found: {text_md_path}")

    if destination_folder is None:
        destination_folder = os.path.dirname(os.path.dirname(text_md_path))

    LOGGER.info("Reading Markdown text from: %s", text_md_path)
    md_text = read_text(text_md_path)
    sections = parse_markdown_sections(md_text)
    sections = build_section_paths(sections)

    tables = load_tables_from_json(document_index.get("tables_paths"))
    LOGGER.info("Assigning tables to sections with heuristic matching.")
    enriched_tables = assign_tables_to_sections(tables, sections, document_index)
    section_path_by_title = {
        s["title"]: s.get("section_path") for s in sections
    }

    enriched_figures = assign_figures_to_sections(
        figures_in_text=[],
        sections=sections,
        doc_figures=document_index.get("figures", []),
    )

    #Build a lookup: image path -> section title + confidence
    _fig_section_map: Dict[str, Dict[str, Any]] = {
        f["path"]: {
            "section": f["belongs_to_section"],
            "section_path": next(
                (s.get("section_path") for s in sections if s["title"] == f["belongs_to_section"]),
                None,
            ),
            "conf": f["link_confidence"],
        }
        for f in enriched_figures
    }

    LOGGER.info("Running NER and standards detection per section.")

    # Step 3 fix: Healthcheck Ollama once, then decide summary strategy
    #ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    #ollama_ok = _ollama_healthcheck(ollama_base)
    #if not ollama_ok:
    #    LOGGER.warning("Ollama healthcheck failed or service not reachable at %s; using extractive summaries.", ollama_base)

    for s in sections:
        # Preserve raw extracted text for audit/traceability
        s["raw_text"] = s.get("text", "")

        # Create a safe cleaned text for indexing/keywording/LLM input (if used later)
        s["text"] = normalize_raw_text(s.get("text", ""))

        mentions, conf_m = detect_mbse_mentions(s["text"], mbse_entities)
        doc_refs = detect_document_refs(s["text"])
        stds, conf_s = detect_standard_refs(s["text"])

        s["mentions_component_ids"] = mentions
        s["mentions_confidence"] = conf_m
        s["document_refs"] = doc_refs
        s["references_standard_refs"] = stds
        s["references_confidence"] = conf_s
        s["nureg_section_ids"] = nureg_section_ids or []
        
        # Keep raw copy for provenance/audit
        s["_raw_text"] = s.get("raw_text", "")

        # Normalize text used for NLP / summarization
        # (strip markdown noise first so links/images don’t pollute summaries/keywords)
        #s["text"] = normalize_extracted_text(strip_markdown_noise(s["text"]))

        """
        try:
            if ollama_ok:
                # Throttle a bit between requests
                time.sleep(0.25)
                s["summary"] = _retry(summarize_text_ollama, max_tries=2, wait_sec=3, text=s["text"])
            else:
                s["summary"] = _extractive_summary(s["text"])
        except Exception as e:
            LOGGER.warning("Summarization failed for section '%s' (%s); using extractive fallback.", s["title"], e)
            s["summary"] = _extractive_summary(s["text"])"""
        
        #s["summary"] = clean_summary_text(s.get("summary", ""))

        s["keywords"] = extract_keywords(s["text"])

    doc_id = document_index.get("doc_id")
    source_path = document_index.get("source_path") or (document_index.get("source") or {}).get("relpath")
    doc_name = document_index.get("doc_name") or os.path.basename(source_path) if source_path else None
    ingest_id = document_index.get("ingest_id")
    
    classification = document_index.get("classification", "internal")

    # canonical content hash for all chunks (fallback: doc_version)
    content_hash = document_index.get("content_hash") or document_index.get("doc_version")

    doc_type = document_index.get("doc_type", "OTHER")

    chunks: list[dict] = build_chunks_for_doc_type(
        sections=sections,
        doc_id=doc_id,
        doc_type=doc_type,
        doc_name=doc_name,
        source_path=source_path,
        ingest_id=ingest_id,
        classification=classification,
        content_hash=content_hash,
        mbse_entities=mbse_entities,
    )

    for tidx, t in enumerate(enriched_tables, start=1):
        chunk_id = f"table:{doc_id}:{tidx}"
        headers = t.get("columns") or []
        rows = t.get("rows") or []
        caption = t.get("caption") or ""
        table_text = textify_table(caption, headers, rows)
        chunks.append({
            "type": "Table",
            "granularity": "table",
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "belongs_to_section": t.get("belongs_to_section"),
            "section_path": section_path_by_title.get(t.get("belongs_to_section")),
            "columns": headers,
            "rows": rows,
            "caption": caption,
            "page": t.get("page"),
            "text": table_text,   # textual representation indexed by Chroma
            "provenance": {
                "source_doc_id": doc_id,
                "extraction_method": document_index.get("extraction", {}).get("tables"),
                "timestamp": _now_iso(),
                "confidence": t.get("link_confidence", 0.6)
            },
            "classification": classification,
            "index_in_vector_store": True,
            "index_in_graph": True,
            "source_path": source_path,
            "doc_name": doc_name,
            "ingest_id": ingest_id,
            "content_hash": document_index.get("content_hash") or document_index.get("doc_version")
        })

    for fidx, f in enumerate(document_index.get("figures", []), start=1):
        fig_path = f.get("path", "")
        section_info = _fig_section_map.get(fig_path, {})
        # Compose indexable alt text from available metadata
        alt_text_parts = [p for p in [f.get("name"), f.get("alt"), f.get("caption")] if p]
        alt_text = " | ".join(alt_text_parts) if alt_text_parts else ""

        chunks.append({
            "type": "Figure",
            "chunk_id": f"figure:{doc_id}:{fidx}",
            "doc_id": doc_id,
            "belongs_to_section": section_info.get("section"),      
            "section_path": section_info.get("section_path"),
            "path": fig_path,
            "alt": f.get("name"),
            "caption": f.get("caption"),
            "page": f.get("page"),
            "text": alt_text,                                       
            "provenance": {
                "source_doc_id": doc_id,
                "extraction_method": document_index.get("extraction", {}).get("text"),
                "timestamp": _now_iso(),
                "confidence": section_info.get("conf", 0.7),
            },
            "classification": classification,
            "index_in_vector_store": bool(alt_text.strip()),
            "index_in_graph": True,
            "source_path": source_path,
            "doc_name": doc_name,
            "ingest_id": ingest_id,
            "content_hash": document_index.get("content_hash") or document_index.get("doc_version")
    })
        
    base_doc_folder = os.path.dirname(os.path.dirname(text_md_path))
    out_struct_path = os.path.join(base_doc_folder, "index", f"{doc_id}_structured_output.json")
    chunks_path = os.path.join(base_doc_folder, "index", f"{doc_id}_chunks.jsonl")

    structured_output = {
        "doc_id": doc_id,
        "sections": sections,
        "tables": enriched_tables,
        "figures": document_index.get("figures", []),
        "timestamp": _now_iso(),
        "provenance": {
            "source_doc_id": doc_id,
            "extraction_method": "mdParser-raw-3.0.0",
            "timestamp": _now_iso()
        },
        "source_path": source_path,
        "doc_name": doc_name,
        "ingest_id": ingest_id,
        "chunks_jsonl_path": chunks_path,
        "structured_output_path": out_struct_path,
    }

    write_json(out_struct_path, structured_output)
    write_jsonl(chunks_path, chunks)

    LOGGER.info("mdParser complete: %s", doc_id)
    return structured_output


import time

def _retry(fn, max_tries=3, wait_sec=2, *args, **kwargs):
    for i in range(max_tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == max_tries - 1:
                raise
            time.sleep(wait_sec * (i + 1))  # exponential-ish backoff

def _estimate_tokens_from_text(text: str) -> int:
    # Heuristic: ~4 chars per token (varies by model)
    return max(1, int(len(text) / 4))

def _truncate_for_context(text: str, num_ctx: int, safety_ratio: float = 0.7) -> str:
    """
    Truncate text so that total prompt stays within ~70% of model context.
    We reserve the rest for system + instruction tokens.
    """
    target_tokens = int(num_ctx * safety_ratio)
    # Convert back to chars (approx 4 chars/token)
    char_budget = target_tokens * 4
    if len(text) <= char_budget:
        return text
    return text[:char_budget].rsplit(" ", 1)[0] + "…"

import re
import unicodedata

TAG_RE = re.compile(r"<[^>]+>")  # simple HTML tag stripper
# conservative normalization for extracted PDF text

def normalize_extracted_text(text: str) -> str:
    if not text:
        return text

    # 1) Unicode normalize, remove invisible chars
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u200b\u200c\u200d\u00ad\u2060]', '', text)

    # 2) Fix hyphenation across line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'-\s*\n\s*', '', text)

    # 3) Collapse line breaks inside paragraphs, preserve blank lines
    text = text.replace('\n\n', '<PARA>')
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = text.replace('<PARA>', '\n\n')

    # 4) Remove obvious layout tags that poison keywords/summaries
    text = TAG_RE.sub(' ', text)          # removes <sup>, <span id=...>, etc.
    text = re.sub(r'\s+', ' ', text).strip()

    # 5) Normalize parentheses spacing: "( ER )" -> "(ER)"
    text = re.sub(r'\(\s*([A-Za-z0-9\s]+?)\s*\)', lambda m: '(' + re.sub(r'\s+', '', m.group(1)) + ')', text)

    # 6) Very conservative split-word repair for lowercase OCR/PDF artifacts only.
    def _fix_lower_split(m: re.Match) -> str:
        left, mid, right = m.group(1), m.group(2), m.group(3)
        token = left + mid + right
        if len(token) < 6:
            return m.group(0)
        return token
    text = re.sub(
        r'\b([a-z]{2,})\s+([a-z])\s+([a-z]{2,})\b',
        _fix_lower_split,
        text)

    #    b) Mid-word splits before common suffix fragments (prevents "To improve" -> "Toimprove")
    _SUFFIX_FRAG = (
        r"(ability|abilities|ational|ation|tions?|ment|ments|ness|nesses|"
        r"ing|ings|ized|ization|izations|ity|ities|ally|al|ive|ives|able|ables|"
        r"ence|ences|ance|ances|ous|ously)")
    
    text = re.sub(
        rf"\b([A-Za-z]{{3,}})\s+({_SUFFIX_FRAG})\b",
        r"\1\2",
        text,
        flags=re.IGNORECASE)

    # 7) Acronym plural join when split: "NPP s" -> "NPPs" (after parentheses cleanup this helps outside parens too)
    text = re.sub(r'\b([A-Z]{2,})\s+(s)\b', r'\1\2', text)

    # 8) Clean spacing around punctuation
    text = re.sub(r'\s+([,.;:?!%])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()

    return text

def clean_summary_text(text: str) -> str:
    """Light cleanup for LLM summaries without reintroducing 'Toimprove' issues."""
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u200b\u200c\u200d\u00ad\u2060]', '', text)
    # collapse "Ne o 4 j" -> "Neo4j", "MB SE" -> "MBSE" when it’s clearly an acronym-like pattern
    text = re.sub(r"\b([A-Z])\s+([A-Z])\b", r"\1\2", text)
    text = re.sub(r"\b([A-Za-z]{2,})\s+(\d+)\s+([A-Za-z]{1,})\b", r"\1\2\3", text)  # e.g., "Ne o 4 j"
    return re.sub(r"\s+", " ", text).strip()

# ------------------------------
# Raw-first text cleaning (safe)
# ------------------------------
# Goal: keep whitespace between words, remove invisible chars / obvious markup noise,
# and fix the most common mid-word split artifact (single-letter splits).
# IMPORTANT: This is intentionally conservative to avoid "To improve" -> "Toimprove".

HTML_TAG_RE = re.compile(r"<[^>]+>")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MD_LINK_RE  = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # keep link text, drop URL
MD_EMPH_RE  = re.compile(r"(\*\*|\*|__|_)(.*?)\1")

def strip_markup_noise(text: str) -> str:
    if not text:
        return text
    # Markdown noise
    text = MD_IMAGE_RE.sub(" ", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = MD_EMPH_RE.sub(r"\2", text)
    # HTML-like tags (marker sometimes emits <sup>, <span id=...>, etc.)
    text = HTML_TAG_RE.sub(" ", text)
    return text

def normalize_raw_text(text: str) -> str:
    if not text:
        return text

    # Unicode normalize + remove invisible/soft hyphen/ZW chars
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u200b\u200c\u200d\u00ad\u2060]', '', text)

    # Normalize newlines; keep paragraph breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'-\s*\n\s*', '', text)       # de-hyphenate line breaks
    text = text.replace('\n\n', '<PARA>')
    text = re.sub(r'\s*\n\s*', ' ', text)       # collapse remaining newlines to spaces
    text = text.replace('<PARA>', '\n\n')

    # Strip markup noise now (keeps human text)
    text = strip_markup_noise(text)

    # Fix spaces inside parentheses ONLY when it looks like an acronym / short token group:
    # "( ER )" -> "(ER)", "( N PP s )" -> "(NPPs)"
    def _paren_fix(m: re.Match) -> str:
        inner = m.group(1).strip()
        # Only compress if it's short and mostly alnum/spaces
        if len(inner) <= 12 and re.fullmatch(r"[A-Za-z0-9 ]+", inner):
            inner = inner.replace(" ", "")
        return f"({inner})"
    text = re.sub(r"\(\s*([^)]+?)\s*\)", _paren_fix, text)

    # Very conservative split-word repair:
    # only merge lowercase fragments like "re li ability" or "rel i ability"
    # and never merge title-case / all-caps plant phrases such as "Train B remained".
    def _fix_split_word(m):
        left = m.group(1)
        mid = m.group(2)
        right = m.group(3)
        token = left + mid + right
        if not (left.islower() and mid.islower() and right.islower()):
            return m.group(0)
        if len(token) < 6:
            return m.group(0)
        return token

    text = re.sub(
        r"\b([a-z]{2,})\s+([a-z])\s+([a-z]{2,})\b",
        _fix_split_word,
        text)

    # Clean spacing around punctuation
    text = re.sub(r"\s+([,.;:?!%])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

########### Splitting chunks into paragraphs ########### 

def split_into_paragraphs(text: str) -> List[str]:
    # Split on blank lines; keep non-empty
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return parts

def chunk_paragraphs(paragraphs: List[str], max_chars: int = 1400, overlap_chars: int = 200) -> List[str]:
    """
    Packs paragraphs into chunks up to max_chars. Overlap is applied between consecutive chunks.
    Char-based is OK since you're storing raw text and embedding later.
    """
    chunks = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            chunks.append(buf)
            # overlap tail
            tail = buf[-overlap_chars:] if overlap_chars > 0 and len(buf) > overlap_chars else ""
            buf = (tail + "\n\n" + p).strip()
    if buf:
        chunks.append(buf)
    return chunks

# ------------------------------
# Table textification (for vector indexing)
# ------------------------------
def textify_table(
    caption: Optional[str],
    headers: List[str],
    rows: List[List[Any]],
    head_rows: int = 4,
    tail_rows: int = 2,
) -> str:
    """
    Compact textual representation for embedding/indexing.

    Always includes:
      - caption
      - column headers
      - first `head_rows` rows  (typical parameter definitions)
      - last  `tail_rows` rows  (often contain limit/summary values)
      - a note when rows are omitted
    """
    lines: List[str] = []
    if caption:
        lines.append(f"Table caption: {caption}")
    if headers:
        lines.append("Columns: " + " | ".join(str(h).strip() for h in headers))

    rows = rows or []
    total = len(rows)

    if total == 0:
        return "\n".join(lines).strip()

    if total <= head_rows + tail_rows:
        # Small enough to show everything
        for i, r in enumerate(rows, start=1):
            lines.append(f"Row {i}: " + " | ".join(str(x).strip() for x in r))
    else:
        for i, r in enumerate(rows[:head_rows], start=1):
            lines.append(f"Row {i}: " + " | ".join(str(x).strip() for x in r))

        omitted = total - head_rows - tail_rows
        lines.append(f"... ({omitted} rows omitted) ...")

        for i, r in enumerate(rows[-tail_rows:], start=total - tail_rows + 1):
            lines.append(f"Row {i}: " + " | ".join(str(x).strip() for x in r))

        lines.append(f"(Table: {total} rows total)")

    return "\n".join(lines).strip()



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mdParser.py <document_index.json> [mbse_entities.json]")
        sys.exit(1)
    doc_index_path = sys.argv[1]
    if not os.path.exists(doc_index_path):
        print(f"document_index.json not found: {doc_index_path}")
        sys.exit(1)
    with open(doc_index_path, "r", encoding="utf-8") as f:
        document_index = json.load(f)
    mbse_entities = None
    if len(sys.argv) >= 3 and os.path.exists(sys.argv[2]):
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            mbse_entities = json.load(f)
    md_parser(document_index=document_index, destination_folder=None, mbse_entities=mbse_entities)
