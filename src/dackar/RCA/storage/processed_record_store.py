from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from .chroma_store import extract_processed_text_record

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(jsonl_path: str) -> Iterable[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("ProcessedRecordStore: parse error in %s line %d: %s", jsonl_path, lineno, exc)
                continue
            if isinstance(obj, dict):
                yield obj

def _is_minimal_processed_record(rec: Dict[str, Any]) -> bool:
    """
    Minimal structural gate for canonical processed_text_record objects.
    """
    if not isinstance(rec, dict):
        return False
    required = ["record_id", "doc_id", "doc_type", "chunk_index", "embedding_text", "metadata", "provenance"]
    for key in required:
        if key not in rec:
            return False
    if not isinstance(rec.get("record_id"), str) or not rec["record_id"].strip():
        return False
    if not isinstance(rec.get("metadata"), dict):
        return False
    if not isinstance(rec.get("provenance"), dict):
        return False
    return True

# ---------------------------------------------------------------------------
# Canonical processed_text_record hydration store
# ---------------------------------------------------------------------------

class ProcessedRecordStore:
    """
    Corpus-level in-memory index of canonical processed_text_record objects.

    Records are indexed by record_id, with a secondary index by provenance.chunk_id
    when available, so downstream components can hydrate either identifier.
    """

    def __init__(self, jsonl_paths: Optional[List[str]] = None) -> None:
        self._by_record_id: Dict[str, Dict[str, Any]] = {}
        self._record_id_by_chunk_id: Dict[str, str] = {}
        if jsonl_paths:
            for path in jsonl_paths:
                self.add_jsonl(path)

    def __len__(self) -> int:
        return len(self._by_record_id)

    def add_jsonl(self, jsonl_path: str) -> int:
        before = len(self._by_record_id)
        for obj in _iter_jsonl(jsonl_path):
            rec = extract_processed_text_record(obj)
            if not rec:
                continue
            self.add_record(rec)
        added = len(self._by_record_id) - before
        LOGGER.info("ProcessedRecordStore: indexed %d processed records from %s.", added, jsonl_path)
        return added

    def add_record(self, rec: Dict[str, Any]) -> bool:
        if not _is_minimal_processed_record(rec):
            LOGGER.warning(
                "ProcessedRecordStore: rejected malformed processed_text_record with keys=%s",
                sorted(rec.keys()) if isinstance(rec, dict) else type(rec).__name__,
            )
            return False

        rid = rec.get("record_id")
        if rid in self._by_record_id:
            LOGGER.warning("ProcessedRecordStore: overwriting existing record_id=%s", rid)
        if not rid:
            return False
        rid = str(rid)
        self._by_record_id[rid] = rec

        provenance = rec.get("provenance") or {}
        chunk_id = provenance.get("chunk_id") or rec.get("chunk_id")

        if chunk_id:
            self._record_id_by_chunk_id[str(chunk_id)] = rid
        return True

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        return self._by_record_id.get(record_id)

    def get_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        rid = self._record_id_by_chunk_id.get(chunk_id)
        return self._by_record_id.get(rid) if rid else None

    def get_many(self, record_ids: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rid in record_ids:
            rec = self._by_record_id.get(rid)
            if rec is not None:
                out.append(rec)
        return out

    def all_record_ids(self) -> List[str]:
        return list(self._by_record_id.keys())


# ---------------------------------------------------------------------------
# Snippet helpers for canonical processed_text_record
# ---------------------------------------------------------------------------

def select_processed_snippet(rec: Optional[Dict[str, Any]], prefer: str = "raw_text") -> str:
    if not rec:
        return ""
    enrichment = rec.get("enrichment") or {}

    if prefer == "summary":
        rs = enrichment.get("retrieval_summary_json") or {}
        scope = rs.get("scope")
        if isinstance(scope, str) and scope.strip():
            return scope.strip()

    if prefer == "raw_text":
        raw = enrichment.get("raw_text")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    if prefer in {"stage5", "raw_text"}:
        stage5 = enrichment.get("stage5_causal_condition") or {}
        causals = stage5.get("extracted_causal_statements") or []
        if causals:
            row = causals[0]
            cause = str(row.get("cause_text") or "").strip()
            connector = str(row.get("connector") or "").strip()
            effect = str(row.get("effect_text") or "").strip()
            sent = str(row.get("sentence_text") or row.get("sentence") or "").strip()
            if sent:
                return sent
            text = " ".join(x for x in [cause, connector, effect] if x).strip()
            if text:
                return text

        cond = stage5.get("condition_state") or {}
        for key in ("as_found", "as_left"):
            value = cond.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    rs = enrichment.get("retrieval_summary_json") or {}
    scope = rs.get("scope")
    if isinstance(scope, str) and scope.strip():
        return scope.strip()

    emb = rec.get("embedding_text")
    if isinstance(emb, str) and emb.strip():
        return emb.strip()
    return ""
