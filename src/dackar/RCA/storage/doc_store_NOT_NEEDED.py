# doc_store.py
from __future__ import annotations

import json
import os
import logging
from typing import Dict, Any, List, Optional

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collapse_ws(s: Optional[str]) -> str:
    """Collapse all whitespace runs in *s* to a single space and strip ends."""
    return " ".join((s or "").split())


def _iter_jsonl(jsonl_path: str):
    """
    Yield parsed JSON objects from a JSONL file, skipping blank lines and
    lines that fail to parse.

    Args:
        jsonl_path: Absolute or relative path to a ``.jsonl`` file.

    Yields:
        dict: One parsed record per non-empty line.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("doc_store: JSON parse error in %s line %d: %s", jsonl_path, lineno, exc)


def _parse_table_payload(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract normalised ``{columns, rows}`` from a Table record.

    Handles two storage layouts:
    - Direct: ``rec["columns"]`` (list) + ``rec["rows"]`` (list of lists).
    - Serialised: ``rec["data"]`` is a JSON string ``{"columns": [...], "rows": [...]}``.

    Args:
        rec: A raw chunk record dict of type ``"Table"``.

    Returns:
        dict: ``{"columns": List[str], "rows": List[List[str]]}``
              Empty lists when neither layout is detected.
    """
    cols = rec.get("columns")
    rows = rec.get("rows")
    if isinstance(cols, list) and isinstance(rows, list):
        return {"columns": cols, "rows": rows}

    data = rec.get("data")
    if isinstance(data, str):
        try:
            obj = json.loads(data)
            if isinstance(obj, dict):
                cols = obj.get("columns") or []
                rows = obj.get("rows") or []
                if isinstance(cols, list) and isinstance(rows, list):
                    return {"columns": cols, "rows": rows}
        except Exception:
            pass

    return {"columns": [], "rows": []}


# ---------------------------------------------------------------------------
# Per-type snippet builders
# ---------------------------------------------------------------------------

def get_text_chunk_content(rec: Dict[str, Any], prefer: str = "text") -> str:
    """
    Return the best available textual content for a ``TextChunk`` record.

    The preference order is controlled by *prefer*:
    - ``"summary"``: tries ``summary`` first, falls back to ``text``.
    - anything else:  tries ``text`` first, falls back to ``summary``.

    Note:
        Summaries are currently disabled in the pipeline; the fallback to
        ``text`` ensures this function is safe to call regardless.

    Args:
        rec:    A raw chunk record dict of type ``"TextChunk"``.
        prefer: Which field to try first (``"summary"`` or ``"text"``).

    Returns:
        str: Whitespace-collapsed textual content, or ``""`` if unavailable.
    """
    if rec.get("type") != "TextChunk":
        return ""
    if prefer == "summary":
        return _collapse_ws(rec.get("summary") or rec.get("text") or "")
    return _collapse_ws(rec.get("text") or rec.get("summary") or "")


def get_table_snippet(rec: Dict[str, Any], max_rows: int = 6) -> str:
    """
    Build a human-readable snippet for a ``Table`` record suitable for
    display in a ``ContextPack`` result.

    Always includes caption (if present), page, column headers, and up to
    *max_rows* data rows.

    Args:
        rec:      A raw chunk record dict of type ``"Table"``.
        max_rows: Maximum number of data rows to include in the snippet.

    Returns:
        str: Pipe-delimited text snippet, or ``""`` if the record is not a Table.
    """
    if rec.get("type") != "Table":
        return ""
    payload = _parse_table_payload(rec)
    cols = [str(c) for c in (payload.get("columns") or [])]
    rows = [[str(x) for x in r] for r in (payload.get("rows") or [])]

    caption = _collapse_ws(rec.get("caption") or "")
    page    = rec.get("page") or rec.get("page_number")
    header  = " | ".join(cols) if cols else ""
    body    = "\n".join(" | ".join(r) for r in rows[:max_rows]) if rows else ""

    parts: List[str] = []
    if caption:        parts.append(f"Table: {caption}")
    if page is not None: parts.append(f"(page {page})")
    if header:         parts.append(header)
    if body:           parts.append(body)
    return "\n".join(parts).strip()


def get_figure_caption(rec: Dict[str, Any]) -> str:
    """
    Build a human-readable caption string for a ``Figure`` record.

    Combines the ``caption`` (or ``alt``) text, page number, and filename
    into a single line for display or embedding.

    Args:
        rec: A raw chunk record dict of type ``"Figure"``.

    Returns:
        str: Caption string, or ``""`` if the record is not a Figure.
    """
    if rec.get("type") != "Figure":
        return ""
    caption = _collapse_ws(rec.get("caption") or rec.get("alt") or "")
    path    = rec.get("path") or rec.get("image_path")
    page    = rec.get("page") or rec.get("page_number")

    fname = ""
    if isinstance(path, str) and path.strip():
        fname = os.path.basename(path.strip())

    parts: List[str] = []
    if caption:          parts.append(f"Figure: {caption}")
    if page is not None: parts.append(f"(page {page})")
    if fname:            parts.append(f"[{fname}]")
    elif path:           parts.append(f"[{path}]")
    return " ".join(parts).strip()


def select_snippet(
    rec: Dict[str, Any],
    prefer_text: str = "text",
    max_table_rows: int = 6,
) -> str:
    """
    Dispatch to the appropriate snippet builder for any recognised chunk type.

    Supported types: ``"TextChunk"``, ``"Table"``, ``"Figure"``.
    Unknown types fall back to the ``text`` field so future chunk types
    (e.g. ``"Requirement"``) are handled gracefully rather than silently
    returning an empty string.

    Args:
        rec:            A raw chunk record dict.
        prefer_text:    Field preference forwarded to ``get_text_chunk_content``.
        max_table_rows: Row limit forwarded to ``get_table_snippet``.

    Returns:
        str: Best available snippet text.
    """
    rtype = rec.get("type")
    if rtype == "TextChunk":
        return get_text_chunk_content(rec, prefer=prefer_text)
    if rtype == "Table":
        return get_table_snippet(rec, max_rows=max_table_rows)
    if rtype == "Figure":
        return get_figure_caption(rec)
    # Graceful fallback for unknown/future chunk types
    return _collapse_ws(rec.get("text") or rec.get("content") or "")


# ---------------------------------------------------------------------------
# DocStore — corpus-level in-memory index  (fixes Issues 6 & 16)
# ---------------------------------------------------------------------------

class DocStore:
    """
    Corpus-level in-memory index of chunk records, supporting O(1) lookup by
    ``chunk_id``.

    Replaces the previous ``load_records_for_ids`` pattern which performed a
    full linear JSONL scan on every retrieval call — a critical latency issue
    for corpora with many documents.

    Usage::

        store = DocStore(jsonl_paths)   # build index once at startup
        rec   = store.get("chunk:abc:sec3:p001")
        recs  = store.get_many(["chunk:abc:sec3", "chunk:xyz:sec1"])

    Args:
        jsonl_paths: One or more paths to ``*_chunks.jsonl`` files produced
                     by ``mdParser``.  All files are indexed into a single
                     shared dict so cross-document hydration works correctly.
    """

    def __init__(self, jsonl_paths: List[str]) -> None:
        self._index: Dict[str, Dict[str, Any]] = {}
        for path in jsonl_paths:
            self._load(path)
        LOGGER.info("DocStore: indexed %d chunks from %d file(s).", len(self._index), len(jsonl_paths))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load(self, jsonl_path: str) -> None:
        """
        Parse *jsonl_path* and insert every record into the index.

        Records without a ``chunk_id`` are skipped with a warning.
        Duplicate ``chunk_id`` values (e.g. from re-ingest) overwrite the
        earlier entry so the index always reflects the latest version.

        Args:
            jsonl_path: Path to a ``*_chunks.jsonl`` file.
        """
        if not os.path.exists(jsonl_path):
            LOGGER.warning("DocStore: JSONL file not found, skipping: %s", jsonl_path)
            return
        count = 0
        for rec in _iter_jsonl(jsonl_path):
            cid = rec.get("chunk_id")
            if not cid:
                continue
            self._index[cid] = rec
            count += 1
        LOGGER.debug("DocStore: loaded %d records from %s", count, jsonl_path)

    def add_jsonl(self, jsonl_path: str) -> int:
        """
        Incrementally add (or update) records from a new JSONL file.

        Useful when new documents are ingested after initial startup without
        rebuilding the entire DocStore.

        Args:
            jsonl_path: Path to a ``*_chunks.jsonl`` file.

        Returns:
            int: Number of records added or updated.
        """
        before = len(self._index)
        self._load(jsonl_path)
        return len(self._index) - before

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single record by ``chunk_id``.

        Args:
            chunk_id: The unique chunk identifier (e.g. ``"chunk:abc123:sec3:p001"``).

        Returns:
            dict | None: The raw chunk record, or ``None`` if not found.
        """
        return self._index.get(chunk_id)

    def get_many(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple records by ``chunk_id`` in O(n) time.

        Missing IDs are silently omitted from the result.  The return order
        matches *chunk_ids* (skipping misses), not insertion order.

        Args:
            chunk_ids: List of chunk identifier strings.

        Returns:
            List[dict]: Found records in the same order as *chunk_ids*.
        """
        out: List[Dict[str, Any]] = []
        missing: List[str] = []
        for cid in chunk_ids:
            rec = self._index.get(cid)
            if rec is not None:
                out.append(rec)
            else:
                missing.append(cid)
        if missing:
            LOGGER.debug("DocStore.get_many: %d id(s) not found: %s", len(missing), missing[:5])
        return out

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._index