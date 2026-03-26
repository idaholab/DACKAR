from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def find_enriched_jsonl_files(root: Path, globs: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for pattern in globs:
        for path in root.glob(pattern):
            if path.is_file() and path.suffix.lower() == ".jsonl":
                rp = path.resolve()
                if rp not in seen:
                    seen.add(rp)
                    found.append(path)
    return sorted(found)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _normalize_doc_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    raw = str(value).strip().upper()
    mapping = {
        "CONDITION_REPORT": "CR",
        "CORRECTIVE_ACTION_PROGRAM": "CR",
        "WORK_ORDER": "WO",
        "STANDARD_OPERATING_PROCEDURE": "SOP",
        "ENGINEERING_CHANGE": "ECA",
        "MAINTENANCE_RULE": "MR",
    }
    return mapping.get(raw, raw)


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _pick_first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None

def _uniq_str(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _extract_processed_text_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enriched JSONL rows are chunk rows; the canonical processed record is nested
    under row['processed_text_record'].
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ptr = row.get("processed_text_record")
        if isinstance(ptr, dict) and ptr.get("record_id"):
            out.append(ptr)
        elif row.get("record_id") and row.get("metadata") and row.get("provenance"):
            # tolerate direct processed_text_record rows too
            out.append(row)
    return out


def derive_document_record(records: Sequence[Dict[str, Any]], source_path: Path) -> Dict[str, Any]:
    first = records[0]
    metadata = first.get("metadata") or {}
    provenance = first.get("provenance") or {}
    equipment_ids = _uniq_str(sorted({
        eq_id
        for rec in records
        for eq_id in _ensure_list((rec.get("metadata") or {}).get("equipment_ids"))
        if eq_id
    }))
    component_names = _uniq_str(sorted({
        comp
        for rec in records
        for comp in _ensure_list((rec.get("metadata") or {}).get("component_names"))
        if comp
    }))
    fm_refs = _uniq_str(sorted({
        fm
        for rec in records
        for fm in (
            _ensure_list((rec.get("metadata") or {}).get("mechanisms")) +
            _ensure_list((rec.get("metadata") or {}).get("failure_outcomes"))
        )
        if fm
    }))

    doc_id = _pick_first(
        first.get("doc_id"),
        metadata.get("doc_id"),
        provenance.get("doc_id"),
        source_path.stem,
    )
    doc_type = _normalize_doc_type(_pick_first(
        first.get("doc_type"),
        metadata.get("doc_type"),
        provenance.get("doc_type"),
    ))

    doc = {
        "doc_id": str(doc_id),
        "doc_type": doc_type or "UNKNOWN",
        "title": _pick_first(metadata.get("title"), provenance.get("title"), source_path.name),
        "equipment_ids": equipment_ids,
        "component_refs": [{"component_id": c} for c in component_names],
        "failure_mode_refs": [{"fm_id": fm} for fm in fm_refs],
        "authority_level": metadata.get("authority_level"),
        "source_file": _pick_first(
            (first.get("source") or {}).get("fullpath"),
            provenance.get("source_file"),
            source_path,
        ),
    }
    return {k: v for k, v in doc.items() if v not in (None, [], {})}


def load_processed_records_from_output(root: Path, globs: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
    files = find_enriched_jsonl_files(root, globs)
    if not files:
        raise FileNotFoundError(f"No enriched JSONL files found under {root}")

    processed_text_records: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []

    for path in files:
        rows = load_jsonl(path)
        if not rows:
            continue
        ptrs = _extract_processed_text_records(rows)
        if not ptrs:
            continue
        processed_text_records.extend(ptrs)
        documents.append(derive_document_record(ptrs, path))

    return documents, processed_text_records, files
