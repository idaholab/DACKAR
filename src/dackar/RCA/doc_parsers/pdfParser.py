
# pdfParser.py
# -------------------------------------------------------------------------
# PDF -> Text (Markdown), Figures (images), Tables (normalized JSON)
# Produces a document index with rich metadata and provenance.
#
# Optional Enhancements implemented:
#  - Robust logging & error handling
#  - Normalized table outputs to JSON (for both marker and pdfplumber modes)
#  - Page-aware figure indexing (parsing Marker filenames)
#  - Document index return + write to disk
# -------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import logging
import datetime
from typing import Dict, List, Any, Optional
import hashlib

# External tools
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.converters.table import TableConverter
except Exception:
    # Allow module import for environments without marker; runtime checks will guard usage
    PdfConverter = None
    TableConverter = None
    create_model_dict = None
    text_from_rendered = None

import pdfplumber
import pandas as pd

from pathlib import Path

# ------------------------------
# Logging configuration
# ------------------------------
LOGGER = logging.getLogger("pdfParser")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

# Canonical filename prefix patterns → doc_type
_DOC_TYPE_PATTERNS = [
    (re.compile(r"\bCR\b",      re.I), "CR"),
    (re.compile(r"\bWO\b",      re.I), "WO"),
    (re.compile(r"\bSOP\b",     re.I), "SOP"),
    (re.compile(r"\bECA\b",     re.I), "ECA"),
    (re.compile(r"\bMR\b",      re.I), "WO"),   # Maintenance Request → WO
    (re.compile(r"\bPM\b",      re.I), "WO"),   # PM task record → WO
    (re.compile(r"procedure",   re.I), "SOP"),
    (re.compile(r"work.?order", re.I), "WO"),
    (re.compile(r"condition.?report", re.I), "CR"),
    (re.compile(r"engineering.?change", re.I), "ECA"),
]

_EARLY_TEXT_PATTERNS = [
     (re.compile(r"condition\s+report",    re.I), "CR"),
    (re.compile(r"apparent\s+cause|immediate\s+actions?|condition\s+description", re.I), "CR"),
     (re.compile(r"work\s+order",          re.I), "WO"),
    (re.compile(r"as[- ]found|as[- ]left|work\s+performed", re.I), "WO"),
     (re.compile(r"standard\s+operating\s+procedure|SOP\b", re.I), "SOP"),
    (re.compile(r"preconditions?|procedure\s+steps?|references", re.I), "SOP"),
     (re.compile(r"engineering\s+change\s+analysis|ECA\b",  re.I), "ECA"),
    (re.compile(r"causal\s+factors?|evidence\s*/\s*data|recommended\s+followups?", re.I), "ECA"),
 ]

def infer_doc_type(filename: str, early_text: str = "") -> str:
    """
    Infer doc_type from filename first, then first ~500 chars of text.
    Returns one of: CR | WO | SOP | ECA | OTHER
    """
    stem = Path(filename).stem if filename else ""
    for pat, dtype in _DOC_TYPE_PATTERNS:
        if pat.search(stem):
            return dtype
    for pat, dtype in _EARLY_TEXT_PATTERNS:
        if pat.search(early_text[:500]):
            return dtype
    return "OTHER"


# ------------------------------
# Helpers
# ------------------------------
def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _clean_table_cell(x: Any) -> str:
    text = "" if x is None else str(x)
    text = re.sub(r'[\u200b\u200c\u200d\u00ad\u2060]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([A-Za-z])\.([A-Za-z])', r'\1\2', text)
    text = re.sub(r'^[\.\,\;\:\-]+\s*', '', text)
    if re.fullmatch(r'.*maintenance', text, flags=re.IGNORECASE):
        text = "Maintenance"
    elif re.fullmatch(r'.*engineering', text, flags=re.IGNORECASE):
        text = "Engineering"
    return text


def _clean_table_obj(t: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(t)
    out["columns"] = [_clean_table_cell(c) for c in (t.get("columns") or [])]
    out["rows"] = [[_clean_table_cell(c) for c in row] for row in (t.get("rows") or [])]
    if out.get("caption") is not None:
        out["caption"] = _clean_table_cell(out["caption"])
    return out

def _estimate_text_extraction_quality(markdown_text: str, page_count: Optional[int]) -> Dict[str, Any]:
    """
    Lightweight extraction-quality heuristic for Stage 1 provenance.
    """
    text = (markdown_text or "").strip()
    char_count = len(text)
    line_count = len(text.splitlines()) if text else 0
    alpha_count = sum(1 for ch in text if ch.isalpha())
    alpha_ratio = (alpha_count / char_count) if char_count > 0 else 0.0

    chars_per_page = None
    if page_count and page_count > 0:
        chars_per_page = round(char_count / page_count, 1)

    image_only_like = bool(page_count and char_count < max(500, 150 * page_count))
    low_confidence_parse = bool(char_count < 300 or alpha_ratio < 0.35)

    return {
        "char_count": char_count,
        "line_count": line_count,
        "alpha_ratio": round(alpha_ratio, 3),
        "chars_per_page": chars_per_page,
        "image_only_like": image_only_like,
        "low_confidence_parse": low_confidence_parse,
        "ocr_used": False,
    }

def get_file_name_no_extension(path: str) -> str:
    """Return filename without extension from a path."""
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_marker_image_key(image_key: str) -> Dict[str, Any]:
    """
    Marker images are commonly saved with names like '_page_12_Picture_3.jpeg'.
    Extract page/index if present; otherwise return None values.
    """
    # Accept keys like '_page_12_Picture_3.jpeg' or 'someprefix_page_12_Picture_3.jpeg'
    m = re.search(r"_page_(\d+)_Picture_(\d+)\.(?:jpg|jpeg|png)$", image_key, re.IGNORECASE)
    page = int(m.group(1)) if m else None
    index = int(m.group(2)) if m else None
    return {"page": page, "index": index}


def _normalize_table_markdown_to_json(md_text: str) -> List[Dict[str, Any]]:
    """
    Convert Marker-produced Markdown tables into a normalized JSON structure:
    [
      { "columns": [...], "rows": [[...], ...], "caption": None, "page": None }
    ]
    Assumes tables are separated in MD by blank lines and follow pipe '|' syntax.
    """
    tables: List[Dict[str, Any]] = []
    lines = [ln.rstrip("\n") for ln in md_text.splitlines()]

    current_table: Optional[Dict[str, Any]] = None
    in_table = False

    def finish_table():
        nonlocal current_table
        if current_table and current_table.get("columns") and current_table.get("rows") is not None:
            tables.append(current_table)
        current_table = None

    for i, line in enumerate(lines):
        # Skip empty lines; they can delimit tables
        if not line.strip():
            if in_table:
                in_table = False
                finish_table()
            continue

        # Detect header line (pipes) and a following separator line (---)
        if "|" in line and re.search(r"\|\s*:?[-]{3,}", line) is None:
            # Potential header (we'll confirm by checking next line)
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if re.search(r"\|\s*:?[-]{3,}", next_line):
                # Start a new table
                in_table = True
                header_cells = [c.strip() for c in line.strip().split("|")]
                # Remove empty cells caused by leading/trailing pipes
                header_cells = [c for c in header_cells if c != ""]
                current_table = {"columns": header_cells, "rows": [], "caption": None, "page": None}
                continue

        # Row lines inside a table
        if in_table and "|" in line:
            row_cells = [c.strip() for c in line.strip().split("|")]
            row_cells = [c for c in row_cells if c != ""]
            # Some rows can be alignment lines like :---:, skip those
            if any(re.match(r"^:?-{3,}:?$", c) for c in row_cells):
                continue
            current_table["rows"].append(row_cells)
        else:
            # Non-table content; if a table was open and we encounter non-pipe content, close the table
            if in_table:
                in_table = False
                finish_table()
            # Could parse captions here if marker emits them, but often it doesn't. Leave as None.
            continue

    # Finish last table
    if in_table:
        finish_table()

    return tables


def _tables_from_pdfplumber(filename: str) -> List[Dict[str, Any]]:
    """Extract tables using pdfplumber and normalize to JSON objects."""
    table_objs: List[Dict[str, Any]] = []
    with pdfplumber.open(filename) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for tbl in tables or []:
                if not tbl or len(tbl) < 2:
                    # Need at least header + one row
                    LOGGER.debug("Skipping a small/invalid table on page %d", page_num)
                    continue
                header = [str(x).strip() for x in tbl[0]]
                rows = [[str(x).strip() for x in row] for row in tbl[1:]]
                table_objs.append({
                    "columns": header,
                    "rows": rows,
                    "caption": None,
                    "page": page_num
                })
    return table_objs


# ------------------------------
# Main API
# ------------------------------
def pdfParser(
    home_folder: Optional[str] = None,
    red_filepath: Optional[str] = None,
    destination_folder: Optional[str] = None,
    text2markdown: str = "marker",
    tableParser: str = "marker",
    classification: str = "internal",
    ingest_id: Optional[str] = None,          # NEW
    source_path: Optional[str] = None,        # NEW (relative path)
) -> Dict[str, Any]:
    """
    Parse a PDF file into:
      - text markdown
      - figures (images)
      - tables (normalized JSON)
    Emit:
      - document_index.json (paths + metadata)
      - text.md
      - tables.json
      - figures/*.jpeg
      - metadata.json (Marker-render metadata when available)
    Return:
      - index dict mirroring document_index.json

    Parameters
    ----------
    home_folder : Optional[str]
        Base folder for relative path `red_filepath`.
    red_filepath : Optional[str]
        File path relative to `home_folder` (or absolute if `home_folder` is None).
    destination_folder : Optional[str]
        Root destination for parsed outputs.
    text2markdown : str
        Only 'marker' supported for now.
    tableParser : str
        'marker' or 'pdfplumber'.
    classification : str
        Data classification tag.

    Returns
    -------
    Dict[str, Any]
        Document index with artifact paths and rich metadata.

    Raises
    ------
    FileNotFoundError
        If input or destination paths are invalid.
    ValueError
        For unsupported options.
    """

    # Resolve paths
    if red_filepath is None:
        raise FileNotFoundError("red_filepath must be provided.")
    fullpath = os.path.join(home_folder, red_filepath) if home_folder else red_filepath
    if not os.path.exists(fullpath):
        print(os.getcwd())
        raise FileNotFoundError(f"PDF file not found: {fullpath}")
    if not source_path:
        source_path = red_filepath
    doc_name = os.path.basename(source_path)

    try:
        with open(fullpath, "rb") as _fh:
            _file_bytes = _fh.read()
        doc_id = hashlib.sha256(_file_bytes).hexdigest()[:12]
        # doc_version reuses the same hash; set it here so it's available early
        doc_version = hashlib.sha256(_file_bytes).hexdigest()
    except Exception:
        doc_id = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]
        doc_version = None

    if destination_folder is None:
        raise FileNotFoundError("destination_folder must be defined.")

    # Prepare output folders
    filename = get_file_name_no_extension(red_filepath)
    file_folder = os.path.dirname(red_filepath) or ""
    # Guard: ensure file_folder is never absolute (prevents escaping destination_folder)
    if os.path.isabs(file_folder):
        # collapse to basename (we intentionally drop parent pieces to keep outputs inside destination_folder)
        file_folder = os.path.basename(file_folder) or ""
    # Build a safe data dump folder rooted under destination_folder
    data_dump_folder = os.path.join(destination_folder, doc_id)

    textFolder = os.path.join(data_dump_folder, "text")
    figuresFolder = os.path.join(data_dump_folder, "figures")
    tablesFolder = os.path.join(data_dump_folder, "tables")
    metaFolder = os.path.join(data_dump_folder, "meta")
    indexFolder = os.path.join(data_dump_folder, "index")

    for path in [textFolder, figuresFolder, tablesFolder, metaFolder, indexFolder]:
        _ensure_dir(path)

    # Convert text to Markdown via Marker
    markdown_text: str = ""
    images_index: List[Dict[str, Any]] = []
    marker_metadata: Dict[str, Any] = {}
    extraction_method_text = None
    extraction_quality: Dict[str, Any] = {}
    page_count: Optional[int] = None

    if text2markdown != "marker":
        LOGGER.error("text2markdown option not allowed. Allowed: 'marker'.")
        raise ValueError("Unsupported text2markdown option.")
    if PdfConverter is None:
        LOGGER.error("Marker PDF converter is not available in this environment.")
        raise RuntimeError("Marker is required for text conversion.")

    try:
        with pdfplumber.open(fullpath) as pdf:
            page_count = len(pdf.pages)
    except Exception as e:
        LOGGER.warning("Could not determine page count for %s: %s", fullpath, e)

    try:
        LOGGER.info("Rendering PDF to Markdown via Marker: %s", fullpath)
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(fullpath)
        markdown_text, _, images = text_from_rendered(rendered)
        marker_metadata = getattr(rendered, "metadata", {}) or {}
        extraction_method_text = "marker"
        extraction_quality = _estimate_text_extraction_quality(markdown_text, page_count)
        if extraction_quality.get("low_confidence_parse"):
            LOGGER.warning(
                "Low-confidence text extraction detected for %s (chars/page=%s, alpha_ratio=%s).",
                fullpath,
                extraction_quality.get("chars_per_page"),
                extraction_quality.get("alpha_ratio"),
            )
    except Exception as e:
        LOGGER.exception("Marker text conversion failed: %s", e)
        raise

    # Save Markdown text
    md_fileID = filename + ".md"
    text_md_path = os.path.join(textFolder, md_fileID)
    with open(text_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)  

    doc_type = infer_doc_type(doc_name, markdown_text[:500])

    # Save figures and build index (parse page/index from filename)
    try:
        for image_key in images:
            img_path = os.path.join(figuresFolder, image_key)
            images[image_key].save(img_path)
            info = _parse_marker_image_key(image_key)
            images_index.append({
                "path": img_path,
                "name": image_key,
                "page": info.get("page"),
                "figure_index": info.get("index"),
                "caption": None,
                "extraction_confidence": 0.8 if info.get("page") is not None else 0.6,
                "source": "marker_image_render",
            })
    except Exception as e:
        LOGGER.exception("Failed saving images: %s", e)
        # Proceed; images may be optional

    # Extract tables
    tables_json_path = os.path.join(tablesFolder, filename + "_tables.json")
    tables_index: List[Dict[str, Any]] = []
    extraction_method_tables = None
    table_extraction_error: Optional[str] = None

    try:
        if tableParser == "marker":
            if TableConverter is None:
                LOGGER.error("Marker TableConverter is unavailable.")
                raise RuntimeError("Marker is required for 'marker' tableParser.")
            LOGGER.info("Extracting tables via Marker: %s", fullpath)
            table_converter = TableConverter(artifact_dict=create_model_dict())
            rendered_tbl = table_converter(fullpath)
            md_table_text, _, tbl_images = text_from_rendered(rendered_tbl)

            # Normalize table markdown to JSON
            tables_objs = _normalize_table_markdown_to_json(md_table_text)
            extraction_method_tables = "marker"

        elif tableParser == "pdfplumber":
            LOGGER.info("Extracting tables via pdfplumber: %s", fullpath)
            tables_objs = _tables_from_pdfplumber(fullpath)
            extraction_method_tables = "pdfplumber"

        else:
            LOGGER.error("tableParser option not allowed. Allowed: 'marker' or 'pdfplumber'.")
            raise ValueError("Unsupported tableParser option.")

        # Persist tables JSON
        tables_objs = [_clean_table_obj(t) for t in tables_objs]
        # Persist tables JSON
        with open(tables_json_path, "w", encoding="utf-8") as f:
            json.dump(tables_objs, f, indent=2)

        # Build tables index entries
        for idx, t in enumerate(tables_objs, start=1):
            tables_index.append({
                "path": tables_json_path,
                "table_id": f"{filename}_table_{idx}",
                "page": t.get("page"),
                "caption": t.get("caption"),
                "columns": t.get("columns"),
                "rows_count": len(t.get("rows") or []),
                "extraction_method": extraction_method_tables,
            })

    except Exception as e:
        LOGGER.exception("Table extraction failed: %s", e)
        tables_json_path = None
        tables_index = []
        table_extraction_error = str(e)

    # Persist marker metadata
    metadata_path = os.path.join(metaFolder, "metadata.json")
    try:
        # Include parser-level fields for provenance
        marker_metadata = marker_metadata or {}
        marker_metadata.update({
            "extraction_method_text": extraction_method_text,
            "extraction_method_tables": extraction_method_tables,
            "timestamp": _now_iso(),
            "classification": classification,
            "source_fullpath": fullpath,
            "source_relpath": red_filepath,
            "parser_version": "pdfParser-2.1.0",
            "page_count": page_count,
            "extraction_quality": extraction_quality,
        })
        with open(metadata_path, "w", encoding="utf-8") as jf:
            json.dump(marker_metadata, jf, indent=2)
    except Exception as e:
        LOGGER.exception("Failed to write metadata.json: %s", e)


    document_index = {
        "doc_id": doc_id,
        "doc_type": doc_type, 
        "doc_type_source": "filename_or_early_text_inference",                                         
        "authority_level": "mandatory" if doc_type == "SOP" else "informational", 
        "source_path": source_path,
        "doc_name": doc_name,
        "ingest_id": ingest_id,
        "text_md_path": text_md_path,
        "figures": images_index,
        "tables_paths": [tables_json_path] if tables_json_path else [],
        "tables_index": tables_index,
        "metadata_path": metadata_path,
        "source": {"fullpath": fullpath, "relpath": red_filepath},
        "content_hash": doc_version,
        "doc_version": doc_version,
        "page_count": page_count,
        "extraction_quality": extraction_quality,
        "low_confidence_parse": extraction_quality.get("low_confidence_parse", False),
        "image_only_like": extraction_quality.get("image_only_like", False),
        "extraction": {
            "text": extraction_method_text,
            "tables": extraction_method_tables,
            "table_extraction_error": table_extraction_error,
            "table_extraction_succeeded": table_extraction_error is None,
            "tables_detected": len(tables_index),
        },
        "timestamp": _now_iso(),
        "classification": classification,
    }

    index_path = os.path.join(indexFolder, "document_index.json")
    try:
        with open(index_path, "w", encoding="utf-8") as idxf:
            json.dump(document_index, idxf, indent=2)
    except Exception as e:
        LOGGER.exception("Failed to write document_index.json: %s", e)

    LOGGER.info("Parsing complete for: %s", fullpath)
    return document_index


if __name__ == "__main__":
    # Example usage scaffold (golden file test idea)
    # python pdfParser.py /path/to/home rel/path/to/file.pdf /destination/root
    import sys
    if len(sys.argv) < 4:
        print("Usage: python pdfParser.py <home_folder or -> <red_filepath> <destination_folder>")
        sys.exit(1)
    home = None if sys.argv[1] == "-" else sys.argv[1]
    rel = sys.argv[2]
    dest = sys.argv[3]
    idx = pdfParser(home_folder=home, red_filepath=rel, destination_folder=dest)
