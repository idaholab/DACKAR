"""
Load RCA artifact bundles from a full-result JSON file or a fixtures directory.

Run the Streamlit app from this directory so ``import loader`` resolves::

    cd DACKAR/src/dackar/RCA/viz && streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

JsonDict = Dict[str, Any]

# basename on disk -> key in the unified ArtifactBundle
_FIXTURE_FILE_MAP: Dict[str, str] = {
    "event.json": "event",
    "telemetry_summary.json": "telemetry_summary",
    "kg_context.json": "kg_context",
    "tskr_patterns.json": "tskr_patterns",
    "causality_candidates.json": "causality_candidates",
    "causality_candidates_pre_refine.json": "causality_candidates_pre_refine",
    "evidence_bundle.json": "evidence_bundle",
    "operational_context.json": "operational_context",
    "pm_compliance.json": "pm_compliance",
    "ishikawa_matrix.json": "ishikawa_matrix",
    "rca_card.json": "rca_card",
    "run_manifest.json": "run_manifest",
    "run_context.json": "run_context",
    "input_validation.json": "input_validation",
    "output_validation.json": "output_validation",
    "evidence_store_rows.json": "evidence_store_rows",
}


def _allowed_path(path: Path) -> bool:
    raw = os.environ.get("RCA_VIZ_ALLOWED_ROOTS", "").strip()
    if not raw:
        return True
    try:
        resolved = path.resolve()
    except OSError:
        return False
    roots = [Path(p.strip()).resolve() for p in raw.split(os.pathsep) if p.strip()]
    return any(str(resolved).startswith(str(root)) for root in roots)


def _read_json(path: Path) -> JsonDict:
    if not _allowed_path(path):
        raise PermissionError(
            f"Path not under RCA_VIZ_ALLOWED_ROOTS: {path}. "
            "Unset the env var to allow any path (local dev only)."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def detect_input_mode(path: str) -> Literal["full_result", "fixtures_dir"]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.is_dir():
        return "fixtures_dir"
    if p.suffix.lower() == ".json":
        return "full_result"
    raise ValueError(f"Unsupported path type (expected directory or .json file): {path}")


def load_from_full_result(path: str) -> JsonDict:
    """Load a single JSON object (e.g. v32_full_result.json). Keys pass through unchanged."""
    fp = Path(path)
    data = _read_json(fp)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at root, got {type(data).__name__}")
    return data


def load_from_fixtures_dir(directory: str) -> JsonDict:
    """
    Merge individual fixture files into one bundle.
    Missing files are omitted (key absent); never raises for missing optional files.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(directory)

    bundle: JsonDict = {}
    for name, key in _FIXTURE_FILE_MAP.items():
        fp = root / name
        if not fp.is_file():
            continue
        try:
            payload = _read_json(fp)
        except (json.JSONDecodeError, OSError) as exc:
            bundle[key] = None
            bundle[f"{key}__load_error"] = str(exc)
            continue
        bundle[key] = payload

    if not bundle:
        for child in sorted(root.iterdir()):
            if child.is_dir():
                nested = load_from_fixtures_dir(str(child))
                if nested:
                    return nested

    return bundle


def load_artifacts(path: str) -> JsonDict:
    """Auto-detect file vs directory and load."""
    mode = detect_input_mode(path)
    if mode == "full_result":
        return load_from_full_result(path)
    return load_from_fixtures_dir(path)


def load_pre_refine_causality(path: str) -> Optional[JsonDict]:
    """
    Load optional pre-refine causality artifact.

    Accepts either a bare ``causality_candidates`` object (has ``candidates``)
    or a full bundle (uses ``causality_candidates`` key if present).
    """
    if not path or not str(path).strip():
        return None
    fp = Path(path.strip())
    if not fp.is_file():
        raise FileNotFoundError(path)
    data = _read_json(fp)
    if isinstance(data.get("candidates"), list) and "run_context" not in data:
        return data
    if isinstance(data.get("causality_candidates"), dict):
        return data["causality_candidates"]
    raise ValueError(
        "Pre-refine file must be causality_candidates.json shape "
        "or a full bundle containing causality_candidates"
    )


def list_bundle_keys(bundle: JsonDict) -> List[str]:
    keys = [k for k in bundle if not k.endswith("__load_error")]
    return sorted(keys)
