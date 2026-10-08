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

# Alias layer for load_from_fixtures_dir: basename on disk -> bundle key, used
# only when a file's name differs from the key we want. Files not listed here
# are keyed by their filename stem, so new artifacts appear without editing this
# map. (Every current entry is stem==key and therefore redundant, kept as
# documentation of the expected fixture set.)
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
    Merge every ``*.json`` in a directory into one bundle, keyed by filename.

    The directory may be a hand-built fixtures folder or a real orchestrator run
    folder (``ArtifactStore`` writes one ``<artifact_name>.json`` per artifact),
    so scanning the directory surfaces whatever artifacts are actually present
    rather than a fixed allow-list. ``_FIXTURE_FILE_MAP`` is consulted as an
    alias layer: a basename it names maps to that key, otherwise the filename
    stem is the key. Missing files are simply absent; never raises for them.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(directory)

    bundle: JsonDict = {}
    for fp in sorted(root.glob("*.json")):
        if not fp.is_file():
            continue
        key = _FIXTURE_FILE_MAP.get(fp.name, fp.stem)
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


def merge_bundles(primary: JsonDict, supplemental: JsonDict) -> JsonDict:
    """
    Return *primary* with any keys it lacks filled in from *supplemental*.

    Primary wins on every shared key, so a complete run bundle is never
    overwritten; the supplement only contributes artifacts the primary is
    missing (e.g. the raw ``event`` / ``telemetry_summary`` inputs that a
    ``v32_full_result.json`` does not carry, or the generated artifacts a
    hand-built fixtures folder lacks). This lets one view combine the data
    provided as input with the data produced by every pipeline stage.
    """
    merged = dict(primary)
    for key, value in supplemental.items():
        merged.setdefault(key, value)
    return merged


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
