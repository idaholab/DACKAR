"""
doc_ref_extractor.py
─────────────────────────────────────────────────────────────────────────────
Extract document cross-reference IDs from nuclear plant text (CR, WO, ECA,
LER, GL, SOP, etc.).

Plant-specific flexibility
──────────────────────────
Every plant uses its own naming conventions.  This extractor is driven by a
JSON *plant profile* (see ``ner/data/plant_profiles/default_plant_profile.json``)
that enumerates document types, their prefix strings, and their regex patterns.

To adapt to a new plant:
  1. Copy ``default_plant_profile.json`` → ``<your-plant-id>_profile.json``
  2. Add / remove entries in ``doc_ref_types``, adjusting prefixes and patterns.
  3. Pass the file path to ``load_doc_ref_profile()`` or directly to
     ``extract_doc_refs(text, profile_path=...)``.

The default profile covers the most common US nuclear site conventions:
  CR / CAP, WO / PM, ECA / EC / DCN, SOP / OP / MP / SP / EOP / AOP,
  LER, GL, IN, BUL, OE / SER / IER, NCR / PER / AR.

Output
──────
Each extracted reference is a ``DocRef`` namedtuple:
  doc_type  - canonical type string (e.g. "CR", "WO", "GL")
  label     - NER label string (e.g. "doc_ref_cr") for schema routing
  raw       - original matched text (before normalization)
  norm      - normalized ID (uppercase, collapsed separators)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

_DEFAULT_PROFILE_PATH = Path(__file__).parent / "data" / "plant_profiles" / "default_plant_profile.json"

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class DocRef(NamedTuple):
    """A single extracted document cross-reference."""
    doc_type: str    # canonical type: "CR", "WO", "GL", …
    label: str       # NER label: "doc_ref_cr", "doc_ref_wo", …
    raw: str         # verbatim matched text
    norm: str        # normalised ID (uppercase, canonical separator)


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

def load_doc_ref_profile(profile_path: Optional[str | Path] = None) -> Dict:
    """Load a plant profile JSON file and return the parsed dict.

    Args:
        profile_path: Path to the plant profile JSON.  When ``None``, the
            bundled ``default_plant_profile.json`` is used.

    Returns:
        Parsed profile dict.

    Raises:
        FileNotFoundError: If *profile_path* is given but does not exist.
        ValueError: If the file is not valid JSON.
    """
    path = Path(profile_path) if profile_path else _DEFAULT_PROFILE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Plant profile not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in plant profile {path}: {exc}") from exc


def _compile_profile(profile: Dict) -> List[Tuple[str, str, List[re.Pattern]]]:
    """Return list of (doc_type, label, [compiled_patterns]) from profile.

    Patterns are compiled case-insensitively.
    """
    result = []
    for entry in profile.get("doc_ref_types", []):
        doc_type = entry.get("type", "UNKNOWN").upper()
        label = entry.get("label", f"doc_ref_{doc_type.lower()}")
        raw_patterns = entry.get("patterns", [])
        compiled = [re.compile(p, re.IGNORECASE) for p in raw_patterns]
        if compiled:
            result.append((doc_type, label, compiled))
    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_doc_ref(raw: str) -> str:
    """Return a canonical document reference string.

    - Uppercase
    - Collapse any internal whitespace to a single hyphen
    - Collapse multiple consecutive hyphens
    - Strip leading/trailing punctuation
    """
    norm = raw.strip().upper()
    norm = re.sub(r"\s+", "-", norm)
    norm = re.sub(r"-{2,}", "-", norm)
    norm = norm.strip("-")
    return norm


# ---------------------------------------------------------------------------
# False positive filtering
# ---------------------------------------------------------------------------

def _make_fp_checker(profile: Dict):
    """Return a callable that returns True when a (doc_type, norm) pair is
    a false positive and should be dropped."""
    fp = profile.get("false_positive_filters", {})
    min_year = int(fp.get("min_year", 1960))
    max_year = int(fp.get("max_year", 2035))
    excluded = {p.upper() for p in fp.get("excluded_prefixes_from_doc_ref", [])}

    _year_re = re.compile(r"(?:19|20)\d{2}")

    def is_fp(doc_type: str, norm: str) -> bool:
        prefix = norm.split("-")[0] if "-" in norm else norm[:3]
        if prefix in excluded:
            return True
        # Year range check: any year fragment outside [min_year, max_year]
        for yr_str in _year_re.findall(norm):
            yr = int(yr_str)
            if yr < min_year or yr > max_year:
                return True
        return False

    return is_fp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_doc_refs(
    text: str,
    *,
    profile_path: Optional[str | Path] = None,
    profile: Optional[Dict] = None,
    normalize: bool = True,
    unique: bool = True,
    max_refs: int = 200,
) -> List[DocRef]:
    """Extract document cross-reference IDs from *text*.

    Args:
        text: Arbitrary chunk or document text.
        profile_path: Path to a plant profile JSON override.  Ignored when
            *profile* is supplied directly.
        profile: Pre-loaded plant profile dict (takes priority over
            *profile_path*).  Pass this when calling the function many times
            on the same plant to avoid repeated file I/O.
        normalize: Normalize matched IDs (uppercase, canonical hyphens).
        unique: Return each normalised ID at most once.
        max_refs: Safety cap on the number of refs returned.

    Returns:
        List of :class:`DocRef` namedtuples in match order.
    """
    if not text:
        return []

    if profile is None:
        profile = load_doc_ref_profile(profile_path)

    compiled_types = _compile_profile(profile)
    is_fp = _make_fp_checker(profile)

    results: List[DocRef] = []
    seen: set = set()

    for doc_type, label, patterns in compiled_types:
        for rx in patterns:
            for m in rx.finditer(text):
                raw = m.group(0)
                norm = _normalize_doc_ref(raw) if normalize else raw.upper()

                if is_fp(doc_type, norm):
                    continue

                key = (doc_type, norm)
                if unique and key in seen:
                    continue
                seen.add(key)

                results.append(DocRef(doc_type=doc_type, label=label, raw=raw, norm=norm))

                if len(results) >= max_refs:
                    return results

    return results


def extract_doc_ref_ids(
    text: str,
    **kwargs,
) -> List[str]:
    """Convenience wrapper — returns only the normalised ID strings.

    Suitable for direct assignment to ``NERSeed.doc_refs``.

    Args:
        text: Source text.
        **kwargs: Forwarded to :func:`extract_doc_refs`.

    Returns:
        List of normalised document reference strings.
    """
    return [r.norm for r in extract_doc_refs(text, **kwargs)]
