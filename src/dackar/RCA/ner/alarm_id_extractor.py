"""
alarm_id_extractor.py
─────────────────────────────────────────────────────────────────────────────
Extract alarm and annunciator IDs from nuclear plant text (ALM, ANN, process
tags, SCRAM/SI/AFAS setpoint conditions, etc.).

Plant-specific flexibility
──────────────────────────
Like ``doc_ref_extractor.py``, this module is driven by the same JSON *plant
profile* (see ``ner/data/plant_profiles/default_plant_profile.json``).  Alarm
patterns live in the ``alarm_patterns`` array; each entry has a ``name``,
``pattern`` (regex string), and ``score`` (0–1 confidence).

False-positive filtering uses the ``false_positive_filters.excluded_prefixes_from_alarm``
list from the same profile, so document-reference prefixes (CR, WO, …) are
never returned as alarm IDs.

Output
──────
Each extracted alarm reference is an ``AlarmRef`` namedtuple:
  pattern_name  - stable pattern identifier (e.g. "short_alarm_id")
  raw           - original matched text (before normalization)
  norm          - normalized ID (uppercase, collapsed separators)
  score         - pattern confidence score (0–1) from the plant profile
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from .doc_ref_extractor import load_doc_ref_profile

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class AlarmRef(NamedTuple):
    """A single extracted alarm or annunciator reference."""
    pattern_name: str   # stable pattern identifier, e.g. "short_alarm_id"
    raw: str            # verbatim matched text
    norm: str           # normalised ID (uppercase, canonical separator)
    score: float        # pattern confidence score from plant profile [0, 1]


# ---------------------------------------------------------------------------
# Profile compilation
# ---------------------------------------------------------------------------

def _compile_alarm_profile(profile: Dict) -> List[tuple]:
    """Return list of (pattern_name, score, compiled_pattern) from profile.

    Patterns are compiled case-insensitively.
    """
    result = []
    for entry in profile.get("alarm_patterns", []):
        name = entry.get("name", "unknown")
        score = float(entry.get("score", 0.5))
        raw_pattern = entry.get("pattern", "")
        if raw_pattern:
            try:
                compiled = re.compile(raw_pattern, re.IGNORECASE)
                result.append((name, score, compiled))
            except re.error:
                pass  # silently skip malformed patterns
    return result


# ---------------------------------------------------------------------------
# False-positive filtering
# ---------------------------------------------------------------------------

def _make_alarm_fp_checker(profile: Dict):
    """Return a callable that returns True when a norm is a false positive
    alarm ID and should be dropped."""
    fp = profile.get("false_positive_filters", {})
    excluded = {p.upper() for p in fp.get("excluded_prefixes_from_alarm", [])}

    def is_fp(norm: str) -> bool:
        prefix = norm.split("-")[0] if "-" in norm else norm[:3]
        return prefix.upper() in excluded

    return is_fp


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_alarm_ref(raw: str) -> str:
    """Return a canonical alarm reference string.

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
# Public API
# ---------------------------------------------------------------------------

def extract_alarm_refs(
    text: str,
    *,
    profile_path: Optional[str | Path] = None,
    profile: Optional[Dict] = None,
    normalize: bool = True,
    unique: bool = True,
    max_ids: int = 200,
    min_score: float = 0.0,
) -> List[AlarmRef]:
    """Extract alarm and annunciator IDs from *text*.

    Args:
        text: Arbitrary chunk or document text.
        profile_path: Path to a plant profile JSON override.  Ignored when
            *profile* is supplied directly.
        profile: Pre-loaded plant profile dict (takes priority over
            *profile_path*).  Pass this when calling the function many times
            on the same plant to avoid repeated file I/O.
        normalize: Normalize matched IDs (uppercase, canonical hyphens).
        unique: Return each normalised ID at most once (keeps highest-score
            pattern hit when the same norm is matched by multiple patterns).
        max_ids: Safety cap on the number of refs returned.
        min_score: Drop any match whose pattern score is below this threshold.

    Returns:
        List of :class:`AlarmRef` namedtuples in match order.
    """
    if not text:
        return []

    if profile is None:
        profile = load_doc_ref_profile(profile_path)

    compiled_patterns = _compile_alarm_profile(profile)
    is_fp = _make_alarm_fp_checker(profile)

    results: List[AlarmRef] = []
    # norm → best (score, AlarmRef) when unique=True
    seen: Dict[str, float] = {}

    for pattern_name, score, rx in compiled_patterns:
        if score < min_score:
            continue
        for m in rx.finditer(text):
            raw = m.group(0)
            norm = _normalize_alarm_ref(raw) if normalize else raw.upper()

            if is_fp(norm):
                continue

            if unique:
                existing_score = seen.get(norm)
                if existing_score is not None:
                    if score <= existing_score:
                        continue
                    # Replace lower-confidence duplicate in results list
                    results = [r for r in results if r.norm != norm]
                seen[norm] = score

            results.append(AlarmRef(pattern_name=pattern_name, raw=raw, norm=norm, score=score))

            if len(results) >= max_ids:
                return results

    return results


def extract_alarm_ref_ids(
    text: str,
    **kwargs,
) -> List[str]:
    """Convenience wrapper — returns only the normalised alarm ID strings.

    Suitable for direct assignment to ``NERSeed.alarm_ids``.

    Args:
        text: Source text.
        **kwargs: Forwarded to :func:`extract_alarm_refs`.

    Returns:
        List of normalised alarm reference strings.
    """
    return [r.norm for r in extract_alarm_refs(text, **kwargs)]
