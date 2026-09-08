"""PMVocabularyLoader — load health-status vocabulary from the DACKAR data directory.

Drives ``analyze_degradation`` and ``compute_pm_found_defect_rate`` in
``effectiveness_analyzer``.  Falls back to hardcoded stems when *data_dir* is
``None`` or the CSV files are absent, so callers without the data path are
unaffected.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Sequence


# Columns to load from each CSV (Adverbs excluded — not informative for as-found text)
_COLS: Sequence[str] = ("Nouns", "Verbs", "Adjectives")

_NEG_FILE = "health_status_keywords_negative.csv"
_POS_FILE = "health_status_keywords_positive.csv"
_NEU_FILE = "health_status_keywords_neutral.csv"

# Fallback hardcoded stems (original behaviour before Wave 2).
# Kept as a safety net when data_dir is unavailable.
_FALLBACK: Dict[str, FrozenSet[str]] = {
    "degrading": frozenset({"degrad", "worse", "increas", "severity", "wear"}),
    "improving": frozenset({"improv", "normal", "no defect", "acceptable"}),
}


def _read_terms(path: Path, cols: Sequence[str]) -> FrozenSet[str]:
    """Return lowercased, stripped terms from *cols* of a keyword CSV."""
    terms: set[str] = set()
    if not path.exists():
        return frozenset()
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in cols:
                val = (row.get(col) or "").strip()
                if val:
                    terms.add(val.lower())
    return frozenset(terms)


def _term_matches(term: str, blob: str) -> bool:
    """Return True if *term* appears in *blob* at a leading word boundary.

    Multi-word phrases (containing spaces) are matched as substrings.
    Single-word terms use a left-side word boundary so that the vocabulary
    term is matched as a prefix of a word — "crack" matches "crack", "cracks",
    "cracked", "cracking" and "leak" matches "leakage", but neither matches in
    the middle of an unrelated word (e.g., "crack" does not match "firecracker"
    because 'c' in "cracker" has no leading word boundary).
    """
    if " " in term:
        return term in blob
    return bool(re.search(r"\b" + re.escape(term), blob))


def matches_any(blob: str, terms: FrozenSet[str]) -> bool:
    """Return True if any term in *terms* matches within *blob*."""
    return any(_term_matches(t, blob) for t in terms)


class PMVocabularyLoader:
    """Load and cache health-status vocabulary keyed by *data_dir* path.

    Usage::

        vocab = PMVocabularyLoader.load(cfg.data_dir)
        is_degrading = matches_any(blob, vocab["degrading"])
    """

    _cache: Dict[str, Dict[str, FrozenSet[str]]] = {}

    @classmethod
    def load(cls, data_dir: Optional[Path]) -> Dict[str, FrozenSet[str]]:
        """Return ``{"degrading": ..., "improving": ...}`` vocabulary sets.

        When *data_dir* is ``None``, returns the hardcoded fallback stems so
        existing behaviour is fully preserved.
        """
        if data_dir is None:
            return dict(_FALLBACK)

        key = str(data_dir)
        if key in cls._cache:
            return cls._cache[key]

        neg = _read_terms(data_dir / _NEG_FILE, _COLS)
        pos = _read_terms(data_dir / _POS_FILE, _COLS)

        vocab: Dict[str, FrozenSet[str]] = {
            "degrading": neg if neg else _FALLBACK["degrading"],
            "improving": pos if pos else _FALLBACK["improving"],
        }
        cls._cache[key] = vocab
        return vocab

    @classmethod
    def clear_cache(cls) -> None:
        """Invalidate the vocabulary cache (useful in tests)."""
        cls._cache.clear()
