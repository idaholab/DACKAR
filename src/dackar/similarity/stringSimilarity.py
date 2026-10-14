# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Lightweight, dependency-free string-similarity measures.

These are surface (lexical) measures that compare the characters or tokens of
two strings directly, with no linguistic resources.  They complement the
WordNet/embedding measures elsewhere in :mod:`dackar.similarity` and make good
fallbacks when those heavier measures are unavailable.

Only the standard library is required.
"""
from __future__ import annotations


def token_jaccard(text_a: str, text_b: str) -> float:
    """Return the Jaccard index of the two strings' lowercased token sets.

    Tokens are whitespace-separated.  Two empty strings are defined as fully
    similar (1.0); one empty and one non-empty are fully dissimilar (0.0).
    """
    a_tokens = set(text_a.lower().split())
    b_tokens = set(text_b.lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def trigram_jaccard(text_a: str, text_b: str) -> float:
    """Return the Jaccard index of the two strings' character-trigram sets.

    Character trigrams give better recall than word Jaccard for short strings
    because they handle partial word matches (e.g. "calibr" in both
    "calibrate" and "calibration") and minor typos (edit distance <= 1 on a
    trigram).  Strings shorter than 3 characters are treated as a single
    "trigram" equal to the whole (lowercased) string.
    """
    def trigrams(s: str) -> set[str]:
        s = s.lower()
        return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

    tg_a = trigrams(text_a)
    tg_b = trigrams(text_b)
    union = tg_a | tg_b
    if not union:
        return 0.0
    return len(tg_a & tg_b) / len(union)
