# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Closed-vocabulary spell corrector.

Unlike the dictionary- and model-based
:class:`~dackar.text_processing.SpellChecker.SpellChecker`, this corrector only
ever substitutes a token with a word that is already present in a
caller-supplied vocabulary.  It never proposes a correction outside that set, so
it is safe for specialised corpora whose terms are absent from general English
dictionaries (domain jargon, part names, un-expanded abbreviations): an unknown
token with no close vocabulary match is left untouched rather than mangled.  It
has no dependencies beyond the standard library.

Algorithm
---------
For each whitespace-separated token in the (lower-cased) text:

1. Pass through unchanged if the token is shorter than ``min_token_len`` or is
   already in the vocabulary.
2. Otherwise take the single nearest vocabulary word via
   :func:`difflib.get_close_matches` at ``cutoff`` (the SequenceMatcher ratio
   threshold).
3. Substitute when a match is found; otherwise keep the original token.

The corrector is deterministic.
"""
from __future__ import annotations

import logging
from difflib import get_close_matches
from typing import Iterable

logger = logging.getLogger(__name__)


class VocabSpellChecker:
    """Correct tokens against a fixed vocabulary using difflib ratios.

    The corrector is intentionally conservative: it only accepts a correction
    whose SequenceMatcher ratio meets ``cutoff`` and only acts on tokens longer
    than ``min_token_len``.  Out-of-vocabulary tokens with no close match are
    left unchanged.

    Args:
        vocabulary: The valid words.  Corrections are only ever drawn from this
            set; entries are lower-cased on construction.
        cutoff: Minimum SequenceMatcher ratio for a correction to be accepted.
            Default 0.85.  Lower values raise recall but also the risk of false
            corrections.
        min_token_len: Tokens shorter than this are never corrected (too short
            to tell a typo from a valid short word).  Default 5.
    """

    def __init__(
        self,
        vocabulary: Iterable[str],
        cutoff: float = 0.85,
        min_token_len: int = 5,
    ) -> None:
        self._cutoff = cutoff
        self._min_len = min_token_len
        self._vocab: frozenset[str] = frozenset(w.lower() for w in vocabulary)
        # Sorted list required by get_close_matches
        self._vocab_list: list[str] = sorted(self._vocab)

        logger.debug(
            "VocabSpellChecker: vocabulary size=%d, cutoff=%.2f, min_len=%d",
            len(self._vocab),
            cutoff,
            min_token_len,
        )

    def transform(self, text: str) -> str:
        """Return *text* with out-of-vocabulary tokens replaced by their nearest
        vocabulary match.  Tokens already in the vocabulary or shorter than
        ``min_token_len`` are passed through unchanged.

        The text is lower-cased before tokenising, so every returned token is
        lower-case — including tokens that were *not* corrected.  Callers that
        surface the result to users (e.g. a ``cleaned_description`` field)
        should expect this normalisation."""
        if not text:
            return text
        return " ".join(self._correct(token) for token in text.lower().split())

    def _correct(self, token: str) -> str:
        """Return the corrected form of a single token."""
        # Pass through: too short to spell-check reliably
        if len(token) < self._min_len:
            return token
        # Pass through: already a known word
        if token in self._vocab:
            return token
        # Attempt correction
        matches = get_close_matches(token, self._vocab_list, n=1, cutoff=self._cutoff)
        if matches and not self._is_affix_variant(token, matches[0]):
            logger.debug("VocabSpellChecker: corrected '%s' to '%s'", token, matches[0])
            return matches[0]
        return token

    @staticmethod
    def _is_affix_variant(token: str, candidate: str) -> bool:
        """Return True when *candidate* is *token* with a prefix/suffix dropped.

        A closed-vocabulary match that is a strict substring of the token (two
        or more characters shorter) is almost never a typo correction — it is
        the token with an affix removed.  For prefixed domain verbs that often
        inverts the meaning (``uninstall`` -> ``install``, ``unisolate`` ->
        ``isolate``, ``reenergize`` -> ``energize``, ``retorque`` -> ``torque``).
        Refusing these keeps the corrector from turning an operation into its
        opposite; a genuine typo (transposition, single substitution) is not a
        substring of the original and is unaffected.
        """
        return candidate in token and len(token) - len(candidate) >= 2
