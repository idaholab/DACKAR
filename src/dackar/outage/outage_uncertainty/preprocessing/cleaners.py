"""
Text cleaning pipeline for outage activity descriptions.

Outage schedules (especially those exported from Primavera P6) contain
descriptions that are notoriously noisy: equipment tag numbers, work-order
codes, abbreviations, spelling errors, and irregular whitespace all compete
with the signal words needed for similarity matching.

This module provides four composable stages:

ComponentIdRemover
    Strips P6-style instrument/equipment tags and work-order codes using
    regular expressions so they don't pollute token-overlap metrics.

TextacyPreprocessor
    Normalises whitespace, unicode, accents, and selective punctuation via
    the ``dackar.text_processing.Preprocessing`` textacy pipeline.  Falls
    back to minimal regex cleaning when the dependency is absent.

AbbreviationResolver  (imported from .abbreviations — not here)
    Expands domain abbreviations using the DACKAR Excel database.

IdentityTransform
    Pass-through no-op used to skip a stage.

ActivityCleaner
    Orchestrates all four stages in the correct order.
"""
from __future__ import annotations

import logging
import re

from outage_uncertainty.domain.activity import ActivityCase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Outage-specific textacy preprocessor list (lighter than the DACKAR default)
# ---------------------------------------------------------------------------
# We omit URL/email/hashtag/phone-number removal (they never appear in P6
# descriptions) and numerize (duration numbers are meaningful).  We keep the
# operations that genuinely help: whitespace, unicode, accents, selective
# punctuation, and hyphenated-word merging.
_OUTAGE_PREPROCESSOR_LIST = [
    "hyphenated_words",   # merge hyphenated compounds before tokenisation
    "whitespace",         # collapse tabs/newlines/multiple spaces
    "unicode",            # NFKC normalisation (removes zero-width chars etc.)
    "accents",            # strip diacritic marks
    "punctuation",        # remove noise punctuation (keep hyphens/slashes)
]
_OUTAGE_PREPROCESSOR_OPTIONS = {
    "unicode": {"form": "NFKC"},
    "accents": {"fast": False},
    # Keep hyphens (component tags), forward slashes (valve codes), and
    # parentheses; remove everything else in the "only" list.
    "punctuation": {"only": ["*", "+", ":", "=", "\\", "^", "_", "|", "~", ".", "...", ".."]},
}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

class IdentityTransform:
    """Pass-through: returns text unchanged.  Used to disable a pipeline stage."""

    def transform(self, text: str) -> str:
        return text


class ComponentIdRemover:
    """Strips P6 equipment/instrument tags and work-order codes from descriptions.

    These codes appear frequently at the start of activity descriptions
    (e.g. ``"PT-1234 CALIBRATE PRESSURE TRANSMITTER"``) or inline and carry
    no similarity signal — two activities on completely different systems can
    share the same text except for the tag number.

    Patterns removed
    ----------------
    - Instrument / equipment tags:
        1–4 uppercase letters, a hyphen, 2–7 digits, optional trailing letter
        Examples: ``PT-123``, ``MOV-4567A``, ``FCV-001``
    - Work-order / work-code prefixes:
        ``WO`` or ``WC`` optionally followed by separator and 4+ digits
        Examples: ``WO-12345``, ``WC 98765``, ``WO#20001``
    """

    _PATTERNS: list[re.Pattern] = [
        re.compile(r"\b[A-Z]{1,4}-\d{2,7}[A-Z]?\b"),         # PT-123, FCV-4567B
        re.compile(r"\bW[OC]\s*[-#]?\s*\d{4,}\b", re.I),     # WO-12345, WC 98765
    ]

    def transform(self, text: str) -> str:
        for pattern in self._PATTERNS:
            text = pattern.sub("", text)
        return re.sub(r"\s+", " ", text).strip()


class TextacyPreprocessor:
    """Normalises text using the DACKAR textacy-based preprocessing pipeline.

    Wraps ``dackar.text_processing.Preprocessing`` with a reduced operation
    list tuned for outage activity descriptions (see module-level constants).

    Falls back to minimal regex cleaning (whitespace collapse only) when
    ``dackar.text_processing`` or ``textacy`` is not installed.

    Args:
        preprocessor_list: Override the default list of textacy operations.
        preprocessor_options: Override the default per-operation options.
    """

    def __init__(
        self,
        preprocessor_list: list[str] | None = None,
        preprocessor_options: dict | None = None,
    ) -> None:
        self._pipeline = None
        pl = preprocessor_list if preprocessor_list is not None else _OUTAGE_PREPROCESSOR_LIST
        po = preprocessor_options if preprocessor_options is not None else _OUTAGE_PREPROCESSOR_OPTIONS
        try:
            from dackar.text_processing.Preprocessing import Preprocessing
            self._pipeline = Preprocessing(preprocessorList=pl, preprocessorOptions=po)
            logger.debug("TextacyPreprocessor: dackar textacy pipeline ready")
        except ImportError as exc:
            logger.warning(
                "TextacyPreprocessor: dackar.text_processing unavailable (%s); "
                "using whitespace-collapse fallback",
                exc,
            )

    def transform(self, text: str) -> str:
        if not text:
            return text
        if self._pipeline is not None:
            return self._pipeline(text)
        # Minimal fallback: collapse whitespace and normalise case
        return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Main cleaner
# ---------------------------------------------------------------------------

class ActivityCleaner:
    """Cleans a raw activity description through a four-stage pipeline.

    Stage order
    -----------
    1. ``component_id_remover`` — strips instrument tags and WO codes
    2. ``preprocessor``         — textacy normalisation (whitespace, unicode …)
    3. ``abbreviation_expander``— expands domain abbreviations
    4. ``spell_checker``        — corrects residual misspellings (Phase 2)

    Each stage must expose ``transform(text: str) -> str``.
    Pass ``IdentityTransform()`` to skip a stage entirely.

    Args:
        component_id_remover: Removes equipment/WO codes. Defaults to
            :class:`ComponentIdRemover`.
        preprocessor: Normalises text. Defaults to :class:`TextacyPreprocessor`.
        abbreviation_expander: Expands abbreviations. Defaults to
            :class:`IdentityTransform` (pass-through); wire in
            :class:`~outage_uncertainty.preprocessing.abbreviations.AbbreviationResolver`
            from the facade.
        spell_checker: Corrects spelling. Defaults to :class:`IdentityTransform`.
    """

    def __init__(
        self,
        component_id_remover=None,
        preprocessor=None,
        abbreviation_expander=None,
        spell_checker=None,
    ) -> None:
        self.component_id_remover = component_id_remover or ComponentIdRemover()
        self.preprocessor = preprocessor or TextacyPreprocessor()
        self.abbreviation_expander = abbreviation_expander or IdentityTransform()
        self.spell_checker = spell_checker or IdentityTransform()

    def clean(self, activity: ActivityCase) -> ActivityCase:
        """Run the cleaning pipeline and store the result in ``cleaned_description``."""
        text: str = activity.raw_description or ""
        text = self.component_id_remover.transform(text)
        text = self.preprocessor.transform(text)
        text = self.abbreviation_expander.transform(text)
        text = self.spell_checker.transform(text)
        activity.cleaned_description = text
        return activity
