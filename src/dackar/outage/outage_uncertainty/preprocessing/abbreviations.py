"""
Abbreviation resolution for outage activity descriptions.

Loads the curated nuclear-outage and P6 abbreviations from the shared DACKAR
Excel file and expands them one whole token at a time.  Whole-token expansion
(rather than the substring/suffix matching in
``dackar.text_processing.Abbreviation.abbreviationSub``) keeps an abbreviation
that is also a substring of a longer word from leaking into ordinary text
(``2pm`` must stay ``2pm``, not become "2 preventive maintenance").

Sources (lowest priority first; later entries override earlier ones)
--------------------------------------------------------------------
1. DACKAR Excel file (``abbreviations_file``)
2. Nuclear-outage supplement (:data:`NUCLEAR_OUTAGE_ABBREVIATIONS`)
3. Extra-abbreviations dict supplied directly by the caller

Negations (:data:`_NEVER_EXPAND`) are dropped from the final map regardless of
source so a token like ``no`` is never rewritten to "number".  If the Excel
file cannot be read the supplement and caller extras still apply; an empty map
means pass-through (no expansion).
"""
from __future__ import annotations

import logging

from dackar.outage.outage_uncertainty.preprocessing.nuclear_abbreviations import NUCLEAR_OUTAGE_ABBREVIATIONS

logger = logging.getLogger(__name__)

# Tokens that must never be expanded in an outage description even when they
# are present in the shared abbreviation sheet.  ``no`` -> "number" lives in
# abbreviations.xlsx for general engineering text, but dropping a negation
# changes the meaning of a P6 description ("no leak found" -> "number leak
# found") for both retrieval and the LabelMapper keyword flags.
_NEVER_EXPAND: frozenset[str] = frozenset({"no", "not", "non", "nil", "none", "never"})


class AbbreviationResolver:
    """Expands abbreviations found in outage activity descriptions.

    The resolver applies abbreviations in the following priority order
    (highest priority last, so later entries override earlier ones):

    1. **Excel file** (``abbreviations_file``) — shared DACKAR engineering and
       nuclear vocabulary loaded via ``dackar.text_processing.Abbreviation``.
    2. **Nuclear outage supplement** (:data:`NUCLEAR_OUTAGE_ABBREVIATIONS`) —
       built-in, applied unconditionally.  A small set of outage-context
       overrides and context-only shorthand that must not live in the shared
       Excel file (they would mis-expand text for other DACKAR consumers).
       Always active; requires no external dependencies.
    3. **Caller-supplied extras** (``extra_abbreviations``) — highest priority;
       overrides both the Excel file and the nuclear supplement.

    Args:
        abbreviations_file: Path to the abbreviations Excel file (.xlsx).
            The file must have columns ``Abbreviation`` and ``Full``.
            Defaults to ``None``; the nuclear supplement and
            ``extra_abbreviations`` still apply.
        extra_abbreviations: Additional ``{abbr: full}`` pairs
            (case-insensitive keys). Merged on top of everything else.
    """

    def __init__(
        self,
        abbreviations_file: str | None = None,
        extra_abbreviations: dict[str, str] | None = None,
    ) -> None:
        # Build effective extras: nuclear supplement first so caller-supplied
        # extra_abbreviations can still override individual entries.
        effective_extra: dict[str, str] = {
            **NUCLEAR_OUTAGE_ABBREVIATIONS,
            **(extra_abbreviations or {}),
        }

        # Single whole-token expansion map, keyed by lowercase token.
        # Resolution order (lowest priority first): shared Excel file, then the
        # nuclear supplement, then caller extras.  Expansion is applied token by
        # token (see transform) rather than via the shared handler's
        # abbreviationSub, which also rewrites the suffix after '-' or a digit
        # ('2pm' -> 'preventive maintenance', 'pump-ts' -> '... technical
        # specification') — meaning-changing for P6 descriptions.
        self._token_map: dict[str, str] = {}
        if abbreviations_file is not None:
            self._token_map.update(self._load_sheet(abbreviations_file))
        for k, v in effective_extra.items():
            self._token_map[k.lower().strip()] = v.lower()

        # Negations are never expanded regardless of source (see _NEVER_EXPAND).
        for token in _NEVER_EXPAND:
            self._token_map.pop(token, None)

        logger.info(
            "AbbreviationResolver: %d whole-token entries (%d nuclear supplement)",
            len(self._token_map),
            len(NUCLEAR_OUTAGE_ABBREVIATIONS),
        )

    # ------------------------------------------------------------------
    # Public interface expected by ActivityCleaner
    # ------------------------------------------------------------------

    def transform(self, text: str) -> str:
        """Return *text* with whole-token abbreviations expanded.

        Only complete whitespace-delimited tokens are matched, so an
        abbreviation that is also a substring of a longer word (``pm`` inside
        ``2pm``, ``ts`` inside ``pump-ts``) is left untouched.  Negations in
        :data:`_NEVER_EXPAND` are never expanded.
        """
        if not text:
            return text
        if not self._token_map:
            return text
        return " ".join(
            self._token_map.get(token.lower(), token) for token in text.split()
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sheet(abbreviations_file: str) -> dict[str, str]:
        """Read ``{abbreviation: full}`` pairs from the Excel file.

        Keys and values are lower-cased and stripped.  Returns an empty dict
        on any failure (missing file, missing pandas, bad columns) so the
        nuclear supplement and caller extras still apply.
        """
        try:
            import pandas as pd

            abbrList = pd.read_excel(abbreviations_file)
            if "Abbreviation" not in abbrList.columns or "Full" not in abbrList.columns:
                raise ValueError(
                    f"Expected columns ['Abbreviation', 'Full'] in {abbreviations_file}; "
                    f"found {list(abbrList.columns)}"
                )

            token_map = dict(
                zip(
                    abbrList["Abbreviation"].astype(str).str.lower().str.strip(),
                    abbrList["Full"].astype(str).str.lower().str.strip(),
                )
            )
            logger.info(
                "AbbreviationResolver: loaded %d entries from %s",
                len(token_map),
                abbreviations_file,
            )
            return token_map

        except FileNotFoundError:
            logger.warning("AbbreviationResolver: file not found: %s", abbreviations_file)
        except ImportError as exc:
            logger.warning("AbbreviationResolver: pandas unavailable (%s)", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AbbreviationResolver: failed to load (%s)", exc)
        return {}
