"""
Abbreviation resolution for outage activity descriptions.

Wraps ``dackar.text_processing.Abbreviation`` which holds a curated database
of nuclear-outage and P6 abbreviations loaded from an Excel file.

Graceful degradation chain
--------------------------
1. DACKAR Excel-backed handler (``dackar.text_processing.Abbreviation``)
2. Extra-abbreviations dict supplied directly by the caller
3. Pass-through (no expansion)

All three levels can coexist: the Excel handler is augmented with any
``extra_abbreviations`` entries before use.
"""
from __future__ import annotations

import logging

from outage_uncertainty.preprocessing.nuclear_abbreviations import NUCLEAR_OUTAGE_ABBREVIATIONS

logger = logging.getLogger(__name__)


class AbbreviationResolver:
    """Expands abbreviations found in outage activity descriptions.

    The resolver applies abbreviations in the following priority order
    (highest priority last, so later entries override earlier ones):

    1. **Excel file** (``abbreviations_file``) — 654-entry general engineering
       vocabulary loaded via ``dackar.text_processing.Abbreviation``.
    2. **Nuclear outage supplement** (:data:`NUCLEAR_OUTAGE_ABBREVIATIONS`) —
       built-in, applied unconditionally.  Overrides wrong duplicate
       resolutions in the Excel file and fills nuclear-specific gaps.
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
        self._handler = None

        # Build effective extras: nuclear supplement first so caller-supplied
        # extra_abbreviations can still override individual entries.
        effective_extra: dict[str, str] = {
            **NUCLEAR_OUTAGE_ABBREVIATIONS,
            **(extra_abbreviations or {}),
        }

        # _fallback is used when the dackar handler is unavailable.
        # Keys are uppercased for case-insensitive lookup in _dict_expand.
        self._fallback: dict[str, str] = {
            k.upper(): v for k, v in effective_extra.items()
        }

        if abbreviations_file is not None:
            self._handler = self._load_handler(abbreviations_file, effective_extra)

        if self._handler is None:
            n = len(self._fallback)
            logger.info(
                "AbbreviationResolver: dict-only mode (%d entries, %d nuclear supplement)",
                n,
                len(NUCLEAR_OUTAGE_ABBREVIATIONS),
            )

    # ------------------------------------------------------------------
    # Public interface expected by ActivityCleaner
    # ------------------------------------------------------------------

    def transform(self, text: str) -> str:
        """Return *text* with abbreviations expanded."""
        if not text:
            return text
        if self._handler is not None:
            return self._handler.abbreviationSub(text)
        return self._dict_expand(text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_handler(self, abbreviations_file: str, extra: dict[str, str]):
        """
        Construct a ``dackar.text_processing.Abbreviation`` instance directly
        from the Excel file, bypassing the ``nlpConfig`` dependency in its
        ``__init__``.  Returns ``None`` on any failure.
        """
        try:
            import pandas as pd
            from dackar.text_processing.Abbreviation import Abbreviation

            abbrList = pd.read_excel(abbreviations_file)
            if "Abbreviation" not in abbrList.columns or "Full" not in abbrList.columns:
                raise ValueError(
                    f"Expected columns ['Abbreviation', 'Full'] in {abbreviations_file}; "
                    f"found {list(abbrList.columns)}"
                )

            # Build the handler without calling __init__ (avoids nlpConfig)
            handler = Abbreviation.__new__(Abbreviation)
            handler.type = handler.name = "Abbreviation"
            handler.abbreviationsFilename = abbreviations_file

            # abbrDict keys must be lowercase strings (as expected by abbreviationSub)
            handler.abbrDict = dict(
                zip(
                    abbrList["Abbreviation"].astype(str).str.lower().str.strip(),
                    abbrList["Full"].astype(str).str.lower().str.strip(),
                )
            )

            # Merge extra abbreviations (updateAbbreviation lowercases keys)
            if extra:
                handler.updateAbbreviation(extra, reset=False)

            logger.info(
                "AbbreviationResolver: loaded %d entries from %s",
                len(handler.abbrDict),
                abbreviations_file,
            )
            return handler

        except FileNotFoundError:
            logger.warning("AbbreviationResolver: file not found: %s", abbreviations_file)
        except ImportError as exc:
            logger.warning("AbbreviationResolver: dackar.text_processing unavailable (%s)", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AbbreviationResolver: failed to load (%s)", exc)
        return None

    def _dict_expand(self, text: str) -> str:
        """Token-level case-insensitive lookup (fallback when no Excel handler)."""
        if not self._fallback:
            return text
        tokens = []
        for token in text.split():
            tokens.append(self._fallback.get(token.upper(), token))
        return " ".join(tokens)
