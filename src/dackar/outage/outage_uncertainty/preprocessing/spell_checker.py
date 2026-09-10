"""
Domain-aware spell checker for nuclear outage activity descriptions.

Design rationale
----------------
Generic spell checkers (pyspellchecker, autocorrect, etc.) are inappropriate
here for two reasons:

1. **Domain vocabulary** — nuclear outage terms such as ``"pressurizer"``,
   ``"impeller"``, ``"torquing"`` are not in standard English dictionaries and
   would be "corrected" to unrelated words.
2. **Abbreviated tokens** — abbreviations that were not fully expanded (or are
   new to the dictionary) must pass through unchanged; a generic checker would
   mangle them.

This checker builds its vocabulary *from the package's own domain data*
(nuclear abbreviation expansions, taxonomy keywords, and a static engineering
word list) so it can only suggest corrections to words that belong to the
nuclear outage domain.

Algorithm
---------
For each token in the cleaned text:

1. **Pass-through** if the token is shorter than ``min_token_len`` (default 5)
   or is already in the domain vocabulary.
2. Otherwise call :func:`difflib.get_close_matches` against the vocabulary
   with a configurable ``cutoff`` (default 0.85).  This is equivalent to
   accepting corrections whose SequenceMatcher ratio ≥ ``cutoff``.
3. If a match is found, substitute; otherwise leave the original token.

The 0.85 threshold was validated on a representative set of P6 typos:

* ``calibartion`` → ``calibration``  (ratio 0.91)
* ``presure``     → ``pressure``     (ratio 0.93)
* ``transmiter``  → ``transmitter``  (ratio 0.95)
* ``maintenace``  → ``maintenance``  (ratio 0.95)
* ``inspcetion``  → ``inspection``   (ratio 0.90)

and confirmed safe against common false-positive pairs
(``primary`` / ``primate`` ratio 0.71; ``install`` / ``instill`` ratio 0.86
which is borderline — resolved by ``install`` matching itself exactly at 1.0).

No external dependencies are required.  The checker is deterministic and
fast enough for the corpus sizes (tens of thousands of activities) encountered
in a single-outage planning cycle.
"""
from __future__ import annotations

import logging
from difflib import get_close_matches

from outage_uncertainty.preprocessing.default_taxonomy import DEFAULT_TAXONOMY_RULES
from outage_uncertainty.preprocessing.nuclear_abbreviations import (
    NUCLEAR_OUTAGE_ABBREVIATIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static engineering / nuclear vocabulary
# ---------------------------------------------------------------------------
# Words (all lowercase) that appear in nuclear outage P6 activity descriptions
# but are not derived from the nuclear abbreviation supplement or the taxonomy
# keyword list.  Only words ≥ 5 characters are necessary here; shorter tokens
# are never sent to get_close_matches (they are too short to distinguish typos
# reliably from valid edits).

_COMMON_ENGINEERING_WORDS: frozenset[str] = frozenset({
    # Action verbs — infinitive and nominal forms
    "replace", "replacement", "replacing",
    "install", "installation", "installing",
    "remove",  "removal",     "removing",
    "inspect", "inspection",  "inspecting",
    "calibrate", "calibration", "calibrating",
    "adjust",  "adjustment",  "adjusting",
    "overhaul", "overhauling",
    "repair",  "repairing",
    "verify",  "verification", "verifying",
    "lubricate", "lubrication", "lubricating",
    "clean",   "cleaning",
    "flush",   "flushing",
    "drain",   "draining",
    "purge",   "purging",
    "vent",    "venting",
    "bleed",   "bleeding",
    "torque",  "torquing",
    "weld",    "welding",
    "fabricate", "fabrication",
    "disassemble", "disassembly",
    "reassemble",  "reassembly",
    "pressurize",  "depressurize",
    "energize", "de-energize",
    "isolate",  "isolation",
    "restore",  "restoration",
    "commission", "decommission",
    "measure",  "measurement",
    "testing",  "retest",
    "document", "documentation",
    "perform",  "performance",
    # Components
    "bearing",   "bearings",
    "shaft",     "shafts",
    "coupling",  "couplings",
    "gasket",    "gaskets",
    "flange",    "flanges",
    "impeller",  "impellers",
    "bracket",   "brackets",
    "nozzle",    "nozzles",
    "strainer",  "strainers",
    "filter",    "filters",
    "housing",   "casing",
    "packing",   "gland",
    "actuator",  "positioner",
    "transmitter", "transducer",
    "sensor",    "sensors",
    "controller", "relay",
    "switch",    "switches",
    "breaker",   "breakers",
    "transformer", "compressor",
    "expansion", "reducer",
    "support",   "hanger",
    "insulation",
    # Systems and descriptors
    "reactor",   "containment",
    "coolant",   "feedwater",
    "turbine",   "condenser",
    "generator", "steam",
    "pressurizer",
    "primary",   "secondary",
    "auxiliary", "emergency",
    "standby",   "redundant",
    "mechanical", "electrical",
    "pneumatic", "hydraulic",
    "thermal",   "nuclear",
    "structural", "functional",
    "visual",    "dimensional",
    "ultrasonic", "radiographic",
    "preventive", "corrective",
    "periodic",  "annual",
    "refuel",    "outage",
    "setpoint",  "setpoints",
    "differential", "pressure",
    "temperature", "voltage",
    "resistance", "current",
    "level",     "flow",
    "safety",    "protective",
    "radioactive", "radiation",
    "critical",  "normal",
    "upper",     "lower",
    "inner",     "outer",
    "discharge", "suction",
    "upstream",  "downstream",
    "train",     "division",
    "system",    "subsystem",
    "assembly",  "subassembly",
    "component", "equipment",
    "station",   "panel",
    "cabinet",   "enclosure",
    "motor",     "pump",
    "valve",     "piping",
    "tubing",    "fitting",
    "connector", "terminal",
    "wiring",    "cabling",
    "ground",    "grounding",
    "torque",    "thread",
    "bolt",      "nut",
    "washer",    "stud",
    "screw",     "anchor",
    "bracket",   "plate",
    "mounting",  "support",
    "scaffold",  "scaffolding",
    "personnel", "qualified",
    "surveillance", "operability",
    "clearance", "permit",
    "procedure", "instruction",
    "checklist", "record",
    "walkdown",  "walkthrough",
})


# ---------------------------------------------------------------------------
# DomainSpellChecker
# ---------------------------------------------------------------------------

class DomainSpellChecker:
    """Correct residual misspellings using a nuclear outage domain vocabulary.

    The checker is intentionally conservative: it only suggests corrections
    whose SequenceMatcher ratio meets the ``cutoff`` threshold and only acts
    on tokens longer than ``min_token_len`` characters.  Unknown tokens with
    no close vocabulary match are left unchanged.

    Args:
        cutoff: Minimum SequenceMatcher ratio for a correction to be
            accepted.  Default 0.85.  Lower values increase recall but
            also increase the risk of false corrections.
        min_token_len: Tokens shorter than this are never corrected
            (too short to disambiguate reliably).  Default 5.
        extra_vocab: Additional domain words to add to the built-in
            vocabulary.  Useful for plant-specific terminology.
    """

    def __init__(
        self,
        cutoff: float = 0.85,
        min_token_len: int = 5,
        extra_vocab: set[str] | None = None,
    ) -> None:
        self._cutoff = cutoff
        self._min_len = min_token_len
        self._vocab: frozenset[str] = self._build_vocab(extra_vocab or set())
        # Sorted list required by get_close_matches
        self._vocab_list: list[str] = sorted(self._vocab)

        logger.debug(
            "DomainSpellChecker: vocabulary size=%d, cutoff=%.2f, min_len=%d",
            len(self._vocab),
            cutoff,
            min_token_len,
        )

    # ------------------------------------------------------------------
    # Public interface (matches ActivityCleaner's expected .transform API)
    # ------------------------------------------------------------------

    def transform(self, text: str) -> str:
        """Return *text* with misspelled tokens replaced by their nearest
        vocabulary match.  Tokens already in the vocabulary or shorter than
        ``min_token_len`` are passed through unchanged."""
        if not text:
            return text

        corrected: list[str] = []
        for token in text.lower().split():
            corrected.append(self._correct(token))
        return " ".join(corrected)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _correct(self, token: str) -> str:
        """Return the corrected form of a single token."""
        # Pass through: too short to spell-check reliably
        if len(token) < self._min_len:
            return token
        # Pass through: already a known domain word
        if token in self._vocab:
            return token
        # Attempt correction
        matches = get_close_matches(token, self._vocab_list, n=1, cutoff=self._cutoff)
        if matches:
            logger.debug("SpellChecker: '%s' → '%s'", token, matches[0])
            return matches[0]
        return token

    def _build_vocab(self, extra: set[str]) -> frozenset[str]:
        """Build the domain vocabulary from three sources:

        1. Individual words from all nuclear abbreviation *expansions*
           (e.g. "main coolant pump" contributes "main", "coolant", "pump").
        2. All keywords from the taxonomy rules (single-word and phrase parts).
        3. The static :data:`_COMMON_ENGINEERING_WORDS` set.
        4. Any ``extra_vocab`` supplied by the caller.
        """
        vocab: set[str] = set()

        # Nuclear abbreviation expansion values
        for phrase in NUCLEAR_OUTAGE_ABBREVIATIONS.values():
            vocab.update(phrase.lower().split())

        # Taxonomy keyword tokens
        for kw in DEFAULT_TAXONOMY_RULES:
            vocab.update(kw.lower().split())

        # Static engineering vocabulary
        vocab.update(w.lower() for w in _COMMON_ENGINEERING_WORDS)

        # Caller-supplied extras
        vocab.update(w.lower() for w in extra)

        return frozenset(vocab)
