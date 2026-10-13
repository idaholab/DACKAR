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

Implementation
--------------
The closed-vocabulary correction algorithm is the generic
:class:`dackar.text_processing.VocabSpellChecker.VocabSpellChecker`.  This module
is the outage adapter: it supplies the nuclear-domain vocabulary and keeps the
``.transform`` interface expected by :class:`ActivityCleaner`.
"""
from __future__ import annotations

import logging

from dackar.text_processing.VocabSpellChecker import VocabSpellChecker

from dackar.outage.outage_uncertainty.preprocessing.default_taxonomy import DEFAULT_TAXONOMY_RULES
from dackar.outage.outage_uncertainty.preprocessing.nuclear_abbreviations import (
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
    # Words formerly contributed by nuclear-abbreviation expansions that were
    # migrated into the shared abbreviations.xlsx (57-entry split).  Listed here
    # so slimming NUCLEAR_OUTAGE_ABBREVIATIONS does not shrink the spell-check
    # vocabulary: these tokens no longer arrive via the module dict.
    "achievable", "analysis", "and", "as", "brief", "controls",
    "cooling", "core", "drive", "examination", "hazard", "high",
    "identification", "job", "low", "material", "non-destructive",
    "operations", "positive", "pre-job", "reasonably", "reinstallation",
    "rod", "spray",
})


# ---------------------------------------------------------------------------
# DomainSpellChecker
# ---------------------------------------------------------------------------

def _build_domain_vocab(extra: set[str]) -> set[str]:
    """Build the nuclear outage vocabulary from four sources:

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

    return vocab


class DomainSpellChecker(VocabSpellChecker):
    """Correct residual misspellings using a nuclear outage domain vocabulary.

    A thin outage adapter over
    :class:`dackar.text_processing.VocabSpellChecker.VocabSpellChecker`: it
    assembles the nuclear-domain vocabulary and leaves the correction algorithm
    (``.transform``) to the base class.  The checker only ever substitutes a
    token with a word drawn from that vocabulary; unknown tokens with no close
    match are left unchanged.

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
        super().__init__(
            vocabulary=_build_domain_vocab(extra_vocab or set()),
            cutoff=cutoff,
            min_token_len=min_token_len,
        )
        logger.debug("DomainSpellChecker: vocabulary size=%d", len(self._vocab))
