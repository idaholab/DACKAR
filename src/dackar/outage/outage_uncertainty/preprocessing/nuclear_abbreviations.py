"""
Nuclear-outage abbreviation layer applied on top of the shared Excel file.

The bulk of the nuclear vocabulary now lives in the shared DACKAR
``data/abbreviations.xlsx`` (loaded by :class:`AbbreviationResolver` and by the
rest of DACKAR via ``dackar.text_processing``).  This module holds only the
entries that *cannot* live in that shared file without changing behaviour for
other, non-outage consumers:

1. **Context overrides** — abbreviations whose shared-file expansion is correct
   in general engineering text but wrong in a nuclear-outage context (e.g.
   ``cont`` → "continuous" generally, but "containment" in an outage).  Listed
   first; they override the shared-file resolution.

2. **Outage-context-only shorthand** — short or English-colliding tokens that
   are deliberately absent from the shared file because adding them there would
   mis-expand ordinary text everywhere else in DACKAR (e.g. ``pm`` →
   "preventive maintenance", ``prev`` → "preventive").  They are safe only
   inside the outage preprocessing pipeline, so they are applied here.

Resolution order (highest priority last):
    shared Excel file  →  this supplement  →  caller-supplied ``extra_abbreviations``

All keys are lowercase; :class:`AbbreviationResolver` handles case
normalisation internally.
"""
from __future__ import annotations

NUCLEAR_OUTAGE_ABBREVIATIONS: dict[str, str] = {

    # ------------------------------------------------------------------
    # Context overrides — present in abbreviations.xlsx with a general
    # meaning that is wrong in a nuclear-outage context.  These win.
    # ------------------------------------------------------------------
    "cont":   "containment",    # sheet: "continuous"
    "rep":    "replace",        # sheet: "report"
    "comp":   "component",      # sheet: "composite"
    "mtg":    "mounting",       # sheet: "meeting"
    "gen":    "generator",      # sheet: "general"
    "cal":    "calibrate",      # sheet: "calibration"
    "surv":   "surveillance",   # sheet: "survey"
    "verif":  "verification",   # sheet: "verify"
    "fdwtr":  "feedwater",      # sheet: "feed water"

    # ------------------------------------------------------------------
    # Outage-context-only shorthand — deliberately NOT in the shared file
    # because these tokens collide with common English words or are too
    # short to disambiguate outside an outage schedule.
    # ------------------------------------------------------------------
    "pm":     "preventive maintenance",
    "prev":   "preventive",
    "corr":   "corrective",
    "func":   "functional",
    "adj":    "adjust",
    "si":     "safety injection",
    "rp":     "radiation protection",
    "dg":     "diesel generator",
    "wo":     "work order",
    "ts":     "technical specification",
    "ot":     "outage",
    "mn":     "main",
}
