"""
Built-in nuclear outage abbreviation supplement.

This dictionary is merged into :class:`AbbreviationResolver` on every
instantiation, regardless of whether the DACKAR Excel file or the
``dackar.text_processing`` library are available.  It serves two roles:

1. **Corrections** — several entries in the general-engineering
   ``abbreviations.xlsx`` file resolve to the wrong expansion in a nuclear
   outage context due to last-wins duplicate handling.  These entries are
   listed first and override those wrong defaults.

2. **Gap-fill** — nuclear-plant-specific component, system, task, and
   work-management abbreviations that are absent from the general-purpose
   Excel file.

Resolution order (highest priority last):
    Excel file  →  this supplement  →  caller-supplied ``extra_abbreviations``

All keys are lowercase; :class:`AbbreviationResolver` handles case
normalisation internally.
"""
from __future__ import annotations

NUCLEAR_OUTAGE_ABBREVIATIONS: dict[str, str] = {

    # ------------------------------------------------------------------
    # Corrections: override wrong last-wins resolutions in abbreviations.xlsx
    # ------------------------------------------------------------------
    # Excel duplicate resolution produces:
    #   cont  → "continuous"   (should be "containment" in outage context)
    #   rep   → "report"       (should be "replace")
    #   comp  → "composite"    (should be "component")
    #   mtg   → "meeting"      (should be "mounting")
    #   gen   → "general"      (should be "generator")
    "cont":   "containment",
    "rep":    "replace",
    "comp":   "component",
    "mtg":    "mounting",
    "gen":    "generator",

    # ------------------------------------------------------------------
    # Reactor coolant and primary systems
    # ------------------------------------------------------------------
    "mcp":    "main coolant pump",
    "rcp":    "reactor coolant pump",
    "rcs":    "reactor coolant system",
    "rhr":    "residual heat removal",
    "pzr":    "pressurizer",        # also in Excel — keep for fallback path

    # ------------------------------------------------------------------
    # Safety systems and injection
    # ------------------------------------------------------------------
    "eccs":   "emergency core cooling system",
    "hpsi":   "high pressure safety injection",
    "lpsi":   "low pressure safety injection",
    "csi":    "containment spray injection",
    "crd":    "control rod drive",

    # ------------------------------------------------------------------
    # Valves
    # ------------------------------------------------------------------
    "msiv":   "main steam isolation valve",
    "mov":    "motor operated valve",
    "aov":    "air operated valve",
    "prv":    "pressure relief valve",
    "psv":    "pressure safety valve",
    "chv":    "check valve",
    "bfv":    "butterfly valve",
    "gtv":    "gate valve",
    "glv":    "globe valve",
    "bav":    "ball valve",

    # ------------------------------------------------------------------
    # Balance-of-plant and auxiliary systems
    # ------------------------------------------------------------------
    "swp":    "service water pump",
    "ccw":    "component cooling water",
    "afw":    "auxiliary feedwater",
    "mfw":    "main feedwater",
    "edg":    "emergency diesel generator",
    "dg":     "diesel generator",
    "htx":    "heat exchanger",     # "hx" already correct in Excel

    # ------------------------------------------------------------------
    # Task / action abbreviations (all absent from Excel)
    # ------------------------------------------------------------------
    "repl":    "replace",
    "tst":     "test",
    "adj":     "adjust",
    "ovhl":    "overhaul",
    "clb":     "calibration",
    "lubr":    "lubrication",
    "lube":    "lubrication",
    "verif":   "verification",      # Excel has "verify" — override to noun form
    "verf":    "verification",
    "rmvl":    "removal",
    "reinstl": "reinstallation",
    "disassy": "disassembly",
    "fabr":    "fabrication",
    "wldg":    "welding",
    "torq":    "torquing",
    "vnt":     "vent",

    # ------------------------------------------------------------------
    # Radiation protection and non-destructive examination
    # ------------------------------------------------------------------
    "rp":     "radiation protection",
    "alara":  "as low as reasonably achievable",
    "nde":    "non-destructive examination",
    "ndt":    "non-destructive testing",
    "pmi":    "positive material identification",

    # ------------------------------------------------------------------
    # Work management
    # ------------------------------------------------------------------
    "wco":    "work control order",
    "jha":    "job hazard analysis",
    "pjb":    "pre-job brief",
    "pm":     "preventive maintenance",

    # ------------------------------------------------------------------
    # Mechanical hardware components
    # ------------------------------------------------------------------
    "sft":    "shaft",
    "cplg":   "coupling",
    "gskt":   "gasket",
    "brkt":   "bracket",
    "impllr": "impeller",

    # ------------------------------------------------------------------
    # P6-scheduler shorthand confirmed in benchmark data
    # (tokens still unresolved after Excel + nuclear supplement pass)
    # ------------------------------------------------------------------
    "xmtr":   "transmitter",        # pressure/temperature transmitter
    "fdtr":   "feedwater",          # feedwater (often "fdwtr" also seen)
    "fdwtr":  "feedwater",
    "mn":     "main",               # "mn fdtr vlv" → "main feedwater valve"
    "vlv":    "valve",              # generic valve shorthand
    "clnt":   "coolant",            # typo-adjacent shorthand for coolant
    "cal":    "calibrate",          # calibrate (Excel has "calibration")
    "rpl":    "replace",            # alternate to "repl"
    "si":     "safety injection",   # "SI SYS" → "safety injection system"
    "aux":    "auxiliary",          # auxiliary building / auxiliary feedwater
    "edgr":   "emergency diesel generator",  # variant of "edg"
    "ts":     "technical specification",     # "per TS surveillance"
    "ot":     "outage",             # outage (telegraphic style)
    "wo":     "work order",         # work order (shorter than "wco")
    "surv":   "surveillance",       # technical-specification surveillance
    "func":   "functional",         # functional test
    "maint":  "maintenance",        # generic maintenance
    "prev":   "preventive",         # preventive maintenance
    "corr":   "corrective",         # corrective maintenance
    "ops":    "operations",         # operations department
    "mech":   "mechanical",         # mechanical discipline
    "elec":   "electrical",         # electrical discipline
    "instru": "instrumentation",    # instrumentation
    "i&c":    "instrumentation and controls",
}
