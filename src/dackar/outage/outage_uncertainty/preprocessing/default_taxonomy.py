"""
Built-in nuclear-outage taxonomy vocabulary for :class:`TaskLabelMapper`.

Entries map a keyword (lowercase, word-boundary matched) to a partial or
complete set of the three label dimensions:

* ``discipline``       – ``mechanical | electrical | I&C | civil | nuclear | operations``
* ``task_family``      – ``inspection | replacement | maintenance | calibration |
                          surveillance | testing | modification | refurbishment |
                          cleaning | lubrication | disassembly | restoration``
* ``component_family`` – ``valve | pump | motor | heat_exchanger | pipe |
                          instrument | cable | filter | fan | compressor |
                          reactor | turbine | generator | transformer |
                          breaker | switchgear | relay | sensor | actuator |
                          structure | condenser``

Ordering is intentional: single-word (less specific) entries come first;
multi-word phrases come last.  Because :class:`TaskLabelMapper` iterates in
insertion order and later matches overwrite earlier ones (last-wins), phrase
entries take precedence over their constituent single-word entries, which is
the desired behaviour for specificity.

User-supplied ``taxonomy_rules`` in :class:`AppConfig` are merged *on top* of
these defaults and therefore always win.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Single-word component keywords  (discipline + component_family)
# ---------------------------------------------------------------------------
_COMPONENT_SINGLE: dict[str, dict[str, str]] = {
    "valve":        {"discipline": "mechanical",  "component_family": "valve"},
    "damper":       {"discipline": "mechanical",  "component_family": "valve"},
    "pump":         {"discipline": "mechanical",  "component_family": "pump"},
    "impeller":     {"discipline": "mechanical",  "component_family": "pump"},
    "motor":        {"discipline": "electrical",  "component_family": "motor"},
    "exchanger":    {"discipline": "mechanical",  "component_family": "heat_exchanger"},
    "condenser":    {"discipline": "mechanical",  "component_family": "condenser"},
    "cooler":       {"discipline": "mechanical",  "component_family": "heat_exchanger"},
    "heater":       {"discipline": "mechanical",  "component_family": "heat_exchanger"},
    "pipe":         {"discipline": "mechanical",  "component_family": "pipe"},
    "piping":       {"discipline": "mechanical",  "component_family": "pipe"},
    "header":       {"discipline": "mechanical",  "component_family": "pipe"},
    "nozzle":       {"discipline": "mechanical",  "component_family": "pipe"},
    "transmitter":  {"discipline": "I&C",         "component_family": "instrument"},
    "indicator":    {"discipline": "I&C",         "component_family": "instrument"},
    "controller":   {"discipline": "I&C",         "component_family": "instrument"},
    "recorder":     {"discipline": "I&C",         "component_family": "instrument"},
    "sensor":       {"discipline": "I&C",         "component_family": "sensor"},
    "detector":     {"discipline": "I&C",         "component_family": "sensor"},
    "probe":        {"discipline": "I&C",         "component_family": "sensor"},
    "thermocouple": {"discipline": "I&C",         "component_family": "sensor"},
    "actuator":     {"discipline": "mechanical",  "component_family": "actuator"},
    "cable":        {"discipline": "electrical",  "component_family": "cable"},
    "wiring":       {"discipline": "electrical",  "component_family": "cable"},
    "conduit":      {"discipline": "electrical",  "component_family": "cable"},
    "filter":       {"discipline": "mechanical",  "component_family": "filter"},
    "strainer":     {"discipline": "mechanical",  "component_family": "filter"},
    "screen":       {"discipline": "mechanical",  "component_family": "filter"},
    "fan":          {"discipline": "mechanical",  "component_family": "fan"},
    "blower":       {"discipline": "mechanical",  "component_family": "fan"},
    "compressor":   {"discipline": "mechanical",  "component_family": "compressor"},
    "reactor":      {"discipline": "nuclear",     "component_family": "reactor"},
    "turbine":      {"discipline": "mechanical",  "component_family": "turbine"},
    "generator":    {"discipline": "electrical",  "component_family": "generator"},
    "transformer":  {"discipline": "electrical",  "component_family": "transformer"},
    "breaker":      {"discipline": "electrical",  "component_family": "breaker"},
    "switchgear":   {"discipline": "electrical",  "component_family": "switchgear"},
    "relay":        {"discipline": "electrical",  "component_family": "relay"},
    "containment":  {"discipline": "civil",       "component_family": "structure"},
    "sump":         {"discipline": "civil",       "component_family": "structure"},
    "building":     {"discipline": "civil",       "component_family": "structure"},
    "structure":    {"discipline": "civil",       "component_family": "structure"},
}

# ---------------------------------------------------------------------------
# Single-word task keywords  (task_family only)
# ---------------------------------------------------------------------------
_TASK_SINGLE: dict[str, dict[str, str]] = {
    # inspection / surveillance
    "inspect":       {"task_family": "inspection"},
    "inspection":    {"task_family": "inspection"},
    "examine":       {"task_family": "inspection"},
    "walkdown":      {"task_family": "inspection"},
    "surveillance":  {"task_family": "surveillance"},
    "walkthrough":   {"task_family": "surveillance"},
    # replacement / installation
    "replace":       {"task_family": "replacement"},
    "replacement":   {"task_family": "replacement"},
    "install":       {"task_family": "replacement"},
    "installation":  {"task_family": "replacement"},
    "swap":          {"task_family": "replacement"},
    # calibration
    "calibrate":     {"task_family": "calibration"},
    "calibration":   {"task_family": "calibration"},
    # testing
    "test":          {"task_family": "testing"},
    "testing":       {"task_family": "testing"},
    # maintenance / repair
    "repair":        {"task_family": "maintenance"},
    "maintenance":   {"task_family": "maintenance"},
    "adjust":        {"task_family": "maintenance"},
    "adjustment":    {"task_family": "maintenance"},
    "tighten":       {"task_family": "maintenance"},
    # refurbishment
    "overhaul":      {"task_family": "refurbishment"},
    "refurbish":     {"task_family": "refurbishment"},
    "rebuild":       {"task_family": "refurbishment"},
    "recondition":   {"task_family": "refurbishment"},
    "repack":        {"task_family": "refurbishment"},
    "disassemble":   {"task_family": "disassembly"},
    "disassembly":   {"task_family": "disassembly"},
    "restore":       {"task_family": "restoration"},
    "restoration":   {"task_family": "restoration"},
    # modification
    "modify":        {"task_family": "modification"},
    "modification":  {"task_family": "modification"},
    "upgrade":       {"task_family": "modification"},
    "retrofit":      {"task_family": "modification"},
    # cleaning
    "clean":         {"task_family": "cleaning"},
    "cleaning":      {"task_family": "cleaning"},
    "flush":         {"task_family": "cleaning"},
    "flushing":      {"task_family": "cleaning"},
    "purge":         {"task_family": "cleaning"},
    "decontaminate": {"task_family": "cleaning"},
    "descale":       {"task_family": "cleaning"},
    # lubrication
    "lube":          {"task_family": "lubrication"},
    "lubricate":     {"task_family": "lubrication"},
    "lubrication":   {"task_family": "lubrication"},
    "grease":        {"task_family": "lubrication"},
}

# ---------------------------------------------------------------------------
# Single-word discipline-only keywords
# ---------------------------------------------------------------------------
_DISCIPLINE_SINGLE: dict[str, dict[str, str]] = {
    "radiation":   {"discipline": "nuclear"},
    "radiological":{"discipline": "nuclear"},
    "nuclear":     {"discipline": "nuclear"},
    "fuel":        {"discipline": "nuclear"},
    "electrical":  {"discipline": "electrical"},
    "mechanical":  {"discipline": "mechanical"},
    "structural":  {"discipline": "civil"},
    "civil":       {"discipline": "civil"},
    "instrumentation": {"discipline": "I&C"},
}

# ---------------------------------------------------------------------------
# Multi-word phrase entries  (more specific — come last, so they win)
# ---------------------------------------------------------------------------
_PHRASE: dict[str, dict[str, str]] = {
    # ── Valve sub-types ────────────────────────────────────────────────────
    "check valve":          {"discipline": "mechanical", "component_family": "valve"},
    "relief valve":         {"discipline": "mechanical", "component_family": "valve"},
    "safety valve":         {"discipline": "mechanical", "component_family": "valve"},
    "control valve":        {"discipline": "mechanical", "component_family": "valve"},
    "gate valve":           {"discipline": "mechanical", "component_family": "valve"},
    "ball valve":           {"discipline": "mechanical", "component_family": "valve"},
    "butterfly valve":      {"discipline": "mechanical", "component_family": "valve"},
    "globe valve":          {"discipline": "mechanical", "component_family": "valve"},
    "needle valve":         {"discipline": "mechanical", "component_family": "valve"},
    # MOV / AOV / SOV override the "motor" → electrical single-word entry
    "motor operated valve": {"discipline": "mechanical", "component_family": "valve"},
    "air operated valve":   {"discipline": "mechanical", "component_family": "valve"},
    "solenoid valve":       {"discipline": "mechanical", "component_family": "valve"},
    "pressure relief valve":{"discipline": "mechanical", "component_family": "valve"},
    "safety relief valve":  {"discipline": "mechanical", "component_family": "valve"},
    # ── Pump sub-types ────────────────────────────────────────────────────
    "charging pump":        {"discipline": "mechanical", "component_family": "pump"},
    "service water pump":   {"discipline": "mechanical", "component_family": "pump"},
    "residual heat pump":   {"discipline": "mechanical", "component_family": "pump"},
    "fire pump":            {"discipline": "mechanical", "component_family": "pump"},
    # ── Generator sub-types (diesel = mechanical discipline) ───────────────
    "diesel generator":     {"discipline": "mechanical", "component_family": "generator"},
    "emergency diesel":     {"discipline": "mechanical", "component_family": "generator"},
    # ── Switchgear / MCC ──────────────────────────────────────────────────
    "motor control center": {"discipline": "electrical", "component_family": "switchgear"},
    "circuit breaker":      {"discipline": "electrical", "component_family": "breaker"},
    "power supply":         {"discipline": "electrical", "component_family": "transformer"},
    "battery charger":      {"discipline": "electrical", "component_family": "transformer"},
    # ── Heat exchangers ───────────────────────────────────────────────────
    "heat exchanger":       {"discipline": "mechanical", "component_family": "heat_exchanger"},
    "tube bundle":          {"discipline": "mechanical", "component_family": "heat_exchanger"},
    # ── I&C specific ──────────────────────────────────────────────────────
    "loop check":           {"discipline": "I&C",        "task_family": "calibration"},
    "set point":            {"discipline": "I&C",        "task_family": "calibration"},
    "channel calibration":  {"discipline": "I&C",        "task_family": "calibration",
                             "component_family": "instrument"},
    "loop calibration":     {"discipline": "I&C",        "task_family": "calibration",
                             "component_family": "instrument"},
    # ── Test phrases (more specific than bare "test") ─────────────────────
    "functional test":      {"task_family": "testing"},
    "leak test":            {"task_family": "testing"},
    "hydrostatic test":     {"task_family": "testing"},
    "performance test":     {"task_family": "testing"},
    "in service inspection":{"task_family": "inspection"},
    "visual inspection":    {"task_family": "inspection"},
    # ── Lubrication phrases ───────────────────────────────────────────────
    "oil change":           {"task_family": "lubrication"},
    "grease fitting":       {"task_family": "lubrication"},
    "grease lubrication":   {"task_family": "lubrication"},
}

# ---------------------------------------------------------------------------
# Assembled default vocabulary (insertion order: least specific → most specific)
# ---------------------------------------------------------------------------
DEFAULT_TAXONOMY_RULES: dict[str, dict[str, str]] = {
    **_COMPONENT_SINGLE,
    **_TASK_SINGLE,
    **_DISCIPLINE_SINGLE,
    **_PHRASE,
}
