"""
cap_config — CAPExportConfig dataclass.

Holds all field-mapping and target-system configuration for the CAP export
serializer.  The bundled ``field_maps/{target_system}_default.json`` file is
the authoritative source for every mapping key (action_type_map, priority_map,
record_endpoint, floc_property, short_description_max_chars, long_text_field);
the module-level Python constants are used only as a fallback when a key (or
the whole file) is missing.  Individual keys can still be overridden per plant
via constructor keyword arguments.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

_FIELD_MAPS_DIR = Path(__file__).parent / "field_maps"

_SUPPORTED_TARGETS = ("maximo", "sap_pm", "generic")

_DEFAULT_ACTION_TYPE_MAP_MAXIMO: Dict[str, str] = {
    "immediate_corrective":   "CAL",
    "long_term_corrective":   "CAP",
    "preventive":             "PM",
    "monitoring":             "SR",
    "procedure_update":       "TQ",
    "engineering_evaluation": "ECR",
    "pm_corrective":          "CM",
}

_DEFAULT_ACTION_TYPE_MAP_SAP: Dict[str, str] = {
    "immediate_corrective":   "M1",
    "long_term_corrective":   "M2",
    "preventive":             "M3",
    "monitoring":             "M4",
    "procedure_update":       "Q3",
    "engineering_evaluation": "Q1",
    "pm_corrective":          "M2",
}

_DEFAULT_ACTION_TYPE_MAP_GENERIC: Dict[str, str] = {
    "immediate_corrective":   "CORRECTIVE",
    "long_term_corrective":   "CORRECTIVE_LT",
    "preventive":             "PREVENTIVE",
    "monitoring":             "MONITORING",
    "procedure_update":       "PROCEDURE",
    "engineering_evaluation": "ENGINEERING",
    "pm_corrective":          "CORRECTIVE_PM",
}

_DEFAULT_PRIORITY_MAP: Dict[str, str] = {
    "critical": "1",
    "high":     "2",
    "medium":   "3",
    "low":      "4",
}

_SHORT_DESC_LIMITS: Dict[str, int] = {
    "maximo": 100,
    "sap_pm": 40,
    "generic": 200,
}


@dataclass
class CAPExportConfig:
    """
    Configuration for CAPExportSerializer.

    On construction the bundled ``field_maps/{target_system}_default.json``
    (or ``field_map_path`` when given) is parsed as the authoritative source
    for the mapping keys; any key absent from the JSON falls back to the
    module-level Python constant, and any value passed explicitly to the
    constructor overrides both.

    Parameters
    ----------
    target_system:
        ``"maximo"`` | ``"sap_pm"`` | ``"generic"``.  Controls which
        field-map defaults are loaded and which system-specific extension
        block (``maximo_ext`` / ``sap_ext``) is populated.  Any other value
        raises ``ValueError``.
    action_type_map:
        Overrides merged *over* the JSON/default map for ``target_system``.
        Example: ``{"monitoring": "PM"}`` to treat monitoring actions as
        PM work orders in a plant with that convention.
    priority_map:
        Overrides merged over the JSON/default ``{"critical": "1", ...}`` map.
    default_work_group:
        Maximo: stamped on every CRRecord as ``maximo_ext.work_group``.
    default_plant_section:
        SAP PM: stamped on every CRRecord as ``sap_ext.plant_section``.
    default_planner_group:
        SAP PM: stamped on every CRRecord as ``sap_ext.planner_group``.
    long_text_header:
        Custom prefix *prepended* to the standard DACKAR long-text header.
        If ``None``, only the standard header is emitted.
    include_rca_run_id_in_description:
        If ``True`` (default), prepends ``"[RCA:{run_id}]"`` to the
        short description, creating a searchable token in the CMMS.
    field_map_path:
        Path to a custom field-map JSON file.  If ``None`` (default) the
        bundled ``field_maps/{target_system}_default.json`` is used.
    record_endpoint, floc_property, short_description_max_chars,
    long_text_field:
        Parsed from the field-map JSON when not passed explicitly.  Exposed
        for adapters (``record_endpoint``, ``long_text_field``) and the
        serializer (``floc_property``, ``short_description_max_chars``).

    Attributes
    ----------
    field_map_source:
        Provenance string identifying the field-map source and a content
        hash, e.g. ``"field_maps/maximo_default.json#sha256:ab12cd34ef56"``.
    """

    target_system: str = "maximo"
    action_type_map: Dict[str, str] = field(default_factory=dict)
    priority_map: Dict[str, str] = field(default_factory=dict)
    default_work_group: Optional[str] = None
    default_plant_section: Optional[str] = None
    default_planner_group: Optional[str] = None
    long_text_header: Optional[str] = None
    include_rca_run_id_in_description: bool = True
    field_map_path: Optional[Path] = None
    record_endpoint: Optional[str] = None
    floc_property: Optional[str] = None
    short_description_max_chars: Optional[int] = None
    long_text_field: Optional[str] = None
    field_map_source: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.target_system not in _SUPPORTED_TARGETS:
            raise ValueError(
                f"Unsupported target_system {self.target_system!r}; "
                f"expected one of {list(_SUPPORTED_TARGETS)}."
            )
        self._field_map_data, self.field_map_source = self._load_field_map()

        fm = self._field_map_data
        if self.record_endpoint is None:
            self.record_endpoint = fm.get("record_endpoint")
        if self.floc_property is None:
            self.floc_property = fm.get("floc_property") or (
                "sap_equipment_id" if self.target_system == "sap_pm" else "maximo_floc"
            )
        if self.short_description_max_chars is None:
            raw = fm.get("short_description_max_chars")
            self.short_description_max_chars = (
                int(raw) if raw is not None else _SHORT_DESC_LIMITS.get(self.target_system, 200)
            )
        if self.long_text_field is None:
            self.long_text_field = fm.get("long_text_field") or "description"

    def _load_field_map(self) -> tuple:
        """
        Return ``(field_map_data, field_map_source)``.

        ``field_map_data`` is the parsed JSON dict (empty when no file is
        found for ``generic``); ``field_map_source`` is a provenance string
        with a content hash.  An explicitly supplied ``field_map_path`` that
        does not exist raises ``FileNotFoundError``.
        """
        if self.field_map_path is not None:
            path = Path(self.field_map_path)
            if not path.exists():
                raise FileNotFoundError(f"field_map_path does not exist: {path}")
        else:
            path = _FIELD_MAPS_DIR / f"{self.target_system}_default.json"
            if not path.exists():
                # generic has no bundled file — constants-only fallback.
                return {}, f"builtin_defaults:{self.target_system}"

        raw = path.read_text()
        data = json.loads(raw)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        try:
            loc = f"field_maps/{path.relative_to(_FIELD_MAPS_DIR)}"
        except ValueError:
            loc = str(path)
        return data, f"{loc}#sha256:{digest}"

    def _constant_action_type_map(self) -> Dict[str, str]:
        if self.target_system == "maximo":
            return _DEFAULT_ACTION_TYPE_MAP_MAXIMO
        if self.target_system == "sap_pm":
            return _DEFAULT_ACTION_TYPE_MAP_SAP
        return _DEFAULT_ACTION_TYPE_MAP_GENERIC

    def resolved_action_type_map(self) -> Dict[str, str]:
        """Return the effective action_type map (constants < JSON < overrides)."""
        base = dict(self._constant_action_type_map())
        base.update(self._field_map_data.get("action_type_map") or {})
        base.update(self.action_type_map or {})
        return base

    def resolved_priority_map(self) -> Dict[str, str]:
        """Return the effective priority map (constants < JSON < overrides)."""
        base = dict(_DEFAULT_PRIORITY_MAP)
        base.update(self._field_map_data.get("priority_map") or {})
        base.update(self.priority_map or {})
        return base

    def short_description_limit(self) -> int:
        """Return the character limit for the CMMS short description field."""
        return int(self.short_description_max_chars)

    def floc_kg_property(self) -> str:
        """KG property name to read for location resolution."""
        return self.floc_property

    @classmethod
    def from_field_map_file(
        cls,
        target_system: str,
        field_map_path: Optional[Path] = None,
        **overrides,
    ) -> "CAPExportConfig":
        """
        Construct a config whose field-map source is a JSON file.

        Parameters
        ----------
        target_system:
            ``"maximo"`` | ``"sap_pm"`` | ``"generic"``
        field_map_path:
            Path to a custom JSON file.  If ``None``, the bundled
            ``field_maps/{target_system}_default.json`` is used.
        **overrides:
            Additional keyword arguments passed to ``CAPExportConfig()``.
            May include ``action_type_map`` / ``priority_map`` (merged over
            the JSON) and any scalar field without collision, because the
            JSON is loaded in ``__post_init__`` rather than through the
            constructor signature.
        """
        return cls(target_system=target_system, field_map_path=field_map_path, **overrides)
