"""
cap_config — CAPExportConfig dataclass.

Holds all field-mapping and target-system configuration for the CAP export
serializer.  Load defaults from a field_maps/*.json file and override
individual keys per plant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

_FIELD_MAPS_DIR = Path(__file__).parent / "field_maps"

_DEFAULT_ACTION_TYPE_MAP_MAXIMO: Dict[str, str] = {
    "immediate_corrective":   "CAL",
    "long_term_corrective":   "CAP",
    "preventive":             "PM",
    "monitoring":             "SR",
    "procedure_update":       "TQ",
    "engineering_evaluation": "ECR",
}

_DEFAULT_ACTION_TYPE_MAP_SAP: Dict[str, str] = {
    "immediate_corrective":   "M1",
    "long_term_corrective":   "M2",
    "preventive":             "M3",
    "monitoring":             "M4",
    "procedure_update":       "Q3",
    "engineering_evaluation": "Q1",
}

_DEFAULT_ACTION_TYPE_MAP_GENERIC: Dict[str, str] = {
    "immediate_corrective":   "CORRECTIVE",
    "long_term_corrective":   "CORRECTIVE_LT",
    "preventive":             "PREVENTIVE",
    "monitoring":             "MONITORING",
    "procedure_update":       "PROCEDURE",
    "engineering_evaluation": "ENGINEERING",
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

    Parameters
    ----------
    target_system:
        ``"maximo"`` | ``"sap_pm"`` | ``"generic"``.
        Controls which field-map defaults are loaded and which
        system-specific extension block (``maximo_ext`` / ``sap_ext``)
        is populated.
    action_type_map:
        Overrides merged *over* the default map for ``target_system``.
        Example: ``{"monitoring": "PM"}`` to treat monitoring actions as
        PM work orders in a plant with that convention.
    priority_map:
        Overrides merged over the default ``{"critical": "1", ...}`` map.
    default_work_group:
        Maximo: stamped on every CRRecord as ``maximo_ext.work_group``.
    default_plant_section:
        SAP PM: stamped on every CRRecord as ``sap_ext.plant_section``.
    default_planner_group:
        SAP PM: stamped on every CRRecord as ``sap_ext.planner_group``.
    long_text_header:
        Custom prefix for the long-text narrative block.  If ``None``,
        the standard DACKAR header is used.
    include_rca_run_id_in_description:
        If ``True`` (default), prepends ``"[RCA:{run_id}]"`` to the
        short description, creating a searchable token in the CMMS.
    """

    target_system: str = "maximo"
    action_type_map: Dict[str, str] = field(default_factory=dict)
    priority_map: Dict[str, str] = field(default_factory=dict)
    default_work_group: Optional[str] = None
    default_plant_section: Optional[str] = None
    default_planner_group: Optional[str] = None
    long_text_header: Optional[str] = None
    include_rca_run_id_in_description: bool = True

    def resolved_action_type_map(self) -> Dict[str, str]:
        """Return the effective action_type map (defaults + overrides)."""
        if self.target_system == "maximo":
            base = dict(_DEFAULT_ACTION_TYPE_MAP_MAXIMO)
        elif self.target_system == "sap_pm":
            base = dict(_DEFAULT_ACTION_TYPE_MAP_SAP)
        else:
            base = dict(_DEFAULT_ACTION_TYPE_MAP_GENERIC)
        base.update(self.action_type_map)
        return base

    def resolved_priority_map(self) -> Dict[str, str]:
        """Return the effective priority map (defaults + overrides)."""
        base = dict(_DEFAULT_PRIORITY_MAP)
        base.update(self.priority_map)
        return base

    def short_description_limit(self) -> int:
        """Return the character limit for the CMMS short description field."""
        return _SHORT_DESC_LIMITS.get(self.target_system, 200)

    def floc_kg_property(self) -> str:
        """KG property name to read for location resolution."""
        if self.target_system == "sap_pm":
            return "sap_equipment_id"
        return "maximo_floc"

    @classmethod
    def from_field_map_file(
        cls,
        target_system: str,
        field_map_path: Optional[Path] = None,
        **overrides,
    ) -> "CAPExportConfig":
        """
        Load defaults from a field_maps JSON file and merge keyword overrides.

        Parameters
        ----------
        target_system:
            ``"maximo"`` | ``"sap_pm"`` | ``"generic"``
        field_map_path:
            Path to a custom JSON file.  If ``None``, the bundled
            ``field_maps/{target_system}_default.json`` is used.
        **overrides:
            Additional keyword arguments passed to ``CAPExportConfig()``.
        """
        if field_map_path is None:
            field_map_path = _FIELD_MAPS_DIR / f"{target_system}_default.json"

        action_type_map: Dict[str, str] = {}
        priority_map: Dict[str, str] = {}

        if field_map_path.exists():
            data = json.loads(field_map_path.read_text())
            action_type_map = data.get("action_type_map", {})
            priority_map = data.get("priority_map", {})

        return cls(
            target_system=target_system,
            action_type_map=action_type_map,
            priority_map=priority_map,
            **overrides,
        )
