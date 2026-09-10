"""
cap_export_serializer — CAPExportSerializer.

Maps a completed (and analyst-approved) rca_card + kg_context into a
CAPExportPackage conforming to schemas/cap_export_package.json.

FLOC resolution uses Option B (KG-augmented): component CMMS IDs are stored
as optional properties on KG element_usage nodes (maximo_floc /
sap_equipment_id) and are returned in kg_context.components[].  No additional
KG query is required at export time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cap_integration.cap_config import CAPExportConfig

JsonDict = Dict[str, Any]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CAPExportSerializer:
    """
    Serializes an approved RCA card into a CAPExportPackage.

    Parameters
    ----------
    config:
        ``CAPExportConfig`` controlling target system, field maps, and
        formatting options.  Defaults to Maximo with standard mappings.

    Usage::

        serializer = CAPExportSerializer(CAPExportConfig(target_system="maximo"))
        package = serializer.serialize(
            rca_card=modified_card,
            kg_context=kg_context,
            run_id="run-001",
            override_id="OVRD::EVT-001::...",
        )
    """

    def __init__(self, config: Optional[CAPExportConfig] = None) -> None:
        self.config = config or CAPExportConfig()
        self._action_type_map = self.config.resolved_action_type_map()
        self._priority_map = self.config.resolved_priority_map()
        self._floc_property = self.config.floc_kg_property()
        self._short_desc_limit = self.config.short_description_limit()

    def serialize(
        self,
        rca_card: JsonDict,
        kg_context: JsonDict,
        run_id: str,
        override_id: Optional[str] = None,
    ) -> JsonDict:
        """
        Build and return a CAPExportPackage dict.

        Parameters
        ----------
        rca_card:
            The analyst-approved RCA card (post ``apply_override()``).
        kg_context:
            The kg_context artifact from the same RCA run.  Used for
            FLOC/equipment_id resolution via components[].maximo_floc or
            components[].sap_equipment_id.
        run_id:
            RCA run identifier.
        override_id:
            ``override_id`` from the AnalystOverride record, if available.

        Returns
        -------
        dict
            Conforms to ``schemas/cap_export_package.json``.

        Raises
        ------
        ValueError
            If ``rca_card.analyst_review.writeback_recommendation`` is not
            ``"ready_if_accepted"`` — export must only be called on an
            approved card.
        """
        self._assert_card_approved(rca_card)

        event_id = rca_card.get("event_id") or "unknown"
        asset_id = rca_card.get("asset_id")
        generated_at = _utcnow_iso()
        export_id = f"CAPEXP::{event_id}::{generated_at}"

        floc_index = self._build_floc_index(kg_context)

        cr_records: List[JsonDict] = []
        unresolved_locations: List[str] = []

        for action in rca_card.get("recommended_actions") or []:
            if not isinstance(action, dict):
                continue
            record, unresolved = self._serialize_action(
                action=action,
                export_id=export_id,
                rca_card=rca_card,
                run_id=run_id,
                floc_index=floc_index,
            )
            cr_records.append(record)
            if unresolved:
                unresolved_locations.append(unresolved)

        return {
            "export_id": export_id,
            "run_id": run_id,
            "event_id": event_id,
            "asset_id": asset_id,
            "generated_at": generated_at,
            "target_system": self.config.target_system,
            "cr_records": cr_records,
            "unresolved_locations": sorted(set(unresolved_locations)),
            "provenance": {
                "generated_by": "CAPExportSerializer",
                "rca_card_id": rca_card.get("rca_id") or rca_card.get("event_id") or "unknown",
                "override_id": override_id,
                "pipeline_version": (rca_card.get("provenance") or {}).get("pipeline_version"),
                "field_map_source": f"{self.config.target_system}_default",
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_card_approved(self, rca_card: JsonDict) -> None:
        rec = (rca_card.get("analyst_review") or {}).get("writeback_recommendation")
        if rec != "ready_if_accepted":
            raise ValueError(
                f"CAP export requires writeback_recommendation == 'ready_if_accepted', "
                f"got {rec!r}. Call apply_override() with writeback_decision='accept' first."
            )

    def _build_floc_index(self, kg_context: JsonDict) -> Dict[str, Optional[str]]:
        """
        Build component_id → FLOC/equipment_id lookup from kg_context.components[].

        Returns a dict where a value of None means the KG property was absent.
        """
        index: Dict[str, Optional[str]] = {}
        for comp in kg_context.get("components") or []:
            if not isinstance(comp, dict):
                continue
            cid = comp.get("component_id")
            if cid:
                index[cid] = comp.get(self._floc_property) or None
        return index

    def _serialize_action(
        self,
        action: JsonDict,
        export_id: str,
        rca_card: JsonDict,
        run_id: str,
        floc_index: Dict[str, Optional[str]],
    ) -> tuple:
        """
        Returns (cr_record_dict, unresolved_component_id_or_None).
        """
        action_id = action.get("action_id") or "ACT-UNKNOWN"
        action_type = action.get("action_type") or ""
        priority = action.get("priority") or "low"
        description = action.get("description") or ""
        rationale = action.get("rationale") or ""
        expected_obs = action.get("expected_observation_if_true") or ""
        target_comp = action.get("target_component_id")
        owner = action.get("owner")
        linked_candidate = action.get("linked_candidate_id")

        cr_type = self._action_type_map.get(action_type, action_type.upper())
        priority_code = self._priority_map.get(priority, priority)

        short_desc = self._build_short_description(
            description=description,
            run_id=run_id,
            action_id=action_id,
        )

        long_text = self._build_long_text(
            rca_card=rca_card,
            run_id=run_id,
            action=action,
        )

        # FLOC resolution (Option B — KG-augmented)
        floc_value: Optional[str] = None
        mapping_status = "unresolved"
        unresolved_comp: Optional[str] = None

        if target_comp:
            if target_comp in floc_index:
                floc_value = floc_index[target_comp]
                mapping_status = "resolved" if floc_value else "unresolved"
            if not floc_value:
                unresolved_comp = target_comp

        # Build system-specific location fields
        if self.config.target_system == "sap_pm":
            functional_location = None
            equipment_id = floc_value
        else:
            functional_location = floc_value
            equipment_id = None

        record: JsonDict = {
            "export_record_id": f"{export_id}::{action_id}",
            "source_action_id": action_id,
            "action_type": action_type,
            "cr_type": cr_type,
            "short_description": short_desc,
            "long_text": long_text,
            "priority": priority,
            "priority_code": priority_code,
            "functional_location": functional_location,
            "equipment_id": equipment_id,
            "target_component_id": target_comp,
            "mapping_status": mapping_status,
            "owner": owner,
            "linked_candidate_id": linked_candidate,
            "maximo_ext": self._build_maximo_ext(),
            "sap_ext": self._build_sap_ext(),
        }

        return record, unresolved_comp

    def _build_short_description(
        self,
        description: str,
        run_id: str,
        action_id: str,
    ) -> str:
        limit = self._short_desc_limit
        if self.config.include_rca_run_id_in_description:
            prefix = f"[RCA:{run_id}] "
            available = limit - len(prefix)
            body = description[:available] if available > 0 else ""
            return (prefix + body).strip()
        return description[:limit].strip()

    def _build_long_text(self, rca_card: JsonDict, run_id: str, action: JsonDict) -> str:
        event_id = rca_card.get("event_id") or ""
        asset_id = rca_card.get("asset_id") or ""
        primary = rca_card.get("primary_hypothesis") or {}
        primary_label = primary.get("cause_label") or ""
        pipeline_version = (rca_card.get("provenance") or {}).get("pipeline_version") or ""

        header = self.config.long_text_header or (
            f"[RCA Run: {run_id}]\n"
            f"[Event: {event_id}] [Asset: {asset_id}]\n"
            f"[Primary Cause: {primary_label}]\n"
        )

        parts = [header]
        if action.get("description"):
            parts.append(f"Action: {action['description']}")
        if action.get("rationale"):
            parts.append(f"\nRationale: {action['rationale']}")
        if action.get("expected_observation_if_true"):
            parts.append(
                f"\nExpected observation if true: {action['expected_observation_if_true']}"
            )
        parts.append(
            f"\n--- Generated by DACKAR RCA {pipeline_version} ---"
        )
        return "\n".join(parts)

    def _build_maximo_ext(self) -> JsonDict:
        if self.config.target_system != "maximo":
            return {}
        ext: JsonDict = {}
        if self.config.default_work_group:
            ext["work_group"] = self.config.default_work_group
        return ext

    def _build_sap_ext(self) -> JsonDict:
        if self.config.target_system != "sap_pm":
            return {}
        ext: JsonDict = {}
        if self.config.default_plant_section:
            ext["plant_section"] = self.config.default_plant_section
        if self.config.default_planner_group:
            ext["planner_group"] = self.config.default_planner_group
        return ext
