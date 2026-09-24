"""
cap_export_serializer — CAPExportSerializer.

Maps a completed (and analyst-accepted) rca_card + kg_context into a
CAPExportPackage conforming to schemas/cap_export_package.json.

Export is gated on an ``AnalystOverride`` record (see schemas/analyst_override.json)
whose ``writeback_decision == "accept"`` — the recommendation flag on the card
alone is not treated as proof of acceptance.  The resulting ``export_id`` is
derived from ``run_id`` + the override's ``override_id`` so regenerating the
package for the same accepted decision is idempotent.

FLOC resolution uses Option B (KG-augmented): component CMMS IDs are stored
as optional properties on KG element_usage nodes (maximo_floc /
sap_equipment_id) and are returned in kg_context.components[].  No additional
KG query is required at export time.

The serializer self-validates its output against schemas/cap_export_package.json
and raises on any schema violation (fail-closed); deep validation of the input
rca_card / kg_context remains the orchestrator's responsibility.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cap_config import CAPExportConfig

JsonDict = Dict[str, Any]

_OUTPUT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "cap_export_package.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=1)
def _output_validator():
    """Return a cached jsonschema validator for the CAPExportPackage schema."""
    import jsonschema

    schema = json.loads(_OUTPUT_SCHEMA_PATH.read_text())
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


class CAPExportSerializer:
    """
    Serializes an analyst-accepted RCA card into a CAPExportPackage.

    Parameters
    ----------
    config:
        ``CAPExportConfig`` controlling target system, field maps, and
        formatting options.  Defaults to Maximo with standard mappings.

    Usage::

        serializer = CAPExportSerializer(CAPExportConfig(target_system="maximo"))
        modified_card, override_record = orchestrator.apply_override(...)
        package = serializer.serialize(
            rca_card=modified_card,
            kg_context=kg_context,
            run_id="run-001",
            override_record=override_record,
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
        override_record: JsonDict,
    ) -> JsonDict:
        """
        Build and return a CAPExportPackage dict.

        Parameters
        ----------
        rca_card:
            The analyst-accepted RCA card (post ``apply_override()``).
        kg_context:
            The kg_context artifact from the same RCA run.  Its ``asset_id``
            is required, and ``components[]`` are used for FLOC/equipment_id
            resolution via ``maximo_floc`` / ``sap_equipment_id``.
        run_id:
            RCA run identifier.
        override_record:
            The ``AnalystOverride`` record returned by ``apply_override()``.
            Must carry ``writeback_decision == "accept"`` and an
            ``override_id``; both gate the export and seed the stable
            ``export_id``.

        Returns
        -------
        dict
            Conforms to ``schemas/cap_export_package.json``.

        Raises
        ------
        ValueError
            If the override record does not represent an accepted writeback,
            if required identifiers (asset_id / event_id / override_id) are
            missing or inconsistent across artifacts, if an ``action_type``
            has no configured mapping, or if the built package fails schema
            validation.
        """
        override_id = self._assert_override_accepted(override_record)
        self._assert_card_approved(rca_card)

        asset_id = kg_context.get("asset_id")
        if not asset_id:
            raise ValueError(
                "kg_context.asset_id is required to build a CAP export package "
                "(Option B FLOC resolution reads asset/component IDs from kg_context)."
            )

        card_event_id = rca_card.get("event_id")
        ovr_event_id = override_record.get("event_id")
        event_id = card_event_id or ovr_event_id
        if not event_id:
            raise ValueError("event_id missing from both rca_card and override_record.")
        if card_event_id and ovr_event_id and card_event_id != ovr_event_id:
            raise ValueError(
                f"event_id mismatch: rca_card={card_event_id!r} vs "
                f"override_record={ovr_event_id!r}; artifacts are from different runs."
            )
        ovr_asset_id = override_record.get("asset_id")
        if ovr_asset_id and ovr_asset_id != asset_id:
            raise ValueError(
                f"asset_id mismatch: kg_context={asset_id!r} vs "
                f"override_record={ovr_asset_id!r}."
            )

        generated_at = _utcnow_iso()
        export_id = f"CAPEXP::{run_id}::{override_id}"

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

        package = {
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
                "field_map_source": self.config.field_map_source,
            },
        }

        self._validate_output(package)
        return package

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_override_accepted(self, override_record: JsonDict) -> str:
        """
        Verify the override record represents an accepted writeback and return
        its ``override_id``.  This is the authoritative acceptance gate.
        """
        if not isinstance(override_record, dict):
            raise ValueError(
                f"override_record must be an AnalystOverride dict, got "
                f"{type(override_record).__name__}."
            )
        decision = override_record.get("writeback_decision")
        if decision != "accept":
            raise ValueError(
                f"CAP export requires an override_record with "
                f"writeback_decision == 'accept', got {decision!r}. "
                f"Call apply_override() with writeback_decision='accept' first."
            )
        override_id = override_record.get("override_id")
        if not override_id:
            raise ValueError("override_record.override_id is required to build a stable export_id.")
        return override_id

    def _assert_card_approved(self, rca_card: JsonDict) -> None:
        """Secondary guard: the pipeline must have flagged the card ready."""
        rec = (rca_card.get("analyst_review") or {}).get("writeback_recommendation")
        if rec != "ready_if_accepted":
            raise ValueError(
                f"CAP export requires writeback_recommendation == 'ready_if_accepted', "
                f"got {rec!r}."
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
        Returns ``(cr_record_dict, unresolved_location_token_or_None)``.
        """
        action_id = action.get("action_id") or "ACT-UNKNOWN"
        action_type = action.get("action_type") or ""
        priority = action.get("priority") or "low"
        description = action.get("description") or ""
        target_comp = action.get("target_component_id")
        owner = action.get("owner")
        linked_candidate = action.get("linked_candidate_id")

        if action_type not in self._action_type_map:
            raise ValueError(
                f"action_type {action_type!r} (action {action_id}) has no mapping for "
                f"target_system {self.config.target_system!r}. Add it to the field map "
                f"(field_maps/{self.config.target_system}_default.json) or "
                f"CAPExportConfig.action_type_map."
            )
        cr_type = self._action_type_map[action_type]
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
        unresolved: Optional[str] = None

        if target_comp:
            floc_value = floc_index.get(target_comp)
            if floc_value:
                mapping_status = "resolved"
            else:
                # target component either absent from kg_context or has no FLOC.
                unresolved = target_comp
        else:
            # I10: an action with no target_component_id must still be surfaced.
            unresolved = f"(no target_component_id for action {action_id})"

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

        return record, unresolved

    def _build_short_description(
        self,
        description: str,
        run_id: str,
        action_id: str,
    ) -> str:
        """
        Build a short description guaranteed to be ``<= short_description_limit``.

        When enabled, a stable ``"[RCA:{run_id}] "`` trace token is prepended;
        if the token alone would exceed the limit it is truncated so the return
        value never overflows the CMMS field.
        """
        limit = self._short_desc_limit
        if self.config.include_rca_run_id_in_description:
            token = f"[RCA:{run_id}] "
            if len(token) >= limit:
                return token[:limit].strip()
            body = description[: limit - len(token)]
            return (token + body)[:limit].strip()
        return description[:limit].strip()

    def _build_long_text(self, rca_card: JsonDict, run_id: str, action: JsonDict) -> str:
        event_id = rca_card.get("event_id") or ""
        asset_id = rca_card.get("asset_id") or ""
        primary = rca_card.get("primary_hypothesis") or {}
        primary_label = primary.get("cause_label") or ""
        pipeline_version = (rca_card.get("provenance") or {}).get("pipeline_version") or ""

        standard_header = (
            f"[RCA Run: {run_id}]\n"
            f"[Event: {event_id}] [Asset: {asset_id}]\n"
            f"[Primary Cause: {primary_label}]\n"
        )
        # I8: a custom header is *prepended* to (not a replacement for) the block.
        if self.config.long_text_header:
            header = f"{self.config.long_text_header}\n{standard_header}"
        else:
            header = standard_header

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

    def _validate_output(self, package: JsonDict) -> None:
        """
        Validate ``package`` against schemas/cap_export_package.json (fail-closed).

        Raises ``ValueError`` with the first offending JSON path and message
        when the built package does not conform.
        """
        validator = _output_validator()
        errors = sorted(validator.iter_errors(package), key=lambda e: list(e.path))
        if errors:
            first = errors[0]
            loc = "/".join(str(p) for p in first.path) or "<root>"
            raise ValueError(
                f"CAPExportPackage failed schema validation at {loc}: "
                f"{first.message} ({len(errors)} error(s) total)."
            )
