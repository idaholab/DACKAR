"""
Unit tests for CAPExportSerializer and CAPExportConfig.
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

import pytest
from cap_integration.cap_config import CAPExportConfig
from cap_integration.cap_export_serializer import CAPExportSerializer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rca_card(
    event_id="EVT-001",
    asset_id="ASSET-A",
    actions=None,
    writeback_recommendation="ready_if_accepted",
    primary_label="Bearing wear",
    pipeline_version="v1.0",
):
    if actions is None:
        actions = [_make_action()]
    return {
        "rca_id": "RCA-001",
        "event_id": event_id,
        "asset_id": asset_id,
        "primary_hypothesis": {"cause_label": primary_label},
        "recommended_actions": actions,
        "analyst_review": {"writeback_recommendation": writeback_recommendation},
        "provenance": {"pipeline_version": pipeline_version},
    }


def _make_action(
    action_id="ACT-001",
    action_type="immediate_corrective",
    priority="high",
    description="Replace worn bearing",
    rationale="Vibration data shows 3σ exceedance",
    expected_observation="Vibration levels within spec post-replacement",
    target_component_id="COMP-001",
    owner="maint-team",
    linked_candidate_id="CAND-001",
):
    return {
        "action_id": action_id,
        "action_type": action_type,
        "priority": priority,
        "description": description,
        "rationale": rationale,
        "expected_observation_if_true": expected_observation,
        "target_component_id": target_component_id,
        "owner": owner,
        "linked_candidate_id": linked_candidate_id,
    }


def _make_kg_context(components=None, asset_id="ASSET-A"):
    if components is None:
        components = [
            {
                "component_id": "COMP-001",
                "component_label": "Bearing assembly",
                "maximo_floc": "PLANT-A/SYS-1/BEARING-01",
                "sap_equipment_id": None,
            }
        ]
    return {"asset_id": asset_id, "components": components}


def _make_override(
    override_id="OVR-001",
    writeback_decision="accept",
    event_id=None,
    asset_id=None,
):
    """
    Minimal AnalystOverride record accepted by ``serialize()``.

    ``event_id`` / ``asset_id`` are omitted by default so the cross-artifact
    consistency checks (I3) are not triggered; pass them to exercise the
    mismatch guards.
    """
    rec = {
        "override_id": override_id,
        "writeback_decision": writeback_decision,
    }
    if event_id is not None:
        rec["event_id"] = event_id
    if asset_id is not None:
        rec["asset_id"] = asset_id
    return rec


# ---------------------------------------------------------------------------
# CAPExportConfig tests
# ---------------------------------------------------------------------------

class TestCAPExportConfig:

    def test_defaults(self):
        cfg = CAPExportConfig()
        assert cfg.target_system == "maximo"
        assert cfg.include_rca_run_id_in_description is True

    def test_resolved_action_type_map_maximo(self):
        cfg = CAPExportConfig(target_system="maximo")
        m = cfg.resolved_action_type_map()
        assert m["immediate_corrective"] == "CAL"
        assert m["preventive"] == "PM"
        assert m["pm_corrective"] == "CM"

    def test_resolved_action_type_map_sap(self):
        cfg = CAPExportConfig(target_system="sap_pm")
        m = cfg.resolved_action_type_map()
        assert m["immediate_corrective"] == "M1"
        assert m["preventive"] == "M3"
        assert m["pm_corrective"] == "M2"

    def test_resolved_action_type_map_generic(self):
        cfg = CAPExportConfig(target_system="generic")
        m = cfg.resolved_action_type_map()
        assert m["immediate_corrective"] == "CORRECTIVE"

    def test_action_type_map_override(self):
        cfg = CAPExportConfig(target_system="maximo", action_type_map={"monitoring": "PM"})
        m = cfg.resolved_action_type_map()
        assert m["monitoring"] == "PM"
        assert m["immediate_corrective"] == "CAL"  # other keys unchanged

    def test_priority_map_defaults(self):
        cfg = CAPExportConfig()
        m = cfg.resolved_priority_map()
        assert m["critical"] == "1"
        assert m["low"] == "4"

    def test_priority_map_override(self):
        cfg = CAPExportConfig(priority_map={"critical": "P1"})
        m = cfg.resolved_priority_map()
        assert m["critical"] == "P1"
        assert m["high"] == "2"  # unchanged

    def test_short_description_limit_maximo(self):
        assert CAPExportConfig(target_system="maximo").short_description_limit() == 100

    def test_short_description_limit_sap(self):
        assert CAPExportConfig(target_system="sap_pm").short_description_limit() == 40

    def test_short_description_limit_generic(self):
        assert CAPExportConfig(target_system="generic").short_description_limit() == 200

    def test_floc_kg_property_maximo(self):
        assert CAPExportConfig(target_system="maximo").floc_kg_property() == "maximo_floc"

    def test_floc_kg_property_sap(self):
        assert CAPExportConfig(target_system="sap_pm").floc_kg_property() == "sap_equipment_id"

    def test_unsupported_target_raises(self):
        with pytest.raises(ValueError, match="Unsupported target_system"):
            CAPExportConfig(target_system="oracle_eam")


# ---------------------------------------------------------------------------
# CAPExportSerializer — package structure
# ---------------------------------------------------------------------------

class TestSerializerPackageStructure:

    def setup_method(self):
        self.serializer = CAPExportSerializer(CAPExportConfig(target_system="maximo"))

    def test_required_top_level_keys(self):
        card = _make_rca_card()
        pkg = self.serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        for key in ("export_id", "run_id", "event_id", "asset_id", "generated_at",
                    "target_system", "cr_records", "unresolved_locations", "provenance"):
            assert key in pkg, f"Missing key: {key}"

    def test_export_id_format(self):
        card = _make_rca_card()
        pkg = self.serializer.serialize(
            card, _make_kg_context(), run_id="run-001",
            override_record=_make_override(override_id="OVR-XYZ"),
        )
        # A1: export_id is derived from run_id + override_id (not event_id).
        assert pkg["export_id"] == "CAPEXP::run-001::OVR-XYZ"

    def test_run_id_preserved(self):
        card = _make_rca_card()
        pkg = self.serializer.serialize(
            card, _make_kg_context(), run_id="run-999", override_record=_make_override()
        )
        assert pkg["run_id"] == "run-999"

    def test_event_id_preserved(self):
        card = _make_rca_card(event_id="EVT-007")
        pkg = self.serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        assert pkg["event_id"] == "EVT-007"

    def test_asset_id_from_kg_context(self):
        # I3: asset_id is sourced from kg_context, not the card.
        card = _make_rca_card()
        pkg = self.serializer.serialize(
            card, _make_kg_context(asset_id="PUMP-42"), run_id="run-001",
            override_record=_make_override(),
        )
        assert pkg["asset_id"] == "PUMP-42"

    def test_target_system_in_package(self):
        pkg = self.serializer.serialize(
            _make_rca_card(), _make_kg_context(), run_id="run-001",
            override_record=_make_override(),
        )
        assert pkg["target_system"] == "maximo"

    def test_cr_records_count_matches_actions(self):
        actions = [_make_action("ACT-001"), _make_action("ACT-002"), _make_action("ACT-003")]
        card = _make_rca_card(actions=actions)
        pkg = self.serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        assert len(pkg["cr_records"]) == 3

    def test_provenance_generated_by(self):
        pkg = self.serializer.serialize(
            _make_rca_card(), _make_kg_context(), run_id="run-001",
            override_record=_make_override(),
        )
        assert pkg["provenance"]["generated_by"] == "CAPExportSerializer"

    def test_provenance_override_id(self):
        pkg = self.serializer.serialize(
            _make_rca_card(), _make_kg_context(), run_id="run-001",
            override_record=_make_override(override_id="OVRD::EVT-001::2026-01-01"),
        )
        assert pkg["provenance"]["override_id"] == "OVRD::EVT-001::2026-01-01"


# ---------------------------------------------------------------------------
# CAPExportSerializer — CR record fields
# ---------------------------------------------------------------------------

class TestSerializerCRRecord:

    def setup_method(self):
        self.serializer = CAPExportSerializer(CAPExportConfig(target_system="maximo"))

    def _get_record(self, action=None, kg_context=None):
        action = action or _make_action()
        kg_context = kg_context or _make_kg_context()
        card = _make_rca_card(actions=[action])
        pkg = self.serializer.serialize(
            card, kg_context, run_id="run-001", override_record=_make_override()
        )
        return pkg["cr_records"][0]

    def test_required_cr_record_keys(self):
        rec = self._get_record()
        for key in ("export_record_id", "source_action_id", "action_type", "cr_type",
                    "short_description", "long_text", "priority", "priority_code", "mapping_status"):
            assert key in rec, f"Missing key: {key}"

    def test_source_action_id(self):
        rec = self._get_record(_make_action(action_id="ACT-XYZ"))
        assert rec["source_action_id"] == "ACT-XYZ"

    def test_export_record_id_format(self):
        rec = self._get_record(_make_action(action_id="ACT-001"))
        assert "ACT-001" in rec["export_record_id"]

    def test_action_type_preserved(self):
        rec = self._get_record(_make_action(action_type="preventive"))
        assert rec["action_type"] == "preventive"

    def test_cr_type_mapped_maximo(self):
        rec = self._get_record(_make_action(action_type="immediate_corrective"))
        assert rec["cr_type"] == "CAL"

    def test_cr_type_mapped_preventive(self):
        rec = self._get_record(_make_action(action_type="preventive"))
        assert rec["cr_type"] == "PM"

    def test_cr_type_mapped_pm_corrective(self):
        # I2: pm_corrective is a recognised action_type (was previously
        # silently upper-cased); maximo maps it to "CM".
        rec = self._get_record(_make_action(action_type="pm_corrective"))
        assert rec["cr_type"] == "CM"

    def test_unmapped_action_type_raises(self):
        # I2: an unknown action_type must raise, not pass through silently.
        card = _make_rca_card(actions=[_make_action(action_type="bogus_type")])
        with pytest.raises(ValueError, match="no mapping"):
            self.serializer.serialize(
                card, _make_kg_context(), run_id="run-001", override_record=_make_override()
            )

    def test_priority_preserved(self):
        rec = self._get_record(_make_action(priority="critical"))
        assert rec["priority"] == "critical"

    def test_priority_code_mapped(self):
        rec = self._get_record(_make_action(priority="critical"))
        assert rec["priority_code"] == "1"

    def test_owner_preserved(self):
        rec = self._get_record(_make_action(owner="ops-team"))
        assert rec["owner"] == "ops-team"

    def test_linked_candidate_id_preserved(self):
        rec = self._get_record(_make_action(linked_candidate_id="CAND-007"))
        assert rec["linked_candidate_id"] == "CAND-007"

    def test_target_component_id_preserved(self):
        rec = self._get_record(_make_action(target_component_id="COMP-X"))
        assert rec["target_component_id"] == "COMP-X"

    def test_maximo_ext_populated_for_maximo(self):
        serializer = CAPExportSerializer(
            CAPExportConfig(target_system="maximo", default_work_group="MECH")
        )
        card = _make_rca_card(actions=[_make_action()])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        assert pkg["cr_records"][0]["maximo_ext"]["work_group"] == "MECH"

    def test_sap_ext_empty_for_maximo(self):
        rec = self._get_record()
        assert rec["sap_ext"] == {}


# ---------------------------------------------------------------------------
# FLOC resolution
# ---------------------------------------------------------------------------

class TestFLOCResolution:

    def test_resolved_maximo(self):
        serializer = CAPExportSerializer(CAPExportConfig(target_system="maximo"))
        kg = _make_kg_context([
            {"component_id": "COMP-001", "maximo_floc": "PLANT/SYS/BEARING-01"}
        ])
        card = _make_rca_card(actions=[_make_action(target_component_id="COMP-001")])
        pkg = serializer.serialize(card, kg, run_id="run-001", override_record=_make_override())
        rec = pkg["cr_records"][0]
        assert rec["functional_location"] == "PLANT/SYS/BEARING-01"
        assert rec["equipment_id"] is None
        assert rec["mapping_status"] == "resolved"
        assert pkg["unresolved_locations"] == []

    def test_resolved_sap(self):
        serializer = CAPExportSerializer(CAPExportConfig(target_system="sap_pm"))
        kg = _make_kg_context([
            {"component_id": "COMP-001", "sap_equipment_id": "EQ-4500012345"}
        ])
        card = _make_rca_card(actions=[_make_action(target_component_id="COMP-001")])
        pkg = serializer.serialize(card, kg, run_id="run-001", override_record=_make_override())
        rec = pkg["cr_records"][0]
        assert rec["equipment_id"] == "EQ-4500012345"
        assert rec["functional_location"] is None
        assert rec["mapping_status"] == "resolved"

    def test_unresolved_component_not_in_kg(self):
        serializer = CAPExportSerializer(CAPExportConfig(target_system="maximo"))
        kg = _make_kg_context([])  # empty components
        card = _make_rca_card(actions=[_make_action(target_component_id="COMP-MISSING")])
        pkg = serializer.serialize(card, kg, run_id="run-001", override_record=_make_override())
        rec = pkg["cr_records"][0]
        assert rec["functional_location"] is None
        assert rec["mapping_status"] == "unresolved"
        assert "COMP-MISSING" in pkg["unresolved_locations"]

    def test_unresolved_floc_property_absent(self):
        serializer = CAPExportSerializer(CAPExportConfig(target_system="maximo"))
        kg = _make_kg_context([
            {"component_id": "COMP-001", "maximo_floc": None}
        ])
        card = _make_rca_card(actions=[_make_action(target_component_id="COMP-001")])
        pkg = serializer.serialize(card, kg, run_id="run-001", override_record=_make_override())
        rec = pkg["cr_records"][0]
        assert rec["functional_location"] is None
        assert rec["mapping_status"] == "unresolved"
        assert "COMP-001" in pkg["unresolved_locations"]

    def test_no_target_component_no_floc(self):
        serializer = CAPExportSerializer(CAPExportConfig(target_system="maximo"))
        action = _make_action(target_component_id=None)
        card = _make_rca_card(actions=[action])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        rec = pkg["cr_records"][0]
        assert rec["functional_location"] is None
        assert rec["mapping_status"] == "unresolved"
        # I10: a targetless action is still surfaced in unresolved_locations.
        assert any("ACT-001" in loc for loc in pkg["unresolved_locations"])

    def test_multiple_unresolved_deduplicated(self):
        serializer = CAPExportSerializer(CAPExportConfig(target_system="maximo"))
        kg = _make_kg_context([])
        actions = [
            _make_action("ACT-001", target_component_id="COMP-X"),
            _make_action("ACT-002", target_component_id="COMP-X"),
        ]
        card = _make_rca_card(actions=actions)
        pkg = serializer.serialize(card, kg, run_id="run-001", override_record=_make_override())
        assert pkg["unresolved_locations"].count("COMP-X") == 1


# ---------------------------------------------------------------------------
# Short description truncation
# ---------------------------------------------------------------------------

class TestShortDescription:

    def test_prefix_included_when_enabled(self):
        cfg = CAPExportConfig(target_system="maximo", include_rca_run_id_in_description=True)
        serializer = CAPExportSerializer(cfg)
        card = _make_rca_card(actions=[_make_action(description="Fix the pump")])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-MYRUN", override_record=_make_override()
        )
        sd = pkg["cr_records"][0]["short_description"]
        assert sd.startswith("[RCA:run-MYRUN]")

    def test_prefix_omitted_when_disabled(self):
        cfg = CAPExportConfig(target_system="maximo", include_rca_run_id_in_description=False)
        serializer = CAPExportSerializer(cfg)
        card = _make_rca_card(actions=[_make_action(description="Fix the pump")])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-MYRUN", override_record=_make_override()
        )
        sd = pkg["cr_records"][0]["short_description"]
        assert not sd.startswith("[RCA:")

    def test_truncated_to_sap_limit(self):
        cfg = CAPExportConfig(target_system="sap_pm", include_rca_run_id_in_description=False)
        serializer = CAPExportSerializer(cfg)
        long_desc = "A" * 100
        card = _make_rca_card(actions=[_make_action(description=long_desc)])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        sd = pkg["cr_records"][0]["short_description"]
        assert len(sd) <= 40

    def test_truncated_to_maximo_limit(self):
        cfg = CAPExportConfig(target_system="maximo", include_rca_run_id_in_description=False)
        serializer = CAPExportSerializer(cfg)
        long_desc = "B" * 200
        card = _make_rca_card(actions=[_make_action(description=long_desc)])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        sd = pkg["cr_records"][0]["short_description"]
        assert len(sd) <= 100

    def test_sap_limit_respected_with_long_run_id(self):
        # I6: the [RCA:{run_id}] token must never push the field over the limit.
        cfg = CAPExportConfig(target_system="sap_pm", include_rca_run_id_in_description=True)
        serializer = CAPExportSerializer(cfg)
        card = _make_rca_card(actions=[_make_action(description="D" * 80)])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="R" * 60, override_record=_make_override()
        )
        assert len(pkg["cr_records"][0]["short_description"]) <= 40


# ---------------------------------------------------------------------------
# Long text content
# ---------------------------------------------------------------------------

class TestLongText:

    def test_long_text_contains_run_id(self):
        serializer = CAPExportSerializer(CAPExportConfig())
        card = _make_rca_card()
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-LONGTEXT", override_record=_make_override()
        )
        lt = pkg["cr_records"][0]["long_text"]
        assert "run-LONGTEXT" in lt

    def test_long_text_contains_description(self):
        serializer = CAPExportSerializer(CAPExportConfig())
        card = _make_rca_card(actions=[_make_action(description="Replace worn seal")])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        lt = pkg["cr_records"][0]["long_text"]
        assert "Replace worn seal" in lt

    def test_long_text_contains_rationale(self):
        serializer = CAPExportSerializer(CAPExportConfig())
        card = _make_rca_card(actions=[_make_action(rationale="Seal is beyond tolerance")])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        lt = pkg["cr_records"][0]["long_text"]
        assert "Seal is beyond tolerance" in lt

    def test_long_text_contains_dackar_footer(self):
        serializer = CAPExportSerializer(CAPExportConfig())
        card = _make_rca_card()
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        lt = pkg["cr_records"][0]["long_text"]
        assert "DACKAR RCA" in lt

    def test_custom_long_text_header(self):
        cfg = CAPExportConfig(long_text_header="CUSTOM HEADER\n")
        serializer = CAPExportSerializer(cfg)
        card = _make_rca_card()
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        lt = pkg["cr_records"][0]["long_text"]
        # I8: the custom header is prepended to (not a replacement for) the block.
        assert lt.startswith("CUSTOM HEADER")
        assert "[RCA Run: run-001]" in lt


# ---------------------------------------------------------------------------
# Approval guard — card-level (secondary) gate
# ---------------------------------------------------------------------------

class TestApprovalGuard:

    def test_raises_if_not_approved(self):
        serializer = CAPExportSerializer(CAPExportConfig())
        card = _make_rca_card(writeback_recommendation="hold_until_review")
        with pytest.raises(ValueError, match="ready_if_accepted"):
            serializer.serialize(
                card, _make_kg_context(), run_id="run-001", override_record=_make_override()
            )

    def test_raises_if_recommendation_missing(self):
        serializer = CAPExportSerializer(CAPExportConfig())
        card = _make_rca_card()
        card["analyst_review"] = {}
        with pytest.raises(ValueError):
            serializer.serialize(
                card, _make_kg_context(), run_id="run-001", override_record=_make_override()
            )

    def test_passes_when_approved(self):
        serializer = CAPExportSerializer(CAPExportConfig())
        card = _make_rca_card(writeback_recommendation="ready_if_accepted")
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        assert "export_id" in pkg


# ---------------------------------------------------------------------------
# Override-record gate (B1) — authoritative acceptance gate
# ---------------------------------------------------------------------------

class TestOverrideGate:

    def setup_method(self):
        self.serializer = CAPExportSerializer(CAPExportConfig())

    def test_export_id_derived_from_override(self):
        pkg = self.serializer.serialize(
            _make_rca_card(), _make_kg_context(), run_id="run-77",
            override_record=_make_override(override_id="OVR-STABLE"),
        )
        assert pkg["export_id"] == "CAPEXP::run-77::OVR-STABLE"

    def test_export_id_is_idempotent(self):
        override = _make_override(override_id="OVR-STABLE")
        a = self.serializer.serialize(
            _make_rca_card(), _make_kg_context(), run_id="run-77", override_record=override
        )
        b = self.serializer.serialize(
            _make_rca_card(), _make_kg_context(), run_id="run-77", override_record=override
        )
        assert a["export_id"] == b["export_id"]

    def test_rejects_reject_decision(self):
        with pytest.raises(ValueError, match="accept"):
            self.serializer.serialize(
                _make_rca_card(), _make_kg_context(), run_id="run-001",
                override_record=_make_override(writeback_decision="reject"),
            )

    def test_rejects_defer_decision(self):
        with pytest.raises(ValueError, match="accept"):
            self.serializer.serialize(
                _make_rca_card(), _make_kg_context(), run_id="run-001",
                override_record=_make_override(writeback_decision="defer"),
            )

    def test_rejects_non_dict_override(self):
        with pytest.raises(ValueError):
            self.serializer.serialize(
                _make_rca_card(), _make_kg_context(), run_id="run-001", override_record=None
            )

    def test_requires_override_id(self):
        override = _make_override()
        del override["override_id"]
        with pytest.raises(ValueError, match="override_id"):
            self.serializer.serialize(
                _make_rca_card(), _make_kg_context(), run_id="run-001", override_record=override
            )


# ---------------------------------------------------------------------------
# Cross-artifact identifier checks (I3)
# ---------------------------------------------------------------------------

class TestCrossArtifactIdentifiers:

    def setup_method(self):
        self.serializer = CAPExportSerializer(CAPExportConfig())

    def test_missing_kg_asset_id_raises(self):
        kg = _make_kg_context()
        del kg["asset_id"]
        with pytest.raises(ValueError, match="asset_id"):
            self.serializer.serialize(
                _make_rca_card(), kg, run_id="run-001", override_record=_make_override()
            )

    def test_event_id_mismatch_raises(self):
        card = _make_rca_card(event_id="EVT-CARD")
        with pytest.raises(ValueError, match="event_id mismatch"):
            self.serializer.serialize(
                card, _make_kg_context(), run_id="run-001",
                override_record=_make_override(event_id="EVT-OTHER"),
            )

    def test_asset_id_mismatch_raises(self):
        with pytest.raises(ValueError, match="asset_id mismatch"):
            self.serializer.serialize(
                _make_rca_card(), _make_kg_context(asset_id="ASSET-A"), run_id="run-001",
                override_record=_make_override(asset_id="ASSET-OTHER"),
            )


# ---------------------------------------------------------------------------
# Empty actions list
# ---------------------------------------------------------------------------

class TestEmptyActions:

    def test_no_actions_produces_empty_cr_records(self):
        serializer = CAPExportSerializer(CAPExportConfig())
        card = _make_rca_card(actions=[])
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        assert pkg["cr_records"] == []
        assert pkg["unresolved_locations"] == []

    def test_none_actions_produces_empty_cr_records(self):
        serializer = CAPExportSerializer(CAPExportConfig())
        card = _make_rca_card(actions=[])
        card["recommended_actions"] = None
        pkg = serializer.serialize(
            card, _make_kg_context(), run_id="run-001", override_record=_make_override()
        )
        assert pkg["cr_records"] == []
