"""
Unit tests for equipment_similarity.equipment_spec_builder.
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from equipment_similarity.equipment_spec_builder import EquipmentSpecBuilder


def test_identity_line_uses_name_and_component_id():
    builder = EquipmentSpecBuilder()
    text = builder.build_spec_text(
        component_id="MCP-A",
        component_name="Main coolant pump",
        definition_props={},
    )
    first = text.splitlines()[0]
    assert first == "Equipment: Main coolant pump (MCP-A)"


def test_identity_line_falls_back_to_component_id_when_name_missing():
    builder = EquipmentSpecBuilder()
    text = builder.build_spec_text(
        component_id="P-101",
        component_name=None,
        definition_props={},
    )
    assert text.splitlines()[0] == "Equipment: P-101"


def test_failure_mechanisms_are_deduplicated_in_output():
    builder = EquipmentSpecBuilder()
    text = builder.build_spec_text(
        component_id="P-101",
        component_name="Pump 101",
        definition_props={},
        failure_mechanisms=["fatigue", "fatigue", "corrosion"],
    )
    assert "Failure mechanisms: fatigue, corrosion" in text
