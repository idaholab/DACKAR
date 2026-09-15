"""
Unit tests for equipment_similarity.kg_equipment_poller.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from equipment_similarity.kg_equipment_poller import KGEquipmentPoller


class _FakeClient:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    def run(self, _query: str):
        return self._rows


class _FakeSpecStore:
    def __init__(self) -> None:
        self.batches: List[List[Dict[str, Any]]] = []

    def upsert_batch(self, batch: List[Dict[str, Any]]) -> int:
        self.batches.append(list(batch))
        return len(batch)


def _base_row() -> Dict[str, Any]:
    return {
        "component_id": "P-101",
        "component_name": "Pump 101",
        "domain_category": None,
        "structural_kind": None,
        "nominal_size": None,
        "design_pressure": None,
        "design_temperature": None,
        "material_spec": None,
        "manufacturer": None,
        "model_number": None,
        "failure_mode_names": [],
        "failure_mechanisms": [],
    }


def test_poll_and_upsert_skips_identity_only_rows():
    row = _base_row()
    poller = KGEquipmentPoller(client=_FakeClient([row]))
    store = _FakeSpecStore()

    total = poller.poll_and_upsert(store, batch_size=10)

    assert total == 0
    assert store.batches == []


def test_poll_and_upsert_persists_rows_with_substantive_data():
    row = _base_row()
    row["domain_category"] = "centrifugal pump"

    poller = KGEquipmentPoller(client=_FakeClient([row]))
    store = _FakeSpecStore()

    total = poller.poll_and_upsert(store, batch_size=10)

    assert total == 1
    assert len(store.batches) == 1
    assert store.batches[0][0]["component_id"] == "P-101"
    assert "Type: centrifugal pump" in store.batches[0][0]["spec_text"]
