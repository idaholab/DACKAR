"""
Unit tests for equipment_similarity.kg_equipment_poller.
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from equipment_similarity.kg_equipment_poller import _SPEC_QUERY, KGEquipmentPoller


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


def test_failure_mode_data_flows_into_spec_text():
    """B2: failure mode names/mechanisms returned by the query flow into the
    spec text (the query reads fm.name / fm.failure_mechanism regardless of
    which relationship shape matched)."""
    row = _base_row()
    row["failure_mode_names"] = ["bearing wear", "seal leak"]
    row["failure_mechanisms"] = ["fatigue", "corrosion"]

    poller = KGEquipmentPoller(client=_FakeClient([row]))
    store = _FakeSpecStore()

    total = poller.poll_and_upsert(store, batch_size=10)

    assert total == 1
    spec = store.batches[0][0]["spec_text"]
    assert "Failure modes: bearing wear, seal leak" in spec
    assert "Failure mechanisms: fatigue, corrosion" in spec


def test_spec_query_matches_canonical_fm_relationships():
    """B2: failure modes are matched via [:subject_to|applies_to] undirected,
    covering both the schema-declared and ingestion-emitted shapes — not the
    non-existent [:has_failure_mode]."""
    assert "subject_to" in _SPEC_QUERY
    assert "applies_to" in _SPEC_QUERY
    assert "has_failure_mode" not in _SPEC_QUERY
    # The failure_mode match must be undirected (no arrowhead on that clause).
    fm_line = next(
        line for line in _SPEC_QUERY.splitlines()
        if "failure_mode" in line and "MATCH" in line
    )
    assert "->" not in fm_line and "<-" not in fm_line


def test_module_importable_as_fully_qualified_package():
    """B1: `import dackar.RCA.equipment_similarity.kg_equipment_poller` must
    succeed via package-relative sibling imports.

    Run in a subprocess with only ``src`` on PYTHONPATH so the top-level
    ``equipment_similarity`` shim this test module adds to sys.path cannot mask
    a regression to absolute sibling imports (which raised ModuleNotFoundError
    under the fully-qualified package path)."""
    src_dir = _RCA_ROOT.parents[1]  # .../src
    proc = subprocess.run(
        [
            sys.executable, "-c",
            "import dackar.RCA.equipment_similarity.kg_equipment_poller as m; "
            "assert hasattr(m, 'KGEquipmentPoller')",
        ],
        env={**os.environ, "PYTHONPATH": str(src_dir)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
