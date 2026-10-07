"""TC-2 integration test — protection-logic-context coverage resolution.

The TC-2 show-and-tell notebook demonstrates (via a live-Neo4j dev
orchestrator for its main run, then two fixture-mode runs) that supplying
``protection_logic_context`` resolves the data-coverage flag that is otherwise
unresolved. Only that offline portion is promoted here: the two fixture-mode
runs that the notebook itself builds with ``build_fixture_orchestrator`` — one
without the PLC fixture, one with it. The live-Neo4j cell is demo-only and is
not part of this test.

The notebook asserted this contract by printing the flag; here it is made an
explicit assertion: the ``protection_logic_context`` coverage status is
unresolved (``"missing"``) when the fixture is withheld and ``"complete"`` when
it is supplied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _plc_coverage_status(result: Dict[str, Any]) -> str:
    cov = (result.get("run_manifest") or {}).get("artifacts", {}).get(
        "data_coverage_summary", {}
    )
    return (cov.get("protection_logic_context") or {}).get("status", "?")


@pytest.mark.integration
def test_case_2_plc_coverage_resolution(tmp_path: Path) -> None:
    fixtures = load_fixtures(_FIXTURE_DIR)

    # Run 1 — withhold protection_logic_context: coverage flag unresolved.
    fixtures_no_plc = dict(fixtures)
    fixtures_no_plc["protection_logic_context"] = None
    orc_no_plc = build_fixture_orchestrator(
        tmp_path / "no_plc", top_k_candidates=5, enable_ishikawa=False
    )
    result_no_plc = run_rca(orc_no_plc, fixtures_no_plc)

    # Run 2 — supply protection_logic_context: coverage flag resolved.
    orc_with_plc = build_fixture_orchestrator(
        tmp_path / "with_plc", top_k_candidates=5, enable_ishikawa=False
    )
    result_with_plc = run_rca(orc_with_plc, fixtures)

    status_no_plc = _plc_coverage_status(result_no_plc)
    status_with_plc = _plc_coverage_status(result_with_plc)

    assert status_no_plc == "missing", (
        f"Without the protection_logic_context fixture, coverage status should be "
        f"'missing'; got {status_no_plc!r}"
    )
    assert status_with_plc == "complete", (
        f"With the protection_logic_context fixture supplied, coverage status should be "
        f"'complete'; got {status_with_plc!r}"
    )
