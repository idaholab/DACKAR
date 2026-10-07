"""TC-4 integration test — spurious neutron-instrument trip.

Promotes the assertions in ``run_test_case_4.ipynb`` to a pytest-collected
end-to-end test. The full Stage A–G orchestrator runs in fixture-only mode
(no live Neo4j / Chroma / LLM); the committed fixtures under ``fixtures/`` are
the inputs and the assertions below are the expected conclusion.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca
from assertion_helpers import (
    assert_primary_cause,
    assert_candidate_present,
    assert_data_coverage_status,
    run_assertion_table,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.integration
def test_case_4_spurious_ni_trip(tmp_path: Path) -> None:
    fixtures = load_fixtures(_FIXTURE_DIR)
    orchestrator = build_fixture_orchestrator(tmp_path, top_k_candidates=5, enable_ishikawa=True)
    result = run_rca(orchestrator, fixtures)

    assertions = [
        {"id": "A4-1", "desc": "FM-NI-SPURIOUS candidate present",
         "fn": lambda r: assert_candidate_present(r, "FM-NI-SPURIOUS")},
        {"id": "A4-2", "desc": "Primary hypothesis is FM-NI-SPURIOUS",
         "fn": lambda r: assert_primary_cause(r, "FM-NI-SPURIOUS")},
        {"id": "A4-3", "desc": "Environmental monitoring complete in coverage",
         "fn": lambda r: assert_data_coverage_status(r, "environmental_monitoring", "complete")},
        {"id": "A4-4", "desc": "Protection logic context complete in coverage",
         "fn": lambda r: assert_data_coverage_status(r, "protection_logic_context", "complete")},
        {"id": "A4-5", "desc": "SOE log complete in coverage",
         "fn": lambda r: assert_data_coverage_status(r, "soe_log", "complete")},
    ]
    run_assertion_table(result, assertions, label="TC-4 Assertions")
