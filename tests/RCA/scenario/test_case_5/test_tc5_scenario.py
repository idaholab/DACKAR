"""TC-5 integration test — HPCI common-cause failure (vendor coupling batch).

Promotes the assertions in ``run_test_case_5.ipynb`` to a pytest-collected
end-to-end test driven by the full orchestrator in fixture-only mode.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca
from assertion_helpers import (
    assert_candidate_present,
    assert_data_coverage_status,
    assert_barrier_analysis_present,
    assert_ap913_completeness_present,
    run_assertion_table,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.integration
def test_case_5_hpci_ccf(tmp_path: Path) -> None:
    fixtures = load_fixtures(_FIXTURE_DIR)
    orchestrator = build_fixture_orchestrator(tmp_path, top_k_candidates=5, enable_ishikawa=True)
    result = run_rca(orchestrator, fixtures)

    assertions = [
        {"id": "A5-1", "desc": "CCF candidate present",
         "fn": lambda r: assert_candidate_present(r, "FM-HPCI-CCF-COUPLING")},
        {"id": "A5-2", "desc": "Vendor supply chain records complete in coverage",
         "fn": lambda r: assert_data_coverage_status(r, "vendor_supply_chain_records", "complete")},
        {"id": "A5-3", "desc": "Barrier analysis present",
         "fn": lambda r: assert_barrier_analysis_present(r)},
        {"id": "A5-4", "desc": "AP-913 completeness present",
         "fn": lambda r: assert_ap913_completeness_present(r)},
    ]
    run_assertion_table(result, assertions, label="TC-5 Assertions")
