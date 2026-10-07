"""TC-6 integration test — human-performance / procedure gap during startup.

Promotes the assertions in ``run_test_case_6.ipynb`` to a pytest-collected
end-to-end test driven by the full orchestrator in fixture-only mode.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca
from assertion_helpers import (
    assert_candidate_present,
    assert_human_perf_applicable,
    assert_human_perf_mode_present,
    assert_ishikawa_category_present,
    assert_data_coverage_status,
    assert_ap913_completeness_present,
    run_assertion_table,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.integration
def test_case_6_human_performance(tmp_path: Path) -> None:
    fixtures = load_fixtures(_FIXTURE_DIR)
    orchestrator = build_fixture_orchestrator(tmp_path, top_k_candidates=5, enable_ishikawa=True)
    result = run_rca(orchestrator, fixtures)

    assertions = [
        {"id": "A6-1", "desc": "Primary hypothesis is execution error",
         "fn": lambda r: assert_candidate_present(r, "FM-MFPB-LUBE-OIL-OMISSION")},
        {"id": "A6-2", "desc": "Human performance assessment applicable",
         "fn": lambda r: assert_human_perf_applicable(r)},
        {"id": "A6-3", "desc": "Execution error finding present",
         "fn": lambda r: assert_human_perf_mode_present(r, "execution_error")},
        {"id": "A6-4", "desc": "Process/procedure Ishikawa row present",
         "fn": lambda r: assert_ishikawa_category_present(r, "process_procedure")},
        {"id": "A6-5", "desc": "Training records coverage complete",
         "fn": lambda r: assert_data_coverage_status(r, "training_records", "complete")},
        {"id": "A6-6", "desc": "AP-913 completeness block present",
         "fn": lambda r: assert_ap913_completeness_present(r)},
    ]
    run_assertion_table(result, assertions, label="TC-6 Assertions")
