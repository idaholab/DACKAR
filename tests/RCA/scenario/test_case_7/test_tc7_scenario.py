"""TC-7 integration test — missing-data scope management across two runs.

Promotes the assertions in ``run_test_case_7.ipynb`` to a pytest-collected
end-to-end test. Two orchestrator runs are driven in fixture-only mode:
Run 1 (ishikawa off) exercises the not-assessed coverage path; Run 2
(ishikawa on) exercises candidate ranking and the similar-event plant match.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca
from assertion_helpers import (
    assert_candidate_present,
    assert_data_coverage_status,
    assert_similar_event_match,
    run_assertion_table,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.integration
def test_case_7_scope_management(tmp_path: Path) -> None:
    fixtures = load_fixtures(_FIXTURE_DIR)

    orc1 = build_fixture_orchestrator(tmp_path / "run1", top_k_candidates=5, enable_ishikawa=False)
    r1 = run_rca(orc1, fixtures)

    orc2 = build_fixture_orchestrator(tmp_path / "run2", top_k_candidates=5, enable_ishikawa=True)
    r2 = run_rca(orc2, fixtures)

    assertions_r1 = [
        {"id": "A7-3", "desc": "Run 1: SOE not assessed",
         "fn": lambda r: assert_data_coverage_status(r, "soe_log", "not_assessed")},
        {"id": "A7-4", "desc": "Run 1: alarm not assessed",
         "fn": lambda r: assert_data_coverage_status(r, "alarm_log", "not_assessed")},
    ]
    run_assertion_table(r1, assertions_r1, label="TC-7 Run 1 Assertions")

    assertions_r2 = [
        {"id": "A7-7", "desc": "Run 2: HX fouling candidate present",
         "fn": lambda r: assert_candidate_present(r, "FM-SWHX4C-FOULING")},
        {"id": "A7-8", "desc": "Run 2: similar event plant match",
         "fn": lambda r: assert_similar_event_match(r, any_plant_match=True)},
    ]
    run_assertion_table(r2, assertions_r2, label="TC-7 Run 2 Assertions")
