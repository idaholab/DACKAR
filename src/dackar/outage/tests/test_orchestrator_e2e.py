"""
End-to-end pipeline tests: A → B → C → D → E(stub) → F → G.

Uses run_pipeline() from the demo module with the two pre-defined scenarios.
Stages A, C, D, F, G run their real production logic.
Stage B runs with a stub KG driver.
Stage E uses the pre-built stub artifact (no LOGOS required).

Coverage targets:
    run_pipeline()           — produces all seven artifact keys
    RCP seal scenario        — ESCALATE decision, regulatory flag set,
                               float_analysis.is_critical_path_impact = True
    Snubber scenario         — PROCEED decision, no regulatory block,
                               non-critical path impact
    Artifact schemas         — required keys present in each stage output
    Stage E fallback         — when schedule_data_root is None, pre-built stub used
    Stage E LOGOS path       — when schedule_data_root is non-existent dir,
                               FileNotFoundError caught, stub returned
    run_id propagation       — custom run_id threaded through all artifacts
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_OUTAGE_ROOT = Path(__file__).parent.parent
_DEMO_ROOT = _OUTAGE_ROOT / "demos" / "unexpected_act_workflow_1"
for _p in (_OUTAGE_ROOT, _DEMO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from demo_scenarios import (
    run_pipeline,
    SCENARIO_RCP_SEAL,
    SCENARIO_SNUBBER_EXT,
)

# ===========================================================================
# Helpers
# ===========================================================================

_EXPECTED_ARTIFACT_KEYS = {
    "intake": ("activity_id", "run_id", "emergence_type", "regulatory_drivers",
               "extracted_entities", "data_quality_score"),
    "timeline": ("activity_id", "run_id", "component_id", "events",
                 "recurrence_indicators", "data_coverage"),
    "temporal": ("activity_id", "run_id", "chain_links"),
    "analogs": ("activity_id", "run_id", "analogs", "duration_distribution"),
    "schedule": ("activity_id", "run_id",),
    "options": ("activity_id", "run_id", "options"),
    "recommendation": ("activity_id", "run_id", "decision_status", "executive_summary"),
}


def _check_artifact_keys(result: Dict[str, Any], artifact: str):
    obj = result[artifact]
    for key in _EXPECTED_ARTIFACT_KEYS.get(artifact, ()):
        assert key in obj, f"{artifact} missing key '{key}'"


# ===========================================================================
# RCP Seal scenario (ESCALATE)
# ===========================================================================

class TestRCPSealScenario:

    @pytest.fixture(scope="class")
    def result(self):
        return run_pipeline(SCENARIO_RCP_SEAL)

    def test_top_level_keys(self, result):
        for key in ("scenario_label", "run_id", "intake", "timeline",
                    "temporal", "analogs", "schedule", "options", "recommendation"):
            assert key in result, f"missing top-level key '{key}'"

    def test_intake_artifact_keys(self, result):
        _check_artifact_keys(result, "intake")

    def test_timeline_artifact_keys(self, result):
        _check_artifact_keys(result, "timeline")

    def test_temporal_artifact_keys(self, result):
        _check_artifact_keys(result, "temporal")

    def test_analogs_artifact_keys(self, result):
        _check_artifact_keys(result, "analogs")

    def test_schedule_artifact_keys(self, result):
        _check_artifact_keys(result, "schedule")

    def test_options_artifact_keys(self, result):
        _check_artifact_keys(result, "options")

    def test_recommendation_artifact_keys(self, result):
        _check_artifact_keys(result, "recommendation")

    def test_decision_status_is_escalate(self, result):
        status = result["recommendation"]["decision_status"]
        assert status == "ESCALATE", f"expected ESCALATE, got {status}"

    def test_regulatory_flag_set(self, result):
        """RCP seal has TS 3.4.6 — at least one defer_prohibited driver."""
        drivers = result["intake"].get("regulatory_drivers", [])
        assert any(d.get("defer_prohibited") for d in drivers), \
            "expected at least one defer_prohibited regulatory driver for RCP seal"

    def test_emergence_type_is_regulatory_driven(self, result):
        # RCP seal description contains TS 3.4.6 — rule 1 fires → regulatory_driven
        et = result["intake"]["emergence_type"]
        assert et == "regulatory_driven", f"expected regulatory_driven, got {et}"

    def test_critical_path_impact(self, result):
        fa = result["schedule"].get("float_analysis", {})
        assert fa.get("is_critical_path_impact") is True, \
            "RCP seal scenario should have critical path impact"

    def test_scenario_label(self, result):
        assert "RCP" in result["scenario_label"] or "Seal" in result["scenario_label"] \
            or "Scenario 1" in result["scenario_label"]

    def test_options_list_not_empty(self, result):
        assert len(result["options"]["options"]) > 0

    def test_run_id_consistent_across_artifacts(self, result):
        run_id = result["run_id"]
        for artifact in ("intake", "timeline", "temporal", "analogs", "schedule",
                         "options", "recommendation"):
            assert result[artifact].get("run_id") == run_id, \
                f"{artifact}.run_id mismatch"


# ===========================================================================
# Snubber scope expansion scenario (PROCEED)
# ===========================================================================

class TestSnubberScenario:

    @pytest.fixture(scope="class")
    def result(self):
        return run_pipeline(SCENARIO_SNUBBER_EXT)

    def test_top_level_keys(self, result):
        for key in ("scenario_label", "run_id", "intake", "timeline",
                    "temporal", "analogs", "schedule", "options", "recommendation"):
            assert key in result

    def test_decision_status_is_proceed(self, result):
        status = result["recommendation"]["decision_status"]
        assert status == "PROCEED", f"expected PROCEED, got {status}"

    def test_no_defer_prohibited_driver(self, result):
        drivers = result["intake"].get("regulatory_drivers", [])
        assert not any(d.get("defer_prohibited") for d in drivers), \
            "snubber scenario should not have defer_prohibited regulatory drivers"

    def test_non_critical_path(self, result):
        fa = result["schedule"].get("float_analysis", {})
        assert fa.get("is_critical_path_impact") is False, \
            "snubber scenario should not have critical path impact"

    def test_options_include_insert_now(self, result):
        option_types = [o.get("option_type") for o in result["options"]["options"]]
        assert "insert_now" in option_types, \
            "snubber scenario should have insert_now as a feasible option"

    def test_run_id_consistent_across_artifacts(self, result):
        run_id = result["run_id"]
        for artifact in ("intake", "timeline", "schedule", "recommendation"):
            assert result[artifact].get("run_id") == run_id


# ===========================================================================
# Custom run_id propagation
# ===========================================================================

class TestCustomRunId:

    def test_custom_run_id_propagated(self):
        custom_id = "CUSTOM-RUN-XYZ-42"
        result = run_pipeline(SCENARIO_RCP_SEAL, run_id=custom_id)
        assert result["run_id"] == custom_id
        assert result["intake"]["run_id"] == custom_id
        assert result["recommendation"]["run_id"] == custom_id

    def test_auto_generated_run_id_has_demo_prefix(self):
        result = run_pipeline(SCENARIO_SNUBBER_EXT)
        assert result["run_id"].startswith("DEMO::")


# ===========================================================================
# Stage E fallback behaviour
# ===========================================================================

class TestStageEFallback:

    def test_no_schedule_root_uses_pre_built_stub(self):
        """Passing schedule_data_root=None must return the pre-built artifact."""
        result = run_pipeline(SCENARIO_RCP_SEAL, schedule_data_root=None)
        # Pre-built artifact has 'schedule_version_id' set to RF-24-WRK-003
        assert result["schedule"].get("schedule_version_id") is not None

    def test_nonexistent_schedule_root_falls_back_to_stub(self):
        """A non-existent data root must trigger FileNotFoundError and return stub."""
        result = run_pipeline(
            SCENARIO_RCP_SEAL,
            schedule_data_root="/nonexistent/path/that/does/not/exist",
        )
        # Should not raise; should return a valid schedule artifact
        assert "activity_id" in result["schedule"]


# ===========================================================================
# Confidence and data quality
# ===========================================================================

class TestDataQuality:

    def test_data_quality_score_in_range(self):
        result = run_pipeline(SCENARIO_RCP_SEAL)
        score = result["intake"]["data_quality_score"]
        assert 0.0 <= score <= 1.0

    def test_analogs_confidence_tier_present(self):
        result = run_pipeline(SCENARIO_SNUBBER_EXT)
        tier = result["analogs"]["duration_distribution"].get("confidence_tier")
        assert tier in ("data_supported", "sme_informed", "low_confidence", None)

    def test_duration_distribution_has_percentiles(self):
        result = run_pipeline(SCENARIO_RCP_SEAL)
        dd = result["analogs"]["duration_distribution"]
        assert dd.get("p50_hours") is not None
        assert dd.get("p80_hours") is not None
