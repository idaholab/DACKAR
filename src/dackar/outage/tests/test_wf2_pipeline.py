"""
End-to-end tests for the workflow 2 pre-outage risk prediction pipeline.

Tests stages A–C (demo fixtures) plus the full run_pipeline() call with ground
truth comparison. Stages D–G are delegated to library services and are not
re-tested here; they have their own unit tests in test_stages_a_c.py and the
outage_uncertainty service tests.

Coverage targets:
    Stage A   — data ingestion, quality gate, regulatory flag detection
    Stage B   — NLP expansion, unknown-token-rate gate, cross-reference extraction
    Stage C   — KG construction: node/edge counts, component_histories shape
    run_pipeline() — full result dict keys, risk register shape
    Ground truth — true_positives, false_positives, false_negatives keys present
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_OUTAGE_ROOT = Path(__file__).parent.parent
_WF2_ROOT = _OUTAGE_ROOT / "demos" / "unexpected_act_workflow_2"
for _p in (_OUTAGE_ROOT, _WF2_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from demo_data import (
    ACTIVITIES,
    COMPONENTS,
    CONDITION_REPORTS,
    SCHEDULE,
    WORK_ORDERS,
    RF22_GROUND_TRUTH,
    NER_GROUND_TRUTH,
    PLANT_ID_PATTERN,
    CR_WO_PATTERN,
)


# ---------------------------------------------------------------------------
# Helpers — import pipeline internals
# ---------------------------------------------------------------------------

import pipeline as _pl


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

class TestStageA:
    def _call(self):
        return _pl._stage_a(COMPONENTS, CONDITION_REPORTS, WORK_ORDERS, ACTIVITIES, SCHEDULE)

    def test_returns_required_keys(self):
        a = self._call()
        for key in ("components_by_id", "crs_by_id", "wos_by_id",
                    "activities_by_id", "schedule_by_id",
                    "quality_summary", "regulatory_component_ids"):
            assert key in a, f"Missing key: {key}"

    def test_component_count(self):
        a = self._call()
        assert len(a["components_by_id"]) == len(COMPONENTS)

    def test_regulatory_flags_detected(self):
        a = self._call()
        # 4 of 5 components have regulatory_constraint_flag = True
        assert len(a["regulatory_component_ids"]) == 4

    def test_non_regulatory_component_excluded(self):
        a = self._call()
        assert "1CCW-P-002A" not in a["regulatory_component_ids"]

    def test_quality_gate_passes(self):
        a = self._call()
        assert a["quality_summary"]["quality_gate_passed"] is True

    def test_emergence_category_counts(self):
        a = self._call()
        counts = a["quality_summary"]["emergence_category_counts"]
        assert counts.get("DISCOVERY", 0) >= 4

    def test_missing_regulatory_flag_raises(self):
        bad_components = [dict(c) for c in COMPONENTS]
        del bad_components[0]["regulatory_constraint_flag"]
        with pytest.raises(ValueError, match="regulatory_constraint_flag missing"):
            _pl._stage_a(bad_components, CONDITION_REPORTS, WORK_ORDERS, ACTIVITIES, SCHEDULE)

    def test_missing_emergence_category_raises(self):
        bad_activities = [dict(a) for a in ACTIVITIES]
        # Find an emergent activity and strip its category
        for act in bad_activities:
            if act.get("emergent_flag"):
                act["emergence_category"] = None
                break
        with pytest.raises(ValueError, match="emergence_category missing"):
            _pl._stage_a(COMPONENTS, CONDITION_REPORTS, WORK_ORDERS, bad_activities, SCHEDULE)


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------

class TestStageB:
    def _stage_a(self):
        return _pl._stage_a(COMPONENTS, CONDITION_REPORTS, WORK_ORDERS, ACTIVITIES, SCHEDULE)

    def _call(self, resolver=None):
        return _pl._stage_b(self._stage_a(), resolver)

    def test_returns_required_keys(self):
        b = self._call()
        for key in ("crs_expanded", "wos_expanded", "nlp_quality"):
            assert key in b

    def test_cr_count_preserved(self):
        b = self._call()
        assert len(b["crs_expanded"]) == len(CONDITION_REPORTS)

    def test_wo_count_preserved(self):
        b = self._call()
        assert len(b["wos_expanded"]) == len(WORK_ORDERS)

    def test_nlp_gate_passes(self):
        b = self._call()
        assert b["nlp_quality"]["quality_gate_passed"] is True

    def test_unknown_token_rate_low(self):
        b = self._call()
        assert b["nlp_quality"]["unknown_token_rate"] < 0.25

    def test_cross_references_full_ids(self):
        """CR_WO_PATTERN fix: cross_references must contain full IDs, not just prefix."""
        b = self._call()
        # CR-2021-00892 references CR-2019-06891
        cr = b["crs_expanded"]["CR-2021-00892"]
        xrefs = cr["cross_references"]
        assert xrefs, "Expected at least one cross-reference"
        for xref in xrefs:
            assert xref.startswith("CR-") or xref.startswith("WO-"), (
                f"Cross-reference '{xref}' is not a full ID — pattern capture group bug"
            )

    def test_plant_ids_extracted(self):
        b = self._call()
        # At least some CRs should have plant element IDs extracted
        total = sum(len(cr.get("plant_element_ids", [])) for cr in b["crs_expanded"].values())
        assert total > 0

    def test_description_expanded_field_present(self):
        b = self._call()
        for cr_id, cr in b["crs_expanded"].items():
            assert "description_expanded" in cr, f"Missing description_expanded on {cr_id}"


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------

class TestStageC:
    def _stages_ab(self):
        a = _pl._stage_a(COMPONENTS, CONDITION_REPORTS, WORK_ORDERS, ACTIVITIES, SCHEDULE)
        b = _pl._stage_b(a, None)
        return a, b

    def _call(self, training_outages=None):
        a, b = self._stages_ab()
        return _pl._stage_c(a, b, training_outages or ["RF-20", "RF-21"])

    def test_returns_required_keys(self):
        c = self._call()
        for key in ("nodes", "edges", "component_histories", "schedule_by_id"):
            assert key in c

    def test_component_nodes_present(self):
        c = self._call()
        for comp in COMPONENTS:
            assert comp["component_id"] in c["nodes"]

    def test_component_node_type(self):
        c = self._call()
        assert c["nodes"]["1RHS-P-001A"]["type"] == "component"

    def test_cr_nodes_present(self):
        c = self._call()
        for cr in CONDITION_REPORTS:
            assert cr["cr_id"] in c["nodes"]

    def test_wo_nodes_present(self):
        c = self._call()
        for wo in WORK_ORDERS:
            assert wo["wo_id"] in c["nodes"]

    def test_has_cr_edges_exist(self):
        c = self._call()
        has_cr_edges = [e for e in c["edges"] if e["edge_type"] == "has_cr"]
        assert len(has_cr_edges) > 0

    def test_component_histories_all_components(self):
        c = self._call()
        for comp in COMPONENTS:
            assert comp["component_id"] in c["component_histories"]

    def test_training_emergent_activities_scoped_to_training_set(self):
        c = self._call(training_outages=["RF-20"])
        hist = c["component_histories"]["1RHS-P-001A"]
        for act in hist["training_emergent_activities"]:
            assert act["outage_id"] == "RF-20", (
                f"Activity from {act['outage_id']} leaked into training set"
            )

    def test_training_outages_parameterised(self):
        """Changing training_outages changes which activities are counted as training emergent."""
        c_both = self._call(training_outages=["RF-20", "RF-21"])
        c_one  = self._call(training_outages=["RF-20"])
        hist_both = c_both["component_histories"]["1RHS-P-001A"]
        hist_one  = c_one["component_histories"]["1RHS-P-001A"]
        assert len(hist_both["training_emergent_activities"]) >= \
               len(hist_one["training_emergent_activities"])

    def test_outage_nodes_created(self):
        c = self._call()
        outage_ids = {a["outage_id"] for a in ACTIVITIES}
        for oid in outage_ids:
            assert oid in c["nodes"], f"Missing outage node for {oid}"


# ---------------------------------------------------------------------------
# Full pipeline run
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @pytest.fixture(scope="class")
    def results(self):
        from pipeline import run_pipeline
        return run_pipeline(include_ground_truth=True)

    def test_top_level_keys(self, results):
        for key in ("pipeline_run_id", "plant", "holdout_outage", "training_outages",
                    "stage_a", "stage_b", "stage_c", "stage_d", "stage_e",
                    "stage_f", "stage_g", "ground_truth_comparison"):
            assert key in results

    def test_risk_register_present(self, results):
        assert len(results["stage_g"]["risk_register"]) == len(COMPONENTS)

    def test_flagged_components_non_empty(self, results):
        assert len(results["stage_g"]["flagged_components"]) > 0

    def test_ground_truth_keys(self, results):
        gt = results["ground_truth_comparison"]
        for key in ("true_positives", "false_positives", "false_negatives",
                    "true_negatives_confirmed", "predicted_flagged",
                    "actual_emergent_component_ids"):
            assert key in gt, f"Missing ground truth key: {key}"

    def test_no_false_negatives(self, results):
        """Both RF-22 emergent components must be flagged."""
        assert results["ground_truth_comparison"]["false_negatives"] == []

    def test_true_positives_are_correct(self, results):
        tp = set(results["ground_truth_comparison"]["true_positives"])
        assert "1RHS-P-001A" in tp
        assert "1RHS-E-001A" in tp

    def test_false_positive_identified(self, results):
        """1CSP-P-001B is a known false positive (trend signal only)."""
        fp = results["ground_truth_comparison"]["false_positives"]
        assert "1CSP-P-001B" in fp

    def test_nlp_gate_passes(self, results):
        assert results["stage_b"]["nlp_quality"]["quality_gate_passed"] is True

    def test_recommendations_have_finding_and_recommendation(self, results):
        for cid, rec in results["stage_g"]["recommendations"].items():
            assert "finding" in rec, f"Missing 'finding' on {cid}"
            assert "recommendation" in rec, f"Missing 'recommendation' on {cid}"
            assert rec["finding"], f"Empty 'finding' on {cid}"
