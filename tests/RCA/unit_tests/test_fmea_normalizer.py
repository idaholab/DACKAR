"""
test_fmea_normalizer.py — standalone unit tests for doc_parsers.fmea_normalizer

Run directly:   python test_fmea_normalizer.py
Or via pytest:  pytest test_fmea_normalizer.py
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from doc_parsers.fmea_normalizer import (
    normalize_fmea_records,
    FIELD_STATUS_DERIVED,
    FIELD_STATUS_MISSING_CRITICAL,
    FIELD_STATUS_MISSING_ENRICHMENT,
    FIELD_STATUS_NLP,
)


def _base_row():
    return {
        "fmea_source_ref": "demo.xlsx",
        "component_type": "centrifugal_pump",
        "failure_mode_name": "seal leakage",
        "failure_mechanism": "thermal fatigue; material degradation",
        "local_effect": "gradual increase in dissolved oxygen",
        "system_effect": "vacuum degradation at condenser train",
        "end_effect": "reduced plant efficiency",
        "severity": 8,
        "occurrence": 4,
        "detection_rating": 3,
    }


def test_rpn_derivation_when_absent():
    row = _base_row()
    row.pop("rpn", None)
    out, _ = normalize_fmea_records([row], profile_name="aiag_4th")
    rec = out[0]
    assert rec["rpn"] == 96
    assert rec["_field_quality"]["rpn"] == FIELD_STATUS_DERIVED
    assert rec.get("_derivation_method", {}).get("rpn") == "_derive_rpn"
    print("  PASS test_rpn_derivation_when_absent")


def test_milstd_criticality_to_severity_derivation():
    row = _base_row()
    row["severity"] = None
    row["criticality"] = "Class II"
    out, _ = normalize_fmea_records([row], profile_name="mil_std_1629a")
    rec = out[0]
    assert rec["severity"] == 8
    assert rec["_field_quality"]["severity"] == FIELD_STATUS_DERIVED
    print("  PASS test_milstd_criticality_to_severity_derivation")


def test_milstd_lambda_to_occurrence_derivation():
    row = _base_row()
    row["occurrence"] = None
    row["failure_rate"] = 0.2
    row["mission_time_hours"] = 10
    out, _ = normalize_fmea_records([row], profile_name="mil_std_1629a")
    rec = out[0]
    assert int(rec["occurrence"]) >= 7
    assert rec["_field_quality"]["occurrence"] == FIELD_STATUS_DERIVED
    print("  PASS test_milstd_lambda_to_occurrence_derivation")


def test_multi_level_effect_separation_preserved():
    row = _base_row()
    out, _ = normalize_fmea_records([row], profile_name="nuclear_generic")
    rec = out[0]
    assert rec["local_effect"] is not None
    assert rec["system_effect"] is not None
    assert rec["end_effect"] is not None
    print("  PASS test_multi_level_effect_separation_preserved")


def test_nlp_pattern_inference_flagged():
    row = _base_row()
    row["expected_anomaly_pattern"] = None
    out, _ = normalize_fmea_records([row], profile_name="aiag_4th")
    rec = out[0]
    assert rec["expected_anomaly_pattern"] in {
        "gradual_drift", "step_change", "spike", "oscillation", "dropout", "sustained_exceedance", "unknown"
    }
    assert rec["_field_quality"]["expected_anomaly_pattern"] == FIELD_STATUS_NLP
    print("  PASS test_nlp_pattern_inference_flagged")


def test_missing_critical_field_marked():
    row = _base_row()
    row["failure_mechanism"] = None
    out, report = normalize_fmea_records([row], profile_name="aiag_4th")
    rec = out[0]
    assert rec["_field_quality"]["failure_mechanism"] == FIELD_STATUS_MISSING_CRITICAL
    assert int(report["critical_field_missing_count"]) >= 1
    print("  PASS test_missing_critical_field_marked")


def test_missing_enrichment_fields_marked_noncritical():
    row = _base_row()
    row["expected_latency_min_hours"] = None
    row["expected_latency_max_hours"] = None
    out, report = normalize_fmea_records([row], profile_name="aiag_4th")
    rec = out[0]
    assert rec["_field_quality"]["expected_latency_min_hours"] == FIELD_STATUS_MISSING_ENRICHMENT
    assert rec["_field_quality"]["expected_latency_max_hours"] == FIELD_STATUS_MISSING_ENRICHMENT
    assert int(report["critical_field_missing_count"]) == 0
    assert int(report["enrichment_field_missing_count"]) >= 2
    print("  PASS test_missing_enrichment_fields_marked_noncritical")


def test_autodetect_profile_milstd_when_lambda_present():
    row = _base_row()
    row["failure_rate"] = 0.05
    row["mission_time_hours"] = 100
    row["occurrence"] = None
    out, report = normalize_fmea_records([row], profile_name="auto")
    rec = out[0]
    assert report["profile_used"] == "mil_std_1629a"
    assert rec["_normalization_profile"] == "mil_std_1629a"
    print("  PASS test_autodetect_profile_milstd_when_lambda_present")


def test_cause_effect_split_marked_nlp_inferred():
    row = _base_row()
    row["potential_causes"] = None
    row["local_effect"] = None
    row["failure_mechanism"] = "Seal wear due to abrasive contamination"
    out, _ = normalize_fmea_records([row], profile_name="aiag_4th")
    rec = out[0]
    assert isinstance(rec.get("potential_causes"), list) and rec["potential_causes"]
    assert rec["_field_quality"]["potential_causes"] == FIELD_STATUS_NLP
    assert rec["_field_quality"]["local_effect"] == FIELD_STATUS_NLP
    methods = rec.get("_derivation_method", {})
    assert methods.get("potential_causes", "").startswith("nlp:")
    assert methods.get("local_effect", "").startswith("nlp:")
    print("  PASS test_cause_effect_split_marked_nlp_inferred")


ALL_TESTS = [
    test_rpn_derivation_when_absent,
    test_milstd_criticality_to_severity_derivation,
    test_milstd_lambda_to_occurrence_derivation,
    test_multi_level_effect_separation_preserved,
    test_nlp_pattern_inference_flagged,
    test_missing_critical_field_marked,
    test_missing_enrichment_fields_marked_noncritical,
    test_autodetect_profile_milstd_when_lambda_present,
    test_cause_effect_split_marked_nlp_inferred,
]


def run_all():
    print(f"\n=== test_fmea_normalizer ({len(ALL_TESTS)} tests) ===")
    passed, failed = 0, 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

