"""
test_fmea_parser_normalization.py — smoke tests for parser+normalizer integration.
"""
import sys
from pathlib import Path
import tempfile

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from doc_parsers.fmeaParser import parse_fmea_file, parse_fmea_files


def test_parse_csv_with_normalization_metadata():
    csv_text = (
        "Component Type,Failure Mode,Failure Mechanism,Severity,Occurrence,Detection,Local Effect\n"
        "centrifugal_pump,seal leakage,thermal fatigue,8,4,3,gradual dissolved oxygen increase\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fmea.csv"
        p.write_text(csv_text, encoding="utf-8")
        rows = parse_fmea_file(p, profile_name="aiag_4th", include_normalization_metadata=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["rpn"] == 96
    assert row["detection_rating"] == 3
    assert row["_normalization_profile"] == "aiag_4th"
    assert isinstance(row.get("_fmea_ingestion_quality"), dict)
    print("  PASS test_parse_csv_with_normalization_metadata")


def test_parse_csv_without_normalization_metadata():
    csv_text = (
        "Component Type,Failure Mode,Failure Mechanism,Severity,Occurrence,Detection\n"
        "centrifugal_pump,seal leakage,thermal fatigue,8,4,3\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fmea.csv"
        p.write_text(csv_text, encoding="utf-8")
        rows = parse_fmea_file(p, profile_name="aiag_4th", include_normalization_metadata=False)
    assert len(rows) == 1
    row = rows[0]
    assert "_field_quality" not in row
    assert "_normalization_profile" not in row
    print("  PASS test_parse_csv_without_normalization_metadata")


def test_profile_specific_column_mapping_milstd():
    csv_text = (
        "Item,Potential Failure Mode,Cause of Failure,Criticality,Failure Rate (λ),Mission Time (Hrs)\n"
        "feed_pump,bearing seizure,loss of lubrication,Class II,0.2,10\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fmea_milstd.csv"
        p.write_text(csv_text, encoding="utf-8")
        rows = parse_fmea_file(p, profile_name="mil_std_1629a", include_normalization_metadata=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["failure_mechanism"] == "loss of lubrication"
    assert row["severity"] == 8
    assert int(row["occurrence"]) >= 7
    assert row["_normalization_profile"] == "mil_std_1629a"
    quality = row.get("_fmea_ingestion_quality") or {}
    assert int(quality.get("critical_field_missing_count", 0)) == 0
    print("  PASS test_profile_specific_column_mapping_milstd")


def test_missing_failure_mechanism_column_hard_fails():
    csv_text = (
        "Component Type,Failure Mode,Severity,Occurrence,Detection\n"
        "centrifugal_pump,seal leakage,8,4,3\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fmea_missing_mech.csv"
        p.write_text(csv_text, encoding="utf-8")
        try:
            parse_fmea_file(p, profile_name="aiag_4th", include_normalization_metadata=True)
            raise AssertionError("Expected ValueError for missing failure_mechanism column.")
        except ValueError as exc:
            assert "failure_mechanism" in str(exc)
    print("  PASS test_missing_failure_mechanism_column_hard_fails")


def test_end_effect_header_not_mapped_to_local_effect():
    csv_text = (
        "Component Type,Failure Mode,Failure Mechanism,End Effect,Severity,Occurrence,Detection\n"
        "centrifugal_pump,seal leakage,thermal fatigue,reduced plant efficiency,8,4,3\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fmea_end_effect.csv"
        p.write_text(csv_text, encoding="utf-8")
        rows = parse_fmea_file(p, profile_name="aiag_4th", include_normalization_metadata=True)
    assert len(rows) == 1
    row = rows[0]
    assert row.get("end_effect") == "reduced plant efficiency"
    assert row.get("local_effect") is None
    print("  PASS test_end_effect_header_not_mapped_to_local_effect")


def test_parse_multiple_files_aggregates_ingestion_report():
    csv_text_1 = (
        "Component Type,Failure Mode,Failure Mechanism,Severity,Occurrence,Detection\n"
        "centrifugal_pump,seal leakage,thermal fatigue,8,4,3\n"
    )
    csv_text_2 = (
        "Component Type,Failure Mode,Failure Mechanism,Severity,Occurrence,Detection\n"
        "control_valve,stiction,packing wear,6,5,4\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "fmea_1.csv"
        p2 = Path(td) / "fmea_2.csv"
        p1.write_text(csv_text_1, encoding="utf-8")
        p2.write_text(csv_text_2, encoding="utf-8")
        rows = parse_fmea_files([p1, p2], profile_name="aiag_4th", include_normalization_metadata=True)
    assert len(rows) == 2
    report = rows[0].get("_fmea_ingestion_quality") or {}
    assert int(report.get("total_fms_ingested", 0)) == 2
    assert rows[1].get("_fmea_ingestion_quality") == report
    print("  PASS test_parse_multiple_files_aggregates_ingestion_report")


ALL_TESTS = [
    test_parse_csv_with_normalization_metadata,
    test_parse_csv_without_normalization_metadata,
    test_profile_specific_column_mapping_milstd,
    test_missing_failure_mechanism_column_hard_fails,
    test_end_effect_header_not_mapped_to_local_effect,
    test_parse_multiple_files_aggregates_ingestion_report,
]


def run_all():
    print(f"\n=== test_fmea_parser_normalization ({len(ALL_TESTS)} tests) ===")
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

