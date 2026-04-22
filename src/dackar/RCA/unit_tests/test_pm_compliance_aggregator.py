"""Tests for ``pm_compliance`` artifact builder and schema conformity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from jsonschema import Draft7Validator, FormatChecker  # type: ignore[import]

from pm_compliance import PMComplianceConfig, build_pm_compliance
from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32, CausalityEngineConfigV32

from pm_compliance.aggregator import _rollup_risk
from pm_compliance.execution_verifier import PMExecutionVerifier


def _schema_validator():
    rca = Path(__file__).resolve().parents[1]
    p = rca / "schemas" / "pm_compliance.json"
    with open(p, encoding="utf-8") as f:
        schema = json.load(f)
    return Draft7Validator(schema, format_checker=FormatChecker())


def test_build_minimal_compliant_artifact():
    event = {
        "asset_id": "ASSET-1",
        "event_id": "EVT-1",
        "timestamp_start": "2024-06-15T12:00:00+00:00",
    }
    v = _schema_validator()
    art = build_pm_compliance(event, kg_context=None, export_rows=[])
    v.validate(art)
    assert art["summary"]["total_checks"] == 0
    assert art["fmea_pm_linkage_available"] is False


def test_build_with_overdue_inspection_fails_governance_relevance():
    event = {
        "asset_id": "ASSET-1",
        "timestamp_start": "2024-06-15T12:00:00+00:00",
    }
    # Next due one month before event → fail + overdue
    rows = [
        {
            "check_id": "PM-INSP-1",
            "check_type": "inspection",
            "next_due_date": "2024-05-15T00:00:00+00:00",
            "last_pm_date": "2023-11-15T00:00:00+00:00",
        }
    ]
    art = build_pm_compliance(
        event,
        export_rows=rows,
        config=PMComplianceConfig(look_back_window_days=365),
    )
    v = _schema_validator()
    v.validate(art)
    assert any(c.get("status") == "fail" for c in art["checks"])
    assert art["summary"]["failed"] >= 1
    if art.get("overdue_items"):
        assert len(art["overdue_items"]) >= 1


def test_schema_required_check_fields_present():
    art = build_pm_compliance(
        {
            "asset_id": "A",
            "timestamp_start": "2024-01-01T00:00:00+00:00",
        },
        export_rows=[{"check_id": "C1", "check_type": "lubrication", "compliance_status": "compliant"}],
    )
    c0 = art["checks"][0]
    for k in ("check_id", "check_type", "status", "overdue_by_days"):
        assert k in c0


def test_rollup_risk_all_clear():
    a, b, c = _rollup_risk(None, set(), False, False)
    assert a == "compliant" and b == "low" and c is False


def test_rollup_risk_high_primary_gap_and_overdue():
    a, b, c = _rollup_risk("FM-1", {"FM-1", "FM-2"}, True, True)
    assert b == "high" and c is True
    a2, b2, c2 = _rollup_risk("FM-1", set(), True, True)
    assert b2 == "medium"  # overdue but not in (empty) gap
    assert c2 is False


def test_verifier_unknown_when_no_dates():
    v = PMExecutionVerifier(event_timestamp_iso="2024-06-15T12:00:00+00:00")
    checks, notes = v.verify_rows([{"check_id": "X", "check_type": "other"}])
    assert checks[0]["status"] == "unknown"
    assert any("no schedule" in n.lower() for n in notes)


def test_fmea_linkage_false_when_only_export_applicable_fm():
    """§3.3: KG FMEA/PM tag absent → fmea_pm_linkage_available False even with applicable_fm_ids."""
    kg = {
        "failure_modes": [{"fm_id": "FM-1", "name": "leakage"}],
        "components": [],
    }
    art = build_pm_compliance(
        {"asset_id": "A1", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context=kg,
        export_rows=[
            {
                "check_id": "PM-1",
                "check_type": "inspection",
                "compliance_status": "compliant",
                "applicable_fm_ids": ["FM-1"],
            }
        ],
    )
    assert art["fmea_pm_linkage_available"] is False
    assert any("applicable_fm_ids" in n for n in art["data_quality_notes"])


def test_not_applicable_row_is_pass_for_governance():
    v = PMExecutionVerifier(event_timestamp_iso="2024-08-01T00:00:00+00:00")
    checks, _ = v.verify_rows(
        [{"check_id": "PM-CBM", "check_type": "other", "compliance_status": "not_applicable"}]
    )
    assert checks[0]["status"] == "pass"


def test_governance_engine_accepts_artifact_from_builder():
    eng = RuleBasedCausalityEngineV32(CausalityEngineConfigV32())
    art = build_pm_compliance(
        {
            "asset_id": "A1",
            "timestamp_start": "2024-08-10T00:00:00+00:00",
        },
        export_rows=[
            {
                "check_id": "PM-1",
                "check_type": "inspection",
                "compliance_status": "fail",
                "overdue_by_days": 40,
            }
        ],
    )
    g = eng._governance_details(art, fm_name="leakage along boundary")
    assert g["pm_data_available"] is True
    assert g["score"] > 0.5


if __name__ == "__main__":  # pragma: no cover
    test_build_minimal_compliant_artifact()
    test_build_with_overdue_inspection_fails_governance_relevance()
    test_schema_required_check_fields_present()
    test_rollup_risk_all_clear()
    test_rollup_risk_high_primary_gap_and_overdue()
    test_fmea_linkage_false_when_only_export_applicable_fm()
    test_not_applicable_row_is_pass_for_governance()
    test_governance_engine_accepts_artifact_from_builder()
    test_verifier_unknown_when_no_dates()
    print("pm_compliance tests OK")
