"""Tests for ``pm_compliance`` artifact builder and schema conformity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from jsonschema import Draft7Validator, FormatChecker  # type: ignore[import]

from pm_compliance import PMComplianceConfig, build_pm_compliance
from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32, CausalityEngineConfigV32

from pm_compliance.aggregator import _rollup_risk
from pm_compliance.execution_verifier import PMExecutionVerifier


def _schema_validator():
    rca = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
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


def test_primary_scope_gap_marks_non_compliant_when_linkage_available():
    kg = {
        "components": [{"component_id": "COMP-1"}],
        "failure_modes": [
            {
                "fm_id": "FM-1",
                "pm_task_ids": ["PM-X"],  # explicit linkage present, but does not match executed task
            }
        ],
    }
    art = build_pm_compliance(
        {
            "asset_id": "ASSET-1",
            "timestamp_start": "2024-08-10T00:00:00+00:00",
        },
        kg_context=kg,
        export_rows=[
            {
                "check_id": "PM-1",
                "check_type": "inspection",
                "component_id": "COMP-1",
                "compliance_status": "compliant",
            }
        ],
        primary_fm_id="FM-1",
    )
    assert art["fmea_pm_linkage_available"] is True
    assert art["summary"]["overall_compliance"] == "non_compliant"
    assert art["summary"]["has_scope_gaps_for_primary_fm"] is True


def test_not_applicable_preserved_in_pm_tasks_narrative_status():
    art = build_pm_compliance(
        {
            "asset_id": "ASSET-1",
            "timestamp_start": "2024-08-10T00:00:00+00:00",
        },
        kg_context={"components": [], "failure_modes": []},
        export_rows=[
            {
                "check_id": "PM-CBM-1",
                "check_type": "other",
                "compliance_status": "not_applicable",
            }
        ],
    )
    assert art["checks"][0]["status"] == "pass"
    assert art["components"][0]["pm_tasks"][0]["compliance_status"] == "not_applicable"


def test_degradation_trend_uses_as_found_fields_from_rows():
    art = build_pm_compliance(
        {
            "asset_id": "ASSET-1",
            "timestamp_start": "2024-08-10T00:00:00+00:00",
        },
        kg_context={"components": [], "failure_modes": []},
        export_rows=[
            {
                "check_id": "PM-1",
                "check_type": "inspection",
                "compliance_status": "compliant",
                "as_found_last": "Found degraded bearing surface and increased wear",
            }
        ],
    )
    assert art["components"][0]["degradation_trend"] == "degrading"


def test_loader_drops_rows_missing_required_identity_or_type_with_note():
    art = build_pm_compliance(
        {
            "asset_id": "ASSET-1",
            "timestamp_start": "2024-08-10T00:00:00+00:00",
        },
        export_rows=[
            {
                "check_id": "PM-1",
                # missing check_type -> dropped
                "compliance_status": "compliant",
            }
        ],
    )
    assert art["summary"]["total_checks"] == 0
    assert any("Dropped PM export row missing required" in n for n in art["data_quality_notes"])


# ---------------------------------------------------------------------------
# Wave 2 — vocabulary loader and pm_found_defect_rate
# ---------------------------------------------------------------------------

def _make_vocab_dir(tmp_path, neg_terms, pos_terms):
    """Write minimal keyword CSVs into *tmp_path* and return the Path."""
    import csv
    for fname, terms in (
        ("health_status_keywords_negative.csv", neg_terms),
        ("health_status_keywords_positive.csv", pos_terms),
    ):
        with open(tmp_path / fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Nouns", "Verbs", "Adjectives"])
            writer.writeheader()
            for t in terms:
                writer.writerow({"Nouns": t, "Verbs": "", "Adjectives": ""})
    return tmp_path


def test_analyze_degradation_uses_vocabulary_csv(tmp_path):
    """Vocabulary-driven matching must classify wear, leak, corrosion as degrading."""
    from pm_compliance.vocabulary_loader import PMVocabularyLoader
    PMVocabularyLoader.clear_cache()
    data_dir = _make_vocab_dir(
        tmp_path,
        neg_terms=["wear", "leak", "corrosion", "crack", "vibration"],
        pos_terms=["acceptable", "normal"],
    )
    from pm_compliance.effectiveness_analyzer import analyze_degradation
    assert analyze_degradation(["bearing wear observed"], data_dir=data_dir) == "degrading"
    assert analyze_degradation(["leak found at seal"], data_dir=data_dir) == "degrading"
    assert analyze_degradation(["corrosion on casing"], data_dir=data_dir) == "degrading"
    assert analyze_degradation(["pump shaft cracked"], data_dir=data_dir) == "degrading"
    assert analyze_degradation(["anomalous vibration"], data_dir=data_dir) == "degrading"
    PMVocabularyLoader.clear_cache()


def test_analyze_degradation_improving_not_shadowed_by_degrading(tmp_path):
    """Improving beats stable but must NOT override an explicit degrading signal."""
    from pm_compliance.vocabulary_loader import PMVocabularyLoader
    PMVocabularyLoader.clear_cache()
    data_dir = _make_vocab_dir(tmp_path, neg_terms=["leak"], pos_terms=["acceptable"])
    from pm_compliance.effectiveness_analyzer import analyze_degradation
    # Improving with no degrading → improving
    assert analyze_degradation(["found acceptable"], data_dir=data_dir) == "improving"
    # Both improving and degrading → degrading wins
    assert analyze_degradation(["acceptable but leak present"], data_dir=data_dir) == "degrading"
    PMVocabularyLoader.clear_cache()


def test_analyze_degradation_fallback_when_no_data_dir():
    """Without data_dir, hardcoded fallback stems still work."""
    from pm_compliance.effectiveness_analyzer import analyze_degradation
    assert analyze_degradation(["pump degraded significantly"]) == "degrading"
    assert analyze_degradation(["no defect found"]) == "improving"
    assert analyze_degradation(["pump running normally"]) == "improving"


def test_word_boundary_prefix_matching():
    """Prefix boundary matching: 'leak' matches 'leakage'; 'crack' matches 'cracks'.
    A term must start at a word boundary — it must not match in the middle of a word.
    """
    from pm_compliance.vocabulary_loader import PMVocabularyLoader, matches_any
    PMVocabularyLoader.clear_cache()
    degrading_terms = frozenset({"leak", "crack"})
    # Prefix of word — intentional matches
    assert matches_any("leakage observed at seal", degrading_terms)
    assert matches_any("several cracks on shaft", degrading_terms)
    # Term in middle of longer word — must not match
    assert not matches_any("bleak outlook", degrading_terms)    # 'leak' has no leading \b
    assert not matches_any("firecracker residue", degrading_terms)  # 'crack' has no leading \b


def test_pm_found_defect_rate_computed_correctly(tmp_path):
    """pm_found_defect_rate must equal defect_rows / total_rows_with_asf."""
    from pm_compliance.vocabulary_loader import PMVocabularyLoader
    PMVocabularyLoader.clear_cache()
    data_dir = _make_vocab_dir(tmp_path, neg_terms=["degraded", "leak"], pos_terms=["acceptable"])
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context={"components": [], "failure_modes": []},
        export_rows=[
            {"check_id": "PM-1", "check_type": "inspection", "compliance_status": "compliant",
             "as_found_last": "component degraded"},     # defect
            {"check_id": "PM-2", "check_type": "inspection", "compliance_status": "compliant",
             "as_found_last": "leak at flange"},         # defect
            {"check_id": "PM-3", "check_type": "inspection", "compliance_status": "compliant",
             "as_found_last": "found acceptable"},       # no defect
            {"check_id": "PM-4", "check_type": "inspection", "compliance_status": "compliant"},  # no asf
        ],
        config=PMComplianceConfig(data_dir=data_dir),
    )
    assert "pm_found_defect_rate" in art["summary"]
    # 2 defects out of 3 rows with as-found data
    assert abs(art["summary"]["pm_found_defect_rate"] - 2/3) < 1e-5
    PMVocabularyLoader.clear_cache()


def test_pm_found_defect_rate_absent_when_no_asf_data():
    """pm_found_defect_rate must be absent (not 0.0) when no as-found data exists."""
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        export_rows=[{"check_id": "PM-1", "check_type": "inspection", "compliance_status": "compliant"}],
    )
    assert "pm_found_defect_rate" not in art["summary"]


def test_schema_validates_artifact_with_pm_found_defect_rate(tmp_path):
    """Schema must accept pm_found_defect_rate in summary."""
    from pm_compliance.vocabulary_loader import PMVocabularyLoader
    PMVocabularyLoader.clear_cache()
    data_dir = _make_vocab_dir(tmp_path, neg_terms=["degraded"], pos_terms=["acceptable"])
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context={"components": [], "failure_modes": []},
        export_rows=[
            {"check_id": "PM-1", "check_type": "inspection", "compliance_status": "compliant",
             "as_found_last": "component degraded"},
        ],
        config=PMComplianceConfig(data_dir=data_dir),
    )
    _schema_validator().validate(art)
    PMVocabularyLoader.clear_cache()


def test_real_data_dir_classifies_domain_texts():
    """Smoke-test against the actual DACKAR data directory (skipped if absent)."""
    import pytest
    data_dir = Path(__file__).resolve().parents[3] / "data"
    if not data_dir.exists():
        pytest.skip("DACKAR data directory not found")
    from pm_compliance.vocabulary_loader import PMVocabularyLoader
    PMVocabularyLoader.clear_cache()
    from pm_compliance.effectiveness_analyzer import analyze_degradation
    # From raw_text.txt / comp_testing_examples.txt observed conditions
    assert analyze_degradation(["Rupture of pump bearings caused pump shaft degradation"], data_dir=data_dir) == "degrading"
    assert analyze_degradation(["Several cracks on pump shaft were observed"], data_dir=data_dir) == "degrading"
    # "Satisfactory" is in the positive CSV adjectives; "acceptable" is in the neutral CSV only
    assert analyze_degradation(["Pump running in satisfactory condition"], data_dir=data_dir) == "improving"
    PMVocabularyLoader.clear_cache()


# ---------------------------------------------------------------------------
# Wave 3 — coverage_type derivation and unknown narrative fix
# ---------------------------------------------------------------------------

def test_coverage_type_preventive_from_preventing_pm_task_ids():
    """§2.3: task linked via preventing_pm_task_ids must get coverage_type='preventive'."""
    kg = {
        "components": [{"component_id": "COMP-1"}],
        "failure_modes": [
            {"fm_id": "FM-1", "preventing_pm_task_ids": ["PM-PREV"]},
        ],
    }
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context=kg,
        export_rows=[
            {"check_id": "PM-PREV", "check_type": "inspection",
             "component_id": "COMP-1", "compliance_status": "compliant"},
        ],
    )
    task = art["components"][0]["pm_tasks"][0]
    assert task["coverage_type"] == "preventive"


def test_coverage_type_detective_from_detecting_pm_task_ids():
    """§2.3: task linked via detecting_pm_task_ids must get coverage_type='detective'."""
    kg = {
        "components": [{"component_id": "COMP-1"}],
        "failure_modes": [
            {"fm_id": "FM-1", "detecting_pm_task_ids": ["PM-DET"]},
        ],
    }
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context=kg,
        export_rows=[
            {"check_id": "PM-DET", "check_type": "surveillance_test",
             "component_id": "COMP-1", "compliance_status": "compliant"},
        ],
    )
    task = art["components"][0]["pm_tasks"][0]
    assert task["coverage_type"] == "detective"


def test_coverage_type_preventive_wins_over_detective_when_both_linked():
    """§2.3: when a task appears in both preventing and detecting fields, 'preventive' wins."""
    kg = {
        "components": [{"component_id": "COMP-1"}],
        "failure_modes": [
            {
                "fm_id": "FM-1",
                "preventing_pm_task_ids": ["PM-BOTH"],
                "detecting_pm_task_ids": ["PM-BOTH"],
            },
        ],
    }
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context=kg,
        export_rows=[
            {"check_id": "PM-BOTH", "check_type": "inspection",
             "component_id": "COMP-1", "compliance_status": "compliant"},
        ],
    )
    task = art["components"][0]["pm_tasks"][0]
    assert task["coverage_type"] == "preventive"


def test_coverage_type_none_when_no_kg_linkage():
    """§2.3: without KG linkage, coverage_type falls back to export row value or 'none'."""
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context={"components": [], "failure_modes": []},
        export_rows=[
            {"check_id": "PM-1", "check_type": "inspection", "compliance_status": "compliant"},
        ],
    )
    task = art["components"][0]["pm_tasks"][0]
    assert task["coverage_type"] == "none"


def test_coverage_type_export_row_preserved_when_no_kg_linkage():
    """§2.3: export row coverage_type is used when KG linkage is absent."""
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context={"components": [], "failure_modes": []},
        export_rows=[
            {"check_id": "PM-1", "check_type": "inspection",
             "compliance_status": "compliant", "coverage_type": "detective"},
        ],
    )
    task = art["components"][0]["pm_tasks"][0]
    assert task["coverage_type"] == "detective"


def test_unknown_status_shows_undetermined_in_narrative():
    """§1.4: a check with no schedule dates must appear as 'undetermined', not 'compliant'."""
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context={"components": [], "failure_modes": []},
        export_rows=[
            # No dates → verifier marks status=unknown
            {"check_id": "PM-NODATE", "check_type": "inspection"},
        ],
    )
    assert art["checks"][0]["status"] == "unknown"          # governance path unchanged
    task = art["components"][0]["pm_tasks"][0]
    assert task["compliance_status"] == "undetermined"      # narrative fixed


def test_assessment_date_is_build_time_not_event_time():
    """§1.1 fix: assessment_date must differ from event timestamp so the staleness guard fires."""
    event_ts = "2024-06-15T12:00:00+00:00"
    art = build_pm_compliance({"asset_id": "A", "timestamp_start": event_ts})
    assert art["assessment_date"] != event_ts, (
        "assessment_date must be build time (utcnow), not the event timestamp"
    )
    # window.end still encodes the event reference time
    assert art["window"]["end"].startswith("2024-06-15")


def test_verifier_tz_naive_next_due_does_not_raise():
    """§1.3 fix: tz-naive next_due_date must not cause TypeError when compared to tz-aware event_dt."""
    v = PMExecutionVerifier(event_timestamp_iso="2024-06-15T12:00:00+00:00")
    rows = [
        {
            "check_id": "PM-TZ",
            "check_type": "inspection",
            # tz-naive — would previously raise TypeError on comparison
            "next_due_date": "2024-05-01T00:00:00",
        }
    ]
    checks, _ = v.verify_rows(rows)
    assert checks[0]["status"] == "fail"
    assert checks[0]["overdue_by_days"] > 0


def test_dq_note_emitted_when_primary_fm_id_given_but_no_kg_linkage():
    """§2.4 fix: silent risk underestimation must surface as a data_quality_note."""
    art = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context={"failure_modes": [{"fm_id": "FM-99"}], "components": []},
        export_rows=[{"check_id": "PM-1", "check_type": "inspection", "compliance_status": "compliant"}],
        primary_fm_id="FM-99",
    )
    assert art["fmea_pm_linkage_available"] is False
    assert any("FM-99" in n and "not evaluable" in n for n in art["data_quality_notes"])


def test_effectiveness_lookback_cycles_limits_rows_used():
    """§3.1 fix: effectiveness_lookback_cycles must cap the as-found rows analysed."""
    from pm_compliance import PMComplianceConfig
    rows = [
        {"check_id": f"PM-{i}", "check_type": "inspection", "compliance_status": "compliant",
         "completed_date": f"2024-0{i}-01T00:00:00+00:00",
         "as_found_last": "degraded" if i <= 2 else "acceptable"}
        for i in range(1, 5)
    ]
    # With max 1 cycle (most recent = PM-4 with "acceptable") trend should not be "degrading"
    art_1 = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context={"components": [], "failure_modes": []},
        export_rows=rows,
        config=PMComplianceConfig(effectiveness_lookback_cycles=1),
    )
    assert art_1["components"][0]["degradation_trend"] in ("improving", "stable")

    # With all 4 cycles, the two "degraded" rows drive the trend to "degrading"
    art_4 = build_pm_compliance(
        {"asset_id": "A", "timestamp_start": "2024-08-01T00:00:00+00:00"},
        kg_context={"components": [], "failure_modes": []},
        export_rows=rows,
        config=PMComplianceConfig(effectiveness_lookback_cycles=4),
    )
    assert art_4["components"][0]["degradation_trend"] == "degrading"


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
    test_primary_scope_gap_marks_non_compliant_when_linkage_available()
    test_not_applicable_preserved_in_pm_tasks_narrative_status()
    test_degradation_trend_uses_as_found_fields_from_rows()
    test_loader_drops_rows_missing_required_identity_or_type_with_note()
    print("pm_compliance tests OK")
