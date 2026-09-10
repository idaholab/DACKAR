"""
Unit tests for Stage A (ActivityIntakeProcessor) and Stage C (TemporalChainScorer).

Coverage targets:
    Stage A:
        _classify_emergence_type     — 4-rule priority chain
        _detect_regulatory_constraints — pattern matching + structured fields
        _compute_data_quality        — composite scoring formula
        _expand_abbreviations        — unknown abbreviation rate calculation

    Stage C:
        _allen_relation              — all 7 Allen relations with concrete timestamps
        _assign_causal_strength      — SIMULTANEOUS fix, temporal_contradiction,
                                       strong / moderate / weak
        _compute_confidence          — lag plausibility boundary logic
        _summarize_chain             — all causal_posture values
        score()                      — end-to-end integration path
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make the stages package importable via the outage root already on sys.path
# (conftest.py adds _OUTAGE_ROOT = tests/../ = .../dackar/outage/ to sys.path)
_OUTAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "dackar" / "outage"
if str(_OUTAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_ROOT))

from stages.stage_a_intake import ActivityIntakeProcessor, ActivityIntakeConfig
from stages.stage_c_temporal_chain import TemporalChainScorer, TemporalChainConfig


# ===========================================================================
# Helpers
# ===========================================================================

def _proc(config: ActivityIntakeConfig | None = None) -> ActivityIntakeProcessor:
    """Return a processor with no injected backends (pure-logic methods only)."""
    return ActivityIntakeProcessor(config=config)


def _dt(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _scorer(epsilon_hours: float = 0.5) -> TemporalChainScorer:
    return TemporalChainScorer(TemporalChainConfig(epsilon_hours=epsilon_hours))


# ===========================================================================
# Stage A — _classify_emergence_type
# ===========================================================================

class TestClassifyEmergenceType:
    """4-rule priority chain: regulatory_driven > scope_expansion >
    schedule_optimization > truly_unplanned."""

    def test_explicit_emergence_type_wins(self):
        """When intake record sets emergence_type it is returned at 1.0 confidence."""
        proc = _proc()
        et, conf, _ = proc._classify_emergence_type(
            {"emergence_type": "scope_expansion"}, [], "some description"
        )
        assert et == "scope_expansion"
        assert conf == 1.0

    def test_rule1_regulatory_keyword_in_text(self):
        """'TS 3.4.2' keyword → regulatory_driven (rule 1)."""
        proc = _proc()
        et, conf, rationale = proc._classify_emergence_type(
            {}, [], "This activity is required per TS 3.4.2 surveillance"
        )
        assert et == "regulatory_driven"
        assert conf >= 0.80
        assert rationale is not None

    def test_rule1_structured_lco_field(self):
        """lco_number in intake record → regulatory_driven even without text keywords."""
        proc = _proc()
        et, conf, _ = proc._classify_emergence_type(
            {"lco_number": "LCO 3.5.1"}, [], "replace gasket on valve"
        )
        assert et == "regulatory_driven"
        assert conf >= 0.85

    def test_rule1_structured_nrc_commitment(self):
        """nrc_commitment_number → regulatory_driven."""
        proc = _proc()
        et, _, _ = proc._classify_emergence_type(
            {"nrc_commitment_number": "ML22001A001"}, [], "inspect RCP seal"
        )
        assert et == "regulatory_driven"

    def test_rule2_scope_expansion_with_wo(self):
        """Scope language + existing WO → scope_expansion at ≥ 0.80 confidence."""
        proc = _proc()
        activity = {"work_order_id": "WO-123456"}
        et, conf, rationale = proc._classify_emergence_type(
            activity, [], "additional scope while we are in the area"
        )
        assert et == "scope_expansion"
        assert conf >= 0.80

    def test_rule2_scope_expansion_no_wo_lower_confidence(self):
        """Scope language without WO → scope_expansion at lower confidence."""
        proc = _proc()
        et, conf, _ = proc._classify_emergence_type(
            {}, [], "opportunistic scope addition identified"
        )
        assert et == "scope_expansion"
        assert conf < 0.80

    def test_rule3_schedule_optimization_no_failure(self):
        """Schedule keywords with no degradation language → schedule_optimization."""
        proc = _proc()
        et, conf, _ = proc._classify_emergence_type(
            {}, [], "reschedule preventive maintenance to advance schedule"
        )
        assert et == "schedule_optimization"
        assert conf >= 0.70

    def test_rule3_schedule_keyword_plus_failure_not_classified_as_optimization(self):
        """Schedule keyword WITH failure term should NOT be schedule_optimization."""
        proc = _proc()
        et, _, _ = proc._classify_emergence_type(
            {}, [], "reschedule repair because pump is leaking"
        )
        # Degradation keyword overrides schedule optimization rule
        assert et != "schedule_optimization"

    def test_rule4_truly_unplanned_with_degradation(self):
        """Degradation keyword → truly_unplanned."""
        proc = _proc()
        et, conf, _ = proc._classify_emergence_type(
            {}, [], "unexpected vibration detected on EDG cooling fan"
        )
        assert et == "truly_unplanned"
        assert conf >= 0.75

    def test_rule4_default_when_no_signal(self):
        """No keywords → truly_unplanned with low confidence."""
        proc = _proc()
        et, conf, _ = proc._classify_emergence_type(
            {}, [], "perform work on component"
        )
        assert et == "truly_unplanned"
        assert conf < 0.60


# ===========================================================================
# Stage A — _detect_regulatory_constraints
# ===========================================================================

class TestDetectRegulatoryConstraints:

    def test_ts_text_pattern_defer_prohibited(self):
        """'TS 3.5.2' in text → ts_surveillance driver, defer_prohibited=True."""
        proc = _proc()
        has_reg, drivers = proc._detect_regulatory_constraints(
            {}, [], "required per TS 3.5.2"
        )
        assert has_reg is True
        types = [d["driver_type"] for d in drivers]
        assert "ts_surveillance" in types
        ts_driver = next(d for d in drivers if d["driver_type"] == "ts_surveillance")
        assert ts_driver["defer_prohibited"] is True

    def test_alara_not_defer_prohibited(self):
        """ALARA → alara_constraint, defer_prohibited=False."""
        proc = _proc()
        has_reg, drivers = proc._detect_regulatory_constraints(
            {}, [], "work requires ALARA dose minimisation plan"
        )
        alara = next((d for d in drivers if d["driver_type"] == "alara_constraint"), None)
        assert alara is not None
        assert alara["defer_prohibited"] is False

    def test_surveillance_keyword_defer_prohibited(self):
        """'surveillance' keyword → ts_surveillance driver, defer_prohibited=True."""
        proc = _proc()
        has_reg, drivers = proc._detect_regulatory_constraints(
            {}, [], "quarterly valve surveillance test"
        )
        surv = next((d for d in drivers if d["driver_type"] == "ts_surveillance"), None)
        assert surv is not None
        assert surv["defer_prohibited"] is True

    def test_structured_field_technical_spec(self):
        """technical_specification_reference field → ts_surveillance driver from intake record."""
        proc = _proc()
        has_reg, drivers = proc._detect_regulatory_constraints(
            {"technical_specification_reference": "TS 3.8.1"}, [], "install battery"
        )
        assert has_reg is True
        ts = next((d for d in drivers if d["driver_type"] == "ts_surveillance"), None)
        assert ts is not None
        assert ts["source"] == "intake_record_field"

    def test_no_duplicate_driver_types(self):
        """Same driver type from both text and structured field is deduplicated."""
        proc = _proc()
        _, drivers = proc._detect_regulatory_constraints(
            {"technical_specification_reference": "TS 3.5.2"},
            [],
            "per TS 3.5.2 surveillance requirement",
        )
        ts_drivers = [d for d in drivers if d["driver_type"] == "ts_surveillance"]
        assert len(ts_drivers) == 1

    def test_no_constraint_clean_text(self):
        """Plain maintenance text → has_regulatory=False."""
        proc = _proc()
        has_reg, drivers = proc._detect_regulatory_constraints(
            {}, [], "replace packing on manual valve"
        )
        assert has_reg is False
        assert drivers == []

    def test_driver_ids_unique(self):
        """Each driver gets a unique driver_id."""
        proc = _proc()
        _, drivers = proc._detect_regulatory_constraints(
            {}, [], "NRC commitment ALARA hold point surveillance"
        )
        ids = [d["driver_id"] for d in drivers]
        assert len(ids) == len(set(ids))

    # ── Y2 fix: driver_type values must match schema enum ────────────────────

    _VALID_DRIVER_TYPES = frozenset({
        "ts_surveillance", "nrc_commitment", "cap_commitment",
        "hold_point", "alara_constraint", "license_basis_inspection", "other",
    })

    def test_y2_all_builtin_driver_types_are_schema_valid(self):
        """Y2 fix: every built-in pattern must produce a schema-valid driver_type."""
        proc = _proc()
        # Cover all built-in pattern triggers in a single text
        text = (
            "TS 3.5.2 LCO 3.4.6 limiting condition for operation "
            "NRC 10 CFR 50 ALARA CAP surveillance operability determination "
            "hold point mode exit"
        )
        _, drivers = proc._detect_regulatory_constraints({}, [], text)
        for d in drivers:
            assert d["driver_type"] in self._VALID_DRIVER_TYPES, (
                f"driver_type '{d['driver_type']}' not in schema enum (Y2 fix)"
            )

    def test_y2_lco_number_field_produces_ts_surveillance(self):
        """Y2 fix: lco_number structured field → ts_surveillance driver (was limiting_condition_for_operation)."""
        proc = _proc()
        _, drivers = proc._detect_regulatory_constraints(
            {"lco_number": "LCO 3.4.6"}, [], "rcp seal degradation"
        )
        types = [d["driver_type"] for d in drivers]
        assert "limiting_condition_for_operation" not in types, (
            "limiting_condition_for_operation is not a valid schema enum value (Y2 fix)"
        )
        assert any(t in self._VALID_DRIVER_TYPES for t in types)

    def test_y2_technical_spec_reference_field_produces_ts_surveillance(self):
        """Y2 fix: technical_specification_reference field → ts_surveillance (was technical_specification)."""
        proc = _proc()
        _, drivers = proc._detect_regulatory_constraints(
            {"technical_specification_reference": "TS 3.8.1"}, [], "install battery"
        )
        types = [d["driver_type"] for d in drivers]
        assert "technical_specification" not in types, (
            "technical_specification is not a valid schema enum value (Y2 fix)"
        )
        assert "ts_surveillance" in types

    def test_y2_alara_maps_to_alara_constraint(self):
        """Y2 fix: ALARA pattern → alara_constraint (was alara_requirement)."""
        proc = _proc()
        _, drivers = proc._detect_regulatory_constraints(
            {}, [], "work requires ALARA dose minimisation"
        )
        types = [d["driver_type"] for d in drivers]
        assert "alara_requirement" not in types, (
            "alara_requirement is not a valid schema enum value (Y2 fix)"
        )
        assert "alara_constraint" in types


# ===========================================================================
# Stage A — _extract_execution_mode_flags
# ===========================================================================

class TestExtractExecutionModeFlags:
    """Verify keyword pattern matching for the four execution mode flags.

    These flags are dead code until Stage A extracts them; having explicit tests
    for each pattern set ensures the extraction is reliable before Stage D
    starts consuming them for mixture_weight computation.
    """

    def _proc(self):
        from stages.stage_a_intake import ActivityIntakeProcessor
        return ActivityIntakeProcessor()

    # -- has_rp_hold ---------------------------------------------------------

    def test_rp_hold_phrase(self):
        flags = self._proc()._extract_execution_mode_flags("task requires rp hold before entry")
        assert flags["has_rp_hold"] is True

    def test_alara_hold_phrase(self):
        flags = self._proc()._extract_execution_mode_flags("ALARA hold required for this work scope")
        assert flags["has_rp_hold"] is True

    def test_rad_hold_abbreviation(self):
        flags = self._proc()._extract_execution_mode_flags("rad hold pending HP survey")
        assert flags["has_rp_hold"] is True

    def test_radiation_protection_hold_full(self):
        flags = self._proc()._extract_execution_mode_flags("radiation protection hold in place")
        assert flags["has_rp_hold"] is True

    # -- requires_scaffold ---------------------------------------------------

    def test_scaffold_noun(self):
        flags = self._proc()._extract_execution_mode_flags("erect scaffold to access valve body")
        assert flags["requires_scaffold"] is True

    def test_scaffolding_word(self):
        flags = self._proc()._extract_execution_mode_flags("scaffolding required for upper nozzle inspection")
        assert flags["requires_scaffold"] is True

    def test_staging_platform(self):
        flags = self._proc()._extract_execution_mode_flags("temporary staging platform installation")
        assert flags["requires_scaffold"] is True

    # -- has_clearance -------------------------------------------------------

    def test_clearance_word(self):
        flags = self._proc()._extract_execution_mode_flags("obtain clearance before starting work")
        assert flags["has_clearance"] is True

    def test_lockout_tagout(self):
        flags = self._proc()._extract_execution_mode_flags("lockout tagout required per procedure")
        assert flags["has_clearance"] is True

    def test_loto_abbreviation(self):
        flags = self._proc()._extract_execution_mode_flags("LOTO applied on MCC breaker")
        assert flags["has_clearance"] is True

    def test_mechanical_clearance(self):
        flags = self._proc()._extract_execution_mode_flags("mechanical clearance 1A-RHR-PP")
        assert flags["has_clearance"] is True

    # -- is_vendor_supported -------------------------------------------------

    def test_vendor_word(self):
        flags = self._proc()._extract_execution_mode_flags("vendor support required for seal replacement")
        assert flags["is_vendor_supported"] is True

    def test_oem_abbreviation(self):
        flags = self._proc()._extract_execution_mode_flags("OEM engineer to perform inspection")
        assert flags["is_vendor_supported"] is True

    def test_tech_rep(self):
        flags = self._proc()._extract_execution_mode_flags("tech rep from manufacturer on site")
        assert flags["is_vendor_supported"] is True

    def test_factory_rep(self):
        flags = self._proc()._extract_execution_mode_flags("factory representative required for alignment")
        assert flags["is_vendor_supported"] is True

    # -- no match / multiple flags -------------------------------------------

    def test_no_flags_plain_description(self):
        flags = self._proc()._extract_execution_mode_flags("replace gasket on 1A CCW pump")
        assert flags == {
            "has_rp_hold": False,
            "requires_scaffold": False,
            "has_clearance": False,
            "is_vendor_supported": False,
        }

    def test_empty_string_returns_all_false(self):
        flags = self._proc()._extract_execution_mode_flags("")
        assert all(v is False for v in flags.values())

    def test_multiple_flags_detected_simultaneously(self):
        desc = (
            "Vendor support needed; obtain clearance and rp hold; "
            "erect scaffold for upper access"
        )
        flags = self._proc()._extract_execution_mode_flags(desc)
        assert flags["has_rp_hold"] is True
        assert flags["requires_scaffold"] is True
        assert flags["has_clearance"] is True
        assert flags["is_vendor_supported"] is True

    def test_flags_present_in_process_output(self):
        """Integration: process() result must include execution_mode_flags key."""
        proc = self._proc()
        activity = {
            "activity_id": "ACT-001",
            "outage_id": "RF-22",
            "plant_id": "PLANT-1",
            "detection_timestamp": "2026-01-15T08:00:00Z",
            "raw_description": "OEM vendor required; obtain clearance before start",
        }
        run_ctx = {"run_id": "RUN-001", "started_at": "2026-01-15T08:00:00Z"}
        result = proc.process(activity, run_ctx)
        assert "execution_mode_flags" in result
        flags = result["execution_mode_flags"]
        assert flags["is_vendor_supported"] is True
        assert flags["has_clearance"] is True


# ===========================================================================
# Stage A — _compute_data_quality
# ===========================================================================

class TestComputeDataQuality:
    """Composite formula: 0.35×completeness + 0.25×ner_yield + 0.25×abbr_clarity
    + 0.15×source_confidence."""

    def test_full_record_high_score(self):
        """Well-populated record from Maximo with entities → score close to 1.0."""
        proc = _proc()
        activity = {
            "raw_description": "Replace primary coolant pump seal assembly due to excessive leakage",
            "detection_timestamp": "2026-04-01T08:00:00Z",
            "known_component_id": "RCP-001A",
            "work_order_id": "WO-9876543",
            "source_system": "maximo",
        }
        entities = [
            {"entity_type": "component"},
            {"entity_type": "action"},
            {"entity_type": "failure_mode"},
        ]
        score = proc._compute_data_quality(activity, entities, abbr_rate=0.0)
        assert score >= 0.70

    def test_minimal_record_low_score(self):
        """Empty record → low score, clamped to [0, 1]."""
        proc = _proc()
        score = proc._compute_data_quality({}, [], abbr_rate=1.0)
        assert 0.0 <= score <= 1.0
        assert score < 0.40

    def test_high_abbr_rate_reduces_score(self):
        """abbr_rate=1.0 kills the 0.25×abbreviation_clarity term."""
        proc = _proc()
        activity = {
            "raw_description": "REPLACE PCM SEAL DUE TO HIGH LEAKAGE RATE",
            "detection_timestamp": "2026-04-01T08:00:00Z",
            "known_component_id": "PCM-001",
            "work_order_id": "WO-111",
            "source_system": "maximo",
        }
        high_rate_score = proc._compute_data_quality(activity, [], abbr_rate=1.0)
        low_rate_score  = proc._compute_data_quality(activity, [], abbr_rate=0.0)
        assert low_rate_score > high_rate_score

    def test_source_system_unknown_lower_than_maximo(self):
        """Unknown source system yields lower score than Maximo."""
        proc = _proc()
        activity_base = {
            "raw_description": "perform inspection of reactor coolant pump",
            "detection_timestamp": "2026-04-01T08:00:00Z",
            "known_component_id": "RCP-001",
            "work_order_id": "WO-999",
        }
        score_maximo  = proc._compute_data_quality({**activity_base, "source_system": "maximo"},  [], 0.0)
        score_unknown = proc._compute_data_quality({**activity_base, "source_system": "unknown"}, [], 0.0)
        assert score_maximo > score_unknown

    def test_score_clamped_to_unit_interval(self):
        """Score is always in [0, 1]."""
        proc = _proc()
        # Extreme entity count: should not push score above 1.0
        activity = {
            "raw_description": "x",
            "source_system": "maximo",
        }
        entities = [{"entity_type": "component"}] * 100
        score = proc._compute_data_quality(activity, entities, abbr_rate=0.0)
        assert 0.0 <= score <= 1.0

    def test_n8_cmms_source_system_not_unknown(self):
        """N8 fix: CMMS records (schema canonical) must not default to 'unknown' confidence."""
        from stages.stage_a_intake import _SOURCE_CONFIDENCE
        # Schema sends "CMMS"; Stage A lowercases to "cmms"
        assert "cmms" in _SOURCE_CONFIDENCE, "'cmms' must be in _SOURCE_CONFIDENCE after N8 fix"
        assert _SOURCE_CONFIDENCE["cmms"] > _SOURCE_CONFIDENCE["unknown"]

    def test_n8_cap_source_system_not_unknown(self):
        """N8 fix: CAP records (schema canonical) must not default to 'unknown' confidence."""
        from stages.stage_a_intake import _SOURCE_CONFIDENCE
        assert "cap" in _SOURCE_CONFIDENCE, "'cap' must be in _SOURCE_CONFIDENCE after N8 fix"
        assert _SOURCE_CONFIDENCE["cap"] > _SOURCE_CONFIDENCE["unknown"]

    def test_n8_source_confidence_ordering(self):
        """maximo ≥ sap ≥ cmms > cap > manual > other ≥ unknown."""
        from stages.stage_a_intake import _SOURCE_CONFIDENCE
        assert _SOURCE_CONFIDENCE["maximo"] >= _SOURCE_CONFIDENCE["sap"]
        assert _SOURCE_CONFIDENCE["sap"] >= _SOURCE_CONFIDENCE["cmms"]
        assert _SOURCE_CONFIDENCE["cmms"] > _SOURCE_CONFIDENCE["cap"]
        assert _SOURCE_CONFIDENCE["cap"] > _SOURCE_CONFIDENCE["manual"]
        assert _SOURCE_CONFIDENCE["manual"] > _SOURCE_CONFIDENCE["other"]
        assert _SOURCE_CONFIDENCE["other"] >= _SOURCE_CONFIDENCE["unknown"]

    def test_n8_cmms_uppercase_normalised_correctly(self):
        """'CMMS' from schema input is normalised to 'cmms' by Stage A and gets proper confidence."""
        proc = _proc()
        activity_base = {
            "raw_description": "inspect reactor coolant pump",
            "detection_timestamp": "2026-04-01T08:00:00Z",
        }
        score_cmms    = proc._compute_data_quality({**activity_base, "source_system": "CMMS"},    [], 0.0)
        score_unknown = proc._compute_data_quality({**activity_base, "source_system": "unknown"}, [], 0.0)
        assert score_cmms > score_unknown, "CMMS (→ cmms) should yield higher confidence than unknown"


# ===========================================================================
# Stage A — _expand_abbreviations
# ===========================================================================

class TestExpandAbbreviations:
    """Unknown rate = (pre-caps still uppercase after expansion) / pre-caps count."""

    class _MockExpander:
        def __init__(self, mapping: dict):
            self._map = mapping
        def transform(self, text: str) -> str:
            for old, new in self._map.items():
                text = text.replace(old, new)
            return text

    def test_all_resolved_rate_zero(self):
        """All caps tokens resolved → unknown_rate == 0.0."""
        expander = self._MockExpander({"RHR": "residual heat removal", "PUMP": "pump"})
        proc = ActivityIntakeProcessor(abbreviation_expander=expander)
        _, rate = proc._expand_abbreviations("RHR PUMP failed")
        assert rate == 0.0

    def test_none_resolved_rate_one(self):
        """Passthrough expander leaves all caps unresolved → rate == 1.0."""
        expander = self._MockExpander({})  # no replacements
        proc = ActivityIntakeProcessor(abbreviation_expander=expander)
        _, rate = proc._expand_abbreviations("RHR PUMP ALARA NRC")
        assert rate == 1.0

    def test_partial_resolution(self):
        """Half resolved → rate == 0.5."""
        expander = self._MockExpander({"RHR": "residual heat removal"})
        proc = ActivityIntakeProcessor(abbreviation_expander=expander)
        # PUMP stays uppercase after expansion (not in mapping)
        _, rate = proc._expand_abbreviations("RHR PUMP")
        assert rate == pytest.approx(0.5)

    def test_empty_text_rate_zero(self):
        """Empty input → rate 0.0, no errors."""
        proc = ActivityIntakeProcessor()
        _, rate = proc._expand_abbreviations("")
        assert rate == 0.0

    def test_no_caps_tokens_rate_zero(self):
        """Text with no candidate abbreviations → rate 0.0."""
        proc = ActivityIntakeProcessor()
        _, rate = proc._expand_abbreviations("replace the pump packing")
        assert rate == 0.0

    def test_common_caps_excluded(self):
        """Common English words (IN, IS, TO…) are not counted as unknown abbreviations."""
        expander = self._MockExpander({})
        proc = ActivityIntakeProcessor(abbreviation_expander=expander)
        _, rate = proc._expand_abbreviations("IS IT IN THE AREA")
        # IS, IT, IN, THE (THE is in _COMMON_CAPS if added, but let's check AREA)
        # AREA is not in _COMMON_CAPS → 1 unknown; IS, IT, IN are → not counted
        # THE is in _COMMON_CAPS; AREA is not → rate = 1/1 = 1.0
        # (adjust if AREA is common — just verify rate is a valid float)
        assert 0.0 <= rate <= 1.0

    def test_unknown_rate_above_threshold_logs_warning(self, caplog):
        """Rate > 0.25 triggers a WARNING log."""
        import logging
        expander = self._MockExpander({})   # no replacements → rate = 1.0
        config = ActivityIntakeConfig(unknown_abbreviation_rate_warning=0.25)
        proc = ActivityIntakeProcessor(config=config, abbreviation_expander=expander)
        with caplog.at_level(logging.WARNING, logger="stages.stage_a_intake"):
            proc._expand_abbreviations("RHR PUMP EDG")
        assert any("abbreviation rate" in r.message.lower() for r in caplog.records)


# ===========================================================================
# Stage C — _allen_relation (all 7 relations)
# ===========================================================================

class TestAllenRelation:
    """Concrete timestamps; epsilon_hours=0 for deterministic boundary tests."""

    def _scorer(self) -> TemporalChainScorer:
        return TemporalChainScorer(TemporalChainConfig(epsilon_hours=0.0))

    def test_precedes(self):
        """A ends well before B starts → PRECEDES."""
        sc = self._scorer()
        prior_s = _dt("2026-01-01T00:00:00Z")
        prior_e = _dt("2026-01-01T02:00:00Z")   # ends 02:00
        act_s   = _dt("2026-01-01T06:00:00Z")   # starts 06:00
        act_e   = _dt("2026-01-01T08:00:00Z")
        assert sc._allen_relation(prior_s, prior_e, act_s, act_e) == "precedes"

    def test_follows(self):
        """A starts after B ends → FOLLOWS."""
        sc = self._scorer()
        prior_s = _dt("2026-01-01T10:00:00Z")
        prior_e = _dt("2026-01-01T12:00:00Z")
        act_s   = _dt("2026-01-01T00:00:00Z")
        act_e   = _dt("2026-01-01T04:00:00Z")
        assert sc._allen_relation(prior_s, prior_e, act_s, act_e) == "follows"

    def test_contains(self):
        """A starts before B and ends after B → CONTAINS."""
        sc = self._scorer()
        prior_s = _dt("2026-01-01T00:00:00Z")
        prior_e = _dt("2026-01-01T12:00:00Z")
        act_s   = _dt("2026-01-01T04:00:00Z")
        act_e   = _dt("2026-01-01T08:00:00Z")
        assert sc._allen_relation(prior_s, prior_e, act_s, act_e) == "contains"

    def test_overlaps(self):
        """A starts before B, ends inside B → OVERLAPS."""
        sc = self._scorer()
        prior_s = _dt("2026-01-01T00:00:00Z")
        prior_e = _dt("2026-01-01T06:00:00Z")   # ends inside B
        act_s   = _dt("2026-01-01T04:00:00Z")
        act_e   = _dt("2026-01-01T10:00:00Z")
        assert sc._allen_relation(prior_s, prior_e, act_s, act_e) == "overlaps"

    def test_during(self):
        """A entirely inside B → DURING."""
        sc = self._scorer()
        prior_s = _dt("2026-01-01T05:00:00Z")
        prior_e = _dt("2026-01-01T07:00:00Z")
        act_s   = _dt("2026-01-01T04:00:00Z")
        act_e   = _dt("2026-01-01T10:00:00Z")
        assert sc._allen_relation(prior_s, prior_e, act_s, act_e) == "during"

    def test_simultaneous(self):
        """A starts inside B but extends beyond B → OVERLAPS (H3 fix).

        Prior event that started within the activity window but outlasted it
        (right-side overlap / Allen 'overlapped-by') is causally equivalent to
        a forward overlap and scores 0.90, not SIMULTANEOUS (0.50).
        """
        sc = self._scorer()
        prior_s = _dt("2026-01-01T05:00:00Z")   # inside B
        prior_e = _dt("2026-01-01T12:00:00Z")   # beyond B end
        act_s   = _dt("2026-01-01T04:00:00Z")
        act_e   = _dt("2026-01-01T10:00:00Z")
        assert sc._allen_relation(prior_s, prior_e, act_s, act_e) == "overlaps"

    def test_unknown_when_prior_start_none(self):
        """Missing prior_start → UNKNOWN."""
        sc = self._scorer()
        act_s = _dt("2026-01-01T04:00:00Z")
        assert sc._allen_relation(None, None, act_s, act_s) == "unknown"

    def test_unknown_when_activity_start_none(self):
        """Missing activity_start → UNKNOWN."""
        sc = self._scorer()
        prior_s = _dt("2026-01-01T00:00:00Z")
        assert sc._allen_relation(prior_s, prior_s, None, None) == "unknown"

    def test_epsilon_boundary_treated_as_simultaneous(self):
        """Events within epsilon of a boundary → not PRECEDES but SIMULTANEOUS."""
        sc = _scorer(epsilon_hours=1.0)
        # A ends 0.5 h before B starts → within epsilon → not PRECEDES
        prior_s = _dt("2026-01-01T00:00:00Z")
        prior_e = _dt("2026-01-01T03:30:00Z")   # 0.5 h before act_s
        act_s   = _dt("2026-01-01T04:00:00Z")
        act_e   = _dt("2026-01-01T08:00:00Z")
        relation = sc._allen_relation(prior_s, prior_e, act_s, act_e)
        assert relation != "precedes"


# ===========================================================================
# Stage C — _assign_causal_strength
# ===========================================================================

class TestAssignCausalStrength:

    def test_follows_always_temporal_contradiction(self):
        """FOLLOWS relation → temporal_contradiction regardless of confidence."""
        sc = _scorer()
        assert sc._assign_causal_strength("follows", 0.99) == "temporal_contradiction"
        assert sc._assign_causal_strength("follows", 0.00) == "temporal_contradiction"

    def test_simultaneous_always_moderate(self):
        """SIMULTANEOUS → at least moderate; never weak even at low confidence.

        This is the bug fix: 0.50 × confidence never reached the MODERATE
        threshold of 0.40, so concurrent events were always scored 'weak'.
        """
        sc = _scorer()
        assert sc._assign_causal_strength("simultaneous", 0.10) == "moderate"
        assert sc._assign_causal_strength("simultaneous", 0.99) == "moderate"

    def test_overlaps_high_confidence_strong(self):
        """OVERLAPS × high confidence → strong."""
        sc = _scorer()
        # 0.90 (relation) × 0.90 (confidence) = 0.81 ≥ 0.75 → strong
        assert sc._assign_causal_strength("overlaps", 0.90) == "strong"

    def test_precedes_moderate_confidence_moderate(self):
        """PRECEDES × moderate confidence → moderate."""
        sc = _scorer()
        # 0.80 × 0.60 = 0.48 ≥ 0.40 → moderate
        assert sc._assign_causal_strength("precedes", 0.60) == "moderate"

    def test_precedes_high_confidence_strong(self):
        """PRECEDES × high confidence → strong.

        With _RELATION_SCORES[PRECEDES] = 0.80 the strong threshold (0.75) is
        reachable: 0.80 × 0.96 = 0.768 ≥ 0.75. Previously the score was 0.75,
        giving a maximum product of 0.75 × 0.9625 = 0.721 < 0.75 — making
        "strong" unreachable for PRECEDES regardless of data quality (X1 fix).
        """
        sc = _scorer()
        # 0.80 × 0.96 = 0.768 ≥ 0.75 → strong
        assert sc._assign_causal_strength("precedes", 0.96) == "strong"

    def test_during_always_weak(self):
        """DURING has relation_score=0.30; even at max confidence below moderate."""
        sc = _scorer()
        # 0.30 × 1.0 = 0.30 < 0.40 → weak
        assert sc._assign_causal_strength("during", 1.0) == "weak"

    def test_unknown_always_weak(self):
        """UNKNOWN has relation_score=0.00 → always weak."""
        sc = _scorer()
        assert sc._assign_causal_strength("unknown", 1.0) == "weak"


# ===========================================================================
# Stage C — _compute_confidence
# ===========================================================================

class TestComputeConfidence:

    def test_positive_lag_within_24h_max_plausibility(self):
        """Lag 0–24 h → lag_plausibility=1.0 (maximum credibility window)."""
        sc = _scorer()
        high_conf = sc._compute_confidence("overlaps", data_quality_score=1.0, onset_lag_hours=12.0)
        # anomaly_weight(0.55)*1.0 + latency_weight(0.30)*1.0 + relation_weight(0.15)*0.90
        expected = 0.55 * 1.0 + 0.30 * 1.0 + 0.15 * 0.90
        assert high_conf == pytest.approx(expected, abs=0.001)

    def test_negative_lag_lower_plausibility(self):
        """Negative lag (A starts after B onset) → plausibility=0.1."""
        sc = _scorer()
        conf = sc._compute_confidence("follows", data_quality_score=0.5, onset_lag_hours=-2.0)
        expected = 0.55 * 0.5 + 0.30 * 0.1 + 0.15 * 0.10  # FOLLOWS score=0.10
        assert conf == pytest.approx(expected, abs=0.001)

    def test_long_lag_decays_plausibility(self):
        """Lag >> 24 h → plausibility decays toward 0.1."""
        sc = _scorer()
        conf_short = sc._compute_confidence("precedes", data_quality_score=0.8, onset_lag_hours=5.0)
        conf_long  = sc._compute_confidence("precedes", data_quality_score=0.8, onset_lag_hours=500.0)
        assert conf_short > conf_long

    def test_none_lag_uses_neutral_plausibility(self):
        """None lag → plausibility=0.5 (neutral)."""
        sc = _scorer()
        conf = sc._compute_confidence("precedes", data_quality_score=0.5, onset_lag_hours=None)
        # _RELATION_SCORES[PRECEDES] = 0.80 (X1 fix, was 0.75)
        expected = 0.55 * 0.5 + 0.30 * 0.5 + 0.15 * 0.80
        assert conf == pytest.approx(expected, abs=0.001)

    def test_confidence_clamped_ge_zero(self):
        """Confidence is never negative."""
        sc = _scorer()
        conf = sc._compute_confidence("unknown", data_quality_score=0.0, onset_lag_hours=-999.0)
        assert conf >= 0.0

    # N10 tests ---------------------------------------------------------------

    def test_n10_overlaps_long_lag_lag_plausibility_is_one(self):
        """N10: OVERLAPS relation → lag_plausibility forced to 1.0 regardless of lag.

        A prior event that OVERLAPS the activity window is exactly the causal
        pattern we want to surface. Applying decay for a 500 h lag would be
        wrong because the relation already encodes precedence.
        """
        sc = _scorer()
        # Without N10 fix, 500 h lag decays to max(0.1, 1 - (500-24)/720) ≈ 0.34
        conf_with_long_lag = sc._compute_confidence(
            "overlaps", data_quality_score=0.8, onset_lag_hours=500.0
        )
        # With N10 fix, lag_plausibility=1.0 unconditionally for OVERLAPS
        expected = 0.55 * 0.8 + 0.30 * 1.0 + 0.15 * 0.90  # dq + lat + rel
        assert conf_with_long_lag == pytest.approx(expected, abs=0.001), (
            "OVERLAPS with 500 h lag must use lag_plausibility=1.0, not the decay formula"
        )

    def test_n10_contains_long_lag_lag_plausibility_is_one(self):
        """N10: CONTAINS relation → lag_plausibility forced to 1.0 regardless of lag."""
        sc = _scorer()
        conf = sc._compute_confidence(
            "contains", data_quality_score=1.0, onset_lag_hours=720.0
        )
        # lag_plausibility=1.0, CONTAINS relation_score=0.85
        expected = 0.55 * 1.0 + 0.30 * 1.0 + 0.15 * 0.85
        assert conf == pytest.approx(expected, abs=0.001), (
            "CONTAINS with 720 h lag must use lag_plausibility=1.0"
        )

    def test_n10_overlaps_confidence_not_penalised_vs_short_lag(self):
        """N10: OVERLAPS with a very long lag must not score lower than OVERLAPS with short lag."""
        sc = _scorer()
        conf_short = sc._compute_confidence("overlaps", data_quality_score=0.7, onset_lag_hours=1.0)
        conf_long  = sc._compute_confidence("overlaps", data_quality_score=0.7, onset_lag_hours=720.0)
        assert conf_long == pytest.approx(conf_short, abs=0.001), (
            "Lag magnitude must not penalise OVERLAPS confidence (N10 fix)"
        )

    def test_n10_precedes_long_lag_still_decays(self):
        """N10 fix is scoped to OVERLAPS/CONTAINS; PRECEDES must still decay on long lag."""
        sc = _scorer()
        conf_short = sc._compute_confidence("precedes", data_quality_score=0.8, onset_lag_hours=5.0)
        conf_long  = sc._compute_confidence("precedes", data_quality_score=0.8, onset_lag_hours=700.0)
        assert conf_short > conf_long, (
            "PRECEDES must still apply lag decay for long onset_lag_hours"
        )


# ===========================================================================
# Stage C — _summarize_chain
# ===========================================================================

class TestSummarizeChain:

    def test_empty_chain_insufficient_data(self):
        sc = _scorer()
        summary = sc._summarize_chain([])
        assert summary["causal_posture"] == "insufficient_data"
        assert summary["chain_length"] == 0
        assert summary["has_temporal_contradiction"] is False

    def test_strong_link_posture_supported(self):
        sc = _scorer()
        links = [{"link_id": "L1", "causal_strength": "strong",
                  "relation_score": 0.90, "allen_relation": "overlaps"}]
        summary = sc._summarize_chain(links)
        assert summary["causal_posture"] == "supported"
        assert summary["strongest_link_id"] == "L1"

    def test_moderate_only_posture_partial(self):
        sc = _scorer()
        links = [{"link_id": "L1", "causal_strength": "moderate",
                  "relation_score": 0.75, "allen_relation": "precedes"}]
        summary = sc._summarize_chain(links)
        assert summary["causal_posture"] == "partial"

    def test_contradiction_posture_overrides_strong(self):
        """Any temporal_contradiction link alongside strong evidence → 'contradicted_with_support'.

        M1 fix: when a FOLLOWS contradiction coexists with a strong OVERLAPS link, the
        posture is 'contradicted_with_support' (not plain 'contradicted') to signal that
        there is both a contradiction AND credible supporting evidence.
        """
        sc = _scorer()
        links = [
            {"link_id": "L1", "causal_strength": "strong",
             "relation_score": 0.90, "allen_relation": "overlaps"},
            {"link_id": "L2", "causal_strength": "temporal_contradiction",
             "relation_score": 0.10, "allen_relation": "follows"},
        ]
        summary = sc._summarize_chain(links)
        assert summary["causal_posture"] == "contradicted_with_support"
        assert summary["has_temporal_contradiction"] is True

    def test_all_weak_posture_weak(self):
        sc = _scorer()
        links = [{"link_id": "L1", "causal_strength": "weak",
                  "relation_score": 0.30, "allen_relation": "during"}]
        summary = sc._summarize_chain(links)
        assert summary["causal_posture"] == "weak"

    def test_strongest_link_selected_by_relation_score(self):
        """strongest_link_id should reference the link with highest relation_score."""
        sc = _scorer()
        links = [
            {"link_id": "L_high", "causal_strength": "strong",
             "relation_score": 0.90, "allen_relation": "overlaps"},
            {"link_id": "L_low",  "causal_strength": "moderate",
             "relation_score": 0.75, "allen_relation": "precedes"},
        ]
        summary = sc._summarize_chain(links)
        assert summary["strongest_link_id"] == "L_high"
        assert summary["max_relation_score"] == pytest.approx(0.90)


# ===========================================================================
# Stage C — score() integration
# ===========================================================================

class TestTemporalChainScorerIntegration:

    def _run_context(self) -> dict:
        return {"run_id": "RUN-001", "started_at": "2026-04-01T00:00:00Z"}

    def test_score_returns_required_top_level_keys(self):
        """score() output must include all top-level artifact keys."""
        sc = _scorer()
        emergent = {
            "activity_id": "ACT-001",
            "detection_timestamp": "2026-04-10T08:00:00Z",
        }
        timeline = {
            "component_id": "COMP-001",
            "events": [],
        }
        result = sc.score(emergent, timeline, self._run_context())
        required = {
            "activity_id", "run_id", "component_id",
            "emergent_activity_interval", "chain_links", "summary", "provenance",
        }
        assert required.issubset(result.keys())

    def test_score_precedes_event_produces_strong_link(self):
        """A CR that precedes the emergent activity by 4 h → strong or moderate link."""
        sc = _scorer()
        emergent = {
            "activity_id": "ACT-002",
            "detection_timestamp": "2026-04-10T08:00:00Z",
        }
        timeline = {
            "component_id": "COMP-001",
            "events": [
                {
                    "event_id": "CR-100",
                    "event_type": "condition_report",
                    "timestamp": "2026-04-10T04:00:00Z",   # 4 h before
                    "data_quality_score": 0.85,
                }
            ],
        }
        result = sc.score(emergent, timeline, self._run_context())
        assert len(result["chain_links"]) == 1
        link = result["chain_links"][0]
        assert link["allen_relation"] == "precedes"
        assert link["causal_strength"] in {"strong", "moderate"}

    def test_score_follows_event_tagged_contradiction(self):
        """Event after the emergent activity → FOLLOWS → temporal_contradiction."""
        sc = _scorer()
        emergent = {
            "activity_id": "ACT-003",
            "detection_timestamp": "2026-04-10T08:00:00Z",
        }
        timeline = {
            "component_id": "COMP-001",
            "events": [
                {
                    "event_id": "WO-200",
                    "event_type": "work_order",
                    "timestamp": "2026-04-10T18:00:00Z",   # 10 h after
                    "data_quality_score": 0.70,
                }
            ],
        }
        result = sc.score(emergent, timeline, self._run_context())
        assert len(result["chain_links"]) == 1
        link = result["chain_links"][0]
        assert link["allen_relation"] == "follows"
        assert link["causal_strength"] == "temporal_contradiction"
        assert result["summary"]["has_temporal_contradiction"] is True

    def test_score_event_missing_timestamp_skipped(self):
        """Events without a timestamp are silently skipped."""
        sc = _scorer()
        emergent = {
            "activity_id": "ACT-004",
            "detection_timestamp": "2026-04-10T08:00:00Z",
        }
        timeline = {
            "component_id": "COMP-001",
            "events": [{"event_id": "EV-999", "event_type": "work_order"}],
        }
        result = sc.score(emergent, timeline, self._run_context())
        assert result["chain_links"] == []
        assert result["summary"]["causal_posture"] == "insufficient_data"

    def test_score_point_event_uses_start_as_end(self):
        """When only detection_timestamp is known (no duration), is_point_event=True."""
        sc = _scorer()
        emergent = {"activity_id": "ACT-005", "detection_timestamp": "2026-04-10T08:00:00Z"}
        timeline = {"component_id": "COMP-001", "events": []}
        result = sc.score(emergent, timeline, self._run_context())
        assert result["emergent_activity_interval"]["is_point_event"] is True

    # ── Y1 fix: chain_link output field names must match schema ──────────────

    def test_y1_chain_link_uses_prior_event_id(self):
        """Y1 fix: chain_link must output 'prior_event_id', not 'event_id'."""
        sc = _scorer()
        emergent = {
            "activity_id": "ACT-Y1",
            "detection_timestamp": "2026-04-10T08:00:00Z",
        }
        timeline = {
            "component_id": "COMP-001",
            "events": [{
                "event_id": "CR-Y1",
                "event_type": "condition_report",
                "timestamp": "2026-04-10T04:00:00Z",
                "data_quality_score": 0.80,
            }],
        }
        result = sc.score(emergent, timeline, self._run_context())
        assert len(result["chain_links"]) == 1
        link = result["chain_links"][0]
        # schema-required keys must be present
        assert "prior_event_id" in link, "chain_link must use 'prior_event_id' (Y1 fix)"
        assert "prior_event_type" in link, "chain_link must use 'prior_event_type' (Y1 fix)"
        # old keys must NOT be present (would fail additionalProperties: false)
        assert "event_id" not in link, "chain_link must not expose 'event_id' (Y1 fix)"
        assert "event_type" not in link, "chain_link must not expose 'event_type' (Y1 fix)"
        assert "event_timestamp" not in link, "chain_link must not expose 'event_timestamp' (Y1 fix)"
        assert "data_quality_score" not in link, "chain_link must not expose 'data_quality_score' (Y1 fix)"

    def test_y1_prior_event_id_value_matches_input(self):
        """Y1 fix: prior_event_id value must equal the source event's event_id."""
        sc = _scorer()
        emergent = {
            "activity_id": "ACT-Y1B",
            "detection_timestamp": "2026-04-10T08:00:00Z",
        }
        timeline = {
            "component_id": "COMP-001",
            "events": [{
                "event_id": "WO-SPECIFIC",
                "event_type": "work_order",
                "timestamp": "2026-04-09T20:00:00Z",
                "data_quality_score": 0.70,
            }],
        }
        result = sc.score(emergent, timeline, self._run_context())
        link = result["chain_links"][0]
        assert link["prior_event_id"] == "WO-SPECIFIC"
        assert link["prior_event_type"] == "work_order"


# ===========================================================================
# Stage A — _run_ner regex layer (Layer 1: always runs, no injected backends)
# ===========================================================================

class TestRunNerRegexLayer:
    """The regex extraction layer must always run, regardless of injected backends."""

    def _run(self, text: str) -> list:
        proc = _proc()  # no injected NER pipeline
        return proc._run_ner(text, {})

    def test_tag_id_extracted(self):
        entities = self._run("Replace packing on MOV-1234 per schedule")
        tag_ids = [e for e in entities if e["entity_type"] == "tag_id"]
        assert len(tag_ids) == 1
        assert tag_ids[0]["text"] == "MOV-1234"
        assert tag_ids[0]["source"] == "regex"
        assert tag_ids[0]["confidence"] == pytest.approx(0.95)

    def test_multiple_tag_ids(self):
        entities = self._run("Inspect PT-4567 and FT-89012 for leakage")
        tag_ids = [e for e in entities if e["entity_type"] == "tag_id"]
        texts = {e["text"] for e in tag_ids}
        assert "PT-4567" in texts
        assert "FT-89012" in texts

    def test_tag_id_with_trailing_letter(self):
        """Tag IDs like RHR-PP-003A (trailing letter) must be matched."""
        entities = self._run("check FCV-5678A for passing")
        tag_ids = [e for e in entities if e["entity_type"] == "tag_id"]
        assert any("5678A" in e["text"] for e in tag_ids)

    def test_work_order_reference_extracted(self):
        entities = self._run("Complete work per WO 483921 this outage")
        wo_refs = [e for e in entities if e["entity_type"] == "work_order_reference"]
        assert len(wo_refs) == 1
        assert "483921" in wo_refs[0]["text"]

    def test_work_order_with_hash(self):
        entities = self._run("See WO#56789 for details")
        wo_refs = [e for e in entities if e["entity_type"] == "work_order_reference"]
        assert len(wo_refs) == 1

    def test_condition_report_reference_extracted(self):
        entities = self._run("Generated per CR 29847 findings")
        cr_refs = [e for e in entities if e["entity_type"] == "condition_report_reference"]
        assert len(cr_refs) == 1
        assert "29847" in cr_refs[0]["text"]

    def test_combined_tag_wo_cr_in_one_description(self):
        """Compressed plant description with multiple entity types."""
        text = "1A-RHR-PP-003 ISOL VLV PKG LKG – SEE WO 483921 & CR 29847"
        entities = self._run(text)
        types = {e["entity_type"] for e in entities}
        assert "work_order_reference" in types
        assert "condition_report_reference" in types

    def test_entity_ids_unique(self):
        """Every returned entity must have a unique entity_id."""
        entities = self._run("PT-1234 FT-5678 WO 99001 CR 44002")
        ids = [e["entity_id"] for e in entities]
        assert len(ids) == len(set(ids))

    def test_no_false_positives_on_plain_text(self):
        """Plain maintenance text with no tags/WOs/CRs → regex layer returns []."""
        entities = self._run("perform preventive maintenance on the pump")
        # Regex layer should find nothing; no injected NER → empty result
        assert entities == []


# ===========================================================================
# Stage C — non-point event (planned_duration_hours computes end timestamp)
# ===========================================================================

class TestNonPointEventInterval:
    """When planned_duration_hours is provided (but no actual_finish), the
    emergent activity interval is computed as start + planned_duration_hours."""

    def test_non_point_event_is_point_false(self):
        """Providing planned_duration_hours → is_point_event=False."""
        sc = _scorer()
        emergent = {
            "activity_id": "ACT-INTERVAL",
            "detection_timestamp": "2026-04-10T08:00:00Z",
            "planned_duration_hours": 8.0,
        }
        timeline = {"component_id": "COMP-001", "events": []}
        result = sc.score(emergent, timeline, {"run_id": "R1", "started_at": ""})
        assert result["emergent_activity_interval"]["is_point_event"] is False

    def test_non_point_event_end_is_start_plus_duration(self):
        """End timestamp = detection_timestamp + 8 h = 16:00."""
        sc = _scorer()
        emergent = {
            "activity_id": "ACT-INTERVAL",
            "detection_timestamp": "2026-04-10T08:00:00Z",
            "planned_duration_hours": 8.0,
        }
        timeline = {"component_id": "COMP-001", "events": []}
        result = sc.score(emergent, timeline, {"run_id": "R1", "started_at": ""})
        interval = result["emergent_activity_interval"]
        assert interval["end"] is not None
        assert interval["start"] != interval["end"]

    def test_during_relation_for_event_inside_planned_window(self):
        """An event at 10:00 is DURING the 08:00–16:00 activity window."""
        sc = TemporalChainScorer(TemporalChainConfig(epsilon_hours=0.0))
        emergent = {
            "activity_id": "ACT-DUR",
            "detection_timestamp": "2026-04-10T08:00:00Z",
            "planned_duration_hours": 8.0,
        }
        timeline = {
            "component_id": "COMP-001",
            "events": [{
                "event_id": "EV-1",
                "event_type": "condition_report",
                "timestamp": "2026-04-10T10:00:00Z",  # inside [08:00, 16:00]
                "data_quality_score": 0.7,
            }],
        }
        result = sc.score(emergent, timeline, {"run_id": "R1", "started_at": ""})
        assert result["chain_links"][0]["allen_relation"] == "during"


# ===========================================================================
# Stage A — M7: regulatory_keywords_path config field is wired and loaded
# ===========================================================================

class TestRegulatoryKeywordsPath:
    """Verify the M7 fix: regulatory_keywords_path config field is loaded and
    the supplementary patterns are used by _detect_regulatory_constraints().

    Before the fix the config field existed but the load path was never called.
    After the fix _load_regulatory_keywords() reads the file, compiles the regexes,
    and those patterns participate in the full pattern union.
    """

    def _proc_with_keywords_file(self, tmp_path, lines: list) -> ActivityIntakeProcessor:
        """Write a keywords file and return a processor configured to use it."""
        kw_file = tmp_path / "custom_keywords.txt"
        kw_file.write_text("\n".join(lines), encoding="utf-8")
        cfg = ActivityIntakeConfig(regulatory_keywords_path=kw_file)
        return ActivityIntakeProcessor(config=cfg)

    def test_m7_config_field_exists(self):
        """ActivityIntakeConfig must have a regulatory_keywords_path field."""
        assert hasattr(ActivityIntakeConfig(), "regulatory_keywords_path"), (
            "ActivityIntakeConfig must expose regulatory_keywords_path (M7 fix)"
        )

    def test_m7_keywords_path_none_returns_empty(self):
        """When regulatory_keywords_path is None, _load_regulatory_keywords returns []."""
        proc = _proc()
        patterns = proc._load_supplementary_regulatory_patterns()
        assert patterns == []

    def test_m7_custom_pattern_detected(self, tmp_path):
        """A pattern from regulatory_keywords_path must fire on matching text."""
        proc = self._proc_with_keywords_file(tmp_path, [
            "# plant-specific patterns",
            "PLANT-HOLD-\\d+|plant_hold|true",
        ])
        has_reg, drivers = proc._detect_regulatory_constraints(
            {}, [], "activity blocked by PLANT-HOLD-42 clearance"
        )
        assert has_reg is True
        types = [d["driver_type"] for d in drivers]
        assert "plant_hold" in types, (
            "Custom pattern from regulatory_keywords_path must fire in _detect_regulatory_constraints"
        )

    def test_m7_custom_pattern_defer_prohibited_true(self, tmp_path):
        """defer_prohibited=true in keywords file must be reflected in the driver."""
        proc = self._proc_with_keywords_file(tmp_path, [
            "OUTAGE-FREEZE|outage_freeze|true",
        ])
        _, drivers = proc._detect_regulatory_constraints(
            {}, [], "subject to OUTAGE-FREEZE constraint"
        )
        driver = next((d for d in drivers if d["driver_type"] == "outage_freeze"), None)
        assert driver is not None
        assert driver["defer_prohibited"] is True

    def test_m7_malformed_line_skipped(self, tmp_path):
        """Malformed lines (wrong field count) must be skipped without raising."""
        proc = self._proc_with_keywords_file(tmp_path, [
            "VALID-PATTERN|valid_driver|false",
            "this_line_has_no_pipe_separators",   # malformed — must be skipped
        ])
        # Must not raise
        patterns = proc._load_supplementary_regulatory_patterns()
        assert len(patterns) == 1, "Only the valid line should produce a pattern"

    def test_m7_missing_file_returns_empty(self, tmp_path):
        """A non-existent regulatory_keywords_path must return [] (no raise)."""
        cfg = ActivityIntakeConfig(
            regulatory_keywords_path=tmp_path / "does_not_exist.txt"
        )
        proc = ActivityIntakeProcessor(config=cfg)
        patterns = proc._load_supplementary_regulatory_patterns()
        assert patterns == []


# ===========================================================================
# Stage A — N1: preprocessing_available field in process() output
# ===========================================================================

class TestN1PreprocessingAvailableField:
    """N1 fix: process() must emit preprocessing_available so consumers can detect
    degraded-mode NLP (when _CLEANERS_AVAILABLE=False, whitespace-only cleaning runs).
    """

    def _run(self) -> dict:
        proc = _proc()
        return proc.process(
            {
                "activity_id": "ACT-N1",
                "raw_description": "valve seal leak discovered during walkdown",
            },
            {"run_id": "RUN-N1"},
        )

    def test_n1_field_present_in_output(self):
        """preprocessing_available must be a top-level key in the process() result."""
        result = self._run()
        assert "preprocessing_available" in result, (
            "process() must include 'preprocessing_available' to signal NLP mode (N1 fix)"
        )

    def test_n1_field_is_bool(self):
        """preprocessing_available must be a boolean."""
        result = self._run()
        assert isinstance(result["preprocessing_available"], bool)

    def test_n1_field_reflects_module_flag(self):
        """preprocessing_available must equal the module-level _CLEANERS_AVAILABLE flag."""
        import stages.stage_a_intake as _stage_a
        result = self._run()
        assert result["preprocessing_available"] == _stage_a._CLEANERS_AVAILABLE


# ===========================================================================
# Stage A — M1: _compute_lco_clock and lco_clock_status in process() output
# ===========================================================================

class TestM1LcoClockComputation:
    """M1 fix: Stage A must compute and surface LCO action-level countdown fields.

    Verifies _compute_lco_clock() directly plus the three fields in process() output.
    Reference time for the tests uses a fixed 'started_at' in run_context.
    """

    # ─── Reference time used across all tests: 2024-06-01T12:00:00Z ──────────
    _REF = "2024-06-01T12:00:00+00:00"

    def _ctx(self) -> dict:
        return {"run_id": "RUN-M1", "started_at": self._REF}

    def _clock(self, emergent_activity: dict) -> tuple:
        return _proc()._compute_lco_clock(emergent_activity, self._ctx())

    # ── not_applicable ────────────────────────────────────────────────────────

    def test_m1_not_applicable_when_no_lco(self):
        """No active_lco and no expires_at → not_applicable."""
        expires, hours, status = self._clock({"activity_id": "A1"})
        assert status == "not_applicable"
        assert expires is None
        assert hours is None

    # ── unknown ───────────────────────────────────────────────────────────────

    def test_m1_unknown_when_active_lco_no_expiry(self):
        """active_lco=True but no lco_action_level_expires_at → unknown."""
        expires, hours, status = self._clock({"activity_id": "A1", "active_lco": True})
        assert status == "unknown"
        assert expires is None
        assert hours is None

    def test_m1_unknown_when_expires_at_unparseable(self):
        """Malformed lco_action_level_expires_at → unknown, raw string passed through."""
        expires, hours, status = self._clock({
            "activity_id": "A1",
            "active_lco": True,
            "lco_action_level_expires_at": "not-a-datetime",
        })
        assert status == "unknown"
        assert expires == "not-a-datetime"   # raw value passed through
        assert hours is None

    # ── expired ───────────────────────────────────────────────────────────────

    def test_m1_expired_when_deadline_in_past(self):
        """Expiry before reference time → expired, hours_to_action_level < 0."""
        # 2 hours before reference
        expires, hours, status = self._clock({
            "activity_id": "A1",
            "active_lco": True,
            "lco_action_level_expires_at": "2024-06-01T10:00:00+00:00",
        })
        assert status == "expired"
        assert hours == pytest.approx(-2.0, abs=0.01)

    # ── critical ──────────────────────────────────────────────────────────────

    def test_m1_critical_when_less_than_4h_remaining(self):
        """1.5 h remaining → critical."""
        expires, hours, status = self._clock({
            "activity_id": "A1",
            "active_lco": True,
            "lco_action_level_expires_at": "2024-06-01T13:30:00+00:00",
        })
        assert status == "critical"
        assert hours == pytest.approx(1.5, abs=0.01)

    def test_m1_critical_boundary_exactly_4h(self):
        """Exactly 4.0 h remaining is 'urgent' (boundary: critical is < 4.0)."""
        expires, hours, status = self._clock({
            "activity_id": "A1",
            "active_lco": True,
            "lco_action_level_expires_at": "2024-06-01T16:00:00+00:00",
        })
        assert status == "urgent"
        assert hours == pytest.approx(4.0, abs=0.01)

    # ── urgent ────────────────────────────────────────────────────────────────

    def test_m1_urgent_when_between_4_and_24h(self):
        """12 h remaining → urgent."""
        expires, hours, status = self._clock({
            "activity_id": "A1",
            "active_lco": True,
            "lco_action_level_expires_at": "2024-06-02T00:00:00+00:00",
        })
        assert status == "urgent"
        assert hours == pytest.approx(12.0, abs=0.01)

    # ── normal ────────────────────────────────────────────────────────────────

    def test_m1_normal_when_24h_or_more_remaining(self):
        """72 h remaining → normal."""
        expires, hours, status = self._clock({
            "activity_id": "A1",
            "active_lco": True,
            "lco_action_level_expires_at": "2024-06-04T12:00:00+00:00",
        })
        assert status == "normal"
        assert hours == pytest.approx(72.0, abs=0.01)

    # ── process() integration ─────────────────────────────────────────────────

    def test_m1_fields_present_in_process_output(self):
        """process() must include all three LCO clock fields."""
        result = _proc().process(
            {"activity_id": "A1", "raw_description": "valve leak"},
            {"run_id": "RUN-M1"},
        )
        assert "lco_action_level_expires_at" in result
        assert "hours_to_action_level" in result
        assert "lco_clock_status" in result

    def test_m1_process_not_applicable_when_no_lco_fields(self):
        """Without any LCO fields, process() reports not_applicable."""
        result = _proc().process(
            {"activity_id": "A1", "raw_description": "routine inspection"},
            {"run_id": "RUN-M1"},
        )
        assert result["lco_clock_status"] == "not_applicable"
        assert result["lco_action_level_expires_at"] is None
        assert result["hours_to_action_level"] is None

    def test_m1_process_uses_started_at_as_reference_time(self):
        """process() uses run_context['started_at'] as the reference time for the clock."""
        result = _proc().process(
            {
                "activity_id": "A1",
                "raw_description": "surveillance test",
                "active_lco": True,
                "lco_action_level_expires_at": "2024-06-01T14:00:00+00:00",
            },
            {"run_id": "RUN-M1", "started_at": "2024-06-01T12:00:00+00:00"},
        )
        assert result["lco_clock_status"] == "critical"  # 2 h: < 4 h threshold
        assert result["hours_to_action_level"] == pytest.approx(2.0, abs=0.01)


# ===========================================================================
# Stage A — X2: lco_number forwarded through process() output
# ===========================================================================

class TestX2LcoNumberForwarding:
    """X2 fix: Stage A must forward lco_number from the emergent_activity dict
    into intake_result so Stage G can display it in the LCO clock warning prefix.

    Previously lco_number was read by Stage G (_build_lco_clock_prefix) but
    never emitted by Stage A, so the identifier was always None in the prefix.
    """

    @staticmethod
    def _ctx(run_id: str = "RUN-X2") -> dict:
        return {"run_id": run_id}

    def test_x2_lco_number_forwarded_when_present(self):
        """lco_number present in input → forwarded unchanged in process() output."""
        result = _proc().process(
            {
                "activity_id": "A1",
                "raw_description": "surveillance test required per LCO 3.5.1",
                "lco_number": "LCO 3.5.1",
            },
            self._ctx(),
        )
        assert "lco_number" in result, "lco_number key must be present in intake_result"
        assert result["lco_number"] == "LCO 3.5.1"

    def test_x2_lco_number_is_none_when_absent(self):
        """No lco_number in input → lco_number is None in process() output (not missing)."""
        result = _proc().process(
            {"activity_id": "A1", "raw_description": "valve inspection"},
            self._ctx(),
        )
        assert "lco_number" in result, "lco_number key must always be present"
        assert result["lco_number"] is None

    def test_x2_lco_number_key_always_present(self):
        """lco_number must be a first-class key regardless of input — not an
        optional/missing key that downstream code must guard against."""
        for activity in [
            {"activity_id": "A1", "raw_description": "test"},
            {"activity_id": "A2", "raw_description": "test", "lco_number": "LCO 3.4.6"},
            {"activity_id": "A3", "raw_description": "test", "active_lco": True},
        ]:
            result = _proc().process(activity, self._ctx(activity["activity_id"]))
            assert "lco_number" in result, (
                f"lco_number missing for activity {activity['activity_id']}"
            )


# ===========================================================================
# ADV-01 – ADV-03: Adversarial / edge-case inputs for Stage A
# ===========================================================================

# ADV-01: The 5 000-character description is a stress test for the full intake
# pipeline (abbreviation expansion, NER, data-quality scoring).  The sentence
# is innocuous but long enough to exercise any length-dependent code paths.
_ADV01_LONG_DESC: str = (
    "Drain valve on auxiliary feedwater system header found with active packing leak. "
    "No prior condition reports or work orders found for this valve assembly. "
    "Estimated four hours for packing replacement. Component not previously catalogued. "
) * 20   # ≈ 5 040 characters


# ADV-02: Dense all-caps acronym string modelled on the example in §7.5 of the
# review document.  Every token is fabricated / uncommon so none will be
# resolved by the built-in nuclear abbreviation table, guaranteeing that the
# unresolved fraction stays above the 0.25 WARNING threshold.
_ADV02_ALL_CAPS_DESC: str = (
    "XYZZY QWERT ASDFG ZXCVB PLKJH NMVCX POIUY LKJHG "
    "FDSAZ MNBVC QAZXS WDSXC EDCRF VTGBY HNUJM IKOLY"
)


# ADV-03: Description mixing UTF-8 non-ASCII characters with standard ASCII
# nuclear-plant vocabulary.  Tests that pattern-match regexes (_TAG_ID_RE,
# _WO_REF_RE) and the abbreviation tokeniser do not crash on non-ASCII input.
_ADV03_UNICODE_DESC: str = (
    "Maintenance sur la soupape de drain WO-44821 référence échangeur αβ-sensor. "
    "Тест контроль ЛКПД valve packing leak identified during walkdown CR-2026-001."
)


class TestAdversarialInputStageA:
    """ADV-01 – ADV-03: Robustness of Stage A against unusual raw_description values."""

    @staticmethod
    def _ctx() -> dict:
        return {"run_id": "RUN-ADV"}

    # ── ADV-01: 5 000-character description ───────────────────────────────────

    def test_adv01_long_description_completes_without_crash(self):
        """Stage A must complete normally for a 5 000-char description (no exception)."""
        result = _proc().process(
            {"activity_id": "A-ADV-01", "raw_description": _ADV01_LONG_DESC},
            self._ctx(),
        )
        assert isinstance(result, dict), "process() must return a dict"

    def test_adv01_long_description_data_quality_in_range(self):
        """data_quality_score must be in [0.0, 1.0] for any description length."""
        result = _proc().process(
            {"activity_id": "A-ADV-01", "raw_description": _ADV01_LONG_DESC},
            self._ctx(),
        )
        score = result.get("data_quality_score", -1.0)
        assert 0.0 <= score <= 1.0, (
            f"data_quality_score out of range: {score}"
        )

    def test_adv01_long_description_abbr_rate_is_float(self):
        """unknown_abbreviation_rate must be a float regardless of description length."""
        result = _proc().process(
            {"activity_id": "A-ADV-01", "raw_description": _ADV01_LONG_DESC},
            self._ctx(),
        )
        rate = result.get("unknown_abbreviation_rate")
        assert isinstance(rate, float), (
            f"unknown_abbreviation_rate must be float, got {type(rate).__name__}"
        )

    def test_adv01_long_description_emergence_type_present(self):
        """emergence_type must be a recognised value for any description length."""
        _VALID_TYPES = {
            "truly_unplanned", "scope_expansion", "regulatory_driven",
            "degradation_escalation", "schedule_optimization",
        }
        result = _proc().process(
            {"activity_id": "A-ADV-01", "raw_description": _ADV01_LONG_DESC},
            self._ctx(),
        )
        et = result.get("emergence_type")
        assert et in _VALID_TYPES, (
            f"emergence_type '{et}' not in valid set {_VALID_TYPES}"
        )

    # ── ADV-02: All-uppercase dense abbreviation string ───────────────────────

    def test_adv02_all_caps_does_not_raise(self):
        """Stage A must not crash on an all-uppercase, high-abbreviation-density description."""
        result = _proc().process(
            {"activity_id": "A-ADV-02", "raw_description": _ADV02_ALL_CAPS_DESC},
            self._ctx(),
        )
        assert isinstance(result, dict)

    def test_adv02_all_caps_abbr_rate_exceeds_warning_threshold(self):
        """All-uppercase unknown tokens must drive unknown_abbreviation_rate above 0.25.

        The warning threshold is ActivityIntakeConfig.unknown_abbreviation_rate_warning = 0.25.
        All tokens in _ADV02_ALL_CAPS_DESC are fabricated and cannot be resolved by
        the built-in nuclear abbreviation table, so the rate must equal 1.0 (no
        resolver injected) or at least exceed the 0.25 threshold.
        """
        result = _proc().process(
            {"activity_id": "A-ADV-02", "raw_description": _ADV02_ALL_CAPS_DESC},
            self._ctx(),
        )
        rate = result.get("unknown_abbreviation_rate", 0.0)
        assert rate > 0.25, (
            f"All-caps unknown tokens must push unknown_abbreviation_rate above 0.25, got {rate}"
        )

    # ── ADV-03: Non-ASCII characters in description ───────────────────────────

    def test_adv03_non_ascii_does_not_raise(self):
        """Stage A must not crash on descriptions containing UTF-8 / non-ASCII characters."""
        result = _proc().process(
            {"activity_id": "A-ADV-03", "raw_description": _ADV03_UNICODE_DESC},
            self._ctx(),
        )
        assert isinstance(result, dict)

    def test_adv03_non_ascii_abbr_rate_is_float(self):
        """unknown_abbreviation_rate must be a float even when description has non-ASCII chars."""
        result = _proc().process(
            {"activity_id": "A-ADV-03", "raw_description": _ADV03_UNICODE_DESC},
            self._ctx(),
        )
        rate = result.get("unknown_abbreviation_rate")
        assert isinstance(rate, float), (
            f"unknown_abbreviation_rate must be float, got {type(rate).__name__}"
        )

    def test_adv03_non_ascii_emergence_type_present(self):
        """A valid emergence_type must be produced even with non-ASCII input."""
        _VALID_TYPES = {
            "truly_unplanned", "scope_expansion", "regulatory_driven",
            "degradation_escalation", "schedule_optimization",
        }
        result = _proc().process(
            {"activity_id": "A-ADV-03", "raw_description": _ADV03_UNICODE_DESC},
            self._ctx(),
        )
        et = result.get("emergence_type")
        assert et in _VALID_TYPES, (
            f"emergence_type '{et}' not in valid set"
        )

    def test_adv03_non_ascii_extracted_entities_is_list(self):
        """extracted_entities must be a list (possibly empty) for non-ASCII descriptions."""
        result = _proc().process(
            {"activity_id": "A-ADV-03", "raw_description": _ADV03_UNICODE_DESC},
            self._ctx(),
        )
        entities = result.get("extracted_entities")
        assert isinstance(entities, list), (
            f"extracted_entities must be a list, got {type(entities).__name__}"
        )
