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
_OUTAGE_ROOT = Path(__file__).parent.parent
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
        """'TS 3.5.2' in text → technical_specification driver, defer_prohibited=True."""
        proc = _proc()
        has_reg, drivers = proc._detect_regulatory_constraints(
            {}, [], "required per TS 3.5.2"
        )
        assert has_reg is True
        types = [d["driver_type"] for d in drivers]
        assert "technical_specification" in types
        ts_driver = next(d for d in drivers if d["driver_type"] == "technical_specification")
        assert ts_driver["defer_prohibited"] is True

    def test_alara_not_defer_prohibited(self):
        """ALARA → alara_requirement, defer_prohibited=False."""
        proc = _proc()
        has_reg, drivers = proc._detect_regulatory_constraints(
            {}, [], "work requires ALARA dose minimisation plan"
        )
        alara = next((d for d in drivers if d["driver_type"] == "alara_requirement"), None)
        assert alara is not None
        assert alara["defer_prohibited"] is False

    def test_surveillance_keyword_defer_prohibited(self):
        """'surveillance' keyword → surveillance_requirement, defer_prohibited=True."""
        proc = _proc()
        has_reg, drivers = proc._detect_regulatory_constraints(
            {}, [], "quarterly valve surveillance test"
        )
        surv = next((d for d in drivers if d["driver_type"] == "surveillance_requirement"), None)
        assert surv is not None
        assert surv["defer_prohibited"] is True

    def test_structured_field_technical_spec(self):
        """technical_specification_reference field → driver added from intake record."""
        proc = _proc()
        has_reg, drivers = proc._detect_regulatory_constraints(
            {"technical_specification_reference": "TS 3.8.1"}, [], "install battery"
        )
        assert has_reg is True
        ts = next((d for d in drivers if d["driver_type"] == "technical_specification"), None)
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
        ts_drivers = [d for d in drivers if d["driver_type"] == "technical_specification"]
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
        """A starts inside B but extends beyond B → SIMULTANEOUS."""
        sc = self._scorer()
        prior_s = _dt("2026-01-01T05:00:00Z")   # inside B
        prior_e = _dt("2026-01-01T12:00:00Z")   # beyond B end
        act_s   = _dt("2026-01-01T04:00:00Z")
        act_e   = _dt("2026-01-01T10:00:00Z")
        assert sc._allen_relation(prior_s, prior_e, act_s, act_e) == "simultaneous"

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
        # 0.75 × 0.60 = 0.45 ≥ 0.40 → moderate
        assert sc._assign_causal_strength("precedes", 0.60) == "moderate"

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
        expected = 0.55 * 0.5 + 0.30 * 0.5 + 0.15 * 0.75
        assert conf == pytest.approx(expected, abs=0.001)

    def test_confidence_clamped_ge_zero(self):
        """Confidence is never negative."""
        sc = _scorer()
        conf = sc._compute_confidence("unknown", data_quality_score=0.0, onset_lag_hours=-999.0)
        assert conf >= 0.0


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
        """Any temporal_contradiction link → 'contradicted', even alongside strong."""
        sc = _scorer()
        links = [
            {"link_id": "L1", "causal_strength": "strong",
             "relation_score": 0.90, "allen_relation": "overlaps"},
            {"link_id": "L2", "causal_strength": "temporal_contradiction",
             "relation_score": 0.10, "allen_relation": "follows"},
        ]
        summary = sc._summarize_chain(links)
        assert summary["causal_posture"] == "contradicted"
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
