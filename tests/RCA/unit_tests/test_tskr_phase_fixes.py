"""
test_tskr_phase_fixes.py — regression tests for Phase 1-3 TSKR fixes.

Coverage (one section per fix):
  Phase 1
    B1  _build_recurrence_profile: component-only fallback guarded by empty matched_fms
    B3  _stage_b_allen_relation_by_component: CAUSAL_PRIORITY wins over last-write
    G1  _normalize_past_events: related_failure_modes / occurred_at remapping

  Phase 2
    B2  _build_recurrence_profile: most_recent_days_ago from live timestamps
    G2  pm_overdue_boost applied when overdue items share component_id
    G3  attention_flags["accelerating_recurrence"] emitted + escalated to RCA card

  Phase 3
    B4  _recurrence_trend: OLS slope-based classification
    B5  _extract_signal_ids_for_fm: per-FM filtering via expected_symptom_types
    G4  contributing_event_ids populated in RecurrenceProfile and pattern output
    B6  unresolved_count aligned to matching set, not dated set
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.tskr_temporal_scorer import (
    TSKRTemporalScorerV1,
    TSKRTemporalScorerConfig,
    RecurrenceProfile,
    parse_dt,
)
from orchestrators.temporal_relations import OVERLAPS, CONTAINS, PRECEDES, DURING, FOLLOWS

# ── shared helpers ─────────────────────────────────────────────────────────────

BASE = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _h(n: float) -> datetime:
    return BASE + timedelta(hours=n)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _scorer() -> TSKRTemporalScorerV1:
    return TSKRTemporalScorerV1()


def _past_event(
    event_id: str,
    fm_ids: list,
    component_id: str,
    days_before: float,
    resolved: bool = True,
) -> dict:
    ts = BASE - timedelta(days=days_before)
    return {
        "event_id": event_id,
        "matched_failure_mode_ids": fm_ids,
        "component_id": component_id,
        "timestamp_start": _iso(ts),
        "timestamp_end": _iso(ts + timedelta(hours=2)),
        "resolved": resolved,
        "time_distance_days": int(days_before),
    }


def _build_profile(scorer, fm_id, component_id, past_events, event_start=None):
    return scorer._build_recurrence_profile(
        fm_id=fm_id,
        component_id=component_id,
        past_events=past_events,
        event_start=event_start,
    )


def _make_fm(fm_id="FM-01", component_id="C-01", symptom_types=None):
    fm = {"fm_id": fm_id, "component_id": component_id}
    if symptom_types is not None:
        fm["expected_symptom_types"] = symptom_types
    return fm


def _make_event(start_h=0.0, end_h=10.0):
    return {
        "event_id": "EVT-001",
        "asset_id": "ASSET-01",
        "timestamp_start": _h(start_h).isoformat(),
        "timestamp_end": _h(end_h).isoformat(),
    }


def _make_telemetry(*sensor_param_pairs):
    """Build telemetry_summary with one signal per (sensor_id, parameter) pair."""
    signals = []
    for sid, param in sensor_param_pairs:
        signals.append({
            "sensor_id": sid,
            "parameter": param,
            "anomalies": [
                {
                    "timestamp_start": _h(-3).isoformat(),
                    "timestamp_end": _h(-1).isoformat(),
                    "severity": "high",
                }
            ],
        })
    return {"asset_id": "ASSET-01", "signals": signals}


def _make_kg(fm, past_events=None, oob_anomalies=None):
    return {
        "failure_modes": [fm] if isinstance(fm, dict) else fm,
        "past_events": past_events or [],
        "out_of_boundary_anomalies": oob_anomalies or [],
    }


def _score_pattern(scorer, fm, past_events=None, telemetry=None, pm_compliance=None):
    """Run the full score() pipeline and return the first pattern."""
    tel = telemetry or {"asset_id": "ASSET-01", "signals": []}
    result = scorer.score(
        event=_make_event(),
        telemetry_summary=tel,
        kg_context=_make_kg(fm, past_events=past_events or []),
        operational_context=None,
        run_context={"run_id": "R-TEST"},
        pm_compliance=pm_compliance,
    )
    return result["patterns"][0]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — B1: OR-matching guard
# ═══════════════════════════════════════════════════════════════════════════════

def test_b1_fm_matched_explicitly_counts():
    """Past event with FM explicitly in matched_failure_mode_ids must be counted."""
    pe = _past_event("CR-001", ["FM-A"], "COMP-1", days_before=30)
    profile = _build_profile(_scorer(), "FM-A", "COMP-1", [pe])
    assert profile.count == 1


def test_b1_fm_not_in_matched_fms_not_counted():
    """Past event lists a different FM — must NOT count toward FM-B even if component matches."""
    pe = _past_event("CR-001", ["FM-A"], "COMP-1", days_before=30)
    profile = _build_profile(_scorer(), "FM-B", "COMP-1", [pe])
    assert profile.count == 0


def test_b1_two_fms_same_component_independent_counts():
    """Two FMs on same component; each past event must only count for its own FM."""
    pe_a = _past_event("CR-A", ["FM-A"], "COMP-X", days_before=60)
    pe_b = _past_event("CR-B", ["FM-B"], "COMP-X", days_before=30)
    profile_a = _build_profile(_scorer(), "FM-A", "COMP-X", [pe_a, pe_b])
    profile_b = _build_profile(_scorer(), "FM-B", "COMP-X", [pe_a, pe_b])
    assert profile_a.count == 1
    assert profile_b.count == 1


def test_b1_component_fallback_used_when_no_fm_ids():
    """When matched_failure_mode_ids is absent/empty, component match is the fallback."""
    pe = {"event_id": "CR-X", "component_id": "COMP-1", "timestamp_start": _iso(BASE - timedelta(days=10))}
    profile = _build_profile(_scorer(), "FM-A", "COMP-1", [pe])
    assert profile.count == 1


def test_b1_component_fallback_not_used_when_fm_ids_present():
    """When matched_failure_mode_ids is non-empty and FM not in it, no component fallback."""
    pe = _past_event("CR-X", ["FM-OTHER"], "COMP-1", days_before=10)
    profile = _build_profile(_scorer(), "FM-A", "COMP-1", [pe])
    assert profile.count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — B3: CAUSAL_PRIORITY-based Allen relation selection
# ═══════════════════════════════════════════════════════════════════════════════

def _oob(component_id, relation):
    return {"component_id": component_id, "allen_relation": relation}


def test_b3_single_entry_returned():
    kg = _make_kg(_make_fm(), oob_anomalies=[_oob("C-01", PRECEDES)])
    mapping = TSKRTemporalScorerV1._stage_b_allen_relation_by_component(kg)
    assert mapping["C-01"] == PRECEDES


def test_b3_higher_priority_replaces_lower():
    """OVERLAPS has higher priority than PRECEDES — must win regardless of order."""
    kg = _make_kg(_make_fm(), oob_anomalies=[
        _oob("C-01", PRECEDES),
        _oob("C-01", OVERLAPS),
    ])
    mapping = TSKRTemporalScorerV1._stage_b_allen_relation_by_component(kg)
    assert mapping["C-01"] == OVERLAPS


def test_b3_lower_priority_does_not_displace_higher():
    """FOLLOWS arriving after OVERLAPS must NOT displace OVERLAPS."""
    kg = _make_kg(_make_fm(), oob_anomalies=[
        _oob("C-01", OVERLAPS),
        _oob("C-01", FOLLOWS),
    ])
    mapping = TSKRTemporalScorerV1._stage_b_allen_relation_by_component(kg)
    assert mapping["C-01"] == OVERLAPS


def test_b3_contains_beats_precedes():
    kg = _make_kg(_make_fm(), oob_anomalies=[
        _oob("C-01", PRECEDES),
        _oob("C-01", CONTAINS),
    ])
    mapping = TSKRTemporalScorerV1._stage_b_allen_relation_by_component(kg)
    assert mapping["C-01"] == CONTAINS


def test_b3_independent_components_tracked_separately():
    kg = _make_kg(_make_fm(), oob_anomalies=[
        _oob("C-01", PRECEDES),
        _oob("C-02", FOLLOWS),
    ])
    mapping = TSKRTemporalScorerV1._stage_b_allen_relation_by_component(kg)
    assert mapping["C-01"] == PRECEDES
    assert mapping["C-02"] == FOLLOWS


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — G1: past_events schema normalization
# ═══════════════════════════════════════════════════════════════════════════════

def test_g1_related_failure_modes_remapped():
    """occurred_at / related_failure_modes fields must be remapped to canonical names."""
    pe_new_schema = {
        "event_id": "EVT-OLD",
        "related_failure_modes": ["FM-A"],
        "component_id": "C-01",
        "occurred_at": _iso(BASE - timedelta(days=20)),
        "similarity_score": 0.8,
    }
    normalized = TSKRTemporalScorerV1._normalize_past_events([pe_new_schema])
    assert normalized[0].get("matched_failure_mode_ids") == ["FM-A"]
    assert normalized[0].get("timestamp_start") == _iso(BASE - timedelta(days=20))


def test_g1_canonical_fields_not_overwritten():
    """Events that already have canonical fields must not be modified."""
    pe = _past_event("EVT-1", ["FM-A"], "C-01", days_before=10)
    original_ts = pe["timestamp_start"]
    normalized = TSKRTemporalScorerV1._normalize_past_events([pe])
    assert normalized[0]["timestamp_start"] == original_ts
    assert normalized[0]["matched_failure_mode_ids"] == ["FM-A"]


def test_g1_normalization_enables_recurrence_count():
    """Recurrence count must be > 0 when past_events uses new-schema field names."""
    scorer = _scorer()
    pe = {
        "event_id": "EVT-TC6",
        "related_failure_modes": ["FM-TARGET"],
        "component_id": "C-01",
        "occurred_at": _iso(BASE - timedelta(days=45)),
    }
    kg = _make_kg(_make_fm("FM-TARGET", "C-01"), past_events=[pe])
    result = scorer.score(
        event=_make_event(),
        telemetry_summary={"asset_id": "A", "signals": []},
        kg_context=kg,
        operational_context=None,
        run_context={"run_id": "R-G1"},
    )
    assert result["patterns"][0]["recurrence_count"] == 1


def test_g1_mixed_schema_events_both_counted():
    """Mix of old-schema and new-schema past_events must both contribute to count."""
    scorer = _scorer()
    pe_old = _past_event("CR-OLD", ["FM-TARGET"], "C-01", days_before=60)
    pe_new = {
        "event_id": "CR-NEW",
        "related_failure_modes": ["FM-TARGET"],
        "component_id": "C-01",
        "occurred_at": _iso(BASE - timedelta(days=30)),
    }
    profile = _build_profile(scorer, "FM-TARGET", "C-01",
                             scorer._normalize_past_events([pe_old, pe_new]))
    assert profile.count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — B2: most_recent_days_ago from live timestamps
# ═══════════════════════════════════════════════════════════════════════════════

def test_b2_days_ago_computed_from_event_start():
    """most_recent_days_ago must equal the delta between event_start and the most recent past event."""
    scorer = _scorer()
    days_before = 30
    pe = _past_event("CR-01", ["FM-A"], "C-01", days_before=days_before)
    profile = _build_profile(scorer, "FM-A", "C-01", [pe], event_start=BASE)
    assert profile.most_recent_days_ago == days_before


def test_b2_days_ago_not_from_time_distance_days_when_event_start_given():
    """time_distance_days must be ignored when event_start is provided."""
    scorer = _scorer()
    pe = _past_event("CR-01", ["FM-A"], "C-01", days_before=30)
    pe["time_distance_days"] = 999  # stale KG value
    profile = _build_profile(scorer, "FM-A", "C-01", [pe], event_start=BASE)
    assert profile.most_recent_days_ago != 999
    assert profile.most_recent_days_ago == 30


def test_b2_fallback_to_time_distance_days_without_event_start():
    """Without event_start, time_distance_days is the fallback."""
    scorer = _scorer()
    pe = _past_event("CR-01", ["FM-A"], "C-01", days_before=30)
    pe["time_distance_days"] = 45
    profile = _build_profile(scorer, "FM-A", "C-01", [pe], event_start=None)
    assert profile.most_recent_days_ago == 45


def test_b2_timezone_naive_event_start_handled():
    """Timezone-naive event_start must not raise when past_event timestamp is aware."""
    scorer = _scorer()
    pe = _past_event("CR-01", ["FM-A"], "C-01", days_before=10)  # aware ts
    naive_start = BASE.replace(tzinfo=None)
    profile = _build_profile(scorer, "FM-A", "C-01", [pe], event_start=naive_start)
    assert profile.most_recent_days_ago == 10


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — G2: pm_overdue_boost
# ═══════════════════════════════════════════════════════════════════════════════

def _overdue_item(component_id, check_id="PM-001", days=30):
    return {"check_id": check_id, "component_id": component_id, "overdue_by_days": float(days)}


def test_g2_no_boost_without_pm_compliance():
    scorer = _scorer()
    pat = _score_pattern(scorer, _make_fm("FM-01", "C-01"))
    assert pat["pm_overdue_boost"] == 0.0


def test_g2_no_boost_when_no_matching_component():
    scorer = _scorer()
    pm = {"overdue_items": [_overdue_item("C-OTHER")]}
    pat = _score_pattern(scorer, _make_fm("FM-01", "C-01"), pm_compliance=pm)
    assert pat["pm_overdue_boost"] == 0.0


def test_g2_single_overdue_item_adds_boost():
    scorer = _scorer()
    pm = {"overdue_items": [_overdue_item("C-01")]}
    pat = _score_pattern(scorer, _make_fm("FM-01", "C-01"), pm_compliance=pm)
    assert pat["pm_overdue_boost"] == 0.05


def test_g2_three_overdue_items_capped_at_015():
    scorer = _scorer()
    pm = {"overdue_items": [
        _overdue_item("C-01", "PM-1"),
        _overdue_item("C-01", "PM-2"),
        _overdue_item("C-01", "PM-3"),
    ]}
    pat = _score_pattern(scorer, _make_fm("FM-01", "C-01"), pm_compliance=pm)
    assert pat["pm_overdue_boost"] == 0.15


def test_g2_boost_capped_at_015_for_many_items():
    scorer = _scorer()
    pm = {"overdue_items": [_overdue_item("C-01", f"PM-{i}") for i in range(10)]}
    pat = _score_pattern(scorer, _make_fm("FM-01", "C-01"), pm_compliance=pm)
    assert pat["pm_overdue_boost"] == 0.15


def test_g2_legacy_overdue_tasks_key_accepted():
    """pm_compliance with overdue_tasks (legacy key) must also trigger boost."""
    scorer = _scorer()
    pm = {"overdue_tasks": [_overdue_item("C-01")]}
    pat = _score_pattern(scorer, _make_fm("FM-01", "C-01"), pm_compliance=pm)
    assert pat["pm_overdue_boost"] == 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — G3: accelerating_recurrence attention flag
# ═══════════════════════════════════════════════════════════════════════════════

def test_g3_attention_flag_absent_when_stable_trend():
    """Stable recurrence trend must not produce an attention flag."""
    scorer = _scorer()
    # Only 1 past event → insufficient_data trend (not increasing)
    pe = _past_event("CR-01", ["FM-01"], "C-01", days_before=30)
    pat = _score_pattern(scorer, _make_fm(), past_events=[pe])
    assert "accelerating_recurrence" not in pat.get("attention_flags", [])


def test_g3_attention_flag_set_in_pattern_when_increasing():
    """Pattern must include accelerating_recurrence flag when OLS slope is negative."""
    scorer = _scorer()
    # 5 events with shrinking intervals: 100, 80, 60, 40, 20 days before
    past_events = [
        _past_event(f"CR-{i}", ["FM-01"], "C-01", days_before=d)
        for i, d in enumerate([300, 200, 120, 60, 20])
    ]
    pat = _score_pattern(scorer, _make_fm(), past_events=past_events)
    assert "accelerating_recurrence" in pat.get("attention_flags", [])


def test_g3_attention_flag_escalated_to_rca_card():
    """_apply_accelerating_recurrence_attention_flags must populate analyst_attention_flags."""
    for _mod in ("neo4j", "py2neo", "chromadb", "langchain_chroma", "langchain_community",
                 "langchain_community.vectorstores", "langchain_community.embeddings"):
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock()

    from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator

    rca_card = {"executive_summary": {}}
    tskr_patterns = {
        "patterns": [
            {"target_id": "FM-FAST", "attention_flags": ["accelerating_recurrence"]},
            {"target_id": "FM-OK",   "attention_flags": []},
        ]
    }
    RCAReasoningOrchestrator._apply_accelerating_recurrence_attention_flags(rca_card, tskr_patterns)
    flags = rca_card["executive_summary"].get("analyst_attention_flags", [])
    assert any("accelerating_recurrence" in f or "Accelerating recurrence" in f for f in flags)


def test_g3_no_flag_in_rca_card_when_all_stable():
    for _mod in ("neo4j", "py2neo", "chromadb", "langchain_chroma", "langchain_community",
                 "langchain_community.vectorstores", "langchain_community.embeddings"):
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock()

    from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator

    rca_card = {"executive_summary": {}}
    tskr_patterns = {
        "patterns": [
            {"target_id": "FM-STABLE", "attention_flags": []},
        ]
    }
    RCAReasoningOrchestrator._apply_accelerating_recurrence_attention_flags(rca_card, tskr_patterns)
    flags = rca_card["executive_summary"].get("analyst_attention_flags", [])
    assert len(flags) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — B4: OLS linear regression trend
# ═══════════════════════════════════════════════════════════════════════════════

def test_b4_insufficient_data_below_3_intervals():
    assert TSKRTemporalScorerV1._recurrence_trend([30.0, 20.0]) == "insufficient_data"


def test_b4_insufficient_data_for_empty():
    assert TSKRTemporalScorerV1._recurrence_trend([]) == "insufficient_data"


def test_b4_clearly_increasing_trend():
    """Sharply shrinking intervals must yield 'increasing'."""
    intervals = [100.0, 70.0, 40.0, 20.0, 5.0]
    assert TSKRTemporalScorerV1._recurrence_trend(intervals) == "increasing"


def test_b4_clearly_decreasing_trend():
    """Sharply growing intervals must yield 'decreasing'."""
    intervals = [5.0, 20.0, 40.0, 70.0, 100.0]
    assert TSKRTemporalScorerV1._recurrence_trend(intervals) == "decreasing"


def test_b4_stable_trend():
    """Flat intervals must yield 'stable'."""
    intervals = [30.0, 30.0, 30.0, 30.0, 30.0]
    assert TSKRTemporalScorerV1._recurrence_trend(intervals) == "stable"


def test_b4_outlier_does_not_flip_clear_increase():
    """A single large outlier in the middle must not flip a clear increasing trend."""
    intervals = [90.0, 70.0, 500.0, 30.0, 10.0]  # overall OLS slope is still negative
    result = TSKRTemporalScorerV1._recurrence_trend(intervals)
    # OLS is robust to this; result should be increasing or stable, NOT decreasing
    assert result != "decreasing"


def test_b4_exactly_3_intervals_accepted():
    """3 intervals (minimum) must not return insufficient_data."""
    result = TSKRTemporalScorerV1._recurrence_trend([60.0, 30.0, 10.0])
    assert result == "increasing"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — B5: per-FM signal_novel
# ═══════════════════════════════════════════════════════════════════════════════

def test_b5_signal_novel_true_when_no_matching_parameter():
    """FM with symptom_type 'pressure' must not match a 'temperature' signal."""
    scorer = _scorer()
    fm = _make_fm(symptom_types=["pressure"])
    tel = _make_telemetry(("S-TEMP", "temperature"))
    result = scorer._extract_signal_ids_for_fm(tel, fm)
    assert result == []


def test_b5_signal_novel_false_when_parameter_matches():
    """FM with symptom_type 'pressure' must match a 'pressure' signal."""
    scorer = _scorer()
    fm = _make_fm(symptom_types=["pressure"])
    tel = _make_telemetry(("S-PRESS", "pressure"))
    result = scorer._extract_signal_ids_for_fm(tel, fm)
    assert "S-PRESS" in result


def test_b5_multiple_symptom_types_any_match():
    """FM with ['pressure', 'flow'] must match a 'flow' signal."""
    scorer = _scorer()
    fm = _make_fm(symptom_types=["pressure", "flow"])
    tel = _make_telemetry(("S-FLOW", "flow"), ("S-VIB", "vibration"))
    result = scorer._extract_signal_ids_for_fm(tel, fm)
    assert "S-FLOW" in result
    assert "S-VIB" not in result


def test_b5_fallback_to_global_when_no_symptom_types():
    """FM without expected_symptom_types must fall back to all anomalous signals."""
    scorer = _scorer()
    fm = _make_fm()  # no symptom_types
    tel = _make_telemetry(("S-ANY", "vibration"))
    result = scorer._extract_signal_ids_for_fm(tel, fm)
    assert "S-ANY" in result


def test_b5_signal_novel_in_pattern_respects_fm_filter():
    """signal_novel in full pattern must be True when FM symptom_types don't match signals."""
    scorer = _scorer()
    fm = _make_fm("FM-01", "C-01", symptom_types=["pressure"])
    tel = _make_telemetry(("S-TEMP", "temperature"))  # no pressure signal
    pat = _score_pattern(scorer, fm, telemetry=tel)
    assert pat["signal_novel"] is True


def test_b5_signal_novel_false_in_pattern_when_matching():
    scorer = _scorer()
    fm = _make_fm("FM-01", "C-01", symptom_types=["pressure"])
    tel = _make_telemetry(("S-PRESS", "pressure"))
    pat = _score_pattern(scorer, fm, telemetry=tel)
    assert pat["signal_novel"] is False


def test_b5_matching_signal_ids_fm_specific():
    """matching_signal_ids must include only FM-relevant sensors."""
    scorer = _scorer()
    fm = _make_fm("FM-01", "C-01", symptom_types=["flow"])
    tel = _make_telemetry(("S-FLOW", "flow"), ("S-TEMP", "temperature"))
    pat = _score_pattern(scorer, fm, telemetry=tel)
    assert "S-FLOW" in pat["matching_signal_ids"]
    assert "S-TEMP" not in pat["matching_signal_ids"]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — G4: contributing_event_ids
# ═══════════════════════════════════════════════════════════════════════════════

def test_g4_contributing_ids_empty_when_no_matches():
    scorer = _scorer()
    profile = _build_profile(scorer, "FM-A", "C-01", [])
    assert profile.contributing_event_ids == []


def test_g4_event_id_captured():
    scorer = _scorer()
    pe = _past_event("CR-ALPHA", ["FM-A"], "C-01", days_before=20)
    profile = _build_profile(scorer, "FM-A", "C-01", [pe])
    assert "CR-ALPHA" in profile.contributing_event_ids


def test_g4_multiple_events_all_captured():
    scorer = _scorer()
    pes = [
        _past_event("CR-001", ["FM-A"], "C-01", days_before=60),
        _past_event("CR-002", ["FM-A"], "C-01", days_before=30),
    ]
    profile = _build_profile(scorer, "FM-A", "C-01", pes)
    assert "CR-001" in profile.contributing_event_ids
    assert "CR-002" in profile.contributing_event_ids


def test_g4_contributing_ids_in_pattern_output():
    """contributing_event_ids must appear in the full pattern dict."""
    scorer = _scorer()
    pe = _past_event("CR-TRACE", ["FM-01"], "C-01", days_before=20)
    pat = _score_pattern(scorer, _make_fm(), past_events=[pe])
    assert "contributing_event_ids" in pat
    assert "CR-TRACE" in pat["contributing_event_ids"]


def test_g4_no_duplicate_ids():
    """Same event_id appearing twice must not produce duplicates."""
    scorer = _scorer()
    pe = _past_event("CR-DUP", ["FM-A"], "C-01", days_before=20)
    profile = _build_profile(scorer, "FM-A", "C-01", [pe, pe])
    assert profile.contributing_event_ids.count("CR-DUP") == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — B6: unresolved_count aligned to matching set
# ═══════════════════════════════════════════════════════════════════════════════

def test_b6_unresolved_counted_in_matching_set():
    """An unresolved event without a timestamp must still increment unresolved_count."""
    scorer = _scorer()
    # Event with no timestamp_start — excluded from dated but still in matching
    pe_no_ts = {
        "event_id": "CR-NO-TS",
        "matched_failure_mode_ids": ["FM-A"],
        "component_id": "C-01",
        "resolved": False,
    }
    profile = _build_profile(scorer, "FM-A", "C-01", [pe_no_ts])
    assert profile.count == 1
    assert profile.unresolved_count == 1


def test_b6_resolved_true_not_counted():
    scorer = _scorer()
    pe = _past_event("CR-R", ["FM-A"], "C-01", days_before=10, resolved=True)
    profile = _build_profile(scorer, "FM-A", "C-01", [pe])
    assert profile.unresolved_count == 0


def test_b6_resolved_none_not_counted():
    """resolved=None (missing) must not count as unresolved."""
    scorer = _scorer()
    pe = _past_event("CR-R", ["FM-A"], "C-01", days_before=10)
    pe.pop("resolved", None)
    profile = _build_profile(scorer, "FM-A", "C-01", [pe])
    assert profile.unresolved_count == 0


def test_b6_count_and_unresolved_same_denominator():
    """count must equal sum(resolved+unresolved) in matching."""
    scorer = _scorer()
    pes = [
        _past_event("CR-1", ["FM-A"], "C-01", days_before=60, resolved=True),
        _past_event("CR-2", ["FM-A"], "C-01", days_before=30, resolved=False),
        {   # no timestamp — excluded from dated but in matching
            "event_id": "CR-3",
            "matched_failure_mode_ids": ["FM-A"],
            "component_id": "C-01",
            "resolved": False,
        },
    ]
    profile = _build_profile(scorer, "FM-A", "C-01", pes)
    assert profile.count == 3
    assert profile.unresolved_count == 2


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
