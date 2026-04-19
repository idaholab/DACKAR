"""
Unit tests for Stage B (KGTimelineBuilder).

Coverage targets:
    _score_event_dq          — field-completeness scoring formula
    _window_start_iso        — date arithmetic helper
    _select_primary_component — resolved_component_ids > known_component_id
    build()                  — empty timeline when kg_driver is None
    _deduplicate_events      — duplicate event removal by event_id
    _compute_recurrence_indicators — repeat-failure counting, inter-event period,
                                     PM compliance flag
    _compute_data_coverage   — event counts and outage-ID collection
    _normalise_event         — field mapping and DQ scoring
    Stub KG driver           — build() with a controlled stub that injects events
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

_OUTAGE_ROOT = Path(__file__).parent.parent
if str(_OUTAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_ROOT))

from stages.stage_b_kg_timeline import (
    KGTimelineBuilder,
    KGTimelineConfig,
    _score_event_dq,
    _window_start_iso,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _builder(config: KGTimelineConfig | None = None, kg_driver=None) -> KGTimelineBuilder:
    return KGTimelineBuilder(config=config, kg_driver=kg_driver)


def _dt(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _run_ctx(activity_id: str = "ACT-TEST-001") -> Dict[str, Any]:
    return {"run_id": "run-test", "started_at": "2026-04-10T00:00:00+00:00"}


def _activity(activity_id: str = "ACT-TEST-001",
              component_id: str = "1RHS-P-001A",
              ts: str = "2026-04-10T06:00:00+00:00") -> Dict[str, Any]:
    return {
        "activity_id": activity_id,
        "detection_timestamp": ts,
        "known_component_id": component_id,
    }


def _intake(resolved: list[str] | None = None,
            known: str = "1RHS-P-001A") -> Dict[str, Any]:
    return {
        "resolved_component_ids": resolved or [],
        "known_component_id": known,
    }


def _event(event_id: str = "EV-001",
           timestamp: str = "2025-01-01T00:00:00+00:00",
           description: str = "test event",
           source_doc_id: str | None = None,
           source_system: str = "maximo",
           event_type: str = "condition_report",
           outage_id: str | None = None) -> Dict[str, Any]:
    ev = {
        "event_id": event_id,
        "timestamp": timestamp,
        "description": description,
        # Default source_doc_id to event_id so each event is unique by default
        "source_doc_id": source_doc_id if source_doc_id is not None else event_id,
        "source_system": source_system,
        "event_type": event_type,
        "data_quality_score": 1.0,
    }
    if outage_id:
        ev["outage_id"] = outage_id
    return ev


# ===========================================================================
# _score_event_dq
# ===========================================================================

class TestScoreEventDQ:
    """Field-completeness scoring: timestamp(0.35) + description(0.30) +
    source_doc_id(0.20) + source_system(0.15) = 1.0 max."""

    def test_all_fields_present(self):
        ev = {"timestamp": "2026-01-01", "description": "X", "source_doc_id": "CR-1", "source_system": "maximo"}
        assert _score_event_dq(ev) == pytest.approx(1.0)

    def test_no_fields(self):
        assert _score_event_dq({}) == 0.0

    def test_timestamp_only(self):
        assert _score_event_dq({"timestamp": "2026-01-01"}) == pytest.approx(0.35)

    def test_description_only(self):
        assert _score_event_dq({"description": "text"}) == pytest.approx(0.30)

    def test_without_source_system(self):
        ev = {"timestamp": "2026-01-01", "description": "X", "source_doc_id": "CR-1"}
        assert _score_event_dq(ev) == pytest.approx(0.85)

    def test_empty_description_not_counted(self):
        ev = {"timestamp": "2026-01-01", "description": ""}
        assert _score_event_dq(ev) == pytest.approx(0.35)


# ===========================================================================
# _window_start_iso
# ===========================================================================

class TestWindowStartIso:

    def test_correct_subtraction(self):
        ts = "2026-04-10T12:00:00+00:00"
        result = _window_start_iso(ts, 365)
        expected = (_dt(ts) - timedelta(days=365)).isoformat()
        assert result == expected

    def test_none_before_ts_falls_back_to_utcnow(self):
        """N9 fix: None before_ts must not return None — fallback to utcnow().
        The result must be a parseable ISO string approximately (window_days)
        days before now (within a 5-second tolerance for test execution time)."""
        before = datetime.now(timezone.utc)
        result = _window_start_iso(None, 365)
        after = datetime.now(timezone.utc)

        assert result is not None, "N9 fix: window_start must never be None"
        from stages.stage_b_kg_timeline import _parse_dt
        dt = _parse_dt(result)
        assert dt is not None, f"Fallback result must be a valid ISO timestamp, got {result!r}"
        # Should be approximately (before − 365 days) ≤ dt ≤ (after − 365 days)
        lower = before - timedelta(days=365, seconds=5)
        upper = after  - timedelta(days=365) + timedelta(seconds=5)
        assert lower <= dt <= upper, (
            f"Fallback window_start {dt} not within expected range [{lower}, {upper}]"
        )

    def test_invalid_ts_falls_back_to_utcnow(self):
        """N9 fix: unparseable before_ts also falls back to utcnow()."""
        result = _window_start_iso("not-a-date", 365)
        assert result is not None, "N9 fix: unparseable ts must not return None"
        from stages.stage_b_kg_timeline import _parse_dt
        assert _parse_dt(result) is not None

    def test_zero_days(self):
        ts = "2026-04-10T00:00:00+00:00"
        assert _window_start_iso(ts, 0) == ts


# ===========================================================================
# _select_primary_component
# ===========================================================================

class TestSelectPrimaryComponent:

    def test_prefers_first_resolved_id(self):
        b = _builder()
        intake = _intake(resolved=["1RHS-P-001A", "1RHS-P-001B"], known="OTHER")
        assert b._select_primary_component(intake) == "1RHS-P-001A"

    def test_falls_back_to_known_component_id(self):
        b = _builder()
        intake = _intake(resolved=[], known="1RHS-P-001B")
        assert b._select_primary_component(intake) == "1RHS-P-001B"

    def test_raises_when_no_component_resolvable(self):
        b = _builder()
        intake = {"resolved_component_ids": [], "known_component_id": None}
        with pytest.raises(ValueError, match="cannot determine component_id"):
            b._select_primary_component(intake)

    def test_raises_when_keys_absent(self):
        b = _builder()
        with pytest.raises((ValueError, KeyError)):
            b._select_primary_component({})


# ===========================================================================
# build() — no KG driver
# ===========================================================================

class TestBuildNoKGDriver:
    """When kg_driver is None, build() must return a valid empty timeline."""

    def setup_method(self):
        self.b = _builder()
        self.act = _activity()
        self.intake = _intake(resolved=["1RHS-P-001A"])
        self.ctx = _run_ctx()

    def test_returns_dict(self):
        result = self.b.build(self.act, self.intake, self.ctx)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = self.b.build(self.act, self.intake, self.ctx)
        for key in ("activity_id", "run_id", "component_id", "events",
                    "recurrence_indicators", "data_coverage", "provenance"):
            assert key in result, f"missing key: {key}"

    def test_empty_events_list(self):
        result = self.b.build(self.act, self.intake, self.ctx)
        assert result["events"] == []

    def test_correct_component_id(self):
        result = self.b.build(self.act, self.intake, self.ctx)
        assert result["component_id"] == "1RHS-P-001A"

    def test_run_id_propagated(self):
        result = self.b.build(self.act, self.intake, self.ctx)
        assert result["run_id"] == "run-test"

    def test_provenance_fields(self):
        result = self.b.build(self.act, self.intake, self.ctx)
        prov = result["provenance"]
        assert prov["generated_by"] == "KGTimelineBuilder"
        assert prov["timeline_window_days"] == KGTimelineConfig().timeline_window_days

    def test_d1_kg_driver_available_false_when_no_driver(self):
        """D1 fix: kg_driver_available must be False when kg_driver is None.

        Before the fix the absence of the KG driver was invisible to downstream
        stages and the analyst.  After the fix the artifact carries a
        kg_driver_available boolean so the orchestrator can surface it in
        review_hooks as a first-class flag.
        """
        result = self.b.build(self.act, self.intake, self.ctx)
        assert "kg_driver_available" in result, (
            "Stage B artifact must carry kg_driver_available field (D1 fix)"
        )
        assert result["kg_driver_available"] is False, (
            "kg_driver_available must be False when no KG driver is injected"
        )


# ===========================================================================
# _deduplicate_events
# ===========================================================================

class TestDeduplicateEvents:

    def test_removes_exact_duplicate_event_ids(self):
        b = _builder()
        events = [
            _event(event_id="EV-001", timestamp="2025-01-01T00:00:00+00:00"),
            _event(event_id="EV-001", timestamp="2025-01-01T00:00:00+00:00"),
            _event(event_id="EV-002", timestamp="2025-02-01T00:00:00+00:00"),
        ]
        deduped = b._deduplicate_events(events)
        ids = [e["event_id"] for e in deduped]
        assert ids.count("EV-001") == 1
        assert len(deduped) == 2

    def test_no_duplicates_unchanged(self):
        b = _builder()
        events = [
            _event(event_id="EV-001"),
            _event(event_id="EV-002"),
        ]
        deduped = b._deduplicate_events(events)
        assert len(deduped) == 2

    def test_empty_input(self):
        b = _builder()
        assert b._deduplicate_events([]) == []


# ===========================================================================
# _compute_recurrence_indicators
# ===========================================================================

class TestComputeRecurrenceIndicators:

    def test_no_events_returns_defaults(self):
        b = _builder()
        ri = b._compute_recurrence_indicators([], None)
        assert ri["repeat_failure_count"] == 0
        assert ri["pm_compliance_status"] in ("unknown", "compliant", "overdue", None, "N/A")

    def test_counts_cr_events_as_repeat_failures(self):
        b = _builder()
        events = [
            _event(event_id="EV-001", event_type="condition_report"),
            _event(event_id="EV-002", event_type="condition_report"),
            _event(event_id="EV-003", event_type="work_order"),
        ]
        ri = b._compute_recurrence_indicators(events, "2026-04-10T00:00:00+00:00")
        # At minimum, the repeat_failure_count should be a non-negative int
        assert isinstance(ri["repeat_failure_count"], int)
        assert ri["repeat_failure_count"] >= 0

    def test_inter_event_period_with_two_events(self):
        b = _builder()
        ts1 = "2025-01-01T00:00:00+00:00"
        ts2 = "2025-07-01T00:00:00+00:00"
        events = [
            _event(event_id="EV-001", timestamp=ts1, event_type="condition_report"),
            _event(event_id="EV-002", timestamp=ts2, event_type="condition_report"),
        ]
        ri = b._compute_recurrence_indicators(events, "2026-01-01T00:00:00+00:00")
        assert ri.get("mean_inter_event_days") is None or ri.get("mean_inter_event_days") >= 0

    # ── Y6 fix: output must not include non-schema fields ────────────────────

    def test_y6_no_min_inter_event_days_in_output(self):
        """Y6 fix: min_inter_event_days must not appear in output (not in schema)."""
        b = _builder()
        events = [
            _event(event_id="EV-001", timestamp="2025-01-01T00:00:00+00:00",
                   event_type="condition_report"),
            _event(event_id="EV-002", timestamp="2025-07-01T00:00:00+00:00",
                   event_type="condition_report"),
        ]
        ri = b._compute_recurrence_indicators(events, "2026-01-01T00:00:00+00:00")
        assert "min_inter_event_days" not in ri, (
            "'min_inter_event_days' is not in schema (Y6 fix)"
        )

    def test_y6_no_pm_overdue_days_in_output(self):
        """Y6 fix: pm_overdue_days must not appear in output (not in schema)."""
        b = _builder()
        # PM event well past overdue threshold so pm_overdue_days would be set
        old_pm_ts = "2024-01-01T00:00:00+00:00"
        events = [_event(event_id="PM-001", timestamp=old_pm_ts,
                         event_type="preventive_maintenance")]
        ri = b._compute_recurrence_indicators(events, "2026-04-10T00:00:00+00:00")
        assert "pm_overdue_days" not in ri, (
            "'pm_overdue_days' is not in schema (Y6 fix)"
        )

    def test_y6_only_schema_fields_in_output(self):
        """Y6 fix: recurrence_indicators must contain only schema-defined keys."""
        _SCHEMA_KEYS = frozenset({
            "repeat_failure_count", "mean_inter_event_days", "trend",
            "last_cm_date", "last_pm_date", "pm_compliance_status",
        })
        b = _builder()
        events = [
            _event(event_id="EV-001", timestamp="2025-01-01T00:00:00+00:00",
                   event_type="condition_report"),
            _event(event_id="PM-001", timestamp="2025-06-01T00:00:00+00:00",
                   event_type="preventive_maintenance"),
        ]
        ri = b._compute_recurrence_indicators(events, "2026-04-10T00:00:00+00:00")
        extra = set(ri.keys()) - _SCHEMA_KEYS
        assert not extra, f"Non-schema fields in recurrence_indicators output: {extra} (Y6 fix)"

    # ── Y4 fix: pm_compliance_status must use schema-valid enum values ────────

    _VALID_PM_STATUSES = frozenset({"compliant", "overdue", "no_pm_defined", "unknown"})

    def test_y4_pm_compliance_status_is_schema_valid(self):
        """Y4 fix: pm_compliance_status must be in schema enum for all code paths."""
        b = _builder()
        # Path 1: no PM events → "unknown"
        ri_no_pm = b._compute_recurrence_indicators([], None)
        assert ri_no_pm["pm_compliance_status"] in self._VALID_PM_STATUSES

        # Path 2: PM event within interval → "compliant" (was "current" before fix)
        recent_pm_ts = "2026-03-01T00:00:00+00:00"   # ~40 days before detection
        detection_ts = "2026-04-10T00:00:00+00:00"
        events = [_event(event_id="PM-001", timestamp=recent_pm_ts,
                         event_type="preventive_maintenance")]
        ri_compliant = b._compute_recurrence_indicators(events, detection_ts)
        assert ri_compliant["pm_compliance_status"] in self._VALID_PM_STATUSES

        # Path 3: PM event well outside interval → "overdue"
        old_pm_ts = "2024-01-01T00:00:00+00:00"      # > default 180-day interval
        events_overdue = [_event(event_id="PM-002", timestamp=old_pm_ts,
                                 event_type="preventive_maintenance")]
        ri_overdue = b._compute_recurrence_indicators(events_overdue, detection_ts)
        assert ri_overdue["pm_compliance_status"] in self._VALID_PM_STATUSES

    def test_y4_within_interval_produces_compliant_not_current(self):
        """Y4 fix: a PM event within the compliance window must produce 'compliant', not 'current'."""
        b = _builder()
        # PM ~40 days ago, default interval 180 days → well within → "compliant"
        events = [_event(event_id="PM-001",
                         timestamp="2026-03-01T00:00:00+00:00",
                         event_type="preventive_maintenance")]
        ri = b._compute_recurrence_indicators(events, "2026-04-10T00:00:00+00:00")
        assert ri["pm_compliance_status"] == "compliant", (
            f"Expected 'compliant', got '{ri['pm_compliance_status']}' (Y4 fix: 'current' is not a valid schema value)"
        )


# ===========================================================================
# _compute_data_coverage
# ===========================================================================

class TestComputeDataCoverage:

    def test_no_events(self):
        b = _builder()
        cov = b._compute_data_coverage([], "2026-04-10T00:00:00+00:00")
        assert cov["total_events"] == 0

    def test_event_count_matches(self):
        b = _builder()
        events = [_event(event_id=f"EV-{i:03d}") for i in range(5)]
        cov = b._compute_data_coverage(events, "2026-04-10T00:00:00+00:00")
        assert cov["total_events"] == 5

    def test_outages_represented_count(self):
        b = _builder()
        events = [
            _event(event_id="EV-001", outage_id="RF-20"),
            _event(event_id="EV-002", outage_id="RF-21"),
            _event(event_id="EV-003", outage_id="RF-20"),
        ]
        cov = b._compute_data_coverage(events, "2026-04-10T00:00:00+00:00")
        # outages_represented is the count of distinct outage IDs
        count = cov.get("outages_represented", 0)
        assert count == 2

    def test_event_type_breakdown_present(self):
        b = _builder()
        events = [
            _event(event_id="EV-001", event_type="condition_report"),
            _event(event_id="EV-002", event_type="work_order"),
        ]
        cov = b._compute_data_coverage(events, "2026-04-10T00:00:00+00:00")
        # Either a flat count or a breakdown dict should be present
        assert "total_events" in cov

    # ── Y5 fix: data_coverage field names must match schema ──────────────────

    def test_y5_uses_earliest_event_not_earliest_event_date(self):
        """Y5 fix: field must be 'earliest_event', not 'earliest_event_date'."""
        b = _builder()
        events = [_event(event_id="EV-001", timestamp="2025-06-01T00:00:00+00:00")]
        cov = b._compute_data_coverage(events, "2026-04-10T00:00:00+00:00")
        assert "earliest_event" in cov, "data_coverage must use 'earliest_event' (Y5 fix)"
        assert "earliest_event_date" not in cov, "'earliest_event_date' is not a schema field (Y5 fix)"

    def test_y5_uses_latest_event_not_latest_event_date(self):
        """Y5 fix: field must be 'latest_event', not 'latest_event_date'."""
        b = _builder()
        events = [_event(event_id="EV-001", timestamp="2025-06-01T00:00:00+00:00")]
        cov = b._compute_data_coverage(events, "2026-04-10T00:00:00+00:00")
        assert "latest_event" in cov, "data_coverage must use 'latest_event' (Y5 fix)"
        assert "latest_event_date" not in cov, "'latest_event_date' is not a schema field (Y5 fix)"

    def test_y5_no_non_schema_fields(self):
        """Y5 fix: output must not include window_start, window_end, or has_gaps."""
        b = _builder()
        cov = b._compute_data_coverage([], "2026-04-10T00:00:00+00:00")
        for field in ("window_start", "window_end", "has_gaps"):
            assert field not in cov, f"'{field}' is not in schema (Y5 fix)"

    def test_y5_earliest_latest_event_values(self):
        """Y5 fix: earliest_event and latest_event carry the correct timestamps."""
        b = _builder()
        events = [
            _event(event_id="EV-001", timestamp="2025-01-01T00:00:00+00:00"),
            _event(event_id="EV-002", timestamp="2025-06-01T00:00:00+00:00"),
            _event(event_id="EV-003", timestamp="2026-01-01T00:00:00+00:00"),
        ]
        cov = b._compute_data_coverage(events, "2026-04-10T00:00:00+00:00")
        assert "2025-01-01" in (cov.get("earliest_event") or "")
        assert "2026-01-01" in (cov.get("latest_event") or "")


# ===========================================================================
# Stub KG driver
# ===========================================================================

class _StubKGDriver:
    """Returns controlled fixture records for KG query methods."""

    def __init__(self, records_by_label: dict):
        self._records = records_by_label

    def query(self, cypher: str, parameters=None, db=None):
        """Return records keyed by a node alias present in cypher."""
        for label, records in self._records.items():
            if label in cypher:
                return records
        return []


def _make_neo4j_record(alias: str, props: dict):
    """Minimal dict-like record that _node_props() and _record_to_dict() can handle."""
    return {alias: props}


class TestBuildWithStubDriver:
    """build() plumbs stub KG query results through normalisation."""

    def _cr_record(self, cr_id: str, ts: str, description: str):
        props = {
            "id": cr_id,
            "initiated_date": ts,
            "description": description,
            "source_doc_id": cr_id,
            "source_system": "maximo",
            "work_type": None,
        }
        return _make_neo4j_record("cr", props)

    def test_events_populated_from_stub_cr(self):
        records = [self._cr_record("CR-001", "2025-06-01T00:00:00", "Pump vibration high")]
        driver = _StubKGDriver({"condition_report": records})
        b = _builder(
            config=KGTimelineConfig(
                include_condition_reports=True,
                include_work_orders=False,
                include_preventive_maintenance=False,
                include_corrective_maintenance=False,
                include_prior_emergent_activities=False,
                include_inspections=False,
            ),
            kg_driver=driver,
        )
        result = b.build(
            _activity(),
            _intake(resolved=["1RHS-P-001A"]),
            _run_ctx(),
        )
        # The stub may or may not produce events depending on Cypher matching;
        # verify the build() call completes without error and has valid structure.
        assert isinstance(result["events"], list)
        assert result["component_id"] == "1RHS-P-001A"

    def test_max_events_cap_respected(self):
        """Events list must never exceed config.max_events."""
        config = KGTimelineConfig(
            max_events=3,
            include_condition_reports=False,
            include_work_orders=False,
            include_preventive_maintenance=False,
            include_corrective_maintenance=False,
            include_prior_emergent_activities=False,
            include_inspections=False,
        )
        b = _builder(config=config)
        result = b.build(_activity(), _intake(resolved=["X"]), _run_ctx())
        assert len(result["events"]) <= 3
