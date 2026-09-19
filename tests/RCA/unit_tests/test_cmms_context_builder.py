"""Unit tests for cmms_integration.cmms_context_builder.CMMSContextBuilder.

These lock down the review fixes on the CMMSContextBuilder:
  * last_pm lookback anchor fires for a schema-shaped kg_context, is scoped to
    the primary asset, and ignores PMs dated after the event;
  * the fallback anchor label is honest when fallback_lookback_days != 90;
  * component_id is resolved from functional_location / equipment_id via the KG;
  * enriched records are projected onto the cmms_context schema whitelist;
  * caps and the recurrence summary order by parsed datetime (mixed offsets);
  * get_chroma_documents emits Chroma-clean metadata (no None / no lists);
  * days_before_event rounds toward zero.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dackar.RCA.cmms_integration.cmms_adapter import MockCMMSAdapter
from dackar.RCA.cmms_integration.cmms_context_builder import (
    _CR_SCHEMA_KEYS,
    CMMSContextBuilder,
    CMMSContextBuilderConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EVENT_TS = "2026-03-01T00:00:00+00:00"


def _event() -> dict:
    return {"event_id": "E1", "asset_id": "ASSET-1", "timestamp_start": EVENT_TS}


def _kg_context(past_events=None) -> dict:
    return {
        "subgraph_id": "SG1",
        "asset_id": "ASSET-1",
        "components": [
            {
                "component_id": "C-PRIMARY",
                "component_label": "Pump A",
                "relation_to_asset": "primary",
                "maximo_floc": "PLANT/PUMP-01",
            },
            {
                "component_id": "C-SISTER",
                "component_label": "Pump B",
                "relation_to_asset": "same_train",
                "maximo_floc": "PLANT/PUMP-02",
                "sap_equipment_id": "EQ-SISTER",
            },
        ],
        "past_events": past_events if past_events is not None else [],
    }


def _build(cr_records=None, wo_records=None, past_events=None, config=None) -> dict:
    adapter = MockCMMSAdapter(cr_records=cr_records or [], wo_records=wo_records or [])
    builder = CMMSContextBuilder(adapter, config=config)
    return builder.build(_event(), _kg_context(past_events), run_id="RUN-1")


# ---------------------------------------------------------------------------
# Lookback anchor: last_pm (I1) + primary-asset filter (N3) + before-event (A1)
# ---------------------------------------------------------------------------

class TestLookbackAnchor:
    def test_last_pm_fires_for_schema_shaped_past_events(self):
        """A schema-shaped PM (timestamp_start) on the primary asset anchors the window."""
        past = [
            {"event_id": "PM_OLD", "asset_id": "ASSET-1", "event_type": "PM",
             "timestamp_start": "2026-01-01T00:00:00+00:00"},
            {"event_id": "PM_LAST", "asset_id": "ASSET-1", "event_type": "preventive_maintenance",
             "timestamp_start": "2026-02-01T00:00:00+00:00"},
        ]
        ctx = _build(past_events=past)
        assert ctx["lookback_anchor"] == "last_pm"
        assert ctx["lookback_from"] == "2026-02-01T00:00:00+00:00"

    def test_pm_on_sister_asset_is_ignored(self):
        """past_events on another asset must not anchor the primary window (N3)."""
        past = [
            {"event_id": "PM_SIS", "asset_id": "ASSET-2", "event_type": "PM",
             "timestamp_start": "2026-02-20T00:00:00+00:00"},
        ]
        ctx = _build(past_events=past)
        # No primary-asset PM → fallback, not the sister's 2026-02-20 date.
        assert ctx["lookback_anchor"] == "event_time_minus_90d"

    def test_pm_after_event_is_ignored(self):
        """A PM dated after the event cannot anchor a backward window (A1)."""
        past = [
            {"event_id": "PM_FUT", "asset_id": "ASSET-1", "event_type": "PM",
             "timestamp_start": "2026-05-01T00:00:00+00:00"},
        ]
        ctx = _build(past_events=past)
        assert ctx["lookback_anchor"] == "event_time_minus_90d"
        # Window is never inverted.
        assert ctx["lookback_from"] <= ctx["lookback_to"]


# ---------------------------------------------------------------------------
# Fallback anchor label honesty (I5)
# ---------------------------------------------------------------------------

class TestFallbackAnchorLabel:
    def test_default_90_day_label(self):
        ctx = _build()  # no PMs, default fallback of 90
        assert ctx["lookback_anchor"] == "event_time_minus_90d"

    def test_non_90_day_reports_custom(self):
        cfg = CMMSContextBuilderConfig(fallback_lookback_days=30)
        ctx = _build(config=cfg)
        assert ctx["lookback_anchor"] == "custom"
        # 30 days before the event, and the real count is preserved in provenance.
        expected = (datetime.fromisoformat(EVENT_TS) - timedelta(days=30)).isoformat()
        assert ctx["lookback_from"] == expected
        assert ctx["provenance"]["query_params"]["fallback_lookback_days"] == 30


# ---------------------------------------------------------------------------
# component_id resolution (I3)
# ---------------------------------------------------------------------------

class TestComponentIdResolution:
    def test_resolved_from_functional_location(self):
        cr = {"cr_id": "CR1", "status": "OPEN", "short_description": "seal leak",
              "created_date": "2026-02-15T00:00:00+00:00", "functional_location": "PLANT/PUMP-01"}
        ctx = _build(cr_records=[cr])
        rec = ctx["cr_records"][0]
        assert rec["component_id"] == "C-PRIMARY"
        assert rec["is_sister_equipment"] is False

    def test_resolved_from_equipment_id_and_marked_sister(self):
        cr = {"cr_id": "CR2", "status": "COMP", "short_description": "bearing",
              "created_date": "2026-02-10T00:00:00+00:00", "equipment_id": "EQ-SISTER"}
        ctx = _build(cr_records=[cr])
        rec = ctx["cr_records"][0]
        assert rec["component_id"] == "C-SISTER"
        assert rec["is_sister_equipment"] is True

    def test_adapter_supplied_component_id_wins(self):
        cr = {"cr_id": "CR3", "status": "OPEN", "short_description": "x",
              "created_date": "2026-02-10T00:00:00+00:00",
              "functional_location": "PLANT/PUMP-01", "component_id": "C-EXPLICIT"}
        ctx = _build(cr_records=[cr])
        assert ctx["cr_records"][0]["component_id"] == "C-EXPLICIT"


# ---------------------------------------------------------------------------
# Schema-whitelist projection (I2)
# ---------------------------------------------------------------------------

class TestSchemaProjection:
    def test_non_schema_fields_dropped(self):
        cr = {
            "cr_id": "CR1", "status": "OPEN", "short_description": "x",
            "created_date": "2026-02-15T00:00:00+00:00",
            "functional_location": "PLANT/PUMP-01",
            # Path-A extras + arbitrary adapter field — none are schema keys.
            "condition_assessment": {"as_found": "worn"},
            "failure_mode_refs": [{"fm_id": "FM-1"}],
            "extracted_causal_statements": [{"cause_text": "wear"}],
            "adapter_internal": "leak-me",
        }
        rec = _build(cr_records=[cr])["cr_records"][0]
        assert set(rec).issubset(_CR_SCHEMA_KEYS)
        for stray in ("condition_assessment", "failure_mode_refs",
                      "extracted_causal_statements", "adapter_internal"):
            assert stray not in rec


# ---------------------------------------------------------------------------
# Datetime-ordered cap + recurrence summary (I4)
# ---------------------------------------------------------------------------

class TestDatetimeOrdering:
    def test_cap_keeps_most_recent_by_parsed_datetime(self):
        # crA offset makes it earlier in real time than crB, but later as a raw
        # string — a string sort would keep the wrong one.
        cr_a = {"cr_id": "A", "status": "OPEN", "short_description": "a",
                "created_date": "2026-02-15T00:00:00+05:00"}   # 2026-02-14T19:00Z
        cr_b = {"cr_id": "B", "status": "OPEN", "short_description": "b",
                "created_date": "2026-02-14T20:00:00+00:00"}   # 2026-02-14T20:00Z (later)
        cfg = CMMSContextBuilderConfig(max_cr_records=1)
        ctx = _build(cr_records=[cr_a, cr_b], config=cfg)
        assert [r["cr_id"] for r in ctx["cr_records"]] == ["B"]

    def test_recurrence_summary_earliest_and_most_recent(self):
        cr_a = {"cr_id": "A", "status": "OPEN", "short_description": "a",
                "created_date": "2026-02-15T00:00:00+05:00"}   # 2026-02-14T19:00Z
        cr_b = {"cr_id": "B", "status": "OPEN", "short_description": "b",
                "created_date": "2026-02-14T20:00:00+00:00"}   # later
        summ = _build(cr_records=[cr_a, cr_b])["recurrence_summary"]
        assert summ["earliest_related_cr_date"] == "2026-02-15T00:00:00+05:00"
        assert summ["most_recent_cr_date"] == "2026-02-14T20:00:00+00:00"


# ---------------------------------------------------------------------------
# Chroma-clean metadata (A2)
# ---------------------------------------------------------------------------

class TestChromaCleanMetadata:
    def test_metadata_has_no_none_or_list_values(self):
        cr = {"cr_id": "CR1", "status": "OPEN", "short_description": "seal leak",
              "created_date": "2026-02-15T00:00:00+00:00",
              "functional_location": "PLANT/PUMP-01", "long_text": "narrative"}
        builder = CMMSContextBuilder(MockCMMSAdapter(cr_records=[cr]))
        ctx = builder.build(_event(), _kg_context(), run_id="RUN-1")
        docs = builder.get_chroma_documents(ctx)
        assert docs, "expected at least one chroma doc"
        meta = docs[0]["metadata"]
        assert all(v is not None for v in meta.values())
        assert all(not isinstance(v, (list, dict)) for v in meta.values())
        # component_ids is a list in the raw dict → JSON-encoded to a string.
        assert isinstance(meta["component_ids"], str)
        assert meta["component_id"] == "C-PRIMARY"

    def test_unresolved_component_id_dropped_from_metadata(self):
        # No functional_location / equipment_id → component_id stays None → dropped.
        cr = {"cr_id": "CR9", "status": "OPEN", "short_description": "x",
              "created_date": "2026-02-15T00:00:00+00:00", "long_text": "n"}
        builder = CMMSContextBuilder(MockCMMSAdapter(cr_records=[cr]))
        ctx = builder.build(_event(), _kg_context(), run_id="RUN-1")
        meta = builder.get_chroma_documents(ctx)[0]["metadata"]
        assert "component_id" not in meta          # None dropped
        assert "component_ids" not in meta          # empty list dropped


# ---------------------------------------------------------------------------
# days_before_event rounds toward zero (N4)
# ---------------------------------------------------------------------------

class TestDaysBeforeEvent:
    def test_hours_after_event_rounds_to_zero(self):
        cr = {"cr_id": "CR1", "status": "OPEN", "short_description": "x",
              "created_date": "2026-03-01T03:00:00+00:00"}   # 3h after the event
        rec = _build(cr_records=[cr])["cr_records"][0]
        assert rec["days_before_event"] == 0

    def test_days_before_event_positive(self):
        cr = {"cr_id": "CR1", "status": "OPEN", "short_description": "x",
              "created_date": "2026-02-27T00:00:00+00:00"}   # 2 days before
        rec = _build(cr_records=[cr])["cr_records"][0]
        assert rec["days_before_event"] == 2
