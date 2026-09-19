"""
Unit tests for CMMSContextBuilder, CMMSContextBuilderConfig,
NoOpCMMSAdapter, and MockCMMSAdapter.

This module has two halves:
  * the baseline behavioural suite (adapters, config, package structure,
    lookback, sisters, enrichment, recurrence, Chroma extraction, caps); and
  * the ``TestLookbackAnchor``/``TestFallbackAnchorLabel``/
    ``TestComponentIdResolution``/``TestSchemaProjection``/
    ``TestDatetimeOrdering``/``TestChromaCleanMetadata``/``TestDaysBeforeEvent``
    classes that lock down the MR #53 review fixes on the builder.
The review-fix classes use ``_rf_``-prefixed fixtures so they do not shadow the
baseline fixtures above them.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from cmms_integration.cmms_adapter import MockCMMSAdapter, NoOpCMMSAdapter
from cmms_integration.cmms_context_builder import (
    _CR_SCHEMA_KEYS,
    CMMSContextBuilder,
    CMMSContextBuilderConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _event(event_id="EVT-001", asset_id="PUMP-01", event_time="2026-01-10T12:00:00+00:00"):
    return {
        "event_id": event_id,
        "asset_id": asset_id,
        "event_time": event_time,
        "event_type": "vibration_exceedance",
    }


def _kg_context(
    subgraph_id="SG-001",
    components=None,
    past_events=None,
):
    if components is None:
        components = [
            {
                "component_id": "COMP-001",
                "component_label": "Bearing assembly",
                "relation_to_asset": "primary",
            },
            {
                "component_id": "COMP-002",
                "component_label": "Sister pump",
                "relation_to_asset": "same_train",
            },
            {
                "component_id": "COMP-003",
                "component_label": "Adjacent valve",
                "relation_to_asset": "adjacent",
            },
            {
                "component_id": "COMP-004",
                "component_label": "Downstream pipe",
                "relation_to_asset": "downstream",
            },
        ]
    if past_events is None:
        past_events = []
    return {
        "subgraph_id": subgraph_id,
        "event_id": "EVT-001",
        "asset_id": "PUMP-01",
        "components": components,
        "past_events": past_events,
        "failure_modes": [],
    }


def _make_cr(
    cr_id="CR-1001",
    status="open",
    priority="2",
    short_description="Bearing vibration high",
    long_text="Vibration sensor exceeded 3σ threshold on bearing assembly.",
    functional_location="PLANT/SYS/PUMP-01/BEARING",
    created_date="2026-01-05T08:00:00+00:00",
    is_sister_equipment=False,
):
    return {
        "cr_id": cr_id,
        "cr_type": "CAL",
        "status": status,
        "priority": priority,
        "short_description": short_description,
        "long_text": long_text,
        "functional_location": functional_location,
        "equipment_id": None,
        "created_date": created_date,
        "closed_date": None,
        "is_sister_equipment": is_sister_equipment,
    }


def _make_wo(
    wo_id="WO-2001",
    status="open",
    short_description="Inspect bearing",
    long_text="Scheduled bearing inspection following vibration alert.",
    created_date="2026-01-06T09:00:00+00:00",
    is_sister_equipment=False,
):
    return {
        "wo_id": wo_id,
        "wo_type": "PM",
        "status": status,
        "priority": "3",
        "short_description": short_description,
        "long_text": long_text,
        "functional_location": "PLANT/SYS/PUMP-01",
        "equipment_id": None,
        "created_date": created_date,
        "closed_date": None,
        "is_sister_equipment": is_sister_equipment,
    }


# ---------------------------------------------------------------------------
# NoOpCMMSAdapter
# ---------------------------------------------------------------------------

class TestNoOpCMMSAdapter:

    def test_returns_empty_records(self):
        adapter = NoOpCMMSAdapter()
        result = adapter.fetch("PUMP-01", [], "2025-01-01", "2026-01-01", {})
        assert result["cr_records"] == []
        assert result["wo_records"] == []

    def test_returns_dict_with_required_keys(self):
        result = NoOpCMMSAdapter().fetch("PUMP-01", [], "2025-01-01", "2026-01-01", {})
        assert "cr_records" in result
        assert "wo_records" in result


# ---------------------------------------------------------------------------
# MockCMMSAdapter
# ---------------------------------------------------------------------------

class TestMockCMMSAdapter:

    def test_returns_configured_records(self):
        cr = _make_cr()
        wo = _make_wo()
        adapter = MockCMMSAdapter(cr_records=[cr], wo_records=[wo])
        result = adapter.fetch("PUMP-01", [], "2025-01-01", "2026-01-01", {})
        assert len(result["cr_records"]) == 1
        assert len(result["wo_records"]) == 1

    def test_empty_by_default(self):
        adapter = MockCMMSAdapter()
        result = adapter.fetch("PUMP-01", [], "2025-01-01", "2026-01-01", {})
        assert result["cr_records"] == []
        assert result["wo_records"] == []

    def test_records_are_copies(self):
        cr = _make_cr()
        adapter = MockCMMSAdapter(cr_records=[cr])
        result1 = adapter.fetch("PUMP-01", [], "2025-01-01", "2026-01-01", {})
        result2 = adapter.fetch("PUMP-01", [], "2025-01-01", "2026-01-01", {})
        assert result1["cr_records"] is not result2["cr_records"]


# ---------------------------------------------------------------------------
# CMMSContextBuilderConfig
# ---------------------------------------------------------------------------

class TestCMMSContextBuilderConfig:

    def test_defaults(self):
        cfg = CMMSContextBuilderConfig()
        assert cfg.fallback_lookback_days == 90
        assert "same_train" in cfg.sister_relation_types
        assert "adjacent" in cfg.sister_relation_types
        assert cfg.include_sister_equipment is True

    def test_custom_fallback(self):
        cfg = CMMSContextBuilderConfig(fallback_lookback_days=60)
        assert cfg.fallback_lookback_days == 60


# ---------------------------------------------------------------------------
# CMMSContextBuilder — package structure
# ---------------------------------------------------------------------------

class TestBuilderPackageStructure:

    def setup_method(self):
        self.adapter = NoOpCMMSAdapter()
        self.builder = CMMSContextBuilder(self.adapter)

    def test_required_top_level_keys(self):
        ctx = self.builder.build(_event(), _kg_context(), run_id="run-001")
        for key in (
            "cmms_context_id", "run_id", "event_id", "asset_id", "generated_at",
            "adapter", "lookback_anchor", "lookback_from", "lookback_to",
            "cr_records", "wo_records", "recurrence_summary", "provenance",
        ):
            assert key in ctx, f"Missing key: {key}"

    def test_cmms_context_id_format(self):
        ctx = self.builder.build(_event(event_id="EVT-42"), _kg_context(), run_id="run-001")
        assert ctx["cmms_context_id"].startswith("CMMSCTX::EVT-42::")

    def test_run_id_preserved(self):
        ctx = self.builder.build(_event(), _kg_context(), run_id="run-XYZ")
        assert ctx["run_id"] == "run-XYZ"

    def test_event_id_preserved(self):
        ctx = self.builder.build(_event(event_id="EVT-007"), _kg_context(), run_id="run-001")
        assert ctx["event_id"] == "EVT-007"

    def test_asset_id_preserved(self):
        ctx = self.builder.build(_event(asset_id="PUMP-99"), _kg_context(), run_id="run-001")
        assert ctx["asset_id"] == "PUMP-99"

    def test_adapter_name(self):
        ctx = self.builder.build(_event(), _kg_context(), run_id="run-001")
        assert ctx["adapter"] == "NoOpCMMSAdapter"

    def test_provenance_generated_by(self):
        ctx = self.builder.build(_event(), _kg_context(), run_id="run-001")
        assert ctx["provenance"]["generated_by"] == "CMMSContextBuilder"

    def test_provenance_kg_context_id(self):
        ctx = self.builder.build(_event(), _kg_context(subgraph_id="SG-XYZ"), run_id="run-001")
        assert ctx["provenance"]["kg_context_id"] == "SG-XYZ"


# ---------------------------------------------------------------------------
# Lookback window resolution
# ---------------------------------------------------------------------------

class TestLookbackResolution:

    def test_fallback_when_no_pm_events(self):
        builder = CMMSContextBuilder(NoOpCMMSAdapter(), CMMSContextBuilderConfig(fallback_lookback_days=90))
        ctx = builder.build(_event(event_time="2026-01-10T12:00:00+00:00"), _kg_context(), run_id="run-001")
        assert ctx["lookback_anchor"] == "event_time_minus_90d"
        # lookback_from should be ~90 days before event
        from_dt = datetime.fromisoformat(ctx["lookback_from"])
        event_dt = datetime.fromisoformat("2026-01-10T12:00:00+00:00")
        delta = event_dt - from_dt
        assert 89 <= delta.days <= 91

    def test_last_pm_anchor_used_when_pm_in_past_events(self):
        # Schema-shaped past_events carry timestamp_start (see kg_context.json);
        # the builder anchors the window on the most recent PM.
        pm_date = "2025-09-15T00:00:00+00:00"
        kg = _kg_context(past_events=[
            {"event_type": "PM", "timestamp_start": pm_date, "description": "Quarterly PM"},
        ])
        ctx = CMMSContextBuilder(NoOpCMMSAdapter()).build(_event(), kg, run_id="run-001")
        assert ctx["lookback_anchor"] == "last_pm"
        assert ctx["lookback_from"] == pm_date

    def test_latest_pm_used_when_multiple(self):
        kg = _kg_context(past_events=[
            {"event_type": "preventive_maintenance", "timestamp_start": "2025-06-01T00:00:00+00:00"},
            {"event_type": "PM", "timestamp_start": "2025-11-01T00:00:00+00:00"},
            {"event_type": "PM", "timestamp_start": "2025-08-15T00:00:00+00:00"},
        ])
        ctx = CMMSContextBuilder(NoOpCMMSAdapter()).build(_event(), kg, run_id="run-001")
        assert "2025-11-01" in ctx["lookback_from"]

    def test_non_pm_past_events_ignored(self):
        kg = _kg_context(past_events=[
            {"event_type": "corrective", "timestamp_start": "2025-01-01T00:00:00+00:00"},
        ])
        ctx = CMMSContextBuilder(NoOpCMMSAdapter()).build(_event(), kg, run_id="run-001")
        assert ctx["lookback_anchor"] == "event_time_minus_90d"

    def test_lookback_to_is_event_time(self):
        ctx = CMMSContextBuilder(NoOpCMMSAdapter()).build(
            _event(event_time="2026-01-10T12:00:00+00:00"), _kg_context(), run_id="run-001"
        )
        assert "2026-01-10" in ctx["lookback_to"]


# ---------------------------------------------------------------------------
# Sister component resolution
# ---------------------------------------------------------------------------

class TestSisterComponentResolution:

    def test_same_train_and_adjacent_included(self):
        ctx = CMMSContextBuilder(NoOpCMMSAdapter()).build(_event(), _kg_context(), run_id="run-001")
        sister_ids = ctx["sister_component_ids"]
        assert "COMP-002" in sister_ids  # same_train
        assert "COMP-003" in sister_ids  # adjacent

    def test_primary_and_downstream_excluded(self):
        ctx = CMMSContextBuilder(NoOpCMMSAdapter()).build(_event(), _kg_context(), run_id="run-001")
        sister_ids = ctx["sister_component_ids"]
        assert "COMP-001" not in sister_ids  # primary
        assert "COMP-004" not in sister_ids  # downstream

    def test_include_sister_false(self):
        cfg = CMMSContextBuilderConfig(include_sister_equipment=False)
        ctx = CMMSContextBuilder(NoOpCMMSAdapter(), cfg).build(_event(), _kg_context(), run_id="run-001")
        assert ctx["sister_component_ids"] == []

    def test_custom_sister_relation_types(self):
        cfg = CMMSContextBuilderConfig(sister_relation_types=["downstream"])
        ctx = CMMSContextBuilder(NoOpCMMSAdapter(), cfg).build(_event(), _kg_context(), run_id="run-001")
        sister_ids = ctx["sister_component_ids"]
        assert "COMP-004" in sister_ids      # downstream
        assert "COMP-002" not in sister_ids  # same_train — excluded


# ---------------------------------------------------------------------------
# Record enrichment
# ---------------------------------------------------------------------------

class TestRecordEnrichment:

    def test_days_before_event_computed(self):
        cr = _make_cr(created_date="2026-01-05T08:00:00+00:00")
        adapter = MockCMMSAdapter(cr_records=[cr])
        ctx = CMMSContextBuilder(adapter).build(_event(event_time="2026-01-10T12:00:00+00:00"), _kg_context(), run_id="run-001")
        assert ctx["cr_records"][0]["days_before_event"] == 5

    def test_status_normalised_open_codes(self):
        for raw_status in ("WAPPR", "INPRG", "APPR", "open"):
            cr = _make_cr(status=raw_status)
            adapter = MockCMMSAdapter(cr_records=[cr])
            ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-001")
            assert ctx["cr_records"][0]["status"] == "open", f"Failed for status: {raw_status}"

    def test_status_normalised_closed_codes(self):
        for raw_status in ("COMP", "CLOSE", "closed"):
            cr = _make_cr(status=raw_status)
            adapter = MockCMMSAdapter(cr_records=[cr])
            ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-001")
            assert ctx["cr_records"][0]["status"] == "closed", f"Failed for status: {raw_status}"

    def test_status_normalised_cancelled(self):
        cr = _make_cr(status="CAN")
        adapter = MockCMMSAdapter(cr_records=[cr])
        ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-001")
        assert ctx["cr_records"][0]["status"] == "cancelled"

    def test_is_sister_equipment_default_false(self):
        cr = {
            "cr_id": "CR-999", "status": "open",
            "short_description": "test", "created_date": "2026-01-01T00:00:00+00:00",
        }
        adapter = MockCMMSAdapter(cr_records=[cr])
        ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-001")
        assert ctx["cr_records"][0]["is_sister_equipment"] is False


# ---------------------------------------------------------------------------
# Recurrence summary
# ---------------------------------------------------------------------------

class TestRecurrenceSummary:

    def test_counts_primary_vs_sister(self):
        crs = [
            _make_cr("CR-1", is_sister_equipment=False),
            _make_cr("CR-2", is_sister_equipment=False),
            _make_cr("CR-3", is_sister_equipment=True),
        ]
        ctx = CMMSContextBuilder(MockCMMSAdapter(cr_records=crs)).build(_event(), _kg_context(), run_id="run-001")
        assert ctx["recurrence_summary"]["cr_count_primary"] == 2
        assert ctx["recurrence_summary"]["cr_count_sister"] == 1

    def test_open_wo_count(self):
        wos = [
            _make_wo("WO-1", status="open"),
            _make_wo("WO-2", status="closed"),
            _make_wo("WO-3", status="open"),
        ]
        ctx = CMMSContextBuilder(MockCMMSAdapter(wo_records=wos)).build(_event(), _kg_context(), run_id="run-001")
        assert ctx["recurrence_summary"]["open_wo_count"] == 2

    def test_open_cr_count(self):
        crs = [
            _make_cr("CR-1", status="open"),
            _make_cr("CR-2", status="closed"),
        ]
        ctx = CMMSContextBuilder(MockCMMSAdapter(cr_records=crs)).build(_event(), _kg_context(), run_id="run-001")
        assert ctx["recurrence_summary"]["open_cr_count"] == 1

    def test_earliest_and_most_recent_dates(self):
        crs = [
            _make_cr("CR-1", created_date="2025-11-01T00:00:00+00:00"),
            _make_cr("CR-2", created_date="2026-01-05T00:00:00+00:00"),
            _make_cr("CR-3", created_date="2025-12-15T00:00:00+00:00"),
        ]
        ctx = CMMSContextBuilder(MockCMMSAdapter(cr_records=crs)).build(_event(), _kg_context(), run_id="run-001")
        s = ctx["recurrence_summary"]
        assert "2025-11-01" in s["earliest_related_cr_date"]
        assert "2026-01-05" in s["most_recent_cr_date"]

    def test_empty_records_produce_zero_counts(self):
        ctx = CMMSContextBuilder(NoOpCMMSAdapter()).build(_event(), _kg_context(), run_id="run-001")
        s = ctx["recurrence_summary"]
        assert s["cr_count_primary"] == 0
        assert s["cr_count_sister"] == 0
        assert s["open_wo_count"] == 0
        assert s["earliest_related_cr_date"] is None


# ---------------------------------------------------------------------------
# Chroma document extraction
# ---------------------------------------------------------------------------

class TestGetChromaDocuments:

    def test_long_text_becomes_document_text(self):
        cr = _make_cr(long_text="Vibration exceeded 3σ on bearing assembly.")
        ctx = CMMSContextBuilder(MockCMMSAdapter(cr_records=[cr])).build(_event(), _kg_context(), run_id="run-001")
        builder = CMMSContextBuilder(MockCMMSAdapter(cr_records=[cr]))
        docs = builder.get_chroma_documents(ctx)
        texts = [d["text"] for d in docs]
        assert any("Vibration exceeded 3σ" in t for t in texts)

    def test_metadata_contains_run_id(self):
        cr = _make_cr(long_text="Some narrative.")
        adapter = MockCMMSAdapter(cr_records=[cr])
        ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-MYRUN")
        docs = CMMSContextBuilder(adapter).get_chroma_documents(ctx)
        assert all(d["metadata"]["run_id"] == "run-MYRUN" for d in docs)

    def test_metadata_source_is_cmms_live(self):
        cr = _make_cr(long_text="narrative")
        adapter = MockCMMSAdapter(cr_records=[cr])
        ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-001")
        docs = CMMSContextBuilder(adapter).get_chroma_documents(ctx)
        assert all(d["metadata"]["source"] == "cmms_live" for d in docs)

    def test_empty_long_text_falls_back_to_short_description(self):
        # long_text="" → builder falls back to short_description
        cr = _make_cr(long_text="", short_description="Short desc fallback")
        adapter = MockCMMSAdapter(cr_records=[cr])
        ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-001")
        docs = CMMSContextBuilder(adapter).get_chroma_documents(ctx)
        assert len(docs) == 1
        assert "Short desc fallback" in docs[0]["text"]

    def test_record_with_no_text_skipped(self):
        # Both long_text and short_description empty → doc skipped
        cr = _make_cr(long_text="", short_description="")
        cr2 = _make_cr("CR-2", long_text="Valid narrative.")
        adapter = MockCMMSAdapter(cr_records=[cr, cr2])
        ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-001")
        docs = CMMSContextBuilder(adapter).get_chroma_documents(ctx)
        assert len(docs) == 1
        assert "Valid narrative" in docs[0]["text"]

    def test_wo_records_also_extracted(self):
        wo = _make_wo(long_text="Inspection found bearing wear beyond tolerance.")
        adapter = MockCMMSAdapter(wo_records=[wo])
        ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-001")
        docs = CMMSContextBuilder(adapter).get_chroma_documents(ctx)
        assert any(d["metadata"]["record_type"] == "wo" for d in docs)

    def test_is_sister_in_metadata(self):
        cr = _make_cr(long_text="Sister pump narrative.", is_sister_equipment=True)
        adapter = MockCMMSAdapter(cr_records=[cr])
        ctx = CMMSContextBuilder(adapter).build(_event(), _kg_context(), run_id="run-001")
        docs = CMMSContextBuilder(adapter).get_chroma_documents(ctx)
        assert docs[0]["metadata"]["is_sister_equipment"] is True

    def test_no_documents_when_no_narratives(self):
        ctx = CMMSContextBuilder(NoOpCMMSAdapter()).build(_event(), _kg_context(), run_id="run-001")
        docs = CMMSContextBuilder(NoOpCMMSAdapter()).get_chroma_documents(ctx)
        assert docs == []

    def test_path_a_metadata_includes_doc_identity_and_asset(self):
        cr = _make_cr(cr_id="CR-42", long_text="Structured CR narrative.")
        adapter = MockCMMSAdapter(cr_records=[cr])
        ctx = CMMSContextBuilder(adapter).build(_event(asset_id="PUMP-77"), _kg_context(), run_id="run-001")
        docs = CMMSContextBuilder(adapter).get_chroma_documents(ctx)
        assert docs
        meta = docs[0]["metadata"]
        assert meta["ingestion_path"] == "path_a_structured"
        assert meta["doc_type"] == "CR"
        assert meta["doc_id"] == "CMMS::CR::CR-42"
        assert meta["asset_id"] == "PUMP-77"

    def test_path_a_structured_fields_are_flattened(self):
        # build() projects records onto the cmms_context schema whitelist, so
        # Path-A extras (condition_assessment / failure_mode_refs /
        # extracted_causal_statements) are dropped from the default build() flow
        # — emitting them on the Chroma path is deferred to the injection MR.
        # The flattening logic itself still applies to any record that reaches
        # get_chroma_documents un-projected, so exercise it on a hand-built
        # context that bypasses the schema projection.
        cr = _make_cr(cr_id="CR-77", long_text="Pump degraded due to lube starvation.")
        cr["condition_assessment"] = {
            "as_found_condition": "DEGRADED",
            "as_left_condition": "ACCEPTABLE",
        }
        cr["failure_mode_refs"] = [{"fm_id": "FM::LUBE-LOSS"}]
        cr["extracted_causal_statements"] = [
            {"cause_text": "loss of lubrication", "connector": "caused", "effect_text": "bearing wear"}
        ]
        ctx = {
            "run_id": "run-001",
            "event_id": "EVT-001",
            "asset_id": "PUMP-01",
            "cr_records": [cr],
            "wo_records": [],
        }
        docs = CMMSContextBuilder(NoOpCMMSAdapter()).get_chroma_documents(ctx)
        meta = docs[0]["metadata"]
        assert meta["ca_as_found_condition"] == "degraded"
        assert meta["ca_as_left_condition"] == "acceptable"
        # A2: list/dict metadata is JSON-encoded for Chroma; the flattened
        # ``*_text`` form is the stable string surface for retrieval.
        assert "FM::LUBE-LOSS" in meta["failure_mode_refs_text"]
        assert "loss of lubrication caused bearing wear" in meta["causal_statements_text"]


# ---------------------------------------------------------------------------
# Record cap
# ---------------------------------------------------------------------------

class TestRecordCap:

    def test_cr_records_capped(self):
        crs = [_make_cr(cr_id=f"CR-{i}", created_date=f"2026-01-0{i+1}T00:00:00+00:00") for i in range(5)]
        cfg = CMMSContextBuilderConfig(max_cr_records=3)
        ctx = CMMSContextBuilder(MockCMMSAdapter(cr_records=crs), cfg).build(_event(), _kg_context(), run_id="run-001")
        assert len(ctx["cr_records"]) == 3

    def test_most_recent_records_kept(self):
        crs = [
            _make_cr("CR-OLD", created_date="2025-06-01T00:00:00+00:00"),
            _make_cr("CR-NEW", created_date="2026-01-08T00:00:00+00:00"),
        ]
        cfg = CMMSContextBuilderConfig(max_cr_records=1)
        ctx = CMMSContextBuilder(MockCMMSAdapter(cr_records=crs), cfg).build(_event(), _kg_context(), run_id="run-001")
        assert ctx["cr_records"][0]["cr_id"] == "CR-NEW"

    def test_no_cap_when_zero(self):
        crs = [_make_cr(cr_id=f"CR-{i}") for i in range(10)]
        cfg = CMMSContextBuilderConfig(max_cr_records=0)
        ctx = CMMSContextBuilder(MockCMMSAdapter(cr_records=crs), cfg).build(_event(), _kg_context(), run_id="run-001")
        assert len(ctx["cr_records"]) == 10


# ===========================================================================
# MR #53 review-fix lock-down suite
# ---------------------------------------------------------------------------
# These classes pin the corrected builder behaviour from the MR #53 review:
#   * last_pm lookback anchor fires for a schema-shaped kg_context, is scoped to
#     the primary asset, and ignores PMs dated after the event (I1/N3/A1);
#   * the fallback anchor label is honest when fallback_lookback_days != 90 (I5);
#   * component_id is resolved from functional_location / equipment_id via the
#     KG (I3);
#   * enriched records are projected onto the cmms_context schema whitelist (I2);
#   * caps and the recurrence summary order by parsed datetime, mixed offsets (I4);
#   * get_chroma_documents emits Chroma-clean metadata, no None / no lists (A2);
#   * days_before_event rounds toward zero (N4).
# Fixtures are ``_rf_``-prefixed so they do not shadow the baseline fixtures.
# ===========================================================================

_RF_EVENT_TS = "2026-03-01T00:00:00+00:00"


def _rf_event() -> dict:
    return {"event_id": "E1", "asset_id": "ASSET-1", "timestamp_start": _RF_EVENT_TS}


def _rf_kg_context(past_events=None) -> dict:
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


def _rf_build(cr_records=None, wo_records=None, past_events=None, config=None) -> dict:
    adapter = MockCMMSAdapter(cr_records=cr_records or [], wo_records=wo_records or [])
    builder = CMMSContextBuilder(adapter, config=config)
    return builder.build(_rf_event(), _rf_kg_context(past_events), run_id="RUN-1")


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
        ctx = _rf_build(past_events=past)
        assert ctx["lookback_anchor"] == "last_pm"
        assert ctx["lookback_from"] == "2026-02-01T00:00:00+00:00"

    def test_pm_on_sister_asset_is_ignored(self):
        """past_events on another asset must not anchor the primary window (N3)."""
        past = [
            {"event_id": "PM_SIS", "asset_id": "ASSET-2", "event_type": "PM",
             "timestamp_start": "2026-02-20T00:00:00+00:00"},
        ]
        ctx = _rf_build(past_events=past)
        # No primary-asset PM → fallback, not the sister's 2026-02-20 date.
        assert ctx["lookback_anchor"] == "event_time_minus_90d"

    def test_pm_after_event_is_ignored(self):
        """A PM dated after the event cannot anchor a backward window (A1)."""
        past = [
            {"event_id": "PM_FUT", "asset_id": "ASSET-1", "event_type": "PM",
             "timestamp_start": "2026-05-01T00:00:00+00:00"},
        ]
        ctx = _rf_build(past_events=past)
        assert ctx["lookback_anchor"] == "event_time_minus_90d"
        # Window is never inverted.
        assert ctx["lookback_from"] <= ctx["lookback_to"]


# ---------------------------------------------------------------------------
# Fallback anchor label honesty (I5)
# ---------------------------------------------------------------------------

class TestFallbackAnchorLabel:
    def test_default_90_day_label(self):
        ctx = _rf_build()  # no PMs, default fallback of 90
        assert ctx["lookback_anchor"] == "event_time_minus_90d"

    def test_non_90_day_reports_custom(self):
        cfg = CMMSContextBuilderConfig(fallback_lookback_days=30)
        ctx = _rf_build(config=cfg)
        assert ctx["lookback_anchor"] == "custom"
        # 30 days before the event, and the real count is preserved in provenance.
        expected = (datetime.fromisoformat(_RF_EVENT_TS) - timedelta(days=30)).isoformat()
        assert ctx["lookback_from"] == expected
        assert ctx["provenance"]["query_params"]["fallback_lookback_days"] == 30


# ---------------------------------------------------------------------------
# component_id resolution (I3)
# ---------------------------------------------------------------------------

class TestComponentIdResolution:
    def test_resolved_from_functional_location(self):
        cr = {"cr_id": "CR1", "status": "OPEN", "short_description": "seal leak",
              "created_date": "2026-02-15T00:00:00+00:00", "functional_location": "PLANT/PUMP-01"}
        ctx = _rf_build(cr_records=[cr])
        rec = ctx["cr_records"][0]
        assert rec["component_id"] == "C-PRIMARY"
        assert rec["is_sister_equipment"] is False

    def test_resolved_from_equipment_id_and_marked_sister(self):
        cr = {"cr_id": "CR2", "status": "COMP", "short_description": "bearing",
              "created_date": "2026-02-10T00:00:00+00:00", "equipment_id": "EQ-SISTER"}
        ctx = _rf_build(cr_records=[cr])
        rec = ctx["cr_records"][0]
        assert rec["component_id"] == "C-SISTER"
        assert rec["is_sister_equipment"] is True

    def test_adapter_supplied_component_id_wins(self):
        cr = {"cr_id": "CR3", "status": "OPEN", "short_description": "x",
              "created_date": "2026-02-10T00:00:00+00:00",
              "functional_location": "PLANT/PUMP-01", "component_id": "C-EXPLICIT"}
        ctx = _rf_build(cr_records=[cr])
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
        rec = _rf_build(cr_records=[cr])["cr_records"][0]
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
        ctx = _rf_build(cr_records=[cr_a, cr_b], config=cfg)
        assert [r["cr_id"] for r in ctx["cr_records"]] == ["B"]

    def test_recurrence_summary_earliest_and_most_recent(self):
        cr_a = {"cr_id": "A", "status": "OPEN", "short_description": "a",
                "created_date": "2026-02-15T00:00:00+05:00"}   # 2026-02-14T19:00Z
        cr_b = {"cr_id": "B", "status": "OPEN", "short_description": "b",
                "created_date": "2026-02-14T20:00:00+00:00"}   # later
        summ = _rf_build(cr_records=[cr_a, cr_b])["recurrence_summary"]
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
        ctx = builder.build(_rf_event(), _rf_kg_context(), run_id="RUN-1")
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
        ctx = builder.build(_rf_event(), _rf_kg_context(), run_id="RUN-1")
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
        rec = _rf_build(cr_records=[cr])["cr_records"][0]
        assert rec["days_before_event"] == 0

    def test_days_before_event_positive(self):
        cr = {"cr_id": "CR1", "status": "OPEN", "short_description": "x",
              "created_date": "2026-02-27T00:00:00+00:00"}   # 2 days before
        rec = _rf_build(cr_records=[cr])["cr_records"][0]
        assert rec["days_before_event"] == 2
