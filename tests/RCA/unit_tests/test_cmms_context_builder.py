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

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import dackar
from dackar.RCA.cmms_integration.cmms_adapter import (
    MockCMMSAdapter,
    normalize_cmms_status,
)
from dackar.RCA.cmms_integration.cmms_context_builder import (
    _CR_SCHEMA_KEYS,
    CMMSContextBuilder,
    CMMSContextBuilderConfig,
)
from dackar.RCA.cmms_integration.maximo_cmms_adapter import MaximoCMMSAdapter
from dackar.RCA.cmms_integration.sap_pm_cmms_adapter import SAPPMCMMSAdapter


# ---------------------------------------------------------------------------
# Schema validation harness (jsonschema is a declared dependency; skip the
# schema-validating tests cleanly where it — or its date-time format validator
# — is unavailable, so the rest of the suite still runs).
# ---------------------------------------------------------------------------

try:
    from jsonschema import Draft7Validator, FormatChecker
    _HAVE_JSONSCHEMA = True
except ImportError:  # pragma: no cover - environment without the optional dep
    _HAVE_JSONSCHEMA = False

requires_jsonschema = pytest.mark.skipif(
    not _HAVE_JSONSCHEMA, reason="jsonschema not installed"
)

_SCHEMA_PATH = Path(dackar.__file__).parent / "RCA" / "schemas" / "cmms_context.json"


def _validate_artifact(ctx: dict) -> None:
    """Validate a built cmms_context against schemas/cmms_context.json with
    date-time format checking.  Raises jsonschema.ValidationError on any
    required-field, type, enum, or additionalProperties violation."""
    schema = json.loads(_SCHEMA_PATH.read_text())
    Draft7Validator(schema, format_checker=FormatChecker()).validate(ctx)


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

    def test_equipment_failure_is_not_treated_as_pm(self):
        """'equipment_failure' contains the letters 'pm' but is not a PM — a
        substring test would wrongly truncate the lookback to the failure date."""
        past = [
            {"event_id": "EF", "asset_id": "ASSET-1", "event_type": "equipment_failure",
             "timestamp_start": "2026-02-20T00:00:00+00:00"},
        ]
        ctx = _build(past_events=past)
        assert ctx["lookback_anchor"] == "event_time_minus_90d"
        assert ctx["lookback_from"] != "2026-02-20T00:00:00+00:00"


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


# ---------------------------------------------------------------------------
# Status normalization: every documented Maximo/SAP code + unknown (blocking)
# ---------------------------------------------------------------------------

# (raw CMMS code, expected cmms_context enum value)
_STATUS_CASES = [
    # Maximo
    ("WAPPR", "open"), ("WMATL", "open"), ("WPCOND", "open"),
    ("INPRG", "open"), ("APPR", "open"),
    ("COMP", "closed"), ("CLOSE", "closed"), ("CAN", "cancelled"),
    # SAP PM
    ("OSNO", "open"), ("OSMA", "open"), ("OSTS", "open"), ("NOCO", "open"),
    ("CLSD", "closed"), ("TECO", "closed"), ("DLFL", "cancelled"),
    # already-normalized + adversarial
    ("open", "open"), ("closed", "closed"), ("cancelled", "cancelled"),
    ("WeirdCode", "unknown"), ("", "unknown"),
]


class TestStatusNormalization:
    @pytest.mark.parametrize("raw,expected", _STATUS_CASES)
    def test_shared_normalizer(self, raw, expected):
        assert normalize_cmms_status(raw) == expected

    @pytest.mark.parametrize("raw,expected", _STATUS_CASES)
    def test_maximo_adapter_delegates(self, raw, expected):
        assert MaximoCMMSAdapter._map_status(raw) == expected

    @pytest.mark.parametrize("raw,expected", _STATUS_CASES)
    def test_sap_adapter_delegates(self, raw, expected):
        assert SAPPMCMMSAdapter._map_status(raw) == expected

    @pytest.mark.parametrize("raw,expected", _STATUS_CASES)
    def test_builder_projects_status_to_enum(self, raw, expected):
        cr = {"cr_id": "CR1", "status": raw, "short_description": "x",
              "created_date": "2026-02-15T00:00:00+00:00"}
        rec = _build(cr_records=[cr])["cr_records"][0]
        assert rec["status"] == expected
        assert rec["status"] in {"open", "closed", "cancelled", "unknown"}


# ---------------------------------------------------------------------------
# Adapter-boundary validation: malformed records skipped with provenance (blocking)
# ---------------------------------------------------------------------------

class TestMalformedRecordSkip:
    def test_missing_required_fields_are_skipped(self):
        good = {"cr_id": "CR1", "status": "OPEN", "short_description": "ok",
                "created_date": "2026-02-15T00:00:00+00:00"}
        bad_no_id   = {"status": "OPEN", "short_description": "x",
                       "created_date": "2026-02-15T00:00:00+00:00"}
        bad_no_desc = {"cr_id": "CR2", "status": "OPEN",
                       "created_date": "2026-02-15T00:00:00+00:00"}
        bad_no_date = {"cr_id": "CR3", "status": "OPEN", "short_description": "x"}
        bad_date    = {"cr_id": "CR4", "status": "OPEN", "short_description": "x",
                       "created_date": "not-a-date"}
        ctx = _build(cr_records=[good, bad_no_id, bad_no_desc, bad_no_date, bad_date])
        assert [r["cr_id"] for r in ctx["cr_records"]] == ["CR1"]
        assert ctx["provenance"]["dropped_records"]["cr"] == 4

    def test_truthy_string_is_sister_is_coerced_to_bool(self):
        cr = {"cr_id": "CR1", "status": "OPEN", "short_description": "x",
              "created_date": "2026-02-15T00:00:00+00:00",
              "is_sister_equipment": "false"}   # truthy string, must read False
        rec = _build(cr_records=[cr])["cr_records"][0]
        assert rec["is_sister_equipment"] is False

    @requires_jsonschema
    def test_adversarial_batch_yields_valid_artifact(self):
        cr  = {"cr_id": "CR1", "status": "OSNO", "short_description": "x",
               "created_date": "2026-02-15T00:00:00+00:00",
               "is_sister_equipment": "false"}
        bad = {"status": "OPEN"}  # no id / short_description / created_date
        ctx = _build(cr_records=[cr], wo_records=[bad])
        _validate_artifact(ctx)
        assert ctx["provenance"]["dropped_records"]["wo"] == 1


# ---------------------------------------------------------------------------
# Recurrence summary is aggregated from the FULL lists, before capping (important)
# ---------------------------------------------------------------------------

class TestRecurrenceBeforeCap:
    def test_summary_counts_full_not_capped(self):
        crs = [
            {"cr_id": f"CR{i}", "status": "OPEN", "short_description": "x",
             "created_date": f"2026-02-{10 + i:02d}T00:00:00+00:00"}
            for i in range(3)
        ]
        cfg = CMMSContextBuilderConfig(max_cr_records=1)
        ctx = _build(cr_records=crs, config=cfg)
        # Detail array is capped …
        assert len(ctx["cr_records"]) == 1
        # … but the recurrence aggregate reflects all three.
        assert ctx["recurrence_summary"]["cr_count_primary"] == 3
        assert ctx["recurrence_summary"]["open_cr_count"] == 3


# ---------------------------------------------------------------------------
# MockCMMSAdapter.filter_by_asset — both modes (important)
# ---------------------------------------------------------------------------

class TestMockAdapterFilterByAsset:
    def test_default_returns_all_untagged(self):
        cr = {"cr_id": "CR1", "functional_location": "PLANT/PUMP-01"}
        out = MockCMMSAdapter(cr_records=[cr]).fetch(
            "ASSET-1", [], "from", "to", {})
        assert out["cr_records"] == [cr]
        assert "is_sister_equipment" not in out["cr_records"][0]

    def test_filter_tags_primary_vs_sister(self):
        primary = {"cr_id": "CR1", "functional_location": "ASSET-1/PUMP"}
        sister  = {"cr_id": "CR2", "functional_location": "OTHER/PUMP"}
        out = MockCMMSAdapter(
            cr_records=[primary, sister], filter_by_asset=True,
        ).fetch("ASSET-1", [], "from", "to", {})
        tags = {r["cr_id"]: r["is_sister_equipment"] for r in out["cr_records"]}
        assert tags == {"CR1": False, "CR2": True}

    def test_filter_matches_on_equipment_id(self):
        rec = {"cr_id": "CR1", "equipment_id": "ASSET-1-EQ"}
        out = MockCMMSAdapter(
            cr_records=[rec], filter_by_asset=True,
        ).fetch("ASSET-1", [], "from", "to", {})
        assert out["cr_records"][0]["is_sister_equipment"] is False


# ---------------------------------------------------------------------------
# A parseable event timestamp is required at the build() boundary (important)
# ---------------------------------------------------------------------------

class TestEventTimestampRequired:
    def test_missing_timestamp_raises(self):
        builder = CMMSContextBuilder(MockCMMSAdapter())
        with pytest.raises(ValueError, match="timestamp"):
            builder.build(
                {"event_id": "E1", "asset_id": "ASSET-1"},
                _kg_context(), run_id="RUN-1",
            )

    def test_unparseable_timestamp_raises(self):
        builder = CMMSContextBuilder(MockCMMSAdapter())
        with pytest.raises(ValueError):
            builder.build(
                {"event_id": "E1", "asset_id": "ASSET-1", "timestamp_start": "nope"},
                _kg_context(), run_id="RUN-1",
            )


# ---------------------------------------------------------------------------
# Topology-only sisters carry embedding_score 0.0, per schema (nit)
# ---------------------------------------------------------------------------

class TestTopologySisterEmbeddingScore:
    def test_topology_only_sister_score_is_zero(self):
        ctx = _build()  # C-SISTER (same_train) is a topology-only sister
        sisters = {s["component_id"]: s for s in ctx["sister_components"]}
        assert "C-SISTER" in sisters
        assert sisters["C-SISTER"]["match_type"] == "topology"
        assert sisters["C-SISTER"]["embedding_score"] == 0.0


# ---------------------------------------------------------------------------
# Similarity-resolver results are projected onto the sister schema (architecture)
# ---------------------------------------------------------------------------

class _FakeSister:
    """Stands in for an EquipmentSimilarityResolver result — exposes both the
    attribute access the old code used and a to_dict()."""

    def __init__(self, **fields):
        self._fields = dict(fields)
        self.__dict__.update(fields)

    def to_dict(self) -> dict:
        return dict(self._fields)


class _FakeResolver:
    def __init__(self, results):
        self._results = results

    def resolve_similar(self, target_component_ids, kg_context):
        return self._results


class TestSimilarityResultProjection:
    def test_extra_fields_are_projected_out(self):
        res = [_FakeSister(component_id="C-EMB", component_label="Emb",
                           match_type="spec_embedding", shared_fm_count=2,
                           embedding_score=0.3, leak_me="secret")]
        cfg = CMMSContextBuilderConfig(similarity_resolver=_FakeResolver(res))
        emb = {s["component_id"]: s for s in _build(config=cfg)["sister_components"]}["C-EMB"]
        assert set(emb).issubset({
            "component_id", "component_label", "match_type",
            "shared_fm_count", "embedding_score",
        })
        assert "leak_me" not in emb

    def test_result_missing_match_type_is_skipped(self):
        res = [_FakeSister(component_id="C-BAD", embedding_score=0.1)]  # no match_type
        cfg = CMMSContextBuilderConfig(similarity_resolver=_FakeResolver(res))
        ids = {s["component_id"] for s in _build(config=cfg)["sister_components"]}
        assert "C-BAD" not in ids

    @requires_jsonschema
    def test_projected_sisters_validate(self):
        res = [_FakeSister(component_id="C-EMB", match_type="spec_embedding",
                           shared_fm_count=2, embedding_score=0.3, leak_me="x")]
        cfg = CMMSContextBuilderConfig(similarity_resolver=_FakeResolver(res))
        _validate_artifact(_build(config=cfg))
