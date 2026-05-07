"""
test_fm_linkage_pipeline.py — unit tests for the FM linkage gap fixes

Covers:
- Phase 3: _compute_kg_governance() FM link coverage metric and yellow status
- Phase 3: _apply_kg_governance_attention_flags() distinguishing messages
- Phase 1 Risk 1 Tier 1: CMMS-vs-CMMS deduplication in _augment_kg_context_with_cmms_past_events()
- Phase 1 Risk 2: exact_doc_ids set is built correctly for cross-pattern exclusion

Run: pytest test_fm_linkage_pipeline.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_orchestrator(**kwargs):
    return RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
        **kwargs,
    )


def _base_event():
    return {"asset_id": "A1", "event_id": "EVT-1", "timestamp_start": "2025-01-01T00:00:00Z"}


def _kg_context_with_past_events(past_events):
    return {
        "subgraph_id": "KG::A1",
        "failure_modes": [{"fm_id": "FM-1", "name": "Bearing wear"}],
        "past_events": past_events,
    }


def _cr_record(cr_id, days=30):
    return {"cr_id": cr_id, "created_date": "2024-12-01T00:00:00Z",
            "days_before_event": days, "status": "closed", "component_id": "C1"}


def _wo_record(wo_id, days=60):
    return {"wo_id": wo_id, "created_date": "2024-11-01T00:00:00Z",
            "days_before_event": days, "status": "closed", "component_id": "C1"}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — _compute_kg_governance() FM link coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeKGGovernanceFMLinkCoverage:

    def _gov(self, past_events, cfg_extra=None):
        o = _make_orchestrator()
        if cfg_extra:
            o.config.extra = cfg_extra
        return o._compute_kg_governance(
            event=_base_event(),
            kg_context=_kg_context_with_past_events(past_events),
        )

    def test_no_past_events_coverage_zero_no_issue(self):
        gov = self._gov([])
        assert gov["past_event_count"] == 0
        assert gov["past_events_with_fm"] == 0
        assert gov["fm_link_coverage"] == 0.0
        assert gov["fm_link_gap"] is False
        assert not any("failure mode link" in i for i in gov["issues"])

    def test_all_events_have_fm_no_gap(self):
        past = [{"fm_id": "FM-1"}, {"fm_id": "FM-2"}]
        gov = self._gov(past)
        assert gov["past_event_count"] == 2
        assert gov["past_events_with_fm"] == 2
        assert gov["fm_link_coverage"] == 1.0
        assert gov["fm_link_gap"] is False

    def test_all_events_missing_fm_gap_detected(self):
        past = [{"fm_id": None}, {"fm_id": None}, {"fm_id": None}]
        gov = self._gov(past)
        assert gov["past_event_count"] == 3
        assert gov["past_events_with_fm"] == 0
        assert gov["fm_link_coverage"] == 0.0
        assert gov["fm_link_gap"] is True
        assert gov["status"] == "yellow"
        assert any("failure mode link" in i for i in gov["issues"])

    def test_partial_fm_coverage_below_threshold_triggers_gap(self):
        past = [{"fm_id": "FM-1"}, {"fm_id": None}, {"fm_id": None}, {"fm_id": None}]
        gov = self._gov(past)  # 1/4 = 0.25 < 0.5 threshold
        assert gov["fm_link_gap"] is True
        assert gov["status"] == "yellow"

    def test_partial_fm_coverage_above_threshold_no_gap(self):
        past = [{"fm_id": "FM-1"}, {"fm_id": "FM-2"}, {"fm_id": "FM-3"}, {"fm_id": None}]
        gov = self._gov(past)  # 3/4 = 0.75 > 0.5 threshold
        assert gov["fm_link_gap"] is False

    def test_custom_threshold_respected(self):
        past = [{"fm_id": "FM-1"}, {"fm_id": None}]  # 0.5 coverage
        gov_strict = self._gov(past, cfg_extra={"kg_governance_fm_link_coverage_threshold": 0.8})
        gov_loose = self._gov(past, cfg_extra={"kg_governance_fm_link_coverage_threshold": 0.3})
        assert gov_strict["fm_link_gap"] is True
        assert gov_loose["fm_link_gap"] is False

    def test_fm_link_gap_does_not_override_red_status(self):
        """A red status (zero failure modes) must not be downgraded by fm_link_gap."""
        o = _make_orchestrator()
        gov = o._compute_kg_governance(
            event=_base_event(),
            kg_context={
                "failure_modes": [],  # triggers red
                "past_events": [{"fm_id": None}, {"fm_id": None}],
            },
        )
        assert gov["status"] == "red"
        assert gov["fm_link_gap"] is True

    def test_output_contains_new_fields(self):
        gov = self._gov([{"fm_id": "FM-1"}])
        for field in ("past_event_count", "past_events_with_fm", "fm_link_coverage", "fm_link_gap"):
            assert field in gov, f"Missing field: {field}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — _apply_kg_governance_attention_flags() distinguishing messages
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyKGGovernanceAttentionFlags:

    def _apply(self, gov):
        rca_card = {"executive_summary": {}}
        RCAReasoningOrchestrator._apply_kg_governance_attention_flags(rca_card, gov)
        return rca_card["executive_summary"].get("analyst_attention_flags", [])

    def test_no_events_emits_no_prior_events_message(self):
        gov = {"status": "green", "issues": [], "past_event_count": 0,
               "past_events_with_fm": 0, "fm_link_gap": False}
        flags = self._apply(gov)
        assert any("No prior KG events" in f for f in flags)

    def test_fm_link_gap_emits_partial_recurrence_message(self):
        gov = {"status": "yellow", "issues": ["2 of 3 past KG event(s) carry no failure mode link — recurrence detection is partial."],
               "past_event_count": 3, "past_events_with_fm": 1, "fm_link_gap": True}
        flags = self._apply(gov)
        assert any("carry no failure mode link" in f for f in flags)
        assert any("recurrence detection is partial" in f for f in flags)
        # Must NOT emit generic "KG governance warning:" for the fm_link_gap issue
        assert not any("KG governance warning:" in f and "failure mode link" in f for f in flags)

    def test_green_status_no_events_still_emits_no_prior_events_message(self):
        """The no-events message is informational — emitted even when status is green."""
        gov = {"status": "green", "issues": [], "past_event_count": 0,
               "past_events_with_fm": 0, "fm_link_gap": False}
        flags = self._apply(gov)
        assert any("No prior KG events" in f for f in flags)

    def test_other_governance_issues_still_emit_standard_prefix(self):
        gov = {"status": "yellow",
               "issues": ["KG failure mode count 0 is below minimum 1."],
               "past_event_count": 2, "past_events_with_fm": 2, "fm_link_gap": False}
        flags = self._apply(gov)
        assert any("KG governance warning:" in f for f in flags)

    def test_events_with_fm_links_no_gap_no_partial_message(self):
        gov = {"status": "green", "issues": [], "past_event_count": 3,
               "past_events_with_fm": 3, "fm_link_gap": False}
        flags = self._apply(gov)
        assert not any("failure mode link" in f for f in flags)
        assert not any("No prior KG events" in f for f in flags)

    def test_flags_are_deduplicated(self):
        gov = {"status": "yellow",
               "issues": ["2 of 3 past KG event(s) carry no failure mode link — recurrence detection is partial."],
               "past_event_count": 3, "past_events_with_fm": 1, "fm_link_gap": True}
        rca_card = {"executive_summary": {}}
        RCAReasoningOrchestrator._apply_kg_governance_attention_flags(rca_card, gov)
        RCAReasoningOrchestrator._apply_kg_governance_attention_flags(rca_card, gov)
        flags = rca_card["executive_summary"]["analyst_attention_flags"]
        partial_msgs = [f for f in flags if "failure mode link" in f]
        assert len(partial_msgs) == 1, "Flag must not be duplicated on repeated calls"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 Risk 1 Tier 1 — CMMS-vs-CMMS deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestCMMSCMMSDeduplication:

    def _augment(self, existing_past_events, cr_records=None, wo_records=None):
        o = _make_orchestrator()
        kg_context = _kg_context_with_past_events(existing_past_events)
        cmms_context = {
            "cr_records": cr_records or [],
            "wo_records": wo_records or [],
        }
        result = o._augment_kg_context_with_cmms_past_events(
            kg_context=kg_context,
            cmms_context=cmms_context,
            event=_base_event(),
        )
        return result.get("past_events", [])

    def test_same_cr_not_injected_twice(self):
        existing = [{"event_id": "CMMS::CR::CR-100", "fm_id": None, "asset_id": "A1"}]
        cr_records = [_cr_record("CR-100")]  # same doc id already present
        past = self._augment(existing, cr_records=cr_records)
        ids = [pe["event_id"] for pe in past]
        assert ids.count("CMMS::CR::CR-100") == 1, "Duplicate CMMS CR must be suppressed"

    def test_same_wo_not_injected_twice(self):
        existing = [{"event_id": "CMMS::WO::WO-200", "fm_id": None, "asset_id": "A1"}]
        wo_records = [_wo_record("WO-200")]
        past = self._augment(existing, wo_records=wo_records)
        ids = [pe["event_id"] for pe in past]
        assert ids.count("CMMS::WO::WO-200") == 1

    def test_different_cr_injected_normally(self):
        existing = [{"event_id": "CMMS::CR::CR-100", "fm_id": None, "asset_id": "A1"}]
        cr_records = [_cr_record("CR-999")]  # different id — should be injected
        past = self._augment(existing, cr_records=cr_records)
        ids = [pe["event_id"] for pe in past]
        assert "CMMS::CR::CR-999" in ids

    def test_kg_native_event_not_affected_by_cmms_dedup(self):
        """KG-native events (no CMMS:: prefix) must never be suppressed by the Tier 1 guard."""
        existing = [{"event_id": "EVT-456", "fm_id": "FM-1", "asset_id": "A1"}]
        cr_records = [_cr_record("CR-456")]  # doc id "CR-456" vs event_id "EVT-456" — no collision
        past = self._augment(existing, cr_records=cr_records)
        ids = [pe["event_id"] for pe in past]
        assert "EVT-456" in ids
        assert "CMMS::CR::CR-456" in ids

    def test_cr_and_wo_with_same_id_only_first_injected(self):
        """If a CR and WO somehow share the same underlying doc id, only the first wins."""
        cr_records = [_cr_record("DOC-1")]
        wo_records = [_wo_record("DOC-1")]
        past = self._augment([], cr_records=cr_records, wo_records=wo_records)
        doc_ids = [
            RCAReasoningOrchestrator._source_doc_id_from_event_id(pe["event_id"])
            for pe in past
        ]
        assert doc_ids.count("DOC-1") == 1, "Same doc id from CR and WO must not inject twice"

    def test_empty_existing_and_unique_records_all_injected(self):
        cr_records = [_cr_record("CR-1"), _cr_record("CR-2")]
        past = self._augment([], cr_records=cr_records)
        ids = {pe["event_id"] for pe in past}
        assert "CMMS::CR::CR-1" in ids
        assert "CMMS::CR::CR-2" in ids


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 Risk 2 — exact_doc_ids exclusion set construction
# ─────────────────────────────────────────────────────────────────────────────

class TestExactDocIdsExclusionSet:

    def test_exclusion_set_contains_cmms_event_doc_ids(self):
        """past_events with CMMS event_ids must contribute to the exact_doc_ids exclusion set."""
        past_events = [
            {"event_id": "CMMS::CR::CR-10", "fm_id": None},
            {"event_id": "CMMS::WO::WO-20", "fm_id": None},
            {"event_id": "EVT-99", "fm_id": "FM-1"},  # KG-native — no doc id
        ]
        kg_context = _kg_context_with_past_events(past_events)

        expected_ids = {"CR-10", "WO-20"}
        actual_ids = {
            doc_id
            for pe in past_events
            for doc_id in [RCAReasoningOrchestrator._source_doc_id_from_event_id(
                str(pe.get("event_id") or "")
            )]
            if doc_id
        }
        assert actual_ids == expected_ids

    def test_cross_pattern_query_receives_exact_doc_ids(self):
        """_build_cross_pattern_evidence must pass exact_doc_ids to DocExtractionStore.query()."""
        store = MagicMock()
        store.query.return_value = ([], [])

        o = _make_orchestrator()
        o.doc_extraction_store = store
        o.cross_pattern_linker = MagicMock()
        o.cross_pattern_linker.run.return_value = {}

        kg_context = _kg_context_with_past_events([
            {"event_id": "CMMS::CR::CR-42", "fm_id": None},
        ])
        hist_eps = {"episodes": []}

        o._build_cross_pattern_evidence(
            historical_signal_episodes=hist_eps,
            causality_candidates={"candidates": []},
            event=_base_event(),
            kg_context=kg_context,
        )

        call_kwargs = store.query.call_args
        passed_exact = call_kwargs.kwargs.get("exact_doc_ids") or (
            call_kwargs.args[4] if len(call_kwargs.args) > 4 else None
        )
        assert passed_exact is not None, "exact_doc_ids must be passed to DocExtractionStore.query()"
        assert "CR-42" in passed_exact

    def test_cross_pattern_query_no_cmms_events_passes_none(self):
        """When no CMMS events in past_events, exact_doc_ids should be None (not empty set)."""
        store = MagicMock()
        store.query.return_value = ([], [])

        o = _make_orchestrator()
        o.doc_extraction_store = store
        o.cross_pattern_linker = MagicMock()
        o.cross_pattern_linker.run.return_value = {}

        kg_context = _kg_context_with_past_events([
            {"event_id": "EVT-99", "fm_id": "FM-1"},  # KG-native only
        ])

        o._build_cross_pattern_evidence(
            historical_signal_episodes={"episodes": []},
            causality_candidates={"candidates": []},
            event=_base_event(),
            kg_context=kg_context,
        )

        call_kwargs = store.query.call_args
        passed_exact = call_kwargs.kwargs.get("exact_doc_ids") or (
            call_kwargs.args[4] if len(call_kwargs.args) > 4 else None
        )
        assert passed_exact is None, "exact_doc_ids must be None when no CMMS events present"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
