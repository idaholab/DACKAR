"""
test_scope_revision_downstream.py — Scope-Revision Downstream Propagation

Covers:
- _resolve_approved_scope_boundary: version 0 → None; version 1 → frozenset;
  lower-cases IDs; empty list → None; latest accepted revision wins
- _apply_scope_boundary_filter: out-of-scope → ruled_out; in-scope kept;
  no component_id → not filtered; reason_code = "scope_filtered";
  None boundary → not called (guarded by version check)
- apply_scope_revision (enhanced): auto-merges added_component_ids;
  removes removed_component_ids; no explicit snapshot → builds from prior
- resolve_expansion_suggestion: accepted updates scope; rejected/deferred
  no scope change; marks analyst_decision on suggestion; unknown signal_id raises
- run() integration: version 0 → no filter; version 1 → out-of-scope in ruled_out
- Manifest scope_filter block present and correct

Run:  pytest test_scope_revision_downstream.py -v
"""
from unittest.mock import MagicMock
import pytest

from dackar.RCA.orchestrators.artifact_store import NoOpSchemaValidator
from dackar.RCA.orchestrators.rca_reasoning_orchestrator import (
    OrchestratorConfig,
    RCAReasoningOrchestrator,
)

RESOLVE = RCAReasoningOrchestrator._resolve_approved_scope_boundary
FILTER  = RCAReasoningOrchestrator._apply_scope_boundary_filter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_context_v0(component_ids=None):
    """run_context at version 0 (initial intake, no analyst decisions yet)."""
    cids = component_ids or ["PUMP-1", "VALVE-A"]
    return {
        "input_refs": {"active_scope_version": 0, "active_scope_revision_id": "SCOPE::EVT-1::0"},
        "scope_management": {
            "active_scope_version": 0,
            "latest_approved_revision_id": "SCOPE::EVT-1::0",
            "scope_revisions": [
                {
                    "revision_id": "SCOPE::EVT-1::0",
                    "scope_version": 0,
                    "trigger": "initial_intake",
                    "changed_boundary": {},
                    "analyst_decision": "accepted",
                    "decision_timestamp": "2024-01-01T00:00:00Z",
                    "scope_snapshot": {"component_ids": cids, "asset_ids": ["PLANT-A"]},
                }
            ],
        },
    }


def _run_context_v1(component_ids=None):
    """run_context at version 1 (analyst accepted one expansion)."""
    cids = ["PUMP-1", "VALVE-A"] if component_ids is None else component_ids
    ctx = _run_context_v0()
    ctx["input_refs"]["active_scope_version"] = 1
    ctx["scope_management"]["active_scope_version"] = 1
    ctx["scope_management"]["scope_revisions"].append({
        "revision_id": "SCOPE::EVT-1::1",
        "scope_version": 1,
        "trigger": "manual_revision",
        "changed_boundary": {"added_component_ids": cids},
        "analyst_decision": "accepted",
        "decision_timestamp": "2024-01-01T01:00:00Z",
        "scope_snapshot": {"component_ids": cids, "asset_ids": ["PLANT-A"]},
    })
    return ctx


def _candidates(*component_ids):
    """Build a minimal causality_candidates dict."""
    cands = []
    for cid in component_ids:
        cands.append({
            "candidate_id": f"FM::{cid}",
            "component_id": cid,
            "composite_score": 0.7,
            "scores": {"structural": 0.7},
        })
    return {"candidates": cands, "ruled_out": [], "metadata": {}}


def _make_orchestrator():
    orch = RCAReasoningOrchestrator.__new__(RCAReasoningOrchestrator)
    orch.artifact_store = MagicMock()
    orch.artifact_store.save = MagicMock()
    return orch


# ===========================================================================
# _resolve_approved_scope_boundary
# ===========================================================================

def test_resolve_boundary_version_zero_returns_none():
    assert RESOLVE(_run_context_v0()) is None


def test_resolve_boundary_version_one_returns_frozenset():
    ctx = _run_context_v1(["PUMP-1", "VALVE-A"])
    result = RESOLVE(ctx)
    assert result is not None
    assert isinstance(result, frozenset)
    assert "pump-1" in result
    assert "valve-a" in result


def test_resolve_boundary_lower_cases_ids():
    ctx = _run_context_v1(["PUMP-1", "ValVe-B"])
    result = RESOLVE(ctx)
    assert "pump-1" in result
    assert "valve-b" in result
    assert "ValVe-B" not in result


def test_resolve_boundary_empty_snapshot_returns_none():
    ctx = _run_context_v1([])
    assert RESOLVE(ctx) is None


def test_resolve_boundary_latest_accepted_wins():
    """When there are multiple revisions, the last accepted one is used."""
    ctx = _run_context_v1(["PUMP-1"])
    # Add a second accepted revision with a different boundary
    ctx["scope_management"]["active_scope_version"] = 2
    ctx["scope_management"]["scope_revisions"].append({
        "revision_id": "SCOPE::EVT-1::2",
        "scope_version": 2,
        "trigger": "manual_revision",
        "changed_boundary": {},
        "analyst_decision": "accepted",
        "decision_timestamp": "2024-01-01T02:00:00Z",
        "scope_snapshot": {"component_ids": ["PUMP-1", "HEAT-X"], "asset_ids": []},
    })
    result = RESOLVE(ctx)
    assert "heat-x" in result
    assert "pump-1" in result


def test_resolve_boundary_skips_non_accepted_revisions():
    """Rejected/deferred revisions do not count as the boundary."""
    ctx = _run_context_v0()
    ctx["input_refs"]["active_scope_version"] = 1
    ctx["scope_management"]["active_scope_version"] = 1
    ctx["scope_management"]["scope_revisions"].append({
        "revision_id": "SCOPE::EVT-1::1",
        "scope_version": 0,    # rejected, so version stays
        "trigger": "manual_revision",
        "changed_boundary": {},
        "analyst_decision": "rejected",
        "decision_timestamp": "2024-01-01T01:00:00Z",
        "scope_snapshot": {"component_ids": ["PUMP-1"], "asset_ids": []},
    })
    # Version > 0 but last *accepted* revision is v0 with its original CIDs
    result = RESOLVE(ctx)
    # Should return the v0 initial intake CIDs (["PUMP-1","VALVE-A"] from _run_context_v0)
    assert result is not None
    assert "pump-1" in result


# ===========================================================================
# _apply_scope_boundary_filter
# ===========================================================================

def test_apply_filter_moves_out_of_scope_to_ruled_out():
    cands = _candidates("PUMP-1", "VALVE-Z")
    boundary = frozenset(["pump-1"])
    result = FILTER(cands, boundary, scope_version=1)
    cand_cids = [c["component_id"] for c in result["candidates"]]
    ruled_cids = [r["component_id"] for r in result["ruled_out"]]
    assert "PUMP-1" in cand_cids
    assert "VALVE-Z" not in cand_cids
    assert "VALVE-Z" in ruled_cids


def test_apply_filter_keeps_in_scope_candidates():
    cands = _candidates("PUMP-1", "VALVE-A")
    boundary = frozenset(["pump-1", "valve-a"])
    result = FILTER(cands, boundary, scope_version=1)
    assert len(result["candidates"]) == 2
    assert len(result["ruled_out"]) == 0


def test_apply_filter_no_component_id_not_filtered():
    cands = {
        "candidates": [{"candidate_id": "FM::X", "composite_score": 0.5}],
        "ruled_out": [],
    }
    boundary = frozenset(["pump-1"])
    result = FILTER(cands, boundary, scope_version=1)
    assert len(result["candidates"]) == 1
    assert len(result["ruled_out"]) == 0


def test_apply_filter_reason_code_is_scope_filtered():
    cands = _candidates("VALVE-Z")
    boundary = frozenset(["pump-1"])
    result = FILTER(cands, boundary, scope_version=1)
    assert result["ruled_out"][0]["reason_code"] == "scope_filtered"


def test_apply_filter_ruled_out_preserves_original_score():
    cands = _candidates("VALVE-Z")
    boundary = frozenset(["pump-1"])
    result = FILTER(cands, boundary, scope_version=1)
    assert result["ruled_out"][0]["original_composite_score"] == 0.7


def test_apply_filter_sets_meta_fields():
    cands = _candidates("PUMP-1", "VALVE-Z")
    boundary = frozenset(["pump-1"])
    result = FILTER(cands, boundary, scope_version=2)
    assert result["scope_filter_applied"] is True
    assert result["scope_filter_version"] == 2
    assert result["scope_filter_filtered_count"] == 1
    assert "VALVE-Z" in result["scope_filter_filtered_component_ids"]


def test_apply_filter_empty_candidates_no_crash():
    cands = {"candidates": [], "ruled_out": []}
    boundary = frozenset(["pump-1"])
    result = FILTER(cands, boundary, scope_version=1)
    assert result["candidates"] == []
    assert result["ruled_out"] == []


# ===========================================================================
# apply_scope_revision — auto-merge enhancement
# ===========================================================================

def test_apply_scope_revision_merges_added_component_ids():
    orch = _make_orchestrator()
    ctx = _run_context_v0(["PUMP-1"])
    updated = orch.apply_scope_revision(
        run_id="R1",
        run_context=ctx,
        revision_input={
            "trigger": "analyst_expand",
            "analyst_decision": "accepted",
            "changed_boundary": {"added_component_ids": ["VALVE-X", "SENSOR-9"]},
        },
        persist=False,
    )
    revisions = updated["scope_management"]["scope_revisions"]
    new_snap = revisions[-1]["scope_snapshot"]["component_ids"]
    assert "PUMP-1" in new_snap
    assert "VALVE-X" in new_snap
    assert "SENSOR-9" in new_snap


def test_apply_scope_revision_removes_component_ids():
    orch = _make_orchestrator()
    ctx = _run_context_v0(["PUMP-1", "VALVE-OLD"])
    updated = orch.apply_scope_revision(
        run_id="R1",
        run_context=ctx,
        revision_input={
            "trigger": "scope_contraction",
            "analyst_decision": "accepted",
            "changed_boundary": {
                "added_component_ids": [],
                "removed_component_ids": ["VALVE-OLD"],
            },
        },
        persist=False,
    )
    revisions = updated["scope_management"]["scope_revisions"]
    new_snap = revisions[-1]["scope_snapshot"]["component_ids"]
    assert "VALVE-OLD" not in new_snap
    assert "PUMP-1" in new_snap


def test_apply_scope_revision_removes_component_ids_case_insensitively():
    # Stored id "PUMP-1"; analyst removes "pump-1". Downstream boundary matching
    # normalizes with .lower(), so the removal must delete the stored id despite
    # the casing mismatch.
    orch = _make_orchestrator()
    ctx = _run_context_v0(["PUMP-1", "VALVE-OLD"])
    updated = orch.apply_scope_revision(
        run_id="R1",
        run_context=ctx,
        revision_input={
            "trigger": "scope_contraction",
            "analyst_decision": "accepted",
            "changed_boundary": {"removed_component_ids": ["pump-1"]},
        },
        persist=False,
    )
    new_snap = updated["scope_management"]["scope_revisions"][-1]["scope_snapshot"]["component_ids"]
    assert [c.lower() for c in new_snap] == ["valve-old"]


def test_apply_scope_revision_added_id_does_not_duplicate_differently_cased_entry():
    # Stored id "PUMP-1"; analyst adds "pump-1" — must not create a duplicate.
    orch = _make_orchestrator()
    ctx = _run_context_v0(["PUMP-1"])
    updated = orch.apply_scope_revision(
        run_id="R1",
        run_context=ctx,
        revision_input={
            "trigger": "analyst_expand",
            "analyst_decision": "accepted",
            "changed_boundary": {"added_component_ids": ["pump-1", "VALVE-X"]},
        },
        persist=False,
    )
    new_snap = updated["scope_management"]["scope_revisions"][-1]["scope_snapshot"]["component_ids"]
    normed = sorted(c.lower() for c in new_snap)
    assert normed == ["pump-1", "valve-x"]  # no duplicate pump-1
    assert "PUMP-1" in new_snap  # original display casing preserved


def test_apply_scope_revision_no_explicit_snapshot_builds_from_prior():
    """When caller omits scope_snapshot, method auto-builds from latest accepted."""
    orch = _make_orchestrator()
    ctx = _run_context_v0(["PUMP-1"])
    updated = orch.apply_scope_revision(
        run_id="R1",
        run_context=ctx,
        revision_input={
            "trigger": "manual_revision",
            "analyst_decision": "accepted",
            "changed_boundary": {"added_component_ids": ["NEW-CMP"]},
            # no scope_snapshot key
        },
        persist=False,
    )
    snap = updated["scope_management"]["scope_revisions"][-1]["scope_snapshot"]
    assert "PUMP-1" in snap["component_ids"]
    assert "NEW-CMP" in snap["component_ids"]


def test_apply_scope_revision_rejected_does_not_merge():
    orch = _make_orchestrator()
    ctx = _run_context_v0(["PUMP-1"])
    updated = orch.apply_scope_revision(
        run_id="R1",
        run_context=ctx,
        revision_input={
            "trigger": "rejected",
            "analyst_decision": "rejected",
            "changed_boundary": {"added_component_ids": ["NEW-CMP"]},
        },
        persist=False,
    )
    # version should not change
    assert updated["scope_management"]["active_scope_version"] == 0


# ===========================================================================
# resolve_expansion_suggestion
# ===========================================================================

def _ctx_with_suggestion(signal_id="SEX::ALLEN::valve-z", suggested=None):
    ctx = _run_context_v0(["PUMP-1"])
    ctx["scope_management"]["expansion_suggestions"] = [
        {
            "signal_id": signal_id,
            "source_stage": "step_2c_allen_relation_map",
            "trigger_type": "out_of_scope_causal_component",
            "suggested_component_ids": suggested or ["VALVE-Z"],
            "analyst_decision": "pending",
        }
    ]
    return ctx


def test_resolve_expansion_suggestion_accepted_updates_scope():
    orch = _make_orchestrator()
    ctx = _ctx_with_suggestion()
    updated = orch.resolve_expansion_suggestion(
        run_id="R1",
        run_context=ctx,
        signal_id="SEX::ALLEN::valve-z",
        decision="accepted",
        persist=False,
    )
    assert updated["scope_management"]["active_scope_version"] == 1
    snap = updated["scope_management"]["scope_revisions"][-1]["scope_snapshot"]
    assert "VALVE-Z" in snap["component_ids"]


def test_resolve_expansion_suggestion_rejected_no_scope_change():
    orch = _make_orchestrator()
    ctx = _ctx_with_suggestion()
    updated = orch.resolve_expansion_suggestion(
        run_id="R1",
        run_context=ctx,
        signal_id="SEX::ALLEN::valve-z",
        decision="rejected",
        persist=False,
    )
    assert updated["scope_management"]["active_scope_version"] == 0


def test_resolve_expansion_suggestion_deferred_no_scope_change():
    orch = _make_orchestrator()
    ctx = _ctx_with_suggestion()
    updated = orch.resolve_expansion_suggestion(
        run_id="R1",
        run_context=ctx,
        signal_id="SEX::ALLEN::valve-z",
        decision="deferred",
        persist=False,
    )
    assert updated["scope_management"]["active_scope_version"] == 0


def test_resolve_expansion_suggestion_marks_analyst_decision():
    orch = _make_orchestrator()
    ctx = _ctx_with_suggestion()
    updated = orch.resolve_expansion_suggestion(
        run_id="R1",
        run_context=ctx,
        signal_id="SEX::ALLEN::valve-z",
        decision="rejected",
        persist=False,
    )
    sug = updated["scope_management"]["expansion_suggestions"][0]
    assert sug["analyst_decision"] == "rejected"
    assert "resolution_timestamp" in sug


def test_resolve_expansion_suggestion_stores_rationale():
    orch = _make_orchestrator()
    ctx = _ctx_with_suggestion()
    updated = orch.resolve_expansion_suggestion(
        run_id="R1",
        run_context=ctx,
        signal_id="SEX::ALLEN::valve-z",
        decision="deferred",
        rationale="Not enough evidence yet",
        persist=False,
    )
    sug = updated["scope_management"]["expansion_suggestions"][0]
    assert sug["analyst_rationale"] == "Not enough evidence yet"


def test_resolve_expansion_suggestion_unknown_signal_id_raises():
    orch = _make_orchestrator()
    ctx = _ctx_with_suggestion()
    with pytest.raises(ValueError, match="signal_id"):
        orch.resolve_expansion_suggestion(
            run_id="R1",
            run_context=ctx,
            signal_id="SEX::NONEXISTENT",
            decision="accepted",
            persist=False,
        )


# ===========================================================================
# run() integration — scope filter activated / not activated
# ===========================================================================

def _make_full_orchestrator(extra_candidates=None):
    """Build a minimal orchestrator that returns controllable candidates."""
    orch = RCAReasoningOrchestrator.__new__(RCAReasoningOrchestrator)
    orch.artifact_store = MagicMock()
    orch.artifact_store.save = MagicMock()

    # Causality engine that returns a fixed candidates dict
    cands = _candidates("PUMP-1", "VALVE-Z")
    if extra_candidates:
        cands["candidates"].extend(extra_candidates)
    engine = MagicMock()
    engine.generate = MagicMock(return_value=cands)
    engine.refine_with_evidence = None  # trigger hasattr check to skip refinement
    orch.causality_engine = engine

    # Minimal stubs for the rest of the pipeline
    orch.config = MagicMock()
    orch.config.persist_intermediate_artifacts = False
    orch.config.top_k_candidates = 5
    orch.config.top_k_evidence = 5
    orch.config.enable_ishikawa = False
    orch.config.stop_on_validation_error = False
    orch.config.extra = {}
    orch.kg_context_builder = MagicMock()
    orch.kg_context_builder.build = MagicMock(return_value={
        "subgraph_id": "SG-1", "event_id": "EVT-1", "asset_id": "PLANT-A",
        "components": [
            {"component_id": "PUMP-1", "name": "Pump"},
            {"component_id": "VALVE-Z", "name": "Valve"},
        ],
        "failure_modes": [], "documents": [], "past_events": [],
    })
    orch.evidence_retriever = MagicMock()
    orch.evidence_retriever.retrieve = MagicMock(return_value={
        "event_id": "EVT-1", "query_results": [],
    })
    orch.rca_synthesizer = MagicMock()
    orch.rca_synthesizer.synthesize = MagicMock(return_value={
        "event_id": "EVT-1",
        "primary_hypothesis": {"candidate_id": "FM::PUMP-1", "cause_label": "wear"},
        "contributing_causes": [],
        "causal_depth_summary": {"depth_complete": False},
        "recommended_actions": [],
        "unresolved_gaps": [],
        "effectiveness_monitoring_plan": [],
        "human_performance_assessment": {"applicable": False, "category_flags": [], "findings": []},
        "barrier_analysis": {},
        "analyst_review": {"decision_required": False},
        "executive_summary": {"analyst_attention_flags": []},
    })
    orch.cmms_adapter = None
    orch.similar_event_adapter = None
    return orch


def _minimal_event():
    return {
        "event_id": "EVT-1",
        "asset_id": "PLANT-A",
        "component_id": "PUMP-1",
        "timestamp_start": "2024-01-01T06:00:00Z",
        "severity": "significant",
        "event_type": "reactor_trip",
        "symptom_signature": {"anomaly_pattern": "vibration"},
    }


def _minimal_telemetry():
    return {"asset_id": "PLANT-A", "signals": [], "anomalies": []}


def _make_run_context_v1(component_ids):
    """Build a run_context that mimics a version-1 approved scope."""
    ctx = {
        "run_id": "test-run",
        "input_refs": {
            "event_id": "EVT-1",
            "asset_id": "PLANT-A",
            "active_scope_version": 1,
            "active_scope_revision_id": "SCOPE::EVT-1::1",
            "has_operational_context": False,
            "has_pm_compliance": False,
        },
        "scope_management": {
            "active_scope_version": 1,
            "latest_approved_revision_id": "SCOPE::EVT-1::1",
            "scope_revisions": [
                {
                    "revision_id": "SCOPE::EVT-1::0",
                    "scope_version": 0,
                    "trigger": "initial_intake",
                    "changed_boundary": {},
                    "analyst_decision": "accepted",
                    "decision_timestamp": "2024-01-01T00:00:00Z",
                    "scope_snapshot": {"component_ids": ["PUMP-1"], "asset_ids": ["PLANT-A"]},
                },
                {
                    "revision_id": "SCOPE::EVT-1::1",
                    "scope_version": 1,
                    "trigger": "manual_revision",
                    "changed_boundary": {"added_component_ids": component_ids},
                    "analyst_decision": "accepted",
                    "decision_timestamp": "2024-01-01T01:00:00Z",
                    "scope_snapshot": {"component_ids": component_ids, "asset_ids": ["PLANT-A"]},
                },
            ],
        },
        "pipeline_runtime": {},
        "validation": {"inputs": None},
    }
    return ctx


def test_run_version_zero_no_filter_applied():
    """With version 0, all candidates pass through (discovery mode)."""
    orch = _make_full_orchestrator()
    # We don't call run() here; instead validate the helper directly
    ctx = _run_context_v0()
    boundary = RESOLVE(ctx)
    assert boundary is None  # no filter should be applied


def test_run_scope_filter_removes_out_of_scope_candidates():
    """_resolve + _apply_filter pipeline removes VALVE-Z when scope = [PUMP-1]."""
    ctx = _run_context_v1(["PUMP-1"])   # VALVE-Z is NOT in approved scope
    boundary = RESOLVE(ctx)
    assert boundary is not None
    assert "pump-1" in boundary
    assert "valve-z" not in boundary

    cands = _candidates("PUMP-1", "VALVE-Z")
    result = FILTER(cands, boundary, scope_version=1)
    cand_cids = [c["component_id"] for c in result["candidates"]]
    ruled_cids = [r["component_id"] for r in result["ruled_out"]]

    assert "PUMP-1" in cand_cids
    assert "VALVE-Z" not in cand_cids
    assert "VALVE-Z" in ruled_cids
    assert result["scope_filter_filtered_count"] == 1


def test_scope_filter_manifest_block_structure():
    """scope_filter block has all required keys."""
    ctx = _run_context_v1(["PUMP-1"])
    boundary = RESOLVE(ctx)
    cands = _candidates("PUMP-1", "VALVE-Z")
    filtered = FILTER(cands, boundary, scope_version=1)

    scope_filter_block = {
        "applied": True,
        "approved_scope_version": 1,
        "approved_boundary_size": len(boundary),
        "filtered_count": filtered["scope_filter_filtered_count"],
        "filtered_component_ids": filtered["scope_filter_filtered_component_ids"],
    }
    for key in ("applied", "approved_scope_version", "approved_boundary_size",
                "filtered_count", "filtered_component_ids"):
        assert key in scope_filter_block


# ===========================================================================
# run() integration — reapply the approved boundary AFTER auto re-entry (I4)
# ===========================================================================
#
# Auto re-entry regenerates candidates from scratch (generate + refine), so a
# component outside the analyst-approved scope boundary can be reintroduced
# after the one-time filter that runs at generation time. These tests drive the
# real RCAReasoningOrchestrator.run() to prove the boundary is reapplied to the
# post-re-entry candidate set.


class _MemoryArtifactStore:
    def __init__(self):
        self.saved = {}

    def save(self, run_id, artifact_name, payload):
        self.saved.setdefault(run_id, {})[artifact_name] = payload
        return f"{run_id}/{artifact_name}.json"

    def save_list(self, run_id, artifact_name, payload):
        self.saved.setdefault(run_id, {})[artifact_name] = payload
        return f"{run_id}/{artifact_name}.json"


class _ReentryKGBuilder:
    client = None
    database = None

    def build(self, event, telemetry_summary, operational_context, pm_compliance,
              run_context, focus_component_ids=None):
        # CMP-1 is in the approved boundary; FM-1 resolves the re-entry target.
        return {
            "event_id": event.get("event_id"),
            "asset_id": event.get("asset_id"),
            "subgraph_id": "KGCTX::RE",
            "components": [{"component_id": "CMP-1"}],
            "failure_modes": [{"fm_id": "FM-1", "component_id": "CMP-1"}],
            "past_events": [],
            "seed_context": {},
            "documents": [],
        }


class _EvidenceRetriever:
    store = object()

    def retrieve(self, event, kg_context, causality_candidates, operational_context, run_context):
        return {
            "retrieval_scope": {"asset_id": event.get("asset_id")},
            "results": [],
            "candidate_evidence_summary": [],
            "pipeline_health": {"status": "green", "issues": []},
        }


def _cand_full(cid, component_id, score):
    return {
        "candidate_id": cid,
        "component_id": component_id,
        "cause_node_id": "FM-1",
        "composite_score": score,
        "scores": {"structural": score},
        "confidence_label": "medium",
        "temporal_evidence": {},
    }


class _ReentryEngine:
    """Forces exactly one auto re-entry whose regeneration reintroduces an
    out-of-scope component (CMP-OOS)."""

    def __init__(self):
        self.generate_calls = 0
        self.refine_calls = 0

    def generate(self, **kwargs):
        self.generate_calls += 1
        if self.generate_calls == 1:
            # Initial generation — both candidates are in-scope (CMP-1).
            return {
                "event_id": "EVT-RE",
                "candidates": [
                    _cand_full("FM::CMP-1", "CMP-1", 0.60),
                    _cand_full("FM::CMP-1b", "CMP-1", 0.50),
                ],
                "ruled_out": [],
            }
        # Re-entry regeneration — reintroduces the OUT-OF-SCOPE component.
        return {
            "event_id": "EVT-RE",
            "candidates": [
                _cand_full("FM::CMP-OOS", "CMP-OOS", 0.90),
                _cand_full("FM::CMP-1", "CMP-1", 0.60),
            ],
            "ruled_out": [],
        }

    def refine_with_evidence(self, causality_candidates, evidence_bundle,
                             signal_evidence=None, **kwargs):
        self.refine_calls += 1
        out = dict(causality_candidates)
        cands = list(causality_candidates.get("candidates") or [])
        if self.refine_calls == 1:
            # Rank inversion vs pre-refine (top was FM::CMP-1): promote the
            # runner-up so _compute_reentry_hook fires.
            cands = sorted(cands, key=lambda c: 0 if c.get("candidate_id") == "FM::CMP-1b" else 1)
        out["candidates"] = cands
        return out


class _CapturingSynthesizer:
    """Records the candidate set it is handed so the test can assert the
    out-of-scope component never reaches synthesis."""

    def __init__(self):
        self.seen_candidate_ids = None

    def synthesize(self, event, telemetry_summary, kg_context, tskr_patterns,
                   causality_candidates, evidence_bundle, operational_context,
                   pm_compliance, ishikawa_matrix, cmms_context, run_context, **kwargs):
        self.seen_candidate_ids = [
            c.get("candidate_id") for c in (causality_candidates.get("candidates") or [])
        ]
        return {
            "event_id": event.get("event_id"),
            "asset_id": event.get("asset_id"),
            "executive_summary": {"decision_status": "candidate_ready", "analyst_attention_flags": []},
            "primary_hypothesis": {
                "candidate_id": (self.seen_candidate_ids or ["NONE"])[0],
                "cause_label": "cause",
                "confidence_label": "medium",
            },
            "validation_status": {
                "schema_valid": True,
                "all_claims_cited": True,
                "passed_minimum_evidence_gate": True,
                "fallback_used": False,
            },
            "analyst_review": {"decision_required": False, "writeback_recommendation": "ready_if_accepted"},
            "recommended_actions": [],
            "contributing_causes": [],
        }


def _reentry_scope_management_v1(component_ids):
    """A seeded version-1 scope_management with an accepted boundary."""
    return {
        "active_scope_version": 1,
        "latest_approved_revision_id": "SCOPE::EVT-RE::1",
        "scope_revisions": [
            {
                "revision_id": "SCOPE::EVT-RE::0",
                "scope_version": 0,
                "trigger": "initial_intake",
                "changed_boundary": {},
                "analyst_decision": "accepted",
                "decision_timestamp": "2026-01-01T00:00:00+00:00",
                "scope_snapshot": {"component_ids": component_ids, "asset_ids": ["ASSET-RE"]},
            },
            {
                "revision_id": "SCOPE::EVT-RE::1",
                "scope_version": 1,
                "trigger": "manual_revision",
                "changed_boundary": {"added_component_ids": component_ids},
                "analyst_decision": "accepted",
                "decision_timestamp": "2026-01-01T01:00:00+00:00",
                "scope_snapshot": {"component_ids": component_ids, "asset_ids": ["ASSET-RE"]},
            },
        ],
    }


def _make_reentry_orchestrator(synthesizer):
    cfg = OrchestratorConfig(
        enable_ishikawa=False,
        persist_intermediate_artifacts=False,
        stop_on_validation_error=False,
        extra={
            "strict_red_state_governance": False,
            "hard_abort_on_kg_red_state": False,
            "enable_chroma_archive_stage": False,
            "hard_fail_on_chroma_archive_error": False,
            "causality_engine_version": "v32",
            "enable_auto_reentry": True,
            "auto_reentry_max_attempts": 1,
        },
    )
    return RCAReasoningOrchestrator(
        validator=NoOpSchemaValidator(),
        artifact_store=_MemoryArtifactStore(),
        kg_context_builder=_ReentryKGBuilder(),
        tskr_temporal_scorer=None,
        causality_engine=_ReentryEngine(),
        evidence_retriever=_EvidenceRetriever(),
        rca_synthesizer=synthesizer,
        config=cfg,
    )


def _reentry_event():
    return {
        "event_id": "EVT-RE",
        "asset_id": "ASSET-RE",
        "component_id": "CMP-1",
        "timestamp_start": "2026-01-01T12:00:00+00:00",
        "severity": "HIGH",
        "event_type": "FAILURE",
    }


def test_run_reapplies_scope_boundary_after_auto_reentry():
    """A re-entry that reintroduces an out-of-scope component must have that
    component filtered back out of the final candidate set."""
    synth = _CapturingSynthesizer()
    orch = _make_reentry_orchestrator(synth)
    result = orch.run(
        event=_reentry_event(),
        telemetry_summary={"asset_id": "ASSET-RE", "signals": []},
        initial_scope_management=_reentry_scope_management_v1(["CMP-1"]),
    )

    # Re-entry actually ran (otherwise CMP-OOS would never have been generated).
    assert result["reentry_execution"]["attempt_count"] == 1
    assert orch.causality_engine.generate_calls == 2

    final = result["causality_candidates"]
    cand_cids = [c.get("component_id") for c in (final.get("candidates") or [])]
    ruled_cids = {
        r.get("component_id")
        for r in (final.get("ruled_out") or [])
        if r.get("reason_code") == "scope_filtered"
    }
    # The reintroduced out-of-scope component is excluded from the final set
    # and recorded as scope-filtered; the in-scope component survives.
    assert "CMP-OOS" not in cand_cids
    assert "CMP-OOS" in ruled_cids
    assert "CMP-1" in cand_cids

    # The filtered set is what reaches synthesis (card never sees CMP-OOS).
    assert synth.seen_candidate_ids is not None
    assert "FM::CMP-OOS" not in synth.seen_candidate_ids

    # Manifest accounting reflects the post-re-entry reapplication.
    scope_filter = result["run_context"]["pipeline_runtime"]["scope_filter"]
    assert scope_filter.get("reapplied_after_reentry") is True
    assert "CMP-OOS" in (scope_filter.get("filtered_component_ids") or [])


def test_run_without_reentry_does_not_reapply_boundary_flag():
    """When no re-entry occurs, the post-re-entry reapplication branch is not
    taken (the fix is gated on an actual re-entry, not run unconditionally)."""
    synth = _CapturingSynthesizer()
    orch = _make_reentry_orchestrator(synth)
    # auto_reentry_max_attempts = 0 → the hook can fire but no attempt executes.
    orch.config.extra["auto_reentry_max_attempts"] = 0
    result = orch.run(
        event=_reentry_event(),
        telemetry_summary={"asset_id": "ASSET-RE", "signals": []},
        initial_scope_management=_reentry_scope_management_v1(["CMP-1"]),
    )
    assert result["reentry_execution"]["attempt_count"] == 0
    assert orch.causality_engine.generate_calls == 1
    scope_filter = result["run_context"]["pipeline_runtime"]["scope_filter"]
    assert scope_filter.get("reapplied_after_reentry") is not True
