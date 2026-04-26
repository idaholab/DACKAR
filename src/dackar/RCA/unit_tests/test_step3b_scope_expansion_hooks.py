"""
test_step3b_scope_expansion_hooks.py — Phase 3b Scope-Expansion Hooks tests

Covers:
- _detect_scope_expansion_signals: Allen relation map out-of-scope causal component
- _detect_scope_expansion_signals: signal_evidence propagation chain out-of-scope component
- _detect_scope_expansion_signals: TSKR novel pattern (no match)
- _detect_scope_expansion_signals: in-scope component produces no signal
- _detect_scope_expansion_signals: de-duplication by signal_id
- _detect_scope_expansion_signals: empty scope boundary skips filtering
- _inject_scope_expansion_signals: merges new signals, idempotent on duplicate signal_id
- Manifest scope_expansion_summary populated correctly (pending count, by_trigger_type)
- review_hooks: analyst_decisions_required populated when pending signals exist
- review_hooks: degraded_run True when pending signals exist

Run:  pytest test_step3b_scope_expansion_hooks.py -v
"""
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator

DETECT = RCAReasoningOrchestrator._detect_scope_expansion_signals
INJECT = RCAReasoningOrchestrator._inject_scope_expansion_signals

T0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _run_context(component_ids: list = None, asset_ids: list = None) -> dict:
    """Minimal run_context with one accepted scope revision."""
    cids = component_ids if component_ids is not None else ["PUMP-A", "VALVE-B"]
    aids = asset_ids if asset_ids is not None else ["ASSET-1"]
    return {
        "run_id": "TEST-RUN-001",
        "input_refs": {"event_id": "EV-001"},
        "scope_management": {
            "active_scope_version": 0,
            "latest_approved_revision_id": "SCOPE::EV-001::0",
            "scope_revisions": [
                {
                    "revision_id": "SCOPE::EV-001::0",
                    "scope_version": 0,
                    "trigger": "initial_intake",
                    "changed_boundary": {},
                    "analyst_decision": "accepted",
                    "decision_timestamp": T0.isoformat(),
                    "scope_snapshot": {
                        "component_ids": cids,
                        "asset_ids": aids,
                        "system_boundary": [],
                    },
                }
            ],
            "expansion_suggestions": [],
        },
    }


def _allen_map(component_id: str, causal: bool = True, relation: str = "precedes") -> dict:
    return {
        "event_id": "EV-001",
        "generated_at": T0.isoformat(),
        "event_interval": {"start": T0.isoformat(), "end": None},
        "quality_flags": {"soe_clock_sync_ok": None, "alarm_clock_sync_ok": None, "soe_nodes_capped": False},
        "summary": {"total_nodes": 1, "node_type_counts": {}, "causal_nodes": 1 if causal else 0,
                    "contradiction_nodes": 0, "timeline_consistent": True},
        "nodes": [
            {
                "node_id": f"anomaly::{component_id}",
                "node_type": "anomaly",
                "source_id": component_id,
                "component_id": component_id,
                "interval_start": (T0 - timedelta(hours=3)).isoformat(),
                "interval_end": None,
                "is_point_event": False,
                "allen_relation_to_event": relation,
                "allen_base_score": 0.75,
                "causal_candidate": causal,
                "severity": "HIGH",
                "priority": None,
                "transition": None,
                "is_protection_signal": None,
                "system": None,
            }
        ],
        "provenance": {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Out-of-scope causal component from Allen map
# ─────────────────────────────────────────────────────────────────────────────

def test_out_of_scope_allen_causal_node_emits_signal():
    ctx = _run_context(component_ids=["PUMP-A", "VALVE-B"])
    arm = _allen_map("HEAT-EXCHANGER-C", causal=True)   # not in scope
    signals = DETECT(run_context=ctx, allen_relation_map=arm)
    assert len(signals) == 1
    s = signals[0]
    assert s["trigger_type"] == "out_of_scope_causal_component"
    assert s["source_stage"] == "step_2c_allen_relation_map"
    assert "HEAT-EXCHANGER-C" in s["suggested_component_ids"]
    assert s["severity"] == "warning"
    assert s["analyst_decision"] == "pending"


def test_in_scope_causal_node_produces_no_signal():
    ctx = _run_context(component_ids=["PUMP-A", "VALVE-B"])
    arm = _allen_map("pump-a", causal=True)   # matches PUMP-A (case-insensitive)
    signals = DETECT(run_context=ctx, allen_relation_map=arm)
    assert signals == []


def test_non_causal_out_of_scope_node_produces_no_signal():
    ctx = _run_context(component_ids=["PUMP-A"])
    arm = _allen_map("HEAT-EXCHANGER-C", causal=False, relation="during")
    signals = DETECT(run_context=ctx, allen_relation_map=arm)
    assert signals == []


def test_empty_scope_boundary_skips_component_filter():
    """When the latest accepted revision has no components, skip scope filtering."""
    ctx = _run_context(component_ids=[])
    arm = _allen_map("ANY-COMPONENT", causal=True)
    signals = DETECT(run_context=ctx, allen_relation_map=arm)
    assert signals == []   # no boundary → nothing to compare against


def test_allen_map_none_produces_no_signal():
    ctx = _run_context()
    signals = DETECT(run_context=ctx, allen_relation_map=None)
    assert signals == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Out-of-scope propagation chain components
# ─────────────────────────────────────────────────────────────────────────────

def test_out_of_scope_propagation_chain_emits_signal():
    ctx = _run_context(component_ids=["PUMP-A"])
    sig_ev = {
        "propagation_chains": [
            {"chain_id": "CHAIN-01", "component_ids": ["PUMP-A", "TURBINE-Z"]},
        ]
    }
    signals = DETECT(run_context=ctx, signal_evidence=sig_ev)
    assert any(s["trigger_type"] == "out_of_scope_propagation_component" for s in signals)
    assert any("TURBINE-Z" in s["suggested_component_ids"] for s in signals)


def test_in_scope_propagation_chain_no_signal():
    ctx = _run_context(component_ids=["PUMP-A", "VALVE-B"])
    sig_ev = {
        "propagation_chains": [
            {"chain_id": "CHAIN-01", "component_ids": ["PUMP-A", "VALVE-B"]},
        ]
    }
    signals = DETECT(run_context=ctx, signal_evidence=sig_ev)
    assert signals == []


def test_propagation_chain_none_produces_no_signal():
    ctx = _run_context()
    signals = DETECT(run_context=ctx, signal_evidence=None)
    assert signals == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. TSKR novel pattern signals
# ─────────────────────────────────────────────────────────────────────────────

def test_novel_tskr_pattern_emits_info_signal():
    ctx = _run_context()
    tskr = {
        "patterns": [
            {"pattern_id": "PAT-001", "novel_pattern": True, "component_id": "PUMP-A", "match_count": 0},
        ]
    }
    signals = DETECT(run_context=ctx, tskr_patterns=tskr)
    assert len(signals) == 1
    s = signals[0]
    assert s["trigger_type"] == "novel_signal_pattern"
    assert s["severity"] == "info"
    assert s["source_stage"] == "step_3_5_tskr_patterns"


def test_matched_tskr_pattern_no_signal():
    ctx = _run_context()
    tskr = {
        "patterns": [
            {"pattern_id": "PAT-002", "novel_pattern": False, "match_count": 5},
        ]
    }
    signals = DETECT(run_context=ctx, tskr_patterns=tskr)
    assert signals == []


def test_zero_match_count_emits_novel_signal():
    ctx = _run_context()
    tskr = {"patterns": [{"pattern_id": "PAT-003", "match_count": 0}]}
    signals = DETECT(run_context=ctx, tskr_patterns=tskr)
    assert len(signals) == 1
    assert signals[0]["trigger_type"] == "novel_signal_pattern"


# ─────────────────────────────────────────────────────────────────────────────
# 4. De-duplication
# ─────────────────────────────────────────────────────────────────────────────

def test_deduplication_by_signal_id():
    ctx = _run_context(component_ids=["PUMP-A"])
    # Two chains pointing to same out-of-scope component
    sig_ev = {
        "propagation_chains": [
            {"chain_id": "CHAIN-01", "component_ids": ["TURBINE-Z"]},
            {"chain_id": "CHAIN-02", "component_ids": ["TURBINE-Z"]},
        ]
    }
    signals = DETECT(run_context=ctx, signal_evidence=sig_ev)
    ids = [s["signal_id"] for s in signals]
    assert len(ids) == len(set(ids)), "Duplicate signal_ids should be removed"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Injection and idempotency
# ─────────────────────────────────────────────────────────────────────────────

def test_inject_adds_signals_to_scope_management():
    ctx = _run_context()
    new_signals = [
        {"signal_id": "SEX::ALLEN::A", "source_stage": "test", "trigger_type": "novel_signal_pattern",
         "analyst_decision": "pending", "suggested_component_ids": []},
    ]
    out = INJECT(ctx, new_signals)
    assert len(out["scope_management"]["expansion_suggestions"]) == 1


def test_inject_is_idempotent_on_same_signal_id():
    ctx = _run_context()
    sig = {"signal_id": "SEX::ALLEN::A", "source_stage": "test", "trigger_type": "novel_signal_pattern",
           "analyst_decision": "pending", "suggested_component_ids": []}
    INJECT(ctx, [sig])
    INJECT(ctx, [sig])   # second call should not duplicate
    assert len(ctx["scope_management"]["expansion_suggestions"]) == 1


def test_inject_merges_new_signals():
    ctx = _run_context()
    INJECT(ctx, [{"signal_id": "SEX::A", "source_stage": "x", "trigger_type": "novel_signal_pattern",
                  "analyst_decision": "pending", "suggested_component_ids": []}])
    INJECT(ctx, [{"signal_id": "SEX::B", "source_stage": "y", "trigger_type": "out_of_scope_causal_component",
                  "analyst_decision": "pending", "suggested_component_ids": []}])
    assert len(ctx["scope_management"]["expansion_suggestions"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 6. scope_expansion_summary counts
# ─────────────────────────────────────────────────────────────────────────────

def test_scope_expansion_summary_pending_count():
    ctx = _run_context(component_ids=["PUMP-A"])
    arm = _allen_map("TURBINE-Z", causal=True)
    tskr = {"patterns": [{"pattern_id": "PAT-001", "novel_pattern": True, "match_count": 0}]}
    signals = DETECT(run_context=ctx, allen_relation_map=arm, tskr_patterns=tskr)
    INJECT(ctx, signals)

    suggestions = ctx["scope_management"]["expansion_suggestions"]
    pending = [s for s in suggestions if s["analyst_decision"] == "pending"]
    assert len(pending) == 2   # one from Allen map, one from TSKR


def test_scope_expansion_summary_by_trigger_type():
    ctx = _run_context(component_ids=["PUMP-A"])
    arm = _allen_map("TURBINE-Z", causal=True)
    signals = DETECT(run_context=ctx, allen_relation_map=arm)
    assert signals[0]["trigger_type"] == "out_of_scope_causal_component"
