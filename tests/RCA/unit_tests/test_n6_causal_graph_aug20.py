"""
test_n6_causal_graph_aug20.py — N-6 unified inspectable causal graph.

Causal reasoning was spread across TSKR chain-position, the telemetry signal-DAG,
common-cause/explain-away links, near-tie competition, and hard gates, so there
was no single directed cause->effect graph the analyst could see and contest.
N-6 adds an additive, ranking-neutral `causal_graph` card block that consolidates
those already-computed signals into one graph: nodes = target event + candidates
(with role + both chain-position views), directed edges = chain-position precedence
vs the event and shared-cause explain-away, undirected edges = near-tie competition.

Run:  pytest test_n6_causal_graph_aug20.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.llm_clients import DummyLLMClient  # noqa: E402
from synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)


def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


_EVENT = {"event_id": "EVT-1", "title": "Pump trip"}


def _cand(cid, cp, score=0.6, **extra):
    c = {
        "candidate_id": cid,
        "cause_label": f"cause {cid}",
        "chain_position": cp,
        "composite_score": score,
        "confidence_label": "medium",
        "canonical_tuple": {"component": "PUMP", "failure_mode": "wear", "causal_category": "A"},
    }
    c.update(extra)
    return c


def _edges_between(block, a, b, relation=None):
    out = []
    for e in block["edges"]:
        if {e["from"], e["to"]} == {a, b} and (relation is None or e["relation"] == relation):
            out.append(e)
    return out


def test_target_event_node_present():
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"}},
        event=_EVENT,
        causality_candidates={"candidates": [_cand("FM::A", "initiating")]},
    )
    ev = [n for n in block["nodes"] if n["node_type"] == "target_event"]
    assert len(ev) == 1 and ev[0]["role"] == "target_event"
    assert block["target_event_id"] == "EVT-1"


def test_initiating_candidate_points_to_event():
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"}},
        event=_EVENT,
        causality_candidates={"candidates": [_cand("FM::A", "initiating")]},
    )
    ev_id = "EVENT::EVT-1"
    e = _edges_between(block, "FM::A", ev_id, "temporal_precedence")
    assert len(e) == 1
    assert e[0]["from"] == "FM::A" and e[0]["to"] == ev_id and e[0]["directed"] is True
    assert block["directionality_committed"] is True


def test_consequence_candidate_points_from_event():
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"}},
        event=_EVENT,
        causality_candidates={"candidates": [_cand("FM::C", "consequence")]},
    )
    ev_id = "EVENT::EVT-1"
    e = _edges_between(block, "FM::C", ev_id, "temporal_precedence")
    assert len(e) == 1
    assert e[0]["from"] == ev_id and e[0]["to"] == "FM::C"  # event -> downstream symptom


def test_primary_and_alternative_roles_assigned():
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"}},
        event=_EVENT,
        causality_candidates={"candidates": [_cand("FM::A", "initiating"), _cand("FM::B", "contributing")]},
    )
    roles = {n["id"]: n["role"] for n in block["nodes"]}
    assert roles["FM::A"] == "primary"
    assert roles["FM::B"] == "alternative"


def test_contributing_role_from_card():
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"},
              "contributing_causes": [{"candidate_id": "FM::B"}]},
        event=_EVENT,
        causality_candidates={"candidates": [_cand("FM::A", "initiating"), _cand("FM::B", "contributing")]},
    )
    roles = {n["id"]: n["role"] for n in block["nodes"]}
    assert roles["FM::B"] == "contributing"


def test_gate_eliminated_candidate_is_eliminated_role_without_causal_edge():
    blocked = _cand("FM::X", "initiating", score=0.9,
                    primary_eligibility="blocked", primary_block_reasons=["barrier_logic_gate_failed"],
                    hard_gates={"barrier_logic": {"passed": False}})
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"}},
        event=_EVENT,
        causality_candidates={"candidates": [_cand("FM::A", "initiating"), blocked]},
    )
    roles = {n["id"]: n["role"] for n in block["nodes"]}
    assert roles["FM::X"] == "eliminated"
    # eliminated candidate keeps a node but no precedence edge to the event
    assert _edges_between(block, "FM::X", "EVENT::EVT-1") == []


def test_explain_away_edge_from_common_cause_summary():
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::SYMPTOM"}},
        event=_EVENT,
        causality_candidates={
            "candidates": [_cand("FM::ROOT", "initiating"), _cand("FM::SYMPTOM", "contributing")],
            "common_cause_summary": {
                "suspected_common_cause": True,
                "top_common_cause_candidate_id": "FM::ROOT",
                "explained_away_candidate_ids": ["FM::SYMPTOM"],
            },
        },
    )
    e = _edges_between(block, "FM::ROOT", "FM::SYMPTOM", "explained_away")
    assert len(e) == 1 and e[0]["from"] == "FM::ROOT" and e[0]["directed"] is True


def test_near_tie_edge_is_undirected_and_deduped():
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"}},
        event=_EVENT,
        causality_candidates={"candidates": [
            _cand("FM::A", "initiating", near_tie_with=["FM::B"]),
            _cand("FM::B", "initiating", near_tie_with=["FM::A"]),
        ]},
    )
    e = _edges_between(block, "FM::A", "FM::B", "near_tie")
    assert len(e) == 1  # deduped despite reciprocal near_tie_with
    assert e[0]["directed"] is False


def test_node_preserves_both_chain_position_views():
    c = _cand("FM::A", "initiating")
    c["scores"] = {"signal_dag_position_type": "convergence_confluence"}
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"}},
        event=_EVENT,
        causality_candidates={"candidates": [c]},
    )
    node = next(n for n in block["nodes"] if n["id"] == "FM::A")
    assert node["chain_position"] == "initiating"
    assert node["signal_dag_position"] == "convergence_confluence"


def test_directionality_uncommitted_when_no_ordering_signal():
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"}},
        event=_EVENT,
        causality_candidates={"candidates": [_cand("FM::A", "")]},  # no chain_position
    )
    assert block["directionality_committed"] is False


def test_schema_shape_is_bounded():
    block = _synth()._build_causal_graph(
        card={"primary_hypothesis": {"candidate_id": "FM::A"}},
        event=_EVENT,
        causality_candidates={"candidates": [_cand("FM::A", "initiating")]},
    )
    for key in ("target_event_id", "directionality_committed", "nodes", "edges", "provenance_note"):
        assert key in block
    for n in block["nodes"]:
        assert set(n).issubset({"id", "label", "node_type", "role", "component", "failure_mode",
                                "causal_category", "chain_position", "signal_dag_position",
                                "composite_score", "confidence_label"})
        assert {"id", "label", "node_type", "role"}.issubset(n)
    for e in block["edges"]:
        assert set(e).issubset({"from", "to", "relation", "directed", "basis"})
        assert {"from", "to", "relation", "directed", "basis"}.issubset(e)
