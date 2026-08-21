"""
test_p5_signal_dag_position_aug20.py — P-5 signal-DAG initiator scoring + wiring.

The telemetry signal-evidence DAG classifies each candidate's anomaly as a root /
common-cause root (initiator), an intermediate node, or a convergence confluence
(downstream symptom). Previously the initiator was scored a flat 1.0 regardless of
propagation quality, co-temporal (OVERLAPS) roots were treated as proven initiators,
and the DAG `position_type` was discarded downstream (only used to zero convergence
evidence). P-5 makes three honest, additive corrections:

  (b) an initiator is scored by its chain `path_score`, not a flat 1.0;
  (c) a root whose onset lead is not established (OVERLAPS / sub-threshold lag) is
      discounted and marked `initiator_lag_established=False`;
  (a) the DAG `position_type` / lag / path_score are wired onto the candidate scores
      (`signal_dag_*`) and surfaced to the analyst by the synthesizer.

Run:  pytest test_p5_signal_dag_position_aug20.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, Dict, List, Optional, Tuple

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from signal_evidence.builder import (  # noqa: E402
    build_signal_evidence,
    _COTEMPORAL_INITIATOR_FACTOR,
)
from signal_evidence.historian_adapter import NullHistorianAdapter  # noqa: E402
from orchestrators.causality_engine_v32 import (  # noqa: E402
    CausalityEngineConfigV32,
    RuleBasedCausalityEngineV32,
)
from orchestrators.llm_clients import DummyLLMClient  # noqa: E402
from synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)


class _FakeNeo4j:
    def __init__(self, upstream_pairs: List[Tuple[str, str]], edge_type: str = "containment") -> None:
        self._pairs = set(upstream_pairs)
        self._edge_type = edge_type

    def query(self, query: str, params: Dict[str, Any], db: Optional[str] = None):
        a = params.get("cid_a")
        b = params.get("cid_b")
        if "collect(DISTINCT type(rel))" in query:
            if (a, b) in self._pairs:
                if self._edge_type == "containment":
                    return [{"rel_types": ["has_part_usage"]}]
                return [{"rel_types": ["owns_port_usage", "connects_port"]}]
            return [{"rel_types": []}]
        return [{"reachable": (a, b) in self._pairs}]


def _kg() -> Dict[str, Any]:
    return {
        "components": [
            {"component_id": "C-A", "monitored_variable_ids": ["S-A"]},
            {"component_id": "C-B", "monitored_variable_ids": ["S-B"]},
        ],
        "failure_modes": [
            {"fm_id": "FM-A", "component_id": "C-A"},
            {"fm_id": "FM-B", "component_id": "C-B"},
        ],
    }


def _telemetry(a_start: str, a_end: str, b_start: str, b_end: str) -> Dict[str, Any]:
    return {
        "asset_id": "ASSET-1",
        "signals": [
            {"sensor_id": "S-A", "anomalies": [
                {"timestamp_start": a_start, "timestamp_end": a_end, "pattern": "spike", "severity": 0.8}]},
            {"sensor_id": "S-B", "anomalies": [
                {"timestamp_start": b_start, "timestamp_end": b_end, "pattern": "spike", "severity": 0.8}]},
        ],
    }


# ── (b)+(c) builder: initiator scored by path_score, co-temporal discounted ──

def test_clean_precedes_root_scored_by_path_score_not_flat_one():
    """A clean PRECEDES root (A ends well before B) is scored by path_score, < 1.0."""
    out = build_signal_evidence(
        run_id="RUN-P5-1",
        event={"timestamp_start": "2026-01-01T12:00:00+00:00"},
        telemetry_summary=_telemetry(
            "2026-01-01T08:00:00+00:00", "2026-01-01T08:30:00+00:00",
            "2026-01-01T10:00:00+00:00", "2026-01-01T10:30:00+00:00"),
        kg_context=_kg(),
        neo4j_client=_FakeNeo4j([("C-A", "C-B")], edge_type="containment"),
        historian_adapter=NullHistorianAdapter(),
    )
    fm_a = out["per_candidate_chain_score"]["FM-A"]
    assert fm_a["position_type"] == "root"
    assert fm_a["initiator_lag_established"] is True
    # scored by chain path_score, not the old flat 1.0
    assert fm_a["chain_position_score"] == fm_a["best_chain_path_score"]
    assert 0.0 < fm_a["chain_position_score"] < 1.0


def test_cotemporal_overlaps_root_discounted_and_flagged():
    """An OVERLAPS root (co-temporal, no clean lead) is discounted and marked unestablished."""
    out = build_signal_evidence(
        run_id="RUN-P5-2",
        event={"timestamp_start": "2026-01-01T12:00:00+00:00"},
        # A overlaps B: A starts before B and ends within B's interval.
        telemetry_summary=_telemetry(
            "2026-01-01T08:00:00+00:00", "2026-01-01T09:15:00+00:00",
            "2026-01-01T09:00:00+00:00", "2026-01-01T10:00:00+00:00"),
        kg_context=_kg(),
        neo4j_client=_FakeNeo4j([("C-A", "C-B")], edge_type="connectivity"),
        historian_adapter=NullHistorianAdapter(),
    )
    fm_a = out["per_candidate_chain_score"]["FM-A"]
    assert fm_a["position_type"] == "root"
    assert fm_a["initiator_lag_established"] is False
    # discounted below the raw chain path_score by the co-temporal factor
    expected = round(fm_a["best_chain_path_score"] * _COTEMPORAL_INITIATOR_FACTOR, 6)
    assert abs(fm_a["chain_position_score"] - expected) < 1e-9
    assert fm_a["chain_position_score"] < fm_a["best_chain_path_score"]


# ── (a) engine mapping + refine wiring ──────────────────────────────────────

def test_signal_dag_position_mapping():
    m = RuleBasedCausalityEngineV32._chain_position_from_signal_dag
    assert m("root") == "initiating"
    assert m("common_cause_root") == "initiating"
    assert m("convergence_confluence") == "consequence"
    assert m("intermediate") == "contributing"
    assert m("absent") is None
    assert m(None) is None


def _engine() -> RuleBasedCausalityEngineV32:
    return RuleBasedCausalityEngineV32(
        CausalityEngineConfigV32(
            minimum_evidence_threshold=0.0,
            minimum_pre_evidence_threshold=0.0,
            minimum_composite_threshold=0.0,
            top_k_candidates=5,
        )
    )


def _candidates() -> Dict[str, Any]:
    return {
        "candidates": [{
            "candidate_id": "FM::FM-A",
            "hypothesis_type": "failure_mode",
            "cause_node_id": "FM-A",
            "scores": {"evidence": 0.2, "governance": 0.5},
            "composite_score": 0.5,
        }],
        "summary": {},
    }


def _bundle() -> Dict[str, Any]:
    return {"candidate_evidence_summary": [{
        "candidate_id": "FM::FM-A", "best_support_score": 0.8,
        "best_contradiction_score": 0.0, "best_context_score": 0.2, "hit_count": 2,
    }]}


def test_refine_wires_signal_dag_fields_onto_candidate():
    out = _engine().refine_with_evidence(
        causality_candidates=_candidates(),
        evidence_bundle=_bundle(),
        kg_context={"failure_modes": []},
        signal_evidence={"per_candidate_chain_score": {"FM-A": {
            "chain_position_score": 0.6,
            "position_type": "root",
            "best_chain_path_score": 0.6,
            "initiator_lag_established": True,
        }}},
    )
    scores = out["candidates"][0]["scores"]
    assert scores["signal_dag_position_type"] == "root"
    assert scores["signal_dag_chain_position"] == "initiating"
    assert scores["signal_dag_initiator_lag_established"] is True
    assert scores["signal_dag_path_score"] == 0.6


def test_refine_wires_convergence_as_consequence_and_zeroes_chain():
    out = _engine().refine_with_evidence(
        causality_candidates=_candidates(),
        evidence_bundle=_bundle(),
        kg_context={"failure_modes": []},
        signal_evidence={"per_candidate_chain_score": {"FM-A": {
            "chain_position_score": 0.9,
            "position_type": "convergence_confluence",
        }}},
    )
    scores = out["candidates"][0]["scores"]
    assert scores["signal_dag_position_type"] == "convergence_confluence"
    assert scores["signal_dag_chain_position"] == "consequence"
    # convergence evidence remains zeroed (pre-existing contract)
    assert float(scores["evidence_chain"]) == 0.0


# ── synthesizer analyst flag ────────────────────────────────────────────────

def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


def _card_cc(scores: Dict[str, Any]):
    card = {
        "primary_hypothesis": {"candidate_id": "P", "uncertainties": []},
        "executive_summary": {"analyst_attention_flags": []},
    }
    cc = {"candidates": [{"candidate_id": "P", "scores": scores}]}
    return card, cc


def test_synth_flags_convergence_confluence_primary():
    s = _synth()
    card, cc = _card_cc({"signal_dag_position_type": "convergence_confluence"})
    s._apply_signal_dag_position_flag(card, cc)
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("convergence" in f.lower() for f in flags)
    assert card["primary_hypothesis"]["uncertainties"]


def test_synth_flags_unestablished_initiator_primary():
    s = _synth()
    card, cc = _card_cc({
        "signal_dag_position_type": "root",
        "signal_dag_initiator_lag_established": False,
    })
    s._apply_signal_dag_position_flag(card, cc)
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("lead was not established" in f.lower() for f in flags)


def test_synth_established_initiator_primary_no_flag():
    s = _synth()
    card, cc = _card_cc({
        "signal_dag_position_type": "root",
        "signal_dag_initiator_lag_established": True,
    })
    s._apply_signal_dag_position_flag(card, cc)
    assert card["executive_summary"]["analyst_attention_flags"] == []
    assert card["primary_hypothesis"]["uncertainties"] == []


def test_synth_absent_signal_dag_no_flag():
    s = _synth()
    card, cc = _card_cc({})
    s._apply_signal_dag_position_flag(card, cc)
    assert card["executive_summary"]["analyst_attention_flags"] == []
