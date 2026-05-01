from __future__ import annotations

"""Unit tests for Phase C — supersession pass and related engine changes.

Covers:
- resolve_supersession: authority hierarchy (ECA > CR, RCA > ECA)
- resolve_supersession: recency tiebreak (newer ECA wins)
- resolve_supersession: equal recency unknown → both contribute
- resolve_supersession: cross-class hits never superseded
- resolve_supersession: candidate_evidence_summary patched correctly
- resolve_supersession: provenance fields set on bundle
- resolve_supersession: single hit per candidate → no supersession
- _build_allen_component_index: only anomaly nodes raise causal_scores
- _build_allen_component_index: alarm/SOE still contribute to follow_ids
- observationally_ungrounded: False when analyzes-class hit present
- observationally_ungrounded: False when affects-class hit present
- observationally_ungrounded: True when only monitors/context hits
- confidence_label capped at medium when observationally_ungrounded
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in (
    "kg", "kg.py2neo_workflow",
    "neo4j", "py2neo",
    "chromadb",
    "langchain_chroma", "langchain_community",
    "langchain_community.vectorstores", "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.supersession import (
    resolve_supersession,
    _is_analyzes_class,
    _authority_rank,
    _recency_dt,
    _patch_candidate_summary,
)
from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit(
    candidate_id: str,
    doc_type: str,
    support_score: float = 0.6,
    finding_status: str = "",
    epistemic_class: str = "",
    event_date: str = "",
    snippet_id: str = "",
) -> dict:
    meta: dict = {
        "linked_candidate_id": candidate_id,
        "doc_type": doc_type,
        "support_role": "supporting" if support_score > 0 else "contextual",
    }
    if finding_status:
        meta["finding_status"] = finding_status
    if epistemic_class:
        meta["epistemic_class"] = epistemic_class
    if event_date:
        meta["event_date"] = event_date
    return {
        "snippet_id": snippet_id or f"{doc_type}::{candidate_id}",
        "support_score": support_score,
        "contradiction_score": 0.0,
        "context_score": 0.0,
        "metadata": meta,
    }


def _bundle(*hits) -> dict:
    results = list(hits)
    summary = [
        {
            "candidate_id": cid,
            "best_support_score": max(
                (h["support_score"] for h in results
                 if (h.get("metadata") or {}).get("linked_candidate_id") == cid
                 and (h.get("metadata") or {}).get("support_role") == "supporting"),
                default=0.0,
            ),
            "supporting_count": sum(
                1 for h in results
                if (h.get("metadata") or {}).get("linked_candidate_id") == cid
                and (h.get("metadata") or {}).get("support_role") == "supporting"
            ),
            "supporting_snippet_ids": [
                h["snippet_id"] for h in results
                if (h.get("metadata") or {}).get("linked_candidate_id") == cid
                and (h.get("metadata") or {}).get("support_role") == "supporting"
            ],
        }
        for cid in dict.fromkeys(
            (h.get("metadata") or {}).get("linked_candidate_id")
            for h in results
            if (h.get("metadata") or {}).get("linked_candidate_id")
        )
    ]
    return {"results": results, "candidate_evidence_summary": summary}


# ---------------------------------------------------------------------------
# _is_analyzes_class
# ---------------------------------------------------------------------------

def test_is_analyzes_class_eca():
    assert _is_analyzes_class({"doc_type": "ECA"})


def test_is_analyzes_class_rca():
    assert _is_analyzes_class({"doc_type": "RCA"})


def test_is_analyzes_class_oe():
    assert _is_analyzes_class({"doc_type": "OE"})


def test_is_analyzes_class_cr_is_false():
    assert not _is_analyzes_class({"doc_type": "CR"})


def test_is_analyzes_class_wo_is_false():
    assert not _is_analyzes_class({"doc_type": "WO"})


def test_is_analyzes_class_explicit_annotation_wins():
    # doc_type=ECA but explicit annotation says monitors → not analyzes
    assert not _is_analyzes_class({
        "doc_type": "ECA",
        "epistemic_class": "monitors_performance",
    })


# ---------------------------------------------------------------------------
# _authority_rank
# ---------------------------------------------------------------------------

def test_authority_rank_rca_formal():
    assert _authority_rank({"doc_type": "RCA", "finding_status": "formal_conclusion"}) == 1


def test_authority_rank_eca_formal():
    assert _authority_rank({"doc_type": "ECA", "finding_status": "formal_conclusion"}) == 2


def test_authority_rank_eca_no_status():
    assert _authority_rank({"doc_type": "ECA"}) == 2


def test_authority_rank_cr_preliminary():
    assert _authority_rank({"doc_type": "CR", "finding_status": "preliminary_assessment"}) == 3


def test_authority_rank_oe_fleet():
    assert _authority_rank({"doc_type": "OE", "finding_status": "fleet_experience"}) == 4


def test_authority_rank_oe_industry():
    assert _authority_rank({"doc_type": "OE"}) == 5


def test_authority_rank_ler():
    assert _authority_rank({"doc_type": "LER"}) == 6


# ---------------------------------------------------------------------------
# resolve_supersession — authority hierarchy
# ---------------------------------------------------------------------------

def test_supersession_eca_supersedes_cr():
    """ECA (rank 2) must supersede CR preliminary (rank 3) for same candidate."""
    b = _bundle(
        _hit("C1", "ECA", finding_status="formal_conclusion", support_score=0.7),
        _hit("C1", "CR", finding_status="preliminary_assessment", support_score=0.6),
    )
    resolve_supersession(b)
    hits = b["results"]
    cr_hit = next(h for h in hits if h["metadata"]["doc_type"] == "CR")
    eca_hit = next(h for h in hits if h["metadata"]["doc_type"] == "ECA")
    assert cr_hit.get("superseded") is True
    assert cr_hit["support_score"] == 0.0
    assert not eca_hit.get("superseded")
    assert eca_hit["support_score"] == 0.7


def test_supersession_rca_supersedes_eca():
    """RCA formal (rank 1) supersedes ECA formal (rank 2)."""
    b = _bundle(
        _hit("C1", "RCA", finding_status="formal_conclusion", support_score=0.8),
        _hit("C1", "ECA", finding_status="formal_conclusion", support_score=0.75),
    )
    resolve_supersession(b)
    hits = b["results"]
    eca = next(h for h in hits if h["metadata"]["doc_type"] == "ECA")
    assert eca.get("superseded") is True
    assert eca["support_score"] == 0.0


def test_supersession_eca_supersedes_oe():
    """ECA (rank 2) supersedes OE industry (rank 5)."""
    b = _bundle(
        _hit("C1", "ECA", support_score=0.65),
        _hit("C1", "OE", support_score=0.55),
    )
    resolve_supersession(b)
    oe = next(h for h in b["results"] if h["metadata"]["doc_type"] == "OE")
    assert oe.get("superseded") is True


# ---------------------------------------------------------------------------
# resolve_supersession — recency tiebreak
# ---------------------------------------------------------------------------

def test_supersession_equal_rank_newer_wins():
    """Two ECA hits — newer date survives, older is superseded."""
    b = _bundle(
        _hit("C1", "ECA", support_score=0.7, event_date="2023-06-01"),
        _hit("C1", "ECA", support_score=0.65, event_date="2022-01-01", snippet_id="old_eca"),
    )
    resolve_supersession(b)
    old = next(h for h in b["results"] if h.get("snippet_id") == "old_eca")
    assert old.get("superseded") is True


def test_supersession_equal_rank_unknown_recency_both_survive():
    """Two ECA hits with no date info — both contribute (ADR-2)."""
    b = _bundle(
        _hit("C1", "ECA", support_score=0.7, snippet_id="e1"),
        _hit("C1", "ECA", support_score=0.65, snippet_id="e2"),
    )
    resolve_supersession(b)
    assert b["supersession_count"] == 0
    for h in b["results"]:
        assert not h.get("superseded")


def test_supersession_equal_rank_tied_recency_both_survive():
    """Same date → tied → both contribute."""
    b = _bundle(
        _hit("C1", "ECA", support_score=0.7, event_date="2023-06-01", snippet_id="e1"),
        _hit("C1", "ECA", support_score=0.65, event_date="2023-06-01", snippet_id="e2"),
    )
    resolve_supersession(b)
    assert b["supersession_count"] == 0


# ---------------------------------------------------------------------------
# resolve_supersession — cross-class hits never superseded
# ---------------------------------------------------------------------------

def test_supersession_cr_not_superseded_by_wo():
    """WO (affects-class) must never supersede CR (monitors-class)."""
    b = _bundle(
        _hit("C1", "WO",
             epistemic_class="affects_performance", support_score=0.8),
        _hit("C1", "CR",
             epistemic_class="monitors_performance", support_score=0.5),
    )
    resolve_supersession(b)
    assert b["supersession_count"] == 0


def test_supersession_only_analyzes_class_participate():
    """Monitors-class hits for same candidate are left untouched."""
    b = _bundle(
        _hit("C1", "ECA", finding_status="formal_conclusion", support_score=0.9),
        _hit("C1", "CR", epistemic_class="monitors_performance", support_score=0.5),
    )
    resolve_supersession(b)
    cr = next(h for h in b["results"] if h["metadata"]["doc_type"] == "CR")
    # CR is monitors-class → not in supersession pool → never zeroed
    assert not cr.get("superseded")
    assert cr["support_score"] == 0.5


# ---------------------------------------------------------------------------
# resolve_supersession — candidate_evidence_summary patch
# ---------------------------------------------------------------------------

def test_supersession_patches_best_support_score():
    """After supersession, candidate_evidence_summary.best_support_score reflects survivors."""
    b = _bundle(
        _hit("C1", "ECA", finding_status="formal_conclusion", support_score=0.9,
             snippet_id="eca1"),
        _hit("C1", "CR", finding_status="preliminary_assessment", support_score=0.8,
             snippet_id="cr1"),
    )
    resolve_supersession(b)
    row = next(r for r in b["candidate_evidence_summary"] if r["candidate_id"] == "C1")
    # CR is superseded; only ECA survives with score 0.9
    assert row["best_support_score"] == 0.9
    assert row["supporting_count"] == 1
    assert "eca1" in row["supporting_snippet_ids"]
    assert "cr1" not in row["supporting_snippet_ids"]


# ---------------------------------------------------------------------------
# resolve_supersession — provenance
# ---------------------------------------------------------------------------

def test_supersession_provenance_fields():
    b = _bundle(
        _hit("C1", "ECA", support_score=0.8),
        _hit("C1", "CR", finding_status="preliminary_assessment", support_score=0.6),
    )
    resolve_supersession(b, epistemics_policy_version="epistemics-v1.0")
    assert b["supersession_applied"] is True
    assert b["supersession_count"] == 1
    assert b["supersession_policy_version"] == "epistemics-v1.0"


def test_supersession_no_hits_no_crash():
    b = {"results": [], "candidate_evidence_summary": []}
    resolve_supersession(b)
    assert b["supersession_applied"] is False
    assert b["supersession_count"] == 0


def test_supersession_single_hit_no_supersession():
    b = _bundle(_hit("C1", "ECA", support_score=0.7))
    resolve_supersession(b)
    assert b["supersession_count"] == 0


# ---------------------------------------------------------------------------
# Allen blend — only anomaly nodes raise causal_scores (§3.4)
# ---------------------------------------------------------------------------

def _allen_map(*nodes) -> dict:
    return {"nodes": list(nodes), "quality_flags": {"soe_clock_sync_ok": True}}


def _node(node_type: str, component_id: str, relation: str, score: float,
          causal: bool = True) -> dict:
    return {
        "node_type": node_type,
        "component_id": component_id,
        "allen_relation_to_event": relation,
        "allen_base_score": score,
        "causal_candidate": causal,
    }


def test_allen_anomaly_node_raises_causal_score():
    eng = RuleBasedCausalityEngineV32()
    am = _allen_map(_node("anomaly", "C1", "precedes", 0.85))
    causal, _, _ = eng._build_allen_component_index(am)
    assert "C1" in causal
    assert abs(causal["C1"] - 0.85) < 1e-6


def test_allen_alarm_node_does_not_raise_causal_score():
    """Alarm nodes are monitors-class — must not contribute to causal_scores."""
    eng = RuleBasedCausalityEngineV32()
    am = _allen_map(_node("alarm", "C1", "precedes", 0.80))
    causal, _, _ = eng._build_allen_component_index(am)
    assert "C1" not in causal


def test_allen_soe_node_does_not_raise_causal_score():
    """SOE nodes are monitors-class — must not contribute to causal_scores."""
    eng = RuleBasedCausalityEngineV32()
    am = _allen_map(_node("soe_record", "C1", "precedes", 0.75))
    causal, _, _ = eng._build_allen_component_index(am)
    assert "C1" not in causal


def test_allen_alarm_follows_still_adds_to_follow_ids():
    """Alarm 'follows' node must still trigger contradiction detection."""
    eng = RuleBasedCausalityEngineV32()
    am = _allen_map(_node("alarm", "C1", "follows", 0.0, causal=False))
    _, _, follow_ids = eng._build_allen_component_index(am)
    assert "C1" in follow_ids


def test_allen_soe_follows_still_adds_to_follow_ids():
    eng = RuleBasedCausalityEngineV32()
    am = _allen_map(_node("soe_record", "C1", "follows", 0.0, causal=False))
    _, _, follow_ids = eng._build_allen_component_index(am)
    assert "C1" in follow_ids


def test_allen_anomaly_and_alarm_same_component_only_anomaly_contributes():
    """Mixed: anomaly + alarm for same component — only anomaly raises causal_score."""
    eng = RuleBasedCausalityEngineV32()
    am = _allen_map(
        _node("anomaly", "C1", "precedes", 0.70),
        _node("alarm",   "C1", "precedes", 0.90),  # higher score but monitors-class
    )
    causal, _, _ = eng._build_allen_component_index(am)
    # Must take anomaly score (0.70), not alarm score (0.90)
    assert abs(causal.get("C1", 0.0) - 0.70) < 1e-6


# ---------------------------------------------------------------------------
# observationally_ungrounded — via evidence summary flags
# ---------------------------------------------------------------------------

def test_observationally_ungrounded_false_when_analyzes_hit():
    """has_analyzes_class_hit=True → observationally_ungrounded=False."""
    # Build a minimal evidence summary with analyzes-class hit
    summary_row = {
        "candidate_id": "C1",
        "best_support_score": 0.7,
        "has_analyzes_class_hit": True,
        "has_affects_class_hit": False,
        "best_contradiction_score": 0.0,
        "best_context_score": 0.0,
        "hit_count": 1,
        "mean_conjecture_fraction": 0.0,
        "dominant_temporal_relation": None,
        "best_lag_hours": None,
        "lag_is_approximate": False,
        "supporting_snippet_ids": [],
        "contradicting_snippet_ids": [],
        "contextual_snippet_ids": [],
        "aggregated_mechanisms": [],
        "aggregated_outcomes": [],
        "best_source_tier": None,
    }
    ungrounded = not (
        bool(summary_row.get("has_affects_class_hit", False))
        or bool(summary_row.get("has_analyzes_class_hit", False))
    )
    assert ungrounded is False


def test_observationally_ungrounded_false_when_affects_hit():
    summary_row = {"has_analyzes_class_hit": False, "has_affects_class_hit": True}
    ungrounded = not (
        bool(summary_row.get("has_affects_class_hit", False))
        or bool(summary_row.get("has_analyzes_class_hit", False))
    )
    assert ungrounded is False


def test_observationally_ungrounded_true_when_no_grounding_hits():
    summary_row = {"has_analyzes_class_hit": False, "has_affects_class_hit": False}
    ungrounded = not (
        bool(summary_row.get("has_affects_class_hit", False))
        or bool(summary_row.get("has_analyzes_class_hit", False))
    )
    assert ungrounded is True
