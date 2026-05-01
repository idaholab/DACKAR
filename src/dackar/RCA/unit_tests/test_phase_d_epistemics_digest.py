from __future__ import annotations

"""Unit tests for Phase D — EpistemicsDigest builder and synthesizer enforcement.

Covers:
- build_epistemics_digests: analyzes_support_count from non-superseded analyzes hits
- build_epistemics_digests: superseded hits excluded from count but kept in items
- build_epistemics_digests: affects_support_present / affects_support_items
- build_epistemics_digests: authority_level mapping (mandatory/guidance/informational)
- build_epistemics_digests: degraded_classification_count (no Phase A annotation)
- build_epistemics_digests: causal_grounding_absent = True when analyzes_count == 0
- build_epistemics_digests: confidence_cap = "medium" when observationally_ungrounded
- build_epistemics_digests: confidence_cap = None when grounded
- build_epistemics_digests: multiple candidates — digests keyed by candidate_id
- build_epistemics_run_summary: class counts, supersession_count, resolution levels
- RCASynthesizer._cap_confidence: level ordering
- RCASynthesizer._apply_epistemics_postprocessing: caps confidence on card
- RCASynthesizer._apply_epistemics_postprocessing: sets causal_grounding_absent flag
- RCASynthesizer._apply_epistemics_postprocessing: adds gap-typed attention flags
- RCASynthesizer._apply_epistemics_postprocessing: no-op when no digest on candidate
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

from orchestrators.epistemics_digest import build_epistemics_digests, build_epistemics_run_summary
from synthesis.rca_synthesizer_v31 import RuleValidatedRCASynthesizerV31 as RCASynthesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit(
    candidate_id: str,
    doc_type: str,
    support_score: float = 0.6,
    finding_status: str = "",
    epistemic_class: str = "",
    superseded: bool = False,
    component_id: str = "",
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
    if component_id:
        meta["component_id"] = component_id
    h = {
        "snippet_id": snippet_id or f"{doc_type}::{candidate_id}",
        "support_score": support_score,
        "contradiction_score": 0.0,
        "context_score": 0.0,
        "metadata": meta,
    }
    if superseded:
        h["superseded"] = True
    return h


def _candidate(
    candidate_id: str,
    observationally_ungrounded: bool = False,
    confidence_label: str = "medium",
) -> dict:
    c = {
        "candidate_id": candidate_id,
        "component_id": candidate_id.replace("FM::", ""),
        "composite_score": 0.50,
        "confidence_label": confidence_label,
    }
    if observationally_ungrounded:
        c["observationally_ungrounded"] = True
    return c


def _candidates(*cands) -> dict:
    return {"candidates": list(cands)}


def _bundle(*hits) -> dict:
    return {"results": list(hits), "supersession_applied": False, "supersession_count": 0}


# ---------------------------------------------------------------------------
# build_epistemics_digests — analyzes hits
# ---------------------------------------------------------------------------

def test_digest_analyzes_count_non_superseded():
    """Only non-superseded analyzes hits count toward analyzes_support_count."""
    results = [
        _hit("C1", "ECA", finding_status="formal_conclusion"),          # non-superseded
        _hit("C1", "RCA", finding_status="formal_conclusion", superseded=True),  # superseded
    ]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    d = digests["C1"]
    assert d["analyzes_support_count"] == 1
    assert len(d["analyzes_support_items"]) == 2  # both in items list
    assert sum(1 for i in d["analyzes_support_items"] if not i["superseded"]) == 1


def test_digest_causal_grounding_absent_when_all_superseded():
    """All analyzes hits superseded → causal_grounding_absent = True."""
    results = [_hit("C1", "ECA", superseded=True)]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["causal_grounding_absent"] is True
    assert digests["C1"]["analyzes_support_count"] == 0


def test_digest_causal_grounding_absent_when_no_analyzes_hits():
    """No analyzes hits at all → causal_grounding_absent = True."""
    results = [_hit("C1", "WO")]  # affects class
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["causal_grounding_absent"] is True


def test_digest_causal_grounding_present_with_rca():
    """RCA formal hit → causal_grounding_absent = False, count = 1."""
    results = [_hit("C1", "RCA", finding_status="formal_conclusion")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    d = digests["C1"]
    assert d["causal_grounding_absent"] is False
    assert d["analyzes_support_count"] == 1


# ---------------------------------------------------------------------------
# build_epistemics_digests — affects hits
# ---------------------------------------------------------------------------

def test_digest_affects_present_from_wo():
    """WO hit → affects_support_present = True."""
    results = [_hit("C1", "WO", component_id="PUMP-01")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    d = digests["C1"]
    assert d["affects_support_present"] is True
    assert len(d["affects_support_items"]) == 1
    assert d["affects_support_items"][0]["component_id"] == "PUMP-01"


def test_digest_affects_absent_when_only_analyzes():
    """Only ECA hit → affects_support_present = False."""
    results = [_hit("C1", "ECA")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["affects_support_present"] is False


# ---------------------------------------------------------------------------
# build_epistemics_digests — authority_level mapping
# ---------------------------------------------------------------------------

def test_digest_authority_level_mandatory_rca_formal():
    results = [_hit("C1", "RCA", finding_status="formal_conclusion")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["analyzes_support_items"][0]["authority_level"] == "mandatory"


def test_digest_authority_level_mandatory_eca_formal():
    results = [_hit("C1", "ECA", finding_status="formal_conclusion")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["analyzes_support_items"][0]["authority_level"] == "mandatory"


def test_digest_authority_level_guidance_cr():
    results = [_hit("C1", "CR", finding_status="preliminary_assessment")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["analyzes_support_items"][0]["authority_level"] == "guidance"


def test_digest_authority_level_guidance_fleet_oe():
    results = [_hit("C1", "OE", finding_status="fleet_experience")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["analyzes_support_items"][0]["authority_level"] == "guidance"


def test_digest_authority_level_informational_oe():
    results = [_hit("C1", "OE")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["analyzes_support_items"][0]["authority_level"] == "informational"


def test_digest_authority_level_informational_ler():
    results = [_hit("C1", "LER")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["analyzes_support_items"][0]["authority_level"] == "informational"


# ---------------------------------------------------------------------------
# build_epistemics_digests — degraded_classification_count
# ---------------------------------------------------------------------------

def test_digest_degraded_count_zero_when_annotated():
    """Phase A annotation present → not counted as degraded."""
    results = [_hit("C1", "ECA", epistemic_class="analyzes_past_degradation")]
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["degraded_classification_count"] == 0


def test_digest_degraded_count_positive_when_no_annotation():
    """No Phase A annotation → doc_type fallback → degraded."""
    results = [_hit("C1", "ECA"), _hit("C1", "WO")]  # no epistemic_class set
    digests = build_epistemics_digests(_candidates(_candidate("C1")), results)
    assert digests["C1"]["degraded_classification_count"] == 2


# ---------------------------------------------------------------------------
# build_epistemics_digests — confidence_cap
# ---------------------------------------------------------------------------

def test_digest_confidence_cap_medium_when_ungrounded():
    cand = _candidate("C1", observationally_ungrounded=True)
    digests = build_epistemics_digests(_candidates(cand), [])
    assert digests["C1"]["confidence_cap"] == "medium"


def test_digest_confidence_cap_none_when_grounded():
    cand = _candidate("C1", observationally_ungrounded=False)
    digests = build_epistemics_digests(_candidates(cand), [])
    assert digests["C1"]["confidence_cap"] is None


# ---------------------------------------------------------------------------
# build_epistemics_digests — multiple candidates
# ---------------------------------------------------------------------------

def test_digest_multiple_candidates_keyed_by_id():
    results = [
        _hit("C1", "ECA"),
        _hit("C2", "WO"),
    ]
    digests = build_epistemics_digests(
        _candidates(_candidate("C1"), _candidate("C2")),
        results,
    )
    assert "C1" in digests
    assert "C2" in digests
    assert digests["C1"]["analyzes_support_count"] == 1
    assert digests["C2"]["affects_support_present"] is True


# ---------------------------------------------------------------------------
# build_epistemics_run_summary
# ---------------------------------------------------------------------------

def test_run_summary_class_counts():
    results = [
        _hit("C1", "ECA"),   # analyzes
        _hit("C1", "WO"),    # affects
        _hit("C1", "OE"),    # analyzes
    ]
    bundle = {"results": results, "supersession_count": 1}
    summary = build_epistemics_run_summary(
        _candidates(_candidate("C1")), results, bundle
    )
    assert summary["hit_counts_by_epistemic_class"]["analyzes_past_degradation"] == 2
    assert summary["hit_counts_by_epistemic_class"]["affects_performance"] == 1
    assert summary["supersession_edge_count"] == 1


def test_run_summary_resolution_levels():
    results = [
        _hit("C1", "ECA", epistemic_class="analyzes_past_degradation"),  # annotation
        _hit("C1", "OE"),  # doc_type_fallback
    ]
    bundle = {"results": results, "supersession_count": 0}
    summary = build_epistemics_run_summary(
        _candidates(_candidate("C1")), results, bundle
    )
    levels = summary["classification_resolution_level_distribution"]
    assert levels["annotation"] == 1
    assert levels["doc_type_fallback"] == 1


# ---------------------------------------------------------------------------
# RCASynthesizer._cap_confidence_label (existing helper, tested via postprocessing)
# ---------------------------------------------------------------------------

def test_cap_confidence_label_high_capped_to_medium():
    synth = _make_synthesizer()
    assert synth._cap_confidence_label("high", "medium") == "medium"


def test_cap_confidence_label_medium_unchanged():
    synth = _make_synthesizer()
    assert synth._cap_confidence_label("medium", "medium") == "medium"


def test_cap_confidence_label_low_not_raised_by_medium_cap():
    synth = _make_synthesizer()
    assert synth._cap_confidence_label("low", "medium") == "low"


def test_cap_confidence_label_speculative_unaffected():
    synth = _make_synthesizer()
    assert synth._cap_confidence_label("speculative", "medium") == "speculative"


# ---------------------------------------------------------------------------
# RCASynthesizer._apply_epistemics_postprocessing
# ---------------------------------------------------------------------------

def _make_card(candidate_id: str, confidence: str = "high") -> dict:
    return {
        "primary_hypothesis": {
            "candidate_id": candidate_id,
            "confidence_label": confidence,
        },
        "executive_summary": {
            "confidence_label": confidence,
            "analyst_attention_flags": [],
        },
    }


def _make_causality_candidates(candidate_id: str, digest: dict) -> dict:
    return {
        "candidates": [{
            "candidate_id": candidate_id,
            "epistemics_digest": digest,
        }]
    }


def _make_synthesizer() -> RCASynthesizer:
    llm_client = MagicMock()
    return RCASynthesizer(llm_client=llm_client)


def test_apply_epistemics_caps_primary_confidence():
    """confidence_cap = 'medium' → primary_hypothesis.confidence_label capped."""
    card = _make_card("C1", confidence="high")
    digest = {"confidence_cap": "medium", "causal_grounding_absent": False,
              "observationally_ungrounded": True}
    synth = _make_synthesizer()
    synth._apply_epistemics_postprocessing(card, _make_causality_candidates("C1", digest))
    assert card["primary_hypothesis"]["confidence_label"] == "medium"
    assert card["primary_hypothesis"].get("confidence_label_cap_reason") == "observationally_ungrounded"


def test_apply_epistemics_caps_executive_summary():
    """confidence_cap also caps executive_summary.confidence_label."""
    card = _make_card("C1", confidence="high")
    digest = {"confidence_cap": "medium", "causal_grounding_absent": False,
              "observationally_ungrounded": True}
    synth = _make_synthesizer()
    synth._apply_epistemics_postprocessing(card, _make_causality_candidates("C1", digest))
    assert card["executive_summary"]["confidence_label"] == "medium"


def test_apply_epistemics_no_cap_when_none():
    """confidence_cap = None → confidence_label unchanged."""
    card = _make_card("C1", confidence="high")
    digest = {"confidence_cap": None, "causal_grounding_absent": False,
              "observationally_ungrounded": False}
    synth = _make_synthesizer()
    synth._apply_epistemics_postprocessing(card, _make_causality_candidates("C1", digest))
    assert card["primary_hypothesis"]["confidence_label"] == "high"


def test_apply_epistemics_sets_causal_grounding_absent_flag():
    """causal_grounding_absent from digest is stamped on primary_hypothesis."""
    card = _make_card("C1")
    digest = {"confidence_cap": None, "causal_grounding_absent": True,
              "observationally_ungrounded": False}
    synth = _make_synthesizer()
    synth._apply_epistemics_postprocessing(card, _make_causality_candidates("C1", digest))
    assert card["primary_hypothesis"]["causal_grounding_absent"] is True


def test_apply_epistemics_sets_observationally_ungrounded_flag():
    card = _make_card("C1")
    digest = {"confidence_cap": "medium", "causal_grounding_absent": False,
              "observationally_ungrounded": True}
    synth = _make_synthesizer()
    synth._apply_epistemics_postprocessing(card, _make_causality_candidates("C1", digest))
    assert card["primary_hypothesis"]["observationally_ungrounded"] is True


def test_apply_epistemics_adds_ungrounded_attention_flag():
    """observationally_ungrounded → attention flag added to executive_summary."""
    card = _make_card("C1")
    digest = {"confidence_cap": "medium", "causal_grounding_absent": False,
              "observationally_ungrounded": True}
    synth = _make_synthesizer()
    synth._apply_epistemics_postprocessing(card, _make_causality_candidates("C1", digest))
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("causally ungrounded" in f for f in flags)


def test_apply_epistemics_adds_analyzes_gap_flag():
    """causal_grounding_absent → analyzes-gap attention flag added."""
    card = _make_card("C1")
    digest = {"confidence_cap": None, "causal_grounding_absent": True,
              "observationally_ungrounded": False}
    synth = _make_synthesizer()
    synth._apply_epistemics_postprocessing(card, _make_causality_candidates("C1", digest))
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("Analyzes gap" in f for f in flags)


def test_apply_epistemics_noop_when_no_digest():
    """No epistemics_digest on candidate → card unchanged."""
    card = _make_card("C1", confidence="high")
    cands = {"candidates": [{"candidate_id": "C1"}]}  # no digest
    synth = _make_synthesizer()
    synth._apply_epistemics_postprocessing(card, cands)
    assert card["primary_hypothesis"]["confidence_label"] == "high"
    assert "causal_grounding_absent" not in card["primary_hypothesis"]


def test_apply_epistemics_noop_when_candidate_not_found():
    """Primary candidate_id not in candidates list → card unchanged."""
    card = _make_card("C99", confidence="high")
    digest = {"confidence_cap": "medium", "causal_grounding_absent": True,
              "observationally_ungrounded": True}
    cands = _make_causality_candidates("C1", digest)  # different id
    synth = _make_synthesizer()
    synth._apply_epistemics_postprocessing(card, cands)
    assert card["primary_hypothesis"]["confidence_label"] == "high"
