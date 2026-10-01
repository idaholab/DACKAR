"""
test_n4_score_interpretation_aug20.py — N-4 composite-score honesty.

`composite_score` is a weighted blend of heuristic sub-scores with hand-set
weights and relation priors; it is NOT calibrated against outcome frequencies,
yet it reads like a probability (0.72 looks like "72% likely"). N-4 adds an
additive, ranking-neutral `score_interpretation` card block that states the
score is a non-probabilistic ordinal ranking number and that any confidence
interval encodes data availability, not statistical uncertainty. This block is
injected on both the LLM and deterministic-fallback paths.

Run:  pytest test_n4_score_interpretation_aug20.py -v
"""
from __future__ import annotations

import copy

from dackar.RCA.orchestrators.llm_clients import DummyLLMClient  # noqa: E402
from dackar.RCA.synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)


def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


def test_score_interpretation_declares_non_probabilistic():
    block = _synth()._build_score_interpretation()
    assert block["score_type"] == "ordinal_ranking"
    assert block["is_probability"] is False
    assert block["is_calibrated"] is False


def test_score_interpretation_note_warns_against_percentage_reading():
    block = _synth()._build_score_interpretation()
    note = block["note"].lower()
    assert "not" in note and ("probability" in note or "likelihood" in note)
    assert "confidence_label" in block["note"]  # points analyst to the ordinal label


def test_interval_meaning_is_data_availability_not_statistical():
    block = _synth()._build_score_interpretation()
    meaning = block["interval_meaning"].lower()
    assert "availability" in meaning or "degrad" in meaning
    assert "not" in meaning and ("statistical" in meaning or "sampling" in meaning)


def test_schema_shape_is_complete_and_bounded():
    block = _synth()._build_score_interpretation()
    for key in ("score_type", "is_probability", "is_calibrated", "note"):
        assert key in block
    assert set(block).issubset(
        {"score_type", "is_probability", "is_calibrated", "interval_meaning", "note"}
    )


def test_block_is_constant_and_ranking_neutral():
    # Two independent builds must be identical (no run-dependent / score-dependent content).
    assert _synth()._build_score_interpretation() == _synth()._build_score_interpretation()


# ── public synthesize() wiring ──────────────────────────────────────────────
#
# The tests above lock the *content* of the block by calling the private
# builder directly. These drive the full public synthesize() so a regression
# that stops attaching the block on either path (LLM or deterministic fallback)
# is caught, and confirm the additive block is ranking-neutral: the primary
# hypothesis is unchanged and the caller's candidates are not mutated.

class _ScriptedLLM:
    """Returns a fixed, valid LLM card (ignores the prompt)."""

    def __init__(self, output):
        self._output = output

    def generate_json(self, model, prompt, temperature=0.1):
        return copy.deepcopy(self._output)


def _event():
    return {"event_id": "EVT-N4", "id": "EVT-N4", "description": "Condenser vacuum degradation"}


def _candidate(cid: str, label: str, score: float) -> dict:
    return {
        "candidate_id": cid,
        "cause_label": label,
        "hypothesis_type": "failure_mode",
        "cause_node_id": cid.replace("FM::", ""),
        "failure_mode_id": cid,
        "component_id": "C::CONDENSER",
        "composite_score": score,
        "confidence_label": "high" if score >= 0.75 else "medium",
        "chain_position": "initiating",
        "primary_causal_category": "A",
        "review_required": False,
        "scores": {"structural": 0.8, "temporal": 0.6, "telemetry": 0.6,
                   "evidence": 0.5, "governance": 0.5},
    }


def _causality(*cands) -> dict:
    return {"candidates": list(cands), "summary": {}}


def _evidence_bundle() -> dict:
    return {
        "bundle_id": "BND-N4",
        "results": [
            {"snippet_id": "SNIP-1", "doc_id": "WO-001",
             "snippet": "Active air in-leakage confirmed at flange."},
        ],
    }


def _valid_llm_card(primary_id: str, label: str = "Air in-leakage") -> dict:
    """A minimal LLM card that passes validation and stays on the LLM path."""
    return {
        "event_id": "EVT-N4",
        "executive_summary": {
            "decision_status": "primary_identified",
            "primary_conclusion": f"{label} is the leading cause of the vacuum degradation.",
            "confidence_label": "medium",
            "analyst_attention_flags": [],
        },
        "primary_hypothesis": {
            "candidate_id": primary_id, "cause_label": label,
            "hypothesis_type": "failure_mode",
            "narrative": "Ingress of air raised condenser back-pressure, degrading vacuum.",
            "why_primary": ["Highest composite score"],
            "uncertainties": [], "composite_score": 0.82,
            "citations": [{"claim_summary": "Leakage confirmed", "source_type": "evidence_snippet",
                           "source_id": "SNIP-1", "excerpt": "Active air in-leakage confirmed at flange."}],
        },
        "alternatives": [], "contributing_causes": [],
        "evidence": [{"evidence_id": "EV-001", "source_type": "evidence_snippet", "source_id": "SNIP-1",
                      "doc_id": "WO-001", "support_role": "supporting",
                      "summary": "Air in-leakage confirmed.",
                      "excerpt": "Active air in-leakage confirmed at flange.",
                      "linked_candidate_id": primary_id}],
        "recommended_actions": [{"action_id": "A001", "action_type": "corrective",
                                 "description": "Locate and seal air in-leakage.", "priority": "high",
                                 "linked_candidate_id": primary_id}],
        "analyst_review": {"decision_required": False, "writeback_recommendation": "hold_until_review",
                           "questions_to_resolve": []},
    }


def _synthesize(s: RuleValidatedRCASynthesizerV31, causality: dict) -> dict:
    return s.synthesize(
        event=_event(),
        telemetry_summary={},
        kg_context={},
        tskr_patterns=None,
        causality_candidates=causality,
        evidence_bundle=_evidence_bundle(),
        operational_context=None,
        pm_compliance=None,
        ishikawa_matrix=None,
        run_context={"run_id": "RUN-N4"},
    )


def test_synthesize_fallback_path_attaches_score_interpretation():
    """Deterministic fallback (DummyLLMClient raises): the public card carries the block."""
    causality = _causality(
        _candidate("FM::AIR-INLEAK", "Air in-leakage", 0.82),
        _candidate("FM::TUBE-FOUL", "Tube fouling", 0.55),
    )
    snapshot = copy.deepcopy(causality)
    s = RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())
    card = _synthesize(s, causality)

    assert card["validation_status"]["fallback_used"] is True
    assert card["score_interpretation"] == _synth()._build_score_interpretation()
    # Ranking-neutral: the additive block did not reorder or mutate the input.
    assert card["primary_hypothesis"]["candidate_id"] == "FM::AIR-INLEAK"
    assert causality == snapshot, "synthesize() mutated the caller's candidates"


def test_synthesize_llm_path_attaches_score_interpretation():
    """LLM path (scripted valid card): the public card still carries the block."""
    causality = _causality(_candidate("FM::AIR-INLEAK", "Air in-leakage", 0.82))
    s = RuleValidatedRCASynthesizerV31(
        llm_client=_ScriptedLLM(_valid_llm_card("FM::AIR-INLEAK")), config=RCASynthesizerConfig()
    )
    card = _synthesize(s, causality)

    assert card["validation_status"]["fallback_used"] is False
    assert card["score_interpretation"] == _synth()._build_score_interpretation()
    assert card["primary_hypothesis"]["candidate_id"] == "FM::AIR-INLEAK"
