"""
test_p8_llm_synthesis_regression_aug20.py — P-8 LLM-narrative-path regression.

`DummyLLMClient` always raises, so every ordinary dev/test run takes the
deterministic fallback; the `OllamaLLMClient` branch (LLM-output normalization,
`_validate_and_repair_llm_sections`, `_validate_card_semantics`, hallucination
hard-reject) had no golden-card regression (Phase-2 P-8 / June F-12).

This harness closes that gap in two layers:

  A. **Scripted-LLM golden regression (always runs).** A canned LLM output is
     driven through the full `synthesize()` LLM path with fixed inputs; the
     post-processed card is canonicalised (volatile ids/timestamps masked) and
     compared to a committed golden (`goldens/p8_llm_card_golden.json`), which is
     self-seeded on first run. Also asserts the deterministic post-processing
     invariants the review relies on — the LLM writes prose, but the *structure*
     (safety routing, human-performance block, CCF, and the WS2 chain-position /
     P-5 signal-DAG analyst flags) is applied deterministically on the LLM card,
     exactly as on the fallback card.

  B. **Live-Ollama semantic regression (opt-in, skipped by default).** When
     `RCA_LLM_GOLDEN=1` and a local Ollama is reachable, the *real* client is
     driven through `synthesize()` and the card must pass `_validate_card_semantics`.
     Skipped cleanly otherwise so CI never depends on a live LLM.

Run:  pytest test_p8_llm_synthesis_regression_aug20.py -v
Live: RCA_LLM_GOLDEN=1 pytest test_p8_llm_synthesis_regression_aug20.py -v
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)

_GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
_GOLDEN_PATH = _GOLDEN_DIR / "p8_llm_card_golden.json"

# Card fields that legitimately vary run-to-run (uuids, wall-clock) and must be
# masked before a golden comparison.
_VOLATILE_KEYS = {
    "rca_id", "run_id", "generated_at", "synthesized_at", "created_at",
    "timestamp", "timestamp_utc", "generated_timestamp",
}


# ── scripted LLM client ─────────────────────────────────────────────────────

class _ScriptedLLM:
    """Returns a fixed, valid LLM card (ignores the prompt) — deterministic."""

    def __init__(self, output: Optional[Dict[str, Any]]):
        self._output = output

    def generate_json(self, model, prompt, temperature=0.1):
        return copy.deepcopy(self._output)


# ── fixtures ────────────────────────────────────────────────────────────────

def _synth(llm_output) -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(
        llm_client=_ScriptedLLM(llm_output), config=RCASynthesizerConfig()
    )


def _event() -> Dict[str, Any]:
    return {"event_id": "EVT-P8", "id": "EVT-P8", "description": "Condenser vacuum degradation"}


def _run_context() -> Dict[str, Any]:
    return {"run_id": "RUN-P8"}


def _candidate(cid: str, label: str, score: float, chain_position: str = "initiating",
               scores: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_scores = {"structural": 0.8, "temporal": 0.6, "telemetry": 0.6,
                   "evidence": 0.5, "governance": 0.5}
    if scores:
        base_scores.update(scores)
    return {
        "candidate_id": cid,
        "cause_label": label,
        "hypothesis_type": "failure_mode",
        "cause_node_id": cid.replace("FM::", ""),
        "failure_mode_id": cid,
        "component_id": "C::CONDENSER",
        "composite_score": score,
        "confidence_label": "high" if score >= 0.75 else "medium",
        "chain_position": chain_position,
        "primary_causal_category": "A",
        "review_required": False,
        "scores": base_scores,
    }


def _causality(*candidates) -> Dict[str, Any]:
    return {"candidates": list(candidates), "summary": {}}


def _evidence_bundle() -> Dict[str, Any]:
    return {
        "bundle_id": "BND-P8",
        "results": [
            {"snippet_id": "SNIP-1", "doc_id": "WO-001", "snippet": "Active air in-leakage confirmed at flange."},
            {"snippet_id": "SNIP-2", "doc_id": "WO-002", "snippet": "Vacuum trend degraded over 6 hours."},
        ],
    }


def _good_llm_card(primary_id: str, cause_label: str = "Air in-leakage") -> Dict[str, Any]:
    return {
        "rca_id": "RCA::EVT-P8::scripted",
        "event_id": "EVT-P8",
        "executive_summary": {
            "decision_status": "primary_identified",
            "primary_conclusion": f"{cause_label} is the leading cause of the vacuum degradation.",
            "confidence_label": "medium",
            "analyst_attention_flags": [],
        },
        "primary_hypothesis": {
            "candidate_id": primary_id,
            "cause_label": cause_label,
            "hypothesis_type": "failure_mode",
            "narrative": "Ingress of air raised condenser back-pressure, degrading vacuum.",
            "why_primary": ["Highest composite score", "Direct telemetry corroboration"],
            "uncertainties": [],
            "composite_score": 0.82,
            "citations": [
                {"claim_summary": "Leakage confirmed", "source_type": "evidence_snippet",
                 "source_id": "SNIP-1", "excerpt": "Active air in-leakage confirmed at flange."}
            ],
        },
        "alternatives": [],
        "contributing_causes": [],
        "evidence": [
            {
                "evidence_id": "EV-001",
                "source_type": "evidence_snippet",
                "source_id": "SNIP-1",
                "doc_id": "WO-001",
                "support_role": "supporting",
                "summary": "Air in-leakage confirmed.",
                "excerpt": "Active air in-leakage confirmed at flange.",
                "linked_candidate_id": primary_id,
            },
            {
                "evidence_id": "EV-002",
                "source_type": "evidence_snippet",
                "source_id": "SNIP-2",
                "doc_id": "WO-002",
                "support_role": "supporting",
                "summary": "Vacuum trend degraded.",
                "excerpt": "Vacuum trend degraded over 6 hours.",
                "linked_candidate_id": primary_id,
            },
        ],
        "recommended_actions": [
            {"action_id": "A001", "action_type": "corrective", "description": "Locate and seal air in-leakage.",
             "priority": "high", "linked_candidate_id": primary_id},
        ],
        "analyst_review": {
            "decision_required": False,
            "writeback_recommendation": "hold_until_review",
            "questions_to_resolve": [],
        },
    }


def _synthesize(s: RuleValidatedRCASynthesizerV31, causality) -> Dict[str, Any]:
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
        run_context=_run_context(),
    )


# ── A. scripted-LLM golden + deterministic-structure parity ─────────────────

def test_llm_path_full_llm_and_deterministic_structure():
    """The scripted LLM card is accepted (full_llm) and carries the deterministic blocks."""
    primary = _candidate("FM::AIR-INLEAK", "Air in-leakage", 0.82)
    s = _synth(_good_llm_card("FM::AIR-INLEAK"))
    card = _synthesize(s, _causality(primary))

    vs = card["validation_status"]
    assert vs["fallback_used"] is False
    assert vs["synthesis_quality"] == "full_llm"
    assert vs["schema_valid"] is True, vs.get("validation_errors")
    # LLM only writes prose; structure is injected deterministically:
    assert "human_performance_assessment" in card
    assert card["primary_hypothesis"]["candidate_id"] == "FM::AIR-INLEAK"


def test_llm_card_chain_position_flag_applied_deterministically():
    """A consequence-as-primary LLM card gets the WS2 chain-position flag injected."""
    consequence = _candidate("FM::VACUUM-LOSS", "Vacuum loss", 0.85, chain_position="consequence")
    initiator = _candidate("FM::AIR-INLEAK", "Air in-leakage", 0.60, chain_position="initiating")
    s = _synth(_good_llm_card("FM::VACUUM-LOSS", cause_label="Vacuum loss"))
    card = _synthesize(s, _causality(consequence, initiator))

    assert card["validation_status"]["fallback_used"] is False  # stayed on LLM path
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("consequence" in f.lower() for f in flags), flags
    assert any("FM::AIR-INLEAK" in f for f in flags), "flag should point to the upstream initiator"


def test_llm_card_signal_dag_flag_applied_deterministically():
    """A primary at a signal-DAG convergence node gets the P-5 analyst flag on the LLM card."""
    primary = _candidate(
        "FM::VACUUM-LOSS", "Vacuum loss", 0.82, chain_position="contributing",
        scores={"signal_dag_position_type": "convergence_confluence"},
    )
    s = _synth(_good_llm_card("FM::VACUUM-LOSS", cause_label="Vacuum loss"))
    card = _synthesize(s, _causality(primary))

    assert card["validation_status"]["fallback_used"] is False
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("convergence" in f.lower() for f in flags), flags


def _canonicalize(obj: Any) -> Any:
    """Recursively drop volatile keys so golden comparison is deterministic."""
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_canonicalize(v) for v in obj]
    return obj


def test_llm_card_golden_snapshot():
    """Full LLM-path card matches the committed golden (self-seeds on first run)."""
    primary = _candidate("FM::AIR-INLEAK", "Air in-leakage", 0.82)
    s = _synth(_good_llm_card("FM::AIR-INLEAK"))
    card = _synthesize(s, _causality(primary))
    canonical = _canonicalize(card)
    serialized = json.dumps(canonical, indent=2, sort_keys=True, default=str)

    if not _GOLDEN_PATH.exists():
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        _GOLDEN_PATH.write_text(serialized, encoding="utf-8")
        pytest.skip(f"seeded golden at {_GOLDEN_PATH} (rerun to compare)")

    golden = _GOLDEN_PATH.read_text(encoding="utf-8")
    assert serialized == golden, (
        "LLM-path card drifted from committed golden "
        f"({_GOLDEN_PATH.name}). If intentional, delete the golden and re-run to reseed."
    )


# ── B. opt-in live Ollama semantic regression ───────────────────────────────

def _ollama_reachable(base_url: str = "http://localhost:11434") -> bool:
    try:
        import requests
        requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    os.environ.get("RCA_LLM_GOLDEN") != "1",
    reason="live-Ollama regression is opt-in (set RCA_LLM_GOLDEN=1)",
)
def test_ollama_llm_semantic_regression():
    base_url = os.environ.get("RCA_OLLAMA_URL", "http://localhost:11434")
    if not _ollama_reachable(base_url):
        pytest.skip(f"Ollama not reachable at {base_url}")

    from orchestrators.llm_clients import OllamaLLMClient

    primary = _candidate("FM::AIR-INLEAK", "Air in-leakage", 0.82)
    s = RuleValidatedRCASynthesizerV31(
        llm_client=OllamaLLMClient(base_url=base_url), config=RCASynthesizerConfig()
    )
    card = _synthesize(s, _causality(primary))

    # Whatever the LLM produced (or the fallback it forced), the emitted card must
    # be semantically valid and structurally complete.
    errors = s._validate_card_semantics(card)
    assert errors == [], f"live-LLM card failed semantic validation: {errors}"
    assert card["primary_hypothesis"]["candidate_id"] in {"FM::AIR-INLEAK", "NONE"}
    assert "human_performance_assessment" in card
