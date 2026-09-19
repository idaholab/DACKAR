"""
mock_llm_clients.py — LLM client stubs for D11 synthesizer resilience tests.

Each class implements the ``LLMClient`` Protocol from ``orchestrators.llm_clients``:

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]: ...

Usage in tests
--------------
    from mock_llm_clients import MalformedLLMClient, TimeoutLLMClient, EmptyLLMClient

    orchestrator = build_fixture_orchestrator(tmp_path, llm_client=MalformedLLMClient())
    result = run_rca(orchestrator, fixtures)
    # synthesizer must have fallen back to the deterministic path
    assert result["rca_card"]["fallback_used"] is True

D11 check mapping
-----------------
    D11-A  WellFormedLLMClient   — baseline: LLM path succeeds, fallback_used=False
    D11-B  MalformedLLMClient    — structurally invalid card; synthesizer falls back
    D11-C  EmptyLLMClient        — empty dict; synthesizer falls back
    D11-D  TimeoutLLMClient      — exception during generation; synthesizer falls back

All four non-Well-Formed cases must satisfy:
    • orchestrator.run() completes without raising
    • result["rca_card"] is present and non-empty
    • result["rca_card"]["fallback_used"] is True
"""

from __future__ import annotations

import time
from typing import Any, Dict


# ---------------------------------------------------------------------------
# D11-A — baseline
# ---------------------------------------------------------------------------

class WellFormedLLMClient:
    """
    Returns a minimal structurally valid RCA card dict.

    The synthesizer's ``_normalize_llm_output()`` should accept this and
    produce a card without invoking the fallback path.  Use this as the
    D11-A baseline to confirm the LLM path works end-to-end before testing
    the three failure modes.

    Note: ``candidate_id`` is set to a placeholder; the normalizer will match
    it against the actual candidates from the run.  If the placeholder is not
    found, the synthesizer may silently correct or fall back — check
    ``fallback_used`` after running.
    """

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        return {
            "executive_summary": {
                "decision_status": "candidate_ready",
                "primary_conclusion": "WellFormedLLMClient: mock LLM synthesis succeeded.",
                "analyst_attention_flags": [],
            },
            "primary_hypothesis": {
                "candidate_id": "__MOCK_PRIMARY__",
                "cause_label": "mock primary cause",
                "hypothesis_type": "failure_mode",
                "narrative": "Mock narrative produced by WellFormedLLMClient.",
                "why_primary": ["Mock supporting reason."],
                "uncertainties": ["Mock uncertainty."],
                "composite_score": 0.75,
                "citations": [],
            },
            "alternatives": [],
            "contributing_causes": [],
            "recommended_actions": [],
            "analyst_review": {
                "decision_required": False,
                "questions_to_resolve": [],
                "writeback_recommendation": "approved_for_cap",
            },
            "evidence": [],
        }


# ---------------------------------------------------------------------------
# D11-B — malformed response
# ---------------------------------------------------------------------------

class MalformedLLMClient:
    """
    Returns a dict that is syntactically valid JSON but structurally invalid
    as an RCA card.

    Missing all required keys (``primary_hypothesis``, ``executive_summary``,
    etc.).  The synthesizer's normalization gate must detect the missing
    structure and fall back to the deterministic path.

    Expected post-condition: ``result["rca_card"]["fallback_used"] is True``
    """

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        return {
            "this_is": "not_a_valid_rca_card",
            "missing": ["primary_hypothesis", "executive_summary", "evidence"],
        }


# ---------------------------------------------------------------------------
# D11-C — empty response
# ---------------------------------------------------------------------------

class EmptyLLMClient:
    """
    Returns an empty dict ``{}``.

    Simulates an LLM that successfully responds but produces no content
    (e.g., token limit exceeded, model refused the prompt).

    Expected post-condition: ``result["rca_card"]["fallback_used"] is True``
    """

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# D11-D — timeout / exception
# ---------------------------------------------------------------------------

class TimeoutLLMClient:
    """
    Raises ``TimeoutError`` to simulate an LLM request that exceeds its
    deadline.

    Parameters
    ----------
    delay_seconds:
        Optional sleep before raising.  Keep at 0.0 in unit tests to avoid
        slowing down CI.  Use a positive value only when testing that the
        synthesizer respects its own wall-clock deadline.

    Expected post-condition: ``result["rca_card"]["fallback_used"] is True``
    """

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self._delay = delay_seconds

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        if self._delay > 0:
            time.sleep(self._delay)
        raise TimeoutError("LLM request timed out (simulated by TimeoutLLMClient).")


# ---------------------------------------------------------------------------
# Registry — convenient for parametrised tests
# ---------------------------------------------------------------------------

#: All three known-fail clients, each expected to trigger ``fallback_used=True``.
FALLBACK_TRIGGERING_CLIENTS = [
    MalformedLLMClient(),
    EmptyLLMClient(),
    TimeoutLLMClient(),
]
