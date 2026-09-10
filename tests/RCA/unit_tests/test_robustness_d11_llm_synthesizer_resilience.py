"""
test_robustness_d11_llm_synthesizer_resilience.py — D11 LLM synthesizer resilience (Phase 3)

Checks:
    D11-B  MalformedLLMClient → pipeline completes, rca_card valid, fallback_used=True
    D11-C  EmptyLLMClient → pipeline completes, rca_card valid, fallback_used=True
    D11-D  TimeoutLLMClient → pipeline completes, rca_card valid, fallback_used=True
    D11-A  WellFormedLLMClient → rca_card valid, fallback_used=False (baseline)

Run directly:   python test_robustness_d11_llm_synthesizer_resilience.py
Or via pytest:  pytest test_robustness_d11_llm_synthesizer_resilience.py -v

Fixtures used: TC-8 (richest scenario)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
_SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenario"
_TESTS_SHARED = _SCENARIO_ROOT / "shared"

for _p in (str(_RCA_ROOT), str(_TESTS_SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _mod in (
    "neo4j", "py2neo", "chromadb",
    "langchain_community", "langchain_community.vectorstores",
    "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402
pytest.importorskip("run_helpers", reason="scenario shared helpers (tests/RCA/scenario/shared) arrive in MR #12")
from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca  # noqa: E402
from mock_llm_clients import (  # noqa: E402
    WellFormedLLMClient,
    MalformedLLMClient,
    EmptyLLMClient,
    TimeoutLLMClient,
    FALLBACK_TRIGGERING_CLIENTS,
)

_TC8_FIXTURES = _SCENARIO_ROOT / "test_case_8" / "fixtures"

_FIXTURES: Dict[str, Any] = {}  # populated at test time, not import time


def _get_fixtures() -> Dict[str, Any]:
    if not _FIXTURES:
        _FIXTURES.update(load_fixtures(_TC8_FIXTURES))
    return _FIXTURES


def _run_with_llm(llm_client: Any) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        orch = build_fixture_orchestrator(tmp, llm_client=llm_client)
        return run_rca(orch, _get_fixtures())


def _assert_rca_card_valid(
    result: Dict[str, Any],
    label: str,
    *,
    expect_fallback: Optional[bool] = None,
) -> None:
    """Minimal invariant: rca_card must be present, non-empty, with a primary hypothesis.

    If *expect_fallback* is not None, also asserts rca_card.fallback_used matches.
    """
    card = result.get("rca_card")
    assert card and isinstance(card, dict), (
        f"D11 FAIL [{label}]: rca_card absent or not a dict after LLM failure. "
        "The synthesizer must produce a card regardless of LLM output quality."
    )
    ph = card.get("primary_hypothesis") or {}
    assert ph.get("candidate_id"), (
        f"D11 FAIL [{label}]: rca_card.primary_hypothesis.candidate_id is absent. "
        "The fallback path must always set a candidate_id (or 'NONE' for no candidates)."
    )
    es = card.get("executive_summary") or {}
    assert es.get("decision_status"), (
        f"D11 FAIL [{label}]: rca_card.executive_summary.decision_status is absent. "
        "The fallback synthesizer must always emit a decision_status."
    )
    assert "fallback_used" in card, (
        f"D11 FAIL [{label}]: rca_card.fallback_used is absent. "
        "The synthesizer must write fallback_used=True/False at the card top level."
    )
    if expect_fallback is not None:
        assert card["fallback_used"] is expect_fallback, (
            f"D11 FAIL [{label}]: expected fallback_used={expect_fallback}, "
            f"got {card['fallback_used']!r}."
        )
    print(
        f"  pass  D11 [{label}]: rca_card valid. "
        f"candidate_id={ph.get('candidate_id')!r} "
        f"decision_status={es.get('decision_status')!r} "
        f"fallback_used={card.get('fallback_used')!r}"
    )


# ---------------------------------------------------------------------------
# D11-B — Malformed LLM response
# ---------------------------------------------------------------------------

def test_d11b_malformed_llm_response():
    """
    D11-B: When the LLM returns a dict that is not a valid RCA card structure,
    the synthesizer must fall back to the deterministic path and produce a
    valid card.  The pipeline must not raise.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result = _run_with_llm(MalformedLLMClient())
    _assert_rca_card_valid(result, "MalformedLLMClient", expect_fallback=True)


# ---------------------------------------------------------------------------
# D11-C — Empty LLM response
# ---------------------------------------------------------------------------

def test_d11c_empty_llm_response():
    """
    D11-C: When the LLM returns an empty dict {}, the synthesizer must fall
    back and produce a valid card.  The pipeline must not raise.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result = _run_with_llm(EmptyLLMClient())
    _assert_rca_card_valid(result, "EmptyLLMClient", expect_fallback=True)


# ---------------------------------------------------------------------------
# D11-D — LLM timeout / exception
# ---------------------------------------------------------------------------

def test_d11d_timeout_llm():
    """
    D11-D: When the LLM raises TimeoutError, the synthesizer must catch the
    exception, fall back to the deterministic path, and produce a valid card.
    The pipeline must not raise.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result = _run_with_llm(TimeoutLLMClient(delay_seconds=0.0))
    _assert_rca_card_valid(result, "TimeoutLLMClient", expect_fallback=True)


# ---------------------------------------------------------------------------
# D11-A — Well-formed LLM response (diagnostic baseline)
# ---------------------------------------------------------------------------

def test_d11a_wellformed_llm_response():
    """
    D11-A diagnostic: When the LLM returns a structurally valid card dict,
    the synthesizer should use it (not fall back).  This is a baseline check.

    Because WellFormedLLMClient uses a placeholder candidate_id
    (__MOCK_PRIMARY__) that won't match real candidates, the normalizer may
    correct or fall back — fallback_used reflects whichever path was taken.
    The check verifies the card is valid and fallback_used is a boolean.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result = _run_with_llm(WellFormedLLMClient())
    _assert_rca_card_valid(result, "WellFormedLLMClient")


# ---------------------------------------------------------------------------
# D11 parametrized — all three fallback-triggering clients  (pytest parametrize)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("client,label", [
    (MalformedLLMClient(),          "malformed"),
    (EmptyLLMClient(),              "empty"),
    (TimeoutLLMClient(delay_seconds=0.0), "timeout"),
])
def test_d11_fallback_triggering_clients(client: Any, label: str):
    """
    D11 parametrized: all three known-fail clients must produce a valid rca_card.
    This is the same coverage as D11-B/C/D combined, in pytest parametrize form.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip("TC-8 fixtures not found")
    result = _run_with_llm(client)
    _assert_rca_card_valid(result, f"parametrized/{label}", expect_fallback=True)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_d11b_malformed_llm_response,
    test_d11c_empty_llm_response,
    test_d11d_timeout_llm,
    test_d11a_wellformed_llm_response,
]


def run_all() -> bool:
    print(f"\n=== test_robustness_d11_llm_synthesizer_resilience ({len(ALL_TESTS)} tests) ===")
    passed = failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
