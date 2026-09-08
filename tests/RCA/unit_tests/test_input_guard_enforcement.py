"""
test_input_guard_enforcement.py — unit tests for strict Stage A guard policy logic.

Run directly:   python test_input_guard_enforcement.py
Or via pytest:  pytest test_input_guard_enforcement.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

# Stub optional heavy dependencies imported by orchestrator module load.
for _mod in (
    "neo4j",
    "py2neo",
    "chromadb",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator


def test_warn_only_mode_never_aborts():
    policy = RCAReasoningOrchestrator._evaluate_input_guard_policy(
        input_guards={"flags": ["telemetry_window_end_before_event"]},
        strict_enabled=False,
        blocking_flags=None,
        hard_stop_on_any_flag=False,
    )
    assert policy["hard_abort_required"] is False
    assert policy["strict_enabled"] is False
    assert "warning-only" in policy["reason"].lower()


def test_strict_mode_aborts_on_default_blocking_flag():
    policy = RCAReasoningOrchestrator._evaluate_input_guard_policy(
        input_guards={"flags": ["telemetry_window_starts_after_event"]},
        strict_enabled=True,
        blocking_flags=None,
        hard_stop_on_any_flag=False,
    )
    assert policy["hard_abort_required"] is True
    assert "telemetry_window_starts_after_event" in policy["triggered_blocking_flags"]


def test_strict_mode_respects_custom_blocking_flags():
    policy = RCAReasoningOrchestrator._evaluate_input_guard_policy(
        input_guards={"flags": ["possible_multi_event_overlap"]},
        strict_enabled=True,
        blocking_flags=["possible_multi_event_overlap"],
        hard_stop_on_any_flag=False,
    )
    assert policy["hard_abort_required"] is True
    assert policy["triggered_blocking_flags"] == ["possible_multi_event_overlap"]


def test_hard_stop_any_flag_aborts_even_when_not_in_blocking_list():
    policy = RCAReasoningOrchestrator._evaluate_input_guard_policy(
        input_guards={"flags": ["operational_context_as_of_may_be_stale"]},
        strict_enabled=True,
        blocking_flags=["telemetry_window_end_before_event"],
        hard_stop_on_any_flag=True,
    )
    assert policy["hard_abort_required"] is True
    assert policy["triggered_blocking_flags"] == []
    assert "any flag" in policy["reason"].lower()


ALL_TESTS = [
    test_warn_only_mode_never_aborts,
    test_strict_mode_aborts_on_default_blocking_flag,
    test_strict_mode_respects_custom_blocking_flags,
    test_hard_stop_any_flag_aborts_even_when_not_in_blocking_list,
]


def run_all():
    print(f"\n=== test_input_guard_enforcement ({len(ALL_TESTS)} tests) ===")
    passed, failed = 0, 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
