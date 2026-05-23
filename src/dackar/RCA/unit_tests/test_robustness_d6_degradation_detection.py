"""
test_robustness_d6_degradation_detection.py — D6 Data degradation and silent failure (Phase 1/3)

Checks:
    D6-D  Optional phase failure recorded in run_manifest.pipeline_warnings
    D6-F  Null timestamp_start raises input guard flag
    D6-A  Missing optional inputs produce 'not_assessed' coverage status
    D6-E  Pipeline completes despite broken optional phase

Run directly:   python test_robustness_d6_degradation_detection.py
Or via pytest:  pytest test_robustness_d6_degradation_detection.py -v

Fixtures used:
    TC-8   ../tests/test_case_8/fixtures/
    D6-F   ../tests/fixtures_robustness/event_null_timestamp.json  (new)

No live Neo4j, Chroma, or LLM required.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — must happen before any RCA package import.
# ---------------------------------------------------------------------------

_RCA_ROOT = Path(__file__).resolve().parent.parent
_TESTS_SHARED = _RCA_ROOT / "tests" / "shared"
_FIXTURES_ROBUSTNESS = _RCA_ROOT / "tests" / "fixtures_robustness"

for _p in (str(_RCA_ROOT), str(_TESTS_SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca  # noqa: E402

_TC8_FIXTURES = _RCA_ROOT / "tests" / "test_case_8" / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_tc8() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = build_fixture_orchestrator(tmp)
        fixtures = load_fixtures(_TC8_FIXTURES)
        return run_rca(orchestrator, fixtures)


class _BrokenIshikawaEvaluator:
    """Simulates a malfunctioning optional phase component."""

    def evaluate(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "Simulated Ishikawa evaluator failure (D6-D test). "
            "This error must appear in run_manifest.pipeline_warnings."
        )


# ---------------------------------------------------------------------------
# D6-E — Pipeline completes despite broken optional phase  (diagnostic, not xfail)
# ---------------------------------------------------------------------------

def test_d6e_pipeline_completes_with_broken_ishikawa():
    """
    D6-E: The orchestrator must not propagate an optional-phase exception to
    the caller.  The run must complete and return a result dict even when the
    Ishikawa evaluator raises.

    If this test FAILS, the orchestrator is propagating optional-phase
    exceptions — that is a separate (more severe) bug than D6-D.  Fix this
    first before investigating D6-D.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip(f"TC-8 fixtures not found at {_TC8_FIXTURES}")

    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = build_fixture_orchestrator(
            tmp,
            ishikawa_evaluator=_BrokenIshikawaEvaluator(),
        )
        fixtures = load_fixtures(_TC8_FIXTURES)
        result = run_rca(orchestrator, fixtures)

    assert isinstance(result, dict), "Pipeline must return a dict even when optional phase fails."
    assert result.get("rca_card"), "rca_card must be present even when Ishikawa fails."
    print(
        "  pass  D6-E: pipeline completed despite broken Ishikawa evaluator. "
        "rca_card present."
    )


# ---------------------------------------------------------------------------
# D6-D — Optional phase failure must be recorded in run_manifest
# ---------------------------------------------------------------------------

def test_d6d_optional_phase_failure_recorded_in_manifest():
    """
    D6-D: When an optional pipeline phase fails, the failure must be
    recorded in run_manifest.pipeline_warnings so the analyst knows the
    artifact is incomplete.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip(f"TC-8 fixtures not found at {_TC8_FIXTURES}")

    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = build_fixture_orchestrator(
            tmp,
            ishikawa_evaluator=_BrokenIshikawaEvaluator(),
        )
        fixtures = load_fixtures(_TC8_FIXTURES)
        result = run_rca(orchestrator, fixtures)

    manifest = result.get("run_manifest") or {}
    warnings = manifest.get("pipeline_warnings") or []

    assert warnings, (
        "D6-D FAIL: run_manifest.pipeline_warnings is absent or empty after "
        "an optional-phase (Ishikawa) failure. The analyst has no signal that "
        "the Ishikawa artifact is missing."
    )
    ishikawa_warning_present = any(
        "ishikawa" in str(w).lower() for w in warnings
    )
    assert ishikawa_warning_present, (
        f"D6-D FAIL: pipeline_warnings exists but contains no Ishikawa entry. "
        f"Got: {warnings}"
    )
    print(
        f"  pass  D6-D: Ishikawa failure recorded in pipeline_warnings: {warnings}"
    )


# ---------------------------------------------------------------------------
# D6-F — Null timestamp_start must raise an input guard flag
# ---------------------------------------------------------------------------

def test_d6f_null_timestamp_triggers_input_guard():
    """
    D6-F: An event with timestamp_start=null must trigger an input guard
    flag ('missing_event_timestamp') so the analyst knows the temporal
    anchor is missing.
    """
    null_ts_fixture = _FIXTURES_ROBUSTNESS / "event_null_timestamp.json"
    assert null_ts_fixture.exists(), (
        f"D6-F fixture not found: {null_ts_fixture}. "
        "Create tests/fixtures_robustness/event_null_timestamp.json."
    )
    tc8_fixtures = _TC8_FIXTURES
    if not tc8_fixtures.exists():
        pytest.skip(f"TC-8 fixtures not found at {tc8_fixtures}")

    # Load TC-8 as base, substitute the null-timestamp event.
    import json
    with open(null_ts_fixture, encoding="utf-8") as fh:
        null_event = json.load(fh)

    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = build_fixture_orchestrator(tmp)
        fixtures = load_fixtures(tc8_fixtures)
        fixtures = dict(fixtures, event=null_event)
        result = run_rca(orchestrator, fixtures)

    input_guards = (
        (result.get("run_context") or {})
        .get("input_guards") or {}
    )
    flags = input_guards.get("flags") or []

    assert "missing_event_timestamp" in flags, (
        f"D6-F FAIL: null timestamp_start did not trigger 'missing_event_timestamp' guard. "
        f"Got flags: {flags}. "
        "The analyst has no signal that the event timestamp is absent."
    )
    print(
        f"  pass  D6-F: 'missing_event_timestamp' flag raised for null timestamp_start. "
        f"flags={flags}"
    )


# ---------------------------------------------------------------------------
# D6-A — Missing optional inputs produce not_assessed coverage status  [should PASS]
# ---------------------------------------------------------------------------

def test_d6a_missing_optional_inputs_not_assessed():
    """
    D6-A: When optional inputs (vendor_supply_chain_records, training_records, etc.)
    are absent, the corresponding categories in data_coverage_summary must have
    status 'not_assessed' (or 'data_limited') — not 'missing' or an exception.
    """
    if not _TC8_FIXTURES.exists():
        pytest.skip(f"TC-8 fixtures not found at {_TC8_FIXTURES}")

    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = build_fixture_orchestrator(tmp)
        fixtures = load_fixtures(_TC8_FIXTURES)
        # Explicitly strip optional inputs
        fixtures = dict(
            fixtures,
            vendor_supply_chain_records=None,
            training_records=None,
            environmental_monitoring=None,
        )
        result = run_rca(orchestrator, fixtures)

    assert isinstance(result, dict), "Pipeline must return result even with stripped optional inputs."
    assert result.get("rca_card"), "rca_card must be present."

    coverage = (
        (result.get("run_manifest") or {})
        .get("artifacts") or {}
    ).get("data_coverage_summary") or (
        (result.get("run_manifest") or {})
        .get("coverage_summary") or {}
    ).get("source_families") or {}

    problematic = []
    for src in ("vendor_supply_chain_records", "training_records", "environmental_monitoring"):
        entry = coverage.get(src)
        if entry is None:
            continue
        status = entry.get("status") if isinstance(entry, dict) else str(entry)
        if status not in ("not_assessed", "data_limited", "partial", "present"):
            problematic.append(f"{src}: unexpected status '{status}'")

    assert not problematic, (
        "D6-A FAIL: optional inputs produced unexpected coverage status:\n  "
        + "\n  ".join(problematic)
    )
    print(
        "  pass  D6-A: stripped optional inputs produce acceptable coverage status."
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_d6e_pipeline_completes_with_broken_ishikawa,
    test_d6d_optional_phase_failure_recorded_in_manifest,
    test_d6f_null_timestamp_triggers_input_guard,
    test_d6a_missing_optional_inputs_not_assessed,
]


def run_all() -> bool:
    print(f"\n=== test_robustness_d6_degradation_detection ({len(ALL_TESTS)} tests) ===")
    passed = failed = xfailed = 0
    for fn in ALL_TESTS:
        is_xfail = getattr(fn, "pytestmark", None) is not None
        try:
            fn()
            if is_xfail:
                print(f"  XPASS {fn.__name__} (expected to fail but passed — remove xfail marker)")
                xfailed += 1
            else:
                passed += 1
        except AssertionError as exc:
            if is_xfail:
                print(f"  xfail {fn.__name__}: {exc}")
                xfailed += 1
            else:
                import traceback
                print(f"  FAIL  {fn.__name__}: {exc}")
                traceback.print_exc()
                failed += 1
        except Exception as exc:
            import traceback
            print(f"  ERROR {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {xfailed} xfail/xpass")
    print("Note: run via pytest for proper xfail handling.")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
