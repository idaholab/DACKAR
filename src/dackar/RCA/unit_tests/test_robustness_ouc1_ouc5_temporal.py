"""
test_robustness_ouc1_ouc5_temporal.py — Tier 2 Scenario correctness: temporal discrimination

OUC-1  Cause vs. consequence
       A pump trips. Vibration (OVERLAPS, lower severity) is the cause.
       Discharge pressure spike (no causal Allen, higher severity) is the consequence.
       The pipeline must rank the vibration candidate first and exclude the
       pressure spike from retained candidates regardless of severity ordering.

OUC-5  Fixation resistance
       A reactor trips. Feedwater flow deviation (OVERLAPS, severity=0.55) is the
       initiating cause. Turbine trip (no causal Allen, severity=0.90) is the
       automatic protection response — highest severity in the event, but a consequence.
       The pipeline must rank feedwater first; the high-severity turbine signal must
       not reach the retained candidate list.

Design notes
------------
- Both OUCs test the same core capability: Allen relation ordering beats raw signal
  severity. If either fails, temporal discrimination is broken.
- Assertion on ``scores["allen_relation"]`` (lowercase) — that is the field name as
  set by the causality engine during ``refine_with_evidence``.
- ``ruled_out[]`` does NOT contain timeline-inconsistent candidates in the current
  implementation; those are filtered via composite/evidence threshold (FOLLOWS/DURING
  candidates receive no causal Allen signal → zero temporal from Allen → evidence
  contradiction → below threshold). The run_manifest artifact summary is used to
  confirm total vs. causal node count.
- Fixture directories: tests/fixtures_robustness/ouc1_cause_vs_consequence/
                       tests/fixtures_robustness/ouc5_fixation_resistance/

Run directly:  python test_robustness_ouc1_ouc5_temporal.py
Or via pytest: pytest test_robustness_ouc1_ouc5_temporal.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

_RCA_ROOT = Path(__file__).resolve().parent.parent
_TESTS_SHARED = _RCA_ROOT / "tests" / "shared"
_FIXTURES_ROOT = _RCA_ROOT / "tests" / "fixtures_robustness"

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

from run_helpers import build_fixture_orchestrator, load_fixtures, run_rca  # noqa: E402

_OUC1_DIR = _FIXTURES_ROOT / "ouc1_cause_vs_consequence"
_OUC5_DIR = _FIXTURES_ROOT / "ouc5_fixation_resistance"

_CAUSAL_ALLEN_RELATIONS = {"overlaps", "contains", "precedes"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run(fixture_dir: Path) -> Dict[str, Any]:
    fixtures = load_fixtures(fixture_dir)
    with tempfile.TemporaryDirectory() as tmp:
        orch = build_fixture_orchestrator(tmp)
        return run_rca(orch, fixtures)


def _candidates(result: Dict[str, Any]):
    return result["causality_candidates"].get("candidates") or []


def _filtered(result: Dict[str, Any]):
    return result["causality_candidates"].get("filtered_out_candidates") or []


def _arm_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return (
        (result.get("run_manifest") or {})
        .get("artifacts", {})
        .get("allen_relation_map") or {}
    )


# ---------------------------------------------------------------------------
# OUC-1 — Cause vs. consequence
# ---------------------------------------------------------------------------

def test_ouc1_cause_is_retained():
    """OUC-1: The vibration (OVERLAPS) candidate must be retained in top candidates."""
    result = _run(_OUC1_DIR)
    cands = _candidates(result)
    assert cands, "OUC-1: No candidates retained — fixture may need adjustment"
    top = cands[0]
    assert top["component_id"] == "U1-PUMP-P101A-BEARING", (
        f"OUC-1: Expected bearing component to rank first, got {top['component_id']!r}"
    )


def test_ouc1_cause_has_causal_allen():
    """OUC-1: The top candidate must have a causal Allen relation (overlaps/contains/precedes)."""
    result = _run(_OUC1_DIR)
    cands = _candidates(result)
    assert cands, "OUC-1: No candidates retained"
    allen_rel = (cands[0].get("scores") or {}).get("allen_relation")
    assert allen_rel in _CAUSAL_ALLEN_RELATIONS, (
        f"OUC-1: Top candidate allen_relation={allen_rel!r}; expected one of "
        f"{_CAUSAL_ALLEN_RELATIONS}. Vibration should be OVERLAPS."
    )


def test_ouc1_consequence_excluded():
    """OUC-1: The discharge pressure spike (consequence) must NOT appear in retained candidates."""
    result = _run(_OUC1_DIR)
    cands = _candidates(result)
    retained_ids = {c["component_id"] for c in cands}
    assert "U1-PUMP-P101A-DISCHARGE" not in retained_ids, (
        "OUC-1: Discharge pressure spike appears in retained candidates — "
        "consequence should be excluded (no causal Allen signal, low evidence)."
    )


def test_ouc1_consequence_has_no_causal_allen():
    """OUC-1: The discharge candidate must have no Allen relation applied (FOLLOWS/DURING → not causal)."""
    result = _run(_OUC1_DIR)
    filtered_cands = _filtered(result)
    discharge = next(
        (c for c in filtered_cands if c.get("component_id") == "U1-PUMP-P101A-DISCHARGE"),
        None,
    )
    assert discharge is not None, (
        "OUC-1: Expected discharge candidate in filtered_out_candidates"
    )
    allen_rel = (discharge.get("scores") or {}).get("allen_relation")
    assert allen_rel is None or allen_rel not in _CAUSAL_ALLEN_RELATIONS, (
        f"OUC-1: Discharge candidate unexpectedly has causal Allen relation {allen_rel!r}"
    )


def test_ouc1_allen_map_one_causal_node():
    """OUC-1: Allen map must show exactly 2 total nodes, exactly 1 causal (vibration only)."""
    result = _run(_OUC1_DIR)
    arm = _arm_summary(result)
    assert arm.get("total_nodes") == 2, (
        f"OUC-1: Expected 2 Allen nodes (vibration + discharge), got {arm.get('total_nodes')}"
    )
    assert arm.get("causal_nodes") == 1, (
        f"OUC-1: Expected 1 causal Allen node (vibration only), got {arm.get('causal_nodes')}. "
        "Discharge (FOLLOWS/DURING) must not be causal."
    )


# ---------------------------------------------------------------------------
# OUC-5 — Fixation resistance
# ---------------------------------------------------------------------------

def test_ouc5_precursor_is_retained():
    """OUC-5: The low-severity feedwater (OVERLAPS) candidate must rank first."""
    result = _run(_OUC5_DIR)
    cands = _candidates(result)
    assert cands, "OUC-5: No candidates retained — fixture may need adjustment"
    top = cands[0]
    assert top["component_id"] == "U1-FEEDWATER-FCV-101", (
        f"OUC-5: Expected feedwater component to rank first, got {top['component_id']!r}. "
        "High-severity turbine trip signal must not override temporal ordering."
    )


def test_ouc5_precursor_has_causal_allen():
    """OUC-5: The feedwater candidate must have a causal Allen relation despite lower severity."""
    result = _run(_OUC5_DIR)
    cands = _candidates(result)
    assert cands, "OUC-5: No candidates retained"
    allen_rel = (cands[0].get("scores") or {}).get("allen_relation")
    assert allen_rel in _CAUSAL_ALLEN_RELATIONS, (
        f"OUC-5: Top candidate allen_relation={allen_rel!r}; expected causal relation. "
        "Feedwater deviation started >30 min before event — must be OVERLAPS."
    )


def test_ouc5_high_severity_consequence_excluded():
    """OUC-5: The high-severity turbine trip (consequence, no causal Allen) must NOT be retained.

    Core fixation-resistance test: the pipeline must not be 'fooled' by a
    severity=0.90 signal that follows the event. Severity alone must not
    override Allen-based temporal ordering.
    """
    result = _run(_OUC5_DIR)
    cands = _candidates(result)
    retained_ids = {c["component_id"] for c in cands}
    assert "U1-TURBINE-TRIP-SIGNAL" not in retained_ids, (
        "OUC-5 FIXATION FAILURE: Turbine trip signal (severity=0.90, DURING/FOLLOWS) "
        "reached retained candidates. High-severity consequence must be excluded when "
        "its Allen relation is not causal. Severity must not override temporal ordering."
    )


def test_ouc5_allen_map_one_causal_node():
    """OUC-5: Allen map must show 2 total nodes, 1 causal (feedwater only, not turbine)."""
    result = _run(_OUC5_DIR)
    arm = _arm_summary(result)
    assert arm.get("total_nodes") == 2, (
        f"OUC-5: Expected 2 Allen nodes (feedwater + turbine), got {arm.get('total_nodes')}"
    )
    assert arm.get("causal_nodes") == 1, (
        f"OUC-5: Expected 1 causal Allen node (feedwater only), got {arm.get('causal_nodes')}. "
        "Turbine trip (DURING/FOLLOWS) must not be a causal node."
    )


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_ouc1_cause_is_retained,
    test_ouc1_cause_has_causal_allen,
    test_ouc1_consequence_excluded,
    test_ouc1_consequence_has_no_causal_allen,
    test_ouc1_allen_map_one_causal_node,
    test_ouc5_precursor_is_retained,
    test_ouc5_precursor_has_causal_allen,
    test_ouc5_high_severity_consequence_excluded,
    test_ouc5_allen_map_one_causal_node,
]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {e}")
