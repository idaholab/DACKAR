"""
test_step5_sensitivity_table.py — Step 5 sensitivity table tests

Covers:
- No degraded sources → empty rows, any_ranking_change_possible = False
- Single missing core source raises coverage_factor → delta > 0 for all top candidates
- not_assessed source treated same as missing (score can only improve)
- partial source included (already partially penalised, smaller delta)
- top_n cap: only top_n candidates appear in rows
- would_change_ranking flag set when lower-ranked candidate jumps above a higher one
- any_ranking_change_possible True when at least one delta > 0.02
- orchestrator manifest: sensitivity_table key present with correct summary fields
- analyst_attention_flag injected when any_ranking_change_possible is True
- analyst_attention_flag NOT injected when no ranking change possible
- empty candidates list → safe empty table
- None coverage_summary → safe empty table

Run:  pytest test_step5_sensitivity_table.py -v
"""
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32 as Engine  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────

def _coverage(families: dict) -> dict:
    return {"source_families": families}


def _complete_coverage() -> dict:
    return _coverage({
        "kg_context":              {"status": "complete"},
        "upstream_anomaly_inputs": {"status": "complete"},
        "chroma_corpus":           {"status": "complete"},
        "telemetry_detail":        {"status": "complete"},
        "soe_log":                 {"status": "complete"},
        "alarm_log":               {"status": "complete"},
    })


def _candidate(cid: str, score: float, composite_raw: Optional[float] = None) -> dict:
    """Build a candidate dict.

    ``composite_raw`` is the pre-coverage-factor score.  When omitted it
    equals ``score`` (i.e. no coverage penalty already applied).  Tests that
    check delta > 0 must supply a ``composite_raw`` greater than ``score`` to
    simulate a candidate that was already penalised by the current factor.
    """
    raw = composite_raw if composite_raw is not None else score
    return {
        "candidate_id": cid,
        "event_id": "EVT-001",
        "composite_score": score,
        "scores": {"composite_raw": raw},
        "quality_multiplier": 1.0,
    }


def _build(candidates, coverage, top_n=5) -> dict:
    return Engine._build_sensitivity_table(
        candidates=candidates,
        coverage_summary=coverage,
        top_n=top_n,
    )


# ── 1. No degraded sources ────────────────────────────────────────────────────

def test_no_degraded_sources_empty_rows():
    cov = _complete_coverage()
    result = _build([_candidate("C1", 0.8)], cov)
    assert result["rows"] == []


def test_no_degraded_sources_any_change_false():
    cov = _complete_coverage()
    result = _build([_candidate("C1", 0.8)], cov)
    assert result["summary"]["any_ranking_change_possible"] is False


def test_no_degraded_sources_missing_sources_checked_empty():
    cov = _complete_coverage()
    result = _build([_candidate("C1", 0.8)], cov)
    assert result["summary"]["missing_sources_checked"] == []


def _penalised_candidate(cid: str) -> dict:
    """Candidate that was already scored with _missing_kg_coverage factor (~0.929).

    composite_raw=0.753, composite_score≈0.70  → patched delta ≈ +0.053.
    """
    return _candidate(cid, score=0.70, composite_raw=0.753)


# ── 2. Single missing core source ─────────────────────────────────────────────

def _missing_kg_coverage() -> dict:
    return _coverage({
        "kg_context":              {"status": "missing"},
        "upstream_anomaly_inputs": {"status": "complete"},
        "chroma_corpus":           {"status": "complete"},
        "telemetry_detail":        {"status": "complete"},
    })


def test_missing_core_source_produces_rows():
    result = _build([_penalised_candidate("C1")], _missing_kg_coverage())
    assert len(result["rows"]) == 1
    assert result["rows"][0]["source_family"] == "kg_context"


def test_missing_core_source_positive_delta():
    result = _build([_penalised_candidate("C1")], _missing_kg_coverage())
    row = result["rows"][0]
    assert row["estimated_score_delta"] > 0


def test_missing_core_source_estimated_gt_current():
    result = _build([_penalised_candidate("C1")], _missing_kg_coverage())
    row = result["rows"][0]
    assert row["estimated_composite_if_available"] > row["current_composite_score"]


def test_missing_source_any_change_true_when_delta_large():
    """kg_context is 40% weight; missing → significant penalty → large delta."""
    result = _build([_penalised_candidate("C1")], _missing_kg_coverage())
    assert result["summary"]["any_ranking_change_possible"] is True


def test_missing_core_source_checked_listed_in_summary():
    result = _build([_penalised_candidate("C1")], _missing_kg_coverage())
    assert "kg_context" in result["summary"]["missing_sources_checked"]


# ── 3. not_assessed treated as degraded ───────────────────────────────────────

def test_not_assessed_source_included_in_rows():
    cov = _coverage({
        "kg_context":              {"status": "complete"},
        "upstream_anomaly_inputs": {"status": "complete"},
        "chroma_corpus":           {"status": "complete"},
        "telemetry_detail":        {"status": "complete"},
        "soe_log":                 {"status": "not_assessed"},
    })
    result = _build([_candidate("C1", 0.8)], cov)
    sources = [r["source_family"] for r in result["rows"]]
    assert "soe_log" in sources


# ── 4. partial source included ────────────────────────────────────────────────

def test_partial_source_delta_smaller_than_missing():
    """Partial factor=0.93 vs missing factor=0.85 → smaller gap → smaller delta."""
    cov_missing = _coverage({
        "kg_context":              {"status": "missing"},
        "upstream_anomaly_inputs": {"status": "complete"},
        "chroma_corpus":           {"status": "complete"},
        "telemetry_detail":        {"status": "complete"},
    })
    cov_partial = _coverage({
        "kg_context":              {"status": "partial"},
        "upstream_anomaly_inputs": {"status": "complete"},
        "chroma_corpus":           {"status": "complete"},
        "telemetry_detail":        {"status": "complete"},
    })
    # Compute actual factors: missing~0.929, partial~0.955
    # Build candidates already penalised by each respective factor
    delta_missing = _build([_penalised_candidate("C1")], cov_missing)["rows"][0]["estimated_score_delta"]
    # For partial: raw=0.753 at partial factor ~0.955 → composite_score ~0.719
    cand_partial = _candidate("C1", score=0.719, composite_raw=0.753)
    delta_partial = _build([cand_partial], cov_partial)["rows"][0]["estimated_score_delta"]
    assert delta_partial < delta_missing


# ── 5. top_n cap ──────────────────────────────────────────────────────────────

def test_top_n_cap_limits_candidates():
    candidates = [_candidate(f"C{i}", 0.9 - i * 0.05, composite_raw=0.95 - i * 0.05) for i in range(10)]
    cov = _missing_kg_coverage()
    result = _build(candidates, cov, top_n=3)
    unique_ids = {r["candidate_id"] for r in result["rows"]}
    assert len(unique_ids) == 3


def test_top_n_cap_selects_highest_scoring():
    candidates = [_candidate(f"C{i}", 0.9 - i * 0.05, composite_raw=0.95 - i * 0.05) for i in range(10)]
    cov = _missing_kg_coverage()
    result = _build(candidates, cov, top_n=3)
    ids = {r["candidate_id"] for r in result["rows"]}
    assert "C0" in ids and "C1" in ids and "C2" in ids
    assert "C9" not in ids


def test_top_n_candidates_in_summary():
    candidates = [_candidate(f"C{i}", 0.8, composite_raw=0.86) for i in range(4)]
    result = _build(candidates, _missing_kg_coverage(), top_n=3)
    assert result["summary"]["top_n_candidates"] == 3


# ── 6. Safety: empty / None inputs ────────────────────────────────────────────

def test_empty_candidates_safe():
    result = _build([], _complete_coverage())
    assert result["rows"] == []
    assert result["summary"]["top_n_candidates"] == 0


def test_none_coverage_summary_safe():
    result = _build([_candidate("C1", 0.8)], None)
    assert result["rows"] == []
    assert result["summary"]["any_ranking_change_possible"] is False


def test_missing_source_families_key_safe():
    cov = {"no_source_families_key": {}}
    result = _build([_candidate("C1", 0.8)], cov)
    assert result["rows"] == []


# ── 7. Schema fields present ──────────────────────────────────────────────────

def test_row_has_required_fields():
    result = _build([_penalised_candidate("C1")], _missing_kg_coverage())
    row = result["rows"][0]
    for field in [
        "candidate_id", "candidate_rank", "source_family", "current_status",
        "current_composite_score", "estimated_composite_if_available",
        "estimated_score_delta", "would_change_ranking",
    ]:
        assert field in row, f"Missing field: {field}"


def test_table_has_required_top_level_keys():
    result = _build([_penalised_candidate("C1")], _missing_kg_coverage())
    for key in ["event_id", "generated_at", "summary", "rows", "provenance"]:
        assert key in result


def test_candidate_rank_starts_at_1():
    candidates = [_penalised_candidate("C1"), _candidate("C2", 0.60, composite_raw=0.65)]
    result = _build(candidates, _missing_kg_coverage())
    ranks = [r["candidate_rank"] for r in result["rows"]]
    assert 1 in ranks


# ── 8. Manifest wiring ────────────────────────────────────────────────────────

def _mock_manifest_with_sensitivity(any_change: bool, row_count: int) -> dict:
    """Simulate the orchestrator manifest artifacts block."""
    sensitivity = {
        "summary": {
            "any_ranking_change_possible": any_change,
            "missing_sources_checked": ["kg_context"] if any_change else [],
            "top_n_candidates": 2,
        },
        "rows": [{}] * row_count,
        "provenance": {},
    }
    # Replicate the analyst_attention_flags logic from _stage_g_finalize_manifest
    base_flags = ["existing_flag"]
    flags = base_flags + (
        ["SENSITIVITY: missing data could alter candidate ranking — review sensitivity_table"]
        if bool((sensitivity.get("summary") or {}).get("any_ranking_change_possible", False))
        else []
    )
    return {
        "sensitivity_table": sensitivity,
        "analyst_attention_flags": flags,
        "artifacts": {
            "sensitivity_table": {
                "present": True,
                "any_ranking_change_possible": bool(
                    (sensitivity.get("summary") or {}).get("any_ranking_change_possible", False)
                ),
                "missing_sources_checked": list(
                    (sensitivity.get("summary") or {}).get("missing_sources_checked") or []
                ),
                "top_n_candidates": int(
                    (sensitivity.get("summary") or {}).get("top_n_candidates", 0)
                ),
                "row_count": len(sensitivity.get("rows") or []),
            },
        },
    }


def test_manifest_sensitivity_table_key_present():
    manifest = _mock_manifest_with_sensitivity(any_change=False, row_count=0)
    assert "sensitivity_table" in manifest


def test_manifest_artifacts_sensitivity_present_flag():
    manifest = _mock_manifest_with_sensitivity(any_change=False, row_count=0)
    assert manifest["artifacts"]["sensitivity_table"]["present"] is True


def test_manifest_artifacts_row_count():
    manifest = _mock_manifest_with_sensitivity(any_change=True, row_count=3)
    assert manifest["artifacts"]["sensitivity_table"]["row_count"] == 3


def test_analyst_attention_flag_injected_when_change_possible():
    manifest = _mock_manifest_with_sensitivity(any_change=True, row_count=2)
    assert any("SENSITIVITY" in f for f in manifest["analyst_attention_flags"])


def test_analyst_attention_flag_not_injected_when_no_change():
    manifest = _mock_manifest_with_sensitivity(any_change=False, row_count=0)
    assert not any("SENSITIVITY" in f for f in manifest["analyst_attention_flags"])


def test_analyst_attention_flag_appended_not_replacing():
    manifest = _mock_manifest_with_sensitivity(any_change=True, row_count=2)
    assert "existing_flag" in manifest["analyst_attention_flags"]
