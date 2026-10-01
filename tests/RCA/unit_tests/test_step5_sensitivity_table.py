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
from typing import Optional
from unittest.mock import MagicMock

import pytest

from dackar.RCA.orchestrators.causality_engine_v32 import RuleBasedCausalityEngineV32 as Engine  # noqa: E402
from dackar.RCA.orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator  # noqa: E402

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


# ── 4b. Ratio-based rescaling keeps independent quality penalties ─────────────

def test_independent_quality_penalty_rescaled_by_ratio():
    """A candidate carrying a quality penalty *beyond* the coverage factor must be
    rescaled by new_factor/current_factor — preserving that penalty — not recomputed
    as composite_raw * new_factor, which silently drops it and overstates the score.
    """
    cov = _missing_kg_coverage()
    current_factor, _ = Engine._coverage_quality_profile(cov)
    patched = {"source_families": dict(cov["source_families"], kg_context={"status": "complete"})}
    new_factor, _ = Engine._coverage_quality_profile(patched)

    raw = 0.90
    # composite_score carries an extra, independent 0.5 penalty on top of coverage.
    current = round(raw * current_factor * 0.5, 6)
    cand = {
        "candidate_id": "C1",
        "event_id": "EVT-001",
        "composite_score": current,
        "scores": {"composite_raw": raw},
        "quality_multiplier": round(current / raw, 6),
    }
    row = _build([cand], cov)["rows"][0]

    expected = round(min(1.0, current * (new_factor / current_factor)), 6)
    assert row["estimated_composite_if_available"] == expected
    # The dropped-penalty (buggy) estimate would be raw * new_factor — far higher.
    buggy = round(min(1.0, raw * new_factor), 6)
    assert row["estimated_composite_if_available"] < buggy - 0.01


def test_two_candidate_inversion_flags_would_change_for_last():
    """The lower-ranked (and here last) candidate's would_change_ranking must be
    evaluated: restoring kg_context lifts C2 above C1's current score.
    """
    cov = _missing_kg_coverage()
    c1 = _candidate("C1", score=0.80, composite_raw=0.86)
    c2 = _candidate("C2", score=0.76, composite_raw=0.82)  # ranks second, overtakes when kg restored
    result = _build([c1, c2], cov)
    by_id = {r["candidate_id"]: r for r in result["rows"]}
    assert by_id["C1"]["candidate_rank"] == 1
    assert by_id["C2"]["candidate_rank"] == 2
    assert by_id["C1"]["would_change_ranking"] is False  # top candidate, nobody above
    assert by_id["C2"]["would_change_ranking"] is True   # last candidate, now evaluated
    assert result["summary"]["any_ranking_change_possible"] is True


def test_rank_two_candidate_not_overtaking_flags_would_change_false():
    """A lower-ranked candidate that IS evaluated (rank > 1) but whose restored-source
    estimate stays below the candidate above it must report would_change_ranking False.
    The sibling inversion test only asserts False for the *top* candidate, where it holds
    trivially (nobody above it); this pins the evaluated-but-no-inversion branch, which a
    regression reinstating the old `rank_idx < len` skip would also wrongly leave False (I7).
    """
    cov = _missing_kg_coverage()
    c1 = _candidate("C1", score=0.95, composite_raw=0.99)
    c2 = _candidate("C2", score=0.70, composite_raw=0.74)  # rank 2; coverage bump cannot overtake C1
    by_id = {r["candidate_id"]: r for r in _build([c1, c2], cov)["rows"]}
    assert by_id["C2"]["candidate_rank"] == 2
    # C2 is evaluated against C1 (rank_idx > 1) yet its estimate stays well below C1.
    assert by_id["C2"]["estimated_composite_if_available"] < 0.95
    assert by_id["C2"]["would_change_ranking"] is False


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


# ── 8. Manifest wiring (real orchestrator, not a mock) ────────────────────────
#
# These exercise the actual _stage_g_finalize_manifest wiring: it builds the
# coverage summary from the stage inputs, feeds it to _build_sensitivity_table,
# surfaces the artifacts.sensitivity_table block, and appends the SENSITIVITY
# analyst-attention flag (preserving pre-existing executive-summary flags) only
# when any_ranking_change_possible is True. The previous version reimplemented
# that flag logic inside the test, so it could not catch a regression in it.

def _orchestrator() -> RCAReasoningOrchestrator:
    return RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )


def _rca_card() -> dict:
    """Minimal valid card carrying one pre-existing analyst-attention flag."""
    return {
        "validation_status": {"schema_valid": True, "all_claims_cited": True,
                              "passed_minimum_evidence_gate": True, "fallback_used": False},
        "analyst_review": {"decision_required": False, "writeback_recommendation": "ready_if_accepted"},
        "executive_summary": {"decision_status": "candidate_ready",
                              "analyst_attention_flags": ["existing_flag"]},
        "primary_hypothesis": {"candidate_id": "FM::CAND-A"},
        "recommended_actions": [],
        "contributing_causes": [],
    }


def _real_manifest(any_change: bool) -> dict:
    """Call the real _stage_g_finalize_manifest.

    any_change=True: empty core families (kg/anomaly/chroma 'missing') with
    candidates whose composite_raw exceeds composite_score, so restoring a
    missing core source lifts the score past the 0.02 threshold →
    any_ranking_change_possible is True.

    any_change=False: fully populated core + optional families so every assessed
    source is 'complete'; the only degraded families (protection_logic_context,
    configuration_change_records) are 'not_assessed', whose restoration leaves the
    quality factor unchanged → no positive delta → flag not injected.
    """
    o = _orchestrator()
    if any_change:
        kg = {"subgraph_id": "KGCTX::1", "components": [], "failure_modes": [], "past_events": []}
        tskr = {"patterns": []}
        evidence = {"results": []}
        cands = {"candidates": [
            {"candidate_id": "FM::CAND-A", "event_id": "EVT-1", "composite_score": 0.80,
             "scores": {"composite_raw": 0.86}, "quality_multiplier": 0.93},
            {"candidate_id": "FM::CAND-B", "event_id": "EVT-1", "composite_score": 0.76,
             "scores": {"composite_raw": 0.82}, "quality_multiplier": 0.927},
        ], "provenance": {}}
        telemetry = soe = alarm = None
    else:
        kg = {"subgraph_id": "KGCTX::1",
              "components": [{"component_id": "C1"}],
              "failure_modes": [{"fm_id": "FM1"}],
              "past_events": [{"event_id": "P1"}]}
        tskr = {"patterns": [{"pattern_id": "T1"}]}
        evidence = {"results": [{"r": 1}, {"r": 2}, {"r": 3}]}
        cands = {"candidates": [
            {"candidate_id": "FM::CAND-A", "event_id": "EVT-1", "composite_score": 0.80,
             "scores": {"composite_raw": 0.80}, "quality_multiplier": 1.0},
        ], "provenance": {}}
        telemetry = {"signals": [{"tag_id": "S1", "data_quality": {"missing_fraction": 0.0}}]}
        soe = {"quality": {"clock_sync_ok": True, "dropped_record_count": 0}, "records": [{"x": 1}]}
        alarm = {"quality": {"clock_sync_ok": True, "missing_fraction": 0.0}, "alarms": [{"a": 1}]}

    return o._stage_g_finalize_manifest(
        run_context={"run_id": "RUN-1", "input_refs": {"event_id": "EVT-1", "asset_id": "ASSET-1"}},
        kg_context=kg,
        tskr_patterns=tskr,
        causality_candidates=cands,
        causality_candidates_pre_refine=None,
        evidence_bundle=evidence,
        ishikawa_matrix=None,
        cmms_context=None,
        rca_card=_rca_card(),
        input_validation={"ok": True},
        output_validation={"ok": True},
        optional_artifact_failures=[],
        kg_governance={"status": "green", "issues": [], "failure_mode_count": 0,
                       "min_failure_modes_required": 0},
        barrier_analysis={"barriers": [], "summary": {"overall_status": "green",
                          "barrier_count": 0, "degraded_barrier_count": 0}},
        reentry_execution={"auto_reentry_enabled": False, "attempt_count": 1,
                           "attempts": [{"attempt_index": 1, "status": "completed"}],
                           "reentry_hook": {"should_reenter": False, "reason": "no_rank_inversion"}},
        reentry_hook={"should_reenter": False, "reason": "no_rank_inversion"},
        telemetry_summary=telemetry,
        soe_log=soe,
        alarm_log=alarm,
    )


def test_manifest_sensitivity_table_key_present():
    manifest = _real_manifest(any_change=False)
    assert "sensitivity_table" in manifest


def test_manifest_artifacts_sensitivity_present_flag():
    manifest = _real_manifest(any_change=False)
    assert manifest["artifacts"]["sensitivity_table"]["present"] is True


def test_manifest_artifacts_row_count():
    """Two penalised candidates × eight degraded source families → 16 rows."""
    manifest = _real_manifest(any_change=True)
    assert manifest["artifacts"]["sensitivity_table"]["row_count"] == 16


def test_analyst_attention_flag_injected_when_change_possible():
    manifest = _real_manifest(any_change=True)
    assert manifest["artifacts"]["sensitivity_table"]["any_ranking_change_possible"] is True
    assert any("SENSITIVITY" in f for f in manifest["analyst_attention_flags"])


def test_analyst_attention_flag_not_injected_when_no_change():
    manifest = _real_manifest(any_change=False)
    assert manifest["artifacts"]["sensitivity_table"]["any_ranking_change_possible"] is False
    assert not any("SENSITIVITY" in f for f in manifest["analyst_attention_flags"])


def test_analyst_attention_flag_appended_not_replacing():
    manifest = _real_manifest(any_change=True)
    assert "existing_flag" in manifest["analyst_attention_flags"]
