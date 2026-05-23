"""D12 — Two-run scope state transfer (checkpoint/resume).

Invariant: When Run 1 detects a scope boundary (out-of-scope causal component
flagged as novel TSKR pattern) and the analyst accepts the scope expansion,
Run 2 must apply the boundary filter deterministically.  Run 2 with a rejected
decision must produce the same scope as Run 1 (no filter active).

Source: TC-7 (RCP-C seal leakoff / HX fouling scope expansion scenario)
         D12 in rca_robustness_cross_check_plan.md
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
import sys

_repo_root = Path(__file__).parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_rca_root = Path(__file__).parents[1]
if str(_rca_root) not in sys.path:
    sys.path.insert(0, str(_rca_root))

from tests.shared.run_helpers import (  # noqa: E402
    build_fixture_orchestrator,
    load_fixtures,
    run_rca,
)

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------
TC7_FIXTURE_DIR = Path(__file__).parents[1] / "tests" / "test_case_7" / "fixtures"

HX_CANDIDATE_ID = "FM::FM-SWHX4C-FOULING"
HX_COMPONENT_ID = "U1-SWP-SEAL-WATER-HX-4C"
SEAL_CANDIDATE_ID = "FM::FM-RCPC-SEAL-CV-DRIFT"
EXPECTED_SIGNAL_PREFIX = "SEX::NOVEL::"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_ids(result: dict, bucket: str = "candidates") -> set[str]:
    return {
        c.get("candidate_id", "")
        for c in (result.get("causality_candidates") or {}).get(bucket) or []
    }


def _scope_filter(result: dict) -> dict:
    return (
        (result.get("run_manifest") or {})
        .get("artifacts", {})
        .get("scope_filter") or {}
    )


def _expansion_suggestions(result: dict) -> list[dict]:
    return (
        (result.get("run_context") or {})
        .get("scope_management") or {}
    ).get("expansion_suggestions") or []


# ---------------------------------------------------------------------------
# Shared Run-1 result (computed once, reused by all D12 subtests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tc7_fixtures():
    return load_fixtures(TC7_FIXTURE_DIR)


@pytest.fixture(scope="module")
def tc7_run1(tc7_fixtures):
    """Run TC-7 with no scope seed.  Returns (orchestrator, result)."""
    with tempfile.TemporaryDirectory() as tmp:
        orc = build_fixture_orchestrator(tmp, top_k_candidates=5, enable_ishikawa=False)
        r1 = run_rca(orc, tc7_fixtures)
        return orc, r1


@pytest.fixture(scope="module")
def tc7_run2_accepted(tc7_fixtures, tc7_run1):
    """Run 2 with accepted scope expansion."""
    orc1, r1 = tc7_run1
    suggestions = _expansion_suggestions(r1)
    assert suggestions, "Run 1 must have generated expansion suggestions"
    sig_id = suggestions[0]["signal_id"]
    run_id = (r1.get("run_manifest") or {}).get("run_id") or "R1"

    with tempfile.TemporaryDirectory() as tmp:
        orc2 = build_fixture_orchestrator(tmp, top_k_candidates=5, enable_ishikawa=False)
        updated_ctx = orc2.resolve_expansion_suggestion(
            run_id=run_id,
            run_context=r1.get("run_context") or {},
            signal_id=sig_id,
            decision="accepted",
            rationale="HX inlet temperature precursor is plausible thermal path to seal degradation",
            persist=False,
        )
        scope_mgmt_v1 = updated_ctx.get("scope_management") or {}
        r2 = run_rca(orc2, tc7_fixtures, initial_scope_management=scope_mgmt_v1)
        return scope_mgmt_v1, r2


@pytest.fixture(scope="module")
def tc7_run2_rejected(tc7_fixtures, tc7_run1):
    """Run 2 with rejected scope expansion (scope unchanged)."""
    orc1, r1 = tc7_run1
    suggestions = _expansion_suggestions(r1)
    assert suggestions, "Run 1 must have generated expansion suggestions"
    sig_id = suggestions[0]["signal_id"]
    run_id = (r1.get("run_manifest") or {}).get("run_id") or "R1"

    with tempfile.TemporaryDirectory() as tmp:
        orc3 = build_fixture_orchestrator(tmp, top_k_candidates=5, enable_ishikawa=False)
        rejected_ctx = orc3.resolve_expansion_suggestion(
            run_id=run_id,
            run_context=r1.get("run_context") or {},
            signal_id=sig_id,
            decision="rejected",
            rationale="Out of scope for this CR — defer to system engineering review",
            persist=False,
        )
        scope_mgmt_rej = rejected_ctx.get("scope_management") or {}
        r2 = run_rca(orc3, tc7_fixtures, initial_scope_management=scope_mgmt_rej)
        return scope_mgmt_rej, r2


# ---------------------------------------------------------------------------
# D12-A: Run 1 scope boundary detection
# ---------------------------------------------------------------------------

class TestD12A:
    """Run 1 must detect the out-of-scope HX signal and surface it as a pending
    expansion suggestion.  No scope filter should be applied in Run 1."""

    def test_d12a_no_scope_filter_applied(self, tc7_run1):
        _, r1 = tc7_run1
        sf = _scope_filter(r1)
        assert sf.get("applied") is False, (
            f"Run 1 scope_filter.applied must be False (no boundary active at v0). Got: {sf}"
        )

    def test_d12a_scope_version_is_zero(self, tc7_run1):
        _, r1 = tc7_run1
        sf = _scope_filter(r1)
        assert sf.get("approved_scope_version") == 0, (
            f"Run 1 approved_scope_version must be 0. Got: {sf}"
        )

    def test_d12a_expansion_suggestion_generated(self, tc7_run1):
        """At least one pending scope expansion suggestion from the HX novel TSKR pattern."""
        _, r1 = tc7_run1
        suggestions = _expansion_suggestions(r1)
        assert len(suggestions) >= 1, (
            "Run 1 must generate at least one scope expansion suggestion. "
            f"suggestions={suggestions}"
        )

    def test_d12a_suggestion_is_pending(self, tc7_run1):
        _, r1 = tc7_run1
        suggestions = _expansion_suggestions(r1)
        pending = [s for s in suggestions if s.get("analyst_decision") == "pending"]
        assert pending, (
            f"At least one suggestion must have analyst_decision='pending'. Got: {suggestions}"
        )

    def test_d12a_suggestion_is_novel_signal(self, tc7_run1):
        """The trigger type must be novel_signal_pattern (Source 3 from TSKR)."""
        _, r1 = tc7_run1
        suggestions = _expansion_suggestions(r1)
        types = {s.get("trigger_type") for s in suggestions}
        assert "novel_signal_pattern" in types, (
            f"Expected trigger_type='novel_signal_pattern' in suggestions. Got triggers: {types}"
        )

    def test_d12a_suggestion_id_starts_with_novel_prefix(self, tc7_run1):
        _, r1 = tc7_run1
        suggestions = _expansion_suggestions(r1)
        ids = [s.get("signal_id", "") for s in suggestions]
        assert any(sid.startswith(EXPECTED_SIGNAL_PREFIX) for sid in ids), (
            f"Expected at least one signal_id starting with '{EXPECTED_SIGNAL_PREFIX}'. Got: {ids}"
        )

    def test_d12a_hx_candidate_present_in_run1(self, tc7_run1):
        """Without a scope filter, the HX fouling candidate must be retained."""
        _, r1 = tc7_run1
        retained = _candidate_ids(r1)
        assert HX_CANDIDATE_ID in retained, (
            f"Run 1 (no filter) must retain {HX_CANDIDATE_ID}. retained={retained}"
        )

    def test_d12a_seal_candidate_present_in_run1(self, tc7_run1):
        """Without a scope filter, the seal drift candidate must also be retained."""
        _, r1 = tc7_run1
        retained = _candidate_ids(r1)
        assert SEAL_CANDIDATE_ID in retained, (
            f"Run 1 must retain {SEAL_CANDIDATE_ID}. retained={retained}"
        )


# ---------------------------------------------------------------------------
# D12-B: Run 2 with accepted scope — boundary filter applied, HX in scope
# ---------------------------------------------------------------------------

class TestD12B:
    """After accepting the scope expansion, Run 2 must activate the scope
    filter.  The HX fouling candidate (the accepted component) must be
    retained; other-component candidates are scope-filtered out."""

    def test_d12b_accepted_scope_version_is_one(self, tc7_run2_accepted):
        scope_mgmt_v1, _ = tc7_run2_accepted
        assert scope_mgmt_v1.get("active_scope_version") == 1, (
            f"Accepted scope must have active_scope_version=1. Got: {scope_mgmt_v1}"
        )

    def test_d12b_scope_filter_applied(self, tc7_run2_accepted):
        _, r2 = tc7_run2_accepted
        sf = _scope_filter(r2)
        assert sf.get("applied") is True, (
            f"Run 2 (accepted) scope_filter.applied must be True. Got: {sf}"
        )

    def test_d12b_approved_scope_version_is_one(self, tc7_run2_accepted):
        _, r2 = tc7_run2_accepted
        sf = _scope_filter(r2)
        assert sf.get("approved_scope_version") == 1, (
            f"Run 2 (accepted) approved_scope_version must be 1. Got: {sf}"
        )

    def test_d12b_hx_candidate_retained(self, tc7_run2_accepted):
        """The HX fouling candidate must survive the scope filter (it is the accepted component)."""
        _, r2 = tc7_run2_accepted
        retained = _candidate_ids(r2)
        assert HX_CANDIDATE_ID in retained, (
            f"Run 2 (accepted) must retain {HX_CANDIDATE_ID}. retained={retained}"
        )

    def test_d12b_hx_component_in_scope_boundary(self, tc7_run2_accepted):
        """The accepted scope boundary must include the HX component."""
        scope_mgmt_v1, _ = tc7_run2_accepted
        revisions = scope_mgmt_v1.get("scope_revisions") or []
        latest_accepted = next(
            (r for r in reversed(revisions) if r.get("analyst_decision") == "accepted"), None
        )
        assert latest_accepted is not None, "No accepted revision found after acceptance"
        scope_cids = [
            c.lower()
            for c in (latest_accepted.get("scope_snapshot") or {}).get("component_ids") or []
        ]
        assert HX_COMPONENT_ID.lower() in scope_cids, (
            f"HX component {HX_COMPONENT_ID!r} must be in scope boundary. "
            f"scope_cids={scope_cids}"
        )

    def test_d12b_seal_candidate_scope_filtered(self, tc7_run2_accepted):
        """Seal drift candidate (U1-RCP-C-SEAL1-PKG) is outside the accepted HX-only scope
        and must appear in ruled_out with reason_code='scope_filtered'."""
        _, r2 = tc7_run2_accepted
        ruled_out = (r2.get("causality_candidates") or {}).get("ruled_out") or []
        scope_filtered = [ro for ro in ruled_out if ro.get("reason_code") == "scope_filtered"]
        scope_filtered_ids = {ro.get("candidate_id") for ro in scope_filtered}
        assert SEAL_CANDIDATE_ID in scope_filtered_ids, (
            f"Run 2 (accepted) must scope-filter {SEAL_CANDIDATE_ID}. "
            f"scope_filtered_ids={scope_filtered_ids}"
        )

    def test_d12b_filter_count_nonzero(self, tc7_run2_accepted):
        _, r2 = tc7_run2_accepted
        sf = _scope_filter(r2)
        assert int(sf.get("filtered_count") or 0) >= 1, (
            f"Run 2 (accepted) scope_filter.filtered_count must be >= 1. Got: {sf}"
        )


# ---------------------------------------------------------------------------
# D12-C: Run 2 with rejected scope — no change, same scope as Run 1
# ---------------------------------------------------------------------------

class TestD12C:
    """When the analyst rejects the expansion, the scope is not updated.
    Run 2 with the rejected scope must behave identically to Run 1."""

    def test_d12c_rejected_scope_version_still_zero(self, tc7_run2_rejected):
        scope_mgmt_rej, _ = tc7_run2_rejected
        assert scope_mgmt_rej.get("active_scope_version") == 0, (
            f"Rejected scope must have active_scope_version=0. Got: {scope_mgmt_rej}"
        )

    def test_d12c_no_scope_filter_applied(self, tc7_run2_rejected):
        _, r2 = tc7_run2_rejected
        sf = _scope_filter(r2)
        assert sf.get("applied") is False, (
            f"Run 2 (rejected) scope_filter.applied must be False (version=0). Got: {sf}"
        )

    def test_d12c_hx_candidate_retained(self, tc7_run2_rejected):
        """Without an active filter, the HX candidate must still be retained."""
        _, r2 = tc7_run2_rejected
        retained = _candidate_ids(r2)
        assert HX_CANDIDATE_ID in retained, (
            f"Run 2 (rejected) must retain {HX_CANDIDATE_ID} (no filter). retained={retained}"
        )

    def test_d12c_seal_candidate_retained(self, tc7_run2_rejected):
        """Without a filter, the seal drift candidate must also be retained."""
        _, r2 = tc7_run2_rejected
        retained = _candidate_ids(r2)
        assert SEAL_CANDIDATE_ID in retained, (
            f"Run 2 (rejected) must retain {SEAL_CANDIDATE_ID} (no filter). retained={retained}"
        )

    def test_d12c_same_candidates_as_run1(self, tc7_run1, tc7_run2_rejected):
        """Run 2 (rejected) retained set must equal Run 1 retained set."""
        _, r1 = tc7_run1
        _, r2 = tc7_run2_rejected
        run1_ids = _candidate_ids(r1)
        run2_ids = _candidate_ids(r2)
        assert run2_ids == run1_ids, (
            f"Run 2 (rejected) must produce the same candidate set as Run 1. "
            f"Run1={sorted(run1_ids)}, Run2={sorted(run2_ids)}"
        )


# ---------------------------------------------------------------------------
# D12-D: Scope state is deterministic (re-running with the same scope gives same result)
# ---------------------------------------------------------------------------

class TestD12D:
    """The scope state transfer is deterministic: running twice with the same
    initial_scope_management yields the same retained candidate set."""

    def test_d12d_accepted_scope_is_deterministic(self, tc7_fixtures, tc7_run1):
        """Two independent runs with the same accepted scope must produce the same retained set."""
        orc1, r1 = tc7_run1
        suggestions = _expansion_suggestions(r1)
        sig_id = suggestions[0]["signal_id"]
        run_id = (r1.get("run_manifest") or {}).get("run_id") or "R1"

        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                orc = build_fixture_orchestrator(tmp, top_k_candidates=5, enable_ishikawa=False)
                updated_ctx = orc.resolve_expansion_suggestion(
                    run_id=run_id,
                    run_context=r1.get("run_context") or {},
                    signal_id=sig_id,
                    decision="accepted",
                    rationale="determinism check",
                    persist=False,
                )
                sm = updated_ctx.get("scope_management") or {}
                r = run_rca(orc, tc7_fixtures, initial_scope_management=sm)
                results.append(_candidate_ids(r))

        assert results[0] == results[1], (
            f"Scope-state-transfer must be deterministic. "
            f"Run A candidates={sorted(results[0])}, Run B candidates={sorted(results[1])}"
        )
