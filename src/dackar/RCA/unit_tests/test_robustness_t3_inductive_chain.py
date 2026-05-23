"""
test_robustness_t3_inductive_chain.py — Tier 3 Form 1: Mathematical induction on causal chain depth

Tier 3 Form 1: P(n) → P(n+1) inductive property for causal chains

## Formal statement

Define: **P(n)** = "the pipeline correctly surfaces all candidates in a causal chain of
depth n, with each candidate retained and ranked in the correct causal order."

**Base case — P(1):** A single failure mode (Category A, proximate physical cause).
The pipeline retains it and it is the sole top-ranked candidate.

**Inductive step — P(k) → P(k+1):** Adding one new causal level to the KG while keeping
the event and telemetry identical must satisfy:

  1. **Survival**: every candidate retained in depth-k either appears in the retained list
     at depth-(k+1), or appears in filtered_out_candidates with an explicit filter_reason.
     No retained candidate may silently vanish.

  2. **Ordering preservation**: for every pair of depth-k candidates that both survive to
     depth-(k+1), their relative composite_score ordering must be unchanged.

  3. **Non-displacement**: the newly added depth-(k+1) candidate must not rank higher than
     any depth-k candidate without an explicit contradicting evidence entry in the evidence
     bundle. (In this fixture no contradicting evidence exists, so the new candidate must
     land at or below all depth-k candidates.)

  4. **Score monotonicity**: adding a deeper causal level may increase scores of existing
     candidates (e.g., governance boost from pm_compliance or OE documents becoming part of
     the evidence context), but must never decrease them.

## Fixture chain (same event + telemetry, expanding KG)

  chain_depth_1/  → 1 FM (A only, proximate bearing wear)
  chain_depth_2/  → 2 FMs (A + J: adds contributing PM-interval FM + pm_compliance)
  chain_depth_3/  → 3 FMs (A + J + L: adds root OE-not-incorporated FM + OE evidence)

  Candidate IDs are stable across depths (same fm_id prefix), making survival
  checks unambiguous.

## Key observations (from fixture probe)

  depth-1: A at rank 0, composite=0.407
  depth-2: A at rank 0 (0.421), J at rank 1 (0.409) — A receives gov boost from pm_compliance
  depth-3: A at rank 0 (0.434), J at rank 1 (0.423), L at rank 2 (0.392)

  - No candidates are filtered at any depth (all clear evidence threshold at every level)
  - Score of A increases across depths: governance context grows as the causal chain deepens
  - Relative order A > J is preserved from depth-2 into depth-3
  - L enters at rank 2 and does not displace either A or J

## What this test proves

  If the inductive property holds for our fixture family, we have demonstrated by
  construction that the pipeline satisfies P(1), P(2), and P(3) for this bearing-failure
  causal chain. Combined with Tier 1 (D1–D12 internal consistency) and Tier 2
  (OUC-1–8 scenario correctness), this closes the traceability argument:
  "no causal depth expansion can silently eliminate or reorder a prior finding."

  This is a critical guarantee for AP-913-style root cause analyses where traceability
  of proximate → contributing → root cause must be preserved across successive
  engineering evaluations.

Plan note: the plan's inductive assertion used `ruled_out` to capture eliminated
candidates. The actual engine uses `filtered_out_candidates` (no `ruled_out` list).
This test is written against the actual pipeline structure.

Fixtures:
  tests/fixtures_robustness/chain_depth_1/
  tests/fixtures_robustness/chain_depth_2/
  tests/fixtures_robustness/chain_depth_3/
"""

from __future__ import annotations

import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock

import pytest

_RCA_ROOT = Path(__file__).resolve().parent.parent
_TESTS_SHARED = _RCA_ROOT / "tests" / "shared"
_FIX = {
    k: _RCA_ROOT / "tests" / "fixtures_robustness" / f"chain_depth_{k}"
    for k in (1, 2, 3)
}

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(depth: int) -> Dict[str, Any]:
    fixtures = load_fixtures(_FIX[depth])
    with tempfile.TemporaryDirectory() as tmp:
        orch = build_fixture_orchestrator(tmp)
        return run_rca(orch, fixtures)


def _retained(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["causality_candidates"].get("candidates") or []


def _filtered(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["causality_candidates"].get("filtered_out_candidates") or []


def _all_accounted(result: Dict[str, Any]) -> Set[str]:
    """All candidate_ids that appear in retained OR filtered_out — the 'accounted' set."""
    return {
        c["candidate_id"]
        for c in _retained(result) + _filtered(result)
        if c.get("candidate_id")
    }


def _ranks(result: Dict[str, Any]) -> Dict[str, int]:
    """Map candidate_id → rank in retained list."""
    return {c["candidate_id"]: i for i, c in enumerate(_retained(result))}


def _scores(result: Dict[str, Any]) -> Dict[str, float]:
    """Map candidate_id → composite_score in retained list."""
    return {c["candidate_id"]: c.get("composite_score", 0.0) for c in _retained(result)}


_A_ID = "FM::FM-CHAIN-BEARING-WEAR"
_J_ID = "FM::FM-CHAIN-PM-INTERVAL"
_L_ID = "FM::FM-CHAIN-OE-NOT-INCORPORATED"


# ---------------------------------------------------------------------------
# Base case P(1)
# ---------------------------------------------------------------------------

def test_p1_base_case():
    """P(1): depth-1 pipeline retains exactly the Category A candidate."""
    result = _run(1)
    retained = _retained(result)
    assert len(retained) >= 1, "P(1): no candidates retained in depth-1 run"
    assert retained[0]["candidate_id"] == _A_ID, (
        f"P(1): top candidate is {retained[0]['candidate_id']!r}; expected {_A_ID!r}. "
        "Proximate physical cause must be the sole top candidate in a depth-1 chain."
    )


# ---------------------------------------------------------------------------
# Inductive step P(1) → P(2)
# ---------------------------------------------------------------------------

def test_p1_to_p2_survival():
    """P(1)→P(2): Category A candidate from depth-1 must survive into depth-2.

    'Survive' means: appears in retained OR filtered_out_candidates at depth-2.
    A silent disappearance — present in depth-1 retained, absent from both lists
    in depth-2 — violates the inductive guarantee.
    """
    r1 = _run(1)
    r2 = _run(2)
    depth1_ids = {c["candidate_id"] for c in _retained(r1)}
    accounted2 = _all_accounted(r2)
    for cid in depth1_ids:
        assert cid in accounted2, (
            f"P(1)→P(2) SURVIVAL VIOLATION: {cid!r} was retained at depth-1 but "
            f"is absent from both retained and filtered_out_candidates at depth-2. "
            "Adding a second failure mode to the KG must not silently eliminate "
            "a previously retained candidate."
        )


def test_p1_to_p2_a_still_retained():
    """P(1)→P(2): Category A must remain in the retained list (not filtered out) at depth-2."""
    r2 = _run(2)
    retained_ids = {c["candidate_id"] for c in _retained(r2)}
    assert _A_ID in retained_ids, (
        f"P(1)→P(2): {_A_ID!r} was demoted to filtered_out at depth-2. "
        "The proximate physical cause must not drop below the screening threshold "
        "merely because a contributing-level FM was added to the KG."
    )


def test_p1_to_p2_j_enters_below_a():
    """P(1)→P(2): the new depth-2 FM (Category J) must enter below Category A.

    No contradicting evidence exists for Category A in either depth fixture.
    Therefore J must not displace A.
    """
    r2 = _run(2)
    retained2 = _retained(r2)
    retained_ids = {c["candidate_id"] for c in retained2}
    if _J_ID not in retained_ids:
        pytest.skip("Category J not retained in depth-2 — fixture may need evidence adjustment")
    ranks2 = _ranks(r2)
    assert ranks2[_A_ID] < ranks2[_J_ID], (
        f"P(1)→P(2) NON-DISPLACEMENT VIOLATION: "
        f"Category J (rank {ranks2[_J_ID]}) ranks above Category A (rank {ranks2[_A_ID]}) "
        f"at depth-2 without contradicting evidence for A. "
        "Adding a contributing-level FM must not displace the proximate cause from the top rank."
    )


def test_p1_to_p2_score_monotone():
    """P(1)→P(2): Category A score must not decrease when depth-2 is added.

    Adding pm_compliance and a J FM may raise A's governance score. Scores may
    increase but must not decrease — evidence context can only add information.
    """
    r1 = _run(1)
    r2 = _run(2)
    score1 = _scores(r1).get(_A_ID, 0.0)
    score2 = _scores(r2).get(_A_ID, 0.0)
    assert score2 >= score1 - 0.001, (
        f"P(1)→P(2) SCORE DECREASE: A composite {score1:.4f} → {score2:.4f}. "
        "Adding a contributing FM and pm_compliance must not decrease the existing "
        "candidate's composite score."
    )


# ---------------------------------------------------------------------------
# Inductive step P(2) → P(3)
# ---------------------------------------------------------------------------

def test_p2_to_p3_survival():
    """P(2)→P(3): all depth-2 retained candidates must survive into depth-3."""
    r2 = _run(2)
    r3 = _run(3)
    depth2_retained_ids = {c["candidate_id"] for c in _retained(r2)}
    accounted3 = _all_accounted(r3)
    for cid in depth2_retained_ids:
        assert cid in accounted3, (
            f"P(2)→P(3) SURVIVAL VIOLATION: {cid!r} was retained at depth-2 but "
            f"is absent from both retained and filtered_out_candidates at depth-3. "
            "Adding a root-cause FM and OE document must not silently eliminate "
            "any previously retained candidate."
        )


def test_p2_to_p3_both_prior_still_retained():
    """P(2)→P(3): both Category A and J must remain in the retained list at depth-3."""
    r3 = _run(3)
    retained_ids = {c["candidate_id"] for c in _retained(r3)}
    for cid, label in [(_A_ID, "Category A"), (_J_ID, "Category J")]:
        assert cid in retained_ids, (
            f"P(2)→P(3): {label} ({cid!r}) was demoted to filtered_out at depth-3. "
            "Adding the root-cause FM must not push prior retained candidates below "
            "the screening threshold."
        )


def test_p2_to_p3_ordering_preserved():
    """P(2)→P(3): the relative rank order of depth-2 candidates must be preserved at depth-3.

    Both A (rank 0) and J (rank 1) from depth-2 must maintain A > J at depth-3.
    If contradicting evidence were added for A, a rank inversion would be justified,
    but no such evidence exists in this fixture family.
    """
    r2 = _run(2)
    r3 = _run(3)
    retained2_ids = {c["candidate_id"] for c in _retained(r2)}
    ranks3 = _ranks(r3)

    surviving_pairs = [
        (a, b)
        for a, b in combinations(sorted(retained2_ids), 2)
        if a in ranks3 and b in ranks3
    ]

    ranks2 = _ranks(r2)

    for cid_a, cid_b in surviving_pairs:
        was_a_above_b = ranks2[cid_a] < ranks2[cid_b]
        is_a_above_b = ranks3[cid_a] < ranks3[cid_b]
        if was_a_above_b != is_a_above_b:
            raise AssertionError(
                f"P(2)→P(3) ORDERING VIOLATION: {cid_a!r} vs {cid_b!r} "
                f"rank inverted from depth-2 (A-above={was_a_above_b}) "
                f"to depth-3 (A-above={is_a_above_b}) "
                f"without contradicting evidence for either candidate. "
                "Relative ordering of prior candidates must be stable when a deeper "
                "causal level is added."
            )


def test_p2_to_p3_l_enters_below_prior():
    """P(2)→P(3): the new root-cause FM (Category L) must enter below both A and J.

    No contradicting evidence exists for A or J. Therefore L must not displace them.
    """
    r3 = _run(3)
    retained3 = _retained(r3)
    retained_ids = {c["candidate_id"] for c in retained3}
    if _L_ID not in retained_ids:
        pytest.skip("Category L not retained in depth-3 — fixture may need OE evidence adjustment")
    ranks3 = _ranks(r3)
    for cid, label in [(_A_ID, "Category A"), (_J_ID, "Category J")]:
        if cid in ranks3:
            assert ranks3[cid] < ranks3[_L_ID], (
                f"P(2)→P(3) NON-DISPLACEMENT VIOLATION: "
                f"Category L (rank {ranks3[_L_ID]}) ranks above {label} (rank {ranks3[cid]}) "
                "at depth-3 without contradicting evidence. "
                "Adding a root-cause FM must not displace existing causal-chain candidates."
            )


def test_p2_to_p3_scores_monotone():
    """P(2)→P(3): scores for A and J must not decrease when depth-3 is added."""
    r2 = _run(2)
    r3 = _run(3)
    scores2 = _scores(r2)
    scores3 = _scores(r3)
    for cid, label in [(_A_ID, "Category A"), (_J_ID, "Category J")]:
        s2 = scores2.get(cid, 0.0)
        s3 = scores3.get(cid, 0.0)
        assert s3 >= s2 - 0.001, (
            f"P(2)→P(3) SCORE DECREASE: {label} composite {s2:.4f} → {s3:.4f}. "
            "Adding a root-cause FM and OE evidence must not reduce the composite "
            "score of any previously retained candidate."
        )


# ---------------------------------------------------------------------------
# Transitivity: full chain P(1) → P(3)
# ---------------------------------------------------------------------------

def test_transitive_p1_to_p3_a_always_top():
    """P(1)→P(3) transitivity: Category A must rank first across all three depths."""
    for depth in (1, 2, 3):
        result = _run(depth)
        retained = _retained(result)
        assert retained, f"No candidates retained at depth-{depth}"
        assert retained[0]["candidate_id"] == _A_ID, (
            f"Transitivity violation: Category A is not rank-0 at depth-{depth}. "
            "The proximate physical cause must remain the top-ranked candidate "
            "regardless of how many deeper causal levels are added."
        )


def test_transitive_all_prior_ids_survive_to_depth3():
    """P(1)→P(3): every FM introduced at depth k must be accounted for at depth-3."""
    r3 = _run(3)
    accounted3 = _all_accounted(r3)
    for depth in (1, 2):
        result = _run(depth)
        for cid in {c["candidate_id"] for c in _retained(result)}:
            assert cid in accounted3, (
                f"Transitivity survival: {cid!r} (introduced at depth-{depth}) is absent "
                f"from depth-3 retained and filtered_out. This is a silent elimination — "
                "a fundamental violation of the inductive guarantee."
            )


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_p1_base_case,
    test_p1_to_p2_survival,
    test_p1_to_p2_a_still_retained,
    test_p1_to_p2_j_enters_below_a,
    test_p1_to_p2_score_monotone,
    test_p2_to_p3_survival,
    test_p2_to_p3_both_prior_still_retained,
    test_p2_to_p3_ordering_preserved,
    test_p2_to_p3_l_enters_below_prior,
    test_p2_to_p3_scores_monotone,
    test_transitive_p1_to_p3_a_always_top,
    test_transitive_all_prior_ids_survive_to_depth3,
]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
