"""
Unit tests for dackar.RCA.ner.causal_condition_adapter._chain_causal_statements.

Locks the normal (sparse) chain-linking behavior and guards the path-enumeration
bound added in Plan B: a dense, mutually-similar statement set previously made the
all-paths DFS blow up exponentially (a ~12-node fully-connected graph hung for
>120s). The bound must keep that terminating without changing sparse output.
"""
from __future__ import annotations

import time

from dackar.RCA.ner.causal_condition_adapter import (
    _chain_causal_statements,
    _MAX_CHAIN_PATHS,
)


# ---------------------------------------------------------------------------
# Normal / sparse behavior (unchanged by the bound)
# ---------------------------------------------------------------------------

def test_linear_chain_linked():
    stmts = [
        {"cause_text": "pump seal wear", "effect_text": "coolant leakage occurred"},
        {"cause_text": "coolant leakage occurred", "effect_text": "low reactor coolant level"},
        {"cause_text": "low reactor coolant level", "effect_text": "reactor scram initiated"},
    ]
    chains = _chain_causal_statements(stmts)
    assert len(chains) == 1
    assert chains[0]["nodes"] == [
        "pump seal wear",
        "coolant leakage occurred",
        "low reactor coolant level",
        "reactor scram initiated",
    ]
    assert chains[0]["length"] >= 2


def test_fewer_than_two_statements_returns_empty():
    assert _chain_causal_statements([]) == []
    assert _chain_causal_statements([{"cause_text": "a", "effect_text": "b"}]) == []


def test_unlinked_statements_return_empty():
    stmts = [
        {"cause_text": "pump seal wear", "effect_text": "bearing overheated"},
        {"cause_text": "valve stuck open", "effect_text": "tank overflowed"},
    ]
    # no effect[i] matches any cause[j] -> no edges -> no chains
    assert _chain_causal_statements(stmts) == []


def test_chain_dicts_have_expected_keys():
    stmts = [
        {"cause_text": "pump seal wear", "effect_text": "coolant leakage occurred"},
        {"cause_text": "coolant leakage occurred", "effect_text": "low reactor coolant level"},
    ]
    ch = _chain_causal_statements(stmts)[0]
    for key in ("chain_id", "length", "nodes", "source_statement_ids",
                "cross_sentence", "min_confidence"):
        assert key in ch


# ---------------------------------------------------------------------------
# Path-enumeration bound (Plan B) — dense graph must terminate quickly
# ---------------------------------------------------------------------------

def test_dense_graph_bounded_by_cap():
    """A fully-connected statement set (every effect == every cause) previously
    enumerated factorially many paths and hung.  With the bound it must return a
    number of chains capped at _MAX_CHAIN_PATHS — and promptly.

    n=12 is chosen so that WITHOUT the cap this enumerates factorially many paths
    and would stall CI (not merely run slow): asserting the count against the cap,
    not just the runtime, makes a removed-cap regression fail hard.
    """
    shared = "reactor coolant system leakage event"
    stmts = [{"cause_text": shared, "effect_text": shared} for _ in range(12)]
    t0 = time.perf_counter()
    chains = _chain_causal_statements(stmts)
    elapsed = time.perf_counter() - t0
    assert 0 < len(chains) <= _MAX_CHAIN_PATHS
    assert elapsed < 5.0, f"chain linking took {elapsed:.1f}s on a dense graph"


def test_dense_cluster_does_not_starve_unrelated_chain(monkeypatch):
    """Per-source path budget (PR#84 review): a dense cluster processed first must
    not consume the whole allowance and drop an unrelated sparse chain.  With a
    tiny cap the dense cluster alone would exhaust a shared budget, yet the
    distinct pump chain must still be returned."""
    import dackar.RCA.ner.causal_condition_adapter as mod
    monkeypatch.setattr(mod, "_MAX_CHAIN_PATHS", 4)

    dense = "reactor coolant system leakage event"
    stmts = [{"cause_text": dense, "effect_text": dense} for _ in range(10)]
    stmts += [
        {"cause_text": "pump seal wear", "effect_text": "auxiliary lube oil loss"},
        {"cause_text": "auxiliary lube oil loss", "effect_text": "pump bearing seizure"},
    ]
    chains = mod._chain_causal_statements(stmts)
    pump_present = any("pump seal wear" in n for c in chains for n in c["nodes"])
    assert pump_present, "unrelated sparse chain was starved by the dense cluster"
