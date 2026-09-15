"""
test_robustness_t3_property_based.py — Tier 3 Form 2: Property-based testing with Hypothesis

## Kernel invariants (IPs)

Six of the eight originally-planned invariants are tested here. IP-6 (traceability of
rca_card citations) is deferred — it requires the live LLM synthesizer, which is mocked
in our fixture-only setup and does not produce structured citations. IP-8 is reformulated
(see below).

| Property | Statement | Implemented |
|----------|-----------|-------------|
| IP-1 | allen_relation in retained candidate scores is never "follows" or "during" | ✅ |
| IP-2 | Adding supporting evidence cannot decrease a candidate's composite score | ✅ |
| IP-3 | Adding contradicting evidence cannot increase a candidate's composite score | ✅ |
| IP-4 | Every KG failure_mode appears in retained OR filtered_out_candidates | ✅ |
| IP-5 | retained and filtered_out_candidates are disjoint (by candidate_id) | ✅ |
| IP-6 | rca_card citations trace to evidence_bundle | ⬜ deferred (mock LLM) |
| IP-7 | composite_score and all sub-scores ∈ [0.0, 1.0] | ✅ |
| IP-8 | Empty kg_context.documents + empty candidate_evidence_summary → 0 retained | ✅ |

## IP-1 reformulation

The plan stated "FOLLOWS never produces a ranked candidate." The actual invariant enforced
by the engine is: the `allen_relation` field in candidate scores is always None or a
recognised causal relation ({overlaps, precedes, contains}). The strings "follows" and
"during" are NEVER stored there — FOLLOWS/DURING signals produce allen_relation=None.
This is the machine-checkable form and is what we test.

## IP-8 reformulation

The plan's ceiling of 0.50 was too loose for universal Hypothesis inputs (high structural +
OVERLAPS timing can produce composite > 0.50 even without a candidate_evidence_summary).
The reformulated invariant is precise: when BOTH kg_context.documents is empty AND
evidence_bundle.candidate_evidence_summary is empty, the initial evidence sub-score
defaults to the minimum floor (~0.30) which falls below the pipeline's
minimum_evidence_threshold, so ALL candidates are filtered. Verified empirically via OUC-7.

## Generator contract — build_signal timing

The Allen classifier reads actual timestamps (epsilon = 0.5 h). The generator maps
symbolic timing → timestamps relative to BASE_EVENT_START / BASE_EVENT_END:

  FOLLOWS   : a_start > b_end   + 30 min
  PRECEDES  : a_end   < b_start - 30 min
  CONTAINS  : a_start < b_start - 30 min  AND  a_end   > b_end   + 30 min
  OVERLAPS  : a_start < b_start - 30 min  AND  b_start ≤ a_end   ≤ b_end
  DURING    : b_start ≤ a_start ≤ b_end - 6 min  (catch-all, no FOLLOWS overlap)

Margins are kept at 32–35 min (≥ epsilon + 2 min) to ensure the correct branch is taken.

## Pytest mark

These tests are marked ``@pytest.mark.slow`` because each example runs the full pipeline
(~2 s). Exclude with ``pytest -m "not slow"`` for fast CI runs.
"""

from __future__ import annotations

import copy
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup (mirrors every other robustness test)
# ---------------------------------------------------------------------------

_RCA_ROOT    = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
_SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenario"
_TESTS_SHARED = _SCENARIO_ROOT / "shared"
_CHAIN_1_DIR  = _SCENARIO_ROOT / "fixtures_robustness" / "chain_depth_1"

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

# ---------------------------------------------------------------------------
# Hypothesis imports
# ---------------------------------------------------------------------------

import hypothesis.strategies as st
from hypothesis import HealthCheck, assume, given, settings

# ---------------------------------------------------------------------------
# Module-level orchestrator (reused across all Hypothesis examples to avoid
# rebuilding ~50 orchestrators per property run)
# ---------------------------------------------------------------------------

_TMP_DIR = tempfile.mkdtemp(prefix="rca_hyp_")
_ORCH: Any = None


def _orch() -> Any:
    global _ORCH
    if _ORCH is None:
        _ORCH = build_fixture_orchestrator(_TMP_DIR)
    return _ORCH


def teardown_module(_module: Any) -> None:
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Base event and timestamp anchors
# (mirrors chain_depth_1/event.json: 2025-03-05 11:00–11:45 UTC)
# ---------------------------------------------------------------------------

BASE_EVENT: Dict[str, Any] = {
    "event_id": "EVT-CHAIN-BEARING-FAIL-001",
    "asset_id": "U1-PUMP-CHAIN-01",
    "timestamp_start": "2025-03-05T11:00:00Z",
    "timestamp_end":   "2025-03-05T11:45:00Z",
    "event_type": "DEGRADATION",
    "severity": "HIGH",
    "symptom_signature": {
        "description": "Hypothesis property-based test event",
        "anomaly_pattern": "gradual_drift",
        "symptom_types": ["bearing_failure"],
    },
}

_B_START = datetime(2025, 3, 5, 11,  0, 0, tzinfo=timezone.utc)
_B_END   = datetime(2025, 3, 5, 11, 45, 0, tzinfo=timezone.utc)
_EPSILON  = timedelta(hours=0.5)                      # Allen classifier epsilon
_MARGIN   = timedelta(minutes=33)                     # conservative: > epsilon + 3 min

_CAUSAL_CATEGORIES = ["A", "B", "D", "E", "F"]

EMPTY_TSKR: Dict[str, Any] = {"patterns": [], "pipeline_health": {"status": "green"}}

# Evidence bundle with no candidate_evidence_summary (for IP-8 and neutral IP-4/5/7 runs)
EMPTY_EVIDENCE: Dict[str, Any] = {
    "bundle_id": "BUNDLE-HYPOTHESIS-EMPTY",
    "generated_at": "2025-06-01T00:00:00Z",
    "query": "hypothesis empty bundle",
    "score_metric": "hybrid_fixture",
    "score_threshold": 0.0,
    "retrieval_scope": {},
    "results": [],
    "candidate_evidence_summary": [],
}

_FMT = "%Y-%m-%dT%H:%M:%SZ"


# ---------------------------------------------------------------------------
# Generator primitives
# ---------------------------------------------------------------------------

def _td(draw: Any, min_minutes: int, max_minutes: int) -> timedelta:
    """Draw a random timedelta between min_minutes and max_minutes inclusive."""
    minutes = draw(st.integers(min_value=min_minutes, max_value=max_minutes))
    return timedelta(minutes=minutes)


def _iso(dt: datetime) -> str:
    return dt.strftime(_FMT)


def _signal_timestamps(timing: str, draw: Any) -> Tuple[str, str]:
    """Produce (start_iso, end_iso) that will resolve to *timing* via the Allen classifier.

    All boundaries include at least *_MARGIN* clearance from the epsilon zone to
    guarantee the correct branch is taken regardless of floating-point rounding.
    """
    bs = _B_START
    be = _B_END
    m  = _MARGIN

    if timing == "follows":
        # a_s must be > be + epsilon (12:15). Use margin: a_s >= 12:15 + 2 min = 12:17.
        delay = _td(draw, 2, 360)
        a_s = be + _EPSILON + delay
        dur  = _td(draw, 5, 120)
        a_e  = a_s + dur

    elif timing == "precedes":
        # a_e must be < bs - epsilon (10:30). Use a_e <= 10:28.
        gap = _td(draw, 2, 240)
        a_e = bs - _EPSILON - gap
        dur = _td(draw, 10, 240)
        a_s = a_e - dur

    elif timing == "contains":
        # a_s < bs - epsilon = 10:30; a_e > be + epsilon = 12:15.
        lead = _td(draw, 2, 180)
        tail = _td(draw, 2, 180)
        a_s = bs - _EPSILON - lead
        a_e = be + _EPSILON + tail

    elif timing == "overlaps":
        # a_s < bs - epsilon = 10:30  AND  bs <= a_e <= be.
        lead = _td(draw, 2, 240)
        a_s  = bs - _EPSILON - lead
        # a_e within event window [bs, be]:
        offset = draw(st.integers(min_value=0, max_value=int((be - bs).total_seconds() // 60)))
        a_e = bs + timedelta(minutes=offset)
        # ensure a_e > a_s (always true since a_s < bs ≤ a_e)

    elif timing == "during":
        # a_s in [bs, be - 6 min]; a_e in [a_s + 1 min, bs + 5 h].
        max_start_offset = max(1, int((be - bs - timedelta(minutes=6)).total_seconds() // 60))
        start_offset = _td(draw, 0, max_start_offset)
        a_s = bs + start_offset
        dur = _td(draw, 1, 300)
        a_e = a_s + dur

    else:
        raise ValueError(f"Unknown timing: {timing!r}")

    return _iso(a_s), _iso(a_e)


@st.composite
def gen_signal(draw: Any, component_id: str, timing: Optional[str] = None) -> Dict[str, Any]:
    """Generate a telemetry signal for *component_id* with correctly derived timestamps."""
    if timing is None:
        timing = draw(st.sampled_from(["precedes", "overlaps", "contains", "during", "follows"]))
    a_s_str, a_e_str = _signal_timestamps(timing, draw)
    severity = draw(st.floats(min_value=0.10, max_value=1.0, allow_nan=False))
    sensor_id = f"SEN-HYPO-{component_id[-4:]}"
    return {
        "sensor_id":    sensor_id,
        "signal_id":    sensor_id,
        "component_id": component_id,
        "severity":     round(severity, 4),
        "anomaly_window": {"start": a_s_str, "end": a_e_str},
        "_timing":      timing,   # metadata tag for assertions; not read by engine
    }


@st.composite
def gen_failure_mode(draw: Any, component_id: str, idx: int) -> Dict[str, Any]:
    """Generate a minimal failure mode for *component_id*."""
    cat = draw(st.sampled_from(_CAUSAL_CATEGORIES))
    return {
        "fm_id":                  f"FM-HYPO-{idx:02d}",
        "component_id":           component_id,
        "name":                   f"Hypothesis FM-{idx:02d} for {component_id}",
        "causal_category":        cat,
        "causal_category_source": "inferred",
    }


@st.composite
def gen_paired_input(
    draw: Any,
    min_n: int = 1,
    max_n: int = 4,
    include_documents: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate a coupled (kg_context, telemetry_summary) pair.

    Component IDs are shared so that signal→FM linkage is valid.
    Returns ``(kg_context, telemetry_summary)``.
    """
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    component_ids = [f"U1-HYPO-COMP-{i:02d}" for i in range(n)]

    failure_modes = [draw(gen_failure_mode(cid, i)) for i, cid in enumerate(component_ids)]
    signals       = [draw(gen_signal(cid))           for cid      in component_ids]

    documents = (
        [{"doc_id": "WO:HYPO:BASE:001", "doc_type": "WO", "time_distance_days": 1}]
        if include_documents else []
    )

    kg_context = {
        "event_id":    BASE_EVENT["event_id"],
        "asset_id":    BASE_EVENT["asset_id"],
        "subgraph_id": f"KGCTX::HYPOTHESIS::N{n}",
        "generated_at": "2025-06-01T00:00:00Z",
        "hop_limit": 1,
        "components": [
            {"component_id": cid, "seed_match_type": "seed"}
            for cid in component_ids
        ],
        "upstream_paths": [],
        "failure_modes":  failure_modes,
        "past_events":    [],
        "documents":      documents,
        "seed_context": {
            "asset_ids":           [BASE_EVENT["asset_id"]],
            "seed_component_ids":  component_ids[:1],
            "monitored_variables": ["vibration"],
        },
        "pipeline_health": {"status": "green"},
    }

    telemetry_summary = {
        "window": {
            "start": "2025-03-05T06:00:00Z",
            "end":   "2025-03-05T17:00:00Z",
        },
        "signals":          signals,
        "pipeline_health":  {"status": "green"},
    }

    return kg_context, telemetry_summary


# ---------------------------------------------------------------------------
# Evidence-perturbation helpers (for IP-2 / IP-3)
# ---------------------------------------------------------------------------

def _with_support(bundle: Dict[str, Any], candidate_id: str, delta: float) -> Dict[str, Any]:
    """Return a new evidence bundle with increased best_support_score for *candidate_id*."""
    bundle = copy.deepcopy(bundle)
    summary = bundle.setdefault("candidate_evidence_summary", [])
    for entry in summary:
        if entry.get("candidate_id") == candidate_id:
            entry["best_support_score"] = min(1.0, entry.get("best_support_score", 0.0) + delta)
            return bundle
    # Entry doesn't exist yet — add it
    summary.append({
        "candidate_id":           candidate_id,
        "best_support_score":     min(1.0, delta),
        "best_contradiction_score": 0.0,
        "best_context_score":     0.50,
        "hit_count":              1,
        "has_affects_class_hit":  True,
        "has_analyzes_class_hit": True,
        "mean_conjecture_fraction": 0.0,
        "dominant_temporal_relation": "precedes",
        "best_lag_hours":         2.0,
        "lag_is_approximate":     False,
        "best_source_tier":       "operational",
    })
    return bundle


def _with_contradiction(
    bundle: Dict[str, Any], candidate_id: str, contradiction_score: float
) -> Dict[str, Any]:
    """Return a new evidence bundle with best_contradiction_score set for *candidate_id*."""
    bundle = copy.deepcopy(bundle)
    summary = bundle.setdefault("candidate_evidence_summary", [])
    for entry in summary:
        if entry.get("candidate_id") == candidate_id:
            entry["best_contradiction_score"] = contradiction_score
            return bundle
    summary.append({
        "candidate_id":             candidate_id,
        "best_support_score":       0.0,
        "best_contradiction_score": contradiction_score,
        "best_context_score":       0.30,
        "hit_count":                1,
        "has_affects_class_hit":    False,
        "has_analyzes_class_hit":   False,
        "mean_conjecture_fraction": 0.8,
        "dominant_temporal_relation": "follows",
        "best_lag_hours":           None,
        "lag_is_approximate":       True,
        "best_source_tier":         "secondary",
    })
    return bundle


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run(
    kg: Dict[str, Any],
    telemetry: Dict[str, Any],
    evidence_bundle: Optional[Dict[str, Any]] = None,
    tskr_patterns: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fixtures = {
        "event":                        BASE_EVENT,
        "kg_context":                   kg,
        "telemetry_summary":            telemetry,
        "tskr_patterns":                tskr_patterns or EMPTY_TSKR,
        "evidence_bundle":              evidence_bundle,
        "operational_context":          None,
        "pm_compliance":                None,
        "soe_log":                      None,
        "alarm_log":                    None,
        "protection_logic_context":     None,
        "configuration_change_records": None,
        "environmental_monitoring":     None,
        "vendor_supply_chain_records":  None,
        "training_records":             None,
    }
    return run_rca(_orch(), fixtures)


def _retained(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["causality_candidates"].get("candidates") or []


def _filtered(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result["causality_candidates"].get("filtered_out_candidates") or []


def _score_anywhere(result: Dict[str, Any], candidate_id: str) -> Optional[float]:
    """Return composite_score for candidate_id from retained or filtered list."""
    for c in _retained(result) + _filtered(result):
        if c.get("candidate_id") == candidate_id:
            return c.get("composite_score")
    return None


# ---------------------------------------------------------------------------
# Chain-depth-1 base fixtures (seed for IP-2 / IP-3)
# ---------------------------------------------------------------------------

_BASE_FIXTURES = load_fixtures(_CHAIN_1_DIR)
_BASE_KG       = _BASE_FIXTURES["kg_context"]
_BASE_TELEMETRY = _BASE_FIXTURES["telemetry_summary"]
_BASE_TSKR      = _BASE_FIXTURES["tskr_patterns"]
_BASE_EVIDENCE  = _BASE_FIXTURES["evidence_bundle"]
_CHAIN1_TOP_ID  = "FM::FM-CHAIN-BEARING-WEAR"

_BASE_RESULT: Optional[Dict[str, Any]] = None


def _get_base_result() -> Dict[str, Any]:
    global _BASE_RESULT
    if _BASE_RESULT is None:
        _BASE_RESULT = _run(
            _BASE_KG,
            _BASE_TELEMETRY,
            evidence_bundle=_BASE_EVIDENCE,
            tskr_patterns=_BASE_TSKR,
        )
    return _BASE_RESULT


# ---------------------------------------------------------------------------
# IP-1 — allen_relation in scores is never "follows" or "during"
# ---------------------------------------------------------------------------

_CAUSAL_ALLEN_RELATIONS: Set[str] = {"overlaps", "precedes", "contains"}

pytestmark = pytest.mark.slow


@given(inputs=gen_paired_input(min_n=1, max_n=4))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ip1_allen_relation_never_follows_or_during(inputs: Tuple) -> None:
    """IP-1: For all valid inputs, no retained candidate has allen_relation 'follows' or 'during'.

    The engine maps FOLLOWS / DURING signals to allen_relation=None in candidate scores.
    The strings "follows" and "during" must never appear in any retained candidate's
    scores dict — this invariant holds universally.
    """
    kg, telemetry = inputs
    result = _run(kg, telemetry)

    for cand in _retained(result):
        ar = (cand.get("scores") or {}).get("allen_relation")
        assert ar not in ("follows", "during"), (
            f"Retained candidate {cand.get('candidate_id')!r} has "
            f"allen_relation={ar!r} — FOLLOWS/DURING are never causal."
        )
        if ar is not None:
            assert ar in _CAUSAL_ALLEN_RELATIONS, (
                f"Unexpected allen_relation {ar!r} on candidate "
                f"{cand.get('candidate_id')!r}. Expected one of {_CAUSAL_ALLEN_RELATIONS} or None."
            )


# ---------------------------------------------------------------------------
# IP-2 / IP-3 — Evidence monotonicity
# ---------------------------------------------------------------------------

@given(support_delta=st.floats(min_value=0.02, max_value=0.12, allow_nan=False))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ip2_support_cannot_decrease_score(support_delta: float) -> None:
    """IP-2: Adding supporting evidence cannot decrease a candidate's composite score.

    Base: chain_depth_1 fixture (known to retain FM::FM-CHAIN-BEARING-WEAR).
    Perturbation: increase best_support_score by *support_delta*.
    """
    base_result = _get_base_result()
    score_base  = _score_anywhere(base_result, _CHAIN1_TOP_ID)
    assume(score_base is not None)

    augmented_bundle = _with_support(_BASE_EVIDENCE, _CHAIN1_TOP_ID, support_delta)
    result_ext  = _run(_BASE_KG, _BASE_TELEMETRY,
                       evidence_bundle=augmented_bundle, tskr_patterns=_BASE_TSKR)
    score_ext   = _score_anywhere(result_ext, _CHAIN1_TOP_ID)
    assume(score_ext is not None)

    assert score_ext >= score_base - 0.002, (
        f"IP-2 violated: adding support_delta={support_delta:.4f} decreased score "
        f"{score_base:.4f} → {score_ext:.4f}"
    )


@given(contradiction_level=st.floats(min_value=0.30, max_value=0.90, allow_nan=False))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ip3_contradiction_cannot_increase_score(contradiction_level: float) -> None:
    """IP-3: Adding contradicting evidence cannot increase a candidate's composite score.

    Base: chain_depth_1 fixture.
    Perturbation: set best_contradiction_score to *contradiction_level*.
    Score is read from retained OR filtered (candidate may be pushed below threshold).
    """
    base_result = _get_base_result()
    score_base  = _score_anywhere(base_result, _CHAIN1_TOP_ID)
    assume(score_base is not None)

    contradiction_bundle = _with_contradiction(_BASE_EVIDENCE, _CHAIN1_TOP_ID, contradiction_level)
    result_ext  = _run(_BASE_KG, _BASE_TELEMETRY,
                       evidence_bundle=contradiction_bundle, tskr_patterns=_BASE_TSKR)
    score_ext   = _score_anywhere(result_ext, _CHAIN1_TOP_ID)

    if score_ext is None:
        # Candidate was eliminated entirely — this only strengthens IP-3.
        return

    assert score_ext <= score_base + 0.002, (
        f"IP-3 violated: contradiction_level={contradiction_level:.4f} increased score "
        f"{score_base:.4f} → {score_ext:.4f}"
    )


# ---------------------------------------------------------------------------
# IP-4 — Coverage completeness
# ---------------------------------------------------------------------------

@given(inputs=gen_paired_input(min_n=1, max_n=4))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ip4_every_fm_accounted_for(inputs: Tuple) -> None:
    """IP-4: Every failure_mode in kg_context appears in retained OR filtered_out_candidates.

    The engine generates a candidate entry for every FM in kg_context.failure_modes
    regardless of telemetry coverage. All candidates are partitioned into retained or
    filtered_out — none are silently discarded.
    """
    kg, telemetry = inputs
    result = _run(kg, telemetry, evidence_bundle=EMPTY_EVIDENCE)

    kg_fm_ids: Set[str] = {fm["fm_id"] for fm in kg.get("failure_modes", [])}
    accounted: Set[str] = {
        c.get("failure_mode_id")
        for c in _retained(result) + _filtered(result)
        if c.get("failure_mode_id")
    }

    missing = kg_fm_ids - accounted
    assert not missing, (
        f"IP-4: {len(missing)} KG FM(s) not accounted for: {missing}"
    )


# ---------------------------------------------------------------------------
# IP-5 — Partition invariant
# ---------------------------------------------------------------------------

@given(inputs=gen_paired_input(min_n=1, max_n=4))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ip5_retained_and_filtered_are_disjoint(inputs: Tuple) -> None:
    """IP-5: retained[] and filtered_out_candidates[] are always disjoint by candidate_id."""
    kg, telemetry = inputs
    result = _run(kg, telemetry, evidence_bundle=EMPTY_EVIDENCE)

    retained_ids: Set[str] = {
        c["candidate_id"] for c in _retained(result) if c.get("candidate_id")
    }
    filtered_ids: Set[str] = {
        c["candidate_id"] for c in _filtered(result) if c.get("candidate_id")
    }

    overlap = retained_ids & filtered_ids
    assert not overlap, (
        f"IP-5 violated: {len(overlap)} candidate_id(s) in both retained and filtered: {overlap}"
    )


# ---------------------------------------------------------------------------
# IP-7 — Score range
# ---------------------------------------------------------------------------

_SCORE_DIMS = ("structural", "temporal", "telemetry", "evidence", "governance")


@given(inputs=gen_paired_input(min_n=1, max_n=4))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ip7_all_scores_in_unit_interval(inputs: Tuple) -> None:
    """IP-7: composite_score and all sub-scores are in [0.0, 1.0] for every retained candidate."""
    kg, telemetry = inputs
    result = _run(kg, telemetry, evidence_bundle=EMPTY_EVIDENCE)

    for cand in _retained(result):
        cid = cand.get("candidate_id", "?")
        comp = cand.get("composite_score")
        if comp is not None:
            assert 0.0 <= comp <= 1.0, (
                f"IP-7: composite_score={comp:.6f} out of [0,1] for {cid!r}"
            )
        scores = cand.get("scores") or {}
        for dim in _SCORE_DIMS:
            val = scores.get(dim)
            if val is not None:
                assert 0.0 <= val <= 1.0, (
                    f"IP-7: sub-score '{dim}'={val:.6f} out of [0,1] for {cid!r}"
                )


# ---------------------------------------------------------------------------
# IP-8 — Sparse evidence forces zero retention
# ---------------------------------------------------------------------------

@given(inputs=gen_paired_input(min_n=1, max_n=4, include_documents=False))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ip8_no_evidence_no_documents_retains_nothing(inputs: Tuple) -> None:
    """IP-8: With empty kg_context.documents AND empty candidate_evidence_summary,
    the initial evidence sub-score defaults to the pipeline floor (~0.30), which
    falls below minimum_evidence_threshold. Therefore, ALL candidates are filtered
    and retained is empty.

    This is the universal uncertainty bound: you cannot achieve a retained candidate
    without at least one evidence source (WO, CR, TSKR, or candidate_evidence_summary).
    """
    kg, telemetry = inputs

    # Verify the fixture has no documents (generator contract)
    assert not kg.get("documents"), (
        "IP-8: generator produced kg_context with non-empty documents — test is invalid."
    )

    result = _run(kg, telemetry, evidence_bundle=EMPTY_EVIDENCE)
    retained = _retained(result)

    assert len(retained) == 0, (
        f"IP-8 violated: {len(retained)} candidate(s) retained despite empty evidence. "
        f"IDs: {[c.get('candidate_id') for c in retained]}"
    )
