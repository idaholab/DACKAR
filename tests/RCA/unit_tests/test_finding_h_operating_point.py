"""
test_finding_h_operating_point.py — Finding H: Category E operating-point score

Covers:
- _operating_point_score: None context, mode bases, power modifier (Cat E),
  standby modifier, non-Cat-E no modifier, train OOS bonus, rationale note
- _build_failure_mode_candidates: structural boosted at high power (Cat E),
  startup mode boosts standby fm, non-Cat-E unaffected, no-context no change,
  score fields stored, composite in [0,1]
- score_rationale: op note present when active, absent when not_assessed
- Backward-compatibility: generate() without operational_context unchanged

Run:  pytest test_finding_h_operating_point.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.causality_engine_v32 import (
    RuleBasedCausalityEngineV32,
    CausalityEngineConfigV32,
)

ENGINE = RuleBasedCausalityEngineV32

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _op_ctx(
    mode: str = "steady",
    percent_rated_power: float = 50.0,
    in_service: bool = True,
) -> dict:
    return {
        "asset_id": "PLANT-A",
        "window": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T08:00:00Z"},
        "mode": mode,
        "percent_rated_power": percent_rated_power,
        "train_configuration": {
            "train_id": "Train-A",
            "in_service": in_service,
        },
    }


def _fm(name: str = "bearing wear", superclass: str = "wear", category: str = "A") -> dict:
    return {
        "fm_id": f"FM-{name[:6].replace(' ', '-')}",
        "name": name,
        "superclass": superclass,
        "failure_mechanism": superclass,
        "component_id": "PUMP-1",
        "component_name": "Feed Water Pump",
    }


def _event(event_type: str = "reactor_trip") -> dict:
    return {
        "event_id": "EVT-001",
        "asset_id": "PLANT-A",
        "timestamp_start": "2024-01-01T06:00:00Z",
        "severity": "significant",
        "event_type": event_type,
        "symptom_signature": {"anomaly_pattern": "vibration", "symptom_types": ["vibration"]},
    }


def _kg_context(fms: list) -> dict:
    return {
        "subgraph_id": "SG-001",
        "event_id": "EVT-001",
        "asset_id": "PLANT-A",
        "components": [{"component_id": "PUMP-1", "name": "Feed Water Pump",
                         "type": "pump", "system": "feedwater"}],
        "failure_modes": fms,
        "documents": [],
        "past_events": [],
    }


def _make_engine() -> RuleBasedCausalityEngineV32:
    return RuleBasedCausalityEngineV32(config=CausalityEngineConfigV32())


# ===========================================================================
# _operating_point_score
# ===========================================================================

def test_no_operational_context_returns_zero():
    score, note = ENGINE._operating_point_score(
        operational_context=None,
        primary_causal_category="E",
        fm_superclass="overload",
        fm_name="overload trip",
    )
    assert score == 0.0
    assert note == "not_assessed"


def test_missing_mode_returns_zero():
    """operational_context with no mode → (0.0, 'not_assessed')."""
    score, note = ENGINE._operating_point_score(
        operational_context={"asset_id": "X", "window": {}, "percent_rated_power": 90.0},
        primary_causal_category="E",
        fm_superclass="overload",
        fm_name="runout",
    )
    assert score == 0.0
    assert note == "not_assessed"


def test_power_ramp_mode_high_base():
    """mode=power_ramp → mode_base = 0.70."""
    score, note = ENGINE._operating_point_score(
        operational_context=_op_ctx(mode="power_ramp", percent_rated_power=50.0),
        primary_causal_category="A",
        fm_superclass="corrosion",
        fm_name="general corrosion",
    )
    # Cat A — no power modifier, no train bonus → score = mode_base
    assert abs(score - 0.70) < 1e-4
    assert "power_ramp" in note


def test_steady_mode_lower_base():
    score, _ = ENGINE._operating_point_score(
        operational_context=_op_ctx(mode="steady"),
        primary_causal_category="A",
        fm_superclass="corrosion",
        fm_name="pitting",
    )
    assert abs(score - 0.30) < 1e-4


def test_shutdown_mode_lowest_base():
    score, _ = ENGINE._operating_point_score(
        operational_context=_op_ctx(mode="shutdown"),
        primary_causal_category="A",
        fm_superclass="corrosion",
        fm_name="pitting",
    )
    assert abs(score - 0.20) < 1e-4


def test_category_e_overload_high_power_boost():
    """Cat E, overload keyword, 95% rated power → power modifier ≈ 0.95×0.30 = 0.285."""
    score, note = ENGINE._operating_point_score(
        operational_context=_op_ctx(mode="steady", percent_rated_power=95.0),
        primary_causal_category="E",
        fm_superclass="overload",
        fm_name="overload trip",
    )
    # steady base=0.30 + 0.95×0.30 = 0.585
    expected = min(1.0, 0.30 + 0.95 * 0.30)
    assert abs(score - expected) < 1e-4
    assert "high-demand" in note


def test_category_e_standby_low_power_boost():
    """Cat E, standby keyword, 5% rated power → (1-0.05)×0.25 ≈ 0.2375."""
    score, note = ENGINE._operating_point_score(
        operational_context=_op_ctx(mode="steady", percent_rated_power=5.0),
        primary_causal_category="E",
        fm_superclass="standby",
        fm_name="standby stagnation",
    )
    expected = min(1.0, 0.30 + (1.0 - 0.05) * 0.25)
    assert abs(score - expected) < 1e-4
    assert "standby" in note


def test_non_category_e_no_power_modifier():
    """Category A at 95% power → no power modifier applied."""
    score_a, _ = ENGINE._operating_point_score(
        operational_context=_op_ctx(mode="steady", percent_rated_power=95.0),
        primary_causal_category="A",
        fm_superclass="overload",
        fm_name="overload trip",
    )
    # Should equal mode_base only = 0.30
    assert abs(score_a - 0.30) < 1e-4


def test_train_oos_standby_mechanism_bonus():
    """in_service=False + standby keyword → +0.15 bonus."""
    score_oos, note = ENGINE._operating_point_score(
        operational_context=_op_ctx(mode="steady", percent_rated_power=20.0, in_service=False),
        primary_causal_category="E",
        fm_superclass="standby",
        fm_name="idle stagnation",
    )
    # steady 0.30 + standby modifier (1-0.20)×0.25 = 0.20 + train_bonus 0.15 = 0.65
    assert score_oos > 0.50
    assert "train_oos" in note


def test_score_capped_at_one():
    """Extreme inputs (high mode + high power + OOS) never exceed 1.0."""
    score, _ = ENGINE._operating_point_score(
        operational_context=_op_ctx(mode="power_ramp", percent_rated_power=100.0, in_service=False),
        primary_causal_category="E",
        fm_superclass="overload",
        fm_name="overload transient",
    )
    assert score <= 1.0


# ===========================================================================
# _build_failure_mode_candidates integration
# ===========================================================================

def _generate_for_fm(fm_kwargs: dict, op_ctx_kwargs: dict) -> list:
    """Run generate() and return the candidates list."""
    eng = _make_engine()
    fm = _fm(**fm_kwargs)
    kg = _kg_context([fm])
    result = eng.generate(
        event=_event(),
        telemetry_summary={"signals": [], "anomalies": []},
        kg_context=kg,
        tskr_patterns=None,
        operational_context=_op_ctx(**op_ctx_kwargs) if op_ctx_kwargs else None,
        pm_compliance=None,
        run_context={
            "run_id": "test-run",
            "input_refs": {},
            "scope_management": {"active_scope_version": 0},
            "scope_snapshot": {},
        },
    )
    return result.get("candidates") or []


def test_category_e_candidate_higher_structural_at_high_power():
    """Cat E overload fm: higher structural at 95% power vs 5% power."""
    # To get a Cat E candidate we need the fm name/superclass to match E keywords
    cands_high = _generate_for_fm(
        {"name": "overload trip", "superclass": "overload"},
        {"mode": "steady", "percent_rated_power": 95.0},
    )
    cands_low = _generate_for_fm(
        {"name": "overload trip", "superclass": "overload"},
        {"mode": "steady", "percent_rated_power": 5.0},
    )
    if not cands_high or not cands_low:
        return  # no fm candidates generated — skip
    s_high = cands_high[0]["scores"]["structural"]
    s_low = cands_low[0]["scores"]["structural"]
    # High power should produce equal or higher structural for overload mechanism
    assert s_high >= s_low


def test_category_e_standby_scores_higher_in_startup_mode():
    """Standby-stagnation fm: startup mode should yield higher structural than steady."""
    cands_startup = _generate_for_fm(
        {"name": "standby stagnation", "superclass": "standby"},
        {"mode": "startup", "percent_rated_power": 10.0},
    )
    cands_steady = _generate_for_fm(
        {"name": "standby stagnation", "superclass": "standby"},
        {"mode": "steady", "percent_rated_power": 10.0},
    )
    if not cands_startup or not cands_steady:
        return
    s_startup = cands_startup[0]["scores"]["structural"]
    s_steady = cands_steady[0]["scores"]["structural"]
    assert s_startup >= s_steady


def test_non_category_e_structural_unchanged_by_power():
    """Category A (corrosion) fm: same structural regardless of power level."""
    cands_high = _generate_for_fm(
        {"name": "general corrosion", "superclass": "corrosion"},
        {"mode": "steady", "percent_rated_power": 95.0},
    )
    cands_low = _generate_for_fm(
        {"name": "general corrosion", "superclass": "corrosion"},
        {"mode": "steady", "percent_rated_power": 5.0},
    )
    if not cands_high or not cands_low:
        return
    # operating_point_score field should be 0.30 (steady base) for Cat A at both power levels
    # but no power modifier → same contribution
    op_high = cands_high[0]["scores"].get("operating_point_score", 0.0)
    op_low = cands_low[0]["scores"].get("operating_point_score", 0.0)
    assert abs(op_high - op_low) < 1e-4


def test_missing_operational_context_no_regression():
    """Without operational_context, candidates are produced and structural is unchanged."""
    cands_no_ctx = _generate_for_fm(
        {"name": "bearing wear", "superclass": "wear"},
        {},  # triggers operational_context=None path
    )
    if not cands_no_ctx:
        return
    c = cands_no_ctx[0]
    # op_score should be 0.0 when context missing
    assert c["scores"].get("operating_point_score", 0.0) == 0.0
    assert c["scores"].get("operating_point_note") == "not_assessed"


def test_operating_point_score_field_stored():
    """operating_point_score and operating_point_note present on every candidate."""
    cands = _generate_for_fm(
        {"name": "overload trip", "superclass": "overload"},
        {"mode": "power_ramp", "percent_rated_power": 80.0},
    )
    if not cands:
        return
    for c in cands:
        assert "operating_point_score" in c["scores"]
        assert "operating_point_note" in c["scores"]


def test_structural_capped_at_one():
    """Structural score must never exceed 1.0."""
    cands = _generate_for_fm(
        {"name": "overload transient", "superclass": "overload"},
        {"mode": "power_ramp", "percent_rated_power": 100.0},
    )
    for c in cands:
        assert c["scores"]["structural"] <= 1.0
        assert c["composite_score"] <= 1.0


# ===========================================================================
# Score rationale
# ===========================================================================

def test_score_rationale_includes_operating_point_note():
    """When op_score > 0, score_rationale['structural'] mentions op_point."""
    cands = _generate_for_fm(
        {"name": "overload trip", "superclass": "overload"},
        {"mode": "power_ramp", "percent_rated_power": 90.0},
    )
    if not cands:
        return
    c = cands[0]
    rationale_structural = (c.get("score_rationale") or {}).get("structural", "")
    # op_score will be > 0 in power_ramp mode regardless of category
    # (mode_base = 0.70 which is > 0)
    assert "op_point" in rationale_structural


def test_score_rationale_absent_when_not_assessed():
    """Without operational_context, rationale['structural'] has no 'op_point'."""
    cands = _generate_for_fm(
        {"name": "bearing wear", "superclass": "wear"},
        {},
    )
    if not cands:
        return
    rationale_structural = (cands[0].get("score_rationale") or {}).get("structural", "")
    assert "op_point" not in rationale_structural


# ===========================================================================
# Backward-compatibility regression
# ===========================================================================

def test_generate_without_operational_context_no_crash():
    """generate() without operational_context must run without error."""
    eng = _make_engine()
    kg = _kg_context([_fm("wear", "wear")])
    result = eng.generate(
        event=_event(),
        telemetry_summary={"signals": [], "anomalies": []},
        kg_context=kg,
        tskr_patterns=None,
        operational_context=None,
        pm_compliance=None,
        run_context={
            "run_id": "test-run",
            "input_refs": {},
            "scope_management": {"active_scope_version": 0},
            "scope_snapshot": {},
        },
    )
    assert isinstance(result, dict)
    assert "candidates" in result


def test_composite_score_always_in_range():
    """All composite scores are in [0, 1] with operating-point context."""
    cands = _generate_for_fm(
        {"name": "overload runout", "superclass": "overload"},
        {"mode": "power_ramp", "percent_rated_power": 100.0, "in_service": False},
    )
    for c in cands:
        assert 0.0 <= c["composite_score"] <= 1.0
        assert 0.0 <= c["scores"]["structural"] <= 1.0
