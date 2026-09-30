"""
Tests for TaskLabelMapper and the built-in DEFAULT_TAXONOMY_RULES vocabulary.
"""
from __future__ import annotations

import pytest

from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.preprocessing.default_taxonomy import DEFAULT_TAXONOMY_RULES
from outage_uncertainty.preprocessing.label_mapper import TaskLabelMapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _activity(description: str, **kwargs) -> ActivityCase:
    return ActivityCase(
        activity_id="T01",
        outage_id="O01",
        plant_id="P01",
        raw_description=description,
        **kwargs,
    )


def _map(description: str, **activity_kwargs) -> ActivityCase:
    mapper = TaskLabelMapper()
    return mapper.map(_activity(description, **activity_kwargs))


# ---------------------------------------------------------------------------
# Smoke: default vocabulary is non-empty
# ---------------------------------------------------------------------------

def test_default_taxonomy_rules_not_empty():
    assert len(DEFAULT_TAXONOMY_RULES) > 50


# ---------------------------------------------------------------------------
# Single-word component keywords
# ---------------------------------------------------------------------------

def test_single_word_valve():
    a = _map("replace valve packing")
    assert a.component_family == "valve"
    assert a.discipline == "mechanical"


def test_single_word_pump():
    a = _map("inspect pump casing for cracks")
    assert a.component_family == "pump"
    assert a.discipline == "mechanical"


def test_single_word_motor():
    a = _map("replace motor bearings")
    assert a.component_family == "motor"
    assert a.discipline == "electrical"


def test_single_word_transmitter():
    a = _map("calibration of pressure transmitter")
    assert a.component_family == "instrument"
    assert a.discipline == "I&C"


def test_single_word_breaker():
    a = _map("test circuit breaker trip function")
    assert a.component_family == "breaker"
    assert a.discipline == "electrical"


# ---------------------------------------------------------------------------
# Single-word task keywords
# ---------------------------------------------------------------------------

def test_task_replacement():
    a = _map("replace pump seal")
    assert a.task_family == "replacement"


def test_task_inspection():
    a = _map("visual inspection of containment liner")
    assert a.task_family == "inspection"


def test_task_calibration():
    a = _map("calibrate level transmitter")
    assert a.task_family == "calibration"


def test_task_testing():
    a = _map("testing of diesel start function")
    assert a.task_family == "testing"


def test_task_lubrication():
    a = _map("lubrication of fan shaft bearings")
    assert a.task_family == "lubrication"


def test_task_cleaning():
    a = _map("flush heat exchanger tubes")
    assert a.task_family == "cleaning"


def test_task_refurbishment():
    a = _map("overhaul turbine control valves")
    assert a.task_family == "refurbishment"


# ---------------------------------------------------------------------------
# Phrase entries win over conflicting single-word entries (last-wins)
# ---------------------------------------------------------------------------

def test_phrase_motor_operated_valve_overrides_motor():
    """'motor operated valve' should give component_family=valve, not motor."""
    a = _map("replace motor operated valve actuator")
    assert a.component_family == "valve"
    assert a.discipline == "mechanical"


def test_phrase_diesel_generator_discipline():
    """Diesel generator is mechanical discipline, not electrical."""
    a = _map("start diesel generator for surveillance test")
    assert a.component_family == "generator"
    assert a.discipline == "mechanical"


def test_phrase_heat_exchanger():
    a = _map("inspect heat exchanger tube bundle")
    assert a.component_family == "heat_exchanger"
    assert a.discipline == "mechanical"


def test_phrase_loop_calibration_sets_task_and_component():
    a = _map("perform loop calibration on flow loop")
    assert a.task_family == "calibration"
    assert a.component_family == "instrument"
    assert a.discipline == "I&C"


def test_phrase_functional_test_overrides_bare_test():
    """'functional test' → task_family=testing (same value but confirmed via phrase path)."""
    a = _map("functional test of feedwater control valve")
    assert a.task_family == "testing"


# ---------------------------------------------------------------------------
# Word-boundary matching
# ---------------------------------------------------------------------------

def test_no_match_for_embedded_word():
    """'protesting' should not trigger the 'test' keyword."""
    a = _map("protesting the schedule change")
    # task_family should not be set to "testing"
    assert a.task_family is None


def test_no_match_latest_for_lube():
    """'latest' should not trigger the 'lube' keyword."""
    a = _map("review latest inspection records")
    assert a.task_family != "lubrication"


def test_no_match_replacement_inside_word():
    """'irreplaceable' must not trigger 'replace'; 'repair' (whole word) does fire."""
    a = _map("component is irreplaceable and must be sent out for repair")
    # 'irreplaceable' must not fire 'replace'; 'repair' (whole word) sets maintenance
    assert a.task_family == "maintenance"


# ---------------------------------------------------------------------------
# Already-labelled activities are NOT overwritten
# ---------------------------------------------------------------------------

def test_caller_supplied_discipline_preserved():
    a = _map("replace pump seal", discipline="operations")
    assert a.discipline == "operations"   # caller-supplied, not overwritten


def test_caller_supplied_task_preserved():
    a = _map("inspect valve stem", task_family="modification")
    assert a.task_family == "modification"


def test_all_three_set_returns_unchanged():
    a = _map(
        "inspect pump discharge valve",
        discipline="nuclear",
        task_family="surveillance",
        component_family="reactor",
    )
    assert a.discipline == "nuclear"
    assert a.task_family == "surveillance"
    assert a.component_family == "reactor"


# ---------------------------------------------------------------------------
# User rules override defaults
# ---------------------------------------------------------------------------

def test_user_rules_override_defaults():
    mapper = TaskLabelMapper({"pump": {"discipline": "operations"}})
    a = mapper.map(_activity("replace pump seal"))
    assert a.discipline == "operations"   # user rule wins over default "mechanical"


def test_use_defaults_false_disables_builtin_vocabulary():
    mapper = TaskLabelMapper(use_defaults=False)
    a = mapper.map(_activity("replace valve packing"))
    assert a.component_family is None
    assert a.task_family is None


def test_use_defaults_false_with_custom_rules():
    mapper = TaskLabelMapper(
        {"valve": {"component_family": "custom_valve"}},
        use_defaults=False,
    )
    a = mapper.map(_activity("inspect valve stem"))
    assert a.component_family == "custom_valve"
    assert a.task_family is None   # no "inspect" in custom rules


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_description():
    a = _map("")
    assert a.discipline is None
    assert a.task_family is None
    assert a.component_family is None


def test_no_matching_keywords():
    a = _map("perform work order as directed by supervision")
    # Generic text — no specific taxonomy signal; all three fields stay None
    assert a.discipline is None
    assert a.task_family is None
    assert a.component_family is None


def test_case_insensitive_matching():
    a = _map("REPLACE PUMP SEAL")
    assert a.task_family == "replacement"
    assert a.component_family == "pump"


def test_cleaned_description_preferred_over_raw():
    activity = _activity("REPL PMP SEAL", cleaned_description="replace pump seal")
    mapper = TaskLabelMapper()
    result = mapper.map(activity)
    assert result.task_family == "replacement"
    assert result.component_family == "pump"
