"""Unit tests for RcaReasoningOrchestrator._apply_pm_corrective_actions (Wave 4)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in (
    "neo4j", "py2neo", "chromadb",
    "langchain_chroma", "langchain_community",
    "langchain_community.vectorstores", "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator

_apply = RCAReasoningOrchestrator._apply_pm_corrective_actions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pm(
    fmea_linkage: bool = True,
    risk: str = "high",
    components: list | None = None,
) -> dict:
    return {
        "fmea_pm_linkage_available": fmea_linkage,
        "summary": {"maintenance_induced_risk": risk},
        "components": components
        if components is not None
        else [
            {
                "component_id": "C-001",
                "scope_gaps": ["FM-001"],
            }
        ],
    }


def _card(primary_fm: str = "FM-001", candidate_id: str = "CAND-1") -> dict:
    return {
        "primary_hypothesis": {
            "fm_id": primary_fm,
            "candidate_id": candidate_id,
        },
        "recommended_actions": [],
    }


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


def test_noop_when_pm_compliance_is_none():
    card = _card()
    _apply(card, None)
    assert card["recommended_actions"] == []


def test_noop_when_fmea_linkage_false():
    card = _card()
    _apply(card, _pm(fmea_linkage=False))
    assert card["recommended_actions"] == []


def test_noop_when_fmea_linkage_missing():
    card = _card()
    pm = _pm()
    del pm["fmea_pm_linkage_available"]
    _apply(card, pm)
    assert card["recommended_actions"] == []


def test_noop_when_components_empty():
    card = _card()
    _apply(card, _pm(components=[]))
    assert card["recommended_actions"] == []


def test_noop_when_components_have_no_gaps():
    card = _card()
    pm = _pm(components=[{"component_id": "C-001", "scope_gaps": []}])
    _apply(card, pm)
    assert card["recommended_actions"] == []


# ---------------------------------------------------------------------------
# Priority rule
# ---------------------------------------------------------------------------


def test_priority_high_when_risk_high():
    card = _card()
    _apply(card, _pm(risk="high"))
    assert card["recommended_actions"][0]["priority"] == "high"


def test_priority_medium_when_risk_medium():
    card = _card()
    _apply(card, _pm(risk="medium"))
    assert card["recommended_actions"][0]["priority"] == "medium"


def test_priority_medium_when_risk_low():
    card = _card()
    _apply(card, _pm(risk="low"))
    assert card["recommended_actions"][0]["priority"] == "medium"


def test_priority_medium_when_risk_missing():
    card = _card()
    pm = _pm()
    pm["summary"].pop("maintenance_induced_risk")
    _apply(card, pm)
    assert card["recommended_actions"][0]["priority"] == "medium"


# ---------------------------------------------------------------------------
# Action structure
# ---------------------------------------------------------------------------


def test_action_type_is_pm_corrective():
    card = _card()
    _apply(card, _pm())
    assert card["recommended_actions"][0]["action_type"] == "pm_corrective"


def test_action_has_required_fields():
    card = _card()
    _apply(card, _pm())
    action = card["recommended_actions"][0]
    for field in ("action_id", "action_type", "description", "priority",
                  "target_component_id", "target_causal_depth", "rationale"):
        assert field in action, f"missing field: {field}"


def test_target_component_id_matches_component():
    card = _card()
    _apply(card, _pm())
    assert card["recommended_actions"][0]["target_component_id"] == "C-001"


def test_target_causal_depth_is_root():
    card = _card()
    _apply(card, _pm())
    assert card["recommended_actions"][0]["target_causal_depth"] == "root"


def test_linked_candidate_id_set_when_primary_hypothesis_present():
    card = _card(candidate_id="CAND-99")
    _apply(card, _pm())
    assert card["recommended_actions"][0].get("linked_candidate_id") == "CAND-99"


def test_linked_candidate_id_absent_when_no_primary_hypothesis():
    card = {"recommended_actions": []}
    _apply(card, _pm())
    assert "linked_candidate_id" not in card["recommended_actions"][0]


def test_description_mentions_gap_fm_id():
    card = _card()
    _apply(card, _pm())
    desc = card["recommended_actions"][0]["description"]
    assert "FM-001" in desc


def test_rationale_mentions_risk_level():
    card = _card()
    _apply(card, _pm(risk="high"))
    rationale = card["recommended_actions"][0]["rationale"]
    assert "high" in rationale


# ---------------------------------------------------------------------------
# Primary FM filtering
# ---------------------------------------------------------------------------


def test_skips_component_when_primary_fm_not_in_gaps():
    card = _card(primary_fm="FM-DIFFERENT")
    pm = _pm(components=[{"component_id": "C-001", "scope_gaps": ["FM-001"]}])
    _apply(card, pm)
    assert card["recommended_actions"] == []


def test_injects_for_component_when_primary_fm_in_gaps():
    card = _card(primary_fm="FM-001")
    pm = _pm(components=[{"component_id": "C-001", "scope_gaps": ["FM-001", "FM-002"]}])
    _apply(card, pm)
    assert len(card["recommended_actions"]) == 1


def test_no_primary_fm_injects_for_all_components_with_gaps():
    card = {"recommended_actions": []}
    pm = _pm(
        components=[
            {"component_id": "C-001", "scope_gaps": ["FM-001"]},
            {"component_id": "C-002", "scope_gaps": ["FM-002"]},
        ]
    )
    _apply(card, pm)
    assert len(card["recommended_actions"]) == 2


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_no_duplicate_when_pm_corrective_already_exists_for_component():
    card = _card()
    card["recommended_actions"] = [
        {
            "action_type": "pm_corrective",
            "target_component_id": "C-001",
            "action_id": "PM-CORR-EXISTING",
        }
    ]
    _apply(card, _pm())
    assert len(card["recommended_actions"]) == 1


def test_injects_for_different_component_even_if_one_already_exists():
    card = {"recommended_actions": [
        {
            "action_type": "pm_corrective",
            "target_component_id": "C-001",
            "action_id": "PM-CORR-EXISTING",
        }
    ]}
    pm = _pm(
        components=[
            {"component_id": "C-001", "scope_gaps": ["FM-001"]},
            {"component_id": "C-002", "scope_gaps": ["FM-001"]},
        ]
    )
    _apply(card, pm)
    comp_ids = [a["target_component_id"] for a in card["recommended_actions"]]
    assert "C-001" in comp_ids
    assert "C-002" in comp_ids
    assert len(card["recommended_actions"]) == 2


def test_idempotent_when_called_twice():
    card = _card()
    pm = _pm()
    _apply(card, pm)
    _apply(card, pm)
    pm_corrective = [a for a in card["recommended_actions"] if a["action_type"] == "pm_corrective"]
    assert len(pm_corrective) == 1


# ---------------------------------------------------------------------------
# Multiple gaps in a single component
# ---------------------------------------------------------------------------


def test_multiple_scope_gaps_mentioned_in_description():
    card = _card(primary_fm="FM-A")
    pm = _pm(
        components=[{"component_id": "C-001", "scope_gaps": ["FM-A", "FM-B"]}]
    )
    _apply(card, pm)
    desc = card["recommended_actions"][0]["description"]
    assert "FM-A" in desc
    assert "FM-B" in desc


# ---------------------------------------------------------------------------
# Action ID uniqueness
# ---------------------------------------------------------------------------


def test_action_ids_unique_across_multiple_components():
    card = {"recommended_actions": []}
    pm = _pm(
        fmea_linkage=True,
        risk="high",
        components=[
            {"component_id": "C-001", "scope_gaps": ["FM-001"]},
            {"component_id": "C-002", "scope_gaps": ["FM-002"]},
        ],
    )
    _apply(card, pm)
    ids = [a["action_id"] for a in card["recommended_actions"]]
    assert len(ids) == len(set(ids)), "action_id values must be unique"


# ---------------------------------------------------------------------------
# Recommended_actions key created when absent
# ---------------------------------------------------------------------------


def test_recommended_actions_key_created_when_absent():
    card = {"primary_hypothesis": {"fm_id": "FM-001", "candidate_id": "C1"}}
    _apply(card, _pm())
    assert "recommended_actions" in card
    assert len(card["recommended_actions"]) == 1
