"""
test_n3_common_cause_explain_away_aug20.py — N-3 common-cause explain-away.

Once the engine's common-cause analysis suspects a shared dependency serving
several candidates, the *other* clustered candidates are co-symptoms of that
common cause, not independent roots. This closes the reasoning-surfacing half of
N-3 (reinforcing P-2):

  * engine (`_build_common_cause_summary`) emits `explained_away_candidate_ids`
    (the clustered co-symptoms minus the top shared-cause candidate) whenever a
    common cause is suspected — additive provenance, ranking unchanged;
  * synthesizer (`_apply_common_cause_explain_away_flag`) raises an analyst flag
    and uncertainty note when the selected primary is one of those co-symptoms,
    pointing at the shared-cause candidate / shared dependency.

Run:  pytest test_n3_common_cause_explain_away_aug20.py -v
"""
from __future__ import annotations

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

from orchestrators.causality_engine_v32 import (  # noqa: E402
    CausalityEngineConfigV32,
    RuleBasedCausalityEngineV32,
)
from orchestrators.llm_clients import DummyLLMClient  # noqa: E402
from synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)


def _engine() -> RuleBasedCausalityEngineV32:
    return RuleBasedCausalityEngineV32(CausalityEngineConfigV32())


def _cc_candidate(cid: str, score: float, confidence: str, deps=("DEP-1",)) -> dict:
    """A candidate carrying common-cause features (as attached by the engine)."""
    return {
        "candidate_id": cid,
        "hypothesis_type": "failure_mode",
        "cause_label": f"cause_{cid}",
        "common_cause_score": score,
        "common_cause_confidence": confidence,
        "shared_dependency_ids": list(deps),
        "converging_candidate_ids": [],
    }


# ── engine: explained_away_candidate_ids provenance ─────────────────────────

def test_summary_lists_explained_away_when_common_cause_suspected():
    e = _engine()
    retained = [
        _cc_candidate("FM::SHARED", 0.70, "high"),
        _cc_candidate("FM::SYMPTOM-A", 0.50, "medium"),
        _cc_candidate("FM::SYMPTOM-B", 0.50, "medium"),
    ]
    summary = e._build_common_cause_summary(retained, [])
    assert summary["suspected_common_cause"] is True
    assert summary["top_common_cause_candidate_id"] == "FM::SHARED"
    ea = summary["explained_away_candidate_ids"]
    # co-symptoms are surfaced; the shared-cause candidate itself is not explained away
    assert "FM::SYMPTOM-A" in ea
    assert "FM::SYMPTOM-B" in ea
    assert "FM::SHARED" not in ea


def test_summary_no_explain_away_when_not_suspected():
    e = _engine()
    # Only one clustered candidate → no suspected common cause
    retained = [
        _cc_candidate("FM::SHARED", 0.70, "high"),
        _cc_candidate("FM::LOW", 0.20, "none"),
    ]
    summary = e._build_common_cause_summary(retained, [])
    assert summary["suspected_common_cause"] is False
    assert summary["explained_away_candidate_ids"] == []


# ── synthesizer: explain-away analyst flag ──────────────────────────────────

def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


def _card(primary_id: str) -> dict:
    return {
        "primary_hypothesis": {"candidate_id": primary_id, "uncertainties": []},
        "executive_summary": {"analyst_attention_flags": []},
    }


def _cc_summary(**over) -> dict:
    base = {
        "suspected_common_cause": True,
        "top_common_cause_candidate_id": "FM::SHARED",
        "explained_away_candidate_ids": ["FM::SYMPTOM-A", "FM::SYMPTOM-B"],
        "shared_dependency_ids": ["DEP-1"],
    }
    base.update(over)
    return base


def test_flag_raised_when_primary_is_co_symptom():
    s = _synth()
    card = _card("FM::SYMPTOM-A")
    s._apply_common_cause_explain_away_flag(card, {"common_cause_summary": _cc_summary()})
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("common cause" in f.lower() for f in flags)
    assert any("FM::SHARED" in f for f in flags)
    assert any("DEP-1" in f for f in flags)
    assert card["primary_hypothesis"]["uncertainties"]


def test_no_flag_when_primary_is_the_shared_cause():
    s = _synth()
    card = _card("FM::SHARED")
    s._apply_common_cause_explain_away_flag(card, {"common_cause_summary": _cc_summary()})
    assert card["executive_summary"]["analyst_attention_flags"] == []
    assert card["primary_hypothesis"]["uncertainties"] == []


def test_no_flag_when_not_suspected():
    s = _synth()
    card = _card("FM::SYMPTOM-A")
    summary = _cc_summary(suspected_common_cause=False, explained_away_candidate_ids=[])
    s._apply_common_cause_explain_away_flag(card, {"common_cause_summary": summary})
    assert card["executive_summary"]["analyst_attention_flags"] == []


def test_no_flag_for_none_primary():
    s = _synth()
    card = _card("NONE")
    s._apply_common_cause_explain_away_flag(card, {"common_cause_summary": _cc_summary()})
    assert card["executive_summary"]["analyst_attention_flags"] == []
