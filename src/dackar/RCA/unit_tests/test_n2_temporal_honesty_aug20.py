"""
test_n2_temporal_honesty_aug20.py — N-2 (part a) temporal-support honesty.

The co-occurrence temporal proxy (anomalies present but NO matched TSKR pattern) is
now explicitly labelled rather than silently masquerading as confirmed temporal
evidence. This is the honest-labelling half of N-2 (post-hoc / cum-hoc guard); the
*magnitude* of the proxy constants is intentionally unchanged here (deferred decision).

Covers:
  - engine `_temporal_score_for_fm`: temporal_basis + temporal_support_unestablished
    are set correctly for (no pattern + anomalies), (pattern), and (no pattern, no anomaly).
  - proxy constant magnitude is unchanged (no ranking/golden shift).
  - synthesizer `_apply_temporal_support_flag`: raises an analyst flag + uncertainty note
    when the primary's temporal support is unestablished, and stays silent otherwise.

Run:  pytest test_n2_temporal_honesty_aug20.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.causality_engine_v32 import (  # noqa: E402
    RuleBasedCausalityEngineV32,
    CausalityEngineConfigV32,
)
from orchestrators.llm_clients import DummyLLMClient  # noqa: E402
from synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)


def _engine() -> RuleBasedCausalityEngineV32:
    return RuleBasedCausalityEngineV32(config=CausalityEngineConfigV32())


_TELEM_WITH_ANOMALY = {
    "signals": [{"sensor_id": "S1", "anomalies": [{"pattern": "spike"}]}]
}
_TELEM_NO_ANOMALY = {"signals": [{"sensor_id": "S1", "anomalies": []}]}


# ── engine labelling ───────────────────────────────────────────────────────

def test_cooccurrence_proxy_is_labelled_unestablished():
    e = _engine()
    parts = e._temporal_score_for_fm(
        {"fm_id": "FM1"}, _TELEM_WITH_ANOMALY, event_time=None, tskr_index={}
    )
    assert parts["temporal_basis"] == "cooccurrence_proxy"
    assert parts["temporal_support_unestablished"] is True
    # magnitude intentionally unchanged (deferred decision): proxy still 0.55
    assert parts["tskr_pattern_match"] == RuleBasedCausalityEngineV32._TEMPORAL_COOCCURRENCE_TSKR_PROXY


def test_matched_pattern_is_established():
    e = _engine()
    tskr_index = {
        "FM1": [{
            "confidence": 0.9, "relation": "precedes",
            "latency_alignment_score": 0.8, "support": 0.7,
        }]
    }
    parts = e._temporal_score_for_fm(
        {"fm_id": "FM1"}, _TELEM_WITH_ANOMALY, event_time=None, tskr_index=tskr_index
    )
    assert parts["temporal_basis"] == "tskr_pattern"
    assert parts["temporal_support_unestablished"] is False


def test_no_pattern_no_anomaly_is_none_not_proxy():
    e = _engine()
    parts = e._temporal_score_for_fm(
        {"fm_id": "FM1"}, _TELEM_NO_ANOMALY, event_time=None, tskr_index={}
    )
    assert parts["temporal_basis"] == "none"
    assert parts["temporal_support_unestablished"] is False
    assert parts["tskr_pattern_match"] == 0.0


# ── synthesizer flag ───────────────────────────────────────────────────────

def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


def test_unestablished_primary_raises_flag():
    s = _synth()
    card = {
        "primary_hypothesis": {"candidate_id": "C1", "uncertainties": []},
        "executive_summary": {"analyst_attention_flags": []},
    }
    cc = {"candidates": [{"candidate_id": "C1", "scores": {"temporal_support_unestablished": True}}]}
    s._apply_temporal_support_flag(card, cc)
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("co-occurrence only" in f for f in flags)
    assert any("proxy" in u.lower() for u in card["primary_hypothesis"]["uncertainties"])


def test_established_primary_no_flag():
    s = _synth()
    card = {
        "primary_hypothesis": {"candidate_id": "C1", "uncertainties": []},
        "executive_summary": {"analyst_attention_flags": []},
    }
    cc = {"candidates": [{"candidate_id": "C1", "scores": {"temporal_support_unestablished": False}}]}
    s._apply_temporal_support_flag(card, cc)
    assert card["executive_summary"]["analyst_attention_flags"] == []
    assert card["primary_hypothesis"]["uncertainties"] == []


def test_unestablished_primary_confidence_capped_to_medium():
    s = _synth()
    card = {
        "primary_hypothesis": {"candidate_id": "C1", "uncertainties": [], "confidence_label": "high"},
        "executive_summary": {"analyst_attention_flags": [], "confidence_label": "high"},
    }
    cc = {"candidates": [{"candidate_id": "C1", "scores": {"temporal_support_unestablished": True}}]}
    s._apply_temporal_support_flag(card, cc)
    assert card["primary_hypothesis"]["confidence_label"] == "medium"
    assert card["primary_hypothesis"]["confidence_label_cap_reason"] == "temporal_support_unestablished"
    assert card["executive_summary"]["confidence_label"] == "medium"


def test_established_primary_confidence_not_capped():
    s = _synth()
    card = {
        "primary_hypothesis": {"candidate_id": "C1", "uncertainties": [], "confidence_label": "high"},
        "executive_summary": {"analyst_attention_flags": [], "confidence_label": "high"},
    }
    cc = {"candidates": [{"candidate_id": "C1", "scores": {"temporal_support_unestablished": False}}]}
    s._apply_temporal_support_flag(card, cc)
    assert card["primary_hypothesis"]["confidence_label"] == "high"
    assert "confidence_label_cap_reason" not in card["primary_hypothesis"]
