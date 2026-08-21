"""
test_p7_data_limited_confidence_aug20.py — P-7 data-limited confidence cap.

The engine already flags `data_limited_conclusion` (a required evidence stream below
the data floor) and reduces the quality multiplier, but that was previously only
*annotated* — the card confidence could still read `high`. §3.5/§7 require conservative
bias under sparse data, so P-7 caps the primary/executive confidence at `medium` when
the primary is data-limited (downward-only; ranking untouched) and raises an analyst
flag.

Run:  pytest test_p7_data_limited_confidence_aug20.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb",
             "langchain_community", "langchain_community.vectorstores",
             "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.llm_clients import DummyLLMClient  # noqa: E402
from synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)


def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


def _card(confidence="high"):
    return {
        "primary_hypothesis": {"candidate_id": "C1", "uncertainties": [], "confidence_label": confidence},
        "executive_summary": {"analyst_attention_flags": [], "confidence_label": confidence},
    }


def test_data_limited_primary_caps_confidence():
    s = _synth()
    card = _card("high")
    cc = {"candidates": [{
        "candidate_id": "C1",
        "data_limited_conclusion": True,
        "critical_streams_below_floor": ["documentary", "oe"],
    }]}
    s._apply_data_limited_confidence_cap(card, cc)
    assert card["primary_hypothesis"]["confidence_label"] == "medium"
    assert card["primary_hypothesis"]["confidence_label_cap_reason"] == "data_limited_conclusion"
    assert card["executive_summary"]["confidence_label"] == "medium"
    flags = card["executive_summary"]["analyst_attention_flags"]
    assert any("data-limited" in f and "documentary" in f for f in flags)


def test_data_limited_does_not_raise_low_confidence():
    s = _synth()
    card = _card("low")
    cc = {"candidates": [{"candidate_id": "C1", "data_limited_conclusion": True, "critical_streams_below_floor": []}]}
    s._apply_data_limited_confidence_cap(card, cc)
    # downward-only: 'low' stays 'low'
    assert card["primary_hypothesis"]["confidence_label"] == "low"


def test_non_data_limited_primary_unchanged():
    s = _synth()
    card = _card("high")
    cc = {"candidates": [{"candidate_id": "C1", "data_limited_conclusion": False}]}
    s._apply_data_limited_confidence_cap(card, cc)
    assert card["primary_hypothesis"]["confidence_label"] == "high"
    assert "confidence_label_cap_reason" not in card["primary_hypothesis"]
    assert card["executive_summary"]["analyst_attention_flags"] == []
