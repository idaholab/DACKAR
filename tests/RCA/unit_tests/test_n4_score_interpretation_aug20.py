"""
test_n4_score_interpretation_aug20.py — N-4 composite-score honesty.

`composite_score` is a weighted blend of heuristic sub-scores with hand-set
weights and relation priors; it is NOT calibrated against outcome frequencies,
yet it reads like a probability (0.72 looks like "72% likely"). N-4 adds an
additive, ranking-neutral `score_interpretation` card block that states the
score is a non-probabilistic ordinal ranking number and that any confidence
interval encodes data availability, not statistical uncertainty. This block is
injected on both the LLM and deterministic-fallback paths.

Run:  pytest test_n4_score_interpretation_aug20.py -v
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

from orchestrators.llm_clients import DummyLLMClient  # noqa: E402
from synthesis.rca_synthesizer_v31 import (  # noqa: E402
    RuleValidatedRCASynthesizerV31,
    RCASynthesizerConfig,
)


def _synth() -> RuleValidatedRCASynthesizerV31:
    return RuleValidatedRCASynthesizerV31(llm_client=DummyLLMClient(), config=RCASynthesizerConfig())


def test_score_interpretation_declares_non_probabilistic():
    block = _synth()._build_score_interpretation()
    assert block["score_type"] == "ordinal_ranking"
    assert block["is_probability"] is False
    assert block["is_calibrated"] is False


def test_score_interpretation_note_warns_against_percentage_reading():
    block = _synth()._build_score_interpretation()
    note = block["note"].lower()
    assert "not" in note and ("probability" in note or "likelihood" in note)
    assert "confidence_label" in block["note"]  # points analyst to the ordinal label


def test_interval_meaning_is_data_availability_not_statistical():
    block = _synth()._build_score_interpretation()
    meaning = block["interval_meaning"].lower()
    assert "availability" in meaning or "degrad" in meaning
    assert "not" in meaning and ("statistical" in meaning or "sampling" in meaning)


def test_schema_shape_is_complete_and_bounded():
    block = _synth()._build_score_interpretation()
    for key in ("score_type", "is_probability", "is_calibrated", "note"):
        assert key in block
    assert set(block).issubset(
        {"score_type", "is_probability", "is_calibrated", "interval_meaning", "note"}
    )


def test_block_is_constant_and_ranking_neutral():
    # Two independent builds must be identical (no run-dependent / score-dependent content).
    assert _synth()._build_score_interpretation() == _synth()._build_score_interpretation()
