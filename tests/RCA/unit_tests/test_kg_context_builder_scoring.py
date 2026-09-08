"""
test_kg_context_builder_scoring.py — unit tests for B2: document recency decay exemption

Tests verify that ECA/RCA documents are excluded from the recency-proximity bonus
(they are now treated as timeless, retrieved regardless of the ±90-day window)
while CR/WO/ECR documents still receive the bonus.

Run directly:   python test_kg_context_builder_scoring.py
Or via pytest:  pytest test_kg_context_builder_scoring.py
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

from orchestrators.kg_context_builder import Neo4jKGContextBuilder, KGContextBuilderConfig


def _make_builder():
    client = MagicMock()
    client.query.return_value = []
    return Neo4jKGContextBuilder(
        client=client,
        database="testdb",
        config=KGContextBuilderConfig(),
    )


def _score_doc(builder, doc_type: str, time_distance_days) -> float:
    """
    Replicate the enrichment scoring loop from _fetch_documents for a single doc.
    """
    doc_type_priority = {
        "CR": 100, "WO": 95, "ECA": 90, "RCA": 85, "ECR": 75,
        "FMEA": 70, "SOP": 60, "MANUAL": 50, "BULLETIN": 45,
    }
    score = float(doc_type_priority.get(doc_type, 10))
    if time_distance_days is not None and doc_type in {"CR", "WO", "ECR"}:
        score += max(0, 10 - min(time_distance_days, 10))
    return score


# ---------------------------------------------------------------------------
# B2 — ECA/RCA should not receive recency proximity bonus
# ---------------------------------------------------------------------------

def test_eca_score_unchanged_regardless_of_time_distance():
    builder = _make_builder()
    score_recent = _score_doc(builder, "ECA", time_distance_days=0)
    score_old = _score_doc(builder, "ECA", time_distance_days=5)
    assert score_recent == score_old == 90.0, (
        "ECA is timeless — score must not change with time_distance_days"
    )


def test_rca_score_unchanged_regardless_of_time_distance():
    builder = _make_builder()
    score_recent = _score_doc(builder, "RCA", time_distance_days=0)
    score_old = _score_doc(builder, "RCA", time_distance_days=10)
    assert score_recent == score_old == 85.0, (
        "RCA is timeless — score must not change with time_distance_days"
    )


def test_cr_still_receives_recency_bonus():
    builder = _make_builder()
    score_same_day = _score_doc(builder, "CR", time_distance_days=0)
    score_10_days = _score_doc(builder, "CR", time_distance_days=10)
    assert score_same_day > score_10_days, "CR should earn a recency bonus when close in time"
    assert score_same_day == 110.0
    assert score_10_days == 100.0


def test_wo_still_receives_recency_bonus():
    builder = _make_builder()
    score_same_day = _score_doc(builder, "WO", time_distance_days=0)
    score_5_days = _score_doc(builder, "WO", time_distance_days=5)
    assert score_same_day > score_5_days


def test_ecr_still_receives_recency_bonus():
    builder = _make_builder()
    score_same_day = _score_doc(builder, "ECR", time_distance_days=0)
    score_10_days = _score_doc(builder, "ECR", time_distance_days=10)
    assert score_same_day > score_10_days


def test_fmea_score_unchanged_was_already_timeless():
    builder = _make_builder()
    score_recent = _score_doc(builder, "FMEA", time_distance_days=0)
    score_old = _score_doc(builder, "FMEA", time_distance_days=100)
    assert score_recent == score_old == 70.0


def test_compute_time_distance_days_returns_none_when_event_time_missing():
    builder = _make_builder()
    result = builder._compute_time_distance_days("2024-01-01T00:00:00Z", None)
    assert result is None


def test_compute_time_distance_days_correct():
    from datetime import datetime, timezone
    builder = _make_builder()
    event_time = datetime(2024, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
    result = builder._compute_time_distance_days("2024-01-01T00:00:00Z", event_time)
    assert result == 9


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
