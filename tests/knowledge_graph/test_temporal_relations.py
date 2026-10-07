"""Unit tests for the dependency-free Allen interval-algebra helper.

These do not require pandas or Neo4j.
"""
import tomllib
from pathlib import Path

import pytest

from dackar.knowledge_graph.temporal_relations import ALLEN_RELATIONS, allen_relation

SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "src" / "dackar" / "knowledge_graph" / "schemas" / "temporalRelationSchema.toml"
)


# A relative to B, interval-interval, covering all 13 relations.
INTERVAL_CASES = [
    ((0, 1), (2, 3), "before"),
    ((2, 3), (0, 1), "after"),
    ((0, 2), (2, 4), "meets"),
    ((2, 4), (0, 2), "met_by"),
    ((0, 2), (1, 3), "overlaps"),
    ((1, 3), (0, 2), "overlapped_by"),
    ((0, 1), (0, 3), "starts"),
    ((0, 3), (0, 1), "started_by"),
    ((1, 2), (0, 3), "during"),
    ((0, 3), (1, 2), "contains"),
    ((2, 3), (0, 3), "finishes"),
    ((0, 3), (2, 3), "finished_by"),
    ((0, 3), (0, 3), "equals"),
]


@pytest.mark.parametrize("a,b,expected", INTERVAL_CASES)
def test_interval_relations(a, b, expected):
    assert allen_relation(a[0], a[1], b[0], b[1]) == expected


# Instant (t_end is None) handling, including coincidence with interval endpoints.
INSTANT_CASES = [
    (0, None, (1, 2), "before"),
    (5, None, (0, 3), "after"),
    (2, None, (2, None), "equals"),      # instant vs instant
    (0, None, (0, 3), "starts"),          # instant at interval start
    (3, None, (0, 3), "finishes"),        # instant at interval end
    (1, None, (0, 3), "during"),          # instant inside interval
]


@pytest.mark.parametrize("a_start,a_end,b,expected", INSTANT_CASES)
def test_instant_relations(a_start, a_end, b, expected):
    assert allen_relation(a_start, a_end, b[0], b[1]) == expected


def test_iso_string_inputs():
    assert allen_relation(
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"
    ) == "before"


def test_invalid_interval_raises():
    with pytest.raises(ValueError):
        allen_relation(5, 1, 0, 3)


def test_outputs_are_valid_enum_members():
    for a, b, _ in INTERVAL_CASES:
        assert allen_relation(a[0], a[1], b[0], b[1]) in ALLEN_RELATIONS


def test_helper_matches_schema_enum():
    """ALLEN_RELATIONS must match the schema's allen_relation enum exactly."""
    with open(SCHEMA, "rb") as f:
        schema = tomllib.load(f)
    props = schema["relation"]["temporally_related"]["relation_properties"]
    enum_vals = next(p["enum_values"] for p in props if p["name"] == "allen_relation")
    assert set(ALLEN_RELATIONS) == set(enum_vals)
    assert len(ALLEN_RELATIONS) == 13
