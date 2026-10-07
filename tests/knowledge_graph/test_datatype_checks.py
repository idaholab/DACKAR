"""
Tests for schema property data-type handling:

  (#1) ingestion-time compatibility between a schema property type and the
       pandas infer_dtype result for an incoming dataframe column, and
  (#2) the single source of truth for allowed types staying in sync with the
       base meta-schema.

These exercise the dependency-free schema_types module, so they run without
pandas or a Neo4j backend. A pandas-backed section validates the mapping
against real infer_dtype outputs when pandas is installed.
"""
import json
from pathlib import Path

import pytest

from dackar.knowledge_graph.schema_types import (
    ALLOWED_SCHEMA_TYPES,
    SCHEMA_TYPE_TO_PANDAS,
    isCompatibleDtype,
    validateValueSemantics,
)

BASE_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "src" / "dackar" / "knowledge_graph" / "schemas" / "baseSchema.json"
)


# ---- compatibility logic (#1) ------------------------------------------------

@pytest.mark.parametrize(
    "schema_type, inferred, expected",
    [
        ("float", "floating", True),
        ("float", "integer", True),      # an int column is a valid float
        ("float", "string", False),
        ("integer", "integer", True),
        ("integer", "floating", False),
        ("boolean", "boolean", True),
        ("boolean", "integer", False),
        ("enum", "string", True),
        ("enum", "integer", False),
        ("json_string", "string", True),
        ("string", "string", True),
        ("string", "mixed", False),
        ("array", "mixed", True),
        ("array", "empty", True),
        ("array", "string", False),
        ("datetime", "datetime", True),
        ("datetime", "datetime64", True),
        ("datetime", "string", False),
    ],
)
def test_is_compatible_dtype(schema_type, inferred, expected):
    assert isCompatibleDtype(schema_type, inferred) is expected


def test_unknown_type_falls_back_to_exact_match():
    assert isCompatibleDtype("mystery", "mystery") is True
    assert isCompatibleDtype("mystery", "string") is False


# ---- per-value semantics (#17 json_string/enum, #18 array) -------------------

def test_json_string_rejects_malformed_json():
    ok, _ = validateValueSemantics("json_string", ['{"k": 1}', '{"k": 2}'])
    assert ok
    ok, bad = validateValueSemantics("json_string", ['{"k": 1}', "not json"])
    assert not ok and bad == "not json"


def test_enum_rejects_value_absent_from_enum_values():
    allowed = ["open", "closed"]
    ok, _ = validateValueSemantics("enum", ["open", "closed"], enumValues=allowed)
    assert ok
    ok, bad = validateValueSemantics("enum", ["open", "ajar"], enumValues=allowed)
    assert not ok and bad == "ajar"


def test_enum_without_declared_values_cannot_be_checked():
    # No enum_values declared -> membership is unknowable, so it passes.
    ok, _ = validateValueSemantics("enum", ["anything"], enumValues=None)
    assert ok


def test_array_rejects_heterogeneous_scalar_cells():
    ok, _ = validateValueSemantics("array", [[1, 2], [3], []])
    assert ok
    # A 'mixed' scalar column (1 and "text") must NOT pass as an array.
    ok, bad = validateValueSemantics("array", [1, "text"])
    assert not ok and bad == 1


def test_value_semantics_skips_nulls():
    ok, _ = validateValueSemantics("json_string", ['{"k": 1}', None, float("nan")])
    assert ok


def test_non_value_validated_types_pass_through():
    ok, _ = validateValueSemantics("string", ["anything", 123, None])
    assert ok


# ---- single source of truth (#2 / drift guard) ------------------------------

def test_mapping_covers_every_allowed_type():
    assert set(SCHEMA_TYPE_TO_PANDAS) == set(ALLOWED_SCHEMA_TYPES)


def test_allowed_types_match_base_schema_enum():
    base = json.loads(BASE_SCHEMA.read_text(encoding="utf-8"))
    enum = base["definitions"]["property"]["properties"]["type"]["enum"]
    assert set(enum) == set(ALLOWED_SCHEMA_TYPES)


# ---- real pandas infer_dtype behavior (skipped if pandas absent) -------------

def test_real_pandas_infer_dtype_round_trip():
    pd = pytest.importorskip("pandas")
    from pandas.api.types import infer_dtype

    cases = [
        ("string", pd.Series(["a", "b", "c"])),
        ("enum", pd.Series(["open", "closed"])),
        ("json_string", pd.Series(['{"k": 1}', '{"k": 2}'])),
        ("integer", pd.Series([1, 2, 3])),
        ("float", pd.Series([1.0, 2.5, 3.1])),
        ("float", pd.Series([1, 2, 3])),           # int column satisfies float
        ("boolean", pd.Series([True, False])),
        ("array", pd.Series([[1, 2], [3]])),
        ("datetime", pd.to_datetime(pd.Series(["2024-01-01", "2024-02-01"]))),
    ]
    for schema_type, series in cases:
        inferred = infer_dtype(series)
        assert isCompatibleDtype(schema_type, inferred), (
            f"{schema_type!r} should accept pandas infer_dtype={inferred!r}"
        )
