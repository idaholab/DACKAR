# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Schema property data types and their pandas dtype compatibility.

This module is intentionally dependency-free (no pandas / neo4j) so the
type-compatibility logic can be unit tested without importing the heavy
knowledge-graph construction stack. KGconstruction imports from here, and
schemas/baseSchema.json must stay in sync with ALLOWED_SCHEMA_TYPES.
"""

# Allowed schema property data types. Must match the type enum in
# schemas/baseSchema.json. 'floating' is retained as an alias of 'float'.
ALLOWED_SCHEMA_TYPES = (
    "string",
    "integer",
    "float",
    "floating",
    "boolean",
    "datetime",
    "enum",
    "array",
    "json_string",
)

# For each schema property type, the set of pandas.api.types.infer_dtype
# results that satisfy it when validating an incoming dataframe column.
#
# Rationale:
#   - enum / json_string are stored as plain strings.
#   - float accepts integer-valued columns (an int column is a valid float),
#     plus the mixed integer/float and decimal inference results.
#   - datetime covers the several labels pandas emits for temporal columns.
#   - array columns (lists per cell) infer as 'mixed' (or 'empty' when blank).
SCHEMA_TYPE_TO_PANDAS = {
    "string":      {"string"},
    "json_string": {"string"},
    "enum":        {"string", "categorical"},
    "integer":     {"integer"},
    "float":       {"floating", "integer", "mixed-integer-float", "decimal"},
    "floating":    {"floating", "integer", "mixed-integer-float", "decimal"},
    "boolean":     {"boolean"},
    "datetime":    {"datetime", "datetime64", "date"},
    "array":       {"mixed", "mixed-integer", "empty"},
}


def isCompatibleDtype(schemaType, inferredDtype):
    """
    Return True if a pandas infer_dtype result satisfies a schema property type.
    @ In, schemaType, string, property data type declared in a graph schema
    @ In, inferredDtype, string, result of pandas.api.types.infer_dtype on the column
    @ Out, bool, whether the inferred dtype is compatible with the schema type
    """
    allowed = SCHEMA_TYPE_TO_PANDAS.get(schemaType)
    if allowed is None:
        # Unknown schema type: fall back to exact-string match (legacy behavior).
        return schemaType == inferredDtype
    return inferredDtype in allowed
