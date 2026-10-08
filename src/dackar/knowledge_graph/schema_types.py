# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Schema property data types and their pandas dtype compatibility.

This module is intentionally dependency-free (no pandas / neo4j) so the
type-compatibility logic can be unit tested without importing the heavy
knowledge-graph construction stack. KGconstruction imports from here, and
schemas/baseSchema.json must stay in sync with ALLOWED_SCHEMA_TYPES.
"""

import json

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


# Schema types whose contract is not fully expressible as a column dtype and
# therefore need a per-value semantic check in addition to isCompatibleDtype.
VALUE_VALIDATED_TYPES = ("json_string", "enum", "array")


def _isNull(value):
    """Return True for None or a NaN float (no pandas dependency)."""
    if value is None:
        return True
    return isinstance(value, float) and value != value


def validateValueSemantics(schemaType, values, enumValues=None):
    """
    Per-value semantic validation beyond the column dtype check.

    The dtype check (isCompatibleDtype) treats enum / json_string columns as
    plain strings and an array column as any 'mixed' column, so malformed JSON,
    out-of-enum values, and heterogeneous scalar columns all slip through. This
    inspects each non-null cell to enforce the declared contract:
      * json_string: every value must be a string that parses as JSON.
      * enum: every value must be a member of enumValues (skipped only when the
        schema declared no enum_values, which cannot be checked).
      * array: every value must be a list/tuple cell, not a scalar that merely
        made pandas infer the column as 'mixed'.
    Every other (or unknown) type is considered already satisfied here.

    @ In, schemaType, string, declared property type
    @ In, values, iterable, the column's values to inspect
    @ In, enumValues, list|None, allowed values when schemaType == 'enum'
    @ Out, tuple(bool, object), (True, None) if every non-null value satisfies
        the contract, else (False, offendingValue)
    """
    if schemaType not in VALUE_VALIDATED_TYPES:
        return True, None

    allowed = set(enumValues) if enumValues else None
    for value in values:
        if _isNull(value):
            continue
        if schemaType == "json_string":
            if not isinstance(value, str):
                return False, value
            try:
                json.loads(value)
            except (ValueError, TypeError):
                return False, value
        elif schemaType == "enum":
            if allowed is not None and value not in allowed:
                return False, value
        elif schemaType == "array":
            if not isinstance(value, (list, tuple)):
                return False, value
    return True, None
