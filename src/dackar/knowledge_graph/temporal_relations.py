"""Allen's interval-algebra classification for the temporal-relation schema.

Dependency-free helper (standard library only) so it can be unit-tested
without pandas or a Neo4j driver. It computes the qualitative temporal
relation between two events for the ``temporally_related.allen_relation``
property defined in ``schemas/temporalRelationSchema.toml``.

Both time-instant events (``t_end`` omitted / None -> treated as a zero-length
interval at ``t_start``) and time-interval events are supported. When an
instant coincides with an interval endpoint, the point is treated as part of
the interval (``starts`` / ``finishes``) rather than as merely touching it
(``meets`` / ``met_by``).

The returned string is always one of :data:`ALLEN_RELATIONS`, matching the
enum values in the schema exactly.
"""
from datetime import date, datetime

# The 13 Allen interval relations, matching the temporalRelationSchema enum.
ALLEN_RELATIONS = (
    "before", "after", "meets", "met_by", "overlaps", "overlapped_by",
    "starts", "started_by", "during", "contains", "finishes", "finished_by",
    "equals",
)


def _coerce(value):
    """Coerce *value* to a comparable scalar.

    ISO-8601 strings are parsed to ``datetime``; ``datetime``/``date`` and
    numeric values are returned unchanged. A string that is not valid ISO-8601
    is rejected with ``ValueError`` rather than returned for lexicographic
    comparison, which would silently produce false temporal relations (e.g.
    ``"9" > "10"`` as strings).
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise ValueError(
                f"malformed timestamp string {value!r}: not ISO-8601 and cannot "
                "be compared temporally"
            ) from None
    return value


def _check_timezone_policy(values):
    """Enforce a uniform timezone policy across the datetime endpoints.

    Timezone-aware and naive ``datetime`` values cannot be compared in Python
    (it raises ``TypeError``), and silently assuming a timezone would invent
    data. The explicit policy is therefore: every ``datetime`` endpoint in a
    single call must be uniformly timezone-aware or uniformly naive.
    """
    datetimes = [v for v in values if isinstance(v, datetime)]
    aware = any(v.tzinfo is not None for v in datetimes)
    naive = any(v.tzinfo is None for v in datetimes)
    if aware and naive:
        raise ValueError(
            "cannot compare timezone-aware and naive timestamps; provide all "
            "datetime endpoints with a consistent timezone policy"
        )


def allen_relation(a_start, a_end, b_start, b_end):
    """Return the Allen relation of interval/instant A to B for edge (A)->(B).

    Args:
        a_start: Start of A (``t_start``). Required.
        a_end: End of A (``t_end``); None for a time-instant event.
        b_start: Start of B (``t_start``). Required.
        b_end: End of B (``t_end``); None for a time-instant event.

    Returns:
        One of :data:`ALLEN_RELATIONS` describing how A relates to B.

    Raises:
        ValueError: if a timestamp string is not valid ISO-8601, if timezone-
            aware and naive datetimes are mixed, if the endpoints are not
            mutually comparable, or if either event's end precedes its start.
    """
    a1 = _coerce(a_start)
    a2 = _coerce(a_start if a_end is None else a_end)
    b1 = _coerce(b_start)
    b2 = _coerce(b_start if b_end is None else b_end)

    _check_timezone_policy((a1, a2, b1, b2))

    try:
        if a2 < a1 or b2 < b1:
            raise ValueError("event t_end must not precede t_start")

        if a1 == b1 and a2 == b2:
            return "equals"
        if a2 < b1:
            return "before"
        if a1 > b2:
            return "after"

        if a1 < b1:
            if a2 == b1:
                return "meets"
            if a2 < b2:
                return "overlaps"
            if a2 == b2:
                return "finished_by"
            return "contains"  # a2 > b2
        if a1 == b1:
            # a2 != b2 (equality handled above)
            return "starts" if a2 < b2 else "started_by"
        # a1 > b1 (and a1 <= b2)
        if a2 == b2:
            return "finishes"
        if a1 == b2:
            return "met_by"
        if a2 < b2:
            return "during"
        return "overlapped_by"  # a2 > b2
    except TypeError as exc:
        # Endpoints of incomparable kinds (e.g. a number against a datetime)
        # otherwise surface as an incidental TypeError; report it clearly.
        raise ValueError(
            f"temporal endpoints are not mutually comparable: {exc}"
        ) from exc
