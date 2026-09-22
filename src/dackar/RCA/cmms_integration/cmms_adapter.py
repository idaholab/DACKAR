"""
cmms_adapter — CMMSContextAdapter Protocol, NoOpCMMSAdapter, MockCMMSAdapter.

Concrete live adapters (MaximoCMMSAdapter, SAPPMCMMSAdapter) live in
separate files and implement the same Protocol.
See CMMS_INTEGRATION_GUIDE.md for the implementation skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

JsonDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Status normalization (shared by the builder and every adapter)
# ---------------------------------------------------------------------------

# The canonical map from documented raw CMMS status codes to the four values
# permitted by cmms_context.json (status enum: open/closed/cancelled/unknown).
# Both live adapters (Maximo, SAP PM) and the builder's record enrichment
# delegate here so a raw code is normalized identically wherever it enters.
_STATUS_OPEN = frozenset({
    "open",
    # Maximo
    "wappr", "wmatl", "wpcond", "inprg", "appr",
    # SAP PM
    "osno", "osma", "osts", "noco",
})
_STATUS_CLOSED = frozenset({
    "closed", "close", "comp", "completed",
    # SAP PM
    "clsd", "teco",
})
_STATUS_CANCELLED = frozenset({
    "cancelled", "canceled", "can",
    # SAP PM
    "dlfl",
})


def normalize_cmms_status(raw_status: Any) -> str:
    """
    Map a raw CMMS status code to the cmms_context schema enum.

    Recognizes every documented Maximo and SAP PM code plus the already-
    normalized values; any unrecognized or empty value maps to ``"unknown"``
    so the artifact never carries a status outside
    ``schemas/cmms_context.json`` (``status`` enum:
    ``open`` / ``closed`` / ``cancelled`` / ``unknown``).
    """
    code = (str(raw_status) if raw_status is not None else "").lower().strip()
    if not code:
        return "unknown"
    if code in _STATUS_OPEN:
        return "open"
    if code in _STATUS_CLOSED:
        return "closed"
    if code in _STATUS_CANCELLED:
        return "cancelled"
    return "unknown"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class CMMSContextAdapter(Protocol):
    """
    Protocol for live CMMS data adapters.

    Implementations must be read-only with respect to the CMMS — no writes.
    Results should be idempotent for the same (asset_id, lookback window)
    inputs: calling fetch() twice with the same arguments must return the
    same records (within CMMS data consistency guarantees).
    """

    def fetch(
        self,
        primary_asset_id: str,
        sister_component_ids: List[str],
        lookback_from: str,
        lookback_to: str,
        event: JsonDict,
    ) -> JsonDict:
        """
        Fetch CR and WO records from the CMMS.

        Parameters
        ----------
        primary_asset_id:
            Asset ID of the event asset.  Used as the primary query scope.
        sister_component_ids:
            KG component IDs of sister equipment (same_train / adjacent).
            These are opaque KG identifiers; a live adapter must resolve them
            to CMMS FLOCs / equipment IDs (via the ``maximo_floc`` /
            ``sap_equipment_id`` KG properties, the same lookup
            ``CAPExportSerializer`` uses) before querying.  This Protocol
            passes only the IDs, so an adapter that needs the mapping must be
            constructed with its own KG/FLOC resolver (or a site config table).
            Threading a schema-shaped query scope (component ID + FLOC +
            equipment ID) or a KG resolver through ``fetch()`` itself is a
            planned contract enhancement, deferred to the live-adapter /
            injection MR — see CMMS_INTEGRATION_GUIDE.md §4.
        lookback_from:
            ISO-8601 UTC timestamp — start of the query window (inclusive).
            Derived from the last PM date on the primary asset, or the
            event_time minus the configured fallback window.
        lookback_to:
            ISO-8601 UTC timestamp — end of the query window (inclusive).
            Typically the event timestamp.
        event:
            The raw event dict, passed for adapter-specific context
            (e.g., failure mode keywords for full-text search).

        Returns
        -------
        dict
            Must contain at minimum:
            ``{"cr_records": [...], "wo_records": [...]}``
            Each record should include at least: an ID field, ``status``,
            ``short_description``, ``created_date``, and
            ``is_sister_equipment``.
        """
        ...


# ---------------------------------------------------------------------------
# NoOpCMMSAdapter
# ---------------------------------------------------------------------------

class NoOpCMMSAdapter:
    """
    Silently returns empty CR and WO lists.

    Used in unit tests, CI, and deployments where no CMMS connection is
    available.  Makes no network calls and has no external dependencies.
    """

    def fetch(
        self,
        primary_asset_id: str,
        sister_component_ids: List[str],
        lookback_from: str,
        lookback_to: str,
        event: JsonDict,
    ) -> JsonDict:
        return {"cr_records": [], "wo_records": []}


# ---------------------------------------------------------------------------
# MockCMMSAdapter
# ---------------------------------------------------------------------------

class MockCMMSAdapter:
    """
    Returns configurable fixture CR and WO records.

    Designed for unit testing ``CMMSContextBuilder`` and the synthesizer
    prompt without a live CMMS connection.

    Parameters
    ----------
    cr_records:
        List of CR record dicts to return from ``fetch()``.
        Each dict should follow the ``cmms_context.json`` schema
        ``cr_records`` item structure (minus derived fields that
        ``CMMSContextBuilder`` computes: ``days_before_event``,
        ``component_id``).
    wo_records:
        List of WO record dicts to return from ``fetch()``.
    filter_by_asset:
        If ``True``, only records whose ``functional_location`` or
        ``equipment_id`` contains ``primary_asset_id`` (case-insensitive
        substring) are returned for the primary scope; all others are
        treated as sister records.  Defaults to ``False`` (all records
        returned regardless of asset).
    """

    def __init__(
        self,
        cr_records: Optional[List[JsonDict]] = None,
        wo_records: Optional[List[JsonDict]] = None,
        filter_by_asset: bool = False,
    ) -> None:
        self._cr_records = cr_records or []
        self._wo_records = wo_records or []
        self._filter_by_asset = filter_by_asset

    def fetch(
        self,
        primary_asset_id: str,
        sister_component_ids: List[str],
        lookback_from: str,
        lookback_to: str,
        event: JsonDict,
    ) -> JsonDict:
        if not self._filter_by_asset:
            return {
                "cr_records": list(self._cr_records),
                "wo_records": list(self._wo_records),
            }
        # filter_by_asset=True: classify each fixture by whether its FLOC /
        # equipment_id contains primary_asset_id (case-insensitive substring);
        # matches are the primary scope, all others are tagged sister.
        return {
            "cr_records": [self._scope(r, primary_asset_id) for r in self._cr_records],
            "wo_records": [self._scope(r, primary_asset_id) for r in self._wo_records],
        }

    @staticmethod
    def _scope(record: JsonDict, primary_asset_id: str) -> JsonDict:
        """Return a copy of ``record`` with ``is_sister_equipment`` set by a
        case-insensitive substring match of ``primary_asset_id`` against the
        record's ``functional_location`` / ``equipment_id``."""
        needle = (primary_asset_id or "").lower()
        haystack = " ".join(
            str(record.get(k) or "") for k in ("functional_location", "equipment_id")
        ).lower()
        tagged = dict(record)
        tagged["is_sister_equipment"] = bool(needle) and needle not in haystack
        return tagged
