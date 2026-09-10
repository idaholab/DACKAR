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
            Implementations should map these to CMMS FLOCs / equipment IDs
            using the same KG property lookup used by ``CAPExportSerializer``
            (``maximo_floc`` / ``sap_equipment_id``).
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
        return {
            "cr_records": list(self._cr_records),
            "wo_records": list(self._wo_records),
        }
