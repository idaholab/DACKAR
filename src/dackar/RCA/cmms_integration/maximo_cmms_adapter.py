"""
maximo_cmms_adapter — MaximoCMMSAdapter skeleton.

Implements CMMSContextAdapter for IBM Maximo using the Maximo REST API
(OSLC / JSON API).  Plant teams fill in the connection details and any
site-specific field mappings.  See CMMS_INTEGRATION_GUIDE.md §4 for the
full implementation guide.

Dependencies (not installed by default — add to your environment):
    pip install requests
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


class MaximoCMMSAdapter:
    """
    Live Maximo adapter — fetches CRs (Service Requests / Work Orders)
    via the Maximo REST API.

    Parameters
    ----------
    base_url:
        Maximo base URL, e.g. ``"https://maximo.plant.corp/maximo"``.
    api_key:
        Maximo API key or session token.
    site_id:
        Maximo site identifier (e.g. ``"PLANT1"``).
    cr_object_name:
        Maximo object name for corrective actions (default ``"SR"``).
    wo_object_name:
        Maximo object name for work orders (default ``"WOTRACK"``).
    floc_field:
        Maximo field containing the Functional Location (default
        ``"SITEID"`` — override per plant convention).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        site_id: Optional[str] = None,
        cr_object_name: str = "SR",
        wo_object_name: str = "WOTRACK",
        floc_field: str = "LOCATION",
        timeout: int = 30,
    ) -> None:
        self.base_url        = base_url.rstrip("/")
        self.api_key         = api_key
        self.site_id         = site_id
        self.cr_object_name  = cr_object_name
        self.wo_object_name  = wo_object_name
        self.floc_field      = floc_field
        self.timeout         = timeout

        # Lazy import so the adapter can be imported without requests installed.
        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MaximoCMMSAdapter requires the 'requests' package. "
                "Install it with: pip install requests"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        primary_asset_id: str,
        sister_component_ids: List[str],
        lookback_from: str,
        lookback_to: str,
        event: JsonDict,
    ) -> JsonDict:
        """
        Fetch CR and WO records from Maximo for the given asset and window.

        TODO — plant teams implement:
        1. Build OSLC WHERE clause: LOCATION in {primary_floc, sister_flocs}
           AND REPORTDATE >= lookback_from AND REPORTDATE <= lookback_to
        2. GET /maximo/oslc/os/{cr_object_name}?oslc.where=...
        3. GET /maximo/oslc/os/{wo_object_name}?oslc.where=...
        4. Map Maximo field names to the cmms_context schema:
             TICKETID  → cr_id
             CLASS     → cr_type
             STATUS    → status  (see _map_status)
             PRIORITY  → priority
             DESCRIPTION → short_description
             LDTEXT    → long_text
             LOCATION  → functional_location
             REPORTDATE → created_date
             SITEID    → (use to filter if multi-site)
        5. Set is_sister_equipment=True for records whose LOCATION matches
           a sister FLOC, False for the primary asset.
        6. Return {"cr_records": [...], "wo_records": [...]}
        """
        raise NotImplementedError(
            "MaximoCMMSAdapter.fetch() is not yet implemented. "
            "See CMMS_INTEGRATION_GUIDE.md §4 for the implementation guide."
        )

    # ------------------------------------------------------------------
    # Helpers (plant teams may reuse or override)
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "apikey":       self.api_key,
            "Accept":       "application/json",
            "Content-Type": "application/json",
        }

    def _build_oslc_where(
        self,
        flocs: List[str],
        lookback_from: str,
        lookback_to: str,
    ) -> str:
        """
        Build an OSLC WHERE clause string for Maximo.

        Example output:
            LOCATION in ["PLANT/SYS/BEARING-01","PLANT/SYS/PUMP-02"]
            and REPORTDATE>="2025-10-01T00:00:00+00:00"
            and REPORTDATE<="2026-01-01T12:00:00+00:00"
        """
        floc_list = ",".join(f'"{f}"' for f in flocs)
        return (
            f'LOCATION in [{floc_list}]'
            f' and REPORTDATE>="{lookback_from}"'
            f' and REPORTDATE<="{lookback_to}"'
        )

    @staticmethod
    def _map_status(maximo_status: str) -> str:
        """Map Maximo status codes to cmms_context schema values."""
        mapping = {
            "WAPPR": "open", "WMATL": "open", "WPCOND": "open",
            "INPRG": "open", "APPR": "open",
            "COMP": "closed", "CLOSE": "closed",
            "CAN": "cancelled",
        }
        return mapping.get((maximo_status or "").upper(), "unknown")
