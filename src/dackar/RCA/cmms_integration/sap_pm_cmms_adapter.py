"""
sap_pm_cmms_adapter — SAPPMCMMSAdapter skeleton.

Implements CMMSContextAdapter for SAP Plant Maintenance using the
SAP OData API (PM Maintenance Notifications / Orders).
Plant teams fill in the connection details and field mappings.
See CMMS_INTEGRATION_GUIDE.md §4 for the full implementation guide.

Dependencies (not installed by default):
    pip install requests
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# SAP PM OData endpoints (override per installation)
_DEFAULT_NOTIFICATION_ENDPOINT = (
    "/sap/opu/odata/sap/PMMAINTNOTIF_ODATA/MaintenanceNotificationSet"
)
_DEFAULT_ORDER_ENDPOINT = (
    "/sap/opu/odata/sap/API_MAINTORDER/MaintenanceOrderSet"
)


class SAPPMCMMSAdapter:
    """
    Live SAP PM adapter — fetches Maintenance Notifications (CRs) and
    Maintenance Orders (WOs) via the SAP OData API.

    Parameters
    ----------
    base_url:
        SAP system base URL, e.g. ``"https://sap.plant.corp"``.
    username:
        SAP user name (basic auth).
    password:
        SAP password (basic auth).  Consider using a service account with
        read-only PM authorisations.
    plant:
        SAP plant code (e.g. ``"1000"``).
    notification_endpoint:
        OData path for maintenance notifications.
    order_endpoint:
        OData path for maintenance orders.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        plant: Optional[str] = None,
        notification_endpoint: str = _DEFAULT_NOTIFICATION_ENDPOINT,
        order_endpoint: str = _DEFAULT_ORDER_ENDPOINT,
        timeout: int = 30,
    ) -> None:
        self.base_url               = base_url.rstrip("/")
        self.username               = username
        self.password               = password
        self.plant                  = plant
        self.notification_endpoint  = notification_endpoint
        self.order_endpoint         = order_endpoint
        self.timeout                = timeout

        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SAPPMCMMSAdapter requires the 'requests' package. "
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
        Fetch Maintenance Notifications and Orders from SAP PM.

        TODO — plant teams implement:
        1. Map sister_component_ids to SAP equipment IDs using the KG
           sap_equipment_id property (same property used for CAP export).
        2. Build OData $filter:
             TechObjNr in ('EQ-001','EQ-002')
             and MaintNotifCreationDate ge datetime'{lookback_from}'
             and MaintNotifCreationDate le datetime'{lookback_to}'
        3. GET {notification_endpoint}?$filter=...&$format=json
        4. GET {order_endpoint}?$filter=...&$format=json
        5. Map SAP field names to the cmms_context schema:
             NotifNo         → cr_id
             MaintNotifType  → cr_type  (e.g. M1, M2, M3, M4)
             UserStatus      → status   (see _map_status)
             Priority        → priority
             ShortText       → short_description
             LongText        → long_text
             FunctLoc        → functional_location
             Equipment       → equipment_id
             MaintNotifCreationDate → created_date
        6. Set is_sister_equipment=True for records on sister equipment IDs.
        7. Return {"cr_records": [...], "wo_records": [...]}
        """
        raise NotImplementedError(
            "SAPPMCMMSAdapter.fetch() is not yet implemented. "
            "See CMMS_INTEGRATION_GUIDE.md §4 for the implementation guide."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth(self):
        """Return (username, password) tuple for requests basic auth."""
        return (self.username, self.password)

    def _build_odata_filter(
        self,
        equipment_ids: List[str],
        lookback_from: str,
        lookback_to: str,
    ) -> str:
        """
        Build an OData $filter string for SAP PM notifications.

        Note: SAP OData datetime literals use the format
        ``datetime'2025-10-01T00:00:00'`` (no timezone — SAP uses plant
        local time by default).
        """
        from_ts = lookback_from[:19]  # strip timezone for SAP literal
        to_ts   = lookback_to[:19]
        eq_list = ",".join(f"'{e}'" for e in equipment_ids)
        return (
            f"Equipment in ({eq_list})"
            f" and MaintNotifCreationDate ge datetime'{from_ts}'"
            f" and MaintNotifCreationDate le datetime'{to_ts}'"
        )

    @staticmethod
    def _map_status(sap_status: str) -> str:
        """Map SAP user status codes to cmms_context schema values."""
        mapping = {
            "OSNO": "open", "OSMA": "open", "OSTS": "open",
            "NOCO": "open",   # not completed
            "CLSD": "closed", "TECO": "closed",
            "DLFL": "cancelled",
        }
        return mapping.get((sap_status or "").upper(), "unknown")
