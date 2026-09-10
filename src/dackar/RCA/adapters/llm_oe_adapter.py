"""
llm_oe_adapter.py — LLM-backed adapter for fleet and industry OE similar-event queries.

Calls a fine-tuned LLM API that has been trained on INPO SOER, EPRI reports,
and NRC LERs (fleet endpoint) or the broader industry database (industry endpoint).

Usage
-----
    adapter = LLMOEAdapter(
        fleet_url="https://oe-api.example.com/fleet",
        industry_url="https://oe-api.example.com/industry",
        api_key=os.environ["OE_API_KEY"],
        timeout_seconds=10.0,
    )
    orchestrator = RCAReasoningOrchestrator(...)
    orchestrator.set_similar_event_adapter(adapter)
    result = orchestrator.run(event=..., ...)

Error contract
--------------
- Any requests.RequestException or json.JSONDecodeError returns ``[]``.
- The ``degraded`` flag is set to True on first failure; cleared on init.
- The ``last_error`` attribute carries the stringified last exception.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

JsonDict = Dict[str, object]


class LLMOEAdapter:
    """Concrete SimilarEventAdapter backed by a fine-tuned LLM REST API.

    The API is expected to accept a POST with a JSON body containing a
    structured ``prompt`` field and return a JSON array of event records.
    """

    def __init__(
        self,
        *,
        fleet_url: str = "",
        industry_url: str = "",
        api_key: str = "",
        timeout_seconds: float = 10.0,
        max_results: int = 5,
        model_name: str = "oe-finetuned-v1",
    ) -> None:
        self.fleet_url = fleet_url
        self.industry_url = industry_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.model_name = model_name
        self.degraded: bool = False
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API (satisfies SimilarEventAdapter Protocol)
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        level: Literal["fleet", "industry"],
        asset_id: Optional[str],
        component_ids: List[str],
        failure_mode_ids: List[str],
        event_type: Optional[str] = None,
        actuation_type: Optional[str] = None,
        max_results: int = 5,
        timeout_seconds: float = 10.0,
    ) -> List[JsonDict]:
        """POST a structured query to the fleet or industry endpoint.

        Returns a list of event dicts on success, ``[]`` on any failure.
        """
        url = self.fleet_url if level == "fleet" else self.industry_url
        if not url:
            logger.debug("LLMOEAdapter: no URL configured for level=%s; skipping.", level)
            return []

        prompt = self._build_query_prompt(
            level=level,
            asset_id=asset_id,
            component_ids=component_ids,
            failure_mode_ids=failure_mode_ids,
            event_type=event_type,
            actuation_type=actuation_type,
            max_results=max_results,
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_results": max_results,
            "response_format": "json_array",
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            import requests  # type: ignore

            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds or self.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self.degraded = True
            self.last_error = str(exc)
            logger.warning(
                "LLMOEAdapter: query failed for level=%s: %s", level, exc
            )
            return []

        return self._parse_response(data, level=level)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_query_prompt(
        self,
        *,
        level: str,
        asset_id: Optional[str],
        component_ids: List[str],
        failure_mode_ids: List[str],
        event_type: Optional[str],
        actuation_type: Optional[str],
        max_results: int,
    ) -> str:
        """Build a structured retrieval prompt for the fine-tuned LLM."""
        db_names = {
            "fleet":    "utility fleet operating experience records",
            "industry": "INPO SOER, EPRI technical reports, and NRC LERs",
        }
        db_label = db_names.get(level, "operating experience database")
        cid_str = ", ".join(component_ids) if component_ids else "unspecified"
        fm_str = ", ".join(failure_mode_ids) if failure_mode_ids else "unspecified"

        return (
            f"Search {db_label} for similar events.\n"
            f"Asset ID: {asset_id or 'unspecified'}\n"
            f"Component IDs: {cid_str}\n"
            f"Failure mode IDs: {fm_str}\n"
            f"Event type: {event_type or 'unspecified'}\n"
            f"Actuation type: {actuation_type or 'unspecified'}\n"
            f"Return up to {max_results} results as a JSON array. "
            f"Each item must include: event_id, date (YYYY-MM-DD), summary, "
            f"root_cause_label, resolution, lessons_learned_ref, "
            f"contributing_categories (array of strings A-L), confidence_weight (0.0-1.0)."
        )

    @staticmethod
    def _parse_response(data: object, *, level: str) -> List[JsonDict]:
        """Normalise the API response into a list of event dicts."""
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Some endpoints wrap results in a key
            records = (
                data.get("events")
                or data.get("results")
                or data.get("data")
                or []
            )
        else:
            return []

        out: List[JsonDict] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            # Ensure required fields; skip malformed records
            event_id = item.get("event_id")
            if not event_id:
                continue
            record: JsonDict = {
                "event_id": str(event_id),
                "source_level": level,
                "confidence_weight": float(item.get("confidence_weight") or 0.50),
                "component_id": item.get("component_id") or item.get("component_ids", [None])[0] if item.get("component_ids") else None,
                "failure_signature": item.get("failure_signature") or item.get("summary"),
                "source_db": item.get("source_db") or ("fleet_oe" if level == "fleet" else "inpo_epri_nrc"),
                "date": item.get("date"),
                "summary": item.get("summary"),
                "actuation_type": item.get("actuation_type"),
                "root_cause_label": item.get("root_cause_label"),
                "resolution": item.get("resolution"),
                "lessons_learned_ref": item.get("lessons_learned_ref"),
                "contributing_categories": list(item.get("contributing_categories") or []),
                "match_dimensions": {},
            }
            out.append(record)
        return out
