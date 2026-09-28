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
- ``query()`` never raises: any transport, decode, or parse failure returns
  ``[]`` and records the reason on the instance.
- ``degraded`` and ``last_error`` are reset at the start of every ``query()``
  call, so they always reflect that one call.  ``degraded`` is set to True when
  the tier is skipped (no URL configured), the request fails, the response body
  cannot be decoded, or the response is unusable (wrong shape, or every record
  malformed).  The orchestrator reads ``degraded`` after each per-tier call.
- ``last_error`` carries the stringified reason for the last degraded call.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

JsonDict = Dict[str, object]


def _opt_str(value: object) -> Optional[str]:
    """Return ``str(value)`` when *value* is not None, otherwise None."""
    return None if value is None else str(value)


def _str_list(value: object) -> List[str]:
    """Coerce *value* to a list of strings; return ``[]`` when it is not a list."""
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]


class LLMOEAdapter:
    """Concrete SimilarEventAdapter backed by a fine-tuned LLM REST API.

    The API is expected to accept a POST with a JSON body containing a
    structured ``prompt`` field and return a JSON array of event records (or a
    dict wrapping that array under ``events``, ``results``, or ``data``).

    Parameters
    ----------
    fleet_url : str, optional
        Endpoint for the utility-fleet OE database.  An empty string means the
        fleet tier is not configured; querying it marks the tier degraded.
    industry_url : str, optional
        Endpoint for the broad industry OE database (INPO SOER, EPRI, NRC LERs).
        An empty string means the industry tier is not configured.
    api_key : str, optional
        Bearer token sent as an ``Authorization`` header when non-empty.
    timeout_seconds : float, optional
        Default per-request timeout, used when ``query()`` is not given an
        explicit ``timeout_seconds``.  Defaults to 10.0.
    max_results : int, optional
        Default maximum number of records to request, used when ``query()`` is
        not given an explicit ``max_results``.  Defaults to 5.
    model_name : str, optional
        Fine-tuned model identifier sent in the request payload.

    Attributes
    ----------
    degraded : bool
        Whether the most recent ``query()`` call failed or was skipped.  Reset
        to False at the start of each call.
    last_error : str or None
        Stringified reason for the most recent degraded call, or None.
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
        max_results: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> List[JsonDict]:
        """POST a structured query to the fleet or industry endpoint.

        Parameters
        ----------
        level : {"fleet", "industry"}
            Which tier to query; selects ``fleet_url`` or ``industry_url`` and
            is stamped onto every returned record as ``source_level``.
        asset_id : str or None
            Target asset identifier included in the retrieval prompt.
        component_ids : list of str
            Candidate component identifiers included in the prompt.
        failure_mode_ids : list of str
            Candidate failure-mode identifiers included in the prompt.
        event_type : str, optional
            Event-type hint for the prompt.
        actuation_type : str, optional
            Actuation-type hint for the prompt.
        max_results : int, optional
            Maximum records to request.  When None (the default), the instance's
            ``max_results`` is used.
        timeout_seconds : float, optional
            Per-request timeout.  When None (the default), the instance's
            ``timeout_seconds`` is used.

        Returns
        -------
        list of dict
            Normalised event records compatible with the
            ``similar_event_list.json`` event-item schema.  Returns ``[]`` on
            any failure or when the tier has no configured URL; in those cases
            ``degraded`` is set and ``last_error`` records the reason.  Each
            record carries ``event_id``, ``source_level``, ``confidence_weight``
            (raw match score in ``[0.0, 1.0]``, before tier discount),
            ``component_id`` (or None), and the descriptive fields.
        """
        # Reset per-call status so degraded/last_error reflect only this call.
        self.degraded = False
        self.last_error = None

        # Resolve constructor defaults when the caller did not override them.
        if max_results is None:
            max_results = self.max_results
        if timeout_seconds is None:
            timeout_seconds = self.timeout_seconds

        url = self.fleet_url if level == "fleet" else self.industry_url
        if not url:
            # Not configured: report the tier as degraded so the orchestrator
            # records it as skipped rather than a healthy "no matches".
            self.degraded = True
            self.last_error = f"no {level} URL configured"
            logger.debug("LLMOEAdapter: no URL configured for level=%s; degraded.", level)
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
                timeout=timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            # Parse inside the fail-safe boundary: an unusable response raises
            # ValueError, which is caught here and reported as degraded.
            return self._parse_response(data, level=level)
        except Exception as exc:
            self.degraded = True
            self.last_error = str(exc)
            logger.warning(
                "LLMOEAdapter: query failed for level=%s: %s", level, exc
            )
            return []

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
            f"Each item must include: event_id, component_id (the canonical "
            f"component identifier this event maps to), failure_signature, "
            f"date (YYYY-MM-DD), summary, root_cause_label, resolution, "
            f"lessons_learned_ref, contributing_categories (array of strings "
            f"A-L), confidence_weight (0.0-1.0)."
        )

    @staticmethod
    def _parse_response(data: object, *, level: str) -> List[JsonDict]:
        """Normalise the endpoint response into a list of event-record dicts.

        Parameters
        ----------
        data : object
            The decoded JSON returned by the endpoint.  Accepted shapes are a
            bare list of record dicts, or a dict wrapping such a list under one
            of the keys ``events``, ``results``, or ``data``.
        level : str
            The tier being queried; stamped onto every record as
            ``source_level``.

        Returns
        -------
        list of dict
            One normalised record per usable input record.  Individual records
            that are malformed (not a dict, missing ``event_id``, or carrying a
            non-numeric / non-finite ``confidence_weight``) are skipped.

        Raises
        ------
        ValueError
            If *data* is not a usable shape, or if it carried records but none
            survived validation.  ``query()`` catches this, marks the tier
            degraded, and returns ``[]``.
        """
        # --- shape validation ------------------------------------------------
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = None
            for key in ("events", "results", "data"):
                if key in data:
                    records = data[key]
                    break
            if records is None:
                raise ValueError("response dict has no events/results/data key")
            if not isinstance(records, list):
                raise ValueError("response wrapper value is not a list")
        else:
            raise ValueError(f"unsupported response type: {type(data).__name__}")

        # --- per-record normalisation (skip bad, keep valid) -----------------
        out: List[JsonDict] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            event_id = item.get("event_id")
            if not event_id:
                continue

            # confidence_weight: default 0.50 only when absent or None; a present
            # 0.0 is preserved.  Skip records whose value is non-numeric or
            # non-finite; clamp a valid value into [0.0, 1.0].
            raw_conf = item.get("confidence_weight")
            if raw_conf is None:
                conf = 0.50
            else:
                try:
                    conf = float(raw_conf)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(conf):
                    continue
                conf = max(0.0, min(1.0, conf))

            # component_id: explicit field first; only when absent fall back to
            # the first entry of a non-empty component_ids list.
            component_id = item.get("component_id")
            if component_id is None:
                cids = item.get("component_ids")
                if isinstance(cids, list) and cids:
                    component_id = cids[0]
            component_id = str(component_id) if component_id is not None else None

            record: JsonDict = {
                "event_id": str(event_id),
                "source_level": level,
                "confidence_weight": conf,
                "component_id": component_id,
                "failure_signature": _opt_str(
                    item.get("failure_signature") or item.get("summary")
                ),
                "source_db": _opt_str(item.get("source_db"))
                or ("fleet_oe" if level == "fleet" else "inpo_epri_nrc"),
                "date": _opt_str(item.get("date")),
                "summary": _opt_str(item.get("summary")),
                "actuation_type": _opt_str(item.get("actuation_type")),
                "root_cause_label": _opt_str(item.get("root_cause_label")),
                "resolution": _opt_str(item.get("resolution")),
                "lessons_learned_ref": _opt_str(item.get("lessons_learned_ref")),
                "contributing_categories": _str_list(item.get("contributing_categories")),
                "match_dimensions": {},
            }
            out.append(record)

        # If the endpoint returned records but none survived, treat the whole
        # response as unusable so the tier is reported degraded.
        if records and not out:
            raise ValueError(f"all {len(records)} record(s) malformed")
        return out
