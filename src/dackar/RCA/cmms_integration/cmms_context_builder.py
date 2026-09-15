"""
cmms_context_builder — CMMSContextBuilder.

Orchestrates live CMMS data retrieval for a single RCA run:
  1. Derives the lookback window from kg_context.past_events[] (last PM)
     or falls back to event_time − fallback_lookback_days.
  2. Identifies sister component IDs from kg_context.components[].
  3. Calls the CMMSContextAdapter.fetch() method.
  4. Enriches raw records: days_before_event, FLOC→KG component match.
  5. Builds the recurrence_summary aggregate.
  6. Returns a dict conforming to schemas/cmms_context.json.

Chroma injection (narrative text → run-scoped embeddings) is handled
separately by the orchestrator, which has access to the evidence store.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _days_between(earlier: Optional[datetime], later: Optional[datetime]) -> Optional[int]:
    if earlier is None or later is None:
        return None
    delta = later - earlier
    return int(delta.total_seconds() // 86400)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CMMSContextBuilderConfig:
    """
    Configuration for CMMSContextBuilder.

    Parameters
    ----------
    fallback_lookback_days:
        Days to look back when no PM date is found in kg_context.past_events[].
        Default: 90.
    sister_relation_types:
        KG ``relation_to_asset`` values that qualify a component as sister
        equipment.  Default: ``["same_train", "adjacent"]``.
    include_sister_equipment:
        Whether to include sister equipment in the CMMS query scope.
        Default: True.
    max_cr_records:
        Cap on how many CR records to retain in the artifact (adapter may
        return more; the most recent are kept).  0 = no cap.
    max_wo_records:
        Cap on WO records.  0 = no cap.
    similarity_resolver:
        Optional ``EquipmentSimilarityResolver`` instance.  When provided,
        Tier 2 (failure mode overlap) and Tier 3 (spec embedding) sisters are
        added to the topology-based sisters already derived from KG topology.
        Set to ``None`` (default) to use topology-only sister resolution.
    """

    fallback_lookback_days: int = 90
    sister_relation_types: List[str] = field(
        default_factory=lambda: ["same_train", "adjacent"]
    )
    include_sister_equipment: bool = True
    max_cr_records: int = 100
    max_wo_records: int = 100
    similarity_resolver: Optional[Any] = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class CMMSContextBuilder:
    """
    Builds a ``cmms_context`` artifact from live CMMS data.

    Parameters
    ----------
    adapter:
        Any object implementing the ``CMMSContextAdapter`` Protocol.
    config:
        ``CMMSContextBuilderConfig`` — defaults to 90-day fallback,
        same_train + adjacent sister scope.
    """

    def __init__(self, adapter: Any, config: Optional[CMMSContextBuilderConfig] = None) -> None:
        self.adapter = adapter
        self.config = config or CMMSContextBuilderConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        event: JsonDict,
        kg_context: JsonDict,
        run_id: str,
    ) -> JsonDict:
        """
        Build and return a ``cmms_context`` dict.

        Parameters
        ----------
        event:
            Raw event dict (used for event_time and asset_id).
        kg_context:
            KG context artifact from Stage 5A of the same run.
            Used to derive the lookback anchor and sister component IDs.
        run_id:
            RCA run identifier.

        Returns
        -------
        dict
            Conforms to ``schemas/cmms_context.json``.
        """
        event_id  = event.get("event_id") or "unknown"
        asset_id  = event.get("asset_id") or kg_context.get("asset_id")
        event_ts  = _parse_iso(
            event.get("timestamp_start")
            or event.get("event_time")
            or event.get("timestamp")
        )
        generated_at = _utcnow_iso()
        cmms_context_id = f"CMMSCTX::{event_id}::{generated_at}"

        # 1. Derive lookback window
        lookback_from, lookback_anchor = self._resolve_lookback(
            kg_context=kg_context,
            event_ts=event_ts,
        )
        lookback_to = event_ts or datetime.now(timezone.utc)
        lookback_from_iso = lookback_from.isoformat()
        lookback_to_iso   = lookback_to.isoformat()

        # 2. Identify sister components
        sister_records = self._resolve_sisters(kg_context)
        sister_ids = [r["component_id"] for r in sister_records]

        # 3. Fetch from CMMS
        primary_asset_id = asset_id or ""
        raw = self.adapter.fetch(
            primary_asset_id=primary_asset_id,
            sister_component_ids=sister_ids,
            lookback_from=lookback_from_iso,
            lookback_to=lookback_to_iso,
            event=event,
        )

        raw_crs = raw.get("cr_records") or []
        raw_wos = raw.get("wo_records") or []

        # 4. Enrich records
        cr_records = [
            self._enrich_record(r, lookback_to, is_cr=True)
            for r in raw_crs
            if isinstance(r, dict)
        ]
        wo_records = [
            self._enrich_record(r, lookback_to, is_cr=False)
            for r in raw_wos
            if isinstance(r, dict)
        ]

        # 5. Cap
        if self.config.max_cr_records:
            cr_records = sorted(
                cr_records, key=lambda r: r.get("created_date") or "", reverse=True
            )[: self.config.max_cr_records]
        if self.config.max_wo_records:
            wo_records = sorted(
                wo_records, key=lambda r: r.get("created_date") or "", reverse=True
            )[: self.config.max_wo_records]

        # 6. Recurrence summary
        recurrence_summary = self._build_recurrence_summary(cr_records, wo_records)

        return {
            "cmms_context_id": cmms_context_id,
            "run_id":          run_id,
            "event_id":        event_id,
            "asset_id":        asset_id,
            "generated_at":    generated_at,
            "adapter":         self.adapter.__class__.__name__,
            "lookback_anchor": lookback_anchor,
            "lookback_from":   lookback_from_iso,
            "lookback_to":     lookback_to_iso,
            "sister_component_ids": sister_ids,
            "sister_components":    sister_records,
            "cr_records":      cr_records,
            "wo_records":      wo_records,
            "recurrence_summary": recurrence_summary,
            "provenance": {
                "generated_by": "CMMSContextBuilder",
                "kg_context_id": kg_context.get("subgraph_id"),
                "query_params": {
                    "primary_asset_id": primary_asset_id,
                    "sister_component_ids": sister_ids,
                    "lookback_from": lookback_from_iso,
                    "lookback_to": lookback_to_iso,
                    "fallback_lookback_days": self.config.fallback_lookback_days,
                    "sister_relation_types": self.config.sister_relation_types,
                },
            },
        }

    def get_chroma_documents(self, cmms_context: JsonDict) -> List[JsonDict]:
        """
        Extract narrative documents suitable for Chroma injection.

        Returns a list of dicts, each with:
        - ``text``: the narrative to embed (``long_text`` field)
        - ``metadata``: source, run_id, record ID, is_sister_equipment

        The orchestrator passes these to ``evidence_store.add_documents()``
        (or equivalent) after calling ``build()``.
        """
        docs: List[JsonDict] = []
        run_id    = cmms_context.get("run_id", "")
        event_id  = cmms_context.get("event_id", "")
        asset_id  = cmms_context.get("asset_id", "")

        for rec in cmms_context.get("cr_records") or []:
            text = rec.get("long_text") or rec.get("short_description") or ""
            if not text.strip():
                continue
            cr_id = str(rec.get("cr_id") or "").strip()
            doc_id = f"CMMS::CR::{cr_id}" if cr_id else ""
            structured = self._extract_structured_fields(rec)
            docs.append({
                "text": text,
                "metadata": {
                    "ingestion_path":      "path_a_structured",
                    "source":              "cmms_live",
                    "source_tier":         "plant_instance",
                    "record_type":         "cr",
                    "doc_type":            "CR",
                    "run_id":              run_id,
                    "event_id":            event_id,
                    "asset_id":            asset_id,
                    "doc_id":              doc_id,
                    "cr_id":               rec.get("cr_id", ""),
                    "component_id":        rec.get("component_id"),
                    "component_ids":       [rec.get("component_id")] if rec.get("component_id") else [],
                    "is_sister_equipment": rec.get("is_sister_equipment", False),
                    "days_before_event":   rec.get("days_before_event"),
                    "status":              rec.get("status", ""),
                    **structured,
                },
            })

        for rec in cmms_context.get("wo_records") or []:
            text = rec.get("long_text") or rec.get("short_description") or ""
            if not text.strip():
                continue
            wo_id = str(rec.get("wo_id") or "").strip()
            doc_id = f"CMMS::WO::{wo_id}" if wo_id else ""
            structured = self._extract_structured_fields(rec)
            docs.append({
                "text": text,
                "metadata": {
                    "ingestion_path":      "path_a_structured",
                    "source":              "cmms_live",
                    "source_tier":         "plant_instance",
                    "record_type":         "wo",
                    "doc_type":            "WO",
                    "run_id":              run_id,
                    "event_id":            event_id,
                    "asset_id":            asset_id,
                    "doc_id":              doc_id,
                    "wo_id":               rec.get("wo_id", ""),
                    "component_id":        rec.get("component_id"),
                    "component_ids":       [rec.get("component_id")] if rec.get("component_id") else [],
                    "is_sister_equipment": rec.get("is_sister_equipment", False),
                    "days_before_event":   rec.get("days_before_event"),
                    "status":              rec.get("status", ""),
                    **structured,
                },
            })

        return docs

    @staticmethod
    def _normalize_token(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).replace("_", " ").replace("-", " ").lower().split()).strip()

    @classmethod
    def _extract_structured_fields(cls, record: JsonDict) -> JsonDict:
        """
        Extract and flatten Path-A structured CMMS fields for retrieval metadata.
        """
        out: JsonDict = {}
        condition = record.get("condition_assessment") or {}
        if isinstance(condition, dict):
            as_found = (
                condition.get("as_found_condition")
                or condition.get("as_found")
                or condition.get("as_found_text")
            )
            as_left = (
                condition.get("as_left_condition")
                or condition.get("as_left")
                or condition.get("as_left_text")
            )
            if as_found is not None:
                out["ca_as_found_condition"] = cls._normalize_token(as_found)
            if as_left is not None:
                out["ca_as_left_condition"] = cls._normalize_token(as_left)

        refs = record.get("failure_mode_refs") or []
        ref_tokens: List[str] = []
        if isinstance(refs, list):
            for row in refs:
                if isinstance(row, dict):
                    token = row.get("fm_id") or row.get("failure_mode_id") or row.get("label")
                    if token:
                        ref_tokens.append(str(token).strip())
                elif row:
                    ref_tokens.append(str(row).strip())
        if ref_tokens:
            deduped = sorted({x for x in ref_tokens if x})
            out["failure_mode_refs"] = deduped
            out["failure_mode_refs_text"] = " | ".join(deduped)

        statements = record.get("extracted_causal_statements") or []
        structured_lines: List[str] = []
        if isinstance(statements, list):
            for row in statements[:8]:
                if not isinstance(row, dict):
                    continue
                cause = str(row.get("cause_text") or "").strip()
                connector = str(row.get("connector") or "").strip()
                effect = str(row.get("effect_text") or "").strip()
                sentence = str(row.get("sentence_text") or row.get("sentence") or "").strip()
                text = sentence or " ".join(x for x in [cause, connector, effect] if x).strip()
                if text:
                    structured_lines.append(re.sub(r"\s+", " ", text))
        if structured_lines:
            out["causal_statements_text"] = " | ".join(structured_lines)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_lookback(
        self,
        kg_context: JsonDict,
        event_ts: Optional[datetime],
    ) -> tuple:
        """
        Returns (lookback_from: datetime, anchor_label: str).

        Searches kg_context.past_events[] for the most recent event of
        type "PM" or "preventive_maintenance" on the primary asset.
        Falls back to event_ts − fallback_lookback_days.
        """
        past_events = kg_context.get("past_events") or []
        pm_dates: List[datetime] = []

        for ev in past_events:
            if not isinstance(ev, dict):
                continue
            ev_type = (ev.get("event_type") or "").lower()
            if "pm" in ev_type or "preventive" in ev_type:
                dt = _parse_iso(ev.get("event_date") or ev.get("timestamp"))
                if dt:
                    pm_dates.append(dt)

        if pm_dates:
            last_pm = max(pm_dates)
            logger.debug("Lookback anchor: last PM at %s", last_pm.isoformat())
            return last_pm, "last_pm"

        # Fallback
        anchor_dt = event_ts or datetime.now(timezone.utc)
        lookback_from = anchor_dt - timedelta(days=self.config.fallback_lookback_days)
        logger.debug(
            "Lookback anchor: event_time − %d days = %s",
            self.config.fallback_lookback_days,
            lookback_from.isoformat(),
        )
        return lookback_from, "event_time_minus_90d"

    def _resolve_sisters(self, kg_context: JsonDict) -> List[JsonDict]:
        """
        Build the sister component list from topology + optional similarity tiers.

        Returns a list of dicts with keys:
        ``component_id``, ``component_label``, ``match_type``,
        ``shared_fm_count``, ``embedding_score``.

        Tier 1 (topology) is always included when ``include_sister_equipment``
        is True — these are components with ``relation_to_asset`` in the
        configured ``sister_relation_types``.

        Tier 2/3 (FM overlap + spec embedding) are added when
        ``config.similarity_resolver`` is set.  Results from both sources are
        merged; a component found in both topology and similarity tiers gets a
        combined ``match_type`` (e.g. ``"topology+failure_mode_overlap"``).
        """
        sisters: dict = {}  # component_id → record dict

        # Tier 1: topology (relation_to_asset in sister_relation_types)
        if self.config.include_sister_equipment:
            allowed = set(self.config.sister_relation_types)
            for comp in kg_context.get("components") or []:
                if not isinstance(comp, dict):
                    continue
                relation = comp.get("relation_to_asset") or ""
                if relation in allowed:
                    cid = comp.get("component_id")
                    if cid:
                        sisters[cid] = {
                            "component_id":    cid,
                            "component_label": comp.get("component_label"),
                            "match_type":      "topology",
                            "shared_fm_count": 0,
                            "embedding_score": 1.0,
                        }

        # Tier 2/3: failure mode overlap + spec embedding
        if self.config.similarity_resolver is not None:
            sister_set = set(sisters.keys())
            # Target = non-topology components (primary asset's parts)
            target_ids = [
                comp.get("component_id")
                for comp in (kg_context.get("components") or [])
                if isinstance(comp, dict)
                and comp.get("component_id")
                and comp.get("component_id") not in sister_set
            ]
            try:
                emb_sisters = self.config.similarity_resolver.resolve_similar(
                    target_component_ids=target_ids,
                    kg_context=kg_context,
                )
                for s in emb_sisters:
                    if s.component_id in sisters:
                        existing = sisters[s.component_id]
                        existing["match_type"] = f"topology+{s.match_type}"
                        existing["shared_fm_count"] = s.shared_fm_count
                        existing["embedding_score"] = s.embedding_score
                    else:
                        sisters[s.component_id] = s.to_dict()
            except Exception as exc:
                logger.warning(
                    "CMMSContextBuilder: similarity_resolver failed: %s", exc
                )

        return list(sisters.values())

    def _enrich_record(
        self,
        record: JsonDict,
        event_dt: Optional[datetime],
        is_cr: bool,
    ) -> JsonDict:
        """
        Add derived fields to a raw CMMS record:
        - ``days_before_event``: int or None
        - ``status``: normalised to open/closed/cancelled/unknown
        - ``is_sister_equipment``: bool (preserved from adapter or default False)
        """
        enriched = dict(record)

        # days_before_event
        created_dt = _parse_iso(record.get("created_date"))
        enriched["days_before_event"] = _days_between(created_dt, event_dt)

        # normalise status
        raw_status = (record.get("status") or "").lower().strip()
        if raw_status in {"open", "wappr", "wmatl", "wpcond", "inprg", "appr"}:
            enriched["status"] = "open"
        elif raw_status in {"comp", "closed", "close", "completed"}:
            enriched["status"] = "closed"
        elif raw_status in {"can", "cancelled", "canceled"}:
            enriched["status"] = "cancelled"
        elif not raw_status:
            enriched["status"] = "unknown"
        else:
            enriched["status"] = raw_status  # pass through unknown codes

        # default is_sister_equipment
        if "is_sister_equipment" not in enriched:
            enriched["is_sister_equipment"] = False

        return enriched

    def _build_recurrence_summary(
        self,
        cr_records: List[JsonDict],
        wo_records: List[JsonDict],
    ) -> JsonDict:
        cr_primary = [r for r in cr_records if not r.get("is_sister_equipment")]
        cr_sister  = [r for r in cr_records if r.get("is_sister_equipment")]
        open_wos   = [r for r in wo_records if r.get("status") == "open"]
        open_crs   = [r for r in cr_records if r.get("status") == "open"]

        all_cr_dates = sorted(
            [r.get("created_date") for r in cr_records if r.get("created_date")]
        )

        return {
            "cr_count_primary":         len(cr_primary),
            "cr_count_sister":          len(cr_sister),
            "open_wo_count":            len(open_wos),
            "open_cr_count":            len(open_crs),
            "earliest_related_cr_date": all_cr_dates[0]  if all_cr_dates else None,
            "most_recent_cr_date":      all_cr_dates[-1] if all_cr_dates else None,
        }
