"""
Stage B — KG Timeline Builder.

Responsibilities:
    1. Query the knowledge graph for all events linked to the component(s)
       resolved in Stage A within the configured time window.
    2. Assemble events from four source types: condition reports (CRs),
       work orders (WOs), preventive/corrective maintenance records,
       and prior emergent activities.
    3. Sort events chronologically and assign data quality scores.
    4. Compute recurrence indicators (repeat failure count, inter-event
       period, PM compliance status).
    5. Summarise data coverage (event count, outages represented, window
       span, any known gaps).

Output schema: outage/schemas/component_event_timeline.json

Reuse targets:
    RCA.kg.py2neo_workflow.Py2Neo              → all KG queries (read-only import)
    RCA.storage.processed_record_store         → CR / WO text retrieval (injected)

KG node labels (defaults match RCA kg_schema_builder_workflow labels):
    mbse_entity       — components, systems, assets
    condition_report  — CR nodes
    work_order        — WO, PM, CM nodes (distinguished by work_type property)
    abnormal_event    — prior emergent activities
    inspection        — standalone inspection records

KG relationship types:
    LINKED_TO   — ConditionReport → Component
    PERFORMED_ON — WorkOrder → Component
    PART_OF     — Component → System → Asset
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_dt(iso_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 string to timezone-aware datetime; returns None on failure."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO string, or None."""
    return dt.isoformat() if dt is not None else None


def _window_start_iso(before_ts: Optional[str], window_days: int) -> Optional[str]:
    """Compute the ISO start of the timeline window."""
    dt = _parse_dt(before_ts)
    if dt is None:
        return None
    return _iso(dt - timedelta(days=window_days))


def _score_event_dq(event: JsonDict) -> float:
    """Compute a [0, 1] data quality score for a raw KG event record.

    Based on field completeness:
        timestamp present:      +0.35  (required for temporal chain)
        description present:    +0.30
        source_doc_id present:  +0.20
        source_system present:  +0.15
    """
    score = 0.0
    if event.get("timestamp"):
        score += 0.35
    if event.get("description", ""):
        score += 0.30
    if event.get("source_doc_id"):
        score += 0.20
    if event.get("source_system"):
        score += 0.15
    return round(score, 4)


def _record_to_dict(record) -> JsonDict:
    """Convert a Neo4j Record object to a plain Python dict."""
    try:
        return dict(record)
    except Exception:  # noqa: BLE001
        return {}


def _node_props(record, key: str = "n") -> JsonDict:
    """Extract properties from a Neo4j node within a record."""
    try:
        node = record[key]
        if node is None:
            return {}
        return dict(node)
    except (KeyError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KGTimelineConfig:
    """Configuration for Stage B."""

    timeline_window_days: int = 1825
    """How far back from the emergent activity detection timestamp to query
    the KG (in days).  Default 5 years."""

    max_events: int = 100
    """Maximum events to include in the timeline. Oldest events are truncated
    when this limit is reached."""

    include_work_orders: bool = True
    include_condition_reports: bool = True
    include_preventive_maintenance: bool = True
    include_corrective_maintenance: bool = True
    include_prior_emergent_activities: bool = True
    include_inspections: bool = True

    pm_overdue_threshold_days: int = 30
    """Number of days past the scheduled PM due date before flagging as overdue."""

    default_pm_interval_days: int = 180
    """Assumed PM interval when no interval is stored on the KG component node."""

    # ── KG schema labels (override to match your deployment's schema) ─────────
    component_label: str = "mbse_entity"
    """Neo4j label for component / system / asset nodes."""

    condition_report_label: str = "condition_report"
    """Neo4j label for condition report nodes."""

    work_order_label: str = "work_order"
    """Neo4j label for work order nodes (PM, CM, and general WOs are
    distinguished by a work_type property on the node)."""

    abnormal_event_label: str = "abnormal_event"
    """Neo4j label for prior emergent-activity / abnormal event nodes."""

    inspection_label: str = "inspection"
    """Neo4j label for standalone inspection record nodes."""

    # ── KG relationship types ─────────────────────────────────────────────────
    cr_component_rel: str = "LINKED_TO"
    wo_component_rel: str = "PERFORMED_ON"
    component_hierarchy_rel: str = "PART_OF"

    # ── Work-type codes used to filter work_order nodes ──────────────────────
    pm_work_type_code: str = "PM"
    cm_work_type_code: str = "CM"

    # ── Neo4j database name (None = driver default) ───────────────────────────
    kg_database: Optional[str] = None


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class KGTimelineBuilder:
    """Concrete Stage B implementation.

    Args:
        config: Stage configuration.
        kg_driver: Py2Neo instance (from RCA.kg.py2neo_workflow).
                   Provides all KG query methods.  When None, all KG queries
                   are skipped and the returned timeline contains zero events.
        record_store: ProcessedRecordStore instance (from RCA.storage).
                      Used to retrieve text snippets for CR/WO events when the
                      KG node itself carries only a source_doc_id.  Optional.
    """

    def __init__(
        self,
        config: Optional[KGTimelineConfig] = None,
        *,
        kg_driver=None,
        record_store=None,
    ) -> None:
        self.config = config or KGTimelineConfig()
        self.kg_driver = kg_driver
        self.record_store = record_store

    # ── Protocol method ───────────────────────────────────────────────────────

    def build(
        self,
        emergent_activity: JsonDict,
        intake_result: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """Execute Stage B for one emergent activity.

        Returns:
            ComponentEventTimeline artifact conforming to
            outage/schemas/component_event_timeline.json.
        """
        run_id: str = run_context["run_id"]
        activity_id: str = emergent_activity["activity_id"]
        component_id = self._select_primary_component(intake_result)
        LOGGER.debug(
            "Stage B building timeline for component %s (run=%s)",
            component_id, run_id,
        )

        detection_ts = emergent_activity.get("detection_timestamp")
        events: List[JsonDict] = []

        if self.kg_driver is None:
            LOGGER.warning(
                "Stage B: kg_driver not injected — returning empty timeline for %s",
                component_id,
            )
        else:
            if self.config.include_condition_reports:
                events.extend(self._query_condition_reports(component_id, detection_ts))
            if self.config.include_work_orders:
                events.extend(self._query_work_orders(component_id, detection_ts))
            if self.config.include_preventive_maintenance:
                events.extend(self._query_preventive_maintenance(component_id, detection_ts))
            if self.config.include_corrective_maintenance:
                events.extend(self._query_corrective_maintenance(component_id, detection_ts))
            if self.config.include_prior_emergent_activities:
                events.extend(
                    self._query_prior_emergent_activities(component_id, detection_ts)
                )
            if self.config.include_inspections:
                events.extend(self._query_inspections(component_id, detection_ts))

        events = self._deduplicate_events(events)
        events = self._sort_and_cap(events)
        recurrence = self._compute_recurrence_indicators(events, detection_ts)
        coverage = self._compute_data_coverage(events, detection_ts)
        component_meta = self._fetch_component_metadata(component_id)

        return {
            "activity_id": activity_id,
            "run_id": run_id,
            "generated_at": run_context.get("started_at", ""),
            "component_id": component_id,
            "component_name": component_meta.get("name"),
            "system_id": component_meta.get("system_id"),
            "system_name": component_meta.get("system_name"),
            "asset_id": component_meta.get("asset_id"),
            "events": events,
            "recurrence_indicators": recurrence,
            "data_coverage": coverage,
            "provenance": {
                "generated_by": self.__class__.__name__,
                "run_id": run_id,
                "kg_query_timestamp": run_context.get("started_at"),
                "timeline_window_days": self.config.timeline_window_days,
            },
        }

    # ── Private step methods ──────────────────────────────────────────────────

    def _select_primary_component(self, intake_result: JsonDict) -> str:
        """Return the primary component_id to build the timeline for.

        Uses the first entry in resolved_component_ids if available;
        falls back to known_component_id from the original intake record.

        Raises ValueError if no component can be resolved.
        """
        resolved = intake_result.get("resolved_component_ids") or []
        if resolved:
            return resolved[0]
        fallback = intake_result.get("known_component_id")
        if fallback:
            return fallback
        raise ValueError(
            "Stage B: cannot determine component_id — "
            "resolved_component_ids is empty and known_component_id is not set"
        )

    def _fetch_component_metadata(self, component_id: str) -> JsonDict:
        """Fetch component name, system_id, system_name, asset_id from the KG.

        Traverses the component → system → asset hierarchy using PART_OF edges.
        Returns an empty dict (with None values) when the KG is unavailable or
        the component node does not exist.

        Reuse: Py2Neo.query() — read-only access, no modification to RCA KG code.
        """
        if self.kg_driver is None:
            return {"name": None, "system_id": None, "system_name": None, "asset_id": None}

        cypher = (
            f"MATCH (c:`{self.config.component_label}` {{id: $component_id}})\n"
            f"OPTIONAL MATCH (c)-[:`{self.config.component_hierarchy_rel}`]->(s:`{self.config.component_label}`)\n"
            f"OPTIONAL MATCH (s)-[:`{self.config.component_hierarchy_rel}`]->(a:`{self.config.component_label}`)\n"
            "RETURN c.name AS name, c.component_type AS component_type,\n"
            "       s.id AS system_id, s.name AS system_name,\n"
            "       a.id AS asset_id"
        )
        try:
            rows = self.kg_driver.query(
                cypher,
                parameters={"component_id": component_id},
                db=self.config.kg_database,
            )
            if not rows:
                return {"name": None, "system_id": None, "system_name": None, "asset_id": None}
            r = _record_to_dict(rows[0])
            return {
                "name": r.get("name"),
                "system_id": r.get("system_id"),
                "system_name": r.get("system_name"),
                "asset_id": r.get("asset_id"),
            }
        except Exception:  # noqa: BLE001
            LOGGER.warning("Stage B: _fetch_component_metadata failed for %s", component_id)
            return {"name": None, "system_id": None, "system_name": None, "asset_id": None}

    def _query_condition_reports(
        self, component_id: str, before_ts: Optional[str]
    ) -> List[JsonDict]:
        """Fetch CRs linked to this component within the timeline window.

        Cypher pattern:
            MATCH (cr:condition_report)-[:LINKED_TO]->(c:mbse_entity {id: $component_id})
            WHERE cr.initiated_date >= $window_start
              AND ($before_ts IS NULL OR cr.initiated_date < $before_ts)
            RETURN cr

        Reuse: Py2Neo.query() from RCA.kg.py2neo_workflow (read-only import).
        """
        window_start = _window_start_iso(before_ts, self.config.timeline_window_days)
        cypher = (
            f"MATCH (cr:`{self.config.condition_report_label}`)"
            f"-[:`{self.config.cr_component_rel}`]->"
            f"(c:`{self.config.component_label}` {{id: $component_id}})\n"
            "WHERE ($window_start IS NULL OR cr.initiated_date >= $window_start)\n"
            "  AND ($before_ts IS NULL OR cr.initiated_date < $before_ts)\n"
            "RETURN cr"
        )
        rows = self._run_query(cypher, {
            "component_id": component_id,
            "window_start": window_start,
            "before_ts": before_ts,
        })
        events = []
        for record in rows:
            props = _node_props(record, "cr")
            if not props:
                continue
            event = self._normalise_event(
                props,
                event_type="condition_report",
                timestamp_field="initiated_date",
            )
            events.append(event)
        return events

    def _query_work_orders(
        self, component_id: str, before_ts: Optional[str]
    ) -> List[JsonDict]:
        """Fetch general WOs (excluding PM/CM — those are queried separately).

        Cypher pattern:
            MATCH (wo:work_order)-[:PERFORMED_ON]->(c:mbse_entity {id: $component_id})
            WHERE wo.initiated_date >= $window_start
              AND NOT wo.work_type IN ['PM', 'CM']
            RETURN wo

        Reuse: Py2Neo.query() from RCA.kg.py2neo_workflow.
        """
        window_start = _window_start_iso(before_ts, self.config.timeline_window_days)
        pm_code = self.config.pm_work_type_code
        cm_code = self.config.cm_work_type_code
        cypher = (
            f"MATCH (wo:`{self.config.work_order_label}`)"
            f"-[:`{self.config.wo_component_rel}`]->"
            f"(c:`{self.config.component_label}` {{id: $component_id}})\n"
            "WHERE ($window_start IS NULL OR wo.initiated_date >= $window_start)\n"
            "  AND ($before_ts IS NULL OR wo.initiated_date < $before_ts)\n"
            f"  AND NOT wo.work_type IN ['{pm_code}', '{cm_code}']\n"
            "RETURN wo"
        )
        rows = self._run_query(cypher, {
            "component_id": component_id,
            "window_start": window_start,
            "before_ts": before_ts,
        })
        events = []
        for record in rows:
            props = _node_props(record, "wo")
            if not props:
                continue
            event = self._normalise_event(
                props,
                event_type="work_order",
                timestamp_field="initiated_date",
            )
            events.append(event)
        return events

    def _query_preventive_maintenance(
        self, component_id: str, before_ts: Optional[str]
    ) -> List[JsonDict]:
        """Fetch scheduled PM records for this component.

        Filters work_order nodes where work_type == config.pm_work_type_code.
        The completion_date (or initiated_date) is stored as the event timestamp.

        Reuse: Py2Neo.query() from RCA.kg.py2neo_workflow.
        """
        window_start = _window_start_iso(before_ts, self.config.timeline_window_days)
        cypher = (
            f"MATCH (wo:`{self.config.work_order_label}`)"
            f"-[:`{self.config.wo_component_rel}`]->"
            f"(c:`{self.config.component_label}` {{id: $component_id}})\n"
            "WHERE wo.work_type = $pm_code\n"
            "  AND ($window_start IS NULL OR wo.initiated_date >= $window_start)\n"
            "  AND ($before_ts IS NULL OR wo.initiated_date < $before_ts)\n"
            "RETURN wo"
        )
        rows = self._run_query(cypher, {
            "component_id": component_id,
            "window_start": window_start,
            "before_ts": before_ts,
            "pm_code": self.config.pm_work_type_code,
        })
        events = []
        for record in rows:
            props = _node_props(record, "wo")
            if not props:
                continue
            # Prefer completion_date for PM events (actual execution timestamp)
            ts_field = "completion_date" if props.get("completion_date") else "initiated_date"
            event = self._normalise_event(
                props,
                event_type="preventive_maintenance",
                timestamp_field=ts_field,
            )
            events.append(event)
        return events

    def _query_corrective_maintenance(
        self, component_id: str, before_ts: Optional[str]
    ) -> List[JsonDict]:
        """Fetch corrective maintenance records for this component.

        Filters work_order nodes where work_type == config.cm_work_type_code.

        Reuse: Py2Neo.query() from RCA.kg.py2neo_workflow.
        """
        window_start = _window_start_iso(before_ts, self.config.timeline_window_days)
        cypher = (
            f"MATCH (wo:`{self.config.work_order_label}`)"
            f"-[:`{self.config.wo_component_rel}`]->"
            f"(c:`{self.config.component_label}` {{id: $component_id}})\n"
            "WHERE wo.work_type = $cm_code\n"
            "  AND ($window_start IS NULL OR wo.initiated_date >= $window_start)\n"
            "  AND ($before_ts IS NULL OR wo.initiated_date < $before_ts)\n"
            "RETURN wo"
        )
        rows = self._run_query(cypher, {
            "component_id": component_id,
            "window_start": window_start,
            "before_ts": before_ts,
            "cm_code": self.config.cm_work_type_code,
        })
        events = []
        for record in rows:
            props = _node_props(record, "wo")
            if not props:
                continue
            ts_field = "completion_date" if props.get("completion_date") else "initiated_date"
            event = self._normalise_event(
                props,
                event_type="corrective_maintenance",
                timestamp_field=ts_field,
            )
            events.append(event)
        return events

    def _query_prior_emergent_activities(
        self, component_id: str, before_ts: Optional[str]
    ) -> List[JsonDict]:
        """Fetch prior emergent activities involving this component.

        Queries abnormal_event nodes linked to this component.  These represent
        historical unexpected-activity records ingested from prior outage runs.

        Cypher pattern:
            MATCH (e:abnormal_event)-[:LINKED_TO]->(c:mbse_entity {id: $component_id})
            WHERE e.timestamp_start >= $window_start
            RETURN e

        Reuse: Py2Neo.query() from RCA.kg.py2neo_workflow.
        """
        window_start = _window_start_iso(before_ts, self.config.timeline_window_days)
        cypher = (
            f"MATCH (e:`{self.config.abnormal_event_label}`)"
            f"-[:`{self.config.cr_component_rel}`]->"
            f"(c:`{self.config.component_label}` {{id: $component_id}})\n"
            "WHERE ($window_start IS NULL OR e.timestamp_start >= $window_start)\n"
            "  AND ($before_ts IS NULL OR e.timestamp_start < $before_ts)\n"
            "RETURN e"
        )
        rows = self._run_query(cypher, {
            "component_id": component_id,
            "window_start": window_start,
            "before_ts": before_ts,
        })
        events = []
        for record in rows:
            props = _node_props(record, "e")
            if not props:
                continue
            event = self._normalise_event(
                props,
                event_type="prior_emergent_activity",
                timestamp_field="timestamp_start",
            )
            events.append(event)
        return events

    def _query_inspections(
        self, component_id: str, before_ts: Optional[str]
    ) -> List[JsonDict]:
        """Fetch standalone inspection records for this component.

        Reuse: Py2Neo.query() from RCA.kg.py2neo_workflow.
        """
        window_start = _window_start_iso(before_ts, self.config.timeline_window_days)
        cypher = (
            f"MATCH (ins:`{self.config.inspection_label}`)"
            f"-[:`{self.config.cr_component_rel}`]->"
            f"(c:`{self.config.component_label}` {{id: $component_id}})\n"
            "WHERE ($window_start IS NULL OR ins.inspection_date >= $window_start)\n"
            "  AND ($before_ts IS NULL OR ins.inspection_date < $before_ts)\n"
            "RETURN ins"
        )
        rows = self._run_query(cypher, {
            "component_id": component_id,
            "window_start": window_start,
            "before_ts": before_ts,
        })
        events = []
        for record in rows:
            props = _node_props(record, "ins")
            if not props:
                continue
            event = self._normalise_event(
                props,
                event_type="inspection",
                timestamp_field="inspection_date",
            )
            events.append(event)
        return events

    def _deduplicate_events(self, events: List[JsonDict]) -> List[JsonDict]:
        """Remove duplicate events by source_doc_id, keeping the highest DQ score.

        Events without a source_doc_id are retained as-is (they are treated as
        distinct since we have no deduplication key for them).
        """
        keyed: Dict[str, JsonDict] = {}
        no_key: List[JsonDict] = []

        for event in events:
            doc_id = event.get("source_doc_id")
            if not doc_id:
                no_key.append(event)
                continue
            existing = keyed.get(doc_id)
            if existing is None or event.get("data_quality_score", 0) > existing.get("data_quality_score", 0):
                keyed[doc_id] = event

        return list(keyed.values()) + no_key

    def _sort_and_cap(self, events: List[JsonDict]) -> List[JsonDict]:
        """Sort events ascending by timestamp and cap at config.max_events.

        Events with a null timestamp are placed at the end of the sorted list
        (they cannot be used in Allen relation scoring but are retained for
        recurrence counting).
        """
        def _sort_key(e: JsonDict):
            ts = _parse_dt(e.get("timestamp"))
            if ts is None:
                return datetime.max.replace(tzinfo=timezone.utc)
            return ts

        sorted_events = sorted(events, key=_sort_key)
        if len(sorted_events) > self.config.max_events:
            LOGGER.debug(
                "Stage B: capping timeline at %d events (%d total retrieved)",
                self.config.max_events,
                len(sorted_events),
            )
            # Keep the most recent max_events (most relevant to Stage C)
            sorted_events = sorted_events[-self.config.max_events:]
        return sorted_events

    def _compute_recurrence_indicators(
        self, events: List[JsonDict], detection_ts: Optional[str]
    ) -> JsonDict:
        """Compute repeat failure count, mean inter-event days, trend, and PM status.

        trend logic:
            Split the timeline window into two halves by event count.
            If second-half count > first-half count × 1.5 → 'increasing'
            If first-half count > second-half count × 1.5 → 'decreasing'
            Otherwise → 'stable'
            If fewer than 3 events → 'insufficient_data'

        PM compliance:
            Locate the most recent preventive_maintenance event in the timeline.
            Compare its timestamp against detection_ts using
            config.default_pm_interval_days.
            If elapsed > interval + pm_overdue_threshold_days → 'overdue'
            Else → 'current'
        """
        # ── Repeat failure count ──────────────────────────────────────────────
        # Count events that represent a problem (CRs + corrective maintenance +
        # prior emergent activities)
        failure_types = {
            "condition_report",
            "corrective_maintenance",
            "prior_emergent_activity",
        }
        failure_events = [e for e in events if e.get("event_type") in failure_types]
        repeat_failure_count = len(failure_events)

        # ── Inter-event periods ───────────────────────────────────────────────
        timed_events = [
            e for e in events if _parse_dt(e.get("timestamp")) is not None
        ]
        inter_event_days: List[float] = []
        for i in range(1, len(timed_events)):
            t1 = _parse_dt(timed_events[i - 1]["timestamp"])
            t2 = _parse_dt(timed_events[i]["timestamp"])
            if t1 and t2:
                delta = (t2 - t1).total_seconds() / 86400.0
                if delta >= 0:
                    inter_event_days.append(delta)

        mean_inter_event_days: Optional[float] = None
        min_inter_event_days: Optional[float] = None
        if inter_event_days:
            mean_inter_event_days = round(sum(inter_event_days) / len(inter_event_days), 1)
            min_inter_event_days = round(min(inter_event_days), 1)

        # ── Trend ─────────────────────────────────────────────────────────────
        if len(timed_events) < 3:
            trend = "insufficient_data"
        else:
            mid = len(timed_events) // 2
            first_half_count = mid
            second_half_count = len(timed_events) - mid
            if second_half_count > first_half_count * 1.5:
                trend = "increasing"
            elif first_half_count > second_half_count * 1.5:
                trend = "decreasing"
            else:
                trend = "stable"

        # ── PM compliance ─────────────────────────────────────────────────────
        pm_events = [
            e for e in timed_events if e.get("event_type") == "preventive_maintenance"
        ]
        last_pm_date: Optional[str] = None
        pm_compliance_status = "unknown"
        pm_overdue_days: Optional[int] = None

        if pm_events:
            last_pm_event = pm_events[-1]  # already sorted ascending
            last_pm_date = last_pm_event.get("timestamp")
            if last_pm_date and detection_ts:
                pm_dt = _parse_dt(last_pm_date)
                detection_dt = _parse_dt(detection_ts)
                if pm_dt and detection_dt:
                    elapsed_days = (detection_dt - pm_dt).total_seconds() / 86400.0
                    interval = self.config.default_pm_interval_days
                    overdue_threshold = interval + self.config.pm_overdue_threshold_days
                    if elapsed_days > overdue_threshold:
                        pm_compliance_status = "overdue"
                        pm_overdue_days = int(elapsed_days - interval)
                    else:
                        pm_compliance_status = "current"

        return {
            "repeat_failure_count": repeat_failure_count,
            "mean_inter_event_days": mean_inter_event_days,
            "min_inter_event_days": min_inter_event_days,
            "trend": trend,
            "last_pm_date": last_pm_date,
            "pm_compliance_status": pm_compliance_status,
            "pm_overdue_days": pm_overdue_days,
        }

    def _compute_data_coverage(
        self, events: List[JsonDict], detection_ts: Optional[str]
    ) -> JsonDict:
        """Summarise total_events, outages_represented, earliest/latest event.

        outages_represented: count of distinct outage_id values across events
        (events without an outage_id are excluded from this count).
        """
        total_events = len(events)

        timed = [e for e in events if _parse_dt(e.get("timestamp")) is not None]
        earliest = _iso(_parse_dt(timed[0]["timestamp"])) if timed else None
        latest = _iso(_parse_dt(timed[-1]["timestamp"])) if timed else None

        outage_ids = {
            e["outage_id"]
            for e in events
            if e.get("outage_id")
        }
        outages_represented = len(outage_ids)

        window_start = _window_start_iso(detection_ts, self.config.timeline_window_days)

        return {
            "total_events": total_events,
            "outages_represented": outages_represented,
            "earliest_event_date": earliest,
            "latest_event_date": latest,
            "window_start": window_start,
            "window_end": detection_ts,
            "has_gaps": False,  # gap detection requires KG completeness metadata
        }

    # ── Private utilities ─────────────────────────────────────────────────────

    def _run_query(
        self, cypher: str, parameters: JsonDict
    ) -> List:
        """Execute a Cypher query via Py2Neo and return the raw record list.

        Returns an empty list on any exception so that a KG connectivity
        problem in one query does not abort the entire pipeline.

        Reuse: Py2Neo.query() from RCA.kg.py2neo_workflow.
        """
        try:
            return self.kg_driver.query(
                cypher,
                parameters=parameters,
                db=self.config.kg_database,
            )
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "Stage B: KG query failed (first 120 chars of Cypher: %s)",
                cypher[:120].replace("\n", " "),
                exc_info=True,
            )
            return []

    def _normalise_event(
        self,
        props: JsonDict,
        event_type: str,
        timestamp_field: str,
    ) -> JsonDict:
        """Convert raw KG node properties into a canonical timeline event dict.

        Canonical fields:
            event_id, event_type, timestamp, end_timestamp,
            description, source_doc_id, source_system,
            outage_id, data_quality_score.

        Attempts to enrich description from record_store when the KG node
        only carries a source_doc_id.

        Reuse: record_store (from RCA.storage) — injected, no modification.
        """
        event_id = (
            props.get("id")
            or props.get("event_id")
            or props.get("cr_id")
            or props.get("wo_id")
            or f"EVT::{event_type}::{uuid.uuid4().hex[:8]}"
        )

        description = props.get("description") or props.get("title") or ""
        source_doc_id = props.get("source_doc_id") or props.get("id")

        # Enrich description from record_store when available
        if not description and source_doc_id and self.record_store is not None:
            try:
                record = self.record_store.get(source_doc_id)
                if record:
                    description = (
                        record.get("summary")
                        or record.get("text", "")[:500]
                    )
            except Exception:  # noqa: BLE001
                pass

        event = {
            "event_id": str(event_id),
            "event_type": event_type,
            "timestamp": props.get(timestamp_field),
            "end_timestamp": props.get("completion_date") or props.get("timestamp_end"),
            "description": description,
            "source_doc_id": source_doc_id,
            "source_system": props.get("source_system"),
            "outage_id": props.get("outage_id"),
            "work_type": props.get("work_type"),
        }

        # Compute DQ score now that all fields are populated
        event["data_quality_score"] = _score_event_dq(event)
        return event
