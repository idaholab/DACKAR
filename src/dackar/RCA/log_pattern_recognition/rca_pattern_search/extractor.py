"""
IncidentExtractor — event normalisation and fingerprint derivation.

Converts raw alarm / SOE / anomaly records into a unified event list and
derives the three fingerprint representations (event_set, event_seq,
freq_vec) used by all downstream similarity metrics.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

from .config import SearchConfig
from .models import IncidentFingerprint, UnifiedEvent

_log = logging.getLogger(__name__)

# Anomaly inclusion: fall back to severity_score when promoted_to_kg_event is absent.
_DEFAULT_SEVERITY_THRESHOLD = 0.5


class IncidentExtractor:
    """
    Converts raw source records to UnifiedEvents and derives IncidentFingerprints.

    Two public entry points:
        to_unified_events()  — normalise all three sources into a flat list
        extract()            — full pipeline: expand window → filter → fingerprint

    _derive_fingerprint() is a staticmethod so that indexer.py can call it
    directly on pre-filtered episode event lists without instantiating an extractor.
    """

    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_unified_events(
        self,
        alarm_log: dict,
        soe_log: dict,
        telemetry_summaries: list[dict],
        *,
        anomaly_severity_threshold: float = _DEFAULT_SEVERITY_THRESHOLD,
    ) -> list[UnifiedEvent]:
        """
        Converts raw schema records to a flat UnifiedEvent list.

        Applies filtering defaults:
            - Alarms with state == "suppressed" are excluded.
            - Anomalies with promoted_to_kg_event == False are excluded.
              If the field is absent, severity_score >= anomaly_severity_threshold
              is used as the inclusion gate.
            - All SOE records are included.

        Args:
            alarm_log:                  Dict with key "alarms": list of alarm dicts.
            soe_log:                    Dict with key "records": list of SOE record dicts.
            telemetry_summaries:        List of telemetry summary dicts, each with
                                        key "anomalies": list of anomaly dicts.
            anomaly_severity_threshold: Fallback gate when promoted_to_kg_event absent.

        Returns:
            Flat list of UnifiedEvent, unsorted.
        """
        events: list[UnifiedEvent] = []
        events.extend(self._parse_alarm_log(alarm_log))
        events.extend(self._parse_soe_log(soe_log))
        for ts in telemetry_summaries:
            events.extend(
                self._parse_telemetry_summary(ts, anomaly_severity_threshold)
            )
        return events

    def extract(
        self,
        alarm_log: dict,
        soe_log: dict,
        telemetry_summaries: list[dict],
        incident_id: str,
        window_start: datetime,
        window_end: datetime,
        metadata: Optional[dict] = None,
    ) -> IncidentFingerprint:
        """
        Full extraction pipeline for a single query incident.

        Steps:
            1. Apply beta buffer to compute expanded window.
            2. Call to_unified_events() for all sources.
            3. Filter events to those with timestamp_start in expanded window.
            4. Compute density over the expanded window (consistent with historical episode density).
            5. Derive event_set, event_seq, freq_vec via _derive_fingerprint().

        Args:
            alarm_log:           Raw alarm log dict.
            soe_log:             Raw SOE log dict.
            telemetry_summaries: List of telemetry summary dicts.
            incident_id:         Identifier for this query incident.
            window_start:        Incident window start (before buffer expansion).
            window_end:          Incident window end (before buffer expansion).
            metadata:            Optional dict; "known_rca" and "asset_id" are read
                                 if present.

        Returns:
            IncidentFingerprint with expanded window stored as window_start/end.
        """
        if metadata is None:
            metadata = {}

        exp_start, exp_end = _expand_window(window_start, window_end, self.config.beta)

        all_events = self.to_unified_events(alarm_log, soe_log, telemetry_summaries)

        window_events = [
            e for e in all_events
            if exp_start <= e.timestamp_start <= exp_end
        ]

        density = _compute_density(all_events, exp_start, exp_end)

        event_set, event_seq, freq_vec = self._derive_fingerprint(
            window_events, self.config.freq_threshold
        )

        asset_id = metadata.get("asset_id") or _dominant_asset(window_events)
        source_types = sorted({ev.source for ev in window_events if ev.source})

        return IncidentFingerprint(
            episode_id=incident_id,
            asset_id=asset_id,
            window_start=exp_start,
            window_end=exp_end,
            density=density,
            event_set=event_set,
            event_seq=event_seq,
            freq_vec=freq_vec,
            known_rca=metadata.get("known_rca"),
            source_types=source_types,
        )

    # ------------------------------------------------------------------
    # Fingerprint derivation (called by indexer.py as well)
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_fingerprint(
        events: list[UnifiedEvent],
        freq_threshold: int,
    ) -> tuple[frozenset[str], list[str], dict[str, int]]:
        """
        Derives the three similarity representations from a list of events.

        High-frequency event types (count > freq_threshold) are excluded from
        event_set and event_seq but retained in freq_vec.

        Args:
            events:          Events belonging to a single episode or incident.
            freq_threshold:  Count above which a type is considered high-frequency.

        Returns:
            (event_set, event_seq, freq_vec)
        """
        if not events:
            return frozenset(), [], {}

        freq_vec: dict[str, int] = Counter(e.event_type for e in events)

        high_freq = {t for t, c in freq_vec.items() if c > freq_threshold}

        # event_set: deduplicated, high-freq excluded
        event_set = frozenset(t for t in freq_vec if t not in high_freq)

        # event_seq: deduplicated, ordered by first occurrence timestamp
        first_seen: dict[str, datetime] = {}
        for e in events:
            if e.event_type not in high_freq:
                if e.event_type not in first_seen or e.timestamp_start < first_seen[e.event_type]:
                    first_seen[e.event_type] = e.timestamp_start

        event_seq = sorted(first_seen, key=lambda t: first_seen[t])

        return event_set, event_seq, dict(freq_vec)

    # ------------------------------------------------------------------
    # Per-source parsers
    # ------------------------------------------------------------------

    def _parse_alarm_log(self, alarm_log: dict) -> list[UnifiedEvent]:
        events: list[UnifiedEvent] = []
        for rec in alarm_log.get("alarms", []):
            if rec.get("state") == "suppressed":
                continue
            ts_start = _parse_ts(rec.get("timestamp"))
            if ts_start is None:
                _log.warning("Alarm record missing timestamp; skipping: %s", rec.get("alarm_id"))
                continue
            events.append(
                UnifiedEvent(
                    raw_id=str(rec.get("alarm_id", "")),
                    asset_id=str(rec.get("asset_id", "")),
                    source="alarm",
                    event_type=str(rec.get("alarm_id", "")),
                    timestamp_start=ts_start,
                    timestamp_end=_parse_ts(rec.get("acknowledged_at")),
                )
            )
        return events

    def _parse_soe_log(self, soe_log: dict) -> list[UnifiedEvent]:
        records = soe_log.get("records", [])
        end_map = _derive_soe_end_timestamps(records)
        events: list[UnifiedEvent] = []
        for rec in records:
            ts_start = _parse_ts(rec.get("timestamp"))
            if ts_start is None:
                _log.warning("SOE record missing timestamp; skipping: %s", rec.get("record_id"))
                continue
            signal_id = str(rec.get("signal_id", ""))
            transition = str(rec.get("transition", ""))
            events.append(
                UnifiedEvent(
                    raw_id=str(rec.get("record_id", "")),
                    asset_id=str(rec.get("asset_id", "")),
                    source="soe",
                    event_type=f"{signal_id}::{transition}",
                    timestamp_start=ts_start,
                    timestamp_end=end_map.get(str(rec.get("record_id", ""))),
                )
            )
        return events

    def _parse_telemetry_summary(
        self,
        summary: dict,
        severity_threshold: float,
    ) -> list[UnifiedEvent]:
        top_asset = str(summary.get("asset_id", ""))
        events: list[UnifiedEvent] = []
        for rec in summary.get("anomalies", []):
            # Inclusion gate: promoted_to_kg_event preferred; severity_score fallback.
            promoted = rec.get("promoted_to_kg_event")
            if promoted is not None:
                if not promoted:
                    continue
            else:
                score = rec.get("severity_score")
                if score is not None and score < severity_threshold:
                    continue

            ts_start = _parse_ts(rec.get("timestamp_start"))
            if ts_start is None:
                _log.warning(
                    "Anomaly record missing timestamp_start; skipping: %s",
                    rec.get("anomaly_id"),
                )
                continue
            sensor_id = str(rec.get("sensor_id", ""))
            pattern = str(rec.get("pattern", ""))
            events.append(
                UnifiedEvent(
                    raw_id=str(rec.get("anomaly_id", "")),
                    asset_id=str(rec.get("asset_id") or top_asset),
                    source="anomaly",
                    event_type=f"{sensor_id}::{pattern}",
                    timestamp_start=ts_start,
                    timestamp_end=_parse_ts(rec.get("timestamp_end")),
                )
            )
        return events


# ------------------------------------------------------------------
# Module-level helpers (used by density.py and indexer.py too)
# ------------------------------------------------------------------

def _expand_window(
    window_start: datetime,
    window_end: datetime,
    beta: float,
) -> tuple[datetime, datetime]:
    """
    Applies beta buffer expansion symmetrically to a time window.

    E_search_start = window_start - beta * duration
    E_search_end   = window_end   + beta * duration
    """
    duration_s = (window_end - window_start).total_seconds()
    delta = timedelta(seconds=beta * duration_s)
    return window_start - delta, window_end + delta


def _compute_density(
    events: list[UnifiedEvent],
    window_start: datetime,
    window_end: datetime,
) -> float:
    """
    Computes event density over the given (unexpanded) window.

    rho = N_events_in_window / window_duration_seconds

    Returns 0.0 if duration is zero or negative.
    """
    duration_s = (window_end - window_start).total_seconds()
    if duration_s <= 0:
        return 0.0
    n = sum(1 for e in events if window_start <= e.timestamp_start <= window_end)
    return n / duration_s


def _dominant_asset(events: list[UnifiedEvent]) -> str:
    """Returns the most frequently occurring asset_id among events, or '' if empty."""
    if not events:
        return ""
    counts: Counter[str] = Counter(e.asset_id for e in events if e.asset_id)
    return counts.most_common(1)[0][0] if counts else ""


def _parse_ts(value) -> Optional[datetime]:
    """
    Coerces a value to datetime.

    Accepts: datetime objects, ISO 8601 strings.
    Returns None for None, pandas NaT, unparseable strings, or unknown types.
    """
    if value is None:
        return None
    # pandas NaT subclasses datetime, so check it BEFORE the isinstance guard.
    if type(value).__name__ == "NaTType":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            _log.debug("Could not parse timestamp string: %r", value)
            return None
    return None


def _derive_soe_end_timestamps(records: list[dict]) -> dict[str, Optional[datetime]]:
    """
    Derives timestamp_end for each SOE record as the timestamp of the next
    record with the same signal_id but a different transition value.

    This models the "opposing transition" pairing (e.g. trip → reset) without
    requiring an explicit transition vocabulary.

    Returns:
        Mapping from record_id (str) to timestamp_end (datetime | None).
    """
    # Group (ts, record_id, transition) by signal_id, sorted by ts.
    by_signal: dict[str, list[tuple[datetime, str, str]]] = defaultdict(list)
    for rec in records:
        ts = _parse_ts(rec.get("timestamp"))
        rec_id = str(rec.get("record_id", ""))
        signal_id = str(rec.get("signal_id", ""))
        transition = str(rec.get("transition", ""))
        if ts is not None and signal_id:
            by_signal[signal_id].append((ts, rec_id, transition))

    for sig in by_signal:
        by_signal[sig].sort(key=lambda x: x[0])

    result: dict[str, Optional[datetime]] = {}
    for events in by_signal.values():
        for i, (_, rec_id, transition) in enumerate(events):
            end_ts: Optional[datetime] = None
            for j in range(i + 1, len(events)):
                if events[j][2] != transition:
                    end_ts = events[j][0]
                    break
            result[rec_id] = end_ts

    return result
