from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

from orchestrators.temporal_relations import (
    Interval,
    allen_relation,
    onset_lag_hours,
    CAUSAL_PRIORITY,
    RELATION_SCORE,
    PRECEDES,
    OVERLAPS,
    CONTAINS,
    DURING,
    FOLLOWS,
)

JsonDict = Dict[str, Any]


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class TSKRTemporalScorerConfig:
    simultaneous_epsilon_hours: float = 0.5
    default_precedes_lag_hours: float = 1.0
    fallback_confidence: float = 0.25
    anomaly_weight: float = 0.55
    latency_weight: float = 0.30
    history_weight: float = 0.15
    min_confidence_for_support: float = 0.35
    anomaly_count_weight: float = 0.20
    lag_consistency_weight: float = 0.15
    telemetry_support_floor: float = 0.35
    tone_vocabulary_version: str = "npp_tone_v1"
    tone_transient_max_minutes: float = 5.0
    tone_watch_min_minutes: float = 5.0
    tone_alert_min_minutes: float = 10.0
    tone_trip_min_minutes: float = 2.0


@dataclass
class RecurrenceProfile:
    """Recurrence statistics for a (failure_mode, component) pair over past events."""
    fm_id: Optional[str]
    component_id: Optional[str]
    count: int                           # total matching past events
    mean_inter_event_days: Optional[float]
    trend: str                           # "increasing"|"decreasing"|"stable"|"insufficient_data"
    unresolved_count: int                # events with resolved == False
    most_recent_days_ago: Optional[int]


class TSKRTemporalScorerV1:
    """Deterministic temporal scorer using interval-based Allen relations.

    Inputs:
      - event interval (timestamp_start / timestamp_end)
      - telemetry anomaly windows (start, end, severity, pattern)
      - kg_context.failure_modes[*].expected_latency_min_hours/max_hours
      - kg_context.past_events for recurrence analysis

    Output:
      - tskr_patterns artifact aligned to tskr_patterns.json schema
    """

    def __init__(self, config: Optional[TSKRTemporalScorerConfig] = None):
        self.config = config or TSKRTemporalScorerConfig()

    # ------------------------------------------------------------------ #
    # Public entry point                                                    #
    # ------------------------------------------------------------------ #

    def score(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
        operational_context: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        event_id = event.get("event_id") or event.get("id")
        asset_id = event.get("asset_id")
        event_start = parse_dt(event.get("timestamp_start"))
        event_end   = parse_dt(event.get("timestamp_end")) or event_start

        anomaly_windows      = self._extract_anomaly_windows(telemetry_summary)
        anomaly_window_summary = self._summarize_anomaly_windows(anomaly_windows)
        signal_ids           = self._extract_signal_ids(telemetry_summary)
        telemetry_support    = self._telemetry_support_score(anomaly_windows)
        tone_summary         = self._summarize_tones(anomaly_windows)
        operator_family      = self._infer_operator_family(event_start, event_end, anomaly_windows)

        past_events = kg_context.get("past_events") or []
        stage_b_allen_by_component = self._stage_b_allen_relation_by_component(kg_context)
        patterns: List[JsonDict] = []
        for fm in kg_context.get("failure_modes", []) or []:
            fm_component_id = fm.get("component_id") or fm.get("applies_to_component_id")
            pattern = self._score_failure_mode_pattern(
                event_id=event_id,
                asset_id=asset_id,
                event_start=event_start,
                event_end=event_end,
                anomaly_windows=anomaly_windows,
                anomaly_window_summary=anomaly_window_summary,
                signal_ids=signal_ids,
                telemetry_support=telemetry_support,
                operator_family=operator_family,
                fm=fm,
                past_events=past_events,
                stage_b_allen_relation=stage_b_allen_by_component.get(str(fm_component_id or "")),
            )
            patterns.append(pattern)

        patterns.sort(key=lambda p: (-float(p.get("confidence") or 0.0), p.get("target_id") or ""))

        supported = [p for p in patterns if float(p.get("confidence") or 0.0) >= self.config.min_confidence_for_support]
        avg_conf = (
            sum(float(p.get("confidence") or 0.0) for p in patterns) / len(patterns)
            if patterns else 0.0
        )
        recurrence_quality = self._recurrence_match_quality_stats(past_events)

        return {
            "event_id": event_id,
            "asset_id": asset_id,
            "patterns": patterns,
            "summary": {
                "has_temporal_support": bool(supported),
                "mode": "deterministic_v1",
                "n_patterns": len(patterns),
                "n_supported_patterns": len(supported),
                "operator_family": operator_family,
                "anomaly_point_count": len(anomaly_windows),
                "signal_count": len(signal_ids),
                "avg_confidence": round(avg_conf, 4),
                "top_supported_targets": [
                    p.get("target_id")
                    for p in supported[:3]
                    if p.get("target_id")
                ],
                "total_cr_count": recurrence_quality["total_cr_count"],
                "unmatched_cr_count": recurrence_quality["unmatched_cr_count"],
                "unmatched_cr_rate": recurrence_quality["unmatched_cr_rate"],
                "high_cr_match_failure_rate": recurrence_quality["high_cr_match_failure_rate"],
                "tone_vocabulary_version": self.config.tone_vocabulary_version,
                "dominant_tone": tone_summary["dominant_tone"],
                "tone_counts": tone_summary["tone_counts"],
                "tone_calibration_uncertainty": tone_summary["tone_calibration_uncertainty"],
            },
            "provenance": {
                "generated_by": "TSKRTemporalScorerV1",
                "run_id": run_context.get("run_id"),
                "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "tone_vocabulary_version": self.config.tone_vocabulary_version,
            },
        }

    # ------------------------------------------------------------------ #
    # Extraction helpers                                                    #
    # ------------------------------------------------------------------ #

    def _extract_anomaly_windows(self, telemetry_summary: JsonDict) -> List[Dict[str, Any]]:
        windows: List[Dict[str, Any]] = []
        for sig in telemetry_summary.get("signals", []) or []:
            sensor_id = sig.get("sensor_id")
            for a in sig.get("anomalies", []) or []:
                start = parse_dt(a.get("timestamp_start"))
                end   = parse_dt(a.get("timestamp_end")) or start
                if start is None:
                    continue
                windows.append({
                    "sensor_id":     sensor_id,
                    "start":         start,
                    "end":           end,
                    "pattern":       a.get("pattern"),
                    "interval_type": self._normalize_interval_type(a.get("interval_type")),
                    "severity":      a.get("severity"),
                    "severity_score": (
                        a.get("severity_score")
                        if isinstance(a.get("severity_score"), (int, float))
                        else a.get("score")
                    ),
                    "tone":          self._classify_tone(
                        start=start,
                        end=end,
                        severity=a.get("severity"),
                        severity_score=(a.get("severity_score") if isinstance(a.get("severity_score"), (int, float)) else a.get("score")),
                    ),
                })
        windows.sort(key=lambda x: x["start"])
        return windows

    @staticmethod
    def _is_cr_like_event(row: JsonDict) -> bool:
        event_type = str((row or {}).get("event_type") or "").strip().lower()
        event_id = str((row or {}).get("event_id") or "").strip().upper()
        return (
            "cmms_cr" in event_type
            or event_type == "cr"
            or event_id.startswith("CMMS::CR::")
        )

    @classmethod
    def _recurrence_match_quality_stats(cls, past_events: List[JsonDict]) -> Dict[str, Any]:
        cr_events = [pe for pe in (past_events or []) if isinstance(pe, dict) and cls._is_cr_like_event(pe)]
        total_cr = len(cr_events)
        unmatched_cr = sum(
            1
            for pe in cr_events
            if not (pe.get("matched_failure_mode_ids") or pe.get("fm_id"))
        )
        unmatched_rate = (float(unmatched_cr) / float(total_cr)) if total_cr > 0 else 0.0
        return {
            "total_cr_count": total_cr,
            "unmatched_cr_count": unmatched_cr,
            "unmatched_cr_rate": round(unmatched_rate, 4),
            "high_cr_match_failure_rate": bool(total_cr > 0 and unmatched_rate > 0.30),
        }

    @staticmethod
    def _normalize_interval_type(value: Any) -> str:
        normalized = str(value or "closed").strip().lower()
        if normalized not in {"closed", "open", "half_open_start", "half_open_end"}:
            return "closed"
        return normalized

    @classmethod
    def _stage_b_allen_relation_by_component(cls, kg_context: JsonDict) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for row in (kg_context.get("out_of_boundary_anomalies") or []):
            if not isinstance(row, dict):
                continue
            comp = str(row.get("component_id") or row.get("related_component_id") or "").strip()
            relation = str(row.get("allen_relation") or "").strip().lower()
            if comp and relation:
                mapping[comp] = relation
        return mapping

    def _summarize_anomaly_windows(
        self, anomaly_windows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not anomaly_windows:
            return {
                "window_start": None, "window_end": None,
                "earliest_start": None, "latest_end": None,
                "duration_hours": None,
            }
        earliest_start = anomaly_windows[0]["start"]
        latest_end     = anomaly_windows[-1]["end"] or anomaly_windows[-1]["start"]
        duration_hours = (latest_end - earliest_start).total_seconds() / 3600.0
        return {
            "window_start":   earliest_start,
            "window_end":     latest_end,
            "earliest_start": earliest_start,
            "latest_end":     latest_end,
            "duration_hours": round(duration_hours, 4),
        }

    def _extract_signal_ids(self, telemetry_summary: JsonDict) -> List[str]:
        ids: List[str] = []
        for sig in telemetry_summary.get("signals", []) or []:
            if not isinstance(sig, dict):
                continue
            sensor_id = sig.get("sensor_id")
            if sensor_id and sig.get("anomalies"):
                ids.append(sensor_id)
        return ids

    # ------------------------------------------------------------------ #
    # Scoring helpers                                                       #
    # ------------------------------------------------------------------ #

    def _telemetry_support_score(self, anomaly_windows: Any) -> float:
        if isinstance(anomaly_windows, dict):
            anomaly_windows = self._extract_anomaly_windows(anomaly_windows)
        anomaly_windows = anomaly_windows or []
        total = 0.0
        count = 0
        for window in anomaly_windows:
            tone = str(window.get("tone") or "").strip().lower()
            base = {
                "trip_band_persistent": 0.90,
                "alert_band_persistent": 0.70,
                "watch_band_persistent": 0.50,
                "transient_excursion": 0.40,
                "unclassified_anomaly": self.config.telemetry_support_floor,
            }.get(tone, self.config.telemetry_support_floor)
            score = window.get("severity_score")
            if isinstance(score, (int, float)):
                base = max(base, min(1.0, float(score)))
            total += base
            count += 1
        if total == 0.0:
            return 0.0
        return clamp01(total / max(1.0, count))

    def _classify_tone(
        self,
        *,
        start: Optional[datetime],
        end: Optional[datetime],
        severity: Any,
        severity_score: Any,
    ) -> str:
        """
        Deterministic tone classification from anomaly severity and duration.
        """
        if start is None:
            return "unclassified_anomaly"
        end_dt = end or start
        duration_min = max(0.0, (end_dt - start).total_seconds() / 60.0)
        sev_text = str(severity or "").strip().lower()
        sev_val = 0.5
        has_known_severity = False
        if isinstance(severity_score, (int, float)):
            sev_val = float(severity_score)
            has_known_severity = True
        elif sev_text == "high":
            sev_val = 0.9
            has_known_severity = True
        elif sev_text == "medium":
            sev_val = 0.7
            has_known_severity = True
        elif sev_text == "low":
            sev_val = 0.5
            has_known_severity = True

        if not has_known_severity:
            return "unclassified_anomaly"

        if duration_min < self.config.tone_transient_max_minutes and sev_val < 0.85:
            return "transient_excursion"
        if sev_val >= 0.85 and duration_min >= self.config.tone_trip_min_minutes:
            return "trip_band_persistent"
        if sev_val >= 0.65 and duration_min >= self.config.tone_alert_min_minutes:
            return "alert_band_persistent"
        if sev_val >= 0.40 and duration_min >= self.config.tone_watch_min_minutes:
            return "watch_band_persistent"
        return "transient_excursion"

    @staticmethod
    def _summarize_tones(anomaly_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not anomaly_windows:
            return {
                "dominant_tone": None,
                "tone_counts": {},
                "tone_calibration_uncertainty": True,
            }
        counts: Dict[str, int] = {}
        for w in anomaly_windows:
            tone = str(w.get("tone") or "unclassified_anomaly")
            counts[tone] = counts.get(tone, 0) + 1
        dominant = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        uncertainty = bool(
            counts.get("unclassified_anomaly", 0) > 0
            or counts.get("transient_excursion", 0) == len(anomaly_windows)
        )
        return {
            "dominant_tone": dominant,
            "tone_counts": counts,
            "tone_calibration_uncertainty": uncertainty,
        }

    def _severity_weight(self, window: Dict[str, Any]) -> float:
        """Return a [0.1, 1.0] weight from a window's severity fields.

        Prefers the numeric ``severity_score`` when present; falls back to
        the string ``severity`` field; defaults to 0.5 (medium).
        """
        raw = window.get("severity_score")
        if isinstance(raw, (int, float)):
            return max(0.1, float(raw))
        sev = str(window.get("severity") or "").lower()
        return {"high": 0.9, "medium": 0.7, "low": 0.5}.get(sev, 0.5)

    def _effective_anomaly_count(self, windows: List[Dict[str, Any]]) -> float:
        """Severity-weighted effective anomaly count (sum of per-window severity weights).

        A single high-severity anomaly (weight 0.9) is treated similarly to
        ~1.8 low-severity ones (weight 0.5 each), preventing noise spikes from
        inflating the count-based score as much as high-confidence detections.
        """
        return sum(self._severity_weight(w) for w in windows)

    def _anomaly_count_score(self, n: float) -> float:
        """Map an effective anomaly count (float) to [0, 1].

        Thresholds are intentionally coarse — the exact breakpoints matter less
        than the monotone shape.  Accepts both raw integer counts (backward
        compatible) and the float effective counts from _effective_anomaly_count.
        """
        if n <= 0:
            return 0.0
        if n < 1.0:
            return 0.5 * n      # partial credit for sub-threshold effective weight
        if n < 2.0:
            return 0.5
        if n < 3.0:
            return 0.7
        if n < 4.0:
            return 0.85
        return 1.0

    def _lag_consistency_score(self, std_lag_hours: Optional[float]) -> float:
        if std_lag_hours is None:
            return 0.5
        if std_lag_hours <= 0.25:
            return 1.0
        if std_lag_hours <= 1.0:
            return 0.8
        if std_lag_hours <= 4.0:
            return 0.55
        return 0.3

    def _infer_operator_family(
        self,
        event_start: Optional[datetime],
        event_end: Optional[datetime],
        anomaly_windows: List[Dict[str, Any]],
    ) -> Optional[str]:
        if event_start and event_end and anomaly_windows:
            return "interval_interval"
        if event_start and event_end:
            return "interval_only"
        if anomaly_windows:
            return "anomaly_only"
        return None

    # ------------------------------------------------------------------ #
    # Allen-relation scoring                                                #
    # ------------------------------------------------------------------ #

    def _score_against_anomalies(
        self,
        *,
        event_start: Optional[datetime],
        event_end: Optional[datetime],
        anomaly_windows: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[float], Optional[float], float]:
        """Compute dominant Allen relation, severity-weighted lag stats, and
        severity-weighted anomaly score across all anomaly windows.

        Returns ``(dominant_relation, mean_lag_hours, std_lag_hours, anomaly_score)``.

        Dominant relation: the highest-priority causal relation present in any
        window (priority order: OVERLAPS > CONTAINS > PRECEDES > DURING > FOLLOWS).
        A single OVERLAPS window beats many DURING windows because it represents
        an anomaly that was already active at event onset.

        Lag stats: severity-weighted mean and std over causally-relevant windows
        (PRECEDES, OVERLAPS, CONTAINS) only.  High-severity anomalies pull the
        mean lag estimate more than low-severity noise spikes.  Weighted std is
        used for lag consistency scoring — tightly clustered high-severity windows
        produce a lower std and therefore a higher lag-consistency score.

        Anomaly score: severity-weighted mean of RELATION_SCORE values across all
        windows.
        """
        if not anomaly_windows or event_start is None:
            return "unknown", None, None, self.config.fallback_confidence

        event_end_dt   = event_end or event_start
        event_interval = Interval(start=event_start, end=event_end_dt)

        present_relations: set[str]                        = set()
        causal_lag_pairs: List[Tuple[float, float]]        = []   # (lag_hours, severity_weight)
        total_weighted_score = 0.0
        total_weight         = 0.0

        for window in anomaly_windows:
            start = window["start"]
            end   = window["end"] or start
            a     = Interval(start=start, end=end)
            rel, base = allen_relation(
                a,
                event_interval,
                self.config.simultaneous_epsilon_hours,
                interval_type=window.get("interval_type", "closed"),
            )
            weight = self._severity_weight(window)

            present_relations.add(rel)
            total_weighted_score += base * weight
            total_weight         += weight

            if rel in (PRECEDES, OVERLAPS, CONTAINS):
                causal_lag_pairs.append((onset_lag_hours(a, event_interval), weight))

        # Dominant relation by causal priority (first match in priority order wins)
        dominant_relation = "unknown"
        for rel in CAUSAL_PRIORITY:
            if rel in present_relations:
                dominant_relation = rel
                break

        # Severity-weighted mean and std lag over causal windows
        mean_lag: Optional[float] = None
        std_lag:  Optional[float] = None
        if causal_lag_pairs:
            total_lag_weight = sum(w for _, w in causal_lag_pairs)
            mean_lag = sum(lag * w for lag, w in causal_lag_pairs) / total_lag_weight
            if len(causal_lag_pairs) > 1:
                variance = (
                    sum(w * (lag - mean_lag) ** 2 for lag, w in causal_lag_pairs)
                    / total_lag_weight
                )
                std_lag = math.sqrt(variance)
            else:
                std_lag = 0.0

        anomaly_score = clamp01(total_weighted_score / max(total_weight, 1e-9))

        return dominant_relation, mean_lag, std_lag, anomaly_score

    # ------------------------------------------------------------------ #
    # Recurrence analysis                                                   #
    # ------------------------------------------------------------------ #

    def _build_recurrence_profile(
        self,
        *,
        fm_id: Optional[str],
        component_id: Optional[str],
        past_events: List[JsonDict],
    ) -> RecurrenceProfile:
        """Build a RecurrenceProfile by matching past_events to the given
        failure mode / component and deriving recurrence statistics."""
        matching = []
        for pe in past_events:
            matched_fms  = set(pe.get("matched_failure_mode_ids") or [])
            matched_comp = pe.get("component_id")
            if fm_id and fm_id in matched_fms:
                matching.append(pe)
            elif component_id and matched_comp == component_id:
                matching.append(pe)

        count = len(matching)

        # Sort by timestamp (ignore events without a parseable timestamp)
        dated = sorted(
            (
                (parse_dt(pe.get("timestamp_start")), pe)
                for pe in matching
                if parse_dt(pe.get("timestamp_start")) is not None
            ),
            key=lambda x: x[0],
        )

        # Inter-event intervals in days
        intervals_days: List[float] = [
            (dated[i][0] - dated[i - 1][0]).total_seconds() / 86400.0
            for i in range(1, len(dated))
        ]
        mean_inter_event_days = (
            sum(intervals_days) / len(intervals_days) if intervals_days else None
        )

        # resolved == False (explicit False, not None) counts as unresolved
        unresolved_count = sum(
            1 for _, pe in dated if pe.get("resolved") is False
        )

        # Recency from the most-recent matching event's time_distance_days field
        most_recent_days_ago: Optional[int] = None
        if dated:
            td = dated[-1][1].get("time_distance_days")
            if isinstance(td, (int, float)):
                most_recent_days_ago = int(td)

        return RecurrenceProfile(
            fm_id=fm_id,
            component_id=component_id,
            count=count,
            mean_inter_event_days=mean_inter_event_days,
            trend=self._recurrence_trend(intervals_days),
            unresolved_count=unresolved_count,
            most_recent_days_ago=most_recent_days_ago,
        )

    @staticmethod
    def _recurrence_trend(intervals: List[float]) -> str:
        """Compare first-half vs second-half inter-event intervals.

        Shrinking intervals mean events are becoming more frequent
        ("increasing" recurrence rate), which is a stronger causal signal.
        Requires at least 3 intervals (4 events) for meaningful comparison.
        """
        if len(intervals) < 3:
            return "insufficient_data"
        mid         = len(intervals) // 2
        first_mean  = sum(intervals[:mid]) / mid
        second_mean = sum(intervals[mid:]) / (len(intervals) - mid)
        if first_mean <= 0:
            return "insufficient_data"
        ratio = second_mean / first_mean
        if ratio < 0.75:
            return "increasing"   # intervals shrinking → accelerating recurrence
        if ratio > 1.33:
            return "decreasing"   # intervals growing → improving / resolved
        return "stable"

    def _score_from_recurrence_profile(self, profile: RecurrenceProfile) -> float:
        """Convert a RecurrenceProfile to a scalar history support score [0, 1].

        Scoring rationale:
          - Base score rises with occurrence count (each repetition adds evidence).
          - Accelerating trend (+0.15): shortening intervals suggest progressive
            degradation, a stronger causal signal than a stable recurrence rate.
          - Unresolved prior events (+0.10): latent condition that was never fixed.
          - Recent occurrence (+0.05): recency increases relevance to the current event.
        """
        if profile.count == 0:
            return 0.0
        if profile.count == 1:
            base = 0.35
        elif profile.count <= 3:
            base = 0.55
        elif profile.count <= 6:
            base = 0.70
        else:
            base = 0.80

        if profile.trend == "increasing":
            base += 0.15
        if profile.unresolved_count > 0:
            base += 0.10
        if profile.most_recent_days_ago is not None and profile.most_recent_days_ago < 90:
            base += 0.05
        return clamp01(base)

    def _score_history_support(
        self,
        *,
        fm_id: Optional[str],
        component_id: Optional[str],
        past_events: List[JsonDict],
    ) -> Tuple[float, RecurrenceProfile]:
        profile = self._build_recurrence_profile(
            fm_id=fm_id, component_id=component_id, past_events=past_events
        )
        return self._score_from_recurrence_profile(profile), profile

    # ------------------------------------------------------------------ #
    # Main pattern scorer                                                   #
    # ------------------------------------------------------------------ #

    def _score_failure_mode_pattern(
        self,
        *,
        event_id: str,
        asset_id: str,
        event_start: Optional[datetime],
        event_end: Optional[datetime],
        anomaly_windows: List[JsonDict],
        anomaly_window_summary: JsonDict,
        signal_ids: List[str],
        telemetry_support: float,
        operator_family: Optional[str],
        fm: JsonDict,
        past_events: List[JsonDict],
        stage_b_allen_relation: Optional[str] = None,
    ) -> JsonDict:

        fm_id        = fm.get("fm_id")
        component_id = fm.get("component_id")

        relation, mean_lag, std_lag, anomaly_score = self._score_against_anomalies(
            event_start=event_start,
            event_end=event_end,
            anomaly_windows=anomaly_windows,
        )

        effective_count      = self._effective_anomaly_count(anomaly_windows)
        anomaly_count_score  = self._anomaly_count_score(effective_count)
        lag_consistency_score = self._lag_consistency_score(std_lag)

        latency_details = self._latency_alignment_details(
            mean_lag_hours=mean_lag,
            expected_min=fm.get("expected_latency_min_hours"),
            expected_max=fm.get("expected_latency_max_hours"),
        )
        latency_score = latency_details["latency_alignment_score"]

        stage_b_temporal_contradiction = str(stage_b_allen_relation or "").lower() == FOLLOWS
        temporal_contradiction = (
            relation == FOLLOWS
            or latency_details["latency_violation_type"] in {"too_fast", "too_slow"}
            or stage_b_temporal_contradiction
        )

        history_score, recurrence_profile = self._score_history_support(
            fm_id=fm_id,
            component_id=component_id,
            past_events=past_events,
        )

        confidence = clamp01(
              self.config.anomaly_weight   * max(anomaly_score, telemetry_support)
            + self.config.latency_weight   * latency_score
            + self.config.history_weight   * history_score
            + self.config.anomaly_count_weight  * anomaly_count_score
            + self.config.lag_consistency_weight * lag_consistency_score
            - (0.20 if temporal_contradiction else 0.0)
        )

        support = clamp01(
              0.35 * history_score
            + 0.35 * telemetry_support
            + 0.15 * anomaly_count_score
            + 0.15 * lag_consistency_score
            - (0.15 if temporal_contradiction else 0.0)
        )

        return {
            "pattern_id":   f"TSKR::{fm_id}",
            "event_id":     event_id,
            "asset_id":     asset_id,
            "target_type":  "failure_mode",
            "target_id":    fm_id,
            "component_id": component_id,
            "relation":     relation,
            "operator_family": operator_family,
            "mean_lag_hours": round(mean_lag, 4) if mean_lag is not None else None,
            "std_lag_hours":  round(std_lag, 4)  if std_lag  is not None else None,
            "support":    round(support, 4),
            "confidence": round(confidence, 4),
            "matching_signal_ids": signal_ids,
            "anomaly_count": len(anomaly_windows),
            "lag_consistency": round(lag_consistency_score, 4),
            "source": "TSKRTemporalScorerV1",
            "window_start": (
                anomaly_window_summary.get("window_start").isoformat()
                if anomaly_window_summary.get("window_start") else None
            ),
            "window_end": (
                anomaly_window_summary.get("window_end").isoformat()
                if anomaly_window_summary.get("window_end") else None
            ),
            "duration_hours": anomaly_window_summary.get("duration_hours"),
            "expected_latency_min_hours": latency_details["expected_min_hours"],
            "expected_latency_max_hours": latency_details["expected_max_hours"],
            "observed_lag_hours":         latency_details["observed_lag_hours"],
            "latency_alignment_score":    latency_details["latency_alignment_score"],
            "latency_violation_type":     latency_details["latency_violation_type"],
            "temporal_contradiction": temporal_contradiction,
            "stage_b_allen_relation": stage_b_allen_relation,
            "stage_b_temporal_contradiction": stage_b_temporal_contradiction,
            # Recurrence fields (new)
            "recurrence_count":            recurrence_profile.count,
            "recurrence_trend":            recurrence_profile.trend,
            "unresolved_recurrence_count": recurrence_profile.unresolved_count,
        }

    # ------------------------------------------------------------------ #
    # Latency alignment                                                     #
    # ------------------------------------------------------------------ #

    def _score_expected_latency(
        self,
        *,
        mean_lag_hours: Optional[float],
        expected_min: Any,
        expected_max: Any,
    ) -> float:
        if mean_lag_hours is None:
            return self.config.fallback_confidence
        try:
            mn = float(expected_min) if expected_min is not None else None
            mx = float(expected_max) if expected_max is not None else None
        except Exception:
            return 0.4

        lag = abs(mean_lag_hours)
        if mn is None and mx is None:
            return 0.5
        if mn is not None and mx is not None:
            if mn <= lag <= mx:
                return 1.0
            if lag < mn:
                return clamp01(1.0 - ((mn - lag) / max(mn, 1.0)))
            return clamp01(1.0 - ((lag - mx) / max(mx, 1.0)))
        if mn is not None:
            return 1.0 if lag >= mn else clamp01(lag / max(mn, 1.0))
        if mx is not None:
            return 1.0 if lag <= mx else clamp01(1.0 - ((lag - mx) / max(mx, 1.0)))
        return 0.5

    def _latency_alignment_details(
        self,
        *,
        mean_lag_hours: Optional[float],
        expected_min: Any,
        expected_max: Any,
    ) -> Dict[str, Any]:
        if mean_lag_hours is None:
            return {
                "expected_min_hours": expected_min,
                "expected_max_hours": expected_max,
                "observed_lag_hours": None,
                "latency_alignment_score": self.config.fallback_confidence,
                "latency_violation_type": "unknown",
            }
        try:
            mn = float(expected_min) if expected_min is not None else None
            mx = float(expected_max) if expected_max is not None else None
        except Exception:
            return {
                "expected_min_hours": expected_min,
                "expected_max_hours": expected_max,
                "observed_lag_hours": round(float(mean_lag_hours), 4),
                "latency_alignment_score": 0.4,
                "latency_violation_type": "unknown",
            }

        lag   = abs(float(mean_lag_hours))
        score = self._score_expected_latency(mean_lag_hours=lag, expected_min=mn, expected_max=mx)

        violation = "none"
        if mn is not None and lag < mn:
            violation = "too_fast"
        elif mx is not None and lag > mx:
            violation = "too_slow"

        return {
            "expected_min_hours":     mn,
            "expected_max_hours":     mx,
            "observed_lag_hours":     round(lag, 4),
            "latency_alignment_score": round(float(score), 4),
            "latency_violation_type": violation,
        }
