from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

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


class TSKRTemporalScorerV1:
    """
    First real deterministic temporal scorer.

    Inputs:
      - event interval
      - telemetry anomaly timestamps
      - kg_context.failure_modes[*].expected_latency_min_hours/max_hours
      - kg_context.past_events for weak historical support

    Output:
      - tskr_patterns artifact aligned to tskr_patterns.json schema
    """

    def __init__(self, config: Optional[TSKRTemporalScorerConfig] = None):
        self.config = config or TSKRTemporalScorerConfig()

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
        event_end = parse_dt(event.get("timestamp_end")) or event_start

        anomaly_points = self._extract_anomaly_points(telemetry_summary)
        anomaly_windows = self._extract_anomaly_windows(telemetry_summary)
        anomaly_window_summary = self._summarize_anomaly_windows(anomaly_windows)
        signal_ids = self._extract_signal_ids(telemetry_summary)
        telemetry_support = self._telemetry_support_score(telemetry_summary)
        operator_family = self._infer_operator_family(event_start, event_end, anomaly_points)

        patterns: List[JsonDict] = []
        for fm in kg_context.get("failure_modes", []) or []:
            pattern = self._score_failure_mode_pattern(
                event_id=event_id,
                asset_id=asset_id,
                event_start=event_start,
                event_end=event_end,
                anomaly_points=anomaly_points,
                anomaly_windows=anomaly_windows,
                anomaly_window_summary=anomaly_window_summary,
                signal_ids=signal_ids,
                telemetry_support=telemetry_support,
                operator_family=operator_family,
                fm=fm,
                past_events=kg_context.get("past_events") or [],
            )
            patterns.append(pattern)

        patterns.sort(key=lambda p: (-float(p.get("confidence") or 0.0), p.get("target_id") or ""))

        supported = [p for p in patterns if float(p.get("confidence") or 0.0) >= self.config.min_confidence_for_support]
        avg_conf = (
            sum(float(p.get("confidence") or 0.0) for p in patterns) / len(patterns)
            if patterns else 0.0
        )
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
                "anomaly_point_count": len(anomaly_points),
                "signal_count": len(signal_ids),
                "avg_confidence": round(avg_conf, 4),
                "top_supported_targets": [
                    p.get("target_id")
                    for p in supported[:3]
                    if p.get("target_id")
                ],
            },
            "provenance": {
                "generated_by": "TSKRTemporalScorerV1",
                "run_id": run_context.get("run_id"),
                "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            },
        }

    def _extract_anomaly_points(self, telemetry_summary: JsonDict) -> List[datetime]:
        points: List[datetime] = []
        for sig in telemetry_summary.get("signals", []) or []:
            for a in sig.get("anomalies", []) or []:
                dt = (
                    parse_dt(a.get("timestamp_start"))
                    or parse_dt(a.get("timestamp_end"))
                    or parse_dt(a.get("t_detection"))
                )
                if dt is not None:
                    points.append(dt)
        points.sort()
        return points

    def _extract_anomaly_windows(self, telemetry_summary: JsonDict) -> List[Dict[str, Any]]:
        windows: List[Dict[str, Any]] = []
        for sig in telemetry_summary.get("signals", []) or []:
            sensor_id = sig.get("sensor_id")
            for a in sig.get("anomalies", []) or []:
                start = parse_dt(a.get("timestamp_start"))
                end = parse_dt(a.get("timestamp_end")) or start
                if start is None:
                    continue
                windows.append(
                    {
                        "sensor_id": sensor_id,
                        "start": start,
                        "end": end,
                        "pattern": a.get("pattern"),
                        "severity": a.get("severity"),
                        "raw_score": a.get("score"),
                    }
                )
        windows.sort(key=lambda x: x["start"])
        return windows

    def _summarize_anomaly_windows(
        self,
        anomaly_windows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not anomaly_windows:
            return {
                "window_start": None,
                "window_end": None,
                "earliest_start": None,
                "latest_end": None,
                "duration_hours": None,
            }

        earliest_start = anomaly_windows[0]["start"]
        latest_end = anomaly_windows[-1]["end"] or anomaly_windows[-1]["start"]
        duration_hours = (latest_end - earliest_start).total_seconds() / 3600.0

        return {
            "window_start": earliest_start,
            "window_end": latest_end,
            "earliest_start": earliest_start,
            "latest_end": latest_end,
            "duration_hours": round(duration_hours, 4),
        }
    
    def _extract_signal_ids(self, telemetry_summary: JsonDict) -> List[str]:
        ids: List[str] = []
        for sig in telemetry_summary.get("signals", []) or []:
            if not isinstance(sig, dict):
                continue
            sensor_id = sig.get("sensor_id")
            anomalies = sig.get("anomalies", []) or []
            if sensor_id and anomalies:
                ids.append(sensor_id)
        return ids

    def _telemetry_support_score(self, telemetry_summary: JsonDict) -> float:
        total = 0.0
        count = 0
        for sig in telemetry_summary.get("signals", []) or []:
            anomalies = sig.get("anomalies", []) or []
            for a in anomalies:
                sev = str(a.get("severity") or "").lower()
                score = a.get("score")
                base = self.config.telemetry_support_floor
                if sev == "high":
                    base = 0.9
                elif sev == "medium":
                    base = 0.7
                elif sev == "low":
                    base = 0.5
                if isinstance(score, (int, float)):
                    base = max(base, min(1.0, float(score)))
                total += base
                count += 1
        if total == 0.0:
            return 0.0
        return clamp01(total / max(1.0, count))

    def _anomaly_count_score(self, anomaly_points: List[datetime]) -> float:
        if not anomaly_points:
            return 0.0
        if len(anomaly_points) == 1:
            return 0.5
        if len(anomaly_points) == 2:
            return 0.7
        if len(anomaly_points) == 3:
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
        anomaly_points: List[datetime],
    ) -> Optional[str]:
        if event_start and event_end and anomaly_points:
            return "interval_point"
        if event_start and event_end:
            return "interval_interval"
        if anomaly_points:
            return "point_point"
        return None

    def _score_failure_mode_pattern(
        self,
        *,
        event_id: str,
        asset_id: str,
        event_start: Optional[datetime],
        event_end: Optional[datetime],
        anomaly_points: List[datetime],
        anomaly_windows: List[JsonDict],
        anomaly_window_summary: JsonDict,
        signal_ids: List[str],
        telemetry_support: float,
        operator_family: Optional[str],
        fm: JsonDict,
        past_events: List[JsonDict],
    ) -> JsonDict:
        
        fm_id = fm.get("fm_id")
        component_id = fm.get("component_id")

        relation, mean_lag, std_lag, anomaly_score = self._score_against_anomalies(
            event_start=event_start,
            event_end=event_end,
            anomaly_points=anomaly_points,
        )

        anomaly_count_score = self._anomaly_count_score(anomaly_points)
        lag_consistency_score = self._lag_consistency_score(std_lag)

        latency_details = self._latency_alignment_details(
            mean_lag_hours=mean_lag,
            expected_min=fm.get("expected_latency_min_hours"),
            expected_max=fm.get("expected_latency_max_hours"),
        )
        latency_score = latency_details["latency_alignment_score"]

        temporal_contradiction = (
            relation == "follows"
            or latency_details["latency_violation_type"] in {"too_fast", "too_slow"}
        )

        history_score = self._score_history_support(
            fm_id=fm_id,
            component_id=component_id,
            past_events=past_events,
        )

        confidence = clamp01(
            self.config.anomaly_weight * max(anomaly_score, telemetry_support)
            + self.config.latency_weight * latency_score
            + self.config.history_weight * history_score
            + self.config.anomaly_count_weight * anomaly_count_score
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
            "pattern_id": f"TSKR::{fm_id}",
            "event_id": event_id,
            "asset_id": asset_id,
            "target_type": "failure_mode",
            "target_id": fm_id,
            "component_id": component_id,
            "relation": relation,
            "operator_family": operator_family,
            "mean_lag_hours": round(mean_lag, 4) if mean_lag is not None else None,
            "std_lag_hours": round(std_lag, 4) if std_lag is not None else None,
            "support": round(support, 4),
            "confidence": round(confidence, 4),
            "matching_signal_ids": signal_ids,
            "anomaly_count": len(anomaly_points),
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
            "observed_lag_hours": latency_details["observed_lag_hours"],
            "latency_alignment_score": latency_details["latency_alignment_score"],
            "latency_violation_type": latency_details["latency_violation_type"],
            "temporal_contradiction": temporal_contradiction,
        }

    def _score_against_anomalies(
        self,
        *,
        event_start: Optional[datetime],
        event_end: Optional[datetime],
        anomaly_points: List[datetime],
    ) -> Tuple[str, Optional[float], Optional[float], float]:
        if not anomaly_points or event_start is None:
            return "unknown", None, None, self.config.fallback_confidence

        lags: List[float] = []
        simultaneous_hits = 0

        for pt in anomaly_points:
            lag_h = (event_start - pt).total_seconds() / 3600.0
            lags.append(lag_h)

            if event_end is not None:
                if event_start <= pt <= event_end:
                    simultaneous_hits += 1
            else:
                if abs(lag_h) <= self.config.simultaneous_epsilon_hours:
                    simultaneous_hits += 1

        mean_lag = sum(lags) / len(lags)
        std_lag = math.sqrt(sum((x - mean_lag) ** 2 for x in lags) / len(lags)) if lags else None

        if simultaneous_hits > 0:
            relation = "simultaneous"
            anomaly_score = 0.85
        elif mean_lag is not None and mean_lag >= 0:
            relation = "precedes"
            anomaly_score = 0.75
        elif mean_lag is not None and mean_lag < 0:
            relation = "follows"
            anomaly_score = 0.35
        else:
            relation = "unknown"
            anomaly_score = self.config.fallback_confidence

        return relation, mean_lag, std_lag, anomaly_score

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

        lag = abs(float(mean_lag_hours))
        score = self._score_expected_latency(
            mean_lag_hours=lag,
            expected_min=mn,
            expected_max=mx,
        )

        violation = "none"
        if mn is not None and lag < mn:
            violation = "too_fast"
        elif mx is not None and lag > mx:
            violation = "too_slow"

        return {
            "expected_min_hours": mn,
            "expected_max_hours": mx,
            "observed_lag_hours": round(lag, 4),
            "latency_alignment_score": round(float(score), 4),
            "latency_violation_type": violation,
        }
    
    def _score_history_support(
        self,
        *,
        fm_id: Optional[str],
        component_id: Optional[str],
        past_events: List[JsonDict],
    ) -> float:
        if not past_events:
            return 0.0

        hits = 0.0
        total = 0.0
        for pe in past_events:
            total += 1.0
            matched_fms = set(pe.get("matched_failure_mode_ids") or [])
            matched_component = pe.get("component_id")
            if fm_id and fm_id in matched_fms:
                hits += 1.0
            elif component_id and matched_component and matched_component == component_id:
                hits += 0.5

        if total == 0:
            return 0.0
        return clamp01(hits / total)