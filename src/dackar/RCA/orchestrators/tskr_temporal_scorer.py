from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)

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
    anomaly_weight: float = 0.45
    latency_weight: float = 0.30
    history_weight: float = 0.10
    chain_weight: float = 0.10
    min_confidence_for_support: float = 0.35
    anomaly_count_weight: float = 0.15
    lag_consistency_weight: float = 0.10
    telemetry_support_floor: float = 0.35
    tone_vocabulary_version: str = "npp_tone_v1"
    tone_transient_max_minutes: float = 5.0
    tone_watch_min_minutes: float = 5.0
    tone_alert_min_minutes: float = 10.0
    tone_trip_min_minutes: float = 2.0
    # Semantic recurrence parameters (§4.3 / §4.5)
    enable_semantic_recurrence: bool = False
    semantic_similarity_threshold: float = 0.75
    near_match_window: float = 0.10
    top_k_semantic: int = 5


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
    contributing_event_ids: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.contributing_event_ids is None:
            self.contributing_event_ids = []


class TSKRTemporalScorerV1:
    _INSTRUMENT_VALIDITY_MULTIPLIER: Dict[str, float] = {
        "calibrated": 1.00,
        "valid": 1.00,
        "in_calibration": 1.00,
        "in_cal": 1.00,
        "ok": 1.00,
        "pass": 1.00,
        "unknown": 0.85,
        "under_investigation": 0.80,
        "unchecked": 0.80,
        "not_available": 0.80,
        "out_of_calibration": 0.55,
        "out_of_cal": 0.55,
        "degraded": 0.65,
        "invalid": 0.55,
        "failed": 0.50,
        "faulted": 0.50,
        "suspect": 0.60,
    }

    """Deterministic temporal scorer using interval-based Allen relations.

    Inputs:
      - event interval (timestamp_start / timestamp_end)
      - telemetry anomaly windows (start, end, severity, pattern)
      - kg_context.failure_modes[*].expected_latency_min_hours/max_hours
      - kg_context.past_events for recurrence analysis

    Output:
      - tskr_patterns artifact aligned to tskr_patterns.json schema
    """

    def __init__(
        self,
        config: Optional[TSKRTemporalScorerConfig] = None,
        doc_extraction_store: Optional[Any] = None,
    ) -> None:
        self.config = config or TSKRTemporalScorerConfig()
        self.doc_extraction_store = doc_extraction_store

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
        signal_evidence: Optional[JsonDict] = None,
        alarm_log: Optional[JsonDict] = None,
        soe_log: Optional[JsonDict] = None,
        pm_compliance: Optional[JsonDict] = None,
    ) -> JsonDict:
        event_id = event.get("event_id") or event.get("id")
        asset_id = event.get("asset_id")
        event_start = parse_dt(event.get("timestamp_start"))
        event_end   = parse_dt(event.get("timestamp_end")) or event_start

        telemetry_windows = (
            self._extract_anomaly_windows_from_signal_evidence(signal_evidence)
            if signal_evidence and int(signal_evidence.get("augmented_anomaly_count", 0) or 0) > 0
            else self._extract_anomaly_windows(telemetry_summary)
        )
        # Alarm and SOE windows are kept separate — they feed onset timing only,
        # not anomaly_score or anomaly_count_score (Phase B restriction).
        alarm_soe_windows: List[Dict[str, Any]] = []
        if alarm_log:
            alarm_soe_windows = alarm_soe_windows + self._extract_alarm_windows(alarm_log)
        if soe_log:
            alarm_soe_windows = alarm_soe_windows + self._extract_soe_windows(soe_log)
        all_windows = sorted(telemetry_windows + alarm_soe_windows, key=lambda x: x["start"])
        anomaly_window_summary = self._summarize_anomaly_windows(all_windows)
        signal_ids           = self._extract_signal_ids(telemetry_summary)
        telemetry_support    = self._telemetry_support_score(telemetry_windows)
        tone_summary         = self._summarize_tones(all_windows)
        operator_family      = self._infer_operator_family(event_start, event_end, all_windows)

        past_events = self._normalize_past_events(kg_context.get("past_events") or [])
        stage_b_allen_by_component = self._stage_b_allen_relation_by_component(kg_context)
        chain_scores = (signal_evidence or {}).get("per_candidate_chain_score") or {}
        patterns: List[JsonDict] = []
        for fm in kg_context.get("failure_modes", []) or []:
            fm_component_id = fm.get("component_id") or fm.get("applies_to_component_id")
            fm_chain = chain_scores.get(str(fm.get("fm_id") or ""), {}) if isinstance(chain_scores, dict) else {}
            fm_signal_ids = self._extract_signal_ids_for_fm(telemetry_summary, fm)
            pattern = self._score_failure_mode_pattern(
                event_id=event_id,
                asset_id=asset_id,
                event_start=event_start,
                event_end=event_end,
                anomaly_windows=telemetry_windows,
                alarm_soe_windows=alarm_soe_windows,
                anomaly_window_summary=anomaly_window_summary,
                signal_ids=signal_ids,
                fm_signal_ids=fm_signal_ids,
                telemetry_support=telemetry_support,
                operator_family=operator_family,
                fm=fm,
                past_events=past_events,
                stage_b_allen_relation=stage_b_allen_by_component.get(str(fm_component_id or "")),
                chain_position_score=float(fm_chain.get("chain_position_score", 0.0) or 0.0),
                chain_position_type=str(fm_chain.get("position_type") or "absent"),
                contributing_cause_role=fm_chain.get("contributing_cause_role"),
                confluence_component_id=fm_chain.get("confluence_component_id"),
                pm_compliance=pm_compliance,
            )
            patterns.append(pattern)

        patterns.sort(key=lambda p: (-float(p.get("confidence") or 0.0), p.get("target_id") or ""))

        supported = [p for p in patterns if float(p.get("confidence") or 0.0) >= self.config.min_confidence_for_support]
        avg_conf = (
            sum(float(p.get("confidence") or 0.0) for p in patterns) / len(patterns)
            if patterns else 0.0
        )
        novel_count = sum(1 for p in patterns if p.get("novel_pattern", False))
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
                "n_novel_patterns": novel_count,
                "has_novel_patterns": novel_count > 0,
                "operator_family": operator_family,
                "anomaly_point_count": len(all_windows),
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
            instrument_validity_flag = sig.get("instrument_validity_flag")
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
                    "instrument_validity_flag": instrument_validity_flag,
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

    def _extract_anomaly_windows_from_signal_evidence(
        self,
        signal_evidence: Optional[JsonDict],
    ) -> List[Dict[str, Any]]:
        windows: List[Dict[str, Any]] = []
        for row in (signal_evidence or {}).get("augmented_anomaly_set", []) or []:
            if not isinstance(row, dict):
                continue
            start = parse_dt(row.get("timestamp_start"))
            end = parse_dt(row.get("timestamp_end")) or start
            if start is None:
                continue
            windows.append(
                {
                    "sensor_id": row.get("sensor_id"),
                    "start": start,
                    "end": end,
                    "pattern": row.get("pattern"),
                    "interval_type": self._normalize_interval_type(row.get("interval_type")),
                    "severity": row.get("severity"),
                    "instrument_validity_flag": row.get("instrument_validity_flag"),
                    "severity_score": row.get("severity"),
                    "tone": self._classify_tone(
                        start=start,
                        end=end,
                        severity=row.get("severity"),
                        severity_score=row.get("severity"),
                    ),
                }
            )
        windows.sort(key=lambda x: x["start"])
        return windows

    @staticmethod
    def _extract_alarm_windows(alarm_log: Optional[JsonDict]) -> List[Dict[str, Any]]:
        """Extract point-event windows from alarm_log.alarms[] for pattern matching.

        Alarms without an ``acknowledged_at`` are treated as point events (end == start).
        Clock-sync failures mark windows as low-severity so they contribute less to scoring.
        """
        windows: List[Dict[str, Any]] = []
        if not isinstance(alarm_log, dict):
            return windows
        clock_ok = (alarm_log.get("quality") or {}).get("clock_sync_ok")
        for alm in (alarm_log.get("alarms") or []):
            if not isinstance(alm, dict):
                continue
            ts = parse_dt(alm.get("activated_at") or alm.get("timestamp"))
            if ts is None:
                continue
            end_raw = alm.get("acknowledged_at") or alm.get("cleared_at")
            end = parse_dt(end_raw) if end_raw else ts
            if end is None:
                end = ts
            severity = alm.get("severity") or ("DEGRADED" if clock_ok is False else "MEDIUM")
            windows.append({
                "sensor_id": alm.get("alarm_id") or alm.get("tag"),
                "start": ts,
                "end": end,
                "pattern": "alarm_activation",
                "interval_type": "closed",
                "severity": severity,
                "instrument_validity_flag": None if clock_ok is not False else "clock_sync_failed",
                "severity_score": 0.4 if clock_ok is False else None,
                "tone": "degraded" if clock_ok is False else "suspect",
                "source_type": "alarm",
            })
        windows.sort(key=lambda x: x["start"])
        return windows

    @staticmethod
    def _extract_soe_windows(soe_log: Optional[JsonDict]) -> List[Dict[str, Any]]:
        """Extract point-event windows from soe_log.records[] for pattern matching.

        All SOE records are point events (end == start).  Clock-sync failure
        marks windows as unreliable; they are still included but flagged.
        """
        windows: List[Dict[str, Any]] = []
        if not isinstance(soe_log, dict):
            return windows
        clock_ok = (soe_log.get("quality") or {}).get("clock_sync_ok")
        for rec in (soe_log.get("records") or []):
            if not isinstance(rec, dict):
                continue
            ts = parse_dt(rec.get("timestamp"))
            if ts is None:
                continue
            windows.append({
                "sensor_id": rec.get("record_id") or rec.get("tag"),
                "start": ts,
                "end": ts,
                "pattern": rec.get("transition") or rec.get("state_change") or "soe_transition",
                "interval_type": "closed",
                "severity": "HIGH" if rec.get("is_protection_signal") else "MEDIUM",
                "instrument_validity_flag": None if clock_ok is not False else "clock_sync_failed",
                "severity_score": 0.3 if clock_ok is False else (0.85 if rec.get("is_protection_signal") else 0.5),
                "tone": "degraded" if clock_ok is False else ("failed" if rec.get("is_protection_signal") else "suspect"),
                "source_type": "soe",
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

    @staticmethod
    def _normalize_past_events(past_events: List[JsonDict]) -> List[JsonDict]:
        """Remap alternate field names produced by different KG schema versions.

        Some KG exports use ``related_failure_modes`` / ``occurred_at`` instead of
        the canonical ``matched_failure_mode_ids`` / ``timestamp_start``.  This
        normalizer makes both schemas transparent to the rest of the scorer without
        mutating caller-owned dicts.
        """
        normalized = []
        for pe in past_events:
            if not isinstance(pe, dict):
                normalized.append(pe)
                continue
            needs_copy = (
                ("related_failure_modes" in pe and "matched_failure_mode_ids" not in pe)
                or ("occurred_at" in pe and "timestamp_start" not in pe)
            )
            if needs_copy:
                pe = dict(pe)
                if "related_failure_modes" in pe and "matched_failure_mode_ids" not in pe:
                    pe["matched_failure_mode_ids"] = pe["related_failure_modes"]
                if "occurred_at" in pe and "timestamp_start" not in pe:
                    pe["timestamp_start"] = pe["occurred_at"]
            normalized.append(pe)
        return normalized

    @classmethod
    def _stage_b_allen_relation_by_component(cls, kg_context: JsonDict) -> Dict[str, str]:
        # priority_rank: lower index = higher causal priority
        priority_rank = {r: i for i, r in enumerate(CAUSAL_PRIORITY)}
        mapping: Dict[str, str] = {}
        for row in (kg_context.get("out_of_boundary_anomalies") or []):
            if not isinstance(row, dict):
                continue
            comp = str(row.get("component_id") or row.get("related_component_id") or "").strip()
            relation = str(row.get("allen_relation") or "").strip().lower()
            if not comp or not relation:
                continue
            existing = mapping.get(comp)
            if existing is None:
                mapping[comp] = relation
            else:
                # Keep whichever relation has higher causal priority (lower rank index).
                # Unknown relations (not in priority_rank) are always displaced.
                existing_rank = priority_rank.get(existing, len(CAUSAL_PRIORITY))
                new_rank      = priority_rank.get(relation, len(CAUSAL_PRIORITY))
                if new_rank < existing_rank:
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

    def _extract_signal_ids_for_fm(
        self, telemetry_summary: JsonDict, fm: JsonDict
    ) -> List[str]:
        """Return sensor IDs with anomalies that are relevant to this FM.

        Relevance is determined by matching the signal's ``parameter`` field
        against the FM's ``expected_symptom_types``.  When the FM carries no
        symptom type information the method falls back to returning all
        anomalous sensor IDs (same behaviour as the global extractor).
        """
        symptom_types = {
            str(t).lower()
            for t in (fm.get("expected_symptom_types") or [])
            if t
        }
        ids: List[str] = []
        for sig in telemetry_summary.get("signals", []) or []:
            if not isinstance(sig, dict) or not sig.get("anomalies"):
                continue
            sensor_id = sig.get("sensor_id")
            if not sensor_id:
                continue
            if not symptom_types:
                ids.append(sensor_id)
            elif str(sig.get("parameter") or "").lower() in symptom_types:
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
            base = max(0.1, float(raw))
        else:
            sev = str(window.get("severity") or "").lower()
            base = {"high": 0.9, "medium": 0.7, "low": 0.5}.get(sev, 0.5)
        validity_mult = self._instrument_validity_multiplier(window.get("instrument_validity_flag"))
        return max(0.1, base * validity_mult)

    def _instrument_validity_multiplier(self, value: Any) -> float:
        key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            return 1.0
        return float(self._INSTRUMENT_VALIDITY_MULTIPLIER.get(key, 0.85))

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

    @staticmethod
    def _normalized_weighted_sum(components: List[Tuple[float, float]]) -> float:
        """Return normalized weighted average in [0, 1].

        This keeps confidence convex even when configured weights do not sum to 1.
        """
        if not components:
            return 0.0
        weighted_sum = 0.0
        total_weight = 0.0
        for score, weight in components:
            w = max(0.0, float(weight))
            if w <= 0.0:
                continue
            weighted_sum += clamp01(score) * w
            total_weight += w
        if total_weight <= 0.0:
            return 0.0
        return clamp01(weighted_sum / total_weight)

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
        event_start: Optional[datetime] = None,
    ) -> RecurrenceProfile:
        """Build a RecurrenceProfile by matching past_events to the given
        failure mode / component and deriving recurrence statistics."""
        matching = []
        for pe in past_events:
            matched_fms  = set(pe.get("matched_failure_mode_ids") or [])
            matched_comp = pe.get("component_id")
            fm_matched = bool(fm_id and fm_id in matched_fms)
            # Fall back to component-level match only when the past event carries no
            # FM attribution at all — prevents inflating scores when multiple FMs
            # share the same component but only one is responsible.
            comp_fallback = bool(component_id and matched_comp == component_id and not matched_fms)
            if fm_matched or comp_fallback:
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

        # resolved == False (explicit False, not None) counts as unresolved.
        # Computed from the full matching set (not just dated) so count and
        # unresolved_count share the same denominator.
        unresolved_count = sum(1 for pe in matching if pe.get("resolved") is False)

        # Recency: prefer computing from actual timestamps so the value is relative
        # to the current event rather than a stale KG-snapshot field.
        most_recent_days_ago: Optional[int] = None
        if dated:
            most_recent_ts = dated[-1][0]
            if event_start is not None:
                # Normalize timezone awareness so subtraction never raises.
                es = event_start
                mr = most_recent_ts
                if es.tzinfo is None and mr.tzinfo is not None:
                    from datetime import timezone as _tz
                    es = es.replace(tzinfo=_tz.utc)
                elif es.tzinfo is not None and mr.tzinfo is None:
                    from datetime import timezone as _tz
                    mr = mr.replace(tzinfo=_tz.utc)
                delta = (es - mr).total_seconds() / 86400.0
                most_recent_days_ago = max(0, int(delta))
            else:
                td = dated[-1][1].get("time_distance_days")
                if isinstance(td, (int, float)):
                    most_recent_days_ago = int(td)

        # Collect IDs from all matched events for downstream traceability.
        contributing_event_ids: List[str] = []
        for pe in matching:
            for field in ("event_id", "source_doc_id", "cr_id", "wo_id"):
                val = pe.get(field)
                if val and str(val) not in contributing_event_ids:
                    contributing_event_ids.append(str(val))
                    break

        return RecurrenceProfile(
            fm_id=fm_id,
            component_id=component_id,
            count=count,
            mean_inter_event_days=mean_inter_event_days,
            trend=self._recurrence_trend(intervals_days),
            unresolved_count=unresolved_count,
            most_recent_days_ago=most_recent_days_ago,
            contributing_event_ids=contributing_event_ids,
        )

    @staticmethod
    def _recurrence_trend(intervals: List[float]) -> str:
        """Fit an OLS line to the inter-event interval sequence and classify the trend.

        A negative slope means intervals are shrinking (events accelerating) →
        "increasing" recurrence rate, the stronger causal signal.  The slope is
        normalised by the mean interval so the threshold is scale-independent.
        Requires at least 3 intervals (4 events) for a meaningful fit.
        """
        n = len(intervals)
        if n < 3:
            return "insufficient_data"
        mean_y = sum(intervals) / n
        if mean_y <= 0:
            return "insufficient_data"
        # OLS slope: cov(x, y) / var(x) with x = 0..n-1
        mean_x = (n - 1) / 2.0
        cov_xy = sum((i - mean_x) * (intervals[i] - mean_y) for i in range(n))
        var_x  = sum((i - mean_x) ** 2 for i in range(n))
        if var_x == 0:
            return "stable"
        slope = cov_xy / var_x
        # Normalise by mean interval → relative change per step
        norm_slope = slope / mean_y
        if norm_slope < -0.10:
            return "increasing"   # intervals shrinking → accelerating recurrence
        if norm_slope > 0.10:
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

    def _score_from_effective_count(
        self, effective_count: float, profile: RecurrenceProfile
    ) -> float:
        """Score from effective (exact + semantic) recurrence count (§4.3).

        Uses floor(effective_count) for threshold bracketing so that fractional
        semantic contributions require at least 0.5 cumulative weight to cross
        into the next tier.
        """
        count_floor = int(effective_count)
        if count_floor == 0:
            base = 0.0
        elif count_floor == 1:
            base = 0.35
        elif count_floor <= 3:
            base = 0.55
        elif count_floor <= 6:
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
        event_start: Optional[datetime] = None,
    ) -> Tuple[float, RecurrenceProfile]:
        profile = self._build_recurrence_profile(
            fm_id=fm_id, component_id=component_id, past_events=past_events,
            event_start=event_start,
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
        alarm_soe_windows: Optional[List[JsonDict]] = None,
        anomaly_window_summary: JsonDict,
        signal_ids: List[str],
        fm_signal_ids: Optional[List[str]] = None,
        telemetry_support: float,
        operator_family: Optional[str],
        fm: JsonDict,
        past_events: List[JsonDict],
        stage_b_allen_relation: Optional[str] = None,
        chain_position_score: float = 0.0,
        chain_position_type: str = "absent",
        contributing_cause_role: Optional[str] = None,
        confluence_component_id: Optional[str] = None,
        pm_compliance: Optional[JsonDict] = None,
    ) -> JsonDict:
        # Per-FM signal IDs: filtered by FM's expected_symptom_types when available;
        # falls back to the global signal_ids list when no FM-level filter is provided.
        effective_signal_ids = fm_signal_ids if fm_signal_ids is not None else signal_ids

        fm_id        = fm.get("fm_id")
        component_id = fm.get("component_id")

        # anomaly_score and anomaly_count_score: telemetry windows only
        relation, mean_lag, std_lag, anomaly_score = self._score_against_anomalies(
            event_start=event_start,
            event_end=event_end,
            anomaly_windows=anomaly_windows,
        )

        effective_count      = self._effective_anomaly_count(anomaly_windows)
        anomaly_count_score  = self._anomaly_count_score(effective_count)

        # lag_consistency_score: include alarm/SOE onset timing (Phase B)
        if alarm_soe_windows:
            _timing_windows = sorted(anomaly_windows + alarm_soe_windows, key=lambda x: x["start"])
            _, _, _std_lag_all, _ = self._score_against_anomalies(
                event_start=event_start,
                event_end=event_end,
                anomaly_windows=_timing_windows,
            )
            lag_consistency_score = self._lag_consistency_score(_std_lag_all)
        else:
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
            event_start=event_start,
        )

        # Semantic recurrence augmentation (§4.3)
        semantic_match_count = 0
        near_match_count = 0
        effective_recurrence_count: float = float(recurrence_profile.count)
        near_match_pattern = False
        semantic_recurrence_capped = False
        fm_resolution_ambiguous = False

        # Collect exact doc IDs from past_events for the double-counting guard.
        # Past events carry source doc references under common field names; if none
        # are present yet, the guard is a no-op (empty set) until past_events are enriched.
        exact_doc_ids: set = set()
        for _pe in (past_events or []):
            for _field in ("source_doc_id", "cr_id", "wo_id", "event_ref", "source_cr_id", "source_wo_id"):
                _val = _pe.get(_field)
                if _val:
                    exact_doc_ids.add(str(_val))
        exact_doc_ids_count = len(exact_doc_ids)

        if (
            self.doc_extraction_store is not None
            and self.config.enable_semantic_recurrence
        ):
            fm_name = fm.get("name") or fm.get("label") or ""
            fm_symptoms = fm.get("expected_symptoms") or ""
            query_text = " | ".join(t for t in (fm_name, fm_symptoms) if t)
            if query_text.strip():
                try:
                    sem_matches, sem_near = self.doc_extraction_store.query(
                        query_text,
                        top_k=self.config.top_k_semantic,
                        similarity_threshold=self.config.semantic_similarity_threshold,
                        near_match_window=self.config.near_match_window,
                        exact_doc_ids=exact_doc_ids,
                    )
                    semantic_contributions = sum(m.semantic_contribution for m in sem_matches)
                    effective_recurrence_count = recurrence_profile.count + semantic_contributions
                    semantic_match_count = len(sem_matches)
                    near_match_count = len(sem_near)

                    # Tier cap: when there are no exact matches, semantic contributions
                    # must not elevate the effective count into tier-1 (floor >= 1).
                    # Cap to 0.99 so _score_from_effective_count stays at base = 0.0.
                    if recurrence_profile.count == 0 and effective_recurrence_count >= 1.0:
                        effective_recurrence_count = 0.99
                        semantic_recurrence_capped = True

                    # Re-score history using the (possibly capped) effective count
                    if semantic_contributions > 0:
                        history_score = self._score_from_effective_count(
                            effective_recurrence_count, recurrence_profile
                        )
                    near_match_pattern = (
                        recurrence_profile.count == 0
                        and not sem_matches
                        and bool(sem_near)
                    )
                    fm_resolution_ambiguous = any(
                        getattr(m, "fm_resolution_status", None) == "ambiguous"
                        for m in sem_matches
                    )
                except Exception as exc:
                    logger.warning(
                        "DocExtractionStore query failed for fm %s: %s — semantic recurrence skipped",
                        fm_id, exc,
                    )

        # PM overdue boost: overdue maintenance on this component is a latent
        # contributor to failure recurrence; apply a small additive boost to
        # history_score (+0.05 per overdue item, capped at +0.15).
        pm_overdue_boost = 0.0
        if pm_compliance and component_id:
            overdue_items = pm_compliance.get("overdue_items") or pm_compliance.get("overdue_tasks") or []
            matching_overdue = [
                item for item in overdue_items
                if isinstance(item, dict) and item.get("component_id") == component_id
            ]
            pm_overdue_boost = min(0.15, 0.05 * len(matching_overdue))
            if pm_overdue_boost > 0:
                history_score = clamp01(history_score + pm_overdue_boost)

        chain_pos = float(chain_position_score or 0.0)
        if str(chain_position_type or "absent") == "convergence_confluence":
            chain_pos = 0.0

        # Phase B intermediates: monitors-class vs analyzes-class sub-scores
        signal_support_score = self._normalized_weighted_sum([
            (max(anomaly_score, telemetry_support), self.config.anomaly_weight),
            (latency_score, self.config.latency_weight),
            (chain_pos, self.config.chain_weight),
            (anomaly_count_score, self.config.anomaly_count_weight),
            (lag_consistency_score, self.config.lag_consistency_weight),
        ])
        recurrence_support_score = clamp01(history_score)

        confidence_base = self._normalized_weighted_sum([
            (max(anomaly_score, telemetry_support), self.config.anomaly_weight),
            (latency_score, self.config.latency_weight),
            (chain_pos, self.config.chain_weight),
            (history_score, self.config.history_weight),
            (anomaly_count_score, self.config.anomaly_count_weight),
            (lag_consistency_score, self.config.lag_consistency_weight),
        ])
        confidence = clamp01(confidence_base - (0.20 if temporal_contradiction else 0.0))

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
            "signal_support_score":     round(signal_support_score, 4),
            "recurrence_support_score": round(recurrence_support_score, 4),
            "matching_signal_ids": effective_signal_ids,
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
            "chain_position_score":       round(chain_pos, 4),
            "chain_position_type":        chain_position_type,
            "contributing_cause_role":    contributing_cause_role,
            "confluence_component_id":    confluence_component_id,
            "temporal_contradiction": temporal_contradiction,
            "stage_b_allen_relation": stage_b_allen_relation,
            "stage_b_temporal_contradiction": stage_b_temporal_contradiction,
            # Recurrence fields
            "recurrence_count":            recurrence_profile.count,
            "effective_recurrence_count":  round(effective_recurrence_count, 4),
            "recurrence_trend":            recurrence_profile.trend,
            "unresolved_recurrence_count": recurrence_profile.unresolved_count,
            "exact_doc_ids_count":         exact_doc_ids_count,
            "contributing_event_ids":      recurrence_profile.contributing_event_ids,
            "semantic_match_count":        semantic_match_count,
            "semantic_doc_ids_count":      semantic_match_count,
            "near_match_count":            near_match_count,
            "semantic_recurrence_capped":  semantic_recurrence_capped,
            "fm_resolution_ambiguous":     fm_resolution_ambiguous,
            # Step 3.5 — novel pattern decomposition (Issue 6 / Phase 2).
            # documentary_novel: no CR/WO history regardless of signal alignment.
            # signal_novel:      no signal IDs map to this failure mode.
            # novel_pattern:     retained for backward compat (AND of both conditions).
            # The dangerous case — documentary_novel=True, signal_novel=False — means
            # the equipment is showing a known signal pattern with no prior RCA record.
            "documentary_novel": bool(
                effective_recurrence_count == 0
                and history_score < 0.20
            ),
            "signal_novel": not bool(effective_signal_ids),
            "novel_pattern": bool(
                effective_recurrence_count == 0
                and history_score < 0.20
                and not bool(effective_signal_ids)
            ),
            "near_match_pattern": near_match_pattern,
            "pm_overdue_boost": round(pm_overdue_boost, 4),
            "attention_flags": (
                ["accelerating_recurrence"] if recurrence_profile.trend == "increasing" else []
            ),
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
                "latency_violation_type": "not_available",
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
                "latency_violation_type": "not_available",
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
