"""
Stage C — Temporal Event Chain Scorer.

Responsibilities:
    1. Parse the emergent activity's detection timestamp (and duration if known)
       into a temporal interval [start, end].
    2. For each prior event in the ComponentEventTimeline, apply Allen interval
       algebra to classify the temporal relationship to the emergent activity:
       PRECEDES, OVERLAPS, CONTAINS, DURING, FOLLOWS, SIMULTANEOUS, UNKNOWN.
    3. Score each link for causal relevance based on the Allen relation, onset
       lag, and data quality of the prior event.
    4. Label each link with a causal_strength: strong / moderate / weak /
       temporal_contradiction.
    5. Summarise the chain: strongest link, causal posture, contradiction flags.

Output schema: outage/schemas/temporal_event_chain.json

Allen relation → causal relevance reference:
    OVERLAPS    → 0.90  prior event was active at emergent activity onset
    CONTAINS    → 0.85  long-running degradation encompasses the activity window
    PRECEDES    → 0.75  classic lead-time: ended before onset
    SIMULTANEOUS→ 0.50  concurrent; possible common cause
    DURING      → 0.30  started after onset; likely a symptom, not a cause
    FOLLOWS     → 0.10  temporal contradiction; flag for analyst review

Reuse targets:
    RCA.orchestrators.tskr_temporal_scorer.TSKRTemporalScorerV1
        Adaptation: operate on CR/WO event intervals rather than telemetry
        anomaly windows. The Allen relation algebra and confidence scoring
        logic are unchanged; only the input data model differs.
    dackar.RCA.orchestrators.temporal_relations
        Interval dataclass and onset_lag_hours utility are imported directly.
        The allen_relation() function is NOT reused because Stage C extends
        the 5-relation RCA vocabulary with SIMULTANEOUS and UNKNOWN.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# Allen relation labels
_PRECEDES = "precedes"
_OVERLAPS = "overlaps"
_CONTAINS = "contains"
_DURING = "during"
_FOLLOWS = "follows"
_SIMULTANEOUS = "simultaneous"
_UNKNOWN = "unknown"

# Causal relevance scores per Allen relation (mirrors TSKR scoring)
_RELATION_SCORES: Dict[str, float] = {
    _OVERLAPS: 0.90,
    _CONTAINS: 0.85,
    _PRECEDES: 0.75,
    _SIMULTANEOUS: 0.50,
    _DURING: 0.30,
    _FOLLOWS: 0.10,
    _UNKNOWN: 0.00,
}

# Causal strength thresholds
_STRONG_THRESHOLD = 0.75
_MODERATE_THRESHOLD = 0.40

# Confidence factor weights — adapted from TSKRTemporalScorerV1
_ANOMALY_WEIGHT = 0.55   # data_quality_score of the prior event
_LATENCY_WEIGHT = 0.30   # onset lag plausibility
_RELATION_WEIGHT = 0.15  # base Allen relation score

# Reuse: Interval dataclass and onset_lag_hours from RCA (read-only import)
try:
    from dackar.RCA.orchestrators.temporal_relations import (  # noqa: F401
        Interval as _RCAInterval,
    )
    _RCA_TEMPORAL_AVAILABLE = True
except ImportError:
    _RCA_TEMPORAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_dt(iso_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string to a timezone-aware datetime.

    Naive datetimes are assumed UTC.  Returns None on invalid input.
    """
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


@dataclass
class TemporalChainConfig:
    """Configuration for Stage C."""

    epsilon_hours: float = 0.5
    """Tolerance for Allen relation boundary comparisons.  Events within
    epsilon_hours of a boundary are treated as simultaneous at that boundary."""

    include_follows_relations: bool = True
    """If True, FOLLOWS links are retained with causal_strength='temporal_contradiction'.
    If False, they are silently dropped from the chain."""

    min_relation_score_threshold: float = 0.0
    """Drop links with relation_score below this value. Default 0.0 (keep all)."""


class TemporalChainScorer:
    """Concrete Stage C implementation.

    Args:
        config: Stage configuration.
    """

    def __init__(self, config: Optional[TemporalChainConfig] = None) -> None:
        self.config = config or TemporalChainConfig()

    # ── Protocol method ───────────────────────────────────────────────────────

    def score(
        self,
        emergent_activity: JsonDict,
        component_event_timeline: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """Execute Stage C for one emergent activity.

        Returns:
            TemporalEventChain artifact conforming to
            outage/schemas/temporal_event_chain.json.
        """
        run_id: str = run_context["run_id"]
        activity_id: str = emergent_activity["activity_id"]
        component_id: str = component_event_timeline["component_id"]
        LOGGER.debug(
            "Stage C temporal chain scoring for %s (run=%s)", activity_id, run_id
        )

        activity_interval, is_point = self._parse_activity_interval(emergent_activity)
        events: List[JsonDict] = component_event_timeline.get("events", [])

        links: List[JsonDict] = []
        for event in events:
            link = self._score_event_link(
                event=event,
                activity_interval=activity_interval,
                is_point_event=is_point,
            )
            if link is None:
                continue
            if link["relation_score"] < self.config.min_relation_score_threshold:
                continue
            if (
                link["allen_relation"] == _FOLLOWS
                and not self.config.include_follows_relations
            ):
                continue
            links.append(link)

        summary = self._summarize_chain(links)

        return {
            "activity_id": activity_id,
            "run_id": run_id,
            "generated_at": run_context.get("started_at", ""),
            "component_id": component_id,
            "emergent_activity_interval": {
                "start": activity_interval[0],
                "end": activity_interval[1],
                "is_point_event": is_point,
            },
            "chain_links": links,
            "summary": summary,
            "provenance": {
                "generated_by": self.__class__.__name__,
                "run_id": run_id,
                "scorer_version": "v1",
                "epsilon_hours": self.config.epsilon_hours,
            },
        }

    # ── Private step methods ──────────────────────────────────────────────────

    def _parse_activity_interval(
        self, emergent_activity: JsonDict
    ) -> Tuple[Tuple[Optional[str], Optional[str]], bool]:
        """Extract the temporal interval of the emergent activity.

        Returns ((start_iso, end_iso), is_point_event).

        is_point_event=True when only a detection timestamp is known (no
        planned duration or actual start/finish), indicating the interval
        collapses to a point for Allen relation purposes.

        Priority for start: actual_start > detection_timestamp > planned_start.
        Priority for end: actual_finish > planned_finish > (start + planned_duration).
        """
        start_iso: Optional[str] = (
            emergent_activity.get("actual_start")
            or emergent_activity.get("detection_timestamp")
            or emergent_activity.get("planned_start")
        )
        end_iso: Optional[str] = (
            emergent_activity.get("actual_finish")
            or emergent_activity.get("planned_finish")
        )

        # Try to derive end from planned duration before deciding on is_point.
        # is_point_event=True only when truly no duration information is available.
        if not end_iso and start_iso:
            planned_hours = emergent_activity.get("planned_duration_hours")
            if planned_hours:
                try:
                    dt_start = _parse_dt(start_iso)
                    if dt_start is not None:
                        end_iso = (
                            dt_start + timedelta(hours=float(planned_hours))
                        ).isoformat()
                except (ValueError, TypeError):
                    pass

        # Collapse to point only if still no end can be determined
        is_point = not end_iso
        if not end_iso:
            end_iso = start_iso

        return (start_iso, end_iso), is_point

    def _score_event_link(
        self,
        event: JsonDict,
        activity_interval: Tuple[Optional[str], Optional[str]],
        is_point_event: bool,
    ) -> Optional[JsonDict]:
        """Score one prior event against the emergent activity interval.

        Returns a chain_link dict conforming to temporal_event_chain.json, or
        None if the event timestamp is missing and the relation cannot be
        determined.

        Steps:
            1. Parse event interval from event['timestamp'] and event duration.
            2. Compute Allen relation via _allen_relation().
            3. Lookup relation_score from _RELATION_SCORES.
            4. Compute onset_lag_hours.
            5. Compute confidence from event data_quality_score and relation score.
            6. Assign causal_strength label.
        """
        event_ts_raw: Optional[str] = event.get("timestamp")
        if not event_ts_raw:
            return None

        event_start: Optional[datetime] = _parse_dt(event_ts_raw)
        if event_start is None:
            return None

        # End timestamp: use explicit end_timestamp if available, else treat as point
        event_end_raw: Optional[str] = event.get("end_timestamp")
        event_end: Optional[datetime] = (
            _parse_dt(event_end_raw) if event_end_raw else event_start
        )

        act_start: Optional[datetime] = _parse_dt(activity_interval[0])
        act_end: Optional[datetime] = (
            act_start
            if is_point_event
            else _parse_dt(activity_interval[1])
        )

        allen_rel = self._allen_relation(event_start, event_end, act_start, act_end)
        relation_score = _RELATION_SCORES.get(allen_rel, 0.0)

        # Signed onset lag: positive means the prior event predates the activity
        lag: Optional[float] = None
        if act_start is not None:
            lag = (act_start - event_start).total_seconds() / 3600.0

        dq: float = event.get("data_quality_score") or 0.5
        confidence = self._compute_confidence(allen_rel, dq, lag)
        causal_strength = self._assign_causal_strength(allen_rel, confidence)

        return {
            "link_id": f"LINK::{event.get('event_id', 'unk')}::{uuid.uuid4().hex[:6]}",
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "event_timestamp": event_ts_raw,
            "allen_relation": allen_rel,
            "relation_score": round(relation_score, 4),
            "onset_lag_hours": round(lag, 2) if lag is not None else None,
            "data_quality_score": dq,
            "confidence": round(confidence, 4),
            "causal_strength": causal_strength,
        }

    def _allen_relation(
        self,
        prior_start: Optional[datetime],
        prior_end: Optional[datetime],
        activity_start: Optional[datetime],
        activity_end: Optional[datetime],
    ) -> str:
        """Classify the Allen interval relation of the prior event relative to
        the emergent activity interval.

        Mirrors the allen_relation() function in TSKRTemporalScorerV1 but
        extends the 5-relation RCA vocabulary with SIMULTANEOUS and UNKNOWN.

        Boundary epsilon: events within self.config.epsilon_hours of a
        boundary are treated as simultaneous at that boundary.

        Logic (interval A = prior event, interval B = activity):
            A.end   < B.start - ε  → PRECEDES
            A.start > B.end   + ε  → FOLLOWS
            A.start < B.start - ε  and A.end > B.end + ε  → CONTAINS
            A.start < B.start - ε  and A.end within B  → OVERLAPS
            A entirely within B    → DURING
            otherwise              → SIMULTANEOUS
        """
        if prior_start is None or activity_start is None:
            return _UNKNOWN

        # Collapse point events to zero-width intervals
        a_end = prior_end if prior_end is not None else prior_start
        b_end = activity_end if activity_end is not None else activity_start

        eps = self.config.epsilon_hours * 3600.0
        a_s = prior_start.timestamp()
        a_e = a_end.timestamp()
        b_s = activity_start.timestamp()
        b_e = b_end.timestamp()

        if a_e < b_s - eps:
            return _PRECEDES
        if a_s > b_e + eps:
            return _FOLLOWS
        if a_s < b_s - eps and a_e > b_e + eps:
            return _CONTAINS
        if a_s < b_s - eps:          # a_e is within [b_s-eps, b_e+eps]
            return _OVERLAPS
        if a_e <= b_e + eps:         # a_s >= b_s - eps: A entirely within B
            return _DURING
        # A started within B (a_s >= b_s - eps) but extends beyond B (a_e > b_e + eps)
        return _SIMULTANEOUS

    def _compute_confidence(
        self,
        allen_relation: str,
        data_quality_score: float,
        onset_lag_hours: Optional[float],
    ) -> float:
        """Compute link confidence combining relation strength and data quality.

        Mirrors TSKR multi-factor confidence:
            anomaly_weight  (0.55) → data_quality_score of the prior event
            latency_weight  (0.30) → onset_lag plausibility
            relation_weight (0.15) → _RELATION_SCORES[allen_relation]
        """
        relation_component = _RELATION_SCORES.get(allen_relation, 0.0)

        # Lag plausibility: positive lags ≤ 24 h are most credible causal leads;
        # negative lags (A after B onset) indicate symptoms rather than causes.
        if onset_lag_hours is None:
            lag_plausibility = 0.5
        elif onset_lag_hours < 0:
            lag_plausibility = 0.1
        elif onset_lag_hours <= 24.0:
            lag_plausibility = 1.0
        else:
            # Smooth decay from 1.0 to 0.1 over 30 days (720 h)
            lag_plausibility = max(0.1, 1.0 - (onset_lag_hours - 24.0) / 720.0)

        return (
            _ANOMALY_WEIGHT * (data_quality_score or 0.5)
            + _LATENCY_WEIGHT * lag_plausibility
            + _RELATION_WEIGHT * relation_component
        )

    def _assign_causal_strength(
        self, relation: str, confidence: float
    ) -> str:
        """Map (relation, confidence) to a causal_strength label.

        temporal_contradiction: relation == FOLLOWS
        strong:   relation in {OVERLAPS, CONTAINS, PRECEDES} and confidence >= 0.75
        moderate: relation in {OVERLAPS, CONTAINS, PRECEDES} and confidence >= 0.40
        weak:     otherwise
        """
        if relation == _FOLLOWS:
            return "temporal_contradiction"
        # SIMULTANEOUS (concurrent; possible common cause) is semantically
        # distinct from DURING/UNKNOWN — always at least moderate regardless
        # of confidence, since concurrent events warrant analyst attention.
        if relation == _SIMULTANEOUS:
            return "moderate"
        score = _RELATION_SCORES.get(relation, 0.0) * confidence
        if score >= _STRONG_THRESHOLD:
            return "strong"
        if score >= _MODERATE_THRESHOLD:
            return "moderate"
        return "weak"

    def _summarize_chain(self, links: List[JsonDict]) -> JsonDict:
        """Aggregate per-link results into a chain summary.

        Computes:
            chain_length, strongest_link_id, strongest_allen_relation,
            max_relation_score, has_temporal_contradiction, causal_posture.

        causal_posture logic:
            'contradicted'       — any link is temporal_contradiction
            'supported'          — any strong link present
            'partial'            — any moderate link, no strong
            'weak'               — all links are weak
            'insufficient_data'  — no links
        """
        if not links:
            return {
                "chain_length": 0,
                "strongest_link_id": None,
                "strongest_allen_relation": None,
                "max_relation_score": 0.0,
                "has_temporal_contradiction": False,
                "causal_posture": "insufficient_data",
            }

        has_contradiction = any(
            lnk["causal_strength"] == "temporal_contradiction" for lnk in links
        )
        strongest = max(links, key=lambda lnk: lnk["relation_score"])
        strengths = {lnk["causal_strength"] for lnk in links}

        if has_contradiction:
            posture = "contradicted"
        elif "strong" in strengths:
            posture = "supported"
        elif "moderate" in strengths:
            posture = "partial"
        elif "weak" in strengths:
            posture = "weak"
        else:
            posture = "insufficient_data"

        return {
            "chain_length": len(links),
            "strongest_link_id": strongest["link_id"],
            "strongest_allen_relation": strongest["allen_relation"],
            "max_relation_score": strongest["relation_score"],
            "has_temporal_contradiction": has_contradiction,
            "causal_posture": posture,
        }
