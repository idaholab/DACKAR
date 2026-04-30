from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UnifiedEvent:
    """
    Canonical representation of a single event from any source.

    All three input sources (alarm, SOE, anomaly) are normalised into this
    structure before any further processing.

    Lifecycle:
        - Created by IncidentExtractor.to_unified_events()
        - episode_id is None until EpisodeDetector assigns membership
        - timestamp_end is nullable and not used in current similarity metrics
          but carried for traceability and future use
    """

    raw_id: str
    # Original record identifier from the source schema.
    # alarm   → alarms[].alarm_id
    # soe     → records[].record_id
    # anomaly → anomalies[].anomaly_id

    asset_id: str
    # Asset this event belongs to.

    source: str
    # One of: "alarm" | "soe" | "anomaly"

    event_type: str
    # Canonical label used in all similarity computations.
    # alarm   → alarm_id
    # soe     → f"{signal_id}::{transition}"   e.g. "SIG_001::trip"
    # anomaly → f"{sensor_id}::{pattern}"      e.g. "TEMP_01::spike"

    timestamp_start: datetime
    # Representative timestamp.  Always populated.  Used as t_start in all
    # ordering and density computations.

    timestamp_end: datetime | None
    # End timestamp.  Nullable — may be NaT depending on source system.
    # NOT used in current similarity metrics.  Carried for future use.

    episode_id: str | None = None
    # Assigned by EpisodeDetector after density-based detection.
    # None if event falls outside any detected episode (background noise).


@dataclass
class IncidentFingerprint:
    """
    Pre-computed similarity representations for a single incident or detected
    historical episode.  This is the unit of comparison in the retrieval
    pipeline.

    Derived from a list of UnifiedEvents by IncidentExtractor.extract() or
    EpisodeDetector after episode boundary assignment.

    The three representations serve distinct metrics:
        event_set  → Jaccard  (what types occurred, ignoring order/repetition)
        event_seq  → NLCS     (what types occurred and in what order)
        freq_vec   → EMD      (how many times each type occurred)

    High-frequency event types (count > freq_threshold) are excluded from
    event_set and event_seq but retained in freq_vec.
    """

    episode_id: str
    asset_id: str
    window_start: datetime       # Expanded window start (after beta applied)
    window_end: datetime         # Expanded window end (after beta applied)
    density: float               # rho = N_events / window_duration_seconds
    event_set: frozenset[str]    # Deduplicated event types, unordered
    event_seq: list[str]         # Deduplicated event types, ordered by first t_start
    freq_vec: dict[str, int]     # event_type → total occurrence count
    known_rca: str | None = None # RCA outcome label if available


@dataclass
class SearchResult:
    """
    A single entry in the ranked retrieval output.

    Returned by PatternSearcher.search() for each matching historical episode.
    Includes all three metric scores for transparency and downstream analysis.
    """

    episode_id: str
    jaccard_score: float                      # Metric 1: set-based [0, 1]
    nlcs_score: float                         # Metric 2: sequence-aware [0, 1]
    emd_score: float                          # Metric 3: frequency-based [0, 1]
    combined_score: float                     # Weighted combination [0, 1]
    weight_profile: str                       # Profile used
    episode_window: tuple[datetime, datetime]
    episode_density: float                    # rho of matched episode
    matched_events: set[str]                  # In both query and episode
    query_only_events: set[str]               # In query, not in episode
    episode_only_events: set[str]             # In episode, not in query
    known_rca: str | None = None              # Known root cause if available
