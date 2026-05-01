from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


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
    known_rca: str | None = None  # RCA outcome label if available
    source_types: list[str] = field(default_factory=list)
    # Sorted list of unique source values present in the episode's events.
    # alarm   → "alarm"
    # soe     → "soe"
    # anomaly → "anomaly"
    # Populated by build_from_history() and IncidentExtractor.extract();
    # defaults to [] for fingerprints created before Phase 1.


@dataclass
class HistoricalSignalEpisode:
    """
    Public output type for PatternSearcher.search().

    Represents a single historical signal episode retrieved for a query incident.
    Carries all three metric scores individually (§5 of the integration plan) and
    an index_status field that governs cross-pattern linkage eligibility (§4.11).

    Sentinel (no_episodes_indexed) instances have episode_id == "" and
    similarity_to_current == 0.0.  Callers must check index_status before
    attempting linkage.
    """

    episode_id: str
    asset_id: str
    window_start: Optional[datetime]       # None for sentinel episodes
    window_end: Optional[datetime]         # None for sentinel episodes
    source_types: list[str]                # ["alarm", "soe", "anomaly"] — present source categories
    event_set: frozenset[str]
    event_seq: list[str]
    freq_vec: dict[str, int]
    similarity_to_current: float           # weighted combined score [0, 1]; 0.0 for sentinel
    jaccard_score: float                   # set-based metric
    nlcs_score: float                      # sequence-aware metric
    emd_score: float                       # frequency-based metric
    weight_profile: str                    # profile used for combined score
    matched_events: set[str]               # event types in both query and episode
    query_only_events: set[str]            # event types in query only
    episode_only_events: set[str]          # event types in episode only
    episode_density: float                 # rho of matched episode
    known_rca: Optional[str]              # known root cause label if available
    linked_doc_ids: list[str]             # populated by cross-pattern linkage (Phase 2)
    index_status: str
    # "indexed"              — episode is from a populated, current index; eligible for linkage
    # "no_episodes_indexed"  — index contained no episodes for the asset; linkage must not run
    # "stale"                — index was built outside the staleness window; linkage allowed
    #                          but link_confidence capped at 0.70 (§4.11)


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
