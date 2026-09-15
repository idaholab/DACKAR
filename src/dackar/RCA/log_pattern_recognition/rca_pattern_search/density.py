"""
EpisodeDetector — KDE-based episode boundary detection.

Detects contiguous periods in a historical event log where the estimated
event rate exceeds a fraction of the query incident's reference density.

KDE implementation uses an event-centric Gaussian kernel: each event
contributes only to grid points within 4σ of its timestamp.  This is
O(N × support_window / grid_res) instead of O(N × G), which is essential
for long historical windows with fine grid resolution.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from .config import SearchConfig
from .extractor import _compute_density, _dominant_asset, _expand_window
from .models import UnifiedEvent

_log = logging.getLogger(__name__)

_SQRT_2PI: float = math.sqrt(2 * math.pi)
# Gaussian kernel is negligible beyond 4σ (< 0.003% of peak).
_KDE_SUPPORT_SIGMAS: float = 4.0


class EpisodeDetector:
    """
    Detects episode boundaries in a continuous historical event log using
    kernel density estimation (KDE) over event timestamps.

    The detection threshold is relative to the query incident density,
    making the method self-calibrating: a busier query requires historically
    busier windows to qualify as matching episodes.

    Pipeline (per detect() call):
        1. Convert event timestamps to float seconds since earliest event.
        2. Evaluate Gaussian KDE on a fine time grid.
        3. Threshold: mask = KDE(t) >= delta * rho_query.
        4. Extract contiguous masked regions as raw episode boundaries.
        5. Apply beta buffer expansion to each boundary.
        6. Merge overlapping expanded boundaries.
        7. Discard episodes shorter than query_duration / 10.
    """

    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    def compute_reference_density(
        self,
        query_events: list[UnifiedEvent],
        window_start: datetime,
        window_end: datetime,
    ) -> float:
        """
        Computes the reference event density for the query incident.

            rho_query = N_query / D_query

        N_query: events with timestamp_start in [window_start, window_end].
        D_query: (window_end - window_start).total_seconds()

        All sources contribute equally.  Returns 0.0 if duration <= 0.
        """
        return _compute_density(query_events, window_start, window_end)

    def detect(
        self,
        historical_events: list[UnifiedEvent],
        rho_query: float,
        query_duration: float,
    ) -> list[tuple[datetime, datetime]]:
        """
        Detects episode boundaries in the historical event log.

        Args:
            historical_events: Full flat event list, all sources merged.
                               No episode_id expected at this stage.
            rho_query:         Reference density from compute_reference_density().
                               Units: events per second.
            query_duration:    D_query in seconds.  Used for KDE bandwidth and
                               minimum episode duration filter.

        Returns:
            List of (episode_start, episode_end) tuples, sorted ascending.
            These are expanded boundaries (beta already applied) ready for
            fingerprinting.  Empty list if no qualifying episodes found.
        """
        if not historical_events or rho_query <= 0.0 or query_duration <= 0.0:
            return []

        # --- Resolve bandwidth ------------------------------------------------
        if self.config.kde_bandwidth == "auto":
            bw = query_duration / 4.0
        else:
            bw = float(self.config.kde_bandwidth)
        if bw <= 0.0:
            _log.warning("KDE bandwidth resolved to %.3f s; using 1.0 s fallback.", bw)
            bw = 1.0

        # --- Convert timestamps to float seconds since earliest event ----------
        t_epoch = min(e.timestamp_start for e in historical_events)
        t_seconds = np.array(
            [(e.timestamp_start - t_epoch).total_seconds() for e in historical_events],
            dtype=np.float64,
        )

        # --- Run detection pipeline with resolved bandwidth -------------------
        return self._run_detection(
            t_seconds, t_epoch, rho_query, query_duration, bw
        )

    def bandwidth_scan(
        self,
        historical_events: list[UnifiedEvent],
        rho_query: float,
        query_duration: float,
        bandwidths: Optional[list[float]] = None,
    ) -> dict[float, int]:
        """
        Multi-scale diagnostic: counts detected episodes at different bandwidths.

        Helps operators validate episode segmentation by showing how many episodes
        are detected when the smoothing scale varies. Useful when query timescale
        may not match historical episode timescales (e.g., fast transient query
        for slow degradation history, or vice versa).

        Args:
            historical_events: Full flat event list, all sources merged.
            rho_query:         Reference density from compute_reference_density().
                               Units: events per second.
            query_duration:    D_query in seconds.
            bandwidths:        Explicit bandwidth list in seconds. If None, defaults to
                               [D/32, D/16, D/8, D/4, D/2, D, 2D, 4D] for broad coverage.

        Returns:
            dict mapping bandwidth (float, seconds) to episode count (int),
            sorted by bandwidth ascending.
        """
        if not historical_events or rho_query <= 0.0 or query_duration <= 0.0:
            return {}

        if bandwidths is None:
            # Default: 8 scales covering 1/32× to 4× query duration.
            bandwidths = [
                query_duration / 32,
                query_duration / 16,
                query_duration / 8,
                query_duration / 4,
                query_duration / 2,
                query_duration,
                2 * query_duration,
                4 * query_duration,
            ]

        t_epoch = min(e.timestamp_start for e in historical_events)
        t_seconds = np.array(
            [(e.timestamp_start - t_epoch).total_seconds() for e in historical_events],
            dtype=np.float64,
        )

        result: dict[float, int] = {}
        for bw in sorted(bandwidths):
            if bw <= 0.0:
                _log.warning(
                    "bandwidth_scan: ignoring non-positive bandwidth %.3f s", bw
                )
                continue
            boundaries = self._run_detection(
                t_seconds, t_epoch, rho_query, query_duration, bw
            )
            result[bw] = len(boundaries)

        _log.debug(
            "bandwidth_scan(): tested %d bandwidths. Summary:\n%s",
            len(result),
            "\n".join(
                f"  {bw:9.1f} s (D/{query_duration/bw:.1f}): {count:3d} episodes"
                for bw, count in result.items()
            ),
        )
        return result

    def _run_detection(
        self,
        t_seconds: np.ndarray,
        t_epoch: datetime,
        rho_query: float,
        query_duration: float,
        bw: float,
    ) -> list[tuple[datetime, datetime]]:
        """
        Core detection pipeline: KDE → threshold → extract → expand → merge → filter.

        Called by detect() and bandwidth_scan() to avoid code duplication.

        Args:
            t_seconds:     Event timestamps as float seconds since t_epoch.
            t_epoch:       Reference time (datetime).
            rho_query:     Reference density (events/second).
            query_duration: Duration of query window in seconds.
            bw:            Bandwidth in seconds (already resolved, > 0).

        Returns:
            Expanded, merged, filtered episode boundaries.
        """
        # --- Time grid --------------------------------------------------------
        t_max = float(t_seconds.max())
        grid_res = min(query_duration / 100.0, 60.0)
        grid = np.arange(0.0, t_max + grid_res, grid_res)

        # --- Gaussian KDE (event-centric, O(N × support / grid_res)) ----------
        kde_values = _kde_evaluate(t_seconds, grid, bw, grid_res)

        # --- Threshold and extract contiguous regions -------------------------
        threshold = self.config.delta * rho_query
        mask = kde_values >= threshold

        boundaries = _extract_contiguous_regions(mask, grid, t_epoch, grid_res)
        if not boundaries:
            return []

        # --- Beta expansion ---------------------------------------------------
        expanded = [_expand_window(s, e, self.config.beta) for s, e in boundaries]

        # --- Merge overlapping ------------------------------------------------
        merged = _merge_overlapping(expanded)

        # --- Minimum duration filter ------------------------------------------
        min_dur_s = query_duration / 10.0
        result = [
            (s, e) for s, e in merged
            if (e - s).total_seconds() >= min_dur_s
        ]

        return result

    def assign_episode_ids(
        self,
        historical_events: list[UnifiedEvent],
        episode_boundaries: list[tuple[datetime, datetime]],
    ) -> list[UnifiedEvent]:
        """
        Assigns episode_id to each historical event based on detected boundaries.

        An event is assigned to the episode whose expanded boundary contains
        its timestamp_start.  Events outside all boundaries retain
        episode_id = None (background noise).

        If an event falls within multiple boundaries (should not occur after
        merging, handled defensively), it is assigned to the first match and
        a WARNING is logged.

        Episode ids: "EP_{asset_id}_{index:05d}" where asset_id is the
        dominant asset among events in that episode.

        Args:
            historical_events:   Full flat event list.
            episode_boundaries:  Output of detect(), sorted (start, end) tuples.

        Returns:
            New list of UnifiedEvents with episode_id populated.
            Original list is not mutated.
        """
        if not episode_boundaries:
            return [dataclasses.replace(e, episode_id=None) for e in historical_events]

        # Pre-pass: count events per episode to compute densities for tie-breaking.
        ep_counts: dict[int, int] = defaultdict(int)
        for ev in historical_events:
            for idx, (ep_s, ep_e) in enumerate(episode_boundaries):
                if ep_s <= ev.timestamp_start <= ep_e:
                    ep_counts[idx] += 1

        ep_densities: dict[int, float] = {}
        for idx, (ep_s, ep_e) in enumerate(episode_boundaries):
            dur = (ep_e - ep_s).total_seconds()
            ep_densities[idx] = ep_counts[idx] / dur if dur > 0.0 else 0.0

        # Pass 1: assign episode index (int) to each event --------------------
        idx_assignments: list[Optional[int]] = []
        ep_event_lists: dict[int, list[UnifiedEvent]] = defaultdict(list)

        for ev in historical_events:
            matches = [
                idx
                for idx, (ep_s, ep_e) in enumerate(episode_boundaries)
                if ep_s <= ev.timestamp_start <= ep_e
            ]
            if not matches:
                idx_assignments.append(None)
            else:
                if len(matches) > 1:
                    chosen = max(matches, key=lambda i: ep_densities[i])
                    _log.warning(
                        "Event %r at %s falls in %d episode boundaries after merging; "
                        "assigning to highest-density episode (ep index %d).",
                        ev.raw_id, ev.timestamp_start, len(matches), chosen,
                    )
                else:
                    chosen = matches[0]
                idx_assignments.append(chosen)
                ep_event_lists[chosen].append(ev)

        # Pass 2: generate episode ID strings per episode index ---------------
        ep_id_strings: dict[int, str] = {}
        for idx in range(len(episode_boundaries)):
            asset_id = _dominant_asset(ep_event_lists.get(idx, [])) or "UNKNOWN"
            ep_id_strings[idx] = f"EP_{asset_id}_{idx:05d}"

        # Pass 3: build result with episode_id populated ----------------------
        result: list[UnifiedEvent] = []
        for ev, ep_idx in zip(historical_events, idx_assignments):
            ep_id = ep_id_strings[ep_idx] if ep_idx is not None else None
            result.append(dataclasses.replace(ev, episode_id=ep_id))

        return result


# ---------------------------------------------------------------------------
# KDE helpers
# ---------------------------------------------------------------------------

def _kde_evaluate(
    t_seconds: np.ndarray,
    grid: np.ndarray,
    bw: float,
    grid_res: float,
) -> np.ndarray:
    """
    Evaluates a Gaussian KDE at each grid point.

    Uses an event-centric approach: each event contributes only to grid
    points within _KDE_SUPPORT_SIGMAS * bw seconds, limiting work to
    O(N × support_window / grid_res) instead of O(N × G).

    Returns rho_hist in events per second:
        rho_hist(t) = Σ_i (1 / (bw √2π)) · exp(-½ · ((t − tᵢ) / bw)²)
    """
    kde = np.zeros(len(grid))
    support_radius = _KDE_SUPPORT_SIGMAS * bw
    inv_bw_sqrt2pi = 1.0 / (bw * _SQRT_2PI)
    grid_start = float(grid[0]) if len(grid) > 0 else 0.0

    for t_i in t_seconds:
        # Compute grid index bounds relative to grid[0], not 0.
        lo = max(0, int((t_i - grid_start - support_radius) / grid_res))
        hi = min(len(grid), int((t_i - grid_start + support_radius) / grid_res) + 2)
        diff = (grid[lo:hi] - t_i) / bw
        kde[lo:hi] += np.exp(-0.5 * diff * diff)

    kde *= inv_bw_sqrt2pi
    return kde


# ---------------------------------------------------------------------------
# Region extraction helpers
# ---------------------------------------------------------------------------

def _extract_contiguous_regions(
    mask: np.ndarray,
    grid: np.ndarray,
    t_epoch: datetime,
    grid_res: float,
) -> list[tuple[datetime, datetime]]:
    """
    Extracts (start, end) datetime pairs for every contiguous True run in mask.

    Pads the mask with False on both ends to detect edges at array boundaries.
    Uses the last grid point of each run as t_end.  If a run is a single grid
    point, t_end is set to t_start + grid_res to give it a non-zero duration.
    """
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    rising  = np.where(diff ==  1)[0]   # first grid index of each run
    falling = np.where(diff == -1)[0]   # first grid index AFTER each run

    regions: list[tuple[datetime, datetime]] = []
    for r, f in zip(rising, falling):
        t_start = t_epoch + timedelta(seconds=float(grid[r]))
        last_idx = f - 1  # last True index in run
        if last_idx > r:
            t_end = t_epoch + timedelta(seconds=float(grid[last_idx]))
        else:
            t_end = t_start + timedelta(seconds=grid_res)
        regions.append((t_start, t_end))

    return regions


def _merge_overlapping(
    boundaries: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """
    Merges overlapping or touching (start, end) intervals.

    Input order does not matter.  Returns intervals sorted by start ascending.
    """
    if not boundaries:
        return []

    sorted_b = sorted(boundaries, key=lambda x: x[0])
    merged = [sorted_b[0]]

    for start, end in sorted_b[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged
