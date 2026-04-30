"""
IncidentIndex — historical episode database.

Stores pre-computed IncidentFingerprints and provides fast candidate
retrieval via an inverted index over event types.

Lifecycle:
    1. build_from_history() — run detector + extractor on raw events_df
    2. add() / add_batch() — incremental updates
    3. get_candidates() — O(|query_event_set|) pre-filter lookup
    4. save() / load() — parquet + JSON persistence
"""
from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import SearchConfig
from .density import EpisodeDetector
from .extractor import IncidentExtractor, _dominant_asset, _parse_ts
from .models import IncidentFingerprint, UnifiedEvent

_log = logging.getLogger(__name__)

# Parquet file name inside the save directory.
_EPISODES_PARQUET = "episodes.parquet"
# Inverted index file name inside the save directory.
_INVERTED_INDEX_JSON = "inverted_index.json"
# EMD normalization metadata file name inside the save directory.
_EMD_META_JSON = "emd_meta.json"

# Columns that store Python objects and need JSON serialisation for parquet.
_COMPLEX_COLS = ("event_set", "event_seq", "freq_vec")


class IncidentIndex:
    """
    Stores and manages pre-computed IncidentFingerprints for the historical
    episode database.

    Internal storage:
        episodes_df       — pd.DataFrame, one row per episode.
        _inverted_index   — dict[str, set[str]]: event_type → set of episode_ids.
                            Used for O(|query_event_set|) Jaccard pre-filtering.
    """

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.episodes_df: pd.DataFrame = pd.DataFrame()
        self._inverted_index: dict[str, set[str]] = {}
        self.emd_normalization_factor: Optional[float] = None

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------

    def build_from_history(
        self,
        events_df: pd.DataFrame,
        rho_query: float,
        query_duration: float,
    ) -> None:
        """
        Populates the index from a raw historical events_df.

        Steps:
            1. Convert events_df rows to UnifiedEvent list.
            2. Run EpisodeDetector.detect() to find episode boundaries.
            3. Group events by episode boundary.
            4. Derive IncidentFingerprint for each episode via
               IncidentExtractor._derive_fingerprint().
            5. Store fingerprints via add_batch().

        Args:
            events_df:      Raw historical event log. Expected columns:
                            raw_id, asset_id, source, event_type,
                            timestamp_start, timestamp_end.
                            No episode_id column required.
            rho_query:      Reference density from the query incident
                            (events per second). Passed to EpisodeDetector.
            query_duration: D_query in seconds. Used for KDE bandwidth and
                            minimum episode duration filter.

        Notes:
            events_df is not modified in place.
            Calling build_from_history() a second time appends to the
            existing index — call reset() first to rebuild from scratch.
        """
        all_events = _df_to_events(events_df)
        if not all_events:
            _log.warning("build_from_history: events_df produced no valid events.")
            return

        detector = EpisodeDetector(self.config)
        boundaries = detector.detect(all_events, rho_query, query_duration)

        if not boundaries:
            _log.info("build_from_history: no episodes detected.")
            return

        # Group events by the first matching episode boundary index.
        ep_event_map: dict[int, list[UnifiedEvent]] = defaultdict(list)
        for ev in all_events:
            for idx, (ep_s, ep_e) in enumerate(boundaries):
                if ep_s <= ev.timestamp_start <= ep_e:
                    ep_event_map[idx].append(ev)
                    break  # assign to first (sorted, non-overlapping after merge)

        # Build a fingerprint per episode.
        fingerprints: list[IncidentFingerprint] = []
        for idx, (ep_s, ep_e) in enumerate(boundaries):
            ep_events = ep_event_map.get(idx, [])
            if not ep_events:
                _log.debug("Episode boundary %d has no events after grouping; skipping.", idx)
                continue

            asset_id = _dominant_asset(ep_events) or "UNKNOWN"
            episode_id = f"EP_{asset_id}_{idx:05d}"

            event_set, event_seq, freq_vec = IncidentExtractor._derive_fingerprint(
                ep_events, self.config.freq_threshold
            )

            duration_s = (ep_e - ep_s).total_seconds()
            density = len(ep_events) / duration_s if duration_s > 0.0 else 0.0

            fingerprints.append(
                IncidentFingerprint(
                    episode_id=episode_id,
                    asset_id=asset_id,
                    window_start=ep_s,
                    window_end=ep_e,
                    density=density,
                    event_set=event_set,
                    event_seq=event_seq,
                    freq_vec=freq_vec,
                    known_rca=None,
                )
            )

        self.add_batch(fingerprints)
        _log.info(
            "build_from_history: detected %d boundaries, built %d fingerprints.",
            len(boundaries), len(fingerprints),
        )

    def reset(self) -> None:
        """Clears all stored episodes and the inverted index."""
        self.episodes_df = pd.DataFrame()
        self._inverted_index = {}

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(self, fingerprint: IncidentFingerprint) -> None:
        """
        Adds a single fingerprint to the index.

        Updates episodes_df and the inverted index incrementally.
        Less efficient than add_batch() for many insertions because the
        inverted index is updated per fingerprint rather than rebuilt once.
        """
        row = _fingerprint_to_row(fingerprint)
        new_df = pd.DataFrame([row])
        if self.episodes_df.empty:
            self.episodes_df = new_df
        else:
            self.episodes_df = pd.concat(
                [self.episodes_df, new_df], ignore_index=True
            )
        for event_type in fingerprint.event_set:
            self._inverted_index.setdefault(event_type, set()).add(
                fingerprint.episode_id
            )

    def add_batch(self, fingerprints: list[IncidentFingerprint]) -> None:
        """
        Adds multiple fingerprints in a single operation.

        Rebuilds the inverted index once after all insertions, which is
        more efficient than repeated add() calls.
        """
        if not fingerprints:
            return
        rows = [_fingerprint_to_row(fp) for fp in fingerprints]
        new_df = pd.DataFrame(rows)
        if self.episodes_df.empty:
            self.episodes_df = new_df
        else:
            self.episodes_df = pd.concat(
                [self.episodes_df, new_df], ignore_index=True
            )
        self._rebuild_inverted_index()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_candidates(self, query_event_set: frozenset[str]) -> list[str]:
        """
        Returns episode_ids of historical episodes that share at least one
        event type with the query.

        Uses the inverted index for O(|query_event_set|) lookup.  Episodes
        with no event-type overlap cannot have Jaccard > 0 and are excluded.

        Args:
            query_event_set: event_set from the query IncidentFingerprint.

        Returns:
            List of episode_ids (may contain duplicates if an episode shares
            multiple event types — callers should treat as a set).
        """
        candidates: set[str] = set()
        for event_type in query_event_set:
            candidates.update(self._inverted_index.get(event_type, set()))
        return list(candidates)

    # ------------------------------------------------------------------
    # EMD Normalization
    # ------------------------------------------------------------------

    def compute_emd_normalization_factor(self, max_pairs: int = 1000) -> float:
        """
        Computes the empirical maximum raw L1 distance across historical episode pairs.

        Used to normalise EMD scores when emd_normalization_mode="empirical_max".
        Should be called once after build_from_history() and before search().

        Algorithm:
            - Extract all freq_vec dicts from episodes_df.
            - If N*(N-1)/2 <= max_pairs: evaluate all pairs.
            - Else: draw max_pairs random pairs without replacement (seeded).
            - For each pair (a, b): raw_l1 = Σ_t |a.get(t,0) - b.get(t,0)|
            - Store the maximum observed L1 distance.

        Args:
            max_pairs: Maximum number of pairs to evaluate. If the index has
                       fewer pairs than this, all pairs are used.

        Returns:
            The empirical maximum raw L1 distance (float >= 0).
            Returns 1.0 if index is empty or contains only one episode
            (to avoid division by zero and provide a sensible fallback).

        Side effect:
            Sets self.emd_normalization_factor to the computed value.
        """
        if self.episodes_df.empty or len(self.episodes_df) <= 1:
            self.emd_normalization_factor = 1.0
            return 1.0

        freq_vecs: list[dict[str, int]] = []
        for _, row in self.episodes_df.iterrows():
            freq_vecs.append({k: int(v) for k, v in dict(row["freq_vec"]).items()})

        n = len(freq_vecs)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        total_possible = len(all_pairs)

        if total_possible <= max_pairs:
            pairs_to_eval = all_pairs
        else:
            random.seed(42)  # Seeded for reproducibility
            pairs_to_eval = random.sample(all_pairs, max_pairs)

        max_l1 = 0.0
        for i, j in pairs_to_eval:
            vocab: set[str] = set(freq_vecs[i]) | set(freq_vecs[j])
            raw_l1 = float(sum(
                abs(freq_vecs[i].get(t, 0) - freq_vecs[j].get(t, 0)) for t in vocab
            ))
            max_l1 = max(max_l1, raw_l1)

        self.emd_normalization_factor = max(max_l1, 1.0)  # Floor at 1.0 to avoid division by near-zero
        _log.info(
            "compute_emd_normalization_factor: evaluated %d/%d pairs, "
            "max raw L1 distance = %.1f",
            len(pairs_to_eval), total_possible, self.emd_normalization_factor,
        )
        return self.emd_normalization_factor

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persists the index to disk.

        episodes_df is saved as parquet with complex columns (frozenset,
        list, dict) JSON-serialised to strings.
        The inverted index is saved as JSON (sets → sorted lists).
        EMD metadata (normalization factor) is saved as JSON.
        All files are written atomically via a .tmp rename.

        Args:
            path: Directory path.  Created if it does not exist.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- episodes.parquet ------------------------------------------------
        parquet_path = save_dir / _EPISODES_PARQUET
        tmp_parquet = str(parquet_path) + ".tmp"
        serialised_df = _serialise_df(self.episodes_df)
        serialised_df.to_parquet(tmp_parquet, index=False)
        os.replace(tmp_parquet, parquet_path)

        # --- inverted_index.json ---------------------------------------------
        json_path = save_dir / _INVERTED_INDEX_JSON
        tmp_json = str(json_path) + ".tmp"
        serialisable = {
            et: sorted(ep_ids)
            for et, ep_ids in self._inverted_index.items()
        }
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)
        os.replace(tmp_json, json_path)

        # --- emd_meta.json ---------------------------------------------------
        emd_path = save_dir / _EMD_META_JSON
        tmp_emd = str(emd_path) + ".tmp"
        emd_meta = {
            "emd_normalization_factor": self.emd_normalization_factor
        }
        with open(tmp_emd, "w", encoding="utf-8") as f:
            json.dump(emd_meta, f, indent=2)
        os.replace(tmp_emd, emd_path)

        _log.info(
            "IncidentIndex saved to %s (%d episodes, emd_factor=%.1f).",
            path, len(self.episodes_df),
            self.emd_normalization_factor if self.emd_normalization_factor else 0.0,
        )

    @classmethod
    def load(cls, path: str, config: SearchConfig) -> "IncidentIndex":
        """
        Loads a persisted index from disk.

        Reconstructs episodes_df (deserialising complex columns from JSON
        strings), the inverted index, and EMD metadata if available.

        Args:
            path:   Directory path written by save().
            config: SearchConfig to attach to the loaded index.

        Returns:
            Populated IncidentIndex.

        Raises:
            FileNotFoundError if core files (parquet, inverted index) are missing.
        """
        save_dir = Path(path)
        parquet_path = save_dir / _EPISODES_PARQUET
        json_path = save_dir / _INVERTED_INDEX_JSON
        emd_path = save_dir / _EMD_META_JSON

        if not parquet_path.exists():
            raise FileNotFoundError(f"Episodes parquet not found: {parquet_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Inverted index JSON not found: {json_path}")

        index = cls(config)

        # Load and deserialise episodes_df.
        raw_df = pd.read_parquet(parquet_path)
        index.episodes_df = _deserialise_df(raw_df)

        # Load inverted index (lists → sets).
        with open(json_path, encoding="utf-8") as f:
            raw_inv = json.load(f)
        index._inverted_index = {et: set(ep_ids) for et, ep_ids in raw_inv.items()}

        # Load EMD metadata if it exists (backward compatible with old indices).
        if emd_path.exists():
            with open(emd_path, encoding="utf-8") as f:
                emd_meta = json.load(f)
            index.emd_normalization_factor = emd_meta.get("emd_normalization_factor")
        else:
            _log.warning(
                "EMD metadata file not found at %s; emd_normalization_factor will be None. "
                "Call compute_emd_normalization_factor() if needed.", emd_path
            )
            index.emd_normalization_factor = None

        _log.info(
            "IncidentIndex loaded from %s (%d episodes, emd_factor=%s).",
            path, len(index.episodes_df),
            index.emd_normalization_factor if index.emd_normalization_factor else "None",
        )
        return index

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_inverted_index(self) -> None:
        """Rebuilds the inverted index from scratch from episodes_df."""
        self._inverted_index = {}
        if self.episodes_df.empty:
            return
        for _, row in self.episodes_df.iterrows():
            ep_id = row["episode_id"]
            event_set = row["event_set"]
            for et in event_set:
                self._inverted_index.setdefault(et, set()).add(ep_id)

    def __len__(self) -> int:
        return len(self.episodes_df)

    def __repr__(self) -> str:
        return (
            f"IncidentIndex(episodes={len(self)}, "
            f"event_types={len(self._inverted_index)})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _df_to_events(events_df: pd.DataFrame) -> list[UnifiedEvent]:
    """Converts a raw events_df to a list of UnifiedEvents."""
    events: list[UnifiedEvent] = []
    for _, row in events_df.iterrows():
        ts_start = _parse_ts(row.get("timestamp_start"))
        if ts_start is None:
            _log.debug(
                "Row with raw_id=%r has no valid timestamp_start; skipping.",
                row.get("raw_id"),
            )
            continue
        events.append(
            UnifiedEvent(
                raw_id=str(row.get("raw_id", "")),
                asset_id=str(row.get("asset_id", "")),
                source=str(row.get("source", "")),
                event_type=str(row.get("event_type", "")),
                timestamp_start=ts_start,
                timestamp_end=_parse_ts(row.get("timestamp_end")),
            )
        )
    return events


def _fingerprint_to_row(fp: IncidentFingerprint) -> dict:
    """Converts an IncidentFingerprint to a flat dict for DataFrame storage."""
    return {
        "episode_id": fp.episode_id,
        "asset_id": fp.asset_id,
        "window_start": fp.window_start,
        "window_end": fp.window_end,
        "density": fp.density,
        "event_set": fp.event_set,    # frozenset — stored as object column
        "event_seq": fp.event_seq,    # list[str]
        "freq_vec": fp.freq_vec,      # dict[str, int]
        "known_rca": fp.known_rca,
    }


def _coerce_ts(value) -> datetime:
    """Coerces a pandas Timestamp or datetime to a plain datetime.

    pandas NaT subclasses datetime, so the NaT check must come first.
    pandas Timestamp also subclasses datetime, so to_pydatetime() must
    be tried before the plain isinstance guard.
    """
    # pandas NaT is a datetime subclass — check it first.
    if type(value).__name__ == "NaTType":
        raise TypeError(f"Cannot coerce NaT to datetime")
    # pandas Timestamp has to_pydatetime(); use it to get a plain datetime.
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    raise TypeError(f"Cannot coerce {type(value)} to datetime")


# ---------------------------------------------------------------------------
# Parquet serialisation / deserialisation for complex columns
# ---------------------------------------------------------------------------

def _serialise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with frozenset/list/dict columns JSON-encoded as strings.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    if "event_set" in out.columns:
        out["event_set"] = out["event_set"].apply(
            lambda x: json.dumps(sorted(x)) if isinstance(x, (frozenset, set)) else x
        )
    if "event_seq" in out.columns:
        out["event_seq"] = out["event_seq"].apply(
            lambda x: json.dumps(list(x)) if isinstance(x, list) else x
        )
    if "freq_vec" in out.columns:
        out["freq_vec"] = out["freq_vec"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else x
        )
    return out


def _deserialise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with JSON-string columns decoded back to Python objects.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    if "event_set" in out.columns:
        out["event_set"] = out["event_set"].apply(
            lambda x: frozenset(json.loads(x)) if isinstance(x, str) else x
        )
    if "event_seq" in out.columns:
        out["event_seq"] = out["event_seq"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    if "freq_vec" in out.columns:
        out["freq_vec"] = out["freq_vec"].apply(
            lambda x: {k: int(v) for k, v in json.loads(x).items()}
            if isinstance(x, str)
            else x
        )
    return out
