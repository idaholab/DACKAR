"""
PatternSearcher — coarse-to-fine similarity retrieval pipeline.

Executes three-step retrieval against a pre-built IncidentIndex:
    1. Inverted-index lookup — O(|query_event_set|) candidate set
    2. Jaccard pre-filter   — discard clearly dissimilar episodes
    3. Full metric scoring  — NLCS + EMD + combined weighted score

All three metric scores are always computed and returned so that callers
can switch weight profiles or analyse individual signals without re-running.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .config import SearchConfig
from .indexer import IncidentIndex, _coerce_ts
from .metrics import combined_score, emd_similarity, jaccard, nlcs
from .models import HistoricalSignalEpisode, IncidentFingerprint, SearchResult

_log = logging.getLogger(__name__)

_INDEX_STATUS_INDEXED = "indexed"
_INDEX_STATUS_NO_DATA = "no_episodes_indexed"
_INDEX_STATUS_STALE   = "stale"


class PatternSearcher:
    """
    Retrieves the top-k most similar historical episodes for a query incident.

    Pipeline per search() call:
        1. Inverted-index lookup: episode_ids sharing ≥ 1 event type with query.
        2. Jaccard pre-filter: discard candidates below config.min_jaccard.
        3. NLCS computation on survivors.
        4. EMD computation on survivors.
        5. Combined score: weighted sum using the resolved weight profile.
        6. Rank descending by combined_score; return top-k SearchResults.

    The coarse-to-fine design avoids computing NLCS and EMD on clearly
    dissimilar episodes (those failing the Jaccard gate).
    """

    def __init__(self, index: IncidentIndex, config: SearchConfig) -> None:
        self.index = index
        self.config = config

    def search(
        self,
        query: IncidentFingerprint,
        weight_profile: Optional[str] = None,
        staleness_window_days: Optional[int] = None,
    ) -> list[HistoricalSignalEpisode]:
        """
        Retrieves top-k most similar historical episodes for a query fingerprint.

        Returns list[HistoricalSignalEpisode] with index_status populated on every
        result (§4.11):
          - "indexed"             — normal result from a current, populated index
          - "no_episodes_indexed" — index is empty; returns a single sentinel episode
          - "stale"               — index is older than staleness_window_days; results
                                    returned but flagged; link_confidence capped downstream

        Args:
            query:                 IncidentFingerprint for the query incident.
            weight_profile:        Weight profile override; None → config.weight_profile.
            staleness_window_days: If set and index.build_timestamp is known, episodes
                                   from an index older than this are marked "stale".
                                   None disables the staleness check.

        Notes:
            - All three metric scores (jaccard, nlcs, emd) are individually visible
              on every returned HistoricalSignalEpisode (§5).
            - matched_events, query_only_events, episode_only_events are derived from
              event_set comparison.
        """
        if self.index.episodes_df.empty:
            return [_make_no_data_sentinel(query.asset_id)]

        # --- Determine index_status ------------------------------------------
        index_status = _compute_index_status(self.index, staleness_window_days)

        # --- Step 1: Inverted-index candidate lookup -------------------------
        candidate_ids = self.index.get_candidates(query.event_set)
        if not candidate_ids:
            _log.debug("search(): no candidates share event types with query.")
            return [_make_no_data_sentinel(query.asset_id, index_status=_INDEX_STATUS_NO_DATA)]

        # Build an episode_id → row mapping for O(1) lookup per candidate.
        ep_lookup: dict[str, pd.Series] = {}
        for _, row in self.index.episodes_df.iterrows():
            ep_lookup[str(row["episode_id"])] = row

        # --- Step 2: Jaccard pre-filter --------------------------------------
        survivors: list[tuple[str, pd.Series, float]] = []
        for ep_id in candidate_ids:
            row = ep_lookup.get(ep_id)
            if row is None:
                _log.warning("Candidate episode_id %r not found in episodes_df.", ep_id)
                continue
            ep_set = frozenset(row["event_set"])
            j = jaccard(query.event_set, ep_set)
            if j >= self.config.min_jaccard:
                survivors.append((ep_id, row, j))

        if not survivors:
            _log.debug(
                "search(): all %d candidates filtered out by Jaccard threshold %.2f.",
                len(candidate_ids), self.config.min_jaccard,
            )
            return [_make_no_data_sentinel(query.asset_id, index_status=_INDEX_STATUS_NO_DATA)]

        # --- Resolve weights -------------------------------------------------
        profile = weight_profile if weight_profile is not None else self.config.weight_profile
        alpha, beta_w, gamma = self._resolve_weights(profile)

        # --- Determine EMD normalization factor ------------------------------
        norm_factor: Optional[float] = None
        if self.config.emd_normalization_mode == "empirical_max":
            if self.index.emd_normalization_factor is None:
                raise RuntimeError(
                    "emd_normalization_mode='empirical_max' requires "
                    "index.compute_emd_normalization_factor() to be called first."
                )
            norm_factor = self.index.emd_normalization_factor

        # --- Steps 3–5: NLCS, EMD, combined score ----------------------------
        episodes: list[HistoricalSignalEpisode] = []
        for ep_id, row, j_score in survivors:
            ep_set   = frozenset(row["event_set"])
            ep_seq   = list(row["event_seq"])
            ep_fvec  = {k: int(v) for k, v in dict(row["freq_vec"]).items()}
            ep_src   = list(row["source_types"]) if "source_types" in row.index else []

            n_score = nlcs(query.event_seq, ep_seq)
            e_score = emd_similarity(query.freq_vec, ep_fvec, normalization_factor=norm_factor)
            c_score = combined_score(j_score, n_score, e_score, alpha, beta_w, gamma)

            rca = row.get("known_rca")
            episodes.append(
                HistoricalSignalEpisode(
                    episode_id=ep_id,
                    asset_id=str(row.get("asset_id", "")),
                    window_start=_coerce_ts(row["window_start"]),
                    window_end=_coerce_ts(row["window_end"]),
                    source_types=ep_src,
                    event_set=ep_set,
                    event_seq=ep_seq,
                    freq_vec=ep_fvec,
                    similarity_to_current=c_score,
                    jaccard_score=j_score,
                    nlcs_score=n_score,
                    emd_score=e_score,
                    weight_profile=profile,
                    matched_events=set(query.event_set & ep_set),
                    query_only_events=set(query.event_set - ep_set),
                    episode_only_events=set(ep_set - query.event_set),
                    episode_density=float(row["density"]),
                    known_rca=rca if (rca is not None and pd.notna(rca)) else None,
                    linked_doc_ids=[],
                    index_status=index_status,
                )
            )

        # --- Step 6: rank and truncate ---------------------------------------
        episodes.sort(key=lambda e: e.similarity_to_current, reverse=True)
        top = episodes[: self.config.top_k]

        _log.debug(
            "search(): %d candidates → %d survivors → returning %d results "
            "(profile=%r, top_k=%d, index_status=%r).",
            len(candidate_ids), len(survivors), len(top), profile, self.config.top_k, index_status,
        )
        return top

    def _resolve_weights(self, weight_profile: str) -> tuple[float, float, float]:
        """
        Returns (alpha, beta_w, gamma) for the given weight profile name.

        Delegates to SearchConfig.resolve_weights() which owns the profile
        registry and "custom" fallback logic.

        Raises:
            ValueError for unrecognised profile names.
        """
        return self.config.resolve_weights(weight_profile)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _compute_index_status(index: IncidentIndex, staleness_window_days: Optional[int]) -> str:
    """Return the index_status string for a given index and staleness window."""
    if staleness_window_days is not None and index.build_timestamp is not None:
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        built = index.build_timestamp.replace(tzinfo=None) if index.build_timestamp.tzinfo else index.build_timestamp
        age_days = (now - built).total_seconds() / 86400.0
        if age_days > staleness_window_days:
            return _INDEX_STATUS_STALE
    return _INDEX_STATUS_INDEXED


def _make_no_data_sentinel(
    asset_id: str,
    index_status: str = _INDEX_STATUS_NO_DATA,
) -> HistoricalSignalEpisode:
    """Return a sentinel HistoricalSignalEpisode for the no-data case.

    Callers must check index_status before attempting cross-pattern linkage.
    A sentinel has episode_id == "" and similarity_to_current == 0.0.
    """
    return HistoricalSignalEpisode(
        episode_id="",
        asset_id=asset_id,
        window_start=None,
        window_end=None,
        source_types=[],
        event_set=frozenset(),
        event_seq=[],
        freq_vec={},
        similarity_to_current=0.0,
        jaccard_score=0.0,
        nlcs_score=0.0,
        emd_score=0.0,
        weight_profile="",
        matched_events=set(),
        query_only_events=set(),
        episode_only_events=set(),
        episode_density=0.0,
        known_rca=None,
        linked_doc_ids=[],
        index_status=index_status,
    )
