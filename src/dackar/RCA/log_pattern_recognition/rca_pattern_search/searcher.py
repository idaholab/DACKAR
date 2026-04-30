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
from typing import Optional

import pandas as pd

from .config import SearchConfig
from .indexer import IncidentIndex, _coerce_ts
from .metrics import combined_score, emd_similarity, jaccard, nlcs
from .models import IncidentFingerprint, SearchResult

_log = logging.getLogger(__name__)


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
    ) -> list[SearchResult]:
        """
        Retrieves top-k most similar historical episodes for a query fingerprint.

        Args:
            query:          IncidentFingerprint for the query incident.
                            Derived by IncidentExtractor.extract().
            weight_profile: Weight profile for the combined score.
                            One of: "equal" | "flooding" | "cascade" | "custom".
                            None falls back to config.weight_profile.

        Returns:
            List of SearchResult sorted by combined_score descending.
            Length <= config.top_k.  Empty if no candidates survive the
            Jaccard pre-filter.

        Notes:
            - matched_events, query_only_events, episode_only_events are
              derived from event_set comparison, not event_seq or freq_vec.
            - All three metric scores are always computed and stored, so
              downstream code can compare profiles without re-searching.
        """
        if self.index.episodes_df.empty:
            return []

        # --- Step 1: Inverted-index candidate lookup -------------------------
        candidate_ids = self.index.get_candidates(query.event_set)
        if not candidate_ids:
            _log.debug("search(): no candidates share event types with query.")
            return []

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
            return []

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
        results: list[SearchResult] = []
        for ep_id, row, j_score in survivors:
            ep_set   = frozenset(row["event_set"])
            ep_seq   = list(row["event_seq"])
            ep_fvec  = {k: int(v) for k, v in dict(row["freq_vec"]).items()}

            n_score = nlcs(query.event_seq, ep_seq)
            e_score = emd_similarity(query.freq_vec, ep_fvec, normalization_factor=norm_factor)
            c_score = combined_score(j_score, n_score, e_score, alpha, beta_w, gamma)

            rca = row.get("known_rca")
            results.append(
                SearchResult(
                    episode_id=ep_id,
                    jaccard_score=j_score,
                    nlcs_score=n_score,
                    emd_score=e_score,
                    combined_score=c_score,
                    weight_profile=profile,
                    episode_window=(
                        _coerce_ts(row["window_start"]),
                        _coerce_ts(row["window_end"]),
                    ),
                    episode_density=float(row["density"]),
                    matched_events=set(query.event_set & ep_set),
                    query_only_events=set(query.event_set - ep_set),
                    episode_only_events=set(ep_set - query.event_set),
                    known_rca=rca if (rca is not None and pd.notna(rca)) else None,
                )
            )

        # --- Step 6: rank and truncate ---------------------------------------
        results.sort(key=lambda r: r.combined_score, reverse=True)
        top = results[: self.config.top_k]

        _log.debug(
            "search(): %d candidates → %d survivors → returning %d results "
            "(profile=%r, top_k=%d).",
            len(candidate_ids), len(survivors), len(top), profile, self.config.top_k,
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
