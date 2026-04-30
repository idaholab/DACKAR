from __future__ import annotations

from dataclasses import dataclass, field


_VALID_PROFILES = {"equal", "flooding", "cascade", "custom"}

# Preset (alpha, beta_w, gamma) tuples for named weight profiles.
_PROFILE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "equal":    (1 / 3, 1 / 3, 1 / 3),
    "flooding": (0.10,  0.10,  0.80),   # high-frequency / alarm-flood dominant
    "cascade":  (0.10,  0.80,  0.10),   # clear ordered sequence dominant
}


@dataclass
class SearchConfig:
    """
    Central configuration for the RCA pattern search pipeline.

    All parameters are tunable and should be validated empirically against
    historical data before production use.
    """

    # --- Window expansion ---
    beta: float = 0.2
    # Fractional buffer applied symmetrically to both sides of an episode or
    # query window.  Captures precursor/tail events outside the dense core.
    # E_search_start = E_start - beta * (E_end - E_start)
    # E_search_end   = E_end   + beta * (E_end - E_start)
    # Recommended range: 0.1–0.3.

    # --- Density-based episode detection ---
    delta: float = 0.5
    # Fraction of rho_query used as detection threshold.
    # episode_mask(t) = 1 if rho_hist(t) >= delta * rho_query
    # Lower delta → more episodes (higher recall, lower precision).
    # Recommended range: 0.3–0.7.

    kde_bandwidth: float | str = "auto"
    # KDE bandwidth in seconds for density estimation over event timestamps.
    # "auto": set to D_query / 4 at runtime (ties smoothing to query size).
    # float:  explicit bandwidth in seconds.

    # --- Repetition handling ---
    freq_threshold: int = 5
    # Count above which an event type is considered high-frequency within an
    # episode.  High-frequency types are excluded from event_set and event_seq
    # (used by Jaccard / NLCS) but kept in freq_vec (used by EMD).

    # --- Retrieval ---
    min_jaccard: float = 0.3
    # Minimum Jaccard score for a candidate to proceed past the pre-filter.

    top_k: int = 5
    # Number of top-ranked results returned per query.

    # --- Combined score weights (used when weight_profile == "custom") ---
    alpha: float = 1 / 3
    # Weight of Jaccard score.

    beta_w: float = 1 / 3
    # Weight of NLCS score.

    gamma: float = 1 / 3
    # Weight of EMD score.
    # alpha + beta_w + gamma must equal 1.0 when weight_profile == "custom".

    weight_profile: str = "equal"
    # Preset profile.  Overrides alpha/beta_w/gamma unless "custom".
    # "equal"    → 1/3 each.
    # "flooding" → 0.10 / 0.10 / 0.80 (alarm-flood dominant).
    # "cascade"  → 0.10 / 0.80 / 0.10 (ordered-sequence dominant).
    # "custom"   → uses alpha, beta_w, gamma as set above.

    # --- EMD normalization ---
    emd_normalization_mode: str = "tv"
    # Strategy for normalizing EMD (Earth Mover's Distance) scores.
    # "tv":            use Total Variation distance on probability distributions.
    #                  emd_score ∈ [0,1] and comparable across queries.
    #                  Default, backward compatible, requires no calibration.
    # "empirical_max": use raw L1 distance normalised by the empirical maximum
    #                  observed across all historical episode pairs.
    #                  Requires index.compute_emd_normalization_factor() to be
    #                  called after build_from_history() and before search().
    #                  More grounded in actual plant data than theoretical bounds.

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if not (0 < self.delta <= 1):
            raise ValueError(f"delta must be in (0, 1], got {self.delta}")
        if not (isinstance(self.kde_bandwidth, str) and self.kde_bandwidth == "auto") and not (
            isinstance(self.kde_bandwidth, (int, float)) and self.kde_bandwidth > 0
        ):
            raise ValueError(
                f"kde_bandwidth must be 'auto' or a positive float, got {self.kde_bandwidth!r}"
            )
        if self.freq_threshold < 1:
            raise ValueError(f"freq_threshold must be >= 1, got {self.freq_threshold}")
        if not (0.0 <= self.min_jaccard <= 1.0):
            raise ValueError(f"min_jaccard must be in [0, 1], got {self.min_jaccard}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.weight_profile not in _VALID_PROFILES:
            raise ValueError(
                f"weight_profile must be one of {_VALID_PROFILES}, got {self.weight_profile!r}"
            )
        if self.weight_profile == "custom":
            total = self.alpha + self.beta_w + self.gamma
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"alpha + beta_w + gamma must equal 1.0 for 'custom' profile, got {total}"
                )
        if self.emd_normalization_mode not in {"tv", "empirical_max"}:
            raise ValueError(
                f"emd_normalization_mode must be 'tv' or 'empirical_max', "
                f"got {self.emd_normalization_mode!r}"
            )

    def resolve_weights(self, profile: str | None = None) -> tuple[float, float, float]:
        """
        Returns (alpha, beta_w, gamma) for the given profile name.

        If profile is None, uses self.weight_profile.
        Raises ValueError for unrecognised profile names.
        """
        p = profile if profile is not None else self.weight_profile
        if p not in _VALID_PROFILES:
            raise ValueError(f"Unknown weight profile {p!r}. Valid: {_VALID_PROFILES}")
        if p == "custom":
            return (self.alpha, self.beta_w, self.gamma)
        return _PROFILE_WEIGHTS[p]
