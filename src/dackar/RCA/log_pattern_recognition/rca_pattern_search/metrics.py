"""
Similarity metrics for the RCA pattern search pipeline.

Three metrics answer complementary questions about two incident fingerprints:

    jaccard        — did the same event types occur?        (set, order-agnostic)
    nlcs           — did they occur in a similar order?     (sequence-aware)
    emd_similarity — did they repeat with similar intensity? (frequency-based)

All functions return a value in [0, 1] where 1 is identical.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Metric 1: Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """
    Set-based similarity between two event sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 when both sets are empty (undefined ratio treated as no
    similarity rather than perfect similarity, which is the safer default
    for retrieval purposes).
    """
    union_size = len(a | b)
    if union_size == 0:
        return 0.0
    return len(a & b) / union_size


# ---------------------------------------------------------------------------
# Metric 2: NLCS (Normalised Longest Common Subsequence)
# ---------------------------------------------------------------------------

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Standard O(m·n) DP for LCS length."""
    m, n = len(a), len(b)
    # Use two rolling rows to keep memory at O(min(m, n)).
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def nlcs(a: list[str], b: list[str]) -> float:
    """
    Normalised Longest Common Subsequence similarity.

    NLCS(A, B) = |LCS(A, B)| / max(|A|, |B|)

    Operates on deduplicated ordered sequences so high-frequency events do not
    dominate the ordering signal.

    Returns 0.0 when both sequences are empty.
    """
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return _lcs_length(a, b) / denom


# ---------------------------------------------------------------------------
# Metric 3: EMD similarity (categorical frequency distributions)
# ---------------------------------------------------------------------------

def emd_similarity(
    a: dict[str, int],
    b: dict[str, int],
    normalization_factor: float | None = None,
) -> float:
    """
    Frequency-based similarity between two event count vectors.

    Captures the repetition signal that Jaccard and NLCS deliberately discard.

    Implementation uses the Total Variation (TV) distance between the two
    probability distributions derived by normalising the count vectors.  For
    categorical distributions with unit ground distance between any two
    distinct types, TV distance equals the (unit-ground) Earth Mover's
    Distance:

        TV(P, Q) = 0.5 · Σ_t |P(t) − Q(t)|   where P(t) = a[t] / Σa

    This is always in [0, 1], so:

        emd_similarity = 1 − TV(P, Q)

    Alternative (raw-count) normalisation:
        If normalization_factor is provided the raw L1 distance between the
        unnormalised count vectors is used instead:

            raw_emd = Σ_t |a.get(t, 0) − b.get(t, 0)|
            emd_similarity = max(0.0, 1 − raw_emd / normalization_factor)

        Suitable when an empirically derived or vocabulary-size-based upper
        bound is available (see spec open point on normalisation).

    Edge cases:
        Both empty  → 1.0  (identical — neither has any events)
        One empty   → 0.0  (maximally dissimilar)
    """
    total_a = sum(a.values()) if a else 0
    total_b = sum(b.values()) if b else 0

    if total_a == 0 and total_b == 0:
        return 1.0
    if total_a == 0 or total_b == 0:
        return 0.0

    vocab: set[str] = set(a) | set(b)

    if normalization_factor is not None:
        if normalization_factor <= 0:
            raise ValueError(
                f"normalization_factor must be positive, got {normalization_factor}"
            )
        raw_l1 = sum(abs(a.get(t, 0) - b.get(t, 0)) for t in vocab)
        return max(0.0, 1.0 - raw_l1 / normalization_factor)

    # Default: TV distance on probability distributions.
    tv = 0.5 * sum(
        abs(a.get(t, 0) / total_a - b.get(t, 0) / total_b) for t in vocab
    )
    return 1.0 - tv


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

def combined_score(
    j: float,
    n: float,
    e: float,
    alpha: float,
    beta_w: float,
    gamma: float,
) -> float:
    """
    Weighted combination of the three metric scores.

    Score = alpha · J + beta_w · NLCS + gamma · EMD

    Inputs are assumed to be valid (alpha + beta_w + gamma ≈ 1).
    No re-normalisation is applied here; the caller (PatternSearcher) is
    responsible for passing a coherent weight triple via SearchConfig.
    """
    return alpha * j + beta_w * n + gamma * e
