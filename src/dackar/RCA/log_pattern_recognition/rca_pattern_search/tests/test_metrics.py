"""Unit tests for rca_pattern_search.metrics"""
import pytest
from ..metrics import jaccard, nlcs, emd_similarity, combined_score


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical(self):
        a = frozenset({"A", "B", "C"})
        assert jaccard(a, a) == pytest.approx(1.0)

    def test_disjoint(self):
        assert jaccard(frozenset({"A"}), frozenset({"B"})) == pytest.approx(0.0)

    def test_partial(self):
        # |{A,B} ∩ {B,C}| / |{A,B,C}| = 1/3
        assert jaccard(frozenset({"A", "B"}), frozenset({"B", "C"})) == pytest.approx(1 / 3)

    def test_one_empty(self):
        assert jaccard(frozenset(), frozenset({"A"})) == pytest.approx(0.0)

    def test_both_empty(self):
        # Both empty → no similarity (safer retrieval default)
        assert jaccard(frozenset(), frozenset()) == pytest.approx(0.0)

    def test_subset(self):
        # |{A}| / |{A,B,C}| = 1/3
        assert jaccard(frozenset({"A"}), frozenset({"A", "B", "C"})) == pytest.approx(1 / 3)

    def test_symmetry(self):
        a = frozenset({"X", "Y"})
        b = frozenset({"Y", "Z"})
        assert jaccard(a, b) == pytest.approx(jaccard(b, a))


# ---------------------------------------------------------------------------
# nlcs
# ---------------------------------------------------------------------------

class TestNlcs:
    def test_identical(self):
        seq = ["A", "B", "C"]
        assert nlcs(seq, seq) == pytest.approx(1.0)

    def test_disjoint(self):
        assert nlcs(["A", "B"], ["C", "D"]) == pytest.approx(0.0)

    def test_one_empty(self):
        assert nlcs([], ["A", "B"]) == pytest.approx(0.0)

    def test_both_empty(self):
        assert nlcs([], []) == pytest.approx(0.0)

    def test_known_lcs(self):
        # LCS of [A,B,C,D] and [A,C,D] is [A,C,D] length 3; max len = 4
        assert nlcs(["A", "B", "C", "D"], ["A", "C", "D"]) == pytest.approx(3 / 4)

    def test_symmetry(self):
        a = ["A", "B", "C"]
        b = ["B", "C", "D"]
        assert nlcs(a, b) == pytest.approx(nlcs(b, a))

    def test_reverse_is_lower(self):
        # Reversed sequence shares all elements but in opposite order
        a = ["A", "B", "C"]
        b = list(reversed(a))
        # LCS length = 1 (any single element), NLCS = 1/3
        assert nlcs(a, b) == pytest.approx(1 / 3)

    def test_normalization_by_longer(self):
        # LCS([A],[A,B,C]) = 1; max len = 3
        assert nlcs(["A"], ["A", "B", "C"]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# emd_similarity
# ---------------------------------------------------------------------------

class TestEmdSimilarity:
    def test_identical(self):
        v = {"A": 3, "B": 1}
        assert emd_similarity(v, v) == pytest.approx(1.0)

    def test_both_empty(self):
        assert emd_similarity({}, {}) == pytest.approx(1.0)

    def test_one_empty(self):
        assert emd_similarity({}, {"A": 2}) == pytest.approx(0.0)
        assert emd_similarity({"A": 2}, {}) == pytest.approx(0.0)

    def test_disjoint_types(self):
        # All mass on different types → TV = 1 → similarity = 0
        assert emd_similarity({"A": 5}, {"B": 5}) == pytest.approx(0.0)

    def test_same_distribution_different_scale(self):
        # {"A": 2, "B": 2} vs {"A": 4, "B": 4} — same proportions → TV = 0 → sim = 1
        a = {"A": 2, "B": 2}
        b = {"A": 4, "B": 4}
        assert emd_similarity(a, b) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # a: A=1, B=1 → P(A)=0.5, P(B)=0.5
        # b: A=1, C=1 → P(A)=0.5, P(C)=0.5
        # TV = 0.5 * (|0.5-0.5| + |0.5-0| + |0-0.5|) = 0.5 * (0 + 0.5 + 0.5) = 0.5
        assert emd_similarity({"A": 1, "B": 1}, {"A": 1, "C": 1}) == pytest.approx(0.5)

    def test_symmetry(self):
        a = {"A": 3, "B": 1}
        b = {"A": 1, "B": 3}
        assert emd_similarity(a, b) == pytest.approx(emd_similarity(b, a))

    def test_raw_normalization(self):
        # With normalization_factor: raw L1 / factor
        # a = {A:3}, b = {A:1}; raw L1 = 2; factor = 4
        # similarity = 1 - 2/4 = 0.5
        assert emd_similarity({"A": 3}, {"A": 1}, normalization_factor=4) == pytest.approx(0.5)

    def test_raw_normalization_clamp_to_zero(self):
        # Large raw L1 → clamp to 0 not negative
        result = emd_similarity({"A": 10}, {"B": 10}, normalization_factor=1)
        assert result == pytest.approx(0.0)

    def test_invalid_normalization_factor(self):
        with pytest.raises(ValueError):
            emd_similarity({"A": 1}, {"A": 1}, normalization_factor=0)


# ---------------------------------------------------------------------------
# combined_score
# ---------------------------------------------------------------------------

class TestCombinedScore:
    def test_equal_weights(self):
        s = combined_score(0.6, 0.9, 0.3, 1 / 3, 1 / 3, 1 / 3)
        assert s == pytest.approx((0.6 + 0.9 + 0.3) / 3)

    def test_flooding_profile(self):
        # gamma dominates
        s = combined_score(0.0, 0.0, 1.0, 0.1, 0.1, 0.8)
        assert s == pytest.approx(0.8)

    def test_cascade_profile(self):
        # beta_w dominates
        s = combined_score(0.0, 1.0, 0.0, 0.1, 0.8, 0.1)
        assert s == pytest.approx(0.8)

    def test_all_zero(self):
        assert combined_score(0, 0, 0, 1 / 3, 1 / 3, 1 / 3) == pytest.approx(0.0)

    def test_all_one(self):
        assert combined_score(1, 1, 1, 1 / 3, 1 / 3, 1 / 3) == pytest.approx(1.0)
