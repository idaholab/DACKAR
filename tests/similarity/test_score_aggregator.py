from dackar.similarity.scoreAggregator import WeightedScoreAggregator


class TestWeightedScoreAggregator:

  def test_weighted_sum(self):
    agg = WeightedScoreAggregator({'x': 0.5, 'y': 0.5})
    assert agg.combine({'x': 1.0, 'y': 0.0}) == 0.5

  def test_missing_scored_dimension_contributes_zero(self):
    agg = WeightedScoreAggregator({'x': 0.5, 'y': 0.5})
    assert agg.combine({'x': 1.0}) == 0.5

  def test_unweighted_score_dimension_is_ignored(self):
    agg = WeightedScoreAggregator({'x': 1.0})
    assert agg.combine({'x': 0.5, 'y': 1.0}) == 0.5

  def test_clamped_to_unit_interval(self):
    # Weights summing to > 1 would exceed 1.0 without clamping.
    agg = WeightedScoreAggregator({'x': 1.0, 'y': 1.0})
    assert agg.combine({'x': 1.0, 'y': 1.0}) == 1.0

  def test_empty_scores(self):
    agg = WeightedScoreAggregator({'x': 0.5})
    assert agg.combine({}) == 0.0
