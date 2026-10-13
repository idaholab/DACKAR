from dackar.similarity.stringSimilarity import token_jaccard, trigram_jaccard


class TestTokenJaccard:

  def test_partial_overlap(self):
    assert token_jaccard('a b c', 'b c d') == 2 / 4

  def test_identical(self):
    assert token_jaccard('pump bearing', 'pump bearing') == 1.0

  def test_disjoint(self):
    assert token_jaccard('pump', 'valve') == 0.0

  def test_both_empty_is_similar(self):
    assert token_jaccard('', '') == 1.0

  def test_one_empty_is_dissimilar(self):
    assert token_jaccard('pump', '') == 0.0

  def test_case_insensitive(self):
    assert token_jaccard('Pump', 'pump') == 1.0


class TestTrigramJaccard:

  def test_shares_root(self):
    # "calibrate" / "calibration" share the "calibr..." trigrams.
    assert trigram_jaccard('calibrate', 'calibration') > 0.4

  def test_identical(self):
    assert trigram_jaccard('pressure', 'pressure') == 1.0

  def test_disjoint(self):
    assert trigram_jaccard('pump', 'xyzzy') == 0.0

  def test_short_strings_treated_as_single_token(self):
    # Both < 3 chars: each becomes {whole-string}; equal strings -> 1.0.
    assert trigram_jaccard('ab', 'ab') == 1.0
    assert trigram_jaccard('ab', 'cd') == 0.0

  def test_both_empty(self):
    # Each empty string -> {''}; the sets are equal -> 1.0.
    assert trigram_jaccard('', '') == 1.0
