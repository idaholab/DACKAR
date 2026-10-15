from dackar.similarity.OllamaEmbedder import (
  OllamaEmbedder,
  cosine_similarity,
  _FALLBACK_EMBEDDING,
)


class TestCosineSimilarity:

  def test_identical(self):
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

  def test_orthogonal(self):
    assert cosine_similarity([1, 0], [0, 1]) == 0.0

  def test_length_mismatch_is_zero(self):
    assert cosine_similarity([1, 2, 3], [1, 2]) == 0.0

  def test_fallback_singleton_is_zero(self):
    assert cosine_similarity(_FALLBACK_EMBEDDING, [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], _FALLBACK_EMBEDDING) == 0.0

  def test_zero_vector_is_zero(self):
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestOllamaEmbedderOffline:
  """Behaviour that does not require a running Ollama server."""

  def test_empty_text_returns_fallback_singleton(self):
    # Same object as the module sentinel, so cosine_similarity's identity
    # check against it fires.
    emb = OllamaEmbedder(model='nomic-embed-text:latest')
    assert emb.encode('') is _FALLBACK_EMBEDDING

  def test_unreachable_server_degrades_to_fallback(self):
    # Point at a port nothing is listening on; encode() must not raise.
    emb = OllamaEmbedder(base_url='http://localhost:1', timeout=1)
    result = emb.encode('replace coolant pump bearing')
    assert result == _FALLBACK_EMBEDDING

  def test_cache_size_tracks_encoded_texts(self):
    # A successful embedding is cached; cache_size reflects it.
    emb = OllamaEmbedder(base_url='http://localhost:1', timeout=1)
    emb._call_api = lambda text: [0.1, 0.2, 0.3]
    assert emb.cache_size() == 0
    emb.encode('something')
    assert emb.cache_size() == 1

  def test_fallback_is_not_cached(self):
    # A failed call returns the fallback but must not be cached, so a later
    # successful call is not pinned to cosine 0.
    emb = OllamaEmbedder(base_url='http://localhost:1', timeout=1)
    assert emb.encode('replace pump') is _FALLBACK_EMBEDDING
    assert emb.cache_size() == 0

  def test_failure_then_success_recovers(self):
    # Ollama is down for the first call, up for the second: the second call
    # must reach the (now healthy) server rather than return the cached failure.
    calls = {'n': 0}

    def flaky(text):
      calls['n'] += 1
      if calls['n'] == 1:
        return _FALLBACK_EMBEDDING
      return [0.5, 0.5, 0.5]

    emb = OllamaEmbedder(base_url='http://localhost:1', timeout=1)
    emb._call_api = flaky
    assert emb.encode('replace pump') is _FALLBACK_EMBEDDING   # failure, not cached
    assert emb.encode('replace pump') == [0.5, 0.5, 0.5]       # retried, succeeded
    assert emb.cache_size() == 1
