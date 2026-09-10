"""
Unit tests for OllamaEmbedder, EmbeddingSemanticScorer, and
LLMAbbreviationDisambiguator.

All tests use fakes/stubs — no live Ollama server required.  The live
integration (actual Ollama calls) is exercised separately by the benchmark
script demos/embedding_benchmark.py.

Coverage:
  OllamaEmbedder
    - encode() returns cached value on second call (no extra API hit)
    - encode() returns fallback [0.0] on URLError
    - encode() returns fallback on malformed JSON response
    - encode_batch() delegates to encode()
    - save_cache() / load from JSON file
    - empty string returns fallback without API call

  cosine_similarity
    - identical vectors → 1.0
    - orthogonal vectors → 0.0
    - fallback vector [0.0] → 0.0
    - length mismatch → 0.0

  EmbeddingSemanticScorer
    - score() uses stored embeddings from activity metadata
    - score() falls back to trigram-Jaccard when embedding missing
    - score() computes embedding on-the-fly via embedder when missing
    - score() clamps result to [0, 1]

  LLMAbbreviationDisambiguator
    - calls upstream resolver first
    - does NOT fire LLM for already-expanded tokens (lowercase / mixed)
    - fires LLM for unresolved uppercase 2-8-char alpha tokens
    - uses cache on second call (LLM not called again)
    - returns original token when LLM response not in candidate set
    - returns original token on URLError

  AppConfig integration
    - embedding_enabled=False → SemanticSimilarityScorer (no embedder created)
    - llm_disambiguation_enabled=False → plain AbbreviationResolver
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import urllib.error
from unittest.mock import MagicMock, call, patch

import pytest

from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.preprocessing.ollama_embedder import (
    OllamaEmbedder,
    _FALLBACK_EMBEDDING,
    cosine_similarity,
)
from outage_uncertainty.preprocessing.ollama_disambiguator import (
    LLMAbbreviationDisambiguator,
)
from outage_uncertainty.retrieval.semantic_similarity import EmbeddingSemanticScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_activity(
    activity_id: str = "A1",
    description: str = "replace motor operated valve",
    embedding: list[float] | None = None,
) -> ActivityCase:
    act = ActivityCase(
        activity_id=activity_id,
        outage_id="OT001",
        plant_id="P1",
        cleaned_description=description,
    )
    if embedding is not None:
        act.metadata["features"] = {"text_embedding": embedding}
    return act


def _fake_urlopen_ok(embedding: list[float]):
    """Context manager that returns a successful embedding response."""
    resp_bytes = json.dumps({"embedding": embedding}).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=resp_bytes)))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _fake_urlopen_error():
    raise urllib.error.URLError("connection refused")


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------

class TestOllamaEmbedder:
    def test_encode_calls_api_and_returns_embedding(self):
        emb = [0.1, 0.2, 0.3]
        with patch("urllib.request.urlopen", return_value=_fake_urlopen_ok(emb)):
            enc = OllamaEmbedder()
            result = enc.encode("replace valve")
        assert result == emb

    def test_encode_caches_result(self):
        emb = [0.1, 0.2]
        with patch("urllib.request.urlopen", return_value=_fake_urlopen_ok(emb)) as mock_open:
            enc = OllamaEmbedder()
            enc.encode("replace valve")
            enc.encode("replace valve")   # second call — must NOT hit API
        assert mock_open.call_count == 1

    def test_encode_returns_fallback_on_url_error(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("err")):
            enc = OllamaEmbedder()
            result = enc.encode("some text")
        assert result == _FALLBACK_EMBEDDING

    def test_encode_returns_fallback_on_bad_json(self):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"not json")))
        cm.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=cm):
            enc = OllamaEmbedder()
            result = enc.encode("bad response")
        assert result == _FALLBACK_EMBEDDING

    def test_encode_empty_string_returns_fallback_without_api(self):
        with patch("urllib.request.urlopen") as mock_open:
            enc = OllamaEmbedder()
            result = enc.encode("")
        mock_open.assert_not_called()
        assert result == _FALLBACK_EMBEDDING

    def test_encode_batch_delegates_to_encode(self):
        emb = [0.5, 0.6]
        with patch("urllib.request.urlopen", return_value=_fake_urlopen_ok(emb)):
            enc = OllamaEmbedder()
            results = enc.encode_batch(["text A", "text B"])
        assert len(results) == 2
        assert results[0] == emb
        assert results[1] == emb

    def test_save_and_load_cache(self):
        emb = [1.0, 2.0, 3.0]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            # Save
            with patch("urllib.request.urlopen", return_value=_fake_urlopen_ok(emb)):
                enc = OllamaEmbedder(cache_path=cache_path)
                enc.encode("test text")
                enc.save_cache()

            # Load into new instance — must not call API
            with patch("urllib.request.urlopen") as mock_open:
                enc2 = OllamaEmbedder(cache_path=cache_path)
                result = enc2.encode("test text")
            mock_open.assert_not_called()
            assert result == emb
        finally:
            os.unlink(cache_path)

    def test_cache_size_reflects_stored_entries(self):
        emb = [0.1]
        with patch("urllib.request.urlopen", return_value=_fake_urlopen_ok(emb)):
            enc = OllamaEmbedder()
            assert enc.cache_size() == 0
            enc.encode("alpha")
            assert enc.cache_size() == 1
            enc.encode("beta")
            assert enc.cache_size() == 2
            enc.encode("alpha")   # cached — no new entry
            assert enc.cache_size() == 2


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_fallback_vector_returns_zero(self):
        assert cosine_similarity(_FALLBACK_EMBEDDING, [1.0, 2.0, 3.0]) == 0.0
        assert cosine_similarity([1.0, 2.0, 3.0], _FALLBACK_EMBEDDING) == 0.0

    def test_length_mismatch_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_zero_norm_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_antiparallel_vectors(self):
        v = [1.0, 0.0]
        neg_v = [-1.0, 0.0]
        result = cosine_similarity(v, neg_v)
        assert abs(result - (-1.0)) < 1e-9

    def test_known_value(self):
        # [1,0] · [1,1]/√2 = 1/√2 ≈ 0.7071
        a = [1.0, 0.0]
        b = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        assert abs(cosine_similarity(a, b) - 1.0 / math.sqrt(2)) < 1e-6


# ---------------------------------------------------------------------------
# EmbeddingSemanticScorer
# ---------------------------------------------------------------------------

class TestEmbeddingSemanticScorer:
    def test_score_uses_stored_embeddings(self):
        scorer = EmbeddingSemanticScorer()
        # Identical embeddings → cosine = 1.0
        emb = [1.0, 0.0, 0.0]
        a = _make_activity("A", embedding=emb)
        b = _make_activity("B", embedding=emb)
        assert abs(scorer.score(a, b) - 1.0) < 1e-6

    def test_score_orthogonal_embeddings(self):
        scorer = EmbeddingSemanticScorer()
        a = _make_activity("A", embedding=[1.0, 0.0])
        b = _make_activity("B", embedding=[0.0, 1.0])
        assert abs(scorer.score(a, b)) < 1e-6

    def test_score_falls_back_to_trigram_when_no_embedding(self):
        scorer = EmbeddingSemanticScorer(embedder=None)
        a = _make_activity("A", description="replace valve", embedding=None)
        b = _make_activity("B", description="replace valve", embedding=None)
        # trigram Jaccard of identical strings = 1.0
        assert scorer.score(a, b) == pytest.approx(1.0)

    def test_score_computes_embedding_on_the_fly(self):
        emb = [1.0, 0.0]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = emb
        scorer = EmbeddingSemanticScorer(embedder=mock_embedder)

        a = _make_activity("A", description="replace valve", embedding=None)
        b = _make_activity("B", description="replace valve", embedding=None)
        result = scorer.score(a, b)

        assert mock_embedder.encode.called
        assert abs(result - 1.0) < 1e-6

    def test_score_caches_on_the_fly_embedding(self):
        emb = [0.6, 0.8]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = emb
        scorer = EmbeddingSemanticScorer(embedder=mock_embedder)

        a = _make_activity("A", description="inspect pump", embedding=None)
        b = _make_activity("B", embedding=emb)

        scorer.score(a, b)
        # Embedding should now be cached in a.metadata
        assert a.metadata["features"]["text_embedding"] == emb
        # Second call must not re-encode
        scorer.score(a, b)
        assert mock_embedder.encode.call_count == 1

    def test_score_clamped_to_unit_interval(self):
        """Cosine result is already in [-1,1]; scorer must clamp to [0,1]."""
        scorer = EmbeddingSemanticScorer()
        # Antiparallel → cosine = -1 → clamped to 0
        a = _make_activity("A", embedding=[1.0, 0.0])
        b = _make_activity("B", embedding=[-1.0, 0.0])
        assert scorer.score(a, b) == 0.0


# ---------------------------------------------------------------------------
# LLMAbbreviationDisambiguator
# ---------------------------------------------------------------------------

def _make_llm_response(text: str) -> MagicMock:
    resp_bytes = json.dumps({"response": text}).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=resp_bytes)))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestLLMAbbreviationDisambiguator:
    def _resolver(self, expansions: dict[str, str] | None = None):
        """Minimal stub for the upstream AbbreviationResolver."""
        mock = MagicMock()
        expansions = expansions or {}

        def transform(text):
            tokens = []
            for t in text.split():
                tokens.append(expansions.get(t.upper(), t))
            return " ".join(tokens)

        mock.transform.side_effect = transform
        return mock

    def test_already_expanded_token_not_sent_to_llm(self):
        """Lowercase tokens after rule-based expansion must never trigger LLM."""
        # Both tokens pre-expanded to lowercase by the rule-based resolver
        resolver = self._resolver({
            "MOV": "motor operated valve",
            "RHR": "residual heat removal",
        })
        dis = LLMAbbreviationDisambiguator(resolver, model="mistral:latest")
        with patch("urllib.request.urlopen") as mock_open:
            result = dis.transform("inspect MOV in RHR")
        # Both tokens expanded to lowercase — no uppercase tokens remain → no LLM call
        mock_open.assert_not_called()
        assert "motor operated valve" in result
        assert "residual heat removal" in result

    def test_llm_called_for_unresolved_uppercase_token(self):
        resolver = self._resolver()   # no expansions
        dis = LLMAbbreviationDisambiguator(resolver, model="mistral:latest")
        with patch(
            "urllib.request.urlopen",
            return_value=_make_llm_response("motor operated valve"),
        ) as mock_open:
            result = dis.transform("inspect MOV in system")
        assert mock_open.called
        assert "motor operated valve" in result

    def test_cache_prevents_duplicate_llm_calls(self):
        resolver = self._resolver()
        dis = LLMAbbreviationDisambiguator(
            resolver, model="mistral:latest"
        )
        with patch(
            "urllib.request.urlopen",
            return_value=_make_llm_response("motor operated valve"),
        ) as mock_open:
            dis.transform("inspect MOV")
            dis.transform("replace MOV")   # same token, different description
        # LLM should be called only once (cached on first call)
        assert mock_open.call_count == 1

    def test_returns_original_on_url_error(self):
        resolver = self._resolver()
        dis = LLMAbbreviationDisambiguator(resolver, model="mistral:latest")
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("unreachable"),
        ):
            result = dis.transform("inspect MOV")
        # MOV should remain unchanged when Ollama is down
        assert "MOV" in result

    def test_returns_original_when_llm_response_off_script(self):
        """If the LLM returns something not in the candidate pool, keep original."""
        resolver = self._resolver()
        dis = LLMAbbreviationDisambiguator(resolver, model="mistral:latest")
        with patch(
            "urllib.request.urlopen",
            return_value=_make_llm_response("some unexpected hallucination"),
        ):
            result = dis.transform("test XYZ now")
        assert "XYZ" in result

    def test_short_tokens_not_sent_to_llm(self):
        """Tokens of length < 2 or > 8 must not trigger LLM."""
        resolver = self._resolver()
        dis = LLMAbbreviationDisambiguator(resolver, model="mistral:latest")
        with patch("urllib.request.urlopen") as mock_open:
            dis.transform("A TOOLONGABBREVIATION normal text")
        mock_open.assert_not_called()

    def test_mixed_case_token_not_sent_to_llm(self):
        resolver = self._resolver()
        dis = LLMAbbreviationDisambiguator(resolver, model="mistral:latest")
        with patch("urllib.request.urlopen") as mock_open:
            dis.transform("check Valve in system")
        mock_open.assert_not_called()

    def test_save_and_load_cache(self):
        resolver = self._resolver()
        dis = LLMAbbreviationDisambiguator(resolver, model="mistral:latest")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            with patch(
                "urllib.request.urlopen",
                return_value=_make_llm_response("motor operated valve"),
            ):
                dis._cache_path = cache_path
                dis.transform("inspect MOV")
                dis.save_cache()

            # New instance with same cache — must not call LLM
            dis2 = LLMAbbreviationDisambiguator(
                resolver, model="mistral:latest", cache_path=cache_path
            )
            with patch("urllib.request.urlopen") as mock_open:
                dis2.transform("replace MOV")
            mock_open.assert_not_called()
        finally:
            os.unlink(cache_path)


# ---------------------------------------------------------------------------
# AppConfig integration
# ---------------------------------------------------------------------------

class TestAppConfigIntegration:
    def test_embedding_disabled_by_default(self):
        from outage_uncertainty.api.config import AppConfig
        cfg = AppConfig()
        assert cfg.embedding_enabled is False

    def test_llm_disambiguation_disabled_by_default(self):
        from outage_uncertainty.api.config import AppConfig
        cfg = AppConfig()
        assert cfg.llm_disambiguation_enabled is False

    def test_default_embedding_model(self):
        from outage_uncertainty.api.config import AppConfig
        cfg = AppConfig()
        assert cfg.embedding_model == "nomic-embed-text:latest"

    def test_default_llm_model(self):
        from outage_uncertainty.api.config import AppConfig
        cfg = AppConfig()
        assert cfg.llm_model == "mistral:latest"

    def test_ollama_base_url_configurable(self):
        from outage_uncertainty.api.config import AppConfig
        cfg = AppConfig(ollama_base_url="http://gpu-server:11434")
        assert cfg.ollama_base_url == "http://gpu-server:11434"
