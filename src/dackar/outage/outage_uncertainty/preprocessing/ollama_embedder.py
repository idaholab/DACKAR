"""
Ollama-backed text embedder for nuclear outage activity descriptions.

Replaces the :class:`SimpleEmbedder` stub in :mod:`preprocessing.feature_builder`
with a real dense-vector embedder served by a locally running Ollama instance.

Embedding model recommendations (from benchmark on outage_cleaning_benchmark.csv):

* ``nomic-embed-text:latest`` — best discrimination ratio (1.415), fastest,
  274 MB.  Recommended default for most deployments.
* ``mxbai-embed-large:latest`` — best Precision@5/10 on category retrieval,
  ~2.4× larger (669 MB).  Preferred when retrieval ranking quality matters most.

The embedder is **stateless except for its cache**.  The same instance can be
shared across the ingestion workflow and the similarity engine.

Caching
-------
Two-level cache:

1. **In-memory** — a plain ``dict`` mapping text → embedding.  Lives for the
   lifetime of the embedder instance.  Avoids redundant API calls within a
   single run (e.g. the same cleaned description appearing in multiple
   historical activities).

2. **Persistent JSON file** (optional) — loaded at construction and saved on
   demand via :meth:`save_cache`.  Allows embeddings to survive across
   service restarts so a large historical corpus need not be re-embedded on
   every cold start.

Graceful degradation
--------------------
If the Ollama server is unreachable (``URLError``, ``TimeoutError``) the
embedder logs a warning and returns a single-element vector ``[0.0]`` so
the rest of the pipeline can continue using the trigram-Jaccard fallback
inside :class:`EmbeddingSemanticScorer`.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_FALLBACK_EMBEDDING: list[float] = [0.0]


class OllamaEmbedder:
    """Dense-vector text embedder backed by a local Ollama server.

    Args:
        model: Ollama model name, e.g. ``"nomic-embed-text:latest"`` or
            ``"mxbai-embed-large:latest"``.
        base_url: Base URL of the Ollama HTTP API.
            Default ``"http://localhost:11434"``.
        cache_path: Optional path to a JSON file used as a persistent cache.
            If the file already exists its contents are loaded on construction.
            Call :meth:`save_cache` to flush in-memory additions back to disk.
        timeout: HTTP request timeout in seconds.  Default 30.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        cache_path: str | None = None,
        timeout: int = 30,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache: dict[str, list[float]] = {}
        self._cache_path = cache_path

        if cache_path and Path(cache_path).exists():
            self._load_cache(cache_path)

        logger.info(
            "OllamaEmbedder: model=%s, base_url=%s, cache_entries=%d",
            model,
            base_url,
            len(self._cache),
        )

    # ------------------------------------------------------------------
    # Public interface (compatible with SimpleEmbedder and feature_builder)
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[float]:
        """Return the dense embedding vector for *text*.

        Results are cached in memory.  Returns ``[0.0]`` if Ollama is
        unavailable (logged at WARNING level).
        """
        if not text:
            return _FALLBACK_EMBEDDING

        if text in self._cache:
            return self._cache[text]

        embedding = self._call_api(text)
        self._cache[text] = embedding
        return embedding

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts, exploiting the in-memory cache."""
        return [self.encode(t) for t in texts]

    def cache_size(self) -> int:
        return len(self._cache)

    def save_cache(self) -> None:
        """Flush the in-memory cache to the JSON file specified at construction."""
        if not self._cache_path:
            return
        path = Path(self._cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)
        logger.info(
            "OllamaEmbedder: saved %d embeddings to %s",
            len(self._cache),
            self._cache_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, text: str) -> list[float]:
        payload = json.dumps({"model": self._model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
                return data["embedding"]
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "OllamaEmbedder: API call failed for model '%s' (%s); "
                "returning fallback embedding — similarity will degrade to trigram-Jaccard",
                self._model,
                exc,
            )
            return _FALLBACK_EMBEDDING
        except (KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                "OllamaEmbedder: unexpected API response (%s); returning fallback",
                exc,
            )
            return _FALLBACK_EMBEDDING

    def _load_cache(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                self._cache = json.load(fh)
            logger.info(
                "OllamaEmbedder: loaded %d cached embeddings from %s",
                len(self._cache),
                path,
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "OllamaEmbedder: could not load cache from %s (%s); starting empty",
                path,
                exc,
            )


# ---------------------------------------------------------------------------
# Cosine similarity utility (used by EmbeddingSemanticScorer)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors.

    Returns 0.0 if either vector is the fallback ``[0.0]`` singleton or if
    either norm is zero.
    """
    if a == _FALLBACK_EMBEDDING or b == _FALLBACK_EMBEDDING:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
