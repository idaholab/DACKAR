"""
Semantic similarity scorers for outage activity descriptions.

Two implementations are provided:

EmbeddingSemanticScorer (recommended)
--------------------------------------
Computes cosine similarity between dense embedding vectors produced by a
local Ollama model (see :mod:`preprocessing.ollama_embedder`).  Embeddings
are pre-computed during ingestion and stored in
``activity.metadata["features"]["text_embedding"]``, so scoring at query
time is a single dot-product — O(d) where d is the embedding dimension.

Benchmark results on ``outage_cleaning_benchmark.csv`` (175 activity pairs):

* ``nomic-embed-text`` — disc. ratio 1.415, P@5 0.619
* ``mxbai-embed-large`` — disc. ratio 1.252, P@5 0.648

Falls back transparently to character-trigram Jaccard when the embedder is
unavailable or when the activity has no stored embedding (e.g. during unit
tests that build minimal ``ActivityCase`` fixtures).

SemanticSimilarityScorer (legacy / WordNet)
--------------------------------------------
Wraps ``dackar.similarity.SentenceSimilarity``, which implements the
Pawar-Mago (2018) method: word-sense disambiguation via ``pywsd``
(simple_lesk by default) followed by bidirectional synset-similarity
vector comparison using WordNet path + Wu-Palmer similarity.

Falls back to character-trigram Jaccard when ``dackar.similarity`` or its
NLTK/pywsd dependencies are absent.
"""
from __future__ import annotations

import logging

from outage_uncertainty.domain.activity import ActivityCase
from outage_uncertainty.preprocessing.ollama_embedder import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticSimilarityScorer:
    """Scores semantic similarity between two cleaned activity descriptions.

    Args:
        disambiguation_method: Word-sense disambiguation strategy passed to
            ``SentenceSimilarity``.  Choices: ``'simple_lesk'`` (default),
            ``'original_lesk'``, ``'cosine_lesk'``, ``'adapted_lesk'``,
            ``'max_similarity'``.
        similarity_method: Synset comparison method.  Default
            ``'semantic_similarity_synsets'`` uses a harmonic mean of WordNet
            path similarity and Wu-Palmer similarity.
    """

    def __init__(
        self,
        disambiguation_method: str = "simple_lesk",
        similarity_method: str = "semantic_similarity_synsets",
    ) -> None:
        self._scorer = None
        self._disambiguation_method = disambiguation_method
        self._similarity_method = similarity_method

        try:
            from dackar.similarity.SentenceSimilarity import SentenceSimilarity

            self._scorer = SentenceSimilarity(
                disambiguationMethod=disambiguation_method,
                similarityMethod=similarity_method,
            )
            logger.info(
                "SemanticSimilarityScorer: WordNet/Pawar-Mago ready "
                "(disambiguation=%s)",
                disambiguation_method,
            )
        except ImportError as exc:
            logger.warning(
                "SemanticSimilarityScorer: dackar.similarity unavailable (%s); "
                "using trigram-Jaccard fallback",
                exc,
            )
        except Exception as exc:  # noqa: BLE001  (e.g. NLTK data missing)
            logger.warning(
                "SemanticSimilarityScorer: SentenceSimilarity initialisation failed (%s); "
                "using trigram-Jaccard fallback",
                exc,
            )

    # ------------------------------------------------------------------
    # Public interface expected by SimilarityEngine
    # ------------------------------------------------------------------

    def score(self, a: ActivityCase, b: ActivityCase) -> float:
        """Return a similarity score in [0, 1] between activities *a* and *b*."""
        text_a = a.cleaned_description or a.raw_description or ""
        text_b = b.cleaned_description or b.raw_description or ""
        if not text_a or not text_b:
            return 0.0
        if self._scorer is not None:
            return self._dackar_score(text_a, text_b)
        return self._trigram_jaccard(text_a, text_b)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dackar_score(self, text_a: str, text_b: str) -> float:
        """Call the DACKAR WordNet scorer; fall back to trigrams on error."""
        try:
            result = self._scorer.sentenceSimilarity(
                text_a, text_b, method="pm_disambiguation"
            )
            return float(max(0.0, min(1.0, result)))
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "SemanticSimilarityScorer: dackar scoring failed (%s); using fallback",
                exc,
            )
            return self._trigram_jaccard(text_a, text_b)

    @staticmethod
    def _trigram_jaccard(text_a: str, text_b: str) -> float:
        """Character-trigram Jaccard index — lightweight dependency-free fallback.

        Gives better recall than word Jaccard for short strings because it
        handles partial word matches (e.g. "calibr" in both "calibrate" and
        "calibration") and minor typos (edit distance ≤ 1 on a trigram).
        """
        def trigrams(s: str) -> set[str]:
            s = s.lower()
            return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

        tg_a = trigrams(text_a)
        tg_b = trigrams(text_b)
        union = tg_a | tg_b
        if not union:
            return 0.0
        return len(tg_a & tg_b) / len(union)


# ---------------------------------------------------------------------------
# EmbeddingSemanticScorer — drop-in replacement backed by Ollama embeddings
# ---------------------------------------------------------------------------

_EMBEDDING_KEY = "text_embedding"


class EmbeddingSemanticScorer:
    """Scores semantic similarity using pre-computed dense embeddings.

    Embeddings are stored in ``activity.metadata["features"]["text_embedding"]``
    by :class:`ActivityFeatureBuilder` during ingestion.  Scoring is a single
    cosine dot-product — no external calls at query time.

    Graceful degradation
    --------------------
    If either activity lacks a stored embedding (e.g. test fixtures, cold-start
    before the first ingestion) the scorer falls back to character-trigram
    Jaccard and logs a DEBUG message.

    Args:
        embedder: An :class:`OllamaEmbedder` instance used to encode the query
            activity on-the-fly when its embedding has not yet been cached.
            When ``None``, the scorer can only use pre-stored embeddings (any
            activity without one will trigger the trigram fallback).
    """

    def __init__(self, embedder=None) -> None:
        self._embedder = embedder

    def score(self, a: ActivityCase, b: ActivityCase) -> float:
        """Return cosine similarity in [0, 1] between *a* and *b*.

        Retrieves embeddings from ``metadata["features"]["text_embedding"]``.
        If the embedding is missing for *a* it is computed on the fly using
        the embedder (and cached back on the activity).  If it is missing for
        *b* (historical activity, should always be pre-computed) the scorer
        falls back to trigram Jaccard.
        """
        emb_a = self._get_embedding(a)
        emb_b = self._get_embedding(b)

        if emb_a is None or emb_b is None:
            text_a = a.cleaned_description or a.raw_description or ""
            text_b = b.cleaned_description or b.raw_description or ""
            logger.debug(
                "EmbeddingSemanticScorer: missing embedding for activity "
                "'%s' or '%s'; using trigram-Jaccard fallback",
                a.activity_id,
                b.activity_id,
            )
            return SemanticSimilarityScorer._trigram_jaccard(text_a, text_b)

        # Dimension mismatch indicates the query and the historical corpus were
        # embedded with different models (e.g. after a model upgrade without
        # re-embedding).  cosine_similarity() returns 0.0 silently on mismatch,
        # which looks like "completely unrelated activities" and causes the
        # selector to flag low coverage across the board.  Warn loudly and fall
        # back to trigram Jaccard so retrieval degrades gracefully rather than
        # silently producing wrong rankings.
        if len(emb_a) != len(emb_b):
            text_a = a.cleaned_description or a.raw_description or ""
            text_b = b.cleaned_description or b.raw_description or ""
            logger.warning(
                "EmbeddingSemanticScorer: embedding dimension mismatch "
                "(%d vs %d) for activities '%s' and '%s'. "
                "This typically means the historical corpus was embedded "
                "with a different model than the query — re-embed the corpus "
                "with the current model to restore full scoring. "
                "Falling back to trigram-Jaccard.",
                len(emb_a), len(emb_b), a.activity_id, b.activity_id,
            )
            return SemanticSimilarityScorer._trigram_jaccard(text_a, text_b)

        sim = cosine_similarity(emb_a, emb_b)
        return float(max(0.0, min(1.0, sim)))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embedding(self, activity: ActivityCase) -> list[float] | None:
        """Return the stored embedding, computing it on-the-fly if needed.

        **Side-effect**: when the embedding is not already cached and the
        embedder produces one, the result is written back into
        ``activity.metadata["features"]["text_embedding"]``.  This mutation
        is intentional — it makes all subsequent ``score()`` calls for the
        same activity free (no re-encoding).

        Callers must be aware of this write-back if they:

        * Pass frozen or read-only ``ActivityCase`` objects (mutation will
          raise ``AttributeError`` on the dict assignment).
        * Share the same ``ActivityCase`` instance across concurrent threads —
          the write is not protected by a lock and is subject to a TOCTOU
          race if two threads encode the same activity simultaneously.

        For the current single-threaded pipeline neither case applies, but
        both should be revisited before introducing concurrency.
        """
        features: dict = activity.metadata.get("features", {})
        emb = features.get(_EMBEDDING_KEY)

        # Already stored and non-trivial — return immediately, no side-effect.
        if emb is not None and len(emb) > 1:
            return emb

        # Try to compute on-the-fly using the embedder.
        if self._embedder is not None:
            text = activity.cleaned_description or activity.raw_description or ""
            if text:
                emb = self._embedder.encode(text)
                if len(emb) > 1:
                    # Write-back cache: store the embedding so future score()
                    # calls for this activity skip re-encoding entirely.
                    # NOTE: mutates activity.metadata in-place (see docstring).
                    if "features" not in activity.metadata:
                        activity.metadata["features"] = {}
                    activity.metadata["features"][_EMBEDDING_KEY] = emb
                    return emb

        return None
