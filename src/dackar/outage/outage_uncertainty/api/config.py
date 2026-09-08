"""
Application configuration for the outage uncertainty service.

All fields have sensible defaults so the service can be constructed with
``AppConfig()`` and work out of the box (using fallback implementations
where heavy dependencies are absent).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AppConfig:
    # ------------------------------------------------------------------
    # Retrieval / similarity
    # ------------------------------------------------------------------
    # ---- Pre-filter (LexicalContextPrescorer / HistoricalActivityIndex) ----
    prescorer_top_k: int = 200
    """
    Number of candidates returned by the cheap pre-filter before the
    expensive full SimilarityEngine runs.  Higher values improve recall
    at the cost of more WordNet calls.
    """

    prescorer_text_weight: float = 0.5
    """
    Share of the pre-filter score attributed to text (token Jaccard).
    The complementary share goes to context (exact-match field average).
    """

    # ---- Final neighbor selection (NeighborSelector) --------------------
    similarity_top_k: int = 30
    """Maximum number of neighbours returned after full scoring."""

    similarity_min_score: float = 0.0
    """
    Hard score floor applied before the top-k cut.  Default 0.0 (disabled).
    Prefer ``similarity_warn_below`` for soft signalling.
    """

    similarity_warn_below: float = 0.4
    """
    If the best match total_score is below this threshold the selector
    flags low coverage, which the estimator surfaces as a warning on the
    ActivityEstimate.
    """

    similarity_weight_exponent: float = 2.0
    """
    α for power-normalised relevance weights: weight ∝ score^α.
    Higher values give stronger preference to top matches (α=1 → linear,
    α→∞ → winner-takes-all).
    """

    # ------------------------------------------------------------------
    # Monte Carlo schedule-risk analysis
    # ------------------------------------------------------------------
    monte_carlo_samples: int = 1000
    """Number of simulation runs for schedule risk analysis."""

    # ------------------------------------------------------------------
    # Taxonomy / label mapping
    # ------------------------------------------------------------------
    taxonomy_rules: dict[str, dict[str, str]] = field(default_factory=dict)
    """
    Keyword-to-taxonomy mapping used by TaskLabelMapper.
    Format::

        {
            "transmitter": {"discipline": "I&C", "task_family": "calibration",
                            "component_family": "transmitter"},
            ...
        }

    These rules are merged *on top of* the built-in vocabulary (when
    ``taxonomy_use_defaults=True``) and take precedence.
    """

    taxonomy_use_defaults: bool = True
    """
    When ``True`` (default) the built-in nuclear-outage taxonomy vocabulary
    (``DEFAULT_TAXONOMY_RULES``) is loaded automatically.  User-supplied
    ``taxonomy_rules`` are merged on top and always take precedence.
    Set to ``False`` to use only the rules in ``taxonomy_rules``.
    """

    # ------------------------------------------------------------------
    # Abbreviation expansion
    # ------------------------------------------------------------------
    abbreviations_file: str | None = None
    """
    Path to the abbreviations Excel file (.xlsx) with columns
    ['Abbreviation', 'Full'].  When ``None`` the facade uses the default
    DACKAR data path (``<project_root>/data/abbreviations.xlsx``).
    Set to an empty string ``""`` to disable file-based loading entirely.
    """

    abbreviations: dict[str, str] = field(default_factory=dict)
    """
    Extra {abbr: full} pairs merged on top of the Excel file.
    Useful for testing or for plant-specific additions not in the master file.
    """

    # ------------------------------------------------------------------
    # Semantic similarity scorer
    # ------------------------------------------------------------------
    semantic_disambiguation_method: str = "simple_lesk"
    """
    Word-sense disambiguation method passed to
    ``dackar.similarity.SentenceSimilarity``.
    Options: ``'simple_lesk'`` (default), ``'original_lesk'``,
    ``'cosine_lesk'``, ``'adapted_lesk'``, ``'max_similarity'``.
    """

    semantic_similarity_method: str = "semantic_similarity_synsets"
    """
    Synset comparison method passed to ``SentenceSimilarity``.
    Default uses harmonic mean of WordNet path similarity and Wu-Palmer.
    """

    # ------------------------------------------------------------------
    # Outlier handling
    # ------------------------------------------------------------------
    outlier_strategy: str = "iqr"
    """
    Strategy used by :class:`OutlierHandler` to separate routine durations
    from extended/disrupted ones.  Options:

    * ``'iqr'`` – upper fence = Q3 + 1.5 × IQR (default).
    * ``'mad'`` – fence = median + ``outlier_mad_scale`` × MAD.
    * ``'trim_symmetric'`` – discard top and bottom ``outlier_trim_pct``
      of samples.
    * ``'keep_all'`` – no separation (single-pool fitting).
    """

    outlier_mad_scale: float = 3.0
    """
    Number of MADs above the median used as the outlier fence when
    ``outlier_strategy='mad'``.
    """

    outlier_trim_pct: float = 0.10
    """
    Fraction of extreme samples removed from each tail when
    ``outlier_strategy='trim_symmetric'``.
    """

    # ------------------------------------------------------------------
    # Ollama integration (shared)
    # ------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    """Base URL of the local Ollama HTTP API server."""

    # ------------------------------------------------------------------
    # Embedding-based semantic similarity (#1)
    # ------------------------------------------------------------------
    embedding_enabled: bool = False
    """
    When ``True`` the :class:`EmbeddingSemanticScorer` (cosine similarity on
    dense vectors) replaces the default WordNet/Pawar-Mago scorer.  Requires
    a locally running Ollama server at ``ollama_base_url``.  Default ``False``
    so the service remains functional without Ollama.
    """

    embedding_model: str = "nomic-embed-text:latest"
    """
    Ollama embedding model.  Benchmarked options:

    * ``"nomic-embed-text:latest"`` — best discrimination ratio (1.415), 274 MB,
      fastest.  Recommended default.
    * ``"mxbai-embed-large:latest"`` — best Precision@5/10 (0.648/0.556),
      669 MB.  Preferred when retrieval ranking quality is the top priority.
    """

    embedding_cache_path: str | None = None
    """
    Optional path to a JSON file for persistent embedding cache.  When set,
    embeddings computed during ingestion are saved to disk and reloaded on the
    next cold start — avoiding redundant Ollama calls for a stable historical
    corpus.  ``None`` means in-memory only.
    """

    # ------------------------------------------------------------------
    # LLM abbreviation disambiguation (#2)
    # ------------------------------------------------------------------
    llm_disambiguation_enabled: bool = False
    """
    When ``True`` the :class:`LLMAbbreviationDisambiguator` is applied after
    rule-based abbreviation expansion to resolve residual uppercase tokens
    using a local Ollama LLM.  Default ``False`` — the rule-based resolver
    alone handles the vast majority of P6 abbreviations.
    """

    llm_model: str = "mistral:latest"
    """
    Ollama generative model used for abbreviation disambiguation.
    ``"mistral:latest"`` (7B) is recommended over ``"llama3.2:latest"`` (3B)
    for nuclear-specific abbreviations.
    """

    llm_disambiguation_cache_path: str | None = None
    """
    Optional path to a JSON file caching (token → expansion) pairs resolved
    by the LLM.  Avoids redundant calls for abbreviations seen in previous
    runs.  ``None`` means in-memory only.
    """

    # ------------------------------------------------------------------
    # Spell checking
    # ------------------------------------------------------------------
    spell_check_enabled: bool = True
    """
    When ``True`` (default) the :class:`DomainSpellChecker` is applied
    after abbreviation expansion to fix residual typos in activity
    descriptions before similarity search.  Set to ``False`` to disable.
    """

    spell_check_cutoff: float = 0.85
    """
    Minimum SequenceMatcher ratio required for a spell correction to be
    accepted.  Default 0.85.  Lower values increase recall at the cost of
    more false corrections; values below 0.80 are not recommended.
    """

    # ------------------------------------------------------------------
    # Hierarchical fallback
    # ------------------------------------------------------------------
    fallback_min_support: int = 3
    """
    Minimum number of historical analogues required at each fallback level
    before the level is accepted.  If no level meets this threshold the
    service falls back to planned duration as the prior.
    """
