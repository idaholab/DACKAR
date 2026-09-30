"""
Public factory for the outage duration-uncertainty service.

Usage (minimal)::

    from outage_uncertainty.api.facade import build_duration_uncertainty_service

    service = build_duration_uncertainty_service()
    estimate = service.estimate_activity(query_row={...}, historical_rows=[...])

The factory wires all components together.  Heavy optional dependencies
(textacy, dackar.similarity, NLTK / pywsd) are loaded lazily inside each
component and fall back gracefully when absent.
"""
from __future__ import annotations

from pathlib import Path

from outage_uncertainty.adapters.pandas_repository import PandasActivityRepository
from outage_uncertainty.api.config import AppConfig
from outage_uncertainty.preprocessing.abbreviations import AbbreviationResolver
from outage_uncertainty.preprocessing.cleaners import (
    ActivityCleaner,
    ComponentIdRemover,
    IdentityTransform,
    TextacyPreprocessor,
)
from outage_uncertainty.preprocessing.ollama_disambiguator import LLMAbbreviationDisambiguator
from outage_uncertainty.preprocessing.ollama_embedder import OllamaEmbedder
from outage_uncertainty.preprocessing.spell_checker import DomainSpellChecker
from outage_uncertainty.retrieval.semantic_similarity import (
    EmbeddingSemanticScorer,
    SemanticSimilarityScorer,
)
from outage_uncertainty.preprocessing.feature_builder import ActivityFeatureBuilder
from outage_uncertainty.preprocessing.label_mapper import TaskLabelMapper
from outage_uncertainty.preprocessing.validators import ActivityValidator
from outage_uncertainty.retrieval.context_similarity import ContextSimilarityScorer
from outage_uncertainty.retrieval.dependency_similarity import DependencyPatternScorer
from outage_uncertainty.retrieval.lexical_similarity import LexicalSimilarityScorer
from outage_uncertainty.retrieval.neighbor_selector import NeighborSelector
from outage_uncertainty.retrieval.retrieval_index import HistoricalActivityIndex, LexicalContextPrescorer
from outage_uncertainty.retrieval.semantic_similarity import SemanticSimilarityScorer
from outage_uncertainty.retrieval.similarity_engine import SimilarityAggregator, SimilarityEngine
from outage_uncertainty.services.duration_service import DurationUncertaintyService
from outage_uncertainty.uncertainty.confidence import ConfidenceEstimator
from outage_uncertainty.uncertainty.distribution_fitter import DistributionFitter
from outage_uncertainty.uncertainty.duration_estimator import DurationEstimator
from outage_uncertainty.uncertainty.fallback_policy import FallbackPolicy
from outage_uncertainty.uncertainty.outlier_handler import OutlierHandler
from outage_uncertainty.workflows.activity_ingestion_workflow import ActivityIngestionWorkflow
from outage_uncertainty.workflows.similarity_assessment_workflow import SimilarityAssessmentWorkflow

# Default path: <DACKAR project root>/data/abbreviations.xlsx
# Resolved relative to this file's location so it works regardless of cwd.
# Layout: outage/outage_uncertainty/api/facade.py
#   parents[0]=api  parents[1]=outage_uncertainty  parents[2]=outage
#   parents[3]=dackar  parents[4]=src  parents[5]=DACKAR/
_DEFAULT_ABBREVIATIONS_FILE = str(
    Path(__file__).parents[5] / "data" / "abbreviations.xlsx"
)


def build_duration_uncertainty_service(
    config: AppConfig | None = None,
    *,
    abbreviations_file: str | None = None,
    abbreviations: dict[str, str] | None = None,
    taxonomy_rules: dict | None = None,
) -> DurationUncertaintyService:
    """Build and wire a fully configured :class:`DurationUncertaintyService`.

    Parameters
    ----------
    config:
        Full :class:`AppConfig` instance.  When provided all other keyword
        arguments are ignored (configure via the ``AppConfig`` fields instead).
    abbreviations_file:
        Path to the abbreviations Excel file.  Defaults to the DACKAR
        ``data/abbreviations.xlsx``.  Pass ``""`` to disable file loading.
    abbreviations:
        Extra ``{abbr: full}`` pairs merged on top of the Excel data.
    taxonomy_rules:
        Keyword-to-taxonomy mapping for :class:`TaskLabelMapper`.

    Returns
    -------
    DurationUncertaintyService
        Ready-to-use service instance.
    """
    if config is None:
        config = AppConfig(
            taxonomy_rules=taxonomy_rules or {},
            abbreviations=abbreviations or {},
            abbreviations_file=abbreviations_file,   # None → use default below
        )

    # Resolve the abbreviations file path (None means "use default")
    abbr_file: str | None = config.abbreviations_file
    if abbr_file is None:
        abbr_file = _DEFAULT_ABBREVIATIONS_FILE

    # ------------------------------------------------------------------
    # Ollama embedder (shared by feature builder and semantic scorer)
    # ------------------------------------------------------------------
    embedder: OllamaEmbedder | None = None
    if config.embedding_enabled:
        embedder = OllamaEmbedder(
            model=config.embedding_model,
            base_url=config.ollama_base_url,
            cache_path=config.embedding_cache_path,
        )

    # ------------------------------------------------------------------
    # Text cleaning pipeline
    # ------------------------------------------------------------------
    base_resolver = AbbreviationResolver(
        abbreviations_file=abbr_file if abbr_file != "" else None,
        extra_abbreviations=config.abbreviations or None,
    )
    abbreviation_expander = (
        LLMAbbreviationDisambiguator(
            abbreviation_resolver=base_resolver,
            model=config.llm_model,
            base_url=config.ollama_base_url,
            cache_path=config.llm_disambiguation_cache_path,
        )
        if config.llm_disambiguation_enabled
        else base_resolver
    )

    cleaner = ActivityCleaner(
        component_id_remover=ComponentIdRemover(),
        preprocessor=TextacyPreprocessor(),
        abbreviation_expander=abbreviation_expander,
        spell_checker=(
            DomainSpellChecker(cutoff=config.spell_check_cutoff)
            if config.spell_check_enabled
            else IdentityTransform()
        ),
    )

    # ------------------------------------------------------------------
    # Ingestion workflow
    # ------------------------------------------------------------------
    ingestion = ActivityIngestionWorkflow(
        repository=PandasActivityRepository(),
        cleaner=cleaner,
        label_mapper=TaskLabelMapper(
            config.taxonomy_rules,
            use_defaults=config.taxonomy_use_defaults,
        ),
        feature_builder=ActivityFeatureBuilder(embedder=embedder),
        validator=ActivityValidator(),
    )

    # ------------------------------------------------------------------
    # Similarity engine
    # ------------------------------------------------------------------
    # When embedding is enabled: EmbeddingSemanticScorer (cosine on dense vectors)
    # Otherwise: WordNet/Pawar-Mago scorer with trigram-Jaccard fallback.
    # Weights rebalanced to include the dependency (schedule-neighbourhood)
    # dimension per the PDF allocation:
    #   text (lexical + semantic) ≈ 45%,  context ≈ 35%,  dependency = 20%
    semantic_scorer = (
        EmbeddingSemanticScorer(embedder=embedder)
        if config.embedding_enabled
        else SemanticSimilarityScorer(
            disambiguation_method=config.semantic_disambiguation_method,
            similarity_method=config.semantic_similarity_method,
        )
    )

    similarity_engine = SimilarityEngine(
        lexical_scorer=LexicalSimilarityScorer(),
        semantic_scorer=semantic_scorer,
        context_scorer=ContextSimilarityScorer(),
        aggregator=SimilarityAggregator(weights={
            "lexical":    0.15,
            "semantic":   0.30,
            "context":    0.35,
            "dependency": 0.20,
        }),
        dependency_scorer=DependencyPatternScorer(),
    )

    # ------------------------------------------------------------------
    # Duration estimation
    # ------------------------------------------------------------------
    prescorer = LexicalContextPrescorer(
        text_weight=config.prescorer_text_weight,
    )

    similarity_workflow = SimilarityAssessmentWorkflow(
        index=HistoricalActivityIndex(prescorer=prescorer),
        similarity_engine=similarity_engine,
        prescorer_top_k=config.prescorer_top_k,
        neighbor_selector=NeighborSelector(
            top_k=config.similarity_top_k,
            min_score=config.similarity_min_score,
            warn_below=config.similarity_warn_below,
            weight_exponent=config.similarity_weight_exponent,
        ),
        duration_estimator=DurationEstimator(
            outlier_handler=OutlierHandler(
                strategy=config.outlier_strategy,
                trim_pct=config.outlier_trim_pct,
                mad_scale=config.outlier_mad_scale,
            ),
            fitter=DistributionFitter(),
            confidence_estimator=ConfidenceEstimator(),
            fallback_policy=FallbackPolicy(
                min_support=config.fallback_min_support,
            ),
        ),
    )

    return DurationUncertaintyService(
        ingestion_workflow=ingestion,
        similarity_workflow=similarity_workflow,
    )
