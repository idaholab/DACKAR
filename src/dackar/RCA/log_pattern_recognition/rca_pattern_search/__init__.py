"""
rca_pattern_search — RCA Temporal Pattern Matching

Two-stage pipeline:
  Stage 1 (offline): KDE-based episode detection from historical event logs
  Stage 2 (online):  Coarse-to-fine similarity retrieval

Typical usage::

    from rca_pattern_search import SearchConfig, IncidentIndex, PatternSearcher
    from rca_pattern_search import IncidentExtractor

    cfg = SearchConfig()

    # --- Stage 1: build historical index (run offline) ---
    index = IncidentIndex(cfg)
    index.build_from_history(events_df, rho_query=rho, query_duration=dur)
    index.save("/path/to/index")

    # --- Stage 2: query at runtime ---
    index = IncidentIndex.load("/path/to/index", cfg)
    extractor = IncidentExtractor(cfg)
    query_fp = extractor.extract(alarm_log, soe_log, telemetry, "INC_001", t0, t1)
    searcher = PatternSearcher(index, cfg)
    results = searcher.search(query_fp)
"""
from .config import SearchConfig
from .density import EpisodeDetector
from .extractor import IncidentExtractor
from .indexer import IncidentIndex
from .metrics import combined_score, emd_similarity, jaccard, nlcs
from .models import IncidentFingerprint, SearchResult, UnifiedEvent
from .searcher import PatternSearcher

__all__ = [
    # Config
    "SearchConfig",
    # Models
    "IncidentFingerprint",
    "SearchResult",
    "UnifiedEvent",
    # Pipeline classes
    "EpisodeDetector",
    "IncidentExtractor",
    "IncidentIndex",
    "PatternSearcher",
    # Metric functions
    "jaccard",
    "nlcs",
    "emd_similarity",
    "combined_score",
]
