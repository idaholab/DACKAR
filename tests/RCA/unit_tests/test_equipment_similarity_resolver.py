"""
Unit tests for EquipmentSimilarityResolver, EquipmentSimilarityConfig,
and SisterComponent.

Uses a MockEquipmentSpecStore (no live Chroma or Neo4j required).
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

import pytest
from equipment_similarity.equipment_similarity_resolver import (
    NON_EMBEDDING_DISTANCE,
    EquipmentSimilarityConfig,
    EquipmentSimilarityResolver,
    SisterComponent,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_doc(component_id: str, score: float = 0.3, label: Optional[str] = None) -> Any:
    """Build a minimal Document-like object returned by Chroma."""
    doc = MagicMock()
    doc.metadata = {
        "component_id": component_id,
        "_score": score,
        "component_label": label,
    }
    return doc


class MockEquipmentSpecStore:
    """
    Configurable stand-in for EquipmentSpecStore.

    ``fixtures`` is a list of (component_id, score, label) triples.
    ``find_similar`` returns them in order, filtering out ``exclude_ids``.
    """

    def __init__(
        self,
        fixtures: Optional[List[tuple]] = None,
        raise_on_query: bool = False,
    ) -> None:
        self._fixtures = fixtures or []
        self._raise_on_query = raise_on_query
        self.last_query_text: Optional[str] = None
        self.last_exclude_ids: Optional[List[str]] = None

    def find_similar(
        self,
        query_text: str,
        top_k: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> list:
        self.last_query_text = query_text
        self.last_exclude_ids = list(exclude_ids or [])
        if self._raise_on_query:
            raise ValueError("collection not initialised")
        exclude = set(exclude_ids or [])
        docs = [
            _make_doc(cid, score, label)
            for cid, score, label in self._fixtures
            if cid not in exclude
        ]
        return docs[:top_k]


def _minimal_kg_context(
    components: Optional[List[Dict]] = None,
    failure_modes: Optional[List[Dict]] = None,
) -> Dict:
    return {
        "subgraph_id": "sg-test-001",
        "asset_id": "ASSET-A",
        "components": components or [],
        "failure_modes": failure_modes or [],
        "past_events": [],
    }


def _comp(cid: str, label: Optional[str] = None) -> Dict:
    return {"component_id": cid, "component_label": label, "component_type": "pump"}


def _fm(comp_id: str, fm_id: str, mechanism: str = "") -> Dict:
    return {
        "component_id": comp_id,
        "fm_id": fm_id,
        "name": fm_id,
        "failure_mechanism": mechanism,
    }


# ---------------------------------------------------------------------------
# SisterComponent dataclass
# ---------------------------------------------------------------------------

class TestSisterComponent:
    def test_defaults(self):
        s = SisterComponent(component_id="C-001")
        assert s.component_label is None
        assert s.match_type == "spec_embedding"
        assert s.shared_fm_count == 0
        assert s.embedding_score == NON_EMBEDDING_DISTANCE

    def test_to_dict(self):
        s = SisterComponent(
            component_id="C-001",
            component_label="Feed pump A",
            match_type="failure_mode_overlap",
            shared_fm_count=3,
            embedding_score=0.0,
        )
        d = s.to_dict()
        assert d["component_id"] == "C-001"
        assert d["component_label"] == "Feed pump A"
        assert d["match_type"] == "failure_mode_overlap"
        assert d["shared_fm_count"] == 3
        assert d["embedding_score"] == NON_EMBEDDING_DISTANCE

    def test_to_dict_has_all_keys(self):
        s = SisterComponent(component_id="X")
        keys = set(s.to_dict().keys())
        assert keys == {"component_id", "component_label", "match_type", "shared_fm_count", "embedding_score"}


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

class TestEmptyContext:
    def test_empty_kg_context_returns_empty(self):
        resolver = EquipmentSimilarityResolver()
        result = resolver.resolve_similar(["C-001"], _minimal_kg_context())
        assert result == []

    def test_no_failure_modes_returns_empty_fm_tier(self):
        kg = _minimal_kg_context(
            components=[_comp("C-001"), _comp("C-002")],
            failure_modes=[],
        )
        resolver = EquipmentSimilarityResolver(config=EquipmentSimilarityConfig(
            include_fm_overlap=True,
            include_spec_embedding=False,
        ))
        result = resolver.resolve_similar(["C-001"], kg)
        assert result == []

    def test_target_ids_not_in_results(self):
        kg = _minimal_kg_context(
            components=[_comp("C-001"), _comp("C-002")],
            failure_modes=[
                _fm("C-001", "fm-bearing"),
                _fm("C-002", "fm-bearing"),
            ],
        )
        resolver = EquipmentSimilarityResolver(config=EquipmentSimilarityConfig(
            fm_overlap_min_shared=1,
            include_spec_embedding=False,
        ))
        result = resolver.resolve_similar(["C-001", "C-002"], kg)
        assert result == []


# ---------------------------------------------------------------------------
# Tier 2 — failure mode overlap
# ---------------------------------------------------------------------------

class TestFMOverlapTier:
    def _make_resolver(self, min_shared: int = 2) -> EquipmentSimilarityResolver:
        return EquipmentSimilarityResolver(
            spec_store=None,
            config=EquipmentSimilarityConfig(
                fm_overlap_min_shared=min_shared,
                include_fm_overlap=True,
                include_spec_embedding=False,
            ),
        )

    def test_meets_threshold(self):
        kg = _minimal_kg_context(
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-001", "fm-2"),
                _fm("C-002", "fm-1"),
                _fm("C-002", "fm-2"),
            ]
        )
        result = self._make_resolver(min_shared=2).resolve_similar(["C-001"], kg)
        assert len(result) == 1
        assert result[0].component_id == "C-002"
        assert result[0].match_type == "failure_mode_overlap"
        assert result[0].shared_fm_count == 2

    def test_below_threshold_excluded(self):
        kg = _minimal_kg_context(
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-001", "fm-2"),
                _fm("C-002", "fm-1"),  # only 1 shared
            ]
        )
        result = self._make_resolver(min_shared=2).resolve_similar(["C-001"], kg)
        assert result == []

    def test_threshold_one_is_inclusive(self):
        kg = _minimal_kg_context(
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-002", "fm-1"),
            ]
        )
        result = self._make_resolver(min_shared=1).resolve_similar(["C-001"], kg)
        assert len(result) == 1

    def test_multiple_targets_union_of_fms(self):
        """Sister should qualify if it shares FMs with ANY target component."""
        kg = _minimal_kg_context(
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-002", "fm-2"),
                _fm("C-003", "fm-1"),
                _fm("C-003", "fm-2"),
            ]
        )
        # Targets are C-001 and C-002; C-003 shares 1 FM with each = 2 total
        result = self._make_resolver(min_shared=2).resolve_similar(["C-001", "C-002"], kg)
        assert len(result) == 1
        assert result[0].component_id == "C-003"
        assert result[0].shared_fm_count == 2

    def test_sorted_by_shared_count_descending(self):
        kg = _minimal_kg_context(
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-001", "fm-2"),
                _fm("C-001", "fm-3"),
                _fm("C-002", "fm-1"),
                _fm("C-002", "fm-2"),
                _fm("C-002", "fm-3"),  # 3 shared
                _fm("C-003", "fm-1"),
                _fm("C-003", "fm-2"),  # 2 shared
            ]
        )
        result = self._make_resolver(min_shared=2).resolve_similar(["C-001"], kg)
        assert len(result) == 2
        assert result[0].component_id == "C-002"  # 3 shared → first
        assert result[1].component_id == "C-003"  # 2 shared → second

    def test_alternate_fm_field_names(self):
        """Supports applies_to_component_id + name field variants."""
        kg = _minimal_kg_context(
            failure_modes=[
                {"applies_to_component_id": "C-001", "name": "bearing_wear"},
                {"applies_to_component_id": "C-002", "name": "bearing_wear"},
                {"applies_to_component_id": "C-001", "name": "seal_leak"},
                {"applies_to_component_id": "C-002", "name": "seal_leak"},
            ]
        )
        result = self._make_resolver(min_shared=2).resolve_similar(["C-001"], kg)
        assert len(result) == 1
        assert result[0].component_id == "C-002"

    def test_fm_overlap_disabled(self):
        kg = _minimal_kg_context(
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-002", "fm-1"),
            ]
        )
        resolver = EquipmentSimilarityResolver(config=EquipmentSimilarityConfig(
            include_fm_overlap=False,
            include_spec_embedding=False,
        ))
        result = resolver.resolve_similar(["C-001"], kg)
        assert result == []

    def test_component_label_populated_from_kg_context(self):
        kg = _minimal_kg_context(
            components=[_comp("C-002", label="Aux feed pump B")],
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-001", "fm-2"),
                _fm("C-002", "fm-1"),
                _fm("C-002", "fm-2"),
            ],
        )
        result = self._make_resolver(min_shared=2).resolve_similar(["C-001"], kg)
        assert result[0].component_label == "Aux feed pump B"


# ---------------------------------------------------------------------------
# Tier 3 — spec embedding
# ---------------------------------------------------------------------------

class TestSpecEmbeddingTier:
    def _kg_with_target(self) -> Dict:
        return _minimal_kg_context(
            components=[
                {"component_id": "C-001", "component_label": "Main feed pump", "component_type": "centrifugal pump"},
            ],
            failure_modes=[
                {"component_id": "C-001", "name": "bearing_wear", "failure_mechanism": "fatigue"},
            ],
        )

    def test_basic_embedding_match(self):
        store = MockEquipmentSpecStore(fixtures=[("C-002", 0.25, "Aux pump B")])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                include_fm_overlap=False,
                include_spec_embedding=True,
                embedding_min_score=0.8,
            ),
        )
        result = resolver.resolve_similar(["C-001"], self._kg_with_target())
        assert len(result) == 1
        assert result[0].component_id == "C-002"
        assert result[0].match_type == "spec_embedding"
        assert result[0].embedding_score == pytest.approx(0.25)

    def test_score_above_threshold_excluded(self):
        """Documents with score > embedding_min_score should be filtered out."""
        store = MockEquipmentSpecStore(fixtures=[("C-002", 0.9, None)])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                include_fm_overlap=False,
                include_spec_embedding=True,
                embedding_min_score=0.8,
            ),
        )
        result = resolver.resolve_similar(["C-001"], self._kg_with_target())
        assert result == []

    def test_score_at_threshold_included(self):
        """Score exactly equal to threshold is included (threshold is inclusive upper bound)."""
        store = MockEquipmentSpecStore(fixtures=[("C-002", 0.8, None)])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                include_fm_overlap=False,
                include_spec_embedding=True,
                embedding_min_score=0.8,
            ),
        )
        result = resolver.resolve_similar(["C-001"], self._kg_with_target())
        assert len(result) == 1
        assert result[0].component_id == "C-002"

    def test_target_ids_excluded_from_store_query(self):
        store = MockEquipmentSpecStore(fixtures=[
            ("C-001", 0.1, "Main feed pump"),   # target — should be excluded
            ("C-002", 0.25, "Aux pump B"),
        ])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                include_fm_overlap=False,
                include_spec_embedding=True,
            ),
        )
        result = resolver.resolve_similar(["C-001"], self._kg_with_target())
        ids = [s.component_id for s in result]
        assert "C-001" not in ids

    def test_exclude_ids_passed_to_store(self):
        store = MockEquipmentSpecStore(fixtures=[("C-002", 0.2, None)])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(include_fm_overlap=False),
        )
        resolver.resolve_similar(["C-001"], self._kg_with_target())
        assert "C-001" in (store.last_exclude_ids or [])

    def test_uninitialized_store_returns_empty(self):
        """ValueError from store (collection not yet populated) → graceful empty."""
        store = MockEquipmentSpecStore(raise_on_query=True)
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(include_fm_overlap=False),
        )
        # Should not raise
        result = resolver.resolve_similar(["C-001"], self._kg_with_target())
        assert result == []

    def test_spec_embedding_disabled_when_store_is_none(self):
        resolver = EquipmentSimilarityResolver(
            spec_store=None,
            config=EquipmentSimilarityConfig(include_spec_embedding=True),
        )
        # No store → Tier 3 silently skipped
        result = resolver.resolve_similar(["C-001"], self._kg_with_target())
        assert result == []

    def test_query_text_built_from_kg_context(self):
        store = MockEquipmentSpecStore(fixtures=[])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(include_fm_overlap=False),
        )
        resolver.resolve_similar(["C-001"], self._kg_with_target())
        assert store.last_query_text is not None
        assert "Main feed pump" in store.last_query_text
        assert "bearing_wear" in store.last_query_text

    def test_query_text_empty_skips_store(self):
        """No matching target components in kg_context → no store query."""
        store = MockEquipmentSpecStore(fixtures=[("C-002", 0.2, None)])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(include_fm_overlap=False),
        )
        # kg_context has no components matching the target IDs
        kg = _minimal_kg_context(components=[], failure_modes=[])
        result = resolver.resolve_similar(["C-001"], kg)
        # Store should not have been queried (or returns empty if queried with empty text)
        assert result == []


# ---------------------------------------------------------------------------
# Merge — component in both tiers
# ---------------------------------------------------------------------------

class TestMergeTiers:
    def test_component_in_both_tiers_gets_combined_match_type(self):
        store = MockEquipmentSpecStore(fixtures=[("C-002", 0.3, "Aux pump B")])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                fm_overlap_min_shared=1,
                include_fm_overlap=True,
                include_spec_embedding=True,
                embedding_min_score=0.8,
            ),
        )
        kg = _minimal_kg_context(
            components=[
                {"component_id": "C-001", "component_label": "Main pump", "component_type": "pump"},
                {"component_id": "C-002", "component_label": "Aux pump B", "component_type": "pump"},
            ],
            failure_modes=[
                _fm("C-001", "fm-bearing"),
                _fm("C-002", "fm-bearing"),
            ],
        )
        result = resolver.resolve_similar(["C-001"], kg)
        assert len(result) == 1
        sister = result[0]
        assert sister.component_id == "C-002"
        assert sister.match_type == "fm_overlap+spec_embedding"
        assert sister.shared_fm_count == 1
        assert sister.embedding_score == pytest.approx(0.3)

    def test_fm_only_match_has_correct_type(self):
        store = MockEquipmentSpecStore(fixtures=[])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                fm_overlap_min_shared=1,
                include_fm_overlap=True,
                include_spec_embedding=True,
            ),
        )
        kg = _minimal_kg_context(
            failure_modes=[
                _fm("C-001", "fm-bearing"),
                _fm("C-002", "fm-bearing"),
            ]
        )
        result = resolver.resolve_similar(["C-001"], kg)
        assert any(s.match_type == "failure_mode_overlap" for s in result)

    def test_embedding_only_match_has_correct_type(self):
        store = MockEquipmentSpecStore(fixtures=[("C-002", 0.2, None)])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                fm_overlap_min_shared=5,  # very high → no FM overlap
                include_fm_overlap=True,
                include_spec_embedding=True,
                embedding_min_score=0.8,
            ),
        )
        kg = _minimal_kg_context(
            components=[
                {"component_id": "C-001", "component_label": "Target pump", "component_type": "pump"},
            ],
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-002", "fm-1"),  # only 1 shared, < threshold of 5
            ],
        )
        result = resolver.resolve_similar(["C-001"], kg)
        emb_matches = [s for s in result if s.component_id == "C-002"]
        assert len(emb_matches) == 1
        assert emb_matches[0].match_type == "spec_embedding"

    def test_no_duplicates_in_result(self):
        store = MockEquipmentSpecStore(fixtures=[
            ("C-002", 0.2, "Aux pump"),
            ("C-003", 0.4, "Other pump"),
        ])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                fm_overlap_min_shared=1,
                include_fm_overlap=True,
                include_spec_embedding=True,
                embedding_min_score=0.8,
            ),
        )
        kg = _minimal_kg_context(
            components=[
                {"component_id": "C-001", "component_label": "Main pump", "component_type": "pump"},
            ],
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-002", "fm-1"),
            ],
        )
        result = resolver.resolve_similar(["C-001"], kg)
        ids = [s.component_id for s in result]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

class TestSorting:
    def test_sorted_by_embedding_score_ascending(self):
        """Lower embedding distance = more similar → should appear first."""
        store = MockEquipmentSpecStore(fixtures=[
            ("C-002", 0.5, None),
            ("C-003", 0.1, None),
        ])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                include_fm_overlap=False,
                include_spec_embedding=True,
                embedding_min_score=0.8,
            ),
        )
        kg = _minimal_kg_context(
            components=[
                {"component_id": "C-001", "component_label": "Main pump", "component_type": "pump"},
            ],
        )
        result = resolver.resolve_similar(["C-001"], kg)
        scores = [s.embedding_score for s in result]
        assert scores == sorted(scores)

    def test_fm_tie_broken_by_shared_count_descending(self):
        """When embedding scores are equal, higher shared_fm_count wins."""
        resolver = EquipmentSimilarityResolver(
            spec_store=None,
            config=EquipmentSimilarityConfig(
                fm_overlap_min_shared=1,
                include_fm_overlap=True,
                include_spec_embedding=False,
            ),
        )
        kg = _minimal_kg_context(
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-001", "fm-2"),
                _fm("C-001", "fm-3"),
                _fm("C-002", "fm-1"),  # 1 shared
                _fm("C-003", "fm-1"),
                _fm("C-003", "fm-2"),
                _fm("C-003", "fm-3"),  # 3 shared
            ]
        )
        result = resolver.resolve_similar(["C-001"], kg)
        assert result[0].component_id == "C-003"
        assert result[1].component_id == "C-002"

    def test_embedding_matches_rank_ahead_of_non_embedding_matches(self):
        store = MockEquipmentSpecStore(fixtures=[("C-003", 0.2, "Emb-only pump")])
        resolver = EquipmentSimilarityResolver(
            spec_store=store,
            config=EquipmentSimilarityConfig(
                fm_overlap_min_shared=1,
                include_fm_overlap=True,
                include_spec_embedding=True,
                embedding_min_score=0.8,
            ),
        )
        kg = _minimal_kg_context(
            components=[
                {"component_id": "C-001", "component_label": "Target pump", "component_type": "pump"},
            ],
            failure_modes=[
                _fm("C-001", "fm-1"),
                _fm("C-002", "fm-1"),  # FM-overlap only -> NON_EMBEDDING_DISTANCE
            ],
        )
        result = resolver.resolve_similar(["C-001"], kg)
        assert [s.component_id for s in result] == ["C-003", "C-002"]
        assert result[0].embedding_score < result[1].embedding_score


# ---------------------------------------------------------------------------
# Query text builder
# ---------------------------------------------------------------------------

class TestBuildQueryText:
    def _resolver(self) -> EquipmentSimilarityResolver:
        return EquipmentSimilarityResolver()

    def test_equipment_label_in_text(self):
        kg = _minimal_kg_context(
            components=[{"component_id": "C-001", "component_label": "Main coolant pump", "component_type": "centrifugal pump"}],
            failure_modes=[_fm("C-001", "bearing_wear")],
        )
        text = self._resolver()._build_query_text(["C-001"], kg)
        assert "Main coolant pump" in text

    def test_component_type_in_text(self):
        kg = _minimal_kg_context(
            components=[{"component_id": "C-001", "component_label": "X", "component_type": "centrifugal pump"}],
        )
        text = self._resolver()._build_query_text(["C-001"], kg)
        assert "centrifugal pump" in text

    def test_failure_modes_in_text(self):
        kg = _minimal_kg_context(
            components=[{"component_id": "C-001", "component_label": "X", "component_type": "pump"}],
            failure_modes=[
                _fm("C-001", "bearing_wear", "fatigue"),
                _fm("C-001", "seal_leak", "corrosion"),
            ],
        )
        text = self._resolver()._build_query_text(["C-001"], kg)
        assert "bearing_wear" in text
        assert "seal_leak" in text
        assert "fatigue" in text
        assert "corrosion" in text

    def test_non_target_fms_excluded(self):
        kg = _minimal_kg_context(
            components=[{"component_id": "C-001", "component_label": "Pump A", "component_type": "pump"}],
            failure_modes=[
                _fm("C-001", "bearing_wear"),
                _fm("C-002", "other_fm"),  # not a target
            ],
        )
        text = self._resolver()._build_query_text(["C-001"], kg)
        assert "other_fm" not in text

    def test_empty_kg_context_returns_empty_string(self):
        text = self._resolver()._build_query_text(["C-001"], _minimal_kg_context())
        assert text.strip() == ""

    def test_duplicate_fm_names_deduplicated(self):
        kg = _minimal_kg_context(
            components=[{"component_id": "C-001", "component_label": "X", "component_type": "pump"}],
            failure_modes=[
                _fm("C-001", "bearing_wear"),
                _fm("C-001", "bearing_wear"),  # duplicate
            ],
        )
        text = self._resolver()._build_query_text(["C-001"], kg)
        # Should not have duplicated "bearing_wear, bearing_wear"
        assert text.count("bearing_wear") == 1
