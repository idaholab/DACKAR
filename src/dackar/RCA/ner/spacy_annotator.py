"""
spacy_annotator.py
──────────────────────────────────────────────────────────────────────────────
Thin wrapper that runs the six plant-specific spaCy pipeline components
(Temporal, TemporalRelation, TemporalAttribute, Location, Conjecture, Unit)
on arbitrary text and returns a structured SpacyAnnotationResult.

Used in two tiers of the RCA workflow:
  Tier 1 (indexing) — called from ner_adapter to enrich NERSeed
  Tier 2 (scoring)  — injected into ChromaEvidenceRetriever to annotate
                       each retrieved snippet inside _assess_hit_against_candidate
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import spacy

# Import pipeline modules so their @Language.factory decorators fire once,
# making the factory names available to nlp.add_pipe().
import dackar.pipelines.TemporalEntity          # noqa: F401  registers "Temporal"
import dackar.pipelines.TemporalRelationEntity   # noqa: F401  registers "temporal_relation_entity"
import dackar.pipelines.TemporalAttributeEntity  # noqa: F401  registers "temporal_attribute_entity"
import dackar.pipelines.LocationEntity           # noqa: F401  registers "location_entity"
import dackar.pipelines.ConjectureEntity         # noqa: F401  registers "conjecture_entity"
import dackar.pipelines.UnitEntity               # noqa: F401  registers "unit_entity"

# ---------------------------------------------------------------------------
# Duration → hours conversion table
# ---------------------------------------------------------------------------

_TO_HOURS: Dict[str, float] = {
    "ms": 1 / 3_600_000,
    "millisecond": 1 / 3_600_000, "milliseconds": 1 / 3_600_000,
    "sec": 1 / 3600, "secs": 1 / 3600,
    "second": 1 / 3600, "seconds": 1 / 3600,
    "min": 1 / 60, "mins": 1 / 60,
    "minute": 1 / 60, "minutes": 1 / 60,
    "hr": 1.0, "hrs": 1.0,
    "hour": 1.0, "hours": 1.0,
    "day": 24.0, "days": 24.0,
    "week": 168.0, "weeks": 168.0,
    "month": 730.0, "months": 730.0,
    "year": 8760.0, "years": 8760.0,
}

_DURATION_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*[-\s]?"
    r"(milliseconds?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)


def _parse_lag_hours(texts: List[str]) -> Optional[float]:
    """Return the first parseable duration in hours from a list of temporal ref texts.

    Returns None if no duration pattern is found.
    """
    for text in texts:
        m = _DURATION_RE.search(text)
        if m:
            value = float(m.group(1))
            factor = _TO_HOURS.get(m.group(2).lower())
            if factor is not None:
                return round(value * factor, 4)
    return None


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class SpacyAnnotationResult:
    """Structured output from SpacyAnnotator.annotate()."""

    measurements: List[Dict[str, Any]] = field(default_factory=list)
    """Physical measurements: {value, unit, entity_type, text}."""

    temporal_refs: List[str] = field(default_factory=list)
    """Absolute dates and durations: 'March 14 2025', '48 hours'."""

    temporal_relations: List[Dict[str, str]] = field(default_factory=list)
    """Ordering words: {text, sub_label} where sub_label is one of
    temporal_relation_order | temporal_relation_reverse_order |
    temporal_relation_concurrency."""

    temporal_qualifiers: List[str] = field(default_factory=list)
    """Fuzzy temporal qualifiers: 'approximately', 'roughly'."""

    locations: List[Dict[str, str]] = field(default_factory=list)
    """Spatial terms: {text, sub_label} where sub_label is one of
    location_proximity | location_up | location_down."""

    conjectures: List[str] = field(default_factory=list)
    """Epistemic hedge markers: 'possibly', 'likely', 'suspected'."""

    lag_hours: Optional[float] = None
    """First parseable duration from temporal_refs, converted to hours."""

    lag_is_approximate: bool = False
    """True when temporal_qualifiers are present alongside a lag_hours value."""

    # ------------------------------------------------------------------
    # Derived helpers used downstream
    # ------------------------------------------------------------------

    def conjecture_fraction(self) -> float:
        """Ratio of conjecture markers to total semantic signals.

        Used as a hedge-density proxy: high values indicate the source text
        is speculative rather than confirmatory.
        """
        total = max(1, len(self.temporal_refs) + len(self.temporal_relations) + len(self.conjectures))
        return round(len(self.conjectures) / total, 4)

    def dominant_temporal_relation(self) -> Optional[str]:
        """Most common temporal ordering label mapped to the schema enum.

        Returns 'precedes', 'follows', 'simultaneous', or None.
        """
        if not self.temporal_relations:
            return None
        _map = {
            "temporal_relation_order": "follows",
            "temporal_relation_reverse_order": "precedes",
            "temporal_relation_concurrency": "simultaneous",
        }
        counts: Dict[str, int] = {}
        for r in self.temporal_relations:
            mapped = _map.get(r["sub_label"])
            if mapped:
                counts[mapped] = counts.get(mapped, 0) + 1
        return max(counts, key=counts.__getitem__) if counts else None


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class SpacyAnnotator:
    """
    One-time-initialised wrapper for the six plant-specific spaCy components.

    Instantiate once per process (model load + pipe setup are expensive) and
    share the same instance across Tier 1 (ner_adapter) and Tier 2
    (evidence_retriever).

    Args:
        nlp_model: spaCy model name.  Must include an NER component so that
            TemporalEntity's Matcher patterns that rely on ENT_TYPE DATE/TIME
            fire correctly.  Defaults to 'en_core_web_sm'.
    """

    _UNIT_LABEL = "unit"
    _TEMPORAL_LABEL = "Temporal"
    _TEMPORAL_RELATION_LABELS = frozenset({
        "temporal_relation_order",
        "temporal_relation_reverse_order",
        "temporal_relation_concurrency",
    })
    _TEMPORAL_ATTRIBUTE_LABEL = "temporal_attribute"
    _LOCATION_LABELS = frozenset({
        "location_proximity",
        "location_up",
        "location_down",
    })
    _CONJECTURE_LABEL = "conjecture"

    def __init__(self, nlp_model: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(nlp_model)
        self.nlp.add_pipe("Temporal", last=True)
        self.nlp.add_pipe("temporal_relation_entity", last=True)
        self.nlp.add_pipe("temporal_attribute_entity", last=True)
        self.nlp.add_pipe("location_entity", last=True)
        self.nlp.add_pipe("conjecture_entity", last=True)
        self.nlp.add_pipe("unit_entity", last=True)

    def annotate(self, text: str) -> SpacyAnnotationResult:
        """Run all six components on *text* and return a structured result.

        Args:
            text: Arbitrary chunk or snippet text.

        Returns:
            :class:`SpacyAnnotationResult` with all signal buckets populated.
        """
        if not text or not text.strip():
            return SpacyAnnotationResult()

        doc = self.nlp(text)

        measurements: List[Dict[str, Any]] = []
        temporal_refs: List[str] = []
        temporal_relations: List[Dict[str, str]] = []
        temporal_qualifiers: List[str] = []
        locations: List[Dict[str, str]] = []
        conjectures: List[str] = []

        for ent in doc.ents:
            label = ent.label_
            span_text = ent.text.strip()

            if label == self._UNIT_LABEL:
                m = ent._.measurement
                if m:
                    measurements.append({**m, "text": span_text})
            elif label == self._TEMPORAL_LABEL:
                temporal_refs.append(span_text)
            elif label in self._TEMPORAL_RELATION_LABELS:
                temporal_relations.append({"text": span_text, "sub_label": label})
            elif label == self._TEMPORAL_ATTRIBUTE_LABEL:
                temporal_qualifiers.append(span_text)
            elif label in self._LOCATION_LABELS:
                locations.append({"text": span_text, "sub_label": label})
            elif label == self._CONJECTURE_LABEL:
                conjectures.append(span_text)

        lag_hours = _parse_lag_hours(temporal_refs)
        lag_is_approximate = bool(temporal_qualifiers) and lag_hours is not None

        return SpacyAnnotationResult(
            measurements=measurements,
            temporal_refs=temporal_refs,
            temporal_relations=temporal_relations,
            temporal_qualifiers=temporal_qualifiers,
            locations=locations,
            conjectures=conjectures,
            lag_hours=lag_hours,
            lag_is_approximate=lag_is_approximate,
        )


def build_spacy_annotator(nlp_model: str = "en_core_web_sm") -> SpacyAnnotator:
    """Factory function — initialise a :class:`SpacyAnnotator` and return it.

    Centralises model selection so callers don't need to import SpacyAnnotator
    directly.

    Args:
        nlp_model: spaCy model name (default: 'en_core_web_sm').

    Returns:
        Configured and ready :class:`SpacyAnnotator` instance.
    """
    return SpacyAnnotator(nlp_model=nlp_model)
