"""
Stage A — Activity Intake Processor.

Responsibilities:
    1. Clean and normalize the raw activity description.
    2. Expand abbreviations and shorthand using the nuclear abbreviation dictionary.
    3. Run the NER pipeline to extract component, system, action, failure-mode,
       tag-ID, WO-reference, and CR-reference entities.
    4. Resolve extracted entity mentions to canonical component / system IDs.
    5. Classify the emergence type (truly_unplanned, scope_expansion,
       regulatory_driven, schedule_optimization).
    6. Detect regulatory constraints (TS surveillance, NRC commitment, CAP
       commitment, hold points, ALARA).
    7. Compute data quality score and unknown abbreviation rate.

Output schema: outage/schemas/activity_intake_result.json

Reuse targets:
    outage_uncertainty.preprocessing.cleaners          → _clean_description()
    outage_uncertainty.preprocessing.abbreviations     → _expand_abbreviations()
    outage_uncertainty.preprocessing.nuclear_abbreviations
    outage_uncertainty.preprocessing.label_mapper      → _classify_labels()
    outage_uncertainty.preprocessing.feature_builder
    RCA.ner.hybrid_ner.pipeline.HybridNERPipeline      → _run_ner()
    RCA.ner.spacy_annotator.SpacyAnnotator             → _run_ner() (temporal, measurements)
    RCA.ner.entity_normalizer.EntityNormalizer         → _resolve_references()
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# Module-level imports for preprocessing — tried once at import time so that
# _clean_description() and _expand_abbreviations() don't re-try on every call.
# ---------------------------------------------------------------------------
try:
    from outage_uncertainty.preprocessing.cleaners import (
        ComponentIdRemover as _ComponentIdRemover,
        TextacyPreprocessor as _TextacyPreprocessor,
        IdentityTransform as _IdentityTransform,
    )
    _CLEANERS_AVAILABLE = True
except ImportError:
    _CLEANERS_AVAILABLE = False

try:
    from outage_uncertainty.preprocessing.abbreviations import (
        AbbreviationResolver as _AbbreviationResolver,
    )
    _ABBR_RESOLVER_AVAILABLE = True
except ImportError:
    _ABBR_RESOLVER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level keyword sets for rule-based classifiers
# ---------------------------------------------------------------------------

# Emergence type: regulatory_driven
_REGULATORY_KEYWORDS_RE = re.compile(
    r"\b(TS\s*[\d.]+|technical\s+specification|LCO\s*[\d.]+|limiting\s+condition"
    r"|NRC|ALARA|CAP\b|corrective\s+action\s+program|surveillance|10\s*CFR"
    r"|operability\s+determination|hold\s+point|quality\s+hold"
    r"|mode\s+change|entry\s+condition)\b",
    re.IGNORECASE,
)

# Emergence type: scope_expansion — typically tied to an existing WO
_SCOPE_KEYWORDS_RE = re.compile(
    r"\b(additional\s+scope|add\s+scope|scope\s+(change|expansion|addition)"
    r"|additional\s+work|extend\s+scope|supplemental|augment"
    r"|while\s+we\s+(have|are)|while\s+in\s+the\s+area|opportunistic)\b",
    re.IGNORECASE,
)

# Emergence type: schedule_optimization — no failure language
_SCHEDULE_OPT_KEYWORDS_RE = re.compile(
    r"\b(reschedule|advance\s+schedule|pull\s+(ahead|in)|optimize\s+schedule"
    r"|schedule\s+(change|optimization|advancement)"
    r"|move\s+up|pull\s+forward)\b",
    re.IGNORECASE,
)

# Emergence type: truly_unplanned — degradation / failure language
_DEGRADATION_KEYWORDS_RE = re.compile(
    r"\b(leak(ing)?|failure|failed|failing|damage[d]?|degraded|degradation"
    r"|broken|inoperable|inoperability|abnormal|emergency|corrective"
    r"|deficiency|defect|unusual|unexpected|unplanned|discovery|finding"
    r"|alarm|trip(ped)?|spurious|vibration|noise|smoke|sparks?|crack[s]?)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Regex patterns for entity extraction
# ---------------------------------------------------------------------------

# Instrument/equipment tag IDs: up to 4 uppercase letters, hyphen, 2–7 digits,
# optional trailing letter  (e.g. PT-1234, MOV-4567A, 1A-RHR-PP-003)
_TAG_ID_RE = re.compile(
    r"\b(?:[0-9][A-Z]-)?[A-Z]{1,4}-\d{2,7}[A-Z]?\b"
)

# Work-order references: WO or WC, optional separator, 4+ digits
_WO_REF_RE = re.compile(r"\bW[OC]\s*[-#]?\s*(\d{4,})\b", re.IGNORECASE)

# Condition-report references: CR, optional separator, 4+ digits
_CR_REF_RE = re.compile(r"\bCR\s*[-#]?\s*(\d{4,})\b", re.IGNORECASE)

# Unknown-abbreviation candidates: isolated ALL-CAPS tokens of 2–8 characters
_ABBR_CANDIDATE_RE = re.compile(r"\b[A-Z]{2,8}\b")

# Common English words that happen to be all-caps after normalisation — not abbreviations
_COMMON_CAPS = frozenset([
    "IN", "IS", "IT", "TO", "OF", "ON", "AS", "AT", "BE", "BY", "DO",
    "GO", "HE", "IF", "ME", "MY", "NO", "OR", "SO", "UP", "US", "WE",
    "AND", "ARE", "BUT", "CAN", "FOR", "GET", "HAS", "HAD", "HIM", "HIS",
    "HOW", "ITS", "LET", "NOT", "NOW", "OLD", "OUR", "OUT", "OWN", "PUT",
    "SAY", "SHE", "THE", "TOO", "TWO", "USE", "WAS", "WAY", "WHO", "WHY",
    "WITH", "WILL", "FROM", "HAVE", "THAT", "THIS", "WHEN", "WERE", "THEY",
    "BEEN", "INTO", "SOME", "THEN", "THAN", "ALSO", "EACH", "MOST",
    "OVER", "SUCH", "TAKE", "THEM", "WELL", "WHAT",
])

# ---------------------------------------------------------------------------
# Regulatory constraint detection patterns
# Each tuple: (compiled_pattern, driver_type, defer_prohibited)
# ---------------------------------------------------------------------------

_REGULATORY_PATTERNS: List[Tuple[re.Pattern, str, bool]] = [
    (re.compile(r"\bTS\s*[\d.]+\b", re.IGNORECASE),
     "technical_specification", True),
    (re.compile(r"\btechnical\s+specification\b", re.IGNORECASE),
     "technical_specification", True),
    (re.compile(r"\bLCO\s*[\d.]+\b", re.IGNORECASE),
     "limiting_condition_for_operation", True),
    (re.compile(r"\blimiting\s+condition\s+for\s+operation\b", re.IGNORECASE),
     "limiting_condition_for_operation", True),
    (re.compile(r"\bNRC\b"),
     "nrc_commitment", True),
    (re.compile(r"\b10\s*CFR\b", re.IGNORECASE),
     "nrc_regulation", True),
    (re.compile(r"\bALARA\b"),
     "alara_requirement", False),
    (re.compile(r"\bCAP\b"),
     "corrective_action_program", False),
    (re.compile(r"\bsurveillance\b", re.IGNORECASE),
     "surveillance_requirement", True),
    (re.compile(r"\boperability\s+determination\b", re.IGNORECASE),
     "operability_determination", True),
    (re.compile(r"\bhold\s+point\b", re.IGNORECASE),
     "hold_point", True),
    (re.compile(r"\bmode\s+(change|entry|exit)\b", re.IGNORECASE),
     "mode_change_constraint", True),
]

# Source confidence mapping (from activity source_system field)
_SOURCE_CONFIDENCE: Dict[str, float] = {
    "maximo": 0.90,
    "primavera": 0.85,
    "p6": 0.85,
    "sap": 0.85,
    "manual": 0.55,
    "unknown": 0.40,
}


@dataclass
class ActivityIntakeConfig:
    """Configuration for Stage A."""

    ner_generator_mode: str = "anchored_np"
    """NER pipeline generator mode. One of: gazetteer_only, anchored_np, full_np.
    anchored_np is the recommended default (precision/recall balance)."""

    np_score_threshold: float = 0.65
    """Minimum score for noun-phrase NER candidates."""

    unknown_abbreviation_rate_warning: float = 0.25
    """Fraction of unresolved tokens above which downstream NER results are
    flagged as unreliable (WBS exit criterion 4.7)."""

    abbreviation_dict_path: Optional[Path] = None
    """Path to the plant-specific abbreviation dictionary (CSV or JSON).
    Falls back to the built-in nuclear abbreviations if None."""

    taxonomy_rules_path: Optional[Path] = None
    """Path to custom taxonomy rules (YAML/JSON).
    Merged on top of DEFAULT_TAXONOMY_RULES if provided."""

    gazetteer_path: Optional[Path] = None
    """Path to the gazetteer Excel file for HybridNERPipeline."""

    regulatory_keywords_path: Optional[Path] = None
    """Path to a file mapping keyword patterns to regulatory driver types.
    Used by _detect_regulatory_constraints(). Falls back to built-in list."""

    entity_normalizer_token_overlap_threshold: float = 0.60
    """Minimum Jaccard token-overlap score for EntityNormalizer phase-1 match."""


class ActivityIntakeProcessor:
    """Concrete Stage A implementation.

    All NLP backends are injected to keep this class testable without a live
    model or dictionary file.  Pass None to defer injection until a backend
    is available; calls to process() will raise NotImplementedError.

    Args:
        config: Stage configuration.
        text_cleaner: Callable or object with a clean(text) → str method.
                      Use outage_uncertainty.preprocessing.cleaners.TextCleaner.
        abbreviation_expander: Callable or object with an expand(text) → str method.
                               Use outage_uncertainty.preprocessing.abbreviations.AbbreviationExpander.
        ner_pipeline: HybridNERPipeline instance from RCA.ner.hybrid_ner.pipeline.
                      Must be pre-built with the outage gazetteer and schema.
        spacy_annotator: SpacyAnnotator from RCA.ner.spacy_annotator.
        entity_normalizer: EntityNormalizer from RCA.ner.entity_normalizer.
        label_mapper: TaskLabelMapper from outage_uncertainty.preprocessing.label_mapper.
    """

    def __init__(
        self,
        config: Optional[ActivityIntakeConfig] = None,
        *,
        text_cleaner=None,
        abbreviation_expander=None,
        ner_pipeline=None,
        spacy_annotator=None,
        entity_normalizer=None,
        label_mapper=None,
    ) -> None:
        self.config = config or ActivityIntakeConfig()
        self.text_cleaner = text_cleaner
        self.abbreviation_expander = abbreviation_expander
        self.ner_pipeline = ner_pipeline
        self.spacy_annotator = spacy_annotator
        self.entity_normalizer = entity_normalizer
        self.label_mapper = label_mapper

        # Fallback cleaning chain — used when text_cleaner is not injected.
        # Built once at init time from the preprocessing module so that
        # _clean_description() has no conditional import logic.
        if _CLEANERS_AVAILABLE:
            self._fallback_id_remover = _ComponentIdRemover()
            self._fallback_preprocessor = _TextacyPreprocessor()
        else:
            self._fallback_id_remover = None  # plain whitespace collapse below
            self._fallback_preprocessor = None

        # Fallback abbreviation resolver — used when abbreviation_expander is not injected.
        if _ABBR_RESOLVER_AVAILABLE:
            abbr_file = (
                str(self.config.abbreviation_dict_path)
                if self.config.abbreviation_dict_path
                else None
            )
            self._fallback_abbr_resolver = _AbbreviationResolver(abbreviations_file=abbr_file)
        else:
            self._fallback_abbr_resolver = None  # passthrough below

    # ── Protocol method ───────────────────────────────────────────────────────

    def process(
        self,
        emergent_activity: JsonDict,
        run_context: JsonDict,
    ) -> JsonDict:
        """Execute Stage A for one emergent activity.

        Returns:
            ActivityIntakeResult artifact conforming to
            outage/schemas/activity_intake_result.json.
        """
        run_id: str = run_context["run_id"]
        activity_id: str = emergent_activity["activity_id"]
        LOGGER.debug("Stage A processing activity %s (run=%s)", activity_id, run_id)

        raw = emergent_activity.get("raw_description", "")

        normalized = self._clean_description(raw)
        expanded, abbr_rate = self._expand_abbreviations(normalized)
        entities = self._run_ner(expanded, emergent_activity)
        (
            resolved_components,
            resolved_systems,
            resolved_wos,
            resolved_crs,
        ) = self._resolve_references(entities, emergent_activity)
        discipline, task_family, component_family = self._classify_labels(
            expanded, entities
        )
        emergence_type, emergence_confidence, emergence_rationale = (
            self._classify_emergence_type(emergent_activity, entities, expanded)
        )
        has_regulatory, regulatory_drivers = self._detect_regulatory_constraints(
            emergent_activity, entities, expanded
        )
        dq_score = self._compute_data_quality(emergent_activity, entities, abbr_rate)

        return {
            "activity_id": activity_id,
            "run_id": run_id,
            "generated_at": run_context.get("started_at", ""),
            "emergence_type": emergence_type,
            "emergence_type_confidence": emergence_confidence,
            "emergence_type_rationale": emergence_rationale,
            "has_regulatory_constraint": has_regulatory,
            "regulatory_drivers": regulatory_drivers,
            "normalized_description": normalized,
            "expanded_description": expanded if expanded != normalized else None,
            "extracted_entities": entities,
            "resolved_component_ids": resolved_components,
            "resolved_system_ids": resolved_systems,
            "resolved_work_order_ids": resolved_wos,
            "resolved_cr_ids": resolved_crs,
            "discipline": discipline,
            "task_family": task_family,
            "component_family": component_family,
            "data_quality_score": dq_score,
            "unknown_abbreviation_rate": abbr_rate,
            "provenance": {
                "generated_by": self.__class__.__name__,
                "run_id": run_id,
                "ner_pipeline_version": None,
                "taxonomy_version": None,
                "abbreviation_dict_version": None,
            },
        }

    # ── Private step methods ──────────────────────────────────────────────────

    def _clean_description(self, raw: str) -> str:
        """Normalize whitespace, punctuation, and case.

        Uses the injected text_cleaner if available.  Falls back to
        ComponentIdRemover → TextacyPreprocessor from the preprocessing module
        (initialised once in __init__).  Does NOT run abbreviation expansion —
        that is a separate step so the abbreviation rate can be computed
        independently.
        """
        if not raw:
            return raw

        if self.text_cleaner is not None:
            # Injected cleaner may accept a plain string or an ActivityCase.
            # Try plain string first; fall back to ActivityCase wrapper.
            try:
                result = self.text_cleaner.clean(raw)
                if isinstance(result, str):
                    return result
                return getattr(result, "cleaned_description", raw) or raw
            except TypeError:
                pass

        # Use preprocessing module components (initialised at __init__ time).
        if self._fallback_id_remover is not None:
            text = self._fallback_id_remover.transform(raw)
            text = self._fallback_preprocessor.transform(text)
            return text

        # Last-resort: collapse whitespace only (preprocessing module absent).
        return re.sub(r"\s+", " ", raw).strip()

    def _expand_abbreviations(self, text: str) -> Tuple[str, float]:
        """Expand nuclear abbreviations and return (expanded_text, unknown_rate).

        unknown_rate: fraction of ALL-CAPS candidate tokens in the *original*
        text that remain unresolved (still ALL-CAPS) after expansion.
        Values > config.unknown_abbreviation_rate_warning trigger analyst
        review flag (WBS exit criterion 4.7).

        Uses the injected abbreviation_expander if available.  Falls back to
        AbbreviationResolver with the built-in NUCLEAR_OUTAGE_ABBREVIATIONS dict.
        """
        if not text:
            return text, 0.0

        # Collect ALL-CAPS candidate tokens in the pre-expansion text
        pre_caps = [
            t for t in _ABBR_CANDIDATE_RE.findall(text) if t not in _COMMON_CAPS
        ]

        # Expand
        if self.abbreviation_expander is not None:
            expanded = self.abbreviation_expander.transform(text)
        elif self._fallback_abbr_resolver is not None:
            expanded = self._fallback_abbr_resolver.transform(text)
        else:
            expanded = text  # preprocessing module absent — passthrough

        # Compute unknown rate: candidates still ALL-CAPS after expansion
        if not pre_caps:
            return expanded, 0.0

        post_caps = {
            t for t in _ABBR_CANDIDATE_RE.findall(expanded) if t not in _COMMON_CAPS
        }
        unresolved = sum(1 for t in pre_caps if t in post_caps)
        abbr_rate = unresolved / len(pre_caps)

        if abbr_rate > self.config.unknown_abbreviation_rate_warning:
            LOGGER.warning(
                "High unknown abbreviation rate %.2f (>%.2f) — NER results may be unreliable",
                abbr_rate,
                self.config.unknown_abbreviation_rate_warning,
            )

        return expanded, round(abbr_rate, 4)

    def _run_ner(
        self, text: str, emergent_activity: JsonDict
    ) -> List[JsonDict]:
        """Extract named entities from the expanded description.

        Returns a list of entity dicts conforming to the extracted_entities
        array in activity_intake_result.json.

        Extraction layers (applied in order, results merged):
            1. Regex patterns for tag IDs, WO references, CR references.
            2. HybridNERPipeline (if injected) for component, action, failure-mode
               entities via gazetteer + anchored NP generators.
            3. SpacyAnnotator (if injected) for temporal references and measurements.

        Each entity dict has keys: entity_id, text, entity_type, start, end,
        source, confidence.
        """
        entities: List[JsonDict] = []

        # ── Layer 1: regex extraction (always runs) ───────────────────────────
        for match in _TAG_ID_RE.finditer(text):
            entities.append(_make_entity(
                text=match.group(),
                entity_type="tag_id",
                start=match.start(),
                end=match.end(),
                source="regex",
                confidence=0.95,
            ))

        for match in _WO_REF_RE.finditer(text):
            entities.append(_make_entity(
                text=match.group(),
                entity_type="work_order_reference",
                start=match.start(),
                end=match.end(),
                source="regex",
                confidence=0.95,
            ))

        for match in _CR_REF_RE.finditer(text):
            entities.append(_make_entity(
                text=match.group(),
                entity_type="condition_report_reference",
                start=match.start(),
                end=match.end(),
                source="regex",
                confidence=0.95,
            ))

        # ── Layer 2: HybridNERPipeline (injected) ────────────────────────────
        if self.ner_pipeline is not None:
            try:
                pipeline_entities = self.ner_pipeline.generate(
                    text,
                    mode=self.config.ner_generator_mode,
                    score_threshold=self.config.np_score_threshold,
                )
                for ent in pipeline_entities:
                    entities.append(_make_entity(
                        text=ent.get("text", ""),
                        entity_type=ent.get("entity_type", "unknown"),
                        start=ent.get("start", 0),
                        end=ent.get("end", 0),
                        source="hybrid_ner",
                        confidence=ent.get("score", ent.get("confidence", 0.7)),
                    ))
            except Exception:  # noqa: BLE001
                LOGGER.warning("HybridNERPipeline.generate() failed; skipping pipeline entities")

        # ── Layer 3: SpacyAnnotator (injected) ───────────────────────────────
        if self.spacy_annotator is not None:
            try:
                spacy_entities = self.spacy_annotator.annotate(text)
                for ent in spacy_entities:
                    entities.append(_make_entity(
                        text=ent.get("text", ""),
                        entity_type=ent.get("entity_type", ent.get("label", "unknown")),
                        start=ent.get("start", 0),
                        end=ent.get("end", 0),
                        source="spacy",
                        confidence=ent.get("confidence", 0.75),
                    ))
            except Exception:  # noqa: BLE001
                LOGGER.warning("SpacyAnnotator.annotate() failed; skipping spacy entities")

        LOGGER.debug("Stage A NER: extracted %d entities", len(entities))
        return entities

    def _resolve_references(
        self,
        entities: List[JsonDict],
        emergent_activity: JsonDict,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Resolve entity mentions to canonical IDs.

        Returns (component_ids, system_ids, work_order_ids, cr_ids).

        Resolution strategy:
            - Pass-through known IDs from emergent_activity (highest confidence).
            - Collect WO/CR regex matches from extracted entities.
            - EntityNormalizer (if injected) for additional component mentions
              found by the NER pipeline (token-overlap → LLM fallback).
        """
        # ── Pass-through known IDs from the intake record ─────────────────────
        comp_ids: List[str] = []
        sys_ids: List[str] = []
        wo_ids: List[str] = []
        cr_ids: List[str] = []

        if emergent_activity.get("known_component_id"):
            comp_ids.append(emergent_activity["known_component_id"])
        if emergent_activity.get("known_system_id"):
            sys_ids.append(emergent_activity["known_system_id"])
        if emergent_activity.get("work_order_id"):
            wo_ids.append(str(emergent_activity["work_order_id"]))
        if emergent_activity.get("condition_report_id"):
            cr_ids.append(str(emergent_activity["condition_report_id"]))

        # ── Collect regex-extracted references ────────────────────────────────
        for ent in entities:
            if ent["entity_type"] == "work_order_reference":
                wo_val = ent["text"]
                if wo_val not in wo_ids:
                    wo_ids.append(wo_val)
            elif ent["entity_type"] == "condition_report_reference":
                cr_val = ent["text"]
                if cr_val not in cr_ids:
                    cr_ids.append(cr_val)

        # ── EntityNormalizer for component / system mentions ──────────────────
        if self.entity_normalizer is not None:
            component_mentions = [
                ent for ent in entities
                if ent["entity_type"] in {"component", "system", "equipment"}
            ]
            for mention in component_mentions:
                try:
                    normalized = self.entity_normalizer.normalize(
                        mention["text"],
                        token_overlap_threshold=self.config.entity_normalizer_token_overlap_threshold,
                    )
                    if normalized:
                        canonical_id = normalized.get("canonical_id") or normalized.get("id")
                        canonical_type = normalized.get("entity_type", "component")
                        if canonical_id:
                            if canonical_type == "system" and canonical_id not in sys_ids:
                                sys_ids.append(canonical_id)
                            elif canonical_id not in comp_ids:
                                comp_ids.append(canonical_id)
                except Exception:  # noqa: BLE001
                    LOGGER.debug("EntityNormalizer failed for mention '%s'", mention["text"])

        return comp_ids, sys_ids, wo_ids, cr_ids

    def _classify_labels(
        self,
        text: str,
        entities: List[JsonDict],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Classify discipline, task_family, component_family.

        Returns (discipline, task_family, component_family).

        Uses the injected label_mapper (TaskLabelMapper) via an ActivityCase
        wrapper when available.  Falls back to direct pattern iteration over
        label_mapper._patterns if ActivityCase cannot be imported, or returns
        (None, None, None) if no label_mapper is injected.
        """
        if self.label_mapper is None:
            return None, None, None

        # Attempt to use the injected TaskLabelMapper via ActivityCase wrapper
        try:
            from dackar.outage.outage_uncertainty.domain.activity import ActivityCase
            activity = ActivityCase.__new__(ActivityCase)
            object.__setattr__(activity, "raw_description", text)
            object.__setattr__(activity, "cleaned_description", text)
            object.__setattr__(activity, "discipline", None)
            object.__setattr__(activity, "task_family", None)
            object.__setattr__(activity, "component_family", None)
            result = self.label_mapper.map(activity)
            return (
                getattr(result, "discipline", None),
                getattr(result, "task_family", None),
                getattr(result, "component_family", None),
            )
        except (ImportError, TypeError, AttributeError):
            pass

        # Fallback: iterate _patterns directly (avoids ActivityCase dependency)
        try:
            inferred: Dict[str, str] = {}
            for pattern, labels in self.label_mapper._patterns:
                if pattern.search(text):
                    inferred.update(labels)
            return (
                inferred.get("discipline"),
                inferred.get("task_family"),
                inferred.get("component_family"),
            )
        except AttributeError:
            LOGGER.debug("label_mapper has no _patterns attribute; skipping label classification")
            return None, None, None

    def _classify_emergence_type(
        self,
        emergent_activity: JsonDict,
        entities: List[JsonDict],
        text: str,
    ) -> Tuple[str, float, Optional[str]]:
        """Classify the emergence type of this activity.

        Returns (emergence_type, confidence, rationale).

        Rule priority (highest to lowest):
            1. regulatory_driven  — TS/NRC/LCO/ALARA/CAP keyword match, or
                                    explicit regulatory fields in intake record.
            2. scope_expansion    — scope-language keywords AND an existing WO
                                    is referenced (expanding known work).
            3. schedule_optimization — schedule-change language, no failure terms.
            4. truly_unplanned    — default; degradation / failure language, or
                                    no other rule matched.
        """
        # Use the lowercase text for keyword matching
        text_lower = text.lower()

        # Check for explicit emergence type in the intake record (highest trust)
        explicit = emergent_activity.get("emergence_type")
        if explicit and explicit not in {"unknown", ""}:
            return explicit, 1.0, "explicitly provided in intake record"

        # ── Rule 1: regulatory_driven ─────────────────────────────────────────
        reg_match = _REGULATORY_KEYWORDS_RE.search(text)
        if reg_match:
            return (
                "regulatory_driven",
                0.85,
                f"regulatory keyword matched: '{reg_match.group()}'",
            )
        # Also check structured regulatory fields in the intake record
        if (
            emergent_activity.get("technical_specification_reference")
            or emergent_activity.get("nrc_commitment_number")
            or emergent_activity.get("lco_number")
        ):
            return (
                "regulatory_driven",
                0.90,
                "structured regulatory reference fields populated in intake record",
            )

        # ── Rule 2: scope_expansion ───────────────────────────────────────────
        has_existing_wo = (
            bool(emergent_activity.get("work_order_id"))
            or any(e["entity_type"] == "work_order_reference" for e in entities)
        )
        scope_match = _SCOPE_KEYWORDS_RE.search(text)
        if scope_match and has_existing_wo:
            return (
                "scope_expansion",
                0.80,
                f"scope language '{scope_match.group()}' with existing WO reference",
            )
        # Scope keyword alone (no existing WO) is weaker
        if scope_match:
            return (
                "scope_expansion",
                0.60,
                f"scope language '{scope_match.group()}' detected (no WO reference)",
            )

        # ── Rule 3: schedule_optimization ────────────────────────────────────
        sched_match = _SCHEDULE_OPT_KEYWORDS_RE.search(text)
        degrad_match = _DEGRADATION_KEYWORDS_RE.search(text)
        if sched_match and not degrad_match:
            return (
                "schedule_optimization",
                0.75,
                f"schedule-optimization language '{sched_match.group()}' without failure terms",
            )

        # ── Rule 4: truly_unplanned (default) ────────────────────────────────
        if degrad_match:
            return (
                "truly_unplanned",
                0.80,
                f"degradation/failure keyword '{degrad_match.group()}' detected",
            )

        # Default: insufficient signal → truly_unplanned with low confidence
        return (
            "truly_unplanned",
            0.45,
            "no strong classification signal; defaulting to truly_unplanned",
        )

    def _detect_regulatory_constraints(
        self,
        emergent_activity: JsonDict,
        entities: List[JsonDict],
        text: str,
    ) -> Tuple[bool, List[JsonDict]]:
        """Identify regulatory constraints that restrict deferral.

        Returns (has_regulatory_constraint, regulatory_drivers).

        Sources checked in order:
            1. Structured regulatory fields in emergent_activity
               (technical_specification_reference, nrc_commitment_number, lco_number).
            2. Pattern matching against expanded description using
               _REGULATORY_PATTERNS (TS, LCO, NRC, ALARA, CAP, surveillance,
               operability determination, hold point).
            3. Optional external keyword vocabulary from
               config.regulatory_keywords_path (not yet wired — placeholder).

        Every driver includes: driver_id, driver_type, matched_text,
        defer_prohibited, source.
        """
        drivers: List[JsonDict] = []
        seen_types: set = set()

        def _add_driver(
            driver_type: str,
            matched_text: str,
            defer_prohibited: bool,
            source: str,
        ) -> None:
            if driver_type not in seen_types:
                seen_types.add(driver_type)
                drivers.append({
                    "driver_id": f"REG::{driver_type}::{uuid.uuid4().hex[:6]}",
                    "driver_type": driver_type,
                    "matched_text": matched_text,
                    "defer_prohibited": defer_prohibited,
                    "source": source,
                })

        # ── Structured fields ─────────────────────────────────────────────────
        if emergent_activity.get("technical_specification_reference"):
            _add_driver(
                "technical_specification",
                str(emergent_activity["technical_specification_reference"]),
                defer_prohibited=True,
                source="intake_record_field",
            )
        if emergent_activity.get("nrc_commitment_number"):
            _add_driver(
                "nrc_commitment",
                str(emergent_activity["nrc_commitment_number"]),
                defer_prohibited=True,
                source="intake_record_field",
            )
        if emergent_activity.get("lco_number"):
            _add_driver(
                "limiting_condition_for_operation",
                str(emergent_activity["lco_number"]),
                defer_prohibited=True,
                source="intake_record_field",
            )

        # ── Text pattern matching ─────────────────────────────────────────────
        for pattern, driver_type, defer_prohibited in _REGULATORY_PATTERNS:
            match = pattern.search(text)
            if match:
                _add_driver(
                    driver_type,
                    match.group(),
                    defer_prohibited=defer_prohibited,
                    source="text_pattern",
                )

        has_regulatory = bool(drivers)
        return has_regulatory, drivers

    def _compute_data_quality(
        self,
        emergent_activity: JsonDict,
        entities: List[JsonDict],
        abbr_rate: float,
    ) -> float:
        """Compute a composite [0, 1] data quality score for this intake record.

        Composite formula (weights sum to 1.0):
            0.35 × field_completeness   — description length, timestamps,
                                          component_id, source fields
            0.25 × ner_yield            — entity count relative to description tokens
            0.25 × abbreviation_clarity — 1.0 − abbr_rate
            0.15 × source_confidence    — source system reliability tier

        All sub-scores are clamped to [0, 1] before weighting.
        """
        # ── Field completeness ────────────────────────────────────────────────
        completeness_score = 0.0
        checks = [
            bool(emergent_activity.get("raw_description", "").strip()),    # has description
            len(emergent_activity.get("raw_description", "")) >= 20,       # non-trivial length
            bool(emergent_activity.get("detection_timestamp")              # has timestamp
                 or emergent_activity.get("actual_start")),
            bool(emergent_activity.get("known_component_id")              # component identified
                 or emergent_activity.get("known_system_id")),
            bool(emergent_activity.get("work_order_id")                    # WO or CR linked
                 or emergent_activity.get("condition_report_id")),
            bool(emergent_activity.get("source_system")),                  # source system known
        ]
        completeness_score = sum(checks) / len(checks)

        # ── NER yield ─────────────────────────────────────────────────────────
        desc = emergent_activity.get("raw_description", "")
        token_count = len(desc.split())
        if token_count > 0:
            # Expected ~1 entity per 5–8 tokens in a typical activity description
            raw_yield = len(entities) / max(token_count / 6.0, 1.0)
            ner_yield = min(1.0, raw_yield)
        else:
            ner_yield = 0.0

        # ── Abbreviation clarity ──────────────────────────────────────────────
        abbr_clarity = max(0.0, 1.0 - abbr_rate)

        # ── Source confidence ─────────────────────────────────────────────────
        source_system = str(emergent_activity.get("source_system", "unknown")).lower()
        source_conf = _SOURCE_CONFIDENCE.get(source_system, _SOURCE_CONFIDENCE["unknown"])

        dq_score = (
            0.35 * completeness_score
            + 0.25 * ner_yield
            + 0.25 * abbr_clarity
            + 0.15 * source_conf
        )
        return round(min(1.0, max(0.0, dq_score)), 4)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_entity(
    *,
    text: str,
    entity_type: str,
    start: int,
    end: int,
    source: str,
    confidence: float,
) -> JsonDict:
    """Construct a standardised entity dict for the extracted_entities array."""
    return {
        "entity_id": f"ENT::{entity_type}::{uuid.uuid4().hex[:6]}",
        "text": text,
        "entity_type": entity_type,
        "start": start,
        "end": end,
        "source": source,
        "confidence": round(confidence, 4),
    }


