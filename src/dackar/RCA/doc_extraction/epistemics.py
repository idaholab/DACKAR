from __future__ import annotations

"""Epistemic classification of document extraction records (Phase A).

The EpistemicClassifier applies the four-way classification (§2 of
epistemics_notes_4.md) to every DocExtractionRecord and SemanticMatch that
enters the pipeline.  It implements the priority chain from §3.3:

    finding_status → authority_level → doc_type → default

Classification is deterministic: the same metadata always produces the same
annotation.  EpistemicsRoutingConfig is a versioned artifact stamped on every
run manifest so that routing decisions are reproducible across pipeline versions.

No scoring changes are made in Phase A.  The annotation fields are carried
through the pipeline for audit and for Phase C consumption.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .schema import (
    AuthorityLevel,
    ClassificationResolutionLevel,
    EpistemicClass,
    FindingStatus,
)

# ---------------------------------------------------------------------------
# Routing tables — the authoritative source for doc_type → epistemic_class
# mapping.  These are the "doc_type" level of the priority chain; they are
# only consulted when finding_status and authority_level are both absent.
# ---------------------------------------------------------------------------

# Primary doc_type routing (§3.2)
_DOC_TYPE_TO_CLASS: Dict[str, str] = {
    # Analyzes past degradation
    "ECA":  EpistemicClass.ANALYZES_PAST_DEGRADATION,
    "RCA":  EpistemicClass.ANALYZES_PAST_DEGRADATION,
    "OE":   EpistemicClass.ANALYZES_PAST_DEGRADATION,
    "LER":  EpistemicClass.ANALYZES_PAST_DEGRADATION,
    # Monitors performance (primary class; secondary causal text → contextual only)
    "CR":   EpistemicClass.MONITORS_PERFORMANCE,
    # Affects performance (activity; as-found observation is secondary)
    "WO":   EpistemicClass.AFFECTS_PERFORMANCE,
    # Characterizes the system — discriminating
    "FMEA": EpistemicClass.CHARACTERIZES_THE_SYSTEM,
    "SOP":  EpistemicClass.CHARACTERIZES_THE_SYSTEM,
    # Characterizes the system — plausibility / reference
    "MAN":  EpistemicClass.CHARACTERIZES_THE_SYSTEM,   # manuals
    "BULL": EpistemicClass.CHARACTERIZES_THE_SYSTEM,   # bulletins
}

# finding_status routing (§3.3, Level 1)
# Maps (finding_status, doc_type) → epistemic_class.
# When doc_type is absent the first-matched entry applies.
_FINDING_STATUS_TO_CLASS: Dict[Tuple[str, str], str] = {
    # Formal conclusions always → analyzes
    (FindingStatus.FORMAL_CONCLUSION, "ECA"):  EpistemicClass.ANALYZES_PAST_DEGRADATION,
    (FindingStatus.FORMAL_CONCLUSION, "RCA"):  EpistemicClass.ANALYZES_PAST_DEGRADATION,
    (FindingStatus.FORMAL_CONCLUSION, "CR"):   EpistemicClass.ANALYZES_PAST_DEGRADATION,
    (FindingStatus.FORMAL_CONCLUSION, "OE"):   EpistemicClass.ANALYZES_PAST_DEGRADATION,
    (FindingStatus.FORMAL_CONCLUSION, "LER"):  EpistemicClass.ANALYZES_PAST_DEGRADATION,
    (FindingStatus.FORMAL_CONCLUSION, ""):     EpistemicClass.ANALYZES_PAST_DEGRADATION,
    # Preliminary assessment: CR → monitors; ECA/RCA → analyzes; others → monitors
    (FindingStatus.PRELIMINARY_ASSESSMENT, "ECA"): EpistemicClass.ANALYZES_PAST_DEGRADATION,
    (FindingStatus.PRELIMINARY_ASSESSMENT, "RCA"): EpistemicClass.ANALYZES_PAST_DEGRADATION,
    (FindingStatus.PRELIMINARY_ASSESSMENT, "CR"):  EpistemicClass.MONITORS_PERFORMANCE,
    (FindingStatus.PRELIMINARY_ASSESSMENT, "WO"):  EpistemicClass.MONITORS_PERFORMANCE,
    (FindingStatus.PRELIMINARY_ASSESSMENT, ""):    EpistemicClass.MONITORS_PERFORMANCE,
    # Observation only always → monitors
    (FindingStatus.OBSERVATION_ONLY, ""):  EpistemicClass.MONITORS_PERFORMANCE,
    (FindingStatus.OBSERVATION_ONLY, "CR"): EpistemicClass.MONITORS_PERFORMANCE,
    (FindingStatus.OBSERVATION_ONLY, "WO"): EpistemicClass.MONITORS_PERFORMANCE,
}

# authority_level routing (§3.3, Level 2)
_AUTHORITY_LEVEL_TO_CLASS: Dict[str, str] = {
    AuthorityLevel.MANDATORY:     EpistemicClass.ANALYZES_PAST_DEGRADATION,
    AuthorityLevel.GUIDANCE:      EpistemicClass.ANALYZES_PAST_DEGRADATION,
    AuthorityLevel.INFORMATIONAL: EpistemicClass.MONITORS_PERFORMANCE,
}

# Default class when no other level resolves
_DEFAULT_CLASS = EpistemicClass.MONITORS_PERFORMANCE


@dataclass
class EpistemicAnnotation:
    """Result of one EpistemicClassifier.classify() call."""
    epistemic_class: str                    # EpistemicClass value
    classification_resolution_level: str    # ClassificationResolutionLevel value
    degraded_classification: bool           # True when doc_type or default was used
    policy_version: str


@dataclass
class EpistemicsRoutingConfig:
    """Versioned configuration artifact for the epistemic routing table.

    policy_version is stamped on run_manifest.pipeline_config and on every
    EpistemicAnnotation so that routing decisions are reproducible.

    The routing tables are embedded in epistemics.py and versioned via
    policy_version; changing any table entry requires a version bump.
    """
    policy_version: str = "epistemics-v1.0"

    # doc_type → epistemic_class overrides (allows plant-specific additions)
    doc_type_overrides: Dict[str, str] = field(default_factory=dict)

    # When True, fall through to doc_type routing even when finding_status is
    # present but not recognized.  When False (default), unrecognized
    # finding_status values are treated as absent, proceeding to authority_level.
    strict_finding_status: bool = False


class EpistemicClassifier:
    """Applies the four-way epistemic classification to a document record.

    Usage
    -----
    config = EpistemicsRoutingConfig(policy_version="epistemics-v1.0")
    classifier = EpistemicClassifier(config)
    annotation = classifier.classify(meta)

    ``meta`` may be a dict (Chroma metadata), a DocExtractionRecord, or a
    SemanticMatch — any object that exposes the relevant fields via attribute
    or dict access.

    Priority chain (§3.3)
    ----------------------
    1. finding_status — semantic; not degraded
    2. authority_level — semantic; not degraded
    3. doc_type — syntactic proxy; degraded_classification = True
    4. default — no metadata; degraded_classification = True
    """

    def __init__(self, config: Optional[EpistemicsRoutingConfig] = None) -> None:
        self.config = config or EpistemicsRoutingConfig()
        # Merge built-in table with any plant-specific overrides
        self._doc_type_table = {**_DOC_TYPE_TO_CLASS, **self.config.doc_type_overrides}

    def classify(self, record: Any) -> EpistemicAnnotation:
        """Classify one record and return an EpistemicAnnotation.

        ``record`` may be:
        - a dict (Chroma metadata dict)
        - a DocExtractionRecord
        - a SemanticMatch
        - any object with attribute access for the relevant fields
        """
        finding_status = _get(record, "finding_status") or ""
        authority_level = _get(record, "authority_level") or ""
        doc_type = (_get(record, "doc_type") or "").upper()

        # ------------------------------------------------------------------
        # Level 1 — finding_status
        # ------------------------------------------------------------------
        if finding_status:
            resolved = self._resolve_finding_status(finding_status, doc_type)
            if resolved is not None:
                return EpistemicAnnotation(
                    epistemic_class=resolved,
                    classification_resolution_level=ClassificationResolutionLevel.FINDING_STATUS,
                    degraded_classification=False,
                    policy_version=self.config.policy_version,
                )

        # ------------------------------------------------------------------
        # Level 2 — authority_level
        # ------------------------------------------------------------------
        if authority_level:
            resolved = _AUTHORITY_LEVEL_TO_CLASS.get(authority_level)
            if resolved is not None:
                return EpistemicAnnotation(
                    epistemic_class=resolved,
                    classification_resolution_level=ClassificationResolutionLevel.AUTHORITY_LEVEL,
                    degraded_classification=False,
                    policy_version=self.config.policy_version,
                )

        # ------------------------------------------------------------------
        # Level 3 — doc_type (syntactic; degraded)
        # ------------------------------------------------------------------
        if doc_type:
            resolved = self._doc_type_table.get(doc_type)
            if resolved is not None:
                return EpistemicAnnotation(
                    epistemic_class=resolved,
                    classification_resolution_level=ClassificationResolutionLevel.DOC_TYPE,
                    degraded_classification=True,
                    policy_version=self.config.policy_version,
                )

        # ------------------------------------------------------------------
        # Level 4 — default (degraded)
        # ------------------------------------------------------------------
        return EpistemicAnnotation(
            epistemic_class=_DEFAULT_CLASS,
            classification_resolution_level=ClassificationResolutionLevel.DEFAULT,
            degraded_classification=True,
            policy_version=self.config.policy_version,
        )

    def annotate_record(self, record: Any) -> None:
        """Classify ``record`` in-place, writing annotation fields back to it.

        Supports DocExtractionRecord and SemanticMatch (both have the three
        annotation fields as attributes).  No-ops silently on other types.
        """
        annotation = self.classify(record)
        try:
            record.epistemic_class = annotation.epistemic_class
            record.classification_resolution_level = annotation.classification_resolution_level
            record.degraded_classification = annotation.degraded_classification
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_finding_status(
        self, finding_status: str, doc_type: str
    ) -> Optional[str]:
        """Look up (finding_status, doc_type) in the routing table.

        Tries exact (finding_status, doc_type) first, then (finding_status, "")
        as the doc_type-agnostic fallback.  Returns None if not found.
        """
        key_specific = (finding_status, doc_type)
        if key_specific in _FINDING_STATUS_TO_CLASS:
            return _FINDING_STATUS_TO_CLASS[key_specific]
        key_generic = (finding_status, "")
        return _FINDING_STATUS_TO_CLASS.get(key_generic)


# ---------------------------------------------------------------------------
# Field accessor
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str) -> Optional[str]:
    """Get a field value from a dict or an object with attribute access."""
    if isinstance(obj, dict):
        return obj.get(key) or None
    return getattr(obj, key, None) or None


# ---------------------------------------------------------------------------
# Manifest summary builder
# ---------------------------------------------------------------------------

def build_epistemics_manifest_summary(
    cross_pattern_evidence: Optional[Dict[str, Any]],
    policy_version: Optional[str],
) -> Dict[str, Any]:
    """Build the epistemics section of run_manifest.artifacts (Phase A).

    Counts epistemic_class distribution, degraded_classification counts by
    doc_type, and classification_resolution_level distribution across all
    doc extractions visible to the pipeline.  Sourced from cross_pattern_evidence
    all_links provenance when available; otherwise returns a minimal stub.

    This function is called by the orchestrator's _stage_g_finalize_manifest().
    It is defined here (not in the orchestrator) so that it can be unit-tested
    independently without the orchestrator's heavy kg dependencies.
    """
    summary: Dict[str, Any] = {
        "present": False,
        "policy_version": policy_version or "not_configured",
        "epistemic_class_distribution": {},
        "classification_resolution_level_distribution": {},
        "degraded_classification_by_doc_type": {},
        "degraded_classification_total": 0,
    }

    if cross_pattern_evidence is None:
        return summary

    try:
        all_links = cross_pattern_evidence.get("all_links") or []
        class_dist: Dict[str, int] = {}
        level_dist: Dict[str, int] = {}
        degraded_by_type: Dict[str, int] = {}
        degraded_total = 0

        for lnk in all_links:
            prov = (lnk.get("provenance") or {}) if isinstance(lnk, dict) else {}
            ep_class = prov.get("epistemic_class") or ""
            ep_level = prov.get("classification_resolution_level") or ""
            ep_doc_type = prov.get("doc_type") or ""
            ep_degraded = bool(prov.get("degraded_classification", False))

            if ep_class:
                class_dist[ep_class] = class_dist.get(ep_class, 0) + 1
            if ep_level:
                level_dist[ep_level] = level_dist.get(ep_level, 0) + 1
            if ep_degraded:
                degraded_total += 1
                key = ep_doc_type or "unknown"
                degraded_by_type[key] = degraded_by_type.get(key, 0) + 1

        summary["present"] = True
        summary["epistemic_class_distribution"] = class_dist
        summary["classification_resolution_level_distribution"] = level_dist
        summary["degraded_classification_by_doc_type"] = degraded_by_type
        summary["degraded_classification_total"] = degraded_total

    except Exception:
        pass

    return summary
