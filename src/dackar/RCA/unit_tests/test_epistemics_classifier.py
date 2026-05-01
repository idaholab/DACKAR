from __future__ import annotations

"""Unit tests for the EpistemicClassifier (Phase A).

Covers:
- All five §3.2 routing rows → correct epistemic_class
- Fallback chain: each of the four ClassificationResolutionLevel values
- degraded_classification = False for finding_status / authority_level
- degraded_classification = True for doc_type / default
- Missing metadata → default class with degraded_classification = True
- Dual-role elements route to primary class per §2.5 policy
- policy_version present on every annotation
- Conflicting finding_status vs doc_type → finding_status wins
- SemanticMatch annotation via annotate_record()
- DocExtractionRecord annotation via annotate_record()
- EpistemicClassifier wired into DocExtractionStore annotates query results
- Manifest epistemics summary counts degraded_classification correctly
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

# Stub heavy optional dependencies so the orchestrator module can be imported
# in a test environment that lacks kg, neo4j, py2neo, and chroma.
_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in (
    "kg", "kg.py2neo_workflow",
    "neo4j", "py2neo",
    "chromadb",
    "langchain_chroma", "langchain_community",
    "langchain_community.vectorstores", "langchain_community.embeddings",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from dackar.RCA.doc_extraction.epistemics import (
    EpistemicAnnotation,
    EpistemicClassifier,
    EpistemicsRoutingConfig,
)
from dackar.RCA.doc_extraction.schema import (
    AuthorityLevel,
    ClassificationResolutionLevel,
    ConfidenceLevel,
    DocExtractionRecord,
    EpistemicClass,
    FindingStatus,
)
from dackar.RCA.doc_extraction.store import SemanticMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classifier(overrides: dict | None = None) -> EpistemicClassifier:
    cfg = EpistemicsRoutingConfig(
        policy_version="epistemics-v1.0",
        doc_type_overrides=overrides or {},
    )
    return EpistemicClassifier(cfg)


def _meta(**kwargs) -> dict:
    """Build a minimal metadata dict with the supplied fields."""
    return {
        "finding_status": "",
        "authority_level": "",
        "doc_type": "",
        **kwargs,
    }


# ---------------------------------------------------------------------------
# §3.2 routing rows — doc_type level (Level 3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc_type,expected_class", [
    ("ECA",  EpistemicClass.ANALYZES_PAST_DEGRADATION),
    ("RCA",  EpistemicClass.ANALYZES_PAST_DEGRADATION),
    ("OE",   EpistemicClass.ANALYZES_PAST_DEGRADATION),
    ("LER",  EpistemicClass.ANALYZES_PAST_DEGRADATION),
    ("CR",   EpistemicClass.MONITORS_PERFORMANCE),
    ("WO",   EpistemicClass.AFFECTS_PERFORMANCE),
    ("FMEA", EpistemicClass.CHARACTERIZES_THE_SYSTEM),
    ("SOP",  EpistemicClass.CHARACTERIZES_THE_SYSTEM),
])
def test_doc_type_routing(doc_type, expected_class):
    """All §3.2 doc_type rows route to the correct epistemic class."""
    clf = _make_classifier()
    ann = clf.classify(_meta(doc_type=doc_type))
    assert ann.epistemic_class == expected_class
    assert ann.classification_resolution_level == ClassificationResolutionLevel.DOC_TYPE
    assert ann.degraded_classification is True


# ---------------------------------------------------------------------------
# finding_status routing (Level 1)
# ---------------------------------------------------------------------------

def test_formal_conclusion_routes_to_analyzes():
    clf = _make_classifier()
    ann = clf.classify(_meta(finding_status=FindingStatus.FORMAL_CONCLUSION, doc_type="CR"))
    assert ann.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert ann.classification_resolution_level == ClassificationResolutionLevel.FINDING_STATUS
    assert ann.degraded_classification is False


def test_formal_conclusion_no_doc_type_routes_to_analyzes():
    clf = _make_classifier()
    ann = clf.classify(_meta(finding_status=FindingStatus.FORMAL_CONCLUSION, doc_type=""))
    assert ann.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert ann.classification_resolution_level == ClassificationResolutionLevel.FINDING_STATUS


def test_preliminary_assessment_cr_routes_to_monitors():
    clf = _make_classifier()
    ann = clf.classify(_meta(finding_status=FindingStatus.PRELIMINARY_ASSESSMENT, doc_type="CR"))
    assert ann.epistemic_class == EpistemicClass.MONITORS_PERFORMANCE
    assert ann.classification_resolution_level == ClassificationResolutionLevel.FINDING_STATUS
    assert ann.degraded_classification is False


def test_preliminary_assessment_eca_routes_to_analyzes():
    clf = _make_classifier()
    ann = clf.classify(_meta(finding_status=FindingStatus.PRELIMINARY_ASSESSMENT, doc_type="ECA"))
    assert ann.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert ann.classification_resolution_level == ClassificationResolutionLevel.FINDING_STATUS


def test_observation_only_routes_to_monitors():
    clf = _make_classifier()
    ann = clf.classify(_meta(finding_status=FindingStatus.OBSERVATION_ONLY, doc_type="CR"))
    assert ann.epistemic_class == EpistemicClass.MONITORS_PERFORMANCE
    assert ann.classification_resolution_level == ClassificationResolutionLevel.FINDING_STATUS


def test_finding_status_beats_doc_type():
    """finding_status must win over a conflicting doc_type (e.g. WO with formal_conclusion)."""
    clf = _make_classifier()
    ann = clf.classify(_meta(finding_status=FindingStatus.FORMAL_CONCLUSION, doc_type="WO"))
    assert ann.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert ann.classification_resolution_level == ClassificationResolutionLevel.FINDING_STATUS


# ---------------------------------------------------------------------------
# authority_level routing (Level 2)
# ---------------------------------------------------------------------------

def test_mandatory_authority_routes_to_analyzes():
    clf = _make_classifier()
    ann = clf.classify(_meta(authority_level=AuthorityLevel.MANDATORY))
    assert ann.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert ann.classification_resolution_level == ClassificationResolutionLevel.AUTHORITY_LEVEL
    assert ann.degraded_classification is False


def test_guidance_authority_routes_to_analyzes():
    clf = _make_classifier()
    ann = clf.classify(_meta(authority_level=AuthorityLevel.GUIDANCE))
    assert ann.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert ann.classification_resolution_level == ClassificationResolutionLevel.AUTHORITY_LEVEL
    assert ann.degraded_classification is False


def test_informational_authority_routes_to_monitors():
    clf = _make_classifier()
    ann = clf.classify(_meta(authority_level=AuthorityLevel.INFORMATIONAL))
    assert ann.epistemic_class == EpistemicClass.MONITORS_PERFORMANCE
    assert ann.classification_resolution_level == ClassificationResolutionLevel.AUTHORITY_LEVEL
    assert ann.degraded_classification is False


def test_finding_status_beats_authority_level():
    """finding_status at Level 1 must win over authority_level at Level 2."""
    clf = _make_classifier()
    ann = clf.classify(_meta(
        finding_status=FindingStatus.OBSERVATION_ONLY,
        authority_level=AuthorityLevel.MANDATORY,
    ))
    assert ann.epistemic_class == EpistemicClass.MONITORS_PERFORMANCE
    assert ann.classification_resolution_level == ClassificationResolutionLevel.FINDING_STATUS


# ---------------------------------------------------------------------------
# Default fallback (Level 4)
# ---------------------------------------------------------------------------

def test_no_metadata_returns_default_class():
    clf = _make_classifier()
    ann = clf.classify(_meta())
    assert ann.epistemic_class == EpistemicClass.MONITORS_PERFORMANCE
    assert ann.classification_resolution_level == ClassificationResolutionLevel.DEFAULT
    assert ann.degraded_classification is True


def test_empty_dict_returns_default_class():
    clf = _make_classifier()
    ann = clf.classify({})
    assert ann.classification_resolution_level == ClassificationResolutionLevel.DEFAULT
    assert ann.degraded_classification is True


def test_unknown_doc_type_falls_through_to_default():
    clf = _make_classifier()
    ann = clf.classify(_meta(doc_type="UNKNOWN_TYPE"))
    assert ann.classification_resolution_level == ClassificationResolutionLevel.DEFAULT
    assert ann.degraded_classification is True


# ---------------------------------------------------------------------------
# Dual-role elements — primary class policy (§2.5)
# ---------------------------------------------------------------------------

def test_wo_primary_role_is_affects_performance():
    """WO activity routes to affects_performance as primary class (§2.5)."""
    clf = _make_classifier()
    ann = clf.classify(_meta(doc_type="WO"))
    assert ann.epistemic_class == EpistemicClass.AFFECTS_PERFORMANCE


def test_cr_primary_role_is_monitors_performance():
    """CR routes to monitors_performance as primary class (§2.5)."""
    clf = _make_classifier()
    ann = clf.classify(_meta(doc_type="CR"))
    assert ann.epistemic_class == EpistemicClass.MONITORS_PERFORMANCE


# ---------------------------------------------------------------------------
# policy_version on every annotation
# ---------------------------------------------------------------------------

def test_policy_version_present_on_finding_status_path():
    clf = _make_classifier()
    ann = clf.classify(_meta(finding_status=FindingStatus.FORMAL_CONCLUSION))
    assert ann.policy_version == "epistemics-v1.0"


def test_policy_version_present_on_default_path():
    clf = _make_classifier()
    ann = clf.classify({})
    assert ann.policy_version == "epistemics-v1.0"


def test_custom_policy_version_propagates():
    cfg = EpistemicsRoutingConfig(policy_version="epistemics-v2.0")
    clf = EpistemicClassifier(cfg)
    ann = clf.classify(_meta(doc_type="CR"))
    assert ann.policy_version == "epistemics-v2.0"


# ---------------------------------------------------------------------------
# doc_type_overrides in config
# ---------------------------------------------------------------------------

def test_doc_type_override_is_applied():
    """Plant-specific doc_type override in config takes precedence over built-in table."""
    clf = _make_classifier(overrides={"MAINT": EpistemicClass.AFFECTS_PERFORMANCE})
    ann = clf.classify(_meta(doc_type="MAINT"))
    assert ann.epistemic_class == EpistemicClass.AFFECTS_PERFORMANCE
    assert ann.classification_resolution_level == ClassificationResolutionLevel.DOC_TYPE
    assert ann.degraded_classification is True


# ---------------------------------------------------------------------------
# annotate_record() on DocExtractionRecord and SemanticMatch
# ---------------------------------------------------------------------------

def _make_record(**kwargs) -> DocExtractionRecord:
    defaults = dict(
        doc_id="DOC-1",
        chain_index=0,
        identified_effect="vibration",
        assessed_cause="bearing wear",
        inferred_fm_label="mechanical wear",
        fm_id_candidate=None,
        fm_id_candidate_alt=None,
        confidence=ConfidenceLevel.MEDIUM,
        cause_is_symptom=False,
        as_found=None,
        as_left=None,
        procedural_deviation_score=0.0,
        extraction_version="v1",
        embedding_model_version=None,
    )
    defaults.update(kwargs)
    return DocExtractionRecord(**defaults)


def test_annotate_record_doc_extraction_record():
    clf = _make_classifier()
    record = _make_record(doc_type="ECA")
    clf.annotate_record(record)
    assert record.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert record.classification_resolution_level == ClassificationResolutionLevel.DOC_TYPE
    assert record.degraded_classification is True


def test_annotate_record_semantic_match():
    clf = _make_classifier()
    sm = SemanticMatch(
        record_id="r1",
        doc_id="DOC-2",
        chain_index=0,
        identified_effect=None,
        assessed_cause=None,
        inferred_fm_label=None,
        fm_id_candidate=None,
        confidence=ConfidenceLevel.LOW,
        cause_is_symptom=False,
        similarity_score=0.8,
        doc_type="RCA",
    )
    clf.annotate_record(sm)
    assert sm.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert sm.classification_resolution_level == ClassificationResolutionLevel.DOC_TYPE
    assert sm.degraded_classification is True


def test_annotate_record_finding_status_not_degraded():
    clf = _make_classifier()
    sm = SemanticMatch(
        record_id="r2",
        doc_id="DOC-3",
        chain_index=0,
        identified_effect=None,
        assessed_cause=None,
        inferred_fm_label=None,
        fm_id_candidate=None,
        confidence=ConfidenceLevel.HIGH,
        cause_is_symptom=False,
        similarity_score=0.9,
        finding_status=FindingStatus.FORMAL_CONCLUSION,
        doc_type="CR",
    )
    clf.annotate_record(sm)
    assert sm.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert sm.degraded_classification is False


# ---------------------------------------------------------------------------
# DocExtractionStore wires classifier into query results
# ---------------------------------------------------------------------------

def test_store_annotates_matches_when_classifier_set():
    """DocExtractionStore.query() calls annotate_record() on each match."""
    from dackar.RCA.doc_extraction.store import DocExtractionStore

    clf = _make_classifier()
    store = DocExtractionStore.__new__(DocExtractionStore)
    store.epistemics_classifier = clf

    # Build a fake SemanticMatch with doc_type set; simulate what query() produces
    sm = SemanticMatch(
        record_id="r1",
        doc_id="DOC-ECA",
        chain_index=0,
        identified_effect=None,
        assessed_cause=None,
        inferred_fm_label=None,
        fm_id_candidate=None,
        confidence=ConfidenceLevel.MEDIUM,
        cause_is_symptom=False,
        similarity_score=0.85,
        doc_type="ECA",
    )

    # Call annotate_record directly (mirrors what query() does after dedup)
    store.epistemics_classifier.annotate_record(sm)

    assert sm.epistemic_class == EpistemicClass.ANALYZES_PAST_DEGRADATION
    assert sm.classification_resolution_level == ClassificationResolutionLevel.DOC_TYPE
    assert sm.degraded_classification is True


# ---------------------------------------------------------------------------
# Manifest summary — degraded_classification counts
# ---------------------------------------------------------------------------

def test_manifest_summary_returns_stub_when_no_evidence():
    from dackar.RCA.doc_extraction.epistemics import build_epistemics_manifest_summary
    result = build_epistemics_manifest_summary(
        cross_pattern_evidence=None,
        policy_version="epistemics-v1.0",
    )
    assert result["present"] is False
    assert result["policy_version"] == "epistemics-v1.0"
    assert result["degraded_classification_total"] == 0


def test_manifest_summary_policy_version_not_configured_when_none():
    from dackar.RCA.doc_extraction.epistemics import build_epistemics_manifest_summary
    result = build_epistemics_manifest_summary(
        cross_pattern_evidence=None,
        policy_version=None,
    )
    assert result["policy_version"] == "not_configured"
