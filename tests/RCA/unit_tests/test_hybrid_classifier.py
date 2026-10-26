"""
Unit tests for dackar.RCA.ner.hybrid_ner.classifier.SpanClassifier.

The classifier is a scikit-learn placeholder that proposes labels for unlabeled
("defer") candidates. Deterministic behaviors tested without any model:
untrained predict() is a no-op; fit() filters examples whose gold label is not
in schema.label_to_group; predict() only ever proposes labels that exist in the
schema (the group-resolution guard at classifier.py:99). A trained fit→predict
round-trip is covered behind importorskip("sklearn").
"""
from __future__ import annotations

import pytest

from dackar.RCA.ner.hybrid_ner.classifier import SpanClassifier, TrainingExample
from dackar.RCA.ner.hybrid_ner.models import CandidateSpan, Document
from dackar.RCA.ner.hybrid_ner.schema import SchemaIndex


_SCHEMA = SchemaIndex(
    label_to_group={"comp": "G1_PHYSICAL_COMPONENT", "deg_mech": "G4_MECHANISM_PROCESS"},
    group_to_labels={"G1_PHYSICAL_COMPONENT": {"comp"}, "G4_MECHANISM_PROCESS": {"deg_mech"}},
    pair_decision={},
    conditional_rules=[],
)


def _doc(text="the pump bearing wear was observed"):
    return Document(doc_id="D1", text=text)


def _unlabeled(start, end, text):
    return CandidateSpan(span_id="c1", doc_id="D1", start=start, end=end, text=text)


# ---------------------------------------------------------------------------
# Untrained → no-op
# ---------------------------------------------------------------------------

def test_untrained_predict_is_noop():
    clf = SpanClassifier()  # never fitted
    cands = [_unlabeled(4, 8, "pump")]
    out = clf.predict(_doc(), cands, _SCHEMA)
    assert out is cands
    assert out[0].proposed_labels == []


def test_fit_with_no_in_schema_examples_leaves_untrained():
    clf = SpanClassifier()
    # all gold labels are outside the schema → nothing to train on
    examples = [TrainingExample("the pump failed", 4, 8, "NOT_A_SCHEMA_LABEL")]
    clf.fit(examples, _SCHEMA)
    # predict stays a no-op because training was skipped
    cands = [_unlabeled(4, 8, "pump")]
    out = clf.predict(_doc(), cands, _SCHEMA)
    assert out[0].proposed_labels == []


# ---------------------------------------------------------------------------
# Trained round-trip (only labels in schema.label_to_group may be proposed)
# ---------------------------------------------------------------------------

def test_trained_predict_drops_labels_absent_from_predict_schema():
    """The guard at classifier.py:99 must drop labels the model knows but that are
    absent from the PREDICT schema.

    Fit on a wide schema (comp + deg_mech), then predict against a narrow schema
    that omits deg_mech.  The model's classes still include deg_mech, so without
    the guard it would be proposed; with the guard only comp survives.  Predicting
    against the same schema used for fit (as a naive test would) can't show this,
    because fit() already filters training examples to that schema.
    """
    pytest.importorskip("sklearn")
    wide = SchemaIndex(
        label_to_group={"comp": "G1_PHYSICAL_COMPONENT", "deg_mech": "G4_MECHANISM_PROCESS"},
        group_to_labels={"G1_PHYSICAL_COMPONENT": {"comp"}, "G4_MECHANISM_PROCESS": {"deg_mech"}},
        pair_decision={}, conditional_rules=[],
    )
    narrow = SchemaIndex(
        label_to_group={"comp": "G1_PHYSICAL_COMPONENT"},  # deg_mech intentionally absent
        group_to_labels={"G1_PHYSICAL_COMPONENT": {"comp"}},
        pair_decision={}, conditional_rules=[],
    )
    clf = SpanClassifier(min_prob=0.0, top_k=5)  # top_k high so deg_mech would surface if not guarded
    clf.fit([
        TrainingExample("the pump was isolated", 4, 8, "comp"),
        TrainingExample("a valve pump unit", 8, 12, "comp"),
        TrainingExample("severe bearing wear found", 15, 19, "deg_mech"),
        TrainingExample("adhesive wear observed", 9, 13, "deg_mech"),
    ], wide)
    # sanity: the trained model really does know deg_mech (so the guard has work to do)
    assert "deg_mech" in set(clf._model.classes_)

    out = clf.predict(_doc(), [_unlabeled(4, 8, "pump")], narrow)
    proposed = sorted(h.label for h in out[0].proposed_labels)
    assert proposed == ["comp"]  # deg_mech dropped by the narrow-schema guard
    assert all(h.rationale == "ml_logreg" for h in out[0].proposed_labels)


def test_trained_predict_leaves_already_labeled_untouched():
    pytest.importorskip("sklearn")
    clf = SpanClassifier(min_prob=0.0)
    clf.fit(
        [TrainingExample("pump isolated", 0, 4, "comp"),
         TrainingExample("bearing wear", 8, 12, "deg_mech")],
        _SCHEMA,
    )
    from dackar.RCA.ner.hybrid_ner.models import LabelHypothesis
    labeled = CandidateSpan(
        span_id="c1", doc_id="D1", start=4, end=8, text="pump",
        proposed_labels=[LabelHypothesis(label="comp")],
    )
    out = clf.predict(_doc(), [labeled], _SCHEMA)
    # already-labeled candidate is not in the 'unlabeled' set → unchanged
    assert [h.label for h in out[0].proposed_labels] == ["comp"]
