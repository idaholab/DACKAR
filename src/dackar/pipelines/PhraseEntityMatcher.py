# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans

import logging
logger = logging.getLogger(__name__)

@Language.factory("phrase_entity_matcher", default_config={"label": "ssc", "patterns":["safety cage", "pump"], "asSpan":True})
def create_phrase_matcher_component(nlp, name, label, patterns, asSpan):
  return PhraseEntityMatcher(nlp, label, patterns, asSpan=asSpan)

class PhraseEntityMatcher(object):
  """
    Phrase Entity Recognition class

    How to use it:

    .. code-block:: python

      from PhraseEntityMatcher import PhraseEntityMatcher
      nlp = spacy.load("en_core_web_sm")
      phraseList = ["safety cage", "cage", "pump"]
      pmatcher = PhraseEntityMatcher(nlp, 'ssc', phraseList)
      doc = nlp("The shaft deflection is causing the safety cage to rattle. Pumps not experiencing enough flow for the pumps to keep the check valves open during test. Pump not experiencing enough flow during test. Shaft made noise. Vibration seems like it is coming from the shaft.")
      updatedDoc = pmatcher(doc)

    or:

    .. code-block:: python

      nlp.add_pipe('phrase_entity_matcher', config={"label": "ssc", "patterns":["safety cage", "pump"], "asSpan":True})
      newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, label, patterns, asSpan=True, callback=None):
    """
    Args:

      label: str, the name/label for the patterns in patterns
      patterns: list, the phrase list, for example, phraseList = ["hello", "world"]
    """
    self.name = 'phrase_entity_matcher'
    self.matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(text) for text in patterns]
    self.matcher.add(label, patterns, on_match=callback)
    self.asSpan = asSpan

  def __call__(self, doc, replace=False):
    """
    Args:

      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines
      replace (bool): if True, relabel duplicated entity with new label
    """
    matches = self.matcher(doc, as_spans=self.asSpan)
    spans = []
    if not self.asSpan:
      for label, start, end in matches:
        span = Span(doc, start, end, label=label)
        spans.append(span)
    else:
      spans.extend(matches)
    # order matters here, for duplicated entities, only the first one will keep.
    if replace:
      doc.ents = filter_spans(spans+list(doc.ents))
    else:
      doc.ents = filter_spans(list(doc.ents)+spans)
    return doc
