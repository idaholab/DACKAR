# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.language import Language
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans

import logging
logger = logging.getLogger(__name__)

@Language.factory("simple_entity_matcher", default_config={"label": "ssc", "patterns":[{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}], "asSpan":True})
def create_simple_matcher_component(nlp, name, label, patterns, asSpan):
  return SimpleEntityMatcher(nlp, label, patterns, asSpan=asSpan)

class SimpleEntityMatcher(object):
  """
    Simple Entity Recognition class

    How to use it:

    .. code-block:: python

      from SimpleEntityMatcher import SimpleEntityMatcher
      nlp = spacy.load("en_core_web_sm")
      patterns = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
      pmatcher = SimpleEntityMatcher(nlp, 'ssc', patterns)
      doc = nlp("The shaft deflection is causing the safety cage to rattle. Pumps not experiencing enough flow for the pumps to keep the check valves open during test. Pump not experiencing enough flow during test. Shaft made noise. Vibration seems like it is coming from the shaft.")
      updatedDoc = pmatcher(doc)

    or:

    .. code-block:: python

      nlp.add_pipe('simple_entity_matcher', config={"label": "ssc", "patterns":[{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}], "asSpan":True})
      newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, label, patterns, asSpan=True, callback=None):
    """
    Args:

      nlp: spacy nlp model
      label: str, the name/label for the patterns in patterns
        patterns, list, the rules used to match the entities, for example,
        patterns = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
    """
    self.name = 'simple_entity_matcher'
    self.matcher = Matcher(nlp.vocab)
    if not isinstance(patterns, list):
      patterns = [patterns]
    if not isinstance(patterns[0], list):
      patterns = [patterns]
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
    # TODO: reorder entities as [existing.ner, new.ner, spacy.ner]
    # In this order, spacy.ner will be always replaced with custom ner, while the existing custom ner is preferred over
    # new custom ner.
    old = []
    ner = []
    spacyNERLabel = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
                     "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL",
                     "CARDINAL"]
    for span in doc.ents:
      if span.label_ in spacyNERLabel:
        ner.append(span)
      else:
        old.append(span)

    if replace:
      doc.ents = filter_spans(spans+list(doc.ents))
    else:
      # directly filtering will not replace existing spacy NERs.
      # doc.ents = filter_spans(list(doc.ents)+spans)
      doc.ents = filter_spans(old+spans+ner)
    return doc
