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
    # Split existing entities into custom NER (kept) and spaCy's built-in NER.
    # Custom entities take precedence over built-in NER (see the else branch below),
    # and existing custom NER is preferred over newly matched custom NER.
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
      # filter_spans keeps the longest span on overlap, so a multi-token built-in
      # span (e.g. PERSON/CARDINAL) would otherwise swallow a single-token custom
      # entity. Resolve custom entities first, then keep only the built-in NER
      # spans that don't overlap them.
      custom = filter_spans(old+spans)
      occupied = {i for span in custom for i in range(span.start, span.end)}
      kept_ner = [span for span in ner if not any(i in occupied for i in range(span.start, span.end))]
      doc.ents = filter_spans(custom+kept_ner)
    return doc
