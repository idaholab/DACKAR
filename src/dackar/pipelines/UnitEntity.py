# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
from quantulum3 import parser
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans

import logging
logging.getLogger('quantulum3').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@Language.factory("unit_entity")
def create_unit_component(nlp, name):
  return UnitEntity(nlp)

class UnitEntity(object):
  """
    How to use it:

    .. code-block:: python

      from UnitEntity import UnitEntity
      nlp = spacy.load("en_core_web_sm")
      unit = UnitEntity(nlp, 'ssc')
      doc = nlp("The shaft deflection is causing the safety cage to rattle. Pumps not experiencing enough flow for the pumps to keep the check valves open during test. Pump not experiencing enough flow during test. Shaft made noise. Vibration seems like it is coming from the shaft.")
      updatedDoc = unit(doc)

    or:

    .. code-block:: python

      nlp.add_pipe('unit_entity', config={"label": "ssc", "asSpan":True})
      newDoc = nlp(doc.text)
  """

  def __init__(self, nlp):
    """
    Args:
      nlp: spacy nlp model
    """
    self.name = 'unit_entity'
    self.label = 'unit'
    self.nlp = nlp

  def __call__(self, doc):
    """
    Args:

      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines
    """
    text = doc.text
    quants = parser.parse(text)
    newEnts = []
    for quant in quants:
      if quant.unit.entity.name not in ['dimensionless', 'time']:
        start, end = quant.span
        span = doc[:].char_span(start, end, label=self.label)
        # When '.' is used at the end of unit, the char_span will return None
        # The workaround is to include '.' in the span
        # Other solution, text can be preprocessed to strip '.'
        if span is None:
          span = doc[:].char_span(start, end+1, label=self.label)
        newEnts.append(span)
    doc.ents = filter_spans(newEnts+list(doc.ents))
    return doc
