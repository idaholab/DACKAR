# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

from spacy.tokens import Span
from spacy.language import Language
from quantulum3 import parser
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans

import logging
logging.getLogger('quantulum3').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Structured measurement data attached to each matched span
if not Span.has_extension('measurement'):
  Span.set_extension('measurement', default=None)


@Language.factory("unit_entity")
def create_unit_component(nlp, name):
  return UnitEntity(nlp)

class UnitEntity(object):
  """
    Unit Entity Recognition class

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
    quants = parser.parse(doc.text)
    newEnts = []
    for quant in quants:
      entity_name = quant.unit.entity.name
      unit_name = quant.unit.name
      # Exclude time — handled by TemporalEntity
      if entity_name == 'time':
        continue
      # Exclude dimensionless except percentages, which are meaningful in plant context
      is_percentage = 'percent' in unit_name.lower() or unit_name.strip() == '%'
      if entity_name == 'dimensionless' and not is_percentage:
        continue
      start, end = quant.span
      # alignment_mode="expand" handles trailing punctuation robustly
      span = doc.char_span(start, end, label=self.label, alignment_mode="expand")
      if span is None:
        continue
      span._.measurement = {
        'value': quant.value,
        'unit': unit_name,
        'entity_type': entity_name,
      }
      newEnts.append(span)
    doc.ents = filter_spans(newEnts + list(doc.ents))
    return doc
