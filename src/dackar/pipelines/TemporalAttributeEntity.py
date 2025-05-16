# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

from spacy.language import Language

from ..utils.nlp.CreatePatterns import CreatePatterns
from ..config import nlpConfig

import logging
logger = logging.getLogger(__name__)

@Language.factory("temporal_attribute_entity", default_config={"patterns": None})
def create_temporal_attribute_component(nlp, name, patterns):
  return TemporalAttributeEntity(nlp, patterns=patterns)


class TemporalAttributeEntity(object):
  """
    How to use it:

    .. code-block:: python

      from TemporalAttributeEntity import TemporalAttributeEntity
      nlp = spacy.load("en_core_web_sm")
      patterns = {'label': 'temporal_attribute', 'pattern': [{'LOWER': 'about'}], 'id': 'temporal_attribute'}
      cmatcher = TemporalAttributeEntity(nlp, patterns)
      doc = nlp("It is close to 5pm.")
      updatedDoc = cmatcher(doc)

    or:

    .. code-block:: python

      nlp.add_pipe('temporal_attribute_entity', config={"patterns": {'label': 'temporal_attribute_entity', 'pattern': [{'LOWER': 'about'}], 'id': 'temporal_attribute_entity'}})
      newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, patterns=None, callback=None):
    """
    Args:

      nlp: spacy nlp model
      patterns: list/dict
    """
    self.name = 'temporal_attribute_entity'
    if patterns is None:
      # update to use config file instead
      filename = nlpConfig['files']['time_keywords_file']
      temporalPatterns = CreatePatterns(filename, entLabel='temporal_attribute', nlp=nlp)
      patterns = temporalPatterns.getPatterns()
    if not isinstance(patterns, list) and isinstance(patterns, dict):
      patterns = [patterns]
    # do we need to pop out other pipes?
    if not nlp.has_pipe('entity_ruler'):
      nlp.add_pipe('entity_ruler')
    self.entityRuler = nlp.get_pipe('entity_ruler')
    self.entityRuler.add_patterns(patterns)

  def __call__(self, doc):
    """
    Args:

      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines
    """
    doc = self.entityRuler(doc)
    return doc
