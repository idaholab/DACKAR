# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

from spacy.language import Language

from ..utils.nlp.CreatePatterns import CreatePatterns
from ..config import nlpConfig

import logging
logger = logging.getLogger(__name__)

@Language.factory("conjecture_entity", default_config={"patterns": None})
def create_conjecture_component(nlp, name, patterns):
  return ConjectureEntity(nlp, patterns=patterns)


class ConjectureEntity(object):
  """
    How to use it:

    .. code-block:: python

      from ConjectureEntity import ConjectureEntity
      nlp = spacy.load("en_core_web_sm")
      patterns = {'label': 'conjecture', 'pattern': [{'LOWER': 'possible'}], 'id': 'conjecture'}
      cmatcher = ConjectureEntity(nlp, patterns)
      doc = nlp("Vibration seems like it is coming from the shaft.")
      updatedDoc = cmatcher(doc)

    or:

    .. code-block:: python

      nlp.add_pipe('conjecture_entity', config={"patterns": {'label': 'conjecture', 'pattern': [{'LOWER': 'possible'}], 'id': 'conjecture'}})
      newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, patterns=None, callback=None):
    """
    Args:

      nlp: spacy nlp model
      patterns: list/dict, patterns for conjecture entity
    """
    self.name = 'conjecture_entity'
    if patterns is None:
      # update to use config file instead
      filename = nlpConfig['files']['conjecture_keywords_file']
      conjecturePatterns = CreatePatterns(filename, entLabel='conjecture', nlp=nlp)
      patterns = conjecturePatterns.getPatterns()
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
