# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED


from spacy.language import Language
from .SimpleEntityMatcher import SimpleEntityMatcher

import logging
logger = logging.getLogger(__name__)

@Language.factory("EmergentActivity")
def create_emergent_activity(nlp, name):
  return EmergentActivity(nlp)

class EmergentActivity(object):
  """
    Emergent Activity Entity Recognition class

    How to use it:

    .. code-block:: python

      from EmergentActivity import EmergentActivity
      nlp = spacy.load("en_core_web_sm")
      pmatcher = EmergentActivity(nlp)
      doc = nlp("The shaft deflection is causing the safety cage to rattle. Pumps not experiencing enough flow for the pumps to keep the check valves open during test. Pump not experiencing enough flow during test. Shaft made noise. Vibration seems like it is coming from the shaft.")
      updatedDoc = pmatcher(doc)

    or:

    .. code-block:: python

      nlp.add_pipe('EmergentActivity')
      newDoc = nlp(doc.text)
  """

  def __init__(self, nlp):
    """
    Args:

      nlp: spacy nlp model
    """
    self.name = 'EmergentActivity'
    # work order identification should always before other identification
    # This is because when spans overlap, the (first) longest span is preferred over shorter spans.
    woPattern = [[{"LOWER": "wo"}, {"IS_PUNCT": True, "OP":"*"}, {"IS_DIGIT": True}], [{"TEXT":{"REGEX":"(?<=wo)\d+"}}]]
    idPattern = [[{"TEXT":{"REGEX":"(?=\S*[a-zA-Z])(?=\S*[0-9])"}}]]
    # idPattern = [[{"TEXT":{"REGEX":"^(?=.*\b(?=\S*[a-zA-Z])(?=\S*[0-9]))"}}]]

    self.matcher = SimpleEntityMatcher(nlp, label='WO', patterns=woPattern)
    self.matcher.matcher.add('ID', idPattern)
    self.asSpan = True

  def __call__(self, doc):
    """
    Args:

      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines
    """
    # 'replace' keyword can be used to prefer the new identified entity when duplicates exist.
    # doc = self.matcher(doc, replace=True)
    doc = self.matcher(doc, replace=False)
    return doc
