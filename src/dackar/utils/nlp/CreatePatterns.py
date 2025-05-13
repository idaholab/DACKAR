# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

import logging
import spacy
import pandas as pd

from .nlp_utils import generatePatternList
from ...config import nlpConfig


class CreatePatterns(object):

  def __init__(self, filename, entLabel, entID=None, nlp=None, *args, **kwargs):
    """

    """
    self.filename = filename
    self.label = entLabel
    if entID is None:
      self.id = entLabel
    else:
      self.id = entID
    self.entities = self.readFile()
    if nlp is None:
      language = nlpConfig['params']['spacy_language_pipeline']
      self.nlp = spacy.load(language, exclude=[])
    else:
      self.nlp = nlp
    self.patterns = self.generatePatterns()


  def readFile(self):
    """
    """
    # assume one column without column name for the csv file
    entList = pd.read_csv(self.filename).values.ravel().tolist()
    return entList


  def generatePatterns(self):
    """
    """
    patterns = generatePatternList(self.entities, label=self.label, id=self.id, nlp=self.nlp, attr="LEMMA")
    return patterns

  def getPatterns(self):
    """
    """
    return self.patterns
