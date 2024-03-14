# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2024

@author: wangc, mandd
"""
import re
import logging
import pandas as pd
from ..config import nlpConfig

logger = logging.getLogger(__name__)

not_acronyms = [
    "level", "was", "for", "ship", "start", "band", "by", "row", "new", "gap",
    "technology", "system", "me", "our", "tag", "low", "east",
    "us", "an", "bolt", "ups", "fan", "team", "hit", "oat", "fit", "well", "same",
    "set", "case", "taps", "flux", "data", "leap", "col", "top", "arc", "side",
    "rooms", "amps", "had", "fleet", "best", "most", "cover", "ice", "min", "tee",
    "matrix", "bat", "rose", "rapid", "pipe", "its", "the", "it", "air", "and",
    "email", "if", "are", "from", "success", "led",'parts','tips','pad','stem','pond','sit','toe','that','get','op','can','feed']

class Abbreviation(object):
  """
    Class to handle abbreviations
  """

  def __init__(self):
    """
      Abbrviation expander constructor

      Args:
        abbreviationsFilename: string, filename of abbreviations data

      Return:
        None
    """
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    logger.info(f'Create instance of {self.name}')
    if 'abbreviation_file' in nlpConfig['files']:
      self.abbreviationsFilename = nlpConfig['files']['abbreviation_file']
      abbrList = pd.read_excel(self.abbreviationsFilename)
      self.abbrDict = dict(zip(abbrList['Abbreviation'], abbrList['Full']))
    else:
      self.abbreviationsFilename = None
      self.abbrDict = {}

  def abbreviationSub(self, text):
    """
      Expands the abbreviations in text

      Args:
        text: string, the text to expand

      Returns:
        expandedText: string, the text with abbreviations expanded
    """
    logger.info('Substitute abbreviations with their full expansions')
    text = text.replace("\n", "").lower()
    textList = [t.strip() for t in text.split('.')]
    expandedText = []
    for sent in textList:
      corrected = sent
      splitSent = sent.split()
      for word in splitSent:
        if word not in not_acronyms:
          if word in self.abbrDict.keys():
            full = self.abbrDict[word]
            if isinstance(full, str):
              corrected = re.sub(r"\b%s\b" % str(word) , full, corrected)
            elif isinstance(full, list) and len(full) == 1:
              corrected = re.sub(r"\b%s\b" % str(word) , full[0], corrected)
            else:
              logger.info(f'Can not replace abbreviation {word}, possible solution {full}')
      expandedText.append(corrected)

    expandedText = '. '.join(expandedText)
    return expandedText


  def updateAbbreviation(self, abbrDict, reset=True):
    """
      Update existing abbreviation dictionary

      Args:
        abbrDict: dict, provided abbreviation dictionary
        reset: boot, True if reset the existing abbreviation dictionary
    """
    updateDict = {}
    for k, v in abbrDict.items():
      if isinstance(v, str):
        updateDict[k.lower().strip()] = v.lower()
      elif isinstance(v, list):
        updateDict[k.lower().strip()] = [e.lower().strip() for e in v]
      else:
        pass
    if reset:
      self.abbrDict = updateDict
    else:
      self.abbrDict.update(updateDict)

  def getAbbreviation(self):
    """
      Get the abbreviation dict
    """
    return self.abbrDict
