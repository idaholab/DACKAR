# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on October, 2022

@author: wangc, mandd
"""
from .SpellChecker import SpellChecker
from .Preprocessing import Preprocessing
import pandas as pd


class AbbrExpander(object):
  """
    Class to expand abbreviations
  """

  def __init__(self, abbreviationsFilename, checkerType='autocorrect', abbrType='mixed'):
    """
      Abbreviation expander constructor

      Args:
        abbreviationsFilename: str, filename of abbreviations data
        checkerType: str, type for spell checker class, i.e., 'autocorrect', 'pyspellchecker', and 'contextual spellcheck', default is 'autocorrect'
        abbrType: str, type of abbreviation method ('spellcheck','hard','mixed') that are employed
        to determine which words are abbreviations that need to be expanded
        * spellcheck: in this case spellchecker is used to identify words that
        are not recognized
        * hard: here we directly search for the abbreviations in the provided
        sentence
        * mixed: here we perform first a "hard" search followed by a "spellcheck"
        search

      Return:
        None
    """
    self.abbrType = abbrType
    self.checkerType = checkerType

    self.abbrList = pd.read_excel(abbreviationsFilename)
    self.preprocessorList = ['hyphenated_words',
                             'whitespace',
                             'numerize']
    self.preprocess = Preprocessing(self.preprocessorList, {})
    self.checker = SpellChecker(checker=self.checkerType)
    self.abbrDict = self.checker.generateAbbrDict(self.abbrList)


  def abbrProcess(self, text, splitToList=False):
    """
      Expands the abbreviations in text

      Args:
        text: string, the text to expand

      Returns:
        expandedText: string, the text with abbreviations expanded
    """
    text = self.preprocess(text)
    if not splitToList:
      expandedText = self.checker.handleAbbreviationsDict(self.abbrDict, text.lower(), type=self.abbrType)
    else:
      text = text.replace("\n", "")
      textList = [t.strip() for t in text.split('.')]
      expandedText = []
      for t in textList:
        cleanedText = self.checker.handleAbbreviationsDict(self.abbrDict, t.lower(), type=self.abbrType)
        expandedText.append(cleanedText)
      expandedText = '. '.join(expandedText)
    return expandedText
