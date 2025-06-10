# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on October, 2022

@author: mandd, wangc
"""
import re
import itertools
import numpy as np
import spacy
from spacy.vocab import Vocab
import logging

logger = logging.getLogger(__name__)

try:
  from contextualSpellCheck.contextualSpellCheck import ContextualSpellCheck
except ModuleNotFoundError as error:
  logger.error(f"Unable to import contextualSpellCheck: {error}")
  logger.info("Please try to install it via: 'pip install contextualSpellCheck'")
try:
  import autocorrect
except ModuleNotFoundError as error:
  logger.error(f"Unable to import autocorrect: {error}")
  logger.info("Please try to install it via: 'pip install autocorrect'")
try:
  from spellchecker import SpellChecker as PySpellChecker
except ModuleNotFoundError as error:
  logger.error(f"Unable to import spellchecker: {error}")
  logger.info("Please try to install it via: 'pip install spellchecker'")


from ..similarity.simUtils import wordsSimilarity
from ..config import nlpConfig

class SpellChecker(object):
  """
    Object to find misspelled words and automatically correct spelling

    Note: when using autocorrect, one need to conduct a spell test to identify the threshold (the word frequencies)
  """

  def __init__(self, checker='autocorrect'):
    """
      SpellChecker object constructor

      Args:
        checker: str, optional, spelling corrector to use ('autocorrect' or 'ContextualSpellCheck')

      Returns:
        None
    """
    self.checker = checker.lower()
    self.addedWords = []
    self.includedWords = []
    if 'extra_vocab' in nlpConfig['files']:
      file2open = nlpConfig['files']['extra_vocab']
      with open(file2open, 'r') as file:
        tmp = file.readlines()
        self.addedWords = list({x.replace('\n', '') for x in tmp})
    # get included and additional dictionary words and update speller dictionary
    if self.checker == 'autocorrect':
      self.speller = autocorrect.Speller()
      self.speller.nlp_data.update({x: 1000000 for x in self.addedWords})
    elif self.checker == 'pyspellchecker':
      self.speller = PySpellChecker()
      self.speller.word_frequency.load_words(self.addedWords)
    else:
      name = 'contextual spellcheck'
      languageModel = nlpConfig['params']['spacy_language_pipeline']
      self.nlp = spacy.load(languageModel)
      self.speller = ContextualSpellCheck(self.nlp, name)
      self.includedWords = list(self.speller.BertTokenizer.get_vocab().keys())
      self.speller.vocab = Vocab(strings=self.includedWords+self.addedWords)

  def addWordsToDictionary(self, words):
    """
      Adds a list of words to the spell check dictionary

      Args:
        words: list, list of words to add to the dictionary

      Returns:
        None
    """
    if self.checker == 'autocorrect':
      self.speller.nlp_data.update({word: 1000000 for word in words})
    elif self.checker == 'pyspellchecker':
      self.speller.word_frequency.load_words(self.addedWords+words)
    else:
      self.speller.vocab = Vocab(strings=self.includedWords+self.addedWords+words)

  def getMisspelledWords(self, text):
    """
      Returns a list of words that are misspelled according to the dictionary used

      Args:
        None

      Returns:
        misspelled: list, list of misspelled words
    """
    if self.checker == 'autocorrect':
      # corrected = self.speller(text.lower())
      original = re.findall(r'[^\s!,.?":;-]+', text)
      # auto = re.findall(r'[^\s!,.?":;-]+', corrected)
      # misspelled = list({w1 if w1.lower() != w2.lower() else None for w1, w2 in zip(original, auto)})
      misspelled = {word for word in original if word not in self.speller.nlp_data}
      if None in misspelled:
        misspelled.remove(None)
    elif self.checker == 'pyspellchecker':
      original = re.findall(r'[^\s!,.?":;-]+', text)
      misspelled = self.speller.unknown(original)
    else:
      doc = self.nlp(text)
      doc = self.speller(doc)
      misspelled = {str(x) for x in doc._.suggestions_spellCheck.keys()}

    return misspelled

  def correct(self, text):
    """
      Performs automatic spelling correction and returns corrected text

      Args:
        None

      Returns:
        corrected: str, spelling corrected text
    """
    if self.checker == 'autocorrect':
      corrected = self.speller(text)
    elif self.checker == 'pyspellchecker':
      l = re.split(r"([A-Za-z]+(?=\s|\.))", text)
      corrected = []
      for elem in l:
        if len(elem) == 0:
          corrected.append(elem)
        elif not re.search(r"[^A-Za-z]+",elem):
          if elem in self.speller:
            corrected.append(elem)
          else:
            corrected.append(self.speller.correction(elem))
        else:
          corrected.append(elem)
      corrected = "".join(corrected)
    else:
      doc = self.nlp(text)
      doc = self.speller(doc)
      corrected = doc._.outcome_spellCheck

    return corrected

  def handleAbbreviations(self, abbrDatabase, text, type):
    """
      Performs automatic correction of abbreviations and returns corrected text
      This method relies on a database of abbreviations located at:
      `src/nlp/data/abbreviations.xlsx`
      This database contains the most common abbreviations collected from literature and
      it provides for each abbreviation its corresponding full word(s); an abbreviation might
      have multiple words associated. In such case the full word that makes more sense given the
      context is chosen (see findOptimalOption method)

      Args:
        abbrDatabase: pandas dataframe, dataframe containing library of abbreviations
        and their corresponding full expression
        text: str, string of text that will be analyzed
        type: string, type of abbreviation method ('spellcheck','hard','mixed') that are employed
        to determine which words are abbreviations that need to be expanded
        * spellcheck: in this case spellchecker is used to identify words that
        are not recognized
        * hard: here we directly search for the abbreviations in the provided
        sentence
        * mixed: here we perform first a "hard" search followed by a "spellcheck"
        search

      Returns:
        options: list, list of corrected text options
    """
    abbreviationSet = set(abbrDatabase['Abbreviation'].values)
    if type == 'spellcheck':
      unknowns = self.getMisspelledWords(text)
    elif type == 'hard' or type=='mixed':
      unknowns = []
      splitSent = text.split()
      for word in splitSent:
        if word.lower() in abbreviationSet:
          unknowns.append(word)
      if type=='mixed':
        set1 = set(self.getMisspelledWords(text))
        set2 = set(unknowns)
        unknowns = list(set1.union(set2))

    corrections={}
    for word in unknowns:
      if word.lower() in abbrDatabase['Abbreviation'].values:
        locs = list(abbrDatabase['Abbreviation'][abbrDatabase['Abbreviation']==word.lower()].index.values)
        if locs:
          corrections[word] = abbrDatabase['Full'][locs].values.tolist()
        else:
          print(word)
      else:
        # Here we are addressing the fact that the abbreviation database will never be complete
        # Given an abbreviation that is not part of the abbreviation database, we are looking for a
        # a subset of abbreviations the abbreviation database that are close enough (and consider
        # them as possible candidates
        from difflib import SequenceMatcher
        corrections[word] = []
        abbreviationDS = abbrDatabase['Abbreviation'].values
        for index,abbr in enumerate(abbreviationDS):
          if SequenceMatcher(None, word, abbr).ratio()>0.8:
            corrections[word].append(abbrDatabase['Full'].values.tolist()[index])
      if not corrections[word]:
        corrections.pop(word)

    combinations = list(itertools.product(*list(corrections.values())))
    options = []
    for comb in combinations:
      corrected = text
      for index,key in enumerate(corrections.keys()):
        corrected = re.sub(r"\b%s\b" % str(key) , comb[index], corrected)
      options.append(corrected)

    if not options:
      return text
    else:
      bestOpt = self.findOptimalOption(options)
      return bestOpt

  def generateAbbrDict(self, abbrDatabase):
    """
      Generates an AbbrDict that can be used by handleAbbreviationsDict

      Args:
        abbrDatabase: pandas dataframe, dataframe containing library of abbreviations
        and their corresponding full expression

      Returns:
        abbrDict: dictionary, a abbreviations dictionary
    """
    abbrDict = {}
    #There may be a more efficient way to do the following
    for row in abbrDatabase.itertuples():
      abbrs = abbrDict.get(row.Abbreviation,[])
      abbrs.append(row.Full)
      abbrDict[row.Abbreviation] = abbrs
    return abbrDict

  def handleAbbreviationsDict(self, abbrDict, text, type):
    """
      Performs automatic correction of abbreviations and returns corrected text
      This method relies on a database of abbreviations located at:
      src/nlp/data/abbreviations.xlsx
      This database contains the most common abbreviations collected from literature and
      it provides for each abbreviation its corresponding full word(s); an abbreviation might
      have multple words associated. In such case the full word that makes more sense given the
      context is chosen (see findOptimalOption method)

      Args:
        abbrDict: dictionary, dictionary containing library of abbreviations
        and their corresponding full expression
        text: str, string of text that will be analyzed
        type: string, type of abbreviation method ('spellcheck','hard','mixed') that are employed
        to determine which words are abbreviations that need to be expanded
        * spellcheck: in this case spellchecker is used to identify words that
        are not recognized
        * hard: here we directly search for the abbreviations in the provided
        sentence
        * mixed: here we perform first a "hard" search followed by a "spellcheck"
        search

      Return:
        options: list, list of corrected text options
    """
    if type == 'spellcheck':
      unknowns = self.getMisspelledWords(text)
    elif type == 'hard' or type=='mixed':
      unknowns = []
      splitSent = text.split()
      for word in splitSent:
        if word.lower() in abbrDict.keys():
          unknowns.append(word)
      if type=='mixed':
        set1 = set(self.getMisspelledWords(text))
        set2 = set(unknowns)
        unknowns = list(set1.union(set2))

    corrections={}
    for word in unknowns:
      if word.lower() in abbrDict.keys():
        if len(abbrDict[word.lower()]) > 0:
          corrections[word] = abbrDict[word.lower()]
      else:
        # Here we are addressing the fact that the abbreviation database will never be complete
        # Given an abbreviation that is not part of the abbreviation database, we are looking for a
        # a subset of abbreviations the abbreviation database that are close enough (and consider
        # them as possible candidates
        from difflib import SequenceMatcher
        corrections[word] = []
        abbreviationDS = list(abbrDict)
        for index,abbr in enumerate(abbreviationDS):
          val=0
          newVal = SequenceMatcher(None, word, abbr).ratio()
          if newVal>=0.75 and newVal>val:
            corrections[word] = abbrDict[abbr]
            val = newVal
      if not corrections[word]:
        corrections.pop(word)

    combinations = list(itertools.product(*list(corrections.values())))
    options = []
    for comb in combinations:
      corrected = text
      for index,key in enumerate(corrections.keys()):
        corrected = re.sub(r"\b%s\b" % str(key) , comb[index], corrected)
      options.append(corrected)

    if not options:
      return text
    else:
      bestOpt = self.findOptimalOption(options)
      return bestOpt

  def findOptimalOption(self,options):
    """
      Method to handle abbreviation with multiple meanings

      Args:
        options: list, list of sentence options

      Return:
        optimalOpt: string, option from the provided options list that fits more the
        possible
    """
    nOpt = len(options)
    combScore = np.zeros(nOpt)
    for index,opt in enumerate(options):
      listOpt = opt.split()
      for i,word in enumerate(listOpt):
        for j in range(i+1,len(listOpt)):
          combScore[index] = combScore[index] + wordsSimilarity(word,listOpt[j])
    optIndex = np.argmax(combScore)
    optimalOpt = options[optIndex]
    return optimalOpt
