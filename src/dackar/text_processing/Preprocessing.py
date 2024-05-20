# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on October, 2022

@author: dgarrett622, wangc, mandd
"""
from cytoolz import functoolz
import re
import textacy.preprocessing as preprocessing
from numerizer import numerize
import spacy
from spacy.vocab import Vocab

try:
  from contextualSpellCheck.contextualSpellCheck import ContextualSpellCheck
except ModuleNotFoundError as error:
  print("ERROR: Unable to import contextualSpellCheck", error)
  print("Please try to install it via: 'pip install contextualSpellCheck'")
try:
  import autocorrect
except ModuleNotFoundError as error:
  print("ERROR: Unable to import autocorrect", error)
  print("Please try to install it via: 'pip install autocorrect'")
try:
  from spellchecker import SpellChecker as PySpellChecker
except ModuleNotFoundError as error:
  print("ERROR: Unable to import spellchecker", error)
  print("Please try to install it via: 'pip install spellchecker'")

import itertools
import os
import numpy as np
import pandas as pd

from ..similarity.simUtils import wordsSimilarity

# list of available preprocessors in textacy.preprocessing.normalize
textacyNormalize = ['bullet_points',
                    'hyphenated_words',
                    'quotation_marks',
                    'repeating_chars',
                    'unicode',
                    'whitespace']
# list of available preprocessors in textacy.preprocessing.remove
textacyRemove = ['accents',
                 'brackets',
                 'html_tags',
                 'punctuation']
# list of available preprocessors in textacy.preprocessing.replace
textacyReplace = ['currency_symbols',
                  'emails',
                  'emojis',
                  'hashtags',
                  'numbers',
                  'phone_numbers',
                  'urls',
                  'user_handles']
# list of available preprocessors from numerizer
numerizer = ['numerize']

preprocessorDefaultList = ['bullet_points',
                    'hyphenated_words',
                    'quotation_marks',
                    'repeating_chars',
                    'whitespace',
                    'unicode',
                    'accents',
                    'html_tags',
                    'punctuation',
                    'emails',
                    'emojis',
                    'hashtags',
                    'urls',
                    'numerize',
                    'whitespace']

preprocessorDefaultOptions = {'repeating_chars': {'chars': ',', 'maxn': 1},
                        'unicode': {'form': 'NFKC'},
                        'accents': {'fast': False},
                        'punctuation': {'only':["*","+",":","=","\\","^","_","|","~", "..", "..."]}}

# TODO: replace & --> and, @ --> at, maybe "/" --> or

class Preprocessing(object):
  """
    NLP Preprocessing class
  """

  def __init__(self, preprocessorList=preprocessorDefaultList, preprocessorOptions=preprocessorDefaultOptions):
    """
      Preprocessing object constructor

      Arg:
        preprocessorList: list, list of preprocessor names as strings
        preprocessorOptions: dict, dictionary of dictionaries containing optional arguments for preprocessors
        top level key is name of preprocessor

      Return:
        None
    """
    self.functionList = [] # list of preprocessor functions
    self.preprocessorNames = textacyNormalize + textacyRemove + textacyReplace + numerizer

    # collect preprocessor functions in a list
    for name in preprocessorList:
      # strip out options for preprocessor
      if name in preprocessorOptions:
        options = preprocessorOptions[name]
      else:
        options = {}
      # build the function to do the preprocessing
      if name in textacyNormalize:
        self.createTextacyNormalizeFunction(name, options)
      elif name in textacyRemove:
        self.createTextacyRemoveFunction(name, options)
      elif name in textacyReplace:
        self.createTextacyReplaceFunction(name, options)
      elif name in numerizer:
        # create function to store in functionList
        self.functionList.append(lambda x: numerize(x, ignore=['a ', 'A', 'second']))
      else:
        print(f'{name} is ignored! \nAvailable preprocessors: {self.preprocessorNames}')

    # create the preprocessor pipeline (composition of functionList)
    self.pipeline = functoolz.compose_left(*self.functionList)

  def createTextacyNormalizeFunction(self, name, options):
    """
      Creates a function from textacy.preprocessing.normalize such that only argument is a string
      and adds it to the functionList

      Args:
        name: str, name of the preprocessor
        options: dict, dictionary of preprocessor options

      Returns:
        None
    """
    # check for optional arguments
    useChars, useMaxn, useForm = False, False, False
    # options for repeating_chars
    if 'chars' in options and isinstance(options['chars'], str):
      # if chars is not str, it gets ignored
      useChars = True
    if 'maxn' in options and isinstance(options['maxn'], int):
      # if maxn is not int, it gets ignored
      useMaxn = True
    # option for unicode
    if 'form' in options and isinstance(options['form'], str):
      # if form is not str, it gets ignored
      useForm = True

    # build function for the pipeline
    if useChars or useMaxn or useForm:
      # include optional arguments
      f = lambda x: getattr(preprocessing.normalize, name)(x, **options)
    else:
      # no options need to be included
      f = lambda x: getattr(preprocessing.normalize, name)(x)

    # add to the functionList
    self.functionList.append(f)

  def createTextacyRemoveFunction(self, name, options):
    """
      Creates a function from textacy.preprocessing.remove such that the only argument is a string
      and adds it to the functionList

      Args:
        name: str, name of the preprocessor
        options: dict, dictionary of preprocessor options

      Returns:
        None
    """
    # check for optional arguments
    useFast, useOnly = False, False
    # option for accents
    if 'fast' in options and isinstance(options['fast'], bool):
      # if fast is not bool, it gets ignored
      useFast = True
    # option for brackets and punctuation
    if 'only' in options and isinstance(options['only'], (str, list, tuple)):
      # if only is not str, list, or tuple, it gets ignored
      useOnly = True

    # build function for the pipeline
    if useFast or useOnly:
      # include optional arguments
      f = lambda x: getattr(preprocessing.remove, name)(x, **options)
    else:
      # no options need to be included
      f = lambda x: getattr(preprocessing.remove, name)(x)

    # add to the functionList
    self.functionList.append(f)

  def createTextacyReplaceFunction(self, name, options):
    """
      Creates a function from textacy.preprocessing.replace such that the only argument is a string
      and adds it to the functionList

      Args:
        name: str, name of the preprocessor
        options: dict, dictionary of preprocessor options

      Returns:
        None
    """
    # check for optional arguments
    useRepl = False
    if 'repl' in options and isinstance(options['repl'], str):
      # if repl is not str, it gets ignored
      useRepl = True

    # build function for the pipeline
    if useRepl:
      # include optional argument
      f = lambda x: getattr(preprocessing.replace, name)(x, **options)
    else:
      # no options need to be included
      f = lambda x: getattr(preprocessing.replace, name)(x)

    # add to the functionList
    self.functionList.append(f)

  def __call__(self, text):
    """
      Performs the preprocessing

      Args:
        text: str, string of text to preprocess

      Returns:
        processed: str, string of processed text
    """
    processed = text.strip('\n')
    processed = re.sub(r'&', ' and ', processed)
    # processed = re.sub(r'/', ' and ', processed)
    processed = re.sub(r'@', ' at ', processed)
    processed = self.pipeline(processed)
    return processed

class SpellChecker(object):
  """
    Object to find misspelled words and automatically correct spelling

    Note: when using autocorrect, one need to conduct a spell test to identify the threshold (the word frequences)
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
    # get included and additional dictionary words and update speller dictionary
    if self.checker == 'autocorrect':
      self.speller = autocorrect.Speller()
      self.includedWords = []
      file2open = os.path.join(os.path.dirname(__file__) , os.pardir, os.pardir, os.pardir, 'data' , 'ac_additional_words.txt')
      with open(file2open, 'r') as file:
        tmp = file.readlines()
      self.addedWords = list({x.replace('\n', '') for x in tmp})
      self.speller.nlp_data.update({x: 1000000 for x in self.addedWords})
    elif self.checker == 'pyspellchecker':
      self.speller = PySpellChecker()
      self.includedWords = []
      file2open = os.path.join(os.path.dirname(__file__) , os.pardir, os.pardir, os.pardir, 'data' , 'psc_additional_words.txt')
      with open(file2open, 'r') as file:
        tmp = file.readlines()
      self.addedWords = list({x.replace('\n', '') for x in tmp})
      self.speller.word_frequency.load_words(self.addedWords)
    else:
      name = 'contextual spellcheck'
      self.nlp = spacy.load('en_core_web_lg')
      self.speller = ContextualSpellCheck(self.nlp, name)
      self.includedWords = list(self.speller.BertTokenizer.get_vocab().keys())
      file2open = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'data' , 'csc_additional_words.txt')
      with open(file2open, 'r') as file:
        tmp = file.readlines()
      self.addedWords = [x.replace('\n', '') for x in tmp]
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
      misspelled = [word for word in original if word not in self.speller.nlp_data]
      if None in misspelled:
        misspelled.remove(None)
    elif self.checker == 'pyspellchecker':
      original = re.findall(r'[^\s!,.?":;-]+', text)
      misspelled = self.speller.unknown(original)
    else:
      doc = self.nlp(text)
      doc = self.speller(doc)
      misspelled = list({str(x) for x in doc._.suggestions_spellCheck.keys()})

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
      l = re.split("([A-Za-z]+(?=\s|\.))", text)
      corrected = []
      for elem in l:
        if len(elem) == 0:
          corrected.append(elem)
        elif not re.search("[^A-Za-z]+",elem):
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
      This database contains the most common abbreviations collected from literarture and
      it provides for each abbreviation its corresponding full word(s); an abbreviation might
      have multple words associated. In such case the full word that makes more sense given the
      context is chosen (see findOptimalOption method)

      Args:
        abbrDatabase: pandas dataframe, dataframe containing library of abbreviations
        and their correspoding full expression
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
        and their correspoding full expression

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
      This database contains the most common abbreviations collected from literarture and
      it provides for each abbreviation its corresponding full word(s); an abbreviation might
      have multple words associated. In such case the full word that makes more sense given the
      context is chosen (see findOptimalOption method)

      Args:
        abbrDict: dictionary, dictionary containing library of abbreviations
        and their correspoding full expression
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


class AbbrExpander(object):
  """
    Class to expand abbreviations
  """

  def __init__(self, abbreviationsFilename, checkerType='autocorrect', abbrType='mixed'):
    """
      Abbreviation expander constructor

      Args:
        abbreviationsFilename: string, filename of abbreviations data

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
