# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on October, 2022

@author: dgarrett622, wangc, mandd
"""
from cytoolz import functoolz
import re
import textacy.preprocessing as preprocessing
from numerizer import numerize
import logging

logger = logging.getLogger('DACKAR.Preprocessing')

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
    logger.info('Preprocess raw text data')
    processed = text.strip('\n')
    processed = re.sub(r'&', ' and ', processed)
    # processed = re.sub(r'/', ' and ', processed)
    processed = re.sub(r'@', ' at ', processed)
    processed = self.pipeline(processed)
    return processed
