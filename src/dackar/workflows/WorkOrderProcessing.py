# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""
import logging
import os
import pandas as pd
import re
import copy
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token
from spacy.tokens import Span
from spacy import displacy
from spacy.matcher import PhraseMatcher
from spacy.matcher import DependencyMatcher
from spacy.util import filter_spans
from collections import deque
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)

from ..utils.nlp.nlp_utils import displayNER, resetPipeline, printDepTree
from ..utils.nlp.nlp_utils import extendEnt
## import pipelines
from ..pipelines.CustomPipelineComponents import normEntities
from ..pipelines.CustomPipelineComponents import initCoref
from ..pipelines.CustomPipelineComponents import aliasResolver
from ..pipelines.CustomPipelineComponents import anaphorCoref
from ..pipelines.CustomPipelineComponents import anaphorEntCoref
from ..pipelines.CustomPipelineComponents import mergePhrase
from ..pipelines.CustomPipelineComponents import pysbdSentenceBoundaries
from ..text_processing.Preprocessing import Preprocessing
from ..utils.utils import getOnlyWords, getShortAcronym
from ..config import nlpConfig

logger = logging.getLogger(__name__)

## temporary add stream handler
# ch = logging.StreamHandler()
# logger.addHandler(ch)
##

## coreferee module for Coreference Resolution
## Q? at which level to perform coreferee? After NER and perform coreferee on collected sentence
_corefAvail = False
try:
  # check the current version spacy>=3.0.0,<=3.3.0
  from packaging.version import Version
  ver = spacy.__version__
  valid = Version(ver)>=Version('3.0.0') and Version(ver)<=Version('3.3.0')
  if valid:
    # https://github.com/msg-systems/coreferee
    import coreferee
    _corefAvail = True
  else:
    logger.info(f'Module coreferee is not compatible with spacy version {ver}')
except ModuleNotFoundError:
  logger.info('Module coreferee can not be imported')

if not Span.has_extension('conjecture'):
  Span.set_extension('conjecture', default=False)
if not Span.has_extension('status'):
  Span.set_extension("status", default=None)
if not Span.has_extension('neg'):
  Span.set_extension("neg", default=None)
if not Span.has_extension('neg_text'):
  Span.set_extension("neg_text", default=None)
if not Span.has_extension('alias'):
  Span.set_extension("alias", default=None)

if not Token.has_extension('conjecture'):
  Token.set_extension('conjecture', default=False)
if not Token.has_extension('status'):
  Token.set_extension("status", default=None)
if not Token.has_extension('neg'):
  Token.set_extension("neg", default=None)
if not Token.has_extension('neg_text'):
  Token.set_extension("neg_text", default=None)
if not Token.has_extension('alias'):
  Token.set_extension("alias", default=None)


class WorkOrderProcessing(object):
  """
    Class to process OPG CWS work order dataset
  """
  def __init__(self, nlp, entLabel='SSC', *args, **kwargs):
    """
      Construct

      Args:

        nlp: spacy.Language object, contains all components and data needed to process text
        args: list, positional arguments
        kwargs: dict, keyword arguments

      Returns:

        None
    """
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    logger.info(f'Create instance of {self.name}')
    # orders of NLP pipeline: 'ner' --> 'normEntities' --> 'merge_entities' --> 'initCoref'
    # --> 'aliasResolver' --> 'coreferee' --> 'anaphorCoref'
    # pipeline 'merge_noun_chunks' can be used to merge phrases (see also displacy option)
    self.nlp = nlp
    self._statusFile = nlpConfig['files']['status_keywords_file']['all']
    self._statusKeywords = self.getKeywords(self._statusFile)
    self._updateStatusKeywords = False
    self._conjectureFile = nlpConfig['files']['conjecture_keywords_file']
    self._conjectureKeywords = self.getKeywords(self._conjectureFile, columnNames=['conjecture-keywords'])
    ## pipelines "merge_entities" and "merge_noun_chunks" can be used to merge noun phrases and entities
    ## for easier analysis
    if _corefAvail:
      self.pipelines = ['pysbdSentenceBoundaries',
                      'mergePhrase', 'normEntities', 'initCoref', 'aliasResolver',
                      'coreferee','anaphorCoref', 'anaphorEntCoref']
    else:
      self.pipelines = ['pysbdSentenceBoundaries',
                      'mergePhrase','normEntities', 'initCoref', 'aliasResolver',
                      'anaphorCoref', 'anaphorEntCoref']
    # ner pipeline is not needed since we are focusing on the keyword matching approach
    if nlp.has_pipe("ner"):
      nlp.remove_pipe("ner")
    nlp = resetPipeline(nlp, self.pipelines)
    self.nlp = nlp
    self._doc = None
    self.entityRuler = None
    self._entityRuler = False
    self._entityRulerMatches = []
    self._matchedSents = [] # collect data of matched sentences
    self._matchedSentsForVis = [] # collect data of matched sentences to be visualized
    self._visualizeMatchedSents = True
    self._coref = _corefAvail # True indicate coreference pipeline is available
    self._entityLabels = {} # labels for rule-based entities
    # reset entity label using toml input
    if 'params' in nlpConfig:
      entLabel = nlpConfig['params'].get('ent_label', entLabel)
    self._labelSSC = entLabel
    self._entHS = None
    self._entStatus = None
    self._textProcess = self.textProcess()

  def reset(self):
    """
      Reset rule-based matcher
    """
    self._matchedSents = []
    self._matchedSentsForVis = []
    self._entHS = None
    self._doc = None

  def textProcess(self):
    """
      Function to clean text

      Args:
        None

      Returns:
        procObj, DACKAR.Preprocessing object
    """
    procList = ['quotation_marks', 'punctuation', 'whitespace']
    procOptions = {'punctuation': {'only':["*","+",":","=","\\","^","_","|","~", "..", "...", ",", ";", "."]}}
    procObj = Preprocessing(preprocessorList=procList, preprocessorOptions=procOptions)
    return procObj


  def getKeywords(self, filename, columnNames=None):
    """
      Get the keywords from given file

      Args:

        filename: str, the file name to read the keywords

      Returns:

        kw: dict, dictionary contains the keywords
    """
    kw = {}
    if columnNames is not None:
      ds = pd.read_csv(filename, skipinitialspace=True, names=columnNames)
    else:
      ds = pd.read_csv(filename, skipinitialspace=True)
    for col in ds.columns:
      vars = set(ds[col].dropna())
      kw[col] = self.extractLemma(vars)
    return kw

  def extractLemma(self, varList):
    """
      Lammatize the variable list

      Args:

        varList: list, list of variables

      Returns:

        lemmaList: list, list of lammatized variables
    """
    lemmaList = []
    for var in varList:
      lemVar = [token.lemma_.lower() for token in self.nlp(var) if token.lemma_ not in ["!", "?", "+", "*"]]
      lemmaList.append(lemVar)
    return lemmaList

  def addKeywords(self, keywords, ktype):
    """
      Method to update self._causalKeywords or self._statusKeywords

      Args:

        keywords: dict, keywords that will be add to self._causalKeywords or self._statusKeywords
        ktype: string, either 'status' or 'causal'
    """
    if type(keywords) != dict:
      raise IOError('"addCausalKeywords" method can only accept dictionary, but got {}'.format(type(keywords)))
    if ktype.lower() == 'status':
      for key, val in keywords.items():
        if type(val) != list:
          val = [val]
        val = self.extractLemma(val)
        if key in self._statusKeywords:
          self._statusKeywords[key].append(val)
        else:
          logger.warning('keyword "{}" cannot be accepted, valid keys for the keywords are "{}"'.format(key, ','.join(list(self._statusKeywords.keys()))))

  def addEntityPattern(self, name, patternList):
    """
      Add entity pattern, to extend doc.ents, similar function to self.extendEnt

      Args:

        name: str, the name for the entity pattern.
        patternList: list, the pattern list, for example:
        {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}
    """
    if not self.nlp.has_pipe('entity_ruler'):
      self.nlp.add_pipe('entity_ruler', before='mergePhrase')
      self.entityRuler = self.nlp.get_pipe("entity_ruler")
    if not isinstance(patternList, list):
      patternList = [patternList]
    # TODO: able to check "id" and "label", able to use "name"
    for pa in patternList:
      label = pa.get('label')
      id = pa.get('id')
      if id is not None:
        if id not in self._entityLabels:
          self._entityLabels[id] = set([label]) if label is not None else set()
        else:
          self._entityLabels[id] = self._entityLabels[id].union(set([label])) if label is not None else set()
    # self._entityLabels += [pa.get('label') for pa in patternList if pa.get('label') is not None]
    self.entityRuler.add_patterns(patternList)
    if not self._entityRuler:
      self._entityRuler = True

  def __call__(self, text):
    """
      Find all token sequences matching the supplied pattern

      Args:

        text: string, the text that need to be processed

      Returns:

        None
    """
    # Merging Entity Tokens
    # We need to consider how to do this, I sugguest to first conduct rule based NER, then collect
    # all related sentences, then create new pipelines to perform NER with "merge_entities" before the
    # conduction of relationship extraction
    # if self.nlp.has_pipe('merge_entities'):
    #   _ = self.nlp.remove_pipe('merge_entities')
    # self.nlp.add_pipe('merge_entities')
    doc = self.nlp(text)
    self._doc = doc
    ## use entity ruler to identify entity
    # if self._entityRuler:
    #   logger.debug('Entity Ruler Matches:')
    #   print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents if ent.label_ in self._entityLabels[self._labelSSC]])

    # First identify coreference through coreferee, then filter it through doc.ents
    if self._coref:
      corefRep = doc._.coref_chains.pretty_representation
      if len(corefRep) != 0:
        logger.debug('Print Coreference Info:')
        print(corefRep)

    # collect sents that contains entities with label self._labelSSC
    matchedSents, matchedSentsForVis = self.collectSents(self._doc)
    # Except reset, add new processed data into existing stored data
    self._matchedSents += matchedSents
    self._matchedSentsForVis += matchedSentsForVis
    ## health status
    logger.info('Start to extract health status')
    self.extractHealthStatus(self._matchedSents)

    ## Access status and output to an ordered csv file
    entList = []
    aliasList = []
    statusList = []
    cjList = []
    negList = []
    negTextList = []
    for sent in self._matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._labelSSC])
      for ent in ents:
        if ent._.status is not None:
          entList.append(ent.text)
          aliasList.append(ent._.alias)
          statusList.append(ent._.status)
          cjList.append(ent._.conjecture)
          negList.append(ent._.neg)
          negTextList.append(ent._.neg_text)

    dfStatus = pd.DataFrame({'entity':entList, 'alias':aliasList, 'status':statusList, 'conjecture':cjList, 'negation':negList, 'negation_text': negTextList})
    dfStatus.to_csv(nlpConfig['files']['output_health_status_file'], columns=['entity', 'alias', 'status', 'conjecture', 'negation', 'negation_text'])

    self._entStatus = dfStatus
    logger.info('End of health status extraction!')
    # df = pd.DataFrame({'entities':entList, 'status keywords':kwList, 'health status':hsList, 'conjecture':cjList, 'sentence':sentList})
    # df.to_csv(nlpConfig['files']['output_health_status_file'], columns=['entities', 'status keywords', 'health statuses', 'conjecture', 'sentence'])

  def visualize(self):
    """
      Visualize the processed document

      Args:

        None

      Returns:

        None
    """
    if self._visualizeMatchedSents:
      # Serve visualization of sentences containing match with displaCy
      # set manual=True to make displaCy render straight from a dictionary
      # (if you're running the code within a Jupyer environment, you can
      # use displacy.render instead)
      # displacy.render(self._matchedSentsForVis, style="ent", manual=True)
      displacy.serve(self._matchedSentsForVis, style="ent", manual=True)

  ##########################
  # methods for relation extraction
  ##########################

  def isPassive(self, token):
    """
      Check the passiveness of the token

      Args:

        token: spacy.tokens.Token, the token of the doc

      Returns:

        isPassive: True, if the token is passive
    """
    if token.dep_.endswith('pass'): # noun
      return True
    for left in token.lefts: # verb
      if left.dep_ == 'auxpass':
        return True
    return False

  def isConjecture(self, token):
    """
      Check the conjecture of the token

      Args:

        token: spacy.tokens.Token, the token of the doc, the token should be the root of the Doc

      Returns:

        isConjecture: True, if the token/sentence indicates conjecture
    """
    for left in token.lefts: # Check modal auxiliary verb: can, could, may, might, must, shall, should, will, would
      if left.dep_.startswith('aux') and left.tag_ in ['MD']:
        return True
    if token.pos_ == 'VERB' and token.tag_ == 'VB': # If it is a verb, and there is no inflectional morphology for the verb
      return True
    # check the keywords
    # FIXME: should we use token.subtree or token.children here
    for child in token.subtree:
      if [child.lemma_.lower()] in self._conjectureKeywords['conjecture-keywords']:
        return True
    return False

  def isNegation(self, token):
    """
      Check negation status of given token

      Args:

        token: spacy.tokens.Token, token from spacy.tokens.doc.Doc

      Returns:

        (neg, text): tuple, the negation status and the token text
    """
    neg = False
    text = ''
    if token.dep_ == 'neg':
      neg = True
      text = token.text
      return neg, text
    # check left for verbs
    for left in token.lefts:
      if left.dep_ == 'neg':
        neg = True
        text = left.text
        return neg, text
    # The following can be used to check the negation status of the sentence
    # # check the subtree
    # for sub in token.subtree:
    #   if sub.dep_ == 'neg':
    #     neg = True
    #     text = sub.text
    #     return neg, text
    return neg, text

  def getCustomEnts(self, ents, labels):
    """
      Get the custom entities

      Args:

        ents: list, all entities from the processed doc
        labels: list, list of labels to be used to get the custom entities out of "ents"

      Returns:

        customEnts: list, the customEnts associates with the "labels"
    """
    customEnts = [ent for ent in ents if ent.label_ in labels]
    if len(customEnts) == 0:
      customEnts = None
    return customEnts


  def extractHealthStatus(self, matchedSents, predSynonyms=[], exclPrepos=[]):
    """
      Extract health status and relation

      Args:

        matchedSents: list, the matched sentences
        predSynonyms: list, predicate synonyms
        exclPrepos: list, exclude the prepositions
    """
    subjList = ['nsubj', 'nsubjpass', 'nsubj:pass']
    objList = ['pobj', 'dobj', 'iobj', 'obj', 'obl', 'oprd']

    # procedure to process OPG CWS data
    # collect status, negation, conjecture information
    for sent in matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._labelSSC])
      root = sent.root
      neg, negText = self.isNegation(root)
      conjecture = self.isConjecture(root)
      if ents is None:
        continue
      for ent in ents:
        ent._.set('neg', neg)
        ent._.set('neg_text', negText)
        ent._.set('conjecture', conjecture)
        if ent._.alias is not None:
          # entity at the beginning of sentence
          if ent.start == sent.start:
            status = sent[ent.end:]
            # some clean up for the text
            text = self._textProcess(status.text)
            ent._.set('status', text)
          # entity at the end of sentence
          elif ent.end == sent.end or (ent.end == sent.end - 1 and sent[-1].is_punct):
            text = sent.text
            # substitute entity ID with its alias
            text = re.sub(r"\b%s\b" % str(ent.text) , ent._.alias, text)
            text = self._textProcess(text)
            ent._.set('status', text)
          # entity in the middle of sentence
          else:
            entRoot = ent.root
            # Only include Pred and Obj info
            if entRoot.dep_ in subjList:
              status = sent[ent.end:]
              # some clean up for the text
              text = self._textProcess(status.text)
              ent._.set('status', text)
            # Include the whole info with alias substitution
            elif entRoot.dep_ in objList:
              text = sent.text
              # substitute entity ID with its alias
              text = re.sub(r"\b%s\b" % str(ent.text) , ent._.alias, text)
              text = getOnlyWords(text)
              text = self._textProcess(text)
              ent._.set('status', text)
        # other type of entities
        else:
          entRoot = ent.root
          if entRoot.dep_ in subjList:
            # depend on the application, can use self.getHealthStatusForSubj to get the status
            status = sent[ent.end:]
            # some clean up for the text
            text = self._textProcess(status.text)
            ent._.set('status', text)
          # Include the whole info with alias substitution
          elif entRoot.dep_ in objList:
            # depend on the application, can use self.getHealthStatusForObj to get the status
            text = sent.text
            text = getOnlyWords(text)
            text = self._textProcess(text)
            ent._.set('status', text)
          else:
            # if the entity not among subj and obj, it may not need to report it
            pass


  ##TODO: how to extend it for entity ruler?
  # @staticmethod
  def collectSents(self, doc):
    """
    collect data of matched sentences that can be used for visualization

      Args:
        doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines
    """
    matchedSents = []
    matchedSentsForVis = []
    for span in doc.ents:
      if span.ent_id_ != self._labelSSC:
        continue
      sent = span.sent
      # Append mock entity for match in displaCy style to matched_sents
      # get the match span by ofsetting the start and end of the span with the
      # start and end of the sentence in the doc
      matchEnts = [{
          "start": span.start_char - sent.start_char,
          "end": span.end_char - sent.start_char,
          "label": span.label_,
      }]
      if sent not in matchedSents:
        matchedSents.append(sent)
      matchedSentsForVis.append({"text": sent.text, "ents": matchEnts})
    return matchedSents, matchedSentsForVis
