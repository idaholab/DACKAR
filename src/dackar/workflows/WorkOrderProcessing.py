# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on March, 2024

@author: wangc, mandd
"""
import logging
import pandas as pd
import re
from spacy.tokens import Token
from spacy.tokens import Span

from ..text_processing.Preprocessing import Preprocessing
from ..utils.utils import getOnlyWords, getShortAcronym
from ..config import nlpConfig
from .WorkflowBase import WorkflowBase

logger = logging.getLogger(__name__)

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


class WorkOrderProcessing(WorkflowBase):
  """
    Class to process CWS work order dataset
  """
  def __init__(self, nlp, entID='SSC', causalKeywordID='causal',*args, **kwargs):
    """
      Construct

      Args:

        nlp: spacy.Language object, contains all components and data needed to process text
        args: list, positional arguments
        kwargs: dict, keyword arguments

      Returns:

        None
    """
    super().__init__(nlp, entID, causalKeywordID='causal', *args, **kwargs)
    self._allRelPairs = []
    self._relationNames = ['Subj_Entity', 'Relation', 'Obj_Entity']

  def reset(self):
    """
      Reset rule-based matcher
    """
    super().reset()
    self._allRelPairs = []
    self._entStatus = None

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

  def extractInformation(self):
    """
      extract information

      Args:

        None

      Returns:

        None
    """
    ## health status
    logger.info('Start to extract health status')
    self.extractHealthStatus(self._matchedSents)

    ## Access status and output to an ordered csv file
    entList = []
    aliasList = []
    entTextList = []
    statusList = []
    cjList = []
    negList = []
    negTextList = []
    for sent in self._matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._entID])
      for ent in ents:
        if ent._.status is not None:
          entList.append(ent.text)
          aliasList.append(ent._.alias)
          if ent._.alias is not None:
            entTextList.append(ent._.alias)
          else:
            entTextList.append(ent.text)
          statusList.append(ent._.status)
          cjList.append(ent._.conjecture)
          negList.append(ent._.neg)
          negTextList.append(ent._.neg_text)

    # Extracted information can be treated as attributes for given entity
    dfStatus = pd.DataFrame({'entity':entList, 'alias':aliasList, 'entity_text': entTextList, 'status':statusList, 'conjecture':cjList, 'negation':negList, 'negation_text': negTextList})
    if 'output_status_file' in nlpConfig['files']:
      dfStatus.to_csv(nlpConfig['files']['output_status_file'], columns=['entity', 'alias', 'entity_text', 'status', 'conjecture', 'negation', 'negation_text'])

    self._entStatus = dfStatus
    logger.info('End of health status extraction!')

    # Extract entity relations
    logger.info('Start to extract causal relation using OPM model information')
    self.extractRelDep(self._matchedSents)
    dfRels = pd.DataFrame(self._allRelPairs, columns=self._relationNames)
    if 'output_relation_file' in nlpConfig['files']:
      dfRels.to_csv(nlpConfig['files']['output_relation_file'], columns=self._relationNames)
    logger.info('End of causal relation extraction!')

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

    # procedure to process CWS data
    # collect status, negation, conjecture information
    for sent in matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._entID])
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

  def extractRelDep(self, matchedSents):
    """

      Args:

        matchedSents: list, the list of matched sentences

      Returns:

        (subject tuple, predicate, object tuple): generator, the extracted causal relation
    """
    subjList = ['nsubj', 'nsubjpass', 'nsubj:pass']
    # objList = ['pobj', 'dobj', 'iobj', 'obj', 'obl', 'oprd']
    for sent in matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._entID])
      if len(ents) <= 1:
        continue
      root = sent.root
      allRelPairs = []
      subjEnt = []
      subjConjEnt = []
      objEnt = []
      objConjEnt = []

      for ent in ents:
        entRoot = ent.root
        if ent._.alias is not None:
          text = ent._.alias
        else:
          text = ent.text
        # entity at the beginning of sentence
        if ent.start == sent.start:
          subjEnt.append(text)
        elif entRoot.dep_ in ['conj'] and entRoot.i < root.i:
          subjConjEnt.append(text)
        elif entRoot.dep_ in subjList:
          subjEnt.append(text)
        elif entRoot.dep_ in ['obj', 'dobj']:
          objEnt.append(text)
        elif entRoot.i > root.i and entRoot.dep_ in ['conj']:
          objConjEnt.append(text)
      # subj
      for subj in subjEnt:
        for subjConj in subjConjEnt:
          allRelPairs.append([subj, 'conj', subjConj])
        for obj in objEnt:
          allRelPairs.append([subj, root, obj])
        for objConj in objConjEnt:
          allRelPairs.append([subj, root, objConj])
      # subjconj
      for subjConj in subjConjEnt:
        for obj in objEnt:
          allRelPairs.append([subjConj, root, obj])
        for objConj in objConjEnt:
          allRelPairs.append([subjConj, root, objConj])
      # obj
      for obj in objEnt:
        for objConj in objConjEnt:
          allRelPairs.append([obj, 'conj', objConj])

      self._allRelPairs += allRelPairs
