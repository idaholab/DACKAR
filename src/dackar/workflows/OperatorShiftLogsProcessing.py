# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on April, 2024

@author: wangc, mandd
"""
import logging
import pandas as pd
import re
from operator import itemgetter
from spacy.tokens import Token
from spacy.tokens import Span

from ..text_processing.Preprocessing import Preprocessing
from ..utils.utils import getOnlyWords, getShortAcronym
from ..config import nlpConfig
from .WorkflowBase import WorkflowBase
from ..pipelines.CustomPipelineComponents import mergeEntitiesWithSameID

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
if not Span.has_extension('action'):
  Span.set_extension("action", default=None)
if not Span.has_extension('edep'):
  Span.set_extension("edep", default=None)

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
if not Token.has_extension('action'):
  Token.set_extension("action", default=None)
if not Token.has_extension('edep'):
  Token.set_extension("edep", default=None)

class OperatorShiftLogs(WorkflowBase):
  """
    Class to process OPG Operator Shift Logs dataset
  """
  def __init__(self, nlp, entID='SSC', causalKeywordID='causal', *args, **kwargs):
    """
      Construct

      Args:

        nlp: spacy.Language object, contains all components and data needed to process text
        args: list, positional arguments
        kwargs: dict, keyword arguments

      Returns:

        None
    """
    super().__init__(nlp, entID, causalKeywordID, *args, **kwargs)
    if not nlp.has_pipe('mergeEntitiesWithSameID'):
      self.nlp.add_pipe('mergeEntitiesWithSameID', after='aliasResolver')

    self._allRelPairs = []
    self._relationNames = ['Subj_Entity', 'Relation', 'Obj_Entity']
    self._subjList = ['nsubj', 'nsubjpass', 'nsubj:pass']
    self._objList = ['pobj', 'dobj', 'iobj', 'obj', 'obl', 'oprd']
    self._entInfoNames = ['Entity', 'Label', 'Status', 'Amod', 'Action', 'Dep', 'Alias', 'Negation', 'Conjecture', 'Sentence']

  def reset(self):
    """
      Reset rule-based matcher
    """
    super().reset()
    self._allRelPairs = []
    self._entInfoNames = None

  def textProcess(self):
    """
      Function to clean text

      Args:
        None

      Returns:
        procObj, DACKAR.Preprocessing object
    """
    procObj = super().textProcess()
    return procObj

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
    self.extractStatus(self._matchedSents)

    if self._screen:
      # print collected info
      for sent in self._matchedSents:
        ents = self.getCustomEnts(sent.ents, self._entityLabels[self._entID])
        if ents is not None:
          print('Sentence:', sent)
          print('... Conjecture:', sent._.conjecture)
          print('... Negation:', sent._.neg, sent._.neg_text)
          print('... Action:', sent._.action)
          for ent in ents:
            print('... Entity:', ent.text)
            print('...... Status:', ent._.status)
            print('...... Amod:', ent._.status_amod)
            print('...... Action:', ent._.action)
            print('...... Dep:', ent._.edep)
            print('...... Alias:', ent._.alias)
            if ent._.neg:
              print('...... Negation', ent._.neg_text)

    entInfo = []
    for sent in self._matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._entID])
      if ents is not None:
        for ent in ents:
            entInfo.append([ent.text, ent.label_, ent._.status, ent._.status_amod, ent._.action, ent._.edep, ent._.alias, ent._.neg_text, sent._.conjecture, sent.text.strip('\n')])
    if len(entInfo) > 0:
      self._entStatus = pd.DataFrame(entInfo, columns=self._entInfoNames)

    # Extract entity relations
    logger.info('Start to extract entity relations')
    self.extractRelDep(self._matchedSents)
    # dfRels = pd.DataFrame(self._allRelPairs, columns=self._relationNames)
    # dfRels.to_csv(nlpConfig['files']['output_relation_file'], columns=self._relationNames)

    if len(self._allRelPairs) > 0:
      self._causalRelationGeneral = pd.DataFrame(self._allRelPairs, columns=self._relationNames)
      if self._screen:
        print(self._causalRelationGeneral)
    logger.info('End of entity relation extraction!')

    if self._causalKeywordID in self._entityLabels:
      # # Extract entity causal relations
      logger.info('Start to extract entity causal relation')
      self.extractCausalRelDep(self._matchedSents)
      logger.info('End of causal relation extraction!')
      if len(self._rawCausalList) > 0:
        for l in self._rawCausalList:
          print(l, l[0].sent)
        # print(self._rawCausalList)



  def extractStatus(self, matchedSents, predSynonyms=[], exclPrepos=[]):
    """
      Extract health status and relation

      Args:

        matchedSents: list, the matched sentences
        predSynonyms: list, predicate synonyms
        exclPrepos: list, exclude the prepositions
    """
    # procedure to process OPG CWS data
    # collect status, negation, conjecture information
    for sent in matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._entID])
      if ents is None:
        continue

      valid = self.validSent(sent)
      if valid:
        self.handleValidSent(sent, ents)
      else:
        self.handleInvalidSent(sent, ents)


  def handleValidSent(self, sent, ents):
    """
      Handle sentence that do not have (subj, predicate, obj)
    """
    root = sent.root
    neg, negText = self.isNegation(root)
    conjecture = self.isConjecture(root)
    sent._.set('neg',neg)
    sent._.set('neg_text',negText)
    sent._.set('conjecture',conjecture)
    root = sent.root
    action = root if root.pos_ in ['VERB', 'AUX'] else None
    sent._.set('action', action)
    for ent in ents:
      neg = None
      negText = None

      status = None        # store health status for identified entities
      entRoot = ent.root

      if entRoot.dep_ in ['nsubj', 'nsubjpass']:
        status, neg, negText = self.getStatusForSubj(ent)
      elif entRoot.dep_ in ['dobj', 'pobj', 'iobj', 'obj', 'obl', 'oprd']:
        status, neg, negText = self.getStatusForObj(ent)
        head = entRoot.head
        if status is None and head.dep_ in ['xcomp', 'advcl', 'relcl']:
          ccomps = [child for child in head.rights if child.dep_ in ['ccomp']]
          status = ccomps[0] if len(ccomps) > 0 else None
      elif entRoot.dep_ in ['compound']:
        status = self.getAmod(ent, ent.start, ent.end, include=False)
        head = entRoot.head
        if status is None and head.dep_ not in ['compound']:
          status = head
      elif entRoot.dep_ in ['conj']:
        # TODO: recursive function to retrieve non-conj
        amod = self.getAmod(ent, ent.start, ent.end, include=False)
        head = entRoot.head
        headStatus = None
        if head.dep_ in ['conj']:
          head = head.head
        headEnt = head.doc[head.i:head.i+1]
        if head.dep_ in ['nsubj', 'nsubjpass']:
          headStatus, neg, negText = self.getStatusForSubj(headEnt)
        elif head.dep_ in ['pobj', 'dobj']:
          headStatus, neg, negText = self.getStatusForObj(headEnt)
          head = entRoot.head
          if headStatus is None and head.dep_ in ['xcomp', 'advcl', 'relcl']:
            ccomps = [child for child in head.rights if child.dep_ in ['ccomp']]
            headStatus = ccomps[0] if len(ccomps) > 0 else None
        if headStatus is None:
          status = amod
        elif isinstance(headStatus, list):
          status = headStatus if amod is None else [amod, headStatus[-1]]
        else:
          status = headStatus if amod is None else [amod, headStatus]

      elif entRoot.dep_ in ['ROOT']:
        status = self.getAmod(ent, ent.start, ent.end, include=False)
        if status is None:
          rights =[tk for tk in list(entRoot.rights) if tk.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV'] and tk.i >= ent.end]
          if len(rights) > 0:
            status = rights[0]
      else:
        status = self.getAmod(ent, ent.start, ent.end, include=False)

      if isinstance(status, list):
        ent._.set('status', status[1])
        ent._.set('status_amod', status[0])
      else:
        ent._.set('status', status)

      ent._.set('neg', neg)
      ent._.set('neg_text', negText)
      ent._.set('edep', ent.root.dep_)

      if ent.root.head.pos_ in ['VERB', 'AUX']:
        ent._.set('action', ent.root.head)
      elif ent.root.head.dep_ in ['prep'] and ent.root.head.head.pos_ in ['VERB', 'AUX']:
        ent._.set('action', ent.root.head.head)

  def handleInvalidSent(self, sent, ents):
    """
      Handle sentence that do not have (subj, predicate, obj)
    """
    root = sent.root
    neg, negText = self.isNegation(root)
    conjecture = self.isConjecture(root)
    sent._.set('neg',neg)
    sent._.set('neg_text',negText)
    sent._.set('conjecture',conjecture)
    root = sent.root
    action = root if root.pos_ in ['VERB', 'AUX'] else None
    sent._.set('action', action)

    for ent in ents:
      ent._.set('neg', neg)
      ent._.set('neg_text', negText)
      ent._.set('conjecture', conjecture)
      entRoot = ent.root
      ent._.set('edep', entRoot.dep_)
      if entRoot.head.pos_ in ['VERB', 'AUX']:
        ent._.set('action', entRoot.head)
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
          if entRoot.dep_ in self._subjList:
            status = sent[ent.end:]
            # some clean up for the text
            text = self._textProcess(status.text)
            ent._.set('status', text)
          # Include the whole info with alias substitution
          elif entRoot.dep_ in self._objList:
            text = sent.text
            # substitute entity ID with its alias
            text = re.sub(r"\b%s\b" % str(ent.text) , ent._.alias, text)
            text = getOnlyWords(text)
            text = self._textProcess(text)
            ent._.set('status', text)
      # other type of entities
      else:
        entRoot = ent.root
        if entRoot.dep_ in self._subjList:
          # depend on the application, can use self.getStatusForSubj to get the status
          status = sent[ent.end:]
          # some clean up for the text
          text = self._textProcess(status.text)
          ent._.set('status', text)
        # Include the whole info with alias substitution
        elif entRoot.dep_ in self._objList:
          # depend on the application, can use self.getstatusForObj to get the status
          text = sent.text
          text = getOnlyWords(text)
          text = self._textProcess(text)
          ent._.set('status', text)
        else:
          # If there is single entity, then report it.
          if len(ents) == 1:
            text = sent.text
            text = re.sub(r"\b%s\b" % str(ent.text) , '', text)
            text = getOnlyWords(text)
            text = self._textProcess(text)
            ent._.set('status', text)
          # if the entity not among subj and obj and there are more than one entity, it may not need to report it
          else:
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
      if ents is None or len(ents) <= 1:
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

  def extractCausalRelDep(self, matchedSents):
    """

      Args:

        matchedSents: list, the list of matched sentences

      Returns:

        (subject tuple, predicate, object tuple): generator, the extracted causal relation
    """
    allCausalPairs = []
    for sent in matchedSents:
      causalPairs = []
      root = sent.root
      passive = self.isPassive(root)
      causalEnts = self.getCustomEnts(sent.ents, self._entityLabels[self._causalKeywordID])
      if causalEnts is None:
        continue
      causalPairs.extend([(ent, ent.start) for ent in causalEnts])

      sscEnts = self.getCustomEnts(sent.ents, self._entityLabels[self._entID])
      if sscEnts is not None:
        causalPairs.extend([(ent, ent.start) for ent in sscEnts])

      subj = self.findSubj(root, passive)
      obj = self.findObj(root)

      if sscEnts is None:
        if subj is not None:
          causalPairs.append((subj, subj.i))
        if obj is not None:
          causalPairs.append((obj, obj.i))
      else:
        if subj is not None:
          if not self.isSubElements(subj, sscEnts+causalEnts):
            causalPairs.append((subj, subj.i))
        if obj is not None:
          if not self.isSubElements(obj, sscEnts+causalEnts):
            causalPairs.append((obj, obj.i))

      # mergePhrase pipelie can merge "( Issue" into single entity.
      causalPairs = sorted(causalPairs, key=itemgetter(1))

      causalPairs = [elem[0] for elem in causalPairs]
      allCausalPairs.append(causalPairs)
    self._rawCausalList.extend(allCausalPairs)


  def isSubElements(self, elem1, elemList):
    """
    """
    isSub = False
    for elem in elemList:
      isSub = self.isSubElement(elem1, elem)
      if isSub:
        return isSub
    return isSub


  def isSubElement(self, elem1, elem2):
    """
      True if elem1 is a subelement of elem2
    """
    if isinstance(elem1, Token):
      s1, e1 = elem1.i, elem1.i
    elif isinstance(elem1, Span):
      s1, e1 = elem1.start, elem1.end
    else:
      raise IOError("Wrong data type is provided!")
    if isinstance(elem2, Token):
      s2, e2 = elem2.i, elem2.i
    elif isinstance(elem2, Span):
      s2, e2 = elem2.start, elem2.end
    else:
      raise IOError("Wrong data type is provided!")
    if s1 >= s2 and e1 <=e2:
      return True
    else:
      return False
