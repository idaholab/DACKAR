# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on April, 2024

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
    self._allRelPairs = []
    self._relationNames = ['Subj_Entity', 'Relation', 'Obj_Entity']
    self._subjList = ['nsubj', 'nsubjpass', 'nsubj:pass']
    self._objList = ['pobj', 'dobj', 'iobj', 'obj', 'obl', 'oprd']

  def reset(self):
    """
      Reset rule-based matcher
    """
    super().reset()
    self._allRelPairs = []
    self._entStatus = None


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
    dfStatus.to_csv(nlpConfig['files']['output_status_file'], columns=['entity', 'alias', 'entity_text', 'status', 'conjecture', 'negation', 'negation_text'])

    self._entStatus = dfStatus
    logger.info('End of health status extraction!')

    # Extract entity relations
    logger.info('Start to extract causal relation using OPM model information')
    self.extractRelDep(self._matchedSents)
    dfRels = pd.DataFrame(self._allRelPairs, columns=self._relationNames)
    dfRels.to_csv(nlpConfig['files']['output_relation_file'], columns=self._relationNames)
    logger.info('End of causal relation extraction!')


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

    for ent in ents:
      status = None        # store health status for identified entities
      statusAmod = None    # store amod for health status
      statusAppend = None  # store some append info for health status (used for short phrase)
      statusAppendAmod = None # store amod info for health status append info
      statusPrepend = None  # store some prepend info
      statusPrependAmod = None # store amod info for prepend info
      statusText = None
      passive = False
      entRoot = ent.root

      if entRoot.dep_ in ['nsubj', 'nsubjpass']:
        status, neg, negText = self.getStatusForSubj(ent)
      elif entRoot.dep_ in ['pobj', 'dobj']:
        status, neg, negText = self.getStatusForObj(ent)


        if len(ents) == 1 or entRoot.dep_ in ['dobj']:
          status, neg, negText = self.getStatusForObj(ent, ent, sent, causalStatus, predSynonyms)
          if entRoot.dep_ in ['pobj']:
            # extract append info for health status
            prep = entRoot.head
            statusAppendAmod = self.getCompoundOnly(ent, ent)
            if len(statusAppendAmod) > 0:
              statusAppendAmod = [prep.text] + statusAppendAmod
              statusAppend = ent
        else:
          status = self.getStatusForPobj(ent, include=False)
        if status is None:
          head = entRoot.head
          if head.dep_ in ['xcomp', 'advcl', 'relcl']:
            for child in head.rights:
              if child.dep_ in ['ccomp']:
                status = child
                break
      elif entRoot.dep_ in ['compound']:
        head = entRoot.head
        if head.pos_ not in ['SPACE', 'PUNCT']:
          if len(ents) == 1:
            if head.dep_ in ['compound']:
              head = head.head
            headEnt = head.doc[head.i:head.i+1]
            if head.dep_ in ['nsubj', 'nsubjpass']:
              status, neg, negText = self.getStatusForSubj(headEnt, ent, sent, causalStatus, predSynonyms, include=True)
              if isinstance(status, Span):
                if entRoot.i >= status.start and entRoot.i < status.end:
                  status = headEnt
              if status is not None:
                statusPrepend = headEnt
                statusPrependAmod = self.getAmodOnly(headEnt)
            elif head.dep_ in ['dobj', 'pobj']:
              status, neg, negText = self.getStatusForObj(headEnt, ent, sent, causalStatus, predSynonyms, include=False)
              if status is not None:
                if isinstance(status, Span):
                  if head not in status:
                    # identify the dobj/pobj, and use it as append info
                    statusAppend = headEnt
                    statusAppendAmod = self.getAmodOnly(headEnt)
                elif isinstance(status, Token):
                  if head != status:
                    # identify the dobj/pobj, and use it as append info
                    statusAppend = headEnt
                    statusAppendAmod = self.getAmodOnly(headEnt)
            if status is None:
              status = headEnt
          else:
            status = entRoot.head
            statusAmod = self.getAmodOnly(status)
            if len(statusAmod) == 0:
              lefts = list(status.lefts)
              # remove entity itself
              for elem in lefts:
                if elem in ent:
                  lefts.remove(elem)
              if len(lefts) != 0:
                statusAmod = [e.text for e in lefts]
            if head.dep_ in ['dobj','pobj','nsubj', 'nsubjpass'] and [root.lemma_.lower()] in predSynonyms:
              ent._.set('hs_keyword', root.lemma_)
        else:
          status = self.getAmod(ent, ent.start, ent.end, include=False)

      elif entRoot.dep_ in ['conj']:
        # TODO: recursive function to retrieve non-conj
        status = self.getAmod(ent, ent.start, ent.end, include=False)
        if status is None:
          head = entRoot.head
          if head.dep_ in ['conj']:
            head = head.head
          headEnt = head.doc[head.i:head.i+1]
          if head.dep_ in ['nsubj', 'nsubjpass']:
            status, neg, negText = self.getStatusForSubj(headEnt, ent, sent, causalStatus, predSynonyms)
          elif head.dep_ in ['pobj', 'dobj']:
            status = self.getStatusForPobj(headEnt, include=False)
            if status is None:
              status, neg, negText = self.getstatusForObj(headEnt, ent, sent, causalStatus, predSynonyms)
      elif entRoot.dep_ in ['ROOT']:
        status = self.getAmod(ent, ent.start, ent.end, include=False)
        if status is None:
          rights =[tk for tk in list(entRoot.rights) if tk.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV'] and tk.i >= ent.end]
          if len(rights) > 0:
            status = rights[0]
      else:
        logger.warning(f'Entity "{ent}" dep_ is "{entRoot.dep_}" is not among valid list "[nsubj, nsubjpass, pobj, dobj, compound]"')
        if entRoot.head == root:
          headEnt = root.doc[root.i:root.i+1]
          if ent.start < root.i:
            if [root.lemma_.lower()] in predSynonyms:
              ent._.set('hs_keyword', root.lemma_)
            else:
              ent._.set('ent_status_verb', root.lemma_)
            neg, negText = self.isNegation(root)
            passive = self.isPassive(root)
            # # last is punct, the one before last is the root
            # if root.nbor().pos_ in ['PUNCT']:
            #   status = root
            status = self.findRightObj(root)
            if status and status.dep_ == 'pobj':
              status = self.getStatusForPobj(status, include=True)
            elif status and status.dep_ == 'dobj':
              subtree = list(status.subtree)
              nbor = self.getNbor(status)
              if nbor is not None and nbor.dep_ in ['prep']:
                status = status.doc[status.i:subtree[-1].i+1]
            # no object is found
            if not status:
              status = self.findRightKeyword(root)
            # last is punct, the one before last is the root
            nbor = self.getNbor(root)
            if not status and nbor is not None and nbor.pos_ in ['PUNCT']:
              status = root
            if status is None:
              status = self.getAmod(ent, ent.start, ent.end, include=False)
            if status is None:
              status = root
          else:
            if [root.lemma_.lower()] in predSynonyms:
              ent._.set('hs_keyword', root.lemma_)
            else:
              ent._.set('ent_status_verb', root.lemma_)
            passive = self.isPassive(root)
            neg, negText = self.isNegation(root)
            status = self.findLeftSubj(root, passive)
            if status is not None:
              status = self.getAmod(status, status.i, status.i+1, include=True)

      if status is None:
        continue

      _status = False
      if isinstance(status, Span):
        conjecture = self.isConjecture(status.root.head)
        if status.root.lemma_ in statusNoun + statusAdj:
          _status = True
      elif isinstance(status, Token):
        conjecture = self.isConjecture(status.head)
        if status.lemma_ in statusNoun + statusAdj:
          _status = True
      if not neg:
        if isinstance(status, Span):
          neg, negText = self.isNegation(status.root)
        else:
          neg, negText = self.isNegation(status)
      # conjecture = self.isConjecture(status.head)
      # neg, negText = self.isNegation(status)
      ent._.set('neg',neg)
      ent._.set('neg_text',negText)
      ent._.set('conjecture',conjecture)
      if _status:
        ent._.set('health_status',status)
        ent._.set('health_status_prepend', statusPrepend)
        ent._.set('health_status_prepend_amod',statusPrependAmod)
        ent._.set('health_status_amod',statusAmod)
        ent._.set('health_status_append',statusAppend)
        ent._.set('health_status_append_amod',statusAppendAmod)
      else:
        ent._.set('status',status)
        ent._.set('status_prepend', statusPrepend)
        ent._.set('status_prepend_amod',statusPrependAmod)
        ent._.set('status_amod',statusAmod)
        ent._.set('status_append',statusAppend)
        ent._.set('status_append_amod',statusAppendAmod)

      prependAmodText = ' '.join(statusPrependAmod) if statusPrependAmod is not None else ''
      prependText = statusPrepend.text if statusPrepend is not None else ''
      amodText = ' '.join(statusAmod) if statusAmod is not None else ''
      appendAmodText = ' '.join(statusAppendAmod) if statusAppendAmod is not None else ''
      if statusAppend is not None and statusAppend != ent:
        appText = statusAppend.root.head.text + ' ' + statusAppend.text if statusAppend.root.dep_ in ['pobj'] else statusAppend.text
      else:
        appText = ''

      statusText = ' '.join(list(filter(None, [prependAmodText, prependText, amodText, status.text, appendAmodText,appText]))).strip()
      if neg:
        statusText = ' '.join([negText,statusText])
      if isinstance(status, Span):
        if ent.start > status.start and ent.end < status.end:
          # remove entity info in statusText
          pn = re.compile(rf'{ent.text}\w*')
          statusText = re.sub(pn, '', statusText).strip()
          # statusText = statusText.replace(ent.text, '')

      logger.debug(f'{ent} health status: {statusText}')
      # ent._.set('health_status', statusText)
      # ent._.set('conjecture',conjecture)



  def handleInvalidSent(self, sent, ents):
    """
      Handle sentence that do not have (subj, predicate, obj)
    """
    root = sent.root
    neg, negText = self.isNegation(root)
    conjecture = self.isConjecture(root)

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



  def validSent(self, sent):
    """
      Check if the sentence has valid structure, either contains subject or object

      Args:

        sent: Span, sentence from user provided text

      Returns:

        valid: bool, False if the sentence has no subject, predicate and object.
    """
    foundSubj = False
    foundObj = False
    foundVerb = False
    valid = False
    root = sent.root
    if root.pos_ in  ['AUX', 'VERB']:
      foundVerb = True
    for tk in sent:
      if tk.dep_.startswith('nsubj'):
        foundSubj = True
      elif tk.dep_.endswith('obj'):
        foundObj = True
    # Is it a strong assumption here?
    if foundSubj and foundObj and foundVerb:
      valid = True
    return valid
