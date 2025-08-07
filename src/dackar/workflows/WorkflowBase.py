# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on April, 2024

@author: wangc, mandd
"""
import abc
import logging
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
from ..config import nlpConfig
from ..text_processing.Preprocessing import Preprocessing

logger = logging.getLogger('DACKAR.WorkflowBase')

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


if not Span.has_extension('health_status'):
  Span.set_extension("health_status", default=None)
if not Token.has_extension('health_status'):
  Token.set_extension("health_status", default=None)
if not Span.has_extension('hs_keyword'):
  Span.set_extension('hs_keyword', default=None)
if not Span.has_extension('ent_status_verb'):
  Span.set_extension('ent_status_verb', default=None)
if not Span.has_extension('conjecture'):
  Span.set_extension('conjecture', default=False)

if not Span.has_extension('status'):
  Span.set_extension("status", default=None)
if not Token.has_extension('status'):
  Token.set_extension("status", default=None)
if not Span.has_extension('health_status_prepend_amod'):
  Span.set_extension("health_status_prepend_amod", default=None)
if not Span.has_extension('health_status_prepend'):
  Span.set_extension("health_status_prepend", default=None)
if not Span.has_extension('health_status_amod'):
  Span.set_extension("health_status_amod", default=None)
if not Span.has_extension('health_status_append_amod'):
  Span.set_extension("health_status_append_amod", default=None)
if not Span.has_extension('health_status_append'):
  Span.set_extension("health_status_append", default=None)
if not Span.has_extension('neg'):
  Span.set_extension("neg", default=None)
if not Span.has_extension('neg_text'):
  Span.set_extension("neg_text", default=None)

if not Span.has_extension('status_prepend_amod'):
  Span.set_extension("status_prepend_amod", default=None)
if not Span.has_extension('status_prepend'):
  Span.set_extension("status_prepend", default=None)
if not Span.has_extension('status_amod'):
  Span.set_extension("status_amod", default=None)
if not Span.has_extension('status_append_amod'):
  Span.set_extension("status_append_amod", default=None)
if not Span.has_extension('status_append'):
  Span.set_extension("status_append", default=None)
if not Span.has_extension('alias'):
  Span.set_extension("alias", default=None)

if not Token.has_extension('health_status_prepend_amod'):
  Token.set_extension("health_status_prepend_amod", default=None)
if not Token.has_extension('health_status_prepend'):
  Token.set_extension("health_status_prepend", default=None)
if not Token.has_extension('health_status_amod'):
  Token.set_extension("health_status_amod", default=None)
if not Token.has_extension('health_status_append_amod'):
  Token.set_extension("health_status_append_amod", default=None)
if not Token.has_extension('health_status_append'):
  Token.set_extension("health_status_append", default=None)
if not Token.has_extension('neg'):
  Token.set_extension("neg", default=None)
if not Token.has_extension('neg_text'):
  Token.set_extension("neg_text", default=None)

if not Token.has_extension('status_prepend_amod'):
  Token.set_extension("status_prepend_amod", default=None)
if not Token.has_extension('status_prepend'):
  Token.set_extension("status_prepend", default=None)
if not Token.has_extension('status_amod'):
  Token.set_extension("status_amod", default=None)
if not Token.has_extension('status_append_amod'):
  Token.set_extension("status_append_amod", default=None)
if not Token.has_extension('status_append'):
  Token.set_extension("status_append", default=None)
if not Token.has_extension('alias'):
  Token.set_extension("alias", default=None)


class WorkflowBase(object):
  """
    Base Class for Workflow Analysis
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
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    logger.info(f'Create instance of {self.name}')
    # orders of NLP pipeline: 'ner' --> 'normEntities' --> 'merge_entities' --> 'initCoref'
    # --> 'aliasResolver' --> 'coreferee' --> 'anaphorCoref'
    # pipeline 'merge_noun_chunks' can be used to merge phrases (see also displacy option)
    self.nlp = nlp
    self._causalFile = nlpConfig['files']['cause_effect_keywords_file']
    # SCONJ->Because, CCONJ->so, ADP->as, ADV->therefore
    self._causalPOS = {'VERB':['VERB'], 'NOUN':['NOUN'], 'TRANSITION':['SCONJ', 'CCONJ', 'ADP', 'ADV']}
    # current columns include: "VERB", "NOUN", "TRANSITION", "causal-relator", "effect-relator", "causal-noun", "effect-noun"
    # For relator, such as becaue, therefore, as, etc.
    #   if the column starts with causal, which means causal entity --> keyword --> effect entity
    #   if the column starts with effect, which means effect entity <-- keyword <-- causal entity
    # For NOUN
    #   if the column starts with causal, which means causal entity --> keyword --> effect entity
    #   if the column starts with effect, the relation is depend on the keyword.dep_
    #   First check the right child of the keyword is ADP with dep_ "prep",
    #   Then, check the dep_ of keyword, if it is "dobj", then causal entity --> keyword --> effect entity
    #   elif it is "nsubj" or "nsubjpass" or "attr", then effect entity <-- keyword <-- causal entity
    self._causalKeywords = self.getKeywords(self._causalFile)
    self._statusFile = nlpConfig['files']['status_keywords_file']['all']
    self._statusKeywords = self.getKeywords(self._statusFile)
    self._updateStatusKeywords = False
    self._updateCausalKeywords = False
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
      entID = nlpConfig['params'].get('ent_id', entID)
    self._entID = entID
    self._causalKeywordID = causalKeywordID
    self._causalNames = ['cause', 'cause health status', 'causal keyword', 'effect', 'effect health status', 'sentence', 'conjecture']
    self._extractedCausals = [] # list of tuples, each tuple represents one causal-effect, i.e., (cause, cause health status, cause keyword, effect, effect health status, sentence)
    self._causalSentsNoEnts = []
    self._rawCausalList = []
    self._causalSentsOneEnt = []
    self._entHS = None
    self._entStatus = None
    self._screen = False
    self.dataframeRelations = None
    self.dataframeEntities = None

    self._textProcess = self.textProcess()

  def reset(self):
    """
      Reset rule-based matcher
    """
    self._matchedSents = []
    self._matchedSentsForVis = []
    self._extractedCausals = []
    self._causalSentsNoEnts = []
    self._rawCausalList = []
    self._causalSentsOneEnt = []
    self._entHS = None
    self._doc = None
    self.dataframeRelations = None
    self.dataframeEntities = None

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
    elif ktype.lower() == 'causal':
      for key, val in keywords.items():
        if type(val) != list:
          val = [val]
        val = self.extractLemma(val)
        if key in self._causalKeywords:
          self._causalKeywords[key].append(val)
        else:
          logger.warning('keyword "{}" cannot be accepted, valid keys for the keywords are "{}"'.format(key, ','.join(list(self._causalKeywords.keys()))))

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

  def __call__(self, text, extract=True, screen=False):
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
    self._screen = screen
    ## use entity ruler to identify entity
    # if self._entityRuler:
    #   logger.debug('Entity Ruler Matches:')
    #   print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents if ent.label_ in self._entityLabels[self._entID]])

    # First identify coreference through coreferee, then filter it through doc.ents
    if self._coref:
      corefRep = doc._.coref_chains.pretty_representation
      if len(corefRep) != 0:
        logger.debug('Print Coreference Info:')
        print(corefRep)

    matchedSents, matchedSentsForVis = self.collectSents(self._doc)
    self._matchedSents += matchedSents
    self._matchedSentsForVis += matchedSentsForVis
    if extract:
      self.extractInformation()

  @abc.abstractmethod
  def extractInformation(self):
    """
      extract information

      Args:

        None

      Returns:

        None
    """

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
    # check the keywords
    # FIXME: should we use token.subtree or token.children here
    for child in token.subtree:
      if [child.lemma_.lower()] in self._conjectureKeywords['conjecture-keywords']:
        return True
    # For short sentences, conjecture can not determined by VERB inflectional morphology
    # if token.pos_ == 'VERB' and token.tag_ == 'VB': # If it is a verb, and there is no inflectional morphology for the verb
    #   return True
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

  def findVerb(self, doc):
    """
      Find the first verb in the doc

      Args:

        doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

      Returns:

        token: spacy.tokens.Token, the token that has VERB pos
    """
    for token in doc:
      if token.pos_ == 'VERB':
        return token
        break
    return None

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

  def getPhrase(self, ent, start, end, include=False):
    """
      Get the phrase for ent with all left children

      Args:

        ent: Span, the ent to amend with all left children
        start: int, the start index of ent
        end: int, the end index of ent
        include: bool, include ent in the returned expression if True

      Returns:

        status: Span or Token, the identified status
    """
    leftInd = list(ent.lefts)[0].i
    if not include:
      status = ent.doc[leftInd:start]
    else:
      status = ent.doc[leftInd:end]
    return status

  def getAmod(self, ent, start, end, include = False):
    """
      Get amod tokens for ent

      Args:

        ent: Span, the ent to amend with all left children
        start: int, the start index of ent
        end: int, the end index of ent
        include: bool, include ent in the returned expression if True

      Returns:

        status: Span or Token, the identified status
    """
    status = None
    deps = [tk.dep_ in ['amod'] for tk in ent.lefts]
    if any(deps):
      status = self.getPhrase(ent, start, end, include)
    else:
      deps = [tk.dep_ in ['compound'] for tk in ent.lefts]
      if any(deps):
        status = self.getPhrase(ent, start, end, include)
        status = self.getAmod(status, status.start, status.end, include=True)
    if status is None and include:
      status = ent
    return status

  def getAmodOnly(self, ent):
    """
      Get amod tokens texts for ent

      Args:

        ent: Span, the ent to amend with all left children

      Returns:

        amod: list, the list of amods for ent
    """
    amod = [tk.text for tk in ent.lefts if tk.dep_ in ['amod']]
    return amod

  def getCompoundOnly(self, headEnt, ent):
    """
      Get the compounds for headEnt except ent

      Args:

        headEnt: Span, the head entity to ent

      Returns:

        compDes: list, the list of compounds for head ent
    """
    compDes = []
    comp = [tk for tk in headEnt.lefts if tk.dep_ in ['compound'] and tk not in ent]
    if len(comp) > 0:
      for elem in comp:
        des = [tk.text for tk in elem.lefts if tk.dep_ in ['amod', 'compound'] and tk not in ent]
        compDes.extend(des)
        compDes.append(elem.text)
    return compDes

  def getNbor(self, token):
    """
      Method to get the nbor from token, return None if nbor is not exist

      Args:

        token: Token, the provided Token to request nbor

      Returns:

        nbor: Token, the requested nbor
    """
    nbor = None
    if token is None:
      return nbor
    try:
      nbor = token.nbor()
    except IndexError:
      pass
    return nbor

  def validSent(self, sent):
    """
      Check if the sentence has valid structure, either contains subject or object

      Args:

        sent: Span, sentence from user provided text

      Returns:

        valid: bool, False if the sentence has no subject and object.
    """
    foundSubj = False
    foundObj = False
    valid = False
    for tk in sent:
      if tk.dep_.startswith('nsubj'):
        foundSubj = True
      elif tk.dep_.endswith('obj'):
        foundObj = True
    if foundSubj or foundObj:
      valid = True
    return valid

  def findLeftSubj(self, pred, passive):
    """
      Find closest subject in predicates left subtree or
      predicates parent's left subtree (recursive).
      Has a filter on organizations.

      Args:

        pred: spacy.tokens.Token, the predicate token
        passive: bool, True if passive

      Returns:

        subj: spacy.tokens.Token, the token that represent subject
    """
    for left in pred.lefts:
      if passive: # if pred is passive, search for passive subject
        subj = self.findHealthStatus(left, ['nsubjpass', 'nsubj:pass'])
      else:
        subj = self.findHealthStatus(left, ['nsubj'])
      if subj is not None: # found it!
        return subj
    if pred.head != pred and not self.isPassive(pred):
      return self.findLeftSubj(pred.head, passive) # climb up left subtree
    else:
      return None

  def findRightObj(self, pred, deps=['dobj', 'pobj', 'iobj', 'obj', 'obl', 'oprd'], exclPrepos=[]):
    """
      Find closest object in predicates right subtree.
      Skip prepositional objects if the preposition is in exclude list.
      Has a filter on organizations.

      Args:
        pred: spacy.tokens.Token, the predicate token
        exclPrepos: list, list of the excluded prepositions
    """
    for right in pred.rights:
      obj = self.findHealthStatus(right, deps)
      if obj is not None:
        if obj.dep_ == 'pobj' and obj.head.lemma_.lower() in exclPrepos: # check preposition
          continue
        return obj
    return None

  def findRightKeyword(self, pred, exclPrepos=[]):
    """
      Find
      Skip prepositional objects if the preposition is in exclude list.
      Has a filter on organizations.

      Args:

        pred: spacy.tokens.Token, the predicate token
        exclPrepos: list, list of the excluded prepositions
    """
    for right in pred.rights:
      pos = right.pos_
      if pos in ['VERB', 'NOUN', 'ADJ']:
        # skip check to remove the limitation of status only in status keywords list
        # if [right.lemma_.lower()] in self._statusKeywords[pos]:
        return right
    return None

  def findHealthStatus(self, root, deps):
    """
      Return first child of root (included) that matches
      dependency list by breadth first search.
      Search stops after first dependency match if firstDepOnly
      (used for subject search - do not "jump" over subjects)

      Args:

        root: spacy.tokens.Token, the root token
        deps: list, the dependency list

      Returns:

        child: token, the token represents the health status
    """
    toVisit = deque([root]) # queue for bfs
    while len(toVisit) > 0:
      child = toVisit.popleft()
      # print("child", child, child.dep_)
      if child.dep_ in deps:
        # to handle preposition
        try:
          nbor = child.nbor()
        except IndexError:
          pass # ignore for now
        # TODO, what else need to be added
        # can not use the first check only, since is nbor is 'during', it will also satisfy the check condition
        # if (nbor.dep_ in ['prep'] and nbor.lemma_.lower() in ['of', 'in']) or nbor.pos_ in ['VERB']:
        #   return self.findRightObj(nbor, deps=['pobj'])
        return child
      elif child.dep_ == 'compound' and \
         child.head.dep_ in deps: # check if contained in compound
        return child
      toVisit.extend(list(child.children))
    return None

  def isValidCausalEnts(self, ent):
    """
    Check the entity if it belongs to the valid causal entities

      Args:

        ent: list, list of entities

      Returns:

        valid: bool, valid cansual ent if True
    """
    valid = False
    validDep = ['nsubj', 'nsubjpass', 'nsubj:pass', 'pobj', 'dobj', 'iobj', 'obj', 'obl', 'oprd']
    for e in ent:
      root = e.root
      if root.dep_ in validDep or root.head.dep_ in validDep:
        valid = True
        break
    return valid

  def getIndex(self, ent, entList):
    """
      Get index for ent in entList

      Args:

        ent: Span, ent that is used to get index
        entList: list, list of entities

      Returns:

        idx: int, the index for ent
    """
    idx = -1
    for i, e in enumerate(entList):
      if isinstance(e, list):
        if ent in e:
          idx = i
      else:
        if e == ent:
          idx = i
    return idx


  def getConjuncts(self, entList):
    """
      Get a list of conjuncts from entity list

      Args:
        entList: list, list of entities

      Returns:
        conjunctList: list, list of conjuncts
    """
    ent = entList[0]
    conjunctList = []
    conjuncts = [ent]
    collected = False
    for i, elem in enumerate(entList[1:]):
      # print('elem', elem, elem.conjuncts)
      # print('ent', ent, ent.conjuncts)
      if elem.root not in ent.conjuncts:
        conjunctList.append(conjuncts)
        conjunctList.extend(self.getConjuncts(entList[i+1:]))
        collected = True
        break
      conjuncts.append(elem)
    if not collected:
      conjunctList.append(conjuncts)
    return conjunctList

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
      if span.ent_id_ != self._entID:
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

#############################################################################
# some useful methods, but currently they are not used

  def extract(self, sents, predSynonyms=[], exclPrepos=[]):
    """
      General extraction method

      Args:
        sents: list, the list of sentences
        predSynonyms: list, the list of predicate synonyms
        exclPrepos: list, the list of exlcuded prepositions

      Returns:
        (subject tuple, predicate, object tuple): generator, the extracted causal relation
    """
    for sent in sents:
      root = sent.root
      if root.pos_ == 'VERB' and [root.lemma_.lower()] in predSynonyms:
        passive = self.isPassive(root)
        subj = self.findSubj(root, passive)
        if subj is not None:
          obj = self.findObj(root, deps=['dobj', 'pobj', 'iobj', 'obj', 'obl'], exclPrepos=[])
          if obj is not None:
            if passive: # switch roles
              obj, subj = subj, obj
            yield ((subj), root, (obj))
      else:
        for token in sent:
          if [token.lemma_.lower()] in predSynonyms:
            root = token
            passive = self.isPassive(root)
            subj = self.findSubj(root, passive)
            if subj is not None:
              obj = self.findObj(root, deps=['dobj', 'pobj', 'iobj', 'obj', 'obl'], exclPrepos=[])
              if obj is not None:
                if passive: # switch roles
                  obj, subj = subj, obj
                yield ((subj), root, (obj))

  def bfs(self, root, deps):
    """
      Return first child of root (included) that matches
      entType and dependency list by breadth first search.
      Search stops after first dependency match if firstDepOnly
      (used for subject search - do not "jump" over subjects)

      Args:
        root: spacy.tokens.Token, the root token
        deps: list, list of dependency

      Returns:
        child: spacy.tokens.Token, the matched token
    """
    toVisit = deque([root]) # queue for bfs
    while len(toVisit) > 0:
      child = toVisit.popleft()
      if child.dep_ in deps:
        # to handle preposition
        nbor = self.getNbor(child)
        if nbor is not None and nbor.dep_ in ['prep'] and nbor.lemma_.lower() in ['of']:
          obj = self.findObj(nbor, deps=['pobj'])
          return obj
        else:
          return child
      elif child.dep_ == 'compound' and \
         child.head.dep_ in deps: # check if contained in compound
        return child
      toVisit.extend(list(child.children))
    return None

  def findSubj(self, pred, passive):
    """
      Find closest subject in predicates left subtree or
      predicates parent's left subtree (recursive).
      Has a filter on organizations.

      Args:
        pred: spacy.tokens.Token, the predicate token
        passive: bool, True if the predicate token is passive

      Returns:
        subj: spacy.tokens.Token, the token that represents subject
    """
    for left in pred.lefts:
      if passive: # if pred is passive, search for passive subject
        subj = self.bfs(left, ['nsubjpass', 'nsubj:pass'])
      else:
        subj = self.bfs(left, ['nsubj'])
      if subj is not None: # found it!
        return subj
    if pred.head != pred and not self.isPassive(pred):
      return self.findSubj(pred.head, passive) # climb up left subtree
    else:
      return None

  def findObj(self, pred, deps=['dobj', 'pobj', 'iobj', 'obj', 'obl'], exclPrepos=[]):
    """
      Find closest object in predicates right subtree.
      Skip prepositional objects if the preposition is in exclude list.
      Has a filter on organizations.

      Args:
        pred: spacy.tokens.Token, the predicate token
        exclPrepos: list, the list of prepositions that will be excluded

      Returns:
        obj: spacy.tokens.Token,, the token that represents the object
    """
    for right in pred.rights:
      obj = self.bfs(right, deps)
      if obj is not None:
        if obj.dep_ == 'pobj' and obj.head.lemma_.lower() in exclPrepos: # check preposition
          continue
        return obj
    return None

  def isValidKeyword(self, var, keywords):
    """

      Args:
        var: token
        keywords: list/dict

      Returns: True if the var is a valid among the keywords
    """
    if isinstance(keywords, dict):
      for _, vals in keywords.items():
        if var.lemma_.lower() in vals:
          return True
    elif isinstance(keywords, list):
      if var.lemma_.lower() in keywords:
        return True
    return False
#######################################################################################

  def getStatusForSubj(self, ent, include=False):
    """
      Get the status for nsubj/nsubjpass ent

      Args:

        ent: Span, the nsubj/nsubjpass ent that will be used to search status
        include: bool, include ent in the returned expression if True

      Returns:

        status: Span or Token, the identified status
    """
    status = None
    neg = False
    negText = ''
    entRoot = ent.root
    root = entRoot.head

    if entRoot.dep_ not in ['nsubj', 'nsubjpass', 'nsubj:pass']:
      raise IOError("Method 'self.getStatusForSubj' can only be used for 'nsubj' or 'nsubjpass'")
    if root.pos_ != 'VERB':
      neg, negText = self.isNegation(root)
      if root.pos_ in ['NOUN', 'ADJ']:
        # TODO: search entRoot.lefts for 'amod' and attach to status
        status = root
      elif root.pos_ in ['AUX']:
        rights = [r for r in root.rights]
        if len(rights) == 0:
          status = None
        elif len(rights) == 1:
          status = root.doc[root.i+1:rights[0].i+1]
        else:
          status = []
          for r in rights:
            s = self.getAmod(r, r.i, r.i+1, include=include)
            status.append(s)
      else:
        logger.warning(f'No status identified for "{ent}" in "{ent.sent}"')
    else:
      rights = root.rights
      valid = [tk.dep_ in ['advcl', 'relcl'] for tk in rights if tk.pos_ not in ['PUNCT', 'SPACE']]
      nbor = self.getNbor(root)
      if nbor is not None and (nbor.dep_ in ['cc'] or nbor.pos_ in ['PUNCT']):
        status = root
      elif len(valid)>0 and all(valid):
        status = root
      else:
        objStatus = None
        amod = self.getAmod(ent, ent.start, ent.end, include=include)
        obj = self.findRightObj(root)
        if obj:
          if obj.dep_ == 'pobj':
            objStatus = self.getStatusForPobj(obj, include=True)
          elif obj.dep_ == 'dobj':
            subtree = list(obj.subtree)
            try:
              if obj.nbor().dep_ in ['prep']:
                objStatus = obj.doc[obj.i:subtree[-1].i+1]
              else:
                objStatus = obj
            except IndexError:
              objStatus = obj
        # no object is found
        else:
          objStatus = self.findRightKeyword(root)
        # last is punct, the one before last is the root
        # if not status and root.nbor().pos_ in ['PUNCT']:
        #   status = root
        if objStatus is None and amod is None:
          extra = [tk for tk in root.rights if tk.pos_ in ['ADP', 'ADJ']]
          # Only select the first ADP and combine with root
          if len(extra) > 0:
            objStatus = root.doc[root.i:extra[0].i+1]
          else:
            objStatus = root
        if amod is not None:
          status = [amod, objStatus]
        else:
          status = objStatus
    return status, neg, negText

  def getStatusForObj(self, ent, include=False):
    """
      Get the status for pobj/dobj ent

      Args:

        ent: Span, the pobj/dobj ent that will be used to search status
        include: bool, include ent in the returned expression if True

      Returns:

        status: Span or Token, the identified status
    """
    status = None
    neg = False
    negText = ''
    entRoot = ent.root
    head = entRoot.head
    prep = False
    if head.pos_ in ['VERB']:
      root = head
    elif head.dep_ in ['prep']:
      root = head.head
      prep = True
    else:
      root = head
    if entRoot.dep_ not in ['pobj', 'dobj', 'iobj', 'obj', 'obl', 'oprd']:
      raise IOError("Method 'self.getStatusForObj' can only be used for 'pobj' or 'dobj'")
    if root.pos_ != 'VERB':
      neg, negText = self.isNegation(root)
      if root.pos_ in ['ADJ']:
        status = root
      elif root.pos_ in ['NOUN', 'PROPN']:
        if root.dep_ in ['pobj']:
          status = root.doc[root.head.head.i:root.i+1]
        else:
          status = root
      elif root.pos_ in ['AUX']:
        leftInd = list(root.lefts)[0].i
        subj = root.doc[leftInd:root.i]
        amod = self.findRightKeyword(root)
        status = [amod, subj]
      else:
        logger.warning(f'No status identified for "{ent}" in "{ent.sent}"')
    else:
      subjStatus = None
      if entRoot.dep_ in ['pobj']:
        amod = self.getStatusForPobj(ent, include=include)
      else:
        amod = self.getAmod(ent, ent.start, ent.end, include=include)
         # status = self.getCompoundOnly(ent, entHS)

      passive = self.isPassive(root)
      neg, negText = self.isNegation(root)
      subjStatus = self.findLeftSubj(root, passive)
      if subjStatus is not None:
        # Corefence can be handled coreferee
        # if subjStatus.pos_ in ['PROPN']:
        #   # coreference resolution
        #   passive = self.isPassive(root.head)
        #   neg, negText = self.isNegation(root.head)
        #   headSubj = self.findLeftSubj(root.head, passive)
        #   if headSubj is not None:
        #     subjStatus = headSubj
        subjStatus = self.getAmod(subjStatus, subjStatus.i, subjStatus.i+1, include=True)

      else:
        subjStatus = root
      # if amod is None:
      #   rights =[tk for tk in list(root.rights) if tk.pos_ not in ['SPACE', 'PUNCT'] and tk.i >= ent.end]
      #   if len(rights) > 0 and rights[0].pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']:
      #     status = rights[0]

      if amod is None:
        status = subjStatus
      else:
        status = [amod, subjStatus]

    return status, neg, negText


  def getStatusForPobj(self, ent, include=False):
    """Get the status for ent root pos ``pobj``

      Args:

        ent: Span, the span of entity
        include: bool, ent will be included in returned status if True

      returns:

        Span or Token, the identified health status
    """
    status = None
    if isinstance(ent, Token):
      root = ent
      start = root.i
      end = start + 1
    elif isinstance(ent, Span):
      root = ent.root
      start = ent.start
      end = ent.end
    if root.dep_ not in ['pobj']:
      raise IOError("Method 'self.getStatusForPobj' can only be used for 'pobj'")

    grandparent = root.head.head
    parent = root.head
    if grandparent.dep_ in ['dobj', 'nsubj', 'nsubjpass', 'pobj']:
      lefts = list(grandparent.lefts)
      if len(lefts) == 0:
        leftInd = grandparent.i
      else:
        leftInd = lefts[0].i
      if not include:
        rights = list(grandparent.rights)
        if grandparent.n_rights > 1 and rights[-1] == parent:
          status = grandparent.doc[leftInd:rights[-1].i]
        else:
          status = grandparent.doc[leftInd:grandparent.i+1]
      else:
        status = grandparent.doc[leftInd:end]
      status = self.getAmod(status, status.start, status.end, include=True)
    elif grandparent.pos_ in ['VERB'] and grandparent.dep_ in ['ROOT']:
      dobj = [tk for tk in grandparent.rights if tk.dep_ in ['dobj'] and tk.i < start]
      if len(dobj) > 0:
        dobjEnt = root.doc[dobj[0].i:dobj[0].i+1]
        status = self.getAmod(dobjEnt, dobjEnt.start, dobjEnt.end, include=True)
      else:
        status = ent
        status = self.getAmod(ent, start, end, include=include)
    elif grandparent.pos_ in ['VERB']:
      status = self.findRightObj(grandparent)
      if status is not None:
        subtree = list(status.subtree)
        nbor = self.getNbor(status)
        if status is not None and nbor is not None and nbor.dep_ in ['prep'] and subtree[-1].i < root.i:
          status = grandparent.doc[status.i:subtree[-1].i+1]
        if not include:
          if isinstance(status, Token) and status.i >= root.i:
            status = None
          elif isinstance(status, Span) and status.end >= root.i:
            status = None

    elif grandparent.pos_ in ['NOUN']:
      grandEnt = grandparent.doc[grandparent.i:grandparent.i+1]
      status = self.getAmod(grandEnt, grandparent.i, grandparent.i+1, include=True)
    elif grandparent.pos_ in ['AUX']:
      status = grandparent.doc[grandparent.i+1:parent.i]
    else: # search lefts for amod
      status = self.getAmod(ent, start, end, include)
    return status


