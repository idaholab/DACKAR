# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""
import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher
from spacy.tokens import Token
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans
import pandas as pd

# use pysbd as a sentencizer component for spacy
import pysbd
from ..config import nlpConfig

import logging


logger = logging.getLogger(__name__)

#### Using spacy's Token extensions for coreferee
if Token.has_extension('ref_n'):
  _ = Token.remove_extension('ref_n')
if Token.has_extension('ref_t'):
  _ = Token.remove_extension('ref_t')
if Token.has_extension('ref_t_'):
  _ = Token.remove_extension('ref_t_')
Token.set_extension('ref_n', default='')
Token.set_extension('ref_t', default='')
if not Token.has_extension('alias'):
  Token.set_extension('alias', default=None)

if not Span.has_extension('health_status'):
  Span.set_extension("health_status", default=None)
if not Span.has_extension('alias'):
  Span.set_extension("alias", default=None)
if not Token.has_extension('ref_ent'):
  Token.set_extension("ref_ent", default=None)
if not Span.has_extension('alias'):
  Span.set_extension("alias", default=None)

customLabel = ['STRUCTURE', 'COMPONENT', 'SYSTEM']
aliasLookup = {}
if 'alias_file' in nlpConfig['files']:
  df = pd.read_csv(nlpConfig['files']['alias_file'], index_col='alias')
  aliasLookup.update(df.to_dict()['name'])

def getEntID():
  """
  """
  if 'params' in nlpConfig:
    entLabel = nlpConfig['params'].get('ent_label', "SSC")
    entID = nlpConfig['params'].get('ent_id', "SSC")
  else:
    entLabel = "SSC"
    entID = "SSC"
  return entID, entLabel

# Use Config File to update aliasLookup Dictionary

# orders of NLP pipeline: 'ner' --> 'normEntities' --> 'merge_entities' --> 'initCoref'
# --> 'aliasResolver' --> 'coreferee' --> 'anaphorCoref'

@Language.component("normEntities")
def normEntities(doc):
  """
    Normalizing Named Entities, remove the leading article and trailing particle

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after the normalizing named entities
  """
  ents = []
  for ent in doc.ents:
    if ent[0].pos_ == "DET": # leading article
      ent = Span(doc, ent.start+1, ent.end, label=ent.label)
    if len(ent) > 0:
      if ent[-1].pos_ == "PART": # trailing particle like 's
        ent = Span(doc, ent.start, ent.end-1, label=ent.label)
      if len(ent) > 0:
        ents.append(ent)
  doc.ents = tuple(ents)
  return doc

@Language.component("initCoref")
def initCoref(doc):
  """
    Initialize the coreference, assign text and label to custom extension ``ref_n`` and ``ref_t``

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after the initializing coreference
  """
  for e in doc.ents:
    #
    # if e.label_ in customLabel:
    e[0]._.ref_n, e[0]._.ref_t = e.text, e.label_
  return doc

@Language.component("aliasResolver")
def aliasResolver(doc):
  """
    Lookup aliases and store result in ``alias``

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after the alias lookup
  """
  for ent in doc.ents:
    alias = ent.text.lower()
    if alias in aliasLookup:
      name = aliasLookup[alias]
      ent._.set('alias', name)
  return doc

def propagateEntType(doc):
  """
    propagate entity type stored in ``ref_t``

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after entity type extension
  """
  ents = []
  for e in doc.ents:
    if e[0]._.ref_n != '': # if e is a coreference
      e = Span(doc, e.start, e.end, label=e[0]._.ref_t)
    ents.append(e)
  doc.ents = tuple(ents)
  return doc

@Language.component("anaphorCoref")
def anaphorCoref(doc):
  """
    Anaphora resolution using coreferee
    This pipeline need to be added after NER.
    The assumption here is: The entities need to be recognized first, then call
    pipeline ``initCoref`` to assign initial custom attribute ``ref_n`` and ``ref_t``,
    then call pipeline ``aliasResolver`` to resolve all the aliases used in the text.
    After all these pre-processes, we can use ``anaphorCoref`` pipeline to resolve the
    coreference.

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after the anaphora resolution using coreferee
  """
  if not Token.has_extension('coref_chains'):
    return doc
  for token in doc:
    coref = token._.coref_chains
    # if token is coref and not already dereferenced
    if coref and token._.ref_n == '':
      # check all the references, if "ref_n" is available (determined by NER and initCoref),
      # the value of "ref_n" will be assigned to current totken
      for chain in coref:
        for ref in chain:
          refToken = doc[ref[0]]
          if refToken._.ref_n != '':
            token._.ref_n = refToken._.ref_n
            token._.ref_t = refToken._.ref_t
            break
  return doc

@Language.component("anaphorEntCoref")
def anaphorEntCoref(doc):
  """
    Anaphora resolution using coreferee for Entities
    This pipeline need to be added after NER.
    The assumption here is: The entities need to be recognized first, then call
    pipeline ``initCoref`` to assign initial custom attribute ``ref_n`` and ``ref_t``,
    then call pipeline ``aliasResolver`` to resolve all the aliases used in the text.
    After all these pre-processes, we can use ``anaphorEntCoref`` pipeline to resolve the
    coreference.

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after the anaphora resolution using coreferee
  """
  if not Token.has_extension('coref_chains'):
    return doc

  for ent in doc.ents:
    for token in ent:
      coref = token._.coref_chains
      if not coref:
        continue
      for chain in coref:
        for ref in chain:
          for index in ref:
            refToken = doc[index]
            if refToken._.ref_ent is None:
              refToken._.ref_ent = ent
  return doc



@Language.component("expandEntities")
def expandEntities(doc):
  """
    Expand the current entities, recursive function to extend entity with all previous NOUN

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after expansion of current entities
  """
  newEnts = []
  isUpdated = False
  entID, _ = getEntID()
  for ent in doc.ents:
    if ent.ent_id_ == entID and ent.start != 0:
      prevToken = doc[ent.start - 1]
      if prevToken.pos_ in ['NOUN']:
        newEnt = Span(doc, ent.start - 1, ent.end, label=ent.label)
        newEnts.append(newEnt)
        isUpdated = True
    else:
      newEnts.append(ent)
  # print(newEnts)
  doc.ents = filter_spans(list(doc.ents) +  newEnts)
  if isUpdated:
    doc = expandEntities(doc)
  return doc


@Language.component("mergeCCWEntities")
def mergeCCWEntities(doc):
  """
    Merge the CCW entities

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after expansion of current entities
  """
  newEnts = []
  isUpdated = False
  ents = list(doc.ents)
  entID, _ = getEntID()
  for i in range(len(ents)-1):
    ent1, ent2 = ents[i], ents[i+1]
    start = ent1.start
    end = ent1.end
    label = ent1.label
    alias = ent1._.alias
    # id = ent1.ent_id
    if ent1.ent_id_ == entID and not isUpdated:
      if start == 1:
        prev = doc[start - 1]
        if prev.pos_ in ['NUM']:
          start = prev.i
      elif start > 1:
        prev1, prev2 = doc[start-1], doc[start-2]
        if prev1.pos_ in ['NUM']:
          start = prev1.i
        elif prev1.dep_ in ['punct'] and prev2.pos_ in ['NUM']:
          start = prev2.i
      if ent2.ent_id_ == entID:
        if end == ent2.start or (end == ent2.start - 1 and doc[end].dep_ in ['punct']):
          end = ent2.end
          label = ent2.label
          if ent2._.alias:
            alias = ent2._.alias
          # id = ent2.ent_id

      if start != ent1.start or end != ent1.end:
        isUpdated = True
        newEnt = Span(doc, start, end, label=label)
        newEnt._.set('alias', alias)
        # print(newEnt, newEnt.ent_id_, newEnt.label_, newEnt._.alias)
        newEnts.append(newEnt)
        # The following can not resolve span attributes

        # with doc.retokenize() as retokenizer:
        #   attrs = {
        #     "tag": newEnt.root.tag,
        #     "dep": newEnt.root.dep,
        #     "ent_type": label,
        #     "ent_id": id,
        #     "_": {
        #           "alias": alias
        #           },
        #   }
        #   retokenizer.merge(newEnt, attrs=attrs)
        # newEnts.append(doc[start:start+1])
        # print("======>: ", newEnts[0], newEnts[0]._.alias)
    if isUpdated:
      break

  doc.ents = filter_spans(list(doc.ents) +  newEnts)
  if isUpdated:
    doc = mergeCCWEntities(doc)
  return doc

@Language.component("mergePhrase")
def mergePhrase(doc):
  """
    Expand the current entities
    This method will keep ``DET`` or ``PART``, using pipeline ``normEntities`` after this pipeline to remove them

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after merge phrase
  """
  def isNum(nounChunks):
    for elem in nounChunks:
      if elem.pos_ == 'NUM':
        return True, elem
        break
    return False, None

  with doc.retokenize() as retokenizer:
    for np in doc.noun_chunks:
      # skip ents since ents are recognized by OPM model and entity_ruler
      # TODO: we may expand the ents, combined with pipeline "expandEntities"
      if len(list(np.ents)) > 1:
        continue
      elif len(list(np.ents)) == 1:
        if np.ents[0].label_ not in ['causal_keywords', 'ORG', 'DATE']:
          # print(np.ents[0].label_)
          continue
      # When a number is provided, we will merge it, but keep the attributes from the number
      num, elem = isNum(np)
      if not num:
        attrs = {
            "tag": np.root.tag_,
            "lemma": np.root.lemma_,
            "pos": np.root.pos_,
            "ent_type": np.root.ent_type_,
            "_": {
                  "ref_n": np.root._.ref_n,
                  "ref_t": np.root._.ref_t,
                  },
        }
      else:
        attrs = {
            "tag": elem.tag_,
            "lemma": elem.lemma_,
            "pos":elem.pos_,
            "ent_type": np.root.ent_type_,
            "_": {
                  "ref_n": np.root._.ref_n,
                  "ref_t": np.root._.ref_t,
                  },
        }
      retokenizer.merge(np, attrs=attrs)
  return doc


@Language.component("pysbdSentenceBoundaries")
def pysbdSentenceBoundaries(doc):
  """
    Use pysbd as a sentencizer component for spacy

    Args:
      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:
      doc: spacy.tokens.doc.Doc, the document after process
  """
  seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
  sentsCharSpans = seg.segment(doc.text)
  charSpans = [doc.char_span(sentSpan.start, sentSpan.end, alignment_mode="contract") for sentSpan in sentsCharSpans]
  startTokenIds = [span[0].idx for span in charSpans if span is not None]
  for token in doc:
      token.is_sent_start = True if token.idx in startTokenIds else False
  return doc
