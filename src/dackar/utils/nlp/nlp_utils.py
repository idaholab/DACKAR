# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""
import pandas as pd
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from spacy.tokens import Span
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans
import logging

import networkx as nx
import matplotlib.pyplot as plt


logger = logging.getLogger('DACKAR.nlp_utils')

###########################################################################

def displayNER(doc, includePunct=False):
  """
    Generate data frame for visualization of spaCy doc with custom attributes.

    Args:

      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines
      includePunct: bool, True if the punctuaction is included

    Returns:

      df: pandas.DataFrame, data frame contains attributes of tokens
  """
  rows = []
  for i, t in enumerate(doc):
    if not t.is_punct or includePunct:
      row = {'token': i,
             'text': t.text, 'lemma': t.lemma_,
             'pos': t.pos_, 'dep': t.dep_, 'ent_type': t.ent_type_,
             'ent_iob_': t.ent_iob_}
      if doc.has_extension('coref_chains'):
        if t.has_extension('coref_chains') and t._.coref_chains: # neuralcoref attributes
          row['coref_chains'] = t._.coref_chains.pretty_representation
        else:
          row['coref_chains'] = None
      if t.has_extension('ref_n'): # referent attribute
        row['ref_n'] = t._.ref_n
        row['ref_t'] = t._.ref_t
      if t.has_extension('ref_ent'): # ref_n/ref_t
        row['ref_ent'] = t._.ref_ent
      rows.append(row)
  df = pd.DataFrame(rows).set_index('token')
  df.index.name = None

  return df

def resetPipeline(nlp, pipes):
  """
    remove all custom pipes, and add new pipes

    Args:

      nlp: spacy.Language object, contains all components and data needed to process text
      pipes: list, list of pipes that will be added to nlp pipeline

    Returns:

      nlp: spacy.Language object, contains updated components and data needed to process text
  """
  customPipes = [pipe for (pipe, _) in nlp.pipeline
                  if pipe not in ['tagger', 'parser',
                                  'tok2vec', 'attribute_ruler', 'lemmatizer', 'ner']]
  for pipe in customPipes:
    _ = nlp.remove_pipe(pipe)
  # re-add specified pipes
  for pipe in pipes:
    if pipe in ['pysbdSentenceBoundaries']:
      # nlp.add_pipe(pipe, before='parser')
      nlp.add_pipe(pipe, first=True)
    elif pipe not in ['tagger', 'parser', 'tok2vec', 'attribute_ruler', 'lemmatizer', 'ner']:
      nlp.add_pipe(pipe)
  logger.info(f"Model: {nlp.meta['name']}, Language: {nlp.meta['lang']}")
  logger.info('Available pipelines:'+', '.join([pipe for (pipe,_) in nlp.pipeline]))
  return nlp

def printDepTree(doc, skipPunct=True):
  """
    Utility function to pretty print the dependency tree.

    Args:

      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines
      skipPunct: bool, True if skip punctuactions

    Returns:

      None
  """
  def printRecursive(root, indent, skipPunct):
    if not root.dep_ == 'punct' or not skipPunct:
      print(" "*indent + f"{root} [{root.pos_}, {root.dep_}]")
    for left in root.lefts:
      printRecursive(left, indent=indent+4, skipPunct=skipPunct)
    for right in root.rights:
      printRecursive(right, indent=indent+4, skipPunct=skipPunct)

  for sent in doc.sents: # iterate over all sentences in a doc
    printRecursive(sent.root, indent=0, skipPunct=skipPunct)


def plotDAG(edges, colors='k'):
  """
    Args:

      edges: list of tuples, [(subj, conj), (..,..)] or [(subj, conj, {"color":"blue"}), (..,..)]
      colors: str or list, list of colors
  """
  g = nx.MultiDiGraph()
  g.add_edges_from(edges)
  nx.draw_networkx(g, edge_color=colors)
  ax = plot.gca()
  plt.axis("off")
  plot.show()

def extractLemma(var, nlp):
  """
    Lammatize the variable list

    Args:

      var: str, string
      nlp: object, preloaded nlp model

    Returns:

      lemVar: list, list of lammatized variables
  """
  mvar = ' '.join(var.split())
  lemVar = [token.lemma_ for token in nlp(mvar)]
  return lemVar

def generatePattern(form, label, id, attr="LOWER"):
  """
    Generate entity pattern

    Args:

      form: str or list, the given str or list of lemmas that will be used to generate pattern
      label: str, the label name for the pattern
      id: str, the id name for the pattern
      attr: str, attribute used for the pattern, either "LOWER" or "LEMMA"

    Returns:

      pattern: dict, pattern will be used by entity matcher
  """
  # if any of "!", "?", "+", "*" present in the provided string "form", we will treat it as determiter for the form
  if attr.lower() == "lower":
    ptn = []
    attr = "LOWER"
    for elem in form.lower().split():
      if "-" in form:
        subList = list(elem.split("-"))
        # slow when using re.split
        # subList = re.split("(-)", elem)
        # for sub in subList:
        #   if sub != '-':
        #     ptn.append({attr:sub} if sub not in ["!", "?", "+", "*"] else {"POS":"DET", "OP":sub})
        #   else:
        #     ptn.append({"ORTH":"-"})
        for i, sub in enumerate(subList):
          ptn.append({attr:sub} if sub not in ["!", "?", "+", "*"] else {"POS":"DET", "OP":sub})
          if i < len(subList)-1:
            ptn.append({"ORTH":"-"})
      else:
        ptn.append({attr:elem} if elem not in ["!", "?", "+", "*"] else {"POS":"DET", "OP":elem})
  elif attr.lower() == "lemma":
    attr = "LEMMA"
    ptn = [{attr:elem} if elem not in ["!", "?", "+", "*"] else {"POS":"DET", "OP":elem} for elem in form]
  else:
    raise IOError(f"Incorrect 'attr={attr}' is provided, valid value for 'attr' is either 'LOWER' or 'LEMMA'")
  pattern = {"label":label, "pattern":ptn, "id": id}
  return pattern

def generatePatternList(entList, label, id, nlp, attr="LOWER"):
  """
    Generate a list of entity patterns

    Args:

      entList: list, list of entities
      label: str, the label name for the pattern
      id: str, the id name for the pattern
      attr: str, attribute used for the pattern, either "LOWER" or "LEMMA"

    Returns:

      ptnList, list, list of patterns will be used by entity matcher
  """
  ptnList = []
  for ent in entList:
    if len(ent.strip()) == 0:
      continue
    # if default is LEMMA, we should also add LOWER to the pattern, since the lemma for "Cooling System"
    # will be "cool System", which can not capture "cool system".
    ptn = generatePattern(ent, label, id, attr="LOWER")
    ptnList.append(ptn)
    if attr.lower() == "lemma":
      try:
        entLemma = extractLemma(ent, nlp)
      except ValueError:
        print(f'Can not add to pattern list {ent}')
        continue
      ptnLemma = generatePattern(entLemma, label, id, attr)
      # print(ptnLemma)
      ptnList.append(ptnLemma)
  return ptnList

###############
# methods can be used for callback in "add" method
###############
def extendEnt(matcher, doc, i, matches):
  """
  Extend the doc's entity

  Args:

    matcher: spacy.Matcher, the spacy matcher instance
    doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines
    i: int, index of the current match (matches[i])
    matches: List[Tuple[int, int, int]], a list of (match_id, start, end) tuples, describing
    the matches. A match tuple describes a span doc[start:end]
  """
  id, start, end = matches[i]
  ent = Span(doc, start, end, label=id)
  logger.debug(ent.label_ + ' ' + ent.text)
  doc.ents = filter_spans(list(doc.ents) +[ent])


def customTokenizer(nlp):
  """custom tokenizer to keep hyphens between letters and digits
  When apply tokenizer, the words with hyphens will be splitted into multiple tokens,
  this function can be used to avoid the split of the words when hyphens are present.

  Args:
      nlp (spacy nlp model): spacy nlp model

  Returns:
      nlp: nlp with custom tokenizer
  """
  inf = list(nlp.Defaults.infixes)               # Default infixes
  inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
  inf = tuple(inf)                               # Convert inf to tuple
  infixes = inf + tuple([r"(?<=[0-9])[+\*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
  infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
  infix_re = compile_infix_regex(infixes)
  nlp.tokenizer = Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                            suffix_search=nlp.tokenizer.suffix_search,
                            infix_finditer=infix_re.finditer,
                            token_match=nlp.tokenizer.token_match,
                            rules=nlp.Defaults.tokenizer_exceptions)
  return nlp


def extractNER(doc):
  """
    Generate data frame for visualization of spaCy doc with custom NER.

    Args:

      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines

    Returns:

      df: pandas.DataFrame, data frame contains attributes of NER tokens
  """
  rows = []
  for ent in doc.ents:
    row = {'Entity':ent, 'ID': ent.ent_id_, 'Label': ent.label_, 'Start': ent.start, 'End': ent.end}
    rows.append(row)
  df = pd.DataFrame(rows)
  df.index.name = None
  return df
