# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on August 1, 2025
@author: wangc, mandd
"""
import os
import logging
import sys
import spacy
import pandas as pd

from .RuleBasedMatcher import RuleBasedMatcher
from .. import config as defaultConfig
from ..utils.nlp.nlp_utils import generatePatternList
# OPL parser to generate object and process lists
from ..utils.opm.OPLparser import OPMobject

class WorkflowManager:
  """_summary_
  """

  def __init__(self, nlp, config):
    self._nlp = nlp
    self._label = config['params']['ent']['label']
    self._entId = config['params']['ent']['id']
    self._patterns = self.processInput(config)
    self._config = config


  def processInput(self, config):
    ents = []
    # Parse OPM model
    # some modifications, bearings --> pump bearings
    if 'opm' in config['files']:
      opmFile = config['files']['opm']
      opmObj = OPMobject(opmFile)
      formList = opmObj.returnObjectList()
      # functionList = opmObj.returnProcessList()
      # attributeList = opmObj.returnAttributeList()
      ents.extend(formList)
    if 'entity' in config['files']:
      entityFile = config['files']['entity']
      entityList = pd.read_csv(entityFile).values.ravel().tolist()
      ents.extend(entityList)

    ents = set(ents)

    # convert opm formList into matcher patternsOPM
    patterns = generatePatternList(ents, label=self._label, id=self._entId, nlp=self._nlp, attr="LEMMA")
    return patterns

  def run(self, doc):
    ########################################################################
    #  Parse causal keywords, and generate patterns for them
    #  The patterns can be used to identify the causal relationships
    causalLabel = "causal_keywords"
    causalID = "causal"
    patternsCausal = []
    causalFilename = defaultConfig.nlpConfig['files']['cause_effect_keywords_file']
    ds = pd.read_csv(causalFilename, skipinitialspace=True)
    for col in ds.columns:
      vars = set(ds[col].dropna())
      patternsCausal.extend(generatePatternList(vars, label=causalLabel, id=causalID, nlp=self._nlp, attr="LEMMA"))

    name = 'ssc_entity_ruler'
    matcher = RuleBasedMatcher(self._nlp, entID=self._entId, causalKeywordID=causalID)
    matcher.addEntityPattern(name, self._patterns)

    causalName = 'causal_keywords_entity_ruler'
    matcher.addEntityPattern(causalName, patternsCausal)

    matcher(doc)

  def get(self):
    pass

  def write(self, fname, style='csv'):
    pass

  def visualize(self):
    pass

  def reset(self):
    pass
