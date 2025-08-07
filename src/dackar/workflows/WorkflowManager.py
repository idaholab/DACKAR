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



# import pipelines
from ..pipelines.ConjectureEntity import ConjectureEntity
from ..pipelines.PhraseEntityMatcher import PhraseEntityMatcher
from ..pipelines.UnitEntity import UnitEntity
from ..pipelines.SimpleEntityMatcher import SimpleEntityMatcher
from dackar.pipelines.TemporalEntity import Temporal
from ..pipelines.TemporalAttributeEntity import TemporalAttributeEntity
from ..pipelines.TemporalRelationEntity import TemporalRelationEntity
from ..pipelines.LocationEntity import LocationEntity
from ..pipelines.EmergentActivityEntity import EmergentActivity
from ..pipelines.GeneralEntity import GeneralEntity
from ..pipelines.CustomPipelineComponents import normEntities
from ..pipelines.CustomPipelineComponents import initCoref
from ..pipelines.CustomPipelineComponents import aliasResolver
from ..pipelines.CustomPipelineComponents import anaphorCoref
from ..pipelines.CustomPipelineComponents import anaphorEntCoref
from ..pipelines.CustomPipelineComponents import expandEntities
from ..pipelines.CustomPipelineComponents import mergePhrase
from ..pipelines.CustomPipelineComponents import pysbdSentenceBoundaries
# import text processing
from ..text_processing.Preprocessing import Preprocessing
# import similarity
from ..similarity import simUtils
from ..similarity import synsetUtils
from ..similarity.SentenceSimilarity import SentenceSimilarity
# import utils
from ..utils.nlp.nlp_utils import generatePatternList
from ..utils.nlp.nlp_utils import resetPipeline
from ..utils.nlp.CreatePatterns import CreatePatterns
# OPL parser to generate object and process lists
from ..utils.opm.OPLparser import OPMobject
from ..utils.mbse.LMLparser import LMLobject

from .RuleBasedMatcher import RuleBasedMatcher
from .. import config as defaultConfig

from ..contrib.lazy import lazy_loader


NERMapping = {'temporal':'Temporal',
              'unit':'unit_entity',
              'temporal_relation':'temporal_relation_entity',
              'temporal_attribute':'temporal_attribute_entity',
              'location':'location_entity',
              'emergent_activity':'EmergentActivity',
              'conjecture':'conjecture_entity',
              'merge':'merge_entities'

}


customPipe = {'norm':'normEntities',
              'alias':'aliasResolver',
              'expand':'expandEntities',
              'merge':'mergePhrase',
              'sbd':'pysbdSentenceBoundaries'
}




logger = logging.getLogger('DACKAR.WorkflowManager')


class WorkflowManager:
  """_summary_
  """

  def __init__(self, nlp, config):
    self._nlp = nlp
    self._label = config['params']['ent']['label']
    self._entId = config['params']['ent']['id']

    self._causalLabel = "causal"
    self._causalID = "causal"

    self._patterns = self.generatePattern(config)
    self._causalPatterns = self.processCausalEnt()


    self._config = config

    # pre-processing
    self._pp = self.preprocessing()
    # add customized NER pipes
    self.ner()
    # setup workflow
    self._workflow = self.setWorkflow()


  def run(self, doc):

    if self._pp is not None:
      doc = self._pp(doc)
    self._workflow(doc)

  def get(self):
    pass

  def write(self, fname, style='csv'):
    pass

  def visualize(self):
    pass

  def reset(self):
    pass




############################################
#      Internal Functions
############################################

  def generatePattern(self, config):
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

  def processCausalEnt(self):
    ########################################################################
    #  Parse causal keywords, and generate patterns for them
    #  The patterns can be used to identify the causal relationships

    patternsCausal = []
    causalFilename = defaultConfig.nlpConfig['files']['cause_effect_keywords_file']
    ds = pd.read_csv(causalFilename, skipinitialspace=True)
    for col in ds.columns:
      cvars = set(ds[col].dropna())
      patternsCausal.extend(generatePatternList(cvars, label=self._causalLabel, id=self._causalID, nlp=self._nlp, attr="LEMMA"))
    return patternsCausal

  def setWorkflow(self):
    name = 'ssc_entity_ruler'
    matcher = RuleBasedMatcher(self._nlp, entID=self._entId, causalKeywordID=self._causalID)
    matcher.addEntityPattern(name, self._patterns)

    causalName = 'causal_keywords_entity_ruler'
    matcher.addEntityPattern(causalName, self._causalPatterns)
    return matcher


  def preprocessing(self):
    """setup text pre-processing pipeline

    Raises:
        IOError: if pipeline option is not available

    Returns:
        Preprocessing Object: Preprocessing pipeline
    """

    logger.info('Set up text pre-processing.')
    ppList = []
    ppOptions = {}
    if 'processing' not in self._config or len(self._config['processing']) == 0:
      return None
    for ptype in self._config['processing']:
      for pp, pval in self._config['processing'][ptype].items():
        if isinstance(pval, bool) and pval:
          ppList.append(pp)
        elif not isinstance(pval, bool):
          ppList.append(pp)
          if pp in ['punctuation', 'brackets']:
            ppOptions.update({pp:{'only':pval}})
          else:
            raise IOError(f'Unrecognized option for {ptype} {pp}!')
            # ppOptions.update({pp:pval})
    preprocess = Preprocessing(ppList, ppOptions)
    return preprocess

  def ner(self):
    """Set up NER pipelines

    Raises:
        IOError: if pipeline is not available
    """
    pipelines = []
    if 'ner' in self._config:
      for pipe in self._config['ner']:
        if pipe in NERMapping:
          pipelines.append(NERMapping[pipe])
        else:
          raise IOError(f'Unrecognized ner {pipe}!')

      self._nlp = resetPipeline(self._nlp, pipes=pipelines)


  # TODO
  def causal(self):
    return None
