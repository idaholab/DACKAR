# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on August 1, 2025
@author: wangc, mandd
"""
import os
from pathlib import Path
import logging
import pandas as pd
from spacy import displacy
import spacy

# import pipelines
from ..pipelines.ConjectureEntity import ConjectureEntity
# from ..pipelines.PhraseEntityMatcher import PhraseEntityMatcher
from ..pipelines.UnitEntity import UnitEntity
# from ..pipelines.SimpleEntityMatcher import SimpleEntityMatcher
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
# # import similarity
# from ..similarity import simUtils
# from ..similarity import synsetUtils
# from ..similarity.SentenceSimilarity import SentenceSimilarity

# import utils
from ..utils.nlp.nlp_utils import generatePatternList
from ..utils.nlp.nlp_utils import resetPipeline
from ..utils.nlp.nlp_utils import extractNER

# OPL parser to generate object and process lists
from ..utils.opm.OPLparser import OPMobject
from ..utils.mbse.LMLparser import LMLobject

from ..causal.CausalSentence import CausalSentence
from ..causal.CausalPhrase import CausalPhrase
from ..causal.OperatorShiftLogsProcessing import OperatorShiftLogs
from .. import config as defaultConfig

from ..validate import validateToml

from ..contrib.lazy import lazy_loader

# load library for neo4j
from ..knowledge_graph.py2neo import Py2Neo
from ..knowledge_graph.graph_utils import set_neo4j_import_folder


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
  """Workflow Manager
  """

  def __init__(self, config):
    logger.info('Initialization')
    self._nlpConfig = None
    self._neo4jConfig = None
    self._config = config
    # validate input
    self._validate(config)
    if 'nlp' in config:
      self._nlpConfig = config['nlp']
      self.initializeNLP()
    elif 'neo4j' in config:
      self._neo4jConfig = config['neo4j']
      self.initializeNeo4j()

  def initializeNLP(self):
    """Initialize NLP calculation
    """
    config = self._nlpConfig
    # load nlp model
    nlp = spacy.load(config['language_model'], exclude=[])
    self._nlp = nlp
    self._label = config['ent']['label']
    self._entId = config['ent']['id']
    self._causalLabel = "causal"
    self._causalID = "causal"
    self._entPatternName = 'dackar_ent'
    self._causalPatternName = 'dackar_causal'
    self._patterns = self.generatePattern(config)
    self._causalPatterns = self.processCausalEnt()

    self._causalFlow = None
    # pre-processing
    self._pp = self.preprocessing()
    # Construct execution logic
    self._mode = config['analysis']['type']
    if self._mode == 'ner':
      # add customized NER pipes
      self.ner()
    elif self._mode == 'causal':
      # setup workflow
      self._causalFlow = self.causal()
    else:
      raise IOError(f'Unrecognized analysis type {self._mode}')

    # text that needs to be processed. either load from file or direct assign
    textFile = config['files']['text']
    with open(textFile, 'r') as ft:
      self._doc = ft.read()

  def initializeNeo4j(self):
    """Initialize NEO4j settings
    """
    self._uri = self._neo4jConfig['uri']
    self._pwd = self._neo4jConfig['pwd']
    self._reset = self._neo4jConfig['reset'] if 'reset' in self._neo4jConfig else False
    # # change import folder to user specific location
    # self._neoConf = self._neo4jConfig['config_file_path'] if 'config_file_path' in self._neo4jConfig else None
    # self._neoImport = self._neo4jConfig['import_folder_path'] if 'import_folder_path' in self._neo4jConfig else None
    # if self._neoConf is not None and self._neoImport is not None:
    #   set_neo4j_import_folder(self._neoConf, self._neoImport)
    # create neo4j driver
    self._py2neo = Py2Neo(uri=self._uri, user='neo4j', pwd=self._pwd)
    if self._reset:
      self._py2neo.reset()

  def runNLP(self):
    """Execute the knowledge extraction

    Args:
        doc (str): raw text data to process
    """
    logger.info('Execute workflow %s', self._mode)
    doc = self._doc
    # pre-processing text
    if self._pp is not None:
        doc = self._pp(doc)
    # Logic for analysis
    if self._mode == 'ner':
      doc = self._nlp(doc)
      # output data
      df = extractNER(doc)
      self.write(df, 'ner.csv', style='csv')
    elif self._mode == 'causal':
      self._causalFlow(doc)
      # output entity data with status
      entHS = self._causalFlow.getAttribute('entHS')
      entStatus = self._causalFlow.getAttribute('entStatus')
      if entHS is not None and len(entHS) != 0:
        self.write(entHS, 'causal_ner_health_status.csv', style='csv')
      if entStatus is not None and len(entStatus) != 0:
        self.write(entStatus, 'causal_ner_status.csv', style='csv')
      # output causal data
      causalRelation = self._causalFlow.getAttribute('causalRelation')
      relationGeneral = self._causalFlow.getAttribute('relationGeneral')
      if causalRelation is not None and len(causalRelation) != 0:
        self.write(causalRelation, 'causal_relation.csv', style='csv')
      if relationGeneral is not None and len(relationGeneral) != 0:
        self.write(relationGeneral, 'relation_general.csv', style='csv')

      doc = self._causalFlow.getAttribute('doc')

    if 'visualize' in self._nlpConfig and 'ner' in self._nlpConfig['visualize']:
      if self._nlpConfig['visualize']['ner']:
        self.visualize(doc)

  def runNeo4j(self):
    """Load data into neo4j
    """
    for node in self._neo4jConfig['node']:
      self._py2neo.load_csv_for_nodes(node['file'], node['label'], node['attribute'])
    for edge in self._neo4jConfig['edge']:
      labelAttr = edge['label_attribute'] if 'label_attribute' in edge else None
      self._py2neo.load_csv_for_relations(edge['file'],
                                          edge['source_label'],
                                          edge['source_attribute'],
                                          edge['target_label'],
                                          edge['target_attribute'],
                                          edge['label'],
                                          labelAttr)


  def run(self):
    """Execute the workflow
    """
    if self._nlpConfig is not None:
      self.runNLP()
    if self._neo4jConfig is not None:
      self.runNeo4j()

  def write(self, data, fname, style='csv'):
    """Dump data

    Args:
        data (pandas.DataFrame): output data to dump
        fname (str): file name to save the data
        style (str, optional): type of file. Defaults to 'csv'.
    """
    if isinstance(data, pd.DataFrame):
      data.to_csv(fname, index=False)
    else:
      pass

  def visualize(self, doc):
    """visual entities

    Args:
        doc (spacy.tokens.doc.Doc): the processed document using nlp pipelines
    """
    cwd = os.getcwd()
    svg = displacy.render(doc, style='ent', page=True, minify=True)
    outputPath = Path(os.path.join(cwd, 'ent.svg'))
    outputPath.open("w", encoding="utf-8").write(svg)


  def reset(self):
    pass

  ############################################
  #      Internal Functions
  ############################################

  def _validate(self, config):
    """validate dackar input file using JSON schema

    Args:
        config (dict): dictionary for dackar input

    Raises:
        IOError: error out if not valid
    """
    # validate
    validate = validateToml(config)
    if not validate:
      logger.error("TOML input file is invalid.")
      raise IOError("TOML input file is invalid.")

  def generatePattern(self, config):
    """Generate patterns using provided OPM and/or entity file

    Args:
        config (dict): input dictionary

    Returns:
        list: list of patterns will be used by entity matcher
    """
    ents = []
    # Parse OPM model
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
    """
    Parse causal keywords, and generate patterns for them
    The patterns can be used to identify the causal relationships

    Returns:
        list: list of patterns will be used by causal entity matcher
    """
    patternsCausal = []
    causalFilename = defaultConfig.nlpConfig['files']['cause_effect_keywords_file']
    ds = pd.read_csv(causalFilename, skipinitialspace=True)
    for col in ds.columns:
      cvars = set(ds[col].dropna())
      patternsCausal.extend(generatePatternList(cvars, label=self._causalLabel, id=self._causalID, nlp=self._nlp, attr="LEMMA"))
    return patternsCausal

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
    if 'processing' not in self._nlpConfig or len(self._nlpConfig['processing']) == 0:
      return None

    for pp, pval in self._nlpConfig['processing'].items():
      if isinstance(pval, bool) and pval:
        ppList.append(pp)
      elif not isinstance(pval, bool):
        ppList.append(pp)
        if pp in ['punctuation', 'brackets']:
          ppOptions.update({pp:{'only':pval}})
        else:
          raise IOError(f'Unrecognized option for processing {pp}!')
          # ppOptions.update({pp:pval})
    preprocess = Preprocessing(ppList, ppOptions)
    return preprocess


  def ner(self):
    """Set up NER pipelines

    Raises:
        NER Object: Object to conduct NER
    """
    pipelines = []
    if 'ner' in self._nlpConfig:
      for pipe in self._nlpConfig['ner']:
        if pipe in NERMapping:
          pipelines.append(NERMapping[pipe])
        else:
          raise IOError(f'Unrecognized ner {pipe}!')
      # add aliasResolver
      pipelines.append('aliasResolver')
      self._nlp = resetPipeline(self._nlp, pipes=pipelines)
    self._nlp.add_pipe("general_entity", config={"patterns": self._patterns}, before='ner')


  def causal(self):
    """Set up causal analysis flow

    Returns:
        Workflow Object: Object to conduct causal analysis
    """
    method = None
    matcher = None
    if 'causal' in self._nlpConfig:
      method = self._nlpConfig['causal']['type'] if 'type' in self._nlpConfig['causal'] else None
    if method is not None:
      if method == 'general':
        matcher = CausalSentence(self._nlp, entID=self._entId, causalKeywordID=self._causalID)
      elif method == 'phrase':
        matcher = CausalPhrase(self._nlp, entID=self._entId, causalKeywordID=self._causalID)
      elif method == 'osl':
        matcher = OperatorShiftLogs(self._nlp, entID=self._entId, causalKeywordID=self._causalID)
      else:
        raise IOError(f'Unrecognized causal type {method}')
      matcher.addEntityPattern(self._entPatternName, self._patterns)
      matcher.addEntityPattern(self._causalPatternName, self._causalPatterns)
    return matcher
