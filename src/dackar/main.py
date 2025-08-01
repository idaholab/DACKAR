# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on July 31, 2025
@author: wangc, mandd
"""
import os
import argparse
import logging
import sys
import spacy
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dackar.utils.utils import readToml, writeToFile
from dackar.workflows.RuleBasedMatcher import RuleBasedMatcher
from dackar import config
from dackar.utils.nlp.nlp_utils import generatePatternList
# OPL parser to generate object and process lists
from dackar.utils.opm.OPLparser import OPMobject

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
# To enable the logging to both file and console, the logger for the main should be the root,
# otherwise, a function to add the file handler and stream handler need to be created and called by each module.
# logger = logging.getLogger(__name__)
logger = logging.getLogger('DACKAR')
# # create file handler which logs debug messages
fh = logging.FileHandler(filename='dackar.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)



def processInput(inputDict, nlp, label, entId):

  ents = []
  # Parse OPM model
  # some modifications, bearings --> pump bearings
  if 'opm' in inputDict['files']:
    opmFile = inputDict['files']['opm']
    opmObj = OPMobject(opmFile)
    formList = opmObj.returnObjectList()
    # functionList = opmObj.returnProcessList()
    # attributeList = opmObj.returnAttributeList()
    ents.extend(formList)
  if 'entity' in inputDict['files']:
    entityFile = inputDict['files']['entity']
    entityList = pd.read_csv(entityFile).values.ravel().tolist()
    ents.extend(entityList)

  ents = set(ents)

  # convert opm formList into matcher patternsOPM
  patterns = generatePatternList(ents, label=label, id=entId, nlp=nlp, attr="LEMMA")

  return patterns


def get_analysis_module(patterns, nlp, entId):

  ########################################################################
  #  Parse causal keywords, and generate patterns for them
  #  The patterns can be used to identify the causal relationships
  causalLabel = "causal_keywords"
  causalID = "causal"
  patternsCausal = []
  causalFilename = config.nlpConfig['files']['cause_effect_keywords_file']
  ds = pd.read_csv(causalFilename, skipinitialspace=True)
  for col in ds.columns:
    vars = set(ds[col].dropna())
    patternsCausal.extend(generatePatternList(vars, label=causalLabel, id=causalID, nlp=nlp, attr="LEMMA"))

  name = 'ssc_entity_ruler'
  matcher = RuleBasedMatcher(nlp, entID=entId, causalKeywordID=causalID)
  matcher.addEntityPattern(name, patterns)

  causalName = 'causal_keywords_entity_ruler'
  matcher.addEntityPattern(causalName, patternsCausal)

  return matcher



def main():
  logger.info('Welcome!')
  # set up argument parser
  parser = argparse.ArgumentParser(description='DACKAR Input ArgumentParser')
  parser.add_argument('file_path', type=str, default='./system_tests/test_opm.toml' ,help='The path to the input file.')
  parser.add_argument('--output-file', type=str, default='output.txt', help='The file to save the output to.')
  # parse the arguments
  args = parser.parse_args()
  logger.info('Input file: %s', args.file_path)
  # read the TOML file
  inputDict = readToml(args.file_path)

  label = inputDict['params']['ent']['label']
  entId = inputDict['params']['ent']['id']

  # doc = "The Pump is not experiencing enough flow for the pumps to keep the check valves open during test."
  # text that needs to be processed. either load from file or direct assign
  textFile = inputDict['files']['text']
  with open(textFile, 'r') as ft:
    doc = ft.read()

  # load nlp model
  nlp = spacy.load(inputDict['params']['language_model'], exclude=[])

  patterns = processInput(inputDict, nlp, label, entId)

  module = get_analysis_module(patterns, nlp, entId)
  module(doc.lower())



  logger.info(' ... Complete!')

if __name__ == '__main__':
  sys.exit(main())
