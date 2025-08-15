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
from dackar.workflows.WorkflowManager import WorkflowManager

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
# To enable the logging to both file and console, the logger for the main should be the root,
# otherwise, a function to add the file handler and stream handler need to be created and called by each module.
# logger = logging.getLogger(__name__)
logger = logging.getLogger('DACKAR')
# # create file handler which logs debug messages
fh = logging.FileHandler(filename='dackar.log', mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)


def main():
  logger.info('Welcome!')
  # set up argument parser
  parser = argparse.ArgumentParser(description='DACKAR Input ArgumentParser')
  parser.add_argument('-i', '--file_path', type=str, default='../../system_tests/test_opm.toml' ,help='The path to the input file.')
  parser.add_argument('-o', '--output-file', type=str, default='output.txt', help='The file to save the output to.')
  # parse the arguments
  args = parser.parse_args()
  logger.info('Input file: %s', args.file_path)
  # read the TOML file
  cwd = os.getcwd()
  configFile = os.path.join(cwd, args.file_path)
  inputDict = readToml(configFile)

  # doc = "The Pump is not experiencing enough flow for the pumps to keep the check valves open during test."
  # text that needs to be processed. either load from file or direct assign
  textFile = inputDict['files']['text']
  with open(textFile, 'r') as ft:
    doc = ft.read()

  # load nlp model
  nlp = spacy.load(inputDict['params']['language_model'], exclude=[])

  # Create logic to switch between different analysis



  module = WorkflowManager(nlp, inputDict)
  module.run(doc.lower())

  logger.info(' ... Complete!')

if __name__ == '__main__':
  sys.exit(main())
