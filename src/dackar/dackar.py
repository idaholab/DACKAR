# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

import argparse
import logging
import sys

from .utils import readToml, writeToFile

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
# To enable the logging to both file and console, the logger for the main should be the root,
# otherwise, a function to add the file handler and stream handler need to be created and called by each module.
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
# # create file handler which logs debug messages
fh = logging.FileHandler(filename='dackar.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

def main():

  # set up argument parser
  parser = argparse.ArgumentParser(description='DACKAR Input ArgumentParser')
  parser.add_argument('file_path', type=str, help='The path to the input file.')
  parser.add_argument('--output-file', type=str, default='output.txt', help='The file to save the output to.')
  # parse the arguments
  args = parser.parse_args()
  # read the TOML file
  inputDict = readToml(args.file_path)



if __name__ == '__main__':
  sys.exit(main())
