# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

import pathlib
import tomli
import os

configFileName = 'nlp_config.toml'
configDefault = 'nlp_config_default.toml'

# configFileName = 'nlp_config_ler.toml'
# configFileName = 'nlp_config_cws.toml'
# configFileName = 'nlp_config_demo.toml'

def getConfig(configFileName):
  path = pathlib.Path(os.path.join(pathlib.Path(__file__).parent, configFileName))
  with path.open(mode="rb") as fp:
    config = tomli.load(fp)
    for file in config['files']:
      if file != 'status_keywords_file':
        config['files'][file] = os.path.join(os.path.dirname(__file__), config['files'][file])
      else:
        for sub in config['files'][file]:
          config['files'][file][sub] = os.path.join(os.path.dirname(__file__), config['files'][file][sub])
  return config

# get the config dictionary
nlpConfig = getConfig(configFileName)
nlpConfigDefault = getConfig(configDefault)
# update the config dictionary with default config
for file in nlpConfigDefault['files']:
  if file not in nlpConfig['files']:
    nlpConfig['files'].update({file: nlpConfigDefault['files'][file]})

