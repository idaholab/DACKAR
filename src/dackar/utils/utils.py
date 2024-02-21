# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on February, 2024

@author: wangc, mandd
"""
import re
import logging
logger = logging.getLogger(__name__)


def getOnlyWords(s):
  """
    Returns a string with only the words (removes things like T8, A-b, etc)

    Args:
      s: string

    Returns:
      string with only the words
  """
  l = re.split("([-A-Za-z0-9]+)", s)
  return "".join([x for x in l if not re.search("[-0-9]+",x)])
