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
  # [-A-Za-z0-9#]+ pattern for any combinations "-", "A-Z", "a-z", "0-9", "#" and "/"
  l = re.split("([-A-Za-z0-9#/%]+)", s)
  # only remove strings that contain "-" or numbers
  return "".join([x for x in l if not re.search("[-0-9]+",x)])

def getShortAcronym(s):
  """
    Remove things like h/s, s/d, etc.

    Args:
      s: string

    Returns:
      string with only the words
  """
  l = re.split(r"(\b[A-Za-z]/[A-Za-z])(?=\s)", s)
  acronym = [x for x in l if re.search("[/]", x) and len(x)==3]
  ns = "".join([x for x in l if x not in acronym])
  return ns, acronym

