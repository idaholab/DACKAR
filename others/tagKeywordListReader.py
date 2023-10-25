# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

import pandas as pd
import numpy as np
import itertools

'''
  These methods are designed to process the file tag_keywords_lists.xlsx
  This file contains several keywords grouped into classes (e.g., material, components, etc)
  For each class, a tag is defined: during the NLP analysis, when these keywords are identified,
  this tag will be associated to such word.
  These methods are designed to run in sequence just once at the beginning of the NLP
  pipeline), e.g.:
    tagsDict = keyWordListGenerator('../../nlp/data/tag_keywords_lists.xlsx')
    unitDict = extractUnits('../../nlp/data/tag_keywords_lists.xlsx')
    tagsDictCleaned, acronymsDict = cleanTagDict(tagsDict)
'''

def keyWordListGenerator(fileName):
  '''
  Method designed to read the file and generate a dictionary which contains, for each tag,
  the set of keywords that should be associate to such tag.
  '''
  # TODO: ==> check if lower, and lemma (e.g. for plural)!!!
  # TODO: subsets of words
  # read excel file .xlsx
  df = pd.read_excel(fileName, None)
  # retrieve list of sheets in excel file
  sheet_list = df.keys()

  tagsDict = {}
  for sheet in sheet_list:
    # retrieve columns of each sheet
    cols = df[sheet].keys()
    for col in cols:
      # retrieve TAG of each column; it should be contained in square brackets [tag]
      first = col.find("[")
      second = col.find("]")
      tagID = col[first+1:second]
      keywordsList = df[sheet][col].dropna().values.tolist()
      keywordsList = [[i] for i in keywordsList if i]
      for index,keyword in enumerate(keywordsList):
        if ',' in keyword[0]:
          keywordsList[index] = keyword[0].split(',')
      tagsDict[tagID] = list(itertools.chain(*keywordsList))
  return tagsDict

def extractUnits(fileName):
  '''
  Method designed to extract measure units from provided file.
  It returns a dictionary which contains, for each quantity, a list of common used units, e.g.,
    {'Pressure': ['pa', ' torr', ' barr', ' atm', ' psi']}
  '''
  measuresDict = {}
  df = pd.read_excel(fileName, None)
  measures = df['operands'][['Properties [prop]','units [unit]']]
  for index,elem in measures.iterrows():
    if not pd.isnull(elem['units [unit]']):
      measuresDict[elem['Properties [prop]']] = elem['units [unit]'].replace(" ", "").split(',')
  return measuresDict

def cleanTagDict(tagsDict):
  '''
  Method designed to clean the dictionary generated by the method keyWordListGenerator(.)
  Here, specific characters or sub strings are removed.
  In addition, if an acronym is defined (within round parentheses), then the acronyms_dict is
  populated {acronym: acronym_definition}
  '''
  acronymsDict = {}
  n_keywords = 0
  for tag in tagsDict:
    for index,elem in enumerate(tagsDict[tag]):
      # clean string
      cleanElem = elem.lower()
      cleanElem = cleanElem.strip().lstrip()
      cleanElem = cleanElem.replace("\xa0", " ")
      cleanElem = cleanElem.replace("\n", " ")
      # Note that here we are removing the hyphen
      cleanElem = cleanElem.replace("-", " ")

      # retrieve acronym if defined
      first = cleanElem.find("(")
      second = cleanElem.find(")")
      if (first==-1 and second>=0) or (second==-1 and first>=0):
        print('Error of acronym definition')
      if (first>=0 and second>=0):
        acronym = cleanElem[first + 1:second].strip().lstrip()
        to_replace = "(" + acronym + ")"
        cleanElem = cleanElem.replace(to_replace,'')
        cleanElem = " ".join(cleanElem.split())
        # save acronym into its own dictionary
        acronymsDict[acronym] = cleanElem.strip().lstrip()
        # remove acronym from tags_dict
        to_replace = "(" + acronym + ")"
        cleanElem = cleanElem.replace(to_replace,'')
      else:
        cleanElem = cleanElem

      tagsDict[tag][index] = " ".join(cleanElem.split()) # clean_elem
    tagsDict[tag] = [i for i in tagsDict[tag] if i]

  for tag in tagsDict:
    n_keywords = n_keywords + len(tagsDict[tag])
  print("Number of listed keywords: " + str(n_keywords))
  tagsDictChecker(tagsDict)
  return tagsDict, acronymsDict

def tagsDictChecker(tagsDict):
  for key1 in tagsDict.keys():
    for key2 in tagsDict.keys():
      commonElements = list(set(tagsDict[key1]).intersection(tagsDict[key2]))
      if key1!=key2 and commonElements:
        print('Elements in common between ' +str(key1)+ ' and ' +str(key2)+ ' are:' + str(commonElements))

