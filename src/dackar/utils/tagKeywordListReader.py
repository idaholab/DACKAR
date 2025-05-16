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

class ERschema():
  """
  Class designed to codify the equipment reliability (ER) schema and classify content of a clause/sentence
  """

  def __init__(self):
    """
    Initialization method
    Args:

      None

    Returns:

      None
    """
    self.matchDict = {}
    self.matchDict['surv_tool']        = ['surv_tool']
    self.matchDict['inspection']       = ['surv_ops','HS_neut']
    self.matchDict['diagnosis']        = ['diagn']
    self.matchDict['maintenance']      = ['mnt_ops']
    self.matchDict['maint_tool']       = ['mnt_tool']
    self.matchDict['location']         = ['arch']
    self.matchDict['material']         = ['chem_cmpd','chem_elem','mat']
    self.matchDict['reaction']         = ['chem_rx']
    self.matchDict['env_agent']        = ['ext_agent']
    self.matchDict['degradation']      = ['deg_mech']
    self.matchDict['function']         = ['opd_elt','opd_hyd_pne']
    self.matchDict['qual_asmnt']       = ['qual_asmnt']
    self.matchDict['asset[anomalous]'] = ['HS_neg','fail_type_n','fail_type_v']
    self.matchDict['asset[OK]']        = ['HS_pos']
    self.matchDict['asset']            = ['comp_mech_fact',
                                          'comp_mech_rot',
                                          'comp_mech_struct',
                                          'comp_mech_spec',
                                          'comp_elt/n',
                                          'comp_hyd/pne',
                                          'ast_mech',
                                          'ast_elt',
                                          'ast_hyd_pne',
                                          'ast_eln',
                                          'ast_I&C',
                                          'ast_fuel']

    self.invMatchDict = {}
    for key in self.matchDict.keys():
      for elem in self.matchDict[key]:
        self.invMatchDict[elem] = key

  def returnERnature(self, labelList):
    """
    Initialization method
    Args:

      labelList, list, list that contains labels identified in a text

    Returns:

      nature, list, list that contains the corresponding elements in the ER schema for each label contained in labelList
    """
    nature = []
    for elem in labelList:
      nature.append(self.invMatchDict[elem])
    return nature


class entityLibrary():
  """
  Class designed to contain all nuclear related entities listed in nlp/data/tag_keywords_lists.xlsx
  """
  def __init__(self,fileName):
    """
    Initialization method
    Args:

      fileName, string, file containing nuclear related entities

    Returns:

      None
    """
    self.library = self.keyWordListGenerator(fileName)
    self.cleanTagDict()
    self.expander()

  def checker(self):
    """
    Method designed to check the structure of the set of nuclear related entities and identify entities
    that might share multiple labels

    Args:

      None

    Returns:

      None
    """
    for key1 in self.library.keys():
      for key2 in self.library.keys():
        commonElements = list(set(self.library[key1]).intersection(self.library[key2]))
        if key1!=key2 and commonElements:
          print('Elements in common between ' +str(key1)+ ' and ' +str(key2)+ ' are:' + str(commonElements))

  def getLibrary(self):
    """
    Method designed to return self.library

    Args:

      None

    Returns:

      self.library, dict, dictionary containing for each label a list of entities
    """
    return self.library

  def getAcronymsDict(self):
    """
    Method designed to return self.acronymsDict

    Args:

      None

    Returns:

      self.acronymsDict, dict, dictionary containing the acronyms contained in the library
    """
    return self.acronymsDict

  def expander(self):
    """
    Method designed to treat those entities that are compounds of two words which are identified as word1-word2.
    These compond words can be written in multiple forms: "word1-word2", "word1word2", "word1 word2".
    Here, these forms are generated for each identified compund word (when '-' is identified in the entity)

    Args:

      None

    Returns:

      None
    """
    for key in self.library.keys():
      for elem in self.library[key]:
        if '-' in elem:
          self.library[key].append(elem.replace('-',''))
          self.library[key].append(elem.replace('-',' '))


  def keyWordListGenerator(self, fileName):
    """
    Method designed to read the file and generate a dictionary which contains, for each tag,
    the set of keywords that should be associate to such tag.

    Args:

      fileName, string, file containing nuclear related entities

    Returns:

      tagsDict, dict, dictionary containing for each label a list of entities
    """

    df = pd.read_excel(fileName, None)
    # retrieve list of sheets in excel file
    sheetList = df.keys()

    tagsDict = {}
    for sheet in sheetList:
      # retrieve columns of each sheet
      cols = df[sheet].keys()
      for col in cols:
        # retrieve TAG of each column; it should be contained in square brackets [tag]
        first = col.find("[")
        second = col.find("]")
        tagID = col[first+1:second]

        if tagID not in ['prop','unit']:
          keywordsList = df[sheet][col].dropna().values.tolist()
          keywordsList = [[i] for i in keywordsList if i]
          for index,keyword in enumerate(keywordsList):
            if ',' in keyword[0]:
              keywordsList[index] = keyword[0].split(',')
          tagsDict[tagID] = list(itertools.chain(*keywordsList))
    return tagsDict


  def cleanTagDict(self):
    """
    Method designed to clean the dictionary generated by the method keyWordListGenerator(.)
    Here, specific characters or sub strings are removed.
    In addition, if an acronym is defined (within round parentheses), then the acronyms_dict is
    populated {acronym: acronym_definition}

    Args:

      None

    Returns:

      None
    """

    self.acronymsDict = {}
    n_keywords = 0
    for tag in self.library:
      for index,elem in enumerate(self.library[tag]):
        # clean string
        cleanElem = elem.lower()
        cleanElem = cleanElem.strip().lstrip()
        cleanElem = cleanElem.replace("\xa0", " ")
        cleanElem = cleanElem.replace("\n", " ")

        # retrieve acronym if defined
        first = cleanElem.find("(")
        second = cleanElem.find(")")
        if (first==-1 and second>=0) or (second==-1 and first>=0):
          print('Error of acronym definition')
        if (first>=0 and second>=0):
          acronym = cleanElem[first + 1:second].strip().lstrip()
          toReplace = "(" + acronym + ")"
          cleanElem = cleanElem.replace(toReplace,'')
          cleanElem = " ".join(cleanElem.split())
          # save acronym into its own dictionary
          self.acronymsDict[acronym] = cleanElem.strip().lstrip()
          # remove acronym from tags_dict
          toReplace = "(" + acronym + ")"
          cleanElem = cleanElem.replace(toReplace,'')
        else:
          cleanElem = cleanElem

        self.library[tag][index] = " ".join(cleanElem.split()) # clean_elem
      self.library[tag] = [i for i in self.library[tag] if i]

    for tag in self.library:
      nKeywords = n_keywords + len(self.library[tag])
    print("Number of listed keywords: " + str(nKeywords))




