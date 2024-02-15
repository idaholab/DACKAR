# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on February, 2024

@author: mandd
"""

# External Imports
import xml.etree.ElementTree as ET
import re
import networkx as nx

class LMLobject(object):
  """
    Class designed to process the MBSE model developed in Lifecycle Modeling Language (LML) using Innoslate.
  """
  def __init__(self, filename):
    """
    Initialization method for the LMLobject class

    Args:

      filename: file, file in xml containing the MBSE model deveolped in LML using Innoslate

    Returns:

      None
    """
    self.filename = filename
    self.entities = {}
    self.link_entities = []
    self.emb_entities = {}
    self.LMLgraph = None
    self.acronyms = {}

  def OPLparser(self):
    """
    This method is designed to parse the xml file containing the MBSE model, to create its corresponding graph
    and to populate:
    - the set of entities: dictionary of assets in the form of 'LML_ID': ('asset name', 'asset ID')
    - the set of links: list containing the LML-IDs of all links between assets
    - the set of embedded entities: dictionary of the components that have been specified in the description text of the 
                                    LML asset or link (e.g., [comp1,comp2,comp3]) in the form of 'LML_ID': [comp1,comp2,comp3]
    """
    self.LMLgraph = nx.MultiDiGraph()

    tree = ET.parse(self.filename)
    root = tree.getroot()

    self.DBnode = root.find('database')

    #  Identify entities and links
    for child in self.DBnode:
      if child.tag=='entity':
        nametxt = child.find('name').text
        if ' to ' in nametxt:
          self.parseLinkEntity(child)
        else:
          self.parseAssetEntity(child)

    self.connetGraph()

  def connetGraph(self):
    """
    This method is designed to actually link the asset entities identified in the OPLparser method
    """
    for child in self.DBnode:
      if child.tag=='relationship':
        sourceID  = child.find('sourceId').text
        targetID  = child.find('targetId').text
        attrNode  = child.find('booleanAttribute')
        if attrNode is not None:
          attribute = attrNode.find('booleanValue').text

          if sourceID in self.link_entities and targetID in self.entities.keys():
            if attribute=='true':
              self.LMLgraph.add_edge(self.entities[targetID], sourceID, color='k', key='link')
            elif attribute=='false':
              self.LMLgraph.add_edge(sourceID, self.entities[targetID], color='k', key='link')
            else:
              print('---error----')


  def parseLinkEntity(self, linkNode):
    """
    This method extracts all required information of the link from the provided xml node.
    It populates the self.link_entities and the self.emb_entities variables.

    Args:

      linkNode: xml node, xml node containing containing all the information of a single link generated 
                          in LML using Innoslate

    Returns:

      None
    """
    linkID = linkNode.find('globalId').text
    self.link_entities.append(linkID)
    self.LMLgraph.add_node(linkID, color='b', key='entity')

    # Parse description
    if linkNode.find('description').text:
      entityDescr = linkNode.find('description').text
      elemList = parseEntityDescription(entityDescr)
      self.emb_entities[linkID] = elemList
    
      for elem in elemList:
        self.LMLgraph.add_node(elem, color='m', key='entity_emb')
      for elem in elemList:
        self.LMLgraph.add_edge(linkID, elem, color='k', key='assoc')


  def parseAssetEntity(self, entityNode):
    """
    This method extracts all required information of the asset from the provided xml node.
    It populates the self.entities and the self.emb_entities variables.

    Args:

      linkNode: xml node, xml node containing containing all the information of a single link generated 
                          in LML using Innoslate

    Returns:

      None
    """
    assetID     = entityNode.find('number').text
    entityID    = entityNode.find('globalId').text
    entityName  = entityNode.find('name').text.strip()
    cleanedName = re.sub(' +', ' ', entityName)

    # Parse description
    if entityNode.find('description').text:
      entityDescr = entityNode.find('description').text
      elemList = parseEntityDescription(entityDescr)
      self.emb_entities[entityID] = elemList

      for elem in elemList:
        self.LMLgraph.add_node(elem, color='r', key='entity_emb')
    
    if assetID:
      self.entities[entityID] = (entityName,assetID)
      self.LMLgraph.add_node((entityName,assetID), color='m', key='entity')
      if entityNode.find('description').text:
        for elem in elemList:
          self.LMLgraph.add_edge((entityName,assetID), elem, color='k', key='assoc')
    else:
      self.entities[entityID] = (entityName,None)
      self.LMLgraph.add_node((entityName,None), color='m', key='entity')
      if entityNode.find('description').text:
        for elem in elemList:
          self.LMLgraph.add_edge((entityName,None), elem, color='k', key='assoc')

  def returnGraph(self):
    """
    This method returns the networkx graph

    Args:

      None

    Returns:

      self.LMLgraph: networkx object, graph containing entities specified in the LML MBSE model
    """ 
    return self.LMLgraph

  def returnEntities(self):
    """
    This method returns the the dictionaries of entities and embedded entities specified in the MBSE model

    Args:

      None

    Returns:

      self.entities     : dict, dict of entities
      self.emb_entities : dict, dict of embedded entities
    """
    return self.entities, self.emb_entities

  def cleanedGraph(self):
    """
    This method is designed to clean the complete MBSE graph by removing the links which are represented as nodes

    Args:

      None

    Returns:

      g: networkx object, cleaned graph containing only asset entities specified in the LML MBSE model
    """
    g = self.LMLgraph.copy()

    for node,degree in g.degree():
      if node in self.link_entities:
        a0,b0 = list(g.in_edges(node))[0]
        a1,b1 = list(g.out_edges(node))[0]

        e0 = a0 if a0!=node else b0
        e1 = a1 if a1!=node else b1

        g.add_edge(e0, e1)

    g.remove_nodes_from(self.link_entities)
    
    return g


def parseEntityDescription(text):
    """
    This method is designed to extract the elements specified in square brackets that are specified in 
    the description node of the MBSE model of a link or entity

    Args:

      text: str, text contained in the description node of the MBSE model

    Returns:

      listOfElems     : list, list of elements specified in square brackets and separated by commas (i.e., ',')
    """
  txtPortion = text[text.find("[")+1:text.find("]")]
  listOfElems = txtPortion.split(',')
  return listOfElems





