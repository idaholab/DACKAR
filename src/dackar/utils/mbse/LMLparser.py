# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on February, 2024

@author: mandd
"""

# External Imports
import xml.etree.ElementTree as ET
import re
import networkx as nx
import pandas as pd
import graphdatascience as gds

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
    self.linkEntities = []
    self.embEntities = {}
    self.LMLgraph = None
    self.acronyms = {}
    self.listIDs = []
    self.linkToMBSEmodels = {}

  def LMLparser(self, diagramName):
    """
      This method is designed to parse the xml file containing the MBSE model, to create its corresponding graph
      and to populate:
      - the set of entities: dictionary of assets in the form of 'LML_ID': ('asset name', 'asset ID')
      - the set of links: list containing the LML-IDs of all links between assets
      - the set of embedded entities: dictionary of the components that have been specified in the description text of the
                                      LML asset or link (e.g., [comp1,comp2,comp3]) in the form of 'LML_ID': [comp1,comp2,comp3]

      Args:

        diagramName: string, original name of the diagram; it is used to remove the correposing node in the graph
    """
    self.diagramName = diagramName

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

          if sourceID in self.linkEntities and targetID in self.entities.keys():
            if attribute=='true':
              self.LMLgraph.add_edge(self.entities[targetID], sourceID, color='k', key='link')
            elif attribute=='false':
              self.LMLgraph.add_edge(sourceID, self.entities[targetID], color='k', key='link')
            else:
               raise IOError('LMLobject object: booleanValue for edge connecting "{}" and "{}" is not defined'.format(sourceID,self.entities[targetID]))


  def parseLinkEntity(self, linkNode):
    """
      This method extracts all required information of the link from the provided xml node.
      It populates the self.linkEntities and the self.embEntities variables.

      Args:

        linkNode: xml node, xml node containing containing all the information of a single link generated
                            in LML using Innoslate

      Returns:

        None
    """
    linkID = linkNode.find('globalId').text
    self.linkEntities.append(linkID)
    self.LMLgraph.add_node(linkID, color='b', key='entity')

    # Parse description
    if linkNode.find('description').text:
      entityDescr = linkNode.find('description').text
      elemList = parseEntityDescription(entityDescr)[0]    # No link to MBSE models allowed for links
      self.embEntities[linkID] = elemList
      self.listIDs = self.listIDs + elemList

      for elem in elemList:
        self.LMLgraph.add_node(elem, color='m', key='entity_emb')
      for elem in elemList:
        self.LMLgraph.add_edge(linkID, elem, color='k', key='assoc')


  def parseAssetEntity(self, entityNode):
    """
      This method extracts all required information of the asset from the provided xml node.
      It populates the self.entities and the self.embEntities variables.

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

    elemList = None
    MBSElink = None

    if entityName == self.diagramName: return

    # Parse description
    entityDescr = entityNode.find('description').text
    if entityDescr:
      (elemList,MBSElink) = parseEntityDescription(entityDescr)

      if elemList:
        self.embEntities[entityID] = elemList
        self.listIDs = self.listIDs + elemList
        for elem in elemList:
          self.LMLgraph.add_node(elem, color='r', key='entity_emb')
      if MBSElink:
        self.LMLgraph.add_node(MBSElink, color='g', key='MBSE_linked_ent')

    if assetID:
      self.entities[entityID] = (entityName,assetID)
      self.listIDs.append(assetID)
      self.LMLgraph.add_node((entityName,assetID), color='m', key='entity')
      if elemList:
        for elem in elemList:
          self.LMLgraph.add_edge((entityName,assetID), elem, color='k', key='assoc')
      if MBSElink:
        self.LMLgraph.add_edge((entityName,assetID), MBSElink, color='g', key='MBSElink')
    else:
      self.entities[entityID] = (entityName,None)
      self.LMLgraph.add_node((entityName,None), color='m', key='entity')
      if elemList:
        for elem in elemList:
          self.LMLgraph.add_edge((entityName,None), elem, color='k', key='assoc')
      if MBSElink:
        self.LMLgraph.add_edge((entityName,None), MBSElink, color='g', key='MBSElink')

  def returnGraph(self):
    """
      This method returns the networkx graph

      Args:

        None

      Returns:

        self.LMLgraph: networkx object, graph containing entities specified in the LML model
    """
    return self.LMLgraph

  def returnEntities(self):
    """
      This method returns the the dictionaries of entities and embedded entities specified in the MBSE model

      Args:

        None

      Returns:

        self.entities     : dict, dict of entities
        self.embEntities : dict, dict of embedded entities
    """
    return self.entities, self.embEntities

  def returnListIDs(self):
    """
      This method returns the list of asset IDs

      Args:

        None

      Returns:

        self.listIDs: list, list of asset IDs specified in the LML MBSE model
    """
    rmv = [str(item) for item in range(1, 40)]
    cleaned = set(self.listIDs).difference(set(rmv))
    return list(cleaned)

  def cleanedGraph(self):
    """
      This method is designed to clean the complete MBSE graph by removing the links which are represented as nodes

      Args:

        None

      Returns:

        g: networkx object, cleaned graph containing only asset entities specified in the LML MBSE model
    """
    self.cleanedGraph = self.LMLgraph.copy()

    for node,degree in self.cleanedGraph.degree():
      if node in self.linkEntities:
        a0,b0 = list(self.cleanedGraph.in_edges(node))[0]
        a1,b1 = list(self.cleanedGraph.out_edges(node))[0]

        e0 = a0 if a0!=node else b0
        e1 = a1 if a1!=node else b1

        self.cleanedGraph.add_edge(e0, e1)

    self.cleanedGraph.remove_nodes_from(self.linkEntities)

    return self.cleanedGraph

  def printOnFile(self, name, csv=True):
    """
      This method is designed to print on file the graph from networkx.
      This is to test a method to import a graph into neo4j as indicated in:
      https://stackoverflow.com/questions/52210619/how-to-import-a-networkx-graph-to-neo4j
      Args:

        None

      Returns:

        None
    """
    if csv:
      name = name + ".csv"
      nx.write_edgelist(self.LMLgraph, name, delimiter=',', data=True, encoding='utf-8')
    else:
      name = name + ".graphml"
      nx.write_graphml(self.LMLgraph, name)


  def dumpDGSgraph(self, name):
    """
      This method is designed to save the graph structure into gds entity
      See Example 3.2 in https://neo4j.com/docs/graph-data-science-client/current/graph-object/
      Args:

        None

      Returns:

        None
    """
    NXnodes = list(self.LMLgraph.nodes(data=True))
    NXedges = list(self.LMLgraph.edges)

    mapping = {}

    nodes = {
            "nodeId": [],
            "labels": [],
            "ID": [],
            "type": []
            }

    for index,node in enumerate(NXnodes):
      nodes['nodeId'].append(index)
      nodeInfo = node

      mapping[index] = node[0]

      if nodeInfo[0] is None:
        nodes['labels'].append(nodeInfo[1])
        nodes['ID'].append(nodeInfo[1])

      elif nodeInfo[1] is None:
        nodes['labels'].append(nodeInfo[0])
        nodes['ID'].append(nodeInfo[0])

      else:
        nodes['labels'].append(nodeInfo[0])
        nodes['ID'].append(nodeInfo[1])

      nodes['type'].append(node[1]['key'])

    nodes = pd.DataFrame(nodes)

    relationships = {
                    "sourceNodeId": [],
                    "targetNodeId": [],
                    "type"        : []
                    }

    for index,edge in enumerate(NXedges):
      father = [key for key, val in mapping.items() if val == edge[0]][0]
      child  = [key for key, val in mapping.items() if val == edge[1]][0]
      relationships['sourceNodeId'].append(father) 
      relationships['targetNodeId'].append(child) 
      relationships['type'].append(edge[2])

    relationships = pd.DataFrame(relationships)

    nodes.to_csv(name+'_nodes.csv')
    relationships.to_csv(name+'_edges.csv')


def parseEntityDescription(text):
  """
    This method is designed to extract the elements specified in square brackets that are specified in
    the description node of the MBSE model of a link or entity

    Args:

      text: str, text contained in the description node of the MBSE model

    Returns:

      out: tuple, tuple containing the list of elements specified in square brackets and separated
           by commas (e.g., ['FV304,'305']) and the link to an external MBSE model
           (e.g., ('centrifugalPumpFull', 'body'))

  """

  if '[' in text:
    txtPortion1 = text[text.find("[")+1:text.find("]")]
    listOfElems = txtPortion1.split(';')
  else:
    listOfElems = None

  if '{' in text:
    MBSElink = text[text.find("{")+1:text.find("}")].split(':')
    MBSEinstance = (MBSElink[0],MBSElink[1])
  else:
    MBSEinstance = None

  out = (listOfElems,MBSEinstance)

  return out





