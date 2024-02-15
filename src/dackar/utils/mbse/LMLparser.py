'''
Created on May 3, 2021

@author: mandd
'''

# External Imports
import xml.etree.ElementTree as ET
import re
import networkx as nx

class LMLobject(object):
  def __init__(self, filename):
    self.filename = filename
    self.entities = {}
    self.link_entities = []
    self.emb_entities = {}
    self.LMLgraph = None
    self.acronyms = {}

  def OPLparser(self):
    '''
    This method translates all the sentences (see self.sentences) and it create a graph structure (self.LMLgraph)
    '''
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
    '''
    This method returns the networkx graph
    ''' 
    return self.LMLgraph

  def returnEntities(self):
    '''
    This method returns the the list of objects
    '''
    return self.entities, self.emb_entities

  def cleanedGraph(self):
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
  txtPortion = text[text.find("[")+1:text.find("]")]
  listOfElems = txtPortion.split(',')
  return listOfElems





