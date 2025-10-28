# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

import os, sys
import re

cwd = os.getcwd()
frameworkDir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, 'src'))

# Internal Modules #
from dackar.knowledge_graph.py2neo import Py2Neo
from dackar.knowledge_graph.graph_utils import set_neo4j_import_folder
from dackar.utils.mbse.customMBSEparser import customMBSEobject

# External Modules #
import pandas as pd
import os, sys

class KG:
    def __init__(self, config_file_path, import_folder_path, uri, pwd, processedDataFolder):
        # Change import folder to user specific location
        set_neo4j_import_folder(config_file_path, import_folder_path)
        
        self.processedDataFolder = processedDataFolder

        # Create python to neo4j driver
        self.py2neo = Py2Neo(uri=uri, user='neo4j', pwd=pwd)

        self.graphSchemas = []

    def graphCheck(self):
        pass

    def mbseWorkflow(self, name, type, nodesFile, edgesFile):
        if type =='customMBSE':
            mbseModel = customMBSEobject(nodesFile,
                                         edgesFile, 
                                         path=self.processedDataFolder)
            
            mbseModel.printEquipmentID()
            mbseModel.plot(name)
    
        elif type =='LML':
            pass
    
    def anomalyWorkflow(self, filename, constructionSchema):
        graphSchemas = BBD

    # Congjian: how to add relations across schemas????
    # use Json/toml for validation

    nodeConstructionSchema = {'nodeLabel1': {'property1': 'node.colA', 'property2': 'node.colB'},
                              'nodeLabel2': {'property1': 'node.colC'}}
    
    edgeConstructionSchema = [{'source': ('nodeLabel1.property1','col1'),
                               'target': ('nodeLabel2.property1','col2'),
                               'type': 'edgeType',
                               'properties': {'property1': 'colAlpha', 'property2': 'colBeta'}}]
    
    constructionSchema = {'nodes': nodeConstructionSchema,
                          'edges': edgeConstructionSchema}

    def kgConstructionWorkflow(self, dataframe, graphSchemas, constructionSchema):
        #TODO: Check constructionSchema against graphSchemas

        for node in constructionSchema['nodes'].keys(): 
            map = {value: key for key, value in constructionSchema['nodes'][node].items()} 
            tempDataframe = dataframe.rename(columns=map)
            self.py2neo.load_dataframe_for_nodes(tempDataframe, node, map.keys())
        
        # Incomplete
        for edge in constructionSchema['edges']:
            self.py2neo.load_dataframe_for_relations(dataframe, 
                                                     l1='sourceLabel', p1='sourceNodeId', 
                                                     l2='targetLabel', p2='targetNodeId', 
                                                     lr='relationshipType', 
                                                     pr=None)
        
'''       
# Load nodes
label = 'anomaly'
self.py2neo.load_csv_for_nodes(file_path, label, nodeAttributes)

# Load edges
l1='anomaly'
p1={'ID':'ID'}
l2='monitored_var'
p2 ={'ID':'monitored_variable'}
lr = 'detected_by'
pr = None
self.py2neo.load_csv_for_relations(file_path, l1, p1, l2, p2, lr, pr)
''' 

'''file_path = 'processed_data/mbse_model_nodes_kg.csv'
label = 'MBSE'
attribute = {'ID':'ID', 'type':'type'}
self.py2neo.load_csv_for_nodes(file_path, label, attribute)

file_path = 'processed_data/mbse_model_edges_kg.csv'
l1='MBSE'
p1={'ID':'sourceNodeId'}
l2='MBSE'
p2 ={'ID':'targetNodeId'}
lr = 'MBSE_link'
pr = {'prop':'type'}
self.py2neo.load_csv_for_relations(file_path, l1, p1, l2, p2, lr, pr)'''

