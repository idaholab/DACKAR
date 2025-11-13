# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on September, 2025

@author: mandd, wangc
"""

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
import tomllib
from jsonschema import validate, ValidationError
import copy
from pathlib import Path
from datetime import datetime

class KG:
    """
    Class designed to automate and check knowledge graph construction
    """
    def __init__(self, config_file_path, import_folder_path, uri, pwd, user):
        """
        Method designed to initialize the KG class
        @ In, config_file_path, string, DBMS database folder
        @ In, import_folder_path, string, folder which contains data to be imported
        @ In, uri, string, uri = "bolt://localhost:7687" for a single instance or uri = "neo4j://localhost:7687" for a cluster
        @ In, user, string, default to 'neo4j'
        @ In, pwd, string, password the the neo4j DBMS database 
        @ Out, None
        """
        # Change import folder to user specific location
        set_neo4j_import_folder(config_file_path, import_folder_path)

        self.datatypes = ['str', 'int', 'float', 'bool', 'datetime']

        # Create python to neo4j driver
        self.py2neo = Py2Neo(uri=uri, user=user, pwd=pwd)

        self.graphSchemas = {} # dictionary containing the set of schemas of the knowledge graph

        # this is the schema for the set of schemas of the knowledge graph
        self.schemaSchema = {"type": "object",
                             "properties": {"title"   : {"type": "string", "description": "Data object that is target of the schema"},
                                            "version" : {"type": "number", "description": "Development version of the schema"},
                                            "node"    : {"description": "Data element encapsulated in the node",
                                                         "type": "object",
                                                         "properties" : {"node_description": {"type": "string", "description": "Type of relationship encapsulated in the relation between two nodes"},
                                                                         "node_properties": {"type": "array", 
                                                                                             "description": "Allowed properties associate with the node", 
                                                                                             "items": {"type": "object",
                                                                                                       "properties": {"name"    : {"type": "string",  "description": "Name of the node property"},
                                                                                                                      "type"    : {"type": "string",  "description": "Type of the node property"},
                                                                                                                      "optional": {"type": "boolean", "description": "Specifies if this property is required or not"}
                                                                                                                      },
                                                                                                       "required":["name","type","optional"]
                                                                                                      }
                                                                                            }
                                                                        },
                                                         "required":["node_description","node_properties"]           
                                                        },
                                            "relation": {"description": "Data element encapsulated in the edge",
                                                         "type": "object",
                                                         "properties" : {"relation_description": {"type": "string", "description": "Type of relationship encapsulated in the relation between two nodes"},
                                                                         "from_entity": {"type": "string", "description": "Label of the departure node"},
                                                                         "to_entity"  : {"type": "string", "description": "Label of the arrival node"},
                                                                         "relation_properties": {"type": "array",
                                                                                                 "description": "Allowed properties associate with the relation",
                                                                                                 "items": {"type": "object",
                                                                                                           "properties": {"name"    : {"type": "string",  "description": "Name of the relation property"},
                                                                                                                          "type"    : {"type": "string",  "description": "Type of the node property"},
                                                                                                                          "optional": {"type": "boolean", "description": "Specifies if this property is required or not"}
                                                                                                                          },
                                                                                                           "required":["name","type","optional"]
                                                                                                      }
                                                                                            }
                                                                        },
                                                         "required":["relation_description","from_entity","to_entity"]
                                                        }
                                            },
                            "required":["title"]}
        
        # set of predefined schemas available in DACKAR egenrated for the RIAM project
        self.predefinedGraphSchemas = {'conditionReportSchema'  : 'conditionReportSchema.toml',
                                       'customMbseSchema'       : 'customMbseSchema.toml',
                                       'monitoringSystemSchema' : 'monitoringSystemSchema.toml',
                                       'nuclearEntitySchema'    : 'nuclearEntitySchema.toml',
                                       'numericPerfomanceSchema': 'numericPerfomanceSchema.toml'}

    def resetGraph(self):
        """
        Method designed to reset knowledge graph
        @ In, None 
        @ Out, None
        """
        self.py2neo.reset()

    def _crossSchemasCheck(self):
        """
        Method designed to perform a series of checks across the defined schemas
        @ In, None 
        @ Out, None
        """
        self.nodeSet = set()
        self.relSet  = set()
        self.relationList = {}

        for schema in self.graphSchemas:
            for node in self.graphSchemas[schema]['node']:
                self.nodeSet.add(node)
        
        for schema in self.graphSchemas:
            for rel in self.graphSchemas[schema]['relation']:
                origin = self.graphSchemas[schema]['relation'][rel]['from_entity']
                destin = self.graphSchemas[schema]['relation'][rel]['to_entity']

                # check that the defined relations link nodes that have been defined
                if origin not in self.nodeSet:
                    print('Schema ' + str(schema) + ' - Relation ' + str(rel) + ': Node label ' + str(origin) + ' is not defined')
                if destin not in self.nodeSet:
                    print('Schema ' + str(schema) + ' - Relation ' + str(rel) + ': Node label ' + str(destin) + ' is not defined')                

    def _checkSchemaStructure(self, importedSchema):
        """
        Method designed to check importedSchema against self.schemaSchema
        @ In, importedSchema, dict, schema parsed by tomllib from .toml file
        @ Out, None
        """
        try:
            validate(instance=importedSchema, schema=self.schemaSchema)
            print("TOML content is valid against the schema.")
        except tomllib.TOMLDecodeError as e:
            print(f"TOML syntax error: {e}")
        except ValidationError as e:
            print(f"TOML schema validation error: {e.message}")
    
    def importGraphSchema(self, graphSchemaName, tomlFilename):         
        """
        Method that imports new schema contained in a .toml file
        @ In, importedSchema, dict, schema parsed by tomllib from .toml file
        @ In, tomlFilename, string, .toml file contained the new schema
        @ Out, None
        """
        full_path = Path(tomlFilename)
        if not full_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {tomlFilename}")

        with open(full_path, 'rb') as f:
            config_data = tomllib.load(f)
        
        # Check structure of imported graphSchema
        self._checkSchemaStructure(config_data)

        #check data types against self.datatypes
        self._checkSchemaDataTypes(config_data)

        # Check imported graphSchema against self.graphSchemas
        # check schema name is not used before
        if graphSchemaName in list(self.graphSchemas.keys()):
            print('Schema ' + str(graphSchemaName) + ' is already defined in the exisiting schemas')
        
        # check nodes are not already defined
        for node in config_data['node'].keys():
            for schema in self.graphSchemas:
                if node in schema['node'].keys():
                    print('Node ' + str(node) + ' defined in the new schema is already defined in the exisiting schema ' + str(schema))
        
        # check relations are not already defined
        for relation in config_data['relation'].keys():
            for schema in self.graphSchemas:
                if relation in schema['relation'].keys():
                    print('Relation ' + str(node) + ' defined in the new schema is already defined in the exisiting schema ' + str(schema))

        self._crossSchemasCheck()
        
        self.graphSchemas[graphSchemaName] = config_data

    def _checkSchemaDataTypes(self, schema):
        """
        Method that checks that the datatypes defined in the new schema are part of the allowed data 
        types contained in self.datatypes
        @ In, schema, dict, schema parsed by tomllib from .toml file
        @ Out, None
        """
        for node in schema['node']:
            for prop in schema['node'][node]['node_properties']:
                if prop['type'] not in self.datatypes:
                    print('Node ' + str(node) + ' - Property ' + str(prop['name']) + ' data type ' + str(prop['type']) + ' is not allowed')

    def _schemaReturnNodeProperties(self, nodeLabel):
        """
        Method that returns the properties of the node nodeLabel
        @ In, nodeLabel, string, ID of the node label 
        @ Out, propdf, dataframe, dataframe containing nodeLabel properties
        """
        for schema in self.graphSchemas:
            if nodeLabel in schema['node'].keys():
                node_properties = schema['node'][nodeLabel]['node_properties']
                propdf = pd.DataFrame(node_properties)
                return propdf
        print('Node not found')
        return None
    
    def _schemaReturnRelationProperties(self, relation):
        """
        Method that returns the properties of the selected relation
        @ In, relation, string, ID of the node label 
        @ Out, propdf, dataframe, dataframe containing relation properties
        """
        for schema in self.graphSchemas:
            if relation in schema['relation'].keys():
                relation_properties = schema['relation'][relation]['relation_properties']
                propdf = pd.DataFrame(relation_properties)
                return propdf
        print('Relation not found')
        return None

    def _constructionSchemaValidation(self, constructionSchema):
        """
        Method that validate the constructionSchema against defined schemas.
        @ In, constructionSchema, dict, construction schema (see above)
        @ Out, None
        """
        # For each node check that required properties are listed
        for node in constructionSchema['nodes']:
            specified_prop = set(constructionSchema['nodes'][node].keys())
            
            prop_df = self._schemaReturnNodeProperties(node)
            allowed_properties = set(prop_df['name'])
            
            selected_prop_df = prop_df[prop_df['optional']==False]
            req_properties = set(selected_prop_df['name'])

            if not req_properties.issubset(specified_prop):
                print('Node ' + str(node) + 'requires all these properties: ' + str(req_properties))

            if not specified_prop.issubset(allowed_properties):
                print('Node ' + str(node) + 'requires these properties: ' + str(allowed_properties))
        
        # For each relation check that required properties are listed
        for rel in constructionSchema['relations']:
            specified_prop = set(rel['properties'])

            prop_df = self._schemaReturnRelationProperties(rel)
            allowed_properties = set(prop_df['name'])
            
            selected_prop_df = prop_df[prop_df['optional']==False]
            req_properties = set(selected_prop_df['name'])

            if not req_properties.issubset(specified_prop):
                print('Relation ' + str(rel) + 'requires all these properties: ' + str(req_properties))

            if not specified_prop.issubset(allowed_properties):
                print('Relation ' + str(rel) + 'requires these properties: ' + str(allowed_properties))

    def genericWorkflow(self, data, constructionSchema):
        """
        Method designed to importa data into knowledge graph according to constructionSchema
        @ In, data, pd.dataframe, pandas dataframe containing data to be imported in the knowledge graph 
        @ Out, constructionSchema, dict, dataframe containing relation properties. A construction schema is defined as follows:

            constructionSchema = {'nodes'    : nodeConstructionSchema,
                                  'relations': edgeConstructionSchema}

            nodeConstructionSchema = {'nodeLabel1': {'property1': 'dataframe.colA', 'property2': 'dataframe.colB'},
                                      'nodeLabel2': {'property1': 'dataframe.colC'}}
            
            edgeConstructionSchema = [{'source': {'nodeLabel1.property1':'dataframe.col1'},
                                       'target': {'nodeLabel2.property1':'dataframe.col2'},
                                       'type'  : 'edgeType',
                                       'properties': {'property1': 'dataframe.colAlpha', 'property2': 'dataframe.colBeta'}}] 
        """
        # Check constructionSchema against self.graphSchemas  
        self._constructionSchemaValidation(self, constructionSchema)

        # Check datatypes of data
        self._checkDataframeDatatypes(data, constructionSchema)

        # Parse data (pd.dataframe) and update KG
        # Nodes
        if 'nodes' in constructionSchema:
            data_temp = copy.deepcopy(data)
            for node in constructionSchema['nodes'].keys(): 
                mapping = {value: key for key, value in constructionSchema['nodes'][node].items()}
                data_renamed = data_temp.rename(columns=mapping)
                self.py2neo.load_dataframe_for_nodes(df=data_renamed, labels=node, properties=list(mapping.values()))
        
        # Relations
        # --> TODO: check nodes exist
        if 'relations' in constructionSchema:
            data_temp = copy.deepcopy(data)
            for rel in constructionSchema['relations']: 
                source_node_label = next(iter(rel['source'])).split('.')[0]
                source_node_prop  = next(iter(rel['source'])).split('.')[1]

                target_node_label = next(iter(rel['target'])).split('.')[0]
                target_node_prop  = next(iter(rel['target'])).split('.')[1]

                mapping = {}
                data_renamed = data_temp.rename(columns={next(iter(rel['source'].values())):source_node_prop,
                                                         next(iter(rel['target'].values())):target_node_prop})
                
                for prop in rel['properties'].keys():
                    data_renamed = data_renamed.rename(columns={rel['properties'][prop]: prop})
                
                data_renamed[source_node_label] = source_node_label
                data_renamed[target_node_label] = target_node_label
                data_renamed[rel['type']] = rel['type']

                self.py2neo.load_dataframe_for_relations(df=data_renamed, 
                                                         l1=source_node_label, p1=source_node_prop, 
                                                         l2=target_node_label, p2=target_node_prop, 
                                                         lr=rel['type'], 
                                                         pr=list(rel['properties'].keys()))
        
    def _checkDataframeDatatypes(self, data, constructionSchema):
        """
        Method that checks that data elements match format specified in the graph schemas
        @ In, data, pd.dataframe, pandas dataframe containing data to be imported in the knowledge graph 
        @ In, constructionSchema, dict, dataframe containing relation properties
        @ Out, None
        """  
        for node in constructionSchema['nodes']:
            for prop in constructionSchema['nodes']:
                allowedDatatype = self._returnNodePropertyDatatype(node,prop)
                df_datatype = data[constructionSchema['nodes'][node][prop]]
                if allowedDatatype != set(df_datatype.map(type)): 
                    print('Node: ' + str(node) + '- Property: ' + str(prop) + '. Dataframe datatype (' + str(df_datatype) + ') does \\'
                          'not match datatype defined in schema (' + str(df_datatype) + ')')

    def _returnNodePropertyDatatype(self, nodeID, propID):
        """
        Method that returns the allowed type of a specified node property
        @ In, node, string, specific node label 
        @ In, prop, string, specific node property
        @ Out, string, allowed type of the specified node property
        """        
        for schema in self.graphSchemas:
            for node in self.graphSchemas[schema]:
                if node==nodeID and propID in self.graphSchemas[schema][node]:
                    allowedtype = self.graphSchemas[schema][node][propID][type]
                    return allowedtype


    def _returnRelationPropertyDatatype(self, node, prop):
        """
        Method that returns the allowed type of a specified relation property.
        @ In, node, string, specific relation 
        @ In, prop, string, specific node property
        @ Out, string, allowed type of the specified relation property
        """
        pass

    
def stringToDatetimeConverter(date_string, format_code):
    datetime_object = datetime.strptime(date_string, format_code)
    return datetime_object

'''      
# These are workflows specific to the RIAM project
def mbseWorkflow(self, name, type, nodesFile, edgesFile):
    if type =='customMBSE':
        mbseModel = customMBSEobject(nodesFile,
                                        edgesFile, 
                                        path=self.processedDataFolder)
        
        self.equipmentIDs = self.equipmentIDs + mbseModel.returnIDs()
        mbseModel.plot(name)

        label = 'MBSE'
        attribute = {'ID':'ID', 'type':'type'}
        self.py2neo.load_csv_for_nodes(os.path.join(self.processedDataFolder, nodesFile), label, attribute)

        l1='MBSE'
        p1={'ID':'sourceNodeId'}
        l2='MBSE'
        p2 ={'ID':'targetNodeId'}
        lr = 'MBSE_link'
        pr = {'prop':'type'}
        self.py2neo.load_csv_for_relations(os.path.join(self.processedDataFolder, edgesFile), l1, p1, l2, p2, lr, pr)

    elif type =='LML':
        pass

def anomalyWorkflow(self, filename, constructionSchema):
    graphSchema = TBD

    #TODO: Check constructionSchema against graphSchemas

    label = 'anomaly'
    attribute = {'ID':'ID', 'time_initial':'start_date', 'time_final':'end_date'}
    self.py2neo.load_csv_for_nodes(filename, label, attribute)

    l1='anomaly'
    p1={'ID':'ID'}
    l2='monitored_var'
    p2 ={'ID':'monitored_variable'}
    lr = 'detected_by'
    pr = None
    self.py2neo.load_csv_for_relations(filename, l1, p1, l2, p2, lr, pr)

    pass

def monitoringWorkflow(self, filename, constructionSchema):
    graphSchema = TBD

    #TODO: Check constructionSchema against graphSchemas

    label = 'monitored_var'
    attribute = {'ID':'varID'}
    self.py2neo.load_csv_for_nodes(filename, label, attribute)

    l1='monitored_var'
    p1={'ID':'varID'}
    l2='MBSE'
    p2 ={'ID':'equip_ID'}
    lr = 'monitors'
    pr = None
    self.py2neo.load_csv_for_relations(filename, l1, p1, l2, p2, lr, pr)


def eventReportWorkflow(self, filename, constructionSchema, pipelines):
    graphSchema = TBD

    #TODO: Check constructionSchema against graphSchemas

    pass

def kgConstructionWorkflow(self, dataframe, graphSchema, constructionSchema):

    self._schemaValidation(self, constructionSchema, graphSchema)

    for node in constructionSchema['nodes'].keys(): 
        map = {value: key for key, value in constructionSchema['nodes'][node].items()} 
        tempDataframe = dataframe.rename(columns=map)
        self.py2neo.load_dataframe_for_nodes(tempDataframe, node, map.keys())
    
    # Incomplete
    for edge in constructionSchema['edges']:
        self.py2neo.load_dataframe_for_relations(dataframe, 
                                                    l1='sourceLabel', p1='sourceNodeId', 
                                                    l2='targetLabel', p2='targetNodeId', 
                                                    lr=edge['type'], 
                                                    pr=edge['properties'])  '''