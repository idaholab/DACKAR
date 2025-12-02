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
from dackar.utils.tagKeywordListReader import entityLibrary

# External Modules #
import pandas as pd
import os, sys
import tomllib
from jsonschema import validate, ValidationError
import copy
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse
from pandas.api.types import infer_dtype

import logging

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
        if import_folder_path is not None:
            set_neo4j_import_folder(config_file_path, import_folder_path)

        self.datatypes = ['string', 'integer', 'floating', 'boolean', 'datetime']

        # Create python to neo4j driver
        self.py2neo = Py2Neo(uri=uri, user=user, pwd=pwd)

        self.graphSchemas = {} # dictionary containing the set of schemas of the knowledge graph

        self.graphMetadata = {} # Metadata container of the knowledge graph

        self.entityLibrary = entityLibrary('../../../data/tag_keywords_lists.xlsx')

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
                                                                                                                      "type"    : {"type": "string",  "description": "Type of the node property", "enum": self.datatypes},
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
                    logging.error('Schema ' + str(schema) + ' - Relation ' + str(rel) + ': Node label ' + str(origin) + ' is not defined')
                if destin not in self.nodeSet:
                    logging.error('Schema ' + str(schema) + ' - Relation ' + str(rel) + ': Node label ' + str(destin) + ' is not defined')

    def _checkSchemaStructure(self, importedSchema):
        """
        Method designed to check importedSchema against self.schemaSchema
        @ In, importedSchema, dict, schema parsed by tomllib from .toml file
        @ Out, None
        """
        try:
            validate(instance=importedSchema, schema=self.schemaSchema)
            logging.info("TOML content is valid against the schema.")
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
            logging.error('Schema ' + str(graphSchemaName) + ' is already defined in the exisiting schemas')

        # check nodes are not already defined
        for node in config_data['node'].keys():
            for schema in self.graphSchemas:
                if node in schema['node'].keys():
                    logging.error('Node ' + str(node) + ' defined in the new schema is already defined in the exisiting schema ' + str(schema))

        # check relations are not already defined
        for relation in config_data['relation'].keys():
            for schema in self.graphSchemas:
                if relation in schema['relation'].keys():
                    logging.error('Relation ' + str(node) + ' defined in the new schema is already defined in the exisiting schema ' + str(schema))

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
                    logging.error('Node ' + str(node) + ' - Property ' + str(prop['name']) + ' data type ' + str(prop['type']) + ' is not allowed')

    def _schemaReturnNodeProperties(self, nodeLabel):
        """
        Method that returns the properties of the node nodeLabel
        @ In, nodeLabel, string, ID of the node label
        @ Out, propdf, dataframe, dataframe containing nodeLabel properties
        """
        for schema in self.graphSchemas:
            if nodeLabel in self.graphSchemas[schema]['node'].keys():
                node_properties = self.graphSchemas[schema]['node'][nodeLabel]['node_properties']
                propdf = pd.DataFrame(node_properties)
                return propdf
        return None

    def _schemaReturnRelationProperties(self, relation):
        """
        Method that returns the properties of the selected relation
        @ In, relation, string, ID of the node label
        @ Out, propdf, dataframe, dataframe containing relation properties
        """
        for schema in self.graphSchemas:
            if relation in self.graphSchemas[schema]['relation']:
                relation_properties = self.graphSchemas[schema]['relation'][relation]['relation_properties']
                propdf = pd.DataFrame(relation_properties)
                return propdf
        return None

    def _constructionSchemaStructureValidation(self, constructionSchema):
        """
        Method that validates the structure of constructionSchema
        @ In, constructionSchema, dict, construction schema
        @ Out, None
        """
        for key in constructionSchema.keys():
            if key=='nodes':
                if isinstance(constructionSchema[key], dict):
                    for kkey in constructionSchema[key].keys():
                        if not isinstance(constructionSchema['nodes'][kkey], dict):
                            logging.error('Key ' + str(kkey) + 'in the construction schema should be a dictionary')
                else:
                    logging.error('Key ' + str(key) + 'in the construction schema should be a dictionary')
            elif key=='relations':
                if isinstance(constructionSchema[key], dict):
                    for kkey in constructionSchema[key].keys():
                        if not isinstance(constructionSchema['relations'][kkey], dict):
                            logging.error('Key ' + str(kkey) + 'in the construction schema should be a dictionary')
                        if list(constructionSchema['relations'][kkey].keys())!=['source','target','properties']:
                            logging.error('Relation ' + str(kkey) + ' needs to contain these keys: source, target, properties')
                else:
                    logging.error('Key ' + str(key) + 'in the construction schema should be a list')
            else:
                logging.error('Key ' + str(key) + 'in the construction schema is not allowed (allowed: nodes, relations)')

    def _constructionSchemaValidation(self, constructionSchema):
        """
        Method that validates the constructionSchema against defined schemas
        @ In, constructionSchema, dict, construction schema
        @ Out, None
        """
        # For each node check that required properties are listed
        if 'nodes' in constructionSchema:
            for node in constructionSchema['nodes']:
                specified_prop = set(constructionSchema['nodes'][node].keys())

                prop_df = self._schemaReturnNodeProperties(node)
                allowed_properties = set(prop_df['name'])

                selected_prop_df = prop_df[prop_df['optional']==False]
                req_properties = set(selected_prop_df['name'])

                if not req_properties.issubset(specified_prop):
                    logging.error('Node ' + str(node) + 'requires all these properties: ' + str(req_properties))

                if not specified_prop.issubset(allowed_properties):
                    logging.error('Node ' + str(node) + 'requires these properties: ' + str(allowed_properties))

        # For each relation check that required properties are listed
        if 'relations' in constructionSchema:
            for rel in constructionSchema['relations']:
                specified_prop = set(constructionSchema['relations'][rel]['properties'])

                prop_df = self._schemaReturnRelationProperties(rel)
                allowed_properties = set(prop_df['name'])

                selected_prop_df = prop_df[prop_df['optional']==False]
                req_properties = set(selected_prop_df['name'])

                if not req_properties.issubset(specified_prop):
                    logging.error('Relation ' + str(rel) + 'requires all these properties: ' + str(req_properties))

                if not specified_prop.issubset(allowed_properties):
                    logging.error('Relation ' + str(rel) + 'requires these properties: ' + str(allowed_properties))

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
        # Check structure of constructionSchema
        self._constructionSchemaStructureValidation(constructionSchema)

        # Check constructionSchema against self.graphSchemas
        self._constructionSchemaValidation(constructionSchema)

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
                source_node_label = next(iter(constructionSchema['relations'][rel]['source'])).split('.')[0]
                source_node_prop  = next(iter(constructionSchema['relations'][rel]['source'])).split('.')[1]

                target_node_label = next(iter(constructionSchema['relations'][rel]['target'])).split('.')[0]
                target_node_prop  = next(iter(constructionSchema['relations'][rel]['target'])).split('.')[1]

                mapping = {}
                data_renamed = data_temp.rename(columns={next(iter(constructionSchema['relations'][rel]['source'].values())):source_node_prop,
                                                         next(iter(constructionSchema['relations'][rel]['target'].values())):target_node_prop})

                for prop in constructionSchema['relations'][rel]['properties'].keys():
                    data_renamed = data_renamed.rename(columns={constructionSchema['relations'][rel]['properties'][prop]: prop})

                data_renamed[source_node_label] = source_node_label
                data_renamed[target_node_label] = target_node_label
                data_renamed[rel] = rel

                self.py2neo.load_dataframe_for_relations(df=data_renamed,
                                                         l1=source_node_label, p1=source_node_prop,
                                                         l2=target_node_label, p2=target_node_prop,
                                                         lr=rel,
                                                         pr=list(constructionSchema['relations'][rel]['properties'].keys()))

    def _checkDataframeDatatypes(self, data, constructionSchema):
        """
        Method that checks that data elements in data match format specified in the graph schemas
        @ In, data, pd.dataframe, pandas dataframe containing data to be imported in the knowledge graph
        @ In, constructionSchema, dict, dataframe containing relation properties
        @ Out, None
        """
        # Check nodes data types
        if 'nodes' in constructionSchema:
            for node in constructionSchema['nodes']:
                for prop in constructionSchema['nodes'][node]:
                    allowedDatatype = self._returnNodePropertyDatatype(node,prop)
                    df_datatype = data[constructionSchema['nodes'][node][prop]]
                    if allowedDatatype != infer_dtype(df_datatype):
                        logging.error('Node: ' + str(node) + '- Property: ' + str(prop) + '. Dataframe datatype (' + str(set(df_datatype.map(type))) + ') does \\'
                                      'not match datatype defined in schema (' + str(allowedDatatype) + ')')

        # Check relations data types
        if 'relations' in constructionSchema:
            for rel in constructionSchema['relations']:
                for prop in constructionSchema['relations'][rel]['properties']:
                    allowedDatatype = self._returnRelationPropertyDatatype(rel,prop)
                    df_datatype = data[constructionSchema['relations'][rel]['properties'][prop]]
                    if allowedDatatype != infer_dtype(df_datatype):
                        logging.error('Relation: ' + str(rel) + '- Property: ' + str(prop) + '. Dataframe datatype (' + str(df_datatype) + ') does \\'
                                      'not match datatype defined in schema (' + str(df_datatype) + ')')

    def _returnNodePropertyDatatype(self, nodeID, propID):
        """
        Method that returns the allowed type of a specified node property
        @ In, node, string, specific node label
        @ In, prop, string, specific node property
        @ Out, string, allowed type of the specified node property
        """
        allowedtype = None
        for schema in self.graphSchemas:
            for node in self.graphSchemas[schema]['node']:
                if node==nodeID:# and propID in self.graphSchemas[schema][node]:
                    for prop in self.graphSchemas[schema]['node'][node]['node_properties']:
                        if prop['name']==propID:
                            allowedtype = prop['type']
                            return allowedtype
        if allowedtype is None:
            logging.error('_returnNodePropertyDatatype error retrieving prop')

    def _returnRelationPropertyDatatype(self, relID, propID):
        """
        Method that returns the allowed type of a specified relation property.
        @ In, node, string, specific relation
        @ In, prop, string, specific node property
        @ Out, string, allowed type of the specified relation property
        """
        allowedtype = None
        for schema in self.graphSchemas:
            for rel in self.graphSchemas[schema]['relation']:
                if rel==relID:
                    for prop in self.graphSchemas[schema]['relation'][rel]['relation_properties']:
                        if prop['name']==propID:
                            allowedtype = prop['type']
                            return allowedtype
        if allowedtype is None:
            logging.error('_returnRelationPropertyDatatype error')


def stringToDatetimeConverterFlexible(date_string, format_code=None):
    """
    Method that convert a string into datetime according to specific format
    @ In, date_string, string, string containing date
    @ In, format_code, string, datetime specific format
    @ Out, datetime_object, datetime, datetime object
    """
    formats = ["%Y-%m-%d %H:%M:%S",
               "%Y/%m/%d %H:%M:%S",
               "%d-%m-%Y %H:%M",
               "%Y-%m-%d"]
    
    if format_code is not None:
        formats.append(format_code)

        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                raise ValueError(f"Unable to parse date string: {date_string}")
    else:
        try:
            return parse(date_string)
        except ValueError:
            raise ValueError(f"Unable to parse date string: {date_string}")

"""
def mbseWorkflow(self, name, type, nodesFile, edgesFile):
    if type =='customMBSE':
        if 'customMbseSchema' not in self.graphSchemas.keys():
            graphSchemaFile = self.predefinedGraphSchemas['customMbseSchema']
            self.importGraphSchema('customMbseSchema', graphSchemaFile)
        
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
        # TODO Implement LML reader
        pass

def anomalyWorkflow(self, dataframe, constructionSchema, monitorVars):
    if 'numericPerfomanceSchema' not in self.graphSchemas.keys():
        graphSchemaFile = self.predefinedGraphSchemas['numericPerfomanceSchema']
        self.importGraphSchema('numericPerfomanceSchema', graphSchemaFile)

    label = 'anomaly'
    if 'ID' in constructionSchema.keys():
        attribute = {'ID':constructionSchema['ID'], 
                     'time_initial':constructionSchema['time_initial'], 
                     'time_final'  :constructionSchema['time_final']}
    else:
        attribute = {'time_initial':constructionSchema['time_initial'], 
                     'time_final'  :constructionSchema['time_final']}
    self.py2neo.load_dataframe_for_nodes(dataframe, label, attribute)

    for var in monitorVars:
        l1 = 'anomaly'
        p1 = constructionSchema['time_initial']
        l2 = 'monitored_variable'
        p2 = var
        lr = 'detected_by'
        pr = None
        self.py2neo.load_dataframe_for_relations(dataframe, l1, p1, l2, p2, lr, pr)


def monitoringWorkflow(self, dataframe, constructionSchema):
    if 'monitoringSystemSchema' not in self.graphSchemas.keys():
        graphSchemaFile = self.predefinedGraphSchemas['monitoringSystemSchema']
        self.importGraphSchema('monitoringSystemSchema', graphSchemaFile)

    #constructionSchema.keys() = [variable, ID, mbse}

    label = 'monitored_variable'
    properties = {'ID': constructionSchema['ID']}
    if 'variable' in constructionSchema.keys():
        properties['variable'] = constructionSchema['variable']
    self.load_dataframe_for_nodes(dataframe, label, properties)

    l1='monitored_variable'
    p1={'ID':constructionSchema['ID']}
    l2='mbse_entity'
    p2 ={'ID':constructionSchema['mbse']}
    lr = 'monitors'
    pr = None
    self.load_dataframe_for_relations(dataframe, l1, p1, l2, p2, lr, pr)


def conditionReportWorkflow(self, dataframe, constructionSchema):
    #constructionSchema = {'date': [],
    #                    'ID': [],
    #                    'conjecture': [],
    #                    'mbse_entity': [],
    #                    'nuclear_entity': [],
    #                    'temporal_entity': []}

    
    if 'conditionReportSchema' not in self.graphSchemas.keys():
        graphSchemaFile = self.predefinedGraphSchemas['conditionReportSchema']
        self.importGraphSchema('conditionReportSchema', graphSchemaFile)

    label = 'condition_report'
    node_properties = {'date': constructionSchema['date'],
                       'ID': constructionSchema['ID']}
    if 'conjecture' in constructionSchema.keys():
        node_properties['conjecture'] = constructionSchema['conjecture']
    self.load_dataframe_for_nodes(dataframe, label, node_properties)

    for index, row in dataframe.iterrows():
        for ent in row[constructionSchema['nuclear_entity']]:
            if self.find_nodes('nuclear_entity',{'ID':ent}):
                # Entity node is already present
                self.create_relation(l1='condition_report', 
                                    p1={'ID': row[constructionSchema['ID']]}, 
                                    l2='nuclear_entity', 
                                    p2={'entity': ent}, 
                                    lr='refers')
            else:
                # Entity node is not present
                derivedClass = self.entityLibrary.searchEntityInfo(ent)
                properties = {'entity': ent,
                              'class': derivedClass}
                self.create_node('nuclear_entity', properties)
                self.create_relation(l1='condition_report', 
                                    p1={'ID': row[constructionSchema['ID']]}, 
                                    l2='nuclear_entity', 
                                    p2={'entity': ent}, 
                                    lr='refers')

        for ent in row[constructionSchema['temporal_entity']]:
            properties = {'datetime': ent}
            self.create_node('temporal_entity', properties)
            self.create_relation(l1='condition_report', 
                                 p1={'ID': row[constructionSchema['ID']]}, 
                                 l2='temporal_entity', 
                                 p2={'datetime': ent}, 
                                 lr='temporal_reference')

        for ent in row[constructionSchema['mbse_entity']]:
            if self.find_nodes('mbse_entity',{'ID':ent}):
                self.create_relation(l1='condition_report', 
                                     p1={'ID': row[constructionSchema['ID']]}, 
                                     l2='mbse_entity', 
                                     p2={'ID':ent}, 
                                     lr='mentions')
            elif self.find_nodes('mbse_entity',{'label':ent}):
                self.create_relation(l1='condition_report', 
                                     p1={'ID': row[constructionSchema['ID']]}, 
                                     l2='mbse_entity', 
                                     p2={'label':ent}, 
                                     lr='mentions')
            else:
                print('Error, MBSE entity not found')
"""