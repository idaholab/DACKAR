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
import tomllib
from jsonschema import validate, ValidationError
import copy
from pathlib import Path

class KG:
    #def __init__(self, config_file_path, import_folder_path, uri, pwd, user, processedDataFolder):
    def __init__(self, config_file_path, uri, pwd, user):    
        # Change import folder to user specific location

        #set_neo4j_import_folder(config_file_path, import_folder_path)
        
        #self.processedDataFolder = processedDataFolder

        self.datatypes = ['string', 'int', 'float', 'boolean', 'datetime']

        # Create python to neo4j driver
        self.py2neo = Py2Neo(uri=uri, user=user, pwd=pwd)

        self.graphSchemas = {}

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
                                                                                                                      "optional": {"type": "boolean", "description": "Specifies if this property is required or not"}},
                                                                                                       "required":["name","type","optional"],
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
                                                                                                                          "optional": {"type": "boolean", "description": "Specifies if this property is required or not"}},
                                                                                                                          "required":["name","type","optional"],
                                                                                                      }
                                                                                            }},
                                                         "required":["relation_description","from_entity","to_entity"],
                                                        }
                                            },
                            "required":["title"]}

        self.predefinedGraphSchemas = {'conditionReportSchema'  : 'conditionReportSchema.toml',
                                       'customMbseSchema'       : 'customMbseSchema.toml',
                                       'monitoringSystemSchema' : 'monitoringSystemSchema.toml',
                                       'nuclearEntitySchema'    : 'nuclearEntitySchema.toml',
                                       'numericPerfomanceSchema': 'numericPerfomanceSchema.toml'}

    def resetGraph(self):
        self.py2neo.reset()

    def _checkSchemaStructure(self, importedSchema):
        try:
            validate(instance=importedSchema, schema=self.schemaSchema)
            print("TOML content is valid against the schema.")
        except tomllib.TOMLDecodeError as e:
            print(f"TOML syntax error: {e}")
        except ValidationError as e:
            print(f"TOML schema validation error: {e.message}")
    
    def importGraphSchema(self, graphSchemaName, tomlFilename):         
        config_path = Path(tomlFilename)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {tomlFilename}")

        with open(config_path, 'rb') as f:
            config_data = tomllib.load(f)
        
        # Check structure of imported graphSchema
        self._checkSchemaStructure(config_data)

        #TODO: check datatypes against self.datatypes

        # Check imported graphSchema against self.graphSchemas
        if graphSchemaName in list(self.graphSchemas.keys()):
            print('Schema ' + str(graphSchemaName) + ' is already defined in the exisiting schemas')

        for node in config_data['node'].keys():
            for schema in self.graphSchemas:
                if node in schema['node'].keys():
                    print('Node ' + str(node) + ' defined in the new schema is already defined in the exisiting schema ' + str(schema))

        for relation in config_data['relation'].keys():
            for schema in self.graphSchemas:
                if relation in schema['relation'].keys():
                    print('Relation ' + str(node) + ' defined in the new schema is already defined in the exisiting schema ' + str(schema))

        self.graphSchemas[graphSchemaName] = config_data
        return config_data

    def _schemaReturnNodeProperties(self, nodeLabel):
        for schema in self.graphSchemas:
            if nodeLabel in schema['node'].keys():
                node_properties = schema['node'][nodeLabel]['node_properties']
                df = pd.DataFrame(node_properties)
                return df
        print('Node not found')
        return None

    def _schemaValidation(self, constructionSchema):
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

    def genericWorkflow(self, data, constructionSchema):
        # Check constructionSchema against self.graphSchemas  

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

        '''
        ---- Example of graph schema (toml file) ----

        title = "Graph Schema for ..."
        version = "1.0"

        # Nodes
        [node.label1]
        description = "This node represents ..."
        properties = {"prop1": {type="date",   required=bool},
                      "prop2": {type="string", required=bool}}

        # Relationships
        [relationships.relation1]
        description = "relation1 indicates ... "
        from_entity = entity1
        to_entity = entity2
        properties = {"prop1": {type="int",   required=bool},
                      "prop2": {type="float", required=bool}}



        ---- Example of construction schema ----

        constructionSchema = {'nodes': nodeConstructionSchema,
                              'relations': edgeConstructionSchema}

        nodeConstructionSchema = {'nodeLabel1': {'property1': 'node.colA', 'property2': 'node.colB'},
                                  'nodeLabel2': {'property1': 'node.colC'}}
        
        edgeConstructionSchema = [{'source': {'nodeLabel1.property1':'col1'},
                                   'target': {'nodeLabel2.property1':'col2'},
                                   'type'  : 'edgeType',
                                   'properties': {'property1': 'colAlpha', 'property2': 'colBeta'}}] 

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
                                                     pr=edge['properties'])
        



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

