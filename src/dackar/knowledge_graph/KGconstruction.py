# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on September, 2025

@author: mandd, wangc
"""
# External Modules #
import re
import pandas as pd
import os, sys
import tomllib
from jsonschema import Draft202012Validator, ValidationError, SchemaError
import json
import copy
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse
from pandas.api.types import infer_dtype

import logging

currentDir = os.path.dirname(__file__)

# Internal Modules #
from dackar.knowledge_graph.py2neo import Py2Neo
from dackar.knowledge_graph.visualize_schema import createInteractiveFile
from dackar.knowledge_graph.graph_utils import set_neo4j_import_folder
from dackar.utils.mbse.customMBSEparser import customMBSEobject
from dackar.utils.tagKeywordListReader import entityLibrary


class KG:
    """
    Class designed to automate and check knowledge graph construction
    """
    def __init__(self, configFilePath, importFolderPath, uri, pwd, user):
        """
        Method designed to initialize the KG class
        @ In, configFilePath, string, DBMS database folder
        @ In, importFolderPath, string, folder which contains data to be imported
        @ In, uri, string, uri = "bolt://localhost:7687" for a single instance or uri = "neo4j://localhost:7687" for a cluster
        @ In, user, string, default to 'neo4j'
        @ In, pwd, string, password the the neo4j DBMS database
        @ Out, None
        """
        # Change import folder to user specific location
        if importFolderPath is not None:
            set_neo4j_import_folder(configFilePath, importFolderPath)

        self.datatypes = ['string', 'integer', 'floating', 'boolean', 'datetime', 'enum']

        # Create python to neo4j driver
        self.py2neo = Py2Neo(uri=uri, user=user, pwd=pwd)

        self.graphSchemas = {} # dictionary containing the set of schemas of the knowledge graph

        self.graphMetadata = {} # Metadata container of the knowledge graph --> TODO: discuss how to manage it

        self.entityLibrary = entityLibrary(os.path.join(currentDir, os.pardir, os.pardir, os.pardir, 'data', 'tag_keywords_lists.xlsx'))

        # this is the base schema for the set of schemas of the knowledge graph
        baseSchemaLocation = os.path.join(currentDir, 'schemas', 'baseSchema.json')
        with open(baseSchemaLocation, "r") as f:
            self.baseSchema = json.load(f)

        # set of predefined schemas available in DACKAR egenrated for the RIAM project
        self.predefinedGraphSchemas = {'conditionReportSchema'  : os.path.join(currentDir,'schemas','conditionReportSchema.toml'),
                                       'customMbseSchema'       : os.path.join(currentDir,'schemas','customMbseSchema.toml'),
                                       'monitoringSystemSchema' : os.path.join(currentDir,'schemas','monitoringSystemSchema.toml'),
                                       'nuclearEntitySchema'    : os.path.join(currentDir,'schemas','nuclearEntitySchema.toml'),
                                       'numericPerfomanceSchema': os.path.join(currentDir,'schemas','numericPerfomanceSchema.toml'),
                                       'causalSchema'           : os.path.join(currentDir,'schemas','causalSchema.toml')}

    def resetGraph(self):
        """
        Method designed to reset knowledge graph
        @ In, None
        @ Out, None
        """
        self.py2neo.reset()

    def crossSchemasCheck(self):
        """
        Perform cross-schema consistency checks:
        - Detect duplicate node and relation names across schemas
        - Ensure relations reference defined nodes
        Logs warnings instead of raising immediately.
        """
        node_set = set()
        relation_set = set()
        errors = []

        # Iterate through all schemas
        for schema_name, schema_data in self.graphSchemas.items():
            nodes = schema_data.get("node", {})
            relations = schema_data.get("relation", {})

            # Validate structure
            if not isinstance(nodes, dict) or not isinstance(relations, dict):
                errors.append(f"Schema '{schema_name}' has invalid structure for 'node' or 'relation'.")
                continue

            # Check nodes for duplicates
            for node in nodes.keys():
                if node in node_set:
                    errors.append(f"Duplicate node '{node}' found in schema '{schema_name}'.")
                else:
                    node_set.add(node)

            # Check relations for duplicates and validate endpoints
            for rel_name, rel_info in relations.items():
                if rel_name in relation_set:
                    errors.append(f"Duplicate relation '{rel_name}' found in schema '{schema_name}'.")
                else:
                    relation_set.add(rel_name)

                # Validate endpoints exist
                origin = rel_info.get("from_node")
                destin = rel_info.get("to_node")
                if origin not in node_set:
                    errors.append(f"Schema '{schema_name}' relation '{rel_name}': origin node '{origin}' is not defined.")
                if destin not in node_set:
                    errors.append(f"Schema '{schema_name}' relation '{rel_name}': destination node '{destin}' is not defined.")

        # Log all errors or success
        if errors:
            for msg in errors:
                logging.warning(msg)
        else:
            logging.info("Cross-schema check passed with no issues.")

    def _checkSchemaStructure(self, schemaName, importedSchema):
        """
        Method designed to check importedSchema against self.baseSchema
        @ In, schemaName, string, name of the schema
        @ In, importedSchema, dict, schema parsed by tomllib from .toml file
        @ Out, None
        """

        # Ensure baseSchema itself is valid
        try:
            Draft202012Validator.check_schema(self.baseSchema)
        except SchemaError as e:
            logging.error("Base schema is invalid: %s", e)
            raise  # propagate; importer should fail

        validator = Draft202012Validator(self.baseSchema)

        # Collect ALL errors so the user can fix them in one pass
        errors = sorted(validator.iter_errors(importedSchema), key=lambda e: e.path)

        if errors:
            # Build a readable message with locations and reasons
            lines = [f"Schema '{schemaName}' failed validation against base schema:"]
            for err in errors:
                location = "/".join(str(p) for p in err.path) or "<root>"
                lines.append(f"  - at {location}: {err.message}")
            message = "\n".join(lines)

            logging.error(message)
            # Raise a single ValidationError carrying the aggregated message
            raise ValidationError(message)

        logging.info("Schema '%s' is valid against the base schema.", schemaName)

    def removeGraphSchema(self, graphSchemaName):
        del self.graphSchemas[graphSchemaName]

    def importGraphSchema(self, graphSchemaName, tomlFilename):
        """
        Method that imports new schema contained in a .toml file
        @ In, graphSchemaName, string, name of the schema to be imported
        @ In, tomlFilename, string, .toml file contained the new schema
        @ Out, None
        """

        # Basic input validation
        if not isinstance(graphSchemaName, str) or not graphSchemaName.strip():
            raise ValueError("graphSchemaName must be a non-empty string.")

        fullPath = Path(tomlFilename)
        if not fullPath.exists():
            raise FileNotFoundError(f"Schema file not found: {tomlFilename}")

        with open(fullPath, 'rb') as f:
            configData = tomllib.load(f)

        # Check structure of imported graphSchema
        self._checkSchemaStructure(graphSchemaName, configData)

        # Access to nodes/relations 
        new_nodes = configData.get("node", {})
        new_relations = configData.get("relation", {})
        if not isinstance(new_nodes, dict) or not isinstance(new_relations, dict):
            raise ValueError("Schema 'node' and 'relation' sections must be dictionaries.")

        # Check imported graphSchema against self.graphSchemas
        # check schema name is not used before
        if graphSchemaName in self.graphSchemas:
            message = f"Schema '{graphSchemaName}' is already defined in the existing schemas."
            raise ValueError(message)


        # Build sets of existing nodes and relations across all registered schemas
        existing_nodes = set()
        existing_relations = set()
        for existing_name, existing_schema in self.graphSchemas.items():
            nodes = existing_schema.get("node", {})
            relations = existing_schema.get("relation", {})
            if not isinstance(nodes, dict) or not isinstance(relations, dict):
                raise ValueError(f"Existing schema '{existing_name}' has invalid 'node'/'relation' structure.")
            existing_nodes.update(nodes.keys())
            existing_relations.update(relations.keys())

        # Check conflicts
        for node in new_nodes.keys():
            if node in existing_nodes:
                message = (
                    f"Node '{node}' defined in the new schema '{graphSchemaName}' "
                    f"is already defined in an existing schema."
                )
                raise ValueError(message)

        for relation in new_relations.keys():
            if relation in existing_relations:
                message = (
                    f"Relation '{relation}' defined in the new schema '{graphSchemaName}' "
                    f"is already defined in an existing schema."
                )
                raise ValueError(message)

        # Cross-schema checks should include the new schema; use a copy to avoid mutation
        prospective_schemas = dict(self.graphSchemas)
        prospective_schemas[graphSchemaName] = configData
        #self._crossSchemasCheck()  --> TODO this should be check when all Schemas are imported

        # All checks passed; commit
        self.graphSchemas[graphSchemaName] = configData
        logging.info(f"Schema '{graphSchemaName}' imported successfully.")


    def _schemaReturnNodeProperties(self, nodeLabel):
        """
        Method that returns the properties of the node nodeLabel
        @ In, nodeLabel, string, ID of the node label
        @ Out, propdf, dataframe, dataframe containing nodeLabel properties
        """
        propdf = None
        for schema in self.graphSchemas:
            if nodeLabel in self.graphSchemas[schema]['node'].keys():
                nodeProperties = self.graphSchemas[schema]['node'][nodeLabel]['node_properties']
                propdf = pd.DataFrame(nodeProperties)
                return propdf

        if propdf is None:
            message = 'Node ' + str(nodeLabel) + ' does not have any property'
            logging.error(message)
            raise ValueError(message)

    def _schemaReturnRelationProperties(self, relation):
        """
        Method that returns the properties of the selected relation
        @ In, relation, string, ID of the node label
        @ Out, propdf, dataframe, dataframe containing relation properties
        """
        propdf = None
        for schema in self.graphSchemas:
            if relation in self.graphSchemas[schema]['relation']:
                relationProperties = self.graphSchemas[schema]['relation'][relation]['relation_properties']
                propdf = pd.DataFrame(relationProperties)
                return propdf

        if propdf is None:
            message = 'Relation ' + str(relation) + ' does not have any property'
            raise ValueError(message)

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
                            message = 'Key ' + str(kkey) + 'in the construction schema should be a dictionary'
                            raise ValueError(message)
                else:
                    message = 'Key ' + str(key) + 'in the construction schema should be a dictionary'
                    raise ValueError(message)
            elif key=='relations':
                if isinstance(constructionSchema[key], dict):
                    for kkey in constructionSchema[key].keys():
                        if not isinstance(constructionSchema['relations'][kkey], dict):
                            message = 'Key ' + str(kkey) + 'in the construction schema should be a dictionary'
                            raise ValueError(message)
                        if list(constructionSchema['relations'][kkey].keys())!=['source','target','properties']:
                            message = 'Relation ' + str(kkey) + ' needs to contain these keys: source, target, properties'
                            raise ValueError(message)
                else:
                    message = 'Key ' + str(key) + 'in the construction schema should be a list'
                    raise ValueError(message)
            else:
                message = 'Key ' + str(key) + 'in the construction schema is not allowed (allowed: nodes, relations)'
                raise ValueError(message)

    def _constructionSchemaValidation(self, constructionSchema):
        """
        Method that validates the constructionSchema against defined schemas
        @ In, constructionSchema, dict, construction schema
        @ Out, None
        """
        # For each node check that required properties are listed
        if 'nodes' in constructionSchema:
            for node in constructionSchema['nodes']:
                specifiedProp = set(constructionSchema['nodes'][node].keys())

                propDf = self._schemaReturnNodeProperties(node)
                allowedProperties = set(propDf['name'])

                selectedPropDf = propDf[propDf['optional']==False]
                reqProperties = set(selectedPropDf['name'])

                if not reqProperties.issubset(specifiedProp):
                    message = 'Node ' + str(node) + 'requires all these properties: ' + str(reqProperties)
                    raise ValueError(message)
                if not specifiedProp.issubset(allowedProperties):
                    message = 'Node ' + str(node) + 'requires these properties: ' + str(allowedProperties)
                    raise ValueError(message)

        # For each relation check that required properties are listed
        if 'relations' in constructionSchema:
            for rel in constructionSchema['relations']:
                specifiedProp = set(constructionSchema['relations'][rel]['properties'])

                propDf = self._schemaReturnRelationProperties(rel)
                allowedProperties = set(propDf['name'])

                selectedPropDf = propDf[propDf['optional']==False]
                reqProperties = set(selectedPropDf['name'])

                if not reqProperties.issubset(specifiedProp):
                    message = 'Relation ' + str(rel) + 'requires all these properties: ' + str(reqProperties)
                    raise ValueError(message)

                if not specifiedProp.issubset(allowedProperties):
                    message = 'Relation ' + str(rel) + 'requires these properties: ' + str(allowedProperties)
                    raise ValueError(message)

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
            dataMasked = copy.deepcopy(data)
            for node in constructionSchema['nodes'].keys():
                mapping = {value: key for key, value in constructionSchema['nodes'][node].items()}
                dataRenamed = dataMasked.rename(columns=mapping)
                self.py2neo.load_dataframe_for_nodes(df=dataRenamed, labels=node, properties=list(mapping.values()))

        # Relations
        # --> TODO: check nodes exist
        if 'relations' in constructionSchema:
            dataMasked = copy.deepcopy(data)
            for rel in constructionSchema['relations']:
                sourceNodeLabel = next(iter(constructionSchema['relations'][rel]['source'])).split('.')[0]
                sourceNodeProp  = next(iter(constructionSchema['relations'][rel]['source'])).split('.')[1]

                targetNodeLabel = next(iter(constructionSchema['relations'][rel]['target'])).split('.')[0]
                targetNodeProp  = next(iter(constructionSchema['relations'][rel]['target'])).split('.')[1]

                mapping = {}
                dataRenamed = dataMasked.rename(columns={next(iter(constructionSchema['relations'][rel]['source'].values())):sourceNodeProp,
                                                         next(iter(constructionSchema['relations'][rel]['target'].values())):targetNodeProp})

                for prop in constructionSchema['relations'][rel]['properties'].keys():
                    dataRenamed = dataRenamed.rename(columns={constructionSchema['relations'][rel]['properties'][prop]: prop})

                dataRenamed[sourceNodeLabel] = sourceNodeLabel
                dataRenamed[targetNodeLabel] = targetNodeLabel
                dataRenamed[rel] = rel

                self.py2neo.load_dataframe_for_relations(df=dataRenamed,
                                                         l1=sourceNodeLabel, p1=sourceNodeProp,
                                                         l2=targetNodeLabel, p2=targetNodeProp,
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
                    dfDatatype = data[constructionSchema['nodes'][node][prop]]
                    if allowedDatatype != infer_dtype(dfDatatype):
                        message = 'Node: ' + str(node) + '- Property: ' + str(prop) + '. Dataframe datatype (' + str(set(dfDatatype.map(type))) + ') does not match datatype defined in schema (' + str(allowedDatatype) + ')'
                        raise ValueError(message)

        # Check relations data types
        if 'relations' in constructionSchema:
            for rel in constructionSchema['relations']:
                for prop in constructionSchema['relations'][rel]['properties']:
                    allowedDatatype = self._returnRelationPropertyDatatype(rel,prop)
                    dfDatatype = data[constructionSchema['relations'][rel]['properties'][prop]]
                    if allowedDatatype != infer_dtype(dfDatatype):
                        message = 'Relation: ' + str(rel) + '- Property: ' + str(prop) + '. Dataframe datatype (' + str(dfDatatype) + ') does not match datatype defined in schema (' + str(dfDatatype) + ')'
                        raise ValueError(message)

    def _returnNodePropertyDatatype(self, nodeID, propID):
        """
        Method that returns the allowed type of a specified node property
        @ In, nodeID, string, specific node label
        @ In, propID, string, specific node property
        @ Out, allowedType, string, allowed type of the specified node property
        """
        allowedType = None
        for schema in self.graphSchemas:
            for node in self.graphSchemas[schema]['node']:
                if node==nodeID:# and propID in self.graphSchemas[schema][node]:
                    for prop in self.graphSchemas[schema]['node'][node]['node_properties']:
                        if prop['name']==propID:
                            allowedType = prop['type']
                            return allowedType
        if allowedType is None:
            ValueError('_returnNodePropertyDatatype error retrieving prop')

    def _returnRelationPropertyDatatype(self, relID, propID):
        """
        Method that returns the allowed type of a specified relation property.
        @ In, relID, string, specific relation
        @ In, propID, string, specific node property
        @ Out, allowedType, string, allowed type of the specified relation property
        """
        allowedType = None
        for schema in self.graphSchemas:
            for rel in self.graphSchemas[schema]['relation']:
                if rel==relID:
                    for prop in self.graphSchemas[schema]['relation'][rel]['relation_properties']:
                        if prop['name']==propID:
                            allowedType = prop['type']
                            return allowedType
        if allowedType is None:
            ValueError('_returnRelationPropertyDatatype error')

    def _createIteractivePlot(self):
        schemaList = list(self.graphSchemas.values())
        createInteractiveFile(schemaList)


def stringToDatetimeConverterFlexible(dateString, formatCode=None):
    """
    Method that convert a string into datetime according to specific format
    @ In, dateString, string, string containing date
    @ In, formatCode, string, datetime specific format
    @ Out, datetimeObject, datetime, datetime object
    """
    formats = ["%Y-%m-%d %H:%M:%S",
               "%Y/%m/%d %H:%M:%S",
               "%d-%m-%Y %H:%M",
               "%Y-%m-%d"]

    if formatCode is not None:
        formats.append(formatCode)

        for fmt in formats:
            try:
                datetimeObject = datetime.strptime(dateString, fmt)
                return datetimeObject
            except ValueError:
                raise ValueError(f"Unable to parse date string: {dateString}")
    else:
        try:
            datetimeObject = parse(dateString)
            return datetimeObject
        except ValueError:
            raise ValueError(f"Unable to parse date string: {dateString}")

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
