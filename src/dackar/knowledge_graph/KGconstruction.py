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
from jsonschema import validate, ValidationError
import json
import copy
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse
from pandas.api.types import infer_dtype

from dackar.knowledge_graph.schema_types import ALLOWED_SCHEMA_TYPES, isCompatibleDtype

import logging

currentDir = os.path.dirname(__file__)

# Internal Modules #
from dackar.knowledge_graph.py2neo import Py2Neo
#from dackar.knowledge_graph.visualize_schema import createIteractiveFile
from dackar.knowledge_graph.graph_utils import set_neo4j_import_folder
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

        # Allowed property data types. Sourced from schema_types so the
        # allowlist, the baseSchema.json type enum, and the dataframe dtype
        # compatibility map share a single definition. 'floating' is retained
        # as an alias of 'float'; 'array'/'json_string' support the document
        # / RAG schema.
        self.datatypes = list(ALLOWED_SCHEMA_TYPES)

        # Create python to neo4j driver
        self.py2neo = Py2Neo(uri=uri, user=user, pwd=pwd)

        self.graphSchemas = {} # dictionary containing the set of schemas of the knowledge graph

        self.graphMetadata = {} # Metadata container of the knowledge graph --> TODO: discuss how to manage it

        self.entityLibrary = entityLibrary(os.path.join(currentDir, os.pardir, os.pardir, os.pardir, 'data', 'tag_keywords_lists.xlsx'))

        # this is the base schema for the set of schemas of the knowledge graph
        baseSchemaLocation = os.path.join(currentDir, 'schemas', 'baseSchema.json')
        with open(baseSchemaLocation, "r") as f:
            self.baseSchema = json.load(f)

        # Curated set of predefined schemas available in DACKAR. Deprecated
        # schemas (customMbseSchema, reqTechspecSchema) are intentionally
        # excluded; their node definitions are superseded by mbseSchema.toml.
        def _schemaPath(name):
            return os.path.join(currentDir, 'schemas', name + '.toml')

        self.predefinedGraphSchemas = {
            name: _schemaPath(name)
            for name in [
                'nuclearEntitySchema',
                'mbseSchema',
                'documentSchema',
                'conditionReportSchema',
                'fmeaSchema',
                'causalSchema',
                'safetyRiskSchema',
                'rootCauseAnalysisSchema',
                'hazopSchema',
                'stpaSchema',
                'workOrderSchema',
                'outageSchema',
                'equipmentOperationSchema',
                'monitoringSystemSchema',
                'numericPerformanceSchema',
                'supplyChainSchema',
                'systemSimulationSchema',
                'temporalRelationSchema',
                'regulatorySchema',
            ]
        }

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
        self.nodeList = []
        self.relationList = []

        for schema in self.graphSchemas:
            for node in self.graphSchemas[schema].get('node', {}):
                # check that the node is not duplicated
                if node in self.nodeList:
                    message = 'Schema ' + str(schema) + ' - Node ' + str(node) + ' has been defined twice'
                    raise ValueError(message)
                else:
                    self.nodeList.append(node)

        for schema in self.graphSchemas:
            for rel in self.graphSchemas[schema].get('relation', {}):
                # check that the defined relations link nodes that have been defined
                origin = self.graphSchemas[schema]['relation'][rel]['from_entity']
                destin = self.graphSchemas[schema]['relation'][rel]['to_entity']

                # A relation is keyed by the (name, from_entity, to_entity)
                # triple, so the same generic verb (e.g. caused_by,
                # targets_element) may be reused across schemas as long as it
                # connects a different pair of node labels.
                relationKey = (rel, origin, destin)
                if relationKey in self.relationList:
                    message = 'Duplicate relation definition encountered: ' + str(relationKey) + ' in schema: ' + str(schema)
                    raise ValueError(message)
                else:
                    self.relationList.append(relationKey)

                if origin not in self.nodeList:
                    message = 'Schema ' + str(schema) + ' - Relation ' + str(rel) + ': Node label ' + str(origin) + ' is not defined'
                    raise ValueError(message)
                if destin not in self.nodeList:
                    message = 'Schema ' + str(schema) + ' - Relation ' + str(rel) + ': Node label ' + str(destin) + ' is not defined'
                    raise ValueError(message)

    def _checkSchemaStructure(self, importedSchema):
        """
        Method designed to check importedSchema against self.baseSchema
        @ In, importedSchema, dict, schema parsed by tomllib from .toml file
        @ Out, None
        """
        try:
            validate(instance=importedSchema, schema=self.baseSchema)
            logging.info("TOML content is valid against the schema.")
        except ValidationError as e:
            logging.error(f"TOML schema validation error: {e.message}")
            raise

    def importGraphSchema(self, graphSchemaName, tomlFilename):
        """
        Method that imports new schema contained in a .toml file
        @ In, graphSchemaName, string, name of the schema to be imported
        @ In, tomlFilename, string, .toml file contained the new schema
        @ Out, None
        """
        fullPath = Path(tomlFilename)

        if not fullPath.exists():
            raise FileNotFoundError(f"Schema file not found: {tomlFilename}")

        with open(fullPath, 'rb') as f:
            configData = tomllib.load(f)

        # Check structure of imported graphSchema
        self._checkSchemaStructure(configData)

        #check data types against self.datatypes
        self._checkSchemaDataTypes(configData)

        # check schema name is not used before
        if graphSchemaName in list(self.graphSchemas.keys()):
            message = 'Schema ' + str(graphSchemaName) + ' is already defined in the exisiting schemas'
            logging.error(message)
            raise ValueError(message)

        self.graphSchemas[graphSchemaName] = configData

        # NOTE: cross-schema integrity (duplicate node labels, duplicate
        # relation triples, and relation endpoints resolving to defined nodes)
        # is a whole-set property and is intentionally NOT checked per import.
        # It tolerates circular cross-references between schemas (e.g. fmea
        # references degradation_mechanism in causal, while causal references
        # failure_mode in fmea), which per-import checking cannot. Call
        # _crossSchemasCheck() (or loadPredefinedGraphSchemas) once all
        # schemas that participate in the set have been imported.

    def loadPredefinedGraphSchemas(self):
        """
        Method that loads the full curated set of predefined DACKAR schemas and
        validates cross-schema integrity once the whole set is in memory.
        @ In, None
        @ Out, None
        """
        for name, path in self.predefinedGraphSchemas.items():
            self.importGraphSchema(name, path)
        self._crossSchemasCheck()

    def _checkSchemaDataTypes(self, schema):
        """
        Method that checks that the datatypes defined in the new schema are part of the allowed data
        types contained in self.datatypes
        @ In, schema, dict, schema parsed by tomllib from .toml file
        @ Out, None
        """
        for node in schema.get('node', {}):
            for prop in schema['node'][node].get('node_properties', []):
                if prop['type'] not in self.datatypes:
                    message = 'Node ' + str(node) + ' - Property ' + str(prop['name']) + ' data type ' + str(prop['type']) + ' is not allowed'
                    logging.error(message)
                    raise ValueError(message)

        for rel in schema.get('relation', {}):
            for prop in schema['relation'][rel].get('relation_properties', []):
                if prop['type'] not in self.datatypes:
                    message = 'Relation ' + str(rel) + ' - Property ' + str(prop['name']) + ' data type ' + str(prop['type']) + ' is not allowed'
                    logging.error(message)
                    raise ValueError(message)

    def _schemaReturnNodeProperties(self, nodeLabel):
        """
        Method that returns the properties of the node nodeLabel
        @ In, nodeLabel, string, ID of the node label
        @ Out, propdf, dataframe, dataframe containing nodeLabel properties
        """
        propdf = None
        for schema in self.graphSchemas:
            if nodeLabel in self.graphSchemas[schema]['node'].keys():
                nodeProperties = self.graphSchemas[schema]['node'][nodeLabel].get('node_properties', [])
                # A node that declares no properties yields an empty frame that
                # still carries the columns callers index ('name', 'optional'),
                # so a propertyless-but-valid label does not KeyError downstream.
                columns = None if nodeProperties else ['name', 'type', 'optional']
                return pd.DataFrame(nodeProperties, columns=columns)

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
                relationProperties = self.graphSchemas[schema]['relation'][relation].get('relation_properties', [])
                # Many curated relations omit relation_properties; an empty frame
                # that still carries the columns callers index ('name', 'optional')
                # keeps a propertyless-but-valid relation from raising KeyError.
                columns = None if relationProperties else ['name', 'type', 'optional']
                return pd.DataFrame(relationProperties, columns=columns)

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
                    if not isCompatibleDtype(allowedDatatype, infer_dtype(dfDatatype)):
                        message = 'Node: ' + str(node) + '- Property: ' + str(prop) + '. Dataframe datatype (' + str(infer_dtype(dfDatatype)) + ') does not match datatype defined in schema (' + str(allowedDatatype) + ')'
                        raise ValueError(message)

        # Check relations data types
        if 'relations' in constructionSchema:
            for rel in constructionSchema['relations']:
                for prop in constructionSchema['relations'][rel]['properties']:
                    allowedDatatype = self._returnRelationPropertyDatatype(rel,prop)
                    dfDatatype = data[constructionSchema['relations'][rel]['properties'][prop]]
                    if not isCompatibleDtype(allowedDatatype, infer_dtype(dfDatatype)):
                        message = 'Relation: ' + str(rel) + '- Property: ' + str(prop) + '. Dataframe datatype (' + str(infer_dtype(dfDatatype)) + ') does not match datatype defined in schema (' + str(allowedDatatype) + ')'
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

    def _createIteractivePlot(self, output_file="knowledge_graph_schema_interactive.html"):
        """
        Method that renders the currently loaded schemas as an interactive HTML graph.
        @ In, output_file, string, path of the HTML file to write
        @ Out, None
        """
        from dackar.knowledge_graph.visualize_schema import createInteractiveFile
        schemaList = list(self.graphSchemas.values())
        createInteractiveFile(schemaList, output_file=output_file, raise_on_collision=False)


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

