# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on February, 2024

@author: mandd
"""

# External Imports
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class customMBSEobject(object):
    """
        Class designed to process the a custom MBSE model from file.
    """
    def __init__(self, nodes_filename, edges_filename):
        """
        Initialization method for the custom MBSE model class

        Args:

            nodes_filename: file, file in .csv format containing all nodes 
            edges_filename: file, file in .csv format containing all edges 

        Returns:

            None
        """
        self.nodes_filename = nodes_filename
        self.edges_filename = edges_filename
        self.entities = []
        self.list_IDs = []

        self.allowed_node_types = ['entity']
        self.allowed_edge_types = ['link','composition','support'] # to be developed: 'opm_instance'

        self.allowed_node_cols = ['nodeID','label','ID','type','description']
        self.allowed_edge_cols = ['sourceNodeId','targetNodeId','type','medium','description']

        self.parseFiles()
        self.checkNodesFile()
        self.checkEdgesFile()
    
    def parseFiles(self):
        """
        Medthods designed to parse the node and edge files

        Args:

            None

        Returns:

            None
        """
        # parse nodes
        self.nodes_df = pd.read_csv(self.nodes_filename, sep=',', skip_blank_lines=True, dtype=str)
        self.nodes_df.dropna(how='all', inplace=True)

        self.entities = self.nodes_df['nodeID'].to_list()
        # parse edges
        self.edges_df = pd.read_csv(self.edges_filename, sep=',', skip_blank_lines=True, dtype=str)
        self.edges_df.dropna(how='all', inplace=True)

    def checkNodesFile(self):
        """
        Medthods designed to check the node file

        Args:

            None

        Returns:

            None
        """
        logger.info('- Check node file -')
        # Check all columns are present
        cols = self.nodes_df.columns.tolist()
        if set(cols)!=set(self.allowed_node_cols):
            raise IOError('Node file structure check - Error: wrong set of provided columns (allowed: nodeID, label, ID, type)')
        else:
             logger.info('Node file structure check - Pass')

        # Check for duplicate IDs
        duplicateIDs = self.nodes_df.duplicated()

        if self.nodes_df[duplicateIDs].empty:
             logger.info("List of node IDs check - Pass")
        else:
             logger.info("List of node IDs check - Error: duplicate IDs were found:")
             logger.info(self.nodes_df[duplicateIDs])
        
        #check for structure of each row
        logger.info("Entity check...")
        for index, row in self.nodes_df.iterrows():
            if row['type'] not in set(self.allowed_node_types):
                raise IOError('Type of row ' + str(index) + ' in node file is not allowed. Allowed types: ' +str(self.allowed_node_types))
            
            if pd.isnull(row['type']) and pd.isnull(row['ID']):
                raise IOError('Entity of row ' + str(index) + ' in node file: Error - neither type nor ID have been specified')
        logger.info("Entities check: Pass")

    def checkEdgesFile(self):
        """
        Medthods designed to check the edge file

        Args:

            None

        Returns:

            None
        """
        logger.info('- Check edge file -')
        # Check all columns are present
        cols = self.edges_df.columns.tolist()
        if set(cols)!=set(self.allowed_edge_cols):
            raise IOError('Edge file structure check - Error: wrong set of provided columns (allowed: sourceNodeId,targetNodeId,type,medium)')
        else:
             logger.info('Edge file structure check - Pass')

        # Check for duplicate edges
        duplicateEdges = self.edges_df[['sourceNodeId','targetNodeId']].duplicated()

        if self.edges_df[duplicateEdges].empty:
             logger.info("List of edges check - Pass")
        else:
            logger.info("List of edges check - Error: duplicate edges were found:")
            logger.info(self.edges_df[duplicateEdges])

        # Check IDs in edge file are defined in node file
        sourceNodeId_list = self.edges_df['sourceNodeId'].to_list()
        diff1 = set(sourceNodeId_list) - set(self.entities)
        if diff1:
            raise IOError('Error - Edge file: not recognized entities: ' + str(diff1))

        targetNodeId_list = self.edges_df['targetNodeId'].to_list()
        diff2 = set(targetNodeId_list) - set(self.entities)
        if diff2:
            raise IOError('Error - Edge file: not recognized entities: ' + str(diff2))

        # Check for structure of each row
        logger.info("Edges check...")
        for index, row in self.edges_df.iterrows():
            if pd.isnull(row['sourceNodeId']) or pd.isnull(row['targetNodeId']):
                logger.info(row)
                raise IOError('Edge ' + str(index) + ' in edge file: Error - both sourceNodeId and targetNodeId need to be specified')
             
            if row['type'] not in set(self.allowed_edge_types):
                logger.info(row)
                raise IOError('Type of row ' + str(index) + ' in edge file is not allowed. Allowed types: ' +str(self.allowed_edge_types))
            
            if row['type']=='link' and pd.isnull(row['medium']):
                logger.info(row)
                raise IOError('Edge ' + str(index) + ' in edge file: Error - link does not have a medium specified')

            if row['type']=='support' and not pd.isnull(row['medium']):
                logger.info(row)
                raise IOError('Edge ' + str(index) + ' in edge file: Error - support does not support medium keyword; specified:' +str(row['medium']))

        # check that entities in the node file have been mentioned in edge file
        entities_edge_list = sourceNodeId_list + targetNodeId_list
        diff3 = set(self.entities) - set(entities_edge_list)
        if diff3:
            raise IOError('Error - Node file: these entities in the node file were not mentioned in the edge file: ' + str(diff3))        
        logger.info("Edges check: Pass")
        
        # Provide info of outgoing only nodes
        outgoingSet = set(sourceNodeId_list) - set(targetNodeId_list)
        logger.info('List of outgoing only nodes:' + str(outgoingSet))
        # Provide info of ingoing only nodes
        ingoingSet = set(targetNodeId_list) - set(sourceNodeId_list)
        logger.info('List of ingoing only nodes:' + str(ingoingSet))

        