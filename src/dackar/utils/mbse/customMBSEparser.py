# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on February, 2024

@author: mandd
"""

# External Imports
import pandas as pd
import logging
import networkx as nx 
import matplotlib.pyplot as plt
import os

logger = logging.getLogger("my logger")
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)
logger.setLevel(logging.DEBUG) 

class customMBSEobject(object):
    """
        Class designed to process the a custom MBSE model from file.
    """
    def __init__(self, nodesFilename, edgesFilename, path=None):
        """
        Initialization method for the custom MBSE model class

        Args:

            nodesFilename: file, file in .csv format containing all nodes 
            edgesFilename: file, file in .csv format containing all edges 

        Returns:

            None
        """
        self.nodesFilename = nodesFilename
        self.edgesFilename = edgesFilename
        self.listIDs = []

        self.allowedNodeTypes = ['entity']
        self.allowedEdgeTypes = ['link','composition','support'] # to be developed: 'opm_instance'

        self.allowedNodeCols   = ['label','ID','type']
        self.allowed_edge_cols = ['sourceNodeId','targetNodeId','type','medium']

        self.parseFiles()
        self.checkNodes()
        self.checkEdges()

        self.path = path

        nodesFileName = os.path.basename(self.nodesFilename) 
        nodesFileSplit = nodesFileName.split('.')
        nodesFileKg = nodesFileSplit[0] + '_kg.' + nodesFileSplit[1]

        edgesFileName = os.path.basename(self.edgesFilename) 
        edgesFileSplit = edgesFileName.split('.')
        edgesFileKg = edgesFileSplit[0] + '_kg.' + edgesFileSplit[1]

        if self.path is not None:
            full_path = os.path.join(os.getcwd(), path)
            os.makedirs(full_path, exist_ok=True)
            nodesFileKg = os.path.join(full_path, nodesFileKg)
            edgesFileKg = os.path.join(full_path, edgesFileKg)

        self.printOnFiles(nodesFileKg,edgesFileKg)
    
    def checkModel(self):
        """
        Method designed to pcheck model consistency

        Args:

            None

        Returns:

            None
        """
        self.checkNodes()
        self.checkEdges()       
    
    def parseFiles(self):
        """
        Method designed to parse the node and edge files

        Args:

            None

        Returns:

            None
        """
        # parse nodes
        self.nodesDf = pd.read_csv(self.nodesFilename, sep=',', skip_blank_lines=True, dtype=str)
        self.nodesDf.dropna(how='all', inplace=True)
        self.nodesDf = self.nodesDf.apply(lambda x: x.astype(str).str.lower())

        self.listIDs = self.nodesDf['ID'].dropna().to_list()
        
        # parse edges
        self.edgesDf = pd.read_csv(self.edgesFilename, sep=',', skip_blank_lines=True, dtype=str)
        self.edgesDf.dropna(how='all', inplace=True)
        self.edgesDf = self.edgesDf.apply(lambda x: x.astype(str).str.lower())


    def checkNodes(self):
        """
        Method designed to check the node file

        Args:

            None

        Returns:

            None
        """
        logger.info('- Check node file -')
        # Check all columns are present
        cols = self.nodesDf.columns.tolist()
        if set(cols)!=set(self.allowedNodeCols):
            raise IOError('Node file structure check - Error: wrong set of provided columns ' + str(cols) + ' (allowed: label, ID, type)')
        else:
             logger.info('Node file structure check - Pass')

        # Check for duplicate IDs
        duplicateIDs = self.nodesDf.duplicated()

        if self.nodesDf[duplicateIDs].empty:
             logger.info("List of node IDs check - Pass")
        else:
             logger.info("List of node IDs check - Error: duplicate IDs were found:")
             logger.info(self.nodesDf[duplicateIDs])
        
        #check for structure of each row
        logger.info("Entity check...")
        for index, row in self.nodesDf.iterrows():
            if row['type'] not in set(self.allowedNodeTypes):
                raise IOError('Type of row ' + str(index) + ' in node file is not allowed. Allowed types: ' +str(self.allowedNodeTypes))
            
            if pd.isnull(row['type']) and pd.isnull(row['ID']):
                raise IOError('Entity of row ' + str(index) + ' in node file: Error - neither type nor ID have been specified')
        logger.info("Entities check: Pass")

    def checkEdges(self):
        """
        Methods designed to check the edge file

        Args:

            None

        Returns:

            None
        """
        logger.info('- Check edge file -')
        # Check all columns are present
        cols = self.edgesDf.columns.tolist()
        if set(cols)!=set(self.allowed_edge_cols):
            raise IOError('Edge file structure check - Error: wrong set of provided columns (allowed: sourceNodeId,targetNodeId,type,medium)')
        else:
             logger.info('Edge file structure check - Pass')

        # Check for duplicate edges
        duplicateEdges = self.edgesDf[['sourceNodeId','targetNodeId']].duplicated()

        if self.edgesDf[duplicateEdges].empty:
             logger.info("List of edges check - Pass")
        else:
            logger.info("List of edges check - Error: duplicate edges were found:")
            logger.info(self.edgesDf[duplicateEdges])

        # Check IDs in edge file are defined in node file
        sourceNodeId_list = self.edgesDf['sourceNodeId'].to_list()
        diff1 = set(sourceNodeId_list) - set(self.listIDs)
        if diff1:
            raise IOError('Error - Edge file: not recognized entities: ' + str(diff1))

        targetNodeId_list = self.edgesDf['targetNodeId'].to_list()
        diff2 = set(targetNodeId_list) - set(self.listIDs)
        if diff2:
            raise IOError('Error - Edge file: not recognized entities: ' + str(diff2))

        # Check for structure of each row
        logger.info("Edges check...")
        for index, row in self.edgesDf.iterrows():
            if pd.isnull(row['sourceNodeId']) or pd.isnull(row['targetNodeId']):
                logger.info(row)
                raise IOError('Edge ' + str(index) + ' in edge file: Error - both sourceNodeId and targetNodeId need to be specified')
             
            if row['type'] not in set(self.allowedEdgeTypes):
                logger.info(row)
                raise IOError('Type of row ' + str(index) + ' in edge file is not allowed. Allowed types: ' +str(self.allowedEdgeTypes))
            
            if row['type']=='link' and pd.isnull(row['medium']):
                logger.info(row)
                raise IOError('Edge ' + str(index) + ' in edge file: Error - link does not have a medium specified')

            if row['type']=='support' and row['medium']!='nan':
                logger.info(row['medium'])
                logger.info(type(row['medium']))
                raise IOError('Edge ' + str(index) + ' in edge file: Error - support does not support medium keyword; specified:' +str(row['medium']))

        # check that entities in the node file have been mentioned in edge file
        entities_edge_list = sourceNodeId_list + targetNodeId_list
        diff3 = set(self.listIDs) - set(entities_edge_list)
        if diff3:
            raise IOError('Error - Node file: these entities in the node file were not mentioned in the edge file: ' + str(diff3))        
        logger.info("Edges check: Pass")
        
        # Provide info of outgoing only nodes
        outgoingSet = set(sourceNodeId_list) - set(targetNodeId_list)
        logger.info('List of outgoing only nodes:' + str(outgoingSet))
        # Provide info of ingoing only nodes
        ingoingSet = set(targetNodeId_list) - set(sourceNodeId_list)
        logger.info('List of ingoing only nodes:' + str(ingoingSet))

    def returnIDs(self):
        """
        Method designed to return list of IDs included in the model

        Args:

            None

        Returns:

            self.listIDs, list, list of IDs specified in the MBSE model
        """
        return self.listIDs
    
    def addNodesEdges(self, new_node_dict, new_edge_dicts):
        """
        Method designed to return list of IDs included in the model

        Args:

            None

        Returns:

            self.listIDs, list, list of IDs specified in the MBSE model
        """        
        self.nodesDf.loc[len(self.nodesDf)] = new_node_dict
        
        for edge in new_edge_dicts:
            self.edgesDf.loc[len(self.edgesDf)] = edge
        
        self.listIDs = self.nodesDf['ID'].dropna().to_list()

    def printOnFiles(self,nodes_file,edges_file):
        """
        Method designed to print on file the set of nodes and edges

        Args:

            None

        Returns:

            self.listIDs, list, list of IDs specified in the MBSE model
        """  

        self.nodesDf.to_csv(nodes_file, index=False)
        self.edgesDf.to_csv(edges_file, index=False)

    def plot(self, fileID):
        """
        Method designed to plot on file the obtained graph

        Args:

            fileID, string, name of the file

        Returns:

            None
        """  
        G = nx.DiGraph()

        for index, row in self.edgesDf.iterrows():
            if row['type'] == 'link':
                G.add_edge(row['sourceNodeId'],	row['targetNodeId'], type='link')
            else: 
                G.add_edge(row['sourceNodeId'],	row['targetNodeId'], type='support')

        edgeTypeColors = {'link': 'blue', 'support': 'green'}

        edgeColors = []
        for u, v, data in G.edges(data=True):
            edgeType = data.get('type')
            edgeColors.append(edgeTypeColors.get(edgeType, 'black'))

        pos = nx.spring_layout(G) # Or any other layout algorithm
        nx.draw(G, pos, with_labels=True, edge_color=edgeColors, node_color='lightblue', font_color='black')

        if self.path is not None:
            full_path = os.path.join(os.getcwd(), self.path)
            os.makedirs(full_path, exist_ok=True)
            fileName = 'fileID' + str('.png')
            fileName = os.path.join(full_path, fileName)
        else: 
            fileName = 'fileID' + str('.png')
        plt.savefig(fileName)

    def printEquipmentID(self):
        """
        Method designed to print on file the list of IDs

        Args:

            None

        Returns:

            None
        """  
        
        if self.path is not None:
            full_path = os.path.join(os.getcwd(), self.path)
            os.makedirs(full_path, exist_ok=True)
            idFile = str(self.__class__.__name__) + "_ID.csv"
            idFile = os.path.join(full_path, idFile)
        else: 
            idFile = str(self.__class__.__name__) + "_ID.csv"
        self.nodesDf['ID'].to_csv(idFile, index=False)