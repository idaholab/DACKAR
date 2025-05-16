# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on March 20, 2025

@author: wangc, mandd
"""

# Source:
# https://neo4j.com/docs/graph-data-science-client/current/
# https://github.com/neo4j/graph-data-science

# Python client for neo4j GDS library
# https://github.com/neo4j/graph-data-science-client
# pip install graphdatascience
# for networkx support
# pip install graphdatascience[networkx]

from graphdatascience import GraphDataScience
import logging

logger = logging.getLogger(__name__)


class PyGDS:

    def __init__(self, uri, user, pwd, database='neo4j'):
        """init method

        Args:
            uri (str): # uri = "bolt://localhost:7687" for a single instance or uri = "neo4j://localhost:7687" for a cluster
            user (str): default to 'neo4j'
            pwd (str): password the the neo4j DBMS database (can be reset in neo4j Desktop app)
        """
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        self.__database = database if database is not None else 'neo4j'
        self.__graph = [] # store the constructed graphs

        try:
            self.__driver = GraphDataScience(self.__uri, auth=(self.__user, self.__pwd), database=self.__database)
        except Exception as e:
            logger.error("Failed to create the driver: %s", e)

    def close(self):
        """Close the python neo4j GDS connection
        """
        if self.__driver is not None:
            self.__driver.close()

    def restart(self):
        """Restart the python neo4j GDS connection
        """
        if self.__driver is not None:
            self.__driver.close()
        try:
            self.__driver = GraphDataScience(self.__uri, auth=(self.__user, self.__pwd), database=self.__database)
        except Exception as e:
            logger.error("Failed to restart the GDS driver: %s", e)

    def query(self, query, params=None, database=None):
        """User provided Cypher query statements for python neo4j driver to use to query database

        Args:
            query (str): user provided Cypher query statements
            parameters (dict, optional): dictionary that provide key/value pairs for query statement to use. Defaults to None.
            db (str, optional): name for database. Defaults to None.

        Returns:
            DataFrame: returned queried results.
        """

        assert self.__driver is not None, "Driver not initialized!"
        response = None

        try:
            response =self.__driver.run_cypher(query, params=params, database=database)
        except Exception as e:
            logger.error("Query failed: %s", e)

        return response

    def project(self, graph_name, node_spec, relationship_spec):
        """Creates a named graph in the catalog for use by algorithms

        Args:
            graph_name (str): graph name
            node_spec (str or dict): Node project, dict option ({nodeLabel: {'properties':[properties]}})
            relationship_spec (str or dict): Relationship projection, dict option ({relationLabel: {'properties':[properties]}})

        Returns:
            graph (Graph object): GDS graph object
            result (pandas.Series): containing metadata from underlying procedure call.
        """
        # exists_result = self.__driver.graph.exists(graph_name)
        # assert exists_result['exists']
        graph, result = self.__driver.graph.project(graph_name, node_spec, relationship_spec)
        self.__graph.append(graph)
        return graph, result


    def load_dataframe(self, graph_name, nodes, relationships, write=False):
        """Constructing a graph from pandas.DataFrames

        Args:
            graph_name (str): Name of the graph to be constructed
            nodes (pandas.DataFrame): one or more dataframes containing node data
            relationships (pandas.DataFrame): one or more dataframes containing relationship data

        Returns:
            graph (Graph object): GDS graph object

        Examples:
            nodes = pandas.DataFrame(
                {
                    "nodeId": [0, 1, 2, 3],
                    "labels":  ["A", "B", "C", "A"],
                    "prop1": [42, 1337, 8, 0],
                    "otherProperty": [0.1, 0.2, 0.3, 0.4]
                }
            )

            relationships = pandas.DataFrame(
                {
                    "sourceNodeId": [0, 1, 2, 3],
                    "targetNodeId": [1, 2, 3, 0],
                    "relationshipType": ["REL", "REL", "REL", "REL"],
                    "weight": [0.0, 0.0, 0.1, 42.0]
                }
            )

        """
        graph = self.__driver.graph.construct(graph_name, nodes, relationships)
        self.__graph.append(graph)
        # if write:
            # self.__driver.graph.nodeProperties.write(graph,list(nodes['labels'].unique()))
            # self.__driver.graph.relationship.write(graph,list(relationships['relationshipType'].unique()))
            # Note: the export can only be used to write the data into a new database
            # self.__driver.graph.export(graph, dbName = self.__database)

        return graph

    def check(self):
        """Print the graph information
        """
        list_graph = self.__driver.graph.list()
        logger.info('List of graphs:')
        logger.info(','.join(list_graph))

        for graph in self.__graph:
            node_count = graph.node_count()
            node_labels = graph.node_labels()
            relationship_count = graph.relationship_count()
            logger.info(f'The graph {graph.name()} has {node_count} nodes')
            logger.info(f'The graph {graph.name()} has labels: {node_labels}')
            logger.info(f'The graph {graph.name()} has {relationship_count} relationships')


    def reset(self):
        """Reset the GDS, delete the graph in the memory
        """
        for graph in self.__graph:
            name = graph.name()
            graph.drop()
            logger.info('Graph with name %s has been removed!', name)

    def centrality(self, method='eigenvector', check=False):
        """Centrality algorithms are used to understand the role or influence of particular nodes in a graph

        Args:
            method (str, optional): centrality algorithm. Defaults to 'eigenvector'.
            'Engenvector centrality' measures the importance or influence of a node based on its connections
            to other nodes in the network.
            'Betweenness centrality' quantifies the importance of a node as a bridge or intermediary
            in the network. It measures how often a node lies on the shortest path between other pairs of nodes
            'Degree centrality' measures the number of connections (edges) a node has in the network
            check (bool, optional): print graph information if True
        """
        centrality_result = None
        for graph in self.__graph:
            if method.lower() == 'eigenvector':
                centrality_result = self.__driver.eigenvector.mutate(graph, maxIterations=100, mutateProperty='eigenvectorCentrality')
                self.__driver.graph.nodeProperties.write(graph, ['eigenvectorCentrality'])
            elif method.lower() == 'betweenness':
                centrality_result = self.__driver.betweenness.mutate(graph, mutateProperty='betweennessCentrality')
                self.__driver.graph.nodeProperties.write(graph, ['betweennessCentrality'])
            elif method.lower() == 'degree':
                centrality_result = self.__driver.degree.mutate(graph, mutateProperty='degreeCentrality')
                self.__driver.graph.nodeProperties.write(graph, ['degreeCentrality'])
            else:
                logger.error("Invalid input for 'method' keyword, available options are: eigenvector, betweenness and degree!")

            if check and centrality_result is not None:
                self.check()
                print(centrality_result.centralityDistribution)

