# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on March, 2025

@author: wangc, mandd
"""

# Source:
# https://github.com/cj2001/bite_sized_data_science/tree/main
# https://github.com/neo4j/neo4j-python-driver
# pip install neo4j


from neo4j import GraphDatabase


class Py2Neo:

    def __init__(self, uri, user, pwd):
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

        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        """Close the python neo4j connection
        """
        if self.__driver is not None:
            self.__driver.close()

    def restart(self):
        """Restart the python neo4j connection
        """
        if self.__driver is not None:
            self.__driver.close()
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to restart the driver:", e)

    def create_node(self, label, properties):
        """Create a new graph node

        Args:
            label (str): node label will be used by neo4j
            properties (dict): node attributes
        """
        # label can be "mbse", "anomaly", "measure"
        with self.__driver.session() as session:
            session.execute_write(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        """Create a new graph node

        Args:
            tx (obj): python neo4j active session that can be used to execute queries
            label (str): node label will be used by neo4j
            properties (dict): node attributes
        """
        query = f"MERGE (n:{label} {{ {', '.join([f'{k}: ${k}' for k in properties.keys()])} }})"
        # print(query)
        tx.run(query, **properties)


    def create_relation(self, l1, p1, l2, p2, lr, pr=None):
        """create graph relation

        Args:
            l1 (str): first node label
            p1 (dict): first node attributes
            l2 (str): second node label
            p2 (dict): second node attributes
            lr (str): relationship label
            pr (dict, optional): attributes for relationship. Defaults to None.
        """
        # label (l1/l2), properties (p1/p2), and relation label (lr), relation properties (pr)
        with self.__driver.session() as session:
            session.execute_write(self._create_relation, l1, p1, l2, p2, lr, pr)

    @staticmethod
    def _create_relation(tx, l1, p1, l2, p2, lr, pr):
        """create graph relation

        Args:
            tx (obj): python neo4j active session that can be used to execute queries
            l1 (str): first node label
            p1 (dict): first node attributes
            l2 (str): second node label
            p2 (dict): second node attributes
            lr (str): relationship label
            pr (dict, optional): attributes for relationship. Defaults to None.
        """
        if pr is not None:
            query = f"""
                MERGE (l1:{l1} {{ {', '.join([f'{k}:"{v}"' for k, v in p1.items()])} }})
                MERGE (l2:{l2} {{ {', '.join([f'{k}:"{v}"' for k, v in p2.items()])} }})
                MERGE (l1)-[r:{lr} {{ {', '.join([f'{k}: ${k}' for k in pr.keys()])} }} ]->(l2)
            """
            tx.run(query, **pr)
        else:
            query = f"""
                MERGE (l1:{l1} {{ {', '.join([f'{k}:"{v}"' for k, v in p1.items()])} }})
                MERGE (l2:{l2} {{ {', '.join([f'{k}:"{v}"' for k, v in p2.items()])} }})
                MERGE (l1)-[r:{lr}]->(l2)
            """
            # print(query)
            tx.run(query)

    def find_nodes(self, label, properties=None):
        """Find the node in neo4j graph database

        Args:
            label (str): node label
            properties (dict, optional): node attributes. Defaults to None.

        Returns:
            list: list of nodes
        """
        with self.__driver.session() as session:
            result = session.execute_read(self._find_nodes, label, properties)
            return result

    @staticmethod
    def _find_nodes(tx, label, properties):
        """Find the node in neo4j graph database

        Args:
            tx (obj): python neo4j active session that can be used to execute queries
            label (str): node label
            properties (dict, optional): node attributes. Defaults to None.

        Returns:
            list: list of nodes
        """
        if properties is None:
            query = f"MATCH (n:{label}) RETURN n"
        else:
            query = f"""MATCH (n:{label} {{ {', '.join([f'{k}:"{v}"' for k, v in properties.items()])} }}) RETURN n"""
        result = tx.run(query)
        values = [record.values() for record in result]
        return values

    def load_csv_for_nodes(self, file_path, label, attribute):
        """Load CSV file to create nodes

        Args:
            file_path (str): file path for CSV file, location is relative to 'dbms.directories.import' or 'server.directories.import' in neo4j.conf file
            label (str): node label
            attribute (dict): node attribute from the CSV column names
        """
        with self.__driver.session() as session:
            session.execute_write(self._load_csv_nodes, file_path, label, attribute)

    @staticmethod
    def _load_csv_nodes(tx, file_path, label, attribute):
        query = f"""
        LOAD CSV WITH HEADERS FROM 'file:///{file_path}' AS row
        MERGE (e:{label} {{ {', '.join([f'{k}:row.{v}' for k,v in attribute.items()])} }});
        """
        # print(query)
        tx.run(query)

    # Load csv function to create relations
    def load_csv_for_relations(self, file_path, l1, p1, l2, p2, lr, pr=None):
        """Load CSV file to create node relations

        Args:
            file_path (str): file path for CSV file, location is relative to 'dbms.directories.import' or 'server.directories.import' in neo4j.conf file
            l1 (str): first node label
            p1 (dict): first node attribute from the CSV column names
            l2 (str): second node label
            p2 (dict): second node attribute from the CSV column names
            lr (str): relationship label
            pr (dict, optional): of attributes for relation. Defaults to None.
        """
        # label (l1/l2), properties (p1/p2), and relation label (lr), relation properties (pr)
        with self.__driver.session() as session:
            session.execute_write(self._load_csv_for_relations, file_path, l1, p1, l2, p2, lr, pr)

    @staticmethod
    def _load_csv_for_relations(tx, file_path, l1, p1, l2, p2, lr, pr):
        if pr is not None:
            query = f"""
                LOAD CSV WITH HEADERS FROM 'file:///{file_path}' AS row
                MERGE (l1:{l1} {{ {', '.join([f'{k}:row.{v}' for k,v in p1.items()])} }})
                MERGE (l2:{l2} {{ {', '.join([f'{k}:row.{v}' for k,v in p2.items()])} }})
                MERGE (l1)-[r:{lr} {{ {', '.join([f'{k}: row.{v}' for k,v in pr.items()])} }} ]->(l2)
            """
            tx.run(query)
        else:
            query = f"""
                LOAD CSV WITH HEADERS FROM 'file:///{file_path}' AS row
                MERGE (l1:{l1} {{ {', '.join([f'{k}:row.{v}' for k,v in p1.items()])} }})
                MERGE (l2:{l2} {{ {', '.join([f'{k}:row.{v}' for k,v in p2.items()])} }})
                MERGE (l1)-[r:{lr}]->(l2)
            """
        # print(query)
        tx.run(query)

    def query(self, query, parameters=None, db=None):
        """User provided Cypher query statements for python neo4j driver to use to query database

        Args:
            query (str): user provided Cypher query statements
            parameters (dict, optional): dictionary that provide key/value pairs for query statement to use. Defaults to None.
            db (str, optional): name for database. Defaults to None.

        Returns:
            list: returned list of queried results.
        """

        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None

        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response

    def reset(self):
        """Reset the database, delete all records, use it with care
        """
        with self.__driver.session() as session:
            session.execute_write(self._reset)

    @staticmethod
    def _reset(tx):
        query = 'MATCH (n) DETACH DELETE n;'
        tx.run(query)

    def get_all(self):
        """Get all records from database

        Returns:
            list: list of all records
        """
        with self.__driver.session() as session:
            result = session.execute_write(self._get_all)
            return result

    @staticmethod
    def _get_all(tx):
        query = 'MATCH (n) RETURN n;'
        result = list(tx.run(query))
        # result = [record.values() for record in result]
        return result

