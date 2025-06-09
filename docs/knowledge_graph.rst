==========================================
Knowledge Graph Construction Through Neo4j
==========================================

Install Additional Required Libraries
-------------------------------------

In order to communicate with neo4j_ Desktop graph database, the neo4j_python_driver_ need to be installed:

.. code-block:: bash

  pip install neo4j


Connect to Neo4j via DACKAR Py2Neo Module
-----------------------------------------

* Set up Neo4j import folder path

  This directory is used to store data files intended for bulk import
  operations. The location of this directory can be configured in the Neo4j configuration file (``neo4j.conf``).
  By default, the import directory is located at: ``$NEO4J_HOME/import``. Here, ``$NEO4J_HOME`` represents
  the root directory of your Neo4j installation. To change the import directory, you can modify the
  ``dbms.directories.import`` setting in the ``neo4j.conf`` file. For example:

  .. code-block:: bash
    dbms.directories.import=/path/to/your/import/directory

  Make sure to replace ``/path/to/your/import/directory`` with the actual path you want to use for your import files.
  After making changes to the ``neo4j.conf`` file, you will need to restart the Neo4j server for the changes to take effect.

* Alternative way to set up Neo4j import folder path

  Open the settings, and replace ``dbms.directories.import`` with the actual path you want to use for your import files.

  .. image:: ./pics/neo4j_settings.png
    :width: 600
    :alt: Set up Neo4j import folder path

* Start Neo4j desktop DBMS, set password if needed, this password will be used in Python driver to connect to
  DBMS database.

  .. image:: ./pics/neo4j.png
    :width: 600
    :alt: Neo4j Desktop Graph Database Management System

* Load DACKAR ``Py2Neo`` module to load ``csv`` files into Neo4j DBMS.

  .. code-block:: python

    from dackar.knowledge_graph.py2neo import Py2Neo
    # Create python to neo4j driver
    uri = "neo4j://localhost:7687" # for a cluster
    pwd = "123456789" # user need to provide the DBMS database password
    py2neo = Py2Neo(uri=uri, user='neo4j', pwd=pwd)

    # clean up the DBMS
    py2neo.reset()

    # Load node data
    file_path = './test_nodes.csv'
    # node label
    label = 'MBSE'
    # node attributes: keys, values are corresponding to neo4j node attributes and csv column names, respectively
    attribute = {'nodeId':'nodeId', 'label':'label', 'ID':'ID', 'type':'type'}
    # API to load csv files to create nodes in neo4j
    py2neo.load_csv_for_nodes(file_path, label, attribute)
    # Load relationship data
    file_path = 'test_edges.csv'
    l1='MBSE' # label for source node
    # node attributes: keys, values are corresponding to neo4j node attributes and csv column names, respectively
    p1={'nodeId':'sourceNodeId'}
    # label for second node
    l2='MBSE'
    # node attributes: keys, values are corresponding to neo4j node attributes and csv column names, respectively
    p2 ={'nodeId':'targetNodeId'}
    # label for neo4j edge
    lr = 'MBSE_link'
    # edge attributes: keys, values are corresponding to neo4j node attributes and csv column names, respectively
    pr = {'prop':'type'}
    # API to load csv files to create relations in neo4j
    py2neo.load_csv_for_relations(file_path, l1, p1, l2, p2, lr, pr)

.. _neo4j: https://neo4j.com/download/
.. _neo4j_python_driver: https://github.com/neo4j/neo4j-python-driver
