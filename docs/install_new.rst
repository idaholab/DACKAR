============
Installation
============

Operating Environments
----------------------

DACKAR can run on Microsoft Windows, Apple OSX and Linux platforms.

Clone DACKAR
------------

The HTTP cloning procedure uses the following clone command:

.. code-block:: bash

  git clone https://github.com/idaholab/DACKAR.git

The SSH cloning procedure requires the user to create a SSH key (See: https://help.github.com/articles/connecting-to-github-with-ssh/).
Once the SSH key has been created, to clone DACKAR the following command can be executed:

.. code-block:: bash

  git clone git@github.com:idaholab/DACKAR.git

Install the Required Libraries with Python 3.11
-----------------------------------------------

.. code-block:: bash

  conda create -n dackar_libs python=3.11

  conda activate dackar_libs

  pip install textacy spacy==3.8.11 stumpy matplotlib nltk==3.8.1 beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] scikit-learn pyspellchecker contextualSpellCheck pandas wordcloud jsonschema toml openai langchain langchain_openai langchain_community langchain-core ollama langchain-ollama cytoolz

.. library conflicts for spacy 3.8: textacy  coreferee
.. fix torch to 2.8.0 for windows

Install the Required Libraries with Python 3.12
-----------------------------------------------

.. code-block:: bash

  conda create -n dackar_libs python=3.12

  conda activate dackar_libs

  pip install torch spacy==3.8.11 stumpy matplotlib nltk==3.8.1 beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] scikit-learn pyspellchecker  pandas wordcloud jsonschema toml openai langchain langchain_openai langchain_community langchain-core ollama langchain-ollama cytoolz

.. library conflicts for spacy 3.8: textacy,  coreferee, contextualSpellCheck
.. fix torch to 2.8.0 for windows

Install the Required Libraries with Python 3.13
-----------------------------------------------

.. code-block:: bash

  conda create -n dackar_libs python=3.13

  conda activate dackar_libs

  pip install torch spacy==3.8.11 stumpy matplotlib nltk==3.8.1 beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] scikit-learn pyspellchecker  pandas wordcloud jsonschema toml openai langchain langchain_openai langchain_community langchain-core ollama langchain-ollama cytoolz

.. library conflicts for spacy 3.8: textacy,  coreferee, contextualSpellCheck
.. fix torch to 2.8.0 for windows


Install the Required Libraries with Python 3.14
-----------------------------------------------

.. code-block:: bash

  conda create -n dackar_libs python=3.14

  conda activate dackar_libs

  pip install torch spacy==3.8.11 stumpy matplotlib nltk==3.8.1 beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] scikit-learn pyspellchecker  pandas wordcloud jsonschema toml openai langchain langchain_openai langchain_community langchain-core ollama langchain-ollama cytoolz

.. library conflicts for spacy 3.8: textacy,  coreferee, contextualSpellCheck
.. fix torch to 2.8.0 for windows


Install Additional Libraries
----------------------------

Library ``neo4j`` is a Python module that is used to communicate with Neo4j database management system,
and ``jupyterlab`` is used to execute notebook examples under ``./examples/`` folder.

.. code-block:: bash

  pip install neo4j jupyterlab

Download Language Model from spaCy
----------------------------------

.. code-block:: bash

  python -m spacy download en_core_web_lg


Required NLTK Data for Similarity Analysis
------------------------------------------

.. code-block:: bash

  python -m nltk.downloader punkt wordnet averaged_perceptron_tagger brown

Retrain Quantulum3 Classifier (Optional)
----------------------------------------

.. code-block:: bash

  quantulum3-training -s




