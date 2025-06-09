============
Installation
============

Operating Environments
----------------------

DACKAR can run on Microsoft Windows, Apple OSX and Linux platforms.

Clone DACKAR
------------

The HTTP cloning procedure uses the following clone command:

.. cond-block:: bash

  git clone https://github.com/idaholab/DACKAR.git

The SSH cloning procedure requires the user to create a SSH key (See: https://help.github.com/articles/connecting-to-github-with-ssh/).
Once the SSH key has been created, to clone DACKAR the following command can be executed:

.. cond-block:: bash

  git clone git@github.com:idaholab/DACKAR.git

Install the Required Libraries
------------------------------

.. code-block:: bash

  conda create -n dackar_libs python=3.11

  conda activate dackar_libs

  pip install spacy==3.5 stumpy textacy matplotlib nltk coreferee beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] numpy==1.26 scikit-learn pyspellchecker contextualSpellCheck pandas

..  conda install -c conda-forge pandas
.. scikit-learn 1.2.2 is required for quantulum3

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

  python -m coreferee install en


Required NLTK Data for Similarity Analysis
------------------------------------------

.. code-block:: bash

  python -m nltk.downloader all

Retrain Quantulum3 Classifier (Optional)
----------------------------------------

.. code-block:: bash

  quantulum3-training -s


Different Approach When There is an Issue with SSLError
-------------------------------------------------------

1. Download en_core_web_lg-3.5.0.whl_, then run

.. code-block:: bash

  python -m pip install ./en_core_web_lg-3.5.0.whl

2. Download coreferee_, then run:

.. code-block:: bash

  python -m pip install ./coreferee_model_en.zip

3. run script DACKAR/nltkDownloader.py to download nltk data:

.. code-block:: bash

  python nltkDownloader.py

or check installing_nltk_data_ on how to manually install nltk data.
For this project, users can also try these steps:

.. code-block:: bash

  cd ~
  mkdir nltk_data
  cd nltk_data
  mkdir corpora
  mkdir taggers
  mkdir tokenizers
  Dowload wordnet, averaged_perceptron_tagger, punkt
  cp -r wordnet ~/nltk_data/corpora/
  cp -r averaged_perceptron_tagger ~/nltk_data/taggers/
  cp -r punkt ~/nltk_data/tokenizers

.. _en_core_web_lg-3.5.0.whl: https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl
.. _coreferee: https://github.com/richardpaulhudson/coreferee/tree/master/models/coreferee_model_en.zip
.. _installing_nltk_data: https://www.nltk.org/data.html



