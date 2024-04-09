============
Installation
============

How to install dependency libraries?
------------------------------------

.. code-block:: bash

  conda create -n dackar_libs python=3.9

  conda activate dackar_libs

  pip install spacy==3.1 textacy matplotlib nltk coreferee beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] numpy scikit-learn==1.2.2 pyspellchecker

.. scikit-learn 1.2.2 is required for quantulum3

Download language model from spacy (can not use INL network)
------------------------------------------------------------

.. code-block:: bash

  python -m spacy download en_core_web_lg

  python -m coreferee install en

Different approach when there is a issue with SSLError
------------------------------------------------------

1. Download en_core_web_lg-3.1.0.tar.gz_, then run

.. code-block:: bash

  python -m pip install ./en_core_web_lg-3.1.0.tar.gz

2. Download coreferee_, then run:

.. code-block:: bash

  python -m pip install ./coreferee_model_en.zip


You may need to install stemming for some of unit parsing
---------------------------------------------------------

.. code-block:: bash

  pip install stemming

Windows machine have issue with pydantic
----------------------------------------

.. See https://github.com/explosion/spaCy/issues/12659. Installing typing_extensions<4.6

.. code-block:: bash

  pip install typing_extensions==4.5.*

Required libraries and nltk data for similarity analysis
--------------------------------------------------------

.. code-block:: bash

  conda install -c conda-forge pandas
  python -m nltk.downloader all

Different approach when there is a issue with SSLError
------------------------------------------------------

Please check installing_nltk_data_ on how to manually install nltk data.
For this project, the users can try the following steps:

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


Required library for preprocessing
----------------------------------

.. code-block:: bash

  pip install contextualSpellCheck

.. _en_core_web_lg-3.1.0.tar.gz: https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-3.1.0
.. _coreferee: https://github.com/richardpaulhudson/coreferee/tree/master/models/coreferee_model_en.zip
.. _installing_nltk_data: https://www.nltk.org/data.html
