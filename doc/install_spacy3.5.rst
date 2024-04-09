============
Installation
============

How to install dependency libraries?
------------------------------------

.. code-block:: bash

  conda create -n dackar_libs python=3.11

  conda activate dackar_libs

  pip install spacy==3.5 textacy matplotlib nltk coreferee beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] numpy scikit-learn spellchecker contextualSpellCheck

  conda install -c conda-forge pandas
.. scikit-learn 1.2.2 is required for quantulum3

Download language model from spacy (can not use INL network)
------------------------------------------------------------

.. code-block:: bash

  python -m spacy download en_core_web_lg

  python -m coreferee install en


Required nltk data for similarity analysis
--------------------------------------------------------

.. code-block:: bash

  python -m nltk.downloader all


Different approach when there is an issue with SSLError
------------------------------------------------------

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
For this project, the users can also try the following steps:

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



