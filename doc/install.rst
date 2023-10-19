============
Installation
============

How to install dependency libraries?
------------------------------------

.. code-block:: bash

  conda create -n dackar_libs python=3.9

  conda activate dackar_libs

  pip install spacy==3.1 textacy matplotlib nltk coreferee beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] numpy scikit-learn==1.2.2

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


.. _en_core_web_lg-3.1.0.tar.gz: https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-3.1.0
.. _coreferee: https://github.com/richardpaulhudson/coreferee/tree/master/models/coreferee_model_en.zip
