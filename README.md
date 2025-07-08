# DACKAR
*Digital Analytics, Causal Knowledge Acquisition and Reasoning for Technical Language Processing*

## Installation

### How to install DACKAR libraries?

- Install dependency libraries

```bash
  conda create -n dackar_libs python=3.11

  conda activate dackar_libs

  pip install spacy==3.5 stumpy textacy matplotlib nltk coreferee beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] numpy==1.26 scikit-learn pyspellchecker contextualSpellCheck pandas
```

- Install additional libraries

Library ``neo4j`` is a Python module that is used to communicate with Neo4j database management system,
and ``jupyterlab`` is used to execute notebook examples under ``./examples/`` folder.

```bash

  pip install neo4j jupyterlab
```

- Download language model from spacy (can not use INL network)

```bash
  python -m spacy download en_core_web_lg
  python -m coreferee install en
```

- Install required nltk data for similarity analysis
--------------------------------------------------------

```bash
  python -m nltk.downloader all
```

## Test

### Test functions with ```pytest```

- Run the following command in your command line to install pytest:

```bash
pip install -U pytest
```

- The tests can be run with:

```bash
cd tests
pytest
```

## How to build documentation, such as html, latex and pdf?

### Install Required Libraries

```bash
  pip install sphinx sphinx_rtd_theme nbsphinx sphinx-copybutton sphinx-autoapi
  conda install pandoc
```

### Build HTML

```bash
  cd docs
  make html
  cd _build/html
  python3 -m http.server
```

open your browser to: http://localhost:8000

### Build Latex and PDF

__Sphinx__ uses latex to export the documentation as a PDF file. Thus one needs the basic
latex dependencies used to write a pdf on the system.

```bash
  cd docs
  make latexpdf
  cd _build/latex/
```

The PDF version of DACKAR is located at ``_build/latex/dackar.pdf``
