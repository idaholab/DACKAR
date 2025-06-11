# DACKAR
*Digital Analytics, Causal Knowledge Acquisition and Reasoning*

A Knowledge Management and Discovery Tool for Equipment Reliability Data

To improve the performance and reliability of high dependable technological systems such as nuclear power plants, advanced monitoring and health management systems are employed to inform system engineers on observed degradation processes and anomalous behaviors of assets and components. This information is captured in the form of large amount of data which can be heterogenous in nature (e.g., numeric, textual). Such a large amount of available data poses challenges when system engineers are required to parse and analyze them to track the historic reliability performance of assets and components. DACKAR tackles this challenge by providing means to organize equipment reliability data in the form of a knowledge graph. DACKAR distinguish itself from current knowledge graph-based methods in that model-based system engineering (MBSE) models are used to capture system architecture and health and performance data. MBSE models are used as skeleton of a knowledge graph; numeric and textual data elements, once processed, are associated to MBSE model elements. Such a feature opens the door to new data analytics methods designed to identify causal relations between observed phenomena.

DACKAR is structured by a set of workflows where each workflow is designed to process raw data elements (i.e., anomalies, events reported in textual form, MBSE models) and construct or update a knowledge graph. For each workflow, the user can specify the sequence of pipelines that are designed to perform specific processing actions on the raw data or the processed data within the same workflow. Specific guidelines on the formats of the raw data are provided. In addition, within the same workflow, a specific data-object is defined; in this respect, each pipeline is tasked to either process portion of the defined data-object or create knowledge graph data. The available workflows are:
* mbse_workflow: Workflow to process system and equipment MBSE models
* anomaly_workflow:	Workflow to process numeric data and anomalies
* tlp_workflow: Workflow to process textual data
* kg_workflow: Workflow to construct and update knowledge graphs


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

### Test functions with ```__pytest__```

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
