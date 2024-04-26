# DACKAR
*Digital Analytics, Causal Knowledge Acquisition and Reasoning for Technical Language Processing*

## Configuration for Sphinx doc/conf.py:

```Python
extensions = ['sphinx.ext.intersphinx',
	'sphinx.ext.autodoc',
	'sphinx.ext.doctest',
	'sphinx.ext.todo',
	"sphinx.ext.autodoc.typehints",
	"sphinx.ext.mathjax",
  "sphinx.ext.autosummary",
	"nbsphinx",  # <- For Jupyter Notebook support
	"sphinx.ext.napoleon",  # <- For Google style docstrings
	"sphinx.ext.imgmath",
	"sphinx.ext.viewcode",
	'autoapi.extension',
  'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = [".rst", ".md"]
autoapi_dirs = ['../src']

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -- NBSphinx options
# Do not execute the notebooks when building the docs
nbsphinx_execute = "never"

autodoc_inherit_docstrings = False
```

## How to build html?

```bash
  pip install sphinx sphinx_rtd_theme nbsphinx sphinx-copybutton sphinx-autoapi
  conda install pandoc
  cd doc
  make html
  cd _build/html
  python3 -m http.server
```

open your brower to: http://localhost:8000

## Installation

### How to install DACKAR libraries with spaCy 3.5?

- Install dependency libraries

```bash
  conda create -n dackar_libs python=3.11

  conda activate dackar_libs

  pip install spacy==3.5 textacy matplotlib nltk coreferee beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] numpy scikit-learn pyspellchecker contextualSpellCheck pandas
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

### Different approach when there is an issue with SSLError

- Download language model from spacy
```bash
  Download en_core_web_lg-3.5.0-py3-none-any.whl from https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl

  python -m pip install ./en_core_web_lg-3.5.0-py3-none-any.whl
```

- Download coreferee model:

```bash
  Download from https://github.com/richardpaulhudson/coreferee/tree/master/models/coreferee_model_en.zip

  python -m pip install ./coreferee_model_en.zip
```

- run script DACKAR/nltkDownloader.py to download nltk data:

```bash
  python nltkDownloader.py
```

or check installing_nltk_data_ on how to manually install nltk data.
For this project, the users can also try the following steps:

```bash
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
```

## Old Installation Process

### How to install DACKAR libraries with spaCy 3.1?

- Install dependency libraries

```bash
  conda create -n nlp_libs python=3.9
  conda activate nlp_libs
  pip install spacy==3.1 textacy matplotlib nltk coreferee beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] numpy scikit-learn==1.2.2 pyspellchecker
```

**scikit-learn 1.2.2 is required for quantulum3**

- Download language model from spacy (can not use INL network)

```bash
  python -m spacy download en_core_web_lg
  python -m coreferee install en
```

- Different approach when there is an issue with SSLError

```bash
  Download en_core_web_lg-3.1.0.tar.gz from https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-3.1.0

  python -m pip install ./en_core_web_lg-3.1.0.tar.gz
```

- Download coreferee model:

```bash
  Download from https://github.com/richardpaulhudson/coreferee/tree/master/models/coreferee_model_en.zip

  python -m pip install ./coreferee_model_en.zip
```

- You may need to install stemming for some of unit parsing

```bash
  pip install stemming
```

- Windows machine have issue with pydantic (See https://github.com/explosion/spaCy/issues/12659)

**Installing typing_extensions<4.6**

```bash
  pip install typing_extensions==4.5.*
```

- Required libraries and nltk data for similarity analysis

```bash
  conda install -c conda-forge pandas
  python -m nltk.downloader all
```

- Different approach when there is an issue with SSLError

As a first alternative, the following command can be used:
```bash
  python nltkDownloader.py
```

If not successful, please check (https://www.nltk.org/data.html) on how to manually install nltk data.
For this project, the users can try the following steps:

```bash
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
```

- Required library for preprocessing

```bash
  pip install contextualSpellCheck
```
