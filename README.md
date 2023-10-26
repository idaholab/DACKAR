# DACKAR
*Digital Analytics, Causal Knowledge Acquisition and Reasoning*

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
- pip install sphinx, sphinx_rtd_theme, nbsphinx, sphinx-copybutton, sphinx-autoapi
- conda install pandoc
- cd doc
- make html
- cd _build/html
- python3 -m http.server
- open your brower to: http://localhost:8000
