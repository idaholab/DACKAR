# DACKAR
Digital Analytics, Causal Knowledge Acquisition and Reasoning

# How to build html
- pip install sphinx
- pip install sphinx_rtd_theme
- pip install nbsphinx (for notebook support)
- cd doc
- make html
- cd _build/html
- python3 -m http.server
- open your brower to: http://localhost:8000

# Use following to generate API

- pip install sphinx-autoapi
## Add and configure AutoAPI in your project/doc's conf.py
- extensions.append('autoapi.extension')
- autoapi_dirs = ['../src']

## to enable Jupyter notebooks inside autodoc, install
- conda install pandoc

