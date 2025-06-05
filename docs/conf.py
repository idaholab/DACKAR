# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
from datetime import datetime
from pathlib import Path
import sys
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# sys.path.insert(0, os.path.abspath('../src'))
# from dackar import __version__
__version__ = '0.1'


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DACKAR'
copyright = 'Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED'
author = 'Congjian Wang, Diego Mandelli, Joshua J. Cogliati'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",  # <- For Google style docstrings
    'sphinx.ext.intersphinx',
	'sphinx.ext.doctest',
	'sphinx.ext.todo',
	"sphinx.ext.autodoc.typehints",
	"sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
	"nbsphinx",  # <- For Jupyter Notebook support
	"sphinx.ext.imgmath",
	"sphinx.ext.viewcode",
	'autoapi.extension',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = [".rst", ".md"]
autoapi_dirs = ['../src']
# autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -- Options for Apidoc
# This can be uncommented to "refresh" the api .rst files.
"""
import os

def run_apidoc(app) -> None:
    '''Generage API documentation'''
    import better_apidoc

    better_apidoc.APP = app
    better_apidoc.main([
        'better-apidoc',
        '-t',
        os.path.join('docs', '_templates'),
        '--force',
        '--separate',
        '-o',
        os.path.join('docs', 'modules'),
        os.path.join('dackar'),
    ])


def setup(app) -> None:
    app.connect('builder-inited', run_apidoc)
# """


def copy_notebooks() -> None:
    for filename in Path("../examples").glob("*.ipynb"):
        shutil.copy2(str(filename), "notebooks")
    for filename in Path("../examples/images").glob("*.png"):
        shutil.copy2(str(filename), "notebooks/images")

copy_notebooks()


# -- NBSphinx options
# Do not execute the notebooks when building the docs
nbsphinx_execute = "never"

autodoc_inherit_docstrings = False

latex_engine = "xelatex"
latex_elements = {
    'printindex': r'\def\twocolumn[#1]{#1}\printindex',
}
