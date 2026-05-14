============
Installation
============

Operating Environments
----------------------

DACKAR runs on Microsoft Windows, Apple macOS, and Linux. Python 3.10–3.12 is supported; CI tests on 3.11.

1. Install uv
-------------

.. code-block:: bash

  # Linux / macOS
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows (PowerShell)
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

  # or via Homebrew on macOS
  brew install uv

2. Clone DACKAR
---------------

.. code-block:: bash

  git clone https://github.com/idaholab/DACKAR.git
  cd DACKAR

For SSH cloning see https://help.github.com/articles/connecting-to-github-with-ssh/.

3. Install Dependencies
-----------------------

Pick the install profile that matches your workflow:

.. code-block:: bash

  uv sync                                                       # core only
  uv sync --group rca --group kg --group nlp-extra --group dev  # full RCA workflow
  uv sync --all-groups                                          # everything

Available optional dependency groups:

============  =========================================================
Group         Use when
============  =========================================================
nlp-extra     Optional NLP pipes (coreferee, pywsd, contextual spell check)
anomaly       Using ``dackar.anomalies`` (matrix-profile, two-sample tests)
kg            Loading data into Neo4j via ``dackar.knowledge_graph``
viz           Word-cloud rendering in ``dackar.utils.visualize``
rca           Running the AI-enhanced RCA demos under ``src/dackar/RCA/``
docs          Building Sphinx documentation
dev           Tests and notebook examples
============  =========================================================

4. Bootstrap Runtime Models
---------------------------

.. code-block:: bash

  uv run python scripts/bootstrap_models.py

This downloads coreferee's English model (if ``nlp-extra`` is installed),
the NLTK corpora used by similarity analysis, and retrains the
quantulum3 classifier. The ``en_core_web_lg`` spaCy model is installed
automatically as a project dependency.

Behind a Corporate SSL Proxy
----------------------------

If model downloads fail with SSL errors, pass ``--insecure-ssl``:

.. code-block:: bash

  uv run python scripts/bootstrap_models.py --insecure-ssl

This disables HTTPS certificate verification for the bootstrap downloads only.

Running DACKAR
--------------

.. code-block:: bash

  uv run python -m dackar.main -i system_tests/ner.toml
  uv run pytest tests/
