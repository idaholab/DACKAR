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

DACKAR uses **uv** as the default environment and dependency manager. A
traditional **conda + pip** install (no uv) is also supported — see
[Alternative: conda + pip](#alternative-conda--pip-no-uv) below. Both paths are
exercised in CI.

### Default: uv

#### 1. Get a Python 3.11 environment

Option A — let uv manage Python (no conda needed):

```bash
uv python install 3.11
```

Option B — use conda just for Python + uv:

```bash
conda create -n dackar python=3.11 uv pip
conda activate dackar
```
Activate the env (`conda activate dackar`) in any new shell before
running the `uv` commands below.

#### 2. Clone and install dependencies

```bash
git clone https://github.com/idaholab/DACKAR.git
cd DACKAR

# Pick the install that matches your workflow:
uv sync                                                # core NLP only
uv sync --group rca --group kg --group nlp-extra --group dev   # full RCA workflow
uv sync --all-groups                                    # everything
```

#### 3. Bootstrap runtime models (one-time)

```bash
uv run python scripts/bootstrap_models.py
```

This downloads the NLTK corpora used for similarity analysis and
retrains the quantulum3 classifier. The `en_core_web_lg` spaCy model
is installed automatically as a project dependency.

### Alternative: conda + pip (no uv)

For users who prefer plain pip. Use Python 3.11 (the editdistance 3.12
build workaround is uv-only).

```bash
conda create -n dackar python=3.11
conda activate dackar

# CPU-only torch, matching uv's routing; omit to get the default PyPI build:
pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cpu

pip install .                              # core (enough to run the test suite)
pip install . --group rca --group kg       # add optional groups (needs pip >= 25.1)

python scripts/bootstrap_models.py
```

With this path, drop the `uv run` prefix from the commands elsewhere in this
README (e.g. run `pytest` and `python scripts/bootstrap_models.py` directly).

### Dependency groups

| Group | Use when |
|---|---|
| _(core, always installed)_ | Running `python -m dackar.main` |
| `nlp-extra` | Optional NLP pipes (pywsd, contextual spell check) |
| `anomaly`   | Using `dackar.anomalies` (matrix-profile / two-sample tests) |
| `kg`        | Loading data into Neo4j via `dackar.knowledge_graph` |
| `viz`       | Word-cloud rendering in `dackar.utils.visualize` |
| `rca`       | Running the AI-enhanced RCA demos under `src/dackar/RCA/` |
| `docs`      | Building Sphinx documentation |
| `dev`       | Running tests and notebook examples |

## Test

The full suite runs on the core dependencies alone — the packages the tests
reach (`stumpy`, `neo4j`, `wordcloud`) are declared in
`[project.dependencies]`, so no dependency group is needed:

```bash
uv sync
```

```bash
uv run pytest tests/                         # full suite
uv run pytest tests/pipelines/test_pipelines.py  # single file
uv run pytest -k temporal                    # by keyword
```

## How to build documentation

### Install dependencies

```bash
uv sync --group docs
# Plus pandoc (system package, not a Python lib):
brew install pandoc          # macOS
# or: sudo apt install pandoc  # Debian/Ubuntu
```

### Build HTML

```bash
cd docs
uv run make html
cd _build/html
python3 -m http.server
```

Open your browser to: http://localhost:8000

### Build LaTeX and PDF

Sphinx uses LaTeX to export documentation as a PDF, so a LaTeX
installation is required on the system.

```bash
cd docs
uv run make latexpdf
```

The PDF is at `docs/_build/latex/dackar.pdf`.
