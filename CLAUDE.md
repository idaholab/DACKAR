# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DACKAR (Digital Analytics, Causal Knowledge Acquisition and Reasoning) is a knowledge-management / discovery toolkit for equipment-reliability data. It is **workflow-driven**: a TOML input file selects one of `mbse_workflow`, `anomaly_workflow`, `tlp_workflow`, or `kg_workflow`, each composing a sequence of pipelines that consume raw inputs (MBSE models, numeric anomalies, textual reports) and produce / update a Neo4j knowledge graph whose skeleton is an MBSE model.

Project layout uses a `src/` layout (`src/dackar/...`); `pytest.ini` sets `pythonpath = src` so tests import as `from dackar.<module> ...`.

## Environment setup

Single `pyproject.toml` at repo root managed by **uv** (≥0.4). One committed `uv.lock` resolves the core NLP stack and the heavier RCA stack together — `pyproject.toml` is the source of truth for which versions actually work, not CI.

```bash
uv sync                                                # core only
uv sync --group rca --group kg --group nlp-extra --group dev   # RCA workflow
uv sync --all-groups                                    # everything

uv run python scripts/bootstrap_models.py              # one-time, post-sync
```

Key constraints worth knowing:
- Python 3.10–3.12 is allowed; `.python-version` pins 3.11 for parity with CI.
- spaCy is pinned to 3.5 because coreferee's `en` model and the quantulum3 classifier weren't retrained for newer versions. numpy stays on 1.x.
- The `en_core_web_lg-3.5.0` model is a URL dep in `pyproject.toml`, not a separate `spacy download` step.
- torch is per-OS via `[tool.uv.sources]` (2.9.1 Linux/macOS, 2.8.0 Windows for pytorch issue #166628). Lockfile records both branches.
- The `rca` group's langchain stack resolves to pydantic-v1-compatible versions naturally because spaCy 3.5 and quantulum3 hold the resolver to pydantic <2. Add explicit upper bounds when bumping core NLP pins.
- coreferee 1.4.1 needs `pkg_resources`, so `setuptools<80` and `pip` are pinned in the `nlp-extra` group.

## Running

```bash
# CLI entry point — pass -i explicitly; default path is a stale relative path
uv run python -m dackar.main -i system_tests/ner.toml
```

Input TOML must contain an `[nlp]` table or a `[neo4j]` table (or both); see `system_tests/ner.toml`, `system_tests/causal.toml`, `system_tests/kg.toml` for the three canonical shapes. `WorkflowManager._validate` runs a JSON-schema check (`src/dackar/validate.py`) before anything else, so schema-level errors surface immediately.

## Testing

```bash
uv run pytest tests/                              # full suite
uv run pytest tests/pipelines/test_pipelines.py   # single file
uv run pytest -k temporal                         # by keyword
```

`tests/` is the gated, CI-run suite. `general_tests/` and `system_tests/` are exploratory scripts / TOML fixtures and not run by `pytest` from the repo root. CI runs `uv run pytest tests/` on Ubuntu, macOS, and Windows via the matrix job in `.github/workflows/github-actions.yml`.

## Documentation

```bash
cd docs && make html        # served via `python3 -m http.server` from docs/_build/html
cd docs && make latexpdf    # output at docs/_build/latex/dackar.pdf
```

Requires `sphinx sphinx_rtd_theme nbsphinx sphinx-copybutton sphinx-autoapi` and pandoc (`conda install pandoc`).

## Architecture

### Workflow execution (the spine)

`dackar.main` → `dackar.utils.utils.readToml` → `dackar.workflows.WorkflowManager.WorkflowManager(config).run()`.

`WorkflowManager` is the single orchestrator. Construction inspects the TOML and branches:

- If `[nlp]` present → `initializeNLP()` loads the spaCy model, builds entity patterns from `[nlp.files].opm` (parsed via `utils.opm.OPLparser.OPMobject`) and `[nlp.files].entity` (CSV), builds causal-keyword patterns from `config.nlpConfig['files']['cause_effect_keywords_file']`, sets up preprocessing, and dispatches on `[nlp.analysis].type`:
  - `ner` → `ner()` swaps in custom spaCy pipes per `[nlp.ner]` (mapping in `NERMapping`) and registers the general entity matcher before the stock `ner` pipe.
  - `causal` → `causal()` returns a `CausalSentence` / `CausalPhrase` / `CausalSimple` flow (from `dackar.causal.*`) seeded with both entity and causal patterns.
- If `[neo4j]` present → `initializeNeo4j()` constructs a `Py2Neo` driver and (optionally) wipes the DB; `runNeo4j()` then loads nodes and edges from CSVs as declared in `[[neo4j.node]]` / `[[neo4j.edge]]` arrays.

The result of an NLP run is CSV output (`ner.csv`, `causal_ner_health_status.csv`, `causal_ner_status.csv`, `causal_relation.csv`, `relation_general.csv`) which is exactly what `runNeo4j` expects to load — i.e., the NLP and Neo4j halves are designed to compose via files, not in-memory.

### Module map

- `dackar/pipelines/` — spaCy pipeline components (entity recognizers + custom components like `normEntities`, `aliasResolver`, `anaphorCoref`, `expandEntities`, `mergePhrase`, `pysbdSentenceBoundaries`). Each `*Entity.py` registers a spaCy factory; `CustomPipelineComponents.py` holds the cross-cutting ones. `NERMapping` / `customPipe` dicts in `WorkflowManager.py` are the TOML-key → factory-name registry.
- `dackar/text_processing/` — `Preprocessing` (textacy-based normalize/remove/replace; configured by `[nlp.processing]`), abbreviation expansion, spell checking. `Preprocessing` is conditionally instantiated and applied before `nlp(doc)`.
- `dackar/causal/` — `CausalBase` and three flow strategies (`CausalSentence`, `CausalPhrase`, `CausalSimple`); each is a callable that mutates internal attributes retrievable via `getAttribute('doc' | 'causalRelation' | 'entHS' | 'entStatus' | 'relationGeneral')`.
- `dackar/knowledge_graph/` — `Py2Neo` wraps the Neo4j Python driver with `load_csv_for_nodes` / `load_csv_for_relations`; `KGconstruction.py` and `pygds.py` cover higher-level KG building / graph data science. JSON / TOML schemas for many domains (FMEA, HAZOP, STPA, work orders, condition reports, etc.) live in `knowledge_graph/schemas/`.
- `dackar/utils/opm/OPLparser.py`, `dackar/utils/mbse/LMLparser.py` — parsers that turn external system-model files (OPM HTML, LML) into object / process / attribute lists used to seed entity patterns.
- `dackar/utils/nlp/nlp_utils.py` — `generatePatternList`, `resetPipeline`, `extractNER`; the small surface area that `WorkflowManager` actually depends on.
- `dackar/anomalies/` — matrix-profile / two-sample-test based anomaly detection (`MatrixProfile`, `kernel_two_sample_test`, `t_score`) used by the anomaly workflow.
- `dackar/similarity/` — sentence and synset similarity over WordNet / spaCy vectors.
- `dackar/config/` — default TOML configs (`nlp_config_default.toml`, plus domain variants `_cws`, `_excavator`, `_ler`, `_typical`) and the `cause_effect_keywords_file` referenced at runtime via `dackar.config.nlpConfig`.
- `dackar/RCA/` — large in-progress AI-enhanced root-cause-analysis subsystem with its own orchestrators (Stages A–G in `RCA/orchestrators/`), KG ingest pipelines, NER, summarizers, viz, and per-test-case directories under `RCA/tests/`. Treat it as a semi-independent subpackage; see `RCA/orchestrators/README.md` and `RCA/orchestrators/README_causality.md`.
- `dackar/outage/` — separate outage-uncertainty skeleton (has its own `pyproject.toml`, `README.md`, and `tests/`); domain models, retrieval, uncertainty, schedule risk, adapters.
- `dackar/validate.py` — `validateToml` (JSON schema) is the gate for every WorkflowManager run; modify the schema here when adding new TOML keys.

### Conventions worth knowing

- Entity label / id come from `[nlp.ent]` and are reused as the label/id for both entity patterns and (separately) causal-keyword patterns.
- The pipeline registry pattern (TOML key → factory-name string → `nlp.add_pipe`) means **new pipelines must be registered as a spaCy factory** (see `pipelines/*.py` for the `@Language.factory` calls) **and** added to `NERMapping` in `WorkflowManager.py`.
- spaCy pattern matching uses `attr="LEMMA"` by default — `generatePatternList` lemmatizes the seed terms.
- Logging is centralized: top-level `logging.getLogger('DACKAR')` with `FileHandler('dackar.log')`; submodules use `getLogger('DACKAR.<Submodule>')`.
