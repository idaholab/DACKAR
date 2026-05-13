# uv Package Management Migration — Design

**Status:** Approved
**Date:** 2026-05-13
**Author:** Mengnan Li
**Branch:** `uv_env_management`

## Problem

DACKAR currently installs via conda + a long hand-rolled `pip install` line (see `.github/workflows/github-actions.yml`). `pyproject.toml` is declarative but not authoritative — CI is the source of truth for which versions actually work. Running the RCA demos in `src/dackar/RCA/demos/` triggers package conflicts: the RCA stack pulls in `langchain-chroma`, `langchain-community`, `langchain-core`, `chromadb`, `pdfplumber`, `panels`, `streamlit`, etc., none of which are declared in `pyproject.toml`, and several of which clash with the spaCy-3.5-era pins (numpy 1.26, pydantic v1) that the core NLP path requires.

## Goals

1. One authoritative dependency manifest (`pyproject.toml`) with one committed lockfile (`uv.lock`).
2. The RCA dependency set resolved together with the core set, so any conflict surfaces at lock time, not at import time.
3. A single command to install (`uv sync`) and a single command to bootstrap models (`uv run python scripts/bootstrap_models.py`).
4. CI installs that take seconds, not minutes, and don't depend on conda.

## Non-goals (deferred)

- Modernizing spaCy / numpy / quantulum3 / coreferee. These pins are carried forward as-is.
- Promoting `src/dackar/outage/` to a uv workspace member. Its existing standalone `pyproject.toml` keeps working as a skeleton.
- Adding pre-commit, ruff, mypy, or any other tooling.
- Editing `src/dackar/RCA/` source code. If an RCA demo still fails after migration for non-packaging reasons, that is a separate change.

## Approach

Single root `pyproject.toml`. Core dependencies in `[project.dependencies]`; everything else split into PEP 735 `[dependency-groups]`. One `uv.lock` resolves all groups together. uv workspaces are not used in this migration.

If `uv lock` cannot resolve all groups together (likely failure: langchain's pydantic-v2 requirements vs. spaCy 3.5's transitive pydantic-v1 requirement), the remediation is to pin the langchain trio and `chromadb` to their last pydantic-v1-compatible releases, with inline comments explaining the constraint. If even that cannot resolve, the migration stops at the Phase 1 gate and the architecture is re-opened (workspace or split projects). This is documented as a fallback, not a planned step.

## Project layout

```
pyproject.toml                  # single root manifest (replaces setuptools+conda flow)
uv.lock                         # committed
.python-version                 # 3.11 (CI parity); pyproject allows 3.10–3.12
scripts/
  bootstrap_models.py           # post-sync runtime artifact downloads
src/dackar/                     # unchanged
src/dackar/outage/pyproject.toml  # unchanged, standalone, not a workspace member
```

## Dependency groups

`requires-python = ">=3.10,<3.13"`. Matches available spaCy 3.5 wheels and existing outage skeleton.

| Group | Contents | Trigger to install |
|---|---|---|
| `[project.dependencies]` (core) | `spacy==3.5.*`, `en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl`, `numpy>=1.26,<2`, `pandas`, `pysbd`, `quantulum3[classifier]`, `scikit-learn`, `textacy`, `numerizer`, `networkx`, `nltk==3.8.1`, `beautifulsoup4`, `matplotlib`, `plotly`, `jsonschema`, `jsonpointer`, `python-dateutil`, `openpyxl`, `xlrd`, `toml`, `tomli; python_version<'3.11'` | Always — needed to run `python -m dackar.main` |
| `nlp-extra` | `coreferee`, `contextualSpellCheck`, `pywsd`, `autocorrect`, `pyspellchecker` | When using optional NLP pipes (the source already guards their imports) |
| `anomaly` | `stumpy` | When using `dackar.anomalies` (sklearn is already in core via quantulum3) |
| `kg` | `neo4j`, `graphdatascience` | When running Neo4j workflows |
| `viz` | `wordcloud` | When using `utils/visualize.py` word-cloud rendering |
| `rca` | `langchain-chroma`, `langchain-community`, `langchain-core`, `chromadb`, `pdfplumber`, `panels`, `streamlit`, `nbformat`, `requests` | When running `src/dackar/RCA/demos` |
| `docs` | `sphinx`, `sphinx_rtd_theme`, `nbsphinx`, `sphinx-copybutton`, `sphinx-autoapi` | When building documentation |
| `dev` | `pytest`, `jupyterlab` | Tests and example notebooks |

### Torch handling

CI today installs `torch==2.9.1` on Linux and `torch==2.8.0` on Windows. This is encoded as a `[tool.uv.sources]` entry with platform markers so the lockfile records the per-OS resolution. No conditional CI step is required.

### Install patterns

```bash
uv sync                                                # core only
uv sync --group rca --group nlp-extra --group kg      # RCA demos
uv sync --all-groups                                   # everything (CI)
```

## Model bootstrap

`scripts/bootstrap_models.py` is one idempotent Python script invoked once after `uv sync`. It handles only what uv cannot lock:

1. `coreferee install en` — skipped if `coreferee` is not importable (i.e., `nlp-extra` group not installed).
2. NLTK corpora: `punkt`, `wordnet`, `averaged_perceptron_tagger`, `brown` — same set CI downloads today.
3. `quantulum3-training -s` — retrains the classifier against the installed scikit-learn version.

Each step short-circuits if already present. The SSL-tolerant context trick from the existing `nltkDownloader.py` is absorbed here and guarded by an `--insecure-ssl` flag.

The spaCy `en_core_web_lg` model is **not** downloaded by this script — it is a URL dependency in `[project.dependencies]`, so `uv sync` installs it.

Flags:
- `--only {coreferee,nltk,quantulum}` — run a single step (useful for CI partial environments).
- `--insecure-ssl` — apply the `nltkDownloader.py` SSL workaround.

## CI workflow

Each of the three OS jobs in `.github/workflows/github-actions.yml` is rewritten to:

```yaml
- uses: actions/checkout@v4
- uses: astral-sh/setup-uv@v6
  with:
    enable-cache: true
- uses: actions/setup-python@v5
  with:
    python-version: "3.11"
- run: uv sync --all-groups
- run: uv run python scripts/bootstrap_models.py
- run: uv run pytest tests/
```

Removed: `conda-incubator/setup-miniconda@v3`, `conda create -n dackar_libs python=3.11`, the long `pip install spacy==3.5 ...` line, the secondary `pip install neo4j jupyterlab pytest`, the per-OS hand-rolled `pip install torch==...`, the `pip install https://...en_core_web_lg-3.5.0...whl` line, all `conda init <shell> && source <rc> && conda activate ...` boilerplate (8 occurrences), and the `use-only-tar-bz2: true` cache hack.

Concurrency / cancel-in-progress blocks at the top of the workflow are unchanged.

Expected install-step time: ~30–60s (down from ~3–5 minutes). Bootstrap step remains the slowest at ~1–2 minutes (model downloads, not packaging).

The Windows torch 2.8 pin (currently flagged in the workflow with `Fix PyTorch to 2.8.0 due to issue: https://github.com/pytorch/pytorch/issues/166628`) is carried over via `[tool.uv.sources]` with a Windows marker — not silently dropped.

## Documentation changes

**`README.md`** — `## Installation` section rewritten to the three-command sequence:

```bash
# 1. install uv (one-time, per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh   # or `brew install uv`

# 2. clone + install
git clone https://github.com/idaholab/DACKAR && cd DACKAR
uv sync                                            # core only
# or: uv sync --group rca --group kg --group nlp-extra --group dev
# or: uv sync --all-groups

# 3. one-time model bootstrap
uv run python scripts/bootstrap_models.py
```

Runtime examples follow:

```bash
uv run python -m dackar.main -i system_tests/ner.toml
uv run pytest tests/
```

A "What's in each group?" table mirrors the table above so users pick the right groups.

`## Test` section: `pip install -U pytest` removed; `cd tests && pytest` → `uv run pytest tests/`.

`## How to build documentation` section: `pip install sphinx ...` → `uv sync --group docs`. `conda install pandoc` line gets a `brew install pandoc` / `apt install pandoc` alternative.

**`CLAUDE.md`** — "Environment setup" and "Running" sections rewritten; the warning that "`pyproject.toml` is not the source of truth for installation" is reversed.

**`docs/install.rst`** and **`docs/install_spacy3.5.rst`** — mirror the new README install flow.

**`src/dackar/outage/README.md`** — "Create a virtual environment and install dependencies you choose" replaced with a pointer to `uv sync` and a `TODO: populate outage group deps` note. Outage subpackage deps are not invented in this migration.

**Untouched:** `tests/readme_pytest.md` (pytest pythonpath config, unaffected), `general_tests/README.md` (one-line stub).

**Deleted:** `nltkDownloader.py` (logic absorbed into bootstrap script).

## Migration sequence

### Phase 1 — Local resolution (gating)

1. Write root `pyproject.toml` with core deps + all groups. No `uv.lock` committed yet.
2. Add `.python-version` (`3.11`) and `scripts/bootstrap_models.py`.
3. Run `uv lock`. **Gate.** If it fails:
   - Pin `langchain-core`, `langchain-community`, `langchain-chroma` to the newest pydantic-v1-compatible releases (most likely `<0.2` line for langchain-core, mid-2024 releases).
   - If still failing, pin `chromadb` similarly.
   - Document chosen pins with inline comments explaining the spaCy-3.5 → pydantic-v1 constraint.
   - If no combination resolves, **stop here** and re-open the brainstorm; this design must be revised before proceeding.
4. `uv sync --all-groups` → `uv run python scripts/bootstrap_models.py` → `uv run pytest tests/` must pass.
5. `uv sync --group rca --group nlp-extra --group kg` → run one RCA demo that previously failed → must execute past the import phase.

### Phase 2 — CI and docs (mechanical)

6. Commit `uv.lock`. Rewrite `.github/workflows/github-actions.yml` per the CI section. Push, watch all three OS jobs go green.
7. Rewrite `README.md`, `CLAUDE.md`, `docs/install.rst`, `docs/install_spacy3.5.rst`, `src/dackar/outage/README.md`.
8. Delete `nltkDownloader.py`.

### Phase 3 — Cleanup

9. Try `uv build`. If the default backend (`hatchling`) works with the `src/` layout, drop the setuptools-only `[build-system]`. Otherwise keep setuptools.
10. Final scrub: `rg -i conda` and `rg dackar_libs` must return nothing in tracked files.

## Rollback

- **Phase 1 failure** — discard the branch. CI, docs, and source untouched. Re-brainstorm with workspace (Approach B) or split-project (Approach C) options.
- **Phase 2 failure (CI red)** — keep `pyproject.toml` + `uv.lock`; revert the workflow file. Ship the local-only migration; CI follow-up later.
- **Post-merge regression** — single-PR revert. Clean because `src/dackar/*` source is not touched.

## Success criteria

- `uv sync --all-groups` succeeds on Linux, macOS, and Windows.
- `uv run pytest tests/` matches the previous CI pass/fail set exactly — no test-count drift, no new skips.
- One RCA demo that previously failed with a package conflict runs past import.
- CI green on all three OSes.
- `grep -ri conda .` returns nothing in tracked files.

## Open question carried into implementation

Whether `nlp-extra` (coreferee, contextualSpellCheck, pywsd) belongs in core. CI installs them unconditionally today, but the source code already guards their imports. Default for this migration: keep them optional (group `nlp-extra`). If CI test failures show the test suite implicitly assumes they are present, promote them to core in the same PR.
