# uv Package Management Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DACKAR's conda+pip installation flow with a single uv-managed project (one `pyproject.toml`, one `uv.lock`) that resolves the core NLP stack and the RCA stack together, eliminating the conflicts encountered when running `src/dackar/RCA/demos/`.

**Architecture:** Single root `pyproject.toml`. Core deps in `[project.dependencies]`; optional features split into PEP 735 `[dependency-groups]` (`nlp-extra`, `anomaly`, `kg`, `viz`, `rca`, `docs`, `dev`). One committed `uv.lock` resolves all groups together. Post-sync model bootstrapping (coreferee, NLTK corpora, quantulum3 classifier) lives in `scripts/bootstrap_models.py`. CI replaces conda+pip with `astral-sh/setup-uv@v6` + `uv sync --all-groups`.

**Tech Stack:** uv (≥0.4 for PEP 735 dependency groups), Python 3.10–3.12 (CI pins 3.11), spaCy 3.5 + `en_core_web_lg` 3.5.0 (URL dep), numpy 1.26, langchain (rca group; version pinned by lock resolution).

**Spec:** [docs/superpowers/specs/2026-05-13-uv-package-management-design.md](../specs/2026-05-13-uv-package-management-design.md)

---

## File Structure

**Created:**
- `pyproject.toml` — replaces the existing one at repo root; single authoritative manifest
- `uv.lock` — committed lockfile
- `.python-version` — `3.11`; was previously gitignored
- `scripts/bootstrap_models.py` — idempotent post-sync model downloader
- `scripts/__init__.py` — empty; makes `scripts` importable for tests
- `tests/scripts/test_bootstrap_models.py` — unit test for bootstrap script helpers

**Modified:**
- `.github/workflows/github-actions.yml` — three OS jobs rewritten to uv-based steps
- `.gitignore` — remove `.python-version` line
- `README.md` — `## Installation`, `## Test`, `## How to build documentation` sections
- `CLAUDE.md` — Environment setup, Running, Testing sections; reverse the "pyproject.toml is not source of truth" warning
- `docs/install.rst` — replace conda flow with uv flow
- `docs/install_spacy3.5.rst` — replace conda flow with uv flow
- `src/dackar/outage/README.md` — point at `uv sync`; add TODO note about populating outage group

**Deleted:**
- `nltkDownloader.py` — logic absorbed into bootstrap script

---

# Phase 1 — Local Resolution (Gating)

> **Gate at Task 5:** If `uv lock` cannot resolve all groups together even after the documented remediation, **stop** and re-open the design. Do not proceed to Phase 2.

---

### Task 1: Set up uv and prepare the workspace

**Files:**
- Modify: `.gitignore` (remove `.python-version` line)
- Create: `.python-version`

- [ ] **Step 1: Verify uv is installed**

Run: `uv --version`

Expected: `uv 0.4.x` or newer (must be ≥0.4 for PEP 735 dependency-groups support).

If missing, install:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # Linux/macOS
# or: brew install uv
# or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- [ ] **Step 2: Remove `.python-version` from `.gitignore`**

Open `.gitignore` and delete the line containing `.python-version` (preserve `.venv` and `venv/`).

Verify:
```bash
grep "^\.python-version$" .gitignore
```
Expected: no output (line is gone).

- [ ] **Step 3: Pin the Python version**

```bash
echo "3.11" > .python-version
```

Verify:
```bash
cat .python-version
```
Expected:
```
3.11
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore .python-version
git commit -m "build: pin Python 3.11 via .python-version

Preparation for uv migration. .python-version is now tracked so all
contributors use the same Python interpreter version."
```

---

### Task 2: Write initial `pyproject.toml` with core dependencies only

**Files:**
- Modify: `pyproject.toml` (replace entirely)

Goal: get `uv lock` succeeding on a minimal manifest before introducing any optional groups. If core itself doesn't resolve, no point adding groups on top.

- [ ] **Step 1: Replace `pyproject.toml` with core-only content**

Overwrite `pyproject.toml` with:

```toml
[project]
name = "dackar"
version = "0.1.0"
description = "Digital Analytics, Causal Knowledge Acquisition and Reasoning"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.10,<3.13"
authors = [
  { name = "Diego Mandelli", email = "diego.mandelli@inl.gov" },
  { name = "Congjian Wang",  email = "congjian.wang@inl.gov" },
  { name = "Joshua J. Cogliati", email = "joshua.cogliati@inl.gov" },
]
classifiers = [
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: Apache Software License",
  "Operating System :: OS Independent",
  "Topic :: Scientific/Engineering",
]

dependencies = [
  # Core NLP — exact versions match CI's known-good install
  "spacy==3.5.*",
  "en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl",
  "numpy>=1.26,<2",
  "pandas",
  "pysbd",
  "quantulum3[classifier]",
  "scikit-learn",
  "textacy",
  "numerizer",
  "networkx",
  "nltk==3.8.1",
  "beautifulsoup4",
  "matplotlib",
  "plotly",
  # Config / validation
  "jsonschema",
  "jsonpointer",
  "python-dateutil",
  "toml",
  "tomli ; python_version < '3.11'",
  # Excel / spreadsheet I/O used in utils
  "openpyxl",
  "xlrd",
]

[project.urls]
Homepage = "https://github.com/idaholab/DACKAR"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Verify `uv lock` succeeds on core**

```bash
uv lock
```

Expected: completes with `Resolved N packages in <time>` and produces a `uv.lock` file. **Do not commit `uv.lock` yet** — we'll commit it after all groups are added.

If `uv lock` fails here, the failure is in core itself (likely the spaCy URL wheel, numpy 1.26 ceiling, or quantulum3[classifier] interaction). Diagnose with `uv lock -v` before continuing.

- [ ] **Step 3: Commit pyproject.toml (lockfile not yet)**

```bash
git add pyproject.toml
git commit -m "build: rewrite pyproject.toml with core dackar dependencies

First step of uv migration. Core dependencies in [project.dependencies],
requires-python pinned to >=3.10,<3.13 to match available spaCy 3.5
wheels. Lockfile committed in a later step once all groups are added."
```

---

### Task 3: Add non-RCA optional groups

**Files:**
- Modify: `pyproject.toml`

These six groups (`nlp-extra`, `anomaly`, `kg`, `viz`, `docs`, `dev`) are all independent of one another and unlikely to introduce a resolution conflict with core. Add them together so the `rca` group (Task 5) is the only remaining unknown.

- [ ] **Step 1: Append `[dependency-groups]` section to `pyproject.toml`**

Add at the end of `pyproject.toml`:

```toml
[dependency-groups]
# Optional NLP pipes. coreferee and pywsd are already guarded by lazy/conditional
# imports in src/dackar/pipelines/CustomPipelineComponents.py and similarity/synsetUtils.py
nlp-extra = [
  "coreferee",
  "contextualSpellCheck",
  "pywsd",
  "autocorrect",
  "pyspellchecker",
]

# Anomaly detection (dackar.anomalies). scikit-learn already pulled in by core via quantulum3[classifier].
anomaly = [
  "stumpy",
]

# Neo4j knowledge-graph backend (dackar.knowledge_graph).
kg = [
  "neo4j",
  "graphdatascience",
]

# Word-cloud rendering in dackar.utils.visualize.
viz = [
  "wordcloud",
]

# Documentation build (cd docs && make html).
docs = [
  "sphinx",
  "sphinx_rtd_theme",
  "nbsphinx",
  "sphinx-copybutton",
  "sphinx-autoapi",
]

# Tests and notebook examples.
dev = [
  "pytest",
  "jupyterlab",
]
```

- [ ] **Step 2: Verify `uv lock` still resolves**

```bash
uv lock
```

Expected: completes successfully. If it fails, the failing group is named in the error — pin offending packages with comments explaining the constraint, then re-lock.

- [ ] **Step 3: Verify each group can be installed in isolation**

```bash
uv sync --group docs --no-default-groups
uv sync --group dev --no-default-groups
uv sync                       # back to core only
```

Expected: each command completes; the `--no-default-groups` ones produce a venv containing only the group's packages plus core.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add non-RCA dependency groups (nlp-extra, anomaly, kg, viz, docs, dev)

Six optional groups for features that don't all need to be installed
together. Each maps to a distinct subpackage or use case; users opt in
via 'uv sync --group <name>'. The RCA group is added separately because
its langchain stack is the main resolution-risk surface."
```

---

### Task 4: Encode the per-OS torch pin via `[tool.uv.sources]`

**Files:**
- Modify: `pyproject.toml`

The current CI installs `torch==2.9.1` on Linux and `torch==2.8.0` on Windows (the Windows pin is documented in the workflow file as a fix for pytorch issue #166628). uv records per-platform resolutions in the lockfile via source markers, so CI doesn't need OS-specific install steps.

- [ ] **Step 1: Add torch as a core dependency and configure the platform-specific source**

Edit `pyproject.toml`. In `[project] dependencies`, add at the end of the list:

```toml
  "torch",
```

Then append a new section after `[tool.setuptools.packages.find]`:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "platform_system != 'Windows'" },
  { index = "pytorch-cpu-win", marker = "platform_system == 'Windows'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu-win"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Then constrain torch versions per OS by adding to `[project] dependencies`:

```toml
  "torch==2.9.1 ; platform_system != 'Windows'",
  "torch==2.8.0 ; platform_system == 'Windows'",
```

(Remove the bare `"torch",` line you added a moment ago — it's replaced by these two marker-conditional pins. Keep both `[tool.uv.sources]` and the two `[[tool.uv.index]]` blocks; the explicit-index pattern is required to avoid uv pulling torch from PyPI when one of those URL wheels is preferable.)

- [ ] **Step 2: Verify `uv lock` resolves on the current host**

```bash
uv lock
```

Expected: completes. On macOS/Linux, the Linux marker resolves to torch 2.9.1; on Windows, torch 2.8.0. uv records both branches in the lockfile so cross-platform CI works from one lockfile.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: pin torch per OS via [tool.uv.sources]

Carries the existing CI behaviour (torch 2.9.1 Linux/macOS, 2.8.0 Windows
per pytorch issue #166628) into the lockfile so CI no longer needs an
OS-conditional install step."
```

---

### Task 5: **THE GATE** — add the `rca` group

**Files:**
- Modify: `pyproject.toml`

This is the step that may fail. The RCA subpackage pulls in langchain (pydantic-v2 era) which conflicts with spaCy 3.5 (pydantic-v1). The remediation in the spec is to pin langchain to its last pydantic-v1-compatible line.

- [ ] **Step 1: Add the `rca` group to `[dependency-groups]`**

In `pyproject.toml`, inside `[dependency-groups]`, add (alphabetical order with the others):

```toml
# AI-enhanced RCA stack (src/dackar/RCA/demos and orchestrators).
# Pinned to pydantic-v1-compatible releases because spaCy 3.5 and
# quantulum3 hold the resolver to pydantic <2. Bump these when the
# core NLP pins move forward.
rca = [
  "langchain-core",
  "langchain-community",
  "langchain-chroma",
  "chromadb",
  "pdfplumber",
  "streamlit",
  "nbformat",
  "requests",
]
```

- [ ] **Step 2: Attempt the full resolution**

```bash
uv lock
```

Three possible outcomes:

**(a) Resolves cleanly** — go directly to Step 5 (commit).

**(b) Fails with a pydantic / langchain conflict** — apply Remediation Pass 1 in Step 3.

**(c) Fails for a different reason** — read the error, identify the offending package, pin it with an explanatory comment, re-lock. If you cannot identify a fix in 30 minutes of attempts, **stop and re-open the design** per the Phase 1 gate.

- [ ] **Step 3: Remediation Pass 1 — pin langchain trio to <0.2**

If Step 2 failed with a pydantic conflict, edit the `rca` group to:

```toml
rca = [
  "langchain-core>=0.1,<0.2",
  "langchain-community>=0.0.30,<0.2",
  "langchain-chroma>=0.1,<0.2",
  "chromadb",
  "pdfplumber",
  "streamlit",
  "nbformat",
  "requests",
]
```

Re-run `uv lock`. If it now succeeds, proceed to Step 5.

- [ ] **Step 4: Remediation Pass 2 — also pin chromadb**

If Step 3 still fails because chromadb pulls a newer langchain transitively, change the chromadb line to:

```toml
  "chromadb>=0.4,<0.5",
```

Re-run `uv lock`. If it now succeeds, proceed to Step 5.

**If Step 4 still fails: STOP.** Do not invent more pins. Document the failing resolver output in a comment on the spec at `docs/superpowers/specs/2026-05-13-uv-package-management-design.md`, surface the failure to the user, and treat this as a gate failure per the spec's "Rollback — Phase 1 failure" path.

- [ ] **Step 5: Commit `pyproject.toml` and `uv.lock` together**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add rca dependency group and commit uv.lock

The langchain stack (langchain-core, -community, -chroma, chromadb) is
pinned to its pydantic-v1-compatible line because spaCy 3.5 and
quantulum3 hold the core resolver to pydantic <2. Bump these pins when
the core NLP versions move forward.

Lockfile committed in this commit because this is the first point at
which all groups successfully resolve together."
```

---

### Task 6: Write the model-bootstrap script

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/bootstrap_models.py`
- Create: `tests/scripts/__init__.py` (empty)
- Create: `tests/scripts/test_bootstrap_models.py`

The script handles three runtime artifacts that uv cannot lock: coreferee's English model, NLTK corpora, and the quantulum3 classifier retrain. The `en_core_web_lg` spaCy model is already a URL dep so `uv sync` installs it — the script must NOT re-download it.

- [ ] **Step 1: Write the test first**

Create `tests/scripts/__init__.py` (empty file):

```bash
touch tests/scripts/__init__.py
```

Create `tests/scripts/test_bootstrap_models.py`:

```python
"""Smoke tests for scripts/bootstrap_models.py.

These don't invoke the actual model downloads — they verify CLI shape and
the idempotency-check helpers, so the test stays under one second and
doesn't need network.
"""
import importlib.util
import pathlib

import pytest


def _load_script():
    """Load the script as a module (it's not a real package)."""
    path = pathlib.Path(__file__).parents[2] / "scripts" / "bootstrap_models.py"
    spec = importlib.util.spec_from_file_location("bootstrap_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exposes_expected_steps():
    """STEPS dict drives --only flag and ordering; both must be present."""
    mod = _load_script()
    assert set(mod.STEPS) == {"coreferee", "nltk", "quantulum"}


def test_only_flag_accepts_known_steps():
    """Argparse rejects unknown --only values."""
    mod = _load_script()
    parser = mod.build_parser()
    parsed = parser.parse_args(["--only", "nltk"])
    assert parsed.only == "nltk"
    with pytest.raises(SystemExit):
        parser.parse_args(["--only", "spacy"])


def test_insecure_ssl_flag_is_off_by_default():
    mod = _load_script()
    parser = mod.build_parser()
    parsed = parser.parse_args([])
    assert parsed.insecure_ssl is False


def test_coreferee_step_skips_when_module_missing(monkeypatch):
    """If `nlp-extra` group not installed, the coreferee step is a no-op."""
    mod = _load_script()
    monkeypatch.setattr(mod, "_has_module", lambda name: False)
    # Should return cleanly without raising, even though coreferee isn't importable.
    mod.install_coreferee()
```

- [ ] **Step 2: Run the test to verify it fails (script doesn't exist yet)**

```bash
uv run pytest tests/scripts/test_bootstrap_models.py -v
```

Expected: FAIL with `FileNotFoundError` or similar on the `spec_from_file_location` call.

- [ ] **Step 3: Create the `scripts/` package and the bootstrap script**

```bash
mkdir -p scripts
touch scripts/__init__.py
```

Create `scripts/bootstrap_models.py`:

```python
"""Post-uv-sync model bootstrap.

Handles the three runtime artifacts that uv cannot lock:
  - coreferee's English model (if `nlp-extra` group is installed)
  - NLTK corpora (punkt, wordnet, averaged_perceptron_tagger, brown)
  - quantulum3 classifier retrain (`quantulum3-training -s`)

Run after `uv sync`:
    uv run python scripts/bootstrap_models.py

The spaCy `en_core_web_lg` model is NOT downloaded here — it's a URL dep
in pyproject.toml, so `uv sync` installs it directly.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import ssl
import subprocess
import sys
from typing import Callable

logger = logging.getLogger("dackar.bootstrap")

# Step name -> function. Ordered for predictable output; --only picks one.
STEPS: dict[str, Callable[[], None]] = {}


def _has_module(name: str) -> bool:
    """True if module is importable in the current environment."""
    return importlib.util.find_spec(name) is not None


def _apply_insecure_ssl() -> None:
    """Disable SSL cert verification globally for downloads behind MITM proxies.

    Same trick as the old nltkDownloader.py. Off by default; opt in with
    --insecure-ssl.
    """
    try:
        _create_unverified = ssl._create_unverified_context  # type: ignore[attr-defined]
    except AttributeError:
        return
    ssl._create_default_https_context = _create_unverified


def install_coreferee() -> None:
    """`python -m coreferee install en` — only if coreferee is importable."""
    if not _has_module("coreferee"):
        logger.info("coreferee not installed (nlp-extra group missing); skipping")
        return
    logger.info("Installing coreferee English model")
    subprocess.check_call([sys.executable, "-m", "coreferee", "install", "en"])


STEPS["coreferee"] = install_coreferee


def install_nltk_corpora() -> None:
    """Download NLTK corpora used by dackar.similarity / dackar.text_processing."""
    import nltk

    corpora = ["punkt", "wordnet", "averaged_perceptron_tagger", "brown"]
    for name in corpora:
        logger.info("nltk.download(%s)", name)
        nltk.download(name, quiet=True)


STEPS["nltk"] = install_nltk_corpora


def retrain_quantulum() -> None:
    """`quantulum3-training -s` — retrains classifier against installed sklearn."""
    logger.info("Retraining quantulum3 classifier")
    subprocess.check_call(["quantulum3-training", "-s"])


STEPS["quantulum"] = retrain_quantulum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap runtime models/corpora for DACKAR.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(STEPS),
        help="Run a single step instead of all (useful for partial CI envs).",
    )
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable SSL cert verification for downloads (MITM proxy workaround).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    args = build_parser().parse_args(argv)

    if args.insecure_ssl:
        logger.warning("SSL verification disabled for this run")
        _apply_insecure_ssl()

    selected = [args.only] if args.only else list(STEPS)
    for name in selected:
        logger.info("=== %s ===", name)
        STEPS[name]()

    logger.info("Bootstrap complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/scripts/test_bootstrap_models.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Sanity-check the script's CLI**

```bash
uv run python scripts/bootstrap_models.py --help
```

Expected output (substrings — exact wording may vary slightly):
```
usage: bootstrap_models.py [-h] [--only {coreferee,nltk,quantulum}] [--insecure-ssl]
...
--only {coreferee,nltk,quantulum}
                      Run a single step instead of all (useful for partial CI envs).
--insecure-ssl        Disable SSL cert verification for downloads (MITM proxy workaround).
```

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/bootstrap_models.py \
        tests/scripts/__init__.py tests/scripts/test_bootstrap_models.py
git commit -m "feat(scripts): add bootstrap_models.py for post-sync model downloads

Replaces the four manual install steps (coreferee, nltk, quantulum3 retrain)
with one idempotent CLI. The en_core_web_lg spaCy model stays a URL dep
in pyproject.toml so uv handles it directly.

Includes --only flag for CI partial envs and --insecure-ssl flag carrying
forward the workaround from the deleted nltkDownloader.py."
```

---

### Task 7: Run the full local validation

This task has no code changes — it's a checkpoint that proves Phase 1 succeeded. If any step here fails, stop and diagnose before touching CI in Phase 2.

- [ ] **Step 1: Full sync**

```bash
uv sync --all-groups
```

Expected: completes; produces `.venv/` containing every dependency from every group.

- [ ] **Step 2: Bootstrap models**

```bash
uv run python scripts/bootstrap_models.py
```

Expected:
```
=== coreferee ===
Installing coreferee English model
...
=== nltk ===
nltk.download(punkt)
nltk.download(wordnet)
nltk.download(averaged_perceptron_tagger)
nltk.download(brown)
=== quantulum ===
Retraining quantulum3 classifier
...
Bootstrap complete.
```

- [ ] **Step 3: Run the full test suite**

```bash
uv run pytest tests/
```

Expected: same pass/fail outcome as the existing conda-based CI. The current CI baseline lives in `.github/workflows/github-actions.yml` (job `Test-DACKAR-Linux`); record the test counts (pass/fail/skip) before this migration started so you can compare. **No new failures or skips.**

- [ ] **Step 4: Confirm `dackar.main` CLI still works**

```bash
uv run python -m dackar.main -i system_tests/ner.toml
```

Expected: produces `ner.csv` in the current directory and logs `... Complete!` at the end. (Some test TOMLs reference files via relative paths; if it complains about a missing file, that's an existing repo quirk, not a migration regression.)

- [ ] **Step 5: No commit needed** — this is a verification checkpoint.

If any of the above failed, **stop and diagnose**. Do not proceed to Phase 2 with a broken local environment.

---

### Task 8: RCA demo smoke test — verify the original conflict is fixed

The whole point of the migration. Pick one RCA demo that previously failed with a package conflict and confirm it now executes past imports.

- [ ] **Step 1: Sync the RCA-relevant groups in isolation**

```bash
uv sync --group rca --group nlp-extra --group kg
```

Expected: completes; produces a venv with core + rca + nlp-extra + kg deps (no dev, no docs, no anomaly, no viz).

- [ ] **Step 2: Run the import-only smoke check**

```bash
uv run python -c "
import sys
sys.path.insert(0, 'src/dackar/RCA')
import langchain_chroma, langchain_community, langchain_core
import chromadb, pdfplumber, nbformat, requests, streamlit
print('All RCA imports OK')
"
```

Expected: `All RCA imports OK`.

- [ ] **Step 3: Run one of the RCA demo entry points past its imports**

```bash
uv run python -c "
import sys
sys.path.insert(0, 'src/dackar/RCA')
sys.path.insert(0, 'src/dackar/RCA/demos')
# Import the demo modules to confirm they're parseable in this env.
# We are NOT running their main() because the demos read demo-specific
# inputs; the goal here is import-graph health.
import importlib
for name in ['kg_population_helpers', 'stage1_6_existing_methods_helpers']:
    importlib.import_module(name)
print(f'Imported {name}')
"
```

Expected: prints `Imported stage1_6_existing_methods_helpers` with no traceback.

- [ ] **Step 4: No commit needed** — verification checkpoint.

If imports still fail, the cause is a missing dep that wasn't listed in the spec's RCA group. Capture the `ModuleNotFoundError`, add the missing package to the `rca` group in `pyproject.toml`, re-run `uv lock` (must still succeed), commit the pyproject + lockfile change with message `build(rca): add <pkg> required by <module>`, then re-run this task.

---

# Phase 2 — CI and Documentation

> Phase 1 is now complete. From here on, changes are mechanical and reversible. A bad CI rewrite reverts cleanly; bad docs aren't load-bearing.

---

### Task 9: Rewrite the CI workflow

**Files:**
- Modify: `.github/workflows/github-actions.yml` (replace entirely)

- [ ] **Step 1: Replace `.github/workflows/github-actions.yml` with the uv-based workflow**

Overwrite the file with:

```yaml
name: GitHub DACKAR test
run-name: ${{ github.actor }} is testing out DACKAR
on: [push, pull_request]

concurrency:
  group: ${{ github.head_ref }}
  cancel-in-progress: true

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: uv sync --all-groups

      - name: Bootstrap runtime models
        run: uv run python scripts/bootstrap_models.py

      - name: Run tests
        working-directory: tests
        run: uv run pytest
```

This single matrix job replaces the three separate jobs (`Test-DACKAR-Linux`, `Test-DACKAR-Macos`, `Test-DACKAR-Windows`) in the previous workflow.

- [ ] **Step 2: Lint the workflow locally (optional but recommended)**

If you have `actionlint` installed:

```bash
actionlint .github/workflows/github-actions.yml
```

Expected: no errors. If you don't have actionlint, skip — GitHub Actions will validate on push.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/github-actions.yml
git commit -m "ci: replace conda+pip with uv across all three OSes

Single matrix job (ubuntu-latest, macos-latest, windows-latest) using
astral-sh/setup-uv@v6 + 'uv sync --all-groups'. Drops conda setup, the
hand-rolled 'pip install spacy==3.5 ...' line, secondary 'pip install
neo4j jupyterlab pytest', per-OS torch install (now in [tool.uv.sources]),
the en_core_web_lg wheel URL (now a project dep), and ~8 'conda init &&
source && conda activate' lines."
```

- [ ] **Step 4: Push the branch and verify all three matrix jobs go green**

```bash
git push -u origin uv_env_management
```

Watch the Actions tab on GitHub. **All three OSes must pass before continuing.** If a single OS fails:
- Read the failure. If it's a packaging issue (missing dep, wrong marker), fix `pyproject.toml`, re-run `uv lock`, commit, re-push.
- If it's a test failure that didn't happen on conda, the test depends on some implicit state from the old environment. File a follow-up issue rather than fixing in this PR — packaging migrations should not also change test behavior.

---

### Task 10: Rewrite the README

**Files:**
- Modify: `README.md` (replace `## Installation`, `## Test`, `## How to build documentation`)

- [ ] **Step 1: Replace the `## Installation` section**

Open `README.md`. Replace the entire section from `## Installation` up to (but not including) `## Test` with:

```markdown
## Installation

### 1. Install uv (one-time, per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # Linux/macOS
# or: brew install uv
# or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/idaholab/DACKAR.git
cd DACKAR

# Pick the install that matches your workflow:
uv sync                                                # core NLP only
uv sync --group rca --group kg --group nlp-extra --group dev   # full RCA workflow
uv sync --all-groups                                    # everything
```

### 3. Bootstrap runtime models (one-time)

```bash
uv run python scripts/bootstrap_models.py
```

This downloads coreferee's English model (if `nlp-extra` is installed),
the NLTK corpora used for similarity analysis, and retrains the
quantulum3 classifier. The `en_core_web_lg` spaCy model is installed
automatically as a project dependency.

### Dependency groups

| Group | Use when |
|---|---|
| _(core, always installed)_ | Running `python -m dackar.main` |
| `nlp-extra` | Optional NLP pipes (coreferee, pywsd, contextual spell check) |
| `anomaly`   | Using `dackar.anomalies` (matrix-profile / two-sample tests) |
| `kg`        | Loading data into Neo4j via `dackar.knowledge_graph` |
| `viz`       | Word-cloud rendering in `dackar.utils.visualize` |
| `rca`       | Running the AI-enhanced RCA demos under `src/dackar/RCA/` |
| `docs`      | Building Sphinx documentation |
| `dev`       | Running tests and notebook examples |
```

- [ ] **Step 2: Replace the `## Test` section**

Replace from `## Test` up to (but not including) `## How to build documentation` with:

```markdown
## Test

```bash
uv run pytest tests/                         # full suite
uv run pytest tests/pipelines/test_pipelines.py  # single file
uv run pytest -k temporal                    # by keyword
```
```

- [ ] **Step 3: Replace the `## How to build documentation` section**

Replace from `## How to build documentation` to end-of-file with:

```markdown
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
```

- [ ] **Step 4: Verify the rewritten README has no leftover `conda` or `pip install` references**

```bash
grep -i "conda\|pip install" README.md
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): replace conda+pip install flow with uv

Three-command install (install uv, uv sync, bootstrap models) plus a
dependency-groups table so users know which group to pick for their
workflow. Test and docs build sections now use 'uv run'."
```

---

### Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (sections: "Environment setup", "Running", "Testing")

The CLAUDE.md written in the previous session contains a warning that "`pyproject.toml` is not the source of truth for installation". After this migration, it *is* the source of truth, so that warning gets reversed.

- [ ] **Step 1: Replace the "Environment setup" section**

In `CLAUDE.md`, find the section `## Environment setup` and replace its body (everything between that header and `## Running`) with:

```markdown
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
- The `rca` group is pinned to pydantic-v1-era langchain because spaCy 3.5 holds pydantic <2. Bump those pins when the core NLP versions move forward.
```

- [ ] **Step 2: Replace the "Running" section**

Find `## Running` and replace its body with:

```markdown
## Running

```bash
# CLI entry point — pass -i explicitly; default path is a stale relative path
uv run python -m dackar.main -i system_tests/ner.toml
```

Input TOML must contain an `[nlp]` table or a `[neo4j]` table (or both); see `system_tests/ner.toml`, `system_tests/causal.toml`, `system_tests/kg.toml` for the three canonical shapes. `WorkflowManager._validate` runs a JSON-schema check (`src/dackar/validate.py`) before anything else, so schema-level errors surface immediately.
```

- [ ] **Step 3: Replace the "Testing" section**

Find `## Testing` and replace its body with:

```markdown
## Testing

```bash
uv run pytest tests/                              # full suite
uv run pytest tests/pipelines/test_pipelines.py   # single file
uv run pytest -k temporal                         # by keyword
```

`tests/` is the gated, CI-run suite. `general_tests/` and `system_tests/` are exploratory scripts / TOML fixtures and not run by `pytest` from the repo root. CI runs `uv run pytest tests/` on Ubuntu, macOS, and Windows via the matrix job in `.github/workflows/github-actions.yml`.
```

- [ ] **Step 4: Verify no leftover conda references**

```bash
grep -i "conda\|dackar_libs" CLAUDE.md
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): update environment setup for uv migration

Reverses the 'pyproject.toml is not the source of truth' warning,
documents the dependency groups, the per-OS torch source, and the
pydantic-v1 constraint on the rca group's langchain pins."
```

---

### Task 12: Update Sphinx install docs

**Files:**
- Modify: `docs/install.rst` (mark archived; remove conda steps)
- Modify: `docs/install_spacy3.5.rst` (rewrite as the primary install page)

- [ ] **Step 1: Rewrite `docs/install_spacy3.5.rst` end-to-end**

Replace the entire file contents with:

```rst
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
```

- [ ] **Step 2: Mark `docs/install.rst` as fully archived**

The file is already titled "Installation with Spacy 3.1 (Archived)". Add a prominent deprecation note at the top.

Replace lines 1–6 of `docs/install.rst`:

```rst
======================================
Installation with Spacy 3.1 (Archived)
======================================

How to install dependency libraries
-----------------------------------
```

with:

```rst
======================================
Installation with Spacy 3.1 (Archived)
======================================

.. deprecated::
   This page is historical. DACKAR no longer supports spaCy 3.1 or the
   conda+pip install flow. Use the current uv-based instructions in
   :doc:`install_spacy3.5` instead.

How to install dependency libraries
-----------------------------------
```

(Leave the rest of the file alone — it's a historical artifact.)

- [ ] **Step 3: Verify no broken references**

```bash
grep -r "conda create\|pip install" docs/install_spacy3.5.rst
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add docs/install.rst docs/install_spacy3.5.rst
git commit -m "docs(install): rewrite install_spacy3.5.rst for uv; archive install.rst

The Sphinx install page now mirrors the README install flow (uv sync +
bootstrap script + dependency-groups table). The older spaCy 3.1 page
gets a sphinx 'deprecated' admonition pointing readers at the new page."
```

---

### Task 13: Update outage README and delete nltkDownloader.py

**Files:**
- Modify: `src/dackar/outage/README.md`
- Delete: `nltkDownloader.py`

- [ ] **Step 1: Update `src/dackar/outage/README.md`**

Open the file. Find the "Suggested next steps" section (currently item 1: "Create a virtual environment and install dependencies you choose.") and replace it with:

```markdown
## Suggested next steps
1. From the repo root, run `uv sync --group dev` (the outage subpackage currently has no extra deps beyond core; this will change as the skeleton is filled in).
2. Replace placeholder logic in preprocessing, retrieval, and uncertainty modules.
3. Add unit tests under `tests/`.
4. Wire the workflow to DACKAR and your outage datasets.

> **TODO:** populate an `outage` dependency group in the root `pyproject.toml` once the outage modules declare their concrete dependencies. The standalone `src/dackar/outage/pyproject.toml` is retained as a skeleton; it is not currently a uv workspace member.
```

- [ ] **Step 2: Delete `nltkDownloader.py`**

```bash
git rm nltkDownloader.py
```

Verify:
```bash
ls nltkDownloader.py 2>&1
```
Expected: `ls: nltkDownloader.py: No such file or directory`.

- [ ] **Step 3: Commit**

```bash
git add src/dackar/outage/README.md
git commit -m "docs(outage): point at uv sync; delete obsolete nltkDownloader.py

The outage README's 'create a venv' step is replaced with a 'uv sync
--group dev' pointer and a TODO to populate an 'outage' dependency
group when the skeleton modules grow real deps.

nltkDownloader.py's SSL-tolerant logic is now in scripts/bootstrap_models.py
behind the --insecure-ssl flag."
```

---

# Phase 3 — Cleanup

---

### Task 14: Try `uv build` and consider dropping setuptools

**Files:**
- Modify (possibly): `pyproject.toml` ([build-system] section)

The spec leaves the build backend choice contingent on whether the default `hatchling` works with the `src/dackar` layout. Try it; revert if it doesn't.

- [ ] **Step 1: Verify the current setuptools build works**

```bash
uv build
```

Expected: produces `dist/dackar-0.1.0.tar.gz` and `dist/dackar-0.1.0-py3-none-any.whl`. If this already errors with the setuptools backend, fix that first — it's a regression from the migration.

- [ ] **Step 2: Try the hatchling default**

Edit `pyproject.toml`. Replace:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

with:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/dackar"]
```

Run:
```bash
rm -rf dist/
uv build
```

Two possible outcomes:

**(a) Build succeeds** — commit the change in Step 4.

**(b) Build fails** — revert the `[build-system]` section to the setuptools version from Step 1 and skip Step 4. This is acceptable; the migration's goal is dependency management, not the build backend.

- [ ] **Step 3: Verify the built wheel imports correctly**

(Only if Step 2 succeeded.)

```bash
python -c "
import zipfile, pathlib
wheel = next(pathlib.Path('dist').glob('dackar-*.whl'))
with zipfile.ZipFile(wheel) as z:
    names = [n for n in z.namelist() if n.startswith('dackar/')]
print(f'Wheel contains {len(names)} dackar/* entries (sample: {names[:3]})')
"
```

Expected: `Wheel contains <N> dackar/* entries (sample: ['dackar/__init__.py', ...])` with N>10.

- [ ] **Step 4: Commit (only if hatchling worked)**

```bash
git add pyproject.toml
git commit -m "build: switch build-backend from setuptools to hatchling

Hatchling produces the same wheel layout from src/dackar/ with less
configuration. No functional change for users; uv build / pip install
behaviour is unchanged."
```

---

### Task 15: Final repo scrub

**Files:** (verification only; remediation is targeted)

- [ ] **Step 1: Confirm no tracked file references conda**

```bash
git ls-files | xargs grep -l -i "conda" 2>/dev/null
```

Expected: no output. If any file is listed:
- If it's an intentional historical reference (e.g., `docs/install.rst` archived page), leave it.
- Otherwise, edit the file to remove the conda reference and commit with message `docs: drop stale conda reference in <file>`.

- [ ] **Step 2: Confirm no tracked file references `dackar_libs`**

```bash
git ls-files | xargs grep -l "dackar_libs" 2>/dev/null
```

Expected: no output. Remediation as in Step 1.

- [ ] **Step 3: Confirm `pyproject.toml` and `uv.lock` are the only manifest files**

```bash
git ls-files | grep -E "(requirements.*\.txt|environment\.ya?ml|Pipfile)" 2>/dev/null
```

Expected: no output. (The `docs/requirements.txt` file may be present if RTD or similar uses it — if so, replace it with an RTD config that uses uv, or leave it documented as a generated artifact from the docs group.)

- [ ] **Step 4: Final test run on a clean venv**

```bash
rm -rf .venv
uv sync --all-groups
uv run python scripts/bootstrap_models.py
uv run pytest tests/
```

Expected: full clean-room migration succeeds end-to-end with no manual fixups.

- [ ] **Step 5: Commit any remediation from Steps 1–3**

If you made fixes in Steps 1 or 2, commit them now:

```bash
git add <changed-files>
git commit -m "chore: remove stale conda / dackar_libs references"
```

Otherwise no commit.

---

## Done

All Phase 1 verification checkpoints passed, CI is green on all three OSes, and the docs reflect the new install flow. Open a PR:

```bash
gh pr create --base main --head uv_env_management \
  --title "Migrate package management from conda+pip to uv" \
  --body "$(cat <<'EOF'
Replaces the conda+pip install flow with a single uv-managed project.
See the design doc at docs/superpowers/specs/2026-05-13-uv-package-management-design.md
and the implementation plan at docs/superpowers/plans/2026-05-13-uv-package-management.md
for full context.

**Verification:**
- [x] uv lock resolves all groups together
- [x] uv run pytest tests/ matches the previous CI pass/fail set
- [x] One RCA demo that previously failed with a package conflict now runs past imports
- [x] CI green on Ubuntu, macOS, Windows
- [x] grep -ri conda returns nothing in tracked files

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
