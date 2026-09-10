# Testing Guide — Outage Activity Pipeline

## Environment

All outage pipeline tests run under the **`base` conda environment**
(Python 3.11, `/opt/anaconda3`).  No LOGOS installation is required — Stage E
tests use duck-typed mock Pert objects.

```bash
conda activate base        # or just use the default shell if base is active
python --version           # expect Python 3.11.x
```

## Running the tests

All commands must be issued from the **outage package root**:

```bash
cd /Users/mandd/projects/DACKAR/src/dackar/outage
```

Run the full suite:

```bash
python -m pytest tests/ -q
```

Run a single test file:

```bash
python -m pytest tests/test_stage_b.py -v
python -m pytest tests/test_stage_e.py -v
python -m pytest tests/test_stages_a_c.py -v
python -m pytest tests/test_stages_f_g.py -v
python -m pytest tests/test_orchestrator_e2e.py -v
```

Run a single test class or method:

```bash
python -m pytest tests/test_stage_e.py::TestCheckResourceConflictsEquipment -v
python -m pytest tests/test_stages_f_g.py::TestAnalystReviewHighAbbrRate -v
```

Expected baseline: **730 tests, 0 failures** (as of April 2026).

## Why NOT the `dackar_libs` environment

The `dackar_libs` conda env contains `spacy 3.5.0`, which depends on
`pydantic.v1.ConstrainedStr` — a symbol removed in Pydantic V2.  When pytest
collects tests it walks up the directory tree and imports
`dackar/__init__.py`, which triggers the spacy import chain and raises
`ImportError: cannot import name 'ConstrainedStr' from 'pydantic'`.

**Fix options (not yet applied):**
- Upgrade spacy to ≥ 3.7 in `dackar_libs` (recommended), or
- Downgrade pydantic to < 2 in `dackar_libs`.

Until then, the outage pipeline tests must run in `base`, and the RCA tests
(see below) must be run as standalone scripts.

## RCA tests (`/Users/mandd/projects/DACKAR/src/dackar/RCA`)

RCA tests live outside the outage package and are affected by the
spacy/Pydantic conflict described above.  Run them as plain Python scripts
inside `dackar_libs`, **not** via pytest:

```bash
conda activate dackar_libs
cd /Users/mandd/projects/DACKAR/src/dackar/RCA
python tests/test_<name>.py
```

All 203 RCA tests pass this way (verified April 2026).

## Test file map

| File | What it covers |
|------|----------------|
| `tests/test_stages_a_c.py` | Stage A + C — intake, NER, regulatory rules, Allen temporal relations |
| `tests/test_stage_b.py` | Stage B — KG timeline builder, DQ scoring, deduplication, stub driver |
| `tests/test_stage_e.py` | Stage E — schedule impact, float analysis, CP drag, resource/equipment/location conflicts |
| `tests/test_stages_f_g.py` | Stage D + F + G — analogs, insertion options, recommendation synthesis, analyst review |
| `tests/test_orchestrator_e2e.py` | End-to-end `run_pipeline()` — both demo scenarios, artifact schemas, fallback paths; imports from `demos/unexpected_act_workflow_1/demo_scenarios.py` |

## Pytest configuration

`pyproject.toml` at the package root sets `testpaths = ["tests"]`, so
`python -m pytest` without arguments picks up all test files automatically
when invoked from the package root.

## Adding new tests

1. Place the file in `tests/test_<stage_or_feature>.py`.
2. Add the `_OUTAGE_ROOT` path block at the top (see any existing test file)
   so that stage imports resolve without installing the package.
3. Avoid importing LOGOS directly — use duck-typed mock classes as in
   `test_stage_e.py` (`_MockPert`, `_FakeEquipmentPool`, `_FakeLocationPool`).
4. Update the test file map in this document and in
   `docs/pipeline_stages_reference.md` section 12.
