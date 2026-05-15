# Testing Guide — Outage Activity Pipeline

## Environment

All outage pipeline tests run under the project uv environment.
No LOGOS installation is required — Stage E tests use duck-typed mock Pert objects.

```bash
uv sync --group rca        # install the RCA dependency group
uv run python --version    # expect Python 3.11.x
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

## Historical note: the old `dackar_libs` conda environment

Previously the project used a conda environment named `dackar_libs`.  That env
had a spaCy 3.5.0 / Pydantic V2 incompatibility that prevented pytest collection.
The project has since migrated to uv (see `pyproject.toml`).  The conflict was
resolved not by upgrading spaCy but by letting uv resolve all dependency groups
together in a single pass: spaCy 3.5 in the core dependencies constrains the
resolver to pydantic-v1-compatible versions, which is exactly what langchain in
the `rca` group needs.  Mixing two separate pip install rounds (as the old conda
workflow did) caused incompatible transitive requirements to collide; uv avoids
this entirely.

## RCA tests

RCA tests live in `/src/dackar/RCA/tests/` and run via pytest through uv:

```bash
uv run pytest tests/
```

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
