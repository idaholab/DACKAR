# Testing Guide — Outage Activity Pipeline

## Environment

All outage pipeline tests run under the **repo-root** uv environment. The
`rca` (langchain) and `dev` (pytest) dependency groups are defined in the root
`pyproject.toml`, not in `src/dackar/outage/pyproject.toml`, so sync from the
repository root. No LOGOS installation is required — Stage E tests use
duck-typed mock Pert objects.

```bash
# from the repository root
uv sync --group rca --group dev    # RCA stack + pytest
uv run python --version            # expect Python 3.11.x
```

## Running the tests

All commands are issued from the **repository root** via `uv run` so the
root environment (with the `rca`/`dev` groups) is used:

Run the full suite:

```bash
uv run pytest src/dackar/outage/tests/ -q
```

Run a single test file:

```bash
uv run pytest src/dackar/outage/tests/test_stage_b.py -v
uv run pytest src/dackar/outage/tests/test_stage_e.py -v
uv run pytest src/dackar/outage/tests/test_stages_a_c.py -v
uv run pytest src/dackar/outage/tests/test_stages_f_g.py -v
uv run pytest src/dackar/outage/tests/test_orchestrator_e2e.py -v
```

Run a single test class or method:

```bash
uv run pytest src/dackar/outage/tests/test_stage_e.py::TestCheckResourceConflictsEquipment -v
uv run pytest src/dackar/outage/tests/test_stages_f_g.py::TestAnalystReviewHighAbbrRate -v
```

Expected baseline: **730 tests, 0 failures** (as of April 2026).

## Historical note: the old `dackar_libs` conda environment

Previously the project used a conda environment named `dackar_libs`, built from
two separate `pip install` rounds. The project has since migrated to uv (see
`pyproject.toml`), which resolves the core NLP stack and the heavier `rca`
(langchain / marker-pdf) stack together in a single pass, so incompatible
transitive requirements can no longer collide the way two ad-hoc pip rounds did.

The original blocker was a `pydantic` conflict: the NLP stack was capped at
`pydantic<2` (via `coreferee 1.4.1`, which required `spacy<3.6`), while
`marker-pdf` needs `pydantic>=2.4.2`. It was resolved by bumping the NLP stack to
spaCy 3.8 / pydantic v2 and removing `coreferee` (which was already inactive at
runtime). spaCy is now pinned to `3.8.*` and the whole project, including the
`rca` group, resolves on pydantic v2.

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
