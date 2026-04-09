# Outage Canonical Model Starter Package

This starter package provides a Python-ready canonical data model for nuclear plant outage schedule analytics.

It is designed to sit between source systems such as Primavera P6 and downstream analytics code.

## What is included

- Canonical entity models built with Pydantic
- A physical schema for pandas tables and parquet/sql persistence
- A dataset container for validated DataFrames
- Generic transforms for P6-like CSV exports
- Example loaders and mock data
- Documentation for the recommended table design

## Core design principles

- Preserve outage schedule versions and baselines
- Keep schedule tasks as the central canonical object
- Preserve WBS, dependencies, calendars, resource assignments, and activity codes
- Keep source lineage for traceability
- Make the model easy to use with pandas, parquet, SQL, and graph analytics

## Package layout

```text
outage_canonical_package/
  pyproject.toml
  README.md
  docs/
    physical_schema.md
  src/
    outage_model/
      __init__.py
      dataset.py
      models.py
      schema.py
      loaders/
        example_loader.py
      transforms/
        common.py
        p6_csv.py
  examples/
    build_mock_dataset.py
    mock_p6_export/
      calendars.csv
      resources.csv
      assignments.csv
      activity_codes.csv
      task_activity_codes.csv
      wbs.csv
      activities.csv
      relationships.csv
```

## Install

```bash
pip install -e .
```

## Quick start

```python
from pathlib import Path

from outage_model.loaders.example_loader import load_mock_dataset

base_dir = Path("examples/mock_p6_export")
dataset = load_mock_dataset(base_dir)

print(dataset.schedule_tasks.head())
print(dataset.dependencies.head())
```

## Expected source data

The package includes a transform for a generic P6-like export represented as CSV files.
The transform expects these files when available:

- `activities.csv`
- `relationships.csv`
- `wbs.csv`
- `resources.csv`
- `assignments.csv`
- `calendars.csv`
- `activity_codes.csv`
- `task_activity_codes.csv`

Column naming is flexible as long as the exported fields map to the expected logical columns. The provided mock files show the expected structure.

## Main canonical tables

- `outages`
- `schedule_versions`
- `wbs`
- `schedule_tasks`
- `dependencies`
- `resources`
- `resource_assignments`
- `calendars`
- `activity_codes`
- `task_activity_codes`
- `task_constraints`
- `work_packages`
- `scope_change_events`
- `delay_events`
- `work_windows`
- `clearances`

## Recommended next extensions

- Add an XER parser or XML parser for your exact P6 export path
- Add mappings from CMMS or work package systems
- Add graph analytics using networkx
- Add outage KPI derivation and feature engineering modules

