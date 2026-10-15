# Outage Canonical Data Model & Pipeline

## Overview

This package provides a Python-ready canonical data model and ingestion pipeline for nuclear power plant outage data, designed to integrate with Primavera P6 schedule exports in both XER and CSV formats.

The goal is to enable:

- Consistent ingestion of outage schedule data
- Preservation of schedule logic (dependencies, WBS, versions)
- Conversion into pandas DataFrames for analytics
- Support for downstream decision-support, RCA, and optimization

## Package Structure

```
P6_adapter/
│
├── outage_model/
│   ├── __init__.py            # Public API (models, dataset, P6XERTransformer)
│   ├── models.py              # Pydantic data models (canonical schema)
│   ├── schema.py              # Physical schema: pandas dtypes, primary keys, foreign keys
│   ├── dataset.py             # OutageDataset container + schema coercion helpers
│   │
│   ├── transforms/
│   │   ├── common.py          # Shared helpers (column normalization, ID generation, CSV reading)
│   │   ├── p6_xer.py          # XER parser (XERParser) and transformer (P6XERTransformer)
│   │   └── p6_csv.py          # Transformer for P6-style CSV exports (P6CsvTransformer)
│   │
│   └── loaders/
│       ├── xer_loader.py      # load_xer_dataset() — convenience entry point for XER files
│       └── example_loader.py  # load_mock_dataset() — convenience entry point for CSV examples
│
├── examples/
│   ├── build_mock_dataset.py  # End-to-end demo using mock CSV export
│   ├── load_sample_xer.py     # End-to-end demo using a sample XER file
│   ├── mock_p6_export/        # Mock P6-like CSV files (activities, relationships, wbs, etc.)
│   └── sample_xer/
│       └── sample_project.xer # Sample Primavera P6 XER file
│
└── docs/
    ├── outage_canonical_model_and_pipeline.md  # This file
    ├── p6_xer_pipeline.md                      # XER ingestion details
    └── physical_schema.md                      # Table design and join conventions
```

## Canonical Data Model (Pandas Schema)

The system uses a multi-table relational structure, stored as pandas DataFrames.

| Table               | Purpose                              |
| ------------------- | ------------------------------------ |
| `outages`           | Outage campaign metadata             |
| `schedule_versions` | Baselines, updates, as-run schedules |
| `wbs`               | Work Breakdown Structure hierarchy   |
| `schedule_tasks`    | Core scheduling activities           |
| `dependencies`      | Task relationships (FS, SS, etc.)    |

Resource & Execution Tables

| Table                  | Purpose                  |
| ---------------------- | ------------------------ |
| `resources`            | Labor, crews, vendors    |
| `resource_assignments` | Task-resource linkage    |
| `calendars`            | Working time definitions |

Classification & Logic

| Table                 | Purpose                        |
| --------------------- | ------------------------------ |
| `activity_codes`      | P6-style classification fields |
| `task_activity_codes` | Many-to-many mapping           |
| `task_constraints`    | Schedule constraints           |

Outage Execution Extensions

| Table                 | Purpose                      |
| --------------------- | ---------------------------- |
| `work_packages`       | Grouping for decision-making |
| `scope_change_events` | Emergent/deferred work       |
| `delay_events`        | Execution delays             |
| `work_windows`        | Access/system windows        |
| `clearances`          | Nuclear clearance tracking   |

## Canonical Dataset Object

All tables are grouped into a single container:

```python
OutageDataset(
    outages: pd.DataFrame
    outage_phases: pd.DataFrame
    schedule_versions: pd.DataFrame
    wbs: pd.DataFrame
    schedule_tasks: pd.DataFrame
    dependencies: pd.DataFrame
    resources: pd.DataFrame
    resource_assignments: pd.DataFrame
    calendars: pd.DataFrame
    activity_codes: pd.DataFrame
    task_activity_codes: pd.DataFrame
    task_constraints: pd.DataFrame
    work_packages: pd.DataFrame
    scope_change_events: pd.DataFrame
    delay_events: pd.DataFrame
    work_windows: pd.DataFrame
    clearances: pd.DataFrame
)
```

`OutageDataset.apply_schema()` coerces all DataFrames to the physical dtypes defined in `schema.py`.
`OutageDataset.as_dict()` returns a `{table_name: DataFrame}` mapping for all tables.

## Data Pipeline Architecture

### Step 1 — Raw Data Ingestion

**XER path** — a single `.xer` file:
```
sample_project.xer
```

**CSV path** — a directory of CSV files:
```
activities.csv
relationships.csv
resources.csv
assignments.csv
wbs.csv
calendars.csv
activity_codes.csv
task_activity_codes.csv
```

Files are loaded as raw pandas DataFrames.

### Step 2 — Standardization

`common.py` normalizes:
- Column names → `snake_case` via `normalize_name()`
- Missing/null-like string values (`"null"`, `"none"`, `"na"`, etc.) via `normalize_text_value()`
- Missing columns are added as `pd.NA` via `ensure_columns()`

### Step 3 — Canonical ID Generation

Stable composite IDs are created to avoid tool dependency:

```python
task_id = build_canonical_id(outage_id, schedule_version_id, activity_id)
# → "RFO-2026-U1:RFO-2026-U1:xer:A1000"
```

Ensures uniqueness across outages and version traceability.

### Step 4 — Transformation (Source → Canonical)

| Source          | Canonical              | Transformer          |
| --------------- | ---------------------- | -------------------- |
| P6 XER `TASK`   | `schedule_tasks`       | `P6XERTransformer`   |
| P6 XER `TASKPRED` | `dependencies`       | `P6XERTransformer`   |
| P6 XER `RSRC`   | `resources`            | `P6XERTransformer`   |
| P6 XER `TASKRSRC` | `resource_assignments` | `P6XERTransformer` |
| P6 XER `PROJWBS` | `wbs`                 | `P6XERTransformer`   |
| P6 XER `CALENDAR` | `calendars`           | `P6XERTransformer`   |
| P6 XER `ACTVCODE` + `ACTVTYPE` | `activity_codes` | `P6XERTransformer` |
| P6 XER `TASKACTV` | `task_activity_codes` | `P6XERTransformer`  |
| P6 CSV `activities.csv` | `schedule_tasks` | `P6CsvTransformer` |
| P6 CSV `relationships.csv` | `dependencies` | `P6CsvTransformer` |

### Step 5 — Validation (Optional but Recommended)

Pydantic models in `models.py` enforce:
- Schema correctness
- Required fields
- Type safety

### Step 6 — Dataset Assembly

All transformed tables are combined into an `OutageDataset` and schema is applied:

```python
dataset = OutageDataset(
    outages=...,
    schedule_tasks=...,
    dependencies=...,
    ...
)
dataset.apply_schema()
```

### Step 7 — Storage (Optional, not yet implemented)

The design supports:
- Parquet (via `pandas.DataFrame.to_parquet`)
- SQL (via `pandas.DataFrame.to_sql`)
- Versioned datasets (each `schedule_version_id` is a separate snapshot)

### Step 8 — Analytics Layer

The dataset supports:
- Graph-based analysis (using networkx)
  - Critical path reconstruction from `dependencies` + `schedule_tasks`
  - Bottleneck detection
  - Dependency risk propagation
- Statistical / ML analysis (using pandas / sklearn)
  - Outage duration prediction
  - Delay clustering
  - Scope growth patterns

## Example Pipeline Execution

**Load from XER:**
```python
from pathlib import Path
from outage_model.loaders.xer_loader import load_xer_dataset

dataset = load_xer_dataset(Path("examples/sample_xer/sample_project.xer"))
print(dataset.schedule_tasks.head())
print(dataset.dependencies.head())
```

**Load from mock CSV export:**
```python
from pathlib import Path
from outage_model.loaders.example_loader import load_mock_dataset

dataset = load_mock_dataset(Path("examples/mock_p6_export"))
for table_name, df in dataset.as_dict().items():
    print(f"{table_name}: {len(df)} rows")
```

## Key Design Principles

1. **Source Independence.** Decouples analytics from P6 or any specific tool. Both `P6XERTransformer` and `P6CsvTransformer` produce the same `OutageDataset` schema.

2. **Version Awareness.** Schedules are never overwritten — baseline, working updates, and as-run schedules each get their own `schedule_version_id`.

3. **Relational (Not Nested).** Better for pandas, SQL, performance, and clarity.

4. **Traceability.** Each `schedule_tasks` row keeps `source_system` and `source_record_id` pointing back to the originating P6 record.

5. **Schedule Logic Preservation.** Dependencies and calendars are first-class citizens in the schema.

6. **Tolerant Ingestion.** Both transformers handle missing XER tables and missing CSV columns gracefully using `ensure_columns()`.

## What This Enables

- **Outage Decision Support:** "Should we defer this work?" / "What is the risk of adding emergent scope?"
- **RCA Integration:** Link delays → tasks → dependencies → systems
- **Predictive Analytics:** Outage duration forecasting, critical path instability detection
- **Optimization:** Resequencing tasks, resource leveling scenarios
