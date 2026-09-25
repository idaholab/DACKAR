# P6 XER ingestion pipeline

## What was added

The package now includes a direct XER parser and transformer for real Primavera P6 exports.

New modules:

- `src/outage_model/transforms/p6_xer.py`
- `src/outage_model/loaders/xer_loader.py`
- `examples/load_sample_xer.py`
- `examples/sample_xer/sample_project.xer`

## Supported XER tables

The parser reads the standard `%T`, `%F`, `%R`, `%E` XER sections and converts each table into a pandas DataFrame.

The transformer currently maps these P6 tables into the canonical schema:

- `PROJECT` -> `outages`, `schedule_versions`
- `PROJWBS` -> `wbs`
- `TASK` -> `schedule_tasks`, `task_constraints`
- `TASKPRED` -> `dependencies`
- `RSRC` -> `resources`
- `TASKRSRC` -> `resource_assignments`
- `CALENDAR` -> `calendars`
- `ACTVTYPE` + `ACTVCODE` -> `activity_codes`
- `TASKACTV` -> `task_activity_codes`

## Pipeline flow

1. Read the XER file line by line.
2. Split it into logical tables using `%T`, `%F`, `%R`, and `%E` markers.
3. Convert each table to a pandas DataFrame with normalized column names.
4. Select the relevant project when the XER contains more than one project.
5. Map P6 table fields into the canonical outage schema.
6. Apply pandas type coercion using the canonical physical schema.
7. Return an `OutageDataset` object.

## Current assumptions

- `ScheduleTask` remains the central canonical object.
- Canonical task IDs are version-aware: `outage_id:schedule_version_id:task_id`.
- Milestone detection is derived from P6 `task_type` values.
- Task status is mapped from common P6 `status_code` values.
- Calendar details are preserved as raw calendar payload in `work_pattern_json`.

## Example usage

```python
from pathlib import Path

from outage_model.loaders.xer_loader import load_xer_dataset

xer_path = Path("examples/sample_xer/sample_project.xer")
dataset = load_xer_dataset(xer_path)

print(dataset.schedule_tasks.head())
print(dataset.dependencies.head())
```

## Notes

This is a strong production starter, but real site exports often vary by:

- P6 version
- export settings
- naming conventions
- optional tables present in the XER

The transformer is written to be tolerant of missing tables and missing columns, but it has not been tuned to a site-specific export profile yet.
