# Physical schema for pandas

This package uses a normalized table design intended for pandas, parquet, and SQL persistence.

## Recommended core tables

### outages
One row per outage.

### schedule_versions
One row per schedule baseline, revision, snapshot, or as-run schedule.

### wbs
One row per WBS element per schedule version.

### schedule_tasks
One row per schedule activity or milestone.

### dependencies
One row per predecessor-successor relationship.

### resources
One row per resource or crew.

### resource_assignments
One row per task-resource assignment.

### calendars
One row per work calendar.

### activity_codes
One row per coded classification value.

### task_activity_codes
Many-to-many bridge between schedule tasks and activity codes.

## Why this design works well in pandas

- It preserves normalized business meaning
- It makes joins explicit and traceable
- It supports version comparison without overwriting history
- It supports graph analytics from the dependencies table
- It is easy to persist to parquet or SQL

## Recommended key conventions

- Use stable canonical keys, not only vendor keys
- Keep all source record identifiers for traceability
- Preserve every schedule version separately
- Use nullable pandas dtypes such as `string`, `Float64`, `Int64`, and `boolean`

## Join examples

### Task to WBS
`schedule_tasks.wbs_id -> wbs.wbs_id`

### Task to outage
`schedule_tasks.outage_id -> outages.outage_id`

### Task to schedule version
`schedule_tasks.schedule_version_id -> schedule_versions.schedule_version_id`

### Dependency predecessor and successor
`dependencies.predecessor_task_id -> schedule_tasks.task_id`
`dependencies.successor_task_id -> schedule_tasks.task_id`

### Task to activity code
`task_activity_codes.task_id -> schedule_tasks.task_id`
`task_activity_codes.activity_code_id -> activity_codes.activity_code_id`

