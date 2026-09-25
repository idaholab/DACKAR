"""
P6 dataset adapter — bridges ``OutageDataset`` → ``Iterable[dict]`` for
``ActivityIngestionWorkflow``.

``OutageDataset`` (from ``P6_adapter``) contains DataFrames for all tables in
the P6 schedule model.  This adapter converts those DataFrames into the
``Iterable[dict]`` format expected by
``PandasActivityRepository.load_activities()`` — the existing ingestion entry
point.  Nothing inside ``outage_uncertainty`` knows whether the dicts came from
a P6 XER file, a CSV export, or a test fixture.

Field sourcing strategy
-----------------------
Sourced directly:
    ``activity_id``, ``outage_id``, ``raw_description``, timestamps,
    ``planned_duration_hours``

Computed in adapter:
    ``actual_duration_hours``  (``actual_finish − actual_start`` in hours)

Joined from related tables:
    ``plant_id``, ``unit_id``          — from the ``Outage`` object (caller-supplied)
    ``predecessor_ids``, ``successor_ids`` — ``Dependency`` table
    ``is_vendor_supported``, ``crew_size`` — ``ResourceAssignment`` ⋈ ``Resource``
    ``contractor_flag``                — ``Resource.org_unit`` membership (opt-in)
    ``is_emergent``, ``is_rework``     — ``ScopeChangeEvent.change_type``
    ``outage_phase``                   — time-window overlap with ``OutagePhase``

Delegated to ``LabelMapper`` NLP:
    ``discipline``, ``task_family``, ``component_family``,
    ``has_rp_hold``, ``requires_scaffold``, ``has_clearance``
    (all emitted as ``None`` / ``False`` by this adapter)

Stashed in ``metadata["p6"]``:
    ``task_code``, ``task_type``, ``critical_flag``, ``total_float_hours``,
    ``scope_origin``, ``wbs_id``, ``schedule_version_id``
    — preserved for audit and downstream use without polluting ``ActivityCase``
      fields.

Performance contract
--------------------
All DataFrame joins are pre-indexed into Python dicts/sets during
``_build_indices()`` at the start of ``iter_activity_rows()`` /
``get_activity_row()``.  Per-task lookup is O(1).  Indices are rebuilt for
each call (i.e. each ``OutageDataset``); no stale state across datasets.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# Normalise change_type strings to this set to detect emergent scope additions.
_DEFAULT_EMERGENT_TYPES: frozenset[str] = frozenset({"emergent", "scope_addition"})


# ---------------------------------------------------------------------------
# Value-coercion helpers — handle pd.NA, pd.NaT, float("nan"), None uniformly
# ---------------------------------------------------------------------------

def _na(v) -> bool:
    """Return True when *v* is any variety of missing value."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _to_float(v) -> float | None:
    if _na(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_str(v) -> str | None:
    if _na(v):
        return None
    s = str(v).strip()
    return s or None


def _to_bool(v, *, default: bool = False) -> bool:
    if _na(v):
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def _to_datetime(v) -> datetime | None:
    if _na(v):
        return None
    if isinstance(v, datetime):
        return v
    try:
        ts = pd.Timestamp(v)
        return None if ts is pd.NaT else ts.to_pydatetime()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class P6DatasetAdapter:
    """Convert a P6 ``OutageDataset`` into ``Iterable[dict]`` for ingestion.

    Parameters
    ----------
    emergent_change_types:
        Set of ``ScopeChangeEvent.change_type`` values (lowercased) that signal
        emergent work.  Defaults to ``{"emergent", "scope_addition"}``.
    contractor_org_units:
        Set of ``Resource.org_unit`` values (lowercased) that indicate a
        contractor resource.  When empty (default) ``contractor_flag`` is left
        as ``None`` (unknown) for all tasks — opt-in only.
    """

    def __init__(
        self,
        emergent_change_types: set[str] | None = None,
        contractor_org_units: set[str] | None = None,
    ) -> None:
        self._emergent_types: frozenset[str] = (
            frozenset(t.lower().strip() for t in emergent_change_types)
            if emergent_change_types is not None
            else _DEFAULT_EMERGENT_TYPES
        )
        self._contractor_orgs: frozenset[str] = (
            frozenset(o.lower().strip() for o in contractor_org_units)
            if contractor_org_units
            else frozenset()
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def iter_activity_rows(
        self,
        dataset,   # OutageDataset
        outage,    # outage_model.models.Outage
        *,
        skip_milestones: bool = True,
    ) -> Iterable[dict]:
        """Yield one ``ActivityCase``-compatible dict per ``ScheduleTask``.

        Parameters
        ----------
        dataset:
            A populated ``OutageDataset`` (from ``P6_adapter``).
        outage:
            The ``Outage`` record that owns this dataset — provides
            ``plant_id`` and ``unit_id``, which are not on ``ScheduleTask``.
        skip_milestones:
            When ``True`` (default), tasks with ``milestone_flag = True`` are
            omitted.  Milestones have zero duration by definition and would
            corrupt the duration distribution in the analog index.

        Yields
        ------
        dict
            Flat dict ready for ``PandasActivityRepository.load_activities()``.
        """
        tasks_df = getattr(dataset, "schedule_tasks", pd.DataFrame())
        if tasks_df.empty:
            logger.warning("P6DatasetAdapter: schedule_tasks is empty; no rows yielded")
            return

        indices = self._build_indices(dataset)
        n_total = 0
        n_skipped = 0

        for row in tasks_df.itertuples(index=False):
            n_total += 1
            if skip_milestones and _to_bool(getattr(row, "milestone_flag", False)):
                n_skipped += 1
                continue
            yield self._row_to_dict(row, outage, indices)

        logger.debug(
            "P6DatasetAdapter.iter_activity_rows: %d tasks → %d yielded, %d milestones skipped",
            n_total,
            n_total - n_skipped,
            n_skipped,
        )

    def get_activity_row(
        self,
        task_id: str,
        dataset,   # OutageDataset
        outage,    # outage_model.models.Outage
    ) -> dict:
        """Return a single ``ActivityCase``-compatible dict for *task_id*.

        Parameters
        ----------
        task_id:
            The ``ScheduleTask.task_id`` to retrieve.
        dataset:
            A populated ``OutageDataset``.
        outage:
            The owning ``Outage`` record.

        Raises
        ------
        KeyError
            If *task_id* is not found in ``dataset.schedule_tasks``.
        """
        tasks_df = getattr(dataset, "schedule_tasks", pd.DataFrame())
        if tasks_df.empty or "task_id" not in tasks_df.columns:
            raise KeyError(f"P6DatasetAdapter: task_id '{task_id}' not found — schedule_tasks is empty")

        matching = tasks_df[tasks_df["task_id"].astype(str) == str(task_id)]
        if matching.empty:
            raise KeyError(f"P6DatasetAdapter: task_id '{task_id}' not found in schedule_tasks")

        indices = self._build_indices(dataset)
        row = next(matching.itertuples(index=False))
        return self._row_to_dict(row, outage, indices)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_indices(self, dataset) -> dict:
        """Pre-index all join tables for O(1) per-task lookup.

        Returns a plain dict of index structures so the adapter has no
        mutable state between calls — each call to ``iter_activity_rows`` /
        ``get_activity_row`` gets a fresh set of indices built from the
        supplied ``dataset``.
        """
        pred_index, succ_index = self._build_dependency_index(
            getattr(dataset, "dependencies", pd.DataFrame())
        )
        vendor_tasks, contractor_tasks, crew_sizes = self._build_resource_indices(
            getattr(dataset, "resource_assignments", pd.DataFrame()),
            getattr(dataset, "resources", pd.DataFrame()),
        )
        emergent_tasks, rework_tasks = self._build_scope_change_index(
            getattr(dataset, "scope_change_events", pd.DataFrame())
        )
        phase_windows = self._build_phase_windows(
            getattr(dataset, "outage_phases", pd.DataFrame())
        )
        return {
            "pred": pred_index,
            "succ": succ_index,
            "vendor": vendor_tasks,
            "contractor": contractor_tasks,
            "crew": crew_sizes,
            "emergent": emergent_tasks,
            "rework": rework_tasks,
            "phases": phase_windows,
        }

    def _build_dependency_index(
        self, deps_df: pd.DataFrame
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Return (predecessor_index, successor_index) keyed by task_id."""
        pred: dict[str, list[str]] = defaultdict(list)  # succ → [preds]
        succ: dict[str, list[str]] = defaultdict(list)  # pred → [succs]

        if deps_df.empty:
            return pred, succ

        req = {"predecessor_task_id", "successor_task_id"}
        if not req.issubset(deps_df.columns):
            logger.debug("P6DatasetAdapter: dependencies missing columns %s; skipping", req - set(deps_df.columns))
            return pred, succ

        for row in deps_df.itertuples(index=False):
            p = _to_str(getattr(row, "predecessor_task_id", None))
            s = _to_str(getattr(row, "successor_task_id", None))
            if p and s:
                pred[s].append(p)
                succ[p].append(s)

        return dict(pred), dict(succ)

    def _build_resource_indices(
        self,
        assignments_df: pd.DataFrame,
        resources_df: pd.DataFrame,
    ) -> tuple[set[str], set[str], dict[str, int]]:
        """Return (vendor_task_ids, contractor_task_ids, crew_sizes_by_task)."""
        vendor_tasks: set[str] = set()
        contractor_tasks: set[str] = set()
        crew_sizes: dict[str, int] = {}

        if assignments_df.empty or "task_id" not in assignments_df.columns:
            return vendor_tasks, contractor_tasks, crew_sizes

        # Crew size: count of distinct resources assigned to each task.
        if "resource_id" in assignments_df.columns:
            for task_id, grp in assignments_df.groupby("task_id"):
                tid = str(task_id)
                crew_sizes[tid] = int(grp["resource_id"].nunique())

        if resources_df.empty or "resource_id" not in resources_df.columns:
            return vendor_tasks, contractor_tasks, crew_sizes

        # Join assignments to resources for vendor / contractor detection.
        res_cols = ["resource_id"]
        if "vendor" in resources_df.columns:
            res_cols.append("vendor")
        if "org_unit" in resources_df.columns:
            res_cols.append("org_unit")

        merged = assignments_df.merge(
            resources_df[res_cols],
            on="resource_id",
            how="left",
        )

        for task_id, grp in merged.groupby("task_id"):
            tid = str(task_id)

            if "vendor" in grp.columns:
                vendors = grp["vendor"].dropna()
                non_empty = vendors[vendors.astype(str).str.strip() != ""]
                if not non_empty.empty:
                    vendor_tasks.add(tid)

            if "org_unit" in grp.columns and self._contractor_orgs:
                orgs = grp["org_unit"].dropna().astype(str).str.strip().str.lower()
                if orgs.isin(self._contractor_orgs).any():
                    contractor_tasks.add(tid)

        return vendor_tasks, contractor_tasks, crew_sizes

    def _build_scope_change_index(
        self, scope_df: pd.DataFrame
    ) -> tuple[set[str], set[str]]:
        """Return (emergent_task_ids, rework_task_ids)."""
        emergent: set[str] = set()
        rework: set[str] = set()

        if scope_df.empty:
            return emergent, rework

        req = {"task_id", "change_type"}
        if not req.issubset(scope_df.columns):
            return emergent, rework

        for row in scope_df.itertuples(index=False):
            tid = _to_str(getattr(row, "task_id", None))
            ct = _to_str(getattr(row, "change_type", None))
            if tid is None or ct is None:
                continue
            ct_lower = ct.lower()
            if ct_lower in self._emergent_types:
                emergent.add(tid)
            if ct_lower == "rework":
                rework.add(tid)

        return emergent, rework

    def _build_phase_windows(
        self, phases_df: pd.DataFrame
    ) -> list[tuple]:
        """Return list of (start, end, phase_name) sorted by phase sequence."""
        windows: list[tuple] = []

        if phases_df.empty:
            return windows

        req = {"phase_name", "start_planned", "finish_planned"}
        if not req.issubset(phases_df.columns):
            return windows

        rows = []
        for row in phases_df.itertuples(index=False):
            start = _to_datetime(getattr(row, "start_planned", None))
            end = _to_datetime(getattr(row, "finish_planned", None))
            name = _to_str(getattr(row, "phase_name", None))
            seq = _to_float(getattr(row, "sequence", None)) or 0.0
            if start is None or end is None or name is None:
                continue
            rows.append((seq, start, end, name))

        rows.sort(key=lambda x: x[0])
        return [(s, e, n) for _, s, e, n in rows]

    # ------------------------------------------------------------------
    # Per-row conversion
    # ------------------------------------------------------------------

    def _row_to_dict(self, row, outage, indices: dict) -> dict:
        """Convert one ``ScheduleTask`` itertuples row to an ActivityCase dict."""
        task_id = _to_str(getattr(row, "task_id", None)) or ""

        # Timestamps
        planned_start = _to_datetime(getattr(row, "planned_start", None))
        planned_finish = _to_datetime(getattr(row, "planned_finish", None))
        actual_start = _to_datetime(getattr(row, "actual_start", None))
        actual_finish = _to_datetime(getattr(row, "actual_finish", None))

        # Compute actual duration from timestamps (more reliable than a stored field)
        actual_duration: float | None = None
        if actual_start is not None and actual_finish is not None:
            delta = actual_finish - actual_start
            actual_duration = delta.total_seconds() / 3600.0
            if actual_duration < 0.0:
                logger.warning(
                    "P6DatasetAdapter: task '%s' has actual_finish < actual_start; "
                    "setting actual_duration_hours = None",
                    task_id,
                )
                actual_duration = None

        # Outage phase via time-window overlap
        outage_phase: str | None = None
        if planned_start is not None:
            for start, end, name in indices["phases"]:
                if start <= planned_start <= end:
                    outage_phase = name
                    break

        # contractor_flag: True/False if contractor detection is configured,
        # None if contractor_org_units was not provided (unknown, not "no").
        if self._contractor_orgs:
            contractor_flag: bool | None = task_id in indices["contractor"]
        else:
            contractor_flag = None

        return {
            # -- identity -------------------------------------------------------
            "activity_id":             task_id,
            "outage_id":               _to_str(getattr(row, "outage_id", None)) or getattr(outage, "outage_id", ""),
            "plant_id":                getattr(outage, "plant_id", None) or "",
            "unit_id":                 getattr(outage, "unit_id", None),
            # -- description (LabelMapper will fill discipline/task_family/component_family)
            "raw_description":         _to_str(getattr(row, "task_name", None)) or "",
            # -- schedule -------------------------------------------------------
            "planned_start":           planned_start,
            "planned_finish":          planned_finish,
            "actual_start":            actual_start,
            "actual_finish":           actual_finish,
            "planned_duration_hours":  _to_float(getattr(row, "planned_duration_hours", None)),
            "actual_duration_hours":   actual_duration,
            # -- taxonomy (delegated to LabelMapper NLP) ------------------------
            "discipline":              None,
            "task_family":             None,
            "component_family":        None,
            # -- execution mode flags (text-based flags delegated to LabelMapper;
            #    structural flags from joined tables set here) ------------------
            "has_rp_hold":             False,   # LabelMapper: keyword "RP hold"
            "requires_scaffold":       False,   # LabelMapper: keyword "scaffold"
            "has_clearance":           False,   # LabelMapper: keyword "clearance"
            "is_vendor_supported":     task_id in indices["vendor"],
            "is_emergent":             task_id in indices["emergent"],
            "is_rework":               task_id in indices["rework"],
            # -- resource -------------------------------------------------------
            "crew_size":               indices["crew"].get(task_id),
            "contractor_flag":         contractor_flag,
            # -- schedule context -----------------------------------------------
            "outage_phase":            outage_phase,
            "predecessor_ids":         list(indices["pred"].get(task_id, [])),
            "successor_ids":           list(indices["succ"].get(task_id, [])),
            # -- provenance stash (not used by UQ pipeline; preserved for audit)
            "metadata": {
                "p6": {
                    "task_code":             _to_str(getattr(row, "task_code", None)),
                    "task_type":             _to_str(getattr(row, "task_type", None)),
                    "critical_flag":         _to_bool(getattr(row, "critical_flag", None)),
                    "total_float_hours":     _to_float(getattr(row, "total_float_hours", None)),
                    "scope_origin":          _to_str(getattr(row, "scope_origin", None)),
                    "wbs_id":                _to_str(getattr(row, "wbs_id", None)),
                    "schedule_version_id":   _to_str(getattr(row, "schedule_version_id", None)),
                }
            },
        }
