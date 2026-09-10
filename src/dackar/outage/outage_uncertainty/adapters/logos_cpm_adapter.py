"""
LOGOS CPM adapter for Stage E schedule impact assessment.

Provides two injectable dependencies consumed by
:class:`~dackar.outage.stages.stage_e_schedule.ScheduleImpactAssessor`:

``LogosCPMScheduleLoader``
    Callable ``(outage_id, version) → OutageData``.
    Resolves the JSON file for the requested outage + version from a
    configurable root directory and delegates to
    ``OutageData.from_json_file()``.

``LogosCPMScheduleGraphBuilder``
    Object with ``.build(outage_data) → Pert``.
    Constructs a ``Pert`` instance from a loaded ``OutageData``, ensures
    ``generateInfo()`` has been called, and returns the ready-to-query Pert.

File-naming convention
----------------------
The loader expects JSON files under *data_root* following::

    <data_root>/<outage_id>/<version>.json

For example::

    schedules/RF-22/working.json
    schedules/RF-22/baseline.json
    schedules/RF-22/as_run.json

If the file is not found under the versioned path, the loader falls back to
``<data_root>/<outage_id>.json`` (legacy single-file layout).

LOGOS import path
-----------------
LOGOS is not installed as a package; its source tree must be on ``sys.path``.
Pass ``logos_src_root`` to :class:`LogosCPMScheduleGraphBuilder` to inject it
at construction time.  The default is ``/Users/mandd/projects/LOGOS/src``.

Usage
-----
::

    from dackar.outage.outage_uncertainty.adapters.logos_cpm_adapter import (
        LogosCPMScheduleLoader,
        LogosCPMScheduleGraphBuilder,
    )
    from dackar.outage.stages.stage_e_schedule import (
        ScheduleImpactAssessor,
        ScheduleImpactConfig,
    )

    loader  = LogosCPMScheduleLoader(data_root="schedules/")
    builder = LogosCPMScheduleGraphBuilder()

    assessor = ScheduleImpactAssessor(
        config=ScheduleImpactConfig(),
        schedule_loader=loader,
        schedule_graph_builder=builder,
    )
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)

# Default path where the LOGOS source tree lives on this machine.
_DEFAULT_LOGOS_SRC = "/Users/mandd/projects/LOGOS/src"


# ---------------------------------------------------------------------------
# Schedule loader
# ---------------------------------------------------------------------------

class LogosCPMScheduleLoader:
    """Load a LOGOS ``OutageData`` from a JSON file on disk.

    Parameters
    ----------
    data_root:
        Directory that contains per-outage schedule files.  See module
        docstring for the expected file layout.
    logos_src_root:
        Path to ``LOGOS/src`` — required so that ``OutageData`` can be
        imported.  Defaults to :data:`_DEFAULT_LOGOS_SRC`.
    """

    def __init__(
        self,
        data_root: str,
        logos_src_root: str = _DEFAULT_LOGOS_SRC,
    ) -> None:
        self._data_root = Path(data_root)
        self._ensure_logos_on_path(logos_src_root)

    # -- callable interface expected by ScheduleImpactAssessor ---------------

    def __call__(self, outage_id: str, version: str = "working") -> Any:
        """Return an ``OutageData`` for *outage_id* at the requested *version*.

        Parameters
        ----------
        outage_id:
            Canonical outage identifier, e.g. ``"RF-22"``.
        version:
            Schedule version: ``"baseline"``, ``"working"`` (default), or
            ``"as_run"``.

        Raises
        ------
        FileNotFoundError
            If no suitable JSON file is found for the outage / version.
        ImportError
            If LOGOS cannot be imported from the configured path.
        """
        from CPM.outage_data import OutageData  # type: ignore[import]

        filepath = self._resolve_filepath(outage_id, version)
        LOGGER.debug(
            "Loading OutageData for outage=%s version=%s from %s",
            outage_id, version, filepath,
        )
        outage_data = OutageData.from_json_file(str(filepath))
        # Inject version_id into outage_config if absent so Stage E can read it
        if "version_id" not in outage_data.outage_config:
            outage_data.outage_config["version_id"] = f"{outage_id}/{version}"
        return outage_data

    # -- internal helpers ----------------------------------------------------

    def _resolve_filepath(self, outage_id: str, version: str) -> Path:
        """Return the JSON path for *outage_id* + *version*."""
        # Primary: <data_root>/<outage_id>/<version>.json
        primary = self._data_root / outage_id / f"{version}.json"
        if primary.exists():
            return primary
        # Fallback: <data_root>/<outage_id>.json (legacy)
        fallback = self._data_root / f"{outage_id}.json"
        if fallback.exists():
            LOGGER.warning(
                "Version-specific file not found at %s; "
                "falling back to legacy layout %s.",
                primary, fallback,
            )
            return fallback
        raise FileNotFoundError(
            f"No LOGOS schedule JSON found for outage '{outage_id}' "
            f"version '{version}'.  Searched:\n"
            f"  {primary}\n  {fallback}"
        )

    @staticmethod
    def _ensure_logos_on_path(logos_src_root: str) -> None:
        """Add *logos_src_root* to ``sys.path`` if not already present."""
        if logos_src_root not in sys.path:
            sys.path.insert(0, logos_src_root)
            LOGGER.debug("Added LOGOS source root to sys.path: %s", logos_src_root)


# ---------------------------------------------------------------------------
# Schedule graph builder
# ---------------------------------------------------------------------------

class LogosCPMScheduleGraphBuilder:
    """Build a LOGOS ``Pert`` from a loaded ``OutageData``.

    The returned ``Pert`` has already had ``generateInfo()`` called, so
    ``infoDict`` is populated with ES/EF/LS/LF/slack values.

    Parameters
    ----------
    logos_src_root:
        Path to ``LOGOS/src``.  Defaults to :data:`_DEFAULT_LOGOS_SRC`.
    """

    def __init__(self, logos_src_root: str = _DEFAULT_LOGOS_SRC) -> None:
        self._ensure_logos_on_path(logos_src_root)

    def build(self, outage_data: Any) -> Any:
        """Construct and return a ``Pert`` for *outage_data*.

        Parameters
        ----------
        outage_data:
            A LOGOS ``OutageData`` instance (as returned by
            :class:`LogosCPMScheduleLoader`).

        Returns
        -------
        ``Pert`` instance with CPM state fully computed (``infoDict``
        populated, ``task_to_activity`` populated).
        """
        from CPM.pert import Pert  # type: ignore[import]

        LOGGER.debug(
            "Building Pert for outage=%s", outage_data.outage_id
        )
        # Pert.__init__ calls _build_graph_from_outage_data() → generateInfo()
        # when outage_data is provided and graph is None.
        pert = Pert(outage_data=outage_data)

        # Defensive: ensure infoDict is populated even if __init__ path changes.
        if not pert.infoDict:
            pert.generateInfo()

        return pert

    @staticmethod
    def _ensure_logos_on_path(logos_src_root: str) -> None:
        if logos_src_root not in sys.path:
            sys.path.insert(0, logos_src_root)
            LOGGER.debug("Added LOGOS source root to sys.path: %s", logos_src_root)
