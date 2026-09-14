from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Protocol

from ._util import clamp01, parse_dt as _parse_dt
from .models import AnomalyRecord


class HistorianAdapter(Protocol):
    """Fetches pre-flagged anomaly records for a set of sensors in a window."""

    def get_anomalies(
        self,
        sensor_ids: list[str],
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[list[AnomalyRecord], list[dict]]:
        """Return ``(anomaly_records, gaps)`` for *sensor_ids* within the window.

        Args:
            sensor_ids: Sensor/tag identifiers to fetch anomalies for.
            window_start: Inclusive UTC-aware start of the query window.
            window_end: Inclusive UTC-aware end of the query window.

        Returns:
            A tuple of the matched :class:`~.models.AnomalyRecord` list and a
            list of gap dicts (one per sensor with no data / a fetch failure).
        """
        ...


class NullHistorianAdapter:
    """Graceful-degradation adapter used when historian is unavailable."""

    def get_anomalies(
        self,
        sensor_ids: list[str],
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[list[AnomalyRecord], list[dict]]:
        gaps = [
            {"sensor_id": sid, "component_id": None, "reason": "historian_unavailable"}
            for sid in sensor_ids
        ]
        return [], gaps


class InfileHistorianAdapter:
    """Reads pre-flagged anomalies from a JSON/CSV export."""

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def get_anomalies(
        self,
        sensor_ids: list[str],
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[list[AnomalyRecord], list[dict]]:
        if not self.source_path.exists():
            return [], [
                {"sensor_id": sid, "component_id": None, "reason": "api_error"}
                for sid in sensor_ids
            ]
        records = self._load_rows()
        allowed = set(sensor_ids)
        out: List[AnomalyRecord] = []
        present_sensor_ids = set()
        for row in records:
            sensor_id = str(row.get("sensor_id") or "").strip()
            if not sensor_id or sensor_id not in allowed:
                continue
            ts_start = _parse_dt(row.get("timestamp_start"))
            ts_end = _parse_dt(row.get("timestamp_end")) or ts_start
            if ts_start is None or ts_end is None:
                continue
            if ts_end < window_start or ts_start > window_end:
                continue
            present_sensor_ids.add(sensor_id)
            out.append(
                AnomalyRecord(
                    sensor_id=sensor_id,
                    component_id=row.get("component_id"),
                    timestamp_start=ts_start,
                    timestamp_end=ts_end,
                    pattern=str(row.get("pattern") or "unknown"),
                    # Clamp at ingest to [0,1] so historian severities match
                    # the baseline anomalies (clamped in builder._baseline_anomalies)
                    # rather than leaving an unclamped latent surprise (MR#49 review).
                    severity=clamp01(float(row.get("severity") or 0.0)),
                    source="historian",
                    raw_value_start=_to_float(row.get("raw_value_start")),
                    raw_value_peak=_to_float(row.get("raw_value_peak")),
                    units=row.get("units"),
                )
            )

        gaps: List[dict] = []
        for sid in sensor_ids:
            if sid not in present_sensor_ids:
                gaps.append(
                    {"sensor_id": sid, "component_id": None, "reason": "no_anomalies_in_window"}
                )
        return out, gaps

    def _load_rows(self) -> List[dict]:
        if self.source_path.suffix.lower() == ".json":
            data = json.loads(self.source_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if isinstance(data.get("records"), list):
                    return [x for x in data["records"] if isinstance(x, dict)]
                return []
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            return []

        rows: List[dict] = []
        with self.source_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
        return rows


class OSIsoftPIHistorianAdapter:
    """NOT IMPLEMENTED — placeholder shim for a future OSIsoft PI integration.

    This adapter is contract-compatible but **not** wired to the PI Web API.
    Every call reports each sensor as ``historian_unavailable`` and returns no
    anomalies, so it must not be mistaken for a functional PI integration. Use
    :class:`InfileHistorianAdapter` for real data until PI is implemented.
    """

    def __init__(self) -> None:
        pass

    def get_anomalies(
        self,
        sensor_ids: list[str],
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[list[AnomalyRecord], list[dict]]:
        # TODO(signal-evidence): wire to the PI Web API. Until then this shim
        # deliberately reports every sensor as unavailable (see class docstring).
        return [], [
            {"sensor_id": sid, "component_id": None, "reason": "historian_unavailable"}
            for sid in sensor_ids
        ]


def _to_float(value: object) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value))
    except Exception:
        return None
