from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

from .models import AnomalyRecord


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


class HistorianAdapter(Protocol):
    def get_anomalies(
        self,
        sensor_ids: list[str],
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[list[AnomalyRecord], list[dict]]:
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
                    severity=float(row.get("severity") or 0.0),
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
    """Production placeholder: contract-compatible PI adapter shim."""

    def __init__(self) -> None:
        pass

    def get_anomalies(
        self,
        sensor_ids: list[str],
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[list[AnomalyRecord], list[dict]]:
        # Not wired to PI Web API in this phase.
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
