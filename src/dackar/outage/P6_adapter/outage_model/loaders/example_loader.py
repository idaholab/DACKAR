from __future__ import annotations

from pathlib import Path

from outage_model.transforms.p6_csv import P6CsvTransformer


DEFAULT_OUTAGE_ID = "RFO-2026-U1"
DEFAULT_OUTAGE_NAME = "Unit 1 Refueling Outage 2026"
DEFAULT_VERSION_ID = "RFO-2026-U1:BL1"
DEFAULT_VERSION_NAME = "Baseline 1"


def load_mock_dataset(base_dir: Path):
    transformer = P6CsvTransformer(
        outage_id=DEFAULT_OUTAGE_ID,
        outage_name=DEFAULT_OUTAGE_NAME,
        schedule_version_id=DEFAULT_VERSION_ID,
        version_name=DEFAULT_VERSION_NAME,
        version_type="baseline",
    )
    return transformer.transform_directory(base_dir)
