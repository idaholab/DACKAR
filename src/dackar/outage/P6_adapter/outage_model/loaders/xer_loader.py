from __future__ import annotations

from pathlib import Path

from outage_model.dataset import OutageDataset
from outage_model.transforms.p6_xer import P6XERTransformer


def load_xer_dataset(
    xer_path: str | Path,
    outage_id: str | None = None,
    outage_name: str | None = None,
    schedule_version_id: str | None = None,
    version_name: str | None = None,
    version_type: str = "working",
    project_id: str | None = None,
) -> OutageDataset:
    transformer = P6XERTransformer(
        outage_id=outage_id,
        outage_name=outage_name,
        schedule_version_id=schedule_version_id,
        version_name=version_name,
        version_type=version_type,
        project_id=project_id,
    )
    return transformer.transform_file(xer_path)
