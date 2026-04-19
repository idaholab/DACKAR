"""
artifact_store — Concrete ArtifactStore and SchemaValidator implementations.

Extracted from rca_reasoning_orchestrator.py.  The parent module re-exports
both classes for backward-compatible imports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

JsonDict = Dict[str, Any]


class NoOpSchemaValidator:
    def validate(self, artifact_name: str, payload: JsonDict) -> None:
        if not isinstance(payload, dict):
            raise TypeError(f"{artifact_name} must be a JSON object")

    def validate_artifact(self, artifact_name: str, payload: JsonDict) -> JsonDict:
        self.validate(artifact_name, payload)
        return {
            "ok": True,
            "issues": [],
            "artifact": artifact_name,
            "mode": "noop",
        }

    def validate_run_bundle(self, **kwargs: Any) -> JsonDict:
        for artifact_name, payload in kwargs.items():
            if payload is None:
                continue
            self.validate(artifact_name, payload)
        return {
            "ok": True,
            "issues": [],
            "artifact": "bundle",
            "mode": "noop",
        }


class FileArtifactStore:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def save(self, run_id: str, artifact_name: str, payload: JsonDict) -> str:
        run_dir = self.root_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{artifact_name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)

    def save_list(self, run_id: str, artifact_name: str, payload: List[JsonDict]) -> str:
        run_dir = self.root_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{artifact_name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)
