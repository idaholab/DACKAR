"""
artifact_store — Concrete ArtifactStore and SchemaValidator implementations.

Extracted from rca_reasoning_orchestrator.py.  The parent module re-exports
both classes for backward-compatible imports.
"""
from __future__ import annotations

import json
import os
import tempfile
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
        return self._write_atomic(run_id, artifact_name, payload)

    def save_list(self, run_id: str, artifact_name: str, payload: List[JsonDict]) -> str:
        return self._write_atomic(run_id, artifact_name, payload)

    def load(self, run_id: str, artifact_name: str) -> Any:
        """Load and return a previously saved artifact, or None if absent."""
        path = self.root_dir / run_id / f"{artifact_name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def is_run_complete(self, run_id: str) -> bool:
        """Return True only if run_status.json exists and run_complete is True.

        External callers (notebooks, replay scripts) should check this before
        loading or re-validating artifacts from a run directory.  A run that
        crashed mid-pipeline will have run_complete=False (or no run_status.json
        at all), and its artifacts should not be treated as authoritative.
        """
        status = self.load(run_id, "run_status")
        return bool((status or {}).get("run_complete", False))

    def _write_atomic(self, run_id: str, artifact_name: str, payload: Any) -> str:
        """Write payload to <run_dir>/<artifact_name>.json via temp-file + rename.

        The rename is atomic on POSIX and near-atomic on Windows (py3.3+).
        A reader can never observe a partial write.
        """
        run_dir = self.root_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / f"{artifact_name}.json"
        text = json.dumps(payload, indent=2, default=str)
        fd, tmp_path = tempfile.mkstemp(dir=run_dir, prefix=f".{artifact_name}_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            Path(tmp_path).replace(target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return str(target)
