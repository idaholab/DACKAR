"""Shared configuration for the RCA unit-test suite.

Central optional-dependency shim — replaces the former per-file
``for _mod in (...): if _mod not in sys.modules: sys.modules[_mod] = MagicMock()``
blocks that appeared in ~70 modules.

Several RCA modules import optional integration packages (``neo4j`` / ``py2neo`` /
``chromadb`` / ``langchain_*``) at import time. When a package is genuinely absent
(for example the pip CI job, which installs only the core distribution) we inject a
``MagicMock`` so that collecting the tests which import production code at module
scope does not fail. When the real package is installed (the uv ``--all-groups`` CI
job, or a full dev environment) we leave it untouched so the tests exercise the real
integration.

Gating on ``importlib.util.find_spec`` (actual availability) instead of the old
``_mod not in sys.modules`` (import order) fixes the two problems the reviewer
flagged: a real installed package is never replaced process-wide by a mock, and
collection order can no longer decide whether a given test sees the mock or the real
module. Because only genuinely-absent modules are shimmed, there is nothing real to
restore afterwards.

This runs from ``pytest_configure`` — after this conftest is imported and before any
sibling test module is collected — so the shim is in place for module-scope
production imports.
"""
from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock

# Union of every optional integration module the per-file blocks used to mock.
# Parents are listed before their submodules.
_OPTIONAL_INTEGRATION_MODULES = (
    "neo4j",
    "py2neo",
    "chromadb",
    "langchain_chroma",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.embeddings",
    "langchain_core",
    "langchain_core.documents",
)


def _absent_optional_modules():
    """Return the optional modules that are not importable in this environment.

    All detection runs before any mock is injected, so a mocked parent can never
    make a real submodule look absent (or vice versa).
    """
    absent = []
    for name in _OPTIONAL_INTEGRATION_MODULES:
        if name in sys.modules:
            continue
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            # A dotted name whose parent is absent raises here — treat as absent.
            found = False
        if not found:
            absent.append(name)
    return absent


def pytest_configure(config):  # noqa: ARG001 - pytest hook signature
    for name in _absent_optional_modules():
        sys.modules.setdefault(name, MagicMock())
