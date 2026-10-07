"""conftest for the RCA scenario integration tests.

The scenario drivers import the shared fixture-mode harness
(``run_helpers``) and the assertion vocabulary (``assertion_helpers``) by
bare module name — matching how the companion show-and-tell notebooks import
them. Those modules live in ``tests/RCA/scenario/shared/`` and reach the
production code via absolute ``dackar.RCA.*`` imports (``src`` is already on
``sys.path`` via ``pytest.ini``'s ``pythonpath = src``).

Putting ``shared/`` on ``sys.path`` here means each driver can simply
``from run_helpers import ...`` / ``from assertion_helpers import ...`` with no
per-file path manipulation.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
