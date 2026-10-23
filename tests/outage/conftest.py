"""
Shared fixtures for outage module tests.

Production imports resolve via ``pythonpath = src`` in ``pytest.ini`` (the
packages are ``dackar.outage.*``); no sys.path manipulation is needed for those.

Demo modules: the two workflow demos live under ``examples/outage/`` (outside
``src/``, matching ``examples/RCA/demos/``), so they are not importable as
``dackar.outage.demos.*``.  This conftest puts each demo's own directory on
``sys.path`` so ``test_orchestrator_e2e`` / ``test_wf2_pipeline`` can import them
by bare module name (``demo_scenarios`` / ``demo_data`` / ``pipeline``) — the
same way the companion demo notebooks do, and mirroring
``tests/RCA/scenario/conftest.py``.

P6 CSV/XER fixtures:
    sample_xer_path   — path to the committed sample XER
                        (P6_adapter/examples/sample_xer/sample_project.xer).
    make_xer          — write an inline XER string to tmp and return the path.
    MINIMAL_XER       — a minimal single-project XER string.
    mock_csv_dir      — path to the mock P6 CSV export.  That directory
                        (P6_adapter/examples/mock_p6_export/) is NOT committed on
                        main; a test that requests this fixture is skipped with a
                        pointer to build_mock_dataset.py when it is absent, rather
                        than failing later with a confusing file-not-found.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTAGE_ROOT = _REPO_ROOT / "src" / "dackar" / "outage"
_P6_ADAPTER_ROOT = _OUTAGE_ROOT / "P6_adapter"

# Expose the two workflow demos (examples/outage/) by bare module name.  Only
# the names the tests import are distinct across the two dirs
# (demo_scenarios / demo_data / pipeline); demo_plots exists in both but no test
# imports it, so the shared-name case does not arise.
_EXAMPLES_OUTAGE = _REPO_ROOT / "examples" / "outage"
for _demo_dir in ("unexpected_act_workflow_1", "unexpected_act_workflow_2"):
    _p = str(_EXAMPLES_OUTAGE / _demo_dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

MOCK_CSV_DIR = _P6_ADAPTER_ROOT / "examples" / "mock_p6_export"
SAMPLE_XER_PATH = _P6_ADAPTER_ROOT / "examples" / "sample_xer" / "sample_project.xer"


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_csv_dir() -> Path:
    if not MOCK_CSV_DIR.is_dir():
        pytest.skip(
            f"mock P6 CSV export not found at {MOCK_CSV_DIR}; it is generated, "
            "not committed — run P6_adapter/examples/build_mock_dataset.py to "
            "create it (the CSVs arrive with the P6 dataset MR)"
        )
    return MOCK_CSV_DIR


@pytest.fixture(scope="session")
def sample_xer_path() -> Path:
    return SAMPLE_XER_PATH


# ---------------------------------------------------------------------------
# Inline XER builder helpers (used by multiple test modules)
# ---------------------------------------------------------------------------

def make_xer(tmp_path: Path, content: str, filename: str = "test.xer") -> Path:
    """Write an XER string to a temporary file and return its path."""
    path = tmp_path / filename
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


MINIMAL_XER = """\
    ERMHDR\t8.4\t2026-01-01\ttest\tuser
    %T\tPROJECT
    %F\tproj_id\tproj_short_name\tproj_name\tlast_recalc_date\tcreate_date\tstatus_code
    %R\t1\tP-ALPHA\tAlpha Project\t2026-01-01 00:00\t2025-12-01 00:00\tTK_Active
    %E
    %T\tTASK
    %F\ttask_id\tproj_id\twbs_id\ttask_code\ttask_name\ttask_type\tstatus_code\ttarget_start_date\ttarget_end_date\ttarget_drtn_hr_cnt\tremain_drtn_hr_cnt\ttotal_float_hr_cnt\tfree_float_hr_cnt\tphys_complete_pct\tclndr_id\tcstr_type\tcstr_date
    %R\t101\t1\t201\tT01\tFirst task\tTT_Task\tTK_Complete\t2026-01-10 06:00\t2026-01-11 18:00\t36\t0\t0\t0\t100\t501\t\t
    %R\t102\t1\t201\tT02\tFinal milestone\tTT_FinMile\tTK_NotStart\t2026-01-15 00:00\t2026-01-15 00:00\t0\t0\t8\t0\t0\t501\t\t
    %E
    %T\tTASKPRED
    %F\ttask_pred_id\ttask_id\tpred_task_id\tpred_type\tlag_hr_cnt
    %R\t401\t102\t101\tFS\t0
    %E
"""
"""A minimal single-project XER string (not yet written to disk)."""
