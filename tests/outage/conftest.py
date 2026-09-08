"""
Shared fixtures for outage module tests.

Adds two directories to sys.path so that packages are importable regardless
of whether they are installed via pip:

- outage/src/          → makes ``outage_uncertainty`` importable
- outage/P6_adapter/   → makes ``outage_model`` importable
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

_OUTAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "dackar" / "outage"

# outage/ → outage_uncertainty package (package lives directly under outage/)
if str(_OUTAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTAGE_ROOT))

# outage/P6_adapter/ → outage_model package
_P6_ADAPTER_ROOT = _OUTAGE_ROOT / "P6_adapter"
if str(_P6_ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_P6_ADAPTER_ROOT))

# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

MOCK_CSV_DIR = _P6_ADAPTER_ROOT / "examples" / "mock_p6_export"
SAMPLE_XER_PATH = _P6_ADAPTER_ROOT / "examples" / "sample_xer" / "sample_project.xer"


@pytest.fixture(scope="session")
def mock_csv_dir() -> Path:
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
