"""
Tests for the built-in nuclear outage abbreviation supplement.

Covers:
  - Correction of wrong last-wins resolutions in the general Excel file
  - Expansion of nuclear-specific component and system abbreviations
  - Expansion of task/action abbreviations
  - Behaviour when dackar.text_processing is unavailable (dict-only path)
  - Priority: extra_abbreviations overrides nuclear supplement
  - Multi-word expansion preserves surrounding tokens
  - Case insensitivity
"""
from __future__ import annotations

import pytest

from outage_uncertainty.preprocessing.abbreviations import AbbreviationResolver
from outage_uncertainty.preprocessing.nuclear_abbreviations import (
    NUCLEAR_OUTAGE_ABBREVIATIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolver(**kwargs) -> AbbreviationResolver:
    """Build a resolver with no Excel file (dict-only path, always available)."""
    return AbbreviationResolver(abbreviations_file=None, **kwargs)


def _expand(text: str, **kwargs) -> str:
    return _resolver(**kwargs).transform(text)


# ===========================================================================
# 1. Corrections — wrong Excel last-wins resolutions overridden
# ===========================================================================

class TestCorrections:
    """Nuclear supplement must override the wrong duplicate resolutions
    produced by pandas dict(zip(...)) on the general abbreviations.xlsx."""

    def test_cont_expands_to_containment_not_continuous(self):
        assert "containment" in _expand("CONT INSPECTION")

    def test_rep_expands_to_replace_not_report(self):
        assert "replace" in _expand("SEAL REP")

    def test_comp_expands_to_component_not_composite(self):
        assert "component" in _expand("COMP REPLACEMENT")

    def test_mtg_expands_to_mounting_not_meeting(self):
        assert "mounting" in _expand("MTG BOLT TORQUE")

    def test_gen_expands_to_generator_not_general(self):
        assert "generator" in _expand("EDG GEN INSPECTION")


# ===========================================================================
# 2. Nuclear system / component abbreviations
# ===========================================================================

class TestNuclearSystemAbbreviations:

    def test_mcp_expands(self):
        result = _expand("MCP SEAL REPLACEMENT")
        assert "main coolant pump" in result

    def test_rcp_expands(self):
        assert "reactor coolant pump" in _expand("RCP OVERHAUL")

    def test_rhr_expands(self):
        assert "residual heat removal" in _expand("RHR PUMP INSPECTION")

    def test_msiv_expands(self):
        assert "main steam isolation valve" in _expand("MSIV FULL STROKE TEST")

    def test_mov_expands(self):
        assert "motor operated valve" in _expand("MOV ACTUATOR CALIBRATION")

    def test_aov_expands(self):
        assert "air operated valve" in _expand("AOV POSITIONER ADJ")

    def test_edg_expands(self):
        assert "emergency diesel generator" in _expand("EDG LOAD TEST")

    def test_dg_expands(self):
        assert "diesel generator" in _expand("DG FUEL FILTER REPL")

    def test_swp_expands(self):
        assert "service water pump" in _expand("SWP 2B OVHL")

    def test_ccw_expands(self):
        assert "component cooling water" in _expand("CCW PUMP BEARING INSP")

    def test_prv_expands(self):
        assert "pressure relief valve" in _expand("PRV SETPOINT TEST")

    def test_psv_expands(self):
        assert "pressure safety valve" in _expand("PSV BENCH TEST")

    def test_hpsi_expands(self):
        assert "high pressure safety injection" in _expand("HPSI PUMP RUN TEST")

    def test_lpsi_expands(self):
        assert "low pressure safety injection" in _expand("LPSI VALVE INSP")

    def test_eccs_expands(self):
        assert "emergency core cooling system" in _expand("ECCS TRAIN A FUNCTIONAL")


# ===========================================================================
# 3. Task / action abbreviations
# ===========================================================================

class TestTaskAbbreviations:

    def test_repl_expands_to_replace(self):
        assert "replace" in _expand("SEAL REPL")

    def test_tst_expands_to_test(self):
        assert "test" in _expand("VALVE TST")

    def test_adj_expands_to_adjust(self):
        assert "adjust" in _expand("GOVERNOR ADJ")

    def test_ovhl_expands_to_overhaul(self):
        assert "overhaul" in _expand("PUMP OVHL")

    def test_clb_expands_to_calibration(self):
        assert "calibration" in _expand("INSTRUMENT CLB")

    def test_lubr_expands(self):
        assert "lubrication" in _expand("BEARING LUBR")

    def test_verif_expands_to_verification(self):
        assert "verification" in _expand("TORQUE VERIF")

    def test_verf_expands_to_verification(self):
        assert "verification" in _expand("SETPOINT VERF")

    def test_rmvl_expands_to_removal(self):
        assert "removal" in _expand("SCAFFOLD RMVL")

    def test_disassy_expands(self):
        assert "disassembly" in _expand("PUMP DISASSY AND INSP")

    def test_wldg_expands(self):
        assert "welding" in _expand("PIPE WLDG REPAIR")

    def test_vnt_expands_to_vent(self):
        assert "vent" in _expand("SYSTEM VNT AND PURGE")


# ===========================================================================
# 4. Radiation protection and NDE abbreviations
# ===========================================================================

class TestRadiationAbbreviations:

    def test_rp_expands_to_radiation_protection(self):
        assert "radiation protection" in _expand("RP HOLD REQUIRED")

    def test_alara_expands(self):
        assert "as low as reasonably achievable" in _expand("ALARA REVIEW")

    def test_nde_expands(self):
        assert "non-destructive examination" in _expand("WELD NDE")

    def test_ndt_expands(self):
        assert "non-destructive testing" in _expand("PIPE NDT")


# ===========================================================================
# 5. Priority: extra_abbreviations overrides nuclear supplement
# ===========================================================================

class TestPriority:

    def test_extra_overrides_nuclear_supplement(self):
        """Caller-supplied abbreviations must take highest priority."""
        result = _expand("CONT INSP", extra_abbreviations={"cont": "continuous"})
        assert "continuous" in result
        assert "containment" not in result

    def test_extra_adds_new_entry_on_top(self):
        result = _expand("CUSTOM ABBR", extra_abbreviations={"custom": "custom expansion"})
        assert "custom expansion" in result

    def test_nuclear_supplement_still_applies_for_other_keys(self):
        """Overriding one key must not suppress other nuclear supplement entries."""
        result = _expand("MCP CONT", extra_abbreviations={"cont": "continuous"})
        assert "main coolant pump" in result   # nuclear supplement still active
        assert "continuous" in result           # override still in effect


# ===========================================================================
# 6. Case insensitivity
# ===========================================================================

class TestCaseInsensitivity:

    def test_uppercase_input(self):
        assert "main coolant pump" in _expand("MCP SEAL REPL")

    def test_lowercase_input(self):
        assert "main coolant pump" in _expand("mcp seal repl")

    def test_mixed_case_input(self):
        assert "main coolant pump" in _expand("Mcp Seal Repl")


# ===========================================================================
# 7. Multi-word expansion preserves surrounding tokens
# ===========================================================================

class TestMultiWordExpansion:

    def test_multiword_expansion_inserts_correctly(self):
        """'MCP' expands to 3 words; surrounding tokens must be preserved."""
        result = _expand("REPLACE MCP SEAL").lower()
        assert "main coolant pump" in result
        assert "seal" in result

    def test_full_sentence_expansion(self):
        result = _expand("MCP OVHL AND SEAL REPL WITH RP HOLD")
        assert "main coolant pump" in result
        assert "overhaul" in result
        assert "replace" in result
        assert "radiation protection" in result

    def test_unknown_tokens_pass_through_unchanged(self):
        result = _expand("ZZZUNKNOWN ABBR")
        assert "ZZZUNKNOWN" in result or "zzzunknown" in result.lower()


# ===========================================================================
# 8. NUCLEAR_OUTAGE_ABBREVIATIONS constant sanity checks
# ===========================================================================

class TestConstantSanity:

    def test_all_keys_lowercase(self):
        for key in NUCLEAR_OUTAGE_ABBREVIATIONS:
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    def test_no_empty_values(self):
        for key, val in NUCLEAR_OUTAGE_ABBREVIATIONS.items():
            assert val.strip(), f"Key '{key}' has empty expansion"

    def test_no_empty_keys(self):
        for key in NUCLEAR_OUTAGE_ABBREVIATIONS:
            assert key.strip(), "Empty key found in NUCLEAR_OUTAGE_ABBREVIATIONS"

    def test_minimum_coverage(self):
        """At least the 5 corrections and key nuclear systems must be present."""
        required = {"cont", "rep", "comp", "mtg", "gen",
                    "mcp", "rhr", "msiv", "mov", "repl", "rp"}
        missing = required - set(NUCLEAR_OUTAGE_ABBREVIATIONS)
        assert not missing, f"Missing required entries: {missing}"
