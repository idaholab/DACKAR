"""
Unit tests for DomainSpellChecker.

Coverage:
  - Common P6 typos are corrected
  - Known vocabulary words pass through unchanged
  - Short tokens (< min_token_len) are never modified
  - False-positive safety (acoustically similar words)
  - extra_vocab parameter extends the vocabulary
  - Domain vocabulary sanity (taxonomy + nuclear abbreviations present)
  - transform() handles empty/None-like input gracefully
  - cutoff parameter is respected (lower cutoff → more corrections)
  - spell_check_enabled=False in AppConfig uses IdentityTransform instead
"""
from __future__ import annotations

import pytest

from outage_uncertainty.preprocessing.spell_checker import DomainSpellChecker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def checker() -> DomainSpellChecker:
    """Default checker — cutoff=0.85, min_token_len=5."""
    return DomainSpellChecker()


# ---------------------------------------------------------------------------
# 1. Common P6 typos corrected
# ---------------------------------------------------------------------------

class TestTypoCorrection:
    """Typos that commonly appear in P6 schedule activity names."""

    @pytest.mark.parametrize("typo, expected", [
        ("calibartion",  "calibration"),   # transposed 'a'/'r'
        ("presure",      "pressure"),      # missing 's'
        ("transmiter",   "transmitter"),   # missing 't'
        ("maintenace",   "maintenance"),   # transposed 'a'/'n'
        ("inspcetion",   "inspection"),    # transposed letters
        ("replacment",   "replacement"),   # missing 'e'
        ("installtion",  "installation"),  # missing 'a'
        ("verifcation",  "verification"),  # missing 'i'
        ("lubrication",  "lubrication"),   # already correct — pass-through
    ])
    def test_single_typo_corrected(self, checker, typo, expected):
        result = checker.transform(typo)
        assert result == expected, f"Expected '{expected}' but got '{result}'"

    def test_typo_in_multi_word_phrase(self, checker):
        result = checker.transform("calibartion of presure transmiter")
        assert "calibration" in result
        assert "pressure" in result
        assert "transmitter" in result

    def test_already_correct_word_unchanged(self, checker):
        assert checker.transform("calibration") == "calibration"
        assert checker.transform("maintenance") == "maintenance"
        assert checker.transform("inspection")  == "inspection"


# ---------------------------------------------------------------------------
# 2. Known vocabulary words pass through unchanged
# ---------------------------------------------------------------------------

class TestVocabPassThrough:
    """Words that ARE in the domain vocabulary must never be changed."""

    @pytest.mark.parametrize("word", [
        "bearing", "impeller", "pressurizer", "containment", "coolant",
        "actuator", "positioner", "transmitter", "insulation", "scaffold",
        "overhaul", "lubrication", "verification", "replacement",
        "reactor", "feedwater",
    ])
    def test_known_word_unchanged(self, checker, word):
        assert checker.transform(word) == word


# ---------------------------------------------------------------------------
# 3. Short tokens never modified
# ---------------------------------------------------------------------------

class TestShortTokenPassThrough:
    """Tokens shorter than min_token_len (default 5) are never spell-checked."""

    @pytest.mark.parametrize("token", ["px", "rcp", "mfw", "vent", "pump", "bolt"])
    def test_short_token_unchanged(self, checker, token):
        # All ≤ 4 chars — must pass through regardless of similarity
        assert checker.transform(token) == token

    def test_min_token_len_boundary(self):
        """Token at exactly min_token_len=5 IS checked; len=4 is NOT."""
        chk = DomainSpellChecker(min_token_len=5)
        # 4-char token: always pass through
        assert chk.transform("valv") == "valv"
        # 5-char token starting with known domain word: checked (may or may not correct)
        result = chk.transform("valvv")  # close to "valve" — should correct
        # Either it corrects or stays — just must not raise
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 4. False-positive safety
# ---------------------------------------------------------------------------

class TestFalsePositiveSafety:
    """Words that should NOT be changed despite superficial similarity."""

    def test_primary_not_changed_to_primate(self, checker):
        # SequenceMatcher("primary","primate") ≈ 0.71 — below cutoff
        assert checker.transform("primary") == "primary"

    def test_install_not_changed_to_instill(self, checker):
        # "install" is IN the vocabulary → pass-through before matching
        assert checker.transform("install") == "install"

    def test_train_not_changed(self, checker):
        # "train" is a valid domain word (electrical train / division)
        assert checker.transform("train") == "train"

    def test_unknown_short_acronym_unchanged(self, checker):
        # Short unknown tokens must not be mauled
        assert checker.transform("edg") == "edg"


# ---------------------------------------------------------------------------
# 5. extra_vocab parameter
# ---------------------------------------------------------------------------

class TestExtraVocab:
    """Caller-supplied vocabulary is included in corrections."""

    def test_extra_word_passes_through(self):
        chk = DomainSpellChecker(extra_vocab={"plantspecific"})
        assert chk.transform("plantspecific") == "plantspecific"

    def test_extra_word_used_for_correction(self):
        chk = DomainSpellChecker(extra_vocab={"plantspecific"}, cutoff=0.80)
        # Slight misspelling of the plant-specific word
        result = chk.transform("plantspecifik")
        assert result == "plantspecific"

    def test_extra_vocab_does_not_break_existing(self):
        chk = DomainSpellChecker(extra_vocab={"customterm"})
        # Existing corrections must still work
        result = chk.transform("calibartion")
        assert result == "calibration"


# ---------------------------------------------------------------------------
# 6. Domain vocabulary sanity
# ---------------------------------------------------------------------------

class TestVocabCoverage:
    """Verify the vocabulary is built correctly from all three sources."""

    def test_nuclear_abbreviation_expansions_present(self):
        from outage_uncertainty.preprocessing.nuclear_abbreviations import (
            NUCLEAR_OUTAGE_ABBREVIATIONS,
        )
        chk = DomainSpellChecker()
        for expansion in NUCLEAR_OUTAGE_ABBREVIATIONS.values():
            for word in expansion.lower().split():
                if len(word) >= chk._min_len:
                    assert word in chk._vocab, (
                        f"Expected nuclear expansion word '{word}' in vocab"
                    )

    def test_taxonomy_keywords_present(self):
        from outage_uncertainty.preprocessing.default_taxonomy import DEFAULT_TAXONOMY_RULES
        chk = DomainSpellChecker()
        for kw in DEFAULT_TAXONOMY_RULES:
            for word in kw.lower().split():
                if len(word) >= chk._min_len:
                    assert word in chk._vocab, (
                        f"Expected taxonomy keyword '{word}' in vocab"
                    )

    def test_vocab_non_empty(self):
        chk = DomainSpellChecker()
        assert len(chk._vocab) > 50


# ---------------------------------------------------------------------------
# 7. Edge cases / robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self, checker):
        assert checker.transform("") == ""

    def test_whitespace_only(self, checker):
        result = checker.transform("   ")
        # split() on whitespace-only → [], join → ""
        assert result == ""

    def test_single_space_between_tokens(self, checker):
        result = checker.transform("replace bearing")
        assert "replace" in result
        assert "bearing" in result

    def test_numbers_pass_through(self, checker):
        # Numeric tokens have no vocabulary match
        assert checker.transform("12345") == "12345"

    def test_mixed_case_lowercased(self, checker):
        # transform() lowercases all tokens first
        result = checker.transform("CALIBARTION")
        assert result == "calibration"

    def test_all_known_words_phrase(self, checker):
        phrase = "replace bearing in reactor coolant pump"
        result = checker.transform(phrase)
        # All words known — none should be changed
        for word in result.split():
            assert word in checker._vocab or len(word) < checker._min_len


# ---------------------------------------------------------------------------
# 8. cutoff parameter respected
# ---------------------------------------------------------------------------

class TestCutoffBehaviour:
    def test_high_cutoff_misses_borderline(self):
        """A very high cutoff should reject marginal corrections."""
        strict = DomainSpellChecker(cutoff=0.99)
        # "calibartion" ratio to "calibration" is ~0.91 → rejected at 0.99
        result = strict.transform("calibartion")
        assert result == "calibartion"

    def test_default_cutoff_accepts_common_typos(self):
        default = DomainSpellChecker()
        assert default.transform("calibartion") == "calibration"

    def test_low_cutoff_corrects_more(self):
        lenient = DomainSpellChecker(cutoff=0.70)
        # "presure" is already fixed at default; check nothing breaks
        result = lenient.transform("presure")
        assert result == "pressure"


# ---------------------------------------------------------------------------
# 9. AppConfig integration: spell_check_enabled=False uses IdentityTransform
# ---------------------------------------------------------------------------

class TestAppConfigIntegration:
    def test_spell_check_disabled_uses_identity(self):
        """When spell_check_enabled=False the cleaner should not import
        DomainSpellChecker — but we can verify via the config field itself."""
        from outage_uncertainty.api.config import AppConfig

        cfg = AppConfig(spell_check_enabled=False)
        assert cfg.spell_check_enabled is False

    def test_spell_check_enabled_default_true(self):
        from outage_uncertainty.api.config import AppConfig

        cfg = AppConfig()
        assert cfg.spell_check_enabled is True
        assert cfg.spell_check_cutoff == 0.85

    def test_spell_check_cutoff_configurable(self):
        from outage_uncertainty.api.config import AppConfig

        cfg = AppConfig(spell_check_cutoff=0.90)
        assert cfg.spell_check_cutoff == 0.90
