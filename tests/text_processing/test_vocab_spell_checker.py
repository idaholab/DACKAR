import pytest

from dackar.text_processing.VocabSpellChecker import VocabSpellChecker


class TestVocabSpellChecker:

  vocab = {'calibration', 'pressure', 'transmitter', 'maintenance', 'inspection'}

  def get_checker(self, **kw):
    return VocabSpellChecker(self.vocab, **kw)

  def test_corrects_close_typos(self):
    checker = self.get_checker()
    assert checker.transform('calibartion') == 'calibration'
    assert checker.transform('presure') == 'pressure'
    assert checker.transform('transmiter') == 'transmitter'

  def test_known_words_pass_through(self):
    checker = self.get_checker()
    assert checker.transform('pressure calibration') == 'pressure calibration'

  def test_short_tokens_pass_through(self):
    # Shorter than min_token_len (default 5): never corrected.
    checker = self.get_checker()
    assert checker.transform('abc') == 'abc'

  def test_out_of_vocab_without_close_match_passes_through(self):
    # No vocabulary word clears the cutoff, so the token is left untouched
    # rather than mangled into an unrelated word.
    checker = self.get_checker()
    assert checker.transform('zzzqwx') == 'zzzqwx'

  def test_empty_string(self):
    assert self.get_checker().transform('') == ''

  def test_multi_token_sentence(self):
    checker = self.get_checker()
    assert checker.transform('presure transmiter calibartion') == \
      'pressure transmitter calibration'

  def test_cutoff_gates_borderline_corrections(self):
    # A low cutoff accepts a correction a high cutoff rejects.
    lax = self.get_checker(cutoff=0.70)
    strict = self.get_checker(cutoff=0.95)
    # 'calibrtn' vs 'calibration' ratio ~0.84: accepted by lax, rejected by strict.
    assert lax.transform('calibrtn') == 'calibration'
    assert strict.transform('calibrtn') == 'calibrtn'


class TestDomainSpellCheckerAdapter:
  """The outage adapter is a thin VocabSpellChecker over the nuclear vocab."""

  def test_is_vocab_spell_checker(self):
    from dackar.outage.outage_uncertainty.preprocessing.spell_checker import (
      DomainSpellChecker,
    )
    assert isinstance(DomainSpellChecker(), VocabSpellChecker)

  def test_corrects_domain_typos(self):
    from dackar.outage.outage_uncertainty.preprocessing.spell_checker import (
      DomainSpellChecker,
    )
    checker = DomainSpellChecker()
    assert checker.transform('maintenace') == 'maintenance'
    assert checker.transform('inspcetion') == 'inspection'

  def test_extra_vocab_is_correctable(self):
    from dackar.outage.outage_uncertainty.preprocessing.spell_checker import (
      DomainSpellChecker,
    )
    checker = DomainSpellChecker(extra_vocab={'gizmotron'})
    assert checker.transform('gizmotrn') == 'gizmotron'
