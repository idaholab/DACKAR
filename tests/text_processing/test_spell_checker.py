from dackar.text_processing.SpellChecker import SpellChecker

from packaging.version import Version
import spacy
import pytest

# Misspelling detection via ContextualSpellCheck relies on spacy's en_core_web_lg tokenization/NER,
# which shifted with the newer model shipped for spacy>=3.6, changing which tokens get flagged.
_newer_ner_model = pytest.mark.skipif(Version(spacy.__version__) >= Version('3.6'), reason='misspelled-word set differs under the newer en_core_web_lg model shipped for spacy>=3.6')


class TestSpellChecker:

  content = """A laek was noticed.
        RCP pump 1A presure gauge was found not operating.
        Pump inspection revieled excessive impeller degradation.
        RCP pump 1A was cavitating.
      """

  def get_spell_checker(self, name):
    spell_checker = SpellChecker(name)
    return spell_checker

  def test_miss_spelled_words_autocorrect(self):
    checker = self.get_spell_checker('autocorrect')
    miss = checker.getMisspelledWords(self.content)
    assert miss == {'presure', 'laek', 'revieled', '1A'}
    checker.addWordsToDictionary(['1A'])
    miss = checker.getMisspelledWords(self.content)
    assert miss == {'presure', 'laek', 'revieled'}

  def test_miss_spelled_words_pyspellchecker(self):
    checker = self.get_spell_checker('pyspellchecker')
    miss = checker.getMisspelledWords(self.content)
    assert miss == {'revieled', 'laek', 'rcp', 'presure', '1a'}
    checker.addWordsToDictionary(['1A', 'RCP'])
    miss = checker.getMisspelledWords(self.content)
    assert miss == {'presure', 'laek', 'revieled'}

  @_newer_ner_model
  def test_miss_spelled_words_contextual_checker(self):
    checker = self.get_spell_checker('ContextualSpellCheck')
    miss = checker.getMisspelledWords(self.content)
    assert miss == {'presure', 'laek', 'revieled', '1A'}
    # can not handle '1A'
    checker.addWordsToDictionary(['1A', 'laek'])
    miss = checker.getMisspelledWords(self.content)
    print(miss)
    assert miss == {'1A', 'presure', 'revieled'}
