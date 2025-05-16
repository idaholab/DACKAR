from dackar.text_processing.SpellChecker import SpellChecker

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

  def test_miss_spelled_words_contextual_checker(self):
    checker = self.get_spell_checker('ContextualSpellCheck')
    miss = checker.getMisspelledWords(self.content)
    assert miss == {'presure', 'laek', 'revieled', '1A'}
    # can not handle '1A'
    checker.addWordsToDictionary(['1A', 'laek'])
    miss = checker.getMisspelledWords(self.content)
    print(miss)
    assert miss == {'1A', 'presure', 'revieled'}
