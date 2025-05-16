from dackar.utils.utils import getOnlyWords, getShortAcronym

def test_get_only_words():
  content = """T8 A-b hello pr12345 hxm4tx 4mfy world"""
  cleaned = getOnlyWords(content)
  assert cleaned == "  hello    world"

def test_get_short_acronym():
  content = """T8 A-b hello pr12345 hxm4tx 4mfy world s/d h/s i/o am"""
  cleaned, acronym = getShortAcronym(content)
  assert cleaned == "T8 A-b hello pr12345 hxm4tx 4mfy world    am"
  assert acronym == ['s/d', 'h/s', 'i/o']
