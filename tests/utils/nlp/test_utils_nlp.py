from dackar.utils.nlp import nlp_utils as utils

import spacy
import pytest

@pytest.fixture
def nlp_obj():
  nlp = spacy.load("en_core_web_lg")
  return nlp

def test_display_ner(nlp_obj):
  content = "The oil is found nearby the pump motor."
  doc = nlp_obj(content)
  df = utils.displayNER(doc)
  assert list(df['pos'].values) == ['DET', 'NOUN', 'AUX', 'VERB', 'ADV', 'DET', 'NOUN', 'NOUN']
  assert list(df['dep'].values) == ['det', 'nsubjpass', 'auxpass', 'ROOT', 'prep', 'det', 'compound', 'pobj']
  assert list(df['lemma'].values) == ['the', 'oil', 'be', 'find', 'nearby', 'the', 'pump', 'motor']


