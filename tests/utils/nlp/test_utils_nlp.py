from dackar.utils.nlp import nlp_utils as utils
from dackar.pipelines.TemporalEntity import Temporal
from dackar.pipelines.UnitEntity import UnitEntity

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

def test_reset_pipeline(nlp_obj):
  pipes = ['Temporal', 'unit_entity']
  nlp = utils.resetPipeline(nlp_obj, pipes)
  updated = [pipe for (pipe,_) in nlp.pipeline]
  assert updated == ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner', 'Temporal', 'unit_entity']
  nlp = utils.resetPipeline(nlp_obj, pipes)
  updated = [pipe for (pipe,_) in nlp.pipeline]
  assert updated == ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner', 'Temporal', 'unit_entity']

def test_generate_pattern_list(nlp_obj):
  entList = ['Pump', 'motor', 'CCW', 'safety cage', 'condensers']
  label = 'SSC'
  id = '13496'
  patterns = utils.generatePatternList(entList, label, id, nlp_obj)
  assert patterns == [{'label': 'SSC', 'pattern': [{'LOWER': 'pump'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LOWER': 'motor'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LOWER': 'ccw'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LOWER': 'safety'}, {'LOWER': 'cage'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LOWER': 'condensers'}], 'id': '13496'}]
  patterns = utils.generatePatternList(entList, label, id, nlp_obj, attr='lemma')
  print(patterns)
  assert patterns == [{'label': 'SSC', 'pattern': [{'LOWER': 'pump'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LEMMA': 'pump'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LOWER': 'motor'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LEMMA': 'motor'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LOWER': 'ccw'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LEMMA': 'CCW'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LOWER': 'safety'}, {'LOWER': 'cage'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LEMMA': 'safety'}, {'LEMMA': 'cage'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LOWER': 'condensers'}], 'id': '13496'},
                      {'label': 'SSC', 'pattern': [{'LEMMA': 'condenser'}], 'id': '13496'}]
