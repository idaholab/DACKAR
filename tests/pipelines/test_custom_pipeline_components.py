from dackar.pipelines.CustomPipelineComponents import aliasResolver
from dackar.pipelines.CustomPipelineComponents import normEntities
from dackar.pipelines.CustomPipelineComponents import anaphorCoref
from dackar.pipelines.CustomPipelineComponents import anaphorEntCoref
from dackar.pipelines.CustomPipelineComponents import expandEntities
from dackar.pipelines.CustomPipelineComponents import mergeEntitiesWithSameID
from dackar.pipelines.CustomPipelineComponents import mergePhrase
from dackar.pipelines.CustomPipelineComponents import pysbdSentenceBoundaries

from dackar.utils.nlp.nlp_utils import resetPipeline
from dackar.utils.nlp.nlp_utils import generatePatternList
from dackar.pipelines.GeneralEntity import GeneralEntity

import spacy
import pytest

@pytest.fixture
def nlp_obj():
  nlp = spacy.load("en_core_web_lg")
  pipeline = [pipe for (pipe,_) in nlp.pipeline]
  if "enity_ruler" in pipeline:
      nlp.remove_pipe("entity_ruler")
  if "ner" in pipeline:
      nlp.remove_pipe("ner")
  return nlp


class TestCustomPipelines:

  def get_custom_entity(self, doc, label):
    ents = [(ent.text, getattr(ent._, label)) for ent in doc.ents if getattr(ent._, label) is not None]
    return ents

  def test_alias_resolver_pipeline(self, nlp_obj):
    """
      Note: NER will identify '1-91120-P1' and '1-91120-PM1 REQUIRES OIL' as 'CARDINAL' entities
      For the testing, we will remove 'NER' for nlp pipelines
    """
    entIDList = ['1-91120-P1', '1-91120-PM1', '91120']
    patterns = generatePatternList(entIDList, label='pump', id='pump', nlp=nlp_obj, attr="LEMMA")
    nlp_obj.add_pipe('general_entity', config={"patterns":patterns})
    content = """1-91120-P1, CLEAN PUMP AND MOTOR. 1-91120-PM1 REQUIRES OIL. 91120, CLEAN TRASH SCREEN"""
    nlp_obj.add_pipe('aliasResolver')
    doc = nlp_obj(content)
    ents = self.get_custom_entity(doc, label='alias')
    assert ents == [('1-91120-P1', 'unit 1 pump'), ('1-91120-PM1', 'unit 1 pump motor'), ('91120', 'pump')]
