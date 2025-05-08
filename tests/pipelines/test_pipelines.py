from dackar.pipelines.ConjectureEntity import ConjectureEntity
from dackar.pipelines.EmergentActivityEntity import EmergentActivity
from dackar.pipelines.GeneralEntity import GeneralEntity
from dackar.pipelines.LocationEntity import LocationEntity
from dackar.pipelines.PhraseEntityMatcher import PhraseEntityMatcher
from dackar.pipelines.SimpleEntityMatcher import SimpleEntityMatcher
from dackar.pipelines.TemporalAttributeEntity import TemporalAttributeEntity
from dackar.pipelines.TemporalEntity import Temporal
from dackar.pipelines.TemporalRelationEntity import TemporalRelationEntity
from dackar.pipelines.UnitEntity import UnitEntity
from dackar.utils.nlp.nlp_utils import resetPipeline

import spacy
import pytest

@pytest.fixture
def nlp_obj():
  nlp = spacy.load("en_core_web_lg")
  return nlp


class TestPipelines:

  def get_entity(self, doc, label=None):
    if label is None:
      ents =[ent.text for ent in doc.ents]
    else:
      ents = [ent.text for ent in doc.ents if ent.label_ == label]
    return ents

  def test_conjecture_entity(self, nlp_obj):
    patterns = {'label': 'conjecture', 'pattern': [{'LOWER': 'seems'}], 'id': 'conjecture'}
    matcher = ConjectureEntity(nlp_obj, patterns)
    doc = nlp_obj("Vibration seems like it is coming from the shaft.")
    updated_doc = matcher(doc)
    ents = self.get_entity(updated_doc, label='conjecture')
    assert ents == ['seems']

  def test_conjecture_entity_pipeline(self, nlp_obj):
    patterns = {'label': 'conjecture', 'pattern': [{'LOWER': 'seems'}], 'id': 'conjecture'}
    nlp_obj.add_pipe('conjecture_entity', config={"patterns":patterns})
    doc = nlp_obj("Vibration seems like it is coming from the shaft.")
    ents = self.get_entity(doc, label='conjecture')
    assert ents == ['seems']

  def test_conjecture_entity_no_pattern_pipeline(self, nlp_obj):
    nlp_obj.add_pipe('conjecture_entity')
    doc = nlp_obj("Vibration is unlikely coming from the shaft.")
    ents = self.get_entity(doc, label='conjecture')
    assert ents == ['unlikely']

  def test_emergent_activity_entity(self, nlp_obj):
    matcher = EmergentActivity(nlp_obj)
    content = """ wo101 wo 102  wo# 103 , wo#104 or wo #105 wo # 106 or a 107 wrong wo .
            ABCD01D hrs- 8hr PERFORM 8-hr REPAIRS IF 10-hrs REQUIRED 24hrs (contingency 24 hrs).
            1EFGH/J08K ERECT AB-7603 FOR IJKL-7148 XYZA7148abc OPGH0248 M-N 100 for WO# 84658 1BC/E08D-34r.
            RELEASE [CLEARANCE] #3693 RED/Replace # the "A" ** (Switch).
            A218-82-9171 -  REMOVE {INSUL}  [ISO].
            """
    doc = nlp_obj(content)
    updated_doc = matcher(doc)
    wo_ents = self.get_entity(updated_doc, label='WO')
    id_ents = self.get_entity(updated_doc, label='ID')
    assert wo_ents == ['wo101', 'wo 102', 'wo# 103', 'wo #105', 'wo # 106']
    assert id_ents == ['wo#104', 'ABCD01D', '8hr', '24hrs', '1EFGH', 'J08', 'AB-7603', 'IJKL-7148', 'XYZA7148abc', 'OPGH0248', 'E08D-34r', 'A218']

  def test_emergent_activity_entity_pipeline(self, nlp_obj):
    nlp_obj.add_pipe('EmergentActivity')
    content = """ wo101 wo 102  wo# 103 , wo#104 or wo #105 wo # 106 or a 107 wrong wo .
        ABCD01D hrs- 8hr PERFORM 8-hr REPAIRS IF 10-hrs REQUIRED 24hrs (contingency 24 hrs).
        1EFGH/J08K ERECT AB-7603 FOR IJKL-7148 XYZA7148abc OPGH0248 M-N 100 for WO# 84658 1BC/E08D-34r.
        RELEASE [CLEARANCE] #3693 RED/Replace # the "A" ** (Switch).
        A218-82-9171 -  REMOVE {INSUL}  [ISO].
        """
    doc = nlp_obj(content)
    wo_ents = self.get_entity(doc, label='WO')
    id_ents = self.get_entity(doc, label='ID')
    assert wo_ents == ['wo101', 'wo 102', 'wo# 103', 'wo #105', 'wo # 106']
    assert id_ents == ['wo#104', 'ABCD01D', '8hr', '24hrs', '1EFGH', 'J08', 'AB-7603', 'IJKL-7148', 'XYZA7148abc', 'OPGH0248', 'E08D-34r', 'A218']


  # def test_location_entity(self, nlp_obj):
  #   patterns = {'label': 'location', 'pattern': [{'LOWER': 'follow'}], 'id': 'location'}
  #   matcher = LocationEntity(nlp_obj, patterns)
  #   doc = nlp_obj("Vibration seems like it is coming from the shaft.")
  #   updated_doc = matcher(doc)
  #   ents = self.get_entity(updated_doc, label='conjecture')
  #   assert ents == ['seems']

  # def test_location_entity_pipeline(self, nlp_obj):
  #   patterns = {'label': 'location', 'pattern': [{'LOWER': 'follow'}], 'id': 'location'}
  #   nlp_obj.add_pipe('conjecture_entity', config={"patterns":patterns})
  #   doc = nlp_obj("Vibration seems like it is coming from the shaft.")
  #   ents = self.get_entity(doc, label='conjecture')
  #   assert ents == ['seems']
