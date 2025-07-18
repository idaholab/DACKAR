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
# from dackar.utils.nlp.nlp_utils import resetPipeline

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

  def test_location_entity(self, nlp_obj):
    patterns = {'label': 'location', 'pattern': [{'LOWER': 'nearby'}], 'id': 'location'}
    matcher = LocationEntity(nlp_obj, patterns)
    doc = nlp_obj("The oil is found nearby the pump motor.")
    updated_doc = matcher(doc)
    ents = self.get_entity(updated_doc, label='location')
    assert ents == ['nearby']

  def test_location_entity_pipeline(self, nlp_obj):
    nlp_obj.add_pipe('location_entity')
    doc = nlp_obj("The oil is found nearby the pump motor. The condenser is behind the pump. The debris is found on top of the rotor")
    ent_proximity = self.get_entity(doc, label='location_proximity')
    ent_up = self.get_entity(doc, label='location_up')
    ent_down = self.get_entity(doc, label='location_down')
    assert ent_proximity == ['nearby']
    assert ent_up == ['on top of']
    assert ent_down == ['behind']

  def test_temporal_attribute_entity(self, nlp_obj):
    patterns = {'label': 'temporal_attribute', 'pattern': [{'LOWER': 'approximately'}], 'id': 'temporal_attribute'}
    matcher = TemporalAttributeEntity(nlp_obj, patterns)
    doc = nlp_obj("It is approximately 5pm.")
    updated_doc = matcher(doc)
    ents = self.get_entity(updated_doc, label='temporal_attribute')
    entTIME = self.get_entity(updated_doc, label='TIME')
    assert ents == []
    assert entTIME == ['approximately 5pm']

  def test_temporal_attribute_entity_pipeline(self, nlp_obj):
    nlp_obj.add_pipe('temporal_attribute_entity')
    doc = nlp_obj("The valve is about a twenty-nine years old. The event occurred almost twice a week")
    ents = self.get_entity(doc, label='temporal_attribute')
    entDATE = self.get_entity(doc, label='DATE')
    assert ents == ['about']
    assert entDATE == ['twenty-nine years old', 'almost twice a week']

  def test_temporal_entity(self, nlp_obj):
    matcher = Temporal(nlp_obj)
    content = """The event is scheduled for 25th August 2023.
                We also have a meeting on 10 September and another one on the twelfth of October and a final one on January fourth.
                yesterday afternoon, before 2022, after 12/2024."""
    doc = nlp_obj(content)
    updated_doc = matcher(doc)
    ents = self.get_entity(updated_doc, label='Temporal')
    print(ents)
    assert ents == ['25th August 2023', 'on 10 September', 'October', 'on January fourth', 'yesterday afternoon', 'before 2022', 'after 12/2024']

  def test_temporal_entity_pipeline(self, nlp_obj):
    nlp_obj.add_pipe('Temporal')
    content = """The event is scheduled for 25th August 2023.
            We also have a meeting on 10 September and another one on the twelfth of October and a final one on January fourth.
            yesterday afternoon, before 2022, after 12/2024."""
    doc = nlp_obj(content)
    ents = self.get_entity(doc, label='Temporal')
    assert ents == ['25th August 2023', 'on 10 September', 'October', 'on January fourth', 'yesterday afternoon', 'before 2022', 'after 12/2024']

  def test_temporal_relation_entity(self, nlp_obj):
    matcher = TemporalRelationEntity(nlp_obj)
    content = """The system failed following the pump failure.
              The pump stopped earlier than the shutoff of the system.
              The pump failed along the broke of pipe."""
    doc = nlp_obj(content)
    updated_doc = matcher(doc)
    ent_reverse = self.get_entity(updated_doc, label='temporal_relation_reverse_order')
    ent = self.get_entity(updated_doc, label='temporal_relation_order')
    ent_con = self.get_entity(updated_doc, label='temporal_relation_concurrency')
    assert ent == ['earlier']
    assert ent_reverse == ['following']
    assert ent_con == ['along']

  def test_temporal_relation_entity_pipeline(self, nlp_obj):
    nlp_obj.add_pipe('temporal_relation_entity')
    content = """The system failed following the pump failure.
              The pump stopped earlier than the shutoff of the system.
              The pump failed along the broke of pipe."""
    doc = nlp_obj(content)
    ent_reverse = self.get_entity(doc, label='temporal_relation_reverse_order')
    ent = self.get_entity(doc, label='temporal_relation_order')
    ent_con = self.get_entity(doc, label='temporal_relation_concurrency')
    assert ent == ['earlier']
    assert ent_reverse == ['following']
    assert ent_con == ['along']

  def test_unit_entity(self, nlp_obj):
    matcher = UnitEntity(nlp_obj)
    content = """I want a gallon of beer.
              The LHC smashes proton beams at 12.8-13.0 TeV,
              The LHC smashes proton beams at 12.9±0.1 TeV.
              Sound travels at 0.34 km/s,
              I want 2 liters of wine.
              I spent 20 pounds on this!
              The average density of the Earth is about 5.5x10-3 kg/cm³,
              Gimme 10e9 GW now!"""
    doc = nlp_obj(content)
    updated_doc = matcher(doc)
    ents = self.get_entity(updated_doc, label='unit')
    assert ents == ['a gallon', '12.8-13.0 TeV', '12.9±0.1 TeV.', '0.34 km/s', '2 liters', '20 pounds', '5.5x10-3 kg/cm³', '10e9 GW']

  def test_unit_entity_pipeline(self, nlp_obj):
    nlp_obj.add_pipe('unit_entity')
    content = """I want a gallon of beer.
              The LHC smashes proton beams at 12.8-13.0 TeV,
              The LHC smashes proton beams at 12.9±0.1 TeV.
              Sound travels at 0.34 km/s,
              I want 2 liters of wine.
              I spent 20 pounds on this!
              The average density of the Earth is about 5.5x10-3 kg/cm³,
              Gimme 10e9 GW now!"""
    doc = nlp_obj(content)
    ents = self.get_entity(doc, label='unit')
    assert ents == ['a gallon', '12.8-13.0 TeV', '12.9±0.1 TeV.', '0.34 km/s', '2 liters', '20 pounds', '5.5x10-3 kg/cm³', '10e9 GW']

  def test_simple_entity(self, nlp_obj):
    patterns = [[{"LOWER": "safety"}, {"LOWER": "cage"}],[{"LOWER": "shaft"}],[{"LOWER": "pump"}]]
    content = """The shaft deflection is causing the safety cage to rattle.
              Pump not experiencing enough flow during test. Shaft made noise.
              Vibration seems like it is coming from the shaft."""
    matcher = SimpleEntityMatcher(nlp_obj,label='simple', patterns=patterns)
    doc = nlp_obj(content)
    updated_doc = matcher(doc)
    ents = self.get_entity(updated_doc, label='simple')
    assert ents == ['shaft', 'safety cage', 'Pump', 'Shaft', 'shaft']

  def test_simple_entity_pipeline(self, nlp_obj):
    patterns = [[{"LOWER": "safety"}, {"LOWER": "cage"}],[{"LOWER": "shaft"}],[{"LOWER": "pump"}]]
    nlp_obj.add_pipe('simple_entity_matcher', config={"label":"simple","patterns":patterns})
    content = """The shaft deflection is causing the safety cage to rattle.
            Pump not experiencing enough flow during test. Shaft made noise.
            Vibration seems like it is coming from the shaft."""
    doc = nlp_obj(content)
    ents = self.get_entity(doc, label='simple')
    assert ents == ['shaft', 'safety cage', 'Pump', 'Shaft', 'shaft']

  def test_phrase_entity(self, nlp_obj):
    patterns = ["safety cage", "cage", "pump", "shaft"]
    matcher = PhraseEntityMatcher(nlp_obj, label='phrase', patterns=patterns)
    content = """The shaft deflection is causing the safety cage to rattle.
              Pump not experiencing enough flow during test."""
    doc = nlp_obj(content)
    updated_doc = matcher(doc)
    ents = self.get_entity(updated_doc, label='phrase')
    assert ents == ['shaft', 'safety cage', 'Pump']

  def test_phrase_entity_pipeline(self, nlp_obj):
    patterns = ["safety cage", "cage", "pump", "shaft"]
    nlp_obj.add_pipe('phrase_entity_matcher', config={"label":"phrase", "patterns":patterns})
    content = """The shaft deflection is causing the safety cage to rattle.
              Pump not experiencing enough flow during test."""
    doc = nlp_obj(content)
    ents = self.get_entity(doc, label='phrase')
    assert ents == ['shaft', 'safety cage', 'Pump']

  def test_general_entity(self, nlp_obj):
    patterns = [{'label': 'general', 'pattern': [{'LOWER': 'shaft'}], 'id': 'general'}, {'label': 'other', 'pattern': [{'LOWER': 'pump'}], 'id': 'other'}]
    matcher = GeneralEntity(nlp_obj, patterns=patterns)
    content = """The shaft deflection is causing the safety cage to rattle.
              Pump not experiencing enough flow during test."""
    doc = nlp_obj(content)
    updated_doc = matcher(doc)
    ents_general = self.get_entity(updated_doc, label='general')
    ents_other = self.get_entity(updated_doc, label='other')
    assert ents_general == ['shaft']
    assert ents_other == ['Pump']

  def test_general_entity_pipeline(self, nlp_obj):
    patterns = {"patterns":[{'label': 'general', 'pattern': [{'LOWER': 'shaft'}], 'id': 'general'}, {'label': 'other', 'pattern': [{'LOWER': 'pump'}], 'id': 'other'}]}
    nlp_obj.add_pipe('general_entity', config=patterns)
    content = """The shaft deflection is causing the safety cage to rattle.
              Pump not experiencing enough flow during test."""
    doc = nlp_obj(content)
    ents_general = self.get_entity(doc, label='general')
    ents_other = self.get_entity(doc, label='other')
    assert ents_general == ['shaft']
    assert ents_other == ['Pump']
