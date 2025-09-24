from dackar.causal.CausalSentence import CausalSentence
from dackar.causal.OperatorShiftLogsProcessing import OperatorShiftLogs
from dackar.causal.CausalPhrase import CausalPhrase
from dackar.config import nlpConfig
from dackar.utils.nlp.nlp_utils import generatePatternList
import spacy
import pandas as pd


class TestCausal:

  nlp = spacy.load("en_core_web_lg", exclude=[])
  entId = 'test'
  entLabel = 'test_label'
  causalLabel = "causal"
  causalID = "causal"
  entPatternName = 'test_ent'
  causalPatternName = 'dackar_causal'
  ents = ['pump', 'shaft', 'impeller', 'diffuser', 'seal', 'motor', 'rotor', 'pump bearings']
  doc = 'Rupture of pump bearings caused shaft degradation.'

  def generatePattern(self):
    """Generate patterns using provided OPM and/or entity file
    """
    # convert opm formList into matcher patternsOPM
    patterns = generatePatternList(self.ents, label=self.entLabel, id=self.entId, nlp=self.nlp, attr="LEMMA")
    return patterns

  def processCausalEnt(self):
    """
    Parse causal keywords, and generate patterns for them
    The patterns can be used to identify the causal relationships

    Returns:
        list: list of patterns will be used by causal entity matcher
    """
    patternsCausal = []
    causalFilename = nlpConfig['files']['cause_effect_keywords_file']
    ds = pd.read_csv(causalFilename, skipinitialspace=True)
    for col in ds.columns:
      cvars = set(ds[col].dropna())
      patternsCausal.extend(generatePatternList(cvars, label=self.causalLabel, id=self.causalID, nlp=self.nlp, attr="LEMMA"))
    return patternsCausal

  def get_matcher(self, method):
    if method == 'general':
      matcher = CausalSentence(self.nlp, entID=self.entId, causalKeywordID=self.causalID)
    elif method == 'wo':
      matcher = CausalPhrase(self.nlp, entID=self.entId, causalKeywordID=self.causalID)
    elif method == 'osl':
      matcher = OperatorShiftLogs(self.nlp, entID=self.entId, causalKeywordID=self.causalID)
    else:
      raise IOError(f'Unrecognized causal type {method}')
    patterns = self.generatePattern()
    causalPatterns = self.processCausalEnt()
    matcher.addEntityPattern(self.entPatternName, patterns)
    matcher.addEntityPattern(self.causalPatternName, causalPatterns)
    return matcher

  def test_general(self):
    matcher = self.get_matcher(method='general')
    matcher(self.doc)
    df = matcher.getAttribute('entStatus')
    dfCausal = matcher.getAttribute('causalRelation')
    assert df['entity'].tolist() == ['pump bearings', 'shaft']
    assert df['label'].tolist() == ['test_label', 'test_label']
    assert df['status'].tolist()[1].text == 'degradation'
    assert dfCausal['cause'].tolist()[0].text == 'pump bearings'
    assert dfCausal['causal keyword'].tolist()[0].text == 'caused'
    assert dfCausal['effect'].tolist()[0].text == 'shaft'


  def test_wo(self):
    matcher = self.get_matcher(method='wo')
    matcher(self.doc)
    df = matcher.getAttribute('entStatus')
    dfCausal = matcher.getAttribute('relationGeneral')
    print(dfCausal)
    assert df['entity'].tolist() == ['pump bearings']
    assert df['label'].tolist() == ['test_label']
    assert df['status'].tolist()[0] == 'Rupture of pump bearings caused shaft degradation'
    assert dfCausal['subject'].tolist()[0] == 'pump bearings'
    assert dfCausal['relation'].tolist()[0] == 'caused'
    assert dfCausal['object'].tolist()[0] == 'shaft'

  def test_osl(self):
    matcher = self.get_matcher(method='osl')
    matcher(self.doc)
    df = matcher.getAttribute('entStatus')
    dfCausal = matcher.getAttribute('relationGeneral')
    print(dfCausal)
    assert df['entity'].tolist() == ['pump bearings', 'shaft']
    assert df['label'].tolist() == ['test_label', 'test_label']
    assert df['status'].tolist()[1].text == 'degradation'
    assert dfCausal['subject'].tolist()[0] == 'pump bearings'
    assert dfCausal['relation'].tolist()[0] == 'caused'
    assert dfCausal['object'].tolist()[0] == 'shaft'
