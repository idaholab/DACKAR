from dackar.text_processing.AbbrExpander import AbbrExpander
from dackar.config import nlpConfig
import os


class TestAbbrExpander:

  abbreviation_file = nlpConfig['files']['abbreviation_file']
  abbreviation = AbbrExpander(abbreviation_file)
  content = """Perf ann sens calib of cyl.
        High concent of hydrogen obs.
        High concent of hydrogen obs every wk.
        Prfr chann calib of chan.
        esf pump room and fuel bldg test.
        cal press xmtr sit elev.
        perform thermography survey of pzr htr terminations.
        plant mods comp iso mode prep.
        drain & rmv pipe."""

  def test_abbreviation_process(self):
    updated = self.abbreviation.abbrProcess(self.content, splitToList='True')
    assert updated.strip() == """perform annual sensor calibration of cylinder. high concentration of hydrogen observe. high concentration of hydrogen observe every work. perform channel calibration of channel. esf pump room and fuel building test. calibration pressure transmitter sit elevation. perform thermography survey of pressurizer heater terminations. plant modifications composition iso mode preparation. drain and remove pipe."""
