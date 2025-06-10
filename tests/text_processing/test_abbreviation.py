from dackar.text_processing.Abbreviation import Abbreviation
class TestAbbreviation:

  abbreviation = Abbreviation()
  content = """Perf ann sens calib of cyl.
        High conc of hydrogen obs.
        High conc of hydrogen obs every wk.
        Prfr chann calib of chan.
        esf pump room and fuel bldg test.
        cal press xmtr sit elev.
        perform thermography survey of pzr htr terminations.
        plant mods comp iso mode prep.
        drain & rmv pipe."""

  def test_abbreviation(self):
    updated = self.abbreviation.abbreviationSub(self.content)
    assert updated.strip() == """perform annual sensor calibration of cylinder. high concentration of hydrogen observe. high concentration of hydrogen observe every work. prfr channel calibration of channel. esf pump room and fuel building test. calibration pressure transmitter sit elevation. perform thermography survey of pressurizer heater terminations. plant modifications composite iso mode prepare. drain & remove pipe."""

  def test_user_defined_abbreviation(self):
    abbrDict = {'perf':'perform', 'ann':'annual', 'sens':'sensor', 'calib':'calibration'}
    self.abbreviation.updateAbbreviation(abbrDict, reset=True)
    updated = self.abbreviation.abbreviationSub(self.content)
    assert updated.strip() == """perform annual sensor calibration of cyl. high conc of hydrogen obs. high conc of hydrogen obs every wk. prfr chann calibration of chan. esf pump room and fuel bldg test. cal press xmtr sit elev. perform thermography survey of pzr htr terminations. plant mods comp iso mode prep. drain & rmv pipe."""
