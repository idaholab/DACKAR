"""Smoke tests for scripts/bootstrap_models.py.

These don't invoke the actual model downloads — they verify CLI shape and
the idempotency-check helpers, so the test stays under one second and
doesn't need network.
"""
import importlib.util
import pathlib

import pytest


def _load_script():
    """Load the script as a module (it's not a real package)."""
    path = pathlib.Path(__file__).parents[2] / "scripts" / "bootstrap_models.py"
    spec = importlib.util.spec_from_file_location("bootstrap_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exposes_expected_steps():
    """STEPS dict drives --only flag and ordering; both must be present."""
    mod = _load_script()
    assert set(mod.STEPS) == {"coreferee", "nltk", "quantulum"}


def test_only_flag_accepts_known_steps():
    """Argparse rejects unknown --only values."""
    mod = _load_script()
    parser = mod.build_parser()
    parsed = parser.parse_args(["--only", "nltk"])
    assert parsed.only == "nltk"
    with pytest.raises(SystemExit):
        parser.parse_args(["--only", "spacy"])


def test_insecure_ssl_flag_is_off_by_default():
    mod = _load_script()
    parser = mod.build_parser()
    parsed = parser.parse_args([])
    assert parsed.insecure_ssl is False


def test_coreferee_step_skips_when_module_missing(monkeypatch):
    """If `nlp-extra` group not installed, the coreferee step is a no-op."""
    mod = _load_script()
    monkeypatch.setattr(mod, "_has_module", lambda name: False)
    # Should return cleanly without raising, even though coreferee isn't importable.
    mod.install_coreferee()
