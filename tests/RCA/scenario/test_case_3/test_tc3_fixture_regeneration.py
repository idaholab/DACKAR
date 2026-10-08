"""
test_tc3_fixture_regeneration.py — guard: ``data_generator.py`` reproduces the
committed TC-3 fixtures byte-for-byte.

wangcj05 review (PR #64, finding 3): the committed fixtures must be exactly what
``data_generator.py`` emits, so that a later edit to the generator — or a hand-edit
to a fixture — cannot silently diverge from the data the scenario advertises. This
re-runs the generator into a throwaway directory and asserts byte-for-byte equality
with the committed fixtures.

``evidence_bundle.json`` is intentionally excluded: it is a curated retrieval-bundle
artifact that the generator does not emit (it is consumed by the fixture-mode harness
in ``tests/RCA/scenario/shared/run_helpers.py``). Guarding it would mean encoding a
hand-authored 6.7 KB bundle as a generator constant; that is tracked as a follow-up
rather than fabricated here. ``test_excluded_artifact_present_but_not_generated``
pins the exclusion so it stays deliberate.

The generator imports only the standard library, so this guard runs in every CI job
(including the pip-install job that omits the optional ``rca``/``kg`` dependencies).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TC3_DIR = Path(__file__).resolve().parent
_FIXTURES = _TC3_DIR / "fixtures"

# The ten files data_generator.dump_fixture_files() writes.
_GENERATED_FIXTURES = (
    "event.json",
    "telemetry_summary.json",
    "kg_context.json",
    "operational_context.json",
    "pm_compliance.json",
    "evidence_store_rows.json",
    "alarm_log.json",
    "configuration_change_records.json",
    "environmental_monitoring.json",
    "processed_records.jsonl",
)

# Curated artifact the generator does not emit — see the module docstring.
_NOT_GENERATED = ("evidence_bundle.json",)


def _load_generator():
    """Import ``data_generator.py`` by path under a unique module name.

    A plain ``import data_generator`` would collide with the sibling
    ``test_case_8/data_generator.py`` under pytest's default (prepend) import mode,
    so the module is loaded explicitly from this directory.
    """
    spec = importlib.util.spec_from_file_location(
        "tc3_data_generator", _TC3_DIR / "data_generator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> Path:
    """Run the generator once into a throwaway directory for the whole module."""
    out = tmp_path_factory.mktemp("tc3_regen")
    _load_generator().dump_fixture_files(out)
    return out


@pytest.mark.parametrize("name", _GENERATED_FIXTURES)
def test_committed_fixture_is_reproducible(name: str, regenerated: Path) -> None:
    committed = (_FIXTURES / name).read_bytes()
    fresh = (regenerated / name).read_bytes()
    assert committed == fresh, (
        f"{name} has drifted: the committed fixture is not byte-identical to a fresh "
        f"`python data_generator.py` run (committed={len(committed)}B, fresh={len(fresh)}B). "
        f"Re-run the generator and commit the result, or revert the generator change."
    )


def test_generator_emits_exactly_the_expected_set(regenerated: Path) -> None:
    """A new or removed generator output must be reflected in ``_GENERATED_FIXTURES``
    so that nothing escapes the byte-equality guard above."""
    emitted = {p.name for p in regenerated.iterdir()}
    assert emitted == set(_GENERATED_FIXTURES), (
        f"generator output set changed: emitted={sorted(emitted)}, "
        f"guarded={sorted(_GENERATED_FIXTURES)}"
    )


def test_excluded_artifact_present_but_not_generated(regenerated: Path) -> None:
    """``evidence_bundle.json`` is committed (consumed by run_helpers) but not emitted
    by the generator; pin that so the exclusion stays intentional, not an oversight."""
    for name in _NOT_GENERATED:
        assert (_FIXTURES / name).exists(), f"{name} missing from committed fixtures"
        assert not (regenerated / name).exists(), (
            f"{name} is now emitted by the generator — add it to _GENERATED_FIXTURES "
            "and drop it from _NOT_GENERATED."
        )
