"""Tests for the Streamlit app's path-containment helpers.

Findings 2 and 3: the app saved uploads under client-controlled names
(``./<upload.name>`` and ``construction_schemas/<cs_name>.json``), so a name
such as ``../target.toml`` escaped the working directory and an absolute
``cs_name`` made ``os.path.join`` discard the base directory. Import could then
overwrite an arbitrary writable path and Remove could delete it.

The remedy writes every upload under a generated basename inside a per-session
temporary directory and confines deletion to that directory via a containment
check. These tests pin the containment predicate and the guarded delete; they
do not drive the Streamlit runtime (which needs a browser session), only the
pure helpers that enforce the boundary.

Skipped if streamlit is not installed (an optional UI dependency).
"""
import os

import pytest

pytest.importorskip("streamlit")

from dackar.knowledge_graph.app import _is_contained, _safe_remove


def test_contained_accepts_child_paths(tmp_path):
    base = str(tmp_path)
    assert _is_contained(base, os.path.join(base, "schema_abc.toml"))
    assert _is_contained(base, os.path.join(base, "sub", "deep.json"))
    # The base directory itself counts as contained.
    assert _is_contained(base, base)


def test_contained_rejects_traversal_and_absolute(tmp_path):
    base = str(tmp_path / "session")
    os.makedirs(base)
    # "../target.toml" climbs out of the session directory.
    assert not _is_contained(base, os.path.join(base, "..", "target.toml"))
    # An absolute path discards the base entirely.
    assert not _is_contained(base, "/etc/passwd")
    # A sibling that merely shares a name prefix is not inside the base.
    assert not _is_contained(base, str(tmp_path / "session_evil" / "x.toml"))


def test_safe_remove_deletes_only_inside_base(tmp_path):
    base = tmp_path / "session"
    base.mkdir()
    inside = base / "schema.toml"
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path / "keepme.toml"
    outside.write_text("y", encoding="utf-8")

    # Inside the base: deleted.
    _safe_remove(str(base), str(inside))
    assert not inside.exists()

    # Outside the base (via traversal): left untouched, no error.
    traversal = os.path.join(str(base), "..", "keepme.toml")
    _safe_remove(str(base), traversal)
    assert outside.exists()
