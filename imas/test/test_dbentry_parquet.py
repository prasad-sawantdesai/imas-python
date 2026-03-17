# This file is part of IMAS-Python.
# You should have received the IMAS-Python LICENSE file with this project.
"""Tests for the Parquet backend (imas:parquet?path=...)."""

import numpy as np
import pytest

import imas

# Skip the whole module if pyarrow is not installed
pyarrow = pytest.importorskip("pyarrow")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_core_profiles(factory):
    """Build a minimal core_profiles IDS for testing."""
    cp = factory.core_profiles()
    cp.ids_properties.homogeneous_time = 1
    cp.ids_properties.comment = "parquet backend test"
    cp.time = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    cp.profiles_1d.resize(3)
    for i in range(3):
        cp.profiles_1d[i].time = float(i + 1)
        cp.profiles_1d[i].electrons.density = np.array(
            [float(j + 1) * (i + 1) for j in range(5)], dtype=np.float64
        )

    return cp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pq_uri(tmp_path):
    """Return a fresh Parquet data entry URI inside a temp directory."""
    return f"imas:parquet?path={tmp_path}/pq_db"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_put_get_roundtrip(pq_uri):
    """Write and read back a core_profiles IDS; check key fields."""
    factory = imas.IDSFactory()

    cp_out = _make_core_profiles(factory)
    with imas.DBEntry(pq_uri, "w") as entry:
        entry.put(cp_out)

    with imas.DBEntry(pq_uri, "r") as entry:
        cp_in = entry.get("core_profiles")

    assert cp_in.ids_properties.comment.value == "parquet backend test"
    np.testing.assert_array_equal(cp_in.time.value, cp_out.time.value)
    assert len(cp_in.profiles_1d) == 3
    np.testing.assert_array_almost_equal(
        cp_in.profiles_1d[1].electrons.density.value,
        cp_out.profiles_1d[1].electrons.density.value,
    )


def test_put_multiple_occurrences(pq_uri):
    """Writing two occurrences and reading them back independently."""
    factory = imas.IDSFactory()

    cp0 = factory.core_profiles()
    cp0.ids_properties.homogeneous_time = 1
    cp0.ids_properties.comment = "occurrence 0"
    cp0.time = np.array([0.0])

    cp1 = factory.core_profiles()
    cp1.ids_properties.homogeneous_time = 1
    cp1.ids_properties.comment = "occurrence 1"
    cp1.time = np.array([1.0])

    with imas.DBEntry(pq_uri, "w") as entry:
        entry.put(cp0, occurrence=0)
        entry.put(cp1, occurrence=1)

    with imas.DBEntry(pq_uri, "r") as entry:
        assert entry.list_all_occurrences("core_profiles") == [0, 1]
        r0 = entry.get("core_profiles", occurrence=0)
        r1 = entry.get("core_profiles", occurrence=1)

    assert r0.ids_properties.comment.value == "occurrence 0"
    assert r1.ids_properties.comment.value == "occurrence 1"


def test_list_filled_paths(pq_uri):
    """list_filled_paths returns canonical slash-separated paths."""
    factory = imas.IDSFactory()
    cp = factory.core_profiles()
    cp.ids_properties.homogeneous_time = 1
    cp.ids_properties.comment = "filled paths test"
    cp.time = np.array([1.0, 2.0])

    with imas.DBEntry(pq_uri, "w") as entry:
        entry.put(cp)

    with imas.DBEntry(pq_uri, "r") as entry:
        paths = entry.list_filled_paths("core_profiles")

    assert "ids_properties/comment" in paths
    assert "ids_properties/homogeneous_time" in paths
    assert "time" in paths


def test_delete_data(pq_uri):
    """Deleting an occurrence removes it from disk."""
    import os

    factory = imas.IDSFactory()
    cp = factory.core_profiles()
    cp.ids_properties.homogeneous_time = 0

    with imas.DBEntry(pq_uri, "w") as entry:
        entry.put(cp, occurrence=0)
        entry.put(cp, occurrence=1)
        entry.delete_data("core_profiles", occurrence=0)
        assert entry.list_all_occurrences("core_profiles") == [1]


def test_mode_x_fails_if_data_exists(pq_uri, tmp_path):
    """Mode 'x' should raise when the directory already contains Parquet files."""
    from imas.exception import DataEntryException

    factory = imas.IDSFactory()
    cp = factory.core_profiles()
    cp.ids_properties.homogeneous_time = 0

    with imas.DBEntry(pq_uri, "w") as entry:
        entry.put(cp)

    with pytest.raises(DataEntryException):
        imas.DBEntry(pq_uri, "x")


def test_get_nonexistent_raises(pq_uri):
    """Getting a non-existent IDS should raise DataEntryException."""
    from imas.exception import DataEntryException

    with imas.DBEntry(pq_uri, "w") as entry:
        with pytest.raises(DataEntryException):
            entry.get("core_profiles")


def test_complex_roundtrip(pq_uri):
    """Complex-valued quantities survive a write/read cycle."""
    factory = imas.IDSFactory()

    ids_name = "waves"  # 'waves' IDS has complex quantities
    try:
        w = factory.new(ids_name)
    except Exception:
        pytest.skip(f"'{ids_name}' IDS not in this DD version")

    w.ids_properties.homogeneous_time = 0

    # Find the first complex field and fill it
    from imas.ids_data_type import IDSDataType

    filled_field = None
    for child_name, child_meta in w.metadata._children.items():
        if child_meta.data_type is IDSDataType.CPX and child_meta.ndim == 1:
            w[child_name].value = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
            filled_field = child_name
            break

    if filled_field is None:
        pytest.skip(f"No CPX_1D top-level field found in '{ids_name}'")

    with imas.DBEntry(pq_uri, "w") as entry:
        entry.put(w)

    with imas.DBEntry(pq_uri, "r") as entry:
        w2 = entry.get(ids_name)

    np.testing.assert_array_almost_equal(
        w2[filled_field].value, w[filled_field].value
    )


def test_context_manager_write_read(pq_uri):
    """Use DBEntry as a context manager for both write and read."""
    factory = imas.IDSFactory()
    cp = _make_core_profiles(factory)

    with imas.DBEntry(pq_uri, "w") as entry:
        entry.put(cp)

    # Entry should be closed after context exit
    with imas.DBEntry(pq_uri, "r") as entry:
        cp2 = entry.get("core_profiles")
        assert cp2.ids_properties.comment.value == cp.ids_properties.comment.value
