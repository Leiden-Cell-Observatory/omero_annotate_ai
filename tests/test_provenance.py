"""Tests for ISCC content provenance (processing/provenance.py)."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omero_annotate_ai.processing import provenance


@pytest.fixture
def fake_iscc(monkeypatch):
    """Inject a stub iscc_bio module and hand back its biocode mock.

    provenance.py imports iscc_bio lazily inside functions, so patching
    sys.modules is enough - no real install needed.
    """
    biocode = MagicMock(
        return_value=[{"iscc_code": "ISCC:AAA", "units": ["ISCC:D", "ISCC:I"]}]
    )
    api_mod = types.ModuleType("iscc_bio.api")
    api_mod.biocode = biocode
    root_mod = types.ModuleType("iscc_bio")
    root_mod.api = api_mod
    monkeypatch.setitem(sys.modules, "iscc_bio", root_mod)
    monkeypatch.setitem(sys.modules, "iscc_bio.api", api_mod)
    return biocode


@pytest.fixture
def no_iscc(monkeypatch):
    """Make `import iscc_bio` raise ImportError, simulating it not installed."""
    monkeypatch.setitem(sys.modules, "iscc_bio", None)


@pytest.mark.unit
class TestIsccAvailability:
    """Guarded-import behaviour."""

    def test_available_when_installed(self, fake_iscc):
        assert provenance.iscc_available() is True

    def test_unavailable_when_missing(self, no_iscc):
        assert provenance.iscc_available() is False


@pytest.mark.unit
class TestComputeFileIscc:
    """compute_file_iscc()."""

    def test_returns_code_for_file(self, fake_iscc, tmp_path):
        img = tmp_path / "image.tif"
        img.write_bytes(b"fake")

        assert provenance.compute_file_iscc(img) == "ISCC:AAA"
        fake_iscc.assert_called_once_with(source=str(img))

    def test_returns_none_when_iscc_missing(self, no_iscc, tmp_path):
        img = tmp_path / "image.tif"
        img.write_bytes(b"fake")

        assert provenance.compute_file_iscc(img) is None

    def test_returns_none_on_compute_error(self, fake_iscc, tmp_path):
        fake_iscc.side_effect = RuntimeError("unreadable")
        img = tmp_path / "image.tif"
        img.write_bytes(b"fake")

        assert provenance.compute_file_iscc(img) is None

    def test_returns_none_on_empty_result(self, fake_iscc, tmp_path):
        fake_iscc.return_value = []
        img = tmp_path / "image.tif"
        img.write_bytes(b"fake")

        assert provenance.compute_file_iscc(img) is None


@pytest.mark.unit
class TestComputeImageIscc:
    """compute_image_iscc() - the canonical source-side code, from raw OMERO pixels."""

    def test_returns_code_for_omero_image(self, fake_iscc):
        conn = MagicMock()

        assert provenance.compute_image_iscc(conn, 123) == "ISCC:AAA"
        fake_iscc.assert_called_once_with(conn=conn, iid=123)

    def test_returns_none_when_iscc_missing(self, no_iscc):
        assert provenance.compute_image_iscc(MagicMock(), 123) is None

    def test_returns_none_on_omero_error(self, fake_iscc):
        fake_iscc.side_effect = RuntimeError("connection lost")

        assert provenance.compute_image_iscc(MagicMock(), 123) is None
