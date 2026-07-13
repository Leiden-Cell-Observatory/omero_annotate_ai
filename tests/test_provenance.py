"""Tests for ISCC content provenance (processing/provenance.py)."""

import sys
import types
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


@pytest.mark.unit
class TestComputeLabelIscc:
    """compute_label_iscc() - content code for annotation masks."""

    def test_returns_code_for_label(self, fake_iscc, monkeypatch, tmp_path):
        """Happy path: fetches mask file and computes ISCC."""
        # Mock ezomero.get_file_annotation to return a path
        mask_file = tmp_path / "mask.tif"
        mask_file.write_bytes(b"fake")

        get_file_annotation = MagicMock(return_value=str(mask_file))

        ezomero_mod = types.ModuleType("ezomero")
        ezomero_mod.get_file_annotation = get_file_annotation
        monkeypatch.setitem(sys.modules, "ezomero", ezomero_mod)

        conn = MagicMock()
        result = provenance.compute_label_iscc(conn, 456)

        assert result == "ISCC:AAA"
        get_file_annotation.assert_called_once()
        # Verify it was called with correct arguments
        call_args = get_file_annotation.call_args
        assert call_args[0][0] is conn
        assert call_args[0][1] == 456

    def test_returns_none_when_iscc_missing(self, no_iscc, monkeypatch):
        """With no_iscc fixture, returns None and does NOT download."""
        get_file_annotation = MagicMock()

        ezomero_mod = types.ModuleType("ezomero")
        ezomero_mod.get_file_annotation = get_file_annotation
        monkeypatch.setitem(sys.modules, "ezomero", ezomero_mod)

        conn = MagicMock()
        result = provenance.compute_label_iscc(conn, 456)

        assert result is None
        # Verify ezomero was NOT called (guarded-import short-circuit)
        get_file_annotation.assert_not_called()

    def test_returns_none_when_mask_path_is_none(self, fake_iscc, monkeypatch):
        """When ezomero returns None, returns None."""
        get_file_annotation = MagicMock(return_value=None)

        ezomero_mod = types.ModuleType("ezomero")
        ezomero_mod.get_file_annotation = get_file_annotation
        monkeypatch.setitem(sys.modules, "ezomero", ezomero_mod)

        conn = MagicMock()
        result = provenance.compute_label_iscc(conn, 456)

        assert result is None
        get_file_annotation.assert_called_once()

    def test_returns_none_on_ezomero_error(self, fake_iscc, monkeypatch):
        """When ezomero raises, returns None and does not propagate."""
        get_file_annotation = MagicMock(side_effect=RuntimeError("download failed"))

        ezomero_mod = types.ModuleType("ezomero")
        ezomero_mod.get_file_annotation = get_file_annotation
        monkeypatch.setitem(sys.modules, "ezomero", ezomero_mod)

        conn = MagicMock()
        result = provenance.compute_label_iscc(conn, 456)

        assert result is None
        get_file_annotation.assert_called_once()


def _config_with(annotations):
    """Build a default config carrying the given annotations."""
    from omero_annotate_ai.core.annotation_config import create_default_config

    config = create_default_config()
    for ann in annotations:
        config.add_annotation(ann)
    return config


@pytest.mark.unit
class TestStampConfig:
    """stamp_config() fills codes, and codes each unique id exactly once."""

    def test_fills_source_and_label_codes(self, fake_iscc, monkeypatch):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        monkeypatch.setattr(provenance, "compute_label_iscc", lambda conn, lid: "ISCC:LBL")
        config = _config_with(
            [ImageAnnotation(image_id=7, image_name="a.tif", label_id=99)]
        )

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc == "ISCC:AAA"
        assert config.annotations[0].label_iscc == "ISCC:LBL"

    def test_codes_each_source_image_only_once(self, fake_iscc):
        """Three patches of one image must trigger exactly one OMERO compute."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = _config_with(
            [
                ImageAnnotation(image_id=7, image_name="a.tif", is_patch=True, patch_x=0),
                ImageAnnotation(image_id=7, image_name="a.tif", is_patch=True, patch_x=1),
                ImageAnnotation(image_id=7, image_name="a.tif", is_patch=True, patch_x=2),
            ]
        )

        provenance.stamp_config(config, MagicMock())

        assert fake_iscc.call_count == 1
        assert all(a.source_iscc == "ISCC:AAA" for a in config.annotations)

    def test_skips_annotations_without_a_label(self, fake_iscc):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = _config_with([ImageAnnotation(image_id=7, image_name="a.tif")])

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc == "ISCC:AAA"
        assert config.annotations[0].label_iscc is None

    def test_does_not_recompute_existing_codes(self, fake_iscc):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = _config_with(
            [ImageAnnotation(image_id=7, image_name="a.tif", source_iscc="ISCC:OLD")]
        )

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc == "ISCC:OLD"
        assert fake_iscc.call_count == 0

    def test_no_op_and_warns_when_iscc_missing(self, no_iscc, caplog):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = _config_with([ImageAnnotation(image_id=7, image_name="a.tif")])

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc is None
        assert "iscc-bio is not installed" in caplog.text

    def test_one_failing_image_does_not_abort_the_pass(self, fake_iscc):
        """A broken image must not cost us the codes of the healthy ones."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        def flaky(conn=None, iid=None, **kwargs):
            if iid == 7:
                raise RuntimeError("corrupt pixels")
            return [{"iscc_code": "ISCC:OK"}]

        fake_iscc.side_effect = flaky
        config = _config_with(
            [
                ImageAnnotation(image_id=7, image_name="bad.tif"),
                ImageAnnotation(image_id=8, image_name="good.tif"),
            ]
        )

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc is None
        assert config.annotations[1].source_iscc == "ISCC:OK"
