"""ISCC Image-Codes (provenance extra): what they tell apart and what they ignore."""

import numpy as np
import pytest
import tifffile

from omero_annotate_ai import provenance

needs_iscc = pytest.mark.skipif(
    not provenance.available(), reason="provenance extra (iscc-lib) not installed"
)


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = (rng.random((128, 150)) * 4000).astype(np.uint16)
    image[30:80, 40:120] += 20000
    return image


def _mask() -> np.ndarray:
    mask = np.zeros((128, 150), np.uint16)
    mask[20:60, 20:60] = 1
    mask[80:110, 70:140] = 2
    return mask


@pytest.mark.unit
class TestNormalisation:
    def test_planes_become_32_by_32_uint8(self):
        for pixels in (_image(), np.stack([_image(), _image(1)]), np.zeros((5, 7, 3))):
            plane = provenance.normalised(pixels)
            assert plane.shape == (32, 32) and plane.dtype == np.uint8

    def test_masks_are_foreground_versus_background(self):
        plane = provenance.normalised(_mask(), mask=True)
        assert (plane.min(), plane.max()) == (0, 255)
        assert provenance.normalised(np.zeros((8, 8)), mask=True).max() == 0


@needs_iscc
@pytest.mark.unit
class TestCodes:
    def test_same_content_same_code(self):
        image = _image()
        assert provenance.image_code(image) == provenance.image_code(image * 2 + 7)

    def test_renumbered_objects_keep_the_mask_code(self):
        renumbered = _mask()
        renumbered[renumbered == 2] = 9
        assert provenance.image_code(_mask(), mask=True) == provenance.image_code(
            renumbered, mask=True
        )

    def test_a_small_edit_is_a_near_duplicate_and_new_content_is_not(self):
        code = provenance.image_code(_mask(), mask=True)
        assert code is not None
        touched = _mask()
        touched[20:24, 20:60] = 0
        other = np.zeros_like(_mask())
        other[5:40, 90:145] = 1

        assert "near-duplicate" in provenance.describe_change(code, touched, mask=True)
        assert "different content" in provenance.describe_change(code, other, mask=True)

    def test_a_volume_is_coded_by_its_projection(self):
        assert provenance.image_code(np.stack([_image(), _image()])) == (
            provenance.image_code(_image())
        )

    def test_file_code_matches_the_pixels(self, tmp_path):
        path = tmp_path / "m.tif"
        tifffile.imwrite(path, _mask())
        assert provenance.file_code(path, mask=True) == provenance.image_code(
            _mask(), mask=True
        )

    def test_uniform_empty_or_damaged_inputs_defer_to_the_digest(self):
        code = provenance.image_code(_mask(), mask=True)
        blank = provenance.image_code(np.zeros((64, 64), np.uint16))
        assert blank is not None
        assert provenance.describe_change(code, np.zeros((64, 64)), mask=True) == ""
        assert provenance.describe_change(blank, _image(), mask=False) == ""
        assert provenance.describe_change("not-an-iscc", _mask(), mask=True) == ""
        assert provenance.image_code(np.zeros((0, 5))) is None

    def test_large_planes_are_reduced_before_conversion(self):
        big = np.zeros((4096, 4096), np.uint16)
        big[:2048] = 60000
        small = big[::128, ::128]
        assert provenance.image_code(big) == provenance.image_code(small)

    def test_distance_is_symmetric_and_zero_for_equal_codes(self):
        a, b = provenance.image_code(_image(0)), provenance.image_code(_image(1))
        assert a is not None and b is not None
        assert provenance.distance(a, a) == 0
        assert provenance.distance(a, b) == provenance.distance(b, a)


@pytest.mark.unit
class TestWithoutTheExtra:
    @pytest.fixture(autouse=True)
    def _no_iscc(self, monkeypatch):
        monkeypatch.setattr(provenance, "_iscc_lib", lambda: None)

    def test_nothing_is_computed(self, tmp_path):
        assert not provenance.available()
        assert provenance.image_code(_image()) is None
        assert provenance.file_code(tmp_path / "never-read.tif") is None
        assert (
            provenance.describe_change("ISCC:EEA4XSZUGQ2MXS6B", _image(), mask=False)
            == ""
        )

    def test_no_recorded_code_means_no_comparison(self):
        assert provenance.describe_change(None, _image(), mask=False) == ""
