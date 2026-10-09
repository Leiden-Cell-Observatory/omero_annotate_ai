"""Tests for the ImageSource seam (omero_annotate_ai.sources)."""

from unittest.mock import Mock

import numpy as np
import pytest

from omero_annotate_ai.sources import (
    ImageMeta,
    ImageSource,
    OmeroImageSource,
    clamp_region,
)


def make_omero_image(planes, image_id=7, name="img.tif", pixels_type="uint16"):
    """Mock ImageWrapper backed by a (t, c, z, y, x) array."""
    size_t, size_c, size_z, size_y, size_x = planes.shape
    image = Mock()
    image.getId.return_value = image_id
    image.getName.return_value = name
    image.getSizeT.return_value = size_t
    image.getSizeC.return_value = size_c
    image.getSizeZ.return_value = size_z
    image.getSizeY.return_value = size_y
    image.getSizeX.return_value = size_x
    image.getPixelsType.return_value = pixels_type
    pixels = Mock()
    pixels.getPlane.side_effect = lambda z, c, t: planes[t, c, z]
    pixels.getTile.side_effect = (
        lambda z, c, t, tile: planes[t, c, z, tile[1]:tile[1] + tile[3], tile[0]:tile[0] + tile[2]]
    )
    image.getPrimaryPixels.return_value = pixels
    return image


@pytest.mark.unit
class TestClampRegion:
    def test_region_inside_image_is_unchanged(self):
        assert clamp_region((2, 3, 4, 5), size_y=20, size_x=20) == (2, 3, 4, 5)

    def test_region_past_the_edge_is_clipped(self):
        assert clamp_region((8, 6, 10, 10), size_y=10, size_x=12) == (8, 6, 4, 4)

    def test_region_starting_outside_image_raises(self):
        with pytest.raises(ValueError, match="outside"):
            clamp_region((12, 0, 4, 4), size_y=10, size_x=12)


@pytest.mark.unit
class TestOmeroImageSource:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.planes = rng.integers(0, 4000, size=(2, 3, 4, 10, 12), dtype=np.uint16)
        self.image = make_omero_image(self.planes)
        self.conn = Mock()
        self.conn.getObject.return_value = self.image
        self.source = OmeroImageSource(self.conn)

    def test_satisfies_protocol(self):
        assert isinstance(self.source, ImageSource)

    def test_get_image_accepts_omero_uri(self):
        assert self.source.get_image("omero:image:7") is self.image
        self.conn.getObject.assert_called_once_with("Image", 7)

    def test_get_image_rejects_non_omero_uri(self):
        with pytest.raises(LookupError, match="not an OMERO image"):
            self.source.get_image("https://h/p.zarr/B/2/0")

    def test_get_image_looks_up_by_id(self):
        assert self.source.get_image(7) is self.image
        self.conn.getObject.assert_called_once_with("Image", 7)

    def test_get_image_missing_raises_lookup_error(self):
        self.conn.getObject.return_value = None
        with pytest.raises(LookupError, match="42"):
            self.source.get_image(42)

    def test_get_image_without_connection_raises(self):
        with pytest.raises(ValueError, match="connection"):
            OmeroImageSource(None).get_image(7)

    def test_metadata(self):
        assert self.source.metadata(self.image) == ImageMeta(
            image_uri="omero:image:7", name="img.tif", size_t=2, size_c=3, size_z=4,
            size_y=10, size_x=12, dtype="uint16",
        )

    @pytest.mark.parametrize(
        "omero_type, numpy_name",
        [("uint8", "uint8"), ("uint16", "uint16"), ("float", "float32"), ("double", "float64")],
    )
    def test_metadata_dtype_is_a_numpy_name(self, omero_type, numpy_name):
        image = make_omero_image(self.planes, pixels_type=omero_type)
        assert self.source.metadata(image).dtype == numpy_name

    def test_read_plane_full(self):
        plane = self.source.read_plane(self.image, t=1, c=2, z=3)
        np.testing.assert_array_equal(plane, self.planes[1, 2, 3])

    def test_read_plane_region_uses_get_tile(self):
        plane = self.source.read_plane(self.image, t=0, c=1, z=2, region=(2, 3, 4, 5))
        np.testing.assert_array_equal(plane, self.planes[0, 1, 2, 3:8, 2:6])
        self.image.getPrimaryPixels().getTile.assert_called_once_with(2, 1, 0, (2, 3, 4, 5))

    def test_read_plane_region_past_edge_is_clipped(self):
        plane = self.source.read_plane(self.image, t=0, c=0, z=0, region=(8, 6, 10, 10))
        assert plane.shape == (4, 4)
        np.testing.assert_array_equal(plane, self.planes[0, 0, 0, 6:10, 8:12])

    def test_read_volume_is_z_y_x_inclusive(self):
        vol = self.source.read_volume(self.image, t=1, c=0, z_start=1, z_end=3)
        assert vol.shape == (3, 10, 12)
        np.testing.assert_array_equal(vol, self.planes[1, 0, 1:4])

    def test_read_volume_region(self):
        vol = self.source.read_volume(self.image, t=0, c=0, z_start=0, z_end=1, region=(0, 0, 5, 6))
        assert vol.shape == (2, 6, 5)

    def test_float_pixels_keep_dtype(self):
        planes = np.full((1, 1, 1, 4, 4), 0.5, dtype=np.float32)
        image = make_omero_image(planes, pixels_type="float")
        plane = self.source.read_plane(image, t=0, c=0, z=0)
        assert plane.dtype == np.float32

    def test_read_error_propagates(self):
        self.image.getPrimaryPixels().getPlane.side_effect = RuntimeError("ice down")
        with pytest.raises(RuntimeError, match="ice down"):
            self.source.read_plane(self.image, t=0, c=0, z=0)


from omero_annotate_ai.sources.uri import (
    annotation_id,
    normalize_zarr_uri,
    omero_uri,
    parse_omero_uri,
    uri_key,
)


@pytest.mark.unit
class TestUri:
    def test_omero_round_trip(self):
        assert omero_uri(42) == "omero:image:42"
        assert parse_omero_uri("omero:image:42") == 42

    @pytest.mark.parametrize("value", ["https://x/y.zarr", "omero:image:abc", 42, None])
    def test_parse_non_omero_returns_none(self, value):
        assert parse_omero_uri(value) is None

    def test_normalize_http_drops_query_and_trailing_slash(self):
        assert (
            normalize_zarr_uri("https://h/p.zarr/B/2/0/?anonymous=true")
            == "https://h/p.zarr/B/2/0"
        )

    def test_normalize_local_path_is_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert normalize_zarr_uri("img.zarr/") == str(tmp_path / "img.zarr")

    def test_uri_key_is_stable_sha1_prefix(self):
        assert uri_key("omero:image:1") == "4ba674d4"
        assert uri_key("omero:image:2") != uri_key("omero:image:1")

    def test_annotation_id_forms(self):
        key = uri_key("omero:image:1")
        assert annotation_id("omero:image:1", 0, 3) == f"{key}_0_3"
        assert annotation_id("omero:image:1", 0, 3, patch=2) == f"{key}_0_3_2"
        assert annotation_id("omero:image:1", 1, 0, volumetric=True) == f"{key}_1_3d"
        assert (
            annotation_id("omero:image:1", 1, 0, volumetric=True, patch=0)
            == f"{key}_1_3d_0"
        )
