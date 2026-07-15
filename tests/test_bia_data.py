"""Tests for prepare_bia_data_from_table (OMERO table -> BIA on-disk layout).

The OMERO connection and the ``ezomero`` module are mocked - no live server.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from tifffile import imread, imwrite

from omero_annotate_ai.core.annotation_config import (
    ImageAnnotation,
    create_default_config,
)
from omero_annotate_ai.core.mifa_export import build_bia_file_lists
from omero_annotate_ai.processing.bia_data import prepare_bia_data_from_table

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

SIZE_X, SIZE_Y = 8, 6  # full-plane size of the fake OMERO image (x, y)


def make_plane(dtype=np.uint16, size_y=SIZE_Y, size_x=SIZE_X):
    """A (y, x) plane with distinct values, spanning more than 8 bits for uint16."""
    plane = np.arange(size_y * size_x, dtype=dtype).reshape(size_y, size_x)
    if np.dtype(dtype) == np.uint16:
        plane = (plane * 100).astype(np.uint16)  # values well above 255
    return plane


def fake_get_image_factory(plane):
    """Mimic ezomero.get_image: XYZCT 5D pixels, or the image object with no_pixels."""

    def fake_get_image(
        conn,
        image_id,
        start_coords=None,
        axis_lengths=None,
        xyzct=False,
        no_pixels=False,
        **kwargs,
    ):
        if no_pixels:
            omero_image = Mock()
            omero_image.getSizeX.return_value = SIZE_X
            omero_image.getSizeY.return_value = SIZE_Y
            omero_image.getSizeZ.return_value = 1
            return omero_image, None

        x0, y0 = int(start_coords[0]), int(start_coords[1])
        width, height = int(axis_lengths[0]), int(axis_lengths[1])
        region = plane[y0 : y0 + height, x0 : x0 + width]  # (y, x)
        # ezomero with xyzct=True returns (x, y, z, c, t)
        pixels = np.swapaxes(region, 0, 1)[:, :, None, None, None]
        return None, pixels

    return fake_get_image


def fake_get_file_annotation(conn, label_id, folder, **kwargs):
    """Mimic ezomero.get_file_annotation: writes the mask into folder, returns its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"downloaded_{label_id}.tif"
    mask = np.full((SIZE_Y, SIZE_X), int(label_id) % 256, dtype=np.uint16)
    imwrite(str(path), mask)
    return str(path)


@pytest.fixture
def mock_conn():
    conn = Mock()
    conn.isConnected.return_value = True
    conn.getObject.return_value = Mock()  # FileAnnotation lookup succeeds
    return conn


@pytest.fixture
def plane():
    return make_plane()


@pytest.fixture
def mock_ezomero(plane):
    """Patched ezomero module inside bia_data."""
    with patch("omero_annotate_ai.processing.bia_data.ezomero") as ez:
        ez.get_image.side_effect = fake_get_image_factory(plane)
        ez.get_file_annotation.side_effect = fake_get_file_annotation
        yield ez


def build_table(annotations, config=None):
    """OMERO table DataFrame for the given ImageAnnotations (via to_dataframe)."""
    source = config or create_default_config()
    source.annotations = list(annotations)
    return source.to_dataframe()


def annotation(**overrides):
    """A processed, full-plane 2D annotation with a mask."""
    fields = dict(
        image_id=101,
        image_name="img.tif",
        timepoint=0,
        z_slice=0,
        channel=0,
        processed=True,
        label_id=555,
    )
    fields.update(overrides)
    return ImageAnnotation(**fields)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def test_writes_input_and_mask_with_annotation_id_names(
    mock_conn, mock_ezomero, tmp_path
):
    """input/{annotation_id}.tif + output/{annotation_id}_mask.tif."""
    table = build_table(
        [
            annotation(image_id=101, label_id=555),
            annotation(image_id=102, z_slice=3, label_id=556),
        ]
    )
    mock_ezomero.get_table.return_value = table

    config = create_default_config()
    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path, config=config)

    mock_ezomero.get_table.assert_called_once_with(mock_conn, 42)

    ids = [ann.annotation_id for ann in config.annotations]
    assert ids == ["101_0_0", "102_0_3"]  # pipeline scheme: {image}_{t}_{z}
    for annotation_id in ids:
        assert (tmp_path / "input" / f"{annotation_id}.tif").exists()
        assert (tmp_path / "output" / f"{annotation_id}_mask.tif").exists()

    assert not (tmp_path / "label_input").exists()
    assert stats["n_images"] == 2
    assert stats["n_masks"] == 2


def test_separate_channels_route_image_to_label_input(mock_conn, mock_ezomero, tmp_path):
    """A label/training channel split puts the image in label_input/, not input/."""
    config = create_default_config()
    config.spatial_coverage.channels = [0, 1]
    config.spatial_coverage.label_channel = 0
    config.spatial_coverage.training_channels = [1]
    assert config.spatial_coverage.uses_separate_channels()

    mock_ezomero.get_table.return_value = build_table([annotation()])

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path, config=config)

    annotation_id = config.annotations[0].annotation_id
    assert (tmp_path / "label_input" / f"{annotation_id}.tif").exists()
    assert not (tmp_path / "input").exists()
    # the mask still goes to output/
    assert (tmp_path / "output" / f"{annotation_id}_mask.tif").exists()
    assert stats["n_images"] == 1


# --------------------------------------------------------------------------- #
# Native bit depth (the reason this is not the training path)
# --------------------------------------------------------------------------- #


def test_uint16_image_is_written_at_native_dtype(mock_conn, mock_ezomero, tmp_path, plane):
    """A uint16 source plane must NOT be squashed to uint8 (training path does that)."""
    assert plane.dtype == np.uint16
    assert plane.max() > 255

    mock_ezomero.get_table.return_value = build_table([annotation()])

    config = create_default_config()
    prepare_bia_data_from_table(mock_conn, 42, tmp_path, config=config)

    written = imread(
        str(tmp_path / "input" / f"{config.annotations[0].annotation_id}.tif")
    )
    assert written.dtype == np.uint16
    assert written.max() == plane.max()
    np.testing.assert_array_equal(written, plane)  # (y, x), values unchanged


# --------------------------------------------------------------------------- #
# Row filtering
# --------------------------------------------------------------------------- #


def test_unprocessed_and_maskless_rows_are_skipped_and_counted(
    mock_conn, mock_ezomero, tmp_path
):
    table = build_table(
        [
            annotation(image_id=101, label_id=555),  # kept
            annotation(image_id=102, processed=False, label_id=556),  # unprocessed
            annotation(image_id=103, processed=False, label_id=None),  # unprocessed
            annotation(image_id=104, label_id=None),  # processed, no mask
        ]
    )
    mock_ezomero.get_table.return_value = table

    config = create_default_config()
    with pytest.warns(UserWarning):
        stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path, config=config)

    assert stats["n_images"] == 1
    assert stats["n_masks"] == 1
    assert stats["skipped_unprocessed"] == 2
    assert stats["skipped_no_mask"] == 1

    assert sorted(p.name for p in (tmp_path / "input").glob("*.tif")) == ["101_0_0.tif"]
    assert sorted(p.name for p in (tmp_path / "output").glob("*.tif")) == [
        "101_0_0_mask.tif"
    ]


def test_skipped_rows_are_pruned_from_the_config(mock_conn, mock_ezomero, tmp_path):
    """Skipped rows must not survive into the config save_bia_package() reads.

    The BIA file lists are built from every annotation in the config, so a skipped row
    left in place would put a path to a file we never downloaded into the submission.
    """
    mock_ezomero.get_table.return_value = build_table(
        [
            annotation(image_id=101, label_id=555),  # exported
            annotation(image_id=102, processed=False, label_id=556),  # unprocessed
            annotation(image_id=103, label_id=None),  # processed, no mask
        ]
    )

    config = create_default_config()
    with pytest.warns(UserWarning):
        prepare_bia_data_from_table(mock_conn, 42, tmp_path, config=config)

    assert [ann.annotation_id for ann in config.annotations] == ["101_0_0"]

    # Package it: every path the file lists reference must exist inside the bundle.
    config.output.output_directory = tmp_path
    bundle = tmp_path / "submission"
    result = config.save_bia_package(bundle, accession="S-BIAD999")

    images_df, annotations_df = build_bia_file_lists(config)
    referenced = (
        set(images_df["Files"])
        | set(annotations_df["Files"])
        | set(annotations_df["source_image"])
    )
    assert referenced == {"images/101_0_0.tif", "annotations/101_0_0_mask.tif"}
    assert result["missing"] == 0
    for rel in referenced:
        assert (bundle / rel).exists(), f"file list references missing file: {rel}"


def test_stats_dict_keys_and_counts(mock_conn, mock_ezomero, tmp_path):
    mock_ezomero.get_table.return_value = build_table(
        [annotation(image_id=101, label_id=555), annotation(image_id=102, label_id=556)]
    )

    stats = prepare_bia_data_from_table(
        mock_conn, 42, tmp_path, config=create_default_config()
    )

    assert set(stats) == {
        "output_dir",
        "n_images",
        "n_masks",
        "skipped_unprocessed",
        "skipped_no_mask",
    }
    assert stats["output_dir"] == Path(tmp_path)
    assert stats["n_images"] == 2
    assert stats["n_masks"] == 2
    assert stats["skipped_unprocessed"] == 0
    assert stats["skipped_no_mask"] == 0


def test_empty_table_raises(mock_conn, mock_ezomero, tmp_path):
    import pandas as pd

    mock_ezomero.get_table.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="empty or not found"):
        prepare_bia_data_from_table(mock_conn, 42, tmp_path)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def test_patch_row_requests_patch_coords_from_ezomero(mock_conn, mock_ezomero, tmp_path):
    """A patch row asks ezomero for exactly its patch region, no full-plane fetch."""
    patch_row = annotation(
        image_id=101,
        timepoint=2,
        z_slice=1,
        channel=3,
        is_patch=True,
        patch_x=2,
        patch_y=1,
        patch_width=4,
        patch_height=3,
    )
    mock_ezomero.get_table.return_value = build_table([patch_row])

    config = create_default_config()
    prepare_bia_data_from_table(mock_conn, 42, tmp_path, config=config)

    # Only the pixel fetch; a patch needs no no_pixels dimension lookup.
    assert mock_ezomero.get_image.call_count == 1
    _, kwargs = mock_ezomero.get_image.call_args
    assert kwargs["start_coords"] == (2, 1, 1, 3, 2)  # (x, y, z, c, t)
    assert kwargs["axis_lengths"] == (4, 3, 1, 1, 1)  # (w, h, 1, 1, 1)
    assert kwargs["xyzct"] is True

    written = imread(
        str(tmp_path / "input" / f"{config.annotations[0].annotation_id}.tif")
    )
    assert written.shape == (3, 4)  # (height, width)
    assert written.dtype == np.uint16


def test_full_plane_row_requests_image_size(mock_conn, mock_ezomero, tmp_path):
    mock_ezomero.get_table.return_value = build_table([annotation()])

    config = create_default_config()
    prepare_bia_data_from_table(mock_conn, 42, tmp_path, config=config)

    # First call resolves the size (no_pixels), second fetches the plane.
    assert mock_ezomero.get_image.call_count == 2
    _, kwargs = mock_ezomero.get_image.call_args
    assert kwargs["start_coords"] == (0, 0, 0, 0, 0)
    assert kwargs["axis_lengths"] == (SIZE_X, SIZE_Y, 1, 1, 1)

    written = imread(
        str(tmp_path / "input" / f"{config.annotations[0].annotation_id}.tif")
    )
    assert written.shape == (SIZE_Y, SIZE_X)


def test_volumetric_row_writes_a_stack(mock_conn, mock_ezomero, tmp_path):
    volume = annotation(
        image_id=101, is_volumetric=True, z_start=0, z_end=2, z_length=3
    )
    mock_ezomero.get_table.return_value = build_table([volume])

    config = create_default_config()
    prepare_bia_data_from_table(mock_conn, 42, tmp_path, config=config)

    annotation_id = config.annotations[0].annotation_id
    assert annotation_id == "101_0_3d"
    written = imread(str(tmp_path / "input" / f"{annotation_id}.tif"))
    assert written.shape == (3, SIZE_Y, SIZE_X)  # (z, y, x)
    assert written.dtype == np.uint16


def test_mask_is_downloaded_from_the_row_label_id(mock_conn, mock_ezomero, tmp_path):
    mock_ezomero.get_table.return_value = build_table([annotation(label_id=777)])

    config = create_default_config()
    prepare_bia_data_from_table(mock_conn, 42, tmp_path, config=config)

    args, _ = mock_ezomero.get_file_annotation.call_args
    assert args[0] is mock_conn
    assert args[1] == 777

    mask = imread(
        str(
            tmp_path / "output" / f"{config.annotations[0].annotation_id}_mask.tif"
        )
    )
    assert mask.dtype == np.uint16
    assert (mask == 777 % 256).all()
    assert not (tmp_path / "tmp").exists()  # scratch dir cleaned up


# --------------------------------------------------------------------------- #
# clean_existing
# --------------------------------------------------------------------------- #


def test_clean_existing_clears_previous_data_dirs(mock_conn, mock_ezomero, tmp_path):
    for name in ("input", "label_input", "output"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "stale.tif").write_bytes(b"stale")

    mock_ezomero.get_table.return_value = build_table([annotation()])

    config = create_default_config()
    prepare_bia_data_from_table(
        mock_conn, 42, tmp_path, config=config, clean_existing=True
    )

    assert not (tmp_path / "input" / "stale.tif").exists()
    assert not (tmp_path / "output" / "stale.tif").exists()
    assert not (tmp_path / "label_input").exists()  # removed, not recreated
    assert (tmp_path / "input" / f"{config.annotations[0].annotation_id}.tif").exists()


def test_without_clean_existing_previous_files_survive(mock_conn, mock_ezomero, tmp_path):
    stale = tmp_path / "input" / "stale.tif"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")

    mock_ezomero.get_table.return_value = build_table([annotation()])

    prepare_bia_data_from_table(
        mock_conn, 42, tmp_path, config=create_default_config()
    )

    assert stale.exists()
