"""Tests for prepare_bia_data_from_table (OMERO table -> BIA on-disk layout).

The OMERO connection, ``ezomero`` and the image source are faked - no live server.
"""

import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from tifffile import imread, imwrite

from omero_annotate_ai.core.annotation_config import (
    create_default_config,
)
from omero_annotate_ai.core.mifa_export import build_bia_file_lists, save_bia_package
from omero_annotate_ai.processing.bia_data import prepare_bia_data_from_table
from omero_annotate_ai.rows import Row, rows_to_frame

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


class PlaneSource:
    """ImageSource serving the same (y, x) plane for every image, t, c and z."""

    def __init__(self, plane):
        self.plane = plane
        self.reads = []

    def get_image(self, image_uri):
        return image_uri

    def read_plane(self, image, *, t, c, z, region=None):
        self.reads.append((image, t, c, z, region))
        if region is None:
            return self.plane
        x, y, width, height = region
        return self.plane[y : y + height, x : x + width]

    def read_volume(self, image, *, t, c, z_start, z_end, region=None):
        return np.stack(
            [
                self.read_plane(image, t=t, c=c, z=z, region=region)
                for z in range(z_start, z_end + 1)
            ]
        )


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
    """Patched ezomero module inside bia_data; pixels come from ``ez.source``."""
    source = PlaneSource(plane)
    with (
        patch("omero_annotate_ai.processing.bia_data.ezomero") as ez,
        patch(
            "omero_annotate_ai.processing.bia_data._read_table",
            side_effect=lambda conn, table_id: ez.get_table(conn, table_id),
        ),
        patch(
            "omero_annotate_ai.processing.bia_data.OmeroImageSource",
            return_value=source,
        ),
    ):
        ez.get_file_annotation.side_effect = fake_get_file_annotation
        ez.source = source
        yield ez


def build_table(annotations):
    """OMERO table DataFrame for the given Rows."""
    return rows_to_frame(list(annotations))


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
    fields.setdefault(
        "annotation_id",
        f"{fields['image_id']}_{fields['timepoint']}_{fields['z_slice']}",
    )
    return Row(**fields)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def test_writes_input_and_mask_with_annotation_id_names(
    mock_conn, mock_ezomero, tmp_path
):
    """annotation_input/{annotation_id}.tif + annotation_output/{annotation_id}_mask.tif."""
    table = build_table(
        [
            annotation(image_id=101, label_id=555),
            annotation(image_id=102, z_slice=3, label_id=556),
        ]
    )
    mock_ezomero.get_table.return_value = table

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    mock_ezomero.get_table.assert_called_once_with(mock_conn, 42)

    ids = [ann.annotation_id for ann in stats["rows"]]
    assert ids == ["101_0_0", "102_0_3"]  # pipeline scheme: {image}_{t}_{z}
    for annotation_id in ids:
        assert (tmp_path / "annotation_input" / f"{annotation_id}.tif").exists()
        assert (tmp_path / "annotation_output" / f"{annotation_id}_mask.tif").exists()

    assert not (tmp_path / "model_input").exists()
    assert stats["n_images"] == 2
    assert stats["n_masks"] == 2


def test_separate_channels_still_write_the_annotated_channel(
    mock_conn, mock_ezomero, tmp_path
):
    """A label/training channel split still exports the annotated channel to annotation_input/.

    The BIA source image is the channel the mask was drawn on; model_input/ is a
    training concern and is not written here.
    """
    config = create_default_config()
    config.spatial_coverage.channels = [0, 1]
    config.spatial_coverage.label_channel = 0
    config.spatial_coverage.training_channels = [1]
    assert config.spatial_coverage.uses_separate_channels()

    mock_ezomero.get_table.return_value = build_table([annotation()])

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    annotation_id = stats["rows"][0].annotation_id
    assert (tmp_path / "annotation_input" / f"{annotation_id}.tif").exists()
    assert not (tmp_path / "model_input").exists()
    assert (tmp_path / "annotation_output" / f"{annotation_id}_mask.tif").exists()
    assert stats["n_images"] == 1


# --------------------------------------------------------------------------- #
# Native bit depth (the reason this is not the training path)
# --------------------------------------------------------------------------- #


def test_uint16_image_is_written_at_native_dtype(
    mock_conn, mock_ezomero, tmp_path, plane
):
    """A uint16 source plane must NOT be squashed to uint8 (training path does that)."""
    assert plane.dtype == np.uint16
    assert plane.max() > 255

    mock_ezomero.get_table.return_value = build_table([annotation()])

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    written = imread(
        str(tmp_path / "annotation_input" / f"{stats['rows'][0].annotation_id}.tif")
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

    with pytest.warns(UserWarning):
        stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    assert stats["n_images"] == 1
    assert stats["n_masks"] == 1
    assert stats["skipped_unprocessed"] == 2
    assert stats["skipped_no_mask"] == 1

    assert sorted(p.name for p in (tmp_path / "annotation_input").glob("*.tif")) == [
        "101_0_0.tif"
    ]
    assert sorted(p.name for p in (tmp_path / "annotation_output").glob("*.tif")) == [
        "101_0_0_mask.tif"
    ]


def test_excluded_rows_are_skipped_silently(mock_conn, mock_ezomero, tmp_path):
    excluded = annotation(image_id=102, label_id=None)
    excluded.mark_excluded(session=1)
    mock_ezomero.get_table.return_value = build_table(
        [annotation(image_id=101, label_id=555), excluded]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    assert [r.annotation_id for r in stats["rows"]] == ["101_0_0"]
    assert (stats["skipped_unprocessed"], stats["skipped_no_mask"]) == (0, 0)


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
        stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    rows = stats["rows"]
    assert [ann.annotation_id for ann in rows] == ["101_0_0"]

    # Package it: every path the file lists reference must exist inside the bundle.
    bundle = tmp_path / "submission"
    result = save_bia_package(config, rows, tmp_path, bundle, accession="S-BIAD999")

    images_df, annotations_df = build_bia_file_lists(rows)
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

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    assert set(stats) == {
        "output_dir",
        "rows",
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


def test_patch_row_reads_only_its_patch(mock_conn, mock_ezomero, tmp_path):
    """A patch row reads exactly its patch region, never the full plane."""
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

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    # (image, t, c, z, (x, y, width, height))
    assert mock_ezomero.source.reads == [("omero:image:101", 2, 3, 1, (2, 1, 4, 3))]

    written = imread(
        str(tmp_path / "annotation_input" / f"{stats['rows'][0].annotation_id}.tif")
    )
    assert written.shape == (3, 4)  # (height, width)
    assert written.dtype == np.uint16


def test_full_plane_row_reads_the_whole_plane(mock_conn, mock_ezomero, tmp_path):
    mock_ezomero.get_table.return_value = build_table([annotation()])

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    assert mock_ezomero.source.reads == [("omero:image:101", 0, 0, 0, None)]

    written = imread(
        str(tmp_path / "annotation_input" / f"{stats['rows'][0].annotation_id}.tif")
    )
    assert written.shape == (SIZE_Y, SIZE_X)


def test_volumetric_row_writes_a_stack(mock_conn, mock_ezomero, tmp_path):
    volume = annotation(
        image_id=101, is_volumetric=True, z_start=0, z_end=2, z_length=3
    )
    mock_ezomero.get_table.return_value = build_table([volume])

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    annotation_id = stats["rows"][0].annotation_id
    written = imread(str(tmp_path / "annotation_input" / f"{annotation_id}.tif"))
    assert written.shape == (3, SIZE_Y, SIZE_X)  # (z, y, x)
    assert written.dtype == np.uint16


def test_mask_is_downloaded_from_the_row_label_id(mock_conn, mock_ezomero, tmp_path):
    mock_ezomero.get_table.return_value = build_table([annotation(label_id=777)])

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    args, _ = mock_ezomero.get_file_annotation.call_args
    assert args[0] is mock_conn
    assert args[1] == 777

    mask = imread(
        str(
            tmp_path
            / "annotation_output"
            / f"{stats['rows'][0].annotation_id}_mask.tif"
        )
    )
    assert mask.dtype == np.uint16
    assert (mask == 777 % 256).all()
    assert not (tmp_path / "tmp").exists()  # scratch dir cleaned up


# --------------------------------------------------------------------------- #
# clean_existing
# --------------------------------------------------------------------------- #


def test_clean_existing_clears_previous_data_dirs(mock_conn, mock_ezomero, tmp_path):
    for name in ("annotation_input", "annotation_output"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "stale.tif").write_bytes(b"stale")

    mock_ezomero.get_table.return_value = build_table([annotation()])

    stats = prepare_bia_data_from_table(mock_conn, 42, tmp_path, clean_existing=True)

    assert not (tmp_path / "annotation_input" / "stale.tif").exists()
    assert not (tmp_path / "annotation_output" / "stale.tif").exists()
    assert (
        tmp_path / "annotation_input" / f"{stats['rows'][0].annotation_id}.tif"
    ).exists()


def test_without_clean_existing_previous_files_survive(
    mock_conn, mock_ezomero, tmp_path
):
    stale = tmp_path / "annotation_input" / "stale.tif"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")

    mock_ezomero.get_table.return_value = build_table([annotation()])

    prepare_bia_data_from_table(mock_conn, 42, tmp_path)

    assert stale.exists()


def test_file_lists_carry_pixel_digests_and_iscc_codes_when_recorded():
    rows = [
        Row(
            image_id=101,
            image_name="a",
            annotation_id="a_0",
            processed=True,
            label_id=5,
            input_digest="in-sha",
            mask_digest="mask-sha",
            input_iscc="ISCC:EEA4XSZUGQ2MXS6B",
            mask_iscc="ISCC:EEA6IF2FC2OREY73",
        )
    ]
    images_df, annotations_df = build_bia_file_lists(rows)
    assert annotations_df.loc[0, "pixel_sha256"] == "mask-sha"
    assert annotations_df.loc[0, "iscc"] == "ISCC:EEA6IF2FC2OREY73"
    assert images_df.loc[0, "iscc"] == "ISCC:EEA4XSZUGQ2MXS6B"

    plain = [Row(image_id=101, image_name="a", annotation_id="a_0", label_id=5)]
    images_df, annotations_df = build_bia_file_lists(plain)
    assert "iscc" not in annotations_df.columns and "iscc" not in images_df.columns


# --------------------------------------------------------------------------- #
# A local training set straight into a bundle
# --------------------------------------------------------------------------- #


def test_local_set_bundle_holds_only_annotated_units(tmp_path):
    pytest.importorskip("bia_mifa_models")
    from tests.fakes import annotate, create_set, omero_set

    config, source, omero = omero_set(tmp_path, n_images=3)
    config.storage = "local"
    ts = create_set(config, source, omero, tmp_path / "home")
    assert annotate(ts, limit=2) == 2

    result = save_bia_package(ts.config, ts.units(), ts.bundle_dir, tmp_path / "bia")

    assert result["n_annotations"] == 2
    assert result["missing"] == 0
    assert len(list((tmp_path / "bia" / "annotations").glob("*_mask.tif"))) == 2
    assert len(list(ts.bundle_dir.glob("annotation_output/*_mask.tif"))) == 2  # copied


def test_local_set_bundle_leaves_excluded_units_out(tmp_path):
    pytest.importorskip("bia_mifa_models")
    from tests.fakes import annotate, create_set, omero_set

    config, source, omero = omero_set(tmp_path, n_images=3)
    config.storage = "local"
    ts = create_set(config, source, omero, tmp_path / "home")
    session = ts.start_session(batch_size=1)
    batch = session.next_batch()
    session.exclude(batch, [batch.items[0].annotation_id])
    session.finish()
    assert annotate(ts) == 2

    result = save_bia_package(ts.config, ts.units(), ts.bundle_dir, tmp_path / "bia")

    assert (result["n_annotations"], result["missing"]) == (2, 0)


def test_mask_changed_on_omero_since_annotation_is_refused(
    mock_conn, mock_ezomero, tmp_path
):
    from omero_annotate_ai.bundle import file_digest
    from omero_annotate_ai.errors import IntegrityError

    # What the fake OMERO serves for label 555, and its digest as recorded at annotation.
    served = tmp_path / "served"
    recorded = file_digest(Path(fake_get_file_annotation(None, 555, served)))
    mock_ezomero.get_table.return_value = build_table(
        [
            annotation(image_id=101, mask_digest=recorded),
            annotation(image_id=102, label_id=556, mask_digest=recorded),  # differs
        ]
    )
    with pytest.raises(IntegrityError, match="102_0_0"):
        prepare_bia_data_from_table(mock_conn, 1, tmp_path / "out")
