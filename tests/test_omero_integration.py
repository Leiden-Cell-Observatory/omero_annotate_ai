"""Integration tests against a real OMERO server (the Docker stack in tests/).

Run with ``pixi run test-omero`` (or ``pytest -m docker``). The fixtures in
conftest.py reuse a running test server or start tests/docker-compose.yml, and
seed a uniquely named project, dataset and plate (tests/omero_seed.py).
"""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
from tests.omero_seed import CHANNEL_NAMES, IMAGE_SHAPE, synthetic_plane



@pytest.mark.docker
@pytest.mark.integration
class TestServer:
    """The test server is reachable and the seed data is in place."""

    def test_login(self, omero_conn):
        assert omero_conn.isConnected()
        assert omero_conn.getUser().getName() == "root"

    def test_seeded_dataset_info(self, omero_conn, omero_seed):
        from omero_annotate_ai.omero.omero_utils import get_container_info

        info = get_container_info(omero_conn, "dataset", omero_seed.dataset_id)

        assert info["total_images"] == len(omero_seed.image_ids)
        assert info["channel_names"] == CHANNEL_NAMES
        assert info["channel_names_vary"] is False
        assert info["dimensions"] == {
            "T": IMAGE_SHAPE["size_t"], "C": IMAGE_SHAPE["size_c"],
            "Z": IMAGE_SHAPE["size_z"], "Y": IMAGE_SHAPE["size_y"],
            "X": IMAGE_SHAPE["size_x"],
        }

    def test_seeded_plate_info(self, omero_conn, omero_seed):
        from omero_annotate_ai.omero.omero_utils import get_container_info

        if omero_seed.plate_id is None:
            pytest.skip("Plate could not be imported on this server")
        info = get_container_info(omero_conn, "plate", omero_seed.plate_id)
        assert info["total_images"] == 4


@pytest.mark.docker
@pytest.mark.integration
class TestOmeroImageSource:
    """The package's OMERO image source reads back the seeded pixels."""

    def _source_and_image(self, conn, seed, index=1):
        from omero_annotate_ai.sources.omero import OmeroImageSource

        source = OmeroImageSource(conn)
        return source, source.get_image(f"omero:image:{seed.image_ids[index]}")

    def test_metadata(self, omero_conn, omero_seed):
        source, image = self._source_and_image(omero_conn, omero_seed)
        meta = source.metadata(image)
        assert meta.image_uri == f"omero:image:{omero_seed.image_ids[1]}"
        assert (meta.size_t, meta.size_c, meta.size_z, meta.size_y, meta.size_x) == (
            IMAGE_SHAPE["size_t"], IMAGE_SHAPE["size_c"], IMAGE_SHAPE["size_z"],
            IMAGE_SHAPE["size_y"], IMAGE_SHAPE["size_x"])
        assert meta.dtype == "uint16"

    def test_plane_region_and_volume(self, omero_conn, omero_seed):
        source, image = self._source_and_image(omero_conn, omero_seed)

        plane = source.read_plane(image, t=1, c=1, z=2)
        np.testing.assert_array_equal(plane, synthetic_plane(2, 1, 1, 1))

        tile = source.read_plane(image, t=0, c=0, z=0, region=(10, 5, 20, 8))
        np.testing.assert_array_equal(tile, synthetic_plane(0, 0, 0, 1)[5:13, 10:30])

        volume = source.read_volume(image, t=0, c=1, z_start=0, z_end=2)
        assert volume.shape == (3, IMAGE_SHAPE["size_y"], IMAGE_SHAPE["size_x"])
        np.testing.assert_array_equal(volume[2], synthetic_plane(2, 1, 0, 1))

    def test_missing_image_raises(self, omero_conn):
        from omero_annotate_ai.sources.omero import OmeroImageSource

        with pytest.raises(LookupError):
            OmeroImageSource(omero_conn).get_image("omero:image:999999999")


@pytest.fixture
def scratch_table(omero_conn):
    """An empty OMERO.tables table (image_uri, processed, roi_id), deleted afterwards."""
    import omero
    import omero.grid

    resources = omero_conn.c.sf.sharedResources()
    repo_id = resources.repositories().descriptions[0].getId().getValue()
    table = resources.newTable(repo_id, "oai_test_inplace.h5")
    columns = [
        omero.grid.StringColumn("image_uri", "", 64, []),
        omero.grid.BoolColumn("processed", "", []),
        omero.grid.LongColumn("roi_id", "", []),
    ]
    table.initialize(columns)
    file_id = table.getOriginalFile().getId().getValue()
    table.close()
    yield resources, file_id
    omero_conn.deleteObjects("OriginalFile", [file_id], wait=True)


def _open(resources, file_id):
    import omero

    return resources.openTable(omero.model.OriginalFileI(file_id, False))


def _rows(table):
    data = table.read(list(range(len(table.getHeaders()))), 0, table.getNumberOfRows())
    return {column.name: list(column.values) for column in data.columns}


@pytest.mark.docker
@pytest.mark.integration
class TestTablesInPlace:
    """OMERO.tables can be changed without replacing the table (same file ID)."""

    def _append(self, table, uris, processed, roi_ids):
        columns = table.getHeaders()
        columns[0].values, columns[1].values, columns[2].values = uris, processed, roi_ids
        table.addData(columns)

    def test_update_and_append_keep_file_id(self, scratch_table):
        resources, file_id = scratch_table

        table = _open(resources, file_id)
        try:
            self._append(table, ["omero:image:1", "omero:image:2"], [False, False], [-1, -1])
        finally:
            table.close()

        table = _open(resources, file_id)
        try:
            data = table.readCoordinates([1])
            for column in data.columns:
                if column.name == "processed":
                    column.values = [True]
                elif column.name == "roi_id":
                    column.values = [42]
            table.update(data)
            self._append(table, ["omero:image:3"], [False], [-1])
        finally:
            table.close()

        table = _open(resources, file_id)
        try:
            assert table.getOriginalFile().getId().getValue() == file_id
            assert _rows(table) == {
                "image_uri": ["omero:image:1", "omero:image:2", "omero:image:3"],
                "processed": [False, True, False],
                "roi_id": [-1, 42, -1],
            }
        finally:
            table.close()

    def test_string_longer_than_column_is_rejected(self, scratch_table):
        import omero

        resources, file_id = scratch_table
        table = _open(resources, file_id)
        try:
            with pytest.raises(omero.ValidationException):
                self._append(table, ["x" * 100], [False], [-1])
            assert table.getNumberOfRows() == 0
        finally:
            table.close()


def _training_set_config(image_ids, categories):
    from omero_annotate_ai.core.annotation_config import ImageAnnotation, create_default_config

    config = create_default_config()
    config.name = "oai_table_layer"
    for image_id, category in zip(image_ids, categories):
        config.add_annotation(ImageAnnotation(
            image_id=image_id, image_name=f"img{image_id}",
            annotation_id=f"{image_id}_0_0", category=category,
            timepoint=0, z_slice=0, channel=0,
        ))
    return config


@pytest.fixture
def training_set(omero_conn, omero_seed):
    """A schema-v2 tracking table on the seeded dataset; everything removed afterwards."""
    from omero_annotate_ai.omero.omero_functions import create_tracking_table, delete_training_set

    image_ids = omero_seed.image_ids[:3]
    config = _training_set_config(image_ids, ["training", "validation", "test"])
    with patch("builtins.print"):
        table_id = create_tracking_table(
            omero_conn, config, "oai_table_layer", "dataset", [omero_seed.dataset_id])
    config.omero.table_id = table_id
    yield config, table_id
    with patch("builtins.print"):
        delete_training_set(omero_conn, table_id, dry_run=False)


@pytest.mark.docker
@pytest.mark.integration
class TestTrackingTableOnOmero:
    """The table layer against a real server: in place, tagged, one config."""

    def test_round_trip_keeps_test_split(self, omero_conn, training_set):
        from omero_annotate_ai.core.annotation_config import create_default_config
        from omero_annotate_ai.omero.omero_functions import sync_omero_table_to_config

        _, table_id = training_set
        loaded = create_default_config()
        with patch("builtins.print"):
            sync_omero_table_to_config(omero_conn, table_id, loaded)
        assert [a.category for a in loaded.annotations] == ["training", "validation", "test"]

    def test_update_and_append_keep_the_table_id(self, omero_conn, omero_seed, training_set):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation
        from omero_annotate_ai.omero import tracking_table
        from omero_annotate_ai.omero.omero_functions import write_config_to_table

        config, table_id = training_set
        file_id = omero_conn.getObject("FileAnnotation", table_id).getFile().getId()
        config.annotations[1].mark_processed(roi_id=None, label_id=None)
        config.annotations[1].session = 1
        extra = omero_seed.image_ids[3]
        config.add_annotation(ImageAnnotation(
            image_id=extra, image_name="extra", annotation_id=f"{extra}_0_0",
            category="training"))

        result = write_config_to_table(omero_conn, table_id, config)

        assert result == {"updated": 3, "added": 1}
        file_ann = omero_conn.getObject("FileAnnotation", table_id)
        assert file_ann.getFile().getId() == file_id
        df = tracking_table.read_table(omero_conn, table_id)
        assert list(df["processed"]) == [False, True, False, False]
        assert list(df["session"]) == [-1, 1, -1, -1]
        assert list(df["image_id"])[-1] == extra

    def test_over_long_string_is_refused_with_a_clear_error(self, omero_conn, training_set):
        from omero_annotate_ai.omero import tracking_table
        from omero_annotate_ai.omero.omero_functions import write_config_to_table

        config, table_id = training_set
        config.annotations[0].image_name = "x" * 600
        with pytest.raises(tracking_table.TableValueTooLongError, match="image_name"):
            write_config_to_table(omero_conn, table_id, config)

    def test_uploads_are_tagged_and_deleted_with_the_set(self, omero_conn, omero_seed, tmp_path):
        import imageio.v3 as imageio
        from omero_annotate_ai.omero.omero_functions import (
            create_tracking_table, delete_training_set, upload_rois_and_labels,
        )
        from omero_annotate_ai.omero.tracking_table import tag_for

        image_id = omero_seed.image_ids[4]
        config = _training_set_config([image_id], ["training"])
        with patch("builtins.print"):
            table_id = create_tracking_table(
                omero_conn, config, "oai_tagged", "dataset", [omero_seed.dataset_id])
        mask = np.zeros((IMAGE_SHAPE["size_y"], IMAGE_SHAPE["size_x"]), np.uint16)
        mask[10:20, 10:30] = 1
        mask_file = tmp_path / "mask.tif"
        imageio.imwrite(mask_file, mask)
        tag = tag_for(table_id, f"{image_id}_0_0")

        with patch("builtins.print"):
            # Uploaded, but (as after a crash) never written to the table.
            label_id, roi_id = upload_rois_and_labels(
                omero_conn, image_id, str(mask_file), trainingset_name=config.name,
                z_slice=0, channel=0, timepoint=0, tag=tag)

        roi = omero_conn.getObject("Roi", roi_id)
        assert roi.getDescription().startswith(tag)
        label = omero_conn.getObject("FileAnnotation", label_id)
        assert label.getDescription().startswith(tag)
        assert label.getFile().getName() == f"oai_table_layer_{image_id}_0_0_mask.tif"

        # The label file round-trips losslessly (zlib-compressed TIFF).
        import ezomero
        import tifffile
        download_dir = tmp_path / "download"
        download_dir.mkdir()
        downloaded = ezomero.get_file_annotation(omero_conn, label_id, str(download_dir))
        assert np.array_equal(tifffile.imread(downloaded), mask)

        with patch("builtins.print"):
            report = delete_training_set(omero_conn, table_id)
            assert report["rois"] == [roi_id] and report["labels"] == [label_id]
            delete_training_set(omero_conn, table_id, dry_run=False)
        assert omero_conn.getObject("Roi", roi_id) is None
        assert omero_conn.getObject("FileAnnotation", label_id) is None
        assert omero_conn.getObject("FileAnnotation", table_id) is None

    def test_one_config_per_table(self, omero_conn, omero_seed, training_set, tmp_path):
        from omero_annotate_ai.omero.omero_functions import (
            CONFIG_NS, find_annotation_config_for_table, replace_table_config,
        )
        from omero_annotate_ai.omero.omero_utils import list_annotations_by_namespace

        config, table_id = training_set
        local = tmp_path / "annotation_config_oai.yaml"
        for model in ("vit_b", "vit_l"):
            config.ai_model.pretrained_from = model
            config.save_yaml(local)
            replace_table_config(omero_conn, table_id, str(local), "dataset",
                                 omero_seed.dataset_id)

        with patch("builtins.print"):
            configs = [a for a in list_annotations_by_namespace(
                omero_conn, "Dataset", omero_seed.dataset_id, CONFIG_NS)
                if a["file_name"] == f"annotation_config_table_{table_id}.yaml"]
        assert len(configs) == 1
        assert find_annotation_config_for_table(omero_conn, table_id).ai_model.pretrained_from == "vit_l"

    def test_tables_from_older_versions_are_refused(self, omero_conn, omero_seed):
        import ezomero
        import pandas as pd
        from omero_annotate_ai.omero import tracking_table

        old_id = ezomero.post_table(
            omero_conn, pd.DataFrame({"image_id": [1], "train": [True]}),
            object_type="Dataset", object_id=omero_seed.dataset_id, title="oai_old_table")
        try:
            with pytest.raises(tracking_table.UnsupportedTableError):
                tracking_table.read_table(omero_conn, old_id)
        finally:
            omero_conn.deleteObjects("Annotation", [old_id], wait=True)


@pytest.mark.unit
class TestOMEROConnectionManagerUnit:
    """Unit tests that don't require OMERO server."""
    
    def test_connection_manager_creation(self):
        """
        Tests that the `SimpleOMEROConnection` manager can be created.
        This is a simple smoke test to ensure that the `SimpleOMEROConnection`
        class can be instantiated without errors.
        """
        conn_manager = SimpleOMEROConnection()
        assert conn_manager is not None
        assert hasattr(conn_manager, 'connect')
        assert hasattr(conn_manager, 'get_last_connection')
    
    def test_config_loading_without_files(self):
        """
        Tests the configuration loading when no config files exist.
        This test ensures that the `load_config_files` method handles the case
        where no environment or configuration files are present and returns a
        dictionary with default empty values.
        """
        conn_manager = SimpleOMEROConnection()
        
        # Environment variables to clear
        env_vars_to_clear = ['HOST', 'USER_NAME', 'GROUP', 'OMERO_USER', 'OMERO_PASSWORD', 'OMERO_HOST', 'OMERO_GROUP', 'OMERO_PORT']
        
        with patch.dict(os.environ, {}, clear=True), \
            patch.object(Path, 'exists', return_value=False), \
            patch.object(conn_manager, 'load_connection_history', return_value=[]), \
            patch('builtins.print'):  # Suppress print statements
            
            config = conn_manager.load_config_files()
            
            assert isinstance(config, dict)
            # Should be completely empty when no config sources exist
            assert config == {}
            # Or if you expect default empty values:
            assert config.get("host", "") == ""
            assert config.get("username", "") == ""
            assert config.get("group", "") == ""