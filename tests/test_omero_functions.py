"""Tests for OMERO integration functions."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from omero_annotate_ai.core.annotation_config import (
    AIModelConfig,
    AnnotationConfig,
    AnnotationMethodology,
    DatasetInfo,
    FeatureType,
    ImageAnnotation,
    OMEROConfig,
    OutputConfig,
    SpatialCoverage,
    StudyContext,
    TrainingConfig,
    WorkflowConfig,
)
from omero_annotate_ai.errors import TableValueTooLongError, UnsupportedTableError
from omero_annotate_ai.omero.biomero import (
    generate_set_id,
    geojson_namespace,
    list_manifests_from_omero,
    load_geojson_from_omero,
    load_manifest_from_omero,
    manifest_namespace,
    merge_geojson_patches,
)
from omero_annotate_ai.omero.omero_functions import (
    CONFIG_NS,
    create_tracking_table,
    generate_unique_table_name,
    link_table_to_containers,
    list_annotation_tables,
    replace_table_config,
    sync_omero_table_to_config,
    upload_rois_and_labels,
    write_config_to_table,
)
from omero_annotate_ai.omero.omero_utils import list_user_tables


@pytest.mark.unit
class TestLinkTableToContainers:
    """Test link_table_to_containers function."""

    def test_link_table_to_containers_dataset(self):
        """Test linking table to dataset containers."""
        # Mock connection and services
        mock_conn = Mock()
        mock_update_service = Mock()
        mock_conn.getUpdateService.return_value = mock_update_service

        # Mock FileAnnotation object
        mock_file_ann = Mock()
        mock_file_ann._obj = Mock()
        mock_conn.getObject.return_value = mock_file_ann

        # Mock save and return
        def mock_save(link):
            mock_saved = Mock()
            mock_saved.getId.return_value.getValue.return_value = 999
            return mock_saved

        mock_update_service.saveAndReturnObject.side_effect = mock_save

        results = link_table_to_containers(
            conn=mock_conn, table_id=100, container_type="dataset", container_ids=[2, 3]
        )

        assert len(results) == 2
        assert results[2] == 999
        assert results[3] == 999
        assert mock_update_service.saveAndReturnObject.call_count == 2

    def test_link_table_to_containers_invalid_type(self):
        """Test that invalid container type returns None results."""
        mock_conn = Mock()

        results = link_table_to_containers(
            conn=mock_conn,
            table_id=100,
            container_type="invalid_type",
            container_ids=[1, 2],
        )

        assert results == {1: None, 2: None}

    def test_link_table_to_containers_file_ann_not_found(self):
        """Test that missing FileAnnotation raises ValueError."""
        mock_conn = Mock()
        mock_conn.getObject.return_value = None

        with pytest.raises(ValueError, match="FileAnnotation 100 not found"):
            link_table_to_containers(
                conn=mock_conn,
                table_id=100,
                container_type="dataset",
                container_ids=[1, 2],
            )


@pytest.mark.unit
class TestCreateTrackingTable:
    """create_tracking_table makes the table once, linked to every container."""

    def _config(self):
        from omero_annotate_ai.core.annotation_config import (
            ImageAnnotation,
            create_default_config,
        )

        config = create_default_config()
        for i, category in enumerate(["training", "validation", "test"]):
            config.add_annotation(
                ImageAnnotation(
                    image_id=i + 1,
                    image_name=f"img{i}",
                    annotation_id=f"a{i}",
                    category=category,
                )
            )
        return config

    def test_links_all_containers_and_prepares_rows(self):
        with (
            patch(
                "omero_annotate_ai.omero.tracking_table.create_table", return_value=100
            ) as create,
            patch("builtins.print"),
        ):
            table_id = create_tracking_table(
                Mock(), self._config(), "run", "plate", [1, 2, 3]
            )

        assert table_id == 100
        conn, df, title, ctype, cids = create.call_args.args
        assert (title, ctype, cids) == ("run", "plate", [1, 2, 3])
        assert list(df["category"]) == ["training", "validation", "test"]
        assert pd.api.types.is_integer_dtype(df["annotation_input_id"])

    def test_empty_config_is_an_error(self):
        from omero_annotate_ai.core.annotation_config import create_default_config

        with pytest.raises(ValueError, match="No annotations"):
            create_tracking_table(Mock(), create_default_config(), "run", "plate", [1])


@pytest.mark.unit
class TestWriteAndSyncTable:
    """Writing rows in place, and reading them back strictly."""

    def test_write_empty_config_writes_nothing(self):
        config = Mock()
        config.to_dataframe.return_value = pd.DataFrame()
        with patch("omero_annotate_ai.omero.tracking_table.write_rows") as write:
            assert write_config_to_table(Mock(), 7, config) == {
                "updated": 0,
                "added": 0,
            }
        write.assert_not_called()

    def test_write_passes_prepared_rows(self):
        config = TestCreateTrackingTable()._config()
        with patch(
            "omero_annotate_ai.omero.tracking_table.write_rows",
            return_value={"updated": 3, "added": 0},
        ) as write:
            write_config_to_table(Mock(), 7, config)
        assert write.call_args.args[1] == 7
        assert list(write.call_args.args[2]["annotation_id"]) == ["a0", "a1", "a2"]

    def test_sync_raises_on_empty_table(self):
        with (
            patch(
                "omero_annotate_ai.omero.omero_functions.read_tracking_table",
                return_value=pd.DataFrame(),
            ),
            pytest.raises(ValueError, match="no rows"),
        ):
            sync_omero_table_to_config(Mock(), 7, Mock())

    def test_sync_raises_when_read_fails(self):
        with (
            patch(
                "omero_annotate_ai.omero.omero_functions.read_tracking_table",
                side_effect=RuntimeError("network"),
            ),
            pytest.raises(RuntimeError),
        ):
            sync_omero_table_to_config(Mock(), 7, Mock())


@pytest.mark.unit
class TestReplaceTableConfig:
    """Each training set keeps exactly one config YAML on OMERO."""

    def test_config_ns_value(self):
        assert CONFIG_NS == "openmicroscopy.org/omero/annotate/config"

    def test_posts_new_then_deletes_old(self, tmp_path):
        local = tmp_path / "annotation_config_run.yaml"
        local.write_text("name: run\n")
        conn = Mock()
        posted = {}

        def post(conn_, file_path, ns, object_type, object_id):
            posted.update(
                name=Path(file_path).name, ns=ns, type=object_type, id=object_id
            )
            return 300

        mod = "omero_annotate_ai.omero.omero_functions"
        listing = [
            {"id": 300, "file_name": "annotation_config_table_7.yaml"},
            {"id": 200, "file_name": "annotation_config_table_7.yaml"},
            {"id": 150, "file_name": "annotation_config_table_8.yaml"},
        ]
        with (
            patch(f"{mod}.ezomero.post_file_annotation", side_effect=post),
            patch(f"{mod}.list_annotations_by_namespace", return_value=listing),
        ):
            new_id = replace_table_config(conn, 7, str(local), "plate", 803)

        assert new_id == 300
        assert posted == {
            "name": "annotation_config_table_7.yaml",
            "ns": CONFIG_NS,
            "type": "Plate",
            "id": 803,
        }
        conn.deleteObjects.assert_called_once_with("Annotation", [200], wait=True)


@pytest.mark.unit
class TestGenerateUniqueTableNameDatetime:
    """Test that generate_unique_table_name uses timezone-aware datetimes."""

    def test_timestamp_in_name_uses_utc(self):
        """Auto-generated table name should use UTC timestamp (no error)."""
        mock_conn = Mock()
        mock_container = Mock()
        mock_container.getName.return_value = "my_dataset"
        mock_conn.getObject.return_value = mock_container

        with patch(
            "omero_annotate_ai.omero.omero_functions.list_annotation_tables",
            return_value=[],
        ):
            name = generate_unique_table_name(mock_conn, "dataset", 1)

        assert "my_dataset" in name
        # Name includes a timestamp in YYYYMMDD_HHMMSS format
        import re

        assert re.search(r"\d{8}_\d{6}", name)

    def test_custom_base_name_no_timestamp(self):
        """When base_name is provided, no timestamp is added."""
        mock_conn = Mock()

        with patch(
            "omero_annotate_ai.omero.omero_functions.list_annotation_tables",
            return_value=[],
        ):
            name = generate_unique_table_name(
                mock_conn, "dataset", 1, base_name="my_table"
            )

        assert name == "my_table"

    def test_versioning_when_name_exists(self):
        """Appends _v1, _v2 etc. when name already exists."""
        mock_conn = Mock()
        existing = [{"name": "my_table"}, {"name": "my_table_v1"}]

        with patch(
            "omero_annotate_ai.omero.omero_functions.list_annotation_tables",
            return_value=existing,
        ):
            name = generate_unique_table_name(
                mock_conn, "dataset", 1, base_name="my_table"
            )

        assert name == "my_table_v2"


@pytest.mark.unit
class TestListUserTablesCreatedField:
    """Test that list_user_tables includes the 'created' field."""

    def test_created_field_populated_when_date_available(self):
        """Tables returned by list_user_tables include a 'created' field."""
        mock_conn = Mock()

        # Mock table data (not None = it's a table)
        mock_df = pd.DataFrame({"col": [1]})

        # Mock file annotation with a date
        mock_file_ann = Mock()
        mock_file_ann.getFile.return_value.getName.return_value = "my_table.h5"
        mock_file_ann.getFile.return_value.getMimetype.return_value = "OMERO.tables"
        mock_file_ann.getDescription.return_value = ""
        mock_file_ann.getNs.return_value = "omero.tables"
        mock_date = Mock()
        mock_date.isoformat.return_value = "2024-01-15T10:00:00"
        mock_file_ann.getDate.return_value = mock_date
        mock_conn.getObject.return_value = mock_file_ann

        with (
            patch(
                "omero_annotate_ai.omero.omero_utils.ezomero.get_file_annotation_ids",
                return_value=[42],
            ),
            patch(
                "omero_annotate_ai.omero.omero_utils.ezomero.get_table",
                return_value=mock_df,
            ),
        ):
            tables = list_user_tables(mock_conn, "dataset", 1)

        assert len(tables) == 1
        assert "created" in tables[0]
        assert tables[0]["created"] == "2024-01-15T10:00:00"

    def test_created_field_empty_when_date_unavailable(self):
        """created field is empty string when file annotation has no date."""
        mock_conn = Mock()
        mock_df = pd.DataFrame({"col": [1]})

        mock_file_ann = Mock()
        mock_file_ann.getFile.return_value.getName.return_value = "my_table.h5"
        mock_file_ann.getFile.return_value.getMimetype.return_value = "OMERO.tables"
        mock_file_ann.getDescription.return_value = ""
        mock_file_ann.getNs.return_value = "omero.tables"
        mock_file_ann.getDate.side_effect = AttributeError("no date")
        mock_conn.getObject.return_value = mock_file_ann

        with (
            patch(
                "omero_annotate_ai.omero.omero_utils.ezomero.get_file_annotation_ids",
                return_value=[42],
            ),
            patch(
                "omero_annotate_ai.omero.omero_utils.ezomero.get_table",
                return_value=mock_df,
            ),
        ):
            tables = list_user_tables(mock_conn, "dataset", 1)

        assert len(tables) == 1
        assert tables[0]["created"] == ""


@pytest.mark.unit
class TestListAnnotationTablesSortsByCreated:
    """Test that list_annotation_tables sorts by the 'created' field."""

    def test_sorts_newest_first(self):
        """Tables are sorted by 'created' descending (newest first)."""
        tables = [
            {"name": "old_table", "created": "2023-01-01T00:00:00"},
            {"name": "new_table", "created": "2024-06-01T00:00:00"},
            {"name": "mid_table", "created": "2023-12-01T00:00:00"},
        ]

        with patch(
            "omero_annotate_ai.omero.omero_functions.list_user_tables",
            return_value=tables,
        ):
            result = list_annotation_tables(Mock(), "dataset", 1)

        assert result[0]["name"] == "new_table"
        assert result[1]["name"] == "mid_table"
        assert result[2]["name"] == "old_table"


# =============================================================================
# Manifest and GeoJSON persistence tests
# =============================================================================


def _make_test_config():
    return AnnotationConfig(
        name="test_workflow",
        study=StudyContext(title="Test", description="Test"),
        dataset=DatasetInfo(source_description="test"),
        annotation_methodology=AnnotationMethodology(annotation_criteria="test"),
        spatial_coverage=SpatialCoverage(channels=[0], timepoints=[0], z_slices=[0]),
        training=TrainingConfig(),
        ai_model=AIModelConfig(),
        workflow=WorkflowConfig(),
        output=OutputConfig(),
        omero=OMEROConfig(container_type="dataset", container_id=1),
        feature_types=[FeatureType(name="cell", color="#FF0000")],
        annotations=[ImageAnnotation(image_id=1, image_name="img1")],
    )


@pytest.mark.unit
class TestManifestPersistence:
    """Tests for JSON manifest save/load/list/delete on OMERO."""

    def test_generate_set_id_is_unique(self):
        ids = {generate_set_id() for _ in range(100)}
        assert len(ids) == 100

    def test_manifest_namespace(self):
        assert manifest_namespace("abc123") == "omero.biomero.manifest.abc123"

    def test_geojson_namespace(self):
        assert geojson_namespace("abc123") == "omero.biomero.annotations.abc123"

    def test_load_manifest_from_omero(self):
        config = _make_test_config()
        json_bytes = config.to_json().encode("utf-8")

        file_ann = MagicMock()
        file_ann.getNs.return_value = manifest_namespace("test_set")
        file_ann.getFileInChunks.return_value = [json_bytes]

        dataset = MagicMock()
        dataset.listAnnotations.return_value = [file_ann]
        conn = MagicMock()
        conn.getObject.return_value = dataset

        loaded = load_manifest_from_omero(conn, "dataset", 1, "test_set")
        assert loaded is not None
        assert loaded.name == "test_workflow"
        assert len(loaded.feature_types) == 1

    def test_load_manifest_not_found(self):
        dataset = MagicMock()
        dataset.listAnnotations.return_value = []
        conn = MagicMock()
        conn.getObject.return_value = dataset

        result = load_manifest_from_omero(conn, "dataset", 1, "nonexistent")
        assert result is None

    def test_list_manifests_from_omero(self):
        config = _make_test_config()
        json_bytes = config.to_json().encode("utf-8")

        file_ann = MagicMock()
        file_ann.getNs.return_value = manifest_namespace("set_001")
        file_ann.getFileInChunks.return_value = [json_bytes]
        file_ann.getId.return_value = 10

        dataset = MagicMock()
        dataset.listAnnotations.return_value = [file_ann]
        conn = MagicMock()
        conn.getObject.return_value = dataset

        manifests = list_manifests_from_omero(conn, "dataset", 1)
        assert len(manifests) == 1
        assert manifests[0]["set_id"] == "set_001"
        assert manifests[0]["name"] == "test_workflow"


@pytest.mark.unit
class TestGeoJSONPersistence:
    """Tests for GeoJSON save/load and patch merging."""

    def _make_geojson(self, patch=None):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
            },
            "properties": {
                "objectType": "annotation",
                "featureType": "cell",
                "plane": {"c": 0, "z": 0, "t": 0},
            },
        }
        if patch:
            feature["properties"]["patch"] = patch
        return {
            "type": "FeatureCollection",
            "channel_presentation": [
                {
                    "channel_index": 0,
                    "visible": True,
                    "contrast_start": 0,
                    "contrast_end": 255,
                    "color": "#FFFFFF",
                }
            ],
            "features": [feature],
        }

    def test_load_geojson_from_omero(self):
        geojson = self._make_geojson()
        json_bytes = json.dumps(geojson).encode("utf-8")

        file_ann = MagicMock()
        file_ann.getNs.return_value = geojson_namespace("set_001")
        file_ann.getFileInChunks.return_value = [json_bytes]

        image = MagicMock()
        image.listAnnotations.return_value = [file_ann]
        conn = MagicMock()
        conn.getObject.return_value = image

        loaded = load_geojson_from_omero(conn, 1, "set_001")
        assert loaded is not None
        assert loaded["type"] == "FeatureCollection"
        assert len(loaded["features"]) == 1

    def test_load_geojson_not_found(self):
        image = MagicMock()
        image.listAnnotations.return_value = []
        conn = MagicMock()
        conn.getObject.return_value = image
        assert load_geojson_from_omero(conn, 1, "nonexistent") is None

    def test_merge_geojson_accumulates_patches(self):
        existing = self._make_geojson(
            patch={"x": 0, "y": 0, "width": 512, "height": 512}
        )
        new_patch = self._make_geojson(
            patch={"x": 512, "y": 0, "width": 512, "height": 512}
        )
        merged = merge_geojson_patches(existing, new_patch)
        assert len(merged["features"]) == 2

    def test_merge_geojson_replaces_same_patch(self):
        existing = self._make_geojson(
            patch={"x": 0, "y": 0, "width": 512, "height": 512}
        )
        updated = self._make_geojson(
            patch={"x": 0, "y": 0, "width": 512, "height": 512}
        )
        updated["features"][0]["properties"]["featureType"] = "nucleus"
        merged = merge_geojson_patches(existing, updated)
        assert len(merged["features"]) == 1
        assert merged["features"][0]["properties"]["featureType"] == "nucleus"

    def test_merge_geojson_updates_channel_presentation(self):
        existing = self._make_geojson()
        new = self._make_geojson()
        new["channel_presentation"][0]["contrast_end"] = 999
        merged = merge_geojson_patches(existing, new)
        assert merged["channel_presentation"][0]["contrast_end"] == 999


@pytest.mark.unit
class TestTypedTableColumns:
    """The tracking table is written with OMERO link columns and fixed widths.

    OMERO.web only renders a table row in an Image's "Tables" panel when the
    table has a column *named* ``Image``, and only links a cell when that
    column is *typed* as ImageColumn/RoiColumn.
    """

    def _make_df(self):
        return pd.DataFrame(
            {
                "image_id": [101, 102],
                "image_name": ["a.tif", "bb.tif"],
                "roi_id": [5012, 0],
                "label_id": [77, 0],
                "channel": [0, 1],
                "processed": [True, False],
                "annotation_type": ["segmentation_mask", "segmentation_mask"],
            }
        )

    def _cols(self):
        from omero_annotate_ai.omero.tracking_table import build_columns

        return {c.name: c for c in build_columns(self._make_df())}

    def test_image_id_becomes_image_column_named_image(self):
        from omero.grid import ImageColumn

        cols = self._cols()
        assert isinstance(cols["Image"], ImageColumn)
        assert cols["Image"].values == [101, 102]
        assert "image_id" not in cols

    def test_roi_id_becomes_roi_column_named_roi(self):
        from omero.grid import RoiColumn

        cols = self._cols()
        assert isinstance(cols["Roi"], RoiColumn)
        assert cols["Roi"].values == [5012, 0]

    def test_remaining_columns_typed_by_dtype(self):
        from omero.grid import BoolColumn, LongColumn, StringColumn

        cols = self._cols()
        assert isinstance(cols["label_id"], LongColumn)
        assert isinstance(cols["channel"], LongColumn)
        assert isinstance(cols["processed"], BoolColumn)
        assert isinstance(cols["image_name"], StringColumn)

    def test_string_columns_get_fixed_generous_widths(self):
        """Widths are fixed up front: later rows may be longer than the first ones."""
        from omero_annotate_ai.omero.tracking_table import STRING_WIDTHS

        cols = self._cols()
        assert cols["image_name"].size == STRING_WIDTHS["image_name"]
        assert STRING_WIDTHS["image_uri"] >= 1024

    def test_dataframe_column_order_is_preserved(self):
        from omero_annotate_ai.omero.tracking_table import build_columns

        names = [c.name for c in build_columns(self._make_df())]
        assert names == [
            "Image",
            "image_name",
            "Roi",
            "label_id",
            "channel",
            "processed",
            "annotation_type",
        ]


@pytest.mark.unit
class TestTrackingTableModule:
    """In-place writes, length checks and the version marker (mocked table)."""

    def _headers(self):
        import omero.grid as g

        return [
            g.StringColumn("annotation_id", "", 256, []),
            g.BoolColumn("processed", "", []),
            g.RoiColumn("Roi", "", []),
        ]

    def _table(self, keys):
        import omero.grid as g

        table = Mock()
        table.getHeaders.side_effect = lambda: self._headers()
        table.getNumberOfRows.return_value = len(keys)
        key_data = Mock()
        key_data.columns = [g.StringColumn("annotation_id", "", 256, list(keys))]
        table.read.return_value = key_data

        def read_coordinates(rows):
            data = Mock()
            data.columns = self._headers()
            return data

        table.readCoordinates.side_effect = read_coordinates
        return table

    def test_updates_existing_rows_and_appends_new_ones(self):
        from omero_annotate_ai.omero import tracking_table as tt

        table = self._table(["a", "b"])
        df = pd.DataFrame(
            {"annotation_id": ["b", "c"], "processed": [True, False], "roi_id": [9, 0]}
        )
        with patch.object(tt, "_open", return_value=table):
            result = tt.write_rows(Mock(), 7, df)

        assert result == {"updated": 1, "added": 1}
        table.readCoordinates.assert_called_once_with([1])
        updated = {c.name: c.values for c in table.update.call_args.args[0].columns}
        assert updated == {"annotation_id": ["b"], "processed": [True], "Roi": [9]}
        added = {c.name: c.values for c in table.addData.call_args.args[0]}
        assert added == {"annotation_id": ["c"], "processed": [False], "Roi": [0]}
        table.close.assert_called_once()

    def test_too_long_string_is_refused_before_writing(self):
        from omero_annotate_ai.omero import tracking_table as tt

        table = self._table([])
        df = pd.DataFrame(
            {"annotation_id": ["x" * 300], "processed": [False], "roi_id": [0]}
        )
        with (
            patch.object(tt, "_open", return_value=table),
            pytest.raises(TableValueTooLongError, match="annotation_id"),
        ):
            tt.write_rows(Mock(), 7, df)
        table.addData.assert_not_called()
        table.close.assert_called_once()

    def test_duplicate_keys_are_refused(self):
        from omero_annotate_ai.omero import tracking_table as tt

        df = pd.DataFrame(
            {"annotation_id": ["a", "a"], "processed": [1, 0], "roi_id": [0, 0]}
        )
        with pytest.raises(ValueError, match="Duplicate"):
            tt.write_rows(Mock(), 7, df)

    def test_tables_without_marker_are_refused(self):
        from omero_annotate_ai.omero import tracking_table as tt

        conn = Mock()
        conn.getObject.return_value.getDescription.return_value = ""
        with pytest.raises(UnsupportedTableError, match="not supported"):
            tt.check_supported(conn, 7)

        conn.getObject.return_value.getDescription.return_value = tt.TABLE_MARKER
        assert tt.check_supported(conn, 7) is conn.getObject.return_value

    def test_tags_and_config_name(self):
        from omero_annotate_ai.omero import tracking_table as tt

        assert tt.tag_for(12, "abc_0_0") == "omero-annotate-ai:12:abc_0_0"
        assert tt.tag_for(123, "x").startswith(tt.tag_prefix(123))
        assert not tt.tag_for(123, "x").startswith(tt.tag_prefix(12))
        assert tt.config_file_name(5) == "annotation_config_table_5.yaml"


@pytest.mark.unit
class TestNormalizeOmeroTableDf:
    """Test the single read-side shim that maps OMERO column names back to internal ones."""

    def test_image_and_roi_are_renamed_back_to_internal_names(self):
        from omero_annotate_ai.omero.omero_functions import _normalize_omero_table_df

        df = pd.DataFrame({"Image": [101], "Roi": [5012], "processed": [True]})

        out = _normalize_omero_table_df(df)

        assert list(out.columns) == ["image_id", "roi_id", "processed"]
        assert out["image_id"].tolist() == [101]
        assert out["roi_id"].tolist() == [5012]

    def test_legacy_table_passes_through_untouched(self):
        """Tables written before this change already use the internal names."""
        from omero_annotate_ai.omero.omero_functions import _normalize_omero_table_df

        df = pd.DataFrame({"image_id": [101], "roi_id": ["None"], "processed": [False]})

        out = _normalize_omero_table_df(df)

        assert out["image_id"].tolist() == [101]
        assert out["roi_id"].tolist() == ["None"]

    def test_non_dataframe_input_is_returned_unchanged(self):
        """ezomero.get_table returns a list-of-lists when pandas is absent."""
        from omero_annotate_ai.omero.omero_functions import _normalize_omero_table_df

        rows = [["Image", "Roi"], [101, 5012]]

        assert _normalize_omero_table_df(rows) == rows


@pytest.mark.unit
class TestPrepareDataframeForOmero:
    """roi_id/label_id must reach OMERO as integers so they can be typed columns."""

    def test_roi_and_label_ids_are_integers_with_zero_sentinel(self):
        from omero_annotate_ai.omero.omero_functions import _prepare_dataframe_for_omero

        df = pd.DataFrame(
            {
                "image_id": [101, 102],
                "roi_id": [5012, None],
                "label_id": [77, None],
            }
        )

        out = _prepare_dataframe_for_omero(df)

        assert out["roi_id"].tolist() == [5012, 0]
        assert out["label_id"].tolist() == [77, 0]
        assert pd.api.types.is_integer_dtype(out["roi_id"])
        assert pd.api.types.is_integer_dtype(out["label_id"])

    def test_legacy_none_string_is_coerced_to_zero(self):
        """to_dataframe used to emit the string "None" for an unset id."""
        from omero_annotate_ai.omero.omero_functions import _prepare_dataframe_for_omero

        df = pd.DataFrame({"image_id": [101], "roi_id": ["None"], "label_id": ["None"]})

        out = _prepare_dataframe_for_omero(df)

        assert out["roi_id"].tolist() == [0]
        assert out["label_id"].tolist() == [0]


@pytest.mark.unit
class TestGetContainerInfoChannels:
    """Channel names reported by get_container_info."""

    @staticmethod
    def _conn(labels_by_image):
        conn = Mock()
        conn.isConnected.return_value = True
        container = Mock()
        container.getName.return_value = "cells"

        def get_object(kind, obj_id):
            if kind != "Image":
                return container
            image = Mock()
            image.getName.return_value = f"img{obj_id}"
            image.getChannelLabels.return_value = labels_by_image[obj_id]
            return image

        conn.getObject.side_effect = get_object
        return conn

    @pytest.mark.parametrize(
        "other, vary", [(["DAPI", "GFP"], False), (["DAPI", "RFP"], True)]
    )
    def test_channel_names_and_variation(self, other, vary):
        from omero_annotate_ai.omero.omero_utils import get_container_info

        labels = {1: ["DAPI", "GFP"], 2: ["DAPI", "GFP"], 3: other}
        with patch(
            "omero_annotate_ai.omero.omero_utils.ezomero.get_image_ids",
            return_value=[1, 2, 3],
        ):
            info = get_container_info(self._conn(labels), "dataset", 5)

        assert info["channel_names"] == ["DAPI", "GFP"]
        assert info["channel_names_vary"] is vary

    def test_spread_sample_includes_last(self):
        from omero_annotate_ai.omero.omero_utils import _spread_sample

        assert _spread_sample(list(range(100)), 4) == [24, 49, 74, 99]
        assert _spread_sample([1, 2], 4) == [1, 2]


@pytest.mark.unit
class TestFindAnnotationConfigForTable:
    """A training set's config is found by table ID only."""

    def test_finds_newest_config_of_this_table(self, tmp_path):
        from omero_annotate_ai.core.annotation_config import create_default_config
        from omero_annotate_ai.omero.omero_functions import (
            find_annotation_config_for_table,
        )

        paths = {}
        for ann_id in (20, 21):
            config = create_default_config()
            config.name = f"run{ann_id}"
            paths[ann_id] = tmp_path / f"{ann_id}.yaml"
            config.save_yaml(paths[ann_id])
        listing = [
            {"id": 20, "file_name": "annotation_config_table_7.yaml"},
            {"id": 21, "file_name": "annotation_config_table_7.yaml"},
            {"id": 99, "file_name": "annotation_config_table_8.yaml"},
        ]
        mod = "omero_annotate_ai.omero.omero_functions"
        with (
            patch(f"{mod}._table_containers", return_value=[("plate", 803)]),
            patch(f"{mod}.list_annotations_by_namespace", return_value=listing),
            patch(
                f"{mod}.ezomero.get_file_annotation",
                side_effect=lambda c, ann_id, folder: str(paths[ann_id]),
            ),
        ):
            config = find_annotation_config_for_table(Mock(), 7)
        assert config.name == "run21"

    def test_none_without_config(self):
        from omero_annotate_ai.omero.omero_functions import (
            find_annotation_config_for_table,
        )

        mod = "omero_annotate_ai.omero.omero_functions"
        with (
            patch(f"{mod}._table_containers", return_value=[("plate", 803)]),
            patch(
                f"{mod}.list_annotations_by_namespace",
                return_value=[{"id": 1, "file_name": "annotation_config_run.yaml"}],
            ),
        ):
            assert find_annotation_config_for_table(Mock(), 7) is None


@pytest.mark.unit
class TestInferSettingsFromTable:
    """Settings recovered from tracking-table rows when no config was saved."""

    def test_patch_table(self):
        from omero_annotate_ai.omero.omero_functions import infer_settings_from_table

        rows = []
        for img, category in [(1, "training"), (2, "training"), (3, "validation")]:
            for patch_index in range(2):
                rows.append(
                    {
                        "image_uri": f"omero:image:{img}",
                        "category": category,
                        "channel": 1,
                        "is_volumetric": False,
                        "is_patch": True,
                        "patch_width": 256,
                        "patch_height": 128,
                        "timepoint": 0,
                        "z_slice": 0,
                        "annotation_id": f"{img}_{patch_index}",
                    }
                )
        settings = infer_settings_from_table(pd.DataFrame(rows))

        assert settings == {
            "channel": 1,
            "three_d": False,
            "use_patches": True,
            "patch_size": [256, 128],
            "patches_per_image": 2,
            "timepoint_mode": "specific",
            "timepoints": [0],
            "z_slice_mode": "specific",
            "z_slices": [0],
            "train_n": 2,
            "validate_n": 1,
            "test_n": 0,
        }

    def test_string_booleans_and_empty_table(self):
        from omero_annotate_ai.omero.omero_functions import infer_settings_from_table

        df = pd.DataFrame(
            [
                {
                    "image_uri": "omero:image:1",
                    "category": "test",
                    "channel": 0,
                    "is_volumetric": "True",
                    "is_patch": "False",
                }
            ]
        )
        settings = infer_settings_from_table(df)
        assert settings["three_d"] is True and settings["use_patches"] is False
        assert "patch_size" not in settings and settings["test_n"] == 1
        assert infer_settings_from_table(pd.DataFrame()) == {}


@pytest.mark.unit
class TestDeleteTrainingSet:
    """delete_training_set removes a table and everything tied to it."""

    TABLE_ID = 100

    def _conn(self, missing=(), rois_on_images=None):
        conn = Mock()

        def get_object(omero_type, object_id):
            if (omero_type, object_id) in missing:
                return None
            return Mock()

        conn.getObject.side_effect = get_object

        def find_by_image(image_id, options, ctx=None):
            result = Mock()
            result.rois = []
            for roi_id, description in (rois_on_images or {}).get(image_id, []):
                roi = Mock()
                roi.getId.return_value.getValue.return_value = roi_id
                roi.getDescription.return_value.getValue.return_value = description
                result.rois.append(roi)
            return result

        conn.getRoiService.return_value.findByImage.side_effect = find_by_image
        return conn

    @staticmethod
    def _rows():
        return pd.DataFrame(
            {
                "annotation_id": ["a", "b", "c"],
                "image_id": [1, 2, 2],
                "roi_id": [11, 12, np.nan],
                "label_id": [21, 22, -1],
                "annotation_input_id": [31, 0, None],
            }
        )

    def _run(self, conn, rows, configs=(), labels_on_images=None, dry_run=True):
        """configs: [(ann_id, file_name)]; labels_on_images: {image_id: [(id, desc)]}."""
        from omero_annotate_ai.omero.omero_functions import (
            CONFIG_NS,
            LABELS_NS,
            delete_training_set,
        )

        def list_ns(conn_, obj_type, obj_id, ns):
            if ns == CONFIG_NS:
                return [{"id": i, "file_name": n} for i, n in configs]
            if ns == LABELS_NS:
                return [
                    {"id": i, "description": d}
                    for i, d in (labels_on_images or {}).get(obj_id, [])
                ]
            return []

        mod = "omero_annotate_ai.omero.omero_functions"
        with (
            patch(f"{mod}._read_any_table", return_value=rows),
            patch(f"{mod}.list_annotations_by_namespace", side_effect=list_ns),
            patch("builtins.print"),
        ):
            return delete_training_set(
                conn, self.TABLE_ID, "plate", 803, dry_run=dry_run
            )

    def test_dry_run_deletes_nothing(self):
        conn = self._conn()
        report = self._run(conn, self._rows())

        conn.deleteObjects.assert_not_called()
        assert report["deleted"] is False
        assert report["rois"] == [11, 12]
        assert report["labels"] == [21, 22]
        assert report["annotation_inputs"] == [31]
        assert report["table"] == [self.TABLE_ID]

    def test_real_run_deletes_children_then_table(self):
        conn = self._conn()
        report = self._run(
            conn,
            self._rows(),
            configs=[
                (41, "annotation_config_table_100.yaml"),
                (42, "annotation_config_table_200.yaml"),
            ],
            dry_run=False,
        )

        calls = conn.deleteObjects.call_args_list
        assert calls[0].args == ("Roi", [11, 12])
        assert calls[1].args == ("Annotation", [21, 22, 31, 41])
        assert calls[-1].args == ("Annotation", [self.TABLE_ID])
        assert report["deleted"] is True
        assert report["configs"] == [41]

    def test_tagged_uploads_missing_from_the_table_are_found(self):
        """A crash between upload and table write leaves tagged, untracked objects."""
        conn = self._conn(
            rois_on_images={
                2: [
                    (13, "omero-annotate-ai:100:c | Training set: run"),
                    (14, "omero-annotate-ai:1000:x"),  # another set (prefix 1000)
                    (15, "something else"),
                ],
            }
        )
        report = self._run(
            conn,
            self._rows(),
            labels_on_images={
                2: [(23, "omero-annotate-ai:100:c"), (24, "omero-annotate-ai:200:c")],
            },
        )
        assert report["rois"] == [11, 12, 13]
        assert report["labels"] == [21, 22, 23]

    def test_missing_objects_are_reported_not_deleted(self):
        conn = self._conn(missing={("Roi", 12), ("FileAnnotation", 22)})
        report = self._run(conn, self._rows(), dry_run=False)

        assert report["rois"] == [11]
        assert report["missing"]["rois"] == [12]
        assert report["missing"]["labels"] == [22]
        assert conn.deleteObjects.call_args_list[0].args == ("Roi", [11])

    def test_missing_table_deletes_nothing(self):
        conn = self._conn(missing={("FileAnnotation", self.TABLE_ID)})
        report = self._run(conn, self._rows(), dry_run=False)
        conn.deleteObjects.assert_not_called()
        assert report["missing"]["table"] == [self.TABLE_ID]

    def test_finds_containers_from_table_links(self):
        from omero_annotate_ai.omero.omero_functions import _table_containers

        conn = Mock()

        def links(container_type, ann_ids):
            if container_type == "Plate":
                return [
                    Mock(**{"getParent.return_value.getId.return_value": pid})
                    for pid in (803, 804)
                ]
            return []

        conn.getAnnotationLinks.side_effect = links
        assert _table_containers(conn, self.TABLE_ID) == [
            ("plate", 803),
            ("plate", 804),
        ]


@pytest.mark.unit
class TestLabelUpload:
    """Label files are uploaded zlib-compressed, losslessly, named after the set."""

    def test_label_file_name(self):
        from omero_annotate_ai.omero.omero_functions import _label_file_name

        assert (
            _label_file_name(
                "/x/seg_00000.tif", "plate run", "omero-annotate-ai:7:42_0_3"
            )
            == "plate_run_42_0_3_mask.tif"
        )
        assert _label_file_name("/x/seg_00000.tif", None, None) == "seg_00000.tif"
        assert _label_file_name("/x/seg_00000.tif", "run", None) == "run_seg_00000.tif"

    def test_upload_is_compressed_lossless_and_tagged(self, tmp_path):
        import tifffile

        mask = np.zeros((40, 50), np.uint16)
        mask[5:15, 5:20] = 1
        mask[20:30, 25:45] = 300
        local = tmp_path / "seg_00000.tif"
        tifffile.imwrite(local, mask)
        uploaded = {}

        def post(conn, file_path, description, ns, object_type, object_id):
            with tifffile.TiffFile(file_path) as tif:
                uploaded.update(
                    name=Path(file_path).name,
                    description=description,
                    compression=tif.pages[0].compression.name,
                    data=tif.asarray(),
                )
            return 55

        mod = "omero_annotate_ai.omero.omero_functions"
        with (
            patch(f"{mod}.ezomero.post_file_annotation", side_effect=post),
            patch(f"{mod}.ezomero.post_roi", return_value=66),
            patch("builtins.print"),
        ):
            label_id, roi_id = upload_rois_and_labels(
                Mock(),
                1,
                str(local),
                trainingset_name="run",
                z_slice=2,
                channel=1,
                timepoint=0,
                tag="omero-annotate-ai:7:1_0_2",
            )

        assert (label_id, roi_id) == (55, 66)
        assert uploaded["name"] == "run_1_0_2_mask.tif"
        assert uploaded["compression"] in ("ADOBE_DEFLATE", "DEFLATE")
        assert np.array_equal(uploaded["data"], mask)
        assert uploaded["description"].startswith("omero-annotate-ai:7:1_0_2 | ")
        assert "t=0" in uploaded["description"] and "z=2" in uploaded["description"]


@pytest.mark.unit
class TestFindTagged:
    """Uploads are found by their exact tag: unit ann1 never adopts ann10's upload."""

    def _conn(self, roi_descriptions, label_descriptions):
        conn = Mock()
        rois = []
        for roi_id, description in roi_descriptions.items():
            roi = Mock()
            roi.getId.return_value.getValue.return_value = roi_id
            roi.getDescription.return_value.getValue.return_value = description
            rois.append(roi)
        conn.getRoiService.return_value.findByImage.return_value.rois = rois
        labels = [
            {"id": label_id, "description": description}
            for label_id, description in label_descriptions.items()
        ]
        return conn, labels

    def test_exact_tag_only(self):
        from omero_annotate_ai.omero import omero_functions

        conn, labels = self._conn(
            {
                1: "omero-annotate-ai:7:ann1 | Training set: x | t=0",
                2: "omero-annotate-ai:7:ann10 | x",
            },
            {3: "omero-annotate-ai:7:ann1 | x", 4: "omero-annotate-ai:8:ann1 | x"},
        )
        with patch.object(
            omero_functions, "list_annotations_by_namespace", return_value=labels
        ):
            assert omero_functions.find_tagged(conn, 5, "omero-annotate-ai:7:ann1") == (
                [3],
                [1],
            )
