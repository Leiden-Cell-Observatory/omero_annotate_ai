"""Tests for OMERO integration functions."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from omero_annotate_ai.omero.omero_functions import (
    upload_rois_and_labels,
    link_table_to_containers,
    create_or_replace_tracking_table,
    download_annotation_config_from_omero,
    sync_config_to_omero_table,
    update_workflow_status_map,
    generate_unique_table_name,
    list_annotation_tables,
    CONFIG_NS,
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
            conn=mock_conn,
            table_id=100,
            container_type="dataset",
            container_ids=[2, 3]
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
            container_ids=[1, 2]
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
                container_ids=[1, 2]
            )


@pytest.mark.unit
class TestCreateOrReplaceTrackingTableMultiContainer:
    """Test create_or_replace_tracking_table with multiple containers."""

    def test_create_table_multiple_containers(self):
        """Test creating table attached to multiple containers."""
        mock_conn = Mock()
        mock_update_service = Mock()
        mock_conn.getUpdateService.return_value = mock_update_service

        # Mock FileAnnotation for linking
        mock_file_ann = Mock()
        mock_file_ann._obj = Mock()
        mock_conn.getObject.return_value = mock_file_ann

        # Mock save link
        def mock_save(link):
            mock_saved = Mock()
            mock_saved.getId.return_value.getValue.return_value = 999
            return mock_saved

        mock_update_service.saveAndReturnObject.side_effect = mock_save

        config_df = pd.DataFrame({
            "image_id": [1, 2, 3],
            "processed": [False, False, False]
        })

        with patch(
            'omero_annotate_ai.omero.omero_functions.post_typed_table'
        ) as mock_post_table:
            mock_post_table.return_value = 100  # Table ID

            table_id = create_or_replace_tracking_table(
                conn=mock_conn,
                config_df=config_df,
                table_title="test_table",
                container_type="dataset",
                container_ids=[1, 2, 3]
            )

            assert table_id == 100
            # Should create table on first container
            mock_post_table.assert_called_once()
            call_kwargs = mock_post_table.call_args.kwargs
            assert call_kwargs['object_id'] == 1  # Primary container

            # Should link to additional containers (2 and 3)
            assert mock_update_service.saveAndReturnObject.call_count == 2

    def test_create_table_single_container_backward_compatible(self):
        """Test that single container_id still works."""
        mock_conn = Mock()
        config_df = pd.DataFrame({"image_id": [1]})

        with patch(
            'omero_annotate_ai.omero.omero_functions.post_typed_table'
        ) as mock_post_table:
            mock_post_table.return_value = 100

            table_id = create_or_replace_tracking_table(
                conn=mock_conn,
                config_df=config_df,
                table_title="test_table",
                container_type="dataset",
                container_id=123
            )

            assert table_id == 100
            call_kwargs = mock_post_table.call_args.kwargs
            assert call_kwargs['object_id'] == 123


@pytest.mark.unit
class TestDownloadAnnotationConfigFromOmero:
    """Test download_annotation_config_from_omero function."""

    def test_returns_none_when_no_config_annotation(self):
        """Returns None when no config FileAnnotation is found."""
        mock_conn = Mock()
        with patch(
            "omero_annotate_ai.omero.omero_utils.list_annotations_by_namespace",
            return_value=[],
        ):
            result = download_annotation_config_from_omero(mock_conn, "Dataset", 1)
        assert result is None

    def test_returns_annotation_config_from_yaml(self, tmp_path):
        """Downloads config YAML and parses it into an AnnotationConfig."""
        from omero_annotate_ai.core.annotation_config import AnnotationConfig

        # Write a minimal valid YAML to tmp_path
        yaml_path = tmp_path / "annotation_config_test.yaml"
        config = AnnotationConfig(name="test_workflow")
        config.save_yaml(yaml_path)

        mock_conn = Mock()
        with (
            patch(
                "omero_annotate_ai.omero.omero_functions.list_annotations_by_namespace",
                return_value=[{"id": 42}],
            ),
            patch(
                "omero_annotate_ai.omero.omero_functions.ezomero.get_file_annotation",
                return_value=str(yaml_path),
            ),
        ):
            result = download_annotation_config_from_omero(mock_conn, "Dataset", 1)

        assert result is not None
        assert result.name == "test_workflow"

    def test_returns_none_when_download_fails(self):
        """Returns None when ezomero.get_file_annotation returns None."""
        mock_conn = Mock()
        with (
            patch(
                "omero_annotate_ai.omero.omero_functions.list_annotations_by_namespace",
                return_value=[{"id": 42}],
            ),
            patch(
                "omero_annotate_ai.omero.omero_functions.ezomero.get_file_annotation",
                return_value=None,
            ),
        ):
            result = download_annotation_config_from_omero(mock_conn, "Dataset", 1)
        assert result is None

    def test_uses_last_annotation_when_multiple(self, tmp_path):
        """Uses the last (most recent) config annotation when multiple exist."""
        from omero_annotate_ai.core.annotation_config import AnnotationConfig

        yaml_path = tmp_path / "config.yaml"
        config = AnnotationConfig(name="latest_config")
        config.save_yaml(yaml_path)

        mock_conn = Mock()
        with (
            patch(
                "omero_annotate_ai.omero.omero_functions.list_annotations_by_namespace",
                return_value=[{"id": 10}, {"id": 20}],
            ),
            patch(
                "omero_annotate_ai.omero.omero_functions.ezomero.get_file_annotation",
                return_value=str(yaml_path),
            ) as mock_get,
        ):
            result = download_annotation_config_from_omero(mock_conn, "Dataset", 1)

        # Should use ann_id 20 (last in list)
        mock_get.assert_called_once()
        assert mock_get.call_args[0][1] == 20
        assert result.name == "latest_config"


@pytest.mark.unit
class TestConfigNSConstant:
    """Test that the CONFIG_NS constant is defined and used consistently."""

    def test_config_ns_value(self):
        """CONFIG_NS has the expected namespace string."""
        assert CONFIG_NS == "openmicroscopy.org/omero/annotate/config"

    def test_upload_uses_config_ns(self):
        """upload_annotation_config_to_omero uses the CONFIG_NS constant."""
        mock_conn = Mock()
        with patch(
            "omero_annotate_ai.omero.omero_functions.ezomero.post_file_annotation",
            return_value=42,
        ) as mock_post:
            from omero_annotate_ai.omero.omero_functions import (
                upload_annotation_config_to_omero,
            )

            result = upload_annotation_config_to_omero(
                mock_conn, "Dataset", 1, file_path="/tmp/config.yaml"
            )

        assert result == 42
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["ns"] == CONFIG_NS

    def test_download_uses_config_ns(self):
        """download_annotation_config_from_omero searches with CONFIG_NS."""
        mock_conn = Mock()
        with patch(
            "omero_annotate_ai.omero.omero_functions.list_annotations_by_namespace",
            return_value=[],
        ) as mock_list:
            result = download_annotation_config_from_omero(mock_conn, "Dataset", 1)

        assert result is None
        mock_list.assert_called_once_with(mock_conn, "Dataset", 1, CONFIG_NS)




@pytest.mark.unit
class TestSyncConfigToOmeroTableEdgeCases:
    """Test edge cases in sync_config_to_omero_table."""

    def test_returns_none_when_empty_config(self):
        """Returns None (not -1) when config has no annotations."""
        mock_conn = Mock()
        mock_config = Mock()
        mock_config.to_dataframe.return_value = pd.DataFrame()

        result = sync_config_to_omero_table(
            conn=mock_conn,
            config=mock_config,
            table_title="test",
            container_type="dataset",
            container_id=1,
        )

        assert result is None

    def test_returns_existing_table_id_when_empty_config_and_existing(self):
        """Returns existing_table_id when config is empty and table already exists."""
        mock_conn = Mock()
        mock_config = Mock()
        mock_config.to_dataframe.return_value = pd.DataFrame()

        result = sync_config_to_omero_table(
            conn=mock_conn,
            config=mock_config,
            table_title="test",
            container_type="dataset",
            container_id=1,
            existing_table_id=99,
        )

        assert result == 99


@pytest.mark.unit
class TestUpdateWorkflowStatusDatetime:
    """Test that update_workflow_status_map uses timezone-aware datetimes."""

    def test_last_updated_is_timezone_aware(self):
        """last_updated field in status map should be a timezone-aware ISO string."""
        from datetime import timezone

        mock_conn = Mock()
        mock_df = pd.DataFrame({"processed": [True, False, True]})

        captured_status_map = {}

        def capture_post_map_annotation(*args, **kwargs):
            captured_status_map.update(kwargs.get("kv_dict", {}))
            return 123

        with patch(
            "omero_annotate_ai.omero.omero_functions.ezomero.get_table",
            return_value=mock_df,
        ), patch(
            "omero_annotate_ai.omero.omero_functions.ezomero.get_map_annotation",
            side_effect=Exception("no annotation"),
        ), patch(
            "omero_annotate_ai.omero.omero_functions.ezomero.post_map_annotation",
            side_effect=capture_post_map_annotation,
        ):
            update_workflow_status_map(mock_conn, "dataset", 1, 10)

        assert "last_updated" in captured_status_map
        from datetime import datetime

        dt = datetime.fromisoformat(captured_status_map["last_updated"])
        assert dt.tzinfo is not None, "last_updated must be timezone-aware"


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

        with patch(
            "omero_annotate_ai.omero.omero_utils.ezomero.get_file_annotation_ids",
            return_value=[42],
        ), patch(
            "omero_annotate_ai.omero.omero_utils.ezomero.get_table",
            return_value=mock_df,
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

        with patch(
            "omero_annotate_ai.omero.omero_utils.ezomero.get_file_annotation_ids",
            return_value=[42],
        ), patch(
            "omero_annotate_ai.omero.omero_utils.ezomero.get_table",
            return_value=mock_df,
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

import json
from omero_annotate_ai.omero.omero_functions import (
    generate_set_id, manifest_namespace, geojson_namespace,
    save_manifest_to_omero, load_manifest_from_omero,
    list_manifests_from_omero, delete_manifest_from_omero,
    save_geojson_to_omero, load_geojson_from_omero, merge_geojson_patches,
)
from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig, StudyContext, DatasetInfo, AnnotationMethodology,
    SpatialCoverage, TrainingConfig, AIModelConfig, WorkflowConfig,
    OutputConfig, OMEROConfig, FeatureType, ImageAnnotation, ChannelPresentation,
)


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
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
            "properties": {"objectType": "annotation", "featureType": "cell", "plane": {"c": 0, "z": 0, "t": 0}},
        }
        if patch:
            feature["properties"]["patch"] = patch
        return {
            "type": "FeatureCollection",
            "channel_presentation": [{"channel_index": 0, "visible": True, "contrast_start": 0, "contrast_end": 255, "color": "#FFFFFF"}],
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
        existing = self._make_geojson(patch={"x": 0, "y": 0, "width": 512, "height": 512})
        new_patch = self._make_geojson(patch={"x": 512, "y": 0, "width": 512, "height": 512})
        merged = merge_geojson_patches(existing, new_patch)
        assert len(merged["features"]) == 2

    def test_merge_geojson_replaces_same_patch(self):
        existing = self._make_geojson(patch={"x": 0, "y": 0, "width": 512, "height": 512})
        updated = self._make_geojson(patch={"x": 0, "y": 0, "width": 512, "height": 512})
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
    """Test that the tracking table is written with OMERO semantic link columns.

    OMERO.web only renders a table row in an Image's "Tables" panel when the table
    has a column *named* ``Image``, and only hyperlinks a cell when that column is
    *typed* as ImageColumn/RoiColumn. See _build_omero_columns.
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

    def test_image_id_becomes_image_column_named_image(self):
        """image_id is emitted as an ImageColumn named 'Image'."""
        from omero.grid import ImageColumn
        from omero_annotate_ai.omero.omero_functions import _build_omero_columns

        cols = {c.name: c for c in _build_omero_columns(self._make_df())}

        assert "Image" in cols, "OMERO.web looks up the column by the name 'Image'"
        assert isinstance(cols["Image"], ImageColumn)
        assert cols["Image"].values == [101, 102]
        assert "image_id" not in cols

    def test_roi_id_becomes_roi_column_named_roi(self):
        """roi_id is emitted as a RoiColumn named 'Roi'."""
        from omero.grid import RoiColumn
        from omero_annotate_ai.omero.omero_functions import _build_omero_columns

        cols = {c.name: c for c in _build_omero_columns(self._make_df())}

        assert "Roi" in cols
        assert isinstance(cols["Roi"], RoiColumn)
        assert cols["Roi"].values == [5012, 0]
        assert "roi_id" not in cols

    def test_unannotated_row_carries_roi_sentinel_zero(self):
        """A row that has not been annotated yet carries Roi == 0, not a string."""
        from omero_annotate_ai.omero.omero_functions import _build_omero_columns

        cols = {c.name: c for c in _build_omero_columns(self._make_df())}

        assert cols["Roi"].values[1] == 0

    def test_remaining_columns_typed_by_dtype(self):
        """Non-link columns fall back to the plain dtype-based column types."""
        from omero.grid import BoolColumn, LongColumn, StringColumn
        from omero_annotate_ai.omero.omero_functions import _build_omero_columns

        cols = {c.name: c for c in _build_omero_columns(self._make_df())}

        assert isinstance(cols["label_id"], LongColumn)
        assert isinstance(cols["channel"], LongColumn)
        assert isinstance(cols["processed"], BoolColumn)
        assert isinstance(cols["image_name"], StringColumn)

    def test_string_column_is_sized_to_longest_value(self):
        """StringColumn size must cover the longest value, or OMERO truncates."""
        from omero_annotate_ai.omero.omero_functions import _build_omero_columns

        cols = {c.name: c for c in _build_omero_columns(self._make_df())}

        assert cols["image_name"].size >= len("bb.tif")

    def test_dataframe_column_order_is_preserved(self):
        """Unlike ezomero, which regroups columns by dtype, we keep DataFrame order."""
        from omero_annotate_ai.omero.omero_functions import _build_omero_columns

        names = [c.name for c in _build_omero_columns(self._make_df())]

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
