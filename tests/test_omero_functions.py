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

        with patch('omero_annotate_ai.omero.omero_functions.ezomero') as mock_ezomero:
            mock_ezomero.post_table.return_value = 100  # Table ID

            table_id = create_or_replace_tracking_table(
                conn=mock_conn,
                config_df=config_df,
                table_title="test_table",
                container_type="dataset",
                container_ids=[1, 2, 3]
            )

            assert table_id == 100
            # Should create table on first container
            mock_ezomero.post_table.assert_called_once()
            call_kwargs = mock_ezomero.post_table.call_args.kwargs
            assert call_kwargs['object_id'] == 1  # Primary container

            # Should link to additional containers (2 and 3)
            assert mock_update_service.saveAndReturnObject.call_count == 2

    def test_create_table_single_container_backward_compatible(self):
        """Test that single container_id still works."""
        mock_conn = Mock()
        config_df = pd.DataFrame({"image_id": [1]})

        with patch('omero_annotate_ai.omero.omero_functions.ezomero') as mock_ezomero:
            mock_ezomero.post_table.return_value = 100

            table_id = create_or_replace_tracking_table(
                conn=mock_conn,
                config_df=config_df,
                table_title="test_table",
                container_type="dataset",
                container_id=123
            )

            assert table_id == 100
            call_kwargs = mock_ezomero.post_table.call_args.kwargs
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
