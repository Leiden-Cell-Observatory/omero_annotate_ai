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
)


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
