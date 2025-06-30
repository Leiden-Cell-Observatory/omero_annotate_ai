"""Tests for OMERO integration functions."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from omero_annotate_ai.omero.omero_functions import (
    initialize_tracking_table,
    get_unprocessed_units,
    update_tracking_table_rows,
    upload_rois_and_labels,
    get_dask_image_multiple
)


class TestTrackingTable:
    """Test tracking table management functions."""
    
    def test_initialize_tracking_table_no_ezomero(self):
        """Test table initialization when ezomero is not available."""
        with patch('omero_annotate_ai.omero.omero_functions.ezomero', None):
            with pytest.raises(ImportError, match="ezomero is required"):
                initialize_tracking_table(
                    None, "test_table", [], "dataset", 1, "test"
                )
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    def test_initialize_tracking_table_success(self, mock_ezomero):
        """Test successful table initialization."""
        # Mock OMERO connection and objects
        mock_conn = Mock()
        mock_image = Mock()
        mock_image.getId.return_value = 123
        mock_image.getName.return_value = "test_image.tif"
        mock_image.getSizeZ.return_value = 1
        mock_image.getSizeT.return_value = 1
        mock_conn.getObject.return_value = mock_image
        
        # Mock ezomero.post_table
        mock_ezomero.post_table.return_value = 456
        
        # Test data
        processing_units = [
            (123, 0, {"category": "training", "model_type": "vit_b_lm"}),
            (124, 1, {"category": "validation", "model_type": "vit_b_lm"})
        ]
        
        table_id = initialize_tracking_table(
            mock_conn, "test_table", processing_units, "dataset", 1, "test"
        )
        
        assert table_id == 456
        mock_ezomero.post_table.assert_called_once()
        
        # Check that DataFrame was created with correct structure
        call_args = mock_ezomero.post_table.call_args
        df = call_args[0][1]  # Second argument is the DataFrame
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "image_id" in df.columns
        assert "processed" in df.columns
        assert "sam_model" in df.columns
    
    def test_get_unprocessed_units_no_ezomero(self):
        """Test getting unprocessed units when ezomero is not available."""
        with patch('omero_annotate_ai.omero.omero_functions.ezomero', None):
            with pytest.raises(ImportError, match="ezomero is required"):
                get_unprocessed_units(None, 123)
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    def test_get_unprocessed_units_success(self, mock_ezomero):
        """Test getting unprocessed units successfully."""
        # Mock table data
        mock_df = pd.DataFrame({
            'image_id': [1, 2, 3, 4],
            'processed': [False, True, False, False],
            'train': [True, True, False, False],
            'metadata': ['{}', '{}', '{}', '{}']
        })
        mock_ezomero.get_table.return_value = mock_df
        
        units = get_unprocessed_units(Mock(), 123)
        
        # Should return 3 unprocessed units (rows 0, 2, 3)
        assert len(units) == 3
        assert units[0][0] == 1  # image_id from row 0
        assert units[1][0] == 3  # image_id from row 2
        assert units[2][0] == 4  # image_id from row 3
    
    def test_update_tracking_table_rows_no_ezomero(self):
        """Test updating table rows when ezomero is not available."""
        with patch('omero_annotate_ai.omero.omero_functions.ezomero', None):
            with pytest.raises(ImportError, match="ezomero is required"):
                update_tracking_table_rows(None, 123, [0, 1], "completed", "test.tif")
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    def test_update_tracking_table_rows_success(self, mock_ezomero):
        """Test successful table row updates using direct OMERO API."""
        # Mock connection and table objects
        mock_conn = Mock()
        mock_file_ann = Mock()
        mock_original_file = Mock()
        mock_table = Mock()
        
        mock_conn.getObject.return_value = mock_file_ann
        mock_file_ann.getFile.return_value = mock_original_file
        mock_conn.c.sf.sharedResources.return_value.openTable.return_value = mock_table
        
        # Mock table operations
        mock_table.getNumberOfRows.return_value = 5
        mock_table.getNumberOfColumns.return_value = 3
        mock_table.getHeaders.return_value = [
            Mock(name="image_id"),
            Mock(name="processed"),
            Mock(name="other_col")
        ]
        
        # Mock read data
        mock_data = Mock()
        mock_data.columns = [Mock(), Mock(), Mock()]
        mock_data.columns[1].values = [False]  # processed column
        mock_table.read.return_value = mock_data
        
        # Mock ezomero table reading for initial check
        mock_df = pd.DataFrame({'processed': [False, False, True]})
        mock_ezomero.get_table.return_value = mock_df
        
        # Test the function
        update_tracking_table_rows(mock_conn, 123, [0, 1], "completed", "test.tif")
        
        # Verify table operations were called
        mock_conn.getObject.assert_called_with("FileAnnotation", 123)
        mock_table.close.assert_called_once()


class TestROIUpload:
    """Test ROI and label upload functions."""
    
    def test_upload_rois_and_labels_no_ezomero(self):
        """Test ROI upload when ezomero is not available."""
        with patch('omero_annotate_ai.omero.omero_functions.ezomero', None):
            with pytest.raises(ImportError, match="ezomero is required"):
                upload_rois_and_labels(None, 123, "test.tif")
    
    def test_upload_rois_and_labels_missing_dependencies(self):
        """Test ROI upload with missing image processing dependencies."""
        with patch('omero_annotate_ai.omero.omero_functions.ezomero', Mock()):
            # Mock missing imports
            with patch.dict('sys.modules', {'omero_annotate_ai.processing.image_functions': None}):
                result = upload_rois_and_labels(Mock(), 123, "test.tif")
                assert result == (None, None)
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.imageio')
    @patch('omero_annotate_ai.omero.omero_functions.label_to_rois')
    def test_upload_rois_and_labels_success(self, mock_label_to_rois, mock_imageio, mock_ezomero):
        """Test successful ROI and label upload."""
        # Mock connection and image
        mock_conn = Mock()
        mock_image = Mock()
        mock_image.getId.return_value = 123
        mock_conn.getObject.return_value = mock_image
        
        # Mock file reading
        label_img = np.array([[[1, 1], [2, 2]]])
        mock_imageio.imread.return_value = label_img
        
        # Mock ROI creation
        mock_shapes = [Mock(), Mock()]
        mock_label_to_rois.return_value = mock_shapes
        
        # Mock ezomero functions
        mock_ezomero.post_file_annotation.return_value = 789
        mock_ezomero.post_roi.return_value = 456
        
        # Test the function
        label_id, roi_id = upload_rois_and_labels(mock_conn, 123, "test.tif")
        
        assert label_id == 789
        assert roi_id == 456
        
        # Verify function calls
        mock_ezomero.post_file_annotation.assert_called_once()
        mock_ezomero.post_roi.assert_called_once_with(mock_conn, 123, mock_shapes)
        mock_label_to_rois.assert_called_once()
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.imageio')
    def test_upload_rois_and_labels_image_not_found(self, mock_imageio, mock_ezomero):
        """Test ROI upload when image is not found."""
        mock_conn = Mock()
        mock_conn.getObject.return_value = None  # Image not found
        
        result = upload_rois_and_labels(mock_conn, 999, "test.tif")
        
        assert result == (None, None)
        mock_imageio.imread.assert_not_called()
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.imageio')
    def test_upload_rois_and_labels_file_read_error(self, mock_imageio, mock_ezomero):
        """Test ROI upload when label file cannot be read."""
        mock_conn = Mock()
        mock_image = Mock()
        mock_conn.getObject.return_value = mock_image
        
        # Mock file read error
        mock_imageio.imread.side_effect = Exception("File not found")
        
        result = upload_rois_and_labels(mock_conn, 123, "nonexistent.tif")
        
        assert result == (None, None)


class TestImageLoading:
    """Test image loading functions."""
    
    def test_get_dask_image_multiple_no_dask(self):
        """Test image loading when dask is not available."""
        with patch.dict('sys.modules', {'dask': None, 'dask.array': None}):
            # Should fall back to direct loading
            with patch('omero_annotate_ai.omero.omero_functions._load_images_direct') as mock_direct:
                mock_direct.return_value = [np.array([1, 2, 3])]
                
                result = get_dask_image_multiple(Mock(), [Mock()], [0], [0], [0])
                
                mock_direct.assert_called_once()
                assert len(result) == 1
    
    @patch('omero_annotate_ai.omero.omero_functions.da')
    @patch('omero_annotate_ai.omero.omero_functions.delayed')
    @patch('omero_annotate_ai.omero.omero_functions._create_dask_array_for_image')
    def test_get_dask_image_multiple_success(self, mock_create_dask, mock_delayed, mock_da):
        """Test successful dask image loading."""
        # Mock image objects
        mock_images = [Mock(), Mock()]
        for i, img in enumerate(mock_images):
            img.getId.return_value = i
            img.getSizeY.return_value = 100
            img.getSizeX.return_value = 100
        
        # Mock dask arrays
        mock_dask_arrays = [Mock(), Mock()]
        for i, arr in enumerate(mock_dask_arrays):
            arr.compute.return_value = np.random.rand(10, 100, 100)
        
        mock_create_dask.side_effect = mock_dask_arrays
        
        result = get_dask_image_multiple(Mock(), mock_images, [0], [0], [0])
        
        assert len(result) == 2
        assert all(isinstance(arr, np.ndarray) for arr in result)
        
        # Verify dask array creation was called
        assert mock_create_dask.call_count == 2
    
    def test_get_dask_image_multiple_empty_list(self):
        """Test image loading with empty image list."""
        result = get_dask_image_multiple(Mock(), [], [0], [0], [0])
        assert result == []
    
    @patch('omero_annotate_ai.omero.omero_functions._create_dask_array_for_image')
    def test_get_dask_image_multiple_with_failures(self, mock_create_dask):
        """Test image loading with some failures."""
        # Mock images
        mock_images = [Mock(), Mock(), Mock()]
        for i, img in enumerate(mock_images):
            img.getId.return_value = i
            img.getSizeY.return_value = 100
            img.getSizeX.return_value = 100
        
        # First image succeeds, second fails, third succeeds
        mock_dask_array = Mock()
        mock_dask_array.compute.return_value = np.random.rand(10, 100, 100)
        
        def side_effect(*args):
            if args[1].getId() == 1:  # Second image fails
                raise Exception("Failed to create dask array")
            return mock_dask_array
        
        mock_create_dask.side_effect = side_effect
        
        with patch('omero_annotate_ai.omero.omero_functions.da'):
            result = get_dask_image_multiple(Mock(), mock_images, [0], [0], [0])
            
            # Should return arrays for successful images plus fallback for failed one
            assert len(result) == 3



class TestMockBehavior:
    """Test behavior when mocking OMERO connections."""
    
    def test_functions_with_none_connection(self):
        """Test that functions handle None connections gracefully."""
        # These should raise ImportError when ezomero is None, not crash on None connection
        with patch('omero_annotate_ai.omero.omero_functions.ezomero', None):
            with pytest.raises(ImportError):
                initialize_tracking_table(None, "test", [], "dataset", 1, "desc")
            
            with pytest.raises(ImportError):
                get_unprocessed_units(None, 123)
            
            with pytest.raises(ImportError):
                update_tracking_table_rows(None, 123, [0], "completed", "file.tif")
            
            with pytest.raises(ImportError):
                upload_rois_and_labels(None, 123, "test.tif")
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    def test_functions_with_mock_connection(self, mock_ezomero):
        """Test that functions work with properly mocked connections."""
        mock_conn = Mock()
        
        # Mock table operations
        mock_ezomero.post_table.return_value = 123
        mock_ezomero.get_table.return_value = pd.DataFrame({'processed': [False]})
        
        # Should not raise exceptions with proper mocks
        table_id = initialize_tracking_table(mock_conn, "test", [], "dataset", 1, "desc")
        assert table_id == 123
        
        units = get_unprocessed_units(mock_conn, 123)
        assert isinstance(units, list)