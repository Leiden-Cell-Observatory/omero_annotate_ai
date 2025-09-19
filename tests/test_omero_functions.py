"""Tests for OMERO integration functions."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from omero_annotate_ai.omero.omero_functions import (
    upload_rois_and_labels,
)


@pytest.mark.unit
class TestROIUpload:
    """Test ROI and label upload functions."""
    
    def test_upload_rois_and_labels_no_ezomero(self):
        """
        Tests that the ROI upload function raises an ImportError if ezomero is not available.
        This test ensures that the function has a proper fallback when a required dependency
        is not installed.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.tif"
            file_path.touch()
            with patch('omero_annotate_ai.omero.omero_functions.ezomero', None):
                with pytest.raises(ImportError, match="ezomero is required"):
                    upload_rois_and_labels(None, 123, str(file_path))
    
    def test_upload_rois_and_labels_missing_dependencies(self):
        """
        Tests the ROI upload function with missing image processing dependencies.
        This test ensures that the function handles the case where the image processing
        dependencies are not available and returns a tuple of (None, None).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.tif"
            file_path.touch()
            with patch('omero_annotate_ai.omero.omero_functions.ezomero', Mock()):
                # Mock missing imports
                with patch.dict('sys.modules', {'omero_annotate_ai.processing.image_functions': None}):
                    result = upload_rois_and_labels(Mock(), 123, str(file_path))
                    assert result == (None, None)
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.imageio')
    @patch('omero_annotate_ai.omero.omero_functions.label_to_rois')
    def test_upload_rois_and_labels_success(self, mock_label_to_rois, mock_imageio, mock_ezomero):
        """
        Tests the successful upload of ROIs and labels.
        This test ensures that the `upload_rois_and_labels` function correctly calls
        the `ezomero` functions to upload the ROIs and labels to OMERO.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.tif"
            file_path.touch()
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
            label_id, roi_id = upload_rois_and_labels(mock_conn, 123, str(file_path))
            
            assert label_id == 789
            assert roi_id == 456
            
            # Verify function calls
            mock_ezomero.post_file_annotation.assert_called_once()
            mock_ezomero.post_roi.assert_called_once_with(mock_conn, 123, mock_shapes, name='Micro-SAM ROIs (vit_b_lm)', description='ROI collection for Micro-SAM segmentation (vit_b_lm)')
            mock_label_to_rois.assert_called_once()
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.imageio')
    def test_upload_rois_and_labels_image_not_found(self, mock_imageio, mock_ezomero):
        """
        Tests the ROI upload when the image is not found in OMERO.
        This test ensures that the function handles the case where the image is not
        found and returns a tuple of (None, None).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.tif"
            file_path.touch()
            mock_conn = Mock()
            mock_conn.getObject.return_value = None  # Image not found
            
            result = upload_rois_and_labels(mock_conn, 999, str(file_path))
            
            assert result == (None, None)
            mock_ezomero.post_file_annotation.assert_not_called()
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.imageio')
    def test_upload_rois_and_labels_file_read_error(self, mock_imageio, mock_ezomero):
        """
        Tests the ROI upload when the label file cannot be read.
        This test ensures that the function handles the case where the label file
        cannot be read and returns a tuple of (None, None).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "nonexistent.tif"
            mock_conn = Mock()
            mock_image = Mock()
            mock_conn.getObject.return_value = mock_image
            
            # Mock file read error
            mock_imageio.imread.side_effect = Exception("File not found")
            
            result = upload_rois_and_labels(mock_conn, 123, str(file_path))
            
            assert result == (None, None)
            mock_label_to_rois.assert_called_once()
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.imageio')
    def test_upload_rois_and_labels_image_not_found(self, mock_imageio, mock_ezomero):
        """
        Tests the ROI upload when the image is not found in OMERO.
        This test ensures that the function handles the case where the image is not
        found and returns a tuple of (None, None).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.tif"
            file_path.touch()
            mock_conn = Mock()
            mock_conn.getObject.return_value = None  # Image not found
            
            result = upload_rois_and_labels(mock_conn, 999, str(file_path))
            
            assert result == (None, None)
            mock_ezomero.post_file_annotation.assert_not_called()
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.imageio')
    def test_upload_rois_and_labels_file_read_error(self, mock_imageio, mock_ezomero):
        """
        Tests the ROI upload when the label file cannot be read.
        This test ensures that the function handles the case where the label file
        cannot be read and returns a tuple of (None, None).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "nonexistent.tif"
            mock_conn = Mock()
            mock_image = Mock()
            mock_conn.getObject.return_value = mock_image
            
            # Mock file read error
            mock_imageio.imread.side_effect = Exception("File not found")
            
            result = upload_rois_and_labels(mock_conn, 123, str(file_path))
            
            assert result == (None, None)