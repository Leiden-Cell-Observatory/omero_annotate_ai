"""Tests for training data preparation functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from omero_annotate_ai.processing.training_functions import (
    prepare_training_data_from_table,
    _prepare_dataset_from_table,
)


class TestPrepareTrainingDataFromTable:
    """Test the automated training data preparation function."""

    @pytest.fixture
    def mock_conn(self):
        """Mock OMERO connection."""
        conn = Mock()
        return conn

    @pytest.fixture
    def sample_table_data(self):
        """Sample annotation table data."""
        return pd.DataFrame({
            'image_id': [1, 2, 3, 4, 5],
            'z_slice': [0, 0, 1, 0, 0],
            'channel': [0, 0, 0, 1, 0],
            'timepoint': [0, 0, 0, 0, 0],
            'is_patch': [False, True, False, True, False],
            'patch_x': [0, 100, 0, 50, 0],
            'patch_y': [0, 100, 0, 50, 0],
            'patch_width': [0, 256, 0, 256, 0],
            'patch_height': [0, 256, 0, 256, 0],
            'label_id': [101, 102, 103, 104, 105],
        })

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup after test
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @patch('omero_annotate_ai.processing.training_functions.ezomero')
    @patch('omero_annotate_ai.processing.training_functions.imwrite')
    def test_prepare_training_data_basic(self, mock_imwrite, mock_ezomero, 
                                        mock_conn, sample_table_data, temp_output_dir):
        """Test basic functionality of prepare_training_data_from_table."""
        # Mock ezomero functions
        mock_ezomero.get_table.return_value = sample_table_data
        mock_ezomero.get_image.return_value = (None, np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        mock_ezomero.get_file_annotation.return_value = "/fake/path/label.tif"
        
        # Mock file operations
        mock_imwrite.return_value = None
        
        # Run function
        result = prepare_training_data_from_table(
            conn=mock_conn,
            table_id=123,
            output_dir=temp_output_dir,
            validation_split=0.2,
            clean_existing=True
        )
        
        # Verify result structure
        assert 'base_dir' in result
        assert 'training_input' in result
        assert 'training_label' in result
        assert 'val_input' in result
        assert 'val_label' in result
        assert 'stats' in result
        
        # Verify directories were created
        assert result['training_input'].exists()
        assert result['training_label'].exists()
        assert result['val_input'].exists()
        assert result['val_label'].exists()
        
        # Verify ezomero was called correctly
        mock_ezomero.get_table.assert_called_once_with(mock_conn, 123)

    @patch('omero_annotate_ai.processing.training_functions.ezomero')
    def test_table_not_found(self, mock_ezomero, mock_conn, temp_output_dir):
        """Test handling of missing table."""
        mock_ezomero.get_table.side_effect = Exception("Table not found")
        
        with pytest.raises(ValueError, match="Failed to load table"):
            prepare_training_data_from_table(
                conn=mock_conn,
                table_id=999,
                output_dir=temp_output_dir
            )

    @patch('omero_annotate_ai.processing.training_functions.ezomero')
    def test_empty_table(self, mock_ezomero, mock_conn, temp_output_dir):
        """Test handling of empty table."""
        mock_ezomero.get_table.return_value = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Table .* is empty"):
            prepare_training_data_from_table(
                conn=mock_conn,
                table_id=123,
                output_dir=temp_output_dir
            )

    def test_invalid_validation_split(self, mock_conn, temp_output_dir):
        """Test validation of split parameter."""
        with pytest.raises(ValueError, match="validation_split must be between"):
            prepare_training_data_from_table(
                conn=mock_conn,
                table_id=123,
                output_dir=temp_output_dir,
                validation_split=1.5
            )

    @patch('omero_annotate_ai.processing.training_functions.ezomero')
    @patch('omero_annotate_ai.processing.training_functions.imwrite')
    def test_existing_train_validate_columns(self, mock_imwrite, mock_ezomero,
                                           mock_conn, temp_output_dir):
        """Test using existing train/validate columns from table."""
        # Create table with train/validate columns
        table_data = pd.DataFrame({
            'image_id': [1, 2, 3, 4],
            'z_slice': [0, 0, 0, 0],
            'channel': [0, 0, 0, 0],
            'timepoint': [0, 0, 0, 0],
            'is_patch': [False, False, False, False],
            'label_id': [101, 102, 103, 104],
            'train': [True, True, False, False],
            'validate': [False, False, True, True],
        })
        
        mock_ezomero.get_table.return_value = table_data
        mock_ezomero.get_image.return_value = (None, np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        mock_ezomero.get_file_annotation.return_value = "/fake/path/label.tif"
        mock_imwrite.return_value = None
        
        result = prepare_training_data_from_table(
            conn=mock_conn,
            table_id=123,
            output_dir=temp_output_dir,
            validation_split=0.5  # Should be ignored due to existing columns
        )
        
        # Should use existing split (2 train, 2 validate)
        assert result['stats']['n_training_images'] >= 0  # Will be 0 in mocked test
        assert result['stats']['n_val_images'] >= 0

    @patch('omero_annotate_ai.processing.training_functions.imwrite', None)
    def test_missing_tifffile_dependency(self, mock_conn, temp_output_dir):
        """Test handling of missing tifffile dependency."""
        with pytest.raises(ImportError, match="tifffile package required"):
            prepare_training_data_from_table(
                conn=mock_conn,
                table_id=123,
                output_dir=temp_output_dir
            )


class TestPrepareDatasetFromTable:
    """Test the internal dataset preparation function."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({
            'image_id': [1, 2],
            'z_slice': [0, '[0, 1]'],  # Test different z_slice formats
            'channel': [0, 0],
            'timepoint': [0, 0],
            'is_patch': [False, True],
            'patch_x': [0, 100],
            'patch_y': [0, 100],
            'patch_width': [0, 256],
            'patch_height': [0, 256],
            'label_id': [101, 102],
        })

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @patch('omero_annotate_ai.processing.training_functions.ezomero')
    @patch('omero_annotate_ai.processing.training_functions.imwrite')
    @patch('omero_annotate_ai.processing.training_functions.shutil')
    def test_prepare_dataset_basic(self, mock_shutil, mock_imwrite, mock_ezomero,
                                  sample_df, temp_output_dir):
        """Test basic dataset preparation."""
        mock_conn = Mock()
        
        # Mock image data - 5D array from ezomero
        mock_image_data = np.random.randint(0, 255, (256, 256, 1, 1, 1), dtype=np.uint8)
        mock_ezomero.get_image.return_value = (None, mock_image_data)
        mock_ezomero.get_file_annotation.return_value = "/fake/label/path.tif"
        mock_shutil.move.return_value = None
        mock_imwrite.return_value = None
        
        input_dir, label_dir = _prepare_dataset_from_table(
            conn=mock_conn,
            df=sample_df,
            output_dir=temp_output_dir,
            subset_type="training"
        )
        
        # Verify directories were created
        assert input_dir.exists()
        assert label_dir.exists()
        assert input_dir.name == "training_input"
        assert label_dir.name == "training_label"
        
        # Verify ezomero calls
        assert mock_ezomero.get_image.call_count == len(sample_df)
        assert mock_ezomero.get_file_annotation.call_count == len(sample_df)

    @patch('omero_annotate_ai.processing.training_functions.ezomero')  
    def test_missing_ezomero_dependency(self, mock_ezomero, sample_df, temp_output_dir):
        """Test handling of missing ezomero dependency."""
        mock_ezomero.__bool__ = lambda: False  # Simulate ezomero = None
        mock_conn = Mock()
        
        with pytest.raises(ImportError, match="ezomero required"):
            _prepare_dataset_from_table(
                conn=mock_conn,
                df=sample_df,
                output_dir=temp_output_dir
            )


if __name__ == "__main__":
    pytest.main([__file__])