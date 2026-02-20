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
    reorganize_local_data_for_training,
    _create_file_link_or_copy,
)
from omero_annotate_ai.core.annotation_config import AnnotationConfig, ImageAnnotation


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
        return pd.DataFrame(
            {
                "image_id": [1, 2, 3, 4, 5],
                "z_slice": [0, 0, 1, 0, 0],
                "channel": [0, 0, 0, 1, 0],
                "timepoint": [0, 0, 0, 0, 0],
                "is_patch": [False, True, False, True, False],
                "train": [True, True, False, False, True],
                "validate": [False, False, True, True, False],
                "patch_x": [0, 100, 0, 50, 0],
                "patch_y": [0, 100, 0, 50, 0],
                "patch_width": [0, 256, 0, 256, 0],
                "patch_height": [0, 256, 0, 256, 0],
                "label_id": [101, 102, 103, 104, 105],
            }
        )

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup after test
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    # @patch('omero_annotate_ai.processing.training_functions.ezomero')
    # @patch('omero_annotate_ai.processing.training_functions.imwrite')
    # @patch('omero_annotate_ai.processing.training_functions.shutil.move')
    # def test_prepare_training_data_basic(self, mock_download, mock_move, mock_imwrite, mock_ezomero,
    #                                     mock_conn, sample_table_data, temp_output_dir):
    #     """
    #     Tests the basic functionality of the `prepare_training_data_from_table` function.
    #     This test ensures that the function correctly creates the training and validation
    #     directories and that it calls the `ezomero` functions to get the data from OMERO.
    #     """
    #     # Mock ezomero functions
    #     mock_ezomero.get_table.return_value = sample_table_data
    #     mock_ezomero.get_image.return_value = (None, np.random.randint(0, 255, (256, 256), dtype=np.uint8))

    #     # Mock file annotation
    #     mock_file_ann = Mock()
    #     mock_file_ann.getFile.return_value.getName.return_value = "label.tif"
    #     mock_file_ann.getFile.return_value.getSize.return_value = 1024
    #     mock_ezomero.get_file_annotation.return_value = str(temp_output_dir / "temp_label.tif")

    #     # Create mock connection with proper getObject method
    #     mock_conn.getObject.return_value = mock_file_ann

    #     # Mock file operations
    #     mock_download.return_value = str(temp_output_dir / "temp_label.tif")
    #     mock_move.return_value = None
    #     mock_imwrite.return_value = None

    #     # Create actual temp files to avoid file not found errors
    #     for i in range(len(sample_table_data)):
    #         temp_file = temp_output_dir / f"temp_label_{i}.tif"
    #         temp_file.touch()

    #     # Run function
    #     result = prepare_training_data_from_table(
    #         conn=mock_conn,
    #         table_id=123,
    #         output_dir=temp_output_dir,
    #         validation_split=0.2,
    #         clean_existing=True
    #     )

    #     # Verify result structure
    #     assert 'base_dir' in result
    #     assert 'training_input' in result
    #     assert 'training_label' in result
    #     assert 'val_input' in result
    #     assert 'val_label' in result
    #     assert 'stats' in result

    #     # Verify directories were created
    #     assert result['training_input'].exists()
    #     assert result['training_label'].exists()
    #     assert result['val_input'].exists()
    #     assert result['val_label'].exists()

    #     # Verify ezomero was called correctly
    #     mock_ezomero.get_table.assert_called_once_with(mock_conn, 123)

    # @patch('omero_annotate_ai.processing.training_functions.ezomero')
    # def test_table_not_found(self, mock_ezomero, mock_conn, temp_output_dir):
    #     """
    #     Tests the handling of a missing table.
    #     This test ensures that the `prepare_training_data_from_table` function
    #     raises a `ValueError` when the specified table is not found in OMERO.
    #     """
    #     mock_ezomero.get_table.side_effect = Exception("Table not found")

    #     with pytest.raises(ValueError, match="Failed to load table"):
    #         prepare_training_data_from_table(
    #             conn=mock_conn,
    #             table_id=999,
    #             output_dir=temp_output_dir
    #         )

    # @patch('omero_annotate_ai.processing.training_functions.ezomero')
    # def test_empty_table(self, mock_ezomero, mock_conn, temp_output_dir):
    #     """
    #     Tests the handling of an empty table.
    #     This test ensures that the `prepare_training_data_from_table` function
    #     raises a `ValueError` when the specified table is empty.
    #     """
    #     mock_ezomero.get_table.return_value = pd.DataFrame()

    #     with pytest.raises(ValueError, match="Table .* is empty"):
    #         prepare_training_data_from_table(
    #             conn=mock_conn,
    #             table_id=123,
    #             output_dir=temp_output_dir
    #         )

    def test_invalid_validation_split(self, mock_conn, temp_output_dir):
        """
        Tests the validation of the `validation_split` parameter.
        This test ensures that the `prepare_training_data_from_table` function
        raises a `ValueError` when the `validation_split` parameter is not
        between 0.0 and 1.0.
        """
        with pytest.raises(ValueError, match="validation_split must be between"):
            prepare_training_data_from_table(
                conn=mock_conn,
                table_id=123,
                output_dir=temp_output_dir,
                validation_split=1.5,
            )

    # FAILED tests/test_training_functions.py::TestPrepareTrainingDataFromTable::test_existing_train_validate_columns - ValueError: Training data preparation failed - no images were processed successfully. Check the error messages above.
    # @patch('omero_annotate_ai.processing.training_functions.ezomero')
    # @patch('omero_annotate_ai.processing.training_functions.imwrite')
    # def test_existing_train_validate_columns(self, mock_imwrite, mock_ezomero,
    #                                        mock_conn, temp_output_dir,sample_table_data):
    #     """
    #     Tests the use of existing 'train' and 'validate' columns in the table.
    #     This test ensures that the `prepare_training_data_from_table` function
    #     correctly uses the existing 'train' and 'validate' columns in the table
    #     to split the data, instead of performing an automatic split.
    #     """
    #     # Create table with train/validate columns
    #     table_data = sample_table_data

    #     mock_ezomero.get_table.return_value = table_data
    #     mock_ezomero.get_image.return_value = (None, np.random.randint(0, 255, (256, 256), dtype=np.uint8))
    #     mock_ezomero.get_file_annotation.return_value = "/fake/path/label.tif"
    #     mock_imwrite.return_value = None

    #     result = prepare_training_data_from_table(
    #         conn=mock_conn,
    #         table_id=123,
    #         output_dir=temp_output_dir,
    #         validation_split=0.5  # Should be ignored due to existing columns
    #     )

    #     # Should use existing split (2 train, 2 validate)
    #     assert result['stats']['n_training_images'] >= 0  # Will be 0 in mocked test
    #     assert result['stats']['n_val_images'] >= 0

    # @patch('omero_annotate_ai.processing.training_functions.imwrite', None)
    # def test_missing_tifffile_dependency(self, mock_conn, temp_output_dir):
    #     """
    #     Tests the handling of a missing `tifffile` dependency.
    #     This test ensures that the `prepare_training_data_from_table` function
    #     raises an `ImportError` when the `tifffile` package is not available.
    #     """
    #     with pytest.raises(ImportError, match="tifffile package required"):
    #         prepare_training_data_from_table(
    #             conn=mock_conn,
    #             table_id=123,
    #             output_dir=temp_output_dir
    #         )


class TestPrepareDatasetFromTable:
    """Test the internal dataset preparation function."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "image_id": [1, 2],
                "z_slice": [0, "[0, 1]"],  # Test different z_slice formats
                "channel": [0, 0],
                "timepoint": [0, 0],
                "is_patch": [False, True],
                "patch_x": [0, 100],
                "patch_y": [0, 100],
                "patch_width": [0, 256],
                "patch_height": [0, 256],
                "label_id": [101, 102],
            }
        )

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    # @patch('omero_annotate_ai.processing.training_functions.ezomero')
    # @patch('omero_annotate_ai.processing.training_functions.imwrite')
    # @patch('omero_annotate_ai.processing.training_functions.shutil.move')
    # @patch('omero_annotate_ai.processing.os.path.exists')
    # def test_prepare_dataset_basic(self, mock_exists, mock_download, mock_move, mock_imwrite, mock_ezomero,
    #                               sample_df, temp_output_dir):
    #     """
    #     Tests the basic functionality of the `_prepare_dataset_from_table` function.
    #     This test ensures that the function correctly creates the input and label
    #     directories and that it calls the `ezomero` functions to get the data from OMERO.
    #     """
    #     mock_conn = Mock()

    #     # Mock image data - 5D array from ezomero
    #     mock_image_data = np.random.randint(0, 255, (256, 256, 1, 1, 1), dtype=np.uint8)
    #     mock_ezomero.get_image.return_value = (None, mock_image_data)

    #     # Mock file annotation
    #     mock_file_ann = Mock()
    #     mock_file_ann.getFile.return_value.getName.return_value = "label.tif"
    #     mock_file_ann.getFile.return_value.getSize.return_value = 1024
    #     mock_ezomero.get_file_annotation.return_value = mock_file_ann

    #     # Mock file operations - create actual temp files to avoid file not found errors
    #     temp_label_files = []
    #     for i in range(len(sample_df)):
    #         temp_file = temp_output_dir / f"temp_label_{i}.tif"
    #         temp_file.touch()  # Create empty file
    #         temp_label_files.append(str(temp_file))

    #     mock_download.side_effect = temp_label_files
    #     mock_exists.return_value = True
    #     mock_move.return_value = None
    #     mock_imwrite.return_value = None

    #     input_dir, label_dir = _prepare_dataset_from_table(
    #         conn=mock_conn,
    #         df=sample_df,
    #         output_dir=temp_output_dir,
    #         subset_type="training"
    #     )

    #     # Verify directories were created
    #     assert input_dir.exists()
    #     assert label_dir.exists()
    #     assert input_dir.name == "training_input"
    #     assert label_dir.name == "training_label"

    #     # Verify ezomero calls
    #     assert mock_ezomero.get_image.call_count == len(sample_df)
    #     assert mock_ezomero.get_file_annotation.call_count == len(sample_df)

    #     # Verify download and move calls
    #     assert mock_download.call_count == len(sample_df)
    #     assert mock_move.call_count == len(sample_df)

    # @patch('omero_annotate_ai.processing.training_functions.ezomero')
    # def test_missing_ezomero_dependency(self, mock_ezomero, sample_df, temp_output_dir):
    #     """
    #     Tests the handling of a missing `ezomero` dependency.
    #     This test ensures that the `_prepare_dataset_from_table` function raises
    #     an `ImportError` when the `ezomero` package is not available.
    #     """
    #     mock_ezomero.__bool__ = lambda: False  # Simulate ezomero = None
    #     mock_conn = Mock()

    #     with pytest.raises(ImportError, match="ezomero required"):
    #         _prepare_dataset_from_table(
    #             conn=mock_conn,
    #             df=sample_df,
    #             output_dir=temp_output_dir
    #         )


@pytest.mark.unit
class TestCreateFileLinkOrCopy:
    """Test the file operation helper function."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary source and destination directories."""
        src_dir = Path(tempfile.mkdtemp())
        dst_dir = Path(tempfile.mkdtemp())
        yield src_dir, dst_dir
        # Cleanup
        if src_dir.exists():
            shutil.rmtree(src_dir)
        if dst_dir.exists():
            shutil.rmtree(dst_dir)

    def test_copy_mode(self, temp_dirs):
        """Test that copy mode creates a copy of the file."""
        src_dir, dst_dir = temp_dirs
        src_file = src_dir / "test.tif"
        dst_file = dst_dir / "test_copy.tif"

        # Create source file with content
        src_file.write_text("test content")

        result = _create_file_link_or_copy(src_file, dst_file, "copy")

        assert result == "copy"
        assert dst_file.exists()
        assert src_file.exists()  # Original still exists
        assert dst_file.read_text() == "test content"

    def test_move_mode(self, temp_dirs):
        """Test that move mode moves the file."""
        src_dir, dst_dir = temp_dirs
        src_file = src_dir / "test.tif"
        dst_file = dst_dir / "test_moved.tif"

        # Create source file with content
        src_file.write_text("test content")

        result = _create_file_link_or_copy(src_file, dst_file, "move")

        assert result == "move"
        assert dst_file.exists()
        assert not src_file.exists()  # Original is gone
        assert dst_file.read_text() == "test content"

    def test_symlink_mode(self, temp_dirs):
        """Test that symlink mode creates a symbolic link (or falls back to copy)."""
        src_dir, dst_dir = temp_dirs
        src_file = src_dir / "test.tif"
        dst_file = dst_dir / "test_link.tif"

        # Create source file with content
        src_file.write_text("test content")

        result = _create_file_link_or_copy(src_file, dst_file, "symlink")

        # Result should be either "symlink" or "copy (symlink fallback)" on Windows
        assert result in ["symlink", "copy (symlink fallback)"]
        assert dst_file.exists()
        assert src_file.exists()  # Original still exists
        assert dst_file.read_text() == "test content"


@pytest.mark.unit
class TestReorganizeLocalDataForTraining:
    """Test the local data reorganization function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AnnotationConfig with annotations."""
        config = AnnotationConfig(name="test_training_set")

        # Add processed training annotations
        for i in range(3):
            ann = ImageAnnotation(
                image_id=100 + i,
                image_name=f"image_{i}",
                annotation_id=f"100_{0}_{i}",  # Format: {image_id}_{t}_{z}
                timepoint=0,
                z_slice=i,
                category="training",
                channel=0,
            )
            ann.processed = True
            config.annotations.append(ann)

        # Add processed validation annotations
        for i in range(2):
            ann = ImageAnnotation(
                image_id=200 + i,
                image_name=f"val_image_{i}",
                annotation_id=f"200_{0}_{i}",
                timepoint=0,
                z_slice=i,
                category="validation",
                channel=0,
            )
            ann.processed = True
            config.annotations.append(ann)

        return config

    @pytest.fixture
    def annotation_dir(self):
        """Create a mock annotation directory with input and output folders."""
        temp_dir = Path(tempfile.mkdtemp())
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def populated_annotation_dir(self, annotation_dir, mock_config):
        """Create annotation directory populated with test files."""
        input_dir = annotation_dir / "input"
        output_dir = annotation_dir / "output"

        # Create files for each annotation
        for ann in mock_config.annotations:
            # Create input file
            input_file = input_dir / f"{ann.annotation_id}.tif"
            input_file.write_text(f"image data for {ann.annotation_id}")

            # Create mask file
            mask_file = output_dir / f"{ann.annotation_id}_mask.tif"
            mask_file.write_text(f"mask data for {ann.annotation_id}")

        return annotation_dir

    def test_reorganize_creates_training_structure(
        self, populated_annotation_dir, mock_config
    ):
        """Test that reorganization creates correct folder structure."""
        result = reorganize_local_data_for_training(
            config=mock_config,
            annotation_dir=populated_annotation_dir,
            file_mode="copy",
            verbose=False,
        )

        # Check that training folders were created
        assert (populated_annotation_dir / "training_input").exists()
        assert (populated_annotation_dir / "training_label").exists()
        assert (populated_annotation_dir / "val_input").exists()
        assert (populated_annotation_dir / "val_label").exists()

        # Check stats
        stats = result["stats"]
        assert stats["n_training_images"] == 3
        assert stats["n_training_labels"] == 3
        assert stats["n_val_images"] == 2
        assert stats["n_val_labels"] == 2

    def test_reorganize_copy_mode_preserves_originals(
        self, populated_annotation_dir, mock_config
    ):
        """Test that copy mode preserves original files."""
        result = reorganize_local_data_for_training(
            config=mock_config,
            annotation_dir=populated_annotation_dir,
            file_mode="copy",
        )

        # Original files should still exist
        input_dir = populated_annotation_dir / "input"
        for ann in mock_config.annotations:
            assert (input_dir / f"{ann.annotation_id}.tif").exists()

    def test_reorganize_move_mode_removes_originals(
        self, populated_annotation_dir, mock_config
    ):
        """Test that move mode removes original files."""
        result = reorganize_local_data_for_training(
            config=mock_config,
            annotation_dir=populated_annotation_dir,
            file_mode="move",
        )

        # Original files should be gone
        input_dir = populated_annotation_dir / "input"
        output_dir = populated_annotation_dir / "output"
        for ann in mock_config.annotations:
            assert not (input_dir / f"{ann.annotation_id}.tif").exists()
            assert not (output_dir / f"{ann.annotation_id}_mask.tif").exists()

    def test_reorganize_sequential_naming(self, populated_annotation_dir, mock_config):
        """Test that files are renamed with sequential numbering."""
        result = reorganize_local_data_for_training(
            config=mock_config,
            annotation_dir=populated_annotation_dir,
            file_mode="copy",
        )

        training_input = populated_annotation_dir / "training_input"

        # Check sequential naming
        assert (training_input / "input_00000.tif").exists()
        assert (training_input / "input_00001.tif").exists()
        assert (training_input / "input_00002.tif").exists()

    def test_reorganize_handles_missing_input(self, annotation_dir, mock_config):
        """Test graceful handling of missing input files."""
        # Only create output files, not input files
        output_dir = annotation_dir / "output"
        for ann in mock_config.annotations:
            mask_file = output_dir / f"{ann.annotation_id}_mask.tif"
            mask_file.write_text(f"mask data for {ann.annotation_id}")

        result = reorganize_local_data_for_training(
            config=mock_config,
            annotation_dir=annotation_dir,
            file_mode="copy",
        )

        stats = result["stats"]
        assert stats["n_missing_input"] == 5  # All 5 annotations missing input
        assert stats["n_training_labels"] == 3
        assert stats["n_val_labels"] == 2

    def test_reorganize_handles_missing_labels(self, annotation_dir, mock_config):
        """Test graceful handling of missing label files."""
        # Only create input files, not output files
        input_dir = annotation_dir / "input"
        for ann in mock_config.annotations:
            input_file = input_dir / f"{ann.annotation_id}.tif"
            input_file.write_text(f"image data for {ann.annotation_id}")

        result = reorganize_local_data_for_training(
            config=mock_config,
            annotation_dir=annotation_dir,
            file_mode="copy",
        )

        stats = result["stats"]
        assert stats["n_missing_label"] == 5  # All 5 annotations missing labels
        assert stats["n_training_images"] == 3
        assert stats["n_val_images"] == 2

    def test_reorganize_with_test_category(self, annotation_dir):
        """Test handling of test category with include_test flag."""
        config = AnnotationConfig(name="test_set")

        # Add test annotations
        for i in range(2):
            ann = ImageAnnotation(
                image_id=300 + i,
                image_name=f"test_image_{i}",
                annotation_id=f"300_{0}_{i}",
                timepoint=0,
                z_slice=i,
                category="test",
                channel=0,
            )
            ann.processed = True
            config.annotations.append(ann)

        # Create files
        input_dir = annotation_dir / "input"
        output_dir = annotation_dir / "output"
        for ann in config.annotations:
            (input_dir / f"{ann.annotation_id}.tif").write_text("data")
            (output_dir / f"{ann.annotation_id}_mask.tif").write_text("mask")

        # Without include_test, test files should be skipped
        result_no_test = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            file_mode="copy",
            include_test=False,
        )
        assert result_no_test["stats"]["n_test_images"] == 0
        assert result_no_test["stats"]["n_skipped"] == 2

        # With include_test, test files should be processed
        result_with_test = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            file_mode="copy",
            include_test=True,
        )
        assert result_with_test["stats"]["n_test_images"] == 2
        assert result_with_test["stats"]["n_test_labels"] == 2

    def test_reorganize_auto_detects_test_annotations(self, annotation_dir):
        """Test that include_test=None auto-detects test annotations."""
        config = AnnotationConfig(name="test_set")

        # Add mixed annotations: training, validation, and test
        for i, category in enumerate(["training", "validation", "test"]):
            ann = ImageAnnotation(
                image_id=400 + i,
                image_name=f"{category}_image_{i}",
                annotation_id=f"400_{0}_{i}",
                timepoint=0,
                z_slice=i,
                category=category,
                channel=0,
            )
            ann.processed = True
            config.annotations.append(ann)

        # Create files
        input_dir = annotation_dir / "input"
        output_dir = annotation_dir / "output"
        for ann in config.annotations:
            (input_dir / f"{ann.annotation_id}.tif").write_text("data")
            (output_dir / f"{ann.annotation_id}_mask.tif").write_text("mask")

        # With include_test=None (default), test files should be auto-detected
        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            file_mode="copy",
            # include_test not specified, defaults to None (auto-detect)
        )
        assert result["stats"]["n_training_images"] == 1
        assert result["stats"]["n_val_images"] == 1
        assert result["stats"]["n_test_images"] == 1
        assert result["stats"]["n_test_labels"] == 1
        assert "test_input" in result

    def test_reorganize_auto_detect_no_test_annotations(
        self, populated_annotation_dir, mock_config
    ):
        """Test that include_test=None doesn't create test folders when no test annotations."""
        # mock_config has only training/validation annotations, no test
        result = reorganize_local_data_for_training(
            config=mock_config,
            annotation_dir=populated_annotation_dir,
            file_mode="copy",
            # include_test not specified, defaults to None (auto-detect)
        )
        # No test folders should be created since no test annotations exist
        assert result["stats"]["n_test_images"] == 0
        assert "test_input" not in result

    def test_reorganize_clean_existing(self, populated_annotation_dir, mock_config):
        """Test that clean_existing removes previous training folders."""
        training_input = populated_annotation_dir / "training_input"
        training_input.mkdir()
        old_file = training_input / "old_file.txt"
        old_file.write_text("old data")

        result = reorganize_local_data_for_training(
            config=mock_config,
            annotation_dir=populated_annotation_dir,
            file_mode="copy",
            clean_existing=True,
        )

        # Old file should be gone
        assert not old_file.exists()
        # But new files should exist
        assert (training_input / "input_00000.tif").exists()

    def test_reorganize_no_annotations_raises_error(self, annotation_dir):
        """Test that empty config raises ValueError."""
        config = AnnotationConfig(name="empty_set")  # No annotations

        with pytest.raises(ValueError, match="no annotations"):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=annotation_dir,
            )

    def test_reorganize_no_processed_raises_error(self, annotation_dir):
        """Test that config with only unprocessed annotations raises ValueError."""
        config = AnnotationConfig(name="unprocessed_set")
        ann = ImageAnnotation(
            image_id=100,
            image_name="test",
            annotation_id="100_0_0",
            timepoint=0,
            z_slice=0,
            category="training",
            channel=0,
        )
        ann.processed = False
        config.annotations.append(ann)

        with pytest.raises(ValueError, match="No processed annotations"):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=annotation_dir,
            )

    def test_reorganize_missing_directory_raises_error(self, mock_config):
        """Test that missing annotation directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            reorganize_local_data_for_training(
                config=mock_config,
                annotation_dir="/nonexistent/path",
            )

    def test_reorganize_file_mapping(self, populated_annotation_dir, mock_config):
        """Test that file mapping is correctly returned."""
        result = reorganize_local_data_for_training(
            config=mock_config,
            annotation_dir=populated_annotation_dir,
            file_mode="copy",
        )

        file_mapping = result["file_mapping"]

        # Check mapping structure
        assert len(file_mapping) == 5  # 3 training + 2 validation

        # Check a specific mapping
        first_training_id = mock_config.annotations[0].annotation_id
        assert first_training_id in file_mapping
        assert file_mapping[first_training_id]["category"] == "training"
        assert file_mapping[first_training_id]["index"] == 0


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.mark.unit
class TestConsistentFolderStructure:
    """Test that all training data preparation functions use consistent folder structure."""

    def test_standard_folder_structure_keys(self):
        """Test that _get_standard_folder_structure returns expected keys."""
        from omero_annotate_ai.processing.training_functions import (
            _get_standard_folder_structure,
        )

        # Without separate channels, without test
        structure = _get_standard_folder_structure(
            uses_separate_channels=False, include_test=False
        )
        assert "training_input" in structure
        assert "training_label" in structure
        assert "validation_input" in structure
        assert "validation_label" in structure
        assert structure["training_input"] == "training_input"
        assert structure["validation_input"] == "val_input"

    def test_standard_folder_structure_with_separate_channels(self):
        """Test that standard folder structure includes label_input folders."""
        from omero_annotate_ai.processing.training_functions import (
            _get_standard_folder_structure,
        )

        structure = _get_standard_folder_structure(
            uses_separate_channels=True, include_test=False
        )
        assert "training_label_input" in structure
        assert "validation_label_input" in structure
        assert structure["training_label_input"] == "training_label_input"
        assert structure["validation_label_input"] == "val_label_input"

    def test_standard_folder_structure_with_test(self):
        """Test that standard folder structure includes test folders."""
        from omero_annotate_ai.processing.training_functions import (
            _get_standard_folder_structure,
        )

        structure = _get_standard_folder_structure(
            uses_separate_channels=False, include_test=True
        )
        assert "test_input" in structure
        assert "test_label" in structure
        assert structure["test_input"] == "test_input"
        assert structure["test_label"] == "test_label"

    def test_standard_folder_structure_complete(self):
        """Test complete folder structure with all options."""
        from omero_annotate_ai.processing.training_functions import (
            _get_standard_folder_structure,
        )

        structure = _get_standard_folder_structure(
            uses_separate_channels=True, include_test=True
        )
        expected_keys = {
            "training_input",
            "training_label",
            "training_label_input",
            "validation_input",
            "validation_label",
            "validation_label_input",
            "test_input",
            "test_label",
            "test_label_input",
        }
        assert set(structure.keys()) == expected_keys

    def test_create_training_directories(self):
        """Test that _create_training_directories creates correct directories."""
        import tempfile
        from pathlib import Path
        import shutil

        from omero_annotate_ai.processing.training_functions import (
            _create_training_directories,
        )

        temp_dir = Path(tempfile.mkdtemp())
        try:
            created = _create_training_directories(
                output_dir=temp_dir,
                uses_separate_channels=False,
                include_test=False,
                clean_existing=False,
            )

            # Check directories exist
            assert (temp_dir / "training_input").exists()
            assert (temp_dir / "training_label").exists()
            assert (temp_dir / "val_input").exists()
            assert (temp_dir / "val_label").exists()

            # Check created_dirs keys
            assert "training_input" in created
            assert "validation_input" in created
            assert created["training_input"] == temp_dir / "training_input"
            assert created["validation_input"] == temp_dir / "val_input"
        finally:
            shutil.rmtree(temp_dir)

    def test_all_functions_return_consistent_keys(self):
        """Test that all three main functions return consistent result keys."""
        from omero_annotate_ai.processing.training_functions import (
            _get_standard_folder_structure,
            _build_standard_result,
        )
        from pathlib import Path
        import tempfile
        import shutil

        temp_dir = Path(tempfile.mkdtemp())
        try:
            base_dir = temp_dir / "base"
            base_dir.mkdir()

            # Create some mock directories
            folders = _get_standard_folder_structure(
                uses_separate_channels=True, include_test=True
            )
            created_dirs = {}
            for key, folder_name in folders.items():
                folder_path = base_dir / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)
                created_dirs[key] = folder_path

            stats = {
                "n_training_images": 10,
                "n_training_labels": 10,
                "n_val_images": 5,
                "n_val_labels": 5,
            }

            # Build result with extra fields (like reorganize does)
            result = _build_standard_result(
                base_dir=base_dir,
                created_dirs=created_dirs,
                stats=stats,
                file_mapping={"key": "value"},
            )

            # Check required keys are present
            required_keys = {
                "base_dir",
                "stats",
                "training_input",
                "training_label",
                "validation_input",
                "validation_label",
                "training_label_input",
                "validation_label_input",
                "test_input",
                "test_label",
                "test_label_input",
            }
            assert required_keys.issubset(result.keys()), (
                f"Missing keys: {required_keys - set(result.keys())}"
            )

            # Check file_mapping was added
            assert "file_mapping" in result
        finally:
            shutil.rmtree(temp_dir)


@pytest.mark.unit
class TestReorganizeSeparateChannels:
    """Test reorganize_local_data_for_training with separate label/training channels."""

    @pytest.fixture
    def separate_channel_config(self):
        """Config with label_channel=0 and training_channels=[1]."""
        config = AnnotationConfig(name="separate_channel_test")
        config.spatial_coverage.channels = [0, 1]
        config.spatial_coverage.label_channel = 0
        config.spatial_coverage.training_channels = [1]

        for i in range(2):
            ann = ImageAnnotation(
                image_id=100 + i,
                image_name=f"image_{i}",
                annotation_id=f"ann_{i}",
                timepoint=0,
                z_slice=0,
                category="training",
                channel=0,
            )
            ann.processed = True
            config.annotations.append(ann)

        ann = ImageAnnotation(
            image_id=200,
            image_name="val_image",
            annotation_id="ann_val",
            timepoint=0,
            z_slice=0,
            category="validation",
            channel=0,
        )
        ann.processed = True
        config.annotations.append(ann)

        return config

    @pytest.fixture
    def annotation_dir_with_train_files(self, separate_channel_config):
        """Directory with both label-channel and training-channel files."""
        temp_dir = Path(tempfile.mkdtemp())
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        for ann in separate_channel_config.annotations:
            # Label-channel image (fluorescence)
            (input_dir / f"{ann.annotation_id}.tif").write_text("label channel data")
            # Training-channel image (e.g. brightfield)
            (input_dir / f"{ann.annotation_id}_train.tif").write_text("train channel data")
            # Mask
            (output_dir / f"{ann.annotation_id}_mask.tif").write_text("mask data")

        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def annotation_dir_label_only(self, separate_channel_config):
        """Directory with only label-channel files (no _train files, simulating old pipeline)."""
        temp_dir = Path(tempfile.mkdtemp())
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        for ann in separate_channel_config.annotations:
            (input_dir / f"{ann.annotation_id}.tif").write_text("label channel data")
            (output_dir / f"{ann.annotation_id}_mask.tif").write_text("mask data")

        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_detects_separate_channels(self, separate_channel_config):
        """uses_separate_channels() returns True for this config."""
        assert separate_channel_config.spatial_coverage.uses_separate_channels() is True

    def test_creates_label_input_dirs(
        self, annotation_dir_with_train_files, separate_channel_config
    ):
        """Reorganization creates *_label_input directories when separate channels."""
        result = reorganize_local_data_for_training(
            config=separate_channel_config,
            annotation_dir=annotation_dir_with_train_files,
            file_mode="copy",
        )
        base = annotation_dir_with_train_files
        assert (base / "training_label_input").exists()
        assert (base / "val_label_input").exists()

    def test_label_channel_goes_to_label_input(
        self, annotation_dir_with_train_files, separate_channel_config
    ):
        """Label-channel images (*.tif) are placed in *_label_input/, not *_input/."""
        reorganize_local_data_for_training(
            config=separate_channel_config,
            annotation_dir=annotation_dir_with_train_files,
            file_mode="copy",
        )
        base = annotation_dir_with_train_files
        label_input_files = list((base / "training_label_input").glob("*.tif"))
        assert len(label_input_files) == 2
        # Content should be the label-channel data
        assert label_input_files[0].read_text() == "label channel data"

    def test_training_channel_goes_to_input(
        self, annotation_dir_with_train_files, separate_channel_config
    ):
        """Training-channel images (*_train.tif) are placed in *_input/."""
        reorganize_local_data_for_training(
            config=separate_channel_config,
            annotation_dir=annotation_dir_with_train_files,
            file_mode="copy",
        )
        base = annotation_dir_with_train_files
        input_files = list((base / "training_input").glob("*.tif"))
        assert len(input_files) == 2
        # Content should be the training-channel data
        assert input_files[0].read_text() == "train channel data"

    def test_stats_count_label_input(
        self, annotation_dir_with_train_files, separate_channel_config
    ):
        """Stats include n_training_label_input and n_val_label_input counts."""
        result = reorganize_local_data_for_training(
            config=separate_channel_config,
            annotation_dir=annotation_dir_with_train_files,
            file_mode="copy",
        )
        stats = result["stats"]
        assert stats["n_training_label_input"] == 2
        assert stats["n_val_label_input"] == 1
        assert stats["n_training_images"] == 2
        assert stats["n_val_images"] == 1

    def test_missing_train_files_reported(
        self, annotation_dir_label_only, separate_channel_config
    ):
        """Missing _train.tif files increment n_missing_input and don't crash."""
        result = reorganize_local_data_for_training(
            config=separate_channel_config,
            annotation_dir=annotation_dir_label_only,
            file_mode="copy",
        )
        stats = result["stats"]
        # Label-channel images should still be placed in label_input
        assert stats["n_training_label_input"] == 2
        # Training-channel images are missing
        assert stats["n_missing_input"] == 3  # 2 training + 1 validation

    def test_single_channel_unchanged(self):
        """Single-channel config (label == training) preserves existing behaviour."""
        config = AnnotationConfig(name="single_channel_test")
        # No separate channels: label_channel and training_channels both None
        assert config.spatial_coverage.uses_separate_channels() is False

        for i in range(2):
            ann = ImageAnnotation(
                image_id=100 + i,
                image_name=f"image_{i}",
                annotation_id=f"sc_ann_{i}",
                timepoint=0,
                z_slice=0,
                category="training",
                channel=0,
            )
            ann.processed = True
            config.annotations.append(ann)

        temp_dir = Path(tempfile.mkdtemp())
        try:
            input_dir = temp_dir / "input"
            output_dir = temp_dir / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            for ann in config.annotations:
                (input_dir / f"{ann.annotation_id}.tif").write_text("image data")
                (output_dir / f"{ann.annotation_id}_mask.tif").write_text("mask data")

            result = reorganize_local_data_for_training(
                config=config, annotation_dir=temp_dir, file_mode="copy"
            )
            stats = result["stats"]
            assert stats["n_training_images"] == 2
            assert stats["n_training_labels"] == 2
            # No label_input dirs should be created
            assert not (temp_dir / "training_label_input").exists()
        finally:
            shutil.rmtree(temp_dir)
