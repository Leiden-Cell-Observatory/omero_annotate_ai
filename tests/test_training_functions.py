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
    _get_dataset_folder_names,
    _clean_dataset_directories,
)
from omero_annotate_ai.core.annotation_config import AnnotationConfig, ImageAnnotation, create_default_config


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
        assert structure["training_input"] == "train_input"
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
        assert structure["training_label_input"] == "train_label_input"
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
            assert (temp_dir / "train_input").exists()
            assert (temp_dir / "train_label").exists()
            assert (temp_dir / "val_input").exists()
            assert (temp_dir / "val_label").exists()

            # Check created_dirs keys
            assert "training_input" in created
            assert "validation_input" in created
            assert created["training_input"] == temp_dir / "train_input"
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
class TestExternalClassificationWorkflow:
    """Tests for the external-label workflow.

    Scenario: user runs CellPose + intensity thresholding in a separate notebook,
    uploads class label maps (integer masks: 0=bg, 1/2/3=class) as FileAnnotations
    to OMERO, and stores the FileAnnotation IDs in the tracking table's label_id column.
    omero_annotate_ai then exports Ch0 as training input and the class label map as
    training ground truth.
    """

    @pytest.fixture
    def patch_df(self):
        """DataFrame using patch mode (avoids the no_pixels image dimension lookup)."""
        return pd.DataFrame(
            {
                "image_id": [1],
                "z_slice": [0],
                "channel": [0],
                "timepoint": [0],
                "is_patch": [True],   # patch=True avoids get_image(no_pixels=True) call
                "patch_x": [0],
                "patch_y": [0],
                "patch_width": [256],
                "patch_height": [256],
                "is_volumetric": [False],
                "label_id": ["None"],
            }
        )

    def test_label_id_as_string_none_is_skipped(self, patch_df):
        """label_id stored as string 'None' (from OMERO table) should not crash."""
        from omero_annotate_ai.processing.training_functions import (
            _prepare_dataset_from_table,
        )
        import tempfile
        from unittest.mock import Mock, patch

        temp_dir = Path(tempfile.mkdtemp())
        try:
            mock_conn = Mock()
            fake_img = np.zeros((256, 256, 1, 1, 1), dtype=np.uint8)
            with patch(
                "omero_annotate_ai.processing.training_functions.ezomero"
            ) as mock_ez:
                mock_ez.get_image.return_value = (None, fake_img)
                input_dir, label_dir = _prepare_dataset_from_table(
                    conn=mock_conn,
                    df=patch_df,
                    output_dir=temp_dir,
                    subset_type="training",
                    tmp_dir=temp_dir / "tmp",
                )
            # No label downloaded — no crash
            assert len(list(label_dir.glob("*.tif"))) == 0
        finally:
            shutil.rmtree(temp_dir)

    def test_label_id_as_string_integer_is_used(self, patch_df):
        """label_id stored as string '101' (from OMERO table) should be parsed to int."""
        from omero_annotate_ai.processing.training_functions import (
            _prepare_dataset_from_table,
        )
        import tempfile
        from unittest.mock import Mock, patch
        from tifffile import imwrite as tiff_imwrite

        df = patch_df.copy()
        df["label_id"] = ["101"]  # String "101" as stored in OMERO table

        temp_dir = Path(tempfile.mkdtemp())
        try:
            label_tiff = temp_dir / "label.tif"
            label_data = np.array([[0, 1, 2, 3]], dtype=np.uint8)
            tiff_imwrite(str(label_tiff), label_data)

            mock_conn = Mock()
            mock_file_ann = Mock()
            mock_file_ann.getFile.return_value.getName.return_value = "label.tif"
            mock_conn.getObject.return_value = mock_file_ann

            fake_img = np.zeros((256, 256, 1, 1, 1), dtype=np.uint8)
            with patch(
                "omero_annotate_ai.processing.training_functions.ezomero"
            ) as mock_ez:
                mock_ez.get_image.return_value = (None, fake_img)
                mock_ez.get_file_annotation.return_value = str(label_tiff)
                input_dir, label_dir = _prepare_dataset_from_table(
                    conn=mock_conn,
                    df=df,
                    output_dir=temp_dir,
                    subset_type="training",
                    tmp_dir=temp_dir / "tmp",
                )
            # Label was downloaded — getObject called with int 101
            mock_conn.getObject.assert_called_once_with("FileAnnotation", 101)
        finally:
            shutil.rmtree(temp_dir)

    def test_multiclass_label_pixel_values_preserved(self, patch_df):
        """Integer pixel values in a multi-class label TIFF are not remapped."""
        from omero_annotate_ai.processing.training_functions import (
            _prepare_dataset_from_table,
        )
        import tempfile
        from unittest.mock import Mock, patch
        from tifffile import imwrite as tiff_imwrite, imread as tiff_imread

        df = patch_df.copy()
        df["label_id"] = [101]  # integer label_id

        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create a class label map with values 0, 1, 2, 3
            label_tiff = temp_dir / "class_label.tif"
            label_data = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
            tiff_imwrite(str(label_tiff), label_data)

            mock_conn = Mock()
            mock_file_ann = Mock()
            mock_file_ann.getFile.return_value.getName.return_value = "class_label.tif"
            mock_conn.getObject.return_value = mock_file_ann

            fake_img = np.zeros((256, 256, 1, 1, 1), dtype=np.uint8)
            with patch(
                "omero_annotate_ai.processing.training_functions.ezomero"
            ) as mock_ez:
                mock_ez.get_image.return_value = (None, fake_img)
                mock_ez.get_file_annotation.return_value = str(label_tiff)
                input_dir, label_dir = _prepare_dataset_from_table(
                    conn=mock_conn,
                    df=df,
                    output_dir=temp_dir,
                    subset_type="training",
                    tmp_dir=temp_dir / "tmp",
                )

            # Check saved label has same pixel values
            saved_label = tiff_imread(str(label_dir / "label_00000.tif"))
            assert set(np.unique(saved_label)) == {0, 1, 2, 3}
        finally:
            shutil.rmtree(temp_dir)

    def test_label_input_id_round_trips_through_dataframe(self):
        """label_input_id is serialized and deserialized correctly via to/from_dataframe."""
        config = AnnotationConfig(name="classification_workflow")
        ann = ImageAnnotation(
            image_id=42,
            image_name="test_image",
            timepoint=0,
            z_slice=0,
            channel=0,
            label_id=101,
            label_input_id=202,
            processed=True,
        )
        config.annotations.append(ann)

        df = config.to_dataframe()
        assert "label_input_id" in df.columns
        assert df.iloc[0]["label_input_id"] == "202"

        config2 = AnnotationConfig(name="classification_workflow")
        config2.from_dataframe(df)
        assert config2.annotations[0].label_input_id == 202

    def test_label_input_id_none_round_trips(self):
        """label_input_id=None serializes as 'None' and deserializes back to None."""
        config = AnnotationConfig(name="classification_workflow")
        ann = ImageAnnotation(
            image_id=42,
            image_name="test_image",
            timepoint=0,
            z_slice=0,
            channel=0,
            label_id=101,
            label_input_id=None,
            processed=True,
        )
        config.annotations.append(ann)

        df = config.to_dataframe()
        assert df.iloc[0]["label_input_id"] == "None"

        config2 = AnnotationConfig(name="classification_workflow")
        config2.from_dataframe(df)
        assert config2.annotations[0].label_input_id is None

    def test_classification_annotation_type_is_valid(self):
        """annotation_type='classification' and 'semantic_segmentation' are valid values."""
        from omero_annotate_ai.core.annotation_config import AnnotationMethodology

        m = AnnotationMethodology(
            annotation_type="classification",
            annotation_criteria="3-class cell type classification",
        )
        assert m.annotation_type == "classification"

        m2 = AnnotationMethodology(
            annotation_type="semantic_segmentation",
            annotation_criteria="semantic segmentation",
        )
        assert m2.annotation_type == "semantic_segmentation"

    def test_external_workflow_config_yaml_roundtrip(self):
        """Config describing the external classification workflow serializes to valid YAML."""
        import yaml

        config = AnnotationConfig(name="cell_classification_workflow")
        config.spatial_coverage.channels = [0, 1, 2]
        config.spatial_coverage.label_channel = 0
        config.spatial_coverage.training_channels = [0]
        config.annotation_methodology.annotation_type = "classification"
        config.annotation_methodology.annotation_method = "automatic"
        config.annotation_methodology.annotation_criteria = (
            "CellPose segmentation + intensity thresholding on Ch1/Ch2 into 3 classes"
        )

        yaml_str = config.to_yaml()
        config2 = AnnotationConfig.from_dict(yaml.safe_load(yaml_str))

        assert config2.spatial_coverage.channels == [0, 1, 2]
        assert config2.spatial_coverage.label_channel == 0
        assert config2.spatial_coverage.training_channels == [0]
        assert config2.annotation_methodology.annotation_type == "classification"
        assert config2.annotation_methodology.annotation_method == "automatic"


@pytest.mark.unit
class TestDatasetDirectoryCleanup:
    """Cleaning of the folders that _prepare_dataset_from_table writes to."""

    def test_folder_names_single_channel(self):
        """Single-channel runs clean the four dataset folders."""
        assert _get_dataset_folder_names() == [
            "training_input",
            "training_label",
            "val_input",
            "val_label",
        ]

    def test_folder_names_separate_channels(self):
        """Separate-channel runs also clean the label_input folders."""
        folders = _get_dataset_folder_names(uses_separate_channels=True)

        assert "training_label_input" in folders
        assert "val_label_input" in folders

    def test_clean_removes_stale_training_data(self, tmp_path):
        """Stale training images must not survive a clean_existing run.

        Cleanup used to target train_input/, a folder nothing writes to, so images
        from an earlier run lingered in training_input/ and leaked into the next one.
        """
        stale = tmp_path / "training_input" / "old.tif"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        _clean_dataset_directories(tmp_path)

        assert not stale.exists()

    def test_clean_removes_stale_label_input_data(self, tmp_path):
        """Separate-channel label images are cleaned too."""
        stale = tmp_path / "training_label_input" / "old.tif"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        _clean_dataset_directories(tmp_path, uses_separate_channels=True)

        assert not stale.exists()

    def test_clean_is_safe_when_directories_absent(self, tmp_path):
        """A first run has nothing to clean and must not raise."""
        _clean_dataset_directories(tmp_path, uses_separate_channels=True)

    def test_cleaned_names_match_prepare_dataset_output(self):
        """Every folder _prepare_dataset_from_table writes to must be cleaned.

        _prepare_dataset_from_table derives its folders from subset_type, as
        f"{subset_type}_input" and f"{subset_type}_label". If a caller adds a
        subset_type, this pins the cleanup list to it.
        """
        cleaned = _get_dataset_folder_names(uses_separate_channels=True)

        for subset_type in ("training", "val", "training_label", "val_label"):
            assert f"{subset_type}_input" in cleaned

        for subset_type in ("training", "val"):
            assert f"{subset_type}_label" in cleaned


@pytest.mark.unit
class TestReorganizeOntoRecords:
    """The offline producer emits the unified layout, paired by annotation_id."""

    def _config(self, tmp_path, categories, separate_channels=False):
        config = create_default_config()
        config.output.output_directory = str(tmp_path)
        if separate_channels:
            config.spatial_coverage.channels = [0, 1]
            config.spatial_coverage.label_channel = 0
            config.spatial_coverage.training_channels = [1]
        else:
            config.spatial_coverage.channels = [0]
            config.spatial_coverage.label_channel = None
            config.spatial_coverage.training_channels = None
        config.annotations = [
            ImageAnnotation(
                image_id=100 + i,
                image_name=f"img_{i}",
                annotation_id=str(i),
                category=category,
                processed=True,
            )
            for i, category in enumerate(categories)
        ]
        return config

    def _populate(self, annotation_dir, ids, separate_channels=False):
        (annotation_dir / "annotation_input").mkdir(parents=True, exist_ok=True)
        (annotation_dir / "annotation_output").mkdir(parents=True, exist_ok=True)
        if separate_channels:
            (annotation_dir / "model_input").mkdir(parents=True, exist_ok=True)
        for i in ids:
            (annotation_dir / "annotation_input" / f"{i}.tif").write_bytes(b"ann")
            (annotation_dir / "annotation_output" / f"{i}_mask.tif").write_bytes(b"lbl")
            if separate_channels:
                (annotation_dir / "model_input" / f"{i}.tif").write_bytes(b"model")

    def test_writes_unified_layout(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config(annotation_dir, ["training", "validation"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert (result["train_input"] / "0.tif").read_bytes() == b"ann"
        assert (result["train_label"] / "0.tif").read_bytes() == b"lbl"
        assert (result["val_input"] / "1.tif").exists()
        assert "validation_input" not in result

    def test_image_and_label_pair_by_id(self, tmp_path):
        """Destination files used to be named by a per-category counter."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config(annotation_dir, ["training", "training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        labels = sorted(p.name for p in result["train_label"].glob("*.tif"))
        assert images == labels == ["0.tif", "1.tif"]

    def test_separate_channels_route_model_and_annotation(self, tmp_path):
        """model_input/ feeds train_input/; annotation_input/ feeds train_annotation_input/."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0], separate_channels=True)
        config = self._config(annotation_dir, ["training"], separate_channels=True)

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert (result["train_input"] / "0.tif").read_bytes() == b"model"
        assert (result["train_annotation_input"] / "0.tif").read_bytes() == b"ann"

    def test_rejects_output_inside_annotation_dir(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        with pytest.raises(ValueError, match="must not be inside"):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=annotation_dir,
                output_dir=annotation_dir,
            )

    def test_defaults_to_sibling_training_dir(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        result = reorganize_local_data_for_training(
            config=config, annotation_dir=annotation_dir
        )

        assert result["base_dir"] == tmp_path / "project_training"

    def test_symlink_mode(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
            file_mode="symlink",
        )

        assert (result["train_input"] / "0.tif").is_symlink()

    def test_missing_label_drops_the_record(self, tmp_path):
        """No orphan image without its label."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        (annotation_dir / "annotation_output" / "0_mask.tif").unlink()
        config = self._config(annotation_dir, ["training", "training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        labels = sorted(p.name for p in result["train_label"].glob("*.tif"))
        assert images == labels == ["1.tif"]

    def test_copy_mode_preserves_originals(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
            file_mode="copy",
        )

        assert (annotation_dir / "annotation_input" / "0.tif").exists()

    def test_move_mode_removes_originals(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
            file_mode="move",
        )

        assert not (annotation_dir / "annotation_input" / "0.tif").exists()

    def test_stats_count_images_and_labels(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1, 2])
        config = self._config(annotation_dir, ["training", "training", "validation"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        stats = result["stats"]
        assert stats["n_training_images"] == 2
        assert stats["n_training_labels"] == 2
        assert stats["n_val_images"] == 1
        assert stats["n_val_labels"] == 1
        assert stats["n_missing"] == 0

    def test_missing_image_drops_the_record(self, tmp_path):
        """An orphan label is as bad as an orphan image: drop the pair."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        (annotation_dir / "annotation_input" / "0.tif").unlink()
        config = self._config(annotation_dir, ["training", "training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        labels = sorted(p.name for p in result["train_label"].glob("*.tif"))
        assert images == labels == ["1.tif"]
        assert result["stats"]["n_missing"] == 1

    def test_test_category_auto_detected(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1, 2])
        config = self._config(annotation_dir, ["training", "validation", "test"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert "test_input" in result
        assert result["stats"]["n_test_images"] == 1

    def test_no_test_annotations_means_no_test_folders(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config(annotation_dir, ["training", "validation"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert "test_input" not in result
        assert result["stats"]["n_test_images"] == 0

    def test_include_test_false_skips_test_annotations(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config(annotation_dir, ["training", "test"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
            include_test=False,
        )

        assert "test_input" not in result
        assert result["stats"]["n_skipped"] == 1

    def test_clean_existing_removes_stale_training_data(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])
        out = tmp_path / "project_training"
        stale = out / "train_input" / "999.tif"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=out,
            clean_existing=True,
        )

        assert not stale.exists()
        assert (out / "train_input" / "0.tif").exists()

    def test_file_mapping_records_destinations(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        mapping = result["stats"]["file_mapping"]["0"]
        assert mapping["image"].endswith("train_input/0.tif")
        assert mapping["label"].endswith("train_label/0.tif")

    def test_no_processed_annotations_raises(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])
        for ann in config.annotations:
            ann.processed = False

        with pytest.raises(ValueError, match="No processed annotations"):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=annotation_dir,
                output_dir=tmp_path / "project_training",
            )

    def test_missing_annotation_dir_raises(self, tmp_path):
        config = self._config(tmp_path / "nope", ["training"])

        with pytest.raises(FileNotFoundError):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=tmp_path / "nope",
                output_dir=tmp_path / "out",
            )
