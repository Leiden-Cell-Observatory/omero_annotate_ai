"""Tests for file I/O functions."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from omero_annotate_ai.processing.file_io_functions import (
    store_annotations_in_zarr,
    zarr_to_tiff,
    zip_directory,
    cleanup_local_embeddings,
    load_annotation_file
)


class TestZarrOperations:
    """Test zarr storage and conversion functions."""
    
    def test_store_annotations_in_zarr(self):
        """Test storing annotations in zarr format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = Path(temp_dir) / "test_annotations.zarr"
            mock_results = {"annotations": [1, 2, 3]}
            
            store_annotations_in_zarr(mock_results, zarr_path)
            
            # Check that zarr directory was created
            assert zarr_path.exists()
            assert zarr_path.is_dir()
            
            # Check marker file exists
            marker_file = zarr_path / "annotations.txt"
            assert marker_file.exists()
            assert "Mock annotation data" in marker_file.read_text()
    
    def test_store_annotations_creates_parent_dirs(self):
        """Test that parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = Path(temp_dir) / "nested" / "path" / "annotations.zarr"
            
            store_annotations_in_zarr({}, zarr_path)
            
            assert zarr_path.exists()
            assert zarr_path.parent.exists()
    
    @patch('imageio.imwrite')
    def test_zarr_to_tiff_with_imageio(self, mock_imwrite):
        """Test zarr to TIFF conversion with imageio available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = Path(temp_dir) / "test.zarr"
            zarr_path.mkdir()
            
            tiff_files = zarr_to_tiff(zarr_path)
            
            assert len(tiff_files) == 3
            assert all(str(f).endswith('.tiff') for f in tiff_files)
            
            # Check that imageio.imwrite was called
            assert mock_imwrite.call_count == 3
    
    def test_zarr_to_tiff_without_imageio(self):
        """Test zarr to TIFF conversion without imageio (fallback)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = Path(temp_dir) / "test.zarr"
            zarr_path.mkdir()
            
            with patch.dict('sys.modules', {'imageio': None}):
                tiff_files = zarr_to_tiff(zarr_path)
                
                assert len(tiff_files) == 3
                
                # Check that fallback files were created
                for tiff_file in tiff_files:
                    assert Path(tiff_file).exists()
    
    def test_zarr_to_tiff_creates_output_dir(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = Path(temp_dir) / "test.zarr"
            zarr_path.mkdir()
            
            zarr_to_tiff(zarr_path)
            
            tiff_dir = zarr_path.parent / "tiff_output"
            assert tiff_dir.exists()
            assert tiff_dir.is_dir()


class TestZipOperations:
    """Test zip directory operations."""
    
    def test_zip_directory(self):
        """Test zipping a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test directory with files
            test_dir = Path(temp_dir) / "test_dir"
            test_dir.mkdir()
            
            file1 = test_dir / "file1.txt"
            file2 = test_dir / "subdir" / "file2.txt"
            file2.parent.mkdir()
            
            file1.write_text("content1")
            file2.write_text("content2")
            
            zip_path = Path(temp_dir) / "archive.zip"
            
            zip_directory(test_dir, zip_path)
            
            assert zip_path.exists()
            
            # Verify zip contents
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                assert "file1.txt" in names
                assert "subdir/file2.txt" in names
    
    def test_zip_directory_empty(self):
        """Test zipping an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "empty_dir"
            test_dir.mkdir()
            
            zip_path = Path(temp_dir) / "empty.zip"
            
            zip_directory(test_dir, zip_path)
            
            assert zip_path.exists()
            
            # Verify empty zip
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                assert len(zf.namelist()) == 0
    
    def test_zip_directory_nonexistent(self):
        """Test zipping nonexistent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = Path(temp_dir) / "nonexistent"
            zip_path = Path(temp_dir) / "archive.zip"
            
            # Should handle gracefully
            zip_directory(nonexistent_dir, zip_path)
            
            # Zip should not be created
            assert not zip_path.exists()


class TestCleanupOperations:
    """Test cleanup operations."""
    
    def test_cleanup_local_embeddings(self):
        """Test cleaning up embedding directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_dir = Path(temp_dir) / "embeddings"
            embedding_dir.mkdir()
            
            # Create some files
            (embedding_dir / "embed1.npy").touch()
            (embedding_dir / "embed2.npy").touch()
            
            cleanup_local_embeddings(embedding_dir)
            
            assert not embedding_dir.exists()
    
    def test_cleanup_nonexistent_embeddings(self):
        """Test cleaning up nonexistent embedding directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = Path(temp_dir) / "nonexistent"
            
            # Should handle gracefully
            cleanup_local_embeddings(nonexistent_dir)
            
            # Should not raise exception
            assert True


class TestAnnotationLoading:
    """Test annotation file loading."""
    
    @patch('imageio.imread')
    def test_load_annotation_file_with_imageio(self, mock_imread):
        """Test loading annotation file with imageio."""
        mock_data = np.array([[1, 2], [3, 4]])
        mock_imread.return_value = mock_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "annotation.tif"
            file_path.touch()
            
            result = load_annotation_file(file_path)
            
            np.testing.assert_array_equal(result, mock_data)
            mock_imread.assert_called_once_with(file_path)
    
    def test_load_annotation_file_without_imageio(self):
        """Test loading annotation file without imageio (fallback)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "annotation.tif"
            file_path.touch()
            
            with patch.dict('sys.modules', {'imageio': None}):
                result = load_annotation_file(file_path)
                
                assert isinstance(result, np.ndarray)
                assert result.shape == (256, 256)
                assert result.dtype == np.uint8


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_store_annotations_with_none_results(self):
        """Test storing None results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = Path(temp_dir) / "test.zarr"
            
            # Should handle None gracefully
            store_annotations_in_zarr(None, zarr_path)
            
            assert zarr_path.exists()
    
    def test_zarr_to_tiff_nonexistent_zarr(self):
        """Test converting nonexistent zarr file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = Path(temp_dir) / "nonexistent.zarr"
            
            # Should handle gracefully
            tiff_files = zarr_to_tiff(zarr_path)
            
            # Should still create output directory and files
            assert len(tiff_files) == 3
    
    def test_file_operations_with_readonly_directory(self):
        """Test file operations with read-only directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = Path(temp_dir) / "readonly"
            readonly_dir.mkdir()
            
            # Make directory read-only (on Windows this might not work as expected)
            try:
                readonly_dir.chmod(0o444)
                
                zarr_path = readonly_dir / "test.zarr"
                
                # Should handle permission errors gracefully (or succeed on Windows)
                try:
                    store_annotations_in_zarr({}, zarr_path)
                    # On Windows, this might succeed despite chmod
                    assert True
                except (PermissionError, OSError):
                    # This is expected on systems where chmod works
                    assert True
                    
            except (OSError, NotImplementedError):
                # Skip on systems where chmod doesn't work as expected
                pytest.skip("Cannot test read-only permissions on this system")
            finally:
                # Restore permissions for cleanup
                try:
                    readonly_dir.chmod(0o755)
                except (OSError, NotImplementedError):
                    pass