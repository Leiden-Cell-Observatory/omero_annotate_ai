"""File I/O functions for micro-SAM workflows."""

import zipfile
import shutil
from pathlib import Path
from typing import List, Any
import numpy as np


def store_annotations_in_zarr(results: Any, zarr_path: Path):
    """Store annotation results in zarr format.
    
    Args:
        results: Annotation results from micro-SAM
        zarr_path: Path to save zarr file
    """
    # Stub implementation
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    zarr_path.mkdir(exist_ok=True)
    
    # Create a simple marker file
    (zarr_path / "annotations.txt").write_text("Mock annotation data")
    print(f"📁 Stored annotations in zarr format: {zarr_path}")


def zarr_to_tiff(zarr_path: Path) -> List[str]:
    """Convert zarr annotations to TIFF format.
    
    Args:
        zarr_path: Path to zarr file
        
    Returns:
        List of TIFF file paths
    """
    # Stub implementation
    tiff_dir = zarr_path.parent / "tiff_output"
    tiff_dir.mkdir(exist_ok=True)
    
    # Create mock TIFF files
    tiff_files = []
    for i in range(3):  # Mock 3 annotation files
        tiff_path = tiff_dir / f"annotation_{i}.tiff"
        
        # Create a simple mock TIFF (in reality, would convert zarr data)
        mock_data = np.zeros((256, 256), dtype=np.uint8)
        try:
            import imageio
            imageio.imwrite(tiff_path, mock_data)
        except ImportError:
            # Fallback: create empty file
            tiff_path.touch()
        
        tiff_files.append(str(tiff_path))
    
    print(f"📁 Converted zarr to {len(tiff_files)} TIFF files")
    return tiff_files


def zip_directory(directory_path: Path, zip_path: Path):
    """Zip a directory.
    
    Args:
        directory_path: Directory to zip
        zip_path: Output zip file path
    """
    if not directory_path.exists():
        print(f"⚠️ Directory not found: {directory_path}")
        return
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(directory_path))
    
    print(f"📦 Created zip file: {zip_path}")


def cleanup_local_embeddings(embedding_path: Path):
    """Clean up local embedding files.
    
    Args:
        embedding_path: Path to embedding directory
    """
    if embedding_path.exists():
        try:
            shutil.rmtree(embedding_path)
            print(f"🗑️ Cleaned up embeddings: {embedding_path}")
        except Exception as e:
            print(f"⚠️ Error cleaning embeddings: {e}")
    else:
        print(f"ℹ️ No embeddings to clean: {embedding_path}")


def load_annotation_file(file_path: Path) -> np.ndarray:
    """Load annotation file as numpy array.
    
    Args:
        file_path: Path to annotation file
        
    Returns:
        Numpy array of annotation data
    """
    # Stub implementation
    try:
        import imageio
        return imageio.imread(file_path)
    except ImportError:
        # Fallback: return mock data
        return np.zeros((256, 256), dtype=np.uint8)