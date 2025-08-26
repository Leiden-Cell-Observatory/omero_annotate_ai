"""File I/O functions for micro-SAM workflows."""

import shutil
import zipfile
from pathlib import Path
from typing import Any, List

import numpy as np



def zip_directory(directory_path: Path, zip_path: Path):
    """Zip a directory.

    Args:
    directory_path: Directory to zip
    zip_path: Output zip file path
    """
    if not directory_path.exists():
        print(f"Directory not found: {directory_path}")
        return

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(directory_path))

    print(f"Created zip file: {zip_path}")


def cleanup_local_embeddings(embedding_path: Path):
    """Clean up local embedding files.

    Args:
    embedding_path: Path to embedding directory
    """
    if embedding_path.exists():
        try:
            shutil.rmtree(embedding_path)
            print(f"Cleaned up embeddings: {embedding_path}")
        except Exception as e:
            print(f"Error cleaning embeddings: {e}")
    else:
        print(f"No embeddings to clean: {embedding_path}")
