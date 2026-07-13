"""On-disk layout for prepared training data.

This module owns every decision about where training files go: folder names,
cleaning, and how a file is materialized at its destination. Both training-data
producers (OMERO download and offline reorganization) route through it, so the
layout is defined exactly once.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from tifffile import imwrite

CATEGORIES = ("training", "validation", "test")

# Per-category folder names: (image, label, annotation-channel image)
_SPLIT_FOLDERS = {
    "training": ("train_input", "train_label", "train_annotation_input"),
    "validation": ("val_input", "val_label", "val_annotation_input"),
    "test": ("test_input", "test_label", "test_annotation_input"),
}


def split_folders_for(category: str) -> tuple:
    """Return (image, label, annotation_image) folder names for a category."""
    if category not in _SPLIT_FOLDERS:
        raise ValueError(f"Unknown category: {category}. Expected one of {CATEGORIES}")
    return _SPLIT_FOLDERS[category]


def layout_folders(
    layout: str = "split",
    uses_separate_channels: bool = False,
    include_test: bool = False,
) -> List[str]:
    """
    Return every folder name a layout writes, relative to the output directory.

    Args:
        layout: "split" (micro-SAM, BiaPy) or "cellpose" (not yet implemented)
        uses_separate_channels: Whether the annotation channel is carried along
        include_test: Whether a test split is written

    Returns:
        Folder names in a stable order

    Raises:
        NotImplementedError: For layout="cellpose"
        ValueError: For an unknown layout
    """
    if layout == "cellpose":
        raise NotImplementedError(
            "The cellpose layout (image and {id}_masks.tif in one folder) is "
            "specified but not implemented. Use layout='split'."
        )
    if layout != "split":
        raise ValueError(f"Unknown layout: {layout}. Expected 'split' or 'cellpose'.")

    categories = ["training", "validation"]
    if include_test:
        categories.append("test")

    folders = []
    for category in categories:
        image_dir, label_dir, annotation_dir = split_folders_for(category)
        folders += [image_dir, label_dir]
        if uses_separate_channels:
            folders.append(annotation_dir)

    return folders


def clean_layout(
    output_dir: Path,
    layout: str = "split",
    uses_separate_channels: bool = False,
    include_test: bool = False,
) -> None:
    """Remove the folders this layout writes, so a run cannot inherit stale data."""
    for folder_name in layout_folders(layout, uses_separate_channels, include_test):
        folder_path = output_dir / folder_name
        if folder_path.exists():
            shutil.rmtree(folder_path)


def assert_output_dir_is_separate(output_dir: Path, annotation_dir: Path) -> None:
    """
    Reject a training output directory that sits inside the annotation directory.

    The annotation phase writes source images the training phase must not touch.
    Cleaning the training layout inside the annotation directory would delete them.

    Raises:
        ValueError: If output_dir is annotation_dir or nested within it
    """
    output_resolved = Path(output_dir).resolve()
    annotation_resolved = Path(annotation_dir).resolve()

    if output_resolved == annotation_resolved or annotation_resolved in output_resolved.parents:
        raise ValueError(
            f"Training output directory must not be inside the annotation directory.\n"
            f"  annotation_dir: {annotation_resolved}\n"
            f"  output_dir:     {output_resolved}\n"
            f"Cleaning the training layout there would delete your source images. "
            f"Use a separate directory, e.g. {annotation_resolved.name}_training/."
        )


def _create_file_link_or_copy(src: Path, dst: Path, mode: str, logger=None) -> str:
    """
    Create a file at destination using the specified mode.

    Args:
        src: Source file path
        dst: Destination file path
        mode: One of "copy", "move", or "symlink"
        logger: Optional logger for messages

    Returns:
        String describing the action taken (e.g. "symlink", "copy (symlink fallback)")
    """
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return "symlink"
        except OSError as e:
            # Windows without developer mode or elevated privileges, or other OS issues
            if logger:
                logger.debug(f"Symlink failed ({e}), falling back to copy")
            shutil.copy2(src, dst)
            return "copy (symlink fallback)"
    elif mode == "move":
        shutil.move(str(src), str(dst))
        return "move"
    else:  # copy (default)
        shutil.copy2(src, dst)
        return "copy"


class FileSource:
    """An image that already exists on disk. Honours file_mode."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def write_to(self, dst: Path, file_mode: str = "copy", logger=None) -> str:
        return _create_file_link_or_copy(self.path, dst, file_mode, logger)


class ArraySource:
    """
    An image that must be fetched (e.g. an OMERO plane).

    The loader is called lazily, at write time, so preparing a large dataset does
    not hold every plane in memory at once. file_mode is meaningless here: there
    is no file on disk to link to.
    """

    def __init__(self, loader: Callable[[], np.ndarray]):
        self.loader = loader

    def exists(self) -> bool:
        return True

    def write_to(self, dst: Path, file_mode: str = "copy", logger=None) -> str:
        imwrite(str(dst), self.loader())
        return "write"


Source = object  # FileSource | ArraySource


@dataclass
class AnnotationRecord:
    """
    One annotated image, ready to be written into a training layout.

    Files are named by annotation_id, never by loop index, so an image and its
    label always pair by name. A record without a label is dropped by the caller
    rather than written, which is what keeps the pairing honest.
    """

    annotation_id: int
    category: str
    image: Source
    label: Source
    annotation_image: Optional[Source] = None

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(
                f"Unknown category: {self.category}. Expected one of {CATEGORIES}"
            )
