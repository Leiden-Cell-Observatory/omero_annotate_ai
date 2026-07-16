"""On-disk layout for prepared training data.

This module owns every decision about where training files go: folder names,
cleaning, and how a file is materialized at its destination. Both training-data
producers (OMERO download and offline reorganization) route through it, so the
layout is defined exactly once.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

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

    def _is_inside(child: Path, parent: Path) -> bool:
        if child == parent or parent in child.parents:
            return True

        # Path comparison alone is not enough: on Windows and (by default) macOS the
        # filesystem is case-insensitive, so /x/Project and /x/project are one
        # directory, and resolve() does not case-fold. Ask the filesystem instead of
        # guessing - samefile() answers correctly on case-insensitive AND
        # case-sensitive filesystems, so a genuine /data/Project vs /data/project on
        # Linux is still allowed through.
        if not parent.exists():
            return False
        for candidate in (child, *child.parents):
            if not candidate.exists():
                continue
            try:
                if candidate.samefile(parent):
                    return True
            except OSError:
                # Race: candidate existed at the .exists() check above but is gone
                # (or otherwise unreadable) by the time samefile() runs. Treat it as
                # not-a-match rather than failing the whole check.
                pass
            # Only the nearest existing ancestor is informative; anything above it
            # is a shared prefix, not containment.
            break
        return False

    if _is_inside(output_resolved, annotation_resolved):
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

    # ImageAnnotation.annotation_id is a str (e.g. "img_001"); the OMERO table may
    # yield an int. Both are accepted and stringified into the filename.
    annotation_id: Union[int, str]
    category: str
    image: Source
    label: Source
    annotation_image: Optional[Source] = None

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(
                f"Unknown category: {self.category}. Expected one of {CATEGORIES}"
            )


def _empty_stats() -> dict:
    stats = {}
    for prefix in ("training", "val", "test"):
        stats[f"n_{prefix}_images"] = 0
        stats[f"n_{prefix}_labels"] = 0
        stats[f"n_{prefix}_annotation_input"] = 0
    stats["n_skipped"] = 0
    stats["file_operations"] = {}
    stats["file_mapping"] = {}
    return stats


# Category -> the prefix used in stats keys
_STATS_PREFIX = {"training": "training", "validation": "val", "test": "test"}


def write_training_layout(
    records: List[AnnotationRecord],
    output_dir: Path,
    layout: str = "split",
    file_mode: str = "copy",
    clean_existing: bool = True,
    include_test: bool = False,
    uses_separate_channels: bool = False,
    logger=None,
) -> tuple:
    """
    Write annotation records into a training layout.

    Files are named {annotation_id}.tif in every folder, so an image and its label
    always pair by name.

    Args:
        records: Annotation records to write
        output_dir: Target directory (must not be inside the annotation directory)
        layout: "split" or "cellpose"
        file_mode: "copy", "move" or "symlink" (ignored by ArraySource records)
        clean_existing: Remove the layout's folders before writing
        include_test: Write the test split; test records are skipped when False
        uses_separate_channels: Write the annotation-channel image alongside
        logger: Optional logger

    Returns:
        (created_dirs, stats)
    """
    output_dir = Path(output_dir)

    if clean_existing:
        clean_layout(output_dir, layout, uses_separate_channels, include_test)

    folder_names = layout_folders(layout, uses_separate_channels, include_test)
    created_dirs = {}
    for folder_name in folder_names:
        folder_path = output_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        created_dirs[folder_name] = folder_path

    stats = _empty_stats()

    def _record_operation(action: str):
        stats["file_operations"][action] = stats["file_operations"].get(action, 0) + 1

    for record in records:
        if record.category == "test" and not include_test:
            stats["n_skipped"] += 1
            continue

        image_folder, label_folder, annotation_folder = split_folders_for(record.category)
        prefix = _STATS_PREFIX[record.category]
        filename = f"{record.annotation_id}.tif"
        mapping = {}

        image_dst = created_dirs[image_folder] / filename
        _record_operation(record.image.write_to(image_dst, file_mode, logger))
        stats[f"n_{prefix}_images"] += 1
        mapping["image"] = str(image_dst)

        label_dst = created_dirs[label_folder] / filename
        _record_operation(record.label.write_to(label_dst, file_mode, logger))
        stats[f"n_{prefix}_labels"] += 1
        mapping["label"] = str(label_dst)

        if uses_separate_channels and record.annotation_image is not None:
            annotation_dst = created_dirs[annotation_folder] / filename
            _record_operation(
                record.annotation_image.write_to(annotation_dst, file_mode, logger)
            )
            stats[f"n_{prefix}_annotation_input"] += 1
            mapping["annotation_image"] = str(annotation_dst)

        stats["file_mapping"][str(record.annotation_id)] = mapping

        if logger:
            logger.debug(f"Wrote annotation {record.annotation_id} to {record.category}")

    return created_dirs, stats
