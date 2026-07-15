"""Download OMERO table data into the on-disk layout a BIA bundle is built from.

:meth:`AnnotationConfig.save_bia_package` copies the referenced images and masks out
of ``config.output.output_directory``, using the paths that
:func:`omero_annotate_ai.core.mifa_export._file_ids` derives from each annotation:

- ``output/{annotation_id}_mask.tif``
- ``input/{annotation_id}.tif`` (or ``label_input/{annotation_id}.tif`` when the
  config annotates one channel and trains on another)

That layout is what ``AnnotationPipeline`` leaves behind on the machine that did the
annotating. A table-driven export starts with the data on OMERO instead, so
:func:`prepare_bia_data_from_table` fetches it and writes exactly that layout.

Unlike the training-data path
(:func:`omero_annotate_ai.processing.training_functions.prepare_training_data_from_table`),
images are written at their **native bit depth**: the training path rescales every
plane to 8-bit, which is lossy and unacceptable for an archive submission.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import ezomero
import numpy as np
from tifffile import imwrite

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.annotation_config import AnnotationConfig, ImageAnnotation

__all__ = ["prepare_bia_data_from_table"]

# Directory holding the mask files (mirrors AnnotationPipeline._setup_directories).
MASK_DIR = "output"

# The two possible image directories (mirrors AnnotationPipeline._get_input_folders).
INPUT_DIR = "input"
LABEL_INPUT_DIR = "label_input"


def _input_dir_name(config: "AnnotationConfig") -> str:
    """``label_input`` when the config annotates and trains on different channels."""
    if config.spatial_coverage.uses_separate_channels():
        return LABEL_INPUT_DIR
    return INPUT_DIR


def _non_negative(value: Optional[int], default: int = 0) -> int:
    """Coerce an ImageAnnotation coordinate to a usable index (``-1``/None -> default)."""
    if value is None:
        return default
    value = int(value)
    return value if value >= 0 else default


def _z_slices(annotation: "ImageAnnotation") -> List[int]:
    """Z indices to fetch: the volume's range for 3D records, one plane otherwise."""
    if not annotation.is_volumetric:
        return [_non_negative(annotation.z_slice)]
    z_start, z_end = annotation.z_start, annotation.z_end
    if z_start >= 0 and z_end >= z_start:
        return list(range(z_start, z_end + 1))
    return [_non_negative(annotation.z_slice)]


def _fetch_plane(
    conn: Any, annotation: "ImageAnnotation", z: int, channel: int, timepoint: int
) -> np.ndarray:
    """Fetch one 2D plane (or patch) from OMERO as a ``(y, x)`` array, native dtype.

    ``ezomero.get_image`` returns an XYZCT-ordered 5D array here, so the singleton
    z/c/t axes are dropped and x/y swapped to give the row-major (y, x) layout
    ``tifffile`` expects.
    """
    image_id = int(annotation.image_id)
    if annotation.is_patch and annotation.patch_width > 0 and annotation.patch_height > 0:
        start_coords = (
            int(annotation.patch_x),
            int(annotation.patch_y),
            int(z),
            int(channel),
            int(timepoint),
        )
        axis_lengths = (
            int(annotation.patch_width),
            int(annotation.patch_height),
            1,
            1,
            1,
        )
    else:
        omero_image, _ = ezomero.get_image(conn, image_id, no_pixels=True)
        if omero_image is None:
            raise ValueError(f"Image {image_id} not found in OMERO")
        start_coords = (0, 0, int(z), int(channel), int(timepoint))
        axis_lengths = (int(omero_image.getSizeX()), int(omero_image.getSizeY()), 1, 1, 1)

    _, pixels = ezomero.get_image(
        conn,
        image_id,
        start_coords=start_coords,
        axis_lengths=axis_lengths,
        xyzct=True,
    )
    if pixels is None:
        raise ValueError(f"No pixel data returned for image {image_id}")

    pixels = np.asarray(pixels)
    if pixels.ndim == 5:
        pixels = pixels[:, :, 0, 0, 0]
        pixels = np.swapaxes(pixels, 0, 1)
    return pixels


def _write_image(conn: Any, annotation: "ImageAnnotation", dest: Path) -> None:
    """Fetch the plane/patch/volume for one annotation and write it to ``dest``.

    No intensity rescaling: the array is written at the dtype OMERO returned.
    """
    channel = _non_negative(annotation.channel)
    timepoint = _non_negative(annotation.timepoint)

    planes = [
        _fetch_plane(conn, annotation, z, channel, timepoint)
        for z in _z_slices(annotation)
    ]
    pixels = np.stack(planes) if annotation.is_volumetric else planes[0]

    dest.parent.mkdir(parents=True, exist_ok=True)
    # photometric="minisblack": a (z, y, x) stack is grayscale pages, never RGB planes.
    imwrite(str(dest), pixels, photometric="minisblack")


def _download_mask(conn: Any, label_id: int, dest: Path, tmp_dir: Path) -> bool:
    """Download the mask FileAnnotation ``label_id`` to ``dest``. False if unavailable."""
    label_id = int(label_id)
    file_ann = conn.getObject("FileAnnotation", label_id)
    if file_ann is None:
        warnings.warn(
            f"FileAnnotation {label_id} not found in OMERO; mask not exported.",
            UserWarning,
            stacklevel=2,
        )
        return False

    tmp_path = ezomero.get_file_annotation(conn, label_id, str(tmp_dir))
    if not tmp_path:
        warnings.warn(
            f"FileAnnotation {label_id} could not be downloaded; mask not exported.",
            UserWarning,
            stacklevel=2,
        )
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(tmp_path), str(dest))
    return True


def _clean_data_dirs(output_dir: Path) -> None:
    """Remove any existing input / label_input / output directories."""
    for name in (INPUT_DIR, LABEL_INPUT_DIR, MASK_DIR):
        directory = output_dir / name
        if directory.exists():
            shutil.rmtree(directory)


def prepare_bia_data_from_table(
    conn: Any,
    table_id: int,
    output_dir: Union[str, Path],
    config: Optional["AnnotationConfig"] = None,
    clean_existing: bool = False,
) -> Dict[str, Any]:
    """Download the data referenced by an OMERO annotation table for a BIA export.

    Loads the table, rebuilds the annotation records from it (each carries a unique
    ``annotation_id``) and writes, for every row that is ``processed`` and has a
    ``label_id``:

    - ``{output_dir}/input/{annotation_id}.tif`` - the image plane, patch or volume
      (``label_input/`` instead when ``config.spatial_coverage.uses_separate_channels()``)
    - ``{output_dir}/output/{annotation_id}_mask.tif`` - the mask FileAnnotation

    which is the layout :meth:`AnnotationConfig.save_bia_package` copies from. Set
    ``config.output.output_directory = output_dir`` afterwards and package it.

    Images are written at their **native dtype** - no 8-bit rescaling.

    Rows that are unprocessed, or processed but carry no mask, are counted and skipped
    with a warning. Individual fetch failures are warned about and skipped too; only a
    missing/empty table is fatal.

    On return, ``config.annotations`` holds **only the exported records** - the ones whose
    image and mask both reached the disk. ``save_bia_package`` derives the BIA file lists
    from every annotation in the config, so leaving a skipped row in place would list a
    file that was never downloaded in the submission manifest.

    Args:
        conn: OMERO connection (``BlitzGateway``).
        table_id: OMERO ID of the annotation table (``OriginalFile``).
        output_dir: Directory to write the ``input``/``label_input`` and ``output`` trees into.
        config: Annotation config, **mutated in place**: its ``annotations`` are rebuilt from
            the table (as :func:`sync_omero_table_to_config` does), then pruned to the
            exported records, so the ids on disk match the ones the MIFA/BIA file lists
            reference. A default config is used when omitted.
        clean_existing: Remove existing ``input``/``label_input``/``output`` directories first.

    Returns:
        Dict with ``output_dir``, ``n_images``, ``n_masks``, ``skipped_unprocessed``
        and ``skipped_no_mask``.

    Raises:
        ValueError: If the table cannot be loaded or is empty.
    """
    output_dir = Path(output_dir)

    if config is None:
        from ..core.annotation_config import create_default_config

        config = create_default_config()

    try:
        table = ezomero.get_table(conn, table_id)
    except Exception as exc:
        raise ValueError(f"Failed to load table {table_id}: {exc}") from exc

    if table is None or len(table) == 0:
        raise ValueError(f"Table {table_id} is empty or not found")

    # Rebuild the annotation records (and their annotation_ids) from the table.
    config.from_dataframe(table)

    if clean_existing:
        _clean_data_dirs(output_dir)

    image_dir = output_dir / _input_dir_name(config)
    mask_dir = output_dir / MASK_DIR
    tmp_dir = output_dir / "tmp"
    for directory in (image_dir, mask_dir, tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_masks = 0
    skipped_unprocessed = 0
    skipped_no_mask = 0
    exported: List["ImageAnnotation"] = []

    try:
        for annotation in config.annotations:
            if not annotation.processed:
                skipped_unprocessed += 1
                continue
            if annotation.label_id is None:
                skipped_no_mask += 1
                continue

            annotation_id = annotation.annotation_id

            try:
                _write_image(conn, annotation, image_dir / f"{annotation_id}.tif")
                n_images += 1
            except Exception as exc:
                warnings.warn(
                    f"Failed to export image for annotation {annotation_id!r} "
                    f"(image {annotation.image_id}): {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            try:
                if _download_mask(
                    conn,
                    annotation.label_id,
                    mask_dir / f"{annotation_id}_mask.tif",
                    tmp_dir,
                ):
                    n_masks += 1
                    exported.append(annotation)
            except Exception as exc:
                warnings.warn(
                    f"Failed to export mask for annotation {annotation_id!r} "
                    f"(label {annotation.label_id}): {exc}",
                    UserWarning,
                    stacklevel=2,
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Keep only what actually made it to disk. save_bia_package() builds the BIA file
    # lists from every annotation in the config, so a skipped row left in place would
    # put a path to a file we never downloaded into the submission manifest.
    config.annotations = exported

    if skipped_unprocessed or skipped_no_mask:
        warnings.warn(
            f"Skipped {skipped_unprocessed} unprocessed row(s) and "
            f"{skipped_no_mask} processed row(s) without a mask (label_id) "
            f"while exporting table {table_id}.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "output_dir": output_dir,
        "n_images": n_images,
        "n_masks": n_masks,
        "skipped_unprocessed": skipped_unprocessed,
        "skipped_no_mask": skipped_no_mask,
    }
