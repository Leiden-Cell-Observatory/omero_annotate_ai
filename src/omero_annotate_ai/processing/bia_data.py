"""Download OMERO table data into the on-disk layout a BIA bundle is built from.

:func:`omero_annotate_ai.core.mifa_export.save_bia_package` copies the referenced
images and masks out of a data folder, using the paths
:func:`omero_annotate_ai.core.mifa_export._file_ids` derives from each row:

- ``annotation_output/{annotation_id}_mask.tif``
- ``annotation_input/{annotation_id}.tif`` - the annotated channel, also when the
  config annotates one channel and trains on another

That layout is a training set's own folder (``TrainingSet.bundle_dir``). A
table-driven export starts with the data on OMERO instead, so
:func:`prepare_bia_data_from_table` fetches it and writes exactly that layout.

Images are written at their **native bit depth**, as the training export does
(:func:`omero_annotate_ai.processing.training_functions.prepare_training_data_from_table`).
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import imwrite

from ..omero._untyped import ezomero
from ..rows import Row, rows_from_frame

__all__ = ["prepare_bia_data_from_table"]

# Directory holding the mask files (see omero_annotate_ai.bundle).
MASK_DIR = "annotation_output"

# Directory holding the annotated channel (see omero_annotate_ai.bundle).
# A separate-channel workflow also writes model_input/, but the BIA source image is the
# channel the mask was drawn on.
IMAGE_DIR = "annotation_input"


def _read_table(conn: Any, table_id: int) -> Any:
    """The tracking table with its link columns named ``image_id`` / ``roi_id``."""
    from ..omero.omero_functions import read_tracking_table  # avoids an import cycle

    return read_tracking_table(conn, table_id)


def _non_negative(value: int | None, default: int = 0) -> int:
    """Coerce an Row coordinate to a usable index (``-1``/None -> default)."""
    if value is None:
        return default
    value = int(value)
    return value if value >= 0 else default


def _z_slices(annotation: Row) -> list[int]:
    """Z indices to fetch: the volume's range for 3D records, one plane otherwise."""
    if not annotation.is_volumetric:
        return [_non_negative(annotation.z_slice)]
    z_start, z_end = annotation.z_start, annotation.z_end
    if z_start >= 0 and z_end >= z_start:
        return list(range(z_start, z_end + 1))
    return [_non_negative(annotation.z_slice)]


def _fetch_plane(
    conn: Any, annotation: Row, z: int, channel: int, timepoint: int
) -> np.ndarray:
    """Fetch one 2D plane (or patch) from OMERO as a ``(y, x)`` array, native dtype.

    ``ezomero.get_image`` returns an XYZCT-ordered 5D array here, so the singleton
    z/c/t axes are dropped and x/y swapped to give the row-major (y, x) layout
    ``tifffile`` expects.
    """
    if annotation.image_id is None:
        raise ValueError(
            f"Annotation {annotation.annotation_id!r} has no OMERO image id"
        )
    image_id = annotation.image_id
    if (
        annotation.is_patch
        and annotation.patch_width > 0
        and annotation.patch_height > 0
    ):
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
        axis_lengths = (
            int(omero_image.getSizeX()),
            int(omero_image.getSizeY()),
            1,
            1,
            1,
        )

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


def _write_image(conn: Any, annotation: Row, dest: Path) -> None:
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
    """Remove any existing annotation_input / annotation_output directories."""
    for name in (IMAGE_DIR, MASK_DIR):
        directory = output_dir / name
        if directory.exists():
            shutil.rmtree(directory)


def prepare_bia_data_from_table(
    conn: Any,
    table_id: int,
    output_dir: str | Path,
    clean_existing: bool = False,
) -> dict[str, Any]:
    """Download the data referenced by an OMERO annotation table for a BIA export.

    Loads the table's rows and writes, for every row that is ``processed`` and has a
    ``label_id``:

    - ``{output_dir}/annotation_input/{annotation_id}.tif`` - the image plane, patch or volume
    - ``{output_dir}/annotation_output/{annotation_id}_mask.tif`` - the mask FileAnnotation

    which is the layout :func:`~omero_annotate_ai.core.mifa_export.save_bia_package`
    copies from: pass it the returned ``rows`` and ``output_dir``.

    Images are written at their **native dtype** - no 8-bit rescaling.

    Rows that are unprocessed, or processed but carry no mask, are counted and skipped
    with a warning. Individual fetch failures are warned about and skipped too; only a
    missing/empty table is fatal.

    The returned ``rows`` hold **only the exported records** - the ones whose image and
    mask both reached the disk - so the BIA file lists never name a file that was not
    downloaded.

    Args:
        conn: OMERO connection (``BlitzGateway``).
        table_id: OMERO ID of the annotation table (``OriginalFile``).
        output_dir: Directory to write the ``annotation_input`` and
            ``annotation_output`` trees into.
        clean_existing: Remove existing ``annotation_input``/``annotation_output``
            directories first.

    Returns:
        Dict with ``output_dir``, ``rows`` (the exported rows), ``n_images``,
        ``n_masks``, ``skipped_unprocessed`` and ``skipped_no_mask``.

    Raises:
        ValueError: If the table cannot be loaded or is empty.
    """
    output_dir = Path(output_dir)

    try:
        table = _read_table(conn, table_id)
    except Exception as exc:
        raise ValueError(f"Failed to load table {table_id}: {exc}") from exc

    if table is None or len(table) == 0:
        raise ValueError(f"Table {table_id} is empty or not found")

    rows = rows_from_frame(table)

    if clean_existing:
        _clean_data_dirs(output_dir)

    image_dir = output_dir / IMAGE_DIR
    mask_dir = output_dir / MASK_DIR
    tmp_dir = output_dir / "tmp"
    for directory in (image_dir, mask_dir, tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_masks = 0
    skipped_unprocessed = 0
    skipped_no_mask = 0
    exported: list[Row] = []

    try:
        for annotation in rows:
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
            except Exception as exc:  # noqa: BLE001 - best effort per record
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
            except Exception as exc:  # noqa: BLE001 - best effort per record
                warnings.warn(
                    f"Failed to export mask for annotation {annotation_id!r} "
                    f"(label {annotation.label_id}): {exc}",
                    UserWarning,
                    stacklevel=2,
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

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
        "rows": exported,
        "n_images": n_images,
        "n_masks": n_masks,
        "skipped_unprocessed": skipped_unprocessed,
        "skipped_no_mask": skipped_no_mask,
    }
