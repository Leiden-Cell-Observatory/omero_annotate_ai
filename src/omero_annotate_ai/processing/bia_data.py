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
from tifffile import imread, imwrite

from ..bundle import file_digest
from ..errors import IntegrityError
from ..omero._untyped import ezomero
from ..provenance import describe_change
from ..rows import Row, rows_from_frame
from ..sources import ImageSource, OmeroImageSource, read_unit

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


def _write_image(source: ImageSource, annotation: Row, dest: Path) -> None:
    """Write the plane/patch/volume of one annotation to ``dest``, at native dtype."""
    pixels = read_unit(source, annotation, max(annotation.channel, 0))
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
        IntegrityError: A downloaded mask no longer matches the ``mask_digest``
            recorded when it was annotated (it was changed on OMERO since).
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
    changed: list[str] = []
    source = OmeroImageSource(conn)

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
                _write_image(source, annotation, image_dir / f"{annotation_id}.tif")
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
                mask_file = mask_dir / f"{annotation_id}_mask.tif"
                if _download_mask(conn, annotation.label_id, mask_file, tmp_dir):
                    recorded = annotation.mask_digest
                    if recorded and file_digest(mask_file) != recorded:
                        change = describe_change(
                            annotation.mask_iscc,
                            np.asarray(imread(mask_file)),
                            mask=True,
                        )
                        changed.append(
                            f"{annotation_id} ({change})" if change else annotation_id
                        )
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

    if changed:
        raise IntegrityError(
            f"{len(changed)} label file(s) on OMERO no longer match the masks recorded "
            f"when they were annotated: {', '.join(changed[:5])}"
            + (" ..." if len(changed) > 5 else "")
        )

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
