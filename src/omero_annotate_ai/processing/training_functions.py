"""Training data preparation functions for micro-SAM workflows."""

import ast
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
)

import numpy as np
import pandas as pd
from tifffile import imread

from ..bundle import file_digest
from ..core.annotation_config import NormalizationConfig
from ..errors import IntegrityError
from ..omero._untyped import ezomero
from ..rows import rows_from_frame
from ..utils.logging import create_training_logger
from .normalization import NormalizationRecorder, write_native_sidecar
from .training_layout import (
    AnnotationRecord,
    ArraySource,
    FileSource,
    assert_output_dir_is_separate,
    write_training_layout,
)
from .utils import validate_table_schema

if TYPE_CHECKING:
    from ..core.annotation_config import AnnotationConfig

_LOGGER = logging.getLogger(__name__)


def _build_standard_result(
    base_dir: Path, created_dirs: dict[str, Path], stats: dict[str, Any], **extra_fields
) -> dict[str, Any]:
    """
    Build the standard result dictionary for training functions.

    Args:
        base_dir: Base output directory
        created_dirs: Dictionary of created directories
        stats: Statistics dictionary
        **extra_fields: Additional function-specific fields

    Returns:
        Standard result dictionary
    """
    result = {
        "base_dir": base_dir,
        "stats": stats,
    }

    # Add directory paths that were created
    for purpose, path in created_dirs.items():
        result[purpose] = path

    # Add any extra fields
    result.update(extra_fields)

    return result


def prepare_training_data_from_table(
    conn: Any,
    table_id: int,
    output_dir: str | Path,
    training_name: str = "micro_sam_training",
    clean_existing: bool = True,
    tmp_dir: str | Path | None = None,
    verbose: bool = False,
    label_channel: int | None = None,
    training_channels: list[int] | None = None,
    upload_annotation_input: bool = False,
    normalization: Union["NormalizationConfig", dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Prepare training data from OMERO annotation table.

    Downloads images and labels from OMERO based on annotation table data and
    organizes them by each row's ``category`` (training, validation, test) into a
    directory structure suitable for micro-SAM training.

    Args:
        conn: OMERO connection object
        table_id: ID of the annotation table in OMERO
        output_dir: Directory to store training data
        training_name: Name for the training session (used in directory naming)
        clean_existing: Whether to clean existing output directories
        tmp_dir: Temporary directory for downloads (optional)
        verbose: If True, show detailed debug information in console output
        label_channel: Optional channel index for label/segmentation images. If provided
            and different from training_channels, downloads label channel images to
            *_annotation_input directories alongside the training data.
        training_channels: Optional list of channel indices for training input images.
            If different from label_channel, downloads from these channels for
            train_input and val_input. Currently uses first channel if multiple specified.
        upload_annotation_input: If True and using separate channels, uploads the annotation-channel
            images back to OMERO as file annotations. Default is False.
        normalization: Normalisation spec (``config.normalization`` or an equivalent dict).
            Images are written at native bit depth unless ``apply_at_export`` is set, in
            which case they are written as float32 and the bounds per image are recorded.
            When given, the spec is saved as ``normalization.json`` in output_dir.

    Returns:
        Dictionary with paths to created directories:
        {
            'base_dir': Path to base output directory,
            'train_input': Path to training images,
            'train_label': Path to training labels (segmentation masks),
            'train_annotation_input': Path to annotation-channel images (separate channels only),
            'val_input': Path to validation images,
            'val_label': Path to validation labels (segmentation masks),
            'val_annotation_input': Path to annotation-channel validation images
                (separate channels only),
            'test_input', 'test_label': Test images and labels (only when the table
                has test rows),
            'stats': Statistics about the prepared data
        }

    Raises:
        ValueError: If table not found or invalid parameters
        ImportError: If required dependencies missing
    """
    # Convert paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger for this training session
    logger = create_training_logger(output_dir, verbose=verbose)
    logger.info(f"Starting training data preparation from table {table_id}")
    logger.debug(
        f"Parameters: output_dir={output_dir}, clean_existing={clean_existing}"
    )

    if tmp_dir is None:
        tmp_dir = output_dir / "tmp"
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        table = _load_table(conn, table_id)
    except Exception as e:  # noqa: BLE001 - any read failure means the table is unusable
        raise ValueError(f"Failed to load table {table_id}: {e}") from e

    if table is None or len(table) == 0:
        raise ValueError(f"Table {table_id} is empty or not found")

    logger.info(f"Loaded table with {len(table)} rows")

    # Save the table locally for inspection
    table_path = output_dir / f"table_{table_id}.csv"
    try:
        table.to_csv(table_path, index=True)
        logger.info(f"Table saved to: {table_path}")
    except OSError as e:
        logger.warning(f"Failed to save table: {e}")

    # Check if 'processed' column exists and filter to only processed rows
    if "processed" in table.columns:
        initial_count = len(table)
        unprocessed_count = len(table[~table["processed"]])

        if unprocessed_count > 0:
            logger.warning(
                f"Found {unprocessed_count} unprocessed rows out of {initial_count} total rows"
            )
            logger.info(
                f"Proceeding with {initial_count - unprocessed_count} processed rows for training"
            )

        # Filter to only processed rows
        table = table.loc[table["processed"].astype(bool)].copy()

        if len(table) == 0:
            raise ValueError(
                "No processed rows found in the table. Cannot proceed with training."
            )

        logger.info(f"Using {len(table)} processed rows for training")

    else:
        logger.warning(
            "No 'processed' column found - assuming all rows are ready for training"
        )

    # Validate table schema and data integrity
    validate_table_schema(table, logger)
    logger.info("Table schema validated for processing")

    # Determine if we're using separate channels for labeling and training
    uses_separate_channels = (
        label_channel is not None
        and training_channels is not None
        and label_channel not in training_channels
    )

    if uses_separate_channels:
        logger.info(
            f"Using separate channels: label_channel={label_channel}, "
            f"training_channels={training_channels}"
        )

    # Determine the effective training channel to use
    effective_train_channel = training_channels[0] if training_channels else None

    # The split was planned when the training set was created; each row carries it.
    counts = table["category"].value_counts()
    include_test = bool(counts.get("test", 0))
    logger.info(
        f"Using {counts.get('training', 0)} training, {counts.get('validation', 0)} "
        f"validation and {counts.get('test', 0)} test rows"
    )

    # Build records. A row whose label cannot be downloaded is dropped whole: writing
    # the image without its label would leave an orphan, and consumers pair images to
    # labels by sorted filename, so one orphan shifts every subsequent pair.
    spec = (
        NormalizationConfig.model_validate(normalization)
        if normalization is not None
        else None
    )
    recorder = (
        NormalizationRecorder(spec)
        if spec is not None and spec.apply_at_export
        else None
    )

    records, n_missing = _records_from_table(
        conn,
        table,
        tmp_dir,
        uses_separate_channels,
        label_channel,
        effective_train_channel,
        logger,
        transform=recorder,
    )

    created_dirs, stats = write_training_layout(
        records,
        output_dir,
        layout="split",
        clean_existing=clean_existing,
        include_test=include_test,
        uses_separate_channels=uses_separate_channels,
        logger=logger,
    )
    stats["n_missing"] = n_missing
    stats["total_rows_processed"] = len(table)
    if recorder is not None:
        recorder.write_sidecar(output_dir)
    elif spec is not None:
        write_native_sidecar(spec, output_dir)

    annotation_input_upload_ids = []
    if uses_separate_channels and upload_annotation_input:
        logger.info("Uploading annotation-channel images to OMERO...")
        rows_by_id = {str(row["annotation_id"]): row for _, row in table.iterrows()}

        for folder_key in ("train_annotation_input", "val_annotation_input"):
            folder = created_dirs.get(folder_key)
            if folder is None or not folder.exists():
                continue

            for tif_file in sorted(folder.glob("*.tif")):
                # Files are named {annotation_id}.tif, so this is a direct lookup.
                row = rows_by_id.get(tif_file.stem)
                if row is None:
                    logger.warning(f"No table row for {tif_file.name}; not uploading")
                    continue
                try:
                    # Lazy import to avoid circular dependency
                    from ..omero.omero_functions import upload_annotation_input_image

                    file_ann_id = upload_annotation_input_image(
                        conn,
                        image_id=int(str(row["image_id"])),
                        annotation_input_file=str(tif_file),
                        trainingset_name=training_name,
                        channel=label_channel,
                        timepoint=_optional_int(row.get("timepoint")),
                        z_slice=_optional_int(row.get("z_slice")),
                    )
                    annotation_input_upload_ids.append(file_ann_id)
                except Exception as e:  # noqa: BLE001 - best effort per file; keep exporting
                    logger.warning(f"Could not upload {tif_file.name}: {e}")

        stats["n_annotation_input_uploaded"] = len(annotation_input_upload_ids)
        logger.info(
            f"Uploaded {len(annotation_input_upload_ids)} annotation-channel images"
        )

    # Clean up temporary directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        logger.debug(f"Cleaned up temporary directory: {tmp_dir}")

    extra_fields = {}
    if annotation_input_upload_ids:
        extra_fields["annotation_input_upload_ids"] = annotation_input_upload_ids

    result = _build_standard_result(
        base_dir=output_dir, created_dirs=created_dirs, stats=stats, **extra_fields
    )

    if stats["n_training_images"] == 0 and stats["n_val_images"] == 0:
        logger.error(f"Training data preparation FAILED in: {output_dir}")
        logger.error(f"Statistics: {stats}")
        raise ValueError(
            "Training data preparation failed - no images were processed successfully. "
            "Check the error messages above."
        )
    logger.info(f"Training data prepared successfully in: {output_dir}")
    logger.info(f"Statistics: {stats}")

    # Close logger handlers to release file locks (important for Windows)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    return result


def _fetch_plane(conn, row, channel: int, logger=None) -> np.ndarray:
    """
    Fetch one annotated image from OMERO at its native bit depth.

    Handles the 2D, 2D-patch and 3D-volumetric cases. Returns the array rather than
    writing it, so the caller decides where it lands - which is what lets files be
    named by annotation_id instead of by loop index.

    Args:
        conn: OMERO connection
        row: One row of the tracking table
        channel: Channel to fetch; the caller has resolved label vs training already
        logger: Optional logger

    Returns:
        The image as a numpy array (2D, or 3D for volumetric data), unscaled.
        Normalisation is defined in the config and applied by the caller or the trainer.

    Raises:
        ValueError: If the image is not found in OMERO
    """
    log = logger or _LOGGER
    # Extract metadata
    image_id = int(row["image_id"])

    # Handle z_slice - could be int, string representation of list, or NaN
    z_slice = row["z_slice"]
    if pd.isna(z_slice):
        z_slice = 0
    elif isinstance(z_slice, str) and z_slice.startswith("["):
        try:
            z_slice = ast.literal_eval(z_slice)
            if isinstance(z_slice, list) and len(z_slice) > 0:
                z_slice = z_slice[0]  # Use first slice for 2D
        except (ValueError, SyntaxError):
            z_slice = 0

    # channel is supplied by the caller, which has already resolved label vs training
    timepoint = int(row["timepoint"]) if pd.notna(row["timepoint"]) else 0
    is_volumetric = (
        bool(row["is_volumetric"])
        if "is_volumetric" in row and pd.notna(row["is_volumetric"])
        else False
    )

    # Get patch information
    is_patch = bool(row["is_patch"])
    patch_x = int(row["patch_x"])
    patch_y = int(row["patch_y"])
    patch_width = int(row["patch_width"])
    patch_height = int(row["patch_height"])

    log.debug(
        f"Image {image_id}: patch={is_patch} {patch_width}x{patch_height} "
        f"at ({patch_x},{patch_y}), volumetric={is_volumetric}, channel={channel}"
    )

    # Process based on whether it's 3D volumetric or 2D
    if is_volumetric:
        # Handle 3D volumetric data
        # Determine which z-slices to load
        if isinstance(z_slice, list):
            z_slices = z_slice
        elif z_slice == "all":
            # Get image object to determine size
            omero_image, _ = ezomero.get_image(conn, image_id, no_pixels=True)
            if not omero_image:
                raise ValueError(f"Image {image_id} not found in OMERO")
            z_slices = range(omero_image.getSizeZ())
        else:
            z_slices = [int(z_slice)]

        # Create empty 3D array to hold all z-slices
        img_3d = []

        # Pre-fetch image dimensions for non-patch full plane extraction
        # (avoids repeated fetches and unreliable locals() check in loop)
        size_x = None
        size_y = None

        # Load each z-slice using ezomero.get_image
        for z in z_slices:
            z_val = int(z)
            if is_patch and patch_width > 0 and patch_height > 0:
                # Debug start_coords and axis_lengths
                log.debug(
                    f"3D patch request: start ({patch_x}, {patch_y}, {z_val}, {channel}, "
                    f"{timepoint}), size {patch_width}x{patch_height}"
                )
                # Use ezomero.get_image to extract the patch for this z-slice
                _, img_slice = ezomero.get_image(
                    conn,
                    image_id,
                    start_coords=(patch_x, patch_y, z_val, channel, timepoint),
                    axis_lengths=(patch_width, patch_height, 1, 1, 1),
                    xyzct=True,  # Use XYZCT ordering
                )

                # Check shape of returned array
                log.debug(
                    f"Returned array shape (before extraction): {img_slice.shape}"
                )
                # The result will be 5D, extract just the 2D slice.
                # ezomero hands back XYZCT, so x is axis 0; swap to (Y, X) like every
                # other branch does. Omitting this swap here transposed volumetric
                # patches against their labels.
                if len(img_slice.shape) == 5:
                    img_slice = img_slice[:, :, 0, 0, 0]
                    img_slice = np.swapaxes(img_slice, 0, 1)
                log.debug(f"Extracted slice shape: {img_slice.shape}")
            else:
                # Get full plane for this z-slice
                # Get image dimensions if not already obtained
                if size_x is None:
                    omero_image, _ = ezomero.get_image(conn, image_id, no_pixels=True)
                    size_x = omero_image.getSizeX()
                    size_y = omero_image.getSizeY()

                _, img_slice = ezomero.get_image(
                    conn,
                    image_id,
                    start_coords=(0, 0, z_val, channel, timepoint),
                    axis_lengths=(size_x, size_y, 1, 1, 1),
                    xyzct=True,  # Use XYZCT ordering
                )
                # Check shape of returned array
                log.debug(f"Full plane shape (before extraction): {img_slice.shape}")
                # The result will be 5D, extract just the 2D slice
                if len(img_slice.shape) == 5:
                    img_slice = img_slice[:, :, 0, 0, 0]
                    img_slice = np.swapaxes(img_slice, 0, 1)
                log.debug(f"Extracted full plane shape: {img_slice.shape}")
            img_3d.append(img_slice)

        # Convert to numpy array
        img_3d = np.array(img_3d)
        log.debug(f"Final 3D array shape: {img_3d.shape}")
        return img_3d

    # Handle 2D data with patch support using ezomero.get_image
    if is_patch and patch_width > 0 and patch_height > 0:
        # Use ezomero.get_image with appropriate coordinates and dimensions
        z_val = z_slice if not isinstance(z_slice, list) else z_slice[0]

        # Debug start_coords and axis_lengths
        log.debug(
            f"2D patch request: start ({patch_x}, {patch_y}, {z_val}, {channel}, "
            f"{timepoint}), size {patch_width}x{patch_height}"
        )
        _, img_data = ezomero.get_image(
            conn,
            image_id,
            start_coords=(patch_x, patch_y, int(z_val), channel, timepoint),
            axis_lengths=(patch_width, patch_height, 1, 1, 1),
            xyzct=True,
        )

        # Check shape of returned array
        log.debug(f"Returned array shape: {img_data.shape}")
        # The array is already in the right dimensions (width, height, z=1, c=1, t=1)
        # We just need to remove the trailing dimensions
        if len(img_data.shape) == 5:
            # Take only the first (and only) z, c, t indices
            img_data = img_data[:, :, 0, 0, 0]
            # swap x and y dimensions in the numpy array
            img_data = np.swapaxes(img_data, 0, 1)

        log.debug(f"Extracted 2D shape: {img_data.shape}")
    else:
        # Get full plane
        z_val = z_slice if not isinstance(z_slice, list) else z_slice[0]

        # Get image dimensions to specify exact plane size
        omero_image, _ = ezomero.get_image(conn, image_id, no_pixels=True)
        size_x = omero_image.getSizeX()
        size_y = omero_image.getSizeY()

        # Debug start_coords
        log.debug(
            f"2D full image request: start (0, 0, {z_val}, {channel}, {timepoint}), "
            f"size {size_x}x{size_y}"
        )
        _, img_data = ezomero.get_image(
            conn,
            image_id,
            start_coords=(0, 0, int(z_val), channel, timepoint),
            axis_lengths=(size_x, size_y, 1, 1, 1),
            xyzct=True,
        )

        # Check shape of returned array
        log.debug(f"Returned array shape: {img_data.shape}")
        # Remove trailing dimensions
        if len(img_data.shape) == 5:
            img_data = img_data[:, :, 0, 0, 0]
            img_data = np.swapaxes(img_data, 0, 1)

        log.debug(f"Extracted 2D shape: {img_data.shape}")
    return img_data


def _optional_int(value) -> int | None:
    """Parse a table cell that may be int, float, NaN, or the string 'None'."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text in ("None", "nan", ""):
        return None
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return None


def _channel_from_row(row) -> int:
    """
    The channel for this row, defaulting to 0 when unset.

    A present-but-unparseable value raises rather than silently falling back to
    channel 0: training on the wrong channel is worse than failing loudly.
    """
    raw = row.get("channel")
    if raw is None or pd.isna(raw):
        return 0
    if str(raw).strip() in ("None", "nan", ""):
        return 0

    parsed = _optional_int(raw)
    if parsed is None:
        raise ValueError(f"Unparseable channel value in tracking table: {raw!r}")
    return parsed


def _load_table(conn, table_id: int) -> pd.DataFrame:
    """Fetch the tracking table as a DataFrame, with its columns named as we expect."""
    from ..omero.omero_functions import read_tracking_table

    return read_tracking_table(conn, table_id)


def _download_label(conn, label_id, tmp_dir: Path, logger=None) -> Path | None:
    """
    Download a label file annotation. Returns None if it is missing or unreadable.

    Returning None rather than raising lets the caller drop the whole record. Writing
    the image without its label would leave an orphan, and consumers pair images to
    labels by sorted filename - so one orphan shifts every subsequent pair.
    """
    if label_id is None:
        return None
    try:
        file_ann = conn.getObject("FileAnnotation", label_id)
        if file_ann is None:
            if logger:
                logger.warning(f"File annotation {label_id} not found in OMERO")
            return None
        file_path = ezomero.get_file_annotation(conn, label_id, str(tmp_dir))
        return Path(file_path) if file_path else None
    except Exception as e:  # noqa: BLE001 - a missing label skips the record
        if logger:
            logger.error(f"Error downloading label {label_id}: {e}")
        return None


def _records_from_table(
    conn,
    df: pd.DataFrame,
    tmp_dir: Path,
    uses_separate_channels: bool,
    label_channel: int | None,
    train_channel: int | None,
    logger=None,
    transform: Callable[[str, np.ndarray], np.ndarray] | None = None,
):
    """
    Build annotation records from a tracking table.

    ``transform(annotation_id, image)`` is applied to the model input image (used for
    normalisation at export); the label and annotation-channel images are untouched.

    A row whose label cannot be downloaded is dropped entirely. The image plane is
    fetched lazily (ArraySource), so dropping the record costs nothing - and nothing
    is written for it.

    Raises:
        IntegrityError: A label file's pixels no longer match the ``mask_digest``
            recorded when it was annotated.

    Returns:
        (records, n_missing)
    """
    records = []
    n_missing = 0
    changed: list[str] = []

    for _, row in df.iterrows():
        annotation_id = str(row["annotation_id"])

        label_id = _optional_int(row.get("label_id"))
        label_path = _download_label(conn, label_id, tmp_dir, logger)
        if label_path is None:
            n_missing += 1
            if logger:
                logger.warning(f"Skipping annotation {annotation_id}: no label")
            continue
        recorded = str(row.get("mask_digest") or "")
        if recorded and recorded != "nan" and file_digest(label_path) != recorded:
            changed.append(annotation_id)

        category = str(row["category"])

        image_channel = (
            train_channel if train_channel is not None else _channel_from_row(row)
        )

        # Bind row/channel per iteration: a bare closure would capture the final values.
        def load_image(row=row, ch=image_channel, annotation_id=str(annotation_id)):
            image = _fetch_plane(conn, row, ch, logger)
            return transform(annotation_id, image) if transform else image

        image_source = ArraySource(load_image)

        annotation_source = None
        if uses_separate_channels and label_channel is not None:
            annotation_source = ArraySource(
                lambda row=row, ch=label_channel: _fetch_plane(conn, row, ch, logger)
            )

        records.append(
            AnnotationRecord(
                annotation_id=annotation_id,
                category=category,
                image=image_source,
                label=FileSource(label_path),
                annotation_image=annotation_source,
            )
        )

    if changed:
        raise IntegrityError(
            f"{len(changed)} label file(s) on OMERO no longer match the masks recorded "
            f"when they were annotated: {', '.join(changed[:5])}"
            + (" ..." if len(changed) > 5 else "")
        )
    return records, n_missing


def reorganize_local_data_for_training(
    config: "AnnotationConfig",
    annotation_dir: str | Path,
    output_dir: str | Path | None = None,
    file_mode: Literal["copy", "move", "symlink"] = "copy",
    clean_existing: bool = True,
    include_test: bool | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Reorganize locally-stored annotation data into the training layout.

    Works entirely offline - no OMERO connection required. Reads the annotation phase's
    folders (annotation_input/, model_input/, annotation_output/) and writes the training
    layout (train_input/, train_label/, val_input/, val_label/).

    Args:
        config: The training set's config (channels, normalisation)
        annotation_dir: The training set's folder (``TrainingSet.bundle_dir``); its
            ``tracking_table.csv`` says which units are annotated and their split
        output_dir: Target directory. Must NOT be inside annotation_dir - cleaning the
            training layout there would delete the source images. Defaults to the sibling
            <annotation_dir>_training/.
        file_mode: "copy" (default), "move", or "symlink". Symlinks fall back to a copy
            where they are unavailable (e.g. Windows without Developer Mode).
        clean_existing: Remove the training folders before writing
        include_test: Write a test split. None (default) auto-detects from the annotations.
        verbose: Show detailed progress

    Normalisation follows ``config.normalization``: with ``apply_at_export`` the model
    input images are written as float32 (file_mode then applies to labels only) and the
    bounds per image go to ``normalization.json``; otherwise images keep their native bit
    depth and ``normalization.json`` records what the trainer should apply.

    Returns:
        Dictionary with the created directories and statistics. Keys include base_dir,
        train_input, train_label, val_input, val_label and stats.

    Raises:
        ValueError: If config has no processed annotations, or output_dir is inside annotation_dir
        FileNotFoundError: If annotation_dir does not exist
    """
    annotation_dir = Path(annotation_dir)
    if output_dir is None:
        output_dir = annotation_dir.parent / f"{annotation_dir.name}_training"
    output_dir = Path(output_dir)

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    assert_output_dir_is_separate(output_dir, annotation_dir)

    logger = create_training_logger(output_dir, verbose=verbose)
    logger.info("Reorganizing local annotation data for training")

    table_file = annotation_dir / "tracking_table.csv"
    if not table_file.exists():
        raise FileNotFoundError(f"No tracking_table.csv in: {annotation_dir}")
    rows = rows_from_frame(pd.read_csv(table_file, keep_default_na=False))
    processed_annotations = [row for row in rows if row.processed]
    if not processed_annotations:
        raise ValueError(f"No annotated units in {table_file}")

    uses_separate_channels = config.spatial_coverage.uses_separate_channels()

    if include_test is None:
        include_test = any(ann.category == "test" for ann in processed_annotations)

    annotation_input_source = annotation_dir / "annotation_input"
    model_input_source = annotation_dir / "model_input"
    label_source = annotation_dir / "annotation_output"

    if not annotation_input_source.exists():
        raise FileNotFoundError(
            f"No annotation_input/ folder found in: {annotation_dir}"
        )
    if not label_source.exists():
        raise FileNotFoundError(f"Output folder not found: {label_source}")

    spec = config.normalization
    recorder = NormalizationRecorder(spec) if spec.apply_at_export else None

    records = []
    n_missing = 0
    for ann in processed_annotations:
        annotation_id = ann.annotation_id

        # The model's input is the model_input channel when there is one; in
        # single-channel mode annotation_input serves both roles.
        annotation_file = annotation_input_source / f"{annotation_id}.tif"
        if uses_separate_channels:
            image_file = model_input_source / f"{annotation_id}.tif"
        else:
            image_file = annotation_file

        label_file = label_source / f"{annotation_id}_mask.tif"

        # Drop the record whole if either half is missing. Writing an image without
        # its label would leave an orphan, and consumers that pair images to labels
        # by sorted filename would then mispair everything after it.
        if not image_file.exists() or not label_file.exists():
            n_missing += 1
            missing = "image" if not image_file.exists() else "label"
            logger.warning(f"Skipping annotation {annotation_id}: missing {missing}")
            continue

        annotation_source = None
        if uses_separate_channels and annotation_file.exists():
            annotation_source = FileSource(annotation_file)

        if recorder is not None:
            image_source = ArraySource(
                lambda f=image_file, a=str(annotation_id): recorder(a, imread(f))
            )
        else:
            image_source = FileSource(image_file)

        records.append(
            AnnotationRecord(
                annotation_id=annotation_id,
                category=ann.category,
                image=image_source,
                label=FileSource(label_file),
                annotation_image=annotation_source,
            )
        )

    created_dirs, stats = write_training_layout(
        records,
        output_dir,
        layout="split",
        file_mode=file_mode,
        clean_existing=clean_existing,
        include_test=include_test,
        uses_separate_channels=uses_separate_channels,
        logger=logger,
    )
    stats["n_missing"] = n_missing
    if recorder is not None:
        recorder.write_sidecar(output_dir)
    else:
        write_native_sidecar(spec, output_dir)

    # An empty training set must be loud. prepare_training_data_from_table() already
    # raises here; without the same guard, a producer/consumer folder mismatch returns
    # valid-looking paths to empty directories and training silently does nothing.
    #
    # Key this off what was actually WRITTEN, not off len(records): a record can be
    # built and then written nowhere (every annotation is category "test" and
    # include_test=False), which leaves the layout just as empty.
    n_written = (
        stats["n_training_images"] + stats["n_val_images"] + stats["n_test_images"]
    )
    if n_written == 0:
        expected_images = annotation_dir / (
            "model_input" if uses_separate_channels else "annotation_input"
        )
        logger.error(
            f"Reorganization FAILED in {output_dir}: 0 images written "
            f"({n_missing} skipped as missing, {stats['n_skipped']} skipped by category)"
        )
        raise ValueError(
            f"Reorganization produced no training data: nothing was written.\n"
            f"  {n_missing} annotation(s) skipped - image or label not found. "
            f"Expected images in {expected_images}/ and masks in "
            f"{annotation_dir / 'annotation_output'}/.\n"
            f"  {stats['n_skipped']} annotation(s) skipped by category "
            f"(test annotations are skipped unless include_test=True)."
        )

    logger.info(
        f"Reorganization complete: {len(records)} annotations written, {n_missing} skipped"
    )

    return _build_standard_result(
        base_dir=output_dir, created_dirs=created_dirs, stats=stats
    )


def default_export_dir(
    ref: int | str | Path, name: str, home: str | Path | None = None
) -> Path:
    """Where a set's training export goes by default: ``<home>/exports/<name>_<ref>``.

    Outside the set's folder, so cleaning an export never touches the set.
    """
    from ..bundle import home_dir

    return (
        home_dir(home)
        / "exports"
        / f"{name}_{ref if isinstance(ref, int) else 'local'}"
    )


def export_training_set(
    ref: int | str | Path,
    output_dir: str | Path,
    conn: Any = None,
    *,
    training_channels: list[int] | None = None,
    label_channel: int | None = None,
    apply_normalization: bool | None = None,
) -> dict[str, Any]:
    """Write a training set's annotated units into the training layout.

    One entry point for the widget, the marimo notebook and scripts. ``ref`` is a
    table ID (the data comes from OMERO, see :func:`prepare_training_data_from_table`)
    or a set's folder (offline, see :func:`reorganize_local_data_for_training`).
    Channels and normalisation default to the set's config; ``apply_normalization``
    overrides ``normalization.apply_at_export``. A local set exports the channels its
    folder holds, so other channels are refused there.

    Returns:
        The export result: the created folders and ``stats``.
    """
    from ..trainingset import read_set_config  # the core imports this module

    config = read_set_config(ref, conn).model_copy(deep=True)
    coverage = config.spatial_coverage
    if not isinstance(ref, int) and (
        (label_channel is not None and label_channel != coverage.get_label_channel())
        or (
            training_channels is not None
            and training_channels != coverage.get_training_channels()
        )
    ):
        raise ValueError(
            "A local training set exports the channels it was annotated with "
            f"(label {coverage.get_label_channel()}, training "
            f"{coverage.get_training_channels()})"
        )
    if label_channel is not None:
        coverage.label_channel = label_channel
    if training_channels is not None:
        coverage.training_channels = training_channels
    if apply_normalization is not None:
        config.normalization.apply_at_export = apply_normalization
    if isinstance(ref, int):
        return prepare_training_data_from_table(
            conn,
            ref,
            output_dir,
            training_name=config.name,
            label_channel=coverage.get_label_channel(),
            training_channels=coverage.get_training_channels(),
            normalization=config.normalization,
        )
    return reorganize_local_data_for_training(config, Path(ref), output_dir)
