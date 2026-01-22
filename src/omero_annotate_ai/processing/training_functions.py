"""Training data preparation functions for micro-SAM workflows."""

import shutil
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import ezomero
import numpy as np
import pandas as pd
from tifffile import imwrite
from tqdm import tqdm

from ..utils.logging import create_training_logger
from .utils import validate_table_schema

if TYPE_CHECKING:
    from ..core.annotation_config import AnnotationConfig


def _get_standard_folder_structure(
    uses_separate_channels: bool = False, include_test: bool = False
) -> Dict[str, str]:
    """
    Get the standard folder structure for training data.

    Args:
        uses_separate_channels: Whether to include label_input folders
        include_test: Whether to include test folders

    Returns:
        Dictionary mapping folder purposes to folder names
    """
    structure = {
        "training_input": "training_input",
        "training_label": "training_label",
        "validation_input": "val_input",
        "validation_label": "val_label",
    }

    if uses_separate_channels:
        structure.update(
            {
                "training_label_input": "training_label_input",
                "validation_label_input": "val_label_input",
            }
        )

    if include_test:
        structure.update(
            {
                "test_input": "test_input",
                "test_label": "test_label",
            }
        )

        if uses_separate_channels:
            structure["test_label_input"] = "test_label_input"

    return structure


def _create_training_directories(
    output_dir: Path,
    uses_separate_channels: bool = False,
    include_test: bool = False,
    clean_existing: bool = True,
) -> Dict[str, Path]:
    """
    Create the standard training directory structure.

    Args:
        output_dir: Base output directory
        uses_separate_channels: Whether to create label_input folders
        include_test: Whether to create test folders
        clean_existing: Whether to remove existing directories first

    Returns:
        Dictionary mapping folder purposes to Path objects
    """
    structure = _get_standard_folder_structure(uses_separate_channels, include_test)
    created_dirs = {}

    # Clean existing directories if requested
    if clean_existing:
        for folder_name in structure.values():
            folder_path = output_dir / folder_name
            if folder_path.exists():
                shutil.rmtree(folder_path)

    # Create all directories
    for purpose, folder_name in structure.items():
        folder_path = output_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        created_dirs[purpose] = folder_path

    return created_dirs


def _build_standard_result(
    base_dir: Path, created_dirs: Dict[str, Path], stats: Dict[str, Any], **extra_fields
) -> Dict[str, Any]:
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
    output_dir: Union[str, Path],
    training_name: str = "micro_sam_training",
    validation_split: float = 0.2,
    clean_existing: bool = True,
    tmp_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    label_channel: Optional[int] = None,
    training_channels: Optional[List[int]] = None,
    upload_label_input: bool = False,
) -> Dict[str, Any]:
    """
    Prepare training data from OMERO annotation table.

    Downloads images and labels from OMERO based on annotation table data,
    splits into training/validation sets, and organizes into directory structure
    suitable for micro-SAM training.

    Args:
        conn: OMERO connection object
        table_id: ID of the annotation table in OMERO
        output_dir: Directory to store training data
        training_name: Name for the training session (used in directory naming)
        validation_split: Fraction of data to use for validation (0.0-1.0) if not already defined in the table
        clean_existing: Whether to clean existing output directories
        tmp_dir: Temporary directory for downloads (optional)
        verbose: If True, show detailed debug information in console output
        label_channel: Optional channel index for label/segmentation images. If provided
            and different from training_channels, downloads label channel images to
            *_label_input directories alongside the training data.
        training_channels: Optional list of channel indices for training input images.
            If different from label_channel, downloads from these channels for
            training_input and val_input. Currently uses first channel if multiple specified.
        upload_label_input: If True and using separate channels, uploads the label_input
            images back to OMERO as file annotations. Default is False.

    Returns:
        Dictionary with paths to created directories:
        {
            'base_dir': Path to base output directory,
            'training_input': Path to training images,
            'training_label': Path to training labels (segmentation masks),
            'training_label_input': Path to label channel images (only if separate channels),
            'val_input': Path to validation images,
            'val_label': Path to validation labels (segmentation masks),
            'val_label_input': Path to label channel images for validation (only if separate channels),
            'stats': Statistics about the prepared data
        }

    Raises:
        ValueError: If table not found or invalid parameters
        ImportError: If required dependencies missing
    """
    # Validate parameters
    if not 0.0 <= validation_split <= 1.0:
        raise ValueError("validation_split must be between 0.0 and 1.0")

    # Convert paths
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create output directory {output_dir}: {e}")

    # Set up logger for this training session
    logger = create_training_logger(output_dir, verbose=verbose)
    logger.info(f"Starting training data preparation from table {table_id}")
    logger.debug(
        f"Parameters: output_dir={output_dir}, validation_split={validation_split}, clean_existing={clean_existing}"
    )

    if tmp_dir is None:
        tmp_dir = output_dir / "tmp"
    tmp_dir = Path(tmp_dir)
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create temporary directory {tmp_dir}: {e}")

    try:
        table = ezomero.get_table(conn, table_id)
    except Exception as e:
        raise ValueError(f"Failed to load table {table_id}: {e}")

    if table is None or len(table) == 0:
        raise ValueError(f"Table {table_id} is empty or not found")

    logger.info(f"Loaded table with {len(table)} rows")

    # Save the table locally for inspection
    table_path = output_dir / f"table_{table_id}.csv"
    try:
        table.to_csv(table_path, index=True)
        logger.info(f"Table saved to: {table_path}")
    except Exception as e:
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
        table = table[table["processed"]].copy()

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
            f"Using separate channels: label_channel={label_channel}, training_channels={training_channels}"
        )

    # Determine the effective training channel to use
    effective_train_channel = training_channels[0] if training_channels else None

    # Create standard directory structure
    created_dirs = _create_training_directories(
        output_dir=output_dir,
        uses_separate_channels=uses_separate_channels,
        include_test=False,  # Table function doesn't support test category
        clean_existing=clean_existing,
    )

    # Split data based on existing 'train'/'validate' columns or automatic split
    if "train" in table.columns and "validate" in table.columns:
        # Use existing split from table
        train_images = table[table["train"]]
        val_images = table[table["validate"]]
        logger.info(f"Using existing train/validate split from table")
    else:
        # Automatic split
        n_val = int(len(table) * validation_split)
        shuffled_indices = np.random.permutation(len(table))
        val_indices = shuffled_indices[:n_val]
        train_indices = shuffled_indices[n_val:]

        train_images = table.iloc[train_indices]
        val_images = table.iloc[val_indices]
        logger.info(f"Applied automatic split with validation_split={validation_split}")

    logger.info(
        f"Using {len(train_images)} training images and {len(val_images)} validation images"
    )

    # Prepare training data (uses training channel if specified)
    training_input_dir, training_label_dir = _prepare_dataset_from_table(
        conn,
        train_images,
        output_dir,
        subset_type="training",
        tmp_dir=tmp_dir,
        train_channel=effective_train_channel,
        logger=logger,
        verbose=verbose,
    )

    # Prepare validation data (uses training channel if specified)
    val_input_dir, val_label_dir = _prepare_dataset_from_table(
        conn,
        val_images,
        output_dir,
        subset_type="val",
        tmp_dir=tmp_dir,
        train_channel=effective_train_channel,
        logger=logger,
        verbose=verbose,
    )

    # Update created_dirs with actual paths from _prepare_dataset_from_table
    created_dirs["training_input"] = training_input_dir
    created_dirs["training_label"] = training_label_dir
    created_dirs["val_input"] = val_input_dir
    created_dirs["val_label"] = val_label_dir

    # Prepare validation data (uses training channel if specified)
    val_input_dir, val_label_dir = _prepare_dataset_from_table(
        conn,
        val_images,
        output_dir,
        subset_type="val",
        tmp_dir=tmp_dir,
        train_channel=effective_train_channel,
        logger=logger,
        verbose=verbose,
    )

    # If using separate channels, also prepare label channel images
    training_label_input_dir = None
    val_label_input_dir = None
    label_input_upload_ids = []
    if uses_separate_channels:
        logger.info(
            f"Preparing label channel ({label_channel}) images for separate channel workflow"
        )
        training_label_input_dir, _ = _prepare_dataset_from_table(
            conn,
            train_images,
            output_dir,
            subset_type="training_label",
            tmp_dir=tmp_dir,
            train_channel=label_channel,
            logger=logger,
            verbose=verbose,
        )
        val_label_input_dir, _ = _prepare_dataset_from_table(
            conn,
            val_images,
            output_dir,
            subset_type="val_label",
            tmp_dir=tmp_dir,
            train_channel=label_channel,
            logger=logger,
            verbose=verbose,
        )

        # Update created_dirs with label_input paths
        created_dirs["training_label_input"] = training_label_input_dir
        created_dirs["val_label_input"] = val_label_input_dir

        # Upload label_input images to OMERO if requested
        if upload_label_input:
            logger.info("Uploading label_input images to OMERO...")
            all_images = pd.concat([train_images, val_images])
            all_label_input_dirs = [training_label_input_dir, val_label_input_dir]

            for label_input_dir in all_label_input_dirs:
                if label_input_dir and label_input_dir.exists():
                    for tif_file in sorted(label_input_dir.glob("*.tif")):
                        # Extract index from filename (e.g., input_00001.tif -> 1)
                        try:
                            file_idx = int(tif_file.stem.split("_")[-1])
                            if file_idx < len(all_images):
                                row = all_images.iloc[file_idx]
                                image_id = int(row["image_id"])
                                timepoint = (
                                    int(row["timepoint"])
                                    if pd.notna(row.get("timepoint"))
                                    else None
                                )
                                z_slice = (
                                    int(row["z_slice"])
                                    if pd.notna(row.get("z_slice"))
                                    else None
                                )

                                # Lazy import to avoid circular dependency
                                from ..omero.omero_functions import (
                                    upload_label_input_image,
                                )

                                file_ann_id = upload_label_input_image(
                                    conn,
                                    image_id=image_id,
                                    label_input_file=str(tif_file),
                                    trainingset_name=training_name,
                                    channel=label_channel,
                                    timepoint=timepoint,
                                    z_slice=z_slice,
                                )
                                label_input_upload_ids.append(file_ann_id)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not upload {tif_file}: {e}")

            logger.info(
                f"Uploaded {len(label_input_upload_ids)} label_input images to OMERO"
            )

    # Clean up temporary directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        logger.debug(f"Cleaned up temporary directory: {tmp_dir}")

    # Collect statistics
    stats = {
        "n_training_images": len(list(training_input_dir.glob("*.tif"))),
        "n_training_labels": len(list(training_label_dir.glob("*.tif"))),
        "n_val_images": len(list(val_input_dir.glob("*.tif"))),
        "n_val_labels": len(list(val_label_dir.glob("*.tif"))),
        "total_rows_processed": len(table),
    }

    # Add label input stats if using separate channels
    if uses_separate_channels:
        stats["n_training_label_input"] = len(
            list(training_label_input_dir.glob("*.tif"))
        )
        stats["n_val_label_input"] = len(list(val_label_input_dir.glob("*.tif")))
        if label_input_upload_ids:
            stats["n_label_input_uploaded"] = len(label_input_upload_ids)

    # Build standard result dictionary
    extra_fields = {}
    if label_input_upload_ids:
        extra_fields["label_input_upload_ids"] = label_input_upload_ids

    result = _build_standard_result(
        base_dir=output_dir, created_dirs=created_dirs, stats=stats, **extra_fields
    )

    # Check if preparation actually succeeded
    if stats["n_training_images"] == 0 and stats["n_val_images"] == 0:
        logger.error(f"Training data preparation FAILED in: {output_dir}")
        logger.error(f"Statistics: {stats}")
        raise ValueError(
            "Training data preparation failed - no images were processed successfully. Check the error messages above."
        )
    else:
        logger.info(f"Training data prepared successfully in: {output_dir}")
        logger.info(f"Statistics: {stats}")

    return result


def prepare_training_data_from_config(
    conn: Any,
    config: "AnnotationConfig",  # Forward reference to avoid circular import
    output_dir: Union[str, Path],
    training_name: str = "micro_sam_training",
    clean_existing: bool = True,
    tmp_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Prepare training data directly from an AnnotationConfig object.

    This function allows preparing training data from a AnnotationConfig object that has been
    loaded from a YAML file with annotations already populated from a previous
    workflow. It avoids the need to have an OMERO table.

    Args:
        conn: OMERO connection object (required for downloading images/labels)
        config: AnnotationConfig with populated annotations list
        output_dir: Directory to store training data
        training_name: Name for the training session (used in directory naming)
        clean_existing: Whether to clean existing output directories
        tmp_dir: Temporary directory for downloads (optional)
        verbose: If True, show detailed debug information

    Returns:
        Dictionary with paths to created directories:
        {
            'base_dir': Path to base output directory,
            'training_input': Path to training images,
            'training_label': Path to training labels,
            'val_input': Path to validation images,
            'val_label': Path to validation labels,
            'stats': Statistics about the prepared data
        }

    Raises:
        ValueError: If config has no annotations or no processed annotations
    """
    # Validate config has annotations
    if not config.annotations:
        raise ValueError("Config has no annotations. Run annotation workflow first.")

    # Convert config annotations to DataFrame
    df = config.to_dataframe()

    if df.empty:
        raise ValueError("Config annotations converted to empty DataFrame")

    # Convert paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = create_training_logger(output_dir, verbose=verbose)
    logger.info(f"Preparing training data from config with {len(df)} annotations")

    if tmp_dir is None:
        tmp_dir = output_dir / "tmp"
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Validate the DataFrame schema
    validate_table_schema(df, logger)
    logger.info("Config DataFrame schema validated")

    # Check if 'processed' column exists and filter to only processed rows
    if "processed" in df.columns:
        initial_count = len(df)
        df = df[df["processed"]].copy()
        if len(df) == 0:
            raise ValueError("No processed annotations found in config")
        logger.info(
            f"Using {len(df)} processed annotations out of {initial_count} total"
        )

    # Determine if using separate channels
    uses_separate_channels = config.spatial_coverage.uses_separate_channels()
    label_channel = (
        config.spatial_coverage.get_label_channel() if uses_separate_channels else None
    )
    training_channels = (
        config.spatial_coverage.get_training_channels()
        if uses_separate_channels
        else None
    )
    effective_train_channel = training_channels[0] if training_channels else None

    if uses_separate_channels:
        logger.info(
            f"Using separate channels: label={label_channel}, training={training_channels}"
        )

    # Create standard directory structure
    created_dirs = _create_training_directories(
        output_dir=output_dir,
        uses_separate_channels=uses_separate_channels,
        include_test=False,  # Config function doesn't support test category
        clean_existing=clean_existing,
    )

    # Split data based on 'train'/'validate' columns
    if "train" in df.columns and "validate" in df.columns:
        train_images = df[df["train"]]
        val_images = df[df["validate"]]
        logger.info(
            f"Using train/validate split from config: {len(train_images)} train, {len(val_images)} val"
        )
    else:
        # All data is training data if no split info
        train_images = df
        val_images = pd.DataFrame()
        logger.warning("No train/validate columns - using all data for training")

    logger.info(
        f"Preparing {len(train_images)} training and {len(val_images)} validation images"
    )

    # Prepare datasets using existing internal function
    training_input_dir, training_label_dir = _prepare_dataset_from_table(
        conn,
        train_images,
        output_dir,
        subset_type="training",
        tmp_dir=tmp_dir,
        train_channel=effective_train_channel,
        logger=logger,
        verbose=verbose,
    )

    val_input_dir, val_label_dir = (
        _prepare_dataset_from_table(
            conn,
            val_images,
            output_dir,
            subset_type="val",
            tmp_dir=tmp_dir,
            train_channel=effective_train_channel,
            logger=logger,
            verbose=verbose,
        )
        if len(val_images) > 0
        else (None, None)
    )

    # Update created_dirs with actual paths from _prepare_dataset_from_table
    created_dirs["training_input"] = training_input_dir
    created_dirs["training_label"] = training_label_dir
    if val_input_dir is not None:
        created_dirs["val_input"] = val_input_dir
        created_dirs["val_label"] = val_label_dir

    # Handle separate channel workflow if needed
    training_label_input_dir = None
    val_label_input_dir = None
    if uses_separate_channels:
        logger.info(f"Preparing label channel ({label_channel}) images")
        training_label_input_dir, _ = _prepare_dataset_from_table(
            conn,
            train_images,
            output_dir,
            subset_type="training_label",
            tmp_dir=tmp_dir,
            train_channel=label_channel,
            logger=logger,
            verbose=verbose,
        )
        if len(val_images) > 0:
            val_label_input_dir, _ = _prepare_dataset_from_table(
                conn,
                val_images,
                output_dir,
                subset_type="val_label",
                tmp_dir=tmp_dir,
                train_channel=label_channel,
                logger=logger,
                verbose=verbose,
            )

        # Update created_dirs with label_input paths
        created_dirs["training_label_input"] = training_label_input_dir
        if val_label_input_dir is not None:
            created_dirs["val_label_input"] = val_label_input_dir

    # Compute statistics
    stats = {
        "n_training_images": len(list(training_input_dir.glob("*.tif")))
        if training_input_dir
        else 0,
        "n_training_labels": len(list(training_label_dir.glob("*.tif")))
        if training_label_dir
        else 0,
        "n_val_images": len(list(val_input_dir.glob("*.tif"))) if val_input_dir else 0,
        "n_val_labels": len(list(val_label_dir.glob("*.tif"))) if val_label_dir else 0,
    }

    # Build standard result dictionary
    result = _build_standard_result(
        base_dir=output_dir, created_dirs=created_dirs, stats=stats
    )

    if stats["n_training_images"] == 0:
        logger.error("Training data preparation FAILED - no images processed")
        raise ValueError("Training data preparation failed - no images were processed")

    logger.info(f"Training data prepared successfully from config: {stats}")
    return result


def _prepare_dataset_from_table(
    conn,
    df: pd.DataFrame,
    output_dir: Path,
    subset_type: str = "training",
    tmp_dir: Optional[Path] = None,
    train_channel: Optional[int] = None,
    logger=None,
    verbose: bool = False,
) -> Tuple[Path, Path]:
    """
    Prepare dataset from annotation table subset.

    Args:
        conn: OMERO connection
        df: DataFrame with annotation info
        output_dir: Base output directory
        subset_type: "training" or "val"
        tmp_dir: Temporary directory for downloading annotations
        train_channel: Optional channel for annotation, then override
        logger: Logger instance for logging messages
        verbose: If True, show debug messages when no logger available

    Returns:
        (input_dir, label_dir): Paths to the input and label directories
    """

    def debug_print(message: str, level: str = "debug"):
        """Helper to print debug messages only if verbose or log to logger."""
        if logger:
            if level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
            else:
                logger.debug(message)
        elif verbose:
            print(f"  {message}")

    if tmp_dir is None:
        tmp_dir = output_dir / "tmp"
        try:
            tmp_dir.mkdir(exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to create temporary directory {tmp_dir}: {e}")

    input_dir = output_dir / f"{subset_type}_input"
    label_dir = output_dir / f"{subset_type}_label"
    try:
        input_dir.mkdir(exist_ok=True)
        label_dir.mkdir(exist_ok=True)
    except Exception as e:
        raise OSError(
            f"Failed to create dataset directories {input_dir}, {label_dir}: {e}"
        )

    if logger:
        logger.info(f"Preparing {subset_type} dataset: {len(df)} items to process")

    for n in tqdm(range(len(df)), desc=f"Preparing {subset_type} data"):
        try:
            # Extract metadata
            image_id = int(df.iloc[n]["image_id"])

            # Handle z_slice - could be int, string representation of list, or NaN
            z_slice = df.iloc[n]["z_slice"]
            if pd.isna(z_slice):
                z_slice = 0
            elif isinstance(z_slice, str) and z_slice.startswith("["):
                try:
                    z_slice = eval(z_slice)
                    if isinstance(z_slice, list) and len(z_slice) > 0:
                        z_slice = z_slice[0]  # Use first slice for 2D
                except Exception:
                    z_slice = 0

            # Handle other metadata columns
            if train_channel is not None:
                channel = train_channel
            else:
                channel = (
                    int(df.iloc[n]["channel"]) if pd.notna(df.iloc[n]["channel"]) else 0
                )
            timepoint = (
                int(df.iloc[n]["timepoint"]) if pd.notna(df.iloc[n]["timepoint"]) else 0
            )
            is_volumetric = (
                bool(df.iloc[n]["is_volumetric"])
                if "is_volumetric" in df.columns
                and pd.notna(df.iloc[n]["is_volumetric"])
                else False
            )

            # Get patch information
            is_patch = bool(df.iloc[n]["is_patch"])
            patch_x = int(df.iloc[n]["patch_x"])
            patch_y = int(df.iloc[n]["patch_y"])
            patch_width = int(df.iloc[n]["patch_width"])
            patch_height = int(df.iloc[n]["patch_height"])

            # Debug patch dimensions
            debug_print(
                f"Item {n} - Image ID: {image_id}, Patch: {is_patch}, Dimensions: {patch_width}x{patch_height} at ({patch_x},{patch_y}), Volumetric: {is_volumetric}"
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
                        if logger:
                            logger.warning(f"Image {image_id} not found, skipping")
                        else:
                            print(f"Warning: Image {image_id} not found, skipping")
                        continue
                    z_slices = range(omero_image.getSizeZ())
                else:
                    z_slices = [int(z_slice)]

                # Create empty 3D array to hold all z-slices
                img_3d = []

                # Load each z-slice using ezomero.get_image
                for z in z_slices:
                    z_val = int(z)
                    if is_patch and patch_width > 0 and patch_height > 0:
                        # Debug start_coords and axis_lengths
                        if logger:
                            logger.debug(
                                f"3D Patch Request - start_coords: ({patch_x}, {patch_y}, {z_val}, {channel}, {timepoint}), dimensions: {patch_width}x{patch_height}"
                            )
                        else:
                            print(
                                f"  3D Patch Request - start_coords: ({patch_x}, {patch_y}, {z_val}, {channel}, {timepoint}), dimensions: {patch_width}x{patch_height}"
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
                        if logger:
                            logger.debug(
                                f"Returned array shape (before extraction): {img_slice.shape}"
                            )
                        else:
                            print(
                                f"  Returned array shape (before extraction): {img_slice.shape}"
                            )

                        # The result will be 5D, extract just the 2D slice
                        img_slice = img_slice[
                            :, :, 0, 0, 0
                        ]  # Extract the single z-slice
                        if logger:
                            logger.debug(f"Extracted slice shape: {img_slice.shape}")
                        else:
                            print(f"  Extracted slice shape: {img_slice.shape}")
                    else:
                        # Get full plane for this z-slice
                        # Get image dimensions if not already obtained
                        if "size_x" not in locals():
                            omero_image, _ = ezomero.get_image(
                                conn, image_id, no_pixels=True
                            )
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
                        if logger:
                            logger.debug(
                                f"Full plane shape (before extraction): {img_slice.shape}"
                            )
                        else:
                            print(
                                f"  Full plane shape (before extraction): {img_slice.shape}"
                            )

                        # The result will be 5D, extract just the 2D slice
                        if len(img_slice.shape) == 5:
                            img_slice = img_slice[:, :, 0, 0, 0]
                            img_slice = np.swapaxes(img_slice, 0, 1)
                        if logger:
                            logger.debug(
                                f"Extracted full plane shape: {img_slice.shape}"
                            )
                        else:
                            print(f"  Extracted full plane shape: {img_slice.shape}")

                    img_3d.append(img_slice)

                # Convert to numpy array
                img_3d = np.array(img_3d)
                if logger:
                    logger.debug(f"Final 3D array shape: {img_3d.shape}")
                else:
                    print(f"  Final 3D array shape: {img_3d.shape}")

                # Normalize to 8-bit
                max_val = img_3d.max()
                if max_val > 0:
                    img_8bit = ((img_3d) * (255.0 / max_val)).astype(np.uint8)
                else:
                    img_8bit = img_3d.astype(np.uint8)

                # Save as multi-page TIFF for 3D data
                output_path = input_dir / f"input_{n:05d}.tif"
                imwrite(str(output_path), img_8bit)
                if logger:
                    logger.debug(
                        f"Saved 3D TIFF to {output_path} with shape {img_8bit.shape}"
                    )
                else:
                    print(
                        f"  Saved 3D TIFF to {output_path} with shape {img_8bit.shape}"
                    )

            else:
                # Handle 2D data with patch support using ezomero.get_image
                if is_patch and patch_width > 0 and patch_height > 0:
                    # Use ezomero.get_image with appropriate coordinates and dimensions
                    z_val = z_slice if not isinstance(z_slice, list) else z_slice[0]

                    # Debug start_coords and axis_lengths
                    if logger:
                        logger.debug(
                            f"2D Patch Request - start_coords: ({patch_x}, {patch_y}, {z_val}, {channel}, {timepoint}), dimensions: {patch_width}x{patch_height}"
                        )
                    else:
                        print(
                            f"  2D Patch Request - start_coords: ({patch_x}, {patch_y}, {z_val}, {channel}, {timepoint}), dimensions: {patch_width}x{patch_height}"
                        )

                    _, img_data = ezomero.get_image(
                        conn,
                        image_id,
                        start_coords=(patch_x, patch_y, int(z_val), channel, timepoint),
                        axis_lengths=(patch_width, patch_height, 1, 1, 1),
                        xyzct=True,
                    )

                    # Check shape of returned array
                    if logger:
                        logger.debug(f"Returned array shape: {img_data.shape}")
                    else:
                        print(f"  Returned array shape: {img_data.shape}")

                    # The array is already in the right dimensions (width, height, z=1, c=1, t=1)
                    # We just need to remove the trailing dimensions
                    if len(img_data.shape) == 5:
                        # Take only the first (and only) z, c, t indices
                        img_data = img_data[:, :, 0, 0, 0]
                        # swap x and y dimensions in the numpy array
                        img_data = np.swapaxes(img_data, 0, 1)

                    if logger:
                        logger.debug(f"Extracted 2D shape: {img_data.shape}")
                    else:
                        print(f"  Extracted 2D shape: {img_data.shape}")
                else:
                    # Get full plane
                    z_val = z_slice if not isinstance(z_slice, list) else z_slice[0]

                    # Get image dimensions to specify exact plane size
                    omero_image, _ = ezomero.get_image(conn, image_id, no_pixels=True)
                    size_x = omero_image.getSizeX()
                    size_y = omero_image.getSizeY()

                    # Debug start_coords
                    if logger:
                        logger.debug(
                            f"2D Full Image Request - start_coords: (0, 0, {z_val}, {channel}, {timepoint}), dimensions: {size_x}x{size_y}"
                        )
                    else:
                        print(
                            f"  2D Full Image Request - start_coords: (0, 0, {z_val}, {channel}, {timepoint}), dimensions: {size_x}x{size_y}"
                        )

                    _, img_data = ezomero.get_image(
                        conn,
                        image_id,
                        start_coords=(0, 0, int(z_val), channel, timepoint),
                        axis_lengths=(size_x, size_y, 1, 1, 1),
                        xyzct=True,
                    )

                    # Check shape of returned array
                    if logger:
                        logger.debug(f"Returned array shape: {img_data.shape}")
                    else:
                        print(f"  Returned array shape: {img_data.shape}")

                    # Remove trailing dimensions
                    if len(img_data.shape) == 5:
                        img_data = img_data[:, :, 0, 0, 0]
                        img_data = np.swapaxes(img_data, 0, 1)

                    if logger:
                        logger.debug(f"Extracted 2D shape: {img_data.shape}")
                    else:
                        print(f"  Extracted 2D shape: {img_data.shape}")

                # Normalize to 8-bit
                # TODO make this optional; not always need 8-bit I guess
                max_val = img_data.max()
                if max_val > 0:
                    img_8bit = ((img_data) * (255.0 / max_val)).astype(np.uint8)
                else:
                    img_8bit = img_data.astype(np.uint8)

                # Save as TIFF
                output_path = input_dir / f"input_{n:05d}.tif"
                imwrite(str(output_path), img_8bit)
                if logger:
                    logger.debug(
                        f"Saved 2D TIFF to {output_path} with shape {img_8bit.shape}"
                    )
                else:
                    print(
                        f"  Saved 2D TIFF to {output_path} with shape {img_8bit.shape}"
                    )

            # Get label file (already normalized to int or NaN)
            label_id_val = df.iloc[n]["label_id"]
            if pd.notna(label_id_val):
                label_id = int(label_id_val)

                try:
                    # First, check if the file annotation exists
                    if logger:
                        logger.debug(
                            f"Attempting to download label with ID: {label_id}"
                        )
                    else:
                        print(f"  Attempting to download label with ID: {label_id}")

                    # Try to get the file annotation object first to validate it exists
                    try:
                        file_ann = conn.getObject("FileAnnotation", label_id)
                        if file_ann is None:
                            if logger:
                                logger.warning(
                                    f"File annotation {label_id} not found in OMERO"
                                )
                            else:
                                print(
                                    f"  Warning: File annotation {label_id} not found in OMERO"
                                )
                            continue
                        if logger:
                            logger.debug(
                                f"File annotation found: {file_ann.getFile().getName()}"
                            )
                        else:
                            print(
                                f"  File annotation found: {file_ann.getFile().getName()}"
                            )
                    except Exception as check_e:
                        if logger:
                            logger.error(
                                f"Error checking file annotation {label_id}: {check_e}"
                            )
                        else:
                            print(
                                f"  Error checking file annotation {label_id}: {check_e}"
                            )
                        continue

                    # Now try to download it using ezomero
                    file_path = ezomero.get_file_annotation(
                        conn, label_id, str(tmp_dir)
                    )
                    if file_path:
                        label_dest = label_dir / f"label_{n:05d}.tif"
                        shutil.move(file_path, str(label_dest))

                        # Check the size of the saved label
                        from tifffile import imread

                        label_img = imread(str(label_dest))
                        if logger:
                            logger.debug(
                                f"Label shape: {label_img.shape} saved to {label_dest}"
                            )
                        else:
                            print(
                                f"  Label shape: {label_img.shape} saved to {label_dest}"
                            )
                    else:
                        if logger:
                            logger.warning(
                                f"Label file for image {image_id} not downloaded (ezomero returned None)"
                            )
                        else:
                            print(
                                f"  Warning: Label file for image {image_id} not downloaded (ezomero returned None)"
                            )

                except Exception as e:
                    if logger:
                        logger.error(f"Error downloading label file {label_id}: {e}")
                        logger.debug(f"Full traceback: {traceback.format_exc()}")
                    else:
                        print(f"  Error downloading label file {label_id}: {e}")
                        # Print more detailed error information for debugging
                        print(f"  Full traceback: {traceback.format_exc()}")
            else:
                if logger:
                    logger.warning(f"No label ID for image {image_id}")
                else:
                    print(f"  Warning: No label ID for image {image_id}")
        except Exception as e:
            if logger:
                logger.error(f"Error processing row {n}: {e}")
                logger.debug(traceback.format_exc())
            else:
                print(f"Error processing row {n}: {e}")
                print(traceback.format_exc())
            raise
    return input_dir, label_dir


def _create_file_link_or_copy(src: Path, dst: Path, mode: str, logger=None) -> str:
    """
    Create a file at destination using the specified mode.

    Args:
        src: Source file path
        dst: Destination file path
        mode: One of "copy", "move", or "symlink"
        logger: Optional logger for messages

    Returns:
        String describing the action taken (e.g., "symlink", "copy", "copy (symlink fallback)")
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


def reorganize_local_data_for_training(
    config: "AnnotationConfig",
    annotation_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    file_mode: Literal["copy", "move", "symlink"] = "copy",
    clean_existing: bool = True,
    include_test: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Reorganize locally-stored annotation data into training folder structure.

    Works entirely offline - no OMERO connection required. This function takes
    the flat folder structure from the annotation pipeline (input/, output/) and
    reorganizes it into the split-based structure expected by training workflows
    (training_input/, training_label/, val_input/, val_label/).

    Args:
        config: AnnotationConfig with populated annotations (contains category info)
        annotation_dir: Directory containing annotation output (input/, output/ folders)
        output_dir: Target directory for training structure (default: same as annotation_dir)
        file_mode: How to handle files:
            - "copy": Copy files (keeps originals) - default
            - "move": Move files (removes originals)
            - "symlink": Create symbolic links (falls back to copy on Windows if symlinks fail)
        clean_existing: Remove existing training folders before reorganization
        include_test: If True, also create test_input/test_label folders for test category
        verbose: Show detailed progress

    Returns:
        Dictionary with paths to created directories and statistics:
        {
            'base_dir': Path to base output directory,
            'training_input': Path to training images,
            'training_label': Path to training labels,
            'training_label_input': Path to label channel images (only if separate channels),
            'val_input': Path to validation images,
            'val_label': Path to validation labels,
            'val_label_input': Path to validation label channel images (only if separate channels),
            'test_input': Path to test images (only if include_test=True),
            'test_label': Path to test labels (only if include_test=True),
            'stats': Statistics about the reorganized data,
            'file_mapping': Mapping of annotation_id to output files
        }

    Raises:
        ValueError: If config has no annotations or no processed annotations
        FileNotFoundError: If annotation_dir doesn't exist or is missing input/output folders
    """
    # Convert paths
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir) if output_dir else annotation_dir

    # Validate annotation directory structure BEFORE setting up logger
    # (logger tries to create directories which would fail for invalid paths)
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    input_source = annotation_dir / "input"
    output_source = annotation_dir / "output"

    if not input_source.exists():
        raise FileNotFoundError(f"Input folder not found: {input_source}")
    if not output_source.exists():
        raise FileNotFoundError(f"Output folder not found: {output_source}")

    # Set up logger (after validation so we know paths are valid)
    logger = create_training_logger(output_dir, verbose=verbose)
    logger.info("Reorganizing local annotation data for training")
    logger.info(f"Source: {annotation_dir}, Target: {output_dir}, Mode: {file_mode}")

    # Validate config has annotations
    if not config.annotations:
        raise ValueError("Config has no annotations. Run annotation workflow first.")

    # Filter to processed annotations only
    processed_annotations = [ann for ann in config.annotations if ann.processed]
    if not processed_annotations:
        raise ValueError("No processed annotations found in config")

    logger.info(
        f"Found {len(processed_annotations)} processed annotations out of {len(config.annotations)} total"
    )

    # Check if using separate channels
    uses_separate_channels = config.spatial_coverage.uses_separate_channels()
    if uses_separate_channels:
        logger.info(
            "Separate channel workflow detected - will create *_label_input folders"
        )

    # Determine which categories we have
    categories = set(ann.category for ann in processed_annotations)
    logger.info(f"Categories found: {categories}")

    # Create standard directory structure using helper
    created_dirs = _create_training_directories(
        output_dir=output_dir,
        uses_separate_channels=uses_separate_channels,
        include_test=include_test,
        clean_existing=clean_existing,
    )

    # Process annotations by category
    stats: Dict[str, Any] = {
        "n_training_images": 0,
        "n_training_labels": 0,
        "n_val_images": 0,
        "n_val_labels": 0,
        "n_test_images": 0,
        "n_test_labels": 0,
        "n_skipped": 0,
        "n_missing_input": 0,
        "n_missing_label": 0,
        "file_operations": {},
    }

    file_mapping: Dict[str, Dict[str, Any]] = {}
    category_counters: Dict[str, int] = {"training": 0, "validation": 0, "test": 0}

    for ann in processed_annotations:
        category = ann.category

        if category == "test" and not include_test:
            stats["n_skipped"] += 1
            continue

        # Only process known categories
        if category not in ("training", "validation", "test"):
            stats["n_skipped"] += 1
            continue

        annotation_id = ann.annotation_id

        # Find source files
        # Input image: input/{annotation_id}.tif
        input_file = input_source / f"{annotation_id}.tif"
        if not input_file.exists():
            # Try .tiff extension
            input_file = input_source / f"{annotation_id}.tiff"

        # Label/mask file: output/{annotation_id}_mask.tif
        label_file = output_source / f"{annotation_id}_mask.tif"
        if not label_file.exists():
            label_file = output_source / f"{annotation_id}_mask.tiff"

        # Get sequential index for this category
        idx = category_counters[category]
        category_counters[category] += 1

        # Determine destination paths using standard folder names from helper
        folder_structure = _get_standard_folder_structure(
            uses_separate_channels, include_test
        )
        input_folder = folder_structure.get(f"{category}_input", f"{category}_input")
        label_folder = folder_structure.get(f"{category}_label", f"{category}_label")

        input_dest = output_dir / input_folder / f"input_{idx:05d}.tif"
        label_dest = output_dir / label_folder / f"label_{idx:05d}.tif"

        # Track mapping
        file_mapping[annotation_id] = {
            "category": category,
            "index": idx,
            "input_dest": str(input_dest),
            "label_dest": str(label_dest),
        }

        # Process input file
        if input_file.exists():
            operation = _create_file_link_or_copy(
                input_file, input_dest, file_mode, logger
            )
            stats["file_operations"][operation] = (
                stats["file_operations"].get(operation, 0) + 1
            )

            # Update stats based on category
            if category == "training":
                stats["n_training_images"] += 1
            elif category == "validation":
                stats["n_val_images"] += 1
            elif category == "test":
                stats["n_test_images"] += 1

            logger.debug(f"[{operation}] {input_file.name} -> {input_dest.name}")
        else:
            stats["n_missing_input"] += 1
            logger.warning(f"Input file not found: {input_file}")

        # Process label file
        if label_file.exists():
            operation = _create_file_link_or_copy(
                label_file, label_dest, file_mode, logger
            )

            if category == "training":
                stats["n_training_labels"] += 1
            elif category == "validation":
                stats["n_val_labels"] += 1
            elif category == "test":
                stats["n_test_labels"] += 1

            logger.debug(f"[{operation}] {label_file.name} -> {label_dest.name}")
        else:
            stats["n_missing_label"] += 1
            logger.warning(f"Label file not found: {label_file}")

    # Build standard result dictionary
    result = _build_standard_result(
        base_dir=output_dir,
        created_dirs=created_dirs,
        stats=stats,
        file_mapping=file_mapping,
    )

    # Log summary
    total_processed = (
        stats["n_training_images"] + stats["n_val_images"] + stats["n_test_images"]
    )
    logger.info(f"Reorganization complete: {total_processed} images processed")
    logger.info(
        f"  Training: {stats['n_training_images']} images, {stats['n_training_labels']} labels"
    )
    logger.info(
        f"  Validation: {stats['n_val_images']} images, {stats['n_val_labels']} labels"
    )
    if include_test:
        logger.info(
            f"  Test: {stats['n_test_images']} images, {stats['n_test_labels']} labels"
        )
    if stats["n_missing_input"] > 0 or stats["n_missing_label"] > 0:
        logger.warning(
            f"  Missing files: {stats['n_missing_input']} inputs, {stats['n_missing_label']} labels"
        )
    logger.info(f"  File operations: {stats['file_operations']}")

    return result
