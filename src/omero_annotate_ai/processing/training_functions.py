"""Training data preparation functions for micro-SAM workflows."""

import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import ezomero
import numpy as np
import pandas as pd
from tifffile import imwrite
from tqdm import tqdm

from ..utils.logging import create_training_logger


def validate_table_schema(df: pd.DataFrame, logger=None) -> None:
    """
    Validate that the table has the required columns and basic data integrity.
    
    Args:
        df: DataFrame from OMERO table
        logger: Optional logger instance for logging messages
        
    Raises:
        ValueError: If required columns are missing or data integrity issues found
    """
    # Required columns for training data preparation
    required_columns = {
        'image_id', 'channel', 'z_slice', 'timepoint', 'label_id', 
        'train', 'validate', 'is_patch', 'patch_x', 'patch_y', 
        'patch_width', 'patch_height'
    }
    
    # Optional columns that enhance functionality
    optional_columns = {'is_volumetric'}
    
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    
    # Check for completely null critical columns
    critical_columns = ['image_id', 'label_id']
    for col in critical_columns:
        if col in df.columns and df[col].isna().all():
            raise ValueError(f"Column '{col}' contains no valid data")
    
    # Basic data type check for image_id (should be numeric)
    if not pd.api.types.is_numeric_dtype(df['image_id']):
        try:
            pd.to_numeric(df['image_id'], errors='raise')
        except (ValueError, TypeError):
            raise ValueError("Column 'image_id' contains non-numeric data")
    
    # Log optional columns that are available
    available_optional = optional_columns.intersection(set(df.columns))
    if available_optional:
        message = f"Optional columns found: {sorted(available_optional)}"
        if logger:
            logger.debug(message)
        else:
            print(message)


def prepare_training_data_from_table(
    conn: Any,
    table_id: int,
    output_dir: Union[str, Path],
    training_name: str = "micro_sam_training",
    validation_split: float = 0.2,
    clean_existing: bool = True,
    tmp_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False
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
    logger.debug(f"Parameters: output_dir={output_dir}, validation_split={validation_split}, clean_existing={clean_existing}")
    
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
    
    # Save the table locally for inspection (without debug in name)
    table_path = output_dir / f"table_{table_id}.csv"
    try:
        table.to_csv(table_path, index=True)
        logger.info(f"Table saved to: {table_path}")
    except Exception as e:
        logger.warning(f"Failed to save table: {e}")
    
    # Check if 'processed' column exists and filter to only processed rows
    if 'processed' in table.columns:
        initial_count = len(table)
        unprocessed_count = len(table[~table['processed']])
        
        if unprocessed_count > 0:
            logger.warning(f"Found {unprocessed_count} unprocessed rows out of {initial_count} total rows")
            logger.info(f"Proceeding with {initial_count - unprocessed_count} processed rows for training")
        
        # Filter to only processed rows
        table = table[table['processed']].copy()
        
        if len(table) == 0:
            raise ValueError("No processed rows found in the table. Cannot proceed with training.")
            
        logger.info(f"Using {len(table)} processed rows for training")
        
    else:
        logger.warning("No 'processed' column found - assuming all rows are ready for training")
    
    # Validate table schema and data integrity
    validate_table_schema(table, logger)
    logger.info("Table schema validated for processing")
    
    # Clean existing directories if requested
    if clean_existing:
        folders = ["training_input", "training_label", "val_input", "val_label"]
        for folder in folders:
            folder_path = output_dir / folder
            if folder_path.exists():
                shutil.rmtree(folder_path)
                
    # Split data based on existing 'train'/'validate' columns or automatic split
    if 'train' in table.columns and 'validate' in table.columns:
        # Use existing split from table
        train_images = table[table['train']]
        val_images = table[table['validate']]
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
    
    logger.info(f"Using {len(train_images)} training images and {len(val_images)} validation images")
    
    # Prepare training data
    training_input_dir, training_label_dir = _prepare_dataset_from_table(
        conn, train_images, output_dir, subset_type="training", tmp_dir=tmp_dir, logger=logger, verbose=verbose
    )
    
    # Prepare validation data
    val_input_dir, val_label_dir = _prepare_dataset_from_table(
        conn, val_images, output_dir, subset_type="val", tmp_dir=tmp_dir, logger=logger, verbose=verbose
    )
    
    # Clean up temporary directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        logger.debug(f"Cleaned up temporary directory: {tmp_dir}")
    
    # Collect statistics
    stats = {
        'n_training_images': len(list(training_input_dir.glob('*.tif'))),
        'n_training_labels': len(list(training_label_dir.glob('*.tif'))),
        'n_val_images': len(list(val_input_dir.glob('*.tif'))),
        'n_val_labels': len(list(val_label_dir.glob('*.tif'))),
        'total_rows_processed': len(table)
    }
    
    result = {
        'base_dir': output_dir,
        'training_input': training_input_dir,
        'training_label': training_label_dir,
        'val_input': val_input_dir,
        'val_label': val_label_dir,
        'stats': stats
    }
    
    # Check if preparation actually succeeded
    if stats['n_training_images'] == 0 and stats['n_val_images'] == 0:
        logger.error(f"Training data preparation FAILED in: {output_dir}")
        logger.error(f"Statistics: {stats}")
        raise ValueError("Training data preparation failed - no images were processed successfully. Check the error messages above.")
    else:
        logger.info(f"Training data prepared successfully in: {output_dir}")
        logger.info(f"Statistics: {stats}")
    
    return result


def _prepare_dataset_from_table(
    conn,
    df: pd.DataFrame,
    output_dir: Path,
    subset_type: str = "training",
    tmp_dir: Optional[Path] = None,
    train_channel: Optional[int] = None,
    logger=None,
    verbose: bool = False
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
        raise OSError(f"Failed to create dataset directories {input_dir}, {label_dir}: {e}")
    
    if logger:
        logger.info(f"Preparing {subset_type} dataset: {len(df)} items to process")
    
    for n in tqdm(range(len(df)), desc=f"Preparing {subset_type} data"):
        try:
            # Extract metadata
            image_id = int(df.iloc[n]['image_id'])
            
            # Handle z_slice - could be int, string representation of list, or NaN
            z_slice = df.iloc[n]['z_slice']
            if pd.isna(z_slice):
                z_slice = 0
            elif isinstance(z_slice, str) and z_slice.startswith('['):
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
                channel = int(df.iloc[n]['channel']) if pd.notna(df.iloc[n]['channel']) else 0
            timepoint = int(df.iloc[n]['timepoint']) if pd.notna(df.iloc[n]['timepoint']) else 0
            is_volumetric = bool(df.iloc[n]['is_volumetric']) if 'is_volumetric' in df.columns and pd.notna(df.iloc[n]['is_volumetric']) else False
            
            # Get patch information  
            is_patch = bool(df.iloc[n]['is_patch'])
            patch_x = int(df.iloc[n]['patch_x'])
            patch_y = int(df.iloc[n]['patch_y'])
            patch_width = int(df.iloc[n]['patch_width'])
            patch_height = int(df.iloc[n]['patch_height'])
            
            # Debug patch dimensions
            debug_print(f"Item {n} - Image ID: {image_id}, Patch: {is_patch}, Dimensions: {patch_width}x{patch_height} at ({patch_x},{patch_y}), Volumetric: {is_volumetric}")
            
            # Process based on whether it's 3D volumetric or 2D
            if is_volumetric:
                # Handle 3D volumetric data
                # Determine which z-slices to load
                if isinstance(z_slice, list):
                    z_slices = z_slice
                elif z_slice == 'all':
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
                            logger.debug(f"3D Patch Request - start_coords: ({patch_x}, {patch_y}, {z_val}, {channel}, {timepoint}), dimensions: {patch_width}x{patch_height}")
                        else:
                            print(f"  3D Patch Request - start_coords: ({patch_x}, {patch_y}, {z_val}, {channel}, {timepoint}), dimensions: {patch_width}x{patch_height}")
                        
                        # Use ezomero.get_image to extract the patch for this z-slice
                        _, img_slice = ezomero.get_image(
                            conn,
                            image_id,
                            start_coords=(patch_x, patch_y, z_val, channel, timepoint),
                            axis_lengths=(patch_width, patch_height, 1, 1, 1),
                            xyzct=True  # Use XYZCT ordering
                        )
                        
                        # Check shape of returned array
                        if logger:
                            logger.debug(f"Returned array shape (before extraction): {img_slice.shape}")
                        else:
                            print(f"  Returned array shape (before extraction): {img_slice.shape}")
                        
                        # The result will be 5D, extract just the 2D slice
                        img_slice = img_slice[:,:,0, 0, 0]  # Extract the single z-slice
                        if logger:
                            logger.debug(f"Extracted slice shape: {img_slice.shape}")
                        else:
                            print(f"  Extracted slice shape: {img_slice.shape}")
                    else:
                        # Get full plane for this z-slice
                        _, img_slice = ezomero.get_image(
                            conn,
                            image_id,
                            start_coords=(0, 0, z_val, channel, timepoint),
                            xyzct=True  # Use XYZCT ordering
                        )
                        # Check shape of returned array
                        if logger:
                            logger.debug(f"Full plane shape (before extraction): {img_slice.shape}")
                        else:
                            print(f"  Full plane shape (before extraction): {img_slice.shape}")
                        
                        # The result will be 5D, extract just the 2D slice
                        if len(img_slice.shape) == 5:
                            img_slice = img_slice[:, :, 0, 0, 0]
                            img_slice = np.swapaxes(img_slice, 0, 1)
                        if logger:
                            logger.debug(f"Extracted full plane shape: {img_slice.shape}")
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
                    logger.debug(f"Saved 3D TIFF to {output_path} with shape {img_8bit.shape}")
                else:
                    print(f"  Saved 3D TIFF to {output_path} with shape {img_8bit.shape}")
                
            else:
                # Handle 2D data with patch support using ezomero.get_image
                if is_patch and patch_width > 0 and patch_height > 0:
                    # Use ezomero.get_image with appropriate coordinates and dimensions
                    z_val = z_slice if not isinstance(z_slice, list) else z_slice[0]
                    
                    # Debug start_coords and axis_lengths
                    if logger:
                        logger.debug(f"2D Patch Request - start_coords: ({patch_x}, {patch_y}, {z_val}, {channel}, {timepoint}), dimensions: {patch_width}x{patch_height}")
                    else:
                        print(f"  2D Patch Request - start_coords: ({patch_x}, {patch_y}, {z_val}, {channel}, {timepoint}), dimensions: {patch_width}x{patch_height}")
                    
                    _, img_data = ezomero.get_image(
                        conn,
                        image_id,
                        start_coords=(patch_x, patch_y, int(z_val), channel, timepoint),
                        axis_lengths=(patch_width, patch_height, 1, 1, 1),
                        xyzct=True
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
                    
                    # Debug start_coords
                    if logger:
                        logger.debug(f"2D Full Image Request - start_coords: (0, 0, {z_val}, {channel}, {timepoint})")
                    else:
                        print(f"  2D Full Image Request - start_coords: (0, 0, {z_val}, {channel}, {timepoint})")
                    
                    _, img_data = ezomero.get_image(
                        conn,
                        image_id,
                        start_coords=(0, 0, int(z_val), channel, timepoint),
                        xyzct=True
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
                max_val = img_data.max()
                if max_val > 0:
                    img_8bit = ((img_data) * (255.0 / max_val)).astype(np.uint8)
                else:
                    img_8bit = img_data.astype(np.uint8)
                
                # Save as TIFF
                output_path = input_dir / f"input_{n:05d}.tif"
                imwrite(str(output_path), img_8bit)
                if logger:
                    logger.debug(f"Saved 2D TIFF to {output_path} with shape {img_8bit.shape}")
                else:
                    print(f"  Saved 2D TIFF to {output_path} with shape {img_8bit.shape}")
            
            # Get label file (already normalized to int or NaN)
            label_id_val = df.iloc[n]['label_id']
            if pd.notna(label_id_val):
                label_id = int(label_id_val)
                
                try:
                    # First, check if the file annotation exists
                    if logger:
                        logger.debug(f"Attempting to download label with ID: {label_id}")
                    else:
                        print(f"  Attempting to download label with ID: {label_id}")
                    
                    # Try to get the file annotation object first to validate it exists
                    try:
                        file_ann = conn.getObject("FileAnnotation", label_id)
                        if file_ann is None:
                            if logger:
                                logger.warning(f"File annotation {label_id} not found in OMERO")
                            else:
                                print(f"  Warning: File annotation {label_id} not found in OMERO")
                            continue
                        if logger:
                            logger.debug(f"File annotation found: {file_ann.getFile().getName()}")
                        else:
                            print(f"  File annotation found: {file_ann.getFile().getName()}")
                    except Exception as check_e:
                        if logger:
                            logger.error(f"Error checking file annotation {label_id}: {check_e}")
                        else:
                            print(f"  Error checking file annotation {label_id}: {check_e}")
                        continue
                    
                    # Now try to download it using ezomero
                    file_path = ezomero.get_file_annotation(conn, label_id, str(tmp_dir))
                    if file_path:
                        label_dest = label_dir / f"label_{n:05d}.tif"
                        shutil.move(file_path, str(label_dest))
                        
                        # Check the size of the saved label
                        from tifffile import imread
                        label_img = imread(str(label_dest))
                        if logger:
                            logger.debug(f"Label shape: {label_img.shape} saved to {label_dest}")
                        else:
                            print(f"  Label shape: {label_img.shape} saved to {label_dest}")
                    else:
                        if logger:
                            logger.warning(f"Label file for image {image_id} not downloaded (ezomero returned None)")
                        else:
                            print(f"  Warning: Label file for image {image_id} not downloaded (ezomero returned None)")
                        
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