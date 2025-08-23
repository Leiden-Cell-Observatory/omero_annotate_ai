"""Training data preparation functions for micro-SAM workflows."""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import ezomero
import numpy as np
import pandas as pd
from tifffile import imwrite
from tqdm import tqdm


def validate_table_schema(df: pd.DataFrame) -> None:
    """
    Validate that the table has the required columns and basic data integrity.
    
    Args:
        df: DataFrame from OMERO table
        
    Raises:
        ValueError: If required columns are missing or data integrity issues found
    """
    # Required columns for training data preparation
    required_columns = {
        'image_id', 'channel', 'z_slice', 'timepoint', 'label_id', 
        'train', 'validate', 'is_patch', 'patch_x', 'patch_y', 
        'patch_width', 'patch_height'
    }
    
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


def prepare_training_data_from_table(
    conn: Any,
    table_id: int,
    output_dir: Union[str, Path],
    validation_split: float = 0.2,
    clean_existing: bool = True,
    tmp_dir: Optional[Union[str, Path]] = None
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
        validation_split: Fraction of data to use for validation (0.0-1.0)
        clean_existing: Whether to clean existing output directories
        tmp_dir: Temporary directory for downloads (optional)
        
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
        
    print(f"Loaded table with {len(table)} rows")
    
    # Validate table schema and data integrity
    validate_table_schema(table)
    print("Table schema validated for processing")
    
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
        train_images = table[table['train'] == True]
        val_images = table[table['validate'] == True]
    else:
        # Automatic split
        n_val = int(len(table) * validation_split)
        shuffled_indices = np.random.permutation(len(table))
        val_indices = shuffled_indices[:n_val]
        train_indices = shuffled_indices[n_val:]
        
        train_images = table.iloc[train_indices]
        val_images = table.iloc[val_indices]
    
    print(f"Using {len(train_images)} training images and {len(val_images)} validation images")
    
    # Prepare training data
    training_input_dir, training_label_dir = _prepare_dataset_from_table(
        conn, train_images, output_dir, subset_type="training", tmp_dir=tmp_dir
    )
    
    # Prepare validation data
    val_input_dir, val_label_dir = _prepare_dataset_from_table(
        conn, val_images, output_dir, subset_type="val", tmp_dir=tmp_dir
    )
    
    # Clean up temporary directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    
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
        print(f"❌ Training data preparation FAILED in: {output_dir}")
        print(f"Statistics: {stats}")
        raise ValueError("Training data preparation failed - no images were processed successfully. Check the error messages above.")
    else:
        print(f"✅ Training data prepared successfully in: {output_dir}")
        print(f"Statistics: {stats}")
    
    return result


def _prepare_dataset_from_table(
    conn,
    df: pd.DataFrame,
    output_dir: Path,
    subset_type: str = "training",
    tmp_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Prepare dataset from annotation table subset.
    
    Args:
        conn: OMERO connection
        df: DataFrame with annotation info
        output_dir: Base output directory
        subset_type: "training" or "val"
        tmp_dir: Temporary directory for downloading annotations
        
    Returns:
        (input_dir, label_dir): Paths to the input and label directories
    """
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
    
    for n in tqdm(range(len(df)), desc=f"Preparing {subset_type} data"):
        # Extract metadata (types already normalized)
        image_id = int(df.iloc[n]['image_id'])
        channel = int(df.iloc[n]['channel'])
        timepoint = int(df.iloc[n]['timepoint'])
        z_val = int(df.iloc[n]['z_slice'])
        
        # Get patch information  
        is_patch = bool(df.iloc[n]['is_patch'])
        patch_x = int(df.iloc[n]['patch_x'])
        patch_y = int(df.iloc[n]['patch_y'])
        patch_width = int(df.iloc[n]['patch_width'])
        patch_height = int(df.iloc[n]['patch_height'])
        
        # Get image data
        try:
            if is_patch and patch_width > 0 and patch_height > 0:
                _, img_data = ezomero.get_image(
                    conn,
                    image_id,
                    start_coords=(patch_x, patch_y, z_val, channel, timepoint),
                    axis_lengths=(patch_width, patch_height, 1, 1, 1),
                    xyzct=True
                )
            else:
                # Get full image without specifying axis_lengths (let ezomero determine size)
                _, img_data = ezomero.get_image(
                    conn,
                    image_id,
                    start_coords=(0, 0, z_val, channel, timepoint),
                    xyzct=True
                )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch image {image_id} from OMERO: {e}")
        
        # Process image data
        if len(img_data.shape) == 5:
            img_data = img_data[:, :, 0, 0, 0]
            img_data = np.swapaxes(img_data, 0, 1)
        
        # Normalize to 8-bit
        max_val = img_data.max()
        if max_val > 0:
            img_8bit = ((img_data) * (255.0 / max_val)).astype(np.uint8)
        else:
            img_8bit = img_data.astype(np.uint8)
        
        # Save image
        output_path = input_dir / f"input_{n:05d}.tif"
        try:
            imwrite(str(output_path), img_8bit)
        except Exception as e:
            raise OSError(f"Failed to save image to {output_path}: {e}")
        
        # Get label file (already normalized to int or NaN)
        label_id_val = df.iloc[n]['label_id']
        if pd.notna(label_id_val):
            label_id = int(label_id_val)
            
            try:
                file_path = ezomero.get_file_annotation(conn, label_id, str(tmp_dir))
            except Exception as e:
                raise RuntimeError(f"Failed to download label file {label_id} from OMERO: {e}")

            if not file_path:
                raise FileNotFoundError(f"Label file {label_id} not found in OMERO")

            label_dest = label_dir / f"label_{n:05d}.tif"
            try:
                shutil.move(file_path, str(label_dest))
            except Exception as e:
                raise OSError(f"Failed to move label file {label_id} to {label_dest}: {e}")
    
    return input_dir, label_dir