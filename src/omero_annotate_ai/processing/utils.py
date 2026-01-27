"""Utility functions for micro-SAM workflows."""

from typing import Any, List
import pandas as pd


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



def interleave_arrays(array1: List[Any], array2: List[Any]) -> List[Any]:
    """Interleave two arrays.

    Args:
        array1: First array
        array2: Second array

    Returns:
        Interleaved array
    """
    result = []
    max_len = max(len(array1), len(array2))

    for i in range(max_len):
        if i < len(array1):
            result.append(array1[i])
        if i < len(array2):
            result.append(array2[i])

    return result


def validate_image_dimensions(image_shape: tuple, patch_size: tuple) -> bool:
    """Validate that image can accommodate patches.

    Args:
        image_shape: (height, width) of image
        patch_size: (height, width) of patch

    Returns:
        True if patches fit in image
    """
    img_h, img_w = image_shape
    patch_h, patch_w = patch_size

    return img_h >= patch_h and img_w >= patch_w


def calculate_optimal_batch_size(
    n_images: int, available_memory_gb: float = 8.0
) -> int:
    """Calculate optimal batch size based on available memory.

    Args:
        n_images: Number of images to process
        available_memory_gb: Available memory in GB

    Returns:
        Recommended batch size
    """
    # Simple heuristic: assume each image needs ~1GB for processing
    max_batch = max(1, int(available_memory_gb))
    return min(n_images, max_batch)


def format_processing_time(seconds: float) -> str:
    """Format processing time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
