"""Utility functions for micro-SAM workflows."""

from typing import Any, List


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
