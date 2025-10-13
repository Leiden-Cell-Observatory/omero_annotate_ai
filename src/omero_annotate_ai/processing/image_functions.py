"""Image processing functions for micro-SAM workflows."""

import random as rnd
from typing import List, Tuple

# Optional dependencies for ROI functionality
import cv2
import ezomero.rois
import numpy as np


def generate_patch_coordinates(
    image_shape: Tuple[int, int],
    patch_size: List[int],
    n_patches: int,
    random_patch: bool = True,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """Generate non-overlapping patch coordinates for an image.

    CRUCIAL: Ensures patches do not overlap when generating multiple patches.

    Args:
        image_shape: (height, width) of the image
        patch_size: (height, width) of patches
        n_patches: Number of patches to generate
        random_patch: Whether to generate random patches or grid-based patches

    Returns:
        Tuple containing:
        - List of (x, y) coordinates for patch top-left corners (non-overlapping)
        - Actual patch size (height, width) to use (adjusted if image smaller than patch)
    """
    height, width = image_shape
    patch_h, patch_w = patch_size

    # Check if image is smaller than patch
    if width < patch_w or height < patch_h:
        # Image smaller than patch, return image size as patch size
        print("⚠️ Image smaller than patch size, using full image")
        actual_patch_size = (height, width)
        return [(0, 0)], actual_patch_size

    # Image is large enough for requested patch size
    actual_patch_size = (patch_h, patch_w)
    
    # Ensure patches fit within image
    max_x = max(0, width - patch_w)
    max_y = max(0, height - patch_h)

    coordinates = []

    if random_patch:
        # Generate random non-overlapping coordinates
        used_areas = []  # Track used rectangular areas
        max_attempts = n_patches * 20  # Limit attempts to avoid infinite loops
        attempts = 0

        while len(coordinates) < n_patches and attempts < max_attempts:
            attempts += 1
            x = rnd.randint(0, max_x)
            y = rnd.randint(0, max_y)

            # Check if this patch overlaps with any existing patch
            new_rect = (x, y, x + patch_w, y + patch_h)
            overlaps = False

            for used_rect in used_areas:
                if _rectangles_overlap(new_rect, used_rect):
                    overlaps = True
                    break

            if not overlaps:
                coordinates.append((x, y))
                used_areas.append(new_rect)

        if len(coordinates) < n_patches:
            print(
                f"Could only place {len(coordinates)} non-overlapping patches out of {n_patches} requested"
            )

    else:
        # Generate grid-based non-overlapping patches
        # Calculate how many patches fit in each dimension
        patches_x = max(1, (width + patch_w - 1) // patch_w)  # Ceiling division
        patches_y = max(1, (height + patch_h - 1) // patch_h)
        max_grid_patches = patches_x * patches_y

        if n_patches > max_grid_patches:
            print(
                f"Requested {n_patches} patches, but only {max_grid_patches} non-overlapping patches fit"
            )
            n_patches = max_grid_patches

        # Calculate spacing to distribute patches evenly
        if patches_x > 1:
            step_x = (width - patch_w) // (patches_x - 1)
        else:
            step_x = 0

        if patches_y > 1:
            step_y = (height - patch_h) // (patches_y - 1)
        else:
            step_y = 0

        # Generate grid coordinates
        patch_count = 0
        for row in range(patches_y):
            for col in range(patches_x):
                if patch_count >= n_patches:
                    break

                x = min(col * step_x, max_x)
                y = min(row * step_y, max_y)
                coordinates.append((x, y))
                patch_count += 1

            if patch_count >= n_patches:
                break

    return coordinates, actual_patch_size


def _rectangles_overlap(
    rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]
) -> bool:
    """Check if two rectangles overlap.

    Args:
        rect1: (x1, y1, x2, y2) coordinates of first rectangle
        rect2: (x1, y1, x2, y2) coordinates of second rectangle

    Returns:
        True if rectangles overlap, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # Check if rectangles do NOT overlap (then return False)
    if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
        return False

    # If not non-overlapping, they must overlap
    return True


def mask_to_contour(mask):
    """Converts a binary mask to a list of ROI coordinates.

    Args:
        mask (np.ndarray): binary mask

    Returns:
        list: list of ROI coordinates
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def process_label_plane(
    label_plane, z_slice, channel, timepoint, model_type, x_offset=0, y_offset=0
):
    """
    Process a single 2D label plane to generate OMERO shapes with optional offset

    Args:
        label_plane: 2D label plane (numpy array)
        z_slice: Z-slice index
        channel: Channel index
        timepoint: Time point index
        model_type: SAM model type identifier
        x_offset: X offset for contour coordinates (default: 0)
        y_offset: Y offset for contour coordinates (default: 0)

    Returns:
        list: List of OMERO shapes
    """

    shapes = []
    unique_labels = np.unique(label_plane)

    # Skip background (label 0)
    for label in unique_labels[1:]:
        # Create binary mask for this label
        mask = (label_plane == label).astype(np.uint8)

        # Get contours
        contours = mask_to_contour(mask)

        # Convert each contour to polygon ROI
        for contour in contours:
            contour = contour[:, 0, :]  # Reshape to (N, 2)

            # Apply offset to contour points if needed
            if x_offset != 0 or y_offset != 0:
                contour = contour + np.array([x_offset, y_offset])

            # Create polygon without text parameter
            poly = ezomero.rois.Polygon(
                points=contour,  # explicitly name the points parameter
                z=z_slice,
                c=channel,
                t=timepoint,
                label=f'micro_sam.{"volumetric" if isinstance(z_slice, (list, range)) or z_slice > 0 else "manual"}_instance_segmentation.{model_type}',
            )
            shapes.append(poly)

    return shapes


def label_to_rois(
    label_img,
    z_slice,
    channel,
    timepoint,
    model_type,
    is_volumetric=False,
    patch_offset=None,
):
    """
    Convert a 2D or 3D label image to OMERO ROI shapes

    Args:
        label_img (np.ndarray): 2D labeled image or 3D labeled stack
        z_slice (int or list): Z-slice index or list/range of Z indices
        channel (int): Channel index
        timepoint (int): Time point index
        model_type (str): SAM model type used
        is_volumetric (bool): Whether the label image is 3D volumetric data
        patch_offset: Optional (x,y) offset for placing ROIs in a larger image

    Returns:
        list: List of OMERO shape objects
    """
    shapes = []

    # Unpack patch offset if provided
    x_offset, y_offset = (0, 0) if patch_offset is None else patch_offset

    if is_volumetric and label_img.ndim > 2:
        # 3D volumetric data - process each z slice
        for z_index, z_plane in enumerate(label_img):
            # If z_slice is a range or list, use the actual z-index from that range
            if isinstance(z_slice, (range, list)):
                actual_z = (
                    z_slice[z_index] if z_index < len(z_slice) else z_slice[0] + z_index
                )
            else:
                actual_z = z_slice + z_index  # Assume z_slice is the starting index

            print(f"Processing volumetric ROIs for z-slice {actual_z}")
            shapes.extend(
                process_label_plane(
                    z_plane,
                    actual_z,
                    channel,
                    timepoint,
                    model_type,
                    x_offset,
                    y_offset,
                )
            )
    else:
        # 2D data - process single plane
        shapes.extend(
            process_label_plane(
                label_img, z_slice, channel, timepoint, model_type, x_offset, y_offset
            )
        )

    return shapes
