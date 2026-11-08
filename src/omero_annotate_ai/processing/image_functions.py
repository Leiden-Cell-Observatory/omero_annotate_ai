"""Image processing functions for annotation workflows."""

import logging
import random as rnd

# Optional dependencies for ROI functionality
import cv2
import ezomero.rois
import numpy as np

logger = logging.getLogger(__name__)


def generate_patch_coordinates(
    image_shape: tuple[int, int],
    patch_size: list[int],
    n_patches: int,
    random_patch: bool = True,
    rng: rnd.Random | None = None,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Generate non-overlapping patch coordinates for an image.

    CRUCIAL: Ensures patches do not overlap when generating multiple patches.

    Args:
        image_shape: (height, width) of the image
        patch_size: (height, width) of patches
        n_patches: Number of patches to generate
        random_patch: Whether to generate random patches or grid-based patches
        rng: Random stream for random patches; the ``random`` module when omitted.

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
        logger.info("Image smaller than patch size, using full image")
        actual_patch_size = (height, width)
        return [(0, 0)], actual_patch_size

    # Image is large enough for requested patch size
    actual_patch_size = (patch_h, patch_w)

    grid = _grid_origins(height, width, patch_h, patch_w)
    if not random_patch:
        if n_patches > len(grid):
            logger.warning(
                f"Requested {n_patches} patches, but only {len(grid)} "
                "non-overlapping patches fit"
            )
        return grid[:n_patches], actual_patch_size

    # Random: rejection sampling; if that leaves too few on a tight image, a random
    # pick of grid slots, which always fits as many patches as the grid holds.
    draw = rng if rng is not None else rnd.Random()
    max_x, max_y = width - patch_w, height - patch_h
    coordinates: list[tuple[int, int]] = []
    for _ in range(n_patches * 50):
        if len(coordinates) == n_patches:
            break
        x, y = draw.randint(0, max_x), draw.randint(0, max_y)
        rect = (x, y, x + patch_w, y + patch_h)
        if not any(
            _rectangles_overlap(rect, (cx, cy, cx + patch_w, cy + patch_h))
            for cx, cy in coordinates
        ):
            coordinates.append((x, y))
    if len(coordinates) < min(n_patches, len(grid)):
        coordinates = draw.sample(grid, min(n_patches, len(grid)))

    if len(coordinates) < n_patches:
        logger.warning(
            f"Could only place {len(coordinates)} non-overlapping patches "
            f"out of {n_patches} requested"
        )
    return coordinates, actual_patch_size


def _grid_origins(
    height: int, width: int, patch_h: int, patch_w: int
) -> list[tuple[int, int]]:
    """Top-left corners of the most non-overlapping patches that fit, spread evenly."""

    def axis(size: int, patch: int) -> list[int]:
        count = size // patch
        if count <= 1:
            return [0]
        return [round(i * (size - patch) / (count - 1)) for i in range(count)]

    return [(x, y) for y in axis(height, patch_h) for x in axis(width, patch_w)]


def _rectangles_overlap(
    rect1: tuple[int, int, int, int], rect2: tuple[int, int, int, int]
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

    # They overlap unless one lies entirely to one side of the other.
    return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)


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
    label_plane, z_slice, channel, timepoint, x_offset=0, y_offset=0
):
    """
    Process a single 2D label plane to generate OMERO shapes with optional offset

    Args:
        label_plane: 2D label plane (numpy array)
        z_slice: Z-slice index
        channel: Channel index
        timepoint: Time point index
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
            segmentation_type = (
                "volumetric"
                if isinstance(z_slice, (list, range)) or z_slice > 0
                else "manual"
            )
            poly = ezomero.rois.Polygon(
                points=contour,  # explicitly name the points parameter
                # Volumetric shapes pass a z range; ezomero types z as int.
                z=z_slice,  # pyright: ignore[reportArgumentType]
                c=channel,
                t=timepoint,
                label=f"micro_sam.{segmentation_type}_instance_segmentation",
            )
            shapes.append(poly)

    return shapes


def label_to_rois(
    label_img,
    z_slice,
    channel,
    timepoint,
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

            logger.debug(f"Processing volumetric ROIs for z-slice {actual_z}")
            shapes.extend(
                process_label_plane(
                    z_plane,
                    actual_z,
                    channel,
                    timepoint,
                    x_offset,
                    y_offset,
                )
            )
    else:
        # 2D data - process single plane
        shapes.extend(
            process_label_plane(
                label_img, z_slice, channel, timepoint, x_offset, y_offset
            )
        )

    return shapes
