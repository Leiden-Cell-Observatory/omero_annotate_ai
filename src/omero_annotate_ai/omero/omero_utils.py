"""
Generic OMERO utility functions for table and annotation management.

This module provides reusable utilities for common OMERO operations that are
useful across different workflows and not specific to micro-SAM annotation.
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import dask.array as da
from dask.delayed import delayed
import ezomero


# =============================================================================
# Table Management Utilities
# =============================================================================


def list_user_tables(
    conn, container_type: str = None, container_id: int = None, namespace: str = None
) -> List[Dict]:
    """List all OMERO.tables accessible to the user.

    Args:
        conn: OMERO connection
        container_type: Optional container type to filter by ('dataset', 'project', etc.)
        container_id: Optional container ID to filter by
        namespace: Optional namespace to filter by (recommended for performance)

    Returns:
        List of dictionaries with table information
    """
    if conn is None:
        print("Cannot list tables: OMERO connection is None")
        return []

    tables = []

    try:
        if container_type and container_id:
            # Get file annotation IDs — pass namespace for server-side filtering
            annotations = ezomero.get_file_annotation_ids(
                conn, container_type.capitalize(), container_id, ns=namespace
            )

            for ann_id in annotations:
                try:
                    file_ann = conn.getObject("FileAnnotation", ann_id)
                    if not file_ann or not hasattr(file_ann, "getFile"):
                        continue

                    original_file = file_ann.getFile()
                    if not original_file:
                        continue

                    # Check mimetype to confirm this is an OMERO.table
                    # without loading the table contents
                    mimetype = original_file.getMimetype()
                    if mimetype != "OMERO.tables":
                        continue

                    file_name = original_file.getName() or f"table_{ann_id}"
                    created = ""
                    try:
                        date = file_ann.getDate()
                        if date:
                            created = date.isoformat()
                    except Exception:
                        pass

                    tables.append(
                        {
                            "id": ann_id,
                            "name": file_name,
                            "created": created,
                            "container_type": container_type,
                            "container_id": container_id,
                            "description": file_ann.getDescription() or "",
                            "namespace": file_ann.getNs() or "",
                        }
                    )
                except Exception:
                    continue
        else:
            print(
                "Tip: Specify container_type and container_id for more efficient search"
            )

    except Exception as e:
        print(f"Error listing tables: {e}")

    return tables


def delete_table(conn, table_id: int) -> bool:
    """Safely delete OMERO table.

    Args:
        conn: OMERO connection
        table_id: ID of table to delete

    Returns:
        True if successful, False otherwise
    """
    if conn is None:
        print("Cannot delete table: OMERO connection is None")
        return False

    try:
        # Get table info first
        file_ann = conn.getObject("FileAnnotation", table_id)
        if not file_ann:
            print(f"Table {table_id} not found")
            return False

        table_name = (
            file_ann.getFile().getName() if file_ann.getFile() else f"ID:{table_id}"
        )
        print(f"Deleting table: {table_name}")

        # Delete the file annotation
        conn.deleteObjects("FileAnnotation", [table_id], wait=True)
        print(f"Successfully deleted table {table_id}")
        return True

    except Exception as e:
        print(f"Error deleting table {table_id}: {e}")
        return False


def validate_table_schema(
    conn, table_id: int, expected_columns: List[str]
) -> Tuple[bool, List[str]]:
    """Validate that table has expected columns.

    Args:
        conn: OMERO connection
        table_id: ID of table to validate
        expected_columns: List of expected column names

    Returns:
        Tuple of (is_valid, missing_columns)
    """
    try:
        from .omero_functions import read_tracking_table

        df = read_tracking_table(conn, table_id)
        actual_columns = set(df.columns)
        expected_set = set(expected_columns)

        missing_columns = list(expected_set - actual_columns)
        extra_columns = list(actual_columns - expected_set)

        is_valid = len(missing_columns) == 0

        if is_valid:
            print(f"Table {table_id} schema is valid")
            if extra_columns:
                print(f"   Extra columns: {extra_columns}")
        else:
            print(f"Table {table_id} schema is invalid")
            print(f"   Missing columns: {missing_columns}")

        return is_valid, missing_columns

    except Exception as e:
        print(f"Error validating table {table_id}: {e}")
        return False, expected_columns


# =============================================================================
# Annotation Management Utilities
# =============================================================================


def list_annotations_by_namespace(
    conn, object_type: str, object_id: int, namespace: str
) -> List[Dict]:
    """List annotations by namespace.

    Args:
        conn: OMERO connection
        object_type: Type of object ('Image', 'Dataset', etc.)
        object_id: ID of object
        namespace: Namespace to filter by

    Returns:
        List of annotation dictionaries
    """
    annotations = []

    try:
        # Pass namespace for server-side filtering
        ann_ids = ezomero.get_file_annotation_ids(
            conn, object_type, object_id, ns=namespace
        )

        for ann_id in ann_ids:
            try:
                file_ann = conn.getObject("FileAnnotation", ann_id)
                if file_ann:
                    annotations.append(
                        {
                            "id": ann_id,
                            "namespace": file_ann.getNs(),
                            "description": file_ann.getDescription() or "",
                            "file_name": file_ann.getFile().getName()
                            if file_ann.getFile()
                            else "",
                            "file_size": file_ann.getFile().getSize()
                            if file_ann.getFile()
                            else 0,
                        }
                    )
            except Exception:
                continue

        print(f"Found {len(annotations)} annotations with namespace '{namespace}'")

    except Exception as e:
        print(f"Error listing annotations: {e}")

    return annotations


def delete_annotations_by_namespace(
    conn, object_type: str, object_id: int, namespace: str
) -> int:
    """Clean up annotations by namespace.

    Args:
        conn: OMERO connection
        object_type: Type of object ('Image', 'Dataset', etc.)
        object_id: ID of object
        namespace: Namespace to delete

    Returns:
        Number of annotations deleted
    """
    try:
        annotations = list_annotations_by_namespace(
            conn, object_type, object_id, namespace
        )

        if not annotations:
            print(f"No annotations found with namespace '{namespace}'")
            return 0

        # Delete annotations
        ann_ids = [ann["id"] for ann in annotations]
        conn.deleteObjects("FileAnnotation", ann_ids, wait=True)

        print(f"Deleted {len(ann_ids)} annotations with namespace '{namespace}'")
        return len(ann_ids)

    except Exception as e:
        print(f"Error deleting annotations: {e}")
        return 0


# =============================================================================
# Connection and Error Handling Utilities
# =============================================================================


def get_container_info(
    conn, container_type: str, container_id: int
) -> Optional[Dict[str, Any]]:
    """Get summary information about a container (Project, Dataset, etc.).

    Args:
        conn: OMERO connection.
        container_type: Type of container ('project', 'dataset', 'plate', 'screen').
        container_id: ID of the container.

    Returns:
        A dictionary with container information or None if an error occurs.
        Dictionary keys include:
        - container_type
        - container_id
        - container_name
        - total_images
        - sample_image_id
        - sample_image_name
        - dimensions
        - pixel_sizes
        - channel_names (from the sample image)
        - channel_names_vary (True if a few other images name channels differently)
    """
    if not conn or not conn.isConnected():
        return None

    try:
        container = conn.getObject(container_type.capitalize(), container_id)
        if not container:
            return None

        image_ids = []
        info = {
            "container_type": container_type,
            "container_id": container_id,
            "container_name": container.getName(),
            "total_images": 0,
            "sample_image_id": None,
            "sample_image_name": None,
            "dimensions": None,
            "pixel_sizes": None,
            "channel_names": None,
            "channel_names_vary": False,
        }

        if container_type == "screen":
            plate_ids = ezomero.get_plate_ids(conn, screen=container_id)
            info["total_plates"] = len(plate_ids)
            all_image_ids = []
            for plate_id in plate_ids:
                all_image_ids.extend(ezomero.get_image_ids(conn, plate=plate_id))
            image_ids = all_image_ids
        else:
            image_ids = ezomero.get_image_ids(conn, **{container_type: container_id})

        total_images = len(image_ids)
        info["total_images"] = total_images

        if total_images > 0:
            sample_image_id = image_ids[0]
            sample_image = conn.getObject("Image", sample_image_id)

            if sample_image:
                info["sample_image_id"] = sample_image_id
                info["sample_image_name"] = sample_image.getName()
                info["dimensions"] = {
                    "T": sample_image.getSizeT(),
                    "C": sample_image.getSizeC(),
                    "Z": sample_image.getSizeZ(),
                    "Y": sample_image.getSizeY(),
                    "X": sample_image.getSizeX(),
                }

                pixel_sizes = {}
                try:
                    size_x = sample_image.getPixelSizeX(units=True)
                    if size_x:
                        pixel_sizes["X"] = (size_x.getValue(), size_x.getSymbol())
                except AttributeError:
                    pass

                try:
                    size_y = sample_image.getPixelSizeY(units=True)
                    if size_y:
                        pixel_sizes["Y"] = (size_y.getValue(), size_y.getSymbol())
                except AttributeError:
                    pass

                try:
                    size_z = sample_image.getPixelSizeZ(units=True)
                    if size_z:
                        pixel_sizes["Z"] = (size_z.getValue(), size_z.getSymbol())
                except AttributeError:
                    pass

                info["pixel_sizes"] = pixel_sizes
                info["channel_names"] = _channel_labels(sample_image)
                info["channel_names_vary"] = any(
                    _channel_labels(conn.getObject("Image", image_id))
                    != info["channel_names"]
                    for image_id in _spread_sample(image_ids[1:], 4)
                )

        return info

    except Exception:
        return None


def _channel_labels(image) -> Optional[List[str]]:
    """Channel names of an OMERO image, or None if they cannot be read."""
    if image is None:
        return None
    try:
        return [str(label) for label in image.getChannelLabels()]
    except Exception:
        return None


def _spread_sample(items: List[Any], n: int) -> List[Any]:
    """Up to n items spread evenly across the list, always including the last."""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[min(len(items) - 1, round((i + 1) * step) - 1)] for i in range(n)]


# =============================================================================
# Generic Image Loading Utilities
# =============================================================================


def _create_dask_array_for_image(
    conn, image, timepoints: List[int], channels: List[int], z_slices: List[int]
):
    """Create a dask array for a single OMERO image with lazy loading.

    This is a generic utility that could be contributed to ezomero in the future.

    Args:
        conn: OMERO connection
        image: OMERO image object
        timepoints: List of timepoint indices
        channels: List of channel indices
        z_slices: List of z-slice indices

    Returns:
        Dask array for the image
    """
    # Use first of each dimension if multiple provided
    t = timepoints[0] if timepoints else 0
    c = channels[0] if channels else 0
    z = z_slices[0] if z_slices else 0

    # Get image dimensions
    height = image.getSizeY()
    width = image.getSizeX()

    # Create delayed function for loading a single plane
    @delayed
    def load_plane(image_id, z_idx, c_idx, t_idx):
        """Delayed function to load a single plane from OMERO."""
        try:
            # Re-get the image object (connections may not be thread-safe)
            img = conn.getObject("Image", image_id)
            if not img:
                return np.zeros((height, width), dtype=np.uint16)

            pix = img.getPrimaryPixels()
            plane_data = pix.getPlane(z_idx, c_idx, t_idx)
            return plane_data

        except Exception as e:
            print(
                f"Error loading plane {z_idx},{c_idx},{t_idx} for image {image_id}: {e}"
            )
            return np.zeros((height, width), dtype=np.uint16)

    # Create delayed loading task
    delayed_plane = load_plane(image.getId(), z, c, t)

    # Convert to dask array with proper chunking
    # Use reasonable chunk size (e.g., 1024x1024 for large images)
    chunk_size = min(1024, height, width)
    chunks = (chunk_size, chunk_size)

    dask_array = da.from_delayed(
        delayed_plane,
        shape=(height, width),
        dtype=np.uint16,
        meta=np.array([], dtype=np.uint16),
    )

    # Rechunk for better performance
    dask_array = dask_array.rechunk(chunks)

    return dask_array
