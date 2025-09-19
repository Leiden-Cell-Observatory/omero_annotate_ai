"""
Generic OMERO utility functions for table and annotation management.

This module provides reusable utilities for common OMERO operations that are
useful across different workflows and not specific to micro-SAM annotation.
"""

import os
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np
import dask.array as da
from dask.delayed import delayed
import ezomero


# =============================================================================
# Table Management Utilities
# =============================================================================

def list_user_tables(conn, container_type: str = None, container_id: int = None) -> List[Dict]:
    """List all tables accessible to the user.
    
    Args:
        conn: OMERO connection
        container_type: Optional container type to filter by ('dataset', 'project', etc.)
        container_id: Optional container ID to filter by
        
    Returns:
        List of dictionaries with table information
    """
    if conn is None:
        print("❌ Cannot list tables: OMERO connection is None")
        return []

    tables = []
    
    try:
        if container_type and container_id:
            # Search within specific container
            annotations = ezomero.get_file_annotation_ids(conn, container_type.capitalize(), container_id)
            
            for ann_id in annotations:
                try:
                    # Try to read as table using ezomero - simpler and more reliable
                    table = ezomero.get_table(conn, ann_id)
                    if table is not None:
                        # This is a valid OMERO.table, get the annotation details
                        file_ann = conn.getObject("FileAnnotation", ann_id)
                        if file_ann and hasattr(file_ann, 'getFile'):
                            original_file = file_ann.getFile()
                            file_name = original_file.getName() if original_file else f"table_{ann_id}"
                            
                            tables.append({
                                'id': ann_id,
                                'name': file_name,
                                'container_type': container_type,
                                'container_id': container_id,
                                'description': file_ann.getDescription() or "",
                                'namespace': file_ann.getNs() or ""
                            })
                except Exception:
                    # Not a valid OMERO.table, skip silently
                    continue
        else:
            # More complex search across user's space would go here
            # For now, we'll return empty list and recommend specifying container
            print("💡 Tip: Specify container_type and container_id for more efficient search")
            
    except Exception as e:
        print(f"❌ Error listing tables: {e}")
    
    return tables


def find_table_by_pattern(conn, container_type: str, container_id: int, pattern: str) -> Optional[Dict]:
    """Find table matching a name pattern.
    
    Args:
        conn: OMERO connection
        container_type: Container type ('dataset', 'project', etc.)
        container_id: Container ID  
        pattern: Pattern to match in table name
        
    Returns:
        Dictionary with table information or None
    """
    tables = list_user_tables(conn, container_type, container_id)
    
    for table in tables:
        if pattern.lower() in table['name'].lower():
            print(f"🔍 Found matching table: {table['name']} (ID: {table['id']})")
            return table
    
    print(f"🔍 No table found matching pattern: {pattern}")
    return None


def delete_table(conn, table_id: int) -> bool:
    """Safely delete OMERO table.
    
    Args:
        conn: OMERO connection
        table_id: ID of table to delete
        
    Returns:
        True if successful, False otherwise
    """
    if conn is None:
        print("❌ Cannot delete table: OMERO connection is None")
        return False

    try:
        # Get table info first
        file_ann = conn.getObject("FileAnnotation", table_id)
        if not file_ann:
            print(f"❌ Table {table_id} not found")
            return False
        
        table_name = file_ann.getFile().getName() if file_ann.getFile() else f"ID:{table_id}"
        print(f"🗑️ Deleting table: {table_name}")
        
        # Delete the file annotation
        conn.deleteObjects("FileAnnotation", [table_id], wait=True)
        print(f"✅ Successfully deleted table {table_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting table {table_id}: {e}")
        return False


def backup_table(conn, table_id: int, backup_path: str) -> bool:
    """Export table data to local backup file.
    
    Args:
        conn: OMERO connection
        table_id: ID of table to backup
        backup_path: Local path for backup file (.csv extension recommended)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get table data
        df = ezomero.get_table(conn, table_id)
        
        # Create backup directory if needed
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(backup_path, index=False)
        
        print(f"💾 Table {table_id} backed up to: {backup_path}")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ Error backing up table {table_id}: {e}")
        return False


def validate_table_schema(conn, table_id: int, expected_columns: List[str]) -> Tuple[bool, List[str]]:
    """Validate that table has expected columns.
    
    Args:
        conn: OMERO connection
        table_id: ID of table to validate
        expected_columns: List of expected column names
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    try:
        df = ezomero.get_table(conn, table_id)
        actual_columns = set(df.columns)
        expected_set = set(expected_columns)
        
        missing_columns = list(expected_set - actual_columns)
        extra_columns = list(actual_columns - expected_set)
        
        is_valid = len(missing_columns) == 0
        
        if is_valid:
            print(f"✅ Table {table_id} schema is valid")
            if extra_columns:
                print(f"   Extra columns: {extra_columns}")
        else:
            print(f"❌ Table {table_id} schema is invalid")
            print(f"   Missing columns: {missing_columns}")
        
        return is_valid, missing_columns
        
    except Exception as e:
        print(f"❌ Error validating table {table_id}: {e}")
        return False, expected_columns


def merge_tables(conn, table_ids: List[int], new_title: str, 
                container_type: str, container_id: int) -> Optional[int]:
    """Merge multiple tracking tables into one.
    
    Args:
        conn: OMERO connection
        table_ids: List of table IDs to merge
        new_title: Title for the new merged table
        container_type: Container type for new table
        container_id: Container ID for new table
        
    Returns:
        New table ID if successful, None otherwise
    """
    try:
        # Load all tables
        dfs = []
        for table_id in table_ids:
            try:
                df = ezomero.get_table(conn, table_id)
                df['source_table_id'] = table_id  # Track source
                dfs.append(df)
                print(f"📊 Loaded table {table_id}: {len(df)} rows")
            except Exception as e:
                print(f"⚠️ Could not load table {table_id}: {e}")
        
        if not dfs:
            print("❌ No tables could be loaded")
            return None
        
        # Merge dataframes
        merged_df = pd.concat(dfs, ignore_index=True, sort=False)
        
        # Remove duplicates based on key columns if they exist
        key_columns = ['image_id', 'timepoint', 'z_slice', 'channel']
        existing_keys = [col for col in key_columns if col in merged_df.columns]
        
        if existing_keys:
            initial_rows = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=existing_keys, keep='first')
            final_rows = len(merged_df)
            
            if initial_rows != final_rows:
                print(f"🔄 Removed {initial_rows - final_rows} duplicate rows")
        
        # Create new table
        new_table_id = ezomero.post_table(
            conn,
            object_type=container_type.capitalize(),
            object_id=container_id,
            table=merged_df,
            title=new_title
        )
        
        print(f"✅ Created merged table '{new_title}' with {len(merged_df)} rows")
        print(f"   New table ID: {new_table_id}")
        
        return new_table_id
        
    except Exception as e:
        print(f"❌ Error merging tables: {e}")
        return None


# =============================================================================
# Annotation Management Utilities  
# =============================================================================

def list_annotations_by_namespace(conn, object_type: str, object_id: int, 
                                namespace: str) -> List[Dict]:
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
        # Get file annotation IDs
        ann_ids = ezomero.get_file_annotation_ids(conn, object_type, object_id)
        
        for ann_id in ann_ids:
            try:
                file_ann = conn.getObject("FileAnnotation", ann_id)
                if file_ann and file_ann.getNs() == namespace:
                    annotations.append({
                        'id': ann_id,
                        'namespace': file_ann.getNs(),
                        'description': file_ann.getDescription() or "",
                        'file_name': file_ann.getFile().getName() if file_ann.getFile() else "",
                        'file_size': file_ann.getFile().getSize() if file_ann.getFile() else 0
                    })
            except Exception:
                continue
        
        print(f"🔍 Found {len(annotations)} annotations with namespace '{namespace}'")
        
    except Exception as e:
        print(f"❌ Error listing annotations: {e}")
    
    return annotations


def delete_annotations_by_namespace(conn, object_type: str, object_id: int, 
                                  namespace: str) -> int:
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
        annotations = list_annotations_by_namespace(conn, object_type, object_id, namespace)
        
        if not annotations:
            print(f"🔍 No annotations found with namespace '{namespace}'")
            return 0
        
        # Delete annotations
        ann_ids = [ann['id'] for ann in annotations]
        conn.deleteObjects("FileAnnotation", ann_ids, wait=True)
        
        print(f"🗑️ Deleted {len(ann_ids)} annotations with namespace '{namespace}'")
        return len(ann_ids)
        
    except Exception as e:
        print(f"❌ Error deleting annotations: {e}")
        return 0


def validate_roi_integrity(conn, image_id: int) -> Dict[str, Any]:
    """Check ROI data integrity for an image.
    
    Args:
        conn: OMERO connection
        image_id: ID of image to check
        
    Returns:
        Dictionary with integrity check results
    """
    results = {
        'image_id': image_id,
        'total_rois': 0,
        'total_shapes': 0,
        'roi_types': {},
        'issues': [],
        'is_valid': True
    }
    
    try:
        # Get image object
        image = conn.getObject("Image", image_id)
        if not image:
            results['issues'].append(f"Image {image_id} not found")
            results['is_valid'] = False
            return results
        
        # Get ROIs
        roi_service = conn.getRoiService()
        result = roi_service.findByImage(image_id, None)
        
        results['total_rois'] = len(result.rois)
        
        for roi in result.rois:
            for shape in roi.getPrimaryIterator():
                results['total_shapes'] += 1
                
                shape_type = type(shape).__name__
                results['roi_types'][shape_type] = results['roi_types'].get(shape_type, 0) + 1
                
                # Check for basic validity
                try:
                    # Try to access basic shape properties
                    _ = shape.getTheZ()
                    _ = shape.getTheC() 
                    _ = shape.getTheT()
                except Exception as e:
                    results['issues'].append(f"Invalid shape properties: {e}")
                    results['is_valid'] = False
        
        print(f"🔍 ROI integrity check for image {image_id}:")
        print(f"   Total ROIs: {results['total_rois']}")
        print(f"   Total shapes: {results['total_shapes']}")
        print(f"   Shape types: {results['roi_types']}")
        
        if results['issues']:
            print(f"   Issues found: {len(results['issues'])}")
            results['is_valid'] = False
        else:
            print("   ✅ No issues found")
        
    except Exception as e:
        results['issues'].append(f"Error during integrity check: {e}")
        results['is_valid'] = False
        print(f"❌ Error checking ROI integrity: {e}")
    
    return results


# =============================================================================
# Connection and Error Handling Utilities
# =============================================================================

def get_server_info(conn) -> Dict[str, Any]:
    """Get OMERO server information and status.
    
    Args:
        conn: OMERO connection
        
    Returns:
        Dictionary with server information
    """
    info = {
        'server_version': 'Unknown',
        'user': 'Unknown',
        'group': 'Unknown',
        'session_id': 'Unknown',
        'is_admin': False,
        'connection_status': 'Unknown'
    }
    
    try:
        # Get basic connection info
        if conn.isConnected():
            info['connection_status'] = 'Connected'
            
            # Get user info
            user = conn.getUser()
            if user:
                info['user'] = user.getName()
                info['is_admin'] = user.isAdmin()
            
            # Get group info
            group = conn.getGroupFromContext()
            if group:
                info['group'] = group.getName()
            
            # Get session info
            session = conn.getSession()
            if session:
                info['session_id'] = str(session.getUuid())
            
            # Try to get server version
            try:
                config = conn.getConfigService()
                info['server_version'] = config.getVersion()
            except Exception:
                pass
                
        else:
            info['connection_status'] = 'Disconnected'
        
    except Exception as e:
        print(f"⚠️ Error getting server info: {e}")
        info['connection_status'] = f'Error: {e}'
    
    return info


def get_container_info(conn, container_type: str, container_id: int) -> Optional[Dict[str, Any]]:
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
    """
    if not conn or not conn.isConnected():
        return None

    try:
        container = conn.getObject(container_type.capitalize(), container_id)
        if not container:
            return None

        image_ids = []
        info = {
            'container_type': container_type,
            'container_id': container_id,
            'container_name': container.getName(),
            'total_images': 0,
            'sample_image_id': None,
            'sample_image_name': None,
            'dimensions': None,
            'pixel_sizes': None,
        }

        if container_type == 'screen':
            plate_ids = ezomero.get_plate_ids(conn, screen=container_id)
            info['total_plates'] = len(plate_ids)
            all_image_ids = []
            for plate_id in plate_ids:
                all_image_ids.extend(ezomero.get_image_ids(conn, plate=plate_id))
            image_ids = all_image_ids
        else:
            image_ids = ezomero.get_image_ids(conn, **{container_type: container_id})

        total_images = len(image_ids)
        info['total_images'] = total_images

        if total_images > 0:
            sample_image_id = image_ids[0]
            sample_image = conn.getObject("Image", sample_image_id)

            if sample_image:
                info['sample_image_id'] = sample_image_id
                info['sample_image_name'] = sample_image.getName()
                info['dimensions'] = {
                    'T': sample_image.getSizeT(),
                    'C': sample_image.getSizeC(),
                    'Z': sample_image.getSizeZ(),
                    'Y': sample_image.getSizeY(),
                    'X': sample_image.getSizeX(),
                }

                pixel_sizes = {}
                try:
                    size_x = sample_image.getPixelSizeX(units=True)
                    if size_x:
                        pixel_sizes['X'] = (size_x.getValue(), size_x.getSymbol())
                except AttributeError:
                    pass

                try:
                    size_y = sample_image.getPixelSizeY(units=True)
                    if size_y:
                        pixel_sizes['Y'] = (size_y.getValue(), size_y.getSymbol())
                except AttributeError:
                    pass

                try:
                    size_z = sample_image.getPixelSizeZ(units=True)
                    if size_z:
                        pixel_sizes['Z'] = (size_z.getValue(), size_z.getSymbol())
                except AttributeError:
                    pass

                info['pixel_sizes'] = pixel_sizes
        
        return info

    except Exception:
        return None


# =============================================================================
# Generic Image Loading Utilities
# =============================================================================

def get_table_by_name(conn, table_name: str, container_type: str = None, container_id: int = None):
    """Get OMERO table by name.
    
    This is a generic utility that could be contributed to ezomero in the future.
    
    Args:
        conn: OMERO connection
        table_name: Name of table to find
        container_type: Optional container type to search within
        container_id: Optional container ID to search within
        
    Returns:
        Table object if found, None otherwise
    """
    print(f"🔍 Searching for table: {table_name}")
    
    try:
        # Search strategy: Look through file annotations for tables
        # OMERO tables are stored as file annotations with specific content type
        
        if container_type and container_id:
            # Search within specific container
            try:
                # Get file annotations for the container
                annotations = ezomero.get_file_annotation_ids(conn, container_type.capitalize(), container_id)
                
                for ann_id in annotations:
                    try:
                        # Get annotation details
                        file_ann = conn.getObject("FileAnnotation", ann_id)
                        if file_ann and hasattr(file_ann, 'getFile'):
                            original_file = file_ann.getFile()
                            if original_file and hasattr(original_file, 'getName'):
                                file_name = original_file.getName()
                                
                                # Check if this is a table file and matches our name
                                if file_name and table_name in file_name:
                                    # Try to get the table
                                    try:
                                        table = conn.getSharedResources().openTable(original_file)
                                        if table:
                                            print(f"✅ Found table: {file_name} (ID: {ann_id})")
                                            return table
                                    except Exception:
                                        continue
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"⚠️ Error searching in container {container_type} {container_id}: {e}")
        
        # For now, we'll return None to trigger new table creation
        print(f"🔍 Table '{table_name}' not found, will create new table")
        return None
        
    except Exception as e:
        print(f"❌ Error searching for table {table_name}: {e}")
        return None


def get_dask_image_multiple(conn, image_list: List, timepoints: List[int], 
                           channels: List[int], z_slices: List[int]):
    """Load image data using dask for memory efficiency.
    
    This function creates dask arrays for lazy loading and only materializes
    them to numpy arrays when needed by downstream processing.
    
    This is a generic utility that could be contributed to ezomero in the future.
    
    Args:
        conn: OMERO connection
        image_list: List of OMERO image objects
        timepoints: List of timepoint indices
        channels: List of channel indices  
        z_slices: List of z-slice indices
        
    Returns:
        List of numpy arrays containing image data (materialized from dask)
    """
    if not image_list:
        return []
    
    print(f"Loading {len(image_list)} images using dask...")
    
    # Create dask arrays for each image
    dask_arrays = []
    for i, image in enumerate(image_list):
        if not image:
            continue
            
        try:
            # Create lazy-loaded dask array for this image
            dask_array = _create_dask_array_for_image(conn, image, timepoints, channels, z_slices)
            dask_arrays.append(dask_array)
            
        except Exception as e:
            print(f"⚠️ Could not create dask array for image {image.getId()}: {e}")
            # Fallback: create zeros array
            height = image.getSizeY()
            width = image.getSizeX()
            fallback_array = np.zeros((height, width), dtype=np.uint16)
            dask_arrays.append(fallback_array)
    
    # Materialize dask arrays to numpy arrays in chunks for memory efficiency
    print("Materializing dask arrays to numpy...")
    materialized_images = []
    
    # Process in smaller chunks to avoid memory issues
    chunk_size = min(3, len(dask_arrays))  # Process max 3 images at a time
    
    for chunk_start in range(0, len(dask_arrays), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(dask_arrays))
        chunk_arrays = dask_arrays[chunk_start:chunk_end]
        
        print(f"   Processing chunk {chunk_start//chunk_size + 1}/{(len(dask_arrays)-1)//chunk_size + 1}")
        
        # Compute this chunk
        for dask_array in chunk_arrays:
            if hasattr(dask_array, 'compute'):
                # It's a dask array, compute it
                numpy_array = dask_array.compute()
            else:
                # It's already a numpy array (fallback case)
                numpy_array = dask_array
            
            materialized_images.append(numpy_array)
    
    print(f"Successfully loaded {len(materialized_images)} images")
    return materialized_images


def get_dask_image_single(conn, image, timepoints: List[int], 
                         channels: List[int], z_slices: List[int]):
    """Load a single image using dask for memory efficiency.
    
    This is a generic utility that could be contributed to ezomero in the future.
    
    Args:
        conn: OMERO connection
        image: Single OMERO image object
        timepoints: List of timepoint indices
        channels: List of channel indices
        z_slices: List of z-slice indices
        
    Returns:
        Numpy array containing image data
    """
    if not image:
        return None
    
    try:
        dask_array = _create_dask_array_for_image(conn, image, timepoints, channels, z_slices)
        
        if hasattr(dask_array, 'compute'):
            return dask_array.compute()
        else:
            return dask_array
            
    except Exception as e:
        print(f"⚠️ Error with dask loading for image {image.getId()}: {e}")
        # Direct loading fallback
        if not image:
            return None
            
        pixels = image.getPrimaryPixels()
        
        # Use first timepoint, channel, z-slice if multiple provided
        t = timepoints[0] if timepoints else 0
        c = channels[0] if channels else 0
        z = z_slices[0] if z_slices else 0
        
        try:
            plane_data = pixels.getPlane(z, c, t)
            return plane_data
        except Exception as e2:
            print(f"⚠️ Could not load plane for image {image.getId()}: {e2}")
            # Fallback to zeros array
            height = image.getSizeY()
            width = image.getSizeX()
            return np.zeros((height, width), dtype=np.uint16)


def _create_dask_array_for_image(conn, image, timepoints: List[int], 
                                channels: List[int], z_slices: List[int]):
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
            print(f"⚠️ Error loading plane {z_idx},{c_idx},{t_idx} for image {image_id}: {e}")
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
        meta=np.array([], dtype=np.uint16)
    )
    
    # Rechunk for better performance
    dask_array = dask_array.rechunk(chunks)
    
    return dask_array
