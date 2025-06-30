"""OMERO integration functions for micro-SAM workflows."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from .omero_utils import delete_table

try:
    import ezomero
except ImportError:
    ezomero = None
    


def initialize_tracking_table(conn, table_title: str, processing_units: List[Tuple], 
                            container_type: str, container_id: int, source_desc: str) -> int:
    """Initialize tracking table for annotation process.
    
    Args:
        conn: OMERO connection
        table_title: Name for the tracking table
        processing_units: List of (image_id, sequence_val, metadata) tuples
        container_type: Type of OMERO container
        container_id: ID of container
        source_desc: Description
        
    Returns:
        Table ID
    """
    if ezomero is None:
        raise ImportError("ezomero is required. Install with: pip install -e .[omero]")
    
    # Create DataFrame from processing units
    df = pd.DataFrame(columns=[
        "image_id", "image_name", "train", "validate", 
        "channel", "z_slice", "timepoint", "sam_model", "embed_id", "label_id", "roi_id", 
        "is_volumetric", "processed", "is_patch", "patch_x", "patch_y", "patch_width", "patch_height",
        "schema_attachment_id"
    ])
    
    for img_id, seq_val, metadata in processing_units:
        # Get image object to get name and dimensions
        image = conn.getObject("Image", img_id)
        if not image:
            continue
            
        # Determine if this is training or validation
        is_train = metadata.get("category", "training") == "training"
        is_validate = not is_train
        
        # Handle patch information
        is_patch = "patch_x" in metadata
        patch_x = metadata.get("patch_x", 0)
        patch_y = metadata.get("patch_y", 0)
        
        if is_patch:
            # Use patch dimensions from config if available
            patch_width = metadata.get("patch_width", 512)
            patch_height = metadata.get("patch_height", 512)
        else:
            # Use full image dimensions
            patch_width = image.getSizeX()
            patch_height = image.getSizeY()
        
        model_type = metadata.get("model_type", "vit_l")
        
        new_row = pd.DataFrame([{
            "image_id": int(img_id),
            "image_name": image.getName(),
            "train": is_train,
            "validate": is_validate,
            "channel": metadata.get("channel", -1),
            "z_slice": metadata.get("z_slice", -1),
            "timepoint": metadata.get("timepoint", -1),
            "sam_model": model_type,
            "embed_id": -1,
            "label_id": -1,
            "roi_id": -1,
            "is_volumetric": metadata.get("three_d", False),
            "processed": False,
            "is_patch": is_patch,
            "patch_x": int(patch_x),
            "patch_y": int(patch_y),
            "patch_width": int(patch_width),
            "patch_height": int(patch_height),
            "schema_attachment_id": -1
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Ensure proper types for OMERO table
    numeric_columns = ['image_id', 'patch_x', 'patch_y', 'patch_width', 'patch_height', 'z_slice', 'timepoint']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
    
    boolean_columns = ['train', 'validate', 'processed', 'is_patch', 'is_volumetric']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).infer_objects(copy=False).astype(bool)
            
    id_columns = ['embed_id', 'label_id', 'roi_id', 'schema_attachment_id']
    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].fillna('None').infer_objects(copy=False).astype(str)
    
    # Create the table
    table_id = ezomero.post_table(
        conn,
        object_type=container_type.capitalize(),
        object_id=container_id,
        table=df,
        title=table_title
    )
    
    print(f"📋 Created tracking table '{table_title}' with {len(df)} units")
    print(f"   Container: {container_type} {container_id}")
    print(f"   Table ID: {table_id}")
    
    return table_id


def get_table_by_name(conn, table_name: str, container_type: str = None, container_id: int = None):
    """Get OMERO table by name.
    
    Args:
        conn: OMERO connection
        table_name: Name of table to find
        container_type: Optional container type to search within
        container_id: Optional container ID to search within
        
    Returns:
        Table object if found, None otherwise
    """
    if ezomero is None:
        print(f"🔍 Would search for table: {table_name}")
        return None
    
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
        
        # Fallback: Search in user's space (this is more complex and may be slow)
        # For now, we'll return None to trigger new table creation
        print(f"🔍 Table '{table_name}' not found, will create new table")
        return None
        
    except Exception as e:
        print(f"❌ Error searching for table {table_name}: {e}")
        return None


def get_annotation_configurations(conn):
    """Get stored annotation configurations."""
    # Stub implementation
    return {}


def get_unprocessed_units(conn, table_id: int) -> List[Tuple]:
    """Get unprocessed units from tracking table.
    
    Args:
        conn: OMERO connection
        table_id: ID of tracking table
        
    Returns:
        List of tuples: (image_id, sequence_val, metadata_dict, row_index)
    """
    if ezomero is None:
        raise ImportError("ezomero is required. Install with: pip install -e .[omero]")
    
    print(f"📋 Getting unprocessed units from table {table_id}")
    
    # Get the table data
    try:
        df = ezomero.get_table(conn, table_id)
    except Exception as e:
        print(f"❌ Error reading table {table_id}: {e}")
        return []
    
    # Filter for unprocessed rows
    unprocessed_df = df[~df['processed']] if 'processed' in df.columns else df
    
    if len(unprocessed_df) == 0:
        print("✅ All units already processed!")
        return []
    
    print(f"📋 Found {len(unprocessed_df)} unprocessed units")
    
    # Convert to processing units format
    processing_units = []
    for idx, row in unprocessed_df.iterrows():
        image_id = int(row['image_id'])
        sequence_val = 0 if row.get('train', True) else 1
        
        # Build metadata dict
        metadata = {
            'timepoint': int(row.get('timepoint', -1)),
            'z_slice': int(row.get('z_slice', -1)),
            'channel': int(row.get('channel', -1)),
            'three_d': bool(row.get('is_volumetric', False)),
            'model_type': str(row.get('sam_model', '')),
            'category': 'training' if row.get('train', True) else 'validation'
        }
        
        # Add patch info if it's a patch
        if row.get('is_patch', False):
            metadata.update({
                'patch_x': int(row.get('patch_x', 0)),
                'patch_y': int(row.get('patch_y', 0)),
                'patch_width': int(row.get('patch_width', 512)),
                'patch_height': int(row.get('patch_height', 512))
            })
        
        processing_units.append((image_id, sequence_val, metadata, idx))
    
    return processing_units


def update_tracking_table_rows(conn, table_id: int, row_indices: List[int], 
                              status: str, annotation_file: str) -> Optional[int]:
    """Update tracking table rows with processing status.
    
    This implementation updates the table by replacing it with a new one.
    
    Args:
        conn: OMERO connection
        table_id: ID of tracking table
        row_indices: List of row indices to update
        status: Status to set ('completed', 'failed', etc.)
        annotation_file: Path to annotation file (currently unused)
        
    Returns:
        New table ID if successful, else original table_id.
    """
    if ezomero is None:
        raise ImportError("ezomero is required for table operations. Install with: pip install -e .[omero]")
    
    try:
        # Get current table data
        df = ezomero.get_table(conn, table_id)
        if df is None:
            print(f"❌ Could not retrieve table {table_id}")
            return table_id

        # Get original table info (title, container)
        file_ann = conn.getObject("FileAnnotation", table_id)
        if not file_ann:
            print(f"❌ Could not find FileAnnotation for table {table_id}")
            return table_id
        
        table_title = file_ann.getFile().getName()
        
        container_type = None
        container_id = None

        # Find container by searching parents of images in the table
        if not df.empty and 'image_id' in df.columns:
            # Get unique image IDs to check
            image_ids = df['image_id'].unique()
            for img_id in image_ids:
                img = conn.getObject("Image", int(img_id))
                if img:
                    parents = list(img.listParents())
                    if parents:
                        parent = parents[0]
                        container_type = parent.__class__.__name__
                        container_id = parent.getId()
                        # Found a container, break loop
                        break
        
        if container_id is None:
            print("❌ Could not determine container for table. Cannot update.")
            return table_id

        # Update the rows in our DataFrame
        for idx in row_indices:
            if idx < len(df):
                df.loc[idx, 'processed'] = (status == 'completed')

        # Prepare DataFrame for OMERO: Handle all columns properly
        df_for_omero = df.copy()
        numeric_columns = ['image_id', 'patch_x', 'patch_y', 'patch_width', 'patch_height', 'z_slice', 'timepoint']
        for col in numeric_columns:
            if col in df_for_omero.columns:
                df_for_omero[col] = pd.to_numeric(df_for_omero[col], errors='coerce').fillna(-1).astype(int)
        
        boolean_columns = ['train', 'validate', 'processed', 'is_patch', 'is_volumetric']
        for col in boolean_columns:
            if col in df_for_omero.columns:
                df_for_omero[col] = df_for_omero[col].fillna(False).astype(bool)
        
        id_columns = ['embed_id', 'label_id', 'roi_id', 'schema_attachment_id']
        for col in id_columns:
            if col in df_for_omero.columns:
                df_for_omero[col] = df_for_omero[col].fillna('None').astype(str)

        # Try to delete the existing table
        if not delete_table(conn, table_id):
            print(f"Warning: Could not delete existing table: {table_id}")
            # If we can't delete the table, try with a new title to avoid conflicts
            table_title = f"{table_title}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Using alternative table title: {table_title}")

        # Create a new table with the updated data
        new_table_id = ezomero.post_table(
            conn,
            object_type=container_type.capitalize(),
            object_id=container_id,
            table=df_for_omero,
            title=table_title
        )
        
        if new_table_id != table_id:
            print(f"📋 Table updated: {table_id} → {new_table_id}")
        else:
            print(f"📋 Table updated with ID: {new_table_id}")
        return new_table_id
        
    except Exception as e:
        print(f"Error creating updated table: {e}")
        return table_id  # Return original values on error


def upload_rois_and_labels(conn, image_id: int, annotation_file: str, 
                          patch_offset: Optional[Tuple[int, int]] = None):
    """Upload ROIs and labels to OMERO image.
    
    Args:
        conn: OMERO connection
        image_id: ID of OMERO image
        annotation_file: Path to annotation file (TIFF with labels)
        patch_offset: Optional (x,y) offset for patch placement
        
    Returns:
        tuple: (label_id, roi_id) - IDs of uploaded label file and ROI collection
    """
    try:
        from ..processing.image_functions import label_to_rois
        import imageio.v3 as imageio
    except ImportError as e:
        print(f"❌ Missing dependencies for ROI creation: {e}")
        print("💡 Install with: pip install opencv-python imageio")
        return None, None
    
    if ezomero is None:
        raise ImportError("ezomero is required for OMERO operations. Install with: pip install -e .[omero]")
    
    try:
        # Get image object
        image = conn.getObject("Image", image_id)
        if not image:
            print(f"❌ Image {image_id} not found")
            return None, None
        
        # Load label image
        try:
            label_img = imageio.imread(annotation_file)
        except Exception as e:
            print(f"❌ Could not read label image {annotation_file}: {e}")
            return None, None
        
        # Get metadata from image for ROI creation
        z_slice = 0  # Default to first z-slice
        channel = 0  # Default to first channel  
        timepoint = 0  # Default to first timepoint
        model_type = "vit_b_lm"  # Default model type
        
        # Create ROI shapes from label image
        try:
            shapes = label_to_rois(
                label_img=label_img,
                z_slice=z_slice,
                channel=channel,
                timepoint=timepoint,
                model_type=model_type,
                is_volumetric=False,  # Assume 2D for now
                patch_offset=patch_offset
            )
        except Exception as e:
            print(f"❌ Error creating ROI shapes: {e}")
            shapes = []
        
        # Upload label file as attachment
        label_desc = f"Micro-SAM segmentation ({model_type})"
        if patch_offset:
            label_desc += f", Patch offset: ({patch_offset[0]}, {patch_offset[1]})"
            
        file_ann_id = ezomero.post_file_annotation(
            conn, 
            file_path=annotation_file, 
            description=label_desc,
            ns="openmicroscopy.org/omero/microsam",
            object_type="Image",
            object_id=image_id
        )
        
        # File annotation is automatically linked when created with post_file_annotation
        # No separate linking step needed in ezomero
        
        # Upload ROI shapes if any were created
        roi_id = None
        if shapes:
            try:
                roi_id = ezomero.post_roi(conn, image_id, shapes)
                print(f"☁️ Created {len(shapes)} ROI shapes for image {image_id}")
            except Exception as e:
                print(f"⚠️ Could not upload ROI shapes: {e}")
                print("   Label file uploaded successfully")
        else:
            print(f"⚠️ No ROI shapes created from {annotation_file}")
        print(f"☁️ Uploaded annotations from {annotation_file} to image {image_id}")
        if patch_offset:
            print(f"   Patch offset: {patch_offset}")
        print(f"   File annotation ID: {file_ann_id}")
        if roi_id:
            print(f"   ROI ID: {roi_id}")
        
        return file_ann_id, roi_id
        
    except Exception as e:
        # Check for specific OMERO API errors
        error_msg = str(e)
        if "findByQuery" in error_msg and "has returned more than one Object" in error_msg:
            print("❌ OMERO API Query Error: Multiple objects found when expecting single result")
            print("   This usually indicates missing object_type/object_id parameters in ezomero calls")
            print(f"   Error details: {error_msg}")
        else:
            print(f"❌ Error uploading annotations to image {image_id}: {e}")
        return None, None


def get_dask_dimensions(conn, image_list: List):
    """Get dimensions for dask array creation."""
    # Stub implementation
    if image_list:
        img = image_list[0]
        return {
            'sizeT': img.getSizeT(),
            'sizeZ': img.getSizeZ(), 
            'sizeY': img.getSizeY(),
            'sizeX': img.getSizeX(),
            'sizeC': img.getSizeC()
        }
    return {}


def get_dask_image_multiple(conn, image_list: List, timepoints: List[int], 
                           channels: List[int], z_slices: List[int]):
    """Load image data using dask for memory efficiency.
    
    This function creates dask arrays for lazy loading and only materializes
    them to numpy arrays when needed by micro-SAM.
    
    Args:
        conn: OMERO connection
        image_list: List of OMERO image objects
        timepoints: List of timepoint indices
        channels: List of channel indices  
        z_slices: List of z-slice indices
        
    Returns:
        List of numpy arrays containing image data (materialized from dask)
    """
    try:
        import dask.array as da
        from dask import delayed
    except ImportError:
        print("⚠️ dask not available, falling back to direct loading")
        return _load_images_direct(conn, image_list, timepoints, channels, z_slices)
    
    if not image_list:
        return []
    
    print(f"📊 Loading {len(image_list)} images using dask...")
    
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
    print("💾 Materializing dask arrays to numpy (required for micro-SAM)...")
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
    
    print(f"✅ Successfully loaded {len(materialized_images)} images")
    return materialized_images


def _create_dask_array_for_image(conn, image, timepoints: List[int], 
                                channels: List[int], z_slices: List[int]):
    """Create a dask array for a single OMERO image with lazy loading.
    
    Args:
        conn: OMERO connection
        image: OMERO image object
        timepoints: List of timepoint indices
        channels: List of channel indices
        z_slices: List of z-slice indices
          Returns:
        Dask array for the image
    """
    try:
        import dask.array as da
        from dask import delayed
    except ImportError:
        # Fallback to direct loading
        return _load_single_image_direct(conn, image, timepoints, channels, z_slices)
    
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


def get_dask_image_single(conn, image, timepoints: List[int], 
                         channels: List[int], z_slices: List[int]):
    """Load a single image using dask for memory efficiency.
    
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
        return _load_single_image_direct(conn, image, timepoints, channels, z_slices)


def _load_single_image_direct(conn, image, timepoints: List[int], 
                             channels: List[int], z_slices: List[int]):
    """Direct loading fallback for a single image."""
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
    except Exception as e:
        print(f"⚠️ Could not load plane for image {image.getId()}: {e}")
        # Fallback to zeros array
        height = image.getSizeY()
        width = image.getSizeX()
        return np.zeros((height, width), dtype=np.uint16)


def _load_images_direct(conn, image_list: List, timepoints: List[int], 
                       channels: List[int], z_slices: List[int]):
    """Direct image loading fallback when dask is not available."""
    import numpy as np
    
    images = []
    for image in image_list:
        if not image:
            continue
            
        pixels = image.getPrimaryPixels()
        
        # Use first timepoint, channel, z-slice if multiple provided
        t = timepoints[0] if timepoints else 0
        c = channels[0] if channels else 0
        z = z_slices[0] if z_slices else 0
        
        try:
            plane_data = pixels.getPlane(z, c, t)
            images.append(plane_data)
        except Exception as e:
            print(f"Warning: Could not load plane for image {image.getId()}: {e}")
            # Fallback to zeros array
            height = image.getSizeY()
            width = image.getSizeX()
            images.append(np.zeros((height, width), dtype=np.uint16))
    
    return images