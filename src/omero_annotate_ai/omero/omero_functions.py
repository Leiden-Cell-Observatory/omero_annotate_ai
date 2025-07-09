"""OMERO integration functions for micro-SAM workflows."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

from .omero_utils import (
    delete_table, 
    get_table_by_name,
    get_dask_image_multiple,
    get_dask_image_single
)

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
                              status: str, annotation_file: str, 
                              container_type: str, container_id: int) -> Optional[int]:
    """Update tracking table rows with processing status.
    
    This implementation updates the table by replacing it with a new one.
    
    Args:
        conn: OMERO connection
        table_id: ID of tracking table
        row_indices: List of row indices to update
        status: Status to set ('completed', 'failed', etc.)
        annotation_file: Path to annotation file (currently unused)
        container_type: Type of OMERO container (e.g. 'dataset', 'project')
        container_id: ID of the container
        
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

        # Generate new table title to avoid conflicts
        table_title = f"microsam_training_{container_type}_{container_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Using table title: {table_title}")

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

        # Create a new table with the updated data using the provided container info
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


# =============================================================================
# Annotation Table Management Functions
# =============================================================================

def list_annotation_tables_for_project(conn, project_id: int) -> List[Dict[str, Any]]:
    """Find all micro-SAM annotation tables for a project.
    
    Args:
        conn: OMERO connection
        project_id: Project ID to search in
        
    Returns:
        List of dictionaries with table information including progress
    """
    return list_annotation_tables_for_container(conn, 'project', project_id)


def list_annotation_tables_for_container(conn, container_type: str, container_id: int) -> List[Dict[str, Any]]:
    """Find all micro-SAM annotation tables for a container.
    
    Args:
        conn: OMERO connection
        container_type: Type of container ('project', 'dataset', 'plate', 'screen')
        container_id: Container ID to search in
        
    Returns:
        List of dictionaries with table information including progress
    """
    if ezomero is None:
        raise ImportError("ezomero is required. Install with: pip install -e .[omero]")
    
    from .omero_utils import list_user_tables
    
    # Get all tables in the container
    all_tables = list_user_tables(conn, container_type=container_type, container_id=container_id)
    
    # Filter for micro-SAM annotation tables
    annotation_tables = []
    
    for table_info in all_tables:
        table_name = table_info.get('name', '')
        
        # Check if this is a micro-SAM annotation table
        if (table_name.startswith('microsam_training_') or 
            table_name.startswith('microsam_annotation_') or
            'microsam' in table_name.lower()):
            
            # Add progress information
            try:
                progress_info = analyze_table_completion_status(conn, table_info['id'])
                table_info.update(progress_info)
            except Exception as e:
                print(f"⚠️ Could not analyze table {table_name}: {e}")
                table_info.update({
                    'total_units': 0,
                    'completed_units': 0,
                    'progress_percent': 0,
                    'is_complete': False,
                    'status': 'error'
                })
            
            annotation_tables.append(table_info)
    
    # Sort by creation date (newest first)
    annotation_tables.sort(key=lambda x: x.get('created_date', ''), reverse=True)
    
    return annotation_tables


def generate_unique_table_name(conn, project_id: int, base_name: str = None) -> str:
    """Generate a unique table name for the project.
    
    Args:
        conn: OMERO connection
        project_id: Project ID
        base_name: Optional base name for the table
        
    Returns:
        Unique table name
    """
    return generate_unique_table_name_for_container(conn, 'project', project_id, base_name)


def generate_unique_table_name_for_container(conn, container_type: str, container_id: int, base_name: str = None) -> str:
    """Generate a unique table name for a container.
    
    Args:
        conn: OMERO connection
        container_type: Type of container ('project', 'dataset', 'plate', 'screen')
        container_id: Container ID
        base_name: Optional base name for the table
        
    Returns:
        Unique table name
    """
    import datetime
    
    # Get container name for better naming
    try:
        container = conn.getObject(container_type.capitalize(), container_id)
        container_name = container.getName() if container else f"{container_type}_{container_id}"
        # Clean container name for use in table name
        container_name = "".join(c for c in container_name if c.isalnum() or c in "_-").lower()
    except:
        container_name = f"{container_type}_{container_id}"
    
    # Create base name if not provided
    if not base_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"microsam_{container_name}_{timestamp}"
    
    # Check if name already exists and make it unique
    existing_tables = list_annotation_tables_for_container(conn, container_type, container_id)
    existing_names = {table['name'] for table in existing_tables}
    
    unique_name = base_name
    counter = 1
    
    while unique_name in existing_names:
        unique_name = f"{base_name}_v{counter}"
        counter += 1
    
    return unique_name


def analyze_table_completion_status(conn, table_id: int) -> Dict[str, Any]:
    """Analyze the completion status of an annotation table.
    
    Args:
        conn: OMERO connection
        table_id: Table ID to analyze
        
    Returns:
        Dictionary with progress information
    """
    if ezomero is None:
        raise ImportError("ezomero is required. Install with: pip install -e .[omero]")
    
    try:
        # Get table data
        table_data = ezomero.get_table(conn, table_id)
        
        if table_data is None or table_data.empty:
            return {
                'total_units': 0,
                'completed_units': 0,
                'progress_percent': 0,
                'is_complete': False,
                'status': 'empty',
                'error': 'Table is empty or could not be read'
            }
        
        # Analyze progress based on table structure
        total_units = len(table_data)
        
        # Check for 'processed' column or similar completion indicators
        completed_units = 0
        completion_columns = ['processed', 'completed', 'finished', 'done']
        
        completion_column = None
        for col in completion_columns:
            if col in table_data.columns:
                completion_column = col
                break
        
        if completion_column:
            # Count completed units
            completed_units = table_data[completion_column].sum() if table_data[completion_column].dtype == bool else \
                             (table_data[completion_column] == True).sum()
        else:
            # Check for roi_id or label_id columns as completion indicators
            if 'roi_id' in table_data.columns:
                completed_units = table_data['roi_id'].notna().sum()
            elif 'label_id' in table_data.columns:
                completed_units = table_data['label_id'].notna().sum()
        
        # Calculate progress
        progress_percent = (completed_units / total_units * 100) if total_units > 0 else 0
        is_complete = progress_percent >= 100
        
        # Determine status
        if is_complete:
            status = 'complete'
        elif completed_units > 0:
            status = 'in_progress'
        else:
            status = 'not_started'
        
        return {
            'total_units': int(total_units),
            'completed_units': int(completed_units),
            'progress_percent': round(progress_percent, 1),
            'is_complete': is_complete,
            'status': status,
            'table_size': len(table_data),
            'columns': list(table_data.columns)
        }
        
    except Exception as e:
        return {
            'total_units': 0,
            'completed_units': 0,
            'progress_percent': 0,
            'is_complete': False,
            'status': 'error',
            'error': str(e)
        }


def get_table_progress_summary(conn, table_id: int) -> str:
    """Get a human-readable progress summary for a table.
    
    Args:
        conn: OMERO connection
        table_id: Table ID
        
    Returns:
        Progress summary string
    """
    progress = analyze_table_completion_status(conn, table_id)
    
    if progress['status'] == 'error':
        return f"❌ Error: {progress.get('error', 'Unknown error')}"
    
    total = progress['total_units']
    completed = progress['completed_units']
    percent = progress['progress_percent']
    
    status_emoji = {
        'complete': '✅',
        'in_progress': '🔄',
        'not_started': '⏳',
        'empty': '📋'
    }
    
    emoji = status_emoji.get(progress['status'], '❓')
    
    return f"{emoji} {completed}/{total} units ({percent:.1f}% complete)"


def create_roi_namespace_for_table(table_name: str) -> str:
    """Create a consistent ROI namespace for a table.
    
    Args:
        table_name: Name of the annotation table
        
    Returns:
        ROI namespace string
    """
    return f"omero_annotate_ai.table.{table_name}"


def get_tables_by_roi_namespace(conn, project_id: int, namespace: str) -> List[int]:
    """Find tables that have ROIs with a specific namespace.
    
    Args:
        conn: OMERO connection
        project_id: Project ID to search in
        namespace: ROI namespace to look for
        
    Returns:
        List of table IDs
    """
    # This would require scanning ROIs across the project
    # Implementation depends on OMERO's ROI search capabilities
    # For now, return empty list - can be enhanced later
    return []


