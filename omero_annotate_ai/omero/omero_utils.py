"""
Generic OMERO utility functions for table and annotation management.

This module provides reusable utilities for common OMERO operations that are
useful across different workflows and not specific to micro-SAM annotation.
"""

import os
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd

try:
    import ezomero
    EZOMERO_AVAILABLE = True
except ImportError:
    EZOMERO_AVAILABLE = False
    ezomero = None


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
    if not EZOMERO_AVAILABLE:
        print("⚠️ ezomero not available - cannot list tables")
        return []
    
    tables = []
    
    try:
        if container_type and container_id:
            # Search within specific container
            annotations = ezomero.get_file_annotation_ids(conn, container_type.capitalize(), container_id)
            
            for ann_id in annotations:
                try:
                    file_ann = conn.getObject("FileAnnotation", ann_id)
                    if file_ann and hasattr(file_ann, 'getFile'):
                        original_file = file_ann.getFile()
                        if original_file:
                            file_name = original_file.getName()
                            # Check if this looks like a table file
                            if file_name and ('.csv' in file_name or 'table' in file_name.lower()):
                                tables.append({
                                    'id': ann_id,
                                    'name': file_name,
                                    'container_type': container_type,
                                    'container_id': container_id,
                                    'description': file_ann.getDescription() or "",
                                    'namespace': file_ann.getNs() or ""
                                })
                except Exception:
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
    if not EZOMERO_AVAILABLE:
        print("⚠️ ezomero not available - cannot delete table")
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
    if not EZOMERO_AVAILABLE:
        print("⚠️ ezomero not available - cannot backup table")
        return False
    
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
    if not EZOMERO_AVAILABLE:
        print("⚠️ ezomero not available - cannot validate table")
        return False, expected_columns
    
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
    if not EZOMERO_AVAILABLE:
        print("⚠️ ezomero not available - cannot merge tables")
        return None
    
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
    if not EZOMERO_AVAILABLE:
        print("⚠️ ezomero not available - cannot list annotations")
        return []
    
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
    if not EZOMERO_AVAILABLE:
        print("⚠️ ezomero not available - cannot delete annotations")
        return 0
    
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

def with_retry(func, max_retries: int = 3, *args, **kwargs):
    """Wrapper for OMERO operations with retry logic.
    
    Args:
        func: Function to execute with retry
        max_retries: Maximum number of retry attempts
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Function result or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                print(f"🔄 Retrying... ({attempt + 2}/{max_retries})")
            else:
                print(f"❌ All {max_retries} attempts failed. Last error: {e}")
                return None


def validate_omero_permissions(conn, operation: str, object_type: str, object_id: int) -> bool:
    """Check if user has required permissions for an operation.
    
    Args:
        conn: OMERO connection
        operation: Operation type ('read', 'write', 'delete')
        object_type: OMERO object type
        object_id: Object ID
        
    Returns:
        True if user has permission, False otherwise
    """
    try:
        obj = conn.getObject(object_type, object_id)
        if not obj:
            print(f"❌ {object_type} {object_id} not found")
            return False
        
        # Basic permission check based on object accessibility
        can_read = obj is not None
        can_write = obj.canEdit() if hasattr(obj, 'canEdit') else False
        can_delete = obj.canDelete() if hasattr(obj, 'canDelete') else False
        
        permissions = {
            'read': can_read,
            'write': can_write, 
            'delete': can_delete
        }
        
        has_permission = permissions.get(operation.lower(), False)
        
        if has_permission:
            print(f"✅ User has {operation} permission for {object_type} {object_id}")
        else:
            print(f"❌ User lacks {operation} permission for {object_type} {object_id}")
        
        return has_permission
        
    except Exception as e:
        print(f"❌ Error checking permissions: {e}")
        return False


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