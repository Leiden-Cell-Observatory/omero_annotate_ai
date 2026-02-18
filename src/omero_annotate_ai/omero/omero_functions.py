"""OMERO integration functions for micro-SAM workflows."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .omero_utils import delete_table

import ezomero

import imageio.v3 as imageio

from ..processing.image_functions import label_to_rois


# =============================================================================
# Helper Functions
# =============================================================================


def _prepare_dataframe_for_omero(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame with proper types for OMERO table storage.

    Args:
        df: DataFrame to prepare

    Returns:
        DataFrame with properly typed columns for OMERO
    """
    df = df.copy()

    numeric_columns = [
        "image_id", "patch_x", "patch_y", "patch_width", "patch_height",
        "z_slice", "timepoint", "z_start", "z_end", "z_length", "channel",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    boolean_columns = ["train", "validate", "processed", "is_patch", "is_volumetric"]
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    id_columns = ["label_id", "roi_id", "schema_attachment_id"]
    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    string_columns = ["annotation_type"]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna("segmentation_mask").astype(str)

    datetime_columns = ["annotation_created_at", "annotation_updated_at"]
    for col in datetime_columns:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    return df


# =============================================================================
# Config-First Table Management Functions
# =============================================================================


def link_table_to_containers(
    conn,
    table_id: int,
    container_type: str,
    container_ids: List[int],
) -> Dict[int, int]:
    """Link an existing table (FileAnnotation) to multiple containers.

    Uses OMERO AnnotationLink objects (e.g., PlateAnnotationLinkI) to attach
    the same FileAnnotation to multiple containers. This allows a single table
    to appear in multiple containers without duplicating the underlying data.

    Note: table_id IS the FileAnnotation ID (ezomero.post_table returns FileAnnotation ID)

    Args:
        conn: OMERO connection
        table_id: The table/FileAnnotation ID to link
        container_type: Type of containers ('dataset', 'plate', 'project', 'screen')
        container_ids: List of container IDs to link the table to

    Returns:
        Dict mapping container_id to link_id

    Raises:
        Exception: If linking fails for any container
    """
    import omero.model as model

    link_classes = {
        'dataset': (model.DatasetAnnotationLinkI, model.DatasetI),
        'plate': (model.PlateAnnotationLinkI, model.PlateI),
        'project': (model.ProjectAnnotationLinkI, model.ProjectI),
        'screen': (model.ScreenAnnotationLinkI, model.ScreenI),
    }

    container_type_lower = container_type.lower()
    if container_type_lower not in link_classes:
        print(f"Warning: Cannot link annotation to container type '{container_type}'")
        return {cid: None for cid in container_ids}

    # table_id is the FileAnnotation ID - get the wrapper object
    file_ann = conn.getObject("FileAnnotation", table_id)
    if not file_ann:
        raise ValueError(f"FileAnnotation {table_id} not found")

    LinkClass, ContainerClass = link_classes[container_type_lower]
    results = {}
    update_service = conn.getUpdateService()

    for container_id in container_ids:
        link = LinkClass()
        link.setParent(ContainerClass(container_id, False))
        link.setChild(file_ann._obj)  # Access underlying OMERO model object
        saved_link = update_service.saveAndReturnObject(link)
        link_id = saved_link.getId().getValue()
        results[container_id] = link_id
        print(f"  Linked table to {container_type} {container_id} (link ID: {link_id})")

    return results


def create_or_replace_tracking_table(
    conn,
    config_df: pd.DataFrame,
    table_title: str,
    container_type: str,
    container_id: Optional[int] = None,
    container_ids: Optional[List[int]] = None,
    existing_table_id: Optional[int] = None,
) -> int:
    """Create new tracking table or replace existing one (delete + recreate pattern).

    Supports attaching the table to multiple containers. The table is created
    attached to the primary container, then linked to additional containers
    using OMERO AnnotationLink objects.

    Args:
        conn: OMERO connection
        config_df: DataFrame from config.to_dataframe()
        table_title: Name for the tracking table
        container_type: Type of OMERO container
        container_id: ID of primary container (legacy, use container_ids for multiple)
        container_ids: List of container IDs to attach the table to
        existing_table_id: Optional existing table to replace

    Returns:
        New table ID
    """
    # Resolve container IDs - container_ids takes precedence
    if container_ids is not None and len(container_ids) > 0:
        all_container_ids = container_ids
    elif container_id is not None and container_id != 0:
        all_container_ids = [container_id]
    else:
        raise ValueError("Either container_id or container_ids must be provided")

    primary_container_id = all_container_ids[0]

    # Delete existing table if provided
    if existing_table_id is not None:
        print(f"Deleting table: {table_title}")
        if not delete_table(conn, existing_table_id):
            print(f"Warning: Could not delete existing table: {existing_table_id}")

    # Create new table attached to primary container
    new_table_id = ezomero.post_table(
        conn,
        object_type=container_type.capitalize(),
        object_id=primary_container_id,
        table=config_df,
        title=table_title,
    )

    if new_table_id is None:
        raise RuntimeError(f"Failed to create table '{table_title}' in {container_type} {primary_container_id}")

    print(f"Created/replaced tracking table '{table_title}' with {len(config_df)} units")
    print(f"   Primary container: {container_type} {primary_container_id}")
    print(f"   Table ID: {new_table_id}")

    # Link to additional containers if multiple are specified
    if len(all_container_ids) > 1:
        additional_ids = all_container_ids[1:]
        print(f"Linking table to {len(additional_ids)} additional container(s)...")
        link_table_to_containers(conn, new_table_id, container_type, additional_ids)

    return new_table_id


def sync_config_to_omero_table(
    conn,
    config,  # AnnotationConfig object
    table_title: str,
    container_type: str,
    container_id: Optional[int] = None,
    container_ids: Optional[List[int]] = None,
    existing_table_id: Optional[int] = None,
) -> int:
    """High-level sync: config.annotations → OMERO table.

    Supports attaching the table to multiple containers.

    Args:
        conn: OMERO connection
        config: AnnotationConfig object with annotations
        table_title: Name for the tracking table
        container_type: Type of OMERO container
        container_id: ID of primary container (legacy, use container_ids for multiple)
        container_ids: List of container IDs to attach the table to
        existing_table_id: Optional existing table to replace

    Returns:
        New table ID
    """
    # Convert config annotations to DataFrame
    config_df = config.to_dataframe()

    if config_df.empty:
        print("Warning: No annotations in config to sync")
        return existing_table_id if existing_table_id else -1

    # Create or replace table
    return create_or_replace_tracking_table(
        conn=conn,
        config_df=config_df,
        table_title=table_title,
        container_type=container_type,
        container_id=container_id,
        container_ids=container_ids,
        existing_table_id=existing_table_id,
    )


def sync_omero_table_to_config(conn, table_id: int, config):
    """High-level sync: OMERO table → config.annotations.
    
    Args:
        conn: OMERO connection
        table_id: ID of OMERO table to load
        config: AnnotationConfig object to populate
    """
    try:
        # Get table data
        df = ezomero.get_table(conn, table_id)
        if df is None or df.empty:
            print(f"Warning: Table {table_id} is empty or could not be read")
            return
            
        # Load into config
        config.from_dataframe(df)
        print(f"Loaded {len(config.annotations)} annotations from table {table_id}")
        
    except Exception as e:
        print(f"Error loading table {table_id} into config: {e}")

def upload_annotation_config_to_omero(
        conn, 
        object_type: str, 
        object_id: int, 
        file_path: Optional[str] = None
) -> int:
    """
    Upload the annotation configuration file to OMERO.

    Args:
        conn: OMERO connection
        object_type: Type of OMERO object (e.g., "Image", "Dataset")
        object_id: ID of the OMERO object
        file_path: Path to the YAML configuration file

    Returns:
        ID of the uploaded file annotation
    """
    id = ezomero.post_file_annotation(conn,
                                      file_path=file_path,
                                      ns="openmicroscopy.org/omero/annotate/config",
                                      object_type=object_type,
                                      object_id=object_id)
    return id


def download_annotation_config_from_omero(conn, object_type: str, object_id: int) -> Optional[Any]:
    """Download and parse the annotation config YAML attached to an OMERO container.

    Finds the most recent FileAnnotation with namespace
    'openmicroscopy.org/omero/annotate/config' on the given container,
    downloads it to a temp file, and parses it into an AnnotationConfig.

    Args:
        conn: OMERO connection
        object_type: Container type ('Dataset', 'Plate', 'Project', 'Screen')
        object_id: Container ID

    Returns:
        AnnotationConfig if found, else None
    """
    import tempfile
    from ..core.annotation_config import AnnotationConfig
    from .omero_utils import list_annotations_by_namespace

    CONFIG_NS = "openmicroscopy.org/omero/annotate/config"
    annotations = list_annotations_by_namespace(conn, object_type, object_id, CONFIG_NS)
    if not annotations:
        return None

    ann_id = annotations[-1]["id"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = ezomero.get_file_annotation(conn, ann_id, tmp_dir)
        if file_path is None:
            return None
        return AnnotationConfig.from_yaml(file_path)


def upload_rois_and_labels(
    conn,
    image_id: int,
    annotation_file: str,
    patch_offset: Optional[Tuple[int, int]] = None,
    trainingset_name: Optional[str] = None,
    trainingset_description: Optional[str] = None,
    timepoint: Optional[int] = None,
    z_slice: Optional[int] = None,
    channel: Optional[int] = None,
    is_volumetric: Optional[bool] = False,
    z_start: Optional[int] = 0,
):
    """Upload ROIs and labels to OMERO image.

    Args:
        conn: OMERO connection
        image_id: ID of OMERO image
        annotation_file: Path to annotation file (TIFF with labels)
        patch_offset: Optional (x,y) offset for patch placement
        trainingset_name: Optional training set name for custom annotation naming
        trainingset_description: Optional training set description for annotation
        timepoint: Optional timepoint for positioning the roi properly in OMERO
        z_slice: Optional z_slice for positioning the roi properly in OMERO
        channel: Optional channel for positioning the roi properly in OMERO
        is_volumetric: check if we need to handle roi as 3D
        z_start: use for volumetric to use z_start as offset in the stack

    Returns:
        tuple: (label_id, roi_id) - IDs of uploaded label file and ROI collection
    """
    # Load label image
    print(f"Step 1: Loading label image from {annotation_file}")
    label_img = imageio.imread(annotation_file)
    print(f"Label image loaded: {label_img.shape}, dtype: {label_img.dtype}")
    unique_labels = np.unique(label_img)
    print(f"Found {len(unique_labels)} unique labels: {unique_labels[:10]}...")

    # Create ROI shapes from label image
    print("Step 2: Converting labels to ROI shapes...")
    effective_z = z_start if is_volumetric else z_slice
    shapes = label_to_rois(
        label_img=label_img,
        z_slice=effective_z,
        channel=channel,
        timepoint=timepoint,
        is_volumetric=is_volumetric,
        patch_offset=patch_offset,
    )
    print(f"Created {len(shapes)} ROI shapes from labels")

    # Upload label file as attachment
    print("Step 3: Uploading label file as attachment")

    # Build description
    label_desc = trainingset_description or ""
    if patch_offset:
        separator = " | " if label_desc else ""
        label_desc += f"{separator}Patch offset: ({patch_offset[0]}, {patch_offset[1]})"

    file_ann_id = ezomero.post_file_annotation(
        conn,
        file_path=annotation_file,
        description=label_desc,
        ns="openmicroscopy.org/omero/annotate/labels",
        object_type="Image",
        object_id=image_id,
    )
    print(f"File annotation uploaded with ID: {file_ann_id}")

    # Upload ROI shapes if any were created
    print("Step 4: Uploading ROI shapes")
    roi_id = None
    if shapes:
        roi_name = f"{trainingset_name}_ROIs" if trainingset_name else "ROIs"
        roi_description = label_desc  # Use same description as label file

        roi_id = ezomero.post_roi(
            conn, image_id, shapes, name=roi_name, description=roi_description
        )
        print(f"Created {len(shapes)} ROI shapes for image {image_id} with ID: {roi_id}")
    else:
        print(f"No ROI shapes created from {annotation_file}")

    print(f"Uploaded annotations from {annotation_file} to image {image_id}")
    if patch_offset:
        print(f"   Patch offset: {patch_offset}")
    print(f"   File annotation ID: {file_ann_id}")
    if roi_id:
        print(f"   ROI ID: {roi_id}")

    return file_ann_id, roi_id


def upload_label_input_image(
    conn,
    image_id: int,
    label_input_file: str,
    trainingset_name: Optional[str] = None,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    z_slice: Optional[int] = None,
) -> int:
    """Upload label input channel image as file annotation to OMERO.

    This function uploads the raw image data from the label channel
    (used for segmentation) as a file annotation. This is useful when
    using separate channels for labeling vs training, allowing the
    label channel images to be stored alongside the segmentation masks.

    Args:
        conn: OMERO connection
        image_id: ID of OMERO image to attach annotation to
        label_input_file: Path to the label input image file (TIFF)
        trainingset_name: Optional training set name for description
        channel: Optional channel index for description
        timepoint: Optional timepoint for description
        z_slice: Optional z-slice for description

    Returns:
        file_ann_id: OMERO file annotation ID
    """
    # Build description with available metadata
    desc_parts = ["Label input image"]
    if channel is not None:
        desc_parts.append(f"channel={channel}")
    if timepoint is not None:
        desc_parts.append(f"t={timepoint}")
    if z_slice is not None:
        desc_parts.append(f"z={z_slice}")
    if trainingset_name:
        desc_parts.append(f"trainingset={trainingset_name}")

    description = " | ".join(desc_parts)

    file_ann_id = ezomero.post_file_annotation(
        conn,
        file_path=label_input_file,
        description=description,
        ns="openmicroscopy.org/omero/annotate/label_input",
        object_type="Image",
        object_id=image_id,
    )

    print(f"Uploaded label input image to OMERO image {image_id}, annotation ID: {file_ann_id}")

    return file_ann_id


# =============================================================================
# Workflow Status Tracking Functions
# =============================================================================


def update_workflow_status_map(
    conn, container_type: str, container_id: int, table_id: int
) -> Optional[int]:
    """Update workflow status map annotation after batch completion.

    Args:
        conn: OMERO connection
        container_type: Type of OMERO container
        container_id: ID of container
        table_id: ID of tracking table

    Returns:
        Map annotation ID if successful, None otherwise
    """
    try:
        # Get current table progress
        df = ezomero.get_table(conn, table_id)
        total_units = len(df)
        completed_units = df["processed"].sum() if "processed" in df.columns else 0

        # Calculate status
        if completed_units == total_units:
            status = "complete"
        elif completed_units > 0:
            status = "incomplete"
        else:
            status = "pending"

        # Create status map
        from datetime import datetime

        status_map = {
            "workflow_status": status,
            "table_id": str(table_id),
            "completed_units": str(completed_units),
            "total_units": str(total_units),
            "progress_percent": (
                str(round(100 * completed_units / total_units, 1))
                if total_units > 0
                else "0.0"
            ),
            "last_updated": datetime.now().isoformat(),
        }

        # Remove any existing workflow status annotation (best-effort cleanup)
        # This may fail if no annotations exist, permissions are missing, or
        # connection issues occur. We intentionally ignore these failures since
        # the new annotation will be created regardless.
        try:
            existing_annotations = ezomero.get_map_annotation(
                conn, container_type.capitalize(), container_id
            )
            for ann_id, ann_data in existing_annotations.items():
                if isinstance(ann_data, dict) and ann_data.get("workflow_status"):
                    ezomero.delete_annotation(conn, ann_id)
                    break
        except (KeyError, ValueError, TypeError):
            pass  # Expected: no existing annotation or malformed data
        except Exception:
            pass  # Connection/permission issues - proceed with creating new annotation

        # Create new status map annotation
        status_ann_id = ezomero.post_map_annotation(
            conn,
            object_type=container_type.capitalize(),
            object_id=container_id,
            kv_dict=status_map,
            ns="openmicroscopy.org/omero/annotate/workflow_status",
        )

        print(
            f"Workflow status updated: {completed_units}/{total_units} ({status_map['progress_percent']}%) - {status}"
        )
        return status_ann_id

    except Exception as e:
        print(f"Could not update workflow status: {e}")
        return None


def get_workflow_status_map(
    conn, container_type: str, container_id: int
) -> Optional[Dict[str, str]]:
    """Get current workflow status from map annotation.

    Args:
        conn: OMERO connection
        container_type: Type of OMERO container
        container_id: ID of container

    Returns:
        Status map dictionary if found, None otherwise
    """
    try:
        # Get map annotations for container
        annotations = ezomero.get_map_annotation(
            conn, container_type.capitalize(), container_id
        )

        # Find workflow status annotation
        for ann_id, ann_data in annotations.items():
            if isinstance(ann_data, dict) and ann_data.get("workflow_status"):
                return ann_data

        return None

    except Exception as e:
        print(f"Could not get workflow status: {e}")
        return None


# =============================================================================
# Annotation Table Management Functions
# =============================================================================


def list_annotation_tables(
    conn, container_type: str, container_id: int
) -> List[Dict[str, Any]]:
    """Find all tables attached to a container.

    Args:
        conn: OMERO connection
        container_type: Type of container ('project', 'dataset', 'plate', 'screen')
        container_id: Container ID to search in

    Returns:
        List of dictionaries with table information, sorted newest first
    """
    from .omero_utils import list_user_tables

    all_tables = list_user_tables(
        conn, container_type=container_type, container_id=container_id
    )
    all_tables.sort(key=lambda x: x.get("created", ""), reverse=True)
    return all_tables


def generate_unique_table_name(
    conn, container_type: str, container_id: int, base_name: str = None
) -> str:
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
        container_name = (
            container.getName() if container else f"{container_type}_{container_id}"
        )
        # Clean container name for use in table name
        container_name = "".join(
            c for c in container_name if c.isalnum() or c in "_-"
        ).lower()
    except Exception as e:
        print(f"Warning: Could not get container name: {e}")
        container_name = f"{container_type}_{container_id}"

    # Create base name if not provided
    if not base_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{container_name}_{timestamp}"

    # Check if name already exists and make it unique
    existing_tables = list_annotation_tables(conn, container_type, container_id)
    existing_names = {table["name"] for table in existing_tables}

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
    try:
        # Get table data
        table_data = ezomero.get_table(conn, table_id)

        if table_data is None or table_data.empty:
            return {
                "total_units": 0,
                "completed_units": 0,
                "progress_percent": 0,
                "is_complete": False,
                "status": "empty",
                "error": "Table is empty or could not be read",
            }

        # Analyze progress based on table structure
        total_units = len(table_data)

        # Check for 'processed' column or similar completion indicators
        completed_units = 0
        completion_columns = ["processed", "completed", "finished", "done"]

        completion_column = None
        for col in completion_columns:
            if col in table_data.columns:
                completion_column = col
                break

        if completion_column:
            # Count completed units
            completed_units = (
                table_data[completion_column].sum()
                if table_data[completion_column].dtype == bool
                else (table_data[completion_column] == True).sum()
            )
        else:
            # Check for roi_id or label_id columns as completion indicators
            if "roi_id" in table_data.columns:
                completed_units = table_data["roi_id"].notna().sum()
            elif "label_id" in table_data.columns:
                completed_units = table_data["label_id"].notna().sum()

        # Calculate progress
        progress_percent = (
            (completed_units / total_units * 100) if total_units > 0 else 0
        )
        is_complete = progress_percent >= 100

        # Determine status
        if is_complete:
            status = "complete"
        elif completed_units > 0:
            status = "in_progress"
        else:
            status = "not_started"

        return {
            "total_units": int(total_units),
            "completed_units": int(completed_units),
            "progress_percent": round(progress_percent, 1),
            "is_complete": is_complete,
            "status": status,
            "table_size": len(table_data),
            "columns": list(table_data.columns),
        }

    except Exception as e:
        return {
            "total_units": 0,
            "completed_units": 0,
            "progress_percent": 0,
            "is_complete": False,
            "status": "error",
            "error": str(e),
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

    if progress["status"] == "error":
        return f"Error: {progress.get('error', 'Unknown error')}"

    total = progress["total_units"]
    completed = progress["completed_units"]
    percent = progress["progress_percent"]

    return f" {completed}/{total} units ({percent:.1f}% complete)"


def create_roi_namespace_for_table(table_name: str) -> str:
    """Create a consistent ROI namespace for a table.

    Args:
        table_name: Name of the annotation table

    Returns:
        ROI namespace string
    """
    return f"omero_annotate_ai.table.{table_name}"



def cleanup_project_annotations(
    conn, project_id: int, trainingset_name: str = ""
    ) -> Dict[str, int]:
    """Clean up all annotations created by omero-annotate-ai from a project.

    Removes annotation tables, ROIs, and map annotations from the project
    and all its underlying datasets and images.

    Args:
        conn: OMERO connection
        project_id: Project ID to clean up
        trainingset_name: Optional - if provided, only clean up annotations
                            matching this training set name. Defaults to empty string.

    Returns:
        Dictionary with counts of deleted items:
        {
            'tables': int,
            'rois': int,
            'map_annotations': int,
            'images_processed': int
        }
    """
    # Import required functions
    from .omero_utils import delete_annotations_by_namespace, delete_table

    # Initialize counters
    results = {"tables": 0, "rois": 0, "map_annotations": 0, "images_processed": 0}

    print(f"Starting cleanup of project {project_id}")
    if trainingset_name:
        print(f"Filtering by training set: {trainingset_name}")

    # Get project and all its datasets and images
    project = conn.getObject("Project", project_id)
    if not project:
        print(f"Project {project_id} not found")
        return results

    # Collect all datasets and images in the project
    all_datasets = []
    all_images = []

    for dataset in project.listChildren():
        all_datasets.append(dataset)
        for image in dataset.listChildren():
            all_images.append(image)

    print(
        f"Found {len(all_datasets)} datasets and {len(all_images)} images in project"
    )

    # 1. Clean up annotation tables
    print("Cleaning up annotation tables...")
    for dataset in all_datasets:
        try:
            tables = list_annotation_tables(conn, "dataset", dataset.getId())
            for table in tables:
                table_id = table["id"]
                table_name = table["name"]

                # Filter by training set name if specified
                if trainingset_name and trainingset_name not in table_name:
                    continue

                # Delete the table
                if delete_table(conn, table_id):
                    results["tables"] += 1
                    print(f"Deleted table: {table_name} (ID: {table_id})")
                else:
                    print(f"Failed to delete table: {table_name} (ID: {table_id})")
        except Exception as e:
            print(f"Error cleaning tables for dataset {dataset.getId()}: {str(e)}")
            raise

    # 2. Clean up ROIs by name patterns
    print("Cleaning up ROIs...")
    for image in all_images:
        try:
            image_id = image.getId()
            rois_to_delete = []

            # Get all ROIs for this image
            roi_service = conn.getRoiService()
            result = roi_service.findByImage(image_id, None)

            for roi in result.rois:
                roi_id = roi.getId().getValue()
                roi_name = roi.getName().getValue() if roi.getName() else ""
                roi_description = (
                    roi.getDescription().getValue() if roi.getDescription() else ""
                )

                # Check if ROI matches our patterns
                is_micro_sam_roi = False

                # Pattern 1: {trainingset_name}_ROIs
                if trainingset_name and f"{trainingset_name}_ROIs" in roi_name:
                    is_micro_sam_roi = True

                # Pattern 2: Generic Micro-SAM ROIs (only if no specific training set filter)
                elif not trainingset_name and "Micro-SAM ROIs" in roi_name:
                    is_micro_sam_roi = True

                # Pattern 3: Check description for Micro-SAM content
                elif not trainingset_name and "Micro-SAM" in roi_description:
                    is_micro_sam_roi = True

                if is_micro_sam_roi:
                    rois_to_delete.append(roi_id)

            # Delete the ROIs
            if rois_to_delete:
                conn.deleteObjects("Roi", rois_to_delete, wait=True)
                results["rois"] += len(rois_to_delete)
                print(f"Deleted {len(rois_to_delete)} ROIs from image {image_id}")

            results["images_processed"] += 1

        except Exception as e:
            print(f"Error cleaning ROIs for image {image.getId()}: {str(e)}")
            raise

    # 3. Clean up map annotations (workflow status)
    print("Cleaning up map annotations...")
    workflow_namespace = "openmicroscopy.org/omero/annotate/workflow_status"

    try:
        # Clean up from datasets
        for dataset in all_datasets:
            count = delete_annotations_by_namespace(
                conn, "Dataset", dataset.getId(), workflow_namespace
            )
            results["map_annotations"] += count

        # Clean up from project
        count = delete_annotations_by_namespace(
            conn, "Project", project_id, workflow_namespace
        )
        results["map_annotations"] += count

    except Exception as e:
        print(f"Error cleaning map annotations: {str(e)}")
        raise

    # Print summary
    print(f"\n Cleanup completed:")
    print(f"   Tables deleted: {results['tables']}")
    print(f"   ROIs deleted: {results['rois']}")
    print(f"   Map annotations deleted: {results['map_annotations']}")
    print(f"   Images processed: {results['images_processed']}")

    return results
