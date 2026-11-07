"""
Utility functions for OMERO Annotation Web Plugin
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import yaml
from omero.gateway import BlitzGateway

logger = logging.getLogger(__name__)


def get_omero_connection(request) -> Optional[BlitzGateway]:
    """
    Extract OMERO connection from Django request session.

    Args:
        request: Django request object

    Returns:
        BlitzGateway connection or None
    """
    try:
        conn = request.session.get('connector')
        if conn and hasattr(conn, 'c') and conn.c and conn.c.isConnected():
            return conn.c
        return None
    except Exception as e:
        logger.error(f"Error getting OMERO connection: {e}")
        return None


def create_config_from_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert form data to AnnotationConfig-compatible dictionary.

    Args:
        form_data: Dictionary from frontend form

    Returns:
        Configuration dictionary
    """
    config = {}

    # Container information
    config['container_type'] = form_data.get('container_type', 'dataset')
    config['container_id'] = int(form_data.get('container_id', 0))

    # Table configuration
    config['table_name'] = form_data.get('table_name', '')
    config['resume_from_table'] = form_data.get('resume_from_table', False)

    # Annotation settings
    config['segment_all'] = form_data.get('segment_all', False)

    if not config['segment_all']:
        config['train_n'] = int(form_data.get('train_n', 5))
        config['validate_n'] = int(form_data.get('validate_n', 2))

    # Channel configuration
    config['channels'] = parse_int_list(form_data.get('channels', '0'))

    # Timepoint configuration
    timepoint_mode = form_data.get('timepoint_mode', 'all')
    if timepoint_mode == 'all':
        config['timepoints'] = None
    elif timepoint_mode == 'first':
        config['timepoints'] = [0]
    else:  # specific
        config['timepoints'] = parse_int_list(form_data.get('timepoints', '0'))

    # Z-slice configuration
    z_mode = form_data.get('z_mode', 'all')
    if z_mode == 'all':
        config['z_slices'] = None
    elif z_mode == 'max_projection':
        config['z_slices'] = 'max_projection'
    else:  # specific
        config['z_slices'] = parse_int_list(form_data.get('z_slices', '0'))

    # 3D processing
    config['process_3d'] = form_data.get('process_3d', False)
    if config['process_3d']:
        config['z_range'] = [
            int(form_data.get('z_range_start', 0)),
            int(form_data.get('z_range_end', 10))
        ]

    # Patch configuration
    config['use_patches'] = form_data.get('use_patches', False)
    if config['use_patches']:
        config['patch_size'] = int(form_data.get('patch_size', 512))
        config['patch_overlap'] = int(form_data.get('patch_overlap', 50))

    # Technical settings
    config['batch_size'] = int(form_data.get('batch_size', 1))
    config['sam_model'] = form_data.get('sam_model', 'vit_b')

    # Workflow settings
    config['read_only'] = form_data.get('read_only', False)

    return config


def parse_int_list(value: str) -> List[int]:
    """
    Parse comma-separated string into list of integers.

    Args:
        value: Comma-separated string (e.g., "0,1,2")

    Returns:
        List of integers
    """
    if not value or value.strip() == '':
        return [0]

    try:
        return [int(x.strip()) for x in value.split(',') if x.strip()]
    except ValueError:
        logger.warning(f"Invalid integer list: {value}, defaulting to [0]")
        return [0]


def config_to_yaml(config: Dict[str, Any]) -> str:
    """
    Convert configuration dictionary to YAML string.

    Args:
        config: Configuration dictionary

    Returns:
        YAML string
    """
    # Create a clean config for YAML (remove None values)
    yaml_config = {k: v for k, v in config.items() if v is not None}

    return yaml.dump(yaml_config, default_flow_style=False, sort_keys=False)


def attach_yaml_to_container(
    conn: BlitzGateway,
    container_type: str,
    container_id: int,
    yaml_content: str,
    description: str = "AI Annotation Workflow Configuration"
) -> Optional[int]:
    """
    Attach YAML configuration as FileAnnotation to OMERO container.

    Args:
        conn: OMERO connection
        container_type: Type of container (project/dataset/plate/screen)
        container_id: OMERO ID of container
        yaml_content: YAML configuration string
        description: Description for the file annotation

    Returns:
        FileAnnotation ID or None on failure
    """
    try:
        import tempfile
        import os

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"annotation_config_{timestamp}.yaml"

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write(yaml_content)
            temp_path = temp_file.name

        try:
            # Upload file to OMERO
            file_ann = conn.createFileAnnfromLocalFile(
                temp_path,
                mimetype='text/yaml',
                ns="omero.annotate_ai.config.v1",
                desc=description
            )

            # Link to container
            container_type_map = {
                'project': 'Project',
                'dataset': 'Dataset',
                'plate': 'Plate',
                'screen': 'Screen'
            }

            omero_type = container_type_map.get(container_type.lower())
            if not omero_type:
                logger.error(f"Unknown container type: {container_type}")
                return None

            container = conn.getObject(omero_type, container_id)
            if not container:
                logger.error(f"Container not found: {omero_type} {container_id}")
                return None

            container.linkAnnotation(file_ann)

            file_ann_id = file_ann.getId()
            logger.info(f"Attached config {file_ann_id} to {omero_type} {container_id}")
            return file_ann_id

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Error attaching YAML to container: {e}", exc_info=True)
        return None


def get_container_metadata(conn: BlitzGateway, container_type: str, container_id: int) -> Optional[Dict[str, Any]]:
    """
    Get detailed metadata about a container.

    Args:
        conn: OMERO connection
        container_type: Type of container
        container_id: Container ID

    Returns:
        Dictionary with container metadata or None
    """
    try:
        # Try to use existing omero_annotate_ai function if available
        try:
            from omero_annotate_ai.omero.omero_utils import get_container_info
            return get_container_info(conn, container_type, container_id)
        except ImportError:
            # Fallback to basic implementation
            pass

        container_type_map = {
            'project': 'Project',
            'dataset': 'Dataset',
            'plate': 'Plate',
            'screen': 'Screen'
        }

        omero_type = container_type_map.get(container_type.lower())
        if not omero_type:
            return None

        container = conn.getObject(omero_type, container_id)
        if not container:
            return None

        metadata = {
            'id': container.getId(),
            'name': container.getName(),
            'description': container.getDescription() or '',
            'type': container_type
        }

        # Count images
        image_count = 0
        if omero_type == 'Project':
            for dataset in container.listChildren():
                image_count += dataset.countChildren()
        elif omero_type == 'Dataset':
            image_count = container.countChildren()
        elif omero_type in ['Plate', 'Screen']:
            # For plates/screens, count images differently
            for well in container.listChildren():
                image_count += well.countWellSample()

        metadata['image_count'] = image_count

        # Get first image for dimension info
        first_image = None
        if omero_type == 'Project':
            for dataset in container.listChildren():
                for img in dataset.listChildren():
                    first_image = img
                    break
                if first_image:
                    break
        elif omero_type == 'Dataset':
            for img in container.listChildren():
                first_image = img
                break

        if first_image:
            metadata['size_x'] = first_image.getSizeX()
            metadata['size_y'] = first_image.getSizeY()
            metadata['size_z'] = first_image.getSizeZ()
            metadata['size_c'] = first_image.getSizeC()
            metadata['size_t'] = first_image.getSizeT()

            # Pixel sizes
            pixels = first_image.getPrimaryPixels()
            if pixels:
                metadata['pixel_size_x'] = pixels.getPhysicalSizeX()
                metadata['pixel_size_y'] = pixels.getPhysicalSizeY()
                metadata['pixel_size_z'] = pixels.getPhysicalSizeZ()

        return metadata

    except Exception as e:
        logger.error(f"Error getting container metadata: {e}")
        return None
