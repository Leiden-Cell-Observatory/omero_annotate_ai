"""
Views for OMERO Annotation Web Plugin

This module provides Django views that expose omero-annotate-ai functionality
through web interfaces, following OMERO.web patterns.
"""

import json
import logging
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
from omeroweb.decorators import login_required

logger = logging.getLogger(__name__)


@login_required()
@ensure_csrf_cookie
def workflow_view(request, conn=None, **kwargs):
    """Main workflow interface view"""
    context = {
        'page_title': 'AI Annotation Workflow Configuration',
        'app_name': 'omero_annotate_web'
    }

    # Add user info to context if connected
    if conn and conn.isConnected():
        user = conn.getUser()
        context['omero_user'] = user.getOmeName()

    return render(request, 'omero_annotate_web/workflow.html', context)


@login_required()
@ensure_csrf_cookie
def index_view(request, conn=None, **kwargs):
    """Center panel integration view (for omero.web.ui.center_plugins)"""
    context = {
        'page_title': 'AI Annotation Workflow Configuration',
        'app_name': 'omero_annotate_web'
    }

    # Add user info to context if connected
    if conn and conn.isConnected():
        user = conn.getUser()
        context['omero_user'] = user.getOmeName()

    return render(request, 'omero_annotate_web/index.html', context)


@login_required()
@require_http_methods(["POST", "GET"])
@csrf_exempt
def api_omero_connection(request, conn=None, **kwargs):
    """API endpoint for OMERO connection testing"""
    try:
        if conn and conn.isConnected():
            user = conn.getUser()
            return JsonResponse({
                'success': True,
                'user': user.getOmeName(),
                'message': 'Connected to OMERO'
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'No OMERO connection available'
            }, status=400)

    except Exception as e:
        logger.error(f"OMERO connection error: {e}")
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["GET"])
def api_containers(request, container_type, conn=None, **kwargs):
    """API endpoint to list OMERO containers"""
    try:
        logger.info(f"API request for container_type: {container_type}")
        logger.info(f"Connection status: {conn is not None and conn.isConnected()}")

        if not conn or not conn.isConnected():
            logger.error("No OMERO connection available")
            return JsonResponse({
                'success': False,
                'message': 'No OMERO connection'
            }, status=400)

        # List containers based on type
        containers = []
        if container_type == 'project':
            for project in conn.getObjects("Project"):
                containers.append({
                    'id': project.getId(),
                    'name': project.getName(),
                    'description': project.getDescription() or ""
                })
        elif container_type == 'dataset':
            for dataset in conn.getObjects("Dataset"):
                containers.append({
                    'id': dataset.getId(),
                    'name': dataset.getName(),
                    'description': dataset.getDescription() or ""
                })
        elif container_type == 'plate':
            for plate in conn.getObjects("Plate"):
                containers.append({
                    'id': plate.getId(),
                    'name': plate.getName(),
                    'description': plate.getDescription() or ""
                })
        elif container_type == 'screen':
            for screen in conn.getObjects("Screen"):
                containers.append({
                    'id': screen.getId(),
                    'name': screen.getName(),
                    'description': screen.getDescription() or ""
                })

        logger.info(f"Found {len(containers)} containers of type {container_type}")
        return JsonResponse({
            'success': True,
            'containers': containers
        })

    except Exception as e:
        logger.error(f"Container listing error: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["POST"])
@csrf_exempt
def api_create_pipeline(request, conn=None, **kwargs):
    """API endpoint to create annotation pipeline"""
    try:
        data = json.loads(request.body)

        if not conn or not conn.isConnected():
            return JsonResponse({
                'success': False,
                'message': 'No OMERO connection'
            }, status=400)

        # Here you would integrate your actual omero-annotate-ai code:
        # from omero_annotate_ai import create_pipeline
        # from omero_annotate_ai.core.annotation_config import AnnotationConfig

        # config = AnnotationConfig.from_dict(data)
        # pipeline = create_pipeline(config, conn)

        # For now, return success with configuration
        return JsonResponse({
            'success': True,
            'pipeline_id': 'placeholder_pipeline_id',
            'message': 'Pipeline created successfully'
        })

    except Exception as e:
        logger.error(f"Pipeline creation error: {e}")
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["POST"])
@csrf_exempt
def api_launch_annotation(request, conn=None, **kwargs):
    """API endpoint to launch annotation workflow"""
    try:
        data = json.loads(request.body)
        pipeline_id = data.get('pipeline_id')

        if not conn or not conn.isConnected():
            return JsonResponse({
                'success': False,
                'message': 'No OMERO connection'
            }, status=400)

        # Here you would:
        # 1. Launch local napari for annotation, or
        # 2. Submit job to BIOMERO for cluster processing

        # For prototype, return success
        return JsonResponse({
            'success': True,
            'message': 'Annotation workflow launched',
            'launch_type': 'local_napari'  # or 'biomero_cluster'
        })

    except Exception as e:
        logger.error(f"Annotation launch error: {e}")
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["GET"])
def api_container_info(request, container_type, container_id, conn=None, **kwargs):
    """API endpoint to get detailed container information"""
    try:
        from .utils import get_container_metadata

        if not conn or not conn.isConnected():
            return JsonResponse({
                'success': False,
                'message': 'No OMERO connection'
            }, status=400)

        metadata = get_container_metadata(conn, container_type, container_id)

        if metadata:
            return JsonResponse({
                'success': True,
                'metadata': metadata
            })
        else:
            return JsonResponse({
                'success': False,
                'message': f'Container {container_type} {container_id} not found'
            }, status=404)

    except Exception as e:
        logger.error(f"Container info error: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["POST"])
@csrf_exempt
def api_tables_scan(request, conn=None, **kwargs):
    """API endpoint to scan for existing annotation tables"""
    try:
        data = json.loads(request.body)
        container_type = data.get('container_type')
        container_id = data.get('container_id')

        if not conn or not conn.isConnected():
            return JsonResponse({
                'success': False,
                'message': 'No OMERO connection'
            }, status=400)

        # Scan for tables attached to the container
        tables = []
        container = conn.getObject(container_type.capitalize(), container_id)

        if container:
            for ann in container.listAnnotations():
                if hasattr(ann, 'getFile') and ann.getFile():
                    # Check if it's a table file
                    file_name = ann.getFile().getName()
                    if 'table' in file_name.lower() or file_name.endswith('.h5'):
                        tables.append({
                            'id': ann.getId(),
                            'name': file_name,
                            'description': ann.getDescription() or ''
                        })

        return JsonResponse({
            'success': True,
            'tables': tables
        })

    except Exception as e:
        logger.error(f"Table scan error: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["POST"])
@csrf_exempt
def api_tables_generate_name(request, conn=None, **kwargs):
    """API endpoint to generate unique table name"""
    try:
        data = json.loads(request.body)
        base_name = data.get('base_name', 'annotation_table')

        # Generate timestamp-based unique name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_name = f"{base_name}_{timestamp}"

        return JsonResponse({
            'success': True,
            'table_name': table_name
        })

    except Exception as e:
        logger.error(f"Table name generation error: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["POST"])
@csrf_exempt
def api_config_validate(request, conn=None, **kwargs):
    """API endpoint to validate configuration"""
    try:
        data = json.loads(request.body)

        # Basic validation
        errors = []
        warnings = []

        # Check required fields
        if not data.get('container_type'):
            errors.append('Container type is required')
        if not data.get('container_id'):
            errors.append('Container ID is required')
        if not data.get('table_name'):
            warnings.append('No table name specified')

        # Validate image selection if not segment_all
        if not data.get('segment_all', False):
            train_n = data.get('train_n', 0)
            validate_n = data.get('validate_n', 0)
            if train_n < 1:
                errors.append('At least 1 training image is required')
            if validate_n < 1:
                warnings.append('No validation images specified')

        return JsonResponse({
            'success': len(errors) == 0,
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        })

    except Exception as e:
        logger.error(f"Config validation error: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["POST"])
@csrf_exempt
def api_config_preview(request, conn=None, **kwargs):
    """API endpoint to preview YAML configuration"""
    try:
        from .utils import create_config_from_form, config_to_yaml

        data = json.loads(request.body)

        # Convert form data to config
        config = create_config_from_form(data)

        # Convert to YAML
        yaml_content = config_to_yaml(config)

        return JsonResponse({
            'success': True,
            'yaml': yaml_content
        })

    except Exception as e:
        logger.error(f"Config preview error: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["POST"])
@csrf_exempt
def api_config_save(request, conn=None, **kwargs):
    """API endpoint to save configuration to OMERO"""
    try:
        from .utils import create_config_from_form, config_to_yaml, attach_yaml_to_container

        data = json.loads(request.body)

        if not conn or not conn.isConnected():
            return JsonResponse({
                'success': False,
                'message': 'No OMERO connection'
            }, status=400)

        # Convert form data to config
        config = create_config_from_form(data)

        # Convert to YAML
        yaml_content = config_to_yaml(config)

        # Attach to container
        container_type = data.get('container_type')
        container_id = data.get('container_id')

        file_ann_id = attach_yaml_to_container(
            conn,
            container_type,
            container_id,
            yaml_content,
            description="AI Annotation Workflow Configuration"
        )

        if file_ann_id:
            return JsonResponse({
                'success': True,
                'file_annotation_id': file_ann_id,
                'message': 'Configuration saved successfully'
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Failed to save configuration'
            }, status=500)

    except Exception as e:
        logger.error(f"Config save error: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)