"""
Views for OMERO Annotation Web Plugin

This module provides Django views that expose omero-annotate-ai functionality
through web interfaces, following OMERO.web patterns.
"""

import json
import logging
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)


@login_required
def annotation_interface(request):
    """Main annotation interface view"""
    context = {
        'page_title': 'OMERO Annotation Workflow',
        'app_name': 'omero_annotate_web'
    }
    return render(request, 'omero_annotate_web/annotation_interface.html', context)


@login_required
@require_http_methods(["POST"])
@csrf_exempt
def api_omero_connection(request):
    """API endpoint for OMERO connection testing"""
    try:
        # Get OMERO connection from OMERO.web session
        conn = request.session.get('connector')

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


@login_required
@require_http_methods(["GET"])
def api_containers(request, container_type):
    """API endpoint to list OMERO containers"""
    try:
        conn = request.session.get('connector')

        if not conn or not conn.isConnected():
            return JsonResponse({
                'success': False,
                'message': 'No OMERO connection'
            }, status=400)

        # List containers based on type
        containers = []
        if container_type == 'projects':
            for project in conn.getObjects("Project"):
                containers.append({
                    'id': project.getId(),
                    'name': project.getName(),
                    'description': project.getDescription() or ""
                })
        elif container_type == 'datasets':
            for dataset in conn.getObjects("Dataset"):
                containers.append({
                    'id': dataset.getId(),
                    'name': dataset.getName(),
                    'description': dataset.getDescription() or ""
                })
        # Add more container types as needed

        return JsonResponse({
            'success': True,
            'containers': containers
        })

    except Exception as e:
        logger.error(f"Container listing error: {e}")
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@login_required
@require_http_methods(["POST"])
@csrf_exempt
def api_create_pipeline(request):
    """API endpoint to create annotation pipeline"""
    try:
        data = json.loads(request.body)

        # Get OMERO connection
        conn = request.session.get('connector')
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


@login_required
@require_http_methods(["POST"])
@csrf_exempt
def api_launch_annotation(request):
    """API endpoint to launch annotation workflow"""
    try:
        data = json.loads(request.body)
        pipeline_id = data.get('pipeline_id')

        # Get OMERO connection
        conn = request.session.get('connector')
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