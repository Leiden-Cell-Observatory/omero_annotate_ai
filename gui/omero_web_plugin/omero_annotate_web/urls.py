"""
URL configuration for OMERO Annotation Web Plugin
"""

from django.urls import path
from . import views

app_name = 'omero_annotate_web'

urlpatterns = [
    # Main workflow interface
    path('', views.workflow_view, name='workflow'),

    # Center panel integration view
    path('index.html', views.index_view, name='index'),

    # API endpoints
    path('api/containers/<str:container_type>/', views.api_containers, name='api_containers'),
    path('api/container/<str:container_type>/<int:container_id>/info/', views.api_container_info, name='api_container_info'),
    path('api/tables/scan/', views.api_tables_scan, name='api_tables_scan'),
    path('api/tables/generate-name/', views.api_tables_generate_name, name='api_tables_generate_name'),
    path('api/config/validate/', views.api_config_validate, name='api_config_validate'),
    path('api/config/preview/', views.api_config_preview, name='api_config_preview'),
    path('api/config/save/', views.api_config_save, name='api_config_save'),

    # Legacy endpoints (keep for compatibility)
    path('api/omero/connection/', views.api_omero_connection, name='api_omero_connection'),
    path('api/pipeline/create/', views.api_create_pipeline, name='api_create_pipeline'),
    path('api/annotation/launch/', views.api_launch_annotation, name='api_launch_annotation'),
]