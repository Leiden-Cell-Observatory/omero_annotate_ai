"""
URL configuration for OMERO Annotation Web Plugin
"""

from django.urls import path
from . import views

app_name = 'omero_annotate_web'

urlpatterns = [
    # Main interface
    path('', views.annotation_interface, name='annotation_interface'),

    # API endpoints
    path('api/omero/connection/', views.api_omero_connection, name='api_omero_connection'),
    path('api/omero/containers/<str:container_type>/', views.api_containers, name='api_containers'),
    path('api/pipeline/create/', views.api_create_pipeline, name='api_create_pipeline'),
    path('api/annotation/launch/', views.api_launch_annotation, name='api_launch_annotation'),
]