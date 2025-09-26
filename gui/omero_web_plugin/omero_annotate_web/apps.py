"""
Django app configuration for OMERO Annotation Web Plugin
"""

from django.apps import AppConfig


class OmeroAnnotateWebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'omero_annotate_web'
    verbose_name = 'OMERO Annotation Workflow'

    def ready(self):
        """Initialize the app when Django starts"""
        pass