"""OMERO Annotate AI: Integration of AI annotation tools with OMERO for automated image segmentation."""

from .core.config import AnnotationConfig, create_default_config, load_config
from .core.pipeline import AnnotationPipeline, create_pipeline
from .widgets.omero_connection_widget import create_omero_connection_widget
from .widgets.workflow_widget import create_workflow_widget

# OMERO utilities
from .omero import omero_utils

__version__ = "0.1.0"
__author__ = "Maarten Paul"
__email__ = "m.w.paul@lumc.nl"

__all__ = [
 "AnnotationConfig",
 "load_config",
 "create_default_config",
 "create_pipeline",
 "AnnotationPipeline",
 "create_omero_connection_widget",
 "create_workflow_widget",
 "omero_utils",
]
