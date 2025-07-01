"""OMERO Annotate AI: Integration of AI annotation tools with OMERO for automated image segmentation."""

from .core.config import AnnotationConfig, load_config, create_default_config
from .core.pipeline import create_pipeline, AnnotationPipeline
from .widgets.config_widget import create_config_widget
from .widgets.omero_connection_widget import create_omero_connection_widget

# OMERO utilities (optional import)
try:
    from .omero import omero_utils
    OMERO_UTILS_AVAILABLE = True
except ImportError:
    OMERO_UTILS_AVAILABLE = False

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "AnnotationConfig",
    "load_config", 
    "create_default_config",
    "create_pipeline",
    "AnnotationPipeline",
    "create_config_widget",
    "create_omero_connection_widget",
]

# Add OMERO utils to exports if available
if OMERO_UTILS_AVAILABLE:
    __all__.append("omero_utils")