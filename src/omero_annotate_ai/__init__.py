"""OMERO Annotate AI: Integration of AI annotation tools with OMERO for automated image segmentation."""

from .core.annotation_config import AnnotationConfig, create_default_config, load_config
from .core.annotation_pipeline import AnnotationPipeline, create_pipeline
from .widgets.omero_connection_widget import create_omero_connection_widget
from .widgets.workflow_widget import create_workflow_widget
from .widgets.training_data_widget import create_training_data_widget

# Processing functions
from .processing.training_functions import prepare_training_data_from_table
from .processing.training_utils import setup_training, run_training

# OMERO utilities
from .omero import omero_utils

__version__ = "0.1.2"
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
    "create_training_data_widget",
    "prepare_training_data_from_table",
    "setup_training",
    "run_training",
    "omero_utils",
]
