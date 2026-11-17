"""OMERO Annotate AI: AI-assisted image annotation and segmentation with OMERO."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Any

from .core.annotation_config import (
    AnnotationConfig,
    create_default_config,
    load_config,
    load_config_from_yaml,
)
from .core.annotation_pipeline import AnnotationPipeline, create_pipeline

# OMERO utilities
from .omero import omero_utils

# Processing functions
from .processing.bia_data import prepare_bia_data_from_table
from .processing.training_functions import (
    export_training_set,
    prepare_training_data_from_table,
    reorganize_local_data_for_training,
)
from .processing.training_utils import run_training, setup_training
from .rows import Row
from .trainingset import OmeroWrites, TrainingSet

if TYPE_CHECKING:
    from .widgets.omero_connection_widget import create_omero_connection_widget
    from .widgets.training_data_widget import create_training_data_widget
    from .widgets.workflow_widget import create_workflow_widget

# The widgets need ipywidgets and IPython (the "notebooks" extra), so they are
# imported on first use; scripts and headless servers can import the package without.
_WIDGET_MODULES = {
    "create_omero_connection_widget": "omero_connection_widget",
    "create_training_data_widget": "training_data_widget",
    "create_workflow_widget": "workflow_widget",
}


def __getattr__(name: str) -> Any:
    module = _WIDGET_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(import_module(f".widgets.{module}", __name__), name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{name} needs the notebook packages: "
            "pip install 'omero-annotate-ai[notebooks]'"
        ) from exc


try:
    __version__ = _pkg_version("omero-annotate-ai")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "unknown"
__author__ = "Maarten Paul"
__email__ = "m.w.paul@lacdr.leidenuniv.nl"

__all__ = [
    "TrainingSet",
    "OmeroWrites",
    "Row",
    "AnnotationConfig",
    "load_config",
    "load_config_from_yaml",
    "create_default_config",
    "create_pipeline",
    "AnnotationPipeline",
    "create_omero_connection_widget",
    "create_workflow_widget",
    "create_training_data_widget",
    "prepare_training_data_from_table",
    "export_training_set",
    "prepare_bia_data_from_table",
    "setup_training",
    "run_training",
    "omero_utils",
    "reorganize_local_data_for_training",
]
