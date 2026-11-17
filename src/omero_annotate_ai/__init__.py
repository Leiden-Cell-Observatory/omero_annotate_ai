"""OMERO Annotate AI: AI-assisted image annotation and segmentation with OMERO."""

import importlib
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
from .processing.trainers import (
    TRAINERS,
    Trainer,
    TrainingData,
    TrainingResult,
    load_training_data,
    train,
)
from .processing.training_functions import (
    export_training_set,
    prepare_training_data_from_table,
    reorganize_local_data_for_training,
)
from .rows import Row
from .trainingset import OmeroWrites, TrainingSet

try:
    __version__ = _pkg_version("omero-annotate-ai")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "unknown"
__author__ = "Maarten Paul"
__email__ = "m.w.paul@lacdr.leidenuniv.nl"

# Static analysis sees the widget names; at runtime __getattr__ resolves them.
if TYPE_CHECKING:
    from .widgets.omero_connection_widget import create_omero_connection_widget
    from .widgets.training_data_widget import create_training_data_widget
    from .widgets.workflow_widget import create_workflow_widget

# Widget factories need the `notebooks` extra (ipywidgets, IPython); resolve them on first access.
_LAZY_WIDGETS = {
    "create_omero_connection_widget": "omero_annotate_ai.widgets.omero_connection_widget",
    "create_training_data_widget": "omero_annotate_ai.widgets.training_data_widget",
    "create_workflow_widget": "omero_annotate_ai.widgets.workflow_widget",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_WIDGETS:
        return getattr(importlib.import_module(_LAZY_WIDGETS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "train",
    "load_training_data",
    "TrainingData",
    "TrainingResult",
    "Trainer",
    "TRAINERS",
    "omero_utils",
    "reorganize_local_data_for_training",
]
