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
from .processing.trainers import (
    TRAINERS,
    Trainer,
    TrainingData,
    TrainingResult,
    load_training_data,
    train,
)
from .rows import Row

try:
    __version__ = _pkg_version("omero-annotate-ai")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "unknown"
__author__ = "Maarten Paul"
__email__ = "m.w.paul@lacdr.leidenuniv.nl"

# Static analysis sees the lazy names; at runtime __getattr__ resolves them.
if TYPE_CHECKING:
    from .core.annotation_pipeline import AnnotationPipeline, create_pipeline
    from .omero import omero_utils
    from .processing.bia_data import prepare_bia_data_from_table
    from .processing.training_functions import (
        export_training_set,
        prepare_training_data_from_table,
        reorganize_local_data_for_training,
    )
    from .trainingset import OmeroWrites, TrainingSet
    from .widgets.omero_connection_widget import create_omero_connection_widget
    from .widgets.training_data_widget import create_training_data_widget
    from .widgets.workflow_widget import create_workflow_widget

# Resolved on first access: widget factories need the `notebooks` extra (ipywidgets,
# IPython), and the OMERO-backed names import omero-py, which the headless CLI must not load.
_LAZY = {
    "AnnotationPipeline": "omero_annotate_ai.core.annotation_pipeline",
    "create_pipeline": "omero_annotate_ai.core.annotation_pipeline",
    "omero_utils": "omero_annotate_ai.omero",
    "prepare_bia_data_from_table": "omero_annotate_ai.processing.bia_data",
    "export_training_set": "omero_annotate_ai.processing.training_functions",
    "prepare_training_data_from_table": "omero_annotate_ai.processing.training_functions",
    "reorganize_local_data_for_training": "omero_annotate_ai.processing.training_functions",
    "OmeroWrites": "omero_annotate_ai.trainingset",
    "TrainingSet": "omero_annotate_ai.trainingset",
    "create_omero_connection_widget": "omero_annotate_ai.widgets.omero_connection_widget",
    "create_training_data_widget": "omero_annotate_ai.widgets.training_data_widget",
    "create_workflow_widget": "omero_annotate_ai.widgets.workflow_widget",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
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
