"""Image and file processing functionality."""

import importlib
from typing import TYPE_CHECKING, Any

from .trainers import (
    TRAINERS,
    Trainer,
    TrainingData,
    TrainingResult,
    load_training_data,
    train,
)
from .utils import (
    validate_table_schema,
)

# Static analysis sees the lazy names; at runtime __getattr__ resolves them.
if TYPE_CHECKING:
    from .bia_data import prepare_bia_data_from_table
    from .image_functions import (
        generate_patch_coordinates,
        label_to_rois,
        mask_to_contour,
        process_label_plane,
    )
    from .training_functions import (
        export_training_set,
        prepare_training_data_from_table,
        reorganize_local_data_for_training,
    )

# These modules import omero-py/ezomero; load them on first access so the headless CLI
# (which only needs ``trainers``) does not.
_LAZY = {
    "prepare_bia_data_from_table": "bia_data",
    "generate_patch_coordinates": "image_functions",
    "label_to_rois": "image_functions",
    "mask_to_contour": "image_functions",
    "process_label_plane": "image_functions",
    "export_training_set": "training_functions",
    "prepare_training_data_from_table": "training_functions",
    "reorganize_local_data_for_training": "training_functions",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # bia_data
    "prepare_bia_data_from_table",
    # image_functions
    "generate_patch_coordinates",
    "label_to_rois",
    "mask_to_contour",
    "process_label_plane",
    # training_functions
    "validate_table_schema",
    "prepare_training_data_from_table",
    "reorganize_local_data_for_training",
    "export_training_set",
    # trainers
    "TRAINERS",
    "Trainer",
    "TrainingData",
    "TrainingResult",
    "load_training_data",
    "train",
]
