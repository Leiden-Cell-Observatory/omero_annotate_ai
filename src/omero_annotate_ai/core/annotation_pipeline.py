"""One-call workflows over :class:`~omero_annotate_ai.trainingset.TrainingSet`.

Deprecated: kept so 0.4 notebooks keep working through 0.5. Use ``TrainingSet``
directly::

    ts = TrainingSet.create(config, conn)      # or TrainingSet.open(table_id, conn)
    ts.annotate(note="nuclei only")
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..annotators import Annotator
from ..events import EventSink
from ..sources import ImageSource
from ..trainingset import OmeroWrites, TrainingSet
from .annotation_config import AnnotationConfig


class AnnotationPipeline:
    """Runs a whole workflow on a training set: create (or open) it, then annotate.

    Opens the set ``config.identity.table_id`` names (raising its image counts to the
    config's), otherwise creates a new one.
    """

    def __init__(
        self,
        config: AnnotationConfig,
        conn: Any = None,
        *,
        annotator: Annotator | None = None,
        omero_writes: OmeroWrites | None = None,
        home: str | Path | None = None,
        events: EventSink | None = None,
        cancel: Callable[[], bool] | None = None,
        image_source: ImageSource | None = None,
    ) -> None:
        self._requested = config
        self.conn = conn
        self.annotator = annotator
        self._options: dict[str, Any] = {
            "omero_writes": omero_writes,
            "home": home,
            "events": events,
            "cancel": cancel,
            "image_source": image_source,
        }
        self._training_set: TrainingSet | None = None

    @property
    def training_set(self) -> TrainingSet:
        """The set this pipeline works on; created or opened on first use."""
        if self._training_set is None:
            config = self._requested
            table_id = config.identity.table_id
            if table_id is None:
                self._training_set = TrainingSet.create(
                    config, self.conn, **self._options
                )
            else:
                ts = TrainingSet.open(table_id, self.conn, **self._options)
                current = ts.config.training
                ts.extend(
                    **{
                        key: wanted
                        for key in ("train_n", "validate_n", "test_n")
                        if (wanted := getattr(config.training, key))
                        > getattr(current, key)
                    }
                )
                self._training_set = ts
        return self._training_set

    @property
    def config(self) -> AnnotationConfig:
        return self.training_set.config

    @property
    def table_id(self) -> int:
        """The tracking table's ID, or -1 for local sets."""
        table_id = self.training_set.table_id
        return -1 if table_id is None else table_id

    @property
    def bundle_dir(self) -> Path:
        return self.training_set.bundle_dir

    @property
    def image_source(self) -> ImageSource:
        return self.training_set.image_source

    def _result(self) -> tuple[int, AnnotationConfig]:
        return self.table_id, self.config

    def run_full_micro_sam_workflow(self) -> tuple[int, AnnotationConfig]:
        """Annotate the pending units (micro-SAM in napari unless an annotator was
        given). Returns ``(table_id, config)``; ``table_id`` is -1 for local sets."""
        requested = self._requested
        self.training_set.annotate(
            self.annotator,
            batch_size=requested.workflow.batch_size or None,
            model=requested.ai_model.pretrained_from,
        )
        return self._result()

    run_full_microsam_workflow = run_full_micro_sam_workflow

    def run_cp_workflow(self) -> tuple[int, AnnotationConfig]:
        """Save the pending units' images for Cellpose (``annotation_input/``).

        Units stay pending until :meth:`collect_annotations_from_disk` finds their
        masks.
        """
        self.training_set.prepare_external()
        return self._result()

    def collect_annotations_from_disk(
        self, folder_pattern: str = "external_output"
    ) -> tuple[int, AnnotationConfig]:
        """Save the masks an external tool wrote; returns ``(n_saved, config)``."""
        return self.training_set.collect_masks(folder_pattern), self.config


def create_pipeline(
    config: AnnotationConfig,
    conn: Any = None,
    *,
    annotator: Annotator | None = None,
    omero_writes: OmeroWrites | None = None,
    home: str | Path | None = None,
    events: EventSink | None = None,
    cancel: Callable[[], bool] | None = None,
) -> AnnotationPipeline:
    """A one-call workflow over a training set.

    Deprecated: use :class:`~omero_annotate_ai.trainingset.TrainingSet` directly.
    """
    warnings.warn(
        "create_pipeline is deprecated; use TrainingSet.create(config, conn) or "
        "TrainingSet.open(table_id, conn) and TrainingSet.annotate()",
        DeprecationWarning,
        stacklevel=2,
    )
    return AnnotationPipeline(
        config,
        conn,
        annotator=annotator,
        omero_writes=omero_writes,
        home=home,
        events=events,
        cancel=cancel,
    )
