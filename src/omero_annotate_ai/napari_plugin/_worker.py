"""Background worker for running the annotation pipeline without blocking the UI.

Threading model
---------------
Qt widgets (napari layers, GL contexts) can only be created/used on the main
thread.  `image_series_annotator` internally creates napari layers, so it must
run on the main thread.

We split the workflow into two parts:

1. **Background thread** (QRunnable) — everything that only touches OMERO and
   the filesystem: initialize_workflow, define_annotation_schema,
   create_tracking_table.  Emits ``ready_to_annotate`` when done.

2. **Main thread** (slot connected to ``ready_to_annotate``) — calls
   run_microsam_annotation, which calls image_series_annotator with the
   existing viewer.

The widget is responsible for connecting ``ready_to_annotate`` to a main-thread
slot before starting the worker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, QRunnable, Signal

from omero_annotate_ai import create_pipeline

if TYPE_CHECKING:
    import napari
    from omero.gateway import BlitzGateway

    from omero_annotate_ai import AnnotationConfig


class _Signals(QObject):
    """Qt signals for AnnotationWorker (must live on QObject)."""

    # Emitted after the OMERO/setup stages complete; carries (pipeline, table_id)
    # so the main thread can proceed with the napari annotation stage.
    ready_to_annotate = Signal(object, int)   # (pipeline, table_id)
    finished = Signal(int)                    # emits final table_id
    error = Signal(str)
    log = Signal(str)


class AnnotationWorker(QRunnable):
    """Run the OMERO setup stages in a background thread.

    After ``ready_to_annotate`` fires the widget must call
    ``run_annotation_on_main_thread(pipeline)`` from the main thread.
    """

    def __init__(
        self,
        config: "AnnotationConfig",
        conn: "BlitzGateway",
    ) -> None:
        super().__init__()
        self.config = config
        self.conn = conn
        self.signals = _Signals()

    def run(self) -> None:
        try:
            self.signals.log.emit("Creating pipeline…")
            pipeline = create_pipeline(self.config, self.conn)

            self.signals.log.emit("Initializing workflow (loading images from OMERO)…")
            table_id, _images_list = pipeline.initialize_workflow()

            self.signals.log.emit("Defining annotation schema…")
            pipeline.define_annotation_schema(_images_list)

            self.signals.log.emit("Creating OMERO tracking table…")
            table_id = pipeline.create_tracking_table()

            self.signals.log.emit("Setup complete. Launching annotation UI on main thread…")
            # Hand off to the main thread — carry the pipeline object via signal
            self.signals.ready_to_annotate.emit(pipeline, table_id if table_id else 0)

        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))
