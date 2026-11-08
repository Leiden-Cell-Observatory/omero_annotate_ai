# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
# (conda's PyQt5 ships no type stubs, so Qt classes are unknown to pyright.)
"""Background threads for the plugin, with results delivered on the Qt main thread.

Two single-thread executors, so nothing runs concurrently that must not:

- the **IO worker** owns every call that touches the OMERO connection or the
  training-set folder (connect, list, open, next batch, commit, finish). A
  BlitzGateway must be used from one thread at a time; routing all calls through
  one thread guarantees that by construction.
- the **compute worker** loads the SAM model once and computes image embeddings,
  so napari stays responsive while an embedding is computed.

Callbacks (``on_done``/``on_error``) always run on the main thread: results travel
back through a Qt signal, which Qt queues onto the thread of the receiving object.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from qtpy.QtCore import QObject, Signal

from ..events import Event

logger = logging.getLogger(__name__)

T = TypeVar("T")
ErrorHandler = Callable[[BaseException], None]


class _Delivery(QObject):
    """Carries a finished job's callback and outcome to the main thread."""

    ready = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.ready.connect(self._call)

    def _call(self, callback: Callable[[Any], None], value: Any) -> None:
        if not self.closed:  # the widget the callback updates may be gone
            callback(value)


class Worker:
    """A single background thread whose results are handled on the main thread."""

    def __init__(self, name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._delivery = _Delivery()

    def run(
        self,
        fn: Callable[[], T],
        on_done: Callable[[T], None],
        on_error: ErrorHandler,
    ) -> Future[T]:
        """Run ``fn`` on the worker thread; ``on_done``/``on_error`` run on the main thread."""

        def job() -> T:
            try:
                value = fn()
            except BaseException as exc:
                logger.exception("Background job failed")
                self._delivery.ready.emit(on_error, exc)
                raise
            self._delivery.ready.emit(on_done, value)
            return value

        return self._executor.submit(job)

    def shutdown(self) -> None:
        """Stop: drop queued jobs, wait for the running one, ignore late results."""
        self._delivery.closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)


class EventBridge(QObject):
    """Turns core events (emitted on the IO thread) into a main-thread Qt signal."""

    event = Signal(object)

    def sink(self, event: Event) -> None:
        """The ``EventSink`` handed to the core: returns at once and never raises."""
        self.event.emit(event)


@dataclass(frozen=True, slots=True)
class Embedding:
    """A computed image embedding, ready for micro-SAM's annotator."""

    path: Path
    predictor: Any
    decoder: Any


class Embedder:
    """Computes SAM embeddings on the compute worker, loading the model once."""

    def __init__(self, worker: Worker) -> None:
        self._worker = worker
        self._model: tuple[str, Any, Any] | None = None

    def compute(
        self,
        image: np.ndarray,
        save_path: Path,
        model_type: str,
        on_done: Callable[[Embedding], None],
        on_error: ErrorHandler,
    ) -> None:
        """Compute (or load, if ``save_path`` exists) the embedding of a 2D image."""
        self._worker.run(
            lambda: self._compute(image, save_path, model_type), on_done, on_error
        )

    def _compute(
        self, image: np.ndarray, save_path: Path, model_type: str
    ) -> Embedding:
        from micro_sam import util  # heavy import, only on the compute thread

        predictor, decoder = self._load(model_type)
        util.precompute_image_embeddings(
            predictor=predictor,
            input_=image,
            save_path=str(save_path),
            ndim=2,
            verbose=False,
        )
        return Embedding(save_path, predictor, decoder)

    def _load(self, model_type: str) -> tuple[Any, Any]:
        if self._model is not None and self._model[0] == model_type:
            return self._model[1], self._model[2]
        from micro_sam import util
        from micro_sam.instance_segmentation import get_decoder

        # return_state=True returns (predictor, state), which the annotation does not say.
        loaded: Any = util.get_sam_model(model_type=model_type, return_state=True)
        predictor, state = loaded
        decoder = (
            get_decoder(
                predictor.model.image_encoder, state["decoder_state"], predictor.device
            )
            if "decoder_state" in state
            else None
        )
        self._model = (model_type, predictor, decoder)
        return predictor, decoder
