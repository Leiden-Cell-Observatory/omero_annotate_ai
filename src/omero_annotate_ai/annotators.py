"""Annotation tools the pipeline can drive: micro-SAM in napari, or a script.

The pipeline loads a batch of units and passes it to an :class:`Annotator`, which
writes ``seg_{i:05}.tif`` for item ``i`` into the batch's own, empty ``mask_dir`` and
returns the files by annotation ID. Units missing from the result were not annotated
(e.g. napari closed early).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import tifffile

from .bundle import collect_masks
from .errors import AnnotatorUnavailable
from .planning import Unit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchItem:
    """One unit to annotate: its pixels ``(y, x)`` or ``(z, y, x)``, context channels,
    the unit itself (image, plane, patch, category) and, when reviewing, the mask
    saved earlier (``initial_mask``) to start from."""

    annotation_id: str
    pixels: np.ndarray
    context: Mapping[str, np.ndarray] = field(default_factory=dict[str, np.ndarray])
    unit: Unit | None = None
    initial_mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Batch:
    """Units handed to an annotator in one go, plus where and how to annotate them.

    ``mask_dir`` and ``embedding_dir`` belong to this batch only; embeddings are
    named ``embedding_{i:05}.zarr`` by position, as micro-SAM writes them.
    """

    index: int
    items: Sequence[BatchItem]
    mask_dir: Path
    embedding_dir: Path | None = None
    model_type: str = ""
    is_volumetric: bool = False
    context_colormaps: Mapping[str, str] = field(default_factory=dict[str, str])


@runtime_checkable
class Annotator(Protocol):
    """An annotation tool: returns ``{annotation_id: mask file}`` once the user is done."""

    @property
    def name(self) -> str: ...

    def annotate(self, batch: Batch) -> Mapping[str, Path]: ...


def _ids(batch: Batch) -> list[str]:
    return [item.annotation_id for item in batch.items]


class MicroSamAnnotator:
    """micro-SAM's image series annotator in napari; blocks until the viewer closes.

    Items with an ``initial_mask`` open with that mask (micro-SAM's
    ``initial_segmentations``), ready to correct.

    napari and micro-SAM are imported on first use; a missing install raises
    :class:`~omero_annotate_ai.errors.AnnotatorUnavailable`.
    """

    name = "micro-sam"

    def annotate(self, batch: Batch) -> Mapping[str, Path]:
        try:
            napari = importlib.import_module("napari")
            sam_annotator = importlib.import_module("micro_sam.sam_annotator")
        except ImportError as error:
            raise AnnotatorUnavailable(
                "micro-SAM and napari are required for interactive annotation: "
                'pip install "omero-annotate-ai[microsam]"'
            ) from error

        settings = napari.settings.get_settings()
        ipy_interactive = settings.application.ipy_interactive
        # Without this napari returns at once in IPython, before any mask exists.
        settings.application.ipy_interactive = False
        settings.application.save_window_geometry = False
        try:
            images = [item.pixels for item in batch.items]
            initial = None
            if any(item.initial_mask is not None for item in batch.items):
                # A label image has no channel axis: (z,) y, x of the pixels.
                ndim = 3 if batch.is_volumetric else 2
                initial = [
                    np.zeros(item.pixels.shape[:ndim], np.uint32)
                    if item.initial_mask is None
                    else item.initial_mask
                    for item in batch.items
                ]
            viewer = sam_annotator.image_series_annotator(
                images=images,
                initial_segmentations=initial,
                output_folder=str(batch.mask_dir),
                model_type=batch.model_type,
                viewer=None,
                return_viewer=True,
                embedding_path=None
                if batch.embedding_dir is None
                else str(batch.embedding_dir),
                is_volumetric=batch.is_volumetric,
                skip_segmented=True,
            )
            install_context_channel_hook(
                viewer,
                images,
                [item.context for item in batch.items],
                batch.context_colormaps,
            )
            napari.run()
        finally:
            settings.application.ipy_interactive = ipy_interactive
        return collect_masks(batch.mask_dir, _ids(batch))


def install_context_channel_hook(
    viewer: Any,
    images: Sequence[np.ndarray],
    contexts: Sequence[Mapping[str, np.ndarray]],
    colormaps: Mapping[str, str],
) -> None:
    """Show context channels as layers that follow the image being annotated.

    The annotator swaps the data of its ``image`` layer on "next image"; this hook
    finds which image is shown and swaps each context layer (one per ``colormaps``
    entry) to match. Does nothing without context channels.
    """
    if not colormaps:
        return
    index_by_id = {id(image): i for i, image in enumerate(images)}

    def add_layer(name: str, data: np.ndarray) -> None:
        viewer.add_image(
            data, name=name, colormap=colormaps[name], blending="additive", visible=True
        )
        viewer.layers.move(len(viewer.layers) - 1, 0)

    def show(index: int) -> None:
        for name in colormaps:
            data = contexts[index].get(name)
            if data is None:
                continue
            if name not in viewer.layers:
                add_layer(name, data)
                continue
            try:
                viewer.layers[name].data = data
            except Exception:  # noqa: BLE001 - napari's error type varies; a new layer always works
                del viewer.layers[name]
                add_layer(name, data)

    def index_of(data: np.ndarray) -> int | None:
        if (index := index_by_id.get(id(data))) is not None:
            return index
        for i, image in enumerate(images):
            if (
                image.shape == data.shape
                and image.dtype == data.dtype
                and np.array_equal(image, data)
            ):
                return i
        return None

    def on_image_changed(_event: object) -> None:
        try:
            if (index := index_of(viewer.layers["image"].data)) is not None:
                show(index)
        except Exception:  # noqa: BLE001 - napari calls this while tearing the viewer down
            logger.debug("Context channel update skipped", exc_info=True)

    if (initial := index_by_id.get(id(viewer.layers["image"].data))) is not None:
        show(initial)
    viewer.layers["image"].events.data.connect(on_image_changed)


@dataclass(frozen=True, slots=True)
class ScriptedCall:
    """One :meth:`ScriptedAnnotator.annotate` call: positions written and skipped."""

    n_images: int
    written: tuple[int, ...]
    skipped: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]


class ScriptedAnnotator:
    """Draws each item's mask with ``mask_fn``, without a display (scripts, tests).

    Writes ``seg_{i:05}.tif`` like micro-SAM, skipping positions whose file exists
    when ``skip_existing``; ``limit`` stops after that many masks, as if the user
    closed napari early. An item with an ``initial_mask`` (a review) gets
    ``review_fn(pixels, initial_mask)``, or the initial mask unchanged without one.
    """

    name = "scripted"

    def __init__(
        self,
        mask_fn: Callable[[np.ndarray], np.ndarray],
        *,
        limit: int | None = None,
        skip_existing: bool = True,
        review_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.mask_fn = mask_fn
        self.review_fn = review_fn
        self.limit = limit
        self.skip_existing = skip_existing
        self.calls: list[ScriptedCall] = []

    def annotate(self, batch: Batch) -> Mapping[str, Path]:
        batch.mask_dir.mkdir(parents=True, exist_ok=True)
        written: list[int] = []
        skipped: list[int] = []
        for position, item in enumerate(batch.items):
            path = batch.mask_dir / f"seg_{position:05}.tif"
            if self.skip_existing and path.exists():
                skipped.append(position)
            elif self.limit is not None and len(written) >= self.limit:
                break
            else:
                tifffile.imwrite(path, self._mask(item))
                written.append(position)
        shapes = tuple(tuple(item.pixels.shape) for item in batch.items)
        self.calls.append(
            ScriptedCall(len(batch.items), tuple(written), tuple(skipped), shapes)
        )
        return collect_masks(batch.mask_dir, _ids(batch))

    def _mask(self, item: BatchItem) -> np.ndarray:
        if item.initial_mask is None:
            return self.mask_fn(item.pixels)
        if self.review_fn is None:
            return item.initial_mask
        return self.review_fn(item.pixels, item.initial_mask)
