"""Drives micro-SAM's 2D annotator inside the user's own napari viewer.

micro-SAM's ``image_series_annotator`` does not fit a plugin: it computes embeddings
on the GUI thread, adds another dock per call and closes the viewer after the last
image. This controller reuses the same building blocks once per session and
switches images itself, the way ``image_series_annotator`` does internally.

It depends on micro-SAM internals (``AnnotatorState``, ``Annotator2d(reset_state=)``,
``Annotator._update_image``, the layer names ``image`` and ``committed_objects``).
``tests/test_napari_plugin.py::TestMicroSamContract`` pins them, so a micro-SAM
release that changes them fails loudly there instead of in a user's session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._workers import Embedding

if TYPE_CHECKING:
    import napari

IMAGE_LAYER = "image"
MASK_LAYER = "committed_objects"


class LayerConflict(RuntimeError):
    """The viewer already holds a layer with a name micro-SAM needs."""


class MicroSamController:
    """Shows one image at a time in micro-SAM's annotator and reads back the mask."""

    def __init__(self, viewer: napari.Viewer) -> None:
        self._viewer = viewer
        self._annotator: Any = None
        self._dock: Any = None
        self._before: set[str] = set()  # the user's layers, left alone at teardown

    def show(
        self,
        pixels: np.ndarray,
        embedding: Embedding,
        model_type: str,
        mask: np.ndarray | None = None,
    ) -> None:
        """Display ``pixels`` using a precomputed embedding.

        The committed objects start empty, or from ``mask`` (a saved mask to review).
        """
        from micro_sam.sam_annotator._state import AnnotatorState
        from micro_sam.sam_annotator.annotator_2d import Annotator2d

        state = AnnotatorState()
        if self._annotator is None:
            if IMAGE_LAYER in self._viewer.layers or MASK_LAYER in self._viewer.layers:
                raise LayerConflict(
                    f"Rename or remove the layers '{IMAGE_LAYER}' and '{MASK_LAYER}' first: "
                    "micro-SAM uses these names."
                )
            self._before = {layer.name for layer in self._viewer.layers}
            self._viewer.add_image(pixels, name=IMAGE_LAYER)
        else:
            self._viewer.layers[IMAGE_LAYER].data = pixels
            if state.amg is not None:
                state.amg.clear_state()

        # The embedding is on disk already, so this only loads it.
        state.initialize_predictor(
            pixels,
            model_type=model_type,
            ndim=2,
            save_path=str(embedding.path),
            predictor=embedding.predictor,
            decoder=embedding.decoder,
            skip_load=False,
            use_cli=True,
        )
        state.image_shape = pixels.shape[:2]

        if self._annotator is None:
            self._annotator = Annotator2d(self._viewer, reset_state=False)
            self._dock = self._viewer.window.add_dock_widget(
                self._annotator, name="micro-SAM", area="right"
            )
        self._annotator._update_image(segmentation_result=mask)

    def current_mask(self) -> np.ndarray:
        """The objects the user has committed for the current image."""
        return np.asarray(self._viewer.layers[MASK_LAYER].data).copy()

    def teardown(self) -> None:
        """Remove micro-SAM's dock and the layers this session added.

        The masks are saved by then; the user's own layers stay. A next session in
        the same viewer starts clean.
        """
        if self._dock is not None:
            self._viewer.window.remove_dock_widget(self._dock)
        for layer in list(self._viewer.layers):
            if layer.name not in self._before:
                self._viewer.layers.remove(layer)
        self._annotator = self._dock = None
