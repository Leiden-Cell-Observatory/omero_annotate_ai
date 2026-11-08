"""Annotation tools behind the Annotator protocol (annotators.py)."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import tifffile

from omero_annotate_ai.annotators import (
    Annotator,
    Batch,
    BatchItem,
    MicroSamAnnotator,
    ScriptedAnnotator,
    install_context_channel_hook,
)


def _batch(tmp_path: Path, n: int = 3) -> Batch:
    items = tuple(
        BatchItem(annotation_id=f"a{i}", pixels=np.full((8, 10), i, dtype=np.uint16))
        for i in range(n)
    )
    return Batch(index=1, items=items, mask_dir=tmp_path / "masks")


def _label(image: np.ndarray) -> np.ndarray:
    return (image + 1).astype(np.uint16)


@pytest.mark.unit
class TestAnnotatorProtocol:
    def test_annotators_satisfy_the_protocol(self):
        assert isinstance(ScriptedAnnotator(_label), Annotator)
        assert isinstance(MicroSamAnnotator(), Annotator)


@pytest.mark.unit
class TestScriptedAnnotator:
    def test_writes_one_mask_per_item(self, tmp_path):
        annotator = ScriptedAnnotator(_label)
        masks = annotator.annotate(_batch(tmp_path))

        assert sorted(masks) == ["a0", "a1", "a2"]
        for i in range(3):
            mask = tifffile.imread(masks[f"a{i}"])
            assert mask.shape == (8, 10) and (mask == i + 1).all()
        call = annotator.calls[0]
        assert (call.n_images, call.written, call.skipped) == (3, (0, 1, 2), ())

    def test_limit_stops_early(self, tmp_path):
        annotator = ScriptedAnnotator(_label, limit=1)
        masks = annotator.annotate(_batch(tmp_path))
        assert list(masks) == ["a0"]
        assert annotator.calls[0].written == (0,)

    def test_skips_existing_files_like_micro_sam(self, tmp_path):
        batch = _batch(tmp_path)
        batch.mask_dir.mkdir()
        tifffile.imwrite(batch.mask_dir / "seg_00000.tif", np.zeros((8, 10)))
        annotator = ScriptedAnnotator(_label)
        annotator.annotate(batch)
        assert annotator.calls[0].skipped == (0,)
        assert annotator.calls[0].written == (1, 2)


def _viewer(image_layer):
    layers = MagicMock()
    viewer = Mock()
    viewer.layers = layers
    return viewer, layers, image_layer


@pytest.mark.unit
class TestContextChannelHook:
    def test_no_op_without_context_channels(self):
        viewer = Mock()
        install_context_channel_hook(viewer, [np.zeros((64, 64))], [{}], {})
        viewer.add_image.assert_not_called()

    def test_adds_layer_for_initial_image(self):
        label = np.zeros((64, 64), dtype=np.uint16)
        dapi = np.ones((64, 64), dtype=np.uint16) * 50
        image_layer = Mock()
        image_layer.data = label
        viewer, layers, _ = _viewer(image_layer)
        layers.__getitem__ = Mock(return_value=image_layer)
        layers.__contains__ = Mock(return_value=False)

        install_context_channel_hook(
            viewer, [label], [{"DAPI": dapi}], {"DAPI": "blue"}
        )

        viewer.add_image.assert_called_once_with(
            dapi, name="DAPI", colormap="blue", blending="additive", visible=True
        )

    def test_follows_the_annotator_to_the_next_image(self):
        labels = [
            np.zeros((64, 64), dtype=np.uint16),
            np.ones((64, 64), dtype=np.uint16),
        ]
        dapi = [np.full((64, 64), 10, np.uint16), np.full((64, 64), 20, np.uint16)]
        dapi_layer = Mock()
        image_layer = Mock()
        image_layer.data = labels[0]
        callbacks = []

        class Events:
            def connect(self, callback):
                callbacks.append(callback)

        image_layer.events = Mock()
        image_layer.events.data = Events()

        def getitem(name):
            return {"image": image_layer, "DAPI": dapi_layer}[name]

        viewer, layers, _ = _viewer(image_layer)
        layers.__getitem__ = Mock(side_effect=getitem)
        layers.__contains__ = Mock(side_effect=lambda name: name == "DAPI")

        install_context_channel_hook(
            viewer, labels, [{"DAPI": d} for d in dapi], {"DAPI": "blue"}
        )
        image_layer.data = labels[1]
        assert len(callbacks) == 1
        callbacks[0](Mock())

        assert dapi_layer.data is dapi[1]


@pytest.mark.unit
class TestMicroSamAnnotator:
    def test_missing_micro_sam_raises_annotator_unavailable(
        self, tmp_path, monkeypatch
    ):
        import importlib

        from omero_annotate_ai.errors import AnnotatorUnavailable

        real_import_module = importlib.import_module

        def import_module(name, *args, **kwargs):
            if name == "napari" or name.startswith("micro_sam"):
                raise ImportError(name)
            return real_import_module(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", import_module)
        with pytest.raises(AnnotatorUnavailable, match="micro-SAM"):
            MicroSamAnnotator().annotate(_batch(tmp_path))

    @staticmethod
    def _fake_micro_sam(monkeypatch) -> Mock:
        import importlib

        sam = Mock()
        modules = {"napari": MagicMock(), "micro_sam.sam_annotator": sam}
        monkeypatch.setattr(importlib, "import_module", lambda name: modules[name])
        return sam

    def test_initial_masks_go_to_micro_sam_as_initial_segmentations(
        self, tmp_path, monkeypatch
    ):
        sam = self._fake_micro_sam(monkeypatch)
        saved = np.ones((8, 10), np.uint16)
        items = (
            BatchItem("a0", np.zeros((8, 10), np.uint16), initial_mask=saved),
            BatchItem("a1", np.zeros((8, 10), np.uint16)),
        )

        MicroSamAnnotator().annotate(Batch(1, items, tmp_path / "masks"))

        initial = sam.image_series_annotator.call_args.kwargs["initial_segmentations"]
        assert initial[0] is saved
        assert initial[1].shape == (8, 10) and not initial[1].any()

    def test_placeholder_mask_has_no_channel_axis(self, tmp_path, monkeypatch):
        sam = self._fake_micro_sam(monkeypatch)
        items = (
            BatchItem("a0", np.zeros((8, 10, 3), np.uint8), initial_mask=None),
            BatchItem(
                "a1", np.zeros((8, 10, 3), np.uint8), initial_mask=np.ones((8, 10))
            ),
        )

        MicroSamAnnotator().annotate(Batch(1, items, tmp_path / "masks"))

        initial = sam.image_series_annotator.call_args.kwargs["initial_segmentations"]
        assert initial[0].shape == (8, 10)

    def test_no_initial_segmentations_when_annotating(self, tmp_path, monkeypatch):
        sam = self._fake_micro_sam(monkeypatch)

        MicroSamAnnotator().annotate(_batch(tmp_path))

        kwargs = sam.image_series_annotator.call_args.kwargs
        assert kwargs["initial_segmentations"] is None


@pytest.mark.unit
class TestScriptedReview:
    @staticmethod
    def _review_batch(tmp_path):
        saved = np.full((8, 10), 5, np.uint16)
        item = BatchItem("a0", np.zeros((8, 10), np.uint16), initial_mask=saved)
        return Batch(1, (item,), tmp_path / "masks")

    def test_initial_mask_is_kept_without_review_fn(self, tmp_path):
        masks = ScriptedAnnotator(_label).annotate(self._review_batch(tmp_path))

        assert (tifffile.imread(masks["a0"]) == 5).all()

    def test_review_fn_edits_the_initial_mask(self, tmp_path):
        annotator = ScriptedAnnotator(_label, review_fn=lambda p, m: m * 2)

        masks = annotator.annotate(self._review_batch(tmp_path))

        assert (tifffile.imread(masks["a0"]) == 10).all()
