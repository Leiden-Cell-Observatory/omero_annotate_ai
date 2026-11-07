"""Tests for the napari plugin (headless Qt, against the plugin's fake core)."""

import inspect
from pathlib import Path

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("pytestqt")

from omero_annotate_ai.napari_plugin import _core  # noqa: E402
from omero_annotate_ai.napari_plugin._core import (  # noqa: E402
    AnnotatorResult,
    FakeBackend,
    UnitCommitted,
    series_mask_name,
)
from omero_annotate_ai.napari_plugin._workers import Embedding  # noqa: E402

pytestmark = pytest.mark.napari


class FakeController:
    """Stands in for micro-SAM's annotator: the test sets the mask the user 'drew'."""

    def __init__(self) -> None:
        self.shown: list[np.ndarray] = []
        self.mask = np.zeros((256, 256), np.uint32)
        self.torn_down = False

    def show(self, pixels, embedding, model_type) -> None:
        self.shown.append(pixels)

    def current_mask(self) -> np.ndarray:
        return self.mask.copy()

    def teardown(self) -> None:
        self.torn_down = True


class InstantEmbedder:
    """Skips SAM: reports an embedding as ready straight away."""

    def compute(self, image, save_path, model_type, on_done, on_error) -> None:
        on_done(Embedding(Path(save_path), None, None))


@pytest.fixture
def dock(make_napari_viewer, qtbot, tmp_path):
    from omero_annotate_ai.napari_plugin._widget import OMEROAnnotateDock

    viewer = make_napari_viewer()
    controller = FakeController()
    widget = OMEROAnnotateDock(
        viewer,
        controller=controller,  # type: ignore[arg-type]
        embedder=InstantEmbedder(),  # type: ignore[arg-type]
        demo_root=tmp_path / "demo",
    )
    qtbot.addWidget(widget)
    return widget, controller, viewer, tmp_path / "demo"


def _open_demo(widget, qtbot) -> None:
    widget.demo_button.click()
    qtbot.waitUntil(lambda: widget.open_button.isEnabled(), timeout=5000)
    widget.open_button.click()
    qtbot.waitUntil(
        lambda: widget._pages.currentIndex() == 1 and widget._ready, timeout=5000
    )


class TestFakeCore:
    """The fake follows the planned core contract the widget relies on."""

    def test_partial_commit_is_idempotent_and_release_returns_units(self, tmp_path):
        events = []
        ts = FakeBackend(tmp_path).open(str(tmp_path), events.append)
        session = ts.start_session()
        batch = session.next_batch()
        assert batch is not None and len(batch.items) == 3

        first = batch.items[0].unit.annotation_id
        mask = batch.mask_dir / series_mask_name(0)
        mask.write_bytes(b"")  # any file stands in for a mask here
        session.commit(batch, AnnotatorResult({first: mask}))
        session.commit(batch, AnnotatorResult({first: mask}))  # second call: no-op
        assert events == [UnitCommitted(first)]
        assert (tmp_path / "annotation_output" / f"{first}_mask.tif").exists()

        session.release(batch)
        again = session.next_batch()
        assert again is not None
        assert first not in {i.unit.annotation_id for i in again.items}
        assert batch.items[1].unit.annotation_id in {
            i.unit.annotation_id for i in again.items
        }

        record = session.finish("nuclei only")
        assert (record.n_annotated, record.note, record.status) == (
            1,
            "nuclei only",
            "finished",
        )
        assert ts.summary().counts["training"] == (1, 4)


class TestDock:
    """The annotate loop end to end, with the core and micro-SAM faked."""

    def test_accept_skip_empty_and_finish_with_note(self, dock, qtbot):
        widget, controller, viewer, root = dock
        _open_demo(widget, qtbot)
        assert widget._pages.currentIndex() == 1
        assert "Training 0/4" in widget.progress_label.text()

        controller.mask[10:20, 10:20] = 1
        widget.accept_button.click()  # image 1: saved
        qtbot.waitUntil(
            lambda: "Training 1/4" in widget.progress_label.text(), timeout=5000
        )

        widget.skip_button.click()  # image 2: skipped
        controller.mask[:] = 0
        widget.accept_button.click()  # image 3: empty, needs confirmation
        assert "Press Accept again" in widget.status.text()
        widget.accept_button.click()
        qtbot.waitUntil(
            lambda: "Training 2/4" in widget.progress_label.text(), timeout=5000
        )

        widget.note.setText("nuclei only, skipped debris")
        widget.finish_button.click()
        qtbot.waitUntil(lambda: widget._pages.currentIndex() == 0, timeout=5000)
        qtbot.waitUntil(lambda: "nuclei only" in widget.summary.text(), timeout=5000)

        saved = sorted(p.name for p in (root / "annotation_output").iterdir())
        assert len(saved) == 2
        assert controller.torn_down
        assert "Training: 2/4" in widget.summary.text()

    def test_prev_goes_back_to_an_uncommitted_image(self, dock, qtbot):
        widget, controller, _, _ = dock
        _open_demo(widget, qtbot)
        widget.skip_button.click()
        assert widget._cursor == 1
        widget.prev_button.click()
        assert widget._cursor == 0
        assert len(controller.shown) == 3

    def test_keys_are_bound_only_while_annotating(self, dock, qtbot):
        widget, _, viewer, _ = dock
        _open_demo(widget, qtbot)
        assert "N" in {str(k) for k in viewer.keymap}
        widget.finish_button.click()
        qtbot.waitUntil(lambda: widget._pages.currentIndex() == 0, timeout=5000)
        assert "N" not in {str(k) for k in viewer.keymap}

    def test_failed_connection_is_reported(self, dock, qtbot, monkeypatch):
        from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection

        widget = dock[0]
        monkeypatch.setattr(
            SimpleOMEROConnection, "connect", lambda self, *a, **k: None
        )
        widget.host.setText("omero.example.org")
        widget.user.setText("someone")
        widget.password.setText("secret")
        widget.connect_button.click()
        qtbot.waitUntil(
            lambda: "Could not connect" in widget.status.text(), timeout=5000
        )
        assert widget.connect_button.isEnabled()


class TestMicroSamContract:
    """The micro-SAM internals the controller drives. If this fails after a micro-SAM
    upgrade, update napari_plugin/_controller.py and the version pin of the `napari`
    extra together."""

    def test_private_api_still_exists(self):
        from micro_sam.sam_annotator._annotator import _AnnotatorBase
        from micro_sam.sam_annotator._state import AnnotatorState
        from micro_sam.sam_annotator.annotator_2d import Annotator2d

        params = inspect.signature(AnnotatorState.initialize_predictor).parameters
        for name in (
            "image_data",
            "model_type",
            "ndim",
            "save_path",
            "predictor",
            "decoder",
        ):
            assert name in params, f"AnnotatorState.initialize_predictor lost '{name}'"
        for name in ("image_shape", "amg", "annotator"):
            assert hasattr(AnnotatorState(), name), f"AnnotatorState lost '{name}'"
        assert "reset_state" in inspect.signature(Annotator2d.__init__).parameters
        assert (
            "segmentation_result"
            in inspect.signature(_AnnotatorBase._update_image).parameters
        )

    @pytest.mark.slow
    def test_controller_shows_an_image_and_reads_the_mask(
        self, make_napari_viewer, qtbot, tmp_path
    ):
        from micro_sam import util

        if not (Path(util.get_cache_directory()) / "models" / "vit_b_lm").exists():
            pytest.skip("vit_b_lm weights not cached; this test does not download them")
        from omero_annotate_ai.napari_plugin._controller import (
            MASK_LAYER,
            MicroSamController,
        )
        from omero_annotate_ai.napari_plugin._workers import Embedder, Worker

        viewer = make_napari_viewer()
        images = [_core.synthetic_image(1), _core.synthetic_image(2)]
        results: list[Embedding] = []
        embedder = Embedder(Worker("test-compute"))
        for i, image in enumerate(images):
            path = tmp_path / f"embedding_{i:05}.zarr"
            embedder.compute(image, path, "vit_b_lm", results.append, pytest.fail)
        qtbot.waitUntil(lambda: len(results) == 2, timeout=600_000)

        controller = MicroSamController(viewer)
        controller.show(images[0], results[0], "vit_b_lm")
        assert MASK_LAYER in viewer.layers
        viewer.layers[MASK_LAYER].data[5:9, 5:9] = 3
        assert controller.current_mask().max() == 3
        controller.show(
            images[1], results[1], "vit_b_lm"
        )  # next image: fresh empty mask
        assert not controller.current_mask().any()
        controller.teardown()


def test_manifest_is_valid_and_points_at_the_widget():
    from npe2 import PluginManifest

    manifest_path = Path(_core.__file__).with_name("napari.yaml")
    manifest = PluginManifest.from_file(manifest_path)
    (command,) = manifest.contributions.commands or []
    module, _, attr = (command.python_name or "").partition(":")
    widget_cls = getattr(__import__(module, fromlist=[attr]), attr)
    assert "napari_viewer" in inspect.signature(widget_cls.__init__).parameters
