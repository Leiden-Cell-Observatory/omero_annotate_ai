"""Tests for the napari plugin: headless Qt, the real core on fake OMERO and images."""

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("pytestqt")

from omero_annotate_ai.napari_plugin._workers import Embedding  # noqa: E402
from omero_annotate_ai.trainingset import summarize  # noqa: E402
from tests.fakes import create_set, omero_set  # noqa: E402

pytestmark = pytest.mark.napari


class FakeController:
    """Stands in for micro-SAM's annotator: the test sets the mask the user 'drew'."""

    def __init__(self) -> None:
        self.shown: list[np.ndarray] = []
        self.mask = np.zeros((16, 16), np.uint32)
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


def _make_set(tmp_path, storage: str, n_images: int = 3):
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    config.storage = storage
    config.workflow.batch_size = 2
    home = tmp_path / "home"
    ts = create_set(config, source, omero, home)
    return ts, source, omero, home


@pytest.fixture
def make_dock(qtbot):
    from napari.components import ViewerModel

    from omero_annotate_ai.napari_plugin._widget import OMEROAnnotateDock

    def make(home, source, omero=None, conn=None):
        # A viewer model without a canvas: the widget only binds keys, micro-SAM is
        # faked, and an OpenGL canvas crashes on headless CI runners (macOS).
        viewer = ViewerModel()
        controller = FakeController()
        options = {"image_source": source}
        if omero is not None:
            options["omero_writes"] = omero
        widget = OMEROAnnotateDock(
            viewer,
            home=home,
            core_options=options,
            controller=controller,  # type: ignore[arg-type]
            embedder=InstantEmbedder(),  # type: ignore[arg-type]
        )
        widget._conn = conn
        qtbot.addWidget(widget)
        qtbot.waitUntil(lambda: widget.open_button.isEnabled(), timeout=5000)
        return widget, controller, viewer

    return make


def _start(widget, qtbot) -> None:
    widget.open_button.click()
    qtbot.waitUntil(
        lambda: widget._pages.currentIndex() == 1 and widget._ready, timeout=10000
    )


@pytest.mark.skipif(
    sys.platform == "darwin" and bool(os.environ.get("CI")),
    reason="Qt widgets with background threads segfault on headless GitHub macOS "
    "runners (not reproduced on Linux/Windows); run these locally on macOS",
)
class TestDock:
    """The annotate loop end to end on the real core; only micro-SAM is faked."""

    def test_local_set_accept_skip_empty_and_finish_with_note(
        self, tmp_path, make_dock, qtbot
    ):
        ts, source, _, home = _make_set(tmp_path, "local")
        widget, controller, _ = make_dock(home, source)
        assert "Training: 0/3" in widget.summary.text()
        _start(widget, qtbot)
        assert "Training 0/3" in widget.progress_label.text()

        controller.mask[2:6, 2:6] = 1
        widget.accept_button.click()  # image 1: saved
        qtbot.waitUntil(
            lambda: "Training 1/3" in widget.progress_label.text(), timeout=5000
        )
        widget.skip_button.click()  # image 2: skipped, back for a later session
        controller.mask[:] = 0
        qtbot.waitUntil(lambda: widget._ready, timeout=5000)  # image 3 (next batch)
        widget.accept_button.click()  # empty: needs confirmation
        assert "Press Accept again" in widget.status.text()
        widget.accept_button.click()
        qtbot.waitUntil(
            lambda: "Training 2/3" in widget.progress_label.text(), timeout=5000
        )

        widget.note.setText("nuclei only, skipped debris")
        widget.finish_button.click()
        qtbot.waitUntil(lambda: widget._pages.currentIndex() == 0, timeout=5000)
        qtbot.waitUntil(lambda: "nuclei only" in widget.summary.text(), timeout=5000)

        summary = summarize(ts.bundle_dir)
        assert summary.progress["training"] == (2, 3)
        assert summary.sessions[-1].note == "nuclei only, skipped debris"
        assert summary.sessions[-1].status == "finished"
        saved = sorted((ts.bundle_dir / "annotation_output").glob("*_mask.tif"))
        assert len(saved) == 2
        assert controller.torn_down

    def test_omero_set_uploads_each_accepted_image_once(
        self, tmp_path, make_dock, qtbot
    ):
        _, source, omero, home = _make_set(tmp_path, "omero", n_images=2)
        widget, controller, _ = make_dock(home, source, omero, conn=omero.conn())
        _start(widget, qtbot)
        controller.mask[1:4, 1:4] = 7
        widget.accept_button.click()
        qtbot.waitUntil(
            lambda: "Training 1/2" in widget.progress_label.text(), timeout=5000
        )
        widget.finish_button.click()  # image 2 stays pending for a later session
        qtbot.waitUntil(lambda: widget._pages.currentIndex() == 0, timeout=5000)

        tags = omero.uploads_per_tag()
        assert list(tags.values()) == [(1, 1)]  # one label and one ROI, tagged
        qtbot.waitUntil(lambda: "Training: 1/2" in widget.summary.text(), timeout=5000)

    def test_prev_goes_back_to_an_uncommitted_image(self, tmp_path, make_dock, qtbot):
        _, source, _, home = _make_set(tmp_path, "local")
        widget, controller, _ = make_dock(home, source)
        _start(widget, qtbot)
        widget.skip_button.click()
        assert widget._cursor == 1
        widget.prev_button.click()
        assert widget._cursor == 0
        assert len(controller.shown) == 3

    def test_keys_are_bound_only_while_annotating(self, tmp_path, make_dock, qtbot):
        _, source, _, home = _make_set(tmp_path, "local")
        widget, _, viewer = make_dock(home, source)
        _start(widget, qtbot)
        assert "N" in {str(k) for k in viewer.keymap}
        widget.finish_button.click()
        qtbot.waitUntil(lambda: widget._pages.currentIndex() == 0, timeout=5000)
        assert "N" not in {str(k) for k in viewer.keymap}

    def test_quitting_mid_session_records_it_as_interrupted(
        self, tmp_path, make_dock, qtbot
    ):
        ts, source, _, home = _make_set(tmp_path, "local")
        widget, _, _ = make_dock(home, source)
        _start(widget, qtbot)
        widget._interrupt()
        assert summarize(ts.bundle_dir).sessions[-1].status == "interrupted"

    def test_failed_connection_is_reported(
        self, tmp_path, make_dock, qtbot, monkeypatch
    ):
        from omero_annotate_ai.errors import ConnectionFailed
        from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection

        _, source, _, home = _make_set(tmp_path, "local")
        widget = make_dock(home, source)[0]

        def refuse(self, *args, **kwargs):
            raise ConnectionFailed("login failed")

        monkeypatch.setattr(SimpleOMEROConnection, "connect", refuse)
        widget.host.setText("omero.example.org")
        widget.user.setText("someone")
        widget.password.setText("secret")
        widget.connect_button.click()
        qtbot.waitUntil(lambda: "login failed" in widget.status.text(), timeout=5000)
        assert widget.connect_button.isEnabled()
        assert not widget.list_button.isEnabled()


def _synthetic_image(seed: int, shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    """A uint16 image with a few bright round 'nuclei' on a noisy background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    image = rng.normal(200, 20, shape)
    for _ in range(8):
        cy, cx = rng.integers(20, shape[0] - 20), rng.integers(20, shape[1] - 20)
        image[(yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(8, 18) ** 2] += 1500
    return np.clip(image, 0, 65535).astype(np.uint16)


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
        images = [_synthetic_image(1), _synthetic_image(2)]
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
        controller.show(images[1], results[1], "vit_b_lm")  # fresh empty mask
        assert not controller.current_mask().any()
        controller.teardown()


def test_manifest_is_valid_and_points_at_the_widget():
    from npe2 import PluginManifest

    import omero_annotate_ai.napari_plugin as plugin

    manifest = PluginManifest.from_file(Path(plugin.__file__).with_name("napari.yaml"))
    (command,) = manifest.contributions.commands or []
    module, _, attr = (command.python_name or "").partition(":")
    widget_cls = getattr(__import__(module, fromlist=[attr]), attr)
    assert "napari_viewer" in inspect.signature(widget_cls.__init__).parameters
