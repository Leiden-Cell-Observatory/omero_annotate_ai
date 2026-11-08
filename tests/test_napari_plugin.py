"""Tests for the napari plugin: headless Qt, the real core on fake OMERO and images."""

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

napari = pytest.importorskip("napari")
pytest.importorskip("pytestqt")

from omero_annotate_ai.napari_plugin._workers import Embedding  # noqa: E402
from omero_annotate_ai.trainingset import summarize  # noqa: E402
from tests.fakes import annotate, create_set, omero_set  # noqa: E402

pytestmark = pytest.mark.napari


class FakeController:
    """Stands in for micro-SAM's annotator: the test edits the mask the user 'drew'.

    Like the real viewer it keeps the array it is given (no copy) and the test edits
    it in place, so a widget that hands over its saved mask would see it change.
    """

    def __init__(self) -> None:
        self.shown: list[np.ndarray] = []
        self.held: list[np.ndarray | None] = []
        self.initial: list[np.ndarray | None] = []
        self.mask = np.zeros((16, 16), np.uint32)
        self.torn_down = False
        self.conflict: str | None = None

    def check_layers(self) -> None:
        # A fresh exception each time: a stored one keeps its traceback (and with it
        # the widget) in a reference cycle that the GC may free off the GUI thread.
        if self.conflict is not None:
            from omero_annotate_ai.napari_plugin._controller import LayerConflict

            raise LayerConflict(self.conflict)

    def hold(self, pixels) -> None:
        self.held.append(pixels)

    def show(self, pixels, embedding, model_type, mask=None) -> None:
        self.shown.append(pixels)
        self.initial.append(None if mask is None else mask.copy())
        self.mask = mask if mask is not None else np.zeros(pixels.shape, np.uint32)

    def current_mask(self) -> np.ndarray:
        return self.mask.copy()

    def teardown(self) -> None:
        self.torn_down = True


class InstantEmbedder:
    """Skips SAM: reports an embedding as ready straight away; records the paths."""

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def compute(self, image, save_path, model_type, on_done, on_error) -> None:
        self.paths.append(Path(save_path))
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
        # Each image is held (shown dimmed) until its embedding is ready; while a
        # next batch loads, the last image is held (None: no new pixels yet).
        assert [h is None for h in controller.held] == [False, False, True, False, True]
        for held, shown in zip(
            [h for h in controller.held if h is not None], controller.shown, strict=True
        ):
            assert held is shown

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

    def test_the_dock_shrinks_to_the_annotate_page(self, tmp_path, make_dock, qtbot):
        _, source, _, home = _make_set(tmp_path, "local")
        widget = make_dock(home, source)[0]
        start = widget.minimumSizeHint().height()
        _start(widget, qtbot)
        # micro-SAM's dock sits below: the short annotate page must not keep the
        # start page's height.
        assert widget.minimumSizeHint().height() < start

    def test_prev_goes_back_to_an_uncommitted_image(self, tmp_path, make_dock, qtbot):
        _, source, _, home = _make_set(tmp_path, "local")
        widget, controller, _ = make_dock(home, source)
        _start(widget, qtbot)
        widget.skip_button.click()
        assert widget._cursor == 1
        widget.prev_button.click()
        assert widget._cursor == 0
        assert len(controller.shown) == 3

    def test_the_next_embedding_is_computed_while_annotating(
        self, tmp_path, make_dock, qtbot
    ):
        _, source, _, home = _make_set(tmp_path, "local")
        widget = make_dock(home, source)[0]
        _start(widget, qtbot)
        names = [p.name for p in widget._embedder.paths]
        assert names == ["embedding_00000.zarr", "embedding_00001.zarr"]

    def test_prev_reopens_a_saved_image_and_saves_corrections(
        self, tmp_path, make_dock, qtbot
    ):
        ts, source, omero, home = _make_set(tmp_path, "omero")
        widget, controller, _ = make_dock(home, source, omero, conn=omero.conn())
        _start(widget, qtbot)
        first = widget._batch.items[0].annotation_id
        controller.mask[2:6, 2:6] = 1
        widget.accept_button.click()
        qtbot.waitUntil(lambda: widget._ready and widget._cursor == 1, timeout=5000)

        widget.prev_button.click()  # back to the saved image, with its mask
        qtbot.waitUntil(lambda: widget._ready and widget._cursor == 0, timeout=5000)
        assert controller.initial[-1].max() == 1
        label = next(r.label_id for r in ts.units() if r.annotation_id == first)
        controller.mask[2:6, 2:6] = 3  # a correction
        widget.accept_button.click()
        qtbot.waitUntil(lambda: widget._ready and widget._cursor == 1, timeout=5000)

        def replaced() -> bool:
            row = next(r for r in widget._session._set.rows if r.annotation_id == first)
            return row.label_id != label

        qtbot.waitUntil(replaced, timeout=5000)
        assert list(omero.uploads_per_tag().values()) == [(1, 1)]

    def test_prev_reaches_images_saved_in_earlier_batches(
        self, tmp_path, make_dock, qtbot
    ):
        _, source, _, home = _make_set(tmp_path, "local")  # batches of 2
        widget, controller, _ = make_dock(home, source)
        _start(widget, qtbot)
        first_batch = widget._batch.index
        for value in (1, 2):
            controller.mask[2:6, 2:6] = value
            widget.accept_button.click()
            qtbot.waitUntil(lambda: widget._ready, timeout=5000)
        assert widget._batch.index == first_batch + 1 and widget._cursor == 0

        widget.prev_button.click()
        qtbot.waitUntil(
            lambda: widget._ready and widget._revisit is not None, timeout=5000
        )
        assert controller.initial[-1].max() == 2  # the last saved image
        widget.prev_button.click()
        qtbot.waitUntil(lambda: controller.initial[-1].max() == 1, timeout=5000)
        widget.skip_button.click()  # unchanged: back to the image still to do
        qtbot.waitUntil(lambda: widget._ready and widget._revisit is None, timeout=5000)
        assert widget._batch.index == first_batch + 1 and widget._cursor == 0

    def test_a_correction_from_an_earlier_batch_is_saved_and_counted_once(
        self, tmp_path, make_dock, qtbot
    ):
        ts, source, _, home = _make_set(tmp_path, "local")  # batches of 2
        widget, controller, _ = make_dock(home, source)
        _start(widget, qtbot)
        for value in (1, 2):
            controller.mask[2:6, 2:6] = value
            widget.accept_button.click()
            qtbot.waitUntil(lambda: widget._ready, timeout=5000)
        assert len(widget._history) == 2

        widget.prev_button.click()
        qtbot.waitUntil(
            lambda: widget._ready and widget._revisit is not None, timeout=5000
        )
        corrected = widget._revisit
        controller.mask[:] = 0
        controller.mask[8:12, 8:12] = 9
        widget.accept_button.click()
        qtbot.waitUntil(lambda: widget._ready and widget._revisit is None, timeout=5000)

        def saved() -> np.ndarray:
            from omero_annotate_ai.bundle import mask_path

            return tifffile.imread(mask_path(ts.bundle_dir, corrected))

        qtbot.waitUntil(lambda: saved().max() == 9, timeout=5000)
        assert "Training 2/3" in widget.progress_label.text()  # not counted twice
        assert not list((ts.bundle_dir / ".scratch").glob("*/*_mask.tif"))

    def test_skip_does_nothing_while_the_next_batch_loads(
        self, tmp_path, make_dock, qtbot
    ):
        _, source, _, home = _make_set(tmp_path, "local")
        widget, controller, _ = make_dock(home, source)
        _start(widget, qtbot)
        for value in (1, 2):
            qtbot.waitUntil(lambda: widget._ready, timeout=5000)
            controller.mask[2:6, 2:6] = value
            widget.accept_button.click()
        widget._skip()  # the next batch is still loading
        widget._prev()
        qtbot.waitUntil(lambda: widget._ready, timeout=5000)
        assert len(widget._history) == 2  # added once
        assert widget._cursor == 0 and widget._revisit is None

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

    def test_review_starts_from_saved_masks_and_saves_only_changes(
        self, tmp_path, make_dock, qtbot
    ):
        ts, source, omero, home = _make_set(tmp_path, "omero", n_images=2)
        annotate(ts)
        before = omero.uploads_per_tag()
        widget, controller, _ = make_dock(home, source, omero, conn=omero.conn())
        widget.review.setChecked(True)
        _start(widget, qtbot)

        first = controller.initial[-1]
        assert first is not None and first.any()  # the saved mask, to correct
        controller.mask[:2, :2] = 0  # a correction
        widget.accept_button.click()
        qtbot.waitUntil(
            lambda: "revised 1" in widget.progress_label.text(), timeout=5000
        )
        qtbot.waitUntil(lambda: widget._ready, timeout=5000)
        assert controller.initial[-1] is not None
        widget.accept_button.click()  # second image accepted unchanged
        qtbot.waitUntil(lambda: not widget._ready, timeout=5000)
        widget.finish_button.click()
        qtbot.waitUntil(lambda: widget._pages.currentIndex() == 0, timeout=5000)

        record = summarize(ts.bundle_dir).sessions[-1]
        assert (record.kind, record.n_annotated) == ("review", 1)
        assert "1 image(s) revised" in widget.status.text()
        after = omero.uploads_per_tag()
        assert list(after.values()) == [(1, 1), (1, 1)]  # replaced, not duplicated
        assert set(after) == set(before)
        revised = [u for u in omero.labels.values() if not u.pixels[:2, :2].any()]
        assert len(revised) == 1

    def test_units_needing_review_and_claims_elsewhere_are_announced(
        self, tmp_path, make_dock, qtbot
    ):
        from datetime import UTC, datetime

        from omero_annotate_ai.session import NEEDS_REVIEW

        ts, source, omero, home = _make_set(tmp_path, "omero", n_images=3)
        table = omero.tables[ts.table_id]
        table.loc[0, "claimed_by"] = "1@someone/elsewhere"
        table.loc[0, "claimed_at"] = datetime.now(UTC).isoformat()
        ts.rows[1].error = NEEDS_REVIEW + "the local mask differs"
        ts.save_local_table()
        widget, _, _ = make_dock(home, source, omero, conn=omero.conn())
        _start(widget, qtbot)

        assert "1 image(s) need review" in widget.notice.text()
        assert "another computer" in widget.notice.text()
        assert widget._batch is not None
        shown = {item.annotation_id for item in widget._batch.items}
        assert shown == {ts.rows[2].annotation_id}

    def test_closing_the_viewer_window_releases_claims(
        self, tmp_path, make_dock, qtbot
    ):
        from qtpy.QtWidgets import QMainWindow

        ts, source, omero, home = _make_set(tmp_path, "omero", n_images=2)
        widget, _, _ = make_dock(home, source, omero, conn=omero.conn())
        window = QMainWindow()  # stands in for napari's window, which hosts the dock
        window.setCentralWidget(widget)
        window.show()
        _start(widget, qtbot)
        table = omero.tables[ts.table_id]
        assert (table["claimed_by"].astype(str) != "").all()  # both shown are claimed

        window.close()  # napari keeps running (e.g. Jupyter); the session must end
        assert summarize(ts.bundle_dir).sessions[-1].status == "interrupted"
        assert (omero.tables[ts.table_id]["claimed_by"].fillna("") == "").all()

    def test_finish_stops_accepting_and_handing_out_batches(
        self, tmp_path, make_dock, qtbot
    ):
        ts, source, _, home = _make_set(tmp_path, "local")
        widget, controller, viewer = make_dock(home, source)
        _start(widget, qtbot)
        controller.mask[2:6, 2:6] = 1
        widget.finish_button.click()
        assert "N" not in {str(k) for k in viewer.keymap}  # unbound at once
        widget._accept()  # a late key press or click does nothing
        widget._skip()
        widget._advance()
        qtbot.waitUntil(lambda: widget._pages.currentIndex() == 0, timeout=5000)

        summary = summarize(ts.bundle_dir)
        assert summary.progress["training"] == (0, 3)
        assert summary.sessions[-1].status == "finished"
        assert not list((ts.bundle_dir / "annotation_output").glob("*_mask.tif"))

    def test_a_failing_first_batch_interrupts_the_session(
        self, tmp_path, make_dock, qtbot, monkeypatch
    ):
        from omero_annotate_ai.session import AnnotationSession

        ts, source, omero, home = _make_set(tmp_path, "omero", n_images=2)
        widget, _, _ = make_dock(home, source, omero, conn=omero.conn())

        def broken(self):
            raise OSError("image server unreachable")

        monkeypatch.setattr(AnnotationSession, "next_batch", broken)
        widget.open_button.click()
        assert not widget.folder_button.isEnabled()  # nothing else starts meanwhile
        assert not widget.review.isEnabled()
        qtbot.waitUntil(lambda: "unreachable" in widget.status.text(), timeout=5000)

        assert summarize(ts.bundle_dir).sessions[-1].status == "interrupted"
        assert widget.open_button.isEnabled() and widget.folder_button.isEnabled()
        assert widget.review.isEnabled() and widget._session is None

    def test_a_layer_conflict_is_reported_before_anything_is_claimed(
        self, tmp_path, make_dock, qtbot
    ):
        ts, source, omero, home = _make_set(tmp_path, "omero", n_images=2)
        widget, controller, _ = make_dock(home, source, omero, conn=omero.conn())
        controller.conflict = "Rename or remove the layers 'image'"
        widget.open_button.click()

        assert "Rename or remove" in widget.status.text()
        assert summarize(ts.bundle_dir).sessions == ()
        assert (omero.tables[ts.table_id]["claimed_by"].fillna("") == "").all()

    def test_review_edits_never_change_the_saved_mask(self, tmp_path, make_dock, qtbot):
        ts, source, omero, home = _make_set(tmp_path, "omero", n_images=2)
        annotate(ts)
        widget, controller, _ = make_dock(home, source, omero, conn=omero.conn())
        widget.review.setChecked(True)
        _start(widget, qtbot)
        saved = controller.initial[-1]
        assert saved is not None
        controller.mask[:] = 0  # edits in the viewer, not accepted
        widget.skip_button.click()
        qtbot.waitUntil(lambda: widget._ready, timeout=5000)
        widget.prev_button.click()  # back to the first image: its saved mask again
        qtbot.waitUntil(lambda: widget._ready, timeout=5000)
        np.testing.assert_array_equal(controller.initial[-1], saved)
        widget.accept_button.click()  # unchanged: the upload stays
        widget.finish_button.click()
        qtbot.waitUntil(lambda: widget._pages.currentIndex() == 0, timeout=5000)
        assert summarize(ts.bundle_dir).sessions[-1].n_annotated == 0

    def test_an_image_that_fails_to_show_can_be_skipped(
        self, tmp_path, make_dock, qtbot
    ):
        _, source, _, home = _make_set(tmp_path, "local")
        widget, controller, _ = make_dock(home, source)

        def broken(*args, **kwargs):
            raise RuntimeError("embedding failed")

        controller.show = broken
        widget.open_button.click()
        qtbot.waitUntil(
            lambda: "embedding failed" in widget.status.text(), timeout=5000
        )
        assert widget.skip_button.isEnabled() and not widget.accept_button.isEnabled()
        widget.skip_button.click()
        assert widget._cursor == 1

    def test_closing_the_window_is_ignored_when_the_close_is_cancelled(
        self, tmp_path, make_dock, qtbot
    ):
        from qtpy.QtWidgets import QMainWindow

        class AskingWindow(QMainWindow):  # napari asks; here the user says "no"
            def closeEvent(self, event):
                event.ignore()

        ts, source, _, home = _make_set(tmp_path, "local")
        widget, _, _ = make_dock(home, source)
        window = AskingWindow()
        window.setCentralWidget(widget)
        window.show()
        _start(widget, qtbot)
        window.close()
        qtbot.wait(50)  # the deferred check runs
        assert widget._session is not None
        assert summarize(ts.bundle_dir).sessions[-1].status == "open"

    def test_removing_the_dock_ends_the_session(self, tmp_path, make_dock, qtbot):
        from qtpy.QtWidgets import QWidget

        ts, source, omero, home = _make_set(tmp_path, "omero", n_images=2)
        widget, controller, viewer = make_dock(home, source, omero, conn=omero.conn())
        holder = QWidget()  # napari's dock widget
        widget.setParent(holder)
        _start(widget, qtbot)
        widget.setParent(None)  # napari removes the dock
        assert summarize(ts.bundle_dir).sessions[-1].status == "interrupted"
        assert (omero.tables[ts.table_id]["claimed_by"].fillna("") == "").all()
        assert controller.torn_down
        assert "N" not in {str(k) for k in viewer.keymap}

    def test_deleting_the_dock_ends_the_session(self, tmp_path, qtbot):
        from napari.components import ViewerModel
        from qtpy.QtWidgets import QWidget

        from omero_annotate_ai.napari_plugin._widget import OMEROAnnotateDock

        ts, source, _, home = _make_set(tmp_path, "local")
        holder = QWidget()
        widget = OMEROAnnotateDock(
            ViewerModel(),
            home=home,
            core_options={"image_source": source},
            controller=FakeController(),  # type: ignore[arg-type]
            embedder=InstantEmbedder(),  # type: ignore[arg-type]
        )
        widget.setParent(holder)
        qtbot.waitUntil(lambda: widget.open_button.isEnabled(), timeout=5000)
        _start(widget, qtbot)
        holder.deleteLater()  # napari deletes the dock with the widget in it
        qtbot.waitUntil(
            lambda: summarize(ts.bundle_dir).sessions[-1].status == "interrupted",
            timeout=5000,
        )

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
        assert widget.password.text() == ""
        assert widget.connect_button.isEnabled()
        assert not widget.list_button.isEnabled()

    def test_the_password_is_never_stored(
        self, tmp_path, make_dock, qtbot, monkeypatch
    ):
        """The real connect() runs against a fake gateway; nothing reaches the
        keychain or the connection history, and the field is cleared."""
        from unittest.mock import MagicMock

        from omero_annotate_ai.omero import simple_connection as sc

        _, source, _, home = _make_set(tmp_path, "local")
        widget = make_dock(home, source)[0]
        history = sc.SimpleOMEROConnection()._get_connections_file()
        before = history.read_bytes() if history.exists() else None
        keyring = MagicMock()
        monkeypatch.setattr(sc, "keyring", keyring, raising=False)
        monkeypatch.setattr(sc, "BlitzGateway", MagicMock())
        widget.host.setText("omero.example.org")
        widget.user.setText("someone")
        widget.password.setText("secret")
        widget.connect_button.click()
        assert widget.password.text() == ""
        qtbot.waitUntil(lambda: widget.list_button.isEnabled(), timeout=5000)
        assert not keyring.set_password.called
        assert (history.read_bytes() if history.exists() else None) == before


@pytest.mark.docker
@pytest.mark.skipif(
    sys.platform == "darwin" and bool(os.environ.get("CI")),
    reason="see TestDock",
)
class TestDockOnOmero:
    """The dock against the Docker test OMERO: annotate, then review, same set."""

    def test_annotate_then_review_on_omero(
        self, omero_conn, omero_seed, qtbot, tmp_path
    ):
        from napari.components import ViewerModel

        from omero_annotate_ai import TrainingSet, create_default_config
        from omero_annotate_ai.napari_plugin._widget import OMEROAnnotateDock

        config = create_default_config()
        config.name = f"{omero_seed.tag}_napari"
        config.omero.container_type = "dataset"
        config.omero.container_id = omero_seed.dataset_id
        config.training.train_n, config.training.validate_n = 2, 0
        config.training.test_n = 0
        config.training.seed = 7
        home = tmp_path / "home"
        ts = TrainingSet.create(config, omero_conn, home=home)

        controller = FakeController()
        widget = OMEROAnnotateDock(
            ViewerModel(),
            home=home,
            controller=controller,  # type: ignore[arg-type]
            embedder=InstantEmbedder(),  # type: ignore[arg-type]
        )
        widget._conn = omero_conn
        qtbot.addWidget(widget)
        qtbot.waitUntil(lambda: widget.open_button.isEnabled(), timeout=20000)

        def session(review: bool, draw) -> None:
            qtbot.waitUntil(lambda: widget.open_button.isEnabled(), timeout=60000)
            widget.review.setChecked(review)
            widget.open_button.click()
            for _ in range(2):
                qtbot.waitUntil(lambda: widget._ready, timeout=60000)
                controller.mask = draw(controller)
                widget.accept_button.click()
            qtbot.waitUntil(lambda: not widget._ready, timeout=60000)
            widget.note.setText("review pass" if review else "first pass")
            widget.finish_button.click()
            qtbot.waitUntil(lambda: widget._pages.currentIndex() == 0, timeout=60000)

        def first(c):
            mask = np.zeros(c.shown[-1].shape, np.uint32)
            mask[4:12, 4:12] = 1
            return mask

        def corrected(c):
            mask = c.initial[-1].astype(np.uint32)  # the saved mask came back
            mask[20:28, 20:28] = 2
            return mask

        session(False, first)
        annotated = TrainingSet.open(ts.table_id, omero_conn, home=home)
        rows = {r.annotation_id: r for r in annotated.units()}
        assert all(r.processed and r.label_id and r.roi_id for r in rows.values())

        session(True, corrected)
        reviewed = TrainingSet.open(ts.table_id, omero_conn, home=home)
        for row in reviewed.units():
            old = rows[row.annotation_id]
            assert row.label_id != old.label_id and row.roi_id != old.roi_id
            assert omero_conn.getObject("FileAnnotation", old.label_id) is None
            assert omero_conn.getObject("Roi", old.roi_id) is None
        kinds = [(s.kind, s.n_annotated, s.note) for s in reviewed.config.sessions]
        assert kinds == [("annotate", 2, "first pass"), ("review", 2, "review pass")]
        assert all(not r.claimed_by for r in reviewed.units())


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
        for name in ("skip_load", "use_cli"):
            assert name in params, f"AnnotatorState.initialize_predictor lost '{name}'"
        for name in ("image_shape", "amg", "annotator"):
            assert hasattr(AnnotatorState(), name), f"AnnotatorState lost '{name}'"
        from micro_sam.instance_segmentation import (
            AMGBase,
            InstanceSegmentationWithDecoder,
        )

        assert callable(AMGBase.clear_state)
        assert callable(InstanceSegmentationWithDecoder.clear_state)
        assert "reset_state" in inspect.signature(Annotator2d.__init__).parameters
        assert (
            "segmentation_result"
            in inspect.signature(_AnnotatorBase._update_image).parameters
        )

    def test_other_micro_sam_versions_are_refused(self, monkeypatch):
        import micro_sam

        from omero_annotate_ai.napari_plugin._controller import (
            check_micro_sam_version,
        )

        check_micro_sam_version()  # the installed one is supported
        monkeypatch.setattr(micro_sam, "__version__", "1.9.0")
        with pytest.raises(RuntimeError, match="needs micro-SAM 1.8"):
            check_micro_sam_version()

    def test_layer_names_already_in_the_viewer_are_a_conflict(self):
        from napari.components import ViewerModel

        from omero_annotate_ai.napari_plugin._controller import (
            LayerConflict,
            MicroSamController,
        )

        viewer = ViewerModel()
        viewer.add_labels(np.zeros((8, 8), np.uint32), name="committed_objects")
        with pytest.raises(LayerConflict):
            MicroSamController(viewer).check_layers()

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

        # A saved embedding is loaded, never computed again.
        predictor = results[0].predictor

        def no_recompute():
            raise AssertionError("embedding computed again")

        reset, predictor.reset_image = predictor.reset_image, no_recompute
        util.precompute_image_embeddings(
            predictor=predictor,
            input_=images[0],
            save_path=str(results[0].path),
            ndim=2,
            verbose=False,
        )
        predictor.reset_image = reset

        viewer.add_image(images[1], name="mine")  # the user's own layer
        controller = MicroSamController(viewer)
        controller.show(images[0], results[0], "vit_b_lm")
        assert MASK_LAYER in viewer.layers and "image" in viewer.layers
        viewer.add_image(images[0], name="added mid-session")
        viewer.layers[MASK_LAYER].data[5:9, 5:9] = 3
        assert controller.current_mask().max() == 3
        # While the next embedding computes: the next image, dimmed; no old objects.
        viewer.layers["image"].opacity = 0.8  # the user's choice survives holds
        controller.hold(images[1])
        np.testing.assert_array_equal(viewer.layers["image"].data, images[1])
        assert viewer.layers["image"].opacity < 1
        assert not viewer.layers[MASK_LAYER].visible
        assert viewer.layers["mine"].visible  # the user's layers stay as they are
        controller.show(images[1], results[1], "vit_b_lm")  # fresh empty mask
        assert not controller.current_mask().any()
        assert viewer.layers[MASK_LAYER].visible
        assert viewer.layers["image"].opacity == 0.8
        saved = np.zeros(images[0].shape, np.uint16)
        saved[30:60, 30:60] = 4
        controller.show(images[0], results[0], "vit_b_lm", saved)  # a review
        np.testing.assert_array_equal(controller.current_mask(), saved)
        controller.teardown()
        assert [layer.name for layer in viewer.layers] == ["mine", "added mid-session"]
        controller.show(images[1], results[1], "vit_b_lm")  # a next session starts
        controller.teardown()


@pytest.mark.slow
def test_dock_with_real_micro_sam_annotates_and_reviews(
    make_napari_viewer, qtbot, tmp_path
):
    """The whole plugin with real micro-SAM in a real viewer (no human: the test
    writes the committed objects a user would draw)."""
    from micro_sam import util

    if not (Path(util.get_cache_directory()) / "models" / "vit_b_lm").exists():
        pytest.skip("vit_b_lm weights not cached; this test does not download them")
    from omero_annotate_ai.napari_plugin._controller import MASK_LAYER
    from omero_annotate_ai.napari_plugin._widget import OMEROAnnotateDock
    from tests.fakes import FakeImageSource

    source = FakeImageSource({i: _synthetic_image(i)[None, None, None] for i in (1, 2)})
    config, _, omero = omero_set(tmp_path, n_images=2)
    config.storage = "local"
    home = tmp_path / "home"
    ts = create_set(config, source, omero, home)
    viewer = make_napari_viewer()
    widget = OMEROAnnotateDock(viewer, home=home, core_options={"image_source": source})
    viewer.window.add_dock_widget(widget)

    def wait(condition, timeout: int) -> None:
        try:
            qtbot.waitUntil(condition, timeout=timeout)
        except qtbot.TimeoutError:
            pytest.fail(f"timed out; dock status: {widget.status.text()!r}")

    wait(lambda: widget.open_button.isEnabled(), 20000)

    def run(review: bool) -> None:
        wait(lambda: widget.open_button.isEnabled(), 60000)  # the list is refreshed
        widget.review.setChecked(review)
        widget.open_button.click()
        for label in (1, 2):
            wait(lambda: widget._ready, 180_000)
            layer = viewer.layers[MASK_LAYER]
            if review:
                assert layer.data.max() == label  # the saved mask is shown
            data = np.zeros(layer.data.shape, layer.data.dtype)
            data[10:40, 10:40] = label
            if review and label == 1:
                data[100:120, 100:120] = 9  # one correction
            layer.data = data
            widget.accept_button.click()
        wait(lambda: not widget._ready, 60000)
        widget.finish_button.click()
        wait(lambda: widget._pages.currentIndex() == 0, 60000)

    run(review=False)
    run(review=True)
    sessions = summarize(ts.bundle_dir).sessions
    assert [(s.kind, s.n_annotated) for s in sessions] == [
        ("annotate", 2),
        ("review", 1),  # the unchanged second image kept its mask
    ]


def test_manifest_is_valid_and_points_at_the_widget():
    from npe2 import PluginManifest

    import omero_annotate_ai.napari_plugin as plugin

    manifest = PluginManifest.from_file(Path(plugin.__file__).with_name("napari.yaml"))
    (command,) = manifest.contributions.commands or []
    module, _, attr = (command.python_name or "").partition(":")
    widget_cls = getattr(__import__(module, fromlist=[attr]), attr)
    assert "napari_viewer" in inspect.signature(widget_cls.__init__).parameters
