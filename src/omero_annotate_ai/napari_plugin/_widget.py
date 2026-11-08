# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
# (conda's PyQt5 ships no type stubs, so Qt classes are unknown to pyright.)
"""The plugin's dock widget: connect, pick a training set, annotate, finish.

Every call into the core (connect, list, open, next batch, commit, finish) runs on
one IO thread, because a BlitzGateway must be used from one thread at a time.

No ``from __future__ import annotations`` here: npe2 reads the ``napari.Viewer``
annotation of ``__init__`` at runtime to inject the viewer.
"""

import contextlib
import functools
import html
import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import napari
import numpy as np
import tifffile
from qtpy.QtCore import QEvent, QObject, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..annotators import Batch, BatchItem
from ..bundle import home_dir
from ..events import (
    ClaimsHeld,
    ClaimsReleased,
    Event,
    ReconcileAction,
    UnitCommitted,
    UnitConflict,
    UploadFailed,
)
from ..omero.simple_connection import SimpleOMEROConnection
from ..session import NEEDS_REVIEW, AnnotationSession
from ..trainingset import (
    TrainingSet,
    TrainingSetSummary,
    list_training_sets,
    summarize,
)
from ._controller import LayerConflict, MicroSamController
from ._workers import Embedder, Embedding, EventBridge, Worker

logger = logging.getLogger(__name__)

ACCEPT_KEY = "n"
SKIP_KEY = "Shift-N"
CATEGORIES = ("training", "validation", "test")
CONTAINER_TYPES = ("dataset", "plate", "project", "screen")

Ref = int | Path
"""A training set as the core opens it: a tracking-table ID or a set folder."""

Opened = tuple[AnnotationSession, Batch | None, dict[str, list[int]], int]
"""The session, its first batch, ``(done, total)`` per split, units needing review."""


def _summary_text(summary: TrainingSetSummary) -> str:
    where = f"table {summary.table_id}" if summary.table_id else summary.storage
    lines = [f"<b>{summary.name}</b> ({where})"]
    for category in CATEGORIES:
        done, total = summary.progress.get(category, (0, 0))
        if total:
            lines.append(f"{category.capitalize()}: {done}/{total}")
    for record in summary.sessions[-3:]:
        note = f" — {record.note}" if record.note else ""
        lines.append(
            f"Session {record.id} ({record.started:%Y-%m-%d}, {record.status}): "
            f"{record.n_annotated} annotated{note}"
        )
    return "<br>".join(lines)


def _describe(item: BatchItem) -> str:
    """One line for the dock: where the image comes from, and which plane."""
    unit = item.unit
    if unit is None:
        return item.annotation_id
    identity = unit.identity
    place = " / ".join(
        p for p in (identity.image_name, identity.well, identity.field) if p
    )
    return (
        f"{place} · t{unit.timepoint} c{unit.channel} z{unit.z_slice} [{unit.category}]"
    )


def _local_sets(home: Path) -> list[tuple[Ref, TrainingSetSummary]]:
    """Training sets with a folder under ``home`` (read-only)."""
    found: list[tuple[Ref, TrainingSetSummary]] = []
    for config in sorted(home.glob("*/*/config.yaml")):
        try:
            found.append((config.parent, summarize(config.parent)))
        except Exception:  # noqa: BLE001 - a broken folder must not hide the others
            logger.warning("Skipping %s: not a readable training set", config.parent)
    return found


class _Live:
    """State that outlives the Qt widget, so a destroyed dock still ends its session."""

    def __init__(self) -> None:
        self.session: AnnotationSession | None = None
        self.closed = False


def _close_dock(
    live: _Live,
    io: Worker,
    compute: Worker,
    viewer: Any,
    controller: Any,
    *_: Any,
) -> None:
    """End an open session as interrupted (releasing its claims), unbind the keys,
    remove micro-SAM's layers and stop the threads. Safe to call more than once and
    after the widget itself is gone (it touches no widget)."""
    if live.closed:
        return
    live.closed = True
    session, live.session = live.session, None
    if session is not None:
        future = io.run(
            lambda: session.finish(status="interrupted"), lambda _: None, lambda _: None
        )
        try:
            future.result(timeout=30)
        except Exception:  # noqa: BLE001 - closing; the next open closes it anyway
            logger.warning("Could not record the session as interrupted")
    for key in (ACCEPT_KEY, SKIP_KEY):
        with contextlib.suppress(Exception):  # the viewer may be closing too
            viewer.bind_key(key, None, overwrite=True)
    try:
        controller.teardown()
    except Exception:  # noqa: BLE001 - the viewer may be closing too
        logger.debug("micro-SAM teardown after close failed", exc_info=True)
    io.shutdown()
    compute.shutdown()


class OMEROAnnotateDock(QWidget):
    """Annotate an omero-annotate-ai training set with micro-SAM in this viewer."""

    def __init__(
        self,
        napari_viewer: napari.Viewer,
        *,
        home: Path | None = None,
        core_options: Mapping[str, Any] | None = None,
        controller: MicroSamController | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._home = home_dir(home)
        # Extra TrainingSet.open() arguments (tests pass an image source and fake OMERO).
        self._core_options = dict(core_options or {})
        self._controller = controller or MicroSamController(napari_viewer)
        self._io = Worker("oai-io")
        self._compute = Worker("oai-compute")
        self._embedder = embedder or Embedder(self._compute)
        self._bridge = EventBridge()
        self._bridge.event.connect(self._on_event)
        self._conn: Any = None
        self._sets: list[tuple[Ref, TrainingSetSummary]] = []
        self._cancel_requested = False

        self._live = _Live()
        self._batch: Batch | None = None
        self._cursor = 0
        self._committed: set[int] = set()
        # Images saved in earlier batches of this session (for Prev), with their model.
        # Images saved in earlier batches of this session, for Prev: (id, model).
        # Their pixels are read back from the set's folder when shown again.
        self._history: list[tuple[str, str]] = []
        self._revisit: str | None = None  # one of those, shown again
        self._unsaved: set[str] = set()  # accepted, but the save failed or conflicted
        self._counted: set[str] = set()  # images counted in the progress line
        self._loading = False  # the next batch is being loaded
        self._skipped: set[str] = set()
        self._done: dict[str, list[int]] = {}
        self._confirm_empty = False
        self._ready = False
        self._revised = 0
        self._finishing = False  # no Accept/next batch once Finish was pressed
        self._docked = False
        self._window: QWidget | None = None  # napari's window, watched for closing

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_start_page())
        self._pages.addWidget(self._build_annotate_page())
        self._pages.currentChanged.connect(self._fit_page)
        self._fit_page(0)
        layout = QVBoxLayout(self)
        layout.addWidget(self._pages)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        app = QApplication.instance()
        if app is not None:  # quitting napari mid-session records it as interrupted
            app.aboutToQuit.connect(self._interrupt)
        # Closing or removing the dock (napari deletes it) ends the session too.
        self.destroyed.connect(
            functools.partial(
                _close_dock,
                self._live,
                self._io,
                self._compute,
                napari_viewer,
                self._controller,
            )
        )
        self._refresh()

    @property
    def _session(self) -> AnnotationSession | None:
        return self._live.session

    @_session.setter
    def _session(self, session: AnnotationSession | None) -> None:
        self._live.session = session

    # --- layout -------------------------------------------------------------------

    def _build_start_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        connection = QGroupBox("OMERO server (not needed for local sets)")
        form = QFormLayout(connection)
        self.host = QLineEdit()
        self.user = QLineEdit()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        usable, backend_name, _ = SimpleOMEROConnection.keyring_backend()
        self.remember = QCheckBox("Remember password in keychain")
        self.remember.setEnabled(usable)
        self.remember.setToolTip(backend_name)
        history = SimpleOMEROConnection().load_connection_history()
        if history:
            self.host.setText(str(history[0].get("host", "")))
            self.user.setText(str(history[0].get("username", "")))
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._connect)
        form.addRow("Host", self.host)
        form.addRow("User", self.user)
        form.addRow("Password", self.password)
        form.addRow(self.remember)
        form.addRow(self.connect_button)
        self.container_type = QComboBox()
        self.container_type.addItems(CONTAINER_TYPES)
        self.container_id = QSpinBox()
        self.container_id.setRange(0, 2**31 - 1)
        self.list_button = QPushButton("List sets")
        self.list_button.setEnabled(False)
        self.list_button.clicked.connect(self._refresh)
        container = QHBoxLayout()
        for widget in (self.container_type, self.container_id, self.list_button):
            container.addWidget(widget)
        form.addRow("Container", container)
        layout.addWidget(connection)

        sets = QGroupBox("Training sets")
        sets_layout = QVBoxLayout(sets)
        self.set_list = QListWidget()
        self.set_list.currentRowChanged.connect(self._on_set_selected)
        self.summary = QLabel(
            "Sets with a folder in the home folder are listed here; connect to "
            "list the sets on an OMERO container."
        )
        self.summary.setWordWrap(True)
        buttons = QHBoxLayout()
        self.folder_button = QPushButton("Open folder…")
        self.folder_button.clicked.connect(self._choose_folder)
        self.open_button = QPushButton("Annotate")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_selected)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._request_cancel)
        for button in (self.folder_button, self.open_button, self.cancel_button):
            buttons.addWidget(button)
        self.review = QCheckBox("Review annotated images (correct their saved masks)")
        sets_layout.addWidget(self.set_list)
        sets_layout.addWidget(self.summary)
        sets_layout.addWidget(self.review)
        sets_layout.addLayout(buttons)
        layout.addWidget(sets)
        layout.addStretch()
        return page

    def _build_annotate_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.progress_label = QLabel()
        self.current_label = QLabel()
        self.current_label.setWordWrap(True)
        self.notice = QLabel()  # lasting warnings: needs review, claims, conflicts
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #d9534f;")
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setVisible(False)
        buttons = QHBoxLayout()
        self.prev_button = QPushButton("◀ Prev")
        self.accept_button = QPushButton("Accept [N]")
        self.skip_button = QPushButton("Skip [Shift-N]")
        self.prev_button.clicked.connect(self._prev)
        self.accept_button.clicked.connect(self._accept)
        self.skip_button.clicked.connect(self._skip)
        for button in (self.prev_button, self.accept_button, self.skip_button):
            button.setEnabled(False)
            buttons.addWidget(button)
        self.note = QLineEdit()
        self.note.setPlaceholderText(
            "Session note (optional), e.g. 'nuclei only, skipped small debris'"
        )
        self.finish_button = QPushButton("Finish session")
        self.finish_button.clicked.connect(self._finish)
        for widget in (self.progress_label, self.current_label, self.notice, self.busy):
            layout.addWidget(widget)
        layout.addLayout(buttons)
        layout.addWidget(self.note)
        layout.addWidget(self.finish_button)
        layout.addStretch()
        return page

    def _fit_page(self, index: int) -> None:
        """Size the dock to the shown page only: a stacked widget otherwise keeps
        the minimum height of its tallest page, so the short annotate page could not
        be shrunk to make room for micro-SAM's dock below it."""
        for i in range(self._pages.count()):
            policy = (
                QSizePolicy.Policy.Preferred
                if i == index
                else QSizePolicy.Policy.Ignored
            )
            self._pages.widget(i).setSizePolicy(policy, policy)
        self._pages.updateGeometry()

    # --- start page ---------------------------------------------------------------

    def _connect(self) -> None:
        host, user, password = (
            self.host.text().strip(),
            self.user.text().strip(),
            self.password.text(),
        )
        remember = self.remember.isChecked()
        if not (host and user and password):
            self._say("Fill in host, user and password.", error=True)
            return

        def connect() -> Any:
            omero = SimpleOMEROConnection()
            conn = omero.connect(host, user, password, verbose=False)
            if remember:
                omero.save_password(host, user, password)
            return conn

        def done(conn: Any) -> None:
            self._conn = conn
            self.connect_button.setEnabled(True)
            self.list_button.setEnabled(True)
            self.password.clear()
            self._say(f"Connected to {host}. Choose a container and list its sets.")

        self.connect_button.setEnabled(False)
        self._say(f"Connecting to {host}…")
        self._io.run(connect, done, self._on_error)

    def _refresh(self) -> None:
        conn, home = self._conn, self._home
        container = (self.container_type.currentText(), self.container_id.value())

        def list_sets() -> list[tuple[Ref, TrainingSetSummary]]:
            found = _local_sets(home)
            if conn is not None and container[1]:
                listed = list_training_sets(conn, *container)
                ids = {info.table_id for info in listed}
                found = [(r, s) for r, s in found if s.table_id not in ids]
                found += [(i.table_id, summarize(i.table_id, conn)) for i in listed]
            return found

        self._io.run(list_sets, self._show_sets, self._on_error)

    def _show_sets(self, sets: list[tuple[Ref, TrainingSetSummary]]) -> None:
        self._sets = sets
        self.set_list.clear()
        for _ref, summary in sets:
            done = sum(d for d, _ in summary.progress.values())
            total = sum(t for _, t in summary.progress.values())
            self.set_list.addItem(QListWidgetItem(f"{summary.name}  ({done}/{total})"))
        if sets:
            self.set_list.setCurrentRow(0)

    def _on_set_selected(self, row: int) -> None:
        valid = 0 <= row < len(self._sets)
        self.open_button.setEnabled(valid)
        if valid:
            self.summary.setText(_summary_text(self._sets[row][1]))

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Training set folder", str(self._home)
        )
        if folder:
            self._open(Path(folder), Path(folder).name)

    def _open_selected(self) -> None:
        row = self.set_list.currentRow()
        if 0 <= row < len(self._sets):
            ref, summary = self._sets[row]
            self._open(ref, summary.name)

    def _open(self, ref: Ref, name: str) -> None:
        conn, home, options = self._conn, self._home, self._core_options
        review = self.review.isChecked()
        try:
            self._controller.check_layers()  # before anything is claimed
        except LayerConflict as exc:
            self._say(str(exc), error=True)
            return
        self._cancel_requested = False
        self._finishing = False
        self.notice.clear()  # events from opening (claims held elsewhere) add to it

        def open_set() -> Opened:
            ts = TrainingSet.open(
                ref,
                conn,
                home=home,
                events=self._bridge.sink,
                cancel=lambda: self._cancel_requested,
                **options,
            )
            done = {c: [0, 0] for c in CATEGORIES}
            flagged = 0
            for row in ts.units():
                done[row.category][1] += 1
                done[row.category][0] += int(row.processed)
                flagged += (row.error or "").startswith(NEEDS_REVIEW)
            session = ts.start_session(review=review)
            try:
                batch = session.next_batch()
            except BaseException:  # incl. Cancelled: record it, release its claims
                session.finish(status="interrupted")
                raise
            return session, batch, done, flagged

        def opened(result: Opened) -> None:
            self._session, batch, self._done, flagged = result
            self._revised = 0
            self._set_start_controls(True)
            self.cancel_button.setVisible(False)
            self._skipped.clear()
            self._history, self._revisit = [], None
            self._unsaved, self._counted = set(), set()
            self._bind_keys(True)
            self._pages.setCurrentIndex(1)
            self._start_batch(batch)
            if flagged:
                self._notify(
                    f"{flagged} image(s) need review: the local mask differs from the "
                    "one on OMERO, so they are skipped (see Troubleshooting)."
                )

        self._set_start_controls(False)
        self.cancel_button.setVisible(True)
        self._say(f"Opening {name} (repairing it if a run was interrupted)…")
        self._io.run(open_set, opened, self._on_error)

    def _request_cancel(self) -> None:
        self._cancel_requested = True
        self._say("Cancelling…")

    # --- annotate page ------------------------------------------------------------

    def _start_batch(self, batch: Batch | None) -> None:
        self._loading = False
        if self._finishing:
            return  # Finish ends every open batch (and releases its claims)
        session = self._session
        if batch is not None and (
            batch.is_volumetric
            or all(i.annotation_id in self._skipped for i in batch.items)
        ):
            # 3D, or only skipped images left: hand the batch back and stop.
            leftover = batch
            if session is not None:
                self._io.run(
                    lambda: session.release(leftover), lambda _: None, self._on_error
                )
            if batch.is_volumetric:
                self._say(
                    "3D training sets are not supported in the plugin yet.", error=True
                )
                self._batch = None
                self._set_buttons(False)
                return
            batch = None
        self._batch, self._committed = batch, set()
        if batch is None:
            self._no_images_left()
            return
        first = next(
            (
                i
                for i, item in enumerate(batch.items)
                if item.annotation_id not in self._skipped
            ),
            0,
        )
        self._show(first)

    def _embedding_path(self, batch: Batch, index: int) -> Path:
        embeddings = batch.embedding_dir or batch.mask_dir / "embeddings"
        embeddings.mkdir(parents=True, exist_ok=True)
        return embeddings / f"embedding_{index:05}.zarr"

    def _saved_ids(self, batch: Batch, indices: Iterable[int]) -> list[str]:
        """The images among ``indices`` that are saved (their save did not fail)."""
        ids = [batch.items[i].annotation_id for i in indices]
        return [i for i in ids if i not in self._unsaved]

    def _show(self, index: int) -> None:
        """Show image ``index`` of the current batch; a saved one with its mask."""
        batch = self._batch
        if batch is None:
            return
        self._cursor, self._revisit = index, None
        item = batch.items[index]
        saved = index in self._committed
        self.current_label.setText(
            f"Image {index + 1}/{len(batch.items)}: {_describe(item)}"
            + (" (saved)" if saved else "")
        )
        path = self._embedding_path(batch, index)

        def current() -> bool:
            return (
                self._batch is batch and self._cursor == index and self._revisit is None
            )

        if saved:
            self._show_saved_unit(item.annotation_id, path, batch.model_type, current)
            return
        # A copy: the viewer's edits never touch the saved mask, which stays the
        # reference for "unchanged".
        start = (
            None if item.initial_mask is None else item.initial_mask.astype(np.uint32)
        )
        self._present(item.pixels, path, batch.model_type, start, False, current)
        self._prefetch(batch, index)

    def _show_earlier(self, annotation_id: str, model_type: str) -> None:
        """Show an image saved in an earlier batch of this session."""
        session = self._session
        if session is None:
            return
        self._revisit = annotation_id
        self.current_label.setText(f"Saved earlier: {annotation_id}")
        path = session.scratch / f"{annotation_id}.zarr"
        self._show_saved_unit(
            annotation_id, path, model_type, lambda: self._revisit == annotation_id
        )

    def _show_saved_unit(
        self,
        annotation_id: str,
        path: Path,
        model_type: str,
        current: Callable[[], bool],
    ) -> None:
        """Read a saved image and its mask from the set's folder (on the IO thread,
        so after its save), then show them."""
        session = self._session
        if session is None:
            return
        self._wait(None)

        def loaded(unit: tuple[np.ndarray, np.ndarray]) -> None:
            if self._finishing or not current():
                return
            pixels, mask = unit
            self._present(
                pixels, path, model_type, mask.astype(np.uint32), True, current
            )

        self._io.run(
            lambda: session.saved_unit(annotation_id), loaded, self._on_image_error
        )

    def _wait(self, pixels: np.ndarray | None) -> None:
        """No Accept/Skip/Prev until the image is ready; the viewer shows it dimmed."""
        self._confirm_empty, self._ready = False, False
        self._update_progress()
        self._set_buttons(False)
        self._controller.hold(pixels)
        self.busy.setVisible(True)

    def _present(
        self,
        pixels: np.ndarray,
        path: Path,
        model_type: str,
        start: np.ndarray | None,
        saved: bool,
        current: Callable[[], bool],
    ) -> None:
        """Compute (or load) the embedding of ``pixels``, then hand it to micro-SAM
        with the objects ``start``; ``current()`` says whether the user is still on
        this image by then."""
        self._wait(pixels)
        self._say("Computing the image embedding…")

        def ready(embedding: Embedding) -> None:
            if self._finishing or not current():
                return  # the user moved on (or finished) meanwhile
            self.busy.setVisible(False)
            try:
                self._controller.show(pixels, embedding, model_type, start)
            except Exception as exc:  # noqa: BLE001 - shown to the user, logged
                self._on_image_error(exc)
                return
            self._ready = True
            self._set_buttons(True)
            self._say(
                "Correct the saved objects, then Accept (unchanged masks stay as they are)."
                if saved or self._reviewing
                else "Segment with micro-SAM, commit objects, then Accept."
            )

        self._embedder.compute(pixels, path, model_type, ready, self._on_image_error)

    def _prefetch(self, batch: Batch, index: int) -> None:
        """Compute the next image's embedding while this one is annotated; showing it
        then only loads it. Errors surface when it is shown."""
        following = self._todo(batch, index)
        if following:
            nxt = following[0]
            self._embedder.compute(
                batch.items[nxt].pixels,
                self._embedding_path(batch, nxt),
                batch.model_type,
                lambda _embedding: None,
                lambda _exc: None,
            )

    def _todo(self, batch: Batch, after: int) -> list[int]:
        """Images of ``batch`` after position ``after`` not yet saved or skipped."""
        return [
            i
            for i in range(after + 1, len(batch.items))
            if i not in self._committed
            and batch.items[i].annotation_id not in self._skipped
        ]

    def _accept(self) -> None:
        batch, session = self._batch, self._session
        if session is None or not self._ready or self._finishing or self._loading:
            return
        if self._revisit is None and batch is None:
            return
        mask = self._controller.current_mask()
        if not mask.any() and not self._confirm_empty:
            self._confirm_empty = True
            self._say(
                "No objects committed. Press Accept again to save this image as empty."
            )
            return
        if self._revisit is not None:
            self._revise(session, self._revisit, mask)
            self._resume()
            return
        assert batch is not None
        index = self._cursor
        item = batch.items[index]
        if index in self._committed:  # saved earlier in this batch: replace it
            self._revise(session, item.annotation_id, mask)
            self._advance()
            return
        path = batch.mask_dir / f"seg_{index:05}.tif"
        initial = item.initial_mask
        if initial is not None and np.array_equal(mask, initial):
            mask = initial  # unchanged: same dtype too, so the review keeps the upload
        else:
            mask = mask.astype(np.uint32)
        tifffile.imwrite(path, mask, compression="zlib")
        self._committed.add(index)
        self._skipped.discard(item.annotation_id)
        result = {item.annotation_id: path}
        # The OMERO config is replaced once, at finish, not after every image.
        self._io.run(
            lambda: session.commit(batch, result, publish_config=False),
            lambda report: self._unsaved.update(report.failed, report.conflicts),
            self._on_error,
        )
        self._advance()

    def _revise(
        self, session: AnnotationSession, annotation_id: str, mask: np.ndarray
    ) -> None:
        """Save corrected objects of an image saved earlier in this session."""

        def revise() -> str:
            # The session's folder is inside the set's folder: the mask moves into
            # place on the same filesystem.
            path = session.scratch / f"{annotation_id}_mask.tif"
            tifffile.imwrite(path, mask.astype(np.uint32), compression="zlib")
            return session.revise(annotation_id, path, publish_config=False)

        def revised(outcome: str) -> None:
            if outcome == "failed":
                self._notify(
                    f"The correction of {annotation_id} could not be uploaded; "
                    "the earlier mask is kept."
                )
            if outcome in ("failed", "conflict"):
                self._unsaved.add(annotation_id)

        self._io.run(revise, revised, self._on_error)

    def _skip(self) -> None:
        # Also without a shown image: an image whose embedding failed can be skipped.
        if self._finishing or self._loading:
            return
        if self._revisit is not None:  # leave the saved image as it is
            self._resume()
            return
        if self._batch is None:
            return
        if self._cursor not in self._committed:
            self._skipped.add(self._batch.items[self._cursor].annotation_id)
        self._advance()

    def _prev(self) -> None:
        """Back one image: within the batch (saved or skipped), then through the
        images saved in earlier batches of this session."""
        if self._finishing or self._loading or self._session is None:
            return
        batch = self._batch
        if self._revisit is None and batch is not None:
            earlier = [
                i
                for i in range(self._cursor)
                if batch.items[i].annotation_id not in self._unsaved
            ]
            if earlier:
                self._show(earlier[-1])
                return
        saved = [e for e in self._history if e[0] not in self._unsaved]
        ids = [annotation_id for annotation_id, _ in saved]
        if self._revisit is not None and self._revisit in ids:
            saved = saved[: ids.index(self._revisit)]
        if saved:
            self._show_earlier(*saved[-1])

    def _resume(self) -> None:
        """From an image of an earlier batch back to the current batch."""
        self._revisit = None
        if self._batch is None:
            self._no_images_left()
            return
        self._cursor = -1
        self._advance()

    def _no_images_left(self) -> None:
        self._ready = False
        self._set_buttons(False)
        self.prev_button.setEnabled(bool(self._history))
        self.current_label.setText("No images left in this session.")
        self._say(
            "All images are done or skipped. Prev goes back to saved images; "
            "add a note and finish the session."
        )

    def _advance(self) -> None:
        batch, session = self._batch, self._session
        if batch is None or session is None or self._finishing or self._loading:
            return
        todo = self._todo(batch, self._cursor)
        if todo:
            self._show(todo[0])
            return
        saved = self._saved_ids(batch, sorted(self._committed))
        self._history += [(annotation_id, batch.model_type) for annotation_id in saved]

        def next_batch() -> Batch | None:
            session.release(batch)
            return session.next_batch()

        self._loading = True
        self._ready = False  # no Accept/Skip (also by key) until the next image shows
        self._set_buttons(False)
        self._controller.hold(None)
        self._io.run(next_batch, self._start_batch, self._on_error)

    def _finish(self) -> None:
        session = self._session
        if session is None:
            return
        note = self.note.text().strip() or None
        batch = self._batch
        self._finishing, self._ready = True, False
        self._bind_keys(False)

        def finish() -> int:
            if batch is not None:
                session.release(batch)
            return session.finish(note)

        def finished(annotated: int) -> None:
            reviewed = self._reviewing
            self._finishing = False
            self._session = self._batch = None
            self._history, self._revisit = [], None
            self._bind_keys(False)
            self._controller.teardown()
            self.note.clear()
            self._pages.setCurrentIndex(0)
            done = "revised" if reviewed else "annotated"
            self._say(f"Session finished: {annotated} image(s) {done}.")
            self._refresh()

        def failed(exc: BaseException) -> None:
            self._finishing = False  # let the user try Finish again
            self._bind_keys(True)
            self._on_error(exc)

        self._set_buttons(False)
        self.finish_button.setEnabled(False)
        self._io.run(finish, finished, failed)

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        """Once docked, watch napari's window: closing it interrupts the session."""
        window = self.window()
        if window is not self and window is not self._window:
            window.installEventFilter(self)
            self._window = window
        super().showEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Closing the viewer window (napari keeps running, e.g. from Jupyter)
        records the session as interrupted and releases its claims - once the close
        really happened: napari may ask for confirmation, and the user may cancel."""
        if watched is self._window:
            kind = event.type()
            if kind == QEvent.Type.Close:
                QTimer.singleShot(0, self._interrupt_if_window_closed)
            elif kind == QEvent.Type.Hide and not event.spontaneous():
                self._interrupt_if_window_closed()  # hidden by close(), not minimised
        return super().eventFilter(watched, event)

    def _interrupt_if_window_closed(self) -> None:
        if self._window is not None and not self._window.isVisible():
            self._interrupt()

    def event(self, event: QEvent) -> bool:
        """Taken out of napari (dock removed): end the session and stop."""
        if event.type() == QEvent.Type.ParentChange:
            if self.parent() is not None:
                self._docked = True
            elif self._docked:
                self._shutdown()
        return super().event(event)

    def _shutdown(self) -> None:
        _close_dock(self._live, self._io, self._compute, self._viewer, self._controller)

    def _interrupt(self) -> None:
        """Record a session still open when napari closes as interrupted.

        Finishing releases the claims of its open batches, so other computers (and
        this one) can take those images at once.
        """
        session = self._session
        if session is None:
            return
        self._session = None
        future = self._io.run(
            lambda: session.finish(status="interrupted"),
            lambda _: None,
            self._on_error,
        )
        try:
            future.result(timeout=30)
        except Exception:  # noqa: BLE001 - quitting; the next open closes it anyway
            logger.warning("Could not record the session as interrupted")

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        """Record an open session as interrupted, then stop the background threads."""
        self._shutdown()
        super().closeEvent(event)

    # --- helpers ------------------------------------------------------------------

    @property
    def _reviewing(self) -> bool:
        return self._session is not None and self._session.review

    def _on_event(self, event: Event) -> None:
        if isinstance(event, UnitCommitted):
            if event.annotation_id in self._counted:
                return  # corrected again with Prev: counted already
            self._counted.add(event.annotation_id)
            if self._reviewing:
                self._revised += 1
            else:
                category = self._category_of(event.annotation_id)
                if category:
                    self._done[category][0] += 1
            self._update_progress()
        elif isinstance(event, UnitConflict):
            self._notify(
                f"{event.annotation_id} was saved on another computer first; yours is "
                f"kept as {event.mask_file}"
            )
        elif isinstance(event, ClaimsHeld):
            self._notify(
                f"{event.units} image(s) are being annotated on another computer "
                "and are skipped."
            )
        elif isinstance(event, ClaimsReleased):
            self._say(f"Released {event.units} image(s) an earlier run left claimed.")
        elif isinstance(event, UploadFailed):
            self._say(
                f"Upload failed for {event.annotation_id}: {event.error}", error=True
            )
        elif isinstance(event, ReconcileAction):
            self._say(f"Repaired {event.annotation_id}: {event.action} {event.detail}")

    def _category_of(self, annotation_id: str) -> str | None:
        if self._batch is None:
            return None
        for item in self._batch.items:
            if item.annotation_id == annotation_id and item.unit is not None:
                return item.unit.category
        return None

    def _update_progress(self) -> None:
        parts = [f"{c.capitalize()} {d}/{t}" for c, (d, t) in self._done.items() if t]
        if self._reviewing:
            parts.append(f"revised {self._revised}")
        self.progress_label.setText(" · ".join(parts))

    def _set_buttons(self, enabled: bool) -> None:
        for button in (self.accept_button, self.skip_button, self.prev_button):
            button.setEnabled(enabled)
        self.finish_button.setEnabled(self._session is not None)

    def _bind_keys(self, on: bool) -> None:
        self._viewer.bind_key(
            ACCEPT_KEY, (lambda _v: self._accept()) if on else None, overwrite=True
        )
        self._viewer.bind_key(
            SKIP_KEY, (lambda _v: self._skip()) if on else None, overwrite=True
        )

    def _on_error(self, exc: BaseException) -> None:
        self.busy.setVisible(False)
        self.cancel_button.setVisible(False)
        self.connect_button.setEnabled(True)
        self._set_start_controls(True)
        self._set_buttons(self._ready)
        self._say(f"{type(exc).__name__}: {exc}", error=True)

    def _on_image_error(self, exc: BaseException) -> None:
        """The image could not be shown: Skip moves on, Prev goes back."""
        self._on_error(exc)
        if self._session is not None and not self._finishing and not self._loading:
            self.skip_button.setEnabled(True)
            self.prev_button.setEnabled(True)

    def _set_start_controls(self, enabled: bool) -> None:
        self.open_button.setEnabled(enabled and bool(self._sets))
        self.folder_button.setEnabled(enabled)
        self.review.setEnabled(enabled)

    def _notify(self, text: str) -> None:
        """Add a lasting warning to the annotate page (status messages get replaced)."""
        lines = [line for line in self.notice.text().split("<br>") if line]
        self.notice.setText("<br>".join([*lines, html.escape(text)]))

    def _say(self, text: str, *, error: bool = False) -> None:
        self.status.setStyleSheet("color: #d9534f;" if error else "")
        self.status.setText(text)
