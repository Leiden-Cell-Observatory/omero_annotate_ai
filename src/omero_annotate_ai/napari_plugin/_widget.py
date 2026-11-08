# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
# (conda's PyQt5 ships no type stubs, so Qt classes are unknown to pyright.)
"""The plugin's dock widget: connect, pick a training set, annotate, finish.

Every call into the core (connect, list, open, next batch, commit, finish) runs on
one IO thread, because a BlitzGateway must be used from one thread at a time.

No ``from __future__ import annotations`` here: npe2 reads the ``napari.Viewer``
annotation of ``__init__`` at runtime to inject the viewer.
"""

import html
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import napari
import numpy as np
import tifffile
from qtpy.QtCore import QEvent, QObject
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
from ._controller import MicroSamController
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

        self._session: AnnotationSession | None = None
        self._batch: Batch | None = None
        self._cursor = 0
        self._committed: set[int] = set()
        self._skipped: set[str] = set()
        self._done: dict[str, list[int]] = {}
        self._confirm_empty = False
        self._ready = False
        self._revised = 0
        self._window: QWidget | None = None  # napari's window, watched for closing

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_start_page())
        self._pages.addWidget(self._build_annotate_page())
        layout = QVBoxLayout(self)
        layout.addWidget(self._pages)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        app = QApplication.instance()
        if app is not None:  # quitting napari mid-session records it as interrupted
            app.aboutToQuit.connect(self._interrupt)
        self._refresh()

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
        self._cancel_requested = False
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
            return session, session.next_batch(), done, flagged

        def opened(result: Opened) -> None:
            self._session, batch, self._done, flagged = result
            self._revised = 0
            self.cancel_button.setVisible(False)
            self._skipped.clear()
            self._bind_keys(True)
            self._pages.setCurrentIndex(1)
            self._start_batch(batch)
            if flagged:
                self._notify(
                    f"{flagged} image(s) need review: the local mask differs from the "
                    "one on OMERO, so they are skipped (see Troubleshooting)."
                )

        self.open_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self._say(f"Opening {name} (repairing it if a run was interrupted)…")
        self._io.run(open_set, opened, self._on_error)

    def _request_cancel(self) -> None:
        self._cancel_requested = True
        self._say("Cancelling…")

    # --- annotate page ------------------------------------------------------------

    def _start_batch(self, batch: Batch | None) -> None:
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
            self._set_buttons(False)
            self.current_label.setText("No images left in this session.")
            self._say(
                "All images are done or skipped. Add a note and finish the session."
            )
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

    def _show(self, index: int) -> None:
        batch = self._batch
        if batch is None:
            return
        self._cursor, self._confirm_empty, self._ready = index, False, False
        item = batch.items[index]
        self.current_label.setText(
            f"Image {index + 1}/{len(batch.items)}: {_describe(item)}"
        )
        self._update_progress()
        self._set_buttons(False)
        self.busy.setVisible(True)
        self._say("Computing the image embedding…")
        embeddings = batch.embedding_dir or batch.mask_dir / "embeddings"
        embeddings.mkdir(parents=True, exist_ok=True)
        path = embeddings / f"embedding_{index:05}.zarr"

        def ready(embedding: Embedding) -> None:
            if self._batch is not batch or self._cursor != index:
                return  # the user moved on meanwhile
            self.busy.setVisible(False)
            try:
                self._controller.show(
                    item.pixels, embedding, batch.model_type, item.initial_mask
                )
            except Exception as exc:  # noqa: BLE001 - shown to the user, logged
                self._on_error(exc)
                return
            self._ready = True
            self._set_buttons(True)
            self._say(
                "Correct the saved objects, then Accept (unchanged masks stay as they are)."
                if self._reviewing
                else "Segment with micro-SAM, commit objects, then Accept."
            )

        self._embedder.compute(
            item.pixels, path, batch.model_type, ready, self._on_error
        )

    def _accept(self) -> None:
        batch, session = self._batch, self._session
        if batch is None or session is None or not self._ready:
            return
        mask = self._controller.current_mask()
        if not mask.any() and not self._confirm_empty:
            self._confirm_empty = True
            self._say(
                "No objects committed. Press Accept again to save this image as empty."
            )
            return
        index = self._cursor
        item = batch.items[index]
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
            lambda _: None,
            self._on_error,
        )
        self._advance()

    def _skip(self) -> None:
        if self._batch is None or not self._ready:
            return
        self._skipped.add(self._batch.items[self._cursor].annotation_id)
        self._advance()

    def _prev(self) -> None:
        if self._batch is None:
            return
        earlier = [i for i in range(self._cursor) if i not in self._committed]
        if earlier:
            self._show(earlier[-1])

    def _advance(self) -> None:
        batch, session = self._batch, self._session
        if batch is None or session is None:
            return
        todo = [
            i
            for i in range(self._cursor + 1, len(batch.items))
            if i not in self._committed
            and batch.items[i].annotation_id not in self._skipped
        ]
        if todo:
            self._show(todo[0])
            return

        def next_batch() -> Batch | None:
            session.release(batch)
            return session.next_batch()

        self._ready = False  # no Accept/Skip (also by key) until the next image shows
        self._set_buttons(False)
        self._io.run(next_batch, self._start_batch, self._on_error)

    def _finish(self) -> None:
        session = self._session
        if session is None:
            return
        note = self.note.text().strip() or None
        batch = self._batch

        def finish() -> int:
            if batch is not None:
                session.release(batch)
            return session.finish(note)

        def finished(annotated: int) -> None:
            reviewed = self._reviewing
            self._session = self._batch = None
            self._bind_keys(False)
            self._controller.teardown()
            self.note.clear()
            self._pages.setCurrentIndex(0)
            done = "revised" if reviewed else "annotated"
            self._say(f"Session finished: {annotated} image(s) {done}.")
            self._refresh()

        self._set_buttons(False)
        self.finish_button.setEnabled(False)
        self._io.run(finish, finished, self._on_error)

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        """Once docked, watch napari's window: closing it interrupts the session."""
        window = self.window()
        if window is not self and window is not self._window:
            window.installEventFilter(self)
            self._window = window
        super().showEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Closing the viewer window (napari keeps running, e.g. from Jupyter) or
        quitting records the session as interrupted and releases its claims."""
        if watched is self._window and event.type() == QEvent.Type.Close:
            self._interrupt()
        return super().eventFilter(watched, event)

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
        self._interrupt()
        self._io.shutdown()
        self._compute.shutdown()
        super().closeEvent(event)

    # --- helpers ------------------------------------------------------------------

    @property
    def _reviewing(self) -> bool:
        return self._session is not None and self._session.review

    def _on_event(self, event: Event) -> None:
        if isinstance(event, UnitCommitted):
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
        self.open_button.setEnabled(bool(self._sets))
        self._set_buttons(self._ready)
        self._say(f"{type(exc).__name__}: {exc}", error=True)

    def _notify(self, text: str) -> None:
        """Add a lasting warning to the annotate page (status messages get replaced)."""
        lines = [line for line in self.notice.text().split("<br>") if line]
        self.notice.setText("<br>".join([*lines, html.escape(text)]))

    def _say(self, text: str, *, error: bool = False) -> None:
        self.status.setStyleSheet("color: #d9534f;" if error else "")
        self.status.setText(text)
