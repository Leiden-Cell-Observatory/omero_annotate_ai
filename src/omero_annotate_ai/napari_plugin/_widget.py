# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
# (conda's PyQt5 ships no type stubs, so Qt classes are unknown to pyright.)
"""The plugin's dock widget: connect, pick a training set, annotate, finish.

No ``from __future__ import annotations`` here: npe2 reads the ``napari.Viewer``
annotation of ``__init__`` at runtime to inject the viewer.
"""

import logging
import tempfile
from pathlib import Path

import napari
import numpy as np
import tifffile
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..omero.simple_connection import SimpleOMEROConnection
from ._controller import MicroSamController
from ._core import (
    CATEGORIES,
    AnnotatorResult,
    Backend,
    Batch,
    Event,
    FakeBackend,
    Session,
    SetSummary,
    UnitCommitted,
    series_mask_name,
)
from ._workers import Embedder, Embedding, EventBridge, Worker

logger = logging.getLogger(__name__)

ACCEPT_KEY = "n"
SKIP_KEY = "Shift-N"


def _summary_text(summary: SetSummary) -> str:
    lines = [f"<b>{summary.name}</b> ({summary.storage}, model {summary.model})"]
    for category in CATEGORIES:
        done, total = summary.counts.get(category, (0, 0))
        if total:
            lines.append(f"{category.capitalize()}: {done}/{total}")
    for record in summary.sessions[-3:]:
        note = f" — {record.note}" if record.note else ""
        lines.append(
            f"Session {record.id} ({record.started[:10]}, {record.status}): "
            f"{record.n_annotated} annotated{note}"
        )
    return "<br>".join(lines)


class OMEROAnnotateDock(QWidget):
    """Annotate an omero-annotate-ai training set with micro-SAM in this viewer."""

    def __init__(
        self,
        napari_viewer: napari.Viewer,
        *,
        backend: Backend | None = None,
        controller: MicroSamController | None = None,
        embedder: Embedder | None = None,
        demo_root: Path | None = None,
    ) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._backend = backend
        self._controller = controller or MicroSamController(napari_viewer)
        self._io = Worker("oai-io")
        self._embedder = embedder or Embedder(Worker("oai-compute"))
        self._bridge = EventBridge()
        self._bridge.event.connect(self._on_event)
        self._demo_root = demo_root
        self._conn: object | None = None
        self._summaries: list[SetSummary] = []

        self._session: Session | None = None
        self._batch: Batch | None = None
        self._cursor = 0
        self._committed: set[int] = set()
        self._skipped: set[str] = set()
        self._done: dict[str, list[int]] = {}
        self._confirm_empty = False
        self._ready = False

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_start_page())
        self._pages.addWidget(self._build_annotate_page())
        layout = QVBoxLayout(self)
        layout.addWidget(self._pages)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    # --- layout -------------------------------------------------------------------

    def _build_start_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        connection = QGroupBox("OMERO server")
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
        layout.addWidget(connection)

        sets = QGroupBox("Training sets")
        sets_layout = QVBoxLayout(sets)
        self.set_list = QListWidget()
        self.set_list.currentRowChanged.connect(self._on_set_selected)
        self.summary = QLabel("Connect, or open the demo training set.")
        self.summary.setWordWrap(True)
        buttons = QHBoxLayout()
        self.demo_button = QPushButton("Open demo set")
        self.demo_button.clicked.connect(self._use_demo)
        self.open_button = QPushButton("Annotate")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_selected)
        buttons.addWidget(self.demo_button)
        buttons.addWidget(self.open_button)
        sets_layout.addWidget(self.set_list)
        sets_layout.addWidget(self.summary)
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
        for widget in (self.progress_label, self.current_label, self.busy):
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

        def connect() -> object | None:
            omero = SimpleOMEROConnection()
            conn = omero.connect(host, user, password)
            if conn is not None and remember:
                omero.save_password(host, user, password)
            return conn

        def done(conn: object | None) -> None:
            self.connect_button.setEnabled(True)
            if conn is None:
                self._say(
                    f"Could not connect to {host}. Check host, user and password.",
                    error=True,
                )
                return
            self._conn = conn
            self.password.clear()
            self._say(f"Connected to {host} as {user}.")
            self.summary.setText(
                "Training sets on OMERO open here once the 0.5.0 core is released. "
                "Until then, use the demo set."
            )

        self.connect_button.setEnabled(False)
        self._say(f"Connecting to {host}…")
        self._io.run(connect, done, self._on_error)

    def _use_demo(self) -> None:
        root = self._demo_root or Path(
            tempfile.mkdtemp(prefix="omero-annotate-ai-demo-")
        )
        self._backend = FakeBackend(root)
        self._refresh()

    def _refresh(self) -> None:
        backend = self._backend
        if backend is None:
            return
        self._io.run(backend.list_sets, self._show_sets, self._on_error)

    def _show_sets(self, summaries: list[SetSummary]) -> None:
        self._summaries = summaries
        self.set_list.clear()
        for summary in summaries:
            done = sum(d for d, _ in summary.counts.values())
            total = sum(t for _, t in summary.counts.values())
            self.set_list.addItem(QListWidgetItem(f"{summary.name}  ({done}/{total})"))
        if summaries:
            self.set_list.setCurrentRow(0)

    def _on_set_selected(self, row: int) -> None:
        valid = 0 <= row < len(self._summaries)
        self.open_button.setEnabled(valid)
        if valid:
            self.summary.setText(_summary_text(self._summaries[row]))

    def _open_selected(self) -> None:
        backend, row = self._backend, self.set_list.currentRow()
        if backend is None or not 0 <= row < len(self._summaries):
            return
        summary = self._summaries[row]

        def open_set() -> tuple[Session, Batch | None]:
            session = backend.open(summary.key, self._bridge.sink).start_session()
            return session, session.next_batch()

        def opened(result: tuple[Session, Batch | None]) -> None:
            self._session, batch = result
            self._done = {c: list(summary.counts.get(c, (0, 0))) for c in CATEGORIES}
            self._skipped.clear()
            self._bind_keys(True)
            self._pages.setCurrentIndex(1)
            self._start_batch(batch)

        self.open_button.setEnabled(False)
        self._say(f"Opening {summary.name}…")
        self._io.run(open_set, opened, self._on_error)

    # --- annotate page ------------------------------------------------------------

    def _start_batch(self, batch: Batch | None) -> None:
        if batch is not None and all(
            i.unit.annotation_id in self._skipped for i in batch.items
        ):
            # Only skipped images are left: hand them back and stop.
            session, leftover = self._session, batch
            if session is not None:
                self._io.run(
                    lambda: session.release(leftover), lambda _: None, self._on_error
                )
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
                if item.unit.annotation_id not in self._skipped
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
            f"Image {index + 1}/{len(batch.items)}: {item.unit.describe()}"
        )
        self._update_progress()
        self._set_buttons(False)
        self.busy.setVisible(True)
        self._say("Computing the image embedding…")
        path = batch.embedding_dir / f"embedding_{index:05}.zarr"

        def ready(embedding: Embedding) -> None:
            if self._batch is not batch or self._cursor != index:
                return  # the user moved on meanwhile
            self.busy.setVisible(False)
            try:
                self._controller.show(item.pixels, embedding, batch.model_type)
            except Exception as exc:  # noqa: BLE001 - shown to the user, logged
                self._on_error(exc)
                return
            self._ready = True
            self._set_buttons(True)
            self._say("Segment with micro-SAM, commit objects, then Accept.")

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
        path = batch.mask_dir / series_mask_name(index)
        tifffile.imwrite(path, mask.astype(np.uint32), compression="zlib")
        self._committed.add(index)
        self._skipped.discard(item.unit.annotation_id)
        result = AnnotatorResult({item.unit.annotation_id: path})
        self._io.run(
            lambda: session.commit(batch, result), lambda _: None, self._on_error
        )
        self._advance()

    def _skip(self) -> None:
        if self._batch is None or not self._ready:
            return
        self._skipped.add(self._batch.items[self._cursor].unit.annotation_id)
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
            and batch.items[i].unit.annotation_id not in self._skipped
        ]
        if todo:
            self._show(todo[0])
            return

        def next_batch() -> Batch | None:
            session.release(batch)
            return session.next_batch()

        self._set_buttons(False)
        self._io.run(next_batch, self._start_batch, self._on_error)

    def _finish(self) -> None:
        session = self._session
        if session is None:
            return
        note = self.note.text().strip()
        batch = self._batch

        def finish() -> object:
            if batch is not None:
                session.release(batch)
            return session.finish(note)

        def finished(_: object) -> None:
            self._session = self._batch = None
            self._bind_keys(False)
            self._controller.teardown()
            self.note.clear()
            self._pages.setCurrentIndex(0)
            self._say("Session finished.")
            self._refresh()

        self._set_buttons(False)
        self.finish_button.setEnabled(False)
        self._io.run(finish, finished, self._on_error)

    # --- helpers ------------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        if isinstance(event, UnitCommitted):
            category = self._category_of(event.annotation_id)
            if category:
                self._done[category][0] += 1
            self._update_progress()
        else:
            self._say(
                f"Upload failed for {event.annotation_id}: {event.error}", error=True
            )

    def _category_of(self, annotation_id: str) -> str | None:
        if self._batch is None:
            return None
        for item in self._batch.items:
            if item.unit.annotation_id == annotation_id:
                return item.unit.category
        return None

    def _update_progress(self) -> None:
        parts = [f"{c.capitalize()} {d}/{t}" for c, (d, t) in self._done.items() if t]
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
        self.connect_button.setEnabled(True)
        self.open_button.setEnabled(bool(self._summaries))
        self._set_buttons(self._ready)
        self._say(f"{type(exc).__name__}: {exc}", error=True)

    def _say(self, text: str, *, error: bool = False) -> None:
        self.status.setStyleSheet("color: #d9534f;" if error else "")
        self.status.setText(text)
