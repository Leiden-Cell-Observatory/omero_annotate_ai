"""Jupyter widget to create or open a training set and follow its annotation.

The widget only holds form state and the OMERO connection. It turns the form into a
request (:class:`NewTrainingSet` or :class:`OpenTrainingSet`) that
:func:`open_or_create` resolves to a :class:`~omero_annotate_ai.trainingset.TrainingSet`.
Annotation progress arrives through :meth:`WorkflowWidget.events`.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import display

from ..annotators import Annotator
from ..bundle import home_dir
from ..core.annotation_config import AnnotationConfig, create_default_config
from ..errors import OmeroAnnotateAIError
from ..events import (
    BatchPrepared,
    ClaimsHeld,
    ClaimsReleased,
    Event,
    Progress,
    ReconcileAction,
    RunCancelled,
    SessionFinished,
    SessionStarted,
    UnitCommitted,
    UnitConflict,
    UploadFailed,
)
from ..omero.omero_utils import get_container_info
from ..planning import Mode, parse_indices
from ..trainingset import (
    TrainingSet,
    TrainingSetSummary,
    summarize,
    training_set_choices,
)

CONTAINER_TYPES = ("dataset", "project", "plate", "screen")
MODELS = (
    ("vit_b_lm (light microscopy, recommended)", "vit_b_lm"),
    ("vit_l_lm", "vit_l_lm"),
    ("vit_b_em_organelles", "vit_b_em_organelles"),
    ("vit_b", "vit_b"),
    ("vit_l", "vit_l"),
    ("vit_h", "vit_h"),
)
SPLITS = (("training", "train_n"), ("validation", "validate_n"), ("test", "test_n"))
LOG_LINES = 12

_STYLE = """
<style>
.oai-card {border: 1px solid var(--jp-border-color2, #d0d7de); border-radius: 8px;
  padding: 8px 12px; margin: 4px 0;}
.oai-title {font-weight: 600; font-size: 1.05em; margin: 0 0 4px 0;}
.oai-muted {opacity: 0.75; font-size: 0.92em;}
.oai-error {color: var(--jp-error-color1, #d1242f);}
.oai-table td {padding: 1px 10px 1px 0;}
</style>
"""

_LABEL = {"description_width": "110px"}


def _wide() -> widgets.Layout:
    return widgets.Layout(width="420px")


def _narrow() -> widgets.Layout:
    return widgets.Layout(width="200px")


@dataclass(frozen=True)
class NewTrainingSet:
    """Create a training set from ``config``."""

    config: AnnotationConfig


@dataclass(frozen=True)
class OpenTrainingSet:
    """Open the set ``ref`` (table ID or folder) and add images per split."""

    ref: int | Path
    add: dict[str, int]


Request = NewTrainingSet | OpenTrainingSet


def open_or_create(
    request: Request,
    conn: Any = None,
    *,
    home: str | Path | None = None,
    events: Callable[[Event], None] | None = None,
    **kwargs: Any,
) -> TrainingSet:
    """Resolve a widget request to a training set.

    ``kwargs`` go to :meth:`TrainingSet.create` / :meth:`TrainingSet.open` (e.g. an
    ``image_source``). For an open request, ``request.add`` images per split are
    added to the set's current counts (zero adds nothing).
    """
    if isinstance(request, NewTrainingSet):
        return TrainingSet.create(
            request.config, conn, home=home, events=events, **kwargs
        )
    ts = TrainingSet.open(request.ref, conn, home=home, events=events, **kwargs)
    if any(request.add.values()):
        current = ts.config.training
        ts.extend(
            **{
                key: getattr(current, key) + request.add.get(category, 0)
                for category, key in SPLITS
            }
        )
    return ts


class _AxisPicker:
    """Which timepoints or z-slices to use: listed ones, all, or a random draw."""

    def __init__(self, what: str) -> None:
        self.what = what
        self.mode = widgets.Dropdown(
            options=[("Specific", "specific"), ("All", "all"), ("Random", "random")],
            value="specific",
            description=what,
            style=_LABEL,
            layout=_narrow(),
        )
        self.listed = widgets.Text(
            value="0",
            placeholder="0, 3, 5-7",
            layout=widgets.Layout(width="130px"),
        )
        self.n_random = widgets.BoundedIntText(
            value=1,
            min=1,
            max=100_000,
            description="n",
            style={"description_width": "20px"},
            layout=widgets.Layout(width="110px"),
        )
        self.mode.observe(lambda _: self._sync(), names="value")
        self.box = widgets.HBox([self.mode, self.listed, self.n_random])
        self._sync()

    def _sync(self) -> None:
        self.listed.disabled = self.mode.value != "specific"
        self.n_random.disabled = self.mode.value != "random"

    def values(self) -> tuple[Mode, list[int], int | None]:
        """``(mode, listed indices, n to draw)`` for :class:`SpatialCoverage`."""
        mode: Mode = self.mode.value
        listed = (
            parse_indices(self.listed.value, self.what) if mode == "specific" else [0]
        )
        return mode, listed, int(self.n_random.value) if mode == "random" else None


def _card(title: str, *children: widgets.Widget) -> widgets.VBox:
    header = widgets.HTML(f"<div class='oai-title'>{html.escape(title)}</div>")
    card = widgets.VBox([header, *children])
    card.add_class("oai-card")
    return card


def container_options(
    connection: Any, container_type: Any, on_error: Callable[[str], None]
) -> list[tuple[str, int]]:
    """``(label, id)`` of the containers of one type; an OMERO error is reported."""
    kind = str(container_type).capitalize()
    try:
        found = list(connection.getObjects(kind))
    except Exception as error:  # noqa: BLE001 - UI boundary: show any OMERO error
        on_error(f"Could not list {kind}s: {error}")
        return []
    return [(f"{c.getName()} (ID {c.getId()})", c.getId()) for c in found]


def render_summary(summary: TrainingSetSummary) -> str:
    """HTML for a set's summary card: progress per split and its sessions."""
    ident = f"table {summary.table_id}" if summary.table_id else "local"
    rows = "".join(
        f"<tr><td>{category}</td><td>{done} / {total} annotated</td></tr>"
        for category, (done, total) in summary.progress.items()
        if total
    )
    sessions = (
        "".join(
            f"<li>Session {s.id} ({'review, ' if s.kind == 'review' else ''}"
            f"{s.status}, {s.n_annotated} {'revised' if s.kind == 'review' else 'saved'}"
            f"{', ' + html.escape(s.user) if s.user else ''})"
            f"{': <i>' + html.escape(s.note) + '</i>' if s.note else ''}</li>"
            for s in summary.sessions
        )
        or "<li class='oai-muted'>No sessions yet</li>"
    )
    return (
        f"<b>{html.escape(summary.name)}</b> <span class='oai-muted'>({ident}, "
        f"storage: {summary.storage})</span>"
        f"<table class='oai-table'>{rows}</table><ul>{sessions}</ul>"
    )


class WorkflowWidget:
    """Create a new training set or open one, then follow the annotation."""

    def __init__(self, connection: Any = None, *, home: str | Path | None = None):
        self.connection = connection
        self.home = home_dir(home)
        self._training_set: TrainingSet | None = None
        self._notes: dict[int, str] = {}  # session ID -> note, for the note editor
        self._log: list[str] = []
        self._build()
        self._load_containers()

    # --- layout -------------------------------------------------------------------

    def _build(self) -> None:
        self.container_type = widgets.Dropdown(
            options=CONTAINER_TYPES,
            description="Container",
            style=_LABEL,
            layout=_narrow(),
        )
        self.container = widgets.Dropdown(options=[], layout=_wide())
        self.container_info = widgets.HTML()
        self.container_type.observe(lambda _: self._load_containers(), names="value")
        self.container.observe(lambda _: self._on_container(), names="value")

        self.mode = widgets.ToggleButtons(
            options=[("New training set", "new"), ("Open training set", "open")],
            value="new",
        )
        self.mode.observe(lambda _: self._show_mode(), names="value")

        def number(
            description: str, value: int, low: int, high: int
        ) -> widgets.BoundedIntText:
            return widgets.BoundedIntText(
                value=value,
                min=low,
                max=high,
                description=description,
                style=_LABEL,
                layout=_narrow(),
            )

        # New
        self.name = widgets.Text(description="Name", style=_LABEL, layout=_wide())
        self.channel = number("Channel", 0, 0, 99)
        self.timepoints = _AxisPicker("Timepoints")
        self.z_slices = _AxisPicker("Z-slices")
        self.three_d = widgets.Checkbox(
            description="3D: one volume over the chosen z-slices", indent=False
        )
        self.use_patches = widgets.Checkbox(description="Patches", indent=False)
        self.patch_size = number("Patch size", 512, 32, 8192)
        self.patches_per_image = number("Per image", 1, 1, 1000)

        def sync_patches(_: object = None) -> None:
            off = not self.use_patches.value
            self.patch_size.disabled = self.patches_per_image.disabled = off

        self.use_patches.observe(sync_patches, names="value")
        sync_patches()
        self.counts = {
            category: number(category.capitalize(), value, 0, 100_000)
            for (category, _), value in zip(SPLITS, (5, 2, 0), strict=True)
        }
        self.seed = widgets.Text(
            placeholder="random", description="Seed", style=_LABEL, layout=_narrow()
        )
        self.storage = widgets.Dropdown(
            options=[
                ("OMERO (table + uploads)", "omero"),
                ("Local folder only", "local"),
            ],
            description="Storage",
            style=_LABEL,
            layout=_wide(),
        )
        self.new_card = _card(
            "New training set",
            self.name,
            self.channel,
            self.timepoints.box,
            self.z_slices.box,
            self.three_d,
            widgets.HBox([self.use_patches, self.patch_size, self.patches_per_image]),
            widgets.HBox(list(self.counts.values())),
            widgets.HBox([self.seed, self.storage]),
        )

        # Open
        self.sets = widgets.Select(options=[], rows=5, layout=_wide())
        self.sets.observe(lambda _: self._on_set(), names="value")
        self.summary = widgets.HTML()
        self.add = {
            category: number(f"+ {category}", 0, 0, 100_000) for category, _ in SPLITS
        }
        self.task = widgets.ToggleButtons(
            options=[
                ("Annotate pending", "annotate"),
                ("Review annotated", "review"),
            ],
            value="annotate",
            tooltips=[
                "Annotate the units not annotated yet",
                "Open annotated units with their saved mask to correct them",
            ],
        )
        self.open_card = _card(
            "Open training set",
            self.sets,
            self.summary,
            widgets.HTML(
                "<span class='oai-muted'>Add images per split (optional)</span>"
            ),
            widgets.HBox(list(self.add.values())),
            self.task,
        )

        # Run settings (both modes)
        self.model = widgets.Dropdown(
            options=MODELS, description="Model", style=_LABEL, layout=_wide()
        )
        self.batch_size = number("Batch size", 0, 0, 10_000)
        self.go = widgets.Button(
            description="Create", button_style="primary", icon="check"
        )
        self.go.on_click(lambda _: self._resolve())
        self.status = widgets.HTML()

        # Progress
        self.progress = widgets.IntProgress(value=0, min=0, max=1, layout=_wide())
        self.log = widgets.HTML()
        self.note = widgets.Textarea(
            placeholder="How did it go? e.g. 'nuclei only, skipped small objects'",
            layout=widgets.Layout(width="420px", height="60px"),
        )
        self.note_session = widgets.Dropdown(
            options=[], description="Session", style=_LABEL, layout=_narrow()
        )
        self.note_session.observe(lambda _: self._show_note(), names="value")
        self.save_note = widgets.Button(
            description="Save session note",
            tooltip="One note per session: saving replaces that session's note",
        )
        self.save_note.on_click(lambda _: self._save_note())

        self.main = widgets.VBox(
            [
                widgets.HTML(_STYLE),
                _card(
                    "Data",
                    widgets.HBox([self.container_type, self.container]),
                    self.container_info,
                ),
                self.mode,
                self.new_card,
                self.open_card,
                _card(
                    "Annotation",
                    widgets.HBox([self.model, self.batch_size]),
                    self.go,
                    self.status,
                ),
                _card(
                    "Progress",
                    self.progress,
                    self.log,
                    widgets.HBox([self.note_session, self.save_note]),
                    self.note,
                ),
            ]
        )
        self._show_mode()

    def _show_mode(self) -> None:
        new = self.mode.value == "new"
        self.new_card.layout.display = None if new else "none"
        self.open_card.layout.display = "none" if new else None
        self.go.description = "Create" if new else "Open"
        self._reset_selection()
        if not new:
            self._load_sets()

    # --- data ---------------------------------------------------------------------

    def _load_containers(self) -> None:
        self._reset_selection()
        if self.connection is None:
            self.container.options = []
            return
        self.container.options = container_options(
            self.connection, self.container_type.value, self._error
        )

    def _on_container(self) -> None:
        self._reset_selection()
        if self.connection is None or self.container.value is None:
            self.container_info.value = ""
            return
        info = (
            get_container_info(
                self.connection,
                str(self.container_type.value),
                int(self.container.value),
            )
            or {}
        )
        self.container_info.value = self._format_info(info)
        sizes = info.get("dimensions") or {}
        if sizes.get("C"):
            self.channel.max = max(0, int(sizes["C"]) - 1)
        if not self.name.value and info.get("container_name"):
            self.name.value = str(info["container_name"])
        if self.mode.value == "open":
            self._load_sets()

    @staticmethod
    def _format_info(info: dict[str, Any]) -> str:
        if not info:
            return ""
        sizes = info.get("dimensions") or {}
        dims = ", ".join(f"{k}={sizes[k]}" for k in "XYZCT" if k in sizes)
        names = info.get("channel_names") or []
        channels = ", ".join(f"{i}: {html.escape(str(n))}" for i, n in enumerate(names))
        if info.get("channel_names_vary"):
            channels += " <i>(names differ between images)</i>"
        parts = [f"{info.get('total_images', 0)} images"]
        if dims:
            parts.append(f"sample {dims}")
        if channels:
            parts.append(f"channels {channels}")
        return f"<span class='oai-muted'>{' · '.join(parts)}</span>"

    def _reset_selection(self) -> None:
        """A new container or mode invalidates the chosen set and its summary."""
        self._training_set = None
        self.sets.options = []
        self.summary.value = ""
        self._notes = {}
        self.note_session.options = []
        for box in self.add.values():
            box.value = 0
        self.task.value = "annotate"
        self.status.value = ""

    def _load_sets(self) -> None:
        options = training_set_choices(
            self.connection,
            str(self.container_type.value),
            self.container.value,
            home=self.home,
            on_error=self._error,
        )
        self.sets.options = options
        if not options:
            self.summary.value = (
                "<span class='oai-muted'>No training sets here yet.</span>"
            )

    def _on_set(self) -> None:
        self._training_set = None
        self._notes = {}
        self.note_session.options = []
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        if self.sets.value is None:
            self.summary.value = ""
            return
        try:
            info = summarize(self.sets.value, self.connection, home=self.home)
        except (OmeroAnnotateAIError, OSError, ValueError) as error:
            self.summary.value = (
                f"<span class='oai-error'>{html.escape(str(error))}</span>"
            )
            return
        self.summary.value = render_summary(info)
        for record in info.sessions:
            self._remember_session(record.id, record.note or "")

    # --- requests -----------------------------------------------------------------

    def get_config(self) -> AnnotationConfig:
        """The config a "New training set" would be created from (a fresh copy)."""
        config = create_default_config()
        config.name = self.name.value.strip() or config.name
        if self.container.value is not None:
            config.omero.container_type = str(self.container_type.value)
            config.omero.container_id = int(self.container.value)
        coverage = config.spatial_coverage
        coverage.channels = [int(self.channel.value)]
        mode, listed, n_random = self.timepoints.values()
        coverage.timepoint_mode, coverage.timepoints = mode, listed
        coverage.n_timepoints = n_random
        mode, listed, n_random = self.z_slices.values()
        coverage.z_slice_mode, coverage.z_slices = mode, listed
        coverage.n_slices = n_random
        coverage.three_d = bool(self.three_d.value)
        coverage.use_patches = bool(self.use_patches.value)
        coverage.patch_size = [int(self.patch_size.value)] * 2
        coverage.patches_per_image = int(self.patches_per_image.value)
        for category, key in SPLITS:
            setattr(config.training, key, int(self.counts[category].value))
        seed = self.seed.value.strip()
        config.training.seed = int(seed) if seed else None
        config.ai_model.pretrained_from = str(self.model.value)
        config.workflow.batch_size = int(self.batch_size.value)
        config.storage = "local" if self.storage.value == "local" else "omero"
        return config

    def request(self) -> Request:
        """The form as a request for :func:`open_or_create`."""
        if self.mode.value == "new":
            return NewTrainingSet(self.get_config())
        if self.sets.value is None:
            raise ValueError("Select a training set to open")
        return OpenTrainingSet(
            self.sets.value, {c: int(box.value) for c, box in self.add.items()}
        )

    def training_set(self, **kwargs: Any) -> TrainingSet:
        """The training set of the form, created or opened on first call.

        ``kwargs`` go to :func:`open_or_create`.
        """
        if self._training_set is None:
            self._training_set = open_or_create(
                self.request(),
                self.connection,
                home=self.home,
                events=self.events,
                **kwargs,
            )
        return self._training_set

    def annotate(
        self, annotator: Annotator | None = None, note: str | None = None
    ) -> int:
        """Run a session with the form's model and batch size.

        Annotates the pending units, or - with *Review annotated* chosen for an
        opened set - opens the annotated units with their masks to correct them.
        """
        ts = self.training_set()
        run = ts.review if self._reviewing() else ts.annotate
        return run(
            annotator,
            model=str(self.model.value),
            batch_size=int(self.batch_size.value) or None,
            note=note,
        )

    def _reviewing(self) -> bool:
        return self.mode.value == "open" and self.task.value == "review"

    def _resolve(self) -> None:
        self._training_set = None
        try:
            ts = self.training_set()
        except (OmeroAnnotateAIError, OSError, ValueError) as error:
            self._error(str(error))
            return
        reviewing = self._reviewing()
        count = sum(row.processed == reviewing for row in ts.units())
        todo = "annotated units to review" if reviewing else "units to annotate"
        where = f"table {ts.table_id}" if ts.table_id else "local"
        self.status.value = (
            f"✅ <b>{html.escape(ts.name)}</b> ({where}): {count} {todo}. "
            f"<span class='oai-muted'>Folder: {html.escape(str(ts.bundle_dir))}</span>"
        )

    # --- progress -----------------------------------------------------------------

    def events(self, event: Event) -> None:
        """Event sink: annotation progress is shown in the Progress card."""
        match event:
            case SessionStarted(session=session, pending=pending):
                self._remember_session(session, "")
                self.progress.max = max(1, pending)
                self.progress.value = 0
                self._add_log(f"Session {session}: {pending} units")
            case BatchPrepared(index=index, annotation_ids=ids):
                self._add_log(f"Batch {index}: {len(ids)} units in napari")
            case UnitCommitted():
                self.progress.value = min(self.progress.max, self.progress.value + 1)
            case UploadFailed(annotation_id=aid, error=error):
                self._add_log(f"Upload failed for {aid}: {error}", error=True)
            case ReconcileAction(annotation_id=aid, action=action, detail=detail):
                self._add_log(f"Repaired {aid}: {action} {detail}".strip())
            case Progress(stage=stage, done=done, total=total):
                self.progress.max = max(1, total)
                self.progress.value = done
                self.progress.description = str(stage)
            case UnitConflict(annotation_id=aid, mask_file=kept):
                self._add_log(
                    f"{aid} was saved by another session first; yours is kept as {kept}",
                    error=True,
                )
            case ClaimsHeld(units=units):
                self._add_log(f"{units} units are being annotated elsewhere (skipped)")
            case ClaimsReleased(units=units):
                self._add_log(f"Released {units} units an earlier run left claimed")
            case RunCancelled(stage=stage):
                self._add_log(f"Cancelled during {stage}", error=True)
            case SessionFinished(session=session, annotated=annotated):
                self._add_log(f"Session {session} finished: {annotated} saved")
                self._remember_session(session, self._session_note(session))
            case _:
                pass

    def _add_log(self, text: str, *, error: bool = False) -> None:
        css = " class='oai-error'" if error else ""
        self._log = [*self._log, f"<div{css}>{html.escape(text)}</div>"][-LOG_LINES:]
        self.log.value = "".join(self._log)

    def _remember_session(self, session: int, note: str) -> None:
        """Offer ``session`` in the note editor and select it (the latest one)."""
        self._notes[session] = note
        self.note_session.options = [(str(s), s) for s in sorted(self._notes)]
        self.note_session.value = max(self._notes)
        self._show_note()

    def _show_note(self) -> None:
        session = self.note_session.value
        self.note.value = "" if session is None else self._notes.get(session, "")

    def _save_note(self) -> None:
        note = self.note.value.strip()
        session = self.note_session.value
        no_set = self._training_set is None and (
            self.mode.value != "open" or self.sets.value is None
        )
        if session is None or not note or no_set:
            self._error(
                "Pick a set with a session (or annotate one), then write a note"
            )
            return
        try:
            ts = self.training_set()  # opens the picked set; never creates one
            ts.set_session_note(session, note)
        except (OmeroAnnotateAIError, OSError, ValueError, KeyError) as error:
            self._error(str(error))
            return
        self._notes[session] = note
        self._add_log(f"Note saved to session {session}")
        if self.mode.value == "open":
            self._refresh_summary()

    def _session_note(self, session: int) -> str:
        """The session's current note (empty if none), to edit and save again."""
        if self._training_set is None:
            return ""
        notes = (s.note for s in self._training_set.config.sessions if s.id == session)
        return next(notes, None) or ""

    def _error(self, text: str) -> None:
        self.status.value = f"<span class='oai-error'>{html.escape(text)}</span>"

    # --- display ------------------------------------------------------------------

    def set_connection(self, connection: Any) -> None:
        """Use another OMERO connection (reloads the containers)."""
        self.connection = connection
        self._load_containers()

    def display(self) -> None:
        """Show the widget in the notebook."""
        display(self.main)


def create_workflow_widget(
    connection: Any = None, *, home: str | Path | None = None
) -> WorkflowWidget:
    """Create the workflow widget (call ``.display()`` to show it)."""
    return WorkflowWidget(connection, home=home)
