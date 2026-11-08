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
    Event,
    Progress,
    ReconcileAction,
    RunCancelled,
    SessionFinished,
    SessionStarted,
    UnitCommitted,
    UploadFailed,
)
from ..omero.omero_utils import get_container_info
from ..trainingset import TrainingSet, TrainingSetSummary, list_training_sets, summarize

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


def _card(title: str, *children: widgets.Widget) -> widgets.VBox:
    header = widgets.HTML(f"<div class='oai-title'>{html.escape(title)}</div>")
    card = widgets.VBox([header, *children])
    card.add_class("oai-card")
    return card


def _local_sets(home: Path) -> list[tuple[str, int | Path]]:
    """Training sets kept only on disk, as ``(label, folder)``."""
    folders = sorted((home / "local").glob("*/config.yaml"))
    return [(f"{f.parent.name} (local)", f.parent) for f in folders]


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
            f"<li>Session {s.id} ({s.status}, {s.n_annotated} saved"
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
        self._last_session: int | None = None
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
        self.timepoint = number("Timepoint", 0, 0, 100_000)
        self.z_slice = number("Z-slice", 0, 0, 100_000)
        self.three_d = widgets.Checkbox(description="3D (all z-slices)", indent=False)
        self.use_patches = widgets.Checkbox(description="Patches", indent=False)
        self.patch_size = number("Patch size", 512, 32, 8192)
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
            widgets.HBox([self.channel, self.timepoint, self.z_slice]),
            widgets.HBox([self.three_d, self.use_patches, self.patch_size]),
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
        self.open_card = _card(
            "Open training set",
            self.sets,
            self.summary,
            widgets.HTML(
                "<span class='oai-muted'>Add images per split (optional)</span>"
            ),
            widgets.HBox(list(self.add.values())),
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
        self.save_note = widgets.Button(description="Save note to last session")
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
                _card("Progress", self.progress, self.log, self.note, self.save_note),
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
        kind = str(self.container_type.value).capitalize()
        try:
            found = list(self.connection.getObjects(kind))
        except Exception as error:  # noqa: BLE001 - UI boundary: show any OMERO error
            self._error(f"Could not list {kind}s: {error}")
            return
        self.container.options = [
            (f"{c.getName()} (ID {c.getId()})", c.getId()) for c in found
        ]

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
        for box in self.add.values():
            box.value = 0
        self.status.value = ""

    def _load_sets(self) -> None:
        options: list[tuple[str, int | Path]] = []
        if self.connection is not None and self.container.value is not None:
            try:
                options += [
                    (f"{s.name} (table {s.table_id})", s.table_id)
                    for s in list_training_sets(
                        self.connection,
                        str(self.container_type.value),
                        int(self.container.value),
                    )
                ]
            except Exception as error:  # noqa: BLE001 - UI boundary
                self._error(f"Could not list training sets: {error}")
        options += _local_sets(self.home)
        self.sets.options = options
        if not options:
            self.summary.value = (
                "<span class='oai-muted'>No training sets here yet.</span>"
            )

    def _on_set(self) -> None:
        self._training_set = None
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
        coverage.timepoints = [int(self.timepoint.value)]
        coverage.z_slices = [int(self.z_slice.value)]
        coverage.three_d = bool(self.three_d.value)
        coverage.use_patches = bool(self.use_patches.value)
        coverage.patch_size = [int(self.patch_size.value)] * 2
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
        """Annotate the set's pending units with the form's model and batch size."""
        return self.training_set().annotate(
            annotator,
            model=str(self.model.value),
            batch_size=int(self.batch_size.value) or None,
            note=note,
        )

    def _resolve(self) -> None:
        self._training_set = None
        try:
            ts = self.training_set()
        except (OmeroAnnotateAIError, OSError, ValueError) as error:
            self._error(str(error))
            return
        pending = sum(not row.processed for row in ts.units())
        where = f"table {ts.table_id}" if ts.table_id else "local"
        self.status.value = (
            f"✅ <b>{html.escape(ts.name)}</b> ({where}): {pending} units to annotate. "
            f"<span class='oai-muted'>Folder: {html.escape(str(ts.bundle_dir))}</span>"
        )

    # --- progress -----------------------------------------------------------------

    def events(self, event: Event) -> None:
        """Event sink: annotation progress is shown in the Progress card."""
        match event:
            case SessionStarted(session=session, pending=pending):
                self._last_session = session
                self.progress.max = max(1, pending)
                self.progress.value = 0
                self._add_log(f"Session {session}: {pending} units to annotate")
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
            case RunCancelled(stage=stage):
                self._add_log(f"Cancelled during {stage}", error=True)
            case SessionFinished(session=session, annotated=annotated):
                self._last_session = session
                self._add_log(f"Session {session} finished: {annotated} saved")
            case _:
                pass

    def _add_log(self, text: str, *, error: bool = False) -> None:
        css = " class='oai-error'" if error else ""
        self._log = [*self._log, f"<div{css}>{html.escape(text)}</div>"][-LOG_LINES:]
        self.log.value = "".join(self._log)

    def _save_note(self) -> None:
        note = self.note.value.strip()
        if self._training_set is None or self._last_session is None or not note:
            self._error("Annotate a session first, then write a note")
            return
        self._training_set.set_session_note(self._last_session, note)
        self._add_log(f"Note saved to session {self._last_session}")
        self.note.value = ""

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
