"""Jupyter widget to pick a training set and export it for training.

The widget holds form state only: it lists the sets on a container (and the local
sets), shows the picked set's progress, and calls
:func:`~omero_annotate_ai.processing.training_functions.export_training_set` with the
chosen output folder, channels and normalisation.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import display

from ..bundle import home_dir
from ..core.annotation_config import AnnotationConfig
from ..errors import OmeroAnnotateAIError
from ..planning import parse_indices
from ..processing.training_functions import default_export_dir, export_training_set
from ..trainingset import read_set_config, summarize, training_set_choices
from .workflow_widget import (
    _LABEL,
    _STYLE,
    CONTAINER_TYPES,
    _card,
    _wide,
    container_options,
    render_summary,
)

NORMALIZATION = (
    ("Keep native bit depth (record the normalisation)", False),
    ("Apply the normalisation at export (float32)", True),
)


class TrainingDataWidget:
    """Pick a training set (OMERO or local) and export it into the training layout."""

    def __init__(
        self,
        connection: Any = None,
        *,
        home: str | Path | None = None,
        show_export: bool = True,
    ):
        self.connection = connection
        self.show_export = show_export
        self.home = home_dir(home)
        self.result: dict[str, Any] | None = None
        self._config: AnnotationConfig | None = None
        self._build()
        self._load_containers()

    # --- layout -------------------------------------------------------------------

    def _build(self) -> None:
        self.container_type = widgets.Dropdown(
            options=CONTAINER_TYPES, description="Container", style=_LABEL
        )
        self.container = widgets.Dropdown(options=[], layout=_wide())
        self.container_type.observe(lambda _: self._load_containers(), names="value")
        self.container.observe(lambda _: self._load_sets(), names="value")
        self.sets = widgets.Select(options=[], rows=5, layout=_wide())
        self.sets.observe(lambda _: self._on_set(), names="value")
        self.summary = widgets.HTML()

        self.output_dir = widgets.Text(
            description="Output folder",
            style=_LABEL,
            layout=widgets.Layout(width="620px"),
        )
        self.training_channels = widgets.Text(
            description="Training channels", placeholder="0, 1", style=_LABEL
        )
        self.label_channel = widgets.BoundedIntText(
            description="Label channel", min=0, max=99, style=_LABEL
        )
        self.normalization = widgets.Dropdown(
            options=NORMALIZATION,
            value=False,
            description="Normalisation",
            style=_LABEL,
            layout=_wide(),
        )
        self.export_button = widgets.Button(
            description="Export", button_style="primary", icon="download"
        )
        self.export_button.on_click(lambda _: self._on_export())
        self.status = widgets.HTML()

        export_card = _card(
            "Export",
            self.output_dir,
            widgets.HBox([self.training_channels, self.label_channel]),
            self.normalization,
            self.export_button,
        )
        export_card.layout.display = None if self.show_export else "none"
        self.main = widgets.VBox(
            [
                widgets.HTML(_STYLE),
                _card(
                    "Training set",
                    widgets.HBox([self.container_type, self.container]),
                    self.sets,
                    self.summary,
                ),
                export_card,
                self.status,
            ]
        )

    # --- data ---------------------------------------------------------------------

    def _load_containers(self) -> None:
        if self.connection is None:
            self.container.options = []
            self._load_sets()  # local sets need no connection
            return
        self.container.options = container_options(
            self.connection, self.container_type.value, self._error
        )

    def _load_sets(self) -> None:
        self.sets.options = training_set_choices(
            self.connection,
            str(self.container_type.value),
            self.container.value,
            home=self.home,
            on_error=self._error,
        )
        if not self.sets.options:
            self.summary.value = (
                "<span class='oai-muted'>No training sets here yet.</span>"
            )

    def _on_set(self) -> None:
        self._config = None
        self.summary.value = self.status.value = ""
        ref = self.sets.value
        if ref is None:
            return
        try:
            self.summary.value = render_summary(summarize(ref, self.connection))
            config = read_set_config(ref, self.connection)
        except (OmeroAnnotateAIError, OSError, ValueError) as error:
            self._error(str(error))
            return
        except Exception as error:  # noqa: BLE001 - UI boundary: e.g. a dropped connection
            self._error(f"Could not read the set: {error}")
            return
        self._config = config
        coverage = config.spatial_coverage
        self.output_dir.value = str(default_export_dir(ref, config.name, self.home))
        self.training_channels.value = ", ".join(
            map(str, coverage.get_training_channels())
        )
        self.label_channel.value = coverage.get_label_channel()
        self.normalization.value = bool(config.normalization.apply_at_export)
        # A local set exports the channels its folder holds.
        local = not isinstance(ref, int)
        self.training_channels.disabled = self.label_channel.disabled = local

    # --- selection and export -----------------------------------------------------

    @property
    def selected(self) -> int | Path | None:
        """The picked set: a table ID (OMERO) or a folder (local); None if none."""
        return self.sets.value

    def get_selected_table_id(self) -> int | None:
        """The picked set's table ID; None for a local set or no selection."""
        return self.selected if isinstance(self.selected, int) else None

    def selected_config(self) -> AnnotationConfig | None:
        """The picked set's config (read-only copy); None if none is picked."""
        return None if self._config is None else self._config.model_copy(deep=True)

    def export(self) -> dict[str, Any]:
        """Export the picked set with the form's options; returns the export result."""
        ref = self.selected
        if ref is None or self._config is None:
            raise ValueError("Pick a training set first")
        if not self.output_dir.value.strip():
            raise ValueError("Choose an output folder")
        self.result = export_training_set(
            ref,
            Path(self.output_dir.value.strip()).expanduser(),
            self.connection,
            training_channels=parse_indices(
                self.training_channels.value, "Training channels"
            ),
            label_channel=int(self.label_channel.value),
            apply_normalization=bool(self.normalization.value),
            home=self.home,
        )
        return self.result

    def _on_export(self) -> None:
        self.export_button.disabled = True
        self.status.value = "<span class='oai-muted'>Exporting…</span>"
        try:
            result = self.export()
        except (OmeroAnnotateAIError, OSError, ValueError) as error:
            self._error(str(error))
        except Exception as error:  # noqa: BLE001 - UI boundary: never leave it "Exporting…"
            self._error(f"Export failed: {error}")
        else:
            stats = result.get("stats", {})
            counts = ", ".join(
                f"{key.removeprefix('n_').replace('_', ' ')}: {value}"
                for key, value in stats.items()
                if key.startswith("n_") and isinstance(value, int)
            )
            self.status.value = (
                f"✅ Exported to <code>{html.escape(str(result['base_dir']))}</code>"
                f"<br><span class='oai-muted'>{html.escape(counts)}</span>"
            )
        finally:
            self.export_button.disabled = False

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


def create_training_data_widget(
    connection: Any = None,
    *,
    home: str | Path | None = None,
    show_export: bool = True,
) -> TrainingDataWidget:
    """Create the training-data widget (call ``.display()`` to show it).

    ``show_export=False`` keeps only the set picker, for notebooks that export
    differently (e.g. a BioImage Archive submission).
    """
    return TrainingDataWidget(connection, home=home, show_export=show_export)
