"""Channel choices of a new training set: what to annotate, train on and show.

One row per channel of the container's images: the channel annotated on, the
channels the model trains on (``sampling.input_channels``) and the extra channels
napari shows as context (``display``), plus the intensity normalisation.
"""

from __future__ import annotations

import html

import ipywidgets as widgets

from ..core.annotation_config import NormalizationConfig
from ..forms import COLORMAPS, ChannelRow, move_label

_LABEL = {"description_width": "110px"}


def _cell(width: str = "auto") -> widgets.Layout:
    return widgets.Layout(width=width)


class _Row:
    def __init__(self, index: int, name: str) -> None:
        self.index = index
        self.name = name
        self.title = widgets.HTML(f"{index}: {html.escape(name)}", layout=_cell())
        self.train = widgets.Checkbox(indent=False, layout=_cell("90px"))
        self.show = widgets.Checkbox(indent=False, layout=_cell("90px"))
        self.colormap = widgets.Dropdown(
            options=COLORMAPS,
            value=ChannelRow(index, name).colormap,
            layout=_cell("110px"),
        )
        self.show.observe(lambda _: self.sync(), names="value")
        self.sync()

    def sync(self) -> None:
        self.colormap.disabled = not self.show.value


class ChannelTable:
    """Annotated, training and napari context channels, and the normalisation."""

    def __init__(self) -> None:
        self.label = widgets.Dropdown(
            options=[("0: Channel 0", 0)],
            value=0,
            description="Annotate on",
            style=_LABEL,
            layout=widgets.Layout(width="420px"),
        )
        self.label.observe(self._on_label, names="value")
        self.grid = widgets.GridBox(
            layout=widgets.Layout(
                grid_template_columns="minmax(160px, 260px) 90px 90px 110px",
                grid_gap="2px 8px",
                align_items="center",
            )
        )
        self.rows: list[_Row] = []

        self.method = widgets.Dropdown(
            options=[("Percentile", "percentile"), ("None (raw)", "none")],
            description="Normalisation",
            style=_LABEL,
            layout=widgets.Layout(width="260px"),
        )
        self.low = widgets.BoundedFloatText(
            value=1.0, min=0, max=100, step=0.1, description="low %",
            style={"description_width": "45px"}, layout=_cell("120px"),
        )  # fmt: skip
        self.high = widgets.BoundedFloatText(
            value=99.8, min=0, max=100, step=0.1, description="high %",
            style={"description_width": "45px"}, layout=_cell("120px"),
        )  # fmt: skip
        self.apply_at_export = widgets.Checkbox(
            description="Apply at export (float32)", indent=False
        )
        self.method.observe(lambda _: self._sync_method(), names="value")
        self._sync_method()

        self.box = widgets.VBox(
            [
                self.label,
                self.grid,
                widgets.HTML(
                    "<span class='oai-muted'>Train on: channels the model gets as "
                    "input. Show: extra channels napari shows next to the annotated "
                    "one (never used for training).</span>"
                ),
                widgets.HBox([self.method, self.low, self.high]),
                self.apply_at_export,
            ]
        )
        self.set_channels(["Channel 0"])

    def set_channels(self, names: list[str]) -> None:
        """One row per channel; the choices are reset."""
        label = self.label.value if self.label.value < len(names) else 0
        self.rows = [_Row(i, name) for i, name in enumerate(names)]
        self.label.unobserve(self._on_label, names="value")
        self.label.options = [(f"{i}: {name}", i) for i, name in enumerate(names)]
        self.label.value = label
        self.label.observe(self._on_label, names="value")
        header = [
            widgets.HTML(f"<b>{text}</b>")
            for text in ("Channel", "Train on", "Show", "Colour")
        ]
        cells = [
            cell
            for row in self.rows
            for cell in (row.title, row.train, row.show, row.colormap)
        ]
        self.grid.children = [*header, *cells]
        self._mark_label(label, train=True)

    def _on_label(self, change: dict) -> None:
        rows = self.rows_data()
        move_label(rows, change["old"], change["new"])
        for row, data in zip(self.rows, rows, strict=True):
            row.train.value, row.show.value = data.train, data.show
        self._mark_label(change["new"], train=False)

    def _mark_label(self, label: int, *, train: bool) -> None:
        for row in self.rows:
            is_label = row.index == label
            row.show.disabled = is_label  # napari shows it anyway
            if is_label:
                row.show.value = False
                row.title.value = (
                    f"<b>{row.index}: {html.escape(row.name)}</b> (annotated)"
                )
                if train:
                    row.train.value = True
            else:
                row.title.value = f"{row.index}: {html.escape(row.name)}"

    def _sync_method(self) -> None:
        self.low.disabled = self.high.disabled = self.method.value == "none"

    # --- values -------------------------------------------------------------------

    def rows_data(self) -> list[ChannelRow]:
        """The rows as :class:`~omero_annotate_ai.forms.ChannelRow`."""
        return [
            ChannelRow(
                r.index,
                r.name,
                train=bool(r.train.value),
                show=bool(r.show.value),
                colormap=str(r.colormap.value),
            )
            for r in self.rows
        ]

    def normalization(self) -> NormalizationConfig:
        return NormalizationConfig(
            method=self.method.value,
            percentile_low=float(self.low.value),
            percentile_high=float(self.high.value),
            apply_at_export=bool(self.apply_at_export.value),
        )
