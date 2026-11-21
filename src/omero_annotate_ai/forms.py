"""The "New training set" form as plain data, shared by the Jupyter widgets and the
napari dock.

A front end fills a :class:`NewSetForm` from its fields and calls
:meth:`NewSetForm.to_config`; the logic behind the pickers (search, channel rows,
well filters) lives here too, so both front ends behave the same.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from bioimage_annotation_schema import datamodel as bas

from .core.annotation_config import (
    AnnotationConfig,
    DisplayConfig,
    NormalizationConfig,
    create_default_config,
)
from .planning import Mode, parse_indices
from .sources.wells import match_well_filter

CONTAINER_TYPES = ("dataset", "project", "plate", "screen")
WELL_CONTAINERS = ("plate", "screen")
MODELS = (
    ("vit_b_lm (light microscopy, recommended)", "vit_b_lm"),
    ("vit_l_lm", "vit_l_lm"),
    ("vit_b_em_organelles", "vit_b_em_organelles"),
    ("vit_b", "vit_b"),
    ("vit_l", "vit_l"),
    ("vit_h", "vit_h"),
)
SPLITS = (("training", "train_n"), ("validation", "validate_n"), ("test", "test_n"))
DEFAULT_COUNTS = {"training": 5, "validation": 2, "test": 0}
COLORMAPS = ("gray", "blue", "green", "red", "magenta", "cyan", "yellow")
_CONTEXT_COLORMAPS = ("blue", "magenta", "green", "yellow", "cyan")
# A key with a different value on (nearly) every well names the well, not a condition.
_UNIQUE_KEY_MIN_WELLS = 12


# --- containers ---------------------------------------------------------------------


def container_label(name: str, cid: int) -> str:
    return f"{name} (ID {cid})"


def matches_search(name: str, cid: int, text: str) -> bool:
    """A container matches when the text is in its name or is its ID."""
    text = text.strip().lower()
    return not text or text in name.lower() or text == str(cid)


# --- channels -----------------------------------------------------------------------


@dataclass
class ChannelRow:
    """One channel of the container's images and what to do with it."""

    index: int
    name: str
    train: bool = False
    show: bool = False
    colormap: str = ""

    def __post_init__(self) -> None:
        if not self.colormap:
            self.colormap = _CONTEXT_COLORMAPS[self.index % len(_CONTEXT_COLORMAPS)]


def channel_rows(names: list[str], label: int = 0) -> list[ChannelRow]:
    """Rows for these channel names: train on the annotated channel only, which
    napari shows in gray unless another colour is picked."""
    return [
        ChannelRow(i, name, train=i == label, colormap="gray" if i == label else "")
        for i, name in enumerate(names)
    ]


def move_label(rows: list[ChannelRow], old: int, new: int) -> None:
    """The annotated channel changed: training on just that channel follows it, and
    so does its default gray; the annotated channel is never also a context channel.
    Choices made by hand (more training channels, another colour) stay."""
    if [r.index for r in rows if r.train] == [old]:
        rows[old].train = False
        rows[new].train = True
    if rows[old].colormap == "gray":
        rows[old].colormap = ChannelRow(old, rows[old].name).colormap
        rows[new].colormap = "gray"
    rows[new].show = False


def input_channels(rows: list[ChannelRow], label: int) -> list[int] | None:
    """The training channels; None when only the annotated one (the default)."""
    chosen = [r.index for r in rows if r.train]
    if not chosen:
        raise ValueError("Pick at least one channel to train on")
    return None if chosen == [label] else chosen


def display_config(rows: list[ChannelRow], label: int) -> DisplayConfig:
    shown = [r for r in rows if r.show and r.index != label]
    annotated = next((r for r in rows if r.index == label), None)
    return DisplayConfig(
        label_colormap=annotated.colormap if annotated else "gray",
        context_channels=[r.index for r in shown],
        context_channel_names=[r.name for r in shown] or None,
        context_channel_colormaps=[r.colormap for r in shown] or None,
    )


# --- wells --------------------------------------------------------------------------


def well_condition_keys(
    wells: Mapping[int, Mapping[str, str]], mihcsme: set[str]
) -> list[tuple[str, str]]:
    """``(label, key)`` to filter on: MIHCSME assay conditions first, keys that name
    every well (well names, positions) left out."""
    values: dict[str, list[str]] = {}
    for kv in wells.values():
        for key, value in kv.items():
            values.setdefault(key, []).append(value)
    usable = [
        key
        for key, seen in values.items()
        if not (len(seen) >= _UNIQUE_KEY_MIN_WELLS and len(set(seen)) == len(seen))
    ]
    usable.sort(key=lambda k: (k not in mihcsme, k.lower()))
    return [(f"{k} (MIHCSME)" if k in mihcsme else k, k) for k in usable]


def well_value_choices(
    wells: Mapping[int, Mapping[str, str]], key: str
) -> list[tuple[str, str]]:
    """``(label, value)`` of one key, with the number of wells per value."""
    counts = Counter(kv[key] for kv in wells.values() if key in kv)
    return [
        (f"{value} ({n} well{'s' if n != 1 else ''})", value)
        for value, n in sorted(counts.items(), key=lambda item: item[0].lower())
    ]


def wells_kept(
    wells: Mapping[int, Mapping[str, str]],
    filters: Mapping[str, list[str]] | None,
    mode: Literal["include", "exclude"],
) -> int:
    """How many wells a filter keeps (all wells without filters)."""
    if not filters:
        return len(wells)
    matching = sum(match_well_filter(dict(kv), dict(filters)) for kv in wells.values())
    return matching if mode == "include" else len(wells) - matching


def describe_filters(filters: Mapping[str, list[str]]) -> str:
    return " AND ".join(f"{k} in {{{', '.join(v)}}}" for k, v in filters.items())


# --- the form -----------------------------------------------------------------------


@dataclass
class Planes:
    """Which timepoints or z-slices: listed ones, all, or ``n`` drawn at random."""

    mode: Mode = "specific"
    listed: str = "0"
    n: int = 1

    def selection(self, what: str) -> dict[str, object]:
        listed = parse_indices(self.listed, what) if self.mode == "specific" else [0]
        n = self.n if self.mode == "random" else None
        return {"mode": self.mode, "values": listed, "n": n}


@dataclass
class NewSetForm:
    """Everything the "New training set" form asks; :meth:`to_config` builds the
    config a set is created from (a fresh one every call)."""

    name: str = ""
    container_type: str = "dataset"
    container_ids: list[int] = field(default_factory=list[int])
    label_channel: int = 0
    channels: list[ChannelRow] = field(default_factory=list[ChannelRow])
    timepoints: Planes = field(default_factory=Planes)
    z_slices: Planes = field(default_factory=Planes)
    volumetric: bool = False
    patch_size: int | None = None  # None: whole images
    patches_per_image: int = 1
    counts: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_COUNTS))
    seed: str = ""
    storage: Literal["omero", "local"] = "omero"
    model: str = "vit_b_lm"
    batch_size: int = 0
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    well_filters: dict[str, list[str]] | None = None
    well_filter_mode: Literal["include", "exclude"] = "include"

    def to_config(self) -> AnnotationConfig:
        """The config for this form.

        Raises:
            ValueError: A field is invalid (with the field's name in the message).
        """
        config = create_default_config()
        config.name = self.name.strip() or config.name
        if self.container_ids:
            config.omero.container_type = self.container_type
            config.omero.container_id = self.container_ids[0]
            ids = list(self.container_ids)
            config.omero.container_ids = ids if len(ids) > 1 else None
        if self.container_type in WELL_CONTAINERS:
            config.omero.well_filters = self.well_filters or None
            config.omero.well_filter_mode = self.well_filter_mode
        rows = self.channels or channel_rows(
            [f"Channel {i}" for i in range(self.label_channel + 1)], self.label_channel
        )
        size = self.patch_size
        config.sampling = bas.Sampling.model_validate(
            {
                "label_channel": self.label_channel,
                "input_channels": input_channels(rows, self.label_channel),
                "timepoints": self.timepoints.selection("Timepoints"),
                "z_slices": self.z_slices.selection("Z-slices"),
                "volumetric": self.volumetric,
                "patches": None
                if size is None
                else {
                    "width": size,
                    "height": size,
                    "per_image": self.patches_per_image,
                },
            }
        )
        for category, key in SPLITS:
            setattr(config.training, key, int(self.counts.get(category, 0)))
        seed = self.seed.strip()
        if seed and not seed.lstrip("-").isdigit():
            raise ValueError(f"Seed must be a whole number, not {seed!r}")
        config.split.seed = int(seed) if seed else None
        config.ai_model.pretrained_from = self.model
        config.workflow.batch_size = int(self.batch_size)
        config.storage = self.storage
        config.display = display_config(rows, self.label_channel)
        config.normalization = self.normalization
        return config
