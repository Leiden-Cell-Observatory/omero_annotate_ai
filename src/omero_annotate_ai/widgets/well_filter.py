"""Pick plate wells by their metadata (``omero.well_filters``).

The keys and values come from the wells' map annotations; keys MIHCSME_py wrote as
assay conditions (cell line, treatment, dose, ...) are offered first. Conditions
combine with AND; a well matches a condition when its value is one of those picked.
The number of matching wells updates as you pick.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Callable
from typing import Any, Literal

import ipywidgets as widgets

from ..omero.omero_utils import plate_well_metadata
from ..sources.wells import match_well_filter

_LABEL = {"description_width": "110px"}
# A key with a different value on (nearly) every well names the well, not a condition.
_UNIQUE_KEY_MIN_WELLS = 12


class _Condition:
    def __init__(
        self,
        keys: list[tuple[str, str]],
        values: Callable[[str], list[tuple[str, str]]],
    ) -> None:
        self._values = values
        self.key = widgets.Dropdown(options=keys, layout=widgets.Layout(width="260px"))
        self.accepted = widgets.SelectMultiple(
            options=[], rows=4, layout=widgets.Layout(width="320px")
        )
        self.remove = widgets.Button(
            icon="trash",
            tooltip="Remove this condition",
            layout=widgets.Layout(width="40px"),
        )
        self.key.observe(lambda _: self._fill(), names="value")
        self._fill()
        self.box = widgets.HBox([self.key, self.accepted, self.remove])

    def _fill(self) -> None:
        self.accepted.options = (
            self._values(str(self.key.value)) if self.key.value else []
        )


class WellFilter:
    """Conditions on well metadata and whether matching wells are kept or left out."""

    def __init__(self, on_error: Callable[[str], None] = lambda _: None) -> None:
        self._on_error = on_error
        self._wells: dict[int, dict[str, str]] = {}
        self._keys: list[tuple[str, str]] = []
        self.conditions: list[_Condition] = []
        self.mode = widgets.ToggleButtons(
            options=[
                ("Keep matching wells", "include"),
                ("Leave matching wells out", "exclude"),
            ],
            value="include",
        )
        self.add = widgets.Button(description="Add condition", icon="plus")
        self.add.on_click(lambda _: self._add())
        self.rows = widgets.VBox()
        self.summary = widgets.HTML()
        self.mode.observe(lambda _: self._update(), names="value")
        self.box = widgets.VBox([self.mode, self.rows, self.add, self.summary])
        self.load(None, "dataset", [])

    def load(self, conn: Any, container_type: str, container_ids: list[int]) -> None:
        """Read the wells of the picked plates (or screens); other types clear it."""
        self._wells = {}
        if conn is not None and container_ids and container_type in ("plate", "screen"):
            try:
                self._wells, mihcsme = plate_well_metadata(
                    conn, container_type, container_ids
                )
            except Exception as error:  # noqa: BLE001 - UI boundary: show any OMERO error
                self._on_error(f"Could not read the well metadata: {error}")
                mihcsme = set()
        else:
            mihcsme = set()
        self._keys = self._condition_keys(mihcsme)
        self.conditions = []
        self.rows.children = []
        self.add.disabled = not self._keys
        self._update()

    def _condition_keys(self, mihcsme: set[str]) -> list[tuple[str, str]]:
        """``(label, key)``, MIHCSME assay conditions first; well-naming keys left out."""
        values: dict[str, list[str]] = {}
        for kv in self._wells.values():
            for key, value in kv.items():
                values.setdefault(key, []).append(value)
        usable = [
            key
            for key, seen in values.items()
            if not (len(seen) >= _UNIQUE_KEY_MIN_WELLS and len(set(seen)) == len(seen))
        ]
        usable.sort(key=lambda k: (k not in mihcsme, k.lower()))
        return [(f"{k} (MIHCSME)" if k in mihcsme else k, k) for k in usable]

    def _values(self, key: str) -> list[tuple[str, str]]:
        counts = Counter(kv[key] for kv in self._wells.values() if key in kv)
        return [
            (f"{value} ({n} well{'s' if n != 1 else ''})", value)
            for value, n in sorted(counts.items(), key=lambda item: item[0].lower())
        ]

    def _add(self) -> None:
        condition = _Condition(self._keys, self._values)
        condition.accepted.observe(lambda _: self._update(), names="value")
        condition.key.observe(lambda _: self._update(), names="value")
        condition.remove.on_click(lambda _: self._remove(condition))
        self.conditions.append(condition)
        self.rows.children = [c.box for c in self.conditions]
        self._update()

    def _remove(self, condition: _Condition) -> None:
        self.conditions.remove(condition)
        self.rows.children = [c.box for c in self.conditions]
        self._update()

    def _update(self) -> None:
        total = len(self._wells)
        if not total:
            self.summary.value = (
                "<span class='oai-muted'>Pick a plate or screen to filter its wells "
                "by metadata.</span>"
            )
            return
        if not self._keys:
            self.summary.value = (
                f"<span class='oai-muted'>{total} wells; none has metadata to filter "
                "on (add it with MIHCSME_py).</span>"
            )
            return
        filters = self.filters()
        if filters is None:
            self.summary.value = f"<span class='oai-muted'>All {total} wells</span>"
            return
        matching = sum(match_well_filter(kv, filters) for kv in self._wells.values())
        kept = matching if self.mode.value == "include" else total - matching
        text = " AND ".join(
            f"{html.escape(k)} in {{{', '.join(map(html.escape, v))}}}"
            for k, v in filters.items()
        )
        css = "oai-error" if kept == 0 else "oai-muted"
        self.summary.value = (
            f"<span class='{css}'>{kept} of {total} wells kept ({text})</span>"
        )

    # --- values -------------------------------------------------------------------

    def filters(self) -> dict[str, list[str]] | None:
        """``{key: accepted values}`` of the conditions with values; None if none."""
        found: dict[str, list[str]] = {}
        for condition in self.conditions:
            if condition.key.value and condition.accepted.value:
                values = found.setdefault(str(condition.key.value), [])
                values += [v for v in condition.accepted.value if v not in values]
        return found or None

    @property
    def filter_mode(self) -> Literal["include", "exclude"]:
        return "exclude" if self.mode.value == "exclude" else "include"
