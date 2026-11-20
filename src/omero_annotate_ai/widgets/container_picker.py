"""Find and pick OMERO containers on a server with many groups, users and datasets.

The picker narrows the list by group (switching the session's group) and owner,
filters it by name or ID as you type, and finds a container by ID in any of the
user's groups. With ``multiple=True`` several containers of one type can be picked
(a multi-container training set); the picks survive changing the filter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ipywidgets as widgets

from ..forms import CONTAINER_TYPES, container_label, matches_search
from ..omero.omero_utils import (
    find_container,
    list_containers,
    list_group_members,
    list_groups,
)
from ..omero.simple_connection import ensure_connected

ALL_MEMBERS = -1
_LABEL = {"description_width": "110px"}
ROWS = 8


class ContainerPicker:
    """Group, owner, container type, a search box and the list of containers.

    ``on_change`` is called when the picked containers change (not when the filter
    does); ``on_error`` receives a message when OMERO cannot be read.
    """

    def __init__(
        self,
        connection: Any = None,
        *,
        multiple: bool = False,
        on_change: Callable[[], None] = lambda: None,
        on_error: Callable[[str], None] = lambda _: None,
    ) -> None:
        self.connection = connection
        self.multiple = multiple
        self._on_change = on_change
        self._on_error = on_error
        self._all: list[tuple[str, int]] = []  # every container of the type and owner
        self._picked: list[int] = []
        self._quiet = False  # set while the widgets are filled programmatically
        self._build()
        self.reload()

    # --- layout -------------------------------------------------------------------

    def _build(self) -> None:
        def dropdown(description: str, **kwargs: Any) -> widgets.Dropdown:
            return widgets.Dropdown(
                description=description,
                style=_LABEL,
                layout=widgets.Layout(width="320px"),
                **kwargs,
            )

        self.group = dropdown("Group", options=[])
        self.owner = dropdown("Owner", options=[])
        self.container_type = dropdown("Container", options=CONTAINER_TYPES)
        self.search = widgets.Text(
            placeholder="Filter by name or ID",
            description="Search",
            style=_LABEL,
            layout=widgets.Layout(width="320px"),
            continuous_update=True,
        )
        self.find = widgets.Button(
            description="Go to ID",
            icon="search",
            tooltip="Find this ID in any of your groups and switch to its group",
            disabled=True,
            layout=widgets.Layout(width="110px"),
        )
        self.count = widgets.HTML()
        kind = widgets.SelectMultiple if self.multiple else widgets.Select
        self.container = kind(
            options=[], rows=ROWS, layout=widgets.Layout(width="650px")
        )
        self.picked_info = widgets.HTML()

        self.group.observe(lambda _: self._on_group(), names="value")
        self.owner.observe(lambda _: self._load(), names="value")
        self.container_type.observe(lambda _: self._on_type(), names="value")
        self.search.observe(lambda _: self._filter(), names="value")
        self.find.on_click(lambda _: self._find())
        self.container.observe(lambda _: self._on_pick(), names="value")

        hint = (
            "<span class='oai-muted'>Ctrl/Shift-click to pick several "
            "(one training set over all of them)</span>"
            if self.multiple
            else ""
        )
        self.box = widgets.VBox(
            [
                widgets.HBox([self.group, self.owner]),
                widgets.HBox([self.container_type]),
                widgets.HBox([self.search, self.find, self.count]),
                self.container,
                widgets.HTML(hint),
                self.picked_info,
            ]
        )

    # --- public -------------------------------------------------------------------

    @property
    def selected(self) -> list[int]:
        """IDs of the picked containers, in the order they were picked."""
        return list(self._picked)

    @property
    def kind(self) -> str:
        return str(self.container_type.value)

    def set_connection(self, connection: Any) -> None:
        self.connection = connection
        self.reload()

    def reload(self) -> None:
        """Read the groups and owners again, then the containers."""
        self._quiet = True
        try:
            self._picked = []
            self.group.options = self.owner.options = []
            if self.connection is not None:
                self._fill_groups()
        finally:
            self._quiet = False
        self._load()

    def pick(self, container_id: int) -> None:
        """Pick a container that is in the list (added to the picks if multiple)."""
        if container_id not in [cid for _, cid in self._all]:
            raise ValueError(f"No {self.kind} with ID {container_id} in this list")
        if self.multiple:
            picked = [*self._picked, container_id]
            self._picked = list(dict.fromkeys(picked))
        else:
            self._picked = [container_id]
        self._show()
        self._on_change()

    # --- data ---------------------------------------------------------------------

    def _fill_groups(self) -> None:
        """Group and owner choices; a server that cannot list them shows neither."""
        conn = self.connection
        try:
            ensure_connected(conn)
            groups = list_groups(conn)
            current = int(conn.getEventContext().groupId)
            me = int(conn.getUserId())
        except Exception as error:  # noqa: BLE001 - UI boundary: show any OMERO error
            self._on_error(f"Could not list groups: {error}")
            self.group.layout.display = self.owner.layout.display = "none"
            return
        self.group.layout.display = self.owner.layout.display = None
        if current not in [gid for _, gid in groups]:  # e.g. root in "system"
            groups.append((conn.getGroupFromContext().getName(), current))
        self.group.options = groups
        self.group.value = current
        self._fill_owners(me)

    def _fill_owners(self, me: int | None = None) -> None:
        conn = self.connection
        if me is None:
            me = int(conn.getUserId())
        group = self.group.value
        try:
            members = [] if group is None else list_group_members(conn, int(group))
        except Exception as error:  # noqa: BLE001 - UI boundary
            self._on_error(f"Could not list the group's members: {error}")
            members = []
        others = [(name, uid) for name, uid in members if uid != me]
        self.owner.options = [("Me", me), ("All members", ALL_MEMBERS), *others]
        self.owner.value = me

    def _on_group(self) -> None:
        if self._quiet or self.group.value is None:
            return
        try:
            self.connection.setGroupForSession(int(self.group.value))
        except Exception as error:  # noqa: BLE001 - UI boundary
            self._on_error(f"Could not switch to the group: {error}")
            return
        self._quiet = True
        try:
            self._picked = []
            self._fill_owners()
        finally:
            self._quiet = False
        self._load()
        self._on_change()

    def _on_type(self) -> None:
        if self._quiet:
            return
        self._picked = []
        self._load()
        self._on_change()

    def _load(self) -> None:
        """Read the containers of the chosen type and owner from OMERO."""
        if self._quiet:
            return
        self._all = []
        if self.connection is not None:
            owner = self.owner.value
            try:
                ensure_connected(self.connection)
                self._all = list_containers(
                    self.connection,
                    self.kind,
                    None if owner in (None, ALL_MEMBERS) else int(owner),
                )
            except Exception as error:  # noqa: BLE001 - UI boundary
                self._on_error(f"Could not list {self.kind}s: {error}")
        known = {cid for _, cid in self._all}
        self._picked = [cid for cid in self._picked if cid in known]
        self._show()

    def _filter(self) -> None:
        text = self.search.value.strip()
        self.find.disabled = not text.isdigit() or self.connection is None
        self._show()

    def _matches(self, name: str, cid: int) -> bool:
        return matches_search(name, cid, self.search.value)

    def _show(self) -> None:
        """Fill the list: the picks first, then the matches of the filter."""
        picked = set(self._picked)
        names = dict((cid, name) for name, cid in self._all)
        shown = [(container_label(names[cid], cid), cid) for cid in self._picked]
        shown += [
            (container_label(name, cid), cid)
            for name, cid in self._all
            if cid not in picked and self._matches(name, cid)
        ]
        matches = sum(self._matches(name, cid) for name, cid in self._all)
        self._quiet = True
        try:
            self.container.options = shown
            if self.multiple:
                self.container.value = tuple(self._picked)
            else:
                self.container.value = self._picked[0] if self._picked else None
        finally:
            self._quiet = False
        total = len(self._all)
        self.count.value = (
            f"<span class='oai-muted'>{matches} of {total} {self.kind}s</span>"
            if self.search.value.strip()
            else f"<span class='oai-muted'>{total} {self.kind}s</span>"
        )
        self.picked_info.value = (
            "<span class='oai-muted'>Picked: "
            + ", ".join(container_label(names[cid], cid) for cid in self._picked)
            + "</span>"
            if self.multiple and len(self._picked) > 1
            else ""
        )

    def _on_pick(self) -> None:
        if self._quiet:
            return
        value = self.container.value
        chosen = list(value) if self.multiple else ([] if value is None else [value])
        # Keep the order of picking: earlier picks first, new ones after.
        picked = [cid for cid in self._picked if cid in chosen]
        picked += [cid for cid in chosen if cid not in picked]
        if picked == self._picked:
            return
        self._picked = picked
        self._show()
        self._on_change()

    def _find(self) -> None:
        """Go to the container whose ID is typed in the search box, in any group."""
        cid = int(self.search.value.strip())
        try:
            ensure_connected(self.connection)
            group = find_container(self.connection, self.kind, cid)
        except Exception as error:  # noqa: BLE001 - UI boundary
            self._on_error(f"Could not look up {self.kind} {cid}: {error}")
            return
        if group is None:
            self._on_error(f"No {self.kind} with ID {cid} that you can read")
            return
        keep = self._picked if group == self.group.value else []
        self._quiet = True
        try:
            if group != self.group.value:
                self.connection.setGroupForSession(group)
                if group not in [gid for _, gid in self.group.options]:
                    self.group.options = [
                        *self.group.options,
                        (f"group {group}", group),
                    ]
                self.group.value = group
                self._fill_owners()
            self.owner.value = ALL_MEMBERS
            self.search.value = ""
            self.find.disabled = True
        finally:
            self._quiet = False
        self._picked = keep
        self._load()
        self.pick(cid)
