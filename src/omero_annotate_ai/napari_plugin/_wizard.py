# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
# (conda's PyQt5 ships no type stubs, so Qt classes are unknown to pyright.)
"""The dock's wizard pieces: find data on OMERO, and the "New training set" form.

The logic behind them (search, channel rows, well filters, form -> config) is in
:mod:`omero_annotate_ai.forms`, shared with the Jupyter widgets. Every OMERO call
goes through the dock's IO worker (``run_io``): a BlitzGateway must be used from one
thread at a time, and napari must stay responsive.
"""

from collections.abc import Callable, Mapping
from typing import Any, Literal

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.annotation_config import NormalizationConfig
from ..forms import (
    COLORMAPS,
    CONTAINER_TYPES,
    DEFAULT_COUNTS,
    MODELS,
    SPLITS,
    WELL_CONTAINERS,
    ChannelRow,
    NewSetForm,
    Planes,
    channel_rows,
    container_label,
    describe_filters,
    matches_search,
    move_label,
    well_condition_keys,
    well_value_choices,
    wells_kept,
)
from ..omero.omero_utils import (
    find_container,
    list_containers,
    list_group_members,
    list_groups,
)

RunIO = Callable[[Callable[[], Any], Callable[[Any], None]], None]
"""Run a function on the IO thread; its result goes to the callback (main thread)."""

ALL_MEMBERS = -1


def combo() -> QComboBox:
    """A combo box that fits a narrow dock: long entries are elided, not widening it."""
    box = QComboBox()
    box.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    box.setMinimumContentsLength(8)
    return box


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: gray")
    return label


def hbox(*widgets: QWidget) -> QWidget:
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    return box


class ContainerBrowser(QWidget):
    """Group, owner, container type, search, *Go to ID* and the containers.

    ``picked`` is emitted when the picked containers change (not on filtering).
    """

    picked = Signal()

    def __init__(self, run_io: RunIO, *, multiple: bool = True) -> None:
        super().__init__()
        self._run_io = run_io
        self._conn: Any = None
        self._all: list[tuple[str, int]] = []
        self._picks: list[int] = []
        self._quiet = False

        self.group = combo()
        self.owner = combo()
        self.container_type = combo()
        self.container_type.addItems(CONTAINER_TYPES)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by name or ID")
        self.search.setClearButtonEnabled(True)
        self.find = QPushButton("Go to ID")
        self.find.setToolTip(
            "Find this ID in any of your groups and switch to its group"
        )
        self.find.setEnabled(False)
        self.count = _muted()
        self.list = QListWidget()
        self.list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
            if multiple
            else QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list.setMinimumHeight(140)
        self.picked_info = _muted()

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Group", self.group)
        form.addRow("Owner", self.owner)
        form.addRow("Type", self.container_type)
        form.addRow("Search", hbox(self.search, self.find))
        form.addRow(self.count)
        form.addRow(self.list)
        if multiple:
            form.addRow(
                _muted("Ctrl/Shift-click to pick several for one training set.")
            )
        form.addRow(self.picked_info)

        self.group.activated.connect(lambda _: self._on_group())
        self.owner.activated.connect(lambda _: self._load())
        self.container_type.activated.connect(lambda _: self._on_type())
        self.search.textChanged.connect(lambda _: self._filter())
        self.search.returnPressed.connect(self._on_return)
        self.find.clicked.connect(self._find)
        self.list.itemSelectionChanged.connect(self._on_pick)
        self._show()

    # --- public -------------------------------------------------------------------

    @property
    def selected(self) -> list[int]:
        return list(self._picks)

    @property
    def kind(self) -> str:
        return self.container_type.currentText()

    def names(self) -> list[str]:
        """``name (ID n)`` of the picked containers."""
        known = dict((cid, name) for name, cid in self._all)
        return [container_label(known.get(c, "?"), c) for c in self._picks]

    def set_connection(self, conn: Any) -> None:
        """Read the groups, owners and the user's containers of the current type."""
        self._conn, self._picks, self._all = conn, [], []
        self._show()
        if conn is None:
            return
        kind = self.kind

        def read() -> tuple[list[tuple[str, int]], int, int, list, list]:
            groups = list_groups(conn)
            current = int(conn.getEventContext().groupId)
            me = int(conn.getUserId())
            if current not in [gid for _, gid in groups]:  # e.g. root in "system"
                groups.append((conn.getGroupFromContext().getName(), current))
            members = list_group_members(conn, current)
            return groups, current, me, members, list_containers(conn, kind, me)

        def show(result: tuple) -> None:
            groups, current, me, members, containers = result
            self._quiet = True
            self.group.clear()
            for name, gid in groups:
                self.group.addItem(name, gid)
            self.group.setCurrentIndex(self.group.findData(current))
            self._fill_owners(me, members)
            self._quiet = False
            self._all = containers
            self._show()
            self.picked.emit()

        self._run_io(read, show)

    def pick(self, cid: int) -> None:
        """Pick a container in the list (added to the picks)."""
        if self.list.selectionMode() == QAbstractItemView.SelectionMode.SingleSelection:
            self._picks = [cid]
        else:
            self._picks = list(dict.fromkeys([*self._picks, cid]))
        self._show()
        self.picked.emit()

    # --- data ---------------------------------------------------------------------

    def _fill_owners(self, me: int, members: list[tuple[str, int]]) -> None:
        self.owner.clear()
        self.owner.addItem("Me", me)
        self.owner.addItem("All members", ALL_MEMBERS)
        for name, uid in members:
            if uid != me:
                self.owner.addItem(name, uid)
        self.owner.setCurrentIndex(0)

    def _owner_filter(self) -> int | None:
        owner = self.owner.currentData()
        return None if owner in (None, ALL_MEMBERS) else int(owner)

    def _load(self, then: Callable[[], None] | None = None) -> None:
        if self._quiet or self._conn is None:
            return
        conn, kind, owner = self._conn, self.kind, self._owner_filter()
        self.count.setText("Loading…")

        def show(found: list[tuple[str, int]]) -> None:
            self._all = found
            known = {cid for _, cid in found}
            dropped = [cid for cid in self._picks if cid not in known]
            self._picks = [cid for cid in self._picks if cid in known]
            self._show()
            if dropped:
                self.picked.emit()
            if then is not None:
                then()

        self._run_io(lambda: list_containers(conn, kind, owner), show)

    def _on_group(self) -> None:
        gid = self.group.currentData()
        if self._quiet or self._conn is None or gid is None:
            return
        conn, kind = self._conn, self.kind

        def switch() -> tuple[int, list, list]:
            conn.setGroupForSession(int(gid))
            me = int(conn.getUserId())
            return (
                me,
                list_group_members(conn, int(gid)),
                list_containers(conn, kind, me),
            )

        def show(result: tuple) -> None:
            me, members, containers = result
            self._quiet = True
            self._fill_owners(me, members)
            self._quiet = False
            self._picks, self._all = [], containers
            self._show()
            self.picked.emit()

        self._run_io(switch, show)

    def _on_type(self) -> None:
        self._picks = []
        self._load()
        self.picked.emit()

    def _filter(self) -> None:
        text = self.search.text().strip()
        self.find.setEnabled(text.isdigit() and self._conn is not None)
        self._show()

    def _on_return(self) -> None:
        if self.find.isEnabled() and not self._matching():
            self._find()

    def _matching(self) -> list[tuple[str, int]]:
        text = self.search.text()
        return [(n, c) for n, c in self._all if matches_search(n, c, text)]

    def _show(self) -> None:
        """Fill the list: the picks first, then the matches of the search."""
        names = dict((cid, name) for name, cid in self._all)
        picks = set(self._picks)
        matching = self._matching()
        shown = [(container_label(names[c], c), c) for c in self._picks]
        shown += [(container_label(n, c), c) for n, c in matching if c not in picks]
        self._quiet = True
        self.list.clear()
        for label, cid in shown:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, cid)
            self.list.addItem(item)
            item.setSelected(cid in picks)
        self._quiet = False
        total, kind = len(self._all), self.kind
        if self._conn is None:
            self.count.setText("Connect to OMERO to browse its data.")
        elif self.search.text().strip():
            self.count.setText(f"{len(matching)} of {total} {kind}s")
        else:
            self.count.setText(f"{total} {kind}s")
        self._update_info()

    def _on_pick(self) -> None:
        if self._quiet:
            return
        chosen = [
            item.data(Qt.ItemDataRole.UserRole) for item in self.list.selectedItems()
        ]
        picks = [c for c in self._picks if c in chosen]
        picks += [c for c in chosen if c not in picks]
        if picks != self._picks:
            self._picks = picks
            self._update_info()
            self.picked.emit()

    def _update_info(self) -> None:
        names = dict((cid, name) for name, cid in self._all)
        self.picked_info.setText(
            "Picked: " + ", ".join(container_label(names[c], c) for c in self._picks)
            if len(self._picks) > 1
            else ""
        )

    def _find(self) -> None:
        conn, kind = self._conn, self.kind
        cid = int(self.search.text().strip())
        current = self.group.currentData()

        def look() -> tuple[int | None, Any]:
            group = find_container(conn, kind, cid)
            if group is None or group == current:
                return group, None
            conn.setGroupForSession(group)
            me = int(conn.getUserId())
            name = conn.getGroupFromContext().getName()
            return group, (me, name, list_group_members(conn, group))

        def found(result: tuple) -> None:
            group, switched = result
            if group is None:
                self.count.setText(f"No {kind} with ID {cid} that you can read.")
                return
            self._quiet = True
            if switched is not None:
                me, name, members = switched
                if self.group.findData(group) < 0:
                    self.group.addItem(name, group)
                self.group.setCurrentIndex(self.group.findData(group))
                self._fill_owners(me, members)
                self._picks = []
            self.owner.setCurrentIndex(self.owner.findData(ALL_MEMBERS))
            self.search.clear()
            self._quiet = False
            self._load(then=lambda: self.pick(cid))

        self._run_io(look, found)


class ChannelTable(QWidget):
    """Which channel to annotate, which to train on and which napari shows, and the
    intensity normalisation."""

    def __init__(self) -> None:
        super().__init__()
        self.label = combo()
        self.label.currentIndexChanged.connect(self._on_label)
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self._rows: list[ChannelRow] = []
        self._widgets: list[tuple[QLabel, QCheckBox, QCheckBox, QComboBox]] = []
        self._label = 0

        self.method = combo()
        self.method.addItem("Percentile", "percentile")
        self.method.addItem("None (raw)", "none")
        self.low = QDoubleSpinBox()
        self.high = QDoubleSpinBox()
        for box, value in ((self.low, 1.0), (self.high, 99.8)):
            box.setRange(0, 100)
            box.setDecimals(1)
            box.setSingleStep(0.1)
            box.setValue(value)
            box.setSuffix(" %")
        self.apply_at_export = QCheckBox("Apply at export (float32)")
        self.method.currentIndexChanged.connect(self._sync_method)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow("Annotate on", self.label)
        grid = QWidget()
        grid.setLayout(self.grid)
        layout.addRow(grid)
        layout.addRow(
            _muted(
                "Train on: the channels the model gets as input. Show: extra channels "
                "napari shows next to the annotated one (never used for training). "
                "Colour: how napari shows a channel, the annotated one included."
            )
        )
        layout.addRow("Normalisation", hbox(self.method, self.low, self.high))
        layout.addRow(self.apply_at_export)
        self.set_channels(["Channel 0"])

    def set_channels(self, names: list[str]) -> None:
        """One row per channel; earlier choices are reset."""
        label = self._label if self._label < len(names) else 0
        self._rows = channel_rows(names, label)
        while self.grid.count():
            widget = self.grid.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        for col, text in enumerate(("Channel", "Train on", "Show", "Colour")):
            self.grid.addWidget(QLabel(f"<b>{text}</b>"), 0, col)
        self._widgets = []
        for row in self._rows:
            title = QLabel()
            train, show = QCheckBox(), QCheckBox()
            colour = combo()
            colour.addItems(COLORMAPS)
            colour.setCurrentText(row.colormap)
            train.toggled.connect(lambda on, r=row: setattr(r, "train", on))
            show.toggled.connect(lambda on, r=row: self._on_show(r, on))
            colour.currentTextChanged.connect(
                lambda t, r=row: setattr(r, "colormap", t)
            )
            for col, widget in enumerate((title, train, show, colour)):
                self.grid.addWidget(widget, row.index + 1, col)
            self._widgets.append((title, train, show, colour))
        self.label.blockSignals(True)
        self.label.clear()
        for i, name in enumerate(names):
            self.label.addItem(f"{i}: {name}", i)
        self.label.setCurrentIndex(label)
        self.label.blockSignals(False)
        self._label = label
        self._sync_rows()

    def _on_show(self, row: ChannelRow, on: bool) -> None:
        row.show = on
        self._widgets[row.index][3].setEnabled(on or row.index == self._label)

    def _on_label(self, index: int) -> None:
        if index < 0 or index == self._label:
            return
        move_label(self._rows, self._label, index)
        self._label = index
        self._sync_rows()

    def _sync_rows(self) -> None:
        for row, (title, train, show, colour) in zip(
            self._rows, self._widgets, strict=True
        ):
            annotated = row.index == self._label
            title.setText(
                f"<b>{row.index}: {row.name}</b> (annotated)"
                if annotated
                else f"{row.index}: {row.name}"
            )
            for box, value in ((train, row.train), (show, row.show)):
                box.blockSignals(True)
                box.setChecked(value)
                box.blockSignals(False)
            colour.blockSignals(True)
            colour.setCurrentText(row.colormap)
            colour.blockSignals(False)
            show.setEnabled(not annotated)  # napari shows it anyway
            colour.setEnabled(row.show or annotated)

    def _sync_method(self) -> None:
        raw = self.method.currentData() == "none"
        self.low.setEnabled(not raw)
        self.high.setEnabled(not raw)

    @property
    def label_channel(self) -> int:
        return self._label

    @property
    def rows(self) -> list[ChannelRow]:
        return self._rows

    def normalization(self) -> NormalizationConfig:
        return NormalizationConfig(
            method=self.method.currentData(),
            percentile_low=self.low.value(),
            percentile_high=self.high.value(),
            apply_at_export=self.apply_at_export.isChecked(),
        )


class PlanesPicker(QWidget):
    """Specific planes (``0, 3, 5-7``), all, or ``n`` at random."""

    def __init__(self) -> None:
        super().__init__()
        self.mode = combo()
        for text, value in (
            ("Specific", "specific"),
            ("All", "all"),
            ("Random", "random"),
        ):
            self.mode.addItem(text, value)
        self.listed = QLineEdit("0")
        self.listed.setPlaceholderText("0, 3, 5-7")
        self.n = QSpinBox()
        self.n.setRange(1, 100_000)
        self.n.setPrefix("n = ")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.mode, self.listed, self.n):
            layout.addWidget(widget)
        self.mode.currentIndexChanged.connect(lambda _: self._sync())
        self._sync()

    def _sync(self) -> None:
        mode = self.mode.currentData()
        self.listed.setEnabled(mode == "specific")
        self.n.setEnabled(mode == "random")

    def planes(self) -> Planes:
        return Planes(self.mode.currentData(), self.listed.text(), self.n.value())


class _Condition(QWidget):
    def __init__(
        self,
        keys: list[tuple[str, str]],
        values: Callable[[str], list[tuple[str, str]]],
        changed: Callable[[], None],
        remove: Callable[["_Condition"], None],
    ) -> None:
        super().__init__()
        self._values = values
        self.key = combo()
        for label, key in keys:
            self.key.addItem(label, key)
        self.accepted = QListWidget()
        self.accepted.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.accepted.setMaximumHeight(90)
        self.remove = QPushButton("✕")
        self.remove.setToolTip("Remove this condition")
        self.remove.setFixedWidth(28)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        left = QVBoxLayout()
        left.addWidget(self.key)
        left.addWidget(self.accepted)
        layout.addLayout(left)
        layout.addWidget(self.remove, alignment=Qt.AlignmentFlag.AlignTop)
        self.key.currentIndexChanged.connect(lambda _: (self._fill(), changed()))
        self.accepted.itemSelectionChanged.connect(changed)
        self.remove.clicked.connect(lambda: remove(self))
        self._fill()

    def _fill(self) -> None:
        self.accepted.clear()
        key = self.key.currentData()
        for label, value in self._values(key) if key else []:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, value)
            self.accepted.addItem(item)

    def chosen(self) -> tuple[str, list[str]]:
        values = [
            i.data(Qt.ItemDataRole.UserRole) for i in self.accepted.selectedItems()
        ]
        return self.key.currentData(), values


class WellFilterBox(QWidget):
    """Conditions on well metadata (MIHCSME first) and whether matches are kept."""

    def __init__(self) -> None:
        super().__init__()
        self._wells: Mapping[int, Mapping[str, str]] = {}
        self._keys: list[tuple[str, str]] = []
        self._conditions: list[_Condition] = []
        self.mode = combo()
        self.mode.addItem("Keep matching wells", "include")
        self.mode.addItem("Leave matching wells out", "exclude")
        self.rows = QVBoxLayout()
        self.add = QPushButton("Add condition")
        self.summary = _muted()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mode)
        layout.addLayout(self.rows)
        layout.addWidget(self.add)
        layout.addWidget(self.summary)
        self.add.clicked.connect(self._add)
        self.mode.currentIndexChanged.connect(lambda _: self._update())
        self.load({}, set())

    def load(self, wells: Mapping[int, Mapping[str, str]], mihcsme: set[str]) -> None:
        self._wells = wells
        self._keys = well_condition_keys(wells, mihcsme)
        for condition in self._conditions:
            condition.deleteLater()
        self._conditions = []
        self.add.setEnabled(bool(self._keys))
        self._update()

    def _add(self) -> None:
        condition = _Condition(
            self._keys,
            lambda key: well_value_choices(self._wells, key),
            self._update,
            self._remove,
        )
        self._conditions.append(condition)
        self.rows.addWidget(condition)
        self._update()

    def _remove(self, condition: _Condition) -> None:
        self._conditions.remove(condition)
        condition.deleteLater()
        self._update()

    def _update(self) -> None:
        total = len(self._wells)
        if not total:
            self.summary.setText("Pick a plate or screen to filter its wells.")
            return
        if not self._keys:
            self.summary.setText(
                f"{total} wells; none has metadata to filter on (add it with MIHCSME)."
            )
            return
        filters = self.filters()
        if filters is None:
            self.summary.setText(f"All {total} wells")
            return
        kept = wells_kept(self._wells, filters, self.filter_mode)
        self.summary.setText(
            f"{kept} of {total} wells kept ({describe_filters(filters)})"
        )
        self.summary.setStyleSheet("color: #d1242f" if kept == 0 else "color: gray")

    def filters(self) -> dict[str, list[str]] | None:
        found: dict[str, list[str]] = {}
        for condition in self._conditions:
            key, values = condition.chosen()
            if key and values:
                kept = found.setdefault(key, [])
                kept += [v for v in values if v not in kept]
        return found or None

    @property
    def filter_mode(self) -> Literal["include", "exclude"]:
        return "exclude" if self.mode.currentData() == "exclude" else "include"


class NewSetPanel(QWidget):
    """The "New training set" form; :meth:`form` reads it as a :class:`NewSetForm`."""

    def __init__(self) -> None:
        super().__init__()
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. hela_nuclei_v1")
        self.channels = ChannelTable()
        self.timepoints = PlanesPicker()
        self.z_slices = PlanesPicker()
        self.volumetric = QCheckBox("3D: one volume over the chosen z-slices")
        self.patches = QCheckBox("Patches")
        self.patch_size = QSpinBox()
        self.patch_size.setRange(32, 8192)
        self.patch_size.setValue(512)
        self.patch_size.setPrefix("size ")
        self.per_image = QSpinBox()
        self.per_image.setRange(1, 1000)
        self.per_image.setPrefix("per image ")
        self.counts = {}
        for category, _ in SPLITS:
            box = QSpinBox()
            box.setRange(0, 100_000)
            box.setValue(DEFAULT_COUNTS[category])
            box.setToolTip(f"Images for {category}")
            self.counts[category] = box
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("random")
        self.storage = combo()
        self.storage.addItem("OMERO (table + uploads)", "omero")
        self.storage.addItem("Local folder only", "local")
        self.wells = WellFilterBox()
        self.wells_title = QLabel("<b>Plate wells</b> (filter by metadata)")

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow("Name", self.name)
        layout.addRow(QLabel("<b>Channels</b>"))
        layout.addRow(self.channels)
        layout.addRow(QLabel("<b>Images</b>"))
        layout.addRow("Timepoints", self.timepoints)
        layout.addRow("Z-slices", self.z_slices)
        layout.addRow(self.volumetric)
        layout.addRow(hbox(self.patches, self.patch_size, self.per_image))
        layout.addRow(
            "Train / val / test",
            hbox(*(self.counts[c] for c, _ in SPLITS)),
        )
        layout.addRow("Seed", self.seed)
        layout.addRow("Storage", self.storage)
        layout.addRow(self.wells_title)
        layout.addRow(self.wells)
        self.patches.toggled.connect(self._sync_patches)
        self.volumetric.toggled.connect(self._sync_patches)
        self._sync_patches()
        self.show_wells(False)

    def _sync_patches(self) -> None:
        on = self.patches.isChecked()
        self.patch_size.setEnabled(on)
        self.per_image.setEnabled(on and not self.volumetric.isChecked())

    def show_wells(self, visible: bool) -> None:
        self.wells_title.setVisible(visible)
        self.wells.setVisible(visible)

    def form(
        self,
        container_type: str,
        container_ids: list[int],
        model: str,
        batch_size: int,
    ) -> NewSetForm:
        return NewSetForm(
            name=self.name.text(),
            container_type=container_type,
            container_ids=container_ids,
            label_channel=self.channels.label_channel,
            channels=self.channels.rows,
            timepoints=self.timepoints.planes(),
            z_slices=self.z_slices.planes(),
            volumetric=self.volumetric.isChecked(),
            patch_size=self.patch_size.value() if self.patches.isChecked() else None,
            patches_per_image=self.per_image.value(),
            counts={c: box.value() for c, box in self.counts.items()},
            seed=self.seed.text(),
            storage=self.storage.currentData(),
            model=model,
            batch_size=batch_size,
            normalization=self.channels.normalization(),
            well_filters=self.wells.filters()
            if container_type in WELL_CONTAINERS
            else None,
            well_filter_mode=self.wells.filter_mode,
        )


def model_box() -> QComboBox:
    box = combo()
    for label, value in MODELS:
        box.addItem(label, value)
    return box
