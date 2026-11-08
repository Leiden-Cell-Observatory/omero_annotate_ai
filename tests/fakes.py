"""In-memory stand-ins for OMERO and image sources, shared by the unit tests.

:class:`FakeOmero` replaces every OMERO call the pipeline makes (it subclasses the
pipeline's :class:`OmeroWrites` seam): a tracking table, label files and ROIs with
their tags. It can fail a call (``fail``) or simulate the process dying mid-call
(``crash``), which raises :class:`Crash` - a ``BaseException`` the pipeline does not
catch, like a killed kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import Mock

import numpy as np
import pandas as pd

from omero_annotate_ai.annotators import ScriptedAnnotator
from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    create_default_config,
)
from omero_annotate_ai.rows import rows_to_frame
from omero_annotate_ai.sources import ImageMeta
from omero_annotate_ai.sources.base import clamp_region
from omero_annotate_ai.sources.uri import omero_uri, parse_omero_uri
from omero_annotate_ai.trainingset import OmeroWrites, TrainingSet


class Crash(BaseException):
    """The process died here (not caught by the pipeline's error handling)."""


class FakeImage:
    """Handle with no OMERO methods: proves the pipeline only talks to the source."""

    def __init__(self, image_id, name):
        self.image_id = image_id
        self.name = name


class FakeImageSource:
    """In-memory ImageSource over (t, c, z, y, x) arrays; records read calls."""

    def __init__(self, arrays):
        self.arrays = arrays  # {image_id: ndarray (t, c, z, y, x)}
        self.reads = []

    def get_image(self, image_id):
        if isinstance(image_id, str):
            image_id = parse_omero_uri(image_id)
        if image_id not in self.arrays:
            raise LookupError(image_id)
        return FakeImage(image_id, f"img{image_id}")

    def metadata(self, image):
        t, c, z, y, x = self.arrays[image.image_id].shape
        return ImageMeta(
            omero_uri(image.image_id),
            image.name,
            t,
            c,
            z,
            y,
            x,
            str(self.arrays[image.image_id].dtype),
        )

    def read_plane(self, image, *, t, c, z, region=None):
        self.reads.append(("plane", image.image_id, t, c, z, region))
        plane = self.arrays[image.image_id][t, c, z]
        if region is not None:
            x, y, w, h = clamp_region(region, *plane.shape)
            plane = plane[y : y + h, x : x + w]
        return plane

    def read_volume(self, image, *, t, c, z_start, z_end, region=None):
        self.reads.append(("volume", image.image_id, t, c, (z_start, z_end), region))
        return np.stack(
            [
                self.read_plane(image, t=t, c=c, z=z, region=region)
                for z in range(z_start, z_end + 1)
            ]
        )


@dataclass
class Upload:
    image_id: int
    tag: str
    pixels: np.ndarray | None = None


@dataclass
class FakeOmero(OmeroWrites):
    """OMERO in memory: one tracking table, label files and ROIs with their tags.

    ``fail`` / ``crash`` map a call name (``label``, ``label_after``, ``roi``,
    ``roi_after``, ``table``, ``read``, ``cells``) to the 1-based call number that fails with a
    ``ConnectionError`` / dies with :class:`Crash`. ``*_after`` fire once the object
    exists on the server, i.e. the upload happened but the caller never heard back.
    """

    fail: dict[str, int] = field(default_factory=dict)
    crash: dict[str, int] = field(default_factory=dict)
    tables: dict[int, pd.DataFrame] = field(default_factory=dict)
    labels: dict[int, Upload] = field(default_factory=dict)
    rois: dict[int, Upload] = field(default_factory=dict)
    configs: list[int] = field(default_factory=list)
    config_text: dict[int, str] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    _next_id: int = 100

    def conn(self) -> Mock:
        """A connection whose getObject finds the tables, labels and ROIs."""
        conn = Mock()

        def get_object(kind, object_id):
            store = {
                "FileAnnotation": {**self.labels, **dict.fromkeys(self.tables, True)},
                "Roi": self.rois,
            }.get(kind, {})
            return Mock() if int(object_id) in store else None

        conn.getObject.side_effect = get_object
        conn.host, conn.port = "fake.omero", 4064
        conn.getUser.return_value.getName.return_value = "tester"
        return conn

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _call(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.crash.get(name) == self.calls[name]:
            raise Crash(name)
        if self.fail.get(name) == self.calls[name]:
            raise ConnectionError(f"simulated {name} failure")

    # tables -------------------------------------------------------------------
    def create_table(self, conn, rows, title, container_type, container_ids):
        table_id = self._new_id()
        self.tables[table_id] = rows_to_frame(rows)
        return table_id

    def read_table(self, conn, table_id):
        self._call("read")
        return self.tables[table_id].copy()

    def write_rows(self, conn, table_id, rows):
        self._call("table")
        new = rows_to_frame(rows).set_index("annotation_id")
        old = self.tables[table_id].set_index("annotation_id")
        added = len(new.index.difference(old.index))
        merged = pd.concat([old.drop(new.index, errors="ignore"), new])
        self.tables[table_id] = merged.reset_index()
        return {"updated": len(new) - added, "added": added}

    def read_rows(self, conn, table_id, keys):
        self._call("read")
        table = self.tables[table_id]
        wanted = [str(k) for k in keys]
        return table[table["annotation_id"].astype(str).isin(wanted)].copy()

    def update_cells(self, conn, table_id, keys, change):
        self._call("cells")
        table = self.tables[table_id]
        position = {str(k): i for i, k in enumerate(table["annotation_id"])}
        changed = []
        for key in (str(k) for k in keys):
            if key not in position:
                continue
            cells = change(table.iloc[position[key]].to_dict())
            if cells:
                for name, value in cells.items():
                    table.loc[table.index[position[key]], name] = value
                changed.append(key)
        return changed

    def replace_config(self, conn, table_id, file_path, container_type, container_id):
        self.configs.append(table_id)
        with open(file_path, encoding="utf-8") as handle:
            self.config_text[table_id] = handle.read()
        return self._new_id()

    def read_config(self, conn, table_id):
        text = self.config_text.get(int(table_id))
        return None if text is None else AnnotationConfig.from_yaml(text)

    # uploads ------------------------------------------------------------------
    def upload_label(self, conn, image_id, label_img, **fields):
        self._call("label")
        label_id = self._new_id()
        self.labels[label_id] = Upload(int(image_id), fields["tag"], label_img.copy())
        self._call("label_after")
        return label_id

    def upload_roi(self, conn, image_id, label_img, **fields):
        self._call("roi")
        if not np.any(label_img):
            return None
        roi_id = self._new_id()
        self.rois[roi_id] = Upload(int(image_id), fields["tag"])
        self._call("roi_after")
        return roi_id

    def find_tagged(self, conn, image_id, tag):
        def ids(store):
            return sorted(
                i for i, u in store.items() if u.image_id == image_id and u.tag == tag
            )

        return ids(self.labels), ids(self.rois)

    def download_label(self, conn, label_id):
        upload = self.labels.get(int(label_id))
        return None if upload is None else upload.pixels

    def object_exists(self, conn, kind, object_id):
        store = self.rois if kind == "Roi" else self.labels
        return int(object_id) in store

    def delete_objects(self, conn, kind, ids):
        store = self.rois if kind == "Roi" else self.labels
        for i in ids:
            store.pop(int(i), None)

    # checks -------------------------------------------------------------------
    def uploads_per_tag(self) -> dict[str, tuple[int, int]]:
        """tag -> (number of label files, number of ROIs)."""
        counts: dict[str, list[int]] = {}
        for upload in self.labels.values():
            counts.setdefault(upload.tag, [0, 0])[0] += 1
        for upload in self.rois.values():
            counts.setdefault(upload.tag, [0, 0])[1] += 1
        return {tag: (n[0], n[1]) for tag, n in counts.items()}


# --- Training sets on the fakes -------------------------------------------------------


def omero_set(tmp_path, n_images=3):
    """A dataset of ``n_images`` 16x16 images (pixel value = image ID), all training."""
    source = FakeImageSource(
        {
            i: np.full((1, 1, 1, 16, 16), i, dtype=np.uint16)
            for i in range(1, n_images + 1)
        }
    )
    config = create_default_config()
    config.name = "fake_set"
    config.omero.container_type = "dataset"
    config.omero.container_id = 1
    config.training.train_n = n_images
    config.training.validate_n = 0
    config.training.test_n = 0
    return config, source, FakeOmero()


def image_mask(pixels):
    """The scripted user's mask: the whole plane, labelled with the image's ID."""
    return np.full(np.asarray(pixels).shape, int(np.asarray(pixels).max()), np.uint16)


def create_set(config, source, omero, home, events=None, **kwargs) -> TrainingSet:
    """A new training set on the fakes (home folder ``home``), over all images."""
    kwargs.setdefault("images", [source.get_image(i) for i in sorted(source.arrays)])
    return TrainingSet.create(
        config,
        None if config.storage == "local" else omero.conn(),
        home=home,
        image_source=source,
        omero_writes=omero,
        events=None if events is None else events.append,
        **kwargs,
    )


def reopen(ts, source, omero, events=None, **kwargs) -> TrainingSet:
    """The same set opened again by its table ID (repairs it first)."""
    return TrainingSet.open(
        ts.table_id,
        omero.conn(),
        home=ts.home,
        image_source=source,
        omero_writes=omero,
        events=None if events is None else events.append,
        **kwargs,
    )


def annotate(ts, limit=None, **kwargs) -> int:
    """Annotate the pending units like a scripted user (``limit`` units at most)."""
    return ts.annotate(ScriptedAnnotator(image_mask, limit=limit), **kwargs)
