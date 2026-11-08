"""Where a training set's files live on disk, and how they are written.

Every training set has one folder (its *bundle*) with the same layout in every
mode::

    config.yaml
    tracking_table.csv
    annotation_input/{annotation_id}.tif
    annotation_output/{annotation_id}_mask.tif
    .scratch/s{session}-b{batch}/      one batch at a time, removed afterwards

Bundles live under a home folder: ``<home>/<server>/<table_id>_<name>/`` for sets
tracked in an OMERO table, ``<home>/local/<id>_<name>/`` for local sets
(``storage: local``, e.g. OME-Zarr). Files are written atomically, so a crash
never leaves half a file.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import tifffile

HOME_ENV = "OMERO_ANNOTATE_AI_HOME"


def home_dir(home: str | Path | None = None) -> Path:
    """The folder holding all bundles: ``home``, else ``$OMERO_ANNOTATE_AI_HOME``,
    else ``~/omero_annotate_ai``."""
    if home is not None:
        return Path(home).expanduser()
    if env := os.environ.get(HOME_ENV):
        return Path(env).expanduser()
    return Path.home() / "omero_annotate_ai"


def _slug(text: str) -> str:
    """Lower-case, filesystem-safe: anything but letters, digits, ``.`` and ``_`` becomes ``-``."""
    return re.sub(r"[^a-z0-9._]+", "-", text.lower()).strip("-.") or "unnamed"


def server_slug(host: str, port: int | None = None) -> str:
    """A folder name for an OMERO server, e.g. ``omero.example.org-4064``.

    A URL host (``wss://example.org/omero-ws``) keeps its host and path, without the
    scheme.
    """
    host = re.sub(r"^[a-z]+://", "", host.strip().lower()).rstrip("/")
    return _slug(f"{host}-{port}" if port else host)


def omero_bundle(home: Path, server: str, table_id: int, name: str) -> Path:
    """Bundle of a set tracked in OMERO table ``table_id`` on ``server``."""
    return home / server / f"{table_id}_{_slug(name)}"


def new_local_id() -> str:
    """ID of a new set kept only on disk (``storage: local``)."""
    return uuid.uuid4().hex[:8]


def local_bundle(home: Path, local_id: str, name: str) -> Path:
    """Bundle of a set kept only on disk (``storage: local``, e.g. OME-Zarr)."""
    return home / "local" / f"{local_id}_{_slug(name)}"


def atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """Write ``path`` via ``write(temp_path)``, then swap it in, so readers never see
    a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    try:
        write(temp)
        with temp.open("r+b") as handle:  # Windows needs write access to fsync
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    if os.name == "posix":  # make the rename itself durable
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def fresh_scratch(bundle: Path, session: int, batch: int) -> Path:
    """An empty scratch folder for one batch (left-overs of a crash are removed)."""
    scratch = bundle / ".scratch" / f"s{session}-b{batch}"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    return scratch


def collect_masks(mask_dir: Path, annotation_ids: Sequence[str]) -> dict[str, Path]:
    """Mask files written for a batch, by exact name: ``seg_{i:05}.tif`` (or
    ``.tiff``) is the mask of item ``i``. Missing files are units left unannotated."""
    masks: dict[str, Path] = {}
    for position, annotation_id in enumerate(annotation_ids):
        for suffix in (".tif", ".tiff"):
            path = mask_dir / f"seg_{position:05}{suffix}"
            if path.is_file():
                masks[annotation_id] = path
                break
    return masks


def input_path(bundle: Path, annotation_id: str) -> Path:
    """Where a unit's annotated image is kept: ``annotation_input/{id}.tif``."""
    return bundle / "annotation_input" / f"{annotation_id}.tif"


def mask_path(bundle: Path, annotation_id: str) -> Path:
    """Where the mask of ``annotation_id`` is kept."""
    return bundle / "annotation_output" / f"{annotation_id}_mask.tif"


def conflict_path(bundle: Path, annotation_id: str, session: int) -> Path:
    """Where a mask is kept that lost to another session's upload of the same unit."""
    return (
        bundle
        / "annotation_output"
        / "conflicts"
        / f"{annotation_id}_session{session}.tif"
    )


def pixel_digest(pixels: np.ndarray) -> str:
    """SHA-256 of an image's pixels: data type, shape and values, not the file.

    A mask saved locally and the same mask uploaded zlib-compressed to OMERO have
    different bytes on disk but the same digest.
    """
    array = np.ascontiguousarray(pixels)
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    digest = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """:func:`pixel_digest` of a TIFF file."""
    return pixel_digest(np.asarray(tifffile.imread(path)))
