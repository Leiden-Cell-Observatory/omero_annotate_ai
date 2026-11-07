"""ImageSource for OME-Zarr images (NGFF 0.4 and 0.5), read through ome-zarr-py.

Handles are :class:`ZarrImage`: the dask array of one multiscale level in the file's
own axis order, plus the axis names. Pixel coordinates are in that level's grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import ImageMeta, Region, clamp_region
from .uri import normalize_zarr_uri


def load_ome_zarr():
    """Import ome-zarr lazily so an OMERO-only install works without it."""
    try:
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Multiscales, Reader
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading OME-Zarr needs ome-zarr: pip install 'omero-annotate-ai[zarr]'"
        ) from exc
    # parse_url logs "version mismatch" for every NGFF 0.4 store; it is harmless.
    logging.getLogger("ome_zarr.io").setLevel(logging.ERROR)
    return parse_url, Reader, Multiscales


@dataclass(frozen=True)
class ZarrImage:
    uri: str
    name: str
    level: int
    data: Any  # dask array of the chosen level
    axes: Tuple[str, ...]


def _default_name(uri: str, metadata_name: Optional[str]) -> str:
    name = (metadata_name or "").strip("/")
    if name:
        return name.replace("/", "_")
    return uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _check_index(axis: str, index: int, size: int) -> None:
    if not 0 <= index < size:
        raise IndexError(f"{axis}={index} is outside the image ({axis} size {size})")


class ZarrImageSource:
    """Read OME-Zarr images at one multiscale ``level``."""

    def __init__(self, level: int = 0):
        self.level = level
        self._cache: Dict[str, ZarrImage] = {}

    def get_image(self, uri: str) -> ZarrImage:
        uri = normalize_zarr_uri(uri)
        if uri not in self._cache:
            self._cache[uri] = self._open(uri)
        return self._cache[uri]

    def _open(self, uri: str) -> ZarrImage:
        parse_url, Reader, Multiscales = load_ome_zarr()
        location = parse_url(uri)
        if location is None:
            raise LookupError(f"No OME-Zarr found at {uri}")
        if "multiscales" not in location.root_attrs:
            raise LookupError(
                f"{uri} is not an image (keys: {sorted(location.root_attrs)})"
            )
        node = next(
            n
            for n in Reader(location)()
            if any(isinstance(s, Multiscales) for s in n.specs)
        )
        levels = node.data
        if self.level >= len(levels):
            raise ValueError(
                f"Level {self.level} requested but {uri} has {len(levels)} levels"
            )
        axes = tuple(
            a["name"] if isinstance(a, dict) else str(a)
            for a in node.metadata.get("axes", [])
        )
        data = levels[self.level]
        if len(axes) != data.ndim:
            raise ValueError(f"{uri} has no usable axes metadata (NGFF >= 0.4 needed)")
        return ZarrImage(
            uri, _default_name(uri, node.metadata.get("name")), self.level, data, axes
        )

    def metadata(self, image: ZarrImage) -> ImageMeta:
        sizes = dict(zip(image.axes, image.data.shape))
        return ImageMeta(
            image_uri=image.uri,
            name=image.name,
            size_t=sizes.get("t", 1),
            size_c=sizes.get("c", 1),
            size_z=sizes.get("z", 1),
            size_y=sizes["y"],
            size_x=sizes["x"],
            dtype=str(image.data.dtype),
        )

    def _index(
        self, image: ZarrImage, t: int, c: int, z: Any, region: Optional[Region]
    ):
        sizes = dict(zip(image.axes, image.data.shape))
        _check_index("t", t, sizes.get("t", 1))
        _check_index("c", c, sizes.get("c", 1))
        for z_value in (z.start, z.stop - 1) if isinstance(z, slice) else (z,):
            _check_index("z", z_value, sizes.get("z", 1))
        if region is None:
            ys, xs = slice(None), slice(None)
        else:
            x, y, width, height = clamp_region(region, sizes["y"], sizes["x"])
            ys, xs = slice(y, y + height), slice(x, x + width)
        picks = {"t": t, "c": c, "z": z, "y": ys, "x": xs}
        return tuple(picks[axis] for axis in image.axes)

    def read_plane(
        self,
        image: ZarrImage,
        *,
        t: int,
        c: int,
        z: int,
        region: Optional[Region] = None,
    ) -> np.ndarray:
        return np.asarray(image.data[self._index(image, t, c, z, region)].compute())

    def read_volume(
        self,
        image: ZarrImage,
        *,
        t: int,
        c: int,
        z_start: int,
        z_end: int,
        region: Optional[Region] = None,
    ) -> np.ndarray:
        volume = np.asarray(
            image.data[
                self._index(image, t, c, slice(z_start, z_end + 1), region)
            ].compute()
        )
        return volume if "z" in image.axes else volume[np.newaxis]
