"""Backend-neutral interface for reading images.

The annotation pipeline reads pixels and image sizes only through an ``ImageSource``.
An image is identified by an opaque *handle* that the source hands out from
``get_image`` - an OMERO ``ImageWrapper`` for :class:`OmeroImageSource` - so callers
never depend on what a handle is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

# (x, y, width, height) in pixels, as OMERO's getTile takes it.
Region = tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageMeta:
    """Name and 5D size of one image."""

    image_uri: str  # omero:image:<id> or the OME-Zarr image URL
    name: str
    size_t: int
    size_c: int
    size_z: int
    size_y: int
    size_x: int
    dtype: str  # numpy dtype name, e.g. "uint16" or "float32"


def clamp_region(region: Region, size_y: int, size_x: int) -> Region:
    """Clip ``region`` to the image, as numpy slicing would.

    Raises:
        ValueError: If the region starts outside the image.
    """
    x, y, width, height = region
    if not (0 <= x < size_x and 0 <= y < size_y):
        raise ValueError(
            f"Region {region} starts outside the image ({size_x}x{size_y})"
        )
    return x, y, min(width, size_x - x), min(height, size_y - y)


@runtime_checkable
class ImageSource(Protocol):
    """Where the pipeline gets images from."""

    def get_image(self, image_id: Any) -> Any:
        """Return the handle for ``image_id``; raise ``LookupError`` if it does not exist."""
        ...

    def metadata(self, image: Any) -> ImageMeta:
        """Return the name and sizes of ``image``."""
        ...

    def read_plane(
        self, image: Any, *, t: int, c: int, z: int, region: Region | None = None
    ) -> np.ndarray:
        """Return one plane as ``(y, x)``, cropped to ``region`` when given."""
        ...

    def read_volume(
        self,
        image: Any,
        *,
        t: int,
        c: int,
        z_start: int,
        z_end: int,
        region: Region | None = None,
    ) -> np.ndarray:
        """Return planes ``z_start..z_end`` (inclusive) as ``(z, y, x)``."""
        ...
