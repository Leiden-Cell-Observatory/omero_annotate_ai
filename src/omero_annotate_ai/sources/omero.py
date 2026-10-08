"""ImageSource backed by an OMERO BlitzGateway connection."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import ImageMeta, Region, clamp_region

# OMERO pixel type names that differ from numpy's (omero.gateway.OMERO_NUMPY_TYPES).
_OMERO_TO_NUMPY_DTYPE = {"float": "float32", "double": "float64", "bit": "bool"}


class OmeroImageSource:
    """Read images from OMERO. Handles are BlitzGateway ``ImageWrapper`` objects."""

    def __init__(self, conn: Any):
        self.conn = conn

    def get_image(self, image_id: int) -> Any:
        if self.conn is None:
            raise ValueError("OMERO connection is not set.")
        image = self.conn.getObject("Image", image_id)
        if image is None:
            raise LookupError(f"OMERO image {image_id} not found or not accessible")
        return image

    def metadata(self, image: Any) -> ImageMeta:
        return ImageMeta(
            image_id=image.getId(),
            name=image.getName(),
            size_t=image.getSizeT(),
            size_c=image.getSizeC(),
            size_z=image.getSizeZ(),
            size_y=image.getSizeY(),
            size_x=image.getSizeX(),
            dtype=_OMERO_TO_NUMPY_DTYPE.get(image.getPixelsType(), str(image.getPixelsType())),
        )

    def read_plane(
        self, image: Any, *, t: int, c: int, z: int, region: Optional[Region] = None
    ) -> np.ndarray:
        pixels = image.getPrimaryPixels()
        if region is None:
            return np.asarray(pixels.getPlane(z, c, t))
        tile = clamp_region(region, image.getSizeY(), image.getSizeX())
        return np.asarray(pixels.getTile(z, c, t, tile))

    def read_volume(
        self,
        image: Any,
        *,
        t: int,
        c: int,
        z_start: int,
        z_end: int,
        region: Optional[Region] = None,
    ) -> np.ndarray:
        return np.stack(
            [
                self.read_plane(image, t=t, c=c, z=z, region=region)
                for z in range(z_start, z_end + 1)
            ]
        )
