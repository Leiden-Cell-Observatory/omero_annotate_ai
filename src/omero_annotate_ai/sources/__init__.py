"""Image sources: where the annotation pipeline reads pixels from."""

from .base import ImageMeta, ImageSource, Region, clamp_region
from .omero import OmeroImageSource
from .zarr import ZarrImage, ZarrImageSource


def make_image_source(config, conn=None) -> ImageSource:
    """The ImageSource a config asks for: OME-Zarr for ``source: zarr``, else OMERO."""
    if getattr(config, "source", "omero") == "zarr":
        return ZarrImageSource(level=config.zarr.level)
    return OmeroImageSource(conn)


__all__ = [
    "ImageMeta",
    "ImageSource",
    "OmeroImageSource",
    "Region",
    "ZarrImage",
    "ZarrImageSource",
    "clamp_region",
    "make_image_source",
]
