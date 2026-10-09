"""Image sources: where the annotation pipeline reads pixels from."""

from .base import ImageMeta, ImageSource, Region, clamp_region
from .omero import OmeroImageSource
from .zarr import ZarrImage, ZarrImageSource

__all__ = [
    "ImageMeta",
    "ImageSource",
    "OmeroImageSource",
    "Region",
    "ZarrImage",
    "ZarrImageSource",
    "clamp_region",
]
