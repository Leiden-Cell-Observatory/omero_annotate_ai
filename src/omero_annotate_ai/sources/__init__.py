"""Image sources: where the annotation pipeline reads pixels from."""

from .base import ImageMeta, ImageSource, Region, clamp_region
from .omero import OmeroImageSource

__all__ = ["ImageMeta", "ImageSource", "OmeroImageSource", "Region", "clamp_region"]
