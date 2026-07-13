"""ISCC content provenance for published annotation datasets.

Computes ISO 24138:2024 content codes via iscc-bio's IMAGEWALK traversal.

IMAGEWALK is an *exact* code: it packs raw dtype bytes with no bit-depth or
intensity canonicalization, aiming for bit-for-bit identity with OMERO's own
pixel representation. It therefore survives a change of container format
(OME-TIFF / OME-Zarr / OMERO) but NOT a change of pixel values.

Consequence: codes are computed on raw source pixels only. The exported 8-bit
training tiles are rescaled (img * 255/max) and channel-restacked, so their
codes would not match the source, and they are never coded here.

iscc-bio is an optional dependency and is imported lazily. Everything in this
module degrades to None with a warning when it is absent.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_MISSING_MSG = (
    "iscc-bio is not installed; skipping ISCC provenance. "
    "Install it with: pip install 'omero-annotate-ai[provenance]'"
)


def _get_biocode():
    """Return iscc_bio.api.biocode, or None if iscc-bio is not installed."""
    try:
        from iscc_bio.api import biocode
    except ImportError:
        return None
    return biocode


def iscc_available() -> bool:
    """Whether iscc-bio is importable and provenance can be computed."""
    return _get_biocode() is not None


def _first_code(results: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Take the ISCC of the first scene.

    biocode() returns one entry per scene. OMERO images and the exported masks
    are single-scene, so the first entry is the one we want.
    """
    if not results:
        return None
    return results[0].get("iscc_code")


def compute_file_iscc(path: Union[str, Path]) -> Optional[str]:
    """Content code for a local image file. Returns None on any failure."""
    biocode = _get_biocode()
    if biocode is None:
        logger.warning(_MISSING_MSG)
        return None

    try:
        return _first_code(biocode(source=str(path)))
    except Exception as exc:  # provenance is best-effort, never fatal
        logger.warning(f"Could not compute ISCC for file {path}: {exc}")
        return None


def compute_image_iscc(conn, image_id: int) -> Optional[str]:
    """Content code for an OMERO image, from its raw pixels.

    This is the canonical source-side code: iscc-bio reads OMERO's pixel data
    directly, so it is directly comparable to any faithful copy of the image.
    """
    biocode = _get_biocode()
    if biocode is None:
        logger.warning(_MISSING_MSG)
        return None

    try:
        return _first_code(biocode(conn=conn, iid=int(image_id)))
    except Exception as exc:
        logger.warning(f"Could not compute ISCC for OMERO image {image_id}: {exc}")
        return None


def compute_label_iscc(conn, label_id: int) -> Optional[str]:
    """Content code for an annotation mask.

    Masks are stored as OMERO file annotations, so this downloads the mask to a
    temporary directory and codes it as a file.
    """
    biocode = _get_biocode()
    if biocode is None:
        logger.warning(_MISSING_MSG)
        return None

    import ezomero

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_path = ezomero.get_file_annotation(
                conn, int(label_id), folder_path=tmp_dir
            )
            if mask_path is None:
                logger.warning(f"Label file annotation {label_id} could not be fetched")
                return None
            return compute_file_iscc(mask_path)
    except Exception as exc:
        logger.warning(f"Could not compute ISCC for label {label_id}: {exc}")
        return None


def stamp_config(config, conn):
    """Fill in source_iscc and label_iscc for every annotation in the config.

    Safe to re-run: codes that are already present are left alone, so this can
    be pointed at an existing config.yaml from a previous run to add provenance
    retroactively.

    Each unique image_id and label_id is coded once and cached - a source image
    is typically shared by many annotation rows (patches, z-slices, timepoints),
    and coding it per row would be wasteful.

    Args:
        config: AnnotationConfig to stamp, mutated in place.
        conn: OMERO BlitzGateway connection.

    Returns:
        The same config, for chaining.
    """
    if not iscc_available():
        logger.warning(_MISSING_MSG)
        return config

    image_cache: Dict[int, Optional[str]] = {}
    label_cache: Dict[int, Optional[str]] = {}

    for annotation in config.annotations:
        if annotation.source_iscc is None:
            image_id = annotation.image_id
            if image_id not in image_cache:
                image_cache[image_id] = compute_image_iscc(conn, image_id)
            annotation.source_iscc = image_cache[image_id]

        if annotation.label_iscc is None and annotation.label_id is not None:
            label_id = annotation.label_id
            if label_id not in label_cache:
                label_cache[label_id] = compute_label_iscc(conn, label_id)
            annotation.label_iscc = label_cache[label_id]

    coded = sum(1 for a in config.annotations if a.source_iscc is not None)
    logger.info(
        f"Stamped ISCC provenance: {coded}/{len(config.annotations)} annotations, "
        f"{len(image_cache)} unique source image(s)"
    )
    return config
