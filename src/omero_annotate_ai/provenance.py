"""ISCC content codes: how similar two images or masks are, not only whether equal.

The SHA-256 pixel digests in the tracking table (:func:`~omero_annotate_ai.bundle.
pixel_digest`) answer "is this exactly the image that was annotated?". An ISCC
Image-Code (ISO 24138) answers "how much does it differ?": codes of similar
images differ in few of their 64 bits. Together they tell a user whether a mask
that changed after upload was touched up (a near-duplicate) or replaced by
something else.

Codes are computed only when the optional extra is installed
(``pip install 'omero-annotate-ai[provenance]'``, which brings ``iscc-lib``);
without it every function here returns ``None`` and nothing else changes.

A plane is normalised before coding: several planes (a volume) are reduced by a
maximum projection, the plane is area-averaged to the 32 x 32 grayscale pixels the
Image-Code takes (colour: mean of its channels), and intensities are stretched to
0-255 between the full plane's minimum and maximum (masks: background 0, every
object 255, so renumbering objects keeps the code). Uniform images all get the
same code, so for them only the SHA-256 digest says anything. Codes are comparable
between images coded by this package; they are not meant to match codes another
tool computed from the image file.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
import tifffile

# Bits (of 64) two Image-Codes may differ in and still count as a near-duplicate.
NEAR_DUPLICATE_BITS = 12

_SIDE = 32  # the Image-Code works on 32 x 32 grayscale pixels


class _IsccLib(Protocol):
    def gen_image_code_v0(self, pixels: bytes, bits: int = 64) -> dict[str, str]: ...

    def iscc_decode(self, iscc: str) -> tuple[object, object, object, int, bytes]: ...


def _iscc_lib() -> _IsccLib | None:
    try:
        return cast(_IsccLib, importlib.import_module("iscc_lib"))
    except ImportError:
        return None


def available() -> bool:
    """True if the ``provenance`` extra (``iscc-lib``) is installed."""
    return _iscc_lib() is not None


# Data types cv2.resize takes; anything else (bool, int32, ...) goes via float32.
_RESIZABLE = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def _plane(pixels: np.ndarray) -> np.ndarray:
    """A 2D (or YXC colour) plane in its own data type; volumes by max projection."""
    array = np.squeeze(np.asarray(pixels))
    colour = array.ndim == 3 and array.shape[-1] in (3, 4)
    while array.ndim > (3 if colour else 2):
        array = array.max(axis=0)
    if colour:
        array = array[..., :3]
    return np.ascontiguousarray(np.atleast_2d(array))


def _shrink(plane: np.ndarray) -> np.ndarray:
    """``plane`` area-averaged to 32 x 32 (colour: mean of the channels), as float."""
    if plane.dtype.type not in _RESIZABLE:
        plane = plane.astype(np.float32)
    small = cv2.resize(plane, (_SIDE, _SIDE), interpolation=cv2.INTER_AREA)
    small = np.asarray(small, dtype=np.float64)
    return small.mean(axis=-1) if small.ndim == 3 else small


def normalised(pixels: np.ndarray, *, mask: bool = False) -> np.ndarray:
    """The 32 x 32 uint8 plane an image (or ``mask``) is coded from.

    The image is reduced in its own data type first and stretched afterwards, so a
    large plane is never copied as floats (area averaging and stretching commute).
    """
    plane = _plane(pixels)
    if mask:
        small = _shrink(np.multiply(plane > 0, 255, dtype=np.uint8))
    else:
        low, high = float(plane.min()), float(plane.max())
        span = high - low
        small = _shrink(plane)
        small = (small - low) * (255.0 / span) if span > 0 else np.zeros_like(small)
    return np.clip(np.rint(small), 0, 255).astype(np.uint8)


def image_code(pixels: np.ndarray, *, mask: bool = False) -> str | None:
    """The ISCC Image-Code of an image (or a label ``mask``).

    None without the extra, or for an empty array.
    """
    lib = _iscc_lib()
    if lib is None or np.asarray(pixels).size == 0:
        return None
    return lib.gen_image_code_v0(normalised(pixels, mask=mask).tobytes())["iscc"]


def file_code(path: Path, *, mask: bool = False) -> str | None:
    """:func:`image_code` of a TIFF file; None without the extra (file not read)."""
    if not available():
        return None
    return image_code(np.asarray(tifffile.imread(path)), mask=mask)


def distance(a: str, b: str) -> int | None:
    """Bits in which two Image-Codes differ (0-64).

    None without the extra, or when a code cannot be decoded.
    """
    lib = _iscc_lib()
    if lib is None:
        return None
    try:
        da, db = lib.iscc_decode(a)[-1], lib.iscc_decode(b)[-1]
    except Exception:  # noqa: BLE001 - a damaged recorded code is reported, not fatal
        return None
    return (int.from_bytes(da, "big") ^ int.from_bytes(db, "big")).bit_count()


def _featureless(code: str) -> bool:
    """True for the code of a uniform plane (blank, or a mask without background)."""
    lib = _iscc_lib()
    if lib is None:
        return False
    uniform = (bytes(_SIDE * _SIDE), bytes([255]) * (_SIDE * _SIDE))
    return code in {lib.gen_image_code_v0(plane)["iscc"] for plane in uniform}


def describe_change(recorded: str | None, pixels: np.ndarray, *, mask: bool) -> str:
    """How pixels whose digest no longer matches relate to the recorded code.

    Returns ``""`` when the codes cannot tell (no recorded code, the extra is not
    installed, an empty or uniform image, or a damaged code): the SHA-256 digest
    alone then says the file changed. Otherwise a short phrase for a message.
    """
    if not recorded:
        return ""
    current = image_code(pixels, mask=mask)
    if current is None or _featureless(current) or _featureless(recorded):
        return ""
    bits = distance(recorded, current)
    if bits is None:
        return ""
    if bits == 0:
        return "a near-duplicate: small edit, same ISCC"
    if bits <= NEAR_DUPLICATE_BITS:
        return f"a near-duplicate: ISCC differs in {bits} of 64 bits"
    return f"different content: ISCC differs in {bits} of 64 bits"
