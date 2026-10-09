"""Image URIs: one string identity for images from any source.

OMERO images are ``omero:image:<id>``; OME-Zarr images are the URL (or local path) of
their image group. Annotation ids are derived from a short hash of the URI so files and
table rows stay linked to the image whatever the backend.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Optional

_OMERO_PREFIX = "omero:image:"


def omero_uri(image_id: int) -> str:
    """URI of an OMERO image."""
    return f"{_OMERO_PREFIX}{int(image_id)}"


def parse_omero_uri(uri: Any) -> Optional[int]:
    """OMERO image id of ``uri``, or ``None`` when it is not an OMERO URI."""
    if isinstance(uri, str) and uri.startswith(_OMERO_PREFIX):
        rest = uri[len(_OMERO_PREFIX):]
        if rest.isdigit():
            return int(rest)
    return None


def normalize_zarr_uri(uri: str) -> str:
    """Canonical form of a Zarr image URI: no query string or trailing slash for
    http(s) URLs, an absolute path with ``/`` separators for local stores (so plate,
    well and field URIs join the same way on Windows)."""
    if re.match(r"^https?://", uri):
        return uri.split("?", 1)[0].rstrip("/")
    return os.path.abspath(os.path.expanduser(uri)).replace(os.sep, "/")


def uri_key(uri: str) -> str:
    """Short stable key for ``uri``: the first 8 hex characters of its SHA-1."""
    return hashlib.sha1(uri.encode("utf-8")).hexdigest()[:8]


def annotation_id(
    uri: str,
    t: int,
    z: int,
    *,
    volumetric: bool = False,
    patch: Optional[int] = None,
) -> str:
    """Annotation id ``{key}_{t}_{z}`` (or ``{key}_{t}_3d``), plus ``_{patch}``."""
    base = f"{uri_key(uri)}_{t}_3d" if volumetric else f"{uri_key(uri)}_{t}_{z}"
    return base if patch is None else f"{base}_{patch}"
