"""JSON manifests and GeoJSON annotations for the BIOMERO OMERO.web plugin.

A manifest is an ``AnnotationConfig`` stored as a JSON FileAnnotation on a container;
GeoJSON annotations are stored per image. Both are keyed by a ``set_id`` (not by the
tracking table) and are written by the BIOMERO web annotation plugin, which is still
in development; nothing in this package calls them.
"""

import contextlib
import json
import random
import string
import tempfile
from datetime import datetime
from pathlib import Path

from ..core.annotation_config import AnnotationConfig

# Namespace prefixes for JSON manifest and GeoJSON storage
MANIFEST_NS_PREFIX = "omero.biomero.manifest."
GEOJSON_NS_PREFIX = "omero.biomero.annotations."


def generate_set_id() -> str:
    """Generate a unique set_id: timestamp with milliseconds + 4-char random suffix."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{suffix}"


def manifest_namespace(set_id: str) -> str:
    """Return the OMERO namespace for a manifest FileAnnotation."""
    return f"{MANIFEST_NS_PREFIX}{set_id}"


def geojson_namespace(set_id: str) -> str:
    """Return the OMERO namespace for GeoJSON FileAnnotations."""
    return f"{GEOJSON_NS_PREFIX}{set_id}"


def save_manifest_to_omero(
    conn, config, container_type: str, container_id: int, set_id: str
) -> int:
    """Save AnnotationConfig as a JSON FileAnnotation on a container.

    Replaces any existing manifest with the same set_id namespace.
    Returns the FileAnnotation ID.
    """
    ns = manifest_namespace(set_id)
    container = conn.getObject(container_type.capitalize(), container_id)
    if not container:
        raise ValueError(f"{container_type} {container_id} not found")

    # Remove existing manifest with same namespace
    to_delete = []
    for ann in container.listAnnotations():
        if hasattr(ann, "getNs") and ann.getNs() == ns:
            to_delete.append(ann.getId())
    if to_delete:
        # Best effort: a stale copy that survives is replaced on the next save.
        with contextlib.suppress(Exception):
            conn.deleteObjects("Annotation", to_delete, wait=True)

    # Write JSON to temp file and upload
    filename = f"{config.name}_{set_id}.json"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=filename.replace(".json", "_"), delete=False
    ) as tmp:
        tmp.write(config.to_json())
        tmp_path = tmp.name

    try:
        file_ann = conn.createFileAnnfromLocalFile(
            tmp_path, mimetype="application/json", ns=ns
        )
        container.linkAnnotation(file_ann)
        return file_ann.getId()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def load_manifest_from_omero(conn, container_type: str, container_id: int, set_id: str):
    """Load an AnnotationConfig manifest by set_id from a container.

    Returns None if not found.
    """
    ns = manifest_namespace(set_id)
    container = conn.getObject(container_type.capitalize(), container_id)
    if not container:
        return None

    for ann in container.listAnnotations():
        if hasattr(ann, "getNs") and ann.getNs() == ns:
            try:
                content = b"".join(ann.getFileInChunks())
                data = json.loads(content)
                return AnnotationConfig.from_json(data)
            except Exception:  # noqa: BLE001 - skip annotations that cannot be read
                continue
    return None


def list_manifests_from_omero(conn, container_type: str, container_id: int) -> list:
    """List all manifest summaries on a container.

    Returns list of dicts: {set_id, name, created, progress, annotation_id}.
    """
    container = conn.getObject(container_type.capitalize(), container_id)
    if not container:
        return []

    manifests = []
    for ann in container.listAnnotations():
        ns = getattr(ann, "getNs", lambda: None)()
        if ns and ns.startswith(MANIFEST_NS_PREFIX):
            set_id = ns[len(MANIFEST_NS_PREFIX) :]
            try:
                content = b"".join(ann.getFileInChunks())
                data = json.loads(content)
                config = AnnotationConfig.from_json(data)
                progress = config.get_progress_summary()
                manifests.append(
                    {
                        "set_id": set_id,
                        "name": config.name,
                        "created": str(config.created),
                        "progress": progress,
                        "annotation_id": ann.getId(),
                    }
                )
            except Exception:  # noqa: BLE001 - skip annotations that cannot be read
                continue
    return manifests


def delete_manifest_from_omero(
    conn, container_type: str, container_id: int, set_id: str
) -> bool:
    """Delete a manifest and all associated GeoJSON FileAnnotations."""
    # Load manifest to find image IDs
    config = load_manifest_from_omero(conn, container_type, container_id, set_id)
    if config:
        image_ids = {ann.image_id for ann in config.annotations}
        ns = geojson_namespace(set_id)
        for image_id in image_ids:
            image = conn.getObject("Image", image_id)
            if not image:
                continue
            to_delete = [
                a.getId()
                for a in image.listAnnotations()
                if hasattr(a, "getNs") and a.getNs() == ns
            ]
            if to_delete:
                # Best effort: leftover GeoJSON does not block deleting the manifest.
                with contextlib.suppress(Exception):
                    conn.deleteObjects("Annotation", to_delete, wait=True)

    # Delete the manifest itself
    container = conn.getObject(container_type.capitalize(), container_id)
    if not container:
        return False
    mns = manifest_namespace(set_id)
    to_delete = [
        a.getId()
        for a in container.listAnnotations()
        if hasattr(a, "getNs") and a.getNs() == mns
    ]
    if to_delete:
        try:
            conn.deleteObjects("Annotation", to_delete, wait=True)
        except Exception:  # noqa: BLE001 - report any OMERO failure as "not deleted"
            return False
    return True


def merge_geojson_patches(existing: dict, new: dict) -> dict:
    """Merge new GeoJSON features into existing, replacing features for the same patch.

    Features are matched by their patch property (x, y, width, height).
    Features without a patch property are treated as full-image and always replaced.
    Channel presentation is taken from the new GeoJSON (latest truth).
    """
    result = {
        "type": "FeatureCollection",
        "channel_presentation": new.get(
            "channel_presentation", existing.get("channel_presentation", [])
        ),
        "features": [],
    }

    def patch_key(feature):
        p = feature.get("properties", {}).get("patch")
        if p:
            return (p["x"], p["y"], p["width"], p["height"])
        return None

    new_patches = {}
    for f in new.get("features", []):
        key = patch_key(f)
        if key not in new_patches:
            new_patches[key] = []
        new_patches[key].append(f)

    # Keep existing features whose patch is NOT in the new set
    for f in existing.get("features", []):
        key = patch_key(f)
        if key not in new_patches:
            result["features"].append(f)

    # Add all new features
    for features in new_patches.values():
        result["features"].extend(features)

    return result


def save_geojson_to_omero(conn, image_id: int, geojson: dict, set_id: str) -> int:
    """Save GeoJSON FeatureCollection as FileAnnotation on an image.

    If a GeoJSON for this set_id already exists, merges patch features.
    Returns the FileAnnotation ID.
    """
    ns = geojson_namespace(set_id)
    image = conn.getObject("Image", image_id)
    if not image:
        raise ValueError(f"Image {image_id} not found")

    # Load existing and merge if present
    existing = load_geojson_from_omero(conn, image_id, set_id)
    if existing:
        geojson = merge_geojson_patches(existing, geojson)

    # Remove old FileAnnotation
    to_delete = []
    for ann in image.listAnnotations():
        if hasattr(ann, "getNs") and ann.getNs() == ns:
            to_delete.append(ann.getId())
    if to_delete:
        # Best effort: a stale copy that survives is replaced on the next save.
        with contextlib.suppress(Exception):
            conn.deleteObjects("Annotation", to_delete, wait=True)

    # Write and upload
    image_name = image.getName() or f"image_{image_id}"
    filename = f"{image_name}_{set_id}.geojson"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".geojson",
        prefix=filename.replace(".geojson", "_"),
        delete=False,
    ) as tmp:
        json.dump(geojson, tmp)
        tmp_path = tmp.name

    try:
        file_ann = conn.createFileAnnfromLocalFile(
            tmp_path, mimetype="application/json", ns=ns
        )
        image.linkAnnotation(file_ann)
        return file_ann.getId()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def load_geojson_from_omero(conn, image_id: int, set_id: str):
    """Load GeoJSON FeatureCollection by set_id namespace from an image.

    Returns None if not found.
    """
    ns = geojson_namespace(set_id)
    image = conn.getObject("Image", image_id)
    if not image:
        return None

    for ann in image.listAnnotations():
        if hasattr(ann, "getNs") and ann.getNs() == ns:
            try:
                content = b"".join(ann.getFileInChunks())
                return json.loads(content)
            except Exception:  # noqa: BLE001 - skip annotations that cannot be read
                continue
    return None
