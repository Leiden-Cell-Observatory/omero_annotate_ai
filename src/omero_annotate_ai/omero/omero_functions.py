"""OMERO integration functions for micro-SAM workflows."""

import json
import os
import random
import string
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import omero.grid as omero_grid
import omero.model as omero_model
import pandas as pd
from omero.constants.namespaces import NSBULKANNOTATIONS
from omero.rtypes import rstring

from .omero_utils import (
    list_annotations_by_namespace,
    list_user_tables,
)

import ezomero

import imageio.v3 as imageio
import tifffile

from ..processing.image_functions import label_to_rois
from ..core.annotation_config import OMERO_ID_UNSET, AnnotationConfig
from . import tracking_table

# Namespace for annotation config file annotations
CONFIG_NS = "openmicroscopy.org/omero/annotate/config"
# Namespace for uploaded label images
LABELS_NS = "openmicroscopy.org/omero/annotate/labels"

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
        try:
            conn.deleteObjects("Annotation", to_delete, wait=True)
        except Exception:
            pass

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
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
            except Exception:
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
            set_id = ns[len(MANIFEST_NS_PREFIX):]
            try:
                content = b"".join(ann.getFileInChunks())
                data = json.loads(content)
                config = AnnotationConfig.from_json(data)
                progress = config.get_progress_summary()
                manifests.append({
                    "set_id": set_id,
                    "name": config.name,
                    "created": str(config.created),
                    "progress": progress,
                    "annotation_id": ann.getId(),
                })
            except Exception:
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
                a.getId() for a in image.listAnnotations()
                if hasattr(a, "getNs") and a.getNs() == ns
            ]
            if to_delete:
                try:
                    conn.deleteObjects("Annotation", to_delete, wait=True)
                except Exception:
                    pass

    # Delete the manifest itself
    container = conn.getObject(container_type.capitalize(), container_id)
    if not container:
        return False
    mns = manifest_namespace(set_id)
    to_delete = [
        a.getId() for a in container.listAnnotations()
        if hasattr(a, "getNs") and a.getNs() == mns
    ]
    if to_delete:
        try:
            conn.deleteObjects("Annotation", to_delete, wait=True)
        except Exception:
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
        "channel_presentation": new.get("channel_presentation", existing.get("channel_presentation", [])),
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
        try:
            conn.deleteObjects("Annotation", to_delete, wait=True)
        except Exception:
            pass

    # Write and upload
    image_name = image.getName() or f"image_{image_id}"
    filename = f"{image_name}_{set_id}.geojson"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".geojson", prefix=filename.replace(".geojson", "_"), delete=False
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
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
            except Exception:
                continue
    return None


# =============================================================================
# Helper Functions
# =============================================================================


def _prepare_dataframe_for_omero(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame with proper types for OMERO table storage.

    Args:
        df: DataFrame to prepare

    Returns:
        DataFrame with properly typed columns for OMERO
    """
    df = df.copy()

    numeric_columns = [
        "image_id", "patch_x", "patch_y", "patch_width", "patch_height",
        "z_slice", "timepoint", "z_start", "z_end", "z_length", "channel",
        "level", "session",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    boolean_columns = ["train", "validate", "processed", "is_patch", "is_volumetric"]
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # Link ids reach OMERO as typed, non-nullable integer columns (see
    # _build_omero_columns). 0 means "not set yet"; a legacy "None" string
    # coerces to it via to_numeric.
    link_id_columns = ["label_id", "roi_id", "annotation_input_id"]
    for col in link_id_columns:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(OMERO_ID_UNSET)
                .astype(int)
            )

    id_columns = ["schema_attachment_id"]
    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    string_columns = ["annotation_type"]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna("segmentation_mask").astype(str)

    if "category" in df.columns:
        df["category"] = df["category"].fillna("training").astype(str)

    datetime_columns = ["annotation_created_at", "annotation_updated_at"]
    for col in datetime_columns:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    return df


def _normalize_omero_table_df(df: Any) -> Any:
    """Map OMERO-side link column names (Image, Roi) back to image_id / roi_id."""
    if not isinstance(df, pd.DataFrame):
        return df
    renames = {
        omero_name: df_name
        for df_name, (omero_name, _, _) in tracking_table.LINK_COLUMNS.items()
        if omero_name in df.columns
    }
    return df.rename(columns=renames) if renames else df


# =============================================================================
# Config-First Table Management Functions
# =============================================================================


def link_table_to_containers(
    conn,
    table_id: int,
    container_type: str,
    container_ids: List[int],
) -> Dict[int, int]:
    """Link an existing table (FileAnnotation) to multiple containers.

    Uses OMERO AnnotationLink objects (e.g., PlateAnnotationLinkI) to attach
    the same FileAnnotation to multiple containers. This allows a single table
    to appear in multiple containers without duplicating the underlying data.

    Note: table_id IS the FileAnnotation ID (ezomero.post_table returns FileAnnotation ID)

    Args:
        conn: OMERO connection
        table_id: The table/FileAnnotation ID to link
        container_type: Type of containers ('dataset', 'plate', 'project', 'screen')
        container_ids: List of container IDs to link the table to

    Returns:
        Dict mapping container_id to link_id

    Raises:
        Exception: If linking fails for any container
    """
    link_classes = {
        'dataset': (omero_model.DatasetAnnotationLinkI, omero_model.DatasetI),
        'plate': (omero_model.PlateAnnotationLinkI, omero_model.PlateI),
        'project': (omero_model.ProjectAnnotationLinkI, omero_model.ProjectI),
        'screen': (omero_model.ScreenAnnotationLinkI, omero_model.ScreenI),
    }

    container_type_lower = container_type.lower()
    if container_type_lower not in link_classes:
        print(f"Warning: Cannot link annotation to container type '{container_type}'")
        return {cid: None for cid in container_ids}

    # table_id is the FileAnnotation ID - get the wrapper object
    file_ann = conn.getObject("FileAnnotation", table_id)
    if not file_ann:
        raise ValueError(f"FileAnnotation {table_id} not found")

    LinkClass, ContainerClass = link_classes[container_type_lower]
    results = {}
    update_service = conn.getUpdateService()

    for container_id in container_ids:
        link = LinkClass()
        link.setParent(ContainerClass(container_id, False))
        link.setChild(file_ann._obj)  # Access underlying OMERO model object
        saved_link = update_service.saveAndReturnObject(link)
        link_id = saved_link.getId().getValue()
        results[container_id] = link_id
        print(f"  Linked table to {container_type} {container_id} (link ID: {link_id})")

    return results


def post_typed_table(
    conn,
    df: pd.DataFrame,
    object_type: str,
    object_id: int,
    title: str,
) -> int:
    """Create a tracking table from a DataFrame and attach it to one container.

    Returns the table's FileAnnotation ID. See :mod:`.tracking_table`.
    """
    return tracking_table.create_table(
        conn, _prepare_dataframe_for_omero(df), title, object_type, [object_id]
    )


def read_tracking_table(conn, table_id: int) -> pd.DataFrame:
    """Read a tracking table (0.5+ only) with link columns named image_id / roi_id."""
    return tracking_table.read_table(conn, table_id)


def _read_any_table(conn, table_id: int) -> Any:
    """Read any OMERO table, without the version check (used for cleanup)."""
    return _normalize_omero_table_df(ezomero.get_table(conn, table_id))


def create_tracking_table(
    conn,
    config,
    table_title: str,
    container_type: str,
    container_ids: List[int],
) -> int:
    """Create the tracking table for a training set from config.annotations.

    The table is linked to every container in ``container_ids`` once, here.
    From then on it is only updated in place (:func:`write_config_to_table`),
    so its ID identifies the training set.

    Returns:
        The table ID.
    """
    df = config.to_dataframe()
    if df.empty:
        raise ValueError("No annotations in the config; define the schema first")
    table_id = tracking_table.create_table(
        conn, _prepare_dataframe_for_omero(df), table_title, container_type, container_ids
    )
    print(f"Created tracking table '{table_title}' with {len(df)} units (ID: {table_id})")
    return table_id


def write_config_to_table(conn, table_id: int, config) -> Dict[str, int]:
    """Write config.annotations to an existing tracking table, in place.

    Rows are matched on annotation_id: existing rows are updated, new rows
    appended, and the table keeps its ID.
    """
    df = config.to_dataframe()
    if df.empty:
        return {"updated": 0, "added": 0}
    return tracking_table.write_rows(conn, table_id, _prepare_dataframe_for_omero(df))


def sync_omero_table_to_config(conn, table_id: int, config) -> None:
    """Load a tracking table's rows into config.annotations.

    Raises instead of returning quietly: a resume that cannot read its table
    must stop, never carry on as if the table were empty.
    """
    df = read_tracking_table(conn, table_id)
    if df is None or df.empty:
        raise ValueError(f"Tracking table {table_id} has no rows")
    config.from_dataframe(df)
    print(f"Loaded {len(config.annotations)} annotations from table {table_id}")


def replace_table_config(
    conn,
    table_id: int,
    file_path: str,
    container_type: str,
    container_id: int,
) -> int:
    """Attach a training set's config YAML, replacing the previous one.

    Every set keeps exactly one config on OMERO, named
    ``annotation_config_table_<id>.yaml``. The new file is posted first and
    the old ones deleted after, so a failed upload never leaves the set
    without a config.

    Args:
        conn: OMERO connection
        table_id: Tracking table ID
        file_path: Local config YAML
        container_type: Container to attach to (the primary container)
        container_id: Container ID

    Returns:
        ID of the new config FileAnnotation.
    """
    object_type = str(container_type).capitalize()
    name = tracking_table.config_file_name(table_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        named = os.path.join(tmp_dir, name)
        with open(file_path, "rb") as src, open(named, "wb") as dst:
            dst.write(src.read())
        new_id = ezomero.post_file_annotation(
            conn, file_path=named, ns=CONFIG_NS, object_type=object_type,
            object_id=int(container_id),
        )
    old = [
        ann["id"] for ann in _table_config_annotations(conn, table_id, [(container_type, container_id)])
        if ann["id"] != new_id
    ]
    if old:
        conn.deleteObjects("Annotation", old, wait=True)
    return new_id


def _table_config_annotations(conn, table_id: int, containers=None) -> List[Dict]:
    """Config FileAnnotations of one table on its containers, newest first."""
    name = tracking_table.config_file_name(table_id)
    containers = containers if containers is not None else _table_containers(conn, table_id)
    found = []
    for ctype, cid in containers:
        for ann in list_annotations_by_namespace(conn, str(ctype).capitalize(), int(cid), CONFIG_NS):
            if ann.get("file_name") == name and ann["id"] not in [f["id"] for f in found]:
                found.append(ann)
    return sorted(found, key=lambda a: a["id"], reverse=True)


def find_annotation_config_for_table(conn, table_id: int) -> Optional[AnnotationConfig]:
    """The config YAML a training set keeps on OMERO, looked up by table ID.

    Returns:
        The AnnotationConfig, or None if the set has no config on OMERO.
    """
    for ann in _table_config_annotations(conn, table_id):
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                file_path = ezomero.get_file_annotation(conn, ann["id"], tmp_dir)
                if file_path:
                    return AnnotationConfig.from_yaml(file_path)
            except Exception:
                continue
    return None


_CONTAINER_TYPES = ("Dataset", "Plate", "Project", "Screen")


def _valid_omero_ids(values) -> List[int]:
    """Positive integer IDs from a column, skipping NaN/None/0/-1, deduplicated."""
    ids = []
    for value in values:
        try:
            if value is None or pd.isna(value):
                continue
            number = int(float(value))
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in ids:
            ids.append(number)
    return ids


def _table_containers(conn, table_id: int) -> List[Tuple[str, int]]:
    """Every container the table FileAnnotation is linked to."""
    containers = []
    for container_type in _CONTAINER_TYPES:
        try:
            links = conn.getAnnotationLinks(container_type, ann_ids=[table_id])
        except Exception:
            continue
        for link in links or []:
            try:
                key = (container_type.lower(), int(link.getParent().getId()))
            except Exception:
                continue
            if key not in containers:
                containers.append(key)
    return containers


def _tagged_objects(conn, image_ids: List[int], table_id: int) -> Tuple[List[int], List[int]]:
    """ROIs and label files on the images whose description carries the set's tag."""
    prefix = tracking_table.tag_prefix(table_id)
    rois, labels = [], []
    roi_service = conn.getRoiService()
    for image_id in image_ids:
        try:
            result = roi_service.findByImage(int(image_id), None, conn.SERVICE_OPTS)
        except Exception:
            continue
        for roi in result.rois:
            description = roi.getDescription().getValue() if roi.getDescription() else ""
            if description.startswith(prefix):
                rois.append(roi.getId().getValue())
        for ann in list_annotations_by_namespace(conn, "Image", int(image_id), LABELS_NS):
            if (ann.get("description") or "").startswith(prefix):
                labels.append(ann["id"])
    return rois, labels


def delete_training_set(
    conn,
    table_id: int,
    container_type: Optional[str] = None,
    container_id: Optional[int] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Delete a tracking table and everything a run created for it.

    Everything is found by ID or by the set's tag
    (``omero-annotate-ai:<table_id>:``), never by name, so another training
    set's objects are never touched:

    - ``rois``: the ``roi_id`` of each row, plus ROIs on the set's images
      whose description carries the tag (uploads a crash left untracked).
    - ``labels``: the ``label_id`` of each row, plus tagged label files.
    - ``annotation_inputs``: the ``annotation_input_id`` of each row.
    - ``configs``: the set's config YAML(s), ``annotation_config_table_<id>.yaml``.
    - ``table``: the tracking table FileAnnotation, deleted last.

    Args:
        conn: OMERO connection
        table_id: Tracking table (FileAnnotation) ID
        container_type: Container to search for configs; defaults to every
            container the table is linked to
        container_id: Container ID that goes with ``container_type``
        dry_run: When True (default) nothing is deleted; the report says what
            would be.

    Returns:
        Report dict with the IDs per kind, ``missing`` (IDs in the table that
        no longer exist on the server), ``containers`` searched, and
        ``deleted`` (False on a dry run).
    """
    kinds = ("rois", "labels", "annotation_inputs", "configs", "table")
    report: Dict[str, Any] = {kind: [] for kind in kinds}
    report["missing"] = {kind: [] for kind in kinds}
    report["containers"] = []
    report["deleted"] = False

    table = conn.getObject("FileAnnotation", table_id)
    if table is None:
        report["missing"]["table"].append(table_id)
        print(f"Table {table_id} not found; nothing to delete")
        return report
    report["table"].append(table_id)

    df = _read_any_table(conn, table_id)
    row_kinds = [
        ("rois", "Roi", "roi_id"),
        ("labels", "FileAnnotation", "label_id"),
        ("annotation_inputs", "FileAnnotation", "annotation_input_id"),
    ]
    for kind, omero_type, column in row_kinds:
        if df is None or column not in df.columns:
            continue
        for object_id in _valid_omero_ids(df[column]):
            if conn.getObject(omero_type, object_id) is None:
                report["missing"][kind].append(object_id)
            else:
                report[kind].append(object_id)

    image_ids = _valid_omero_ids(df["image_id"]) if df is not None and "image_id" in df.columns else []
    tagged_rois, tagged_labels = _tagged_objects(conn, image_ids, table_id)
    report["rois"].extend(i for i in tagged_rois if i not in report["rois"])
    report["labels"].extend(i for i in tagged_labels if i not in report["labels"])

    if container_type and container_id:
        containers = [(str(container_type).lower(), int(container_id))]
    else:
        containers = _table_containers(conn, table_id)
    report["containers"] = containers
    report["configs"] = [a["id"] for a in _table_config_annotations(conn, table_id, containers)]

    summary = ", ".join(f"{len(report[k])} {k}" for k in kinds)
    missing = sum(len(v) for v in report["missing"].values())
    if dry_run:
        print(f"Dry run, nothing deleted. Would delete: {summary}"
              + (f" ({missing} IDs in the table no longer exist)" if missing else "")
              + ". Pass dry_run=False to delete.")
        return report

    # Children first, the table last.
    if report["rois"]:
        conn.deleteObjects("Roi", report["rois"], wait=True)
    annotations = report["labels"] + report["annotation_inputs"] + report["configs"]
    if annotations:
        conn.deleteObjects("Annotation", annotations, wait=True)
    conn.deleteObjects("Annotation", [table_id], wait=True)
    report["deleted"] = True
    print(f"Deleted: {summary}")
    return report


def infer_settings_from_table(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer the run settings that a tracking table's rows pin down.

    Used when no saved config matches the table. Returns only what the rows
    show: the annotated channel, 3D and patch settings, and the number of
    distinct images per category.
    """
    settings: Dict[str, Any] = {}
    if df is None or df.empty:
        return settings

    def first(column):
        if column not in df.columns:
            return None
        values = df[column].dropna()
        return values.iloc[0] if len(values) else None

    def as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    channel = first("channel")
    if channel is not None:
        settings["channel"] = int(channel)
    volumetric = first("is_volumetric")
    if volumetric is not None:
        settings["three_d"] = as_bool(volumetric)
    is_patch = first("is_patch")
    if is_patch is not None:
        settings["use_patches"] = as_bool(is_patch)
        if settings["use_patches"]:
            patches = df[df["is_patch"].map(as_bool)]
            width, height = first("patch_width"), first("patch_height")
            if width is not None and height is not None:
                settings["patch_size"] = [int(width), int(height)]
            image_col = "image_uri" if "image_uri" in df.columns else "image_id"
            plane = [c for c in (image_col, "timepoint", "z_slice") if c in patches.columns]
            if image_col in plane and len(patches):
                settings["patches_per_image"] = int(patches.groupby(plane).size().max())

    for column, mode_key, list_key in (("timepoint", "timepoint_mode", "timepoints"),
                                       ("z_slice", "z_slice_mode", "z_slices")):
        if column in df.columns:
            values = sorted({int(v) for v in df[column].dropna()})
            if values:
                settings[mode_key] = "specific"
                settings[list_key] = values

    image_col = "image_uri" if "image_uri" in df.columns else "image_id"
    if "category" in df.columns and image_col in df.columns:
        counts = df.groupby("category")[image_col].nunique()
        settings["train_n"] = int(counts.get("training", 0))
        settings["validate_n"] = int(counts.get("validation", 0))
        settings["test_n"] = int(counts.get("test", 0))
    return settings


def _label_file_name(annotation_file: str, trainingset_name: Optional[str],
                     tag: Optional[str]) -> str:
    """``<trainingset>_<annotation_id>_mask.tif``, falling back to the local name.

    The annotation ID comes from the tag (``omero-annotate-ai:<table>:<id>``).
    Characters that are unsafe in file names are replaced by ``_``.
    """
    def safe(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)

    annotation_id = tag.split(":", 2)[2] if tag and tag.count(":") >= 2 else None
    stem = f"{annotation_id}_mask" if annotation_id else os.path.splitext(
        os.path.basename(annotation_file))[0]
    prefix = f"{trainingset_name}_" if trainingset_name else ""
    return safe(prefix + stem) + ".tif"


def upload_rois_and_labels(
    conn,
    image_id: int,
    annotation_file: str,
    patch_offset: Optional[Tuple[int, int]] = None,
    trainingset_name: Optional[str] = None,
    trainingset_description: Optional[str] = None,
    timepoint: Optional[int] = None,
    z_slice: Optional[int] = None,
    channel: Optional[int] = None,
    is_volumetric: Optional[bool] = False,
    z_start: Optional[int] = 0,
    tag: Optional[str] = None,
):
    """Upload ROIs and labels to OMERO image.

    Args:
        conn: OMERO connection
        image_id: ID of OMERO image
        annotation_file: Path to annotation file (TIFF with labels)
        patch_offset: Optional (x,y) offset for patch placement
        trainingset_name: Optional training set name for custom annotation naming
        trainingset_description: Optional training set description for annotation
        timepoint: Optional timepoint for positioning the roi properly in OMERO
        z_slice: Optional z_slice for positioning the roi properly in OMERO
        channel: Optional channel for positioning the roi properly in OMERO
        is_volumetric: check if we need to handle roi as 3D
        z_start: use for volumetric to use z_start as offset in the stack
        tag: Training-set tag (``tracking_table.tag_for``) written first in the
            ROI and label descriptions, so the upload can be traced back to its
            table even if the table never recorded it.

    Returns:
        tuple: (label_id, roi_id) - IDs of uploaded label file and ROI collection
    """
    # Load label image
    print(f"Step 1: Loading label image from {annotation_file}")
    label_img = imageio.imread(annotation_file)
    print(f"Label image loaded: {label_img.shape}, dtype: {label_img.dtype}")
    unique_labels = np.unique(label_img)
    print(f"Found {len(unique_labels)} unique labels: {unique_labels[:10]}...")

    # Create ROI shapes from label image
    print("Step 2: Converting labels to ROI shapes...")
    effective_z = z_start if is_volumetric else z_slice
    shapes = label_to_rois(
        label_img=label_img,
        z_slice=effective_z,
        channel=channel,
        timepoint=timepoint,
        is_volumetric=is_volumetric,
        patch_offset=patch_offset,
    )
    print(f"Created {len(shapes)} ROI shapes from labels")

    # Upload label file as attachment
    print("Step 3: Uploading label file as attachment")

    # Description: the set's tag first (so the upload can be traced to its
    # table), then the training set and the plane the labels belong to.
    desc_parts = [tag] if tag else []
    if trainingset_description:
        desc_parts.append(trainingset_description)
    for key, value in (("t", timepoint), ("z", z_start if is_volumetric else z_slice), ("c", channel)):
        if value is not None:
            desc_parts.append(f"{key}={value}")
    if patch_offset:
        desc_parts.append(f"Patch offset: ({patch_offset[0]}, {patch_offset[1]})")
    label_desc = " | ".join(desc_parts)

    # The uploaded file is written losslessly compressed (label images compress
    # well) and named after the training set and annotation, so label files of
    # different sets on the same image can be told apart in OMERO.
    with tempfile.TemporaryDirectory() as tmp:
        upload_path = os.path.join(tmp, _label_file_name(annotation_file, trainingset_name, tag))
        tifffile.imwrite(upload_path, label_img, compression="zlib")
        file_ann_id = ezomero.post_file_annotation(
            conn,
            file_path=upload_path,
            description=label_desc,
            ns=LABELS_NS,
            object_type="Image",
            object_id=image_id,
        )
    print(f"File annotation uploaded with ID: {file_ann_id}")

    # Upload ROI shapes if any were created
    print("Step 4: Uploading ROI shapes")
    roi_id = None
    if shapes:
        roi_name = f"{trainingset_name}_ROIs" if trainingset_name else "ROIs"
        roi_description = label_desc  # Use same description as label file

        roi_id = ezomero.post_roi(
            conn, image_id, shapes, name=roi_name, description=roi_description
        )
        print(f"Created {len(shapes)} ROI shapes for image {image_id} with ID: {roi_id}")
    else:
        print(f"No ROI shapes created from {annotation_file}")

    print(f"Uploaded annotations from {annotation_file} to image {image_id}")
    if patch_offset:
        print(f"   Patch offset: {patch_offset}")
    print(f"   File annotation ID: {file_ann_id}")
    if roi_id:
        print(f"   ROI ID: {roi_id}")

    return file_ann_id, roi_id


def upload_annotation_input_image(
    conn,
    image_id: int,
    annotation_input_file: str,
    trainingset_name: Optional[str] = None,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    z_slice: Optional[int] = None,
) -> int:
    """Upload label input channel image as file annotation to OMERO.

    This function uploads the raw image data from the label channel
    (used for segmentation) as a file annotation. This is useful when
    using separate channels for labeling vs training, allowing the
    label channel images to be stored alongside the segmentation masks.

    Args:
        conn: OMERO connection
        image_id: ID of OMERO image to attach annotation to
        annotation_input_file: Path to the annotation-channel image file (TIFF)
        trainingset_name: Optional training set name for description
        channel: Optional channel index for description
        timepoint: Optional timepoint for description
        z_slice: Optional z-slice for description

    Returns:
        file_ann_id: OMERO file annotation ID
    """
    # Build description with available metadata
    desc_parts = ["Label input image"]
    if channel is not None:
        desc_parts.append(f"channel={channel}")
    if timepoint is not None:
        desc_parts.append(f"t={timepoint}")
    if z_slice is not None:
        desc_parts.append(f"z={z_slice}")
    if trainingset_name:
        desc_parts.append(f"trainingset={trainingset_name}")

    description = " | ".join(desc_parts)

    file_ann_id = ezomero.post_file_annotation(
        conn,
        file_path=annotation_input_file,
        description=description,
        ns="openmicroscopy.org/omero/annotate/annotation_input",
        object_type="Image",
        object_id=image_id,
    )

    print(f"Uploaded label input image to OMERO image {image_id}, annotation ID: {file_ann_id}")

    return file_ann_id


# =============================================================================
# Annotation Table Management Functions
# =============================================================================


def list_annotation_tables(
    conn, container_type: str, container_id: int
) -> List[Dict[str, Any]]:
    """Find all tables attached to a container.

    Args:
        conn: OMERO connection
        container_type: Type of container ('project', 'dataset', 'plate', 'screen')
        container_id: Container ID to search in

    Returns:
        List of dictionaries with table information, sorted newest first
    """
    all_tables = list_user_tables(
        conn, container_type=container_type, container_id=container_id
    )
    all_tables.sort(key=lambda x: x.get("created", ""), reverse=True)
    return all_tables


def generate_unique_table_name(
    conn, container_type: str, container_id: int, base_name: str = None
) -> str:
    """Generate a unique table name for a container.

    Args:
        conn: OMERO connection
        container_type: Type of container ('project', 'dataset', 'plate', 'screen')
        container_id: Container ID
        base_name: Optional base name for the table

    Returns:
        Unique table name
    """
    # Get container name for better naming
    try:
        container = conn.getObject(container_type.capitalize(), container_id)
        container_name = (
            container.getName() if container else f"{container_type}_{container_id}"
        )
        # Clean container name for use in table name
        container_name = "".join(
            c for c in container_name if c.isalnum() or c in "_-"
        ).lower()
    except Exception as e:
        print(f"Warning: Could not get container name: {e}")
        container_name = f"{container_type}_{container_id}"

    # Create base name if not provided
    if not base_name:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_name = f"{container_name}_{timestamp}"

    # Check if name already exists and make it unique
    existing_tables = list_annotation_tables(conn, container_type, container_id)
    existing_names = {table["name"] for table in existing_tables}

    unique_name = base_name
    counter = 1

    while unique_name in existing_names:
        unique_name = f"{base_name}_v{counter}"
        counter += 1

    return unique_name


def analyze_table_completion_status(conn, table_id: int) -> Dict[str, Any]:
    """Analyze the completion status of an annotation table.

    Args:
        conn: OMERO connection
        table_id: Table ID to analyze

    Returns:
        Dictionary with progress information
    """
    try:
        # Get table data
        table_data = read_tracking_table(conn, table_id)

        if table_data is None or table_data.empty:
            return {
                "total_units": 0,
                "completed_units": 0,
                "progress_percent": 0,
                "is_complete": False,
                "status": "empty",
                "error": "Table is empty or could not be read",
            }

        # Analyze progress based on table structure
        total_units = len(table_data)

        # Check for 'processed' column or similar completion indicators
        completed_units = 0
        completion_columns = ["processed", "completed", "finished", "done"]

        completion_column = None
        for col in completion_columns:
            if col in table_data.columns:
                completion_column = col
                break

        if completion_column:
            # Count completed units
            completed_units = (
                table_data[completion_column].sum()
                if table_data[completion_column].dtype == bool
                else (table_data[completion_column] == True).sum()
            )
        else:
            # Check for roi_id or label_id columns as completion indicators
            if "roi_id" in table_data.columns:
                completed_units = table_data["roi_id"].notna().sum()
            elif "label_id" in table_data.columns:
                completed_units = table_data["label_id"].notna().sum()

        # Calculate progress
        progress_percent = (
            (completed_units / total_units * 100) if total_units > 0 else 0
        )
        is_complete = progress_percent >= 100

        # Determine status
        if is_complete:
            status = "complete"
        elif completed_units > 0:
            status = "in_progress"
        else:
            status = "not_started"

        return {
            "total_units": int(total_units),
            "completed_units": int(completed_units),
            "progress_percent": round(progress_percent, 1),
            "is_complete": is_complete,
            "status": status,
            "table_size": len(table_data),
            "columns": list(table_data.columns),
        }

    except Exception as e:
        return {
            "total_units": 0,
            "completed_units": 0,
            "progress_percent": 0,
            "is_complete": False,
            "status": "error",
            "error": str(e),
        }


def get_table_progress_summary(conn, table_id: int) -> str:
    """Get a human-readable progress summary for a table.

    Args:
        conn: OMERO connection
        table_id: Table ID

    Returns:
        Progress summary string
    """
    progress = analyze_table_completion_status(conn, table_id)

    if progress["status"] == "error":
        return f"Error: {progress.get('error', 'Unknown error')}"

    total = progress["total_units"]
    completed = progress["completed_units"]
    percent = progress["progress_percent"]

    return f" {completed}/{total} units ({percent:.1f}% complete)"


def create_roi_namespace_for_table(table_name: str) -> str:
    """Create a consistent ROI namespace for a table.

    Args:
        table_name: Name of the annotation table

    Returns:
        ROI namespace string
    """
    return f"omero_annotate_ai.table.{table_name}"


