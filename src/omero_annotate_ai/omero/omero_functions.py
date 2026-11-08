"""OMERO integration functions for micro-SAM workflows."""

import logging
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import ezomero
import imageio.v3 as imageio
import numpy as np
import omero.model as omero_model
import pandas as pd
import tifffile

from ..core.annotation_config import AnnotationConfig
from ..processing.image_functions import label_to_rois
from ..rows import Row, rows_from_frame, rows_to_frame
from . import tracking_table
from .omero_utils import (
    list_annotations_by_namespace,
    list_user_tables,
)

# Namespace for annotation config file annotations
CONFIG_NS = "openmicroscopy.org/omero/annotate/config"
# Namespace for uploaded label images
LABELS_NS = "openmicroscopy.org/omero/annotate/labels"

# =============================================================================
# Helper Functions
# =============================================================================

logger = logging.getLogger(__name__)


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
    container_ids: list[int],
) -> dict[int, int]:
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
        "dataset": (omero_model.DatasetAnnotationLinkI, omero_model.DatasetI),
        "plate": (omero_model.PlateAnnotationLinkI, omero_model.PlateI),
        "project": (omero_model.ProjectAnnotationLinkI, omero_model.ProjectI),
        "screen": (omero_model.ScreenAnnotationLinkI, omero_model.ScreenI),
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


def read_tracking_table(conn, table_id: int) -> pd.DataFrame:
    """Read a tracking table (0.5+ only) with link columns named image_id / roi_id."""
    return tracking_table.read_table(conn, table_id)


def _read_any_table(conn, table_id: int) -> Any:
    """Read any OMERO table, without the version check (used for cleanup)."""
    return _normalize_omero_table_df(ezomero.get_table(conn, table_id))


def create_tracking_table(
    conn,
    rows: Sequence[Row],
    table_title: str,
    container_type: str,
    container_ids: list[int],
) -> int:
    """Create the tracking table for a training set's rows.

    The table is linked to every container in ``container_ids`` once, here.
    From then on it is only updated in place (:func:`write_rows_to_table`),
    so its ID identifies the training set.

    Returns:
        The table ID.
    """
    if not rows:
        raise ValueError("No rows to track; plan the training set first")
    table_id = tracking_table.create_table(
        conn, rows_to_frame(rows), table_title, container_type, container_ids
    )
    logger.info(
        "Created tracking table '%s' with %d units (ID: %d)",
        table_title,
        len(rows),
        table_id,
    )
    return table_id


def write_rows_to_table(conn, table_id: int, rows: Sequence[Row]) -> dict[str, int]:
    """Write rows to an existing tracking table, in place.

    Rows are matched on annotation_id: existing rows are updated, new rows
    appended, and the table keeps its ID.
    """
    if not rows:
        return {"updated": 0, "added": 0}
    return tracking_table.write_rows(conn, table_id, rows_to_frame(rows))


def read_table_rows(conn, table_id: int) -> list[Row]:
    """A tracking table's rows.

    Raises instead of returning quietly: a resume that cannot read its table
    must stop, never carry on as if the table were empty.
    """
    df = read_tracking_table(conn, table_id)
    if df is None or df.empty:
        raise ValueError(f"Tracking table {table_id} has no rows")
    return rows_from_frame(df)


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
            conn,
            file_path=named,
            ns=CONFIG_NS,
            object_type=object_type,
            object_id=int(container_id),
        )
    old = [
        ann["id"]
        for ann in _table_config_annotations(
            conn, table_id, [(container_type, container_id)]
        )
        if ann["id"] != new_id
    ]
    if old:
        conn.deleteObjects("Annotation", old, wait=True)
    return new_id


def _table_config_annotations(conn, table_id: int, containers=None) -> list[dict]:
    """Config FileAnnotations of one table on its containers, newest first."""
    name = tracking_table.config_file_name(table_id)
    containers = (
        containers if containers is not None else _table_containers(conn, table_id)
    )
    found = []
    for ctype, cid in containers:
        for ann in list_annotations_by_namespace(
            conn, str(ctype).capitalize(), int(cid), CONFIG_NS
        ):
            if ann.get("file_name") == name and ann["id"] not in [
                f["id"] for f in found
            ]:
                found.append(ann)
    return sorted(found, key=lambda a: a["id"], reverse=True)


def find_annotation_config_for_table(conn, table_id: int) -> AnnotationConfig | None:
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


def _valid_omero_ids(values) -> list[int]:
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


def _table_containers(conn, table_id: int) -> list[tuple[str, int]]:
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


def _tagged_on_image(
    conn, image_id: int
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """``(label files, ROIs)`` on an image as ``(id, tag)``, for tagged uploads only."""
    rois: list[tuple[int, str]] = []
    labels: list[tuple[int, str]] = []
    try:
        result = conn.getRoiService().findByImage(image_id, None, conn.SERVICE_OPTS)
    except Exception:  # noqa: BLE001 - an unreadable image has no uploads to find
        return labels, rois
    for roi in result.rois:
        description = roi.getDescription().getValue() if roi.getDescription() else ""
        if tag := _description_tag(description):
            rois.append((roi.getId().getValue(), tag))
    for ann in list_annotations_by_namespace(conn, "Image", image_id, LABELS_NS):
        if tag := _description_tag(ann.get("description") or ""):
            labels.append((ann["id"], tag))
    return labels, rois


def _tagged_objects(
    conn, image_ids: list[int], table_id: int
) -> tuple[list[int], list[int]]:
    """ROIs and label files on the images whose description carries the set's tag."""
    prefix = tracking_table.tag_prefix(table_id)
    rois, labels = [], []
    for image_id in image_ids:
        image_labels, image_rois = _tagged_on_image(conn, int(image_id))
        rois.extend(i for i, tag in image_rois if tag.startswith(prefix))
        labels.extend(i for i, tag in image_labels if tag.startswith(prefix))
    return rois, labels


# Large ROI deletes can take a while; wait up to 10 minutes per call.
_DELETE_WAIT_LOOPS = 1200
_DELETE_WAIT_MS = 500

# Report keys and the OMERO type each kind is checked (and owned) as.
_DELETE_KINDS = {
    "rois": "Roi",
    "labels": "FileAnnotation",
    "annotation_inputs": "FileAnnotation",
    "configs": "FileAnnotation",
    "table": "FileAnnotation",
}


def _owner_id(obj: Any) -> int | None:
    """The owner's experimenter ID of a gateway object, if it can be read."""
    try:
        return int(obj.getDetails().getOwner().getId())
    except (AttributeError, TypeError, ValueError):
        return None


def _delete_and_wait(conn: Any, graph: str, ids: list[int]) -> None:
    """Delete ``ids`` and wait until the server reports the command done.

    ``deleteObjects(wait=True)`` gives up silently after a few seconds; this
    waits much longer and raises if the server reports an error or never
    finishes, so a report never says "deleted" for work still running.
    """
    handle = conn.deleteObjects(graph, ids, wait=False)
    conn._waitOnCmd(
        handle,
        loops=_DELETE_WAIT_LOOPS,
        ms=_DELETE_WAIT_MS,
        failonerror=True,
        failontimeout=True,
        closehandle=True,
    )


def delete_training_set(
    conn,
    table_id: int,
    container_type: str | None = None,
    container_id: int | None = None,
    dry_run: bool = True,
    force_others: bool = False,
) -> dict[str, Any]:
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

    Objects are looked up in every group the user can see and deleted in the
    table's group. Objects owned by another user (possible in read-write groups)
    are listed under ``others`` and block the deletion unless ``force_others``
    is True.

    Args:
        conn: OMERO connection
        table_id: Tracking table (FileAnnotation) ID
        container_type: Container to search for configs; defaults to every
            container the table is linked to
        container_id: Container ID that goes with ``container_type``
        dry_run: When True (default) nothing is deleted; the report says what
            would be.
        force_others: Also delete objects owned by other users.

    Returns:
        Report dict with the IDs per kind, ``missing`` (IDs in the table that
        no longer exist), ``others`` (IDs owned by another user),
        ``containers`` searched, and ``deleted`` (False on a dry run).

    Raises:
        PermissionError: Objects belong to another user and ``force_others``
            is False (nothing is deleted).
        omero.CmdError: The server reported an error while deleting, or did
            not finish in time.
    """
    kinds = tuple(_DELETE_KINDS)
    report: dict[str, Any] = {kind: [] for kind in kinds}
    report["missing"] = {kind: [] for kind in kinds}
    report["others"] = {kind: [] for kind in kinds}
    report["containers"] = []
    report["deleted"] = False

    original_group = conn.SERVICE_OPTS.getOmeroGroup()
    conn.SERVICE_OPTS.setOmeroGroup("-1")  # find the table in any group
    try:
        table = conn.getObject("FileAnnotation", table_id)
        if table is None:
            report["missing"]["table"].append(table_id)
            print(f"Table {table_id} not found; nothing to delete")
            return report
        # Work in the table's group from here on, so its objects are found and
        # deleted there instead of being reported missing.
        conn.SERVICE_OPTS.setOmeroGroup(str(table.getDetails().getGroup().getId()))
        report["table"].append(table_id)

        df = _read_any_table(conn, table_id)
        row_kinds = [
            ("rois", "roi_id"),
            ("labels", "label_id"),
            ("annotation_inputs", "annotation_input_id"),
        ]
        for kind, column in row_kinds:
            if df is None or column not in df.columns:
                continue
            for object_id in _valid_omero_ids(df[column]):
                if conn.getObject(_DELETE_KINDS[kind], object_id) is None:
                    report["missing"][kind].append(object_id)
                else:
                    report[kind].append(object_id)

        image_ids = (
            _valid_omero_ids(df["image_id"])
            if df is not None and "image_id" in df.columns
            else []
        )
        tagged_rois, tagged_labels = _tagged_objects(conn, image_ids, table_id)
        report["rois"].extend(i for i in tagged_rois if i not in report["rois"])
        report["labels"].extend(i for i in tagged_labels if i not in report["labels"])

        if container_type and container_id:
            containers = [(str(container_type).lower(), int(container_id))]
        else:
            containers = _table_containers(conn, table_id)
        report["containers"] = containers
        report["configs"] = [
            a["id"] for a in _table_config_annotations(conn, table_id, containers)
        ]

        me = conn.getUserId()
        for kind in kinds:
            for object_id in report[kind]:
                obj = conn.getObject(_DELETE_KINDS[kind], object_id)
                owner = _owner_id(obj) if obj is not None else None
                if owner is not None and owner != me:
                    report["others"][kind].append(object_id)

        summary = ", ".join(f"{len(report[k])} {k}" for k in kinds)
        missing = sum(len(v) for v in report["missing"].values())
        others = sum(len(v) for v in report["others"].values())
        notes = (
            f" ({missing} IDs in the table no longer exist)" if missing else ""
        ) + (f" ({others} owned by another user)" if others else "")
        if dry_run:
            print(
                f"Dry run, nothing deleted. Would delete: {summary}{notes}. "
                "Pass dry_run=False to delete."
            )
            return report
        if others and not force_others:
            raise PermissionError(
                f"{others} objects of training set {table_id} belong to another "
                "user; nothing was deleted. Pass force_others=True to delete them "
                "too (needs the rights to do so)."
            )

        # Children first, the table last.
        if report["rois"]:
            _delete_and_wait(conn, "Roi", report["rois"])
        annotations = report["labels"] + report["annotation_inputs"] + report["configs"]
        if annotations:
            _delete_and_wait(conn, "Annotation", annotations)
        _delete_and_wait(conn, "Annotation", [table_id])
        report["deleted"] = conn.getObject("FileAnnotation", table_id) is None
        if not report["deleted"]:
            raise RuntimeError(f"Table {table_id} still exists after deleting it")
        print(f"Deleted: {summary}")
        return report
    finally:
        conn.SERVICE_OPTS.setOmeroGroup(original_group)


def _label_file_name(
    annotation_file: str, trainingset_name: str | None, tag: str | None
) -> str:
    """``<trainingset>_<annotation_id>_mask.tif``, falling back to the local name.

    The annotation ID comes from the tag (``omero-annotate-ai:<table>:<id>``).
    Characters that are unsafe in file names are replaced by ``_``.
    """

    def safe(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)

    annotation_id = tag.split(":", 2)[2] if tag and tag.count(":") >= 2 else None
    stem = (
        f"{annotation_id}_mask"
        if annotation_id
        else os.path.splitext(os.path.basename(annotation_file))[0]
    )
    prefix = f"{trainingset_name}_" if trainingset_name else ""
    return safe(prefix + stem) + ".tif"


def _label_description(
    tag: str | None,
    trainingset_description: str | None,
    timepoint: int | None,
    z: int | None,
    channel: int | None,
    patch_offset: tuple[int, int] | None,
) -> str:
    """``<tag> | <training set> | t=.. | z=.. | c=.. | Patch offset: (x, y)``.

    The set's tag comes first, so an upload can be traced to its table.
    """
    parts = [tag] if tag else []
    if trainingset_description:
        parts.append(trainingset_description)
    for key, value in (("t", timepoint), ("z", z), ("c", channel)):
        if value is not None:
            parts.append(f"{key}={value}")
    if patch_offset:
        parts.append(f"Patch offset: ({patch_offset[0]}, {patch_offset[1]})")
    return " | ".join(parts)


def upload_label_file(
    conn,
    image_id: int,
    label_img: np.ndarray,
    *,
    tag: str | None = None,
    trainingset_name: str | None = None,
    trainingset_description: str | None = None,
    timepoint: int | None = None,
    z: int | None = None,
    channel: int | None = None,
    patch_offset: tuple[int, int] | None = None,
    local_name: str = "mask.tif",
) -> int:
    """Attach a label image to an OMERO image as a file; returns its FileAnnotation ID.

    The file is written losslessly compressed (zlib) and named after the training
    set and annotation, so label files of different sets on one image stay apart.
    """
    description = _label_description(
        tag, trainingset_description, timepoint, z, channel, patch_offset
    )
    with tempfile.TemporaryDirectory() as tmp:
        upload_path = os.path.join(
            tmp, _label_file_name(local_name, trainingset_name, tag)
        )
        tifffile.imwrite(upload_path, label_img, compression="zlib")
        return ezomero.post_file_annotation(
            conn,
            file_path=upload_path,
            description=description,
            ns=LABELS_NS,
            object_type="Image",
            object_id=image_id,
        )


def upload_label_rois(
    conn,
    image_id: int,
    label_img: np.ndarray,
    *,
    tag: str | None = None,
    trainingset_name: str | None = None,
    trainingset_description: str | None = None,
    timepoint: int | None = None,
    z: int | None = None,
    channel: int | None = None,
    is_volumetric: bool = False,
    patch_offset: tuple[int, int] | None = None,
) -> int | None:
    """Post one ROI holding a shape per label; returns its ID, or None for an empty mask."""
    shapes = label_to_rois(
        label_img=label_img,
        z_slice=z,
        channel=channel,
        timepoint=timepoint,
        is_volumetric=is_volumetric,
        patch_offset=patch_offset,
    )
    if not shapes:
        return None
    return ezomero.post_roi(
        conn,
        image_id,
        shapes,
        name=f"{trainingset_name}_ROIs" if trainingset_name else "ROIs",
        description=_label_description(
            tag, trainingset_description, timepoint, z, channel, patch_offset
        ),
    )


def upload_rois_and_labels(
    conn,
    image_id: int,
    annotation_file: str,
    patch_offset: tuple[int, int] | None = None,
    trainingset_name: str | None = None,
    trainingset_description: str | None = None,
    timepoint: int | None = None,
    z_slice: int | None = None,
    channel: int | None = None,
    is_volumetric: bool | None = False,
    z_start: int | None = 0,
    tag: str | None = None,
):
    """Upload a mask file as a label file plus ROIs (:func:`upload_label_file`, then
    :func:`upload_label_rois`).

    For volumes, ``z_start`` is the first plane; otherwise ``z_slice`` is the plane.

    Returns:
        tuple: (label_id, roi_id); roi_id is None for an empty mask.
    """
    label_img = imageio.imread(annotation_file)
    common = {
        "tag": tag,
        "trainingset_name": trainingset_name,
        "trainingset_description": trainingset_description,
        "timepoint": timepoint,
        "z": z_start if is_volumetric else z_slice,
        "channel": channel,
        "patch_offset": patch_offset,
    }
    label_id = upload_label_file(
        conn,
        image_id,
        label_img,
        local_name=os.path.basename(annotation_file),
        **common,
    )
    roi_id = upload_label_rois(
        conn, image_id, label_img, is_volumetric=bool(is_volumetric), **common
    )
    return label_id, roi_id


def _description_tag(description: str) -> str:
    """The tag an upload's description starts with ("" if none)."""
    return description.split(" | ", 1)[0].strip()


def find_tagged(conn, image_id: int, tag: str) -> tuple[list[int], list[int]]:
    """Label files and ROIs on an image whose description carries exactly ``tag``.

    Returns:
        ``(label_ids, roi_ids)``, each sorted.
    """
    labels, rois = _tagged_on_image(conn, int(image_id))
    return (
        sorted(i for i, t in labels if t == tag),
        sorted(i for i, t in rois if t == tag),
    )


def download_label(conn, label_id: int) -> np.ndarray | None:
    """Pixels of an uploaded label file, or None if it no longer exists."""
    if conn.getObject("FileAnnotation", int(label_id)) is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        return np.asarray(
            tifffile.imread(ezomero.get_file_annotation(conn, int(label_id), tmp))
        )


def upload_annotation_input_image(
    conn,
    image_id: int,
    annotation_input_file: str,
    trainingset_name: str | None = None,
    channel: int | None = None,
    timepoint: int | None = None,
    z_slice: int | None = None,
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

    print(
        f"Uploaded label input image to OMERO image {image_id}, annotation ID: {file_ann_id}"
    )

    return file_ann_id


# =============================================================================
# Annotation Table Management Functions
# =============================================================================


def list_annotation_tables(
    conn, container_type: str, container_id: int
) -> list[dict[str, Any]]:
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
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
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


def analyze_table_completion_status(conn, table_id: int) -> dict[str, Any]:
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
                else table_data[completion_column].eq(True).sum()
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
