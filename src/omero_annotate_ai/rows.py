"""The rows of a training set: one per annotation unit, and their one table codec.

Rows live in the set's ``tracking_table.csv`` and its OMERO tracking table - never in
the config. :func:`rows_to_frame` and :func:`rows_from_frame` are the only mapping
between a :class:`Row` and a table row, used for both.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from .planning import ImageIdentity, Unit
from .sources.uri import omero_uri

Category = Literal["training", "validation", "test"]
SPLIT_CATEGORIES: tuple[Category, ...] = ("training", "validation", "test")

OMERO_ID_UNSET = 0
"""OMERO id columns are non-nullable: 0 means "not uploaded" (no object has id 0)."""


class Row(BaseModel):
    """One annotation unit and its state: what to annotate, and what was saved."""

    # The image
    image_uri: str = Field(default="", description="omero:image:<id> or OME-Zarr URL")
    image_id: int | None = Field(default=None, description="OMERO image ID")
    image_name: str = Field(description="Image name")
    annotation_id: str = Field(description="Unique annotation identifier")
    level: int = Field(default=0, description="Multiscale level of the coordinates")
    well: str = Field(default="", description="Plate well, e.g. B2")
    field: str = Field(default="", description="Field (well sample) path, e.g. 0")
    well_metadata: dict[str, str] = Field(default_factory=dict)

    # The unit
    category: Category = "training"
    timepoint: int = -1
    z_slice: int = -1
    channel: int = -1
    is_volumetric: bool = False
    z_start: int = -1
    z_end: int = -1
    z_length: int = 1
    is_patch: bool = False
    patch_x: int = 0
    patch_y: int = 0
    patch_width: int = 0
    patch_height: int = 0

    # State
    processed: bool = False
    annotation_created_at: str | None = None
    annotation_updated_at: str | None = None
    annotation_type: str = "segmentation_mask"
    session: int = Field(default=-1, description="Session that annotated it (-1: none)")
    roi_id: int | None = None
    label_id: int | None = None
    annotation_input_id: int | None = None
    input_digest: str | None = Field(default=None, description="Pixel SHA-256 of input")
    mask_digest: str | None = Field(default=None, description="Pixel SHA-256 of mask")
    error: str | None = Field(
        default=None, description="Last upload error (local tracking_table.csv only)"
    )

    @model_validator(mode="after")
    def _fill_image_uri(self) -> Row:
        if not self.image_uri and self.image_id is not None:
            self.image_uri = omero_uri(self.image_id)
        return self

    @classmethod
    def from_unit(cls, unit: Unit) -> Row:
        """A new, not yet annotated row for a planned unit."""
        x, y, width, height = unit.patch if unit.patch is not None else (0, 0, 0, 0)
        image = unit.identity
        return cls(
            image_uri=image.image_uri,
            image_id=image.image_id,
            image_name=image.image_name,
            level=image.level,
            well=image.well,
            field=image.field,
            well_metadata=dict(image.well_metadata),
            annotation_id=unit.annotation_id,
            category=unit.category,
            timepoint=unit.timepoint,
            z_slice=unit.z_slice,
            z_start=unit.z_start,
            z_end=unit.z_end,
            z_length=unit.z_length,
            channel=unit.channel,
            is_volumetric=unit.is_volumetric,
            is_patch=unit.patch is not None,
            patch_x=x,
            patch_y=y,
            patch_width=width,
            patch_height=height,
        )

    def unit(self) -> Unit:
        """The planned unit this row describes."""
        patch = (
            (self.patch_x, self.patch_y, self.patch_width, self.patch_height)
            if self.is_patch
            else None
        )
        return Unit(
            annotation_id=self.annotation_id,
            identity=ImageIdentity(
                image_uri=self.image_uri,
                image_name=self.image_name,
                image_id=self.image_id,
                level=self.level,
                well=self.well,
                field=self.field,
                well_metadata=dict(self.well_metadata),
            ),
            category=self.category,
            timepoint=self.timepoint,
            z_start=self.z_start,
            z_end=self.z_end,
            z_length=self.z_length,
            channel=self.channel,
            is_volumetric=self.is_volumetric,
            patch=patch,
        )

    def mark_processed(self, annotation_type: str = "segmentation_mask") -> None:
        """Mark the unit annotated, keeping its first annotation time."""
        now = datetime.now().isoformat()
        self.processed = True
        self.annotation_updated_at = now
        if self.annotation_created_at is None:
            self.annotation_created_at = now
        self.annotation_type = annotation_type


# --- Table codec ----------------------------------------------------------------

COLUMNS: tuple[str, ...] = (
    "image_id",
    "image_uri",
    "image_name",
    "annotation_id",
    "category",
    "channel",
    "z_slice",
    "timepoint",
    "label_id",
    "roi_id",
    "is_volumetric",
    "processed",
    "is_patch",
    "patch_x",
    "patch_y",
    "patch_width",
    "patch_height",
    "annotation_type",
    "annotation_created_at",
    "annotation_updated_at",
    "annotation_input_id",
    "session",
    "z_start",
    "z_end",
    "z_length",
    "level",
    "well",
    "field",
    "well_metadata",
    "input_digest",
    "mask_digest",
)
_INT_COLUMNS = (
    "image_id",
    "patch_x",
    "patch_y",
    "patch_width",
    "patch_height",
    "z_slice",
    "timepoint",
    "z_start",
    "z_end",
    "z_length",
    "level",
    "session",
)
_BOOL_COLUMNS = ("processed", "is_patch", "is_volumetric")


def _omero_id(value: int | None) -> int:
    return OMERO_ID_UNSET if value is None else int(value)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False  # array-like cells are never "missing"


def _int(row: Any, key: str, default: int) -> int:
    value = row.get(key, default)
    if _is_missing(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _str(row: Any, key: str) -> str:
    value = row.get(key, "")
    return "" if _is_missing(value) else str(value)


def _optional_str(row: Any, key: str) -> str | None:
    value = _str(row, key)
    return None if value in ("", "None") else value


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return False if _is_missing(value) else bool(value)


def _id(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return None if parsed in (OMERO_ID_UNSET, -1) else parsed


def rows_to_frame(rows: Sequence[Row], *, local: bool = False) -> pd.DataFrame:
    """The table holding ``rows``; ``local=True`` adds the CSV-only ``error`` column."""
    records = [
        {
            "image_id": _omero_id(r.image_id),
            "image_uri": r.image_uri,
            "image_name": r.image_name,
            "annotation_id": r.annotation_id,
            "category": r.category,
            "channel": r.channel,
            "z_slice": r.z_slice,
            "timepoint": r.timepoint,
            "label_id": _omero_id(r.label_id),
            "roi_id": _omero_id(r.roi_id),
            "is_volumetric": r.is_volumetric,
            "processed": r.processed,
            "is_patch": r.is_patch,
            "patch_x": r.patch_x,
            "patch_y": r.patch_y,
            "patch_width": r.patch_width,
            "patch_height": r.patch_height,
            "annotation_type": r.annotation_type,
            "annotation_created_at": r.annotation_created_at or "None",
            "annotation_updated_at": r.annotation_updated_at or "None",
            "annotation_input_id": _omero_id(r.annotation_input_id),
            "session": r.session,
            "z_start": r.z_start,
            "z_end": r.z_end,
            "z_length": r.z_length,
            "level": r.level,
            "well": r.well,
            "field": r.field,
            "well_metadata": json.dumps(r.well_metadata, sort_keys=True),
            "input_digest": r.input_digest or "",
            "mask_digest": r.mask_digest or "",
            **({"error": r.error or ""} if local else {}),
        }
        for r in rows
    ]
    columns = [*COLUMNS, "error"] if local else list(COLUMNS)
    frame = pd.DataFrame(records, columns=columns)
    for column in _INT_COLUMNS:
        numbers = pd.Series(pd.to_numeric(frame[column], errors="coerce"))
        frame[column] = numbers.fillna(-1).astype(int)
    for column in _BOOL_COLUMNS:
        frame[column] = frame[column].astype(bool)
    return frame


def rows_from_frame(frame: pd.DataFrame | None) -> list[Row]:
    """The rows of a tracking table (OMERO or CSV).

    Raises:
        ValueError: The table has no ``annotation_id`` column (tables written before
            0.5.0 are not supported).
    """
    if frame is None or len(frame) == 0:
        return []
    if "annotation_id" not in frame.columns:
        raise ValueError(
            "This tracking table has no annotation_id column; tables made before "
            "omero-annotate-ai 0.5.0 are not supported"
        )
    rows: list[Row] = []
    for _, row in frame.iterrows():
        category = row.get("category")
        rows.append(
            Row(
                image_id=_id(row.get("image_id")),
                image_uri=_str(row, "image_uri"),
                image_name=_str(row, "image_name"),
                annotation_id=_str(row, "annotation_id"),
                level=_int(row, "level", 0),
                well=_str(row, "well"),
                field=_str(row, "field"),
                well_metadata=json.loads(_str(row, "well_metadata") or "{}"),
                category=category if category in SPLIT_CATEGORIES else "training",
                timepoint=_int(row, "timepoint", -1),
                z_slice=_int(row, "z_slice", -1),
                channel=_int(row, "channel", -1),
                is_volumetric=_bool(row.get("is_volumetric", False)),
                z_start=_int(row, "z_start", -1),
                z_end=_int(row, "z_end", -1),
                z_length=_int(row, "z_length", 1),
                is_patch=_bool(row.get("is_patch", False)),
                patch_x=_int(row, "patch_x", 0),
                patch_y=_int(row, "patch_y", 0),
                patch_width=_int(row, "patch_width", 0),
                patch_height=_int(row, "patch_height", 0),
                processed=_bool(row.get("processed", False)),
                annotation_type=_str(row, "annotation_type") or "segmentation_mask",
                annotation_created_at=_optional_str(row, "annotation_created_at"),
                annotation_updated_at=_optional_str(row, "annotation_updated_at"),
                session=_int(row, "session", -1),
                roi_id=_id(row.get("roi_id")),
                label_id=_id(row.get("label_id")),
                annotation_input_id=_id(row.get("annotation_input_id")),
                input_digest=_str(row, "input_digest") or None,
                mask_digest=_str(row, "mask_digest") or None,
                error=_str(row, "error") or None,
            )
        )
    return rows


# --- Reports --------------------------------------------------------------------


def progress_summary(rows: Sequence[Row]) -> dict[str, int | float]:
    """Units in total, annotated and pending, and the percentage done."""
    total = len(rows)
    done = sum(r.processed for r in rows)
    return {
        "total_units": total,
        "completed_units": done,
        "pending_units": total - done,
        "progress_percent": round(100 * done / total, 1) if total else 0,
    }


def _split_level(rows: Iterable[Row], key: Callable[[Row], str]) -> dict[str, Any]:
    """Per split, how many distinct keys it has; and the keys found in several."""
    splits: dict[str, set[str]] = {}
    for row in rows:
        splits.setdefault(key(row), set()).add(row.category)
    return {
        "total": len(splits),
        "per_split": {
            c: sum(c in s for s in splits.values()) for c in SPLIT_CATEGORIES
        },
        "spanning_splits": sorted(k for k, s in splits.items() if len(s) > 1),
    }


def split_report(
    rows: Sequence[Row],
    split_unit: str,
    stratify_by: Sequence[str] = (),
    split_rationale: str | None = None,
) -> dict[str, Any]:
    """How images and wells are distributed over the splits.

    Records rather than judges: for the declared split unit and every level above it,
    lists the units whose rows fall in more than one split. The ``wells`` section is
    present only when rows carry well information.
    """
    report: dict[str, Any] = {
        "split_unit": split_unit,
        "split_rationale": split_rationale,
        "images": _split_level(rows, lambda r: r.image_uri),
    }
    if any(r.well for r in rows):
        report["wells"] = _split_level([r for r in rows if r.well], lambda r: r.well)
    if stratify_by:
        unit: Callable[[Row], str] = (
            (lambda r: r.well) if split_unit == "well" else (lambda r: r.image_uri)
        )
        by_stratum: dict[str, list[Row]] = {}
        for r in rows:
            label = ", ".join(
                f"{k}={r.well if k == 'well' else r.well_metadata.get(k, '(missing)')}"
                for k in stratify_by
            )
            by_stratum.setdefault(label, []).append(r)
        wanted = {r.category for r in rows}
        report["stratify_by"] = list(stratify_by)
        report["strata"] = {
            label: _split_level(group, unit)["per_split"]
            for label, group in by_stratum.items()
        }
        report["strata_missing_splits"] = {
            label: [c for c in SPLIT_CATEGORIES if c in wanted and not counts[c]]
            for label, counts in report["strata"].items()
            if any(c in wanted and not counts[c] for c in SPLIT_CATEGORIES)
        }
    return report


def split_summary(report: dict[str, Any]) -> str:
    """One-paragraph, human-readable version of a :func:`split_report`."""
    parts = [f"Split unit: {report['split_unit']}."]
    for level in ("images", "wells"):
        if level not in report:
            continue
        section = report[level]
        spanning = section["spanning_splits"]
        if spanning:
            parts.append(
                f"{len(spanning)} of {section['total']} {level} have annotations in "
                f"more than one split ({', '.join(spanning)})."
            )
        else:
            parts.append(
                f"None of {section['total']} {level} span more than one split."
            )
    if "strata" in report:
        stratified = f"Stratified by {', '.join(report['stratify_by'])}"
        missing = report["strata_missing_splits"]
        if missing:
            parts.append(
                f"{stratified}: {len(missing)} of {len(report['strata'])} strata are "
                "not in every split ("
                + "; ".join(f"{s} lacks {', '.join(m)}" for s, m in missing.items())
                + ")."
            )
        else:
            parts.append(
                f"{stratified}: every stratum ({len(report['strata'])}) is in every split."
            )
    if report["split_rationale"]:
        parts.append(f"Rationale: {report['split_rationale'].rstrip('.')}.")
    return " ".join(parts)
