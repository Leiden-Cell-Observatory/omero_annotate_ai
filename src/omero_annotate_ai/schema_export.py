"""A training set as Bioimage Annotation Schema documents, and the exports built on them.

The config's schema sections are already the schema's models; this module adds what
only the tool knows: the sessions, the normalisation and model it was set up with,
and one record per row of the tracking table. The RO-Crate, MIFA and BioImage
Archive exports then come from the schema library (``bioimage_annotation_schema``),
so every tool that writes these documents produces the same exports.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from bioimage_annotation_schema import (
    RECORDS_FILE,
    SET_FILE,
    SchemaError,
    check_documents,
    write_records,
    write_set,
)
from bioimage_annotation_schema import datamodel as bas
from bioimage_annotation_schema.crate import write_ro_crate as _write_ro_crate
from bioimage_annotation_schema.mifa import save_bia_package as _save_bia_package

from . import provenance
from .bundle import input_path, mask_path
from .core.annotation_config import AnnotationConfig, SessionRecord
from .rows import Row

SCHEMA_VERSION = "0.1.0"
SOFTWARE = "omero-annotate-ai"
_ANNOTATION_TYPES = {t.value for t in bas.AnnotationType}
_TOOL_NAMES = {"micro_sam": "micro-sam", "cellpose": "cellpose"}

__all__ = [
    "SCHEMA_VERSION",
    "annotation_records",
    "annotation_set",
    "save_bia_package",
    "write_ro_crate",
    "write_schema_documents",
]


def _aware(moment: datetime) -> datetime:
    """The time with its zone; the tool records local times without one."""
    return moment if moment.tzinfo is not None else moment.astimezone()


def _session(record: SessionRecord) -> bas.Session:
    version = f" {record.software_version}" if record.software_version else ""
    return bas.Session.model_validate(
        {
            "id": record.id,
            "kind": record.kind,
            "started": _aware(record.started),
            "ended": _aware(record.ended) if record.ended else None,
            "status": record.status,
            "annotator": record.user,
            "software": f"{SOFTWARE}{version}",
            "model": record.model,
            "n_annotated": record.n_annotated,
            "note": record.note,
        }
    )


def _tools(config: AnnotationConfig) -> list[bas.Tool]:
    """The annotation tools: the config's own, else the AI model it was set up with."""
    if config.methodology.tools:
        return list(config.methodology.tools)
    ai = config.ai_model
    if ai.framework not in _TOOL_NAMES:
        return []
    model_name = ai.model_name or ai.pretrained_from
    model = (
        bas.Model(
            name=model_name,
            version=None if ai.model_version == "latest" else ai.model_version,
            url=str(ai.model_url) if ai.model_url else None,
            doi=ai.model_doi,
        )
        if model_name
        else None
    )
    return [
        bas.Tool.model_validate(
            {
                "name": _TOOL_NAMES[ai.framework],
                "version": ai.framework_version,
                "mode": "interactive",
                "model": model,
            }
        )
    ]


def annotation_set(
    config: AnnotationConfig, rows: Sequence[Row] = ()
) -> bas.AnnotationSet:
    """The config as an :class:`AnnotationSet`: its schema sections plus the
    sessions, normalisation and tools the tool recorded, and, when ``rows`` carry
    ISCC codes, how those codes were made."""
    norm = config.normalization
    methodology = config.methodology.model_copy(
        update={"tools": _tools(config) or None}
    )
    schema_fields: dict[str, Any] = {
        name: getattr(config, name)
        for name in (
            "name",
            "description",
            "created",
            "license",
            "version",
            "changes",
            "previous_version",
            "keywords",
            "links",
            "ai_models_trained",
            "study",
            "source_dataset",
            "classes",
            "sampling",
            "split",
            "notes",
        )
    }
    schema_fields["created"] = _aware(config.created)
    return bas.AnnotationSet(
        schema_version=SCHEMA_VERSION,
        **{k: v for k, v in schema_fields.items() if v not in (None, [], "")},
        authors=config.authors or None,
        methodology=methodology,
        normalization=bas.Normalization.model_validate(
            norm.model_dump(
                include={"method", "percentile_low", "percentile_high", "clip"}
            )
        ),
        sessions=[_session(s) for s in config.sessions] or None,
        content_codes=bas.ContentCodes(region=provenance.code_method())
        if any(r.input_iscc or r.mask_iscc for r in rows)
        else None,
        extensions=config.extensions or None,
    )


def _status(row: Row) -> str:
    if row.excluded:
        return "excluded"
    return "annotated" if row.processed else "pending"


def _time(text: str | None) -> datetime | None:
    return _aware(datetime.fromisoformat(text)) if text else None


def _index(value: int) -> int | None:
    """``None`` for the tracking table's "not set" value (-1)."""
    return value if value >= 0 else None


def annotation_records(rows: Sequence[Row]) -> list[bas.AnnotationRecord]:
    """One :class:`AnnotationRecord` per row of the tracking table."""
    records = []
    for row in rows:
        status = _status(row)
        external = {
            key: str(value)
            for key, value in (
                ("omero_roi", row.roi_id),
                ("omero_label_image", row.label_id),
                ("omero_annotation_input", row.annotation_input_id),
            )
            if value
        }
        fields: dict[str, Any] = {
            "annotation_id": row.annotation_id,
            "image_uri": row.image_uri,
            "image_name": row.image_name or None,
            "level": row.level,
            "well": row.well or None,
            "field": row.field or None,
            "well_metadata": json.dumps(row.well_metadata)
            if row.well_metadata
            else None,
            "channel": _index(row.channel),
            "timepoint": _index(row.timepoint),
            "split": row.category,
            "status": status,
            "created_at": _time(row.annotation_created_at),
            "updated_at": _time(row.annotation_updated_at),
            "session": _index(row.session),
            "input_digest": row.input_digest,
            "input_iscc": row.input_iscc,
            "external_ids": json.dumps(external) if external else None,
        }
        if row.is_volumetric:
            fields.update(z_start=_index(row.z_start), z_end=_index(row.z_end))
        else:
            fields["z_slice"] = _index(row.z_slice)
        if row.is_patch:
            fields.update(
                patch_x=row.patch_x,
                patch_y=row.patch_y,
                patch_width=row.patch_width,
                patch_height=row.patch_height,
            )
        if row.processed:
            fields["input_file"] = input_path(Path(), row.annotation_id).as_posix()
        if status == "annotated":
            fields.update(
                mask_file=mask_path(Path(), row.annotation_id).as_posix(),
                mask_digest=row.mask_digest,
                mask_iscc=row.mask_iscc,
                annotation_type=row.annotation_type
                if row.annotation_type in _ANNOTATION_TYPES
                else "other",
            )
        records.append(
            bas.AnnotationRecord(**{k: v for k, v in fields.items() if v is not None})
        )
    return records


def write_schema_documents(
    config: AnnotationConfig, rows: Sequence[Row], folder: str | Path
) -> tuple[Path, Path]:
    """Write ``annotation-set.yaml`` and ``annotation-records.csv`` into ``folder``
    (validated against the schema, and against each other); returns their paths."""
    folder = Path(folder)
    described, records = annotation_set(config, rows), annotation_records(rows)
    if problems := check_documents(described, records):
        raise SchemaError("The set and its records", problems)
    return (
        write_set(described, folder / SET_FILE),
        write_records(records, folder / RECORDS_FILE),
    )


def write_ro_crate(
    config: AnnotationConfig, rows: Sequence[Row], folder: str | Path
) -> Path:
    """Write the schema documents and ``ro-crate-metadata.json`` into a set's folder."""
    write_schema_documents(config, rows, folder)
    return _write_ro_crate(
        annotation_set(config, rows),
        annotation_records(rows),
        folder,
        metadata_files=(SET_FILE, RECORDS_FILE, "config.yaml", "tracking_table.csv"),
    )


def save_bia_package(
    config: AnnotationConfig,
    rows: Sequence[Row],
    data_dir: str | Path,
    dest_dir: str | Path,
    *,
    accession: str | None = None,
    move_data: bool = False,
) -> dict[str, Any]:
    """A BioImage Archive submission bundle (MIFA metadata, file lists, images and
    masks) of the annotated rows, built from the files in ``data_dir`` (a set's
    folder, or one filled by
    :func:`~omero_annotate_ai.processing.bia_data.prepare_bia_data_from_table`).

    Raises ``MifaNotReady`` naming the fields MIFA requires that the config lacks.
    """
    return _save_bia_package(
        annotation_set(config, rows),
        annotation_records(rows),
        data_dir,
        dest_dir,
        accession=accession,
        move_data=move_data,
    )
