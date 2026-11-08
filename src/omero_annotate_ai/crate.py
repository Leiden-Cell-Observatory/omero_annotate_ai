"""Describe a training set's folder as an RO-Crate (``ro-crate-metadata.json``).

An `RO-Crate <https://w3id.org/ro/crate/1.1>`_ is a JSON-LD file next to the data
that says what the folder holds and how it came about, in a form repositories and
workflow tools read. For a training set it lists:

- the dataset (name, description, licence, authors) and the software that made it,
- the files: ``config.yaml``, ``tracking_table.csv``, each annotated image in
  ``annotation_input/`` and each mask in ``annotation_output/``, with the pixel
  SHA-256 digest and, with the ``provenance`` extra, the ISCC code of each,
- the source images by URI (``omero:image:<id>`` or an OME-Zarr URL), which the
  local images are based on; a local OME-Zarr store gets a ``#source-<key>`` id
  and its name only, so no local path (which may hold a user name) is shared,
- every annotation session as an action: who, when, which model, its note, and
  the masks it produced (an annotate session is a ``CreateAction``, a review an
  ``UpdateAction``).

The crate is written on request (:meth:`TrainingSet.write_ro_crate
<omero_annotate_ai.trainingset.TrainingSet.write_ro_crate>`), not after every
commit: it describes the folder at the moment you share or archive it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .bundle import atomic_write, input_path, mask_path
from .sources.uri import parse_omero_uri, uri_key

if TYPE_CHECKING:
    from .core.annotation_config import AnnotationConfig, SessionRecord
    from .rows import Row

METADATA_FILE = "ro-crate-metadata.json"
SPEC = "https://w3id.org/ro/crate/1.1"
SOFTWARE_ID = "https://github.com/Leiden-Cell-Observatory/omero_annotate_ai"

Entity = dict[str, Any]


def _ref(entity_id: str) -> dict[str, str]:
    return {"@id": entity_id}


def _local_id(kind: str, text: str) -> str:
    """A stable, unique ``#kind-...`` id for ``text`` (readable part plus a hash)."""
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:40] or "x"
    return f"#{kind}-{readable}-{uri_key(text)}"


def _source_id(uri: str) -> str:
    """The id of a source image: its URI if it is one (OMERO, a URL), otherwise a
    local id, so no local path (which may hold a user name) enters the crate."""
    if parse_omero_uri(uri) is not None or re.match(r"^[a-z][a-z0-9+.-]*://", uri):
        return uri
    return f"#source-{uri_key(uri)}"


def _license(text: str) -> Entity:
    """An SPDX licence (e.g. ``CC-BY-4.0``) by its URL, anything else by name."""
    if re.fullmatch(r"[A-Za-z0-9.+-]+", text):
        return {
            "@id": f"https://spdx.org/licenses/{text}",
            "@type": "CreativeWork",
            "name": text,
        }
    return {"@id": "#license", "@type": "CreativeWork", "name": text}


def _published(config: AnnotationConfig) -> str:
    """The last time the set changed: its newest session end, else its creation."""
    times = [config.created, *(s.ended or s.started for s in config.sessions)]
    latest = max(t if t.tzinfo else t.replace(tzinfo=UTC) for t in times)
    return latest.isoformat()


class _Graph:
    """Collects entities by ``@id``; the first one written for an id wins."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}

    def add(self, entity: Entity) -> dict[str, str]:
        self.entities.setdefault(entity["@id"], entity)
        return _ref(entity["@id"])

    def digests(self, owner: str, digest: str | None, iscc: str | None) -> list[Any]:
        """``identifier`` values of a file: its pixel digest and ISCC code."""
        found: list[Any] = []
        if digest:
            found.append(
                self.add(
                    {
                        "@id": f"#{owner}-pixel-sha256",
                        "@type": "PropertyValue",
                        "propertyID": "pixel-sha256",
                        "name": "SHA-256 of the pixels (dtype, shape and values)",
                        "value": digest,
                    }
                )
            )
        if iscc:
            found.append(
                self.add(
                    {
                        "@id": f"#{owner}-iscc",
                        "@type": "PropertyValue",
                        "propertyID": "iscc",
                        "name": "ISCC Image-Code (ISO 24138)",
                        "value": iscc,
                    }
                )
            )
        return found


def _file(path: Path, bundle_dir: Path, **extra: Any) -> Entity:
    rel = path.relative_to(bundle_dir).as_posix()
    entity: Entity = {"@id": rel, "@type": "File", "name": path.name}
    if path.suffix in (".tif", ".tiff"):
        entity["encodingFormat"] = "image/tiff"
    elif path.suffix == ".csv":
        entity["encodingFormat"] = "text/csv"
    elif path.suffix in (".yaml", ".yml"):
        entity["encodingFormat"] = "application/yaml"
    entity["contentSize"] = str(path.stat().st_size)
    entity.update({k: v for k, v in extra.items() if v})
    return entity


def _session_action(
    graph: _Graph, record: SessionRecord, masks: list[dict[str, str]]
) -> Entity:
    action: Entity = {
        "@id": f"#session-{record.id}",
        "@type": "UpdateAction" if record.kind == "review" else "CreateAction",
        "name": f"{'Review' if record.kind == 'review' else 'Annotation'} session "
        f"{record.id}",
        "startTime": record.started.isoformat(),
        "actionStatus": {
            "open": "http://schema.org/ActiveActionStatus",
            "finished": "http://schema.org/CompletedActionStatus",
            "interrupted": "http://schema.org/FailedActionStatus",
        }[record.status],
        "instrument": [_ref(SOFTWARE_ID)],
        "result": masks,
    }
    if record.ended:
        action["endTime"] = record.ended.isoformat()
    if record.user:
        action["agent"] = graph.add(
            {
                "@id": _local_id("user", record.user),
                "@type": "Person",
                "name": record.user,
            }
        )
    if record.model:
        action["instrument"].append(
            graph.add(
                {
                    "@id": _local_id("model", record.model),
                    "@type": "SoftwareApplication",
                    "name": record.model,
                }
            )
        )
    if record.note:
        action["description"] = record.note
    return action


def build_ro_crate(
    config: AnnotationConfig,
    rows: Sequence[Row],
    bundle_dir: Path,
    *,
    software_version: str = "unknown",
) -> dict[str, Any]:
    """The RO-Crate metadata (JSON-LD) of a training set's folder.

    Only files that exist are listed; images not saved locally (e.g. never shown)
    appear as source images only.
    """
    graph = _Graph()
    graph.add(
        {
            "@id": SOFTWARE_ID,
            "@type": "SoftwareApplication",
            "name": "omero-annotate-ai",
            "version": software_version,
            "url": SOFTWARE_ID,
        }
    )
    parts: list[dict[str, str]] = []
    for name in ("config.yaml", "tracking_table.csv"):
        if (bundle_dir / name).exists():
            parts.append(graph.add(_file(bundle_dir / name, bundle_dir)))

    masks_by_session: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        source = graph.add(
            {
                "@id": _source_id(row.image_uri),
                "@type": "ImageObject",
                "name": row.image_name,
            }
        )
        image = input_path(bundle_dir, row.annotation_id)
        based_on = [source]
        if image.exists():
            image_ref = graph.add(
                _file(
                    image,
                    bundle_dir,
                    isBasedOn=source,
                    identifier=graph.digests(
                        f"{row.annotation_id}-input", row.input_digest, row.input_iscc
                    ),
                )
            )
            parts.append(image_ref)
            based_on = [image_ref, source]
        mask = mask_path(bundle_dir, row.annotation_id)
        if row.processed and mask.exists():
            mask_ref = graph.add(
                _file(
                    mask,
                    bundle_dir,
                    isBasedOn=based_on,
                    identifier=graph.digests(
                        f"{row.annotation_id}-mask", row.mask_digest, row.mask_iscc
                    ),
                )
            )
            parts.append(mask_ref)
            masks_by_session.setdefault(row.session, []).append(mask_ref)

    actions = [
        graph.add(_session_action(graph, s, masks_by_session.get(s.id, [])))
        for s in config.sessions
    ]
    authors = [
        graph.add(
            {
                "@id": str(a.orcid) if a.orcid else f"#author-{i}",
                "@type": "Person",
                "name": a.name,
                **(
                    {
                        "affiliation": graph.add(
                            {
                                "@id": _local_id("org", a.affiliation),
                                "@type": "Organization",
                                "name": a.affiliation,
                            }
                        )
                    }
                    if a.affiliation
                    else {}
                ),
            }
        )
        for i, a in enumerate(config.authors)
        if a.name
    ]
    root: Entity = {
        "@id": "./",
        "@type": "Dataset",
        "name": config.name,
        "description": config.study.description
        or config.dataset.source_description
        or f"Annotations of the training set {config.name}, made with omero-annotate-ai.",
        "datePublished": _published(config),
        "license": graph.add(_license(config.dataset.license)),
        "hasPart": parts,
        "mentions": actions,
    }
    if authors:
        root["author"] = authors
    if config.study.keywords:
        root["keywords"] = config.study.keywords
    descriptor: Entity = {
        "@id": METADATA_FILE,
        "@type": "CreativeWork",
        "conformsTo": _ref(SPEC),
        "about": _ref("./"),
    }
    entities = [descriptor, root, *graph.entities.values()]
    return {"@context": f"{SPEC}/context", "@graph": [_single(e) for e in entities]}


def _single(entity: Entity) -> Entity:
    """One value instead of a one-item list (the RO-Crate 1.1 JSON-LD convention)."""
    single: Entity = {}
    for key, value in entity.items():
        items = cast(list[Any], value) if isinstance(value, list) else None
        single[key] = items[0] if items is not None and len(items) == 1 else value
    return single


def write_ro_crate(
    config: AnnotationConfig,
    rows: Sequence[Row],
    bundle_dir: Path,
    *,
    software_version: str = "unknown",
) -> Path:
    """Write ``ro-crate-metadata.json`` into ``bundle_dir`` (atomically); its path."""
    crate = build_ro_crate(config, rows, bundle_dir, software_version=software_version)
    target = bundle_dir / METADATA_FILE
    atomic_write(
        target,
        lambda tmp: tmp.write_text(
            json.dumps(crate, indent=2, ensure_ascii=False), encoding="utf-8"
        ),
    )
    return target
