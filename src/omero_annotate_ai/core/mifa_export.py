"""Export an :class:`AnnotationConfig` to MIFA (BioImage Archive) metadata documents.

MIFA is the BioImage Archive's "Metadata, Incentives, Formats and Accessibility"
annotation-metadata model (https://github.com/BioImage-Archive/bia-mifa-models).
A MIFA submission consists of three standalone YAML documents:

- ``Study_<accession>.yaml``       - general study / dataset metadata
- ``Annotations_<accession>.yaml`` - annotation metadata + one record per annotation file
- ``Version_<accession>.yaml``     - version / changelog metadata

This module builds those documents from an ``AnnotationConfig`` using the upstream
LinkML *dataclasses* (``bia_mifa_models.datamodel``). The upstream generated
*pydantic* model is intentionally **not** used: it is emitted with the legacy
pydantic-v1 pattern (``WeakRefShimBaseModel`` + class-keyword config) and fails to
import under pydantic v2. The LinkML dataclasses import cleanly, enforce required
slots at construction (our validation gate), and serialize via linkml-runtime's
``yaml_dumper`` to the exact MIFA submission YAML shape.

``bia-mifa-models`` is an optional dependency, imported lazily so that importing
this module - and using the pure mapping helpers below - never requires it.
"""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

# Link used for the MIFA Study ``link_url`` slot (required) when the config carries
# no repository / dataset / documentation URL of its own.
_PROJECT_REPO_URL = "https://github.com/Leiden-Cell-Observatory/omero_annotate_ai"

# --------------------------------------------------------------------------- #
# Pure mapping helpers (no upstream dependency required)
# --------------------------------------------------------------------------- #

# Our AnnotationMethodology.annotation_type literals -> MIFA AnnotationType codes.
_ANNOTATION_TYPE_MAP = {
    "segmentation_mask": "segmentation_mask",
    "semantic_segmentation": "segmentation_mask",
    "bounding_box": "bounding_boxes",
    "point": "point_annotations",
    "classification": "class_labels",
}

# Our dataset license strings -> MIFA LicenseType codes (only CC0 / CC_BY exist).
_LICENSE_MAP = {
    "CC-BY-4.0": "CC_BY",
    "CC-BY": "CC_BY",
    "CC_BY": "CC_BY",
    "CCBY": "CC_BY",
    "CC0": "CC0",
    "CC0-1.0": "CC0",
    "CC-0": "CC0",
}


def map_license(license_str: Optional[str]) -> str:
    """Map a free-form license string to a MIFA ``LicenseType`` code."""
    key = (license_str or "").strip()
    if key in _LICENSE_MAP:
        return _LICENSE_MAP[key]
    warnings.warn(
        f"Unrecognized license {license_str!r}; defaulting MIFA license to 'CC_BY'.",
        UserWarning,
        stacklevel=2,
    )
    return "CC_BY"


def map_annotation_type(annotation_type: Optional[str]) -> str:
    """Map an annotation-type string to a MIFA ``AnnotationType`` code."""
    key = (annotation_type or "").strip()
    if key in _ANNOTATION_TYPE_MAP:
        return _ANNOTATION_TYPE_MAP[key]
    warnings.warn(
        f"Unrecognized annotation type {annotation_type!r}; using MIFA 'other'.",
        UserWarning,
        stacklevel=2,
    )
    return "other"


def split_author_name(name: Optional[str]) -> Optional[Tuple[str, str]]:
    """Split a single author name string into ``(first_name, last_name)``.

    MIFA requires both an author first and last name. Returns ``None`` when the
    name is empty/blank (such authors are dropped from the export). A single-token
    name becomes ``(token, ".")`` so the required last-name slot stays non-empty;
    surname particles ("van der") are kept together in the last name.
    """
    if not name or not name.strip():
        return None
    tokens = name.split()
    if len(tokens) == 1:
        return (tokens[0], ".")
    return (tokens[0], " ".join(tokens[1:]))


def version_to_float(version: Optional[str]) -> float:
    """Coerce a (possibly semver) version string to a float for the MIFA Version doc.

    The upstream ``Version.version`` dataclass slot is mistyped as ``float`` (the
    schema declares it a string), so semver strings like ``"1.0.0"`` crash there.
    We derive ``major.minor`` (e.g. ``"1.0.0" -> 1.0``, ``"v1.1.0" -> 1.1``) and
    fall back to ``1.0`` with a warning when nothing numeric can be extracted.
    """
    s = re.sub(r"^[vV]", "", (version or "").strip())
    parts = s.split(".")
    try:
        if len(parts) >= 2:
            return float(f"{int(parts[0])}.{int(parts[1])}")
        return float(int(parts[0]))
    except (ValueError, IndexError):
        pass
    try:
        return float(s)
    except ValueError:
        warnings.warn(
            f"Could not derive a numeric MIFA version from {version!r}; using 1.0.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0


def compose_overview(config) -> str:
    """Build the required MIFA ``annotation_overview`` free-text from the config."""
    study = config.study
    meth = config.annotation_methodology
    dataset = config.dataset
    base = (
        study.description or study.title or config.name or "Annotation dataset"
    ).strip()
    parts = [base.rstrip(".") + "."]
    parts.append(
        f"Annotation type: {meth.annotation_type}; coverage: {meth.annotation_coverage}."
    )
    if meth.annotation_criteria:
        parts.append(f"Criteria: {meth.annotation_criteria.rstrip('.')}.")
    if study.organism:
        parts.append(f"Organism: {study.organism}.")
    if study.imaging_method:
        parts.append(f"Imaging method: {study.imaging_method}.")
    src = dataset.source_dataset_id or dataset.source_description
    if src:
        parts.append(f"Source dataset: {src}.")
    return " ".join(parts)


def compose_method(config) -> str:
    """Build the required MIFA ``annotation_method`` free-text from the config.

    MIFA has no structured slot for manual-vs-automated or AI-model provenance, so
    that information is captured here as free text.
    """
    meth = config.annotation_methodology
    ai = config.ai_model
    tool = meth.annotation_tool
    how = meth.annotation_method or (tool.mode if tool else "unspecified")
    parts = [f"Annotation method: {how}."]
    if tool:
        version = f" v{tool.version}" if tool.version else ""
        parts.append(f"Annotation tool: {tool.name}{version} (mode: {tool.mode}).")
    model_bits = ai.model_name or ai.pretrained_from or ""
    model_clause = f" model {model_bits}" if model_bits else ""
    parts.append(
        f"AI framework: {ai.framework}{model_clause} "
        f"(version {ai.model_version}, training mode {ai.training_mode})."
    )
    if ai.model_url:
        parts.append(f"Model URL: {ai.model_url}.")
    if ai.model_doi:
        parts.append(f"Model DOI: {ai.model_doi}.")
    return " ".join(parts)


def _slugify(text: Optional[str]) -> str:
    """Filesystem-safe identifier (keeps alphanumerics, ``.`` ``_`` ``-``)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip()).strip("_")
    return slug or "export"


def resolve_accession(config, accession: Optional[str] = None) -> str:
    """Pick the accession used in MIFA filenames: explicit > source id > name slug."""
    return accession or config.dataset.source_dataset_id or _slugify(config.name)


# --------------------------------------------------------------------------- #
# Builders that use the upstream MIFA LinkML dataclasses (optional dependency)
# --------------------------------------------------------------------------- #


def _import_mifa():
    """Lazily import the upstream MIFA dataclasses and the linkml YAML dumper.

    Imported here (not at module scope) so the pure helpers above stay usable
    without the optional ``bia-mifa-models`` dependency installed.
    """
    try:
        from bia_mifa_models.datamodel import bia_mifa_models as models
        from linkml_runtime.dumpers import yaml_dumper
    except ImportError as exc:  # pragma: no cover - exercised only when dep missing
        raise ImportError(
            "MIFA export requires the 'bia-mifa-models' package and 'linkml-runtime'. "
            "Install with: pip install "
            "'git+https://github.com/BioImage-Archive/bia-mifa-models.git@v1.1.5' "
            "(or use the pixi 'mifa' / 'dev' environment)."
        ) from exc
    return models, yaml_dumper


def build_mifa_authors(config) -> List[Any]:
    """Map ``config.authors`` to MIFA ``Author`` objects (blank-name authors dropped)."""
    models, _ = _import_mifa()
    authors = []
    for author in config.authors:
        split = split_author_name(author.name)
        if split is None:
            continue
        first, last = split
        kwargs: Dict[str, Any] = {
            "author_first_name": first,
            "author_last_name": last,
        }
        if author.email:
            kwargs["email"] = author.email
        if author.orcid:
            kwargs["orcid_id"] = str(author.orcid)
        if author.affiliation:
            kwargs["organisation"] = [
                models.OrganisationInfo(organisation_name=author.affiliation)
            ]
        authors.append(models.Author(**kwargs))
    return authors


def _spatial_information(img) -> str:
    """Encode plane / patch / volume position for a MIFA ``spatial_information`` slot."""
    bits = [f"t={img.timepoint}", f"z={img.z_slice}", f"c={img.channel}"]
    if img.is_patch:
        bits.append(
            f"patch=(x={img.patch_x},y={img.patch_y},"
            f"w={img.patch_width},h={img.patch_height})"
        )
    if img.is_volumetric:
        bits.append(f"z_range=({img.z_start}-{img.z_end})")
    return ",".join(bits)


def _source_paths(config, img) -> Tuple[str, str]:
    """``(mask, image)`` paths relative to ``config.output.output_directory``.

    This is the annotation pipeline's own on-disk layout, whose folder names describe
    the pipeline's roles for the files: ``input`` is what was fed to the annotation
    model, ``output`` is the masks it produced.
    """
    input_dir = (
        "label_input" if config.spatial_coverage.uses_separate_channels() else "input"
    )
    return (
        f"output/{img.annotation_id}_mask.tif",
        f"{input_dir}/{img.annotation_id}.tif",
    )


def _bundle_paths(img) -> Tuple[str, str]:
    """``(mask, image)`` paths relative to the BIA submission root.

    The BioImage Archive prescribes no folder names - the file lists just carry paths
    relative to the submission root. So the bundle uses names that describe the content
    to someone browsing the archive, rather than the pipeline's internal input/output
    vocabulary.
    """
    return (
        f"{BUNDLE_ANNOTATION_DIR}/{img.annotation_id}_mask.tif",
        f"{BUNDLE_IMAGE_DIR}/{img.annotation_id}.tif",
    )


def _file_ids(config, img, file_id_source: str) -> Tuple[str, str]:
    """Return ``(annotation_id, source_image_id)`` for one ImageAnnotation record.

    ``file_id_source`` is one of ``"omero"``, ``"local"`` or ``"auto"`` (default).
    ``auto`` decides per record: OMERO ids when the record has been uploaded
    (``label_id`` set), otherwise the file's path inside the BIA submission bundle.
    """
    use_omero = file_id_source == "omero" or (
        file_id_source == "auto" and img.label_id is not None
    )
    if use_omero:
        annotation_id = (
            str(img.label_id) if img.label_id is not None else img.annotation_id
        )
        return annotation_id, str(img.image_id)
    return _bundle_paths(img)


def build_mifa_file_metadata(config, *, file_id_source: str = "auto") -> List[Any]:
    """Map each ``ImageAnnotation`` to a MIFA ``FileLevelMetadata`` record."""
    models, _ = _import_mifa()
    records = []
    for img in config.annotations:
        annotation_id, source_image_id = _file_ids(config, img, file_id_source)
        kwargs: Dict[str, Any] = {
            "annotation_id": annotation_id,
            "annotation_type": [map_annotation_type(img.annotation_type)],
            "source_image_id": source_image_id,
            "spatial_information": _spatial_information(img),
        }
        if img.annotation_created_at:
            kwargs["annotation_creation_time"] = img.annotation_created_at
        records.append(models.FileLevelMetadata(**kwargs))
    return records


def build_mifa_study(config, *, funding_statement: Optional[str] = None) -> Any:
    """Build the MIFA ``Study`` document from the config (with fallbacks for requireds)."""
    models, _ = _import_mifa()
    study = config.study
    title = study.title or config.name or "Untitled annotation study"
    description = study.description or study.title or config.name or title
    keywords = list(study.keywords) or list(config.tags) or ["bioimage annotation"]
    links = [
        str(url)
        for url in (
            config.repository,
            config.dataset.source_dataset_url,
            config.documentation,
        )
        if url
    ] or [_PROJECT_REPO_URL]
    funding = funding_statement or study.funding_statement or "Not specified."

    kwargs: Dict[str, Any] = {
        "title": title,
        "description": description,
        "keywords": keywords,
        "license": map_license(config.dataset.license),
        "funding_statement": funding,
        "link_url": links,
    }
    authors = build_mifa_authors(config)
    if authors:
        kwargs["authors"] = authors
    if config.ai_model.model_url:
        kwargs["ai_models_trained"] = [str(config.ai_model.model_url)]
    return models.Study(**kwargs)


def build_mifa_annotations(config, *, file_id_source: str = "auto") -> Any:
    """Build the MIFA ``Annotations`` document (per-image records + summary metadata)."""
    models, _ = _import_mifa()
    meth = config.annotation_methodology
    kwargs: Dict[str, Any] = {
        "annotation_overview": compose_overview(config),
        "annotation_method": compose_method(config),
    }
    if meth.annotation_criteria:
        kwargs["annotation_criteria"] = meth.annotation_criteria
    if meth.annotation_coverage:
        kwargs["annotation_coverage"] = (
            f"{meth.annotation_coverage} subset of the dataset"
        )
    if config.training.quality_threshold is not None:
        kwargs["annotation_confidence_level"] = (
            f"Quality threshold: {config.training.quality_threshold}"
        )
    authors = build_mifa_authors(config)
    if authors:
        kwargs["authors"] = authors
    kwargs["file_metadata"] = build_mifa_file_metadata(
        config, file_id_source=file_id_source
    )
    return models.Annotations(**kwargs)


def build_mifa_version(config) -> Any:
    """Build the MIFA ``Version`` document from the config."""
    models, _ = _import_mifa()
    return models.Version(
        version=version_to_float(config.version),
        timestamp=config.created.isoformat(),
    )


def to_mifa(
    config, *, file_id_source: str = "auto", funding_statement: Optional[str] = None
) -> Dict[str, Any]:
    """Build all three MIFA documents as upstream dataclass objects.

    Returns a dict with keys ``"study"``, ``"annotations"`` and ``"version"``.
    Construction validates required slots, so a successful call is the validation.
    """
    return {
        "study": build_mifa_study(config, funding_statement=funding_statement),
        "annotations": build_mifa_annotations(config, file_id_source=file_id_source),
        "version": build_mifa_version(config),
    }


def save_mifa(
    config,
    directory: Union[str, Path],
    *,
    accession: Optional[str] = None,
    file_id_source: str = "auto",
    funding_statement: Optional[str] = None,
) -> Dict[str, Path]:
    """Write the three MIFA YAML documents to ``directory``.

    Files are named ``<Class>_<accession>.yaml``. Returns a dict mapping
    ``"study"``/``"annotations"``/``"version"`` to the written paths.
    """
    _, yaml_dumper = _import_mifa()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    accession = _slugify(resolve_accession(config, accession))
    docs = to_mifa(
        config, file_id_source=file_id_source, funding_statement=funding_statement
    )
    paths: Dict[str, Path] = {}
    for key, class_name in (
        ("study", "Study"),
        ("annotations", "Annotations"),
        ("version", "Version"),
    ):
        path = directory / f"{class_name}_{accession}.yaml"
        path.write_text(yaml_dumper.dumps(docs[key]), encoding="utf-8")
        paths[key] = path
    return paths


def to_mifa_dicts(config, *, funding_statement: Optional[str] = None) -> Dict[str, Any]:
    """Return the three MIFA documents as plain dicts (study/annotations/version)."""
    _, yaml_dumper = _import_mifa()
    docs = to_mifa(config, funding_statement=funding_statement)
    return {key: yaml.safe_load(yaml_dumper.dumps(obj)) for key, obj in docs.items()}


# --------------------------------------------------------------------------- #
# BioImage Archive submission bundle (file lists + MIFA metadata + data copy)
# --------------------------------------------------------------------------- #

# BIA file-list columns that are always kept (never dropped as "constant").
_BIA_REQUIRED_COLUMNS = ("Files", "source_image")

# Folder names inside the submission bundle. See _bundle_paths().
BUNDLE_IMAGE_DIR = "images"
BUNDLE_ANNOTATION_DIR = "annotations"


def _drop_constant_optional_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Drop optional BIA columns with < 2 distinct values (BIA guidance)."""
    for col in [c for c in df.columns if c not in _BIA_REQUIRED_COLUMNS]:
        if df[col].nunique(dropna=False) < 2:
            df = df.drop(columns=col)
    return df


def build_bia_file_lists(config) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Build the BioImage Archive file lists (images, annotations) as DataFrames.

    The annotation list has ``Files`` (the mask path) as the first column and a required
    ``source_image`` column linking each annotation to its raw image. Optional columns
    (Channel/Timepoint/Z/Category) are dropped when they carry fewer than two distinct
    values. Paths are local, matching the pipeline's on-disk layout.
    """
    ann_rows: List[Dict[str, Any]] = []
    img_rows: List[Dict[str, Any]] = []
    seen = set()
    for img in config.annotations:
        annotation_id, source_image_id = _file_ids(config, img, "local")
        ann_rows.append(
            {
                "Files": annotation_id,
                "source_image": source_image_id,
                "Channel": img.channel,
                "Timepoint": img.timepoint,
                "Z": img.z_slice,
                "Category": img.category,
            }
        )
        if source_image_id not in seen:
            seen.add(source_image_id)
            img_rows.append({"Files": source_image_id, "Channel": img.channel})

    annotations_df = pd.DataFrame(
        ann_rows,
        columns=["Files", "source_image", "Channel", "Timepoint", "Z", "Category"],
    )
    images_df = pd.DataFrame(img_rows, columns=["Files", "Channel"])
    return (
        _drop_constant_optional_columns(images_df),
        _drop_constant_optional_columns(annotations_df),
    )


def save_bia_package(
    config,
    dest_dir: Union[str, Path],
    *,
    accession: Optional[str] = None,
    funding_statement: Optional[str] = None,
    move_data: bool = False,
) -> Dict[str, Any]:
    """Assemble a self-contained BioImage Archive submission bundle in ``dest_dir``.

    Writes ``metadata/`` (the three MIFA YAMLs), ``file_list_images.tsv`` and
    ``file_list_annotations.tsv``, and transfers every referenced image/mask out of
    ``config.output.output_directory`` into the bundle, from the pipeline's on-disk
    layout (:func:`_source_paths`) to the bundle's own (:func:`_bundle_paths`):

    - ``input/{id}.tif`` (or ``label_input/{id}.tif``) -> ``images/{id}.tif``
    - ``output/{id}_mask.tif``                          -> ``annotations/{id}_mask.tif``

    Returns a summary dict (paths + counts). Missing source files are skipped with a
    warning rather than failing the whole bundle.

    Args:
        config: The annotation config describing the submission.
        dest_dir: Where to build the bundle.
        accession: Accession used in the MIFA filenames.
        funding_statement: Optional override for the Study funding statement.
        move_data: **Moves** the data into the bundle instead of copying it, leaving
            ``config.output.output_directory`` empty of the transferred files. Only use
            this when that directory is a disposable staging area (e.g. one just filled
            by :func:`~omero_annotate_ai.processing.bia_data.prepare_bia_data_from_table`).
            It is destructive against a real annotation run's output directory, which is
            why it defaults to copying.
    """
    dest = Path(dest_dir)
    metadata_dir = dest / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = save_mifa(
        config, metadata_dir, accession=accession, funding_statement=funding_statement
    )

    images_df, annotations_df = build_bia_file_lists(config)
    images_path = dest / "file_list_images.tsv"
    annotations_path = dest / "file_list_annotations.tsv"
    images_df.to_csv(images_path, sep="\t", index=False)
    annotations_df.to_csv(annotations_path, sep="\t", index=False)

    # Map each file from where the pipeline left it to where the bundle wants it.
    transfers: Dict[str, str] = {}
    for img in config.annotations:
        source_mask, source_image = _source_paths(config, img)
        bundle_mask, bundle_image = _bundle_paths(img)
        transfers[source_image] = bundle_image
        transfers[source_mask] = bundle_mask

    source_root = Path(config.output.output_directory)
    transfer = shutil.move if move_data else shutil.copy2
    copied = 0
    missing = 0
    for source_rel, bundle_rel in sorted(transfers.items()):
        src = source_root / source_rel
        if src.exists():
            target = dest / bundle_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            transfer(str(src), str(target))
            copied += 1
        else:
            missing += 1
    if missing:
        verb = "moved" if move_data else "copied"
        warnings.warn(
            f"{missing} referenced data file(s) were not found under {source_root} "
            f"and were not {verb} into the BIA bundle.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "dir": dest,
        "metadata": metadata_paths,
        "file_list_images": images_path,
        "file_list_annotations": annotations_path,
        "n_images": len(images_df),
        "n_annotations": len(annotations_df),
        "copied": copied,
        "missing": missing,
    }
