"""Map an AnnotationConfig onto BioImage Model Zoo (BMZ) export metadata.

Two export targets consume the result:
- micro-sam: ``export_sam_model(..., **to_micro_sam_kwargs(meta))``
- BiaPy:     ``BiaPy(...).export_model_to_bmz(folder, bmz_cfg=to_biapy_bmz_cfg(meta))``

This module is pure: it imports neither micro_sam nor biapy, so it stays importable and
unit-testable without either. It only reads an AnnotationConfig.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Dataset license string -> SPDX identifier (BMZ wants SPDX, not the MIFA enum codes).
_SPDX_LICENSE_MAP = {
    "CC-BY-4.0": "CC-BY-4.0",
    "CC-BY": "CC-BY-4.0",
    "CC_BY": "CC-BY-4.0",
    "CC0": "CC0-1.0",
    "CC0-1.0": "CC0-1.0",
    "CC-0": "CC0-1.0",
}
_DEFAULT_SPDX_LICENSE = "CC-BY-4.0"


def _to_spdx_license(license_str: Optional[str]) -> str:
    """Map a dataset license string to an SPDX identifier (default CC-BY-4.0)."""
    return _SPDX_LICENSE_MAP.get((license_str or "").strip(), _DEFAULT_SPDX_LICENSE)


def _build_authors(config) -> List[Dict[str, str]]:
    """Map config authors to BMZ author dicts; drop authors without a name."""
    authors: List[Dict[str, str]] = []
    for author in config.authors:
        if not author.name or not author.name.strip():
            continue
        entry = {"name": author.name.strip()}
        if author.affiliation:
            entry["affiliation"] = author.affiliation
        authors.append(entry)
    return authors


def _build_tags(config) -> List[str]:
    """Merge tags + study keywords + framework + organism, de-duplicated."""
    tags: List[str] = []
    candidates = [
        *config.tags,
        *config.study.keywords,
        config.ai_model.framework,
        config.study.organism,
    ]
    for tag in candidates:
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _build_cite(config) -> List[Dict[str, str]]:
    """Build the BMZ citation list from the model DOI, if present."""
    if config.ai_model.model_doi:
        return [{"text": "model", "doi": config.ai_model.model_doi}]
    return []


def _build_data(config) -> Dict[str, str]:
    """Build the BMZ training-data reference (name / doi / image_modality)."""
    data: Dict[str, str] = {}
    name = config.dataset.source_description or config.study.title
    if name:
        data["name"] = name
    doi = config.dataset.source_dataset_id or (
        str(config.dataset.source_dataset_url)
        if config.dataset.source_dataset_url
        else None
    )
    if doi:
        data["doi"] = doi
    if config.study.imaging_method:
        data["image_modality"] = config.study.imaging_method
    return data


def config_to_bmz_metadata(config) -> Dict[str, Any]:
    """Build the common BMZ metadata dict from an AnnotationConfig.

    Scalar fields fall back to ``config.name`` when blank; list/dict fields may be
    empty so each export tool's own defaults apply.
    """
    return {
        "model_name": config.ai_model.model_name or config.name,
        "model_type": config.ai_model.pretrained_from,
        "description": config.study.description or config.study.title or config.name,
        "authors": _build_authors(config),
        "license": _to_spdx_license(config.dataset.license),
        "tags": _build_tags(config),
        "cite": _build_cite(config),
        "data": _build_data(config),
    }


def to_micro_sam_kwargs(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Metadata kwargs for ``micro_sam ... export_sam_model(**kwargs)``.

    ``name``/``model_type`` are passed positionally by the caller; the dataset link
    rides on ``cite`` (export_sam_model has no training_data kwarg). Empty list fields
    are omitted so micro-sam's defaults apply.
    """
    kwargs: Dict[str, Any] = {
        "description": meta["description"],
        "license": meta["license"],
    }
    if meta["authors"]:
        kwargs["authors"] = meta["authors"]
    if meta["tags"]:
        kwargs["tags"] = meta["tags"]
    if meta["cite"]:
        kwargs["cite"] = meta["cite"]
    return kwargs


def to_biapy_bmz_cfg(meta: Dict[str, Any]) -> Dict[str, Any]:
    """The ``bmz_cfg`` dict for ``BiaPy(...).export_model_to_bmz(folder, bmz_cfg=...)``."""
    cfg: Dict[str, Any] = {
        "model_name": meta["model_name"],
        "description": meta["description"],
        "license": meta["license"],
        "authors": meta["authors"],
        "tags": meta["tags"],
        "cite": meta["cite"],
    }
    if meta["data"]:
        cfg["data"] = meta["data"]
    return cfg
