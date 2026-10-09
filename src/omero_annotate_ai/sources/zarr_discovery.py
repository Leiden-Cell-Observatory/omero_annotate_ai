"""Expand OME-Zarr plate / well / image URIs into the images to annotate.

Reads only NGFF metadata, never pixels. Well selection (a ``wells`` list, then a
metadata CSV with optional filters) happens before a well's fields are listed. OME-Zarr
stores no link between such a CSV and its wells, so the match we make is written to
``well_matching.csv``.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from .uri import normalize_zarr_uri
from .wells import match_well_filter, normalize_well
from .zarr import load_ome_zarr

WellKey = Union[str, Tuple[str, str]]


@dataclasses.dataclass(frozen=True)
class ZarrImageRef:
    uri: str
    plate: str = ""
    well: str = ""
    field: str = ""
    well_metadata: Dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def name(self) -> str:
        """``{plate}_{well}_{field}`` for plate images, else empty."""
        if not self.well:
            return ""
        return "_".join(part for part in (self.plate, self.well, self.field) if part)


def _attrs(uri: str) -> Optional[dict]:
    parse_url, _, _ = load_ome_zarr()
    location = parse_url(uri)
    return None if location is None else location.root_attrs


def _read_well_metadata(path: str) -> Tuple[Dict[WellKey, Dict[str, str]], bool]:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "well" not in df.columns:
        raise ValueError(f"{path} needs a 'well' column (found: {list(df.columns)})")
    by_plate = "plate" in df.columns
    rows: Dict[WellKey, Dict[str, str]] = {}
    for record in df.to_dict("records"):
        well = normalize_well(record.pop("well"))
        key = (record.pop("plate"), well) if by_plate else well
        rows[key] = {k: str(v) for k, v in record.items()}
    return rows, by_plate


class _Selector:
    """Decides per well whether it is used, and records every decision."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.wanted: Optional[Set[str]] = (
            {normalize_well(w) for w in cfg.wells} if cfg.wells else None
        )
        self.rows, self.by_plate = (
            _read_well_metadata(cfg.well_metadata) if cfg.well_metadata else ({}, False)
        )
        self.seen: Set[WellKey] = set()
        self.records: List[dict] = []

    def select(self, plate: str, well: str, well_uri: str) -> Optional[Dict[str, str]]:
        """The well's metadata if it is selected, else None."""
        key: WellKey = (plate, well) if self.by_plate else well
        self.seen.add(key)
        if self.wanted is not None and well not in self.wanted:
            return None
        metadata = self.rows.get(key)
        matched = metadata is not None
        kv = metadata or {}
        if self.cfg.well_filters:
            hit = matched and match_well_filter(kv, self.cfg.well_filters)
            included = hit if self.cfg.well_filter_mode == "include" else not hit
        else:
            included = True
        self.records.append({"plate": plate, "well": well, "well_uri": well_uri,
                             "matched": matched, "included": included, **kv})
        return kv if included else None

    def report(self, output_dir: Optional[Path]) -> None:
        if self.cfg.well_metadata:
            no_row = [r["well"] for r in self.records if not r["matched"]]
            if no_row:
                warnings.warn(f"{len(no_row)} wells have no row in {self.cfg.well_metadata}: "
                              f"{', '.join(no_row[:5])}")
            no_well = [str(k) for k in self.rows if k not in self.seen]
            if no_well:
                warnings.warn(f"{len(no_well)} rows in {self.cfg.well_metadata} match no "
                              f"well: {', '.join(no_well[:5])}")
        if output_dir is not None and self.records:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.records).to_csv(output_dir / "well_matching.csv", index=False)


def _field_refs(selector: _Selector, well_uri: str, plate: str, well: str,
                well_attrs: Optional[dict] = None) -> List[ZarrImageRef]:
    metadata = selector.select(plate, well, well_uri)
    if metadata is None:
        return []
    if well_attrs is None:
        well_attrs = _attrs(well_uri) or {}
    fields = [image["path"] for image in well_attrs.get("well", {}).get("images", [])]
    if selector.cfg.fields == "first":
        fields = fields[:1]
    return [ZarrImageRef(f"{well_uri}/{f}", plate, well, f, dict(metadata)) for f in fields]


def discover_zarr_images(cfg, output_dir: Optional[Path] = None) -> List[ZarrImageRef]:
    """Images to annotate for a :class:`ZarrConfig`, in plate row/column order."""
    selector = _Selector(cfg)
    refs: List[ZarrImageRef] = []
    for raw in cfg.uris:
        uri = normalize_zarr_uri(raw)
        attrs = _attrs(uri)
        if attrs and "plate" in attrs:
            plate = attrs["plate"]
            name = str(plate.get("name", ""))
            wells = sorted(plate.get("wells", []),
                           key=lambda w: (w.get("rowIndex", 0), w.get("columnIndex", 0)))
            for well in wells:
                refs += _field_refs(selector, f"{uri}/{well['path']}", name,
                                    normalize_well(well["path"]))
        elif attrs and "well" in attrs:
            parent, row, col = uri.rsplit("/", 2)
            parent_attrs = _attrs(parent) or {}
            name = str(parent_attrs.get("plate", {}).get("name", ""))
            refs += _field_refs(selector, uri, name, normalize_well(f"{row}/{col}"), attrs)
        elif attrs and "multiscales" in attrs:
            refs.append(ZarrImageRef(uri))
        else:
            raise ValueError(
                f"{uri} is not an OME-Zarr plate, well or image "
                f"(found keys: {sorted(attrs or {})})"
            )
    selector.report(output_dir)
    return refs
