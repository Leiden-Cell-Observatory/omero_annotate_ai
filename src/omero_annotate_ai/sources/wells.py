"""Well names and well filters, shared by the OMERO and OME-Zarr paths."""

from __future__ import annotations

import re
from typing import Dict, List

_WELL = re.compile(r"^\s*([A-Za-z]{1,2})\s*[/_\- ]?\s*0*(\d+)\s*$")


def normalize_well(name: str) -> str:
    """Canonical well name: ``B2`` for ``B2``, ``b02``, ``B/2`` or ``B_02``."""
    match = _WELL.match(str(name))
    if not match:
        raise ValueError(f"Not a well name: {name!r} (expected e.g. B2, B02 or B/2)")
    return f"{match.group(1).upper()}{int(match.group(2))}"


def match_well_filter(kv: Dict[str, str], filters: Dict[str, List[str]]) -> bool:
    """True when ``kv`` meets ALL filter conditions (key present, value accepted)."""
    for key, accepted in filters.items():
        if key not in kv or kv[key] not in accepted:
            return False
    return True
