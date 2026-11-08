"""Which images to annotate, in which split, and which units each image gives.

Everything here is pure: inputs (including the seed) in, plan out; no config, OMERO
or disk access. :func:`rng_for` derives a separate random stream per purpose from the
training set's seed, so each choice is reproducible on its own.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .errors import SelectionError
from .sources.base import ImageMeta, Region
from .sources.uri import annotation_id as make_annotation_id

Category = Literal["training", "validation", "test"]
CATEGORIES: tuple[Category, ...] = ("training", "validation", "test")
Mode = Literal["all", "random", "specific"]


def rng_for(seed: int, purpose: str, *parts: object) -> random.Random:
    """A random stream for ``purpose`` (and e.g. an image URI), reproducible from ``seed``."""
    return random.Random(":".join(str(part) for part in (seed, purpose, *parts)))


@dataclass(frozen=True, slots=True)
class ImageIdentity:
    """Which image a unit belongs to, as stored on every row of that image."""

    image_uri: str
    image_name: str
    image_id: int | None = None
    level: int = 0
    well: str = ""
    field: str = ""
    well_metadata: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])


@dataclass(frozen=True, slots=True)
class Unit:
    """One annotation unit: a plane, a volume, or a patch of either.

    ``z_slice`` is the plane (the first slice of a volume); ``patch`` is
    ``(x, y, width, height)`` or ``None`` for the whole plane.
    """

    annotation_id: str
    identity: ImageIdentity
    category: Category
    timepoint: int
    z_start: int
    z_end: int
    z_length: int
    channel: int
    is_volumetric: bool = False
    patch: Region | None = None

    @property
    def z_slice(self) -> int:
        return self.z_start


# --- Splits -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """An image as split planning sees it; ``well=None`` means no well information."""

    uri: str
    plate: str = ""
    well: str | None = None
    well_metadata: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """How images are divided over the splits (``config.training``)."""

    split_unit: Literal["image", "well"] = "image"
    stratify_by: tuple[str, ...] = ()
    segment_all: bool = False
    train_n: int = 0
    validate_n: int = 0
    test_n: int = 0
    validation_fraction: float = 0.0
    test_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """``assignment`` maps each selected URI to its split, training first.

    ``shortfalls`` lists, per condition, the requested splits it did not reach
    (only with ``stratify_by``).
    """

    assignment: Mapping[str, Category]
    n_units: int
    unit_name: str
    n_strata: int
    shortfalls: Mapping[str, tuple[Category, ...]]

    def shortfall_message(self) -> str | None:
        if not self.shortfalls:
            return None
        lacking = "; ".join(
            f"{s} lacks {', '.join(m)}" for s, m in self.shortfalls.items()
        )
        return (
            f"{len(self.shortfalls)} of {self.n_strata} conditions are not in every "
            f"split (too few {self.unit_name}): {lacking}"
        )


def _unit_key(image: ImageInfo, spec: SplitSpec) -> str:
    if spec.split_unit == "image":
        return image.uri
    if image.well is None:
        raise SelectionError(
            f"split_unit='well' needs plate images with well information, "
            f"but {image.uri} has none. Use split_unit='image' for non-plate data."
        )
    return f"{image.plate}/{image.well}"


def _stratum(image: ImageInfo, spec: SplitSpec) -> str:
    well = image.well or ""
    return ", ".join(
        f"{key}={well if key == 'well' else image.well_metadata.get(key, '(missing)')}"
        for key in spec.stratify_by
    )


def _check_stratify_keys(images: Sequence[ImageInfo], spec: SplitSpec) -> None:
    plates = [image for image in images if image.well is not None]
    for key in spec.stratify_by:
        if not any(key == "well" or key in image.well_metadata for image in plates):
            available = sorted({k for image in plates for k in image.well_metadata})
            raise SelectionError(
                f"stratify_by key '{key}' is not found on any image. "
                f"Available well keys: {[*available, 'well']}"
            )


def plan_split(
    images: Sequence[ImageInfo], spec: SplitSpec, rng: random.Random
) -> SplitPlan:
    """Assign images to splits, keeping every split unit (image or well) in one split.

    Counts and fractions refer to split units. With ``stratify_by``, units are
    divided over the splits per condition; count mode fills test and validation
    first so they reach as many conditions as possible.

    Raises:
        SelectionError: Too few units, an unknown ``stratify_by`` key, or
            ``split_unit='well'`` on images without wells.
    """
    unit_name = f"{spec.split_unit}s"
    _check_stratify_keys(images, spec)
    units: dict[str, list[ImageInfo]] = {}
    for image in images:
        units.setdefault(_unit_key(image, spec), []).append(image)
    strata: dict[str, list[str]] = {}
    for key, members in units.items():
        strata.setdefault(_stratum(members[0], spec), []).append(key)
    for keys in strata.values():
        rng.shuffle(keys)

    assignment: dict[str, Category] = {}
    if spec.segment_all:
        for keys in strata.values():
            n = len(keys)
            n_val = min(math.floor(n * spec.validation_fraction + 0.5), n)
            n_test = min(math.floor(n * spec.test_fraction + 0.5), n - n_val)
            for i, key in enumerate(keys):
                assignment[key] = (
                    "validation"
                    if i < n_val
                    else "test"
                    if i < n_val + n_test
                    else "training"
                )
        wanted = {"training"} | {
            c
            for c, f in (
                ("validation", spec.validation_fraction),
                ("test", spec.test_fraction),
            )
            if f > 0
        }
    else:
        counts: tuple[tuple[Category, int], ...] = (
            ("test", spec.test_n),
            ("validation", spec.validate_n),
            ("training", spec.train_n),
        )
        n_requested = sum(n for _, n in counts)
        if n_requested > len(units):
            raise SelectionError(
                f"Not enough {unit_name}: requested {spec.train_n} training + "
                f"{spec.validate_n} validation + {spec.test_n} test = {n_requested}, "
                f"but only {len(units)} {unit_name} available. Reduce train_n, "
                f"validate_n, or test_n in the annotation settings."
            )
        pools = [list(keys) for keys in strata.values()]
        rng.shuffle(pools)
        turn = 0
        for category, n in counts:
            taken = 0
            while taken < n:
                pool = pools[turn % len(pools)]
                turn += 1
                if pool:
                    assignment[pool.pop()] = category
                    taken += 1
        wanted = {category for category, n in counts if n > 0}

    shortfalls: dict[str, tuple[Category, ...]] = {}
    if spec.stratify_by:
        for stratum, keys in strata.items():
            missing = wanted - {assignment[k] for k in keys if k in assignment}
            if missing:
                shortfalls[stratum] = tuple(c for c in CATEGORIES if c in missing)

    ordered = sorted(assignment.items(), key=lambda item: CATEGORIES.index(item[1]))
    return SplitPlan(
        assignment={
            image.uri: category for key, category in ordered for image in units[key]
        },
        n_units=len(units),
        unit_name=unit_name,
        n_strata=len(strata),
        shortfalls=shortfalls,
    )


# --- Units of one image -------------------------------------------------------

PatchFn = Callable[
    [tuple[int, int], random.Random], tuple[list[tuple[int, int]], tuple[int, int]]
]
"""``((height, width), rng) -> (patch origins (x, y), patch (height, width))``."""


@dataclass(frozen=True, slots=True)
class CoverageSpec:
    """What to annotate in each image (``config.spatial_coverage``).

    ``patches`` draws patch origins for an image; ``None`` annotates whole planes.
    """

    channel: int
    timepoint_mode: Mode = "specific"
    timepoints: tuple[int, ...] = (0,)
    n_timepoints: int | None = None
    z_slice_mode: Mode = "specific"
    z_slices: tuple[int, ...] = (0,)
    n_slices: int | None = None
    three_d: bool = False
    patches: PatchFn | None = None


def select_indices(
    size: int,
    mode: Mode,
    listed: Sequence[int],
    n_random: int | None,
    rng: random.Random,
) -> list[int]:
    """Indices of an axis of ``size`` to use: all, a random draw, or the listed ones."""
    if mode == "all":
        return list(range(size))
    if mode == "random":
        return rng.sample(
            range(size), min(n_random if n_random is not None else len(listed), size)
        )
    return [i for i in listed if i < size]


def timepoints_for(meta: ImageMeta, coverage: CoverageSpec, seed: int) -> list[int]:
    rng = rng_for(seed, "timepoints", meta.image_uri)
    return select_indices(
        meta.size_t,
        coverage.timepoint_mode,
        coverage.timepoints,
        coverage.n_timepoints,
        rng,
    )


def z_slices_for(meta: ImageMeta, coverage: CoverageSpec, seed: int) -> list[int]:
    rng = rng_for(seed, "z_slices", meta.image_uri)
    return select_indices(
        meta.size_z, coverage.z_slice_mode, coverage.z_slices, coverage.n_slices, rng
    )


def units_for_image(
    meta: ImageMeta,
    identity: ImageIdentity,
    coverage: CoverageSpec,
    category: Category,
    seed: int,
) -> list[Unit]:
    """Every unit (plane, volume or patch) one image gives; the same for the same seed."""
    z_slices = z_slices_for(meta, coverage, seed)
    if coverage.three_d:
        z_ranges = (
            [(min(z_slices), max(z_slices), len(z_slices))] if z_slices else [(0, 0, 1)]
        )
    else:
        z_ranges = [(z, z, 1) for z in z_slices]
    patch_rng = rng_for(seed, "patches", meta.image_uri)
    units: list[Unit] = []
    for t in timepoints_for(meta, coverage, seed):
        for z_start, z_end, z_length in z_ranges:
            if coverage.patches is None:
                patches: list[Region | None] = [None]
            else:
                origins, (height, width) = coverage.patches(
                    (meta.size_y, meta.size_x), patch_rng
                )
                patches = [(x, y, width, height) for x, y in origins]
            for i, patch in enumerate(patches):
                aid = make_annotation_id(
                    identity.image_uri,
                    t,
                    z_start,
                    volumetric=coverage.three_d,
                    patch=None if patch is None else i,
                )
                units.append(
                    Unit(
                        aid,
                        identity,
                        category,
                        t,
                        z_start,
                        z_end,
                        z_length,
                        coverage.channel,
                        coverage.three_d,
                        patch,
                    )
                )
    return units


# --- Extending a training set -------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtensionPlan:
    """New image URIs per split, and splits that got fewer than asked as ``(got, asked)``."""

    picks: Mapping[Category, tuple[str, ...]]
    shortfalls: Mapping[Category, tuple[int, int]]


def plan_extension(
    existing: Iterable[tuple[str, Category]],
    wanted: Mapping[Category, int],
    available: Callable[[], Sequence[str]],
    rng: random.Random,
) -> ExtensionPlan:
    """Images to add when a set is asked for more images per split than it holds.

    ``existing`` holds ``(image_uri, category)`` per unit. Images are never removed,
    so a lower count changes nothing. ``available`` lists the images the set may
    draw from; it is only called when images must be added.
    """
    have: dict[Category, set[str]] = {c: set() for c in CATEGORIES}
    for uri, category in existing:
        have[category].add(uri)
    missing = {c: max(0, wanted.get(c, 0) - len(have[c])) for c in CATEGORIES}
    if not any(missing.values()):
        return ExtensionPlan(picks={}, shortfalls={})
    used = set[str]().union(*have.values())
    unused = [uri for uri in available() if uri not in used]
    rng.shuffle(unused)
    picks: dict[Category, tuple[str, ...]] = {}
    shortfalls: dict[Category, tuple[int, int]] = {}
    for category in CATEGORIES:
        if n := missing[category]:
            picks[category], unused = tuple(unused[:n]), unused[n:]
            if len(picks[category]) < n:
                shortfalls[category] = (len(picks[category]), n)
    return ExtensionPlan(picks=picks, shortfalls=shortfalls)
