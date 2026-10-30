"""Train/validation/test split assignment and leakage reporting.

Annotations are assigned to a split category per *group*, where the group is set
by the split level:

- ``timepoint``: one frame of one image (z-slices and patches of that frame stay
  together). Frames of the same field of view can end up in different splits, which
  is the leakiest option and mainly useful as a baseline.
- ``image``: one OMERO image (all its timepoints, z-slices and patches).
- ``well``: one plate well (all fields of view in it).

Groups are shuffled with a seeded random generator, so a split is reproducible from
the recorded ``random_seed``. With ``stratify_by``, groups are allocated stratum by
stratum (a stratum being a combination of well metadata values such as cell line and
treatment), so every category receives a proportional share of each stratum where
the stratum is large enough.
"""

import random
from collections import defaultdict
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

SPLIT_CATEGORIES = ("training", "validation", "test")
SPLIT_LEVELS = ("timepoint", "image", "well")

MISSING_METADATA = "<missing>"

# How units at each split level are named in messages
_UNIT_NAMES = {"timepoint": "frames", "image": "images", "well": "wells"}


def group_key(annotation: Any, level: str) -> Tuple[Hashable, ...]:
    """Return the key of the split group an annotation belongs to at ``level``."""
    if level == "timepoint":
        return (annotation.image_id, annotation.timepoint)
    if level == "image":
        return (annotation.image_id,)
    if level == "well":
        if annotation.well_id is None:
            raise ValueError(
                f"split_level='well' needs well information, but annotation "
                f"{annotation.annotation_id!r} (image {annotation.image_id}) has none. "
                "Well-level splits are only available for plate containers."
            )
        return (annotation.well_id,)
    raise ValueError(f"Unknown split level {level!r}; expected one of {SPLIT_LEVELS}")


def stratum_key(annotation: Any, stratify_by: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Return the stratum of an annotation: its well metadata values for ``stratify_by``."""
    if not stratify_by:
        return ()
    metadata = annotation.well_metadata or {}
    return tuple(str(metadata.get(key, MISSING_METADATA)) for key in stratify_by)


def category_targets(
    n_groups: int, train_fraction: float, validation_fraction: float, test_fraction: float
) -> Dict[str, int]:
    """Number of groups per category in fraction mode.

    Validation and test are rounded down; every remaining group goes to training,
    so all groups are used even when the fractions sum to less than one.
    """
    n_val = int(n_groups * validation_fraction)
    n_test = int(n_groups * test_fraction)
    return {"training": n_groups - n_val - n_test, "validation": n_val, "test": n_test}


def _interleave_strata(
    strata: Dict[Tuple[str, ...], List[Tuple[Hashable, ...]]], rng: random.Random
) -> List[Tuple[Hashable, ...]]:
    """Round-robin over shuffled strata, so any prefix covers the strata evenly."""
    order = list(strata)
    rng.shuffle(order)
    queues = [list(strata[s]) for s in order]
    interleaved = []
    while any(queues):
        for queue in queues:
            if queue:
                interleaved.append(queue.pop(0))
    return interleaved


def _allocate(
    ordered_groups: List[Tuple[Hashable, ...]],
    group_stratum: Dict[Tuple[Hashable, ...], Tuple[str, ...]],
    targets: Dict[str, int],
) -> Dict[Tuple[Hashable, ...], str]:
    """Assign groups to categories, hitting ``targets`` exactly.

    Groups are visited stratum by stratum. Each group goes to the category that is
    furthest behind its proportional share at that point, which spreads every
    stratum over the categories in proportion to the targets.
    """
    total = sum(targets.values())
    by_stratum: Dict[Tuple[str, ...], List[Tuple[Hashable, ...]]] = defaultdict(list)
    for group in ordered_groups:
        by_stratum[group_stratum[group]].append(group)

    assigned = {category: 0 for category in SPLIT_CATEGORIES}
    assignment = {}
    position = 0
    for stratum_groups in by_stratum.values():
        for group in stratum_groups:
            position += 1
            open_categories = [c for c in SPLIT_CATEGORIES if assigned[c] < targets[c]]
            category = max(
                open_categories,
                key=lambda c: targets[c] * position / total - assigned[c],
            )
            assignment[group] = category
            assigned[category] += 1
    return assignment


def assign_split_categories(
    annotations: Sequence[Any],
    *,
    level: str = "image",
    seed: int,
    fractions: Optional[Tuple[float, float, float]] = None,
    counts: Optional[Tuple[int, int, int]] = None,
    stratify_by: Optional[Sequence[str]] = None,
) -> List[Any]:
    """Set ``category`` on annotations by group and return the annotations kept.

    Exactly one of ``fractions`` (train, validation, test; all groups are used) or
    ``counts`` (number of groups per category; the rest is dropped) must be given.

    Args:
        annotations: ImageAnnotation-like objects (need image_id, timepoint, well_id,
            well_metadata, annotation_id and a writable category).
        level: Split level, one of SPLIT_LEVELS.
        seed: Seed for the random generator; the same seed gives the same split.
        fractions: Category fractions for fraction mode.
        counts: Group counts for count mode.
        stratify_by: Well metadata keys to stratify on.

    Returns:
        The annotations that were assigned a category, in their original order.
    """
    if (fractions is None) == (counts is None):
        raise ValueError("Give exactly one of fractions or counts")

    rng = random.Random(seed)

    group_members: Dict[Tuple[Hashable, ...], List[Any]] = {}
    group_stratum: Dict[Tuple[Hashable, ...], Tuple[str, ...]] = {}
    for annotation in annotations:
        key = group_key(annotation, level)
        group_members.setdefault(key, []).append(annotation)
        group_stratum.setdefault(key, stratum_key(annotation, stratify_by))

    strata: Dict[Tuple[str, ...], List[Tuple[Hashable, ...]]] = defaultdict(list)
    for key in group_members:
        strata[group_stratum[key]].append(key)
    for groups in strata.values():
        rng.shuffle(groups)

    ordered = _interleave_strata(strata, rng)

    if fractions is not None:
        targets = category_targets(len(ordered), *fractions)
    else:
        targets = dict(zip(SPLIT_CATEGORIES, counts))
        n_requested = sum(counts)
        if n_requested > len(ordered):
            unit = _UNIT_NAMES[level]
            raise ValueError(
                f"Not enough {unit}: requested {counts[0]} training + "
                f"{counts[1]} validation + {counts[2]} test = {n_requested}, "
                f"but only {len(ordered)} {unit} available. "
                "Reduce train_n, validate_n, or test_n in the annotation settings."
            )
        ordered = ordered[:n_requested]

    assignment = _allocate(ordered, group_stratum, targets)

    kept = []
    for annotation in annotations:
        category = assignment.get(group_key(annotation, level))
        if category is not None:
            annotation.category = category
            kept.append(annotation)
    return kept


def split_leakage_report(
    annotations: Sequence[Any],
    levels: Sequence[str] = SPLIT_LEVELS,
    metadata_keys: Sequence[str] = (),
) -> Dict[str, Dict[str, Any]]:
    """Report, per level, how many groups have members in more than one split.

    A group shared between splits means that closely related data (the same frame,
    field of view or well, or the same metadata value such as a treatment) is seen
    both in training and in evaluation.

    Args:
        annotations: Annotations with a category assigned.
        levels: Split levels to check. ``well`` is skipped when no annotation has
            well information.
        metadata_keys: Well metadata keys to check as well (e.g. ``treatment``).
            Sharing these is often intended; the report makes it explicit.

    Returns:
        ``{name: {"n_groups", "n_shared", "shared": {group: [categories]}}}``.
    """
    report: Dict[str, Dict[str, Any]] = {}

    def summarise(name, key_of):
        categories: Dict[Hashable, set] = defaultdict(set)
        for annotation in annotations:
            categories[key_of(annotation)].add(annotation.category)
        shared = {
            group: sorted(cats, key=SPLIT_CATEGORIES.index)
            for group, cats in categories.items()
            if len(cats) > 1
        }
        report[name] = {"n_groups": len(categories), "n_shared": len(shared), "shared": shared}

    for level in levels:
        if level == "well" and all(a.well_id is None for a in annotations):
            continue
        summarise(level, lambda a, level=level: group_key(a, level))

    for key in metadata_keys:
        summarise(key, lambda a, key=key: (a.well_metadata or {}).get(key, MISSING_METADATA))

    return report
