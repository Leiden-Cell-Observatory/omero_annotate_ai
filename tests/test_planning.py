"""Pure planning of splits, units and extensions (planning.py)."""

import random

import pytest

from omero_annotate_ai.errors import SelectionError
from omero_annotate_ai.planning import (
    CoverageSpec,
    ImageIdentity,
    ImageInfo,
    SplitSpec,
    plan_extension,
    plan_split,
    rng_for,
    select_indices,
    units_for_image,
)
from omero_annotate_ai.sources.base import ImageMeta

URIS = [f"omero:image:{i}" for i in range(1, 9)]
IMAGES = [ImageInfo(uri=uri) for uri in URIS]
META = ImageMeta(
    "omero:image:1",
    "img",
    size_t=5,
    size_c=1,
    size_z=4,
    size_y=100,
    size_x=120,
    dtype="uint16",
)
IDENTITY = ImageIdentity("omero:image:1", "img", image_id=1)


def _plate_images(conditions: dict[str, int]) -> list[ImageInfo]:
    """Two fields per well; ``conditions`` maps cell line -> number of wells."""
    images = []
    for line, n_wells in conditions.items():
        for w in range(n_wells):
            for field in range(2):
                images.append(
                    ImageInfo(
                        uri=f"omero:image:{line}{w}{field}",
                        plate="1",
                        well=f"{line}{w}",
                        well_metadata={"cell_line": line},
                    )
                )
    return images


@pytest.mark.unit
class TestRngFor:
    def test_same_purpose_same_stream(self):
        assert rng_for(1, "split").random() == rng_for(1, "split").random()

    def test_purposes_and_parts_differ(self):
        draws = {
            rng_for(1, "split").random(),
            rng_for(1, "planes").random(),
            rng_for(1, "planes", "omero:image:2").random(),
            rng_for(2, "split").random(),
        }
        assert len(draws) == 4


@pytest.mark.unit
class TestPlanSplit:
    def test_counts_golden(self):
        plan = plan_split(
            IMAGES, SplitSpec(train_n=3, validate_n=2, test_n=1), rng_for(42, "split")
        )
        assert dict(plan.assignment) == {
            "omero:image:1": "training",
            "omero:image:6": "training",
            "omero:image:2": "training",
            "omero:image:3": "validation",
            "omero:image:8": "validation",
            "omero:image:4": "test",
        }
        assert plan.n_units == 8 and plan.unit_name == "images"

    def test_fractions_golden(self):
        spec = SplitSpec(segment_all=True, validation_fraction=0.25, test_fraction=0.25)
        plan = plan_split(IMAGES, spec, rng_for(42, "split"))
        categories = list(plan.assignment.values())
        assert categories == ["training"] * 4 + ["validation"] * 2 + ["test"] * 2
        assert set(plan.assignment) == set(URIS)

    def test_same_seed_same_plan(self):
        spec = SplitSpec(train_n=2, validate_n=2)
        assert plan_split(IMAGES, spec, rng_for(3, "split")) == plan_split(
            IMAGES, spec, rng_for(3, "split")
        )

    def test_too_many_requested(self):
        with pytest.raises(SelectionError, match="Not enough images"):
            plan_split(IMAGES, SplitSpec(train_n=9), random.Random(0))

    def test_selection_error_is_a_value_error(self):
        with pytest.raises(ValueError):
            plan_split(IMAGES, SplitSpec(train_n=9), random.Random(0))

    def test_well_unit_keeps_fields_together(self):
        images = _plate_images({"A": 3})
        plan = plan_split(
            images,
            SplitSpec(split_unit="well", train_n=2, validate_n=1),
            random.Random(0),
        )
        assert plan.unit_name == "wells" and plan.n_units == 3
        by_well: dict[str, set[str]] = {}
        for image in images:
            by_well.setdefault(image.well or "", set()).add(plan.assignment[image.uri])
        assert all(len(categories) == 1 for categories in by_well.values())

    def test_well_unit_needs_wells(self):
        with pytest.raises(SelectionError, match="split_unit='well'"):
            plan_split(
                IMAGES, SplitSpec(split_unit="well", train_n=1), random.Random(0)
            )

    def test_unknown_stratify_key(self):
        images = _plate_images({"A": 2})
        with pytest.raises(SelectionError, match="treatment"):
            plan_split(
                images,
                SplitSpec(stratify_by=("treatment",), train_n=1),
                random.Random(0),
            )

    def test_stratified_shortfall_is_reported(self):
        images = _plate_images({"A": 3, "B": 1})
        spec = SplitSpec(
            split_unit="well", stratify_by=("cell_line",), train_n=2, validate_n=2
        )
        plan = plan_split(images, spec, random.Random(0))
        message = plan.shortfall_message()
        assert message is not None and "cell_line=B" in message
        assert plan.n_strata == 2

    def test_no_shortfall_message_without_stratify(self):
        plan = plan_split(IMAGES, SplitSpec(train_n=1), random.Random(0))
        assert plan.shortfall_message() is None


@pytest.mark.unit
class TestSelectIndices:
    def test_modes(self):
        rng = random.Random(0)
        assert select_indices(4, "all", [0], None, rng) == [0, 1, 2, 3]
        assert select_indices(4, "specific", [1, 3, 9], None, rng) == [1, 3]
        drawn = select_indices(10, "random", [0, 0, 0], None, rng)
        assert len(drawn) == 3 and len(set(drawn)) == 3

    def test_random_never_more_than_available(self):
        assert len(select_indices(2, "random", [], 5, random.Random(0))) == 2


@pytest.mark.unit
class TestUnitsForImage:
    def test_random_planes_golden(self):
        coverage = CoverageSpec(
            channel=0,
            timepoint_mode="random",
            n_timepoints=2,
            z_slice_mode="random",
            n_slices=2,
        )
        units = units_for_image(META, IDENTITY, coverage, "training", seed=7)
        assert [(u.timepoint, u.z_slice) for u in units] == [
            (0, 3),
            (0, 2),
            (4, 3),
            (4, 2),
        ]
        assert [u.annotation_id for u in units] == [
            "4ba674d4_0_3",
            "4ba674d4_0_2",
            "4ba674d4_4_3",
            "4ba674d4_4_2",
        ]

    def test_volume_spans_selected_slices(self):
        coverage = CoverageSpec(channel=1, z_slice_mode="all", three_d=True)
        (unit,) = units_for_image(META, IDENTITY, coverage, "validation", seed=0)
        assert (unit.z_start, unit.z_end, unit.z_length) == (0, 3, 4)
        assert unit.is_volumetric and unit.annotation_id.endswith("_0_3d")
        assert (unit.channel, unit.category, unit.identity.image_id) == (
            1,
            "validation",
            1,
        )

    def test_patches_get_their_own_ids(self):
        def two_patches(shape, rng):
            return [(0, 0), (60, 50)], (50, 60)

        coverage = CoverageSpec(channel=0, patches=two_patches)
        units = units_for_image(META, IDENTITY, coverage, "training", seed=0)
        assert [u.patch for u in units] == [(0, 0, 60, 50), (60, 50, 60, 50)]
        assert [u.annotation_id[-2:] for u in units] == ["_0", "_1"]

    def test_identity_is_copied_to_every_unit(self):
        identity = ImageIdentity(
            "omero:image:1",
            "img",
            image_id=1,
            well="B2",
            field="0",
            well_metadata={"k": "v"},
        )
        (unit,) = units_for_image(
            META, identity, CoverageSpec(channel=0), "test", seed=0
        )
        assert (
            unit.identity.well,
            unit.identity.field,
            dict(unit.identity.well_metadata),
        ) == (
            "B2",
            "0",
            {"k": "v"},
        )


@pytest.mark.unit
class TestPlanExtension:
    def test_golden(self):
        plan = plan_extension(
            [("omero:image:1", "training")],
            {"training": 3, "validation": 1},
            lambda: URIS,
            rng_for(7, "extend", 1),
        )
        assert dict(plan.picks) == {
            "training": ("omero:image:4", "omero:image:2"),
            "validation": ("omero:image:3",),
        }
        assert not plan.shortfalls

    def test_nothing_missing_does_not_list_images(self):
        def no_listing():
            raise AssertionError("images listed although nothing is missing")

        plan = plan_extension(
            [("omero:image:1", "training")],
            {"training": 1},
            no_listing,
            random.Random(0),
        )
        assert not plan.picks

    def test_shortfall_when_images_run_out(self):
        plan = plan_extension([], {"training": 5}, lambda: URIS[:2], random.Random(0))
        assert plan.shortfalls == {"training": (2, 5)}

    def test_never_reuses_images_in_the_set(self):
        existing = [(uri, "training") for uri in URIS[:6]]
        plan = plan_extension(
            existing, {"training": 6, "test": 2}, lambda: URIS, random.Random(0)
        )
        assert set(plan.picks["test"]) == set(URIS[6:])
