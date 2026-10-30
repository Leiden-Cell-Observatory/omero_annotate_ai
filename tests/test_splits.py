"""Tests for grouped, stratified and seeded train/validation/test splits."""

from unittest.mock import Mock

import pytest
import yaml

from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    ImageAnnotation,
    create_default_config,
)
from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline
from omero_annotate_ai.core.splits import (
    assign_split_categories,
    category_targets,
    split_leakage_report,
)

CONDITIONS = [
    ("lineA", "DMSO"),
    ("lineA", "drug1"),
    ("lineB", "DMSO"),
    ("lineB", "drug1"),
]


def _plate_units(n_wells_per_condition=3, fields=4, frames=5):
    """Units for a plate: wells x fields (images) x frames, with well metadata."""
    units = []
    well_id = 100
    image_id = 1000
    for cellline, treatment in CONDITIONS:
        for _ in range(n_wells_per_condition):
            well_id += 1
            for _ in range(fields):
                image_id += 1
                for t in range(frames):
                    units.append(
                        ImageAnnotation(
                            image_id=image_id,
                            image_name=f"img{image_id}",
                            annotation_id=f"{image_id}_{t}_0",
                            timepoint=t,
                            well_id=well_id,
                            well=f"W{well_id}",
                            well_metadata={"cellline": cellline, "treatment": treatment},
                        )
                    )
    return units


def _categories_per(units, key):
    groups = {}
    for unit in units:
        groups.setdefault(key(unit), set()).add(unit.category)
    return groups


FRACTIONS = (0.6, 0.2, 0.2)


@pytest.mark.unit
class TestAssignSplitCategories:
    def test_same_seed_same_split(self):
        a = assign_split_categories(_plate_units(), level="image", seed=7, fractions=FRACTIONS)
        b = assign_split_categories(_plate_units(), level="image", seed=7, fractions=FRACTIONS)
        assert [u.category for u in a] == [u.category for u in b]

    def test_different_seed_different_split(self):
        a = assign_split_categories(_plate_units(), level="image", seed=1, fractions=FRACTIONS)
        b = assign_split_categories(_plate_units(), level="image", seed=2, fractions=FRACTIONS)
        assert [u.category for u in a] != [u.category for u in b]

    @pytest.mark.parametrize(
        "level,key",
        [
            ("timepoint", lambda u: (u.image_id, u.timepoint)),
            ("image", lambda u: u.image_id),
            ("well", lambda u: u.well_id),
        ],
    )
    def test_groups_never_span_categories(self, level, key):
        units = assign_split_categories(_plate_units(), level=level, seed=3, fractions=FRACTIONS)
        assert all(len(c) == 1 for c in _categories_per(units, key).values())

    def test_timepoint_level_spreads_images_over_splits(self):
        units = assign_split_categories(_plate_units(), level="timepoint", seed=3, fractions=FRACTIONS)
        per_image = _categories_per(units, lambda u: u.image_id)
        assert any(len(c) > 1 for c in per_image.values())

    def test_fraction_mode_uses_all_groups_with_exact_targets(self):
        units = assign_split_categories(_plate_units(), level="well", seed=5, fractions=FRACTIONS)
        assert len(units) == len(_plate_units())
        wells = {u.well_id: u.category for u in units}
        expected = category_targets(len(wells), *FRACTIONS)
        for category, n in expected.items():
            assert list(wells.values()).count(category) == n

    def test_stratified_well_split_covers_every_condition(self):
        # 5 wells per condition at 0.6/0.2/0.2 -> 3/1/1 of each condition
        units = assign_split_categories(
            _plate_units(n_wells_per_condition=5),
            level="well",
            seed=11,
            fractions=FRACTIONS,
            stratify_by=["cellline", "treatment"],
        )
        per_condition = _categories_per(
            units, lambda u: (u.well_metadata["cellline"], u.well_metadata["treatment"])
        )
        for categories in per_condition.values():
            assert categories == {"training", "validation", "test"}

    def test_count_mode_selects_groups(self):
        units = assign_split_categories(_plate_units(), level="well", seed=2, counts=(3, 2, 1))
        wells = {u.well_id: u.category for u in units}
        assert len(wells) == 6
        assert sorted(wells.values()).count("training") == 3
        # every unit of a selected well is kept
        assert len(units) == 6 * 4 * 5

    def test_count_mode_too_few_groups(self):
        with pytest.raises(ValueError, match="Not enough wells"):
            assign_split_categories(_plate_units(), level="well", seed=2, counts=(10, 2, 1))

    def test_well_level_without_well_info_raises(self):
        units = [ImageAnnotation(image_id=1, image_name="a", annotation_id="1_0_0")]
        with pytest.raises(ValueError, match="well information"):
            assign_split_categories(units, level="well", seed=0, fractions=FRACTIONS)


@pytest.mark.unit
class TestLeakageReport:
    def test_report_per_level(self):
        units = assign_split_categories(_plate_units(), level="timepoint", seed=4, fractions=FRACTIONS)
        report = split_leakage_report(units, metadata_keys=["treatment"])
        assert report["timepoint"]["n_shared"] == 0
        assert report["image"]["n_shared"] > 0
        assert report["well"]["n_shared"] > 0
        assert report["treatment"]["n_groups"] == 2

    def test_well_level_skipped_without_well_info(self):
        units = [
            ImageAnnotation(image_id=1, image_name="a", annotation_id="1_0_0", category="training")
        ]
        assert "well" not in split_leakage_report(units)


def _mock_image(image_id, size_t=5):
    image = Mock()
    image.getId.return_value = image_id
    image.getName.return_value = f"img{image_id}"
    image.getSizeT.return_value = size_t
    image.getSizeZ.return_value = 1
    return image


@pytest.mark.unit
class TestPipelineSplits:
    def _pipeline(self, **training):
        config = create_default_config()
        config.training.segment_all = True
        config.training.train_fraction = 0.6
        config.training.validation_fraction = 0.2
        config.training.test_fraction = 0.2
        config.spatial_coverage.timepoint_mode = "all"
        for key, value in training.items():
            setattr(config.training, key, value)
        return AnnotationPipeline(config, conn=Mock())

    def _well_info(self, pipeline, n_wells=10, fields=2):
        images = []
        for w in range(n_wells):
            for f in range(fields):
                image_id = 100 + w * fields + f
                pipeline._well_info[image_id] = {
                    "well_id": w + 1,
                    "well": f"A{w + 1}",
                    "well_metadata": {"treatment": "DMSO" if w % 2 else "drug"},
                }
                images.append(_mock_image(image_id))
        return images

    def test_seed_is_generated_and_recorded(self):
        pipeline = self._pipeline()
        pipeline._prepare_processing_units([_mock_image(i) for i in range(10)])
        assert isinstance(pipeline.config.training.random_seed, int)

    def test_recorded_seed_reproduces_split(self):
        first = self._pipeline(random_seed=42)
        first._prepare_processing_units([_mock_image(i) for i in range(20)])
        second = self._pipeline(random_seed=42)
        second._prepare_processing_units([_mock_image(i) for i in range(20)])
        assert [a.category for a in first.config.annotations] == [
            a.category for a in second.config.annotations
        ]

    def test_fraction_mode_is_shuffled_not_in_listing_order(self):
        pipeline = self._pipeline(random_seed=0)
        pipeline._prepare_processing_units([_mock_image(i) for i in range(20)])
        by_image = {a.image_id: a.category for a in pipeline.config.annotations}
        in_order = [by_image[i] for i in range(20)]
        assert in_order != sorted(in_order, key=["training", "validation", "test"].index)

    def test_annotations_carry_well_context(self):
        pipeline = self._pipeline(random_seed=1)
        images = self._well_info(pipeline)
        pipeline._prepare_processing_units(images)
        annotation = pipeline.config.annotations[0]
        assert annotation.well_id is not None
        assert annotation.well.startswith("A")
        assert annotation.well_metadata["treatment"] in ("DMSO", "drug")

    def test_well_split_keeps_wells_together(self):
        pipeline = self._pipeline(random_seed=1, split_level="well", stratify_by=["treatment"])
        pipeline._prepare_processing_units(self._well_info(pipeline))
        report = pipeline.config.split_leakage_report()
        assert report["well"]["n_shared"] == 0
        assert report["image"]["n_shared"] == 0
        assert report["timepoint"]["n_shared"] == 0


@pytest.mark.unit
class TestTableRoundTrip:
    def test_test_category_and_well_context_survive_table(self):
        config = AnnotationConfig(name="roundtrip")
        config.annotations = [
            ImageAnnotation(
                image_id=1,
                image_name="a",
                annotation_id="1_0_0",
                category="test",
                well_id=7,
                well="C3",
                well_metadata={"treatment": "DMSO"},
            ),
            ImageAnnotation(image_id=2, image_name="b", annotation_id="2_0_0", category="validation"),
        ]
        restored = AnnotationConfig(name="restored")
        restored.from_dataframe(config.to_dataframe())
        first, second = restored.annotations
        assert first.category == "test"
        assert (first.well_id, first.well, first.well_metadata) == (7, "C3", {"treatment": "DMSO"})
        assert second.category == "validation"
        assert (second.well_id, second.well, second.well_metadata) == (None, None, {})

    def test_legacy_table_row_with_neither_flag_is_test(self):
        config = AnnotationConfig(name="legacy")
        config.annotations = [
            ImageAnnotation(image_id=1, image_name="a", annotation_id="1_0_0", category="test")
        ]
        df = config.to_dataframe().drop(columns=["category", "well_id", "well", "well_metadata"])
        restored = AnnotationConfig(name="restored")
        restored.from_dataframe(df)
        assert restored.annotations[0].category == "test"


@pytest.mark.unit
def test_notes_and_custom_metadata_round_trip_yaml():
    config = AnnotationConfig(
        name="notes",
        notes="Wells A1-B2 use the key 'celline' instead of 'cellline'.",
        custom_metadata={"incucyte_objective": "10x", "plate_barcode": "1406"},
    )
    restored = AnnotationConfig.from_dict(yaml.safe_load(config.to_yaml()))
    assert restored.notes == config.notes
    assert restored.custom_metadata == config.custom_metadata
