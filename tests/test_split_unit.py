"""Tests for the declared train/validation/test split unit (image or well).

The split unit is the level at which data are assigned to a split: every
annotation of one unit lands in the same split. It is recorded in the config,
enforced by the pipeline, checked by validation and reported at export.
"""

import pytest
from pydantic import ValidationError

from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    TrainingConfig,
    create_default_config,
)
from omero_annotate_ai.rows import Row, split_report, split_summary
from omero_annotate_ai.sources import ImageMeta
from omero_annotate_ai.sources.uri import omero_uri
from omero_annotate_ai.trainingset import TrainingSet

pytestmark = pytest.mark.unit


class FakeImage:
    def __init__(self, image_id):
        self.image_id = image_id
        self.name = f"img{image_id}"


class FakeImageSource:
    """Single-plane images, one per id."""

    def __init__(self, image_ids):
        self.image_ids = image_ids

    def get_image(self, image_id):
        return FakeImage(image_id)

    def metadata(self, image):
        return ImageMeta(
            omero_uri(image.image_id), image.name, 1, 1, 1, 16, 16, "uint16"
        )


def _plate_pipeline(tmp_path, wells, **training):
    """Pipeline over a fake plate; ``wells`` maps well name -> list of image ids."""
    config = create_default_config()
    config.storage = "local"
    for key, value in training.items():
        setattr(config.training, key, value)
    ids = [i for ids in wells.values() for i in ids]
    source = FakeImageSource(ids)
    pipeline = TrainingSet(config, None, image_source=source, home=tmp_path)
    for well, well_ids in wells.items():
        for field, image_id in enumerate(well_ids):
            pipeline._omero_wells[omero_uri(image_id)] = {
                "well": well,
                "field": str(field),
                "well_metadata": {"treatment": f"t_{well}"},
            }
    return pipeline, [source.get_image(i) for i in ids]


def _ann(image_id, well, category, t=0):
    return Row(
        image_uri=omero_uri(image_id),
        image_id=image_id,
        image_name=f"img{image_id}",
        annotation_id=f"a{image_id}_{t}",
        timepoint=t,
        well=well,
        category=category,
    )


def _plan(training_set, images):
    """Plan a new set's rows from ``images`` (what TrainingSet.create does)."""
    by_uri = {training_set.image_source.metadata(i).image_uri: i for i in images}
    plan = training_set._plan_split(list(by_uri))
    training_set.rows = [
        row
        for uri, category in plan.items()
        for row in training_set._units_for_image(by_uri[uri], category)
    ]


def _report(rows, training):
    return split_report(
        rows, training.split_unit, training.stratify_by, training.split_rationale
    )


class TestSplitUnitConfig:
    def test_default_is_image(self):
        assert TrainingConfig().split_unit == "image"
        assert TrainingConfig().split_rationale is None

    def test_well_is_accepted(self):
        training = TrainingConfig(
            split_unit="well", split_rationale="wells independent"
        )
        assert training.split_unit == "well"

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(split_unit="pixel")

    def test_survives_yaml_round_trip(self):
        config = create_default_config()
        config.training.split_unit = "well"
        config.training.split_rationale = "fields overlap"
        loaded = AnnotationConfig.from_yaml(config.to_yaml())
        assert loaded.training.split_unit == "well"
        assert loaded.training.split_rationale == "fields overlap"


class TestWellSplit:
    WELLS = {"B2": [1, 2], "B3": [3, 4], "B4": [5, 6], "B5": [7, 8]}

    def test_count_mode_selects_whole_wells(self, tmp_path):
        pipeline, images = _plate_pipeline(
            tmp_path, self.WELLS, split_unit="well", train_n=2, validate_n=1, test_n=1
        )
        _plan(pipeline, images)

        anns = pipeline.rows
        assert len(anns) == 8
        by_well = {}
        for a in anns:
            by_well.setdefault(a.well, set()).add(a.category)
        assert all(len(cats) == 1 for cats in by_well.values())
        cats = sorted(next(iter(c)) for c in by_well.values())
        assert cats == ["test", "training", "training", "validation"]

    def test_count_mode_takes_subset_of_wells(self, tmp_path):
        pipeline, images = _plate_pipeline(
            tmp_path, self.WELLS, split_unit="well", train_n=1, validate_n=1, test_n=0
        )
        _plan(pipeline, images)

        wells = {a.well for a in pipeline.rows}
        assert len(wells) == 2
        assert len(pipeline.rows) == 4

    def test_fraction_mode_splits_wells(self, tmp_path):
        pipeline, images = _plate_pipeline(
            tmp_path,
            self.WELLS,
            split_unit="well",
            segment_all=True,
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
        )
        _plan(pipeline, images)

        report = _report(pipeline.rows, pipeline.config.training)
        assert report["wells"]["spanning_splits"] == []
        counts = report["wells"]["per_split"]
        assert counts == {"training": 2, "validation": 1, "test": 1}

    def test_too_many_wells_requested(self, tmp_path):
        pipeline, images = _plate_pipeline(
            tmp_path, self.WELLS, split_unit="well", train_n=4, validate_n=1
        )
        with pytest.raises(ValueError, match="only 4 wells available"):
            _plan(pipeline, images)

    def test_well_unit_needs_well_information(self, tmp_path):
        pipeline, images = _plate_pipeline(
            tmp_path, self.WELLS, split_unit="well", train_n=1, validate_n=1
        )
        pipeline._omero_wells.clear()
        with pytest.raises(ValueError, match="split_unit='well'"):
            _plan(pipeline, images)

    def test_plan_split_returns_every_image_of_selected_wells(self, tmp_path):
        pipeline, images = _plate_pipeline(
            tmp_path, self.WELLS, split_unit="well", train_n=1, validate_n=1
        )
        uris = [omero_uri(i) for ids in self.WELLS.values() for i in ids]
        plan = pipeline._plan_split(uris)
        assert len(plan) == 4
        wells = {pipeline._omero_wells[uri]["well"] for uri in plan}
        assert len(wells) == 2

    def test_image_unit_keeps_per_image_split(self, tmp_path):
        pipeline, images = _plate_pipeline(
            tmp_path, self.WELLS, train_n=3, validate_n=2, test_n=1
        )
        _plan(pipeline, images)

        cats = [a.category for a in pipeline.rows]
        assert (
            cats.count("training"),
            cats.count("validation"),
            cats.count("test"),
        ) == (3, 2, 1)


class TestSplitReport:
    def _config(self, anns, unit="image", rationale=None):
        config = create_default_config()
        config.training.split_unit = unit
        config.training.split_rationale = rationale
        self.rows = list(anns)
        return config

    def test_reports_wells_spanning_splits(self):
        config = self._config(
            [
                _ann(1, "B2", "training"),
                _ann(2, "B2", "validation"),
                _ann(3, "B3", "test"),
                _ann(4, "B3", "test"),
            ]
        )
        report = _report(self.rows, config.training)
        assert report["split_unit"] == "image"
        assert report["images"] == {
            "total": 4,
            "per_split": {"training": 1, "validation": 1, "test": 2},
            "spanning_splits": [],
        }
        assert report["wells"]["total"] == 2
        assert report["wells"]["spanning_splits"] == ["B2"]

    def test_image_spanning_splits_is_detected(self):
        config = self._config(
            [
                _ann(1, "B2", "training", t=0),
                _ann(1, "B2", "validation", t=1),
            ]
        )
        assert _report(self.rows, config.training)["images"]["spanning_splits"] == [
            "omero:image:1"
        ]

    def test_no_wells_section_without_well_information(self):
        config = self._config([_ann(1, "", "training")])
        assert "wells" not in _report(self.rows, config.training)

    def test_summary_sentence(self):
        config = self._config(
            [
                _ann(1, "B2", "training"),
                _ann(2, "B2", "validation"),
                _ann(3, "B3", "test"),
            ],
            rationale="Fields do not overlap.",
        )
        summary = split_summary(_report(self.rows, config.training))
        assert "Split unit: image" in summary
        assert "1 of 2 wells" in summary
        assert "B2" in summary
        assert "Fields do not overlap." in summary


def test_mifa_method_text_states_split_unit():
    from omero_annotate_ai.core.mifa_export import compose_method

    config = create_default_config()
    config.training.split_unit = "well"
    config.training.split_rationale = "Wells are the experimental unit."
    rows = [_ann(1, "B2", "training"), _ann(2, "B3", "test")]
    method = compose_method(config, rows)
    assert "Split unit: well" in method
    assert "Wells are the experimental unit." in method


def _condition_pipeline(tmp_path, n_fields=4, **training):
    """2 cell lines x 2 treatments, one well each, ``n_fields`` fields per well."""
    config = create_default_config()
    config.storage = "local"
    for key, value in training.items():
        setattr(config.training, key, value)
    pipeline = TrainingSet(
        config, None, image_source=FakeImageSource([]), home=tmp_path
    )
    images, image_id = [], 0
    for well, (line, drug) in zip(
        ["B2", "B3", "C2", "C3"],
        [("MDA", "DMSO"), ("MDA", "drug"), ("HCC", "DMSO"), ("HCC", "drug")],
        strict=True,
    ):
        for field in range(n_fields):
            image_id += 1
            pipeline._omero_wells[omero_uri(image_id)] = {
                "well": well,
                "field": str(field),
                "well_metadata": {"cell_line": line, "treatment": drug},
            }
            images.append(FakeImage(image_id))
    return pipeline, images


def _categories_per_well(pipeline):
    per_well = {}
    for a in pipeline.rows:
        per_well.setdefault(a.well, []).append(a.category)
    return per_well


class TestFractionShuffle:
    def test_fraction_mode_is_not_in_list_order(self, tmp_path):
        pipeline, images = _condition_pipeline(
            tmp_path,
            segment_all=True,
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
        )
        orders = set()
        for seed in range(5):  # the order is random per seed, fixed for one seed
            pipeline._config.training.seed = seed
            _plan(pipeline, images)
            orders.add(
                tuple(
                    a.category for a in sorted(pipeline.rows, key=lambda a: a.image_id)
                )
            )
        assert len(orders) > 1


class TestStratifiedSplit:
    def test_fraction_mode_covers_every_condition(self, tmp_path):
        pipeline, images = _condition_pipeline(
            tmp_path,
            segment_all=True,
            stratify_by=["cell_line", "treatment"],
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        _plan(pipeline, images)

        for cats in _categories_per_well(pipeline).values():
            assert sorted(cats) == ["test", "training", "training", "validation"]

    def test_count_mode_spreads_over_conditions(self, tmp_path):
        pipeline, images = _condition_pipeline(
            tmp_path,
            stratify_by=["cell_line", "treatment"],
            train_n=8,
            validate_n=4,
            test_n=4,
        )
        _plan(pipeline, images)

        for cats in _categories_per_well(pipeline).values():
            assert cats.count("validation") == 1
            assert cats.count("test") == 1
            assert cats.count("training") == 2

    def test_well_can_be_a_stratum(self, tmp_path):
        pipeline, images = _condition_pipeline(
            tmp_path,
            segment_all=True,
            stratify_by=["well"],
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
        )
        _plan(pipeline, images)
        for cats in _categories_per_well(pipeline).values():
            assert set(cats) == {"training", "validation", "test"}

    def test_impossible_coverage_warns_but_splits(self, tmp_path):
        pipeline, images = _condition_pipeline(
            tmp_path,
            n_fields=1,
            split_unit="well",
            segment_all=True,
            stratify_by=["cell_line", "treatment"],
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        with pytest.warns(UserWarning, match="not in every split"):
            _plan(pipeline, images)
        assert len(pipeline.rows) == 4

    def test_unknown_key_is_an_error(self, tmp_path):
        pipeline, images = _condition_pipeline(
            tmp_path,
            segment_all=True,
            stratify_by=["celline"],
        )
        with pytest.raises(ValueError, match="celline"):
            _plan(pipeline, images)

    def test_key_missing_on_some_images_is_its_own_stratum(self, tmp_path):
        pipeline, images = _condition_pipeline(
            tmp_path,
            segment_all=True,
            stratify_by=["treatment"],
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
        )
        for well in pipeline._omero_wells.values():
            if well["well"] == "B2":
                well["well_metadata"] = {}
        _plan(pipeline, images)
        assert len(pipeline.rows) == 16

    def test_report_lists_strata_coverage(self, tmp_path):
        pipeline, images = _condition_pipeline(
            tmp_path,
            segment_all=True,
            stratify_by=["cell_line"],
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
        )
        _plan(pipeline, images)
        report = _report(pipeline.rows, pipeline.config.training)
        assert report["stratify_by"] == ["cell_line"]
        assert set(report["strata"]) == {"cell_line=MDA", "cell_line=HCC"}
        assert report["strata_missing_splits"] == {}
        assert "every stratum" in split_summary(
            _report(pipeline.rows, pipeline.config.training)
        )
