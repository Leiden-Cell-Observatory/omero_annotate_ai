"""Tests for training data preparation functions."""

import json
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    create_default_config,
)
from omero_annotate_ai.errors import InvalidTrainingData
from omero_annotate_ai.processing import training_functions as tf
from omero_annotate_ai.processing.normalization import (
    SIDECAR_NAME,
    write_native_sidecar,
)
from omero_annotate_ai.processing.trainers import load_training_data, train
from omero_annotate_ai.processing.training_functions import (
    reorganize_local_data_for_training,
)
from omero_annotate_ai.rows import Row, rows_from_frame, rows_to_frame


@pytest.mark.unit
class TestExternalClassificationWorkflow:
    """Tests for the external-label workflow.

    Scenario: user runs CellPose + intensity thresholding in a separate notebook,
    uploads class label maps (integer masks: 0=bg, 1/2/3=class) as FileAnnotations
    to OMERO, and stores the FileAnnotation IDs in the tracking table's label_id column.
    omero_annotate_ai then exports Ch0 as training input and the class label map as
    training ground truth.
    """

    def test_label_id_as_string_none_is_skipped(self):
        """label_id stored as the string 'None' (from an OMERO table) must not crash."""
        from omero_annotate_ai.processing.training_functions import _optional_int

        assert _optional_int("None") is None
        assert _optional_int("nan") is None
        assert _optional_int("") is None
        assert _optional_int(None) is None

    def test_label_id_as_string_integer_is_used(self):
        """label_id stored as the string '101' must parse to int 101."""
        from omero_annotate_ai.processing.training_functions import _optional_int

        assert _optional_int("101") == 101
        assert _optional_int(101.0) == 101
        assert _optional_int(101) == 101

    def test_no_label_id_means_no_download_and_no_record(self, tmp_path):
        """A row with no label is dropped, not written as an orphan image."""
        from omero_annotate_ai.processing.training_functions import _download_label

        assert _download_label(Mock(), None, tmp_path) is None

    def test_multiclass_label_pixel_values_preserved(self, tmp_path):
        """Integer pixel values in a multi-class label TIFF are not remapped.

        The label is copied verbatim; only the image plane is normalized to 8-bit.
        """
        from tifffile import imread, imwrite

        from omero_annotate_ai.processing.training_layout import FileSource

        label = np.array([[0, 1], [2, 7]], dtype=np.uint8)
        src = tmp_path / "src.tif"
        imwrite(str(src), label)
        dst = tmp_path / "dst.tif"

        FileSource(src).write_to(dst)

        np.testing.assert_array_equal(imread(str(dst)), label)

    def test_classification_annotation_type_is_valid(self):
        """annotation_type='classification' and 'semantic_segmentation' are valid values."""
        from omero_annotate_ai.core.annotation_config import AnnotationMethodology

        m = AnnotationMethodology(
            annotation_type="classification",
            annotation_criteria="3-class cell type classification",
        )
        assert m.annotation_type == "classification"

        m2 = AnnotationMethodology(
            annotation_type="semantic_segmentation",
            annotation_criteria="semantic segmentation",
        )
        assert m2.annotation_type == "semantic_segmentation"

    def test_external_workflow_config_yaml_roundtrip(self):
        """Config describing the external classification workflow serializes to valid YAML."""
        import yaml

        config = AnnotationConfig(name="cell_classification_workflow")
        config.spatial_coverage.channels = [0, 1, 2]
        config.spatial_coverage.label_channel = 0
        config.spatial_coverage.training_channels = [0]
        config.annotation_methodology.annotation_type = "classification"
        config.annotation_methodology.annotation_method = "automatic"
        config.annotation_methodology.annotation_criteria = (
            "CellPose segmentation + intensity thresholding on Ch1/Ch2 into 3 classes"
        )

        yaml_str = config.to_yaml()
        config2 = AnnotationConfig.from_dict(yaml.safe_load(yaml_str))

        assert config2.spatial_coverage.channels == [0, 1, 2]
        assert config2.spatial_coverage.label_channel == 0
        assert config2.spatial_coverage.training_channels == [0]
        assert config2.annotation_methodology.annotation_type == "classification"
        assert config2.annotation_methodology.annotation_method == "automatic"


def _write_rows(folder, rows):
    """A training set's tracking_table.csv, as reorganize reads it."""
    folder.mkdir(parents=True, exist_ok=True)
    rows_to_frame(rows, local=True).to_csv(folder / "tracking_table.csv", index=False)


@pytest.mark.unit
class TestReorganizeOntoRecords:
    """The offline producer emits the unified layout, paired by annotation_id."""

    def _config(self, tmp_path, categories, separate_channels=False):
        config = create_default_config()
        if separate_channels:
            config.spatial_coverage.channels = [0, 1]
            config.spatial_coverage.label_channel = 0
            config.spatial_coverage.training_channels = [1]
        else:
            config.spatial_coverage.channels = [0]
            config.spatial_coverage.label_channel = None
            config.spatial_coverage.training_channels = None
        rows = [
            Row(
                image_id=100 + i,
                image_name=f"img_{i}",
                annotation_id=str(i),
                category=category,
                processed=True,
            )
            for i, category in enumerate(categories)
        ]
        _write_rows(tmp_path, rows)
        return config

    def _populate(self, annotation_dir, ids, separate_channels=False):
        (annotation_dir / "annotation_input").mkdir(parents=True, exist_ok=True)
        (annotation_dir / "annotation_output").mkdir(parents=True, exist_ok=True)
        if separate_channels:
            (annotation_dir / "model_input").mkdir(parents=True, exist_ok=True)
        for i in ids:
            (annotation_dir / "annotation_input" / f"{i}.tif").write_bytes(b"ann")
            (annotation_dir / "annotation_output" / f"{i}_mask.tif").write_bytes(b"lbl")
            if separate_channels:
                (annotation_dir / "model_input" / f"{i}.tif").write_bytes(b"model")

    def test_excluded_units_are_left_out_silently(self, tmp_path, caplog):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training", "training"])
        rows = rows_from_frame(pd.read_csv(annotation_dir / "tracking_table.csv"))
        rows[1].mark_excluded(session=1)  # no files in the folder
        _write_rows(annotation_dir, rows)

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert [p.name for p in result["train_input"].glob("*.tif")] == ["0.tif"]
        assert result["stats"]["n_missing"] == 0
        assert "Skipping annotation 1" not in caplog.text

    def test_writes_unified_layout(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config(annotation_dir, ["training", "validation"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert (result["train_input"] / "0.tif").read_bytes() == b"ann"
        assert (result["train_label"] / "0.tif").read_bytes() == b"lbl"
        assert "validation_input" not in result

    def test_image_and_label_pair_by_id(self, tmp_path):
        """Destination files used to be named by a per-category counter."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config(annotation_dir, ["training", "training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        labels = sorted(p.name for p in result["train_label"].glob("*.tif"))
        assert images == labels == ["0.tif", "1.tif"]

    def test_separate_channel_stats_count_annotation_input(self, tmp_path):
        """The *_annotation_input counters were asserted by the old suite and by nothing since."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1], separate_channels=True)
        config = self._config(
            annotation_dir, ["training", "validation"], separate_channels=True
        )

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        stats = result["stats"]
        assert stats["n_training_annotation_input"] == 1
        assert stats["n_val_annotation_input"] == 1

    def test_separate_channels_route_model_and_annotation(self, tmp_path):
        """model_input/ feeds train_input/; annotation_input/ feeds train_annotation_input/."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0], separate_channels=True)
        config = self._config(annotation_dir, ["training"], separate_channels=True)

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert (result["train_input"] / "0.tif").read_bytes() == b"model"
        assert (result["train_annotation_input"] / "0.tif").read_bytes() == b"ann"

    def test_rejects_output_inside_annotation_dir(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        with pytest.raises(ValueError, match="must not be inside"):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=annotation_dir,
                output_dir=annotation_dir,
            )

    def test_defaults_to_sibling_training_dir(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        result = reorganize_local_data_for_training(
            config=config, annotation_dir=annotation_dir
        )

        assert result["base_dir"] == tmp_path / "project_training"

    def test_symlink_mode(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
            file_mode="symlink",
        )

        assert (result["train_input"] / "0.tif").is_symlink()

    def test_missing_label_drops_the_record(self, tmp_path):
        """No orphan image without its label."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        (annotation_dir / "annotation_output" / "0_mask.tif").unlink()
        config = self._config(annotation_dir, ["training", "training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        labels = sorted(p.name for p in result["train_label"].glob("*.tif"))
        assert images == labels == ["1.tif"]

    def test_copy_mode_preserves_originals(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
            file_mode="copy",
        )

        assert (annotation_dir / "annotation_input" / "0.tif").exists()

    def test_move_mode_removes_originals(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
            file_mode="move",
        )

        assert not (annotation_dir / "annotation_input" / "0.tif").exists()

    def test_stats_count_images_and_labels(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1, 2])
        config = self._config(annotation_dir, ["training", "training", "validation"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        stats = result["stats"]
        assert stats["n_training_images"] == 2
        assert stats["n_training_labels"] == 2
        assert stats["n_val_images"] == 1
        assert stats["n_val_labels"] == 1
        assert stats["n_missing"] == 0

    def test_missing_image_drops_the_record(self, tmp_path):
        """An orphan label is as bad as an orphan image: drop the pair."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        (annotation_dir / "annotation_input" / "0.tif").unlink()
        config = self._config(annotation_dir, ["training", "training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        labels = sorted(p.name for p in result["train_label"].glob("*.tif"))
        assert images == labels == ["1.tif"]
        assert result["stats"]["n_missing"] == 1

    def test_test_category_auto_detected(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1, 2])
        config = self._config(annotation_dir, ["training", "validation", "test"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert "test_input" in result
        assert result["stats"]["n_test_images"] == 1

    def test_no_test_annotations_means_no_test_folders(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config(annotation_dir, ["training", "validation"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert "test_input" not in result
        assert result["stats"]["n_test_images"] == 0

    def test_include_test_false_skips_test_annotations(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config(annotation_dir, ["training", "test"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
            include_test=False,
        )

        assert "test_input" not in result
        assert result["stats"]["n_skipped"] == 1

    def test_clean_existing_removes_stale_training_data(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])
        out = tmp_path / "project_training"
        stale = out / "train_input" / "999.tif"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=out,
            clean_existing=True,
        )

        assert not stale.exists()
        assert (out / "train_input" / "0.tif").exists()

    def test_file_mapping_records_destinations(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        mapping = result["stats"]["file_mapping"]["0"]
        # Compare as Paths: file_mapping stores str(Path), whose separator is
        # backslash on Windows.
        assert Path(mapping["image"]) == result["train_input"] / "0.tif"
        assert Path(mapping["label"]) == result["train_label"] / "0.tif"

    def test_no_processed_annotations_raises(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])
        _write_rows(
            annotation_dir,
            [Row(image_id=100, image_name="img_0", annotation_id="0")],
        )

        with pytest.raises(ValueError, match="No annotated units"):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=annotation_dir,
                output_dir=tmp_path / "project_training",
            )

    def test_missing_annotation_dir_raises(self, tmp_path):
        config = self._config(tmp_path / "nope", ["training"])

        with pytest.raises(FileNotFoundError):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=tmp_path / "nope",
                output_dir=tmp_path / "out",
            )


@pytest.mark.unit
class TestPrepareFromTableRecords:
    """The OMERO producer pairs images and labels by annotation_id."""

    def _table(self):
        rows = []
        for i in (0, 1):
            rows.append(
                {
                    "image_id": 100 + i,
                    "annotation_id": str(i),
                    "category": "training",
                    "channel": 0,
                    "z_slice": 0,
                    "timepoint": 0,
                    "label_id": 900 + i,
                    "is_volumetric": False,
                    "is_patch": False,
                    "patch_x": 0,
                    "patch_y": 0,
                    "patch_width": 0,
                    "patch_height": 0,
                    "processed": True,
                }
            )
        return pd.DataFrame(rows)

    def _run(self, tmp_path, table, missing_label_ids=()):
        from tifffile import imwrite as _imwrite

        from omero_annotate_ai.processing import training_functions as tf

        plane = np.ones((4, 4), dtype=np.uint8)

        def fake_download(conn, label_id, tmp_dir, logger=None):
            if label_id in missing_label_ids:
                return None
            path = Path(tmp_dir) / f"{label_id}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            _imwrite(str(path), plane)
            return path

        with (
            patch.object(tf, "read_unit", return_value=plane),
            patch.object(tf, "_download_label", side_effect=fake_download),
            patch.object(tf, "_load_table", return_value=table),
            patch.object(tf.ezomero, "get_table", return_value=table),
        ):
            return tf.prepare_training_data_from_table(
                conn=Mock(),
                table_id=1,
                output_dir=tmp_path / "training",
            )

    def test_a_missing_label_does_not_shift_later_pairs(self, tmp_path):
        """The regression this refactor exists to kill.

        Files used to be named by loop index, and the image was written before the
        label download could fail. One missing label left an orphan image; consumers
        pair raw_paths to label_paths by sorted filename, so every subsequent pair
        silently shifted by one.
        """
        result = self._run(tmp_path, self._table(), missing_label_ids=(900,))

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        labels = sorted(p.name for p in result["train_label"].glob("*.tif"))

        # Annotation 0 lost its label, so it is dropped whole - not left as an orphan.
        assert images == labels == ["1.tif"]
        assert result["stats"]["n_missing"] == 1

    def test_excluded_rows_are_not_exported(self, tmp_path):
        table = self._table()
        table["annotation_type"] = ["segmentation_mask", "excluded"]
        table.loc[1, "label_id"] = 0

        result = self._run(tmp_path, table)

        assert [p.name for p in result["train_input"].glob("*.tif")] == ["0.tif"]
        assert result["stats"]["n_missing"] == 0

    def test_files_named_by_annotation_id(self, tmp_path):
        result = self._run(tmp_path, self._table())

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        assert images == ["0.tif", "1.tif"]

    def test_test_rows_get_their_own_folders(self, tmp_path):
        table = self._table()
        table.loc[1, "category"] = "test"

        result = self._run(tmp_path, table)

        assert (result["train_input"] / "0.tif").exists()
        assert (result["test_input"] / "1.tif").exists()
        assert (result["test_label"] / "1.tif").exists()

    def test_unknown_category_is_rejected(self, tmp_path):
        table = self._table()
        table.loc[1, "category"] = "holdout"

        with pytest.raises(ValueError, match="Unknown category"):
            self._run(tmp_path, table)

    def test_label_that_no_longer_matches_its_digest_stops_the_export(self, tmp_path):
        from omero_annotate_ai.bundle import pixel_digest
        from omero_annotate_ai.errors import IntegrityError

        table = self._table()
        table["mask_digest"] = pixel_digest(np.ones((4, 4), dtype=np.uint8))
        table.loc[1, "mask_digest"] = pixel_digest(np.zeros((4, 4), dtype=np.uint8))

        with pytest.raises(IntegrityError, match="1"):
            self._run(tmp_path, table)

    def test_matching_digests_export_normally(self, tmp_path):
        from omero_annotate_ai.bundle import pixel_digest

        table = self._table()
        table["mask_digest"] = pixel_digest(np.ones((4, 4), dtype=np.uint8))

        result = self._run(tmp_path, table)

        assert sorted(p.name for p in result["train_label"].glob("*.tif")) == [
            "0.tif",
            "1.tif",
        ]


@pytest.mark.unit
class TestEmptyTrainingSetIsLoud:
    """An empty training set must raise, not return valid-looking empty folders."""

    def test_reorganize_raises_when_every_record_is_dropped(self, tmp_path):
        """A producer/consumer folder mismatch used to return silently.

        reorganize returned base_dir/train_input/... pointing at empty directories,
        stats all zero, and no exception - so training ran on nothing.
        """
        annotation_dir = tmp_path / "project"
        (annotation_dir / "annotation_input").mkdir(parents=True)
        (annotation_dir / "annotation_output").mkdir(parents=True)

        config = create_default_config()
        config.spatial_coverage.label_channel = None
        config.spatial_coverage.training_channels = None
        _write_rows(
            annotation_dir,
            [
                Row(
                    image_id=100,
                    image_name="img",
                    annotation_id="0",
                    category="training",
                    processed=True,
                )
            ],
        )
        # No files on disk: every record will be dropped as missing.

        with pytest.raises(ValueError, match="no training data"):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=annotation_dir,
                output_dir=tmp_path / "out",
            )

    def test_raises_when_records_exist_but_nothing_is_written(self, tmp_path):
        """Records can be built and then written nowhere.

        Every annotation is category "test" and include_test=False, so the records are
        valid but the layout gets nothing. Keying the guard off len(records) would miss
        this and return empty folders silently.
        """
        annotation_dir = tmp_path / "project"
        (annotation_dir / "annotation_input").mkdir(parents=True)
        (annotation_dir / "annotation_output").mkdir(parents=True)
        (annotation_dir / "annotation_input" / "0.tif").write_bytes(b"img")
        (annotation_dir / "annotation_output" / "0_mask.tif").write_bytes(b"lbl")

        config = create_default_config()
        config.spatial_coverage.label_channel = None
        config.spatial_coverage.training_channels = None
        _write_rows(
            annotation_dir,
            [
                Row(
                    image_id=100,
                    image_name="img",
                    annotation_id="0",
                    category="test",
                    processed=True,
                )
            ],
        )

        with pytest.raises(ValueError, match="no training data"):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=annotation_dir,
                output_dir=tmp_path / "out",
                include_test=False,
            )


@pytest.mark.unit
class TestChannelParsing:
    """The channel column must not fail soft."""

    def test_missing_channel_defaults_to_zero(self):
        from omero_annotate_ai.processing.training_functions import _channel_from_row

        assert _channel_from_row(pd.Series({"channel": None})) == 0
        assert _channel_from_row(pd.Series({"channel": "None"})) == 0
        assert _channel_from_row(pd.Series({})) == 0

    def test_valid_channel_is_parsed(self):
        from omero_annotate_ai.processing.training_functions import _channel_from_row

        assert _channel_from_row(pd.Series({"channel": 2})) == 2
        assert _channel_from_row(pd.Series({"channel": "2"})) == 2

    def test_unparseable_channel_raises(self):
        """Silently fetching channel 0 would train on the wrong channel."""
        from omero_annotate_ai.processing.training_functions import _channel_from_row

        with pytest.raises(ValueError, match="Unparseable channel"):
            _channel_from_row(pd.Series({"channel": "not-a-channel"}))


@pytest.mark.unit
class TestExportTrainingSet:
    """One entry point: table ID -> OMERO export, folder -> local reorganise."""

    def test_table_id_uses_the_set_config_and_overrides(self, tmp_path):
        from omero_annotate_ai.core.annotation_config import create_default_config

        config = create_default_config()
        config.name = "nuclei"
        config.spatial_coverage.channels = [2]
        with (
            patch("omero_annotate_ai.trainingset.read_set_config", return_value=config),
            patch.object(tf, "prepare_training_data_from_table") as prepare,
        ):
            tf.export_training_set(
                7, tmp_path, "conn", training_channels=[3], apply_normalization=True
            )
        args, kwargs = prepare.call_args
        assert args == ("conn", 7, tmp_path)
        assert kwargs["training_name"] == "nuclei"
        assert kwargs["label_channel"] == 2
        assert kwargs["training_channels"] == [3]
        assert kwargs["normalization"].apply_at_export is True
        assert config.normalization.apply_at_export is False  # caller's copy untouched

    def test_local_set_refuses_other_channels(self, tmp_path):
        from omero_annotate_ai.core.annotation_config import create_default_config

        with (
            patch(
                "omero_annotate_ai.trainingset.read_set_config",
                return_value=create_default_config(),
            ),
            pytest.raises(ValueError, match="exports the channels it was annotated"),
        ):
            tf.export_training_set(tmp_path / "set", tmp_path / "out", label_channel=3)

    def test_omero_export_refuses_a_folder_inside_the_sets_folder(self, tmp_path):
        from omero_annotate_ai.core.annotation_config import create_default_config

        set_folder = tmp_path / "omero.example.org" / "7_nuclei"
        set_folder.mkdir(parents=True)
        with (
            patch(
                "omero_annotate_ai.trainingset.read_set_config",
                return_value=create_default_config(),
            ),
            patch.object(tf, "prepare_training_data_from_table") as prepare,
        ):
            with pytest.raises(ValueError):
                tf.export_training_set(7, set_folder / "out", "conn", home=tmp_path)
            tf.export_training_set(7, tmp_path / "exports" / "x", "conn", home=tmp_path)
        assert prepare.call_count == 1

    def test_omero_export_refuses_several_training_channels(self, tmp_path):
        from omero_annotate_ai.core.annotation_config import create_default_config

        with (
            patch(
                "omero_annotate_ai.trainingset.read_set_config",
                return_value=create_default_config(),
            ),
            pytest.raises(ValueError, match="trains on one channel"),
        ):
            tf.export_training_set(7, tmp_path, "conn", training_channels=[0, 1])

    def test_local_set_accepts_its_own_channels(self, tmp_path):
        from omero_annotate_ai.core.annotation_config import create_default_config

        config = create_default_config()
        with (
            patch("omero_annotate_ai.trainingset.read_set_config", return_value=config),
            patch.object(tf, "reorganize_local_data_for_training") as reorganize,
        ):
            tf.export_training_set(
                tmp_path / "set",
                tmp_path / "out",
                training_channels=config.spatial_coverage.get_training_channels(),
                label_channel=config.spatial_coverage.get_label_channel(),
            )
        assert reorganize.called


@pytest.mark.unit
class TestDefaultExportDir:
    def test_one_folder_per_set_and_safe_names(self, tmp_path):
        omero_a = tf.default_export_dir(7, "Nuclei/HeLa v1", tmp_path)
        omero_b = tf.default_export_dir(8, "Nuclei/HeLa v1", tmp_path)
        local_a = tf.default_export_dir(
            tmp_path / "local" / "ab12_nuclei", "nuclei", tmp_path
        )
        local_b = tf.default_export_dir(
            tmp_path / "local" / "cd34_nuclei", "nuclei", tmp_path
        )

        assert omero_a == tmp_path / "exports" / "nuclei-hela-v1_7"
        assert len({omero_a, omero_b, local_a, local_b}) == 4
        assert local_a.name == "ab12_nuclei"
        assert all(p.parent == tmp_path / "exports" for p in (omero_a, local_b))


def _export(
    tmp_path: Path,
    *,
    shape: tuple[int, ...] = (32, 32),
    n_train: int = 2,
    n_val: int = 1,
    test: bool = False,
    sidecar: bool = True,
    dtype: type = np.uint16,
    applied_at_export: bool = False,
) -> Path:
    """Write a training export (image/label pairs named <id>.tif) and return its root."""
    root = tmp_path / "export"
    label_shape = shape[:-1] if len(shape) >= 3 and shape[-1] <= 4 else shape
    splits = {"train": n_train, "val": n_val, "test": 1 if test else 0}
    next_id = 0
    for split, count in splits.items():
        if split == "test" and not test:
            continue
        (root / f"{split}_input").mkdir(parents=True, exist_ok=True)
        (root / f"{split}_label").mkdir(parents=True, exist_ok=True)
        for _ in range(count):
            label = np.zeros(label_shape, np.uint16)
            label[..., :4, :4] = 1
            label[..., 8:12, 8:12] = 2
            image = np.ones(shape, np.float32 if applied_at_export else dtype)
            if applied_at_export:
                image *= 0.5
            tifffile.imwrite(root / f"{split}_input" / f"{next_id}.tif", image)
            tifffile.imwrite(root / f"{split}_label" / f"{next_id}.tif", label)
            next_id += 1
    if sidecar:
        norm = create_default_config().normalization
        if applied_at_export:
            (root / SIDECAR_NAME).write_text(
                json.dumps({"applied_at_export": True, "method": "percentile"})
            )
        else:
            write_native_sidecar(norm, root)
    return root


@pytest.mark.unit
class TestLoadTrainingData:
    def test_reads_layout_axes_dtype_and_sidecar(self, tmp_path):
        data = load_training_data(_export(tmp_path))
        assert data.axes == "YX" and data.n_channels == 1 and data.dtype == "uint16"
        assert data.normalization["applied_at_export"] is False
        assert len(data.pairs("train")) == 2 and len(data.pairs("val")) == 1
        assert data.test_input is None

    def test_axes_for_channels_and_volumes(self, tmp_path):
        assert (
            load_training_data(_export(tmp_path / "c", shape=(32, 32, 3))).axes == "YXC"
        )
        assert (
            load_training_data(_export(tmp_path / "z", shape=(8, 32, 32))).axes == "ZYX"
        )
        assert (
            load_training_data(
                _export(tmp_path / "zc", shape=(8, 32, 32, 2))
            ).n_channels
            == 2
        )

    def test_missing_sidecar_is_empty_dict(self, tmp_path):
        assert load_training_data(_export(tmp_path, sidecar=False)).normalization == {}

    def test_missing_val_pair_is_refused(self, tmp_path):
        with pytest.raises(InvalidTrainingData, match="val_input"):
            load_training_data(_export(tmp_path, n_val=0))

    def test_unpaired_file_is_refused(self, tmp_path):
        root = _export(tmp_path)
        (root / "train_label" / "extra.tif").write_bytes(b"")
        with pytest.raises(InvalidTrainingData, match="extra.tif"):
            load_training_data(root)

    def test_half_test_split_is_refused(self, tmp_path):
        root = _export(tmp_path, test=True)
        shutil.rmtree(root / "test_label")
        with pytest.raises(InvalidTrainingData, match="test_label"):
            load_training_data(root)

    def test_shape_mismatch_is_refused(self, tmp_path):
        root = _export(tmp_path)
        first = sorted((root / "train_label").glob("*.tif"))[0]
        tifffile.imwrite(first, np.zeros((16, 16), np.uint16))
        with pytest.raises(InvalidTrainingData, match=first.name):
            load_training_data(root)

    def test_local_export_copies_tracking_table(self, tmp_path):
        annotation_dir = tmp_path / "project"
        (annotation_dir / "annotation_input").mkdir(parents=True)
        (annotation_dir / "annotation_output").mkdir()
        (annotation_dir / "annotation_input" / "0.tif").write_bytes(b"ann")
        (annotation_dir / "annotation_output" / "0_mask.tif").write_bytes(b"lbl")
        config = create_default_config()
        config.spatial_coverage.channels = [0]
        config.spatial_coverage.label_channel = None
        config.spatial_coverage.training_channels = None
        _write_rows(
            annotation_dir,
            [
                Row(
                    image_id=100,
                    image_name="img_0",
                    annotation_id="0",
                    category="training",
                    processed=True,
                )
            ],
        )
        output_dir = tmp_path / "project_training"
        reorganize_local_data_for_training(
            config=config, annotation_dir=annotation_dir, output_dir=output_dir
        )
        assert (output_dir / "tracking_table.csv").exists()

    def test_train_split_without_pairs_is_refused(self, tmp_path):
        with pytest.raises(InvalidTrainingData, match="train_input"):
            load_training_data(_export(tmp_path, n_train=0))

    def test_unreadable_tiff_names_the_file(self, tmp_path):
        root = _export(tmp_path)
        first = sorted((root / "train_input").glob("*.tif"))[0]
        first.write_bytes(b"not a tiff")
        with pytest.raises(InvalidTrainingData, match=first.name):
            load_training_data(root)

    def test_broken_sidecar_names_the_file(self, tmp_path):
        root = _export(tmp_path)
        (root / SIDECAR_NAME).write_text("{not json")
        with pytest.raises(InvalidTrainingData, match=SIDECAR_NAME):
            load_training_data(root)


def _stub_microsam(monkeypatch, calls):
    """Install fake micro_sam.training / torch / torch_em modules recording calls."""
    calls["loader"] = []

    def default_sam_loader(**kwargs):
        calls["loader"].append(kwargs)
        return object()

    def train_sam(**kwargs):
        calls["train_sam"] = kwargs
        best = Path(kwargs["save_root"]) / kwargs["name"] / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"weights")

    micro_sam = types.ModuleType("micro_sam")
    micro_sam.__version__ = "1.8.2"
    training = types.ModuleType("micro_sam.training")
    training.default_sam_loader = default_sam_loader
    training.train_sam = train_sam
    micro_sam.training = training
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_em = types.ModuleType("torch_em")
    torch_em_data = types.ModuleType("torch_em.data")
    torch_em_data.MinInstanceSampler = lambda **kwargs: ("sampler", kwargs)
    torch_em.data = torch_em_data
    for name, module in {
        "micro_sam": micro_sam,
        "micro_sam.training": training,
        "torch": torch,
        "torch_em": torch_em,
        "torch_em.data": torch_em_data,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


@pytest.mark.unit
class TestTrain:
    def test_unknown_trainer_lists_names(self, tmp_path):
        with pytest.raises(ValueError, match="microsam, cellpose, biapy"):
            train(_export(tmp_path), "stardist", tmp_path / "out")

    def test_accepts_folder_or_training_data(self, tmp_path, monkeypatch):
        _stub_microsam(monkeypatch, {})
        r1 = train(_export(tmp_path / "a"), "microsam", tmp_path / "o1", n_iterations=1)
        r2 = train(
            load_training_data(_export(tmp_path / "b")),
            "microsam",
            tmp_path / "o2",
            n_iterations=1,
        )
        assert r1.trainer == r2.trainer == "microsam"

    def test_result_to_json_has_string_paths(self, tmp_path, monkeypatch):
        _stub_microsam(monkeypatch, {})
        r = train(_export(tmp_path), "microsam", tmp_path / "out", n_iterations=1)
        assert isinstance(r.to_json()["model_path"], str)


@pytest.mark.unit
class TestMicroSamTrainer:
    def test_calls_train_sam_with_loaders_and_sidecar_transform(
        self, tmp_path, monkeypatch
    ):
        calls = _stub_microsam(monkeypatch, {})
        r = train(
            _export(tmp_path),
            "microsam",
            tmp_path / "out",
            model_name="m",
            n_iterations=3,
            learning_rate=3e-4,
            patch_shape=(64, 64),
        )
        assert calls["train_sam"]["n_iterations"] == 3
        assert calls["train_sam"]["lr"] == 3e-4
        assert calls["train_sam"]["save_root"] == str(tmp_path / "out")
        assert calls["loader"][0]["patch_shape"] == (64, 64)
        assert callable(calls["loader"][0]["raw_transform"])
        assert r.model_path == tmp_path / "out" / "m_final.pt" and r.model_path.exists()
        assert r.package_version == "1.8.2" and r.params["model_type"] == "vit_b_lm"

    def test_refuses_multichannel(self, tmp_path, monkeypatch):
        _stub_microsam(monkeypatch, {})
        with pytest.raises(InvalidTrainingData, match="one channel"):
            train(_export(tmp_path, shape=(32, 32, 3)), "microsam", tmp_path / "out")

    def test_volume_passes_patch_shape_through(self, tmp_path, monkeypatch):
        calls = _stub_microsam(monkeypatch, {})
        train(
            _export(tmp_path, shape=(4, 32, 32)),
            "microsam",
            tmp_path / "out",
            n_iterations=1,
            patch_shape=(2, 16, 16),
        )
        assert calls["loader"][0]["patch_shape"] == (2, 16, 16)

    def test_epochs_zero_without_iterations_is_refused(self, tmp_path, monkeypatch):
        _stub_microsam(monkeypatch, {})
        with pytest.raises(ValueError, match="epochs"):
            train(_export(tmp_path), "microsam", tmp_path / "out", epochs=0)

    def test_missing_package_names_the_extra(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "micro_sam", None)
        with pytest.raises(ImportError, match="microsam"):
            train(_export(tmp_path), "microsam", tmp_path / "out")

    def test_no_checkpoint_written_is_an_error(self, tmp_path, monkeypatch):
        calls = _stub_microsam(monkeypatch, {})
        monkeypatch.setattr(
            sys.modules["micro_sam.training"], "train_sam", lambda **kw: None
        )
        with pytest.raises(RuntimeError, match="no checkpoint"):
            train(_export(tmp_path), "microsam", tmp_path / "out", n_iterations=1)
        assert "train_sam" not in calls


def _stub_cellpose(monkeypatch, calls):
    """Install fake cellpose / cellpose.models / cellpose.train / torch modules."""

    class CellposeModel:
        def __init__(self, gpu=False, pretrained_model=""):
            calls["model"] = {"gpu": gpu, "pretrained_model": pretrained_model}
            self.net = "net"

    def train_seg(net, **kwargs):
        calls["train_seg"] = kwargs
        path = Path(kwargs["save_path"]) / "models" / kwargs["model_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return path, [1.0], [1.0]

    cellpose = types.ModuleType("cellpose")
    cellpose.version = "4.0.5"
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = CellposeModel
    train_mod = types.ModuleType("cellpose.train")
    train_mod.train_seg = train_seg
    cellpose.models = models
    cellpose.train = train_mod
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    for name, module in {
        "cellpose": cellpose,
        "cellpose.models": models,
        "cellpose.train": train_mod,
        "torch": torch,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


@pytest.mark.unit
class TestCellposeTrainer:
    def test_passes_arrays_and_names(self, tmp_path, monkeypatch):
        calls = _stub_cellpose(monkeypatch, {})
        r = train(
            _export(tmp_path), "cellpose", tmp_path / "out", model_name="cp", epochs=2
        )
        kw = calls["train_seg"]
        assert len(kw["train_data"]) == 2 and len(kw["test_data"]) == 1
        assert kw["train_data"][0].shape == (32, 32)
        assert kw["train_labels"][0].dtype.kind in "iu"
        assert kw["n_epochs"] == 2 and kw["normalize"] is True
        assert kw["model_name"] == "cp"
        assert r.model_path.exists() and r.package_version == "4.0.5"

    def test_normalised_export_turns_cellpose_normalisation_off(
        self, tmp_path, monkeypatch
    ):
        calls = _stub_cellpose(monkeypatch, {})
        root = _export(tmp_path, applied_at_export=True)
        r = train(root, "cellpose", tmp_path / "out")
        assert calls["train_seg"]["normalize"] is False
        assert r.params["normalize"] is False

    def test_volume_is_sliced_into_planes(self, tmp_path, monkeypatch):
        calls = _stub_cellpose(monkeypatch, {})
        r = train(_export(tmp_path, shape=(4, 32, 32)), "cellpose", tmp_path / "out")
        assert len(calls["train_seg"]["train_data"]) == 8
        assert calls["train_seg"]["train_data"][0].shape == (32, 32)
        assert r.extras["n_planes"] == 8

    def test_more_than_three_channels_refused(self, tmp_path, monkeypatch):
        _stub_cellpose(monkeypatch, {})
        with pytest.raises(InvalidTrainingData, match="3 channels"):
            train(_export(tmp_path, shape=(32, 32, 4)), "cellpose", tmp_path / "out")

    def test_missing_package_names_the_extra(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "cellpose", None)
        with pytest.raises(ImportError, match=r"omero-annotate-ai\[cellpose\]"):
            train(_export(tmp_path), "cellpose", tmp_path / "out")


def _stub_biapy(monkeypatch, calls, *, write_checkpoint=True):
    """Install fake biapy / torch modules."""

    class BiaPy:
        def __init__(self, config, result_dir, name, run_id, gpu):
            calls["init"] = {
                "config": config,
                "result_dir": result_dir,
                "name": name,
                "run_id": run_id,
                "gpu": gpu,
            }
            self._result_dir, self._name = result_dir, name

        def run_job(self):
            if write_checkpoint:
                ckpt = Path(self._result_dir) / self._name / "checkpoints"
                ckpt.mkdir(parents=True, exist_ok=True)
                (ckpt / f"{self._name}_1-checkpoint-best.pth").write_bytes(b"w")

    biapy = types.ModuleType("biapy")
    biapy.__version__ = "3.6.8"
    biapy.BiaPy = BiaPy
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "biapy", biapy)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


@pytest.mark.unit
class TestBiaPyTrainer:
    def test_writes_config_and_runs_job(self, tmp_path, monkeypatch):
        calls = _stub_biapy(monkeypatch, {})
        r = train(
            _export(tmp_path), "biapy", tmp_path / "out", model_name="bp", epochs=3
        )
        cfg = yaml.safe_load((tmp_path / "out" / "biapy.yaml").read_text())
        assert cfg["PROBLEM"]["TYPE"] == "INSTANCE_SEG"
        assert cfg["PROBLEM"]["NDIM"] == "2D"
        assert cfg["DATA"]["TRAIN"]["PATH"] == str(tmp_path / "export" / "train_input")
        assert cfg["DATA"]["VAL"]["FROM_TRAIN"] is False
        assert cfg["TRAIN"]["EPOCHS"] == 3
        assert cfg["DATA"]["NORMALIZATION"]["PERC_CLIP"]["ENABLE"] is True
        assert calls["init"]["name"] == "bp" and calls["init"]["run_id"] == 1
        assert calls["init"]["gpu"] is None
        assert r.model_path == (
            tmp_path / "out" / "bp" / "checkpoints" / "bp_1-checkpoint-best.pth"
        )
        assert r.package_version == "3.6.8"
        assert r.extras["log_dir"] == tmp_path / "out" / "bp" / "logs"

    def test_volume_and_channels_set_ndim_and_patch(self, tmp_path, monkeypatch):
        _stub_biapy(monkeypatch, {})
        train(_export(tmp_path, shape=(8, 32, 32, 2)), "biapy", tmp_path / "out")
        cfg = yaml.safe_load((tmp_path / "out" / "biapy.yaml").read_text())
        assert cfg["PROBLEM"]["NDIM"] == "3D"
        assert cfg["DATA"]["PATCH_SIZE"] == [16, 128, 128, 2]

    def test_user_patch_size_is_spatial_only(self, tmp_path, monkeypatch):
        _stub_biapy(monkeypatch, {})
        train(_export(tmp_path), "biapy", tmp_path / "out", patch_size=(64, 64))
        cfg = yaml.safe_load((tmp_path / "out" / "biapy.yaml").read_text())
        assert cfg["DATA"]["PATCH_SIZE"] == [64, 64, 1]

    def test_user_config_overrides_nested_keys(self, tmp_path, monkeypatch):
        _stub_biapy(monkeypatch, {})
        train(
            _export(tmp_path),
            "biapy",
            tmp_path / "out",
            config={"MODEL": {"ARCHITECTURE": "resunet"}, "TRAIN": {"BATCH_SIZE": 4}},
        )
        cfg = yaml.safe_load((tmp_path / "out" / "biapy.yaml").read_text())
        assert cfg["MODEL"]["ARCHITECTURE"] == "resunet"
        assert cfg["TRAIN"]["EPOCHS"] == 100 and cfg["TRAIN"]["BATCH_SIZE"] == 4

    def test_missing_checkpoint_names_log_dir(self, tmp_path, monkeypatch):
        _stub_biapy(monkeypatch, {}, write_checkpoint=False)
        with pytest.raises(RuntimeError, match="logs"):
            train(_export(tmp_path), "biapy", tmp_path / "out", model_name="bp")

    def test_job_failure_names_log_dir(self, tmp_path, monkeypatch):
        _stub_biapy(monkeypatch, {})
        monkeypatch.setattr(
            sys.modules["biapy"].BiaPy,
            "run_job",
            lambda self: (_ for _ in ()).throw(ValueError("boom")),
        )
        with pytest.raises(RuntimeError, match="logs"):
            train(_export(tmp_path), "biapy", tmp_path / "out", model_name="bp")

    def test_missing_package_names_the_extra(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "biapy", None)
        with pytest.raises(ImportError, match=r"omero-annotate-ai\[biapy\]"):
            train(_export(tmp_path), "biapy", tmp_path / "out")
