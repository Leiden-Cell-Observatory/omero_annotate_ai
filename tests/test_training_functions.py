"""Tests for training data preparation functions."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    create_default_config,
)
from omero_annotate_ai.processing import training_functions as tf
from omero_annotate_ai.processing.training_functions import (
    reorganize_local_data_for_training,
)
from omero_annotate_ai.rows import Row, rows_to_frame


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

    def test_result_satisfies_setup_training_contract(self, tmp_path):
        """Both producers must emit the keys setup_training requires.

        The old suite checked this across both producers; afterwards only the OMERO
        one was covered. reorganize used to return validation_input/validation_label,
        so feeding it to setup_training raised ValueError.
        """
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config(annotation_dir, ["training", "validation"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        for key in ("train_input", "train_label", "val_input", "val_label"):
            assert key in result, f"setup_training requires {key}"
            assert result[key].exists()

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

    def test_files_named_by_annotation_id(self, tmp_path):
        result = self._run(tmp_path, self._table())

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        assert images == ["0.tif", "1.tif"]

    def test_result_keys_feed_setup_training(self, tmp_path):
        table = self._table()
        table.loc[1, "category"] = "validation"

        result = self._run(tmp_path, table)

        for key in ("train_input", "train_label", "val_input", "val_label"):
            assert key in result, f"setup_training requires {key}"
        assert (result["val_input"] / "1.tif").exists()
        assert (result["val_input"] / "1.tif").exists()

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
class TestRunTrainingLength:
    """How long micro-SAM trains: epochs, or n_iterations when given."""

    def _result(self, tmp_path):
        folders = {}
        for key in ("train_input", "train_label", "val_input", "val_label"):
            folders[key] = tmp_path / key
            folders[key].mkdir()
        return folders

    def test_epochs_zero_without_iterations_is_rejected(self, tmp_path):
        from omero_annotate_ai.processing.training_utils import setup_training

        with pytest.raises(ValueError, match="epochs > 0"):
            setup_training(self._result(tmp_path), epochs=0)

    def test_n_iterations_reaches_train_sam(self, tmp_path, monkeypatch):
        sam_training = pytest.importorskip("micro_sam.training")
        from omero_annotate_ai.processing.training_utils import (
            run_training,
            setup_training,
        )

        seen = {}
        monkeypatch.setattr(sam_training, "default_sam_loader", lambda **_: object())
        monkeypatch.setattr(
            sam_training, "train_sam", lambda **kwargs: seen.update(kwargs)
        )

        config = setup_training(self._result(tmp_path), epochs=5, n_iterations=3)
        outcome = run_training(config)

        assert seen["n_iterations"] == 3
        assert seen["n_epochs"] == 5  # micro-SAM ignores it when n_iterations is set
        assert outcome["training_stats"]["n_iterations"] == 3

    def test_epochs_only_leaves_iterations_unset(self, tmp_path, monkeypatch):
        sam_training = pytest.importorskip("micro_sam.training")
        from omero_annotate_ai.processing.training_utils import (
            run_training,
            setup_training,
        )

        seen = {}
        monkeypatch.setattr(sam_training, "default_sam_loader", lambda **_: object())
        monkeypatch.setattr(
            sam_training, "train_sam", lambda **kwargs: seen.update(kwargs)
        )

        run_training(setup_training(self._result(tmp_path), epochs=2))

        assert seen["n_epochs"] == 2
        assert seen["n_iterations"] is None


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
