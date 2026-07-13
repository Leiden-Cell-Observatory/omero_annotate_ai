"""Tests for training data preparation functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from omero_annotate_ai.processing.training_functions import (
    prepare_training_data_from_table,
    reorganize_local_data_for_training,
)
from omero_annotate_ai.core.annotation_config import AnnotationConfig, ImageAnnotation, create_default_config


class TestPrepareTrainingDataFromTable:
    """Test the automated training data preparation function."""

    @pytest.fixture
    def mock_conn(self):
        """Mock OMERO connection."""
        return Mock()

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_invalid_validation_split(self, mock_conn, temp_output_dir):
        """validation_split outside 0.0-1.0 must raise."""
        with pytest.raises(ValueError, match="validation_split must be between"):
            prepare_training_data_from_table(
                conn=mock_conn,
                table_id=123,
                output_dir=temp_output_dir,
                validation_split=1.5,
            )

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

    def test_label_input_id_round_trips_through_dataframe(self):
        """label_input_id is serialized and deserialized correctly via to/from_dataframe."""
        config = AnnotationConfig(name="classification_workflow")
        ann = ImageAnnotation(
            image_id=42,
            image_name="test_image",
            timepoint=0,
            z_slice=0,
            channel=0,
            label_id=101,
            label_input_id=202,
            processed=True,
        )
        config.annotations.append(ann)

        df = config.to_dataframe()
        assert "label_input_id" in df.columns
        assert df.iloc[0]["label_input_id"] == "202"

        config2 = AnnotationConfig(name="classification_workflow")
        config2.from_dataframe(df)
        assert config2.annotations[0].label_input_id == 202

    def test_label_input_id_none_round_trips(self):
        """label_input_id=None serializes as 'None' and deserializes back to None."""
        config = AnnotationConfig(name="classification_workflow")
        ann = ImageAnnotation(
            image_id=42,
            image_name="test_image",
            timepoint=0,
            z_slice=0,
            channel=0,
            label_id=101,
            label_input_id=None,
            processed=True,
        )
        config.annotations.append(ann)

        df = config.to_dataframe()
        assert df.iloc[0]["label_input_id"] == "None"

        config2 = AnnotationConfig(name="classification_workflow")
        config2.from_dataframe(df)
        assert config2.annotations[0].label_input_id is None

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


@pytest.mark.unit
class TestReorganizeOntoRecords:
    """The offline producer emits the unified layout, paired by annotation_id."""

    def _config(self, tmp_path, categories, separate_channels=False):
        config = create_default_config()
        config.output.output_directory = str(tmp_path)
        if separate_channels:
            config.spatial_coverage.channels = [0, 1]
            config.spatial_coverage.label_channel = 0
            config.spatial_coverage.training_channels = [1]
        else:
            config.spatial_coverage.channels = [0]
            config.spatial_coverage.label_channel = None
            config.spatial_coverage.training_channels = None
        config.annotations = [
            ImageAnnotation(
                image_id=100 + i,
                image_name=f"img_{i}",
                annotation_id=str(i),
                category=category,
                processed=True,
            )
            for i, category in enumerate(categories)
        ]
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
        assert (result["val_input"] / "1.tif").exists()
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
        assert mapping["image"].endswith("train_input/0.tif")
        assert mapping["label"].endswith("train_label/0.tif")

    def test_no_processed_annotations_raises(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config(annotation_dir, ["training"])
        for ann in config.annotations:
            ann.processed = False

        with pytest.raises(ValueError, match="No processed annotations"):
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
                    "train": True,
                    "validate": False,
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

        with patch.object(tf, "_fetch_plane", return_value=plane), patch.object(
            tf, "_download_label", side_effect=fake_download
        ), patch.object(tf, "_load_table", return_value=table), patch.object(
            tf.ezomero, "get_table", return_value=table
        ):
            return tf.prepare_training_data_from_table(
                conn=Mock(),
                table_id=1,
                output_dir=tmp_path / "training",
                validation_split=0.0,
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
        table.loc[1, "train"] = False
        table.loc[1, "validate"] = True

        result = self._run(tmp_path, table)

        for key in ("train_input", "train_label", "val_input", "val_label"):
            assert key in result, f"setup_training requires {key}"
        assert (result["val_input"] / "1.tif").exists()
