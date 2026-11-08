"""Tests for the config-defined intensity normalisation of model input images."""

import json

import numpy as np
import pytest
from tifffile import imread, imwrite

from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    NormalizationConfig,
)
from omero_annotate_ai.processing.normalization import (
    SIDECAR_NAME,
    biapy_normalization,
    percentile_normalize,
)
from omero_annotate_ai.processing.training_functions import (
    reorganize_local_data_for_training,
)
from omero_annotate_ai.rows import Row, rows_to_frame


@pytest.mark.unit
class TestPercentileNormalize:
    def test_maps_percentiles_to_zero_and_one(self):
        image = np.arange(1001, dtype=np.uint16).reshape(7, 143)[:, :143]
        out, bounds = percentile_normalize(image, 1.0, 99.0)
        low, high = bounds[0]
        assert out.dtype == np.float32
        assert out.min() == 0.0 and out.max() == 1.0
        assert low == pytest.approx(np.percentile(image, 1.0))
        assert high == pytest.approx(np.percentile(image, 99.0))

    def test_independent_of_bit_depth(self):
        """12-bit data in a 16-bit container normalises like the same data as 8-bit."""
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, size=(32, 32))
        out8, _ = percentile_normalize(base.astype(np.uint8), 1.0, 99.8)
        out16, _ = percentile_normalize((base * 16).astype(np.uint16), 1.0, 99.8)
        np.testing.assert_allclose(out8, out16, atol=1e-6)

    def test_per_channel_bounds(self):
        image = np.stack(
            [np.full((4, 4), 10.0), np.arange(16.0).reshape(4, 4)], axis=-1
        )
        out, bounds = percentile_normalize(image, 0.0, 100.0, channel_axis=-1)
        assert len(bounds) == 2
        assert np.all(out[..., 0] == 0)  # constant channel: no division by zero
        assert out[..., 1].max() == 1.0

    def test_no_clip_keeps_outliers(self):
        image = np.array([[0, 1, 2, 3, 100]], dtype=np.float32)
        out, _ = percentile_normalize(image, 0.0, 75.0, clip=False)
        assert out.max() > 1.0


@pytest.mark.unit
class TestNormalizationConfig:
    def test_legacy_spatial_coverage_fields_are_refused(self):
        from omero_annotate_ai.errors import RemovedConfigKey

        with pytest.raises(RemovedConfigKey, match="normalization.method"):
            AnnotationConfig.from_dict(
                {"name": "legacy", "spatial_coverage": {"normalization_method": "none"}}
            )

    def test_low_must_be_below_high(self):
        with pytest.raises(ValueError, match="percentile_low"):
            NormalizationConfig(percentile_low=99.0, percentile_high=1.0)


@pytest.mark.unit
class TestBiapyNormalization:
    def test_applied_at_export_is_identity(self):
        block = biapy_normalization(NormalizationConfig(apply_at_export=True))
        assert block["TYPE"] == "zero_mean_unit_variance"
        assert block["ZERO_MEAN_UNIT_VAR"] == {"MEAN_VAL": 0.0, "STD_VAL": 1.0}
        assert block["PERC_CLIP"]["ENABLE"] is False

    def test_trainer_side_percentiles_avoid_bit_depth_guessing(self):
        block = biapy_normalization(
            NormalizationConfig(percentile_low=2.0, percentile_high=99.0)
        )
        assert block["TYPE"] not in ("div", "scale_range")
        assert block["PERC_CLIP"] == {
            "ENABLE": True,
            "LOWER_PERC": 2.0,
            "UPPER_PERC": 99.0,
        }


def _local_annotation_set(tmp_path, apply_at_export):
    annotation_dir = tmp_path / "annotations"
    (annotation_dir / "annotation_input").mkdir(parents=True)
    (annotation_dir / "annotation_output").mkdir()
    config = AnnotationConfig(
        name="norm",
        normalization={
            "apply_at_export": apply_at_export,
            "percentile_low": 0.0,
            "percentile_high": 100.0,
        },
    )
    rows = []
    for i, category in enumerate(["training", "validation"]):
        annotation_id = f"img{i}"
        imwrite(
            annotation_dir / "annotation_input" / f"{annotation_id}.tif",
            (np.arange(64).reshape(8, 8) * 50 + i).astype(np.uint16),
        )
        imwrite(
            annotation_dir / "annotation_output" / f"{annotation_id}_mask.tif",
            np.ones((8, 8), dtype=np.uint16),
        )
        rows.append(
            Row(
                image_id=i,
                image_name=annotation_id,
                annotation_id=annotation_id,
                category=category,
                processed=True,
            )
        )
    rows_to_frame(rows).to_csv(annotation_dir / "tracking_table.csv", index=False)
    return config, annotation_dir


@pytest.mark.unit
class TestExportNormalization:
    def test_native_export_keeps_dtype_and_records_spec(self, tmp_path):
        config, annotation_dir = _local_annotation_set(tmp_path, apply_at_export=False)
        result = reorganize_local_data_for_training(
            config, annotation_dir, tmp_path / "training"
        )
        image = imread(result["train_input"] / "img0.tif")
        assert image.dtype == np.uint16
        sidecar = json.loads((tmp_path / "training" / SIDECAR_NAME).read_text())
        assert sidecar["applied_at_export"] is False
        assert sidecar["dtype"] == "native"

    def test_export_normalisation_writes_float32_and_bounds(self, tmp_path):
        config, annotation_dir = _local_annotation_set(tmp_path, apply_at_export=True)
        result = reorganize_local_data_for_training(
            config, annotation_dir, tmp_path / "training"
        )
        image = imread(result["train_input"] / "img0.tif")
        assert image.dtype == np.float32
        assert image.min() == 0.0 and image.max() == 1.0
        sidecar = json.loads((tmp_path / "training" / SIDECAR_NAME).read_text())
        assert sidecar["applied_at_export"] is True
        assert sidecar["bounds_per_image"]["img0"] == [[0.0, 63 * 50.0]]
        assert sidecar["bounds_per_image"]["img1"] == [[1.0, 63 * 50.0 + 1]]
        # labels are never normalised
        assert imread(result["train_label"] / "img0.tif").dtype == np.uint16


@pytest.mark.unit
class TestMicroSamRawTransform:
    """micro-SAM training needs inputs in [0, 255]; exports keep native bit depth."""

    def test_percentile_scales_16bit_to_8bit_range(self):
        from omero_annotate_ai.processing.normalization import microsam_raw_transform

        image = (
            np.linspace(0, 60000, 1000, dtype=np.float32)
            .astype(np.uint16)
            .reshape(20, 50)
        )
        out = microsam_raw_transform(NormalizationConfig())(image)
        assert out.min() >= 0 and out.max() <= 255
        assert out.max() > 250

    def test_applied_at_export_scales_unit_range(self):
        from omero_annotate_ai.processing.normalization import microsam_raw_transform

        spec = NormalizationConfig(apply_at_export=True)
        out = microsam_raw_transform(spec)(
            np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        )
        assert np.allclose(out, [[0, 127.5, 255]])

    def test_none_rejects_values_above_255(self):
        from omero_annotate_ai.processing.normalization import microsam_raw_transform

        transform = microsam_raw_transform(NormalizationConfig(method="none"))
        assert transform(np.array([[0, 200]], dtype=np.uint8)).max() == 200
        with pytest.raises(ValueError, match="0-255"):
            transform(np.array([[0, 4000]], dtype=np.uint16))

    def test_setup_training_reads_sidecar(self, tmp_path):
        from omero_annotate_ai.processing.normalization import write_native_sidecar
        from omero_annotate_ai.processing.training_utils import setup_training

        for name in ("train_input", "train_label", "val_input", "val_label"):
            (tmp_path / name).mkdir()
        write_native_sidecar(NormalizationConfig(percentile_high=99.0), tmp_path)
        result = {
            name: tmp_path / name
            for name in ("train_input", "train_label", "val_input", "val_label")
        }
        result["output_dir"] = tmp_path

        config = setup_training(result)
        assert config["normalization"]["percentile_high"] == 99.0
        assert config["normalization"]["applied_at_export"] is False
