"""Tests for the annotation pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import tifffile

from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    ImageAnnotation,
    create_default_config,
)
from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline
from omero_annotate_ai.sources import ImageMeta, OmeroImageSource, clamp_region
from omero_annotate_ai.sources.uri import (
    annotation_id as make_annotation_id,
)
from omero_annotate_ai.sources.uri import (
    omero_uri,
    parse_omero_uri,
)


def _read_only_config():
    """Default config that needs no OMERO connection (nothing is written back)."""
    config = create_default_config()
    config.workflow.read_only_mode = True
    return config


class FakeImage:
    """Handle with no OMERO methods: proves the pipeline only talks to the source."""

    def __init__(self, image_id, name):
        self.image_id = image_id
        self.name = name


class FakeImageSource:
    """In-memory ImageSource over (t, c, z, y, x) arrays; records read calls."""

    def __init__(self, arrays):
        self.arrays = arrays  # {image_id: ndarray (t, c, z, y, x)}
        self.reads = []

    def get_image(self, image_id):
        if isinstance(image_id, str):
            image_id = parse_omero_uri(image_id)
        if image_id not in self.arrays:
            raise LookupError(image_id)
        return FakeImage(image_id, f"img{image_id}")

    def metadata(self, image):
        t, c, z, y, x = self.arrays[image.image_id].shape
        return ImageMeta(
            omero_uri(image.image_id),
            image.name,
            t,
            c,
            z,
            y,
            x,
            str(self.arrays[image.image_id].dtype),
        )

    def read_plane(self, image, *, t, c, z, region=None):
        self.reads.append(("plane", image.image_id, t, c, z, region))
        plane = self.arrays[image.image_id][t, c, z]
        if region is not None:
            x, y, w, h = clamp_region(region, *plane.shape)
            plane = plane[y : y + h, x : x + w]
        return plane

    def read_volume(self, image, *, t, c, z_start, z_end, region=None):
        self.reads.append(("volume", image.image_id, t, c, (z_start, z_end), region))
        return np.stack(
            [
                self.read_plane(image, t=t, c=c, z=z, region=region)
                for z in range(z_start, z_end + 1)
            ]
        )


@pytest.mark.unit
class TestImageSourceSeam:
    def test_annotations_carry_uri_and_hash_ids(self, tmp_path):
        source = FakeImageSource({1: np.zeros((1, 1, 2, 16, 16), np.uint16)})
        config = _read_only_config()
        config.output.output_directory = tmp_path
        config.spatial_coverage.z_slice_mode = "all"
        config.training.segment_all = True
        pipeline = AnnotationPipeline(config, conn=None, image_source=source)

        pipeline._prepare_processing_units([source.get_image(1)])

        anns = pipeline.config.annotations
        assert {a.image_uri for a in anns} == {"omero:image:1"}
        assert {a.image_id for a in anns} == {1}
        assert sorted(a.annotation_id for a in anns) == sorted(
            make_annotation_id("omero:image:1", 0, z) for z in (0, 1)
        )

    def test_processing_units_are_keyed_by_uri(self, tmp_path):
        config = _read_only_config()
        config.output.output_directory = tmp_path
        config.annotations.append(
            ImageAnnotation(image_id=3, image_name="a", annotation_id="k_0_0")
        )
        pipeline = AnnotationPipeline(
            config, conn=None, image_source=FakeImageSource({})
        )
        ((key, ann_id, meta, row),) = pipeline._convert_annotations_to_processing_units(
            config.annotations
        )
        assert key == "omero:image:3"
        assert (meta["image_uri"], meta["image_id"]) == ("omero:image:3", 3)

    """The pipeline reads image sizes and pixels only through its ImageSource."""

    def test_default_source_is_omero_on_the_same_connection(self):
        conn = Mock()
        pipeline = AnnotationPipeline(create_default_config(), conn=conn)
        assert isinstance(pipeline.image_source, OmeroImageSource)
        assert pipeline.image_source.conn is conn

    def test_prepare_units_uses_source_metadata_without_a_connection(self, tmp_path):
        arrays = {
            1: np.zeros((2, 1, 3, 32, 32), np.uint16),
            2: np.zeros((2, 1, 3, 32, 32), np.uint16),
        }
        source = FakeImageSource(arrays)
        config = create_default_config()
        config.output.output_directory = tmp_path
        config.spatial_coverage.timepoint_mode = "all"
        config.spatial_coverage.z_slice_mode = "all"
        config.training.segment_all = True
        config.workflow.read_only_mode = True
        pipeline = AnnotationPipeline(config, conn=None, image_source=source)

        pipeline._prepare_processing_units([source.get_image(1), source.get_image(2)])

        units = {
            (a.image_id, a.timepoint, a.z_slice) for a in pipeline.config.annotations
        }
        assert units == {(i, t, z) for i in (1, 2) for t in range(2) for z in range(3)}
        assert {a.image_name for a in pipeline.config.annotations} == {"img1", "img2"}

    def test_patch_coordinates_use_source_size(self):
        source = FakeImageSource({1: np.zeros((1, 1, 1, 50, 80), np.uint16)})
        config = create_default_config()
        config.spatial_coverage.patch_size = [20, 30]
        config.spatial_coverage.patches_per_image = 3
        config.spatial_coverage.random_patches = False  # random placement may fit fewer
        config.workflow.read_only_mode = True
        pipeline = AnnotationPipeline(config, conn=None, image_source=source)

        coords, _ = pipeline._generate_patch_coordinates(source.get_image(1))

        assert len(coords) == 3
        assert all(0 <= x <= 80 - 30 and 0 <= y <= 50 - 20 for x, y in coords)

    def test_no_connection_requires_read_only_mode(self):
        """Without a connection nothing can be written back, so fail before annotating."""
        source = FakeImageSource({1: np.zeros((1, 1, 1, 4, 4), np.uint16)})
        with pytest.raises(ValueError, match="OMERO connection is required"):
            AnnotationPipeline(create_default_config(), conn=None, image_source=source)

    def test_omero_source_with_its_own_connection_is_accepted(self):
        config = create_default_config()
        config.workflow.read_only_mode = True
        pipeline = AnnotationPipeline(
            config, conn=None, image_source=OmeroImageSource(Mock())
        )
        assert isinstance(pipeline.image_source, OmeroImageSource)

    def test_default_omero_source_without_connection_raises_in_read_only_mode(self):
        config = create_default_config()
        config.workflow.read_only_mode = True
        with pytest.raises(ValueError, match="OMERO connection is required"):
            AnnotationPipeline(config, conn=None)

    def _pipeline(self, arrays, use_patches=False):
        config = create_default_config()
        config.spatial_coverage.use_patches = use_patches
        source = FakeImageSource(arrays)
        config.workflow.read_only_mode = True
        return AnnotationPipeline(config, conn=None, image_source=source), source

    def test_load_plane(self):
        data = np.arange(2 * 2 * 3 * 8 * 8, dtype=np.uint16).reshape(2, 2, 3, 8, 8)
        pipeline, source = self._pipeline({1: data})
        image = source.get_image(1)
        out = pipeline._load_image_data(
            image, {"timepoint": 1, "z_slice": 2, "channel_override": 1}
        )
        np.testing.assert_array_equal(out, data[1, 1, 2])

    def test_load_volume_is_3d(self):
        data = np.ones((1, 1, 5, 8, 8), dtype=np.uint16)
        pipeline, source = self._pipeline({1: data})
        out = pipeline._load_image_data(
            source.get_image(1),
            {
                "timepoint": 0,
                "is_volumetric": True,
                "z_start": 1,
                "z_end": 3,
                "z_length": 3,
            },
        )
        assert out.shape == (3, 8, 8)

    def test_patch_is_read_as_region(self):
        data = np.arange(64, dtype=np.uint16).reshape(1, 1, 1, 8, 8)
        pipeline, source = self._pipeline({1: data}, use_patches=True)
        out = pipeline._load_image_data(
            source.get_image(1),
            {
                "timepoint": 0,
                "z_slice": 0,
                "patch_x": 2,
                "patch_y": 1,
                "patch_width": 3,
                "patch_height": 4,
            },
        )
        np.testing.assert_array_equal(out, data[0, 0, 0, 1:5, 2:5])
        assert source.reads[-1][-1] == (
            2,
            1,
            3,
            4,
        )  # region passed, not cropped afterwards

    def test_get_images_by_ids_uses_source_and_skips_missing(self):
        pipeline, source = self._pipeline({1: np.zeros((1, 1, 1, 4, 4), np.uint16)})
        images = pipeline.get_images_by_ids([1, 99])
        assert [i.image_id for i in images] == [1]


@pytest.mark.unit
class TestAnnotationPipeline:
    """Test the main annotation pipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with config."""
        config = create_default_config()
        config.omero.container_id = 123

        # Should raise ValueError when no connection provided
        with pytest.raises(ValueError, match="OMERO connection is required"):
            AnnotationPipeline(config, conn=None)

    def test_pipeline_initialization_with_connection(self):
        """Test pipeline initialization with mock connection."""
        config = create_default_config()
        mock_conn = Mock()

        pipeline = AnnotationPipeline(config, conn=mock_conn)

        assert pipeline.config == config
        assert pipeline.conn == mock_conn

    def test_pipeline_validation_success(self):
        """Test successful pipeline creation with valid config."""
        config = create_default_config()
        config.omero.container_id = 123

        # Should create pipeline successfully with valid config and connection
        pipeline = AnnotationPipeline(config, conn=Mock())

        assert pipeline.config == config
        assert pipeline.conn is not None

    def test_config_pydantic_validation(self):
        """Test that Pydantic validates config fields properly."""
        config = create_default_config()

        # Test valid assignment
        config.omero.container_id = 123
        assert config.omero.container_id == 123

        # Test that certain fields have expected defaults/validation
        assert config.training.train_fraction >= 0.1
        assert config.training.train_fraction <= 0.9
        assert config.workflow.batch_size >= 0

    @patch("omero_annotate_ai.core.annotation_pipeline.ezomero")
    def test_get_images_from_container_dataset(self, mock_ezomero):
        """Test getting images from dataset."""
        mock_ezomero.get_image_ids.return_value = [1, 2, 3]
        config = create_default_config()
        config.omero.container_type = "dataset"
        config.omero.container_id = 123
        mock_conn = Mock()
        mock_images = [Mock() for _ in range(3)]
        mock_conn.getObject = Mock(side_effect=mock_images)
        pipeline = AnnotationPipeline(config, conn=mock_conn)
        images = pipeline.get_images_from_container()
        assert len(images) == 3
        mock_ezomero.get_image_ids.assert_called_once_with(pipeline.conn, dataset=123)

    @patch("omero_annotate_ai.core.annotation_pipeline.ezomero")
    def test_get_images_from_container_project(self, mock_ezomero):
        """Test getting images from project (via datasets)."""
        mock_ezomero.get_dataset_ids.return_value = [10, 20]
        mock_ezomero.get_image_ids.side_effect = [
            [1, 2],  # Images in dataset 10
            [3, 4, 5],  # Images in dataset 20
        ]
        config = create_default_config()
        config.omero.container_type = "project"
        config.omero.container_id = 789
        mock_conn = Mock()
        mock_images = [Mock() for _ in range(5)]
        mock_conn.getObject = Mock(side_effect=mock_images)
        pipeline = AnnotationPipeline(config, conn=mock_conn)
        images = pipeline.get_images_from_container()
        assert len(images) == 5
        mock_ezomero.get_dataset_ids.assert_called_once_with(pipeline.conn, project=789)
        assert mock_ezomero.get_image_ids.call_count == 2

    def test_get_images_from_container_single_image(self):
        """Test getting single image."""
        config = create_default_config()
        config.omero.container_type = "image"
        config.omero.container_id = 456

        mock_image = Mock()
        mock_conn = Mock()
        mock_conn.getObject.return_value = mock_image

        pipeline = AnnotationPipeline(config, conn=mock_conn)
        images = pipeline.get_images_from_container()

        assert len(images) == 1
        assert images[0] == mock_image
        mock_conn.getObject.assert_called_once_with("Image", 456)

    def test_bundle_dir_uses_explicit_output_directory(self, tmp_path):
        config = create_default_config()
        config.output.output_directory = tmp_path / "my_set"

        pipeline = AnnotationPipeline(config, conn=Mock())

        assert pipeline.bundle_dir == tmp_path / "my_set"

    def test_bundle_dir_of_omero_set_is_named_after_server_and_table(self, tmp_path):
        config = create_default_config()
        config.name = "nuclei"
        pipeline = AnnotationPipeline(
            config, conn=Mock(host="omero.example.org", port=4064), home=tmp_path
        )
        with pytest.raises(RuntimeError, match="does not exist yet"):
            _ = pipeline.bundle_dir

        pipeline.set_table_id(42)

        expected = tmp_path / "omero.example.org-4064" / "42_nuclei"
        assert pipeline.bundle_dir == expected
        assert config.output.output_directory == expected  # stored for resume

    def test_bundle_dir_of_read_only_set_is_local(self, tmp_path):
        config = create_default_config()
        config.name = "nuclei"
        config.workflow.read_only_mode = True

        pipeline = AnnotationPipeline(config, conn=Mock(), home=tmp_path)

        assert pipeline.bundle_dir.parent == tmp_path / "local"
        assert pipeline.bundle_dir.name.endswith("_nuclei")
        assert pipeline.bundle_dir == pipeline.bundle_dir  # chosen once

    def test_get_table_title(self):
        """Test table title generation - should use config name directly."""
        config = create_default_config()
        config.name = "my_annotation_workflow_20250101_120000"

        pipeline = AnnotationPipeline(config, conn=Mock())
        title = pipeline._get_table_title()

        # Should return config name directly without modification
        assert title == "my_annotation_workflow_20250101_120000"

    def test_get_table_title_fallback(self):
        """Test table title generation with fallback when config name is empty."""
        config = create_default_config()
        config.name = ""  # Empty name triggers fallback
        config.omero.container_type = "dataset"
        config.omero.container_id = 123

        pipeline = AnnotationPipeline(config, conn=Mock())
        title = pipeline._get_table_title()

        # Should use clean naming
        assert title.startswith("dataset_123_")
        # Should include timestamp
        assert len(title.split("_")) >= 3  # format: dataset_123_timestamp

    def test_table_naming_consistency_throughout_pipeline(self):
        """Test that table naming is consistent throughout the pipeline.

        This test ensures that:
        1. generate_unique_table_name creates names with proper format
        2. _get_table_title preserves the name from config
        3. The table title used in create_tracking_table matches config.name
        """
        from omero_annotate_ai.omero.omero_functions import generate_unique_table_name

        # Setup mock connection and container
        mock_conn = Mock()
        mock_container = Mock()
        mock_container.getName.return_value = "TestDataset"
        mock_conn.getObject.return_value = mock_container

        # Test 1: generate_unique_table_name with custom name
        unique_name = generate_unique_table_name(
            mock_conn, "Dataset", 123, base_name="my_custom_name"
        )
        assert "my_custom_name" in unique_name

        # Test 2: generate_unique_table_name uses container name when no custom name
        unique_name_no_custom = generate_unique_table_name(mock_conn, "Dataset", 123)
        # Container name is lowercased by generate_unique_table_name
        assert unique_name_no_custom.startswith("testdataset_")

        # Test 3: Pipeline uses config name directly without modification
        config = create_default_config()
        config.name = unique_name

        pipeline = AnnotationPipeline(config, conn=mock_conn)
        title = pipeline._get_table_title()

        # Should be exactly the same as config.name - no prefix added
        assert title == config.name
        assert title == unique_name

        # Test 4: Verify create_tracking_table receives the exact config name
        config.training.train_n = 1
        config.training.validate_n = 1
        config.training.test_n = 0
        config.training.segment_all = False
        mock_images = []
        for i in range(2):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"image_{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)

        pipeline.define_annotation_schema(mock_images)

        with patch(
            "omero_annotate_ai.core.annotation_pipeline.create_omero_tracking_table"
        ) as mock_create_table:
            mock_create_table.return_value = 999
            pipeline.create_tracking_table()

            # Verify the table was created with the exact name from config
            mock_create_table.assert_called_once()
            call_args = mock_create_table.call_args
            table_title_arg = call_args.kwargs.get("table_title")

            assert table_title_arg == unique_name
            assert table_title_arg == config.name

    def test_prepare_processing_units_basic(self):
        """Test preparing processing units from images."""
        config = create_default_config()
        config.training.train_n = 2
        config.training.validate_n = 1
        config.training.test_n = 0  # No test images for this test
        config.training.segment_all = False

        # Mock images
        mock_images = []
        for i in range(5):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"image_{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        # Should create annotations for train_n + validate_n images
        assert len(pipeline.config.annotations) == 3

        # Check categories
        categories = [ann.category for ann in pipeline.config.annotations]
        assert categories.count("training") == 2
        assert categories.count("validation") == 1

    def test_prepare_processing_units_segment_all(self):
        """Test preparing processing units with segment_all=True."""
        config = create_default_config()
        config.training.segment_all = True
        config.training.train_fraction = 0.6  # 60% training
        config.training.validation_fraction = 0.4  # 40% validation
        config.training.test_fraction = 0.0  # No test for this test

        # Mock images
        mock_images = []
        for i in range(5):  # 5 images: 3 training (60%), 2 validation (40%)
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"image_{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        # Should create annotations for all images
        assert len(pipeline.config.annotations) == 5

        # Check categories based on train_fraction
        categories = [ann.category for ann in pipeline.config.annotations]
        assert categories.count("training") == 3  # 60% of 5 = 3
        assert categories.count("validation") == 2  # 40% of 5 = 2

    def test_prepare_processing_units_with_patches(self):
        """Test preparing processing units with patch processing."""
        config = create_default_config()
        config.spatial_coverage.use_patches = True
        config.spatial_coverage.patches_per_image = 2
        config.training.train_n = 2
        config.training.validate_n = 1
        config.training.test_n = 0  # No test images for this test
        config.training.segment_all = False

        # Mock images - create distinct images
        mock_images = []
        for i in range(5):  # Provide more images than needed so selection works
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"test_image_{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_img.getSizeY.return_value = 2000
            mock_img.getSizeX.return_value = 2000
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        # Should create 2 patches per image × 3 selected images (2 train + 1 val) = 6 annotations
        assert len(pipeline.config.annotations) == 6

        # All should be patches
        assert all(ann.is_patch for ann in pipeline.config.annotations)

        # Check categories
        categories = [ann.category for ann in pipeline.config.annotations]
        assert categories.count("training") == 4  # 2 images × 2 patches each
        assert categories.count("validation") == 2  # 1 image × 2 patches

    def test_prepare_processing_units_with_patches_too_many(self):
        """Patch processing when more patches are requested than fit in the image."""
        config = create_default_config()
        config.spatial_coverage.use_patches = True
        config.spatial_coverage.patches_per_image = 2
        config.training.train_n = 2
        config.spatial_coverage.patch_size = [512, 512]
        config.training.validate_n = 1
        config.training.test_n = 0  # No test images for this test
        config.training.segment_all = False

        # Mock images - create distinct images
        mock_images = []
        for i in range(5):  # Provide more images than needed so selection works
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"test_image_{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_img.getSizeY.return_value = 1000
            mock_img.getSizeX.return_value = 1000
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        # Should create 2 patches per image × 3 selected images (2 train + 1 val) = 6 annotations
        assert len(pipeline.config.annotations) == 3

        # All should be patches
        assert all(ann.is_patch for ann in pipeline.config.annotations)

        # Check categories
        categories = [ann.category for ann in pipeline.config.annotations]
        assert categories.count("training") == 2  # 2 images × 1 patches each
        assert categories.count("validation") == 1  # 1 image × 1 patches

    def test_prepare_processing_units_with_patches_small_image(self):
        """Patch processing when the patches are larger than the image."""
        config = create_default_config()
        config.spatial_coverage.use_patches = True
        config.spatial_coverage.patches_per_image = 2
        config.training.train_n = 2
        config.spatial_coverage.patch_size = [512, 512]
        config.training.validate_n = 1
        config.training.test_n = 0  # No test images for this test
        config.training.segment_all = False

        # Mock images - create distinct images
        mock_images = []
        for i in range(5):  # Provide more images than needed so selection works
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"test_image_{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_img.getSizeY.return_value = 400
            mock_img.getSizeX.return_value = 500
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        # Should create 2 patches per image × 3 selected images (2 train + 1 val) = 6 annotations
        assert len(pipeline.config.annotations) == 3
        assert all(ann.patch_width == 500 for ann in pipeline.config.annotations)
        assert all(ann.patch_height == 400 for ann in pipeline.config.annotations)
        # All should be patches
        assert all(ann.is_patch for ann in pipeline.config.annotations)
        # TODO actually they are not patches because patch size > image size

        # Check categories
        categories = [ann.category for ann in pipeline.config.annotations]
        assert categories.count("training") == 2  # 2 images × 1 patches each
        assert categories.count("validation") == 1  # 1 image × 1 patches

    def test_load_image_data_2d(self):
        """A 2D unit loads one (y, x) plane at its z-slice."""
        source = FakeImageSource({1: np.random.rand(1, 1, 10, 100, 100)})
        pipeline = AnnotationPipeline(
            _read_only_config(), conn=None, image_source=source
        )
        metadata = {"timepoint": 0, "z_slice": 5, "is_volumetric": False}

        result = pipeline._load_image_data(source.get_image(1), metadata)

        assert result.shape == (100, 100)
        assert len(source.reads) == 1

    def test_load_image_data_3d(self):
        """A volumetric unit loads its whole z-range as (z, y, x)."""
        source = FakeImageSource({1: np.random.rand(1, 1, 10, 100, 100)})
        pipeline = AnnotationPipeline(
            _read_only_config(), conn=None, image_source=source
        )
        metadata = {
            "timepoint": 0,
            "z_start": 0,
            "z_end": 9,
            "z_length": 10,
            "is_volumetric": True,
        }

        result = pipeline._load_image_data(source.get_image(1), metadata)

        assert result.shape == (10, 100, 100)

    def test_config_annotation_management(self):
        """Test config annotation management methods."""
        config = create_default_config()
        AnnotationPipeline(config, conn=Mock())

        # Add test annotations
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        ann1 = ImageAnnotation(image_id=1, image_name="test1")
        ann2 = ImageAnnotation(image_id=2, image_name="test2", processed=True)

        config.add_annotation(ann1)
        config.add_annotation(ann2)

        # Test getters
        unprocessed = config.get_unprocessed()
        processed = config.get_processed()

        assert len(unprocessed) == 1
        assert len(processed) == 1
        assert unprocessed[0].image_id == 1
        assert processed[0].image_id == 2

    def test_process_annotation_results_pairs_tiff_by_name_not_glob_order(
        self, tmp_path
    ):
        """seg_0000i.tif must upload to metadata[i]'s image, regardless of glob order.

        Regression: _process_annotation_results paired metadata[i] with the raw
        glob() result tiff_files[i]. glob() returns arbitrary filesystem order, so
        when the filesystem yielded files in reverse, image N received image
        (count-1-N)'s mask. Here glob is forced to return reverse order to
        reproduce that failure deterministically.
        """
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = create_default_config()
        config.workflow.read_only_mode = False
        config.spatial_coverage.use_patches = False
        config.output.output_directory = tmp_path / "set"
        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline.set_table_id(7)  # uploads need the table to exist first
        pipeline._save_original_image_for_annotation = Mock()

        # Three annotations + parallel seg files + metadata, all in order 0,1,2
        annotations_path = tmp_path / "annotations"
        annotations_path.mkdir()
        metadata = []
        for i in range(3):
            config.add_annotation(
                ImageAnnotation(
                    image_id=100 + i, image_name=f"img{i}", annotation_id=f"ann{i}"
                )
            )
            (annotations_path / f"seg_{i:05d}.tif").write_bytes(f"mask{i}".encode())
            metadata.append((f"ann{i}", {"image_id": 100 + i}, i))

        captured = []

        def fake_upload(conn, image_id, annotation_file, **kwargs):
            captured.append((image_id, Path(annotation_file).read_bytes()))
            return (image_id * 10, image_id * 100)

        # Force glob to return descending name order, a valid (worst-case)
        # filesystem ordering that the buggy positional pairing mishandles.
        real_glob = Path.glob

        def reversed_glob(self, pattern):
            return sorted(real_glob(self, pattern), reverse=True)

        with (
            patch(
                "omero_annotate_ai.core.annotation_pipeline.upload_rois_and_labels",
                side_effect=fake_upload,
            ),
            patch.object(Path, "glob", reversed_glob),
        ):
            pipeline._process_annotation_results(
                {"metadata": metadata, "annotations_path": annotations_path}
            )

        # Each seg_0000i.tif must have been uploaded to image 100+i
        assert captured == [(100, b"mask0"), (101, b"mask1"), (102, b"mask2")]
        for i in range(3):
            kept = tmp_path / "set" / "annotation_output" / f"ann{i}_mask.tif"
            assert kept.read_bytes() == f"mask{i}".encode()

    def test_three_way_split_with_counts(self):
        """Test train/validation/test split with specific counts."""
        config = create_default_config()
        config.training.train_n = 3
        config.training.validate_n = 2
        config.training.test_n = 1
        config.training.segment_all = False

        mock_images = []
        for i in range(10):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"img{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        categories = [ann.category for ann in config.annotations]
        assert categories.count("training") == 3
        assert categories.count("validation") == 2
        assert categories.count("test") == 1

    def test_three_way_split_with_fractions(self):
        """Test train/validation/test split with fractions (segment_all=True)."""
        config = create_default_config()
        config.training.train_fraction = 0.5
        config.training.validation_fraction = 0.3
        config.training.test_fraction = 0.2
        config.training.segment_all = True

        mock_images = []
        for i in range(10):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"img{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        categories = [ann.category for ann in config.annotations]
        # 50% train = 5, 30% val = 3, remaining 20% = 2
        assert categories.count("training") == 5
        assert categories.count("validation") == 3
        assert categories.count("test") == 2

    def test_zero_test_fraction_with_rounding_remainder(self):
        """Test that test_fraction=0.0 creates no test annotations even with rounding.

        Regression test for bug where n_test was calculated as remainder instead
        of using test_fraction, causing test annotations when test_fraction=0.0
        with non-divisible image counts (e.g., 11 images with 0.7/0.3 split).
        """
        config = create_default_config()
        config.training.train_fraction = 0.7
        config.training.validation_fraction = 0.3
        config.training.test_fraction = 0.0  # Explicitly no test
        config.training.segment_all = True

        # Use 11 images to trigger rounding remainder:
        # int(11 * 0.7) = 7, int(11 * 0.3) = 3, remainder = 1
        # Bug: remainder went to test instead of respecting test_fraction=0.0
        mock_images = []
        for i in range(11):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"img{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        categories = [ann.category for ann in config.annotations]
        # With test_fraction=0.0, there should be NO test annotations
        assert categories.count("test") == 0, (
            f"Expected 0 test annotations with test_fraction=0.0, got {categories.count('test')}"
        )
        # Remainder should go to training (7 + 1 = 8)
        assert categories.count("training") == 8
        assert categories.count("validation") == 3
        # Total should equal number of images
        assert len(categories) == 11

    def test_optional_validation_zero(self):
        """Test that validation can be set to 0."""
        config = create_default_config()
        config.training.train_n = 5
        config.training.validate_n = 0  # No validation
        config.training.test_n = 2
        config.training.segment_all = False

        mock_images = []
        for i in range(10):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"img{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        categories = [ann.category for ann in config.annotations]
        assert categories.count("training") == 5
        assert categories.count("validation") == 0  # No validation images
        assert categories.count("test") == 2

    def test_optional_test_zero(self):
        """Test that test can be set to 0."""
        config = create_default_config()
        config.training.train_n = 5
        config.training.validate_n = 2
        config.training.test_n = 0  # No test
        config.training.segment_all = False

        mock_images = []
        for i in range(10):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"img{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline._prepare_processing_units(mock_images)

        categories = [ann.category for ann in config.annotations]
        assert categories.count("training") == 5
        assert categories.count("validation") == 2
        assert categories.count("test") == 0  # No test images

    def test_fraction_validation_error(self):
        """Test that fractions > 1.0 raise validation error when segment_all=True."""
        from omero_annotate_ai.core.annotation_config import TrainingConfig

        with pytest.raises(ValueError, match="must sum to <= 1.0"):
            # Validation happens at construction time when segment_all=True
            TrainingConfig(
                segment_all=True,
                train_fraction=0.6,
                validation_fraction=0.3,
                test_fraction=0.3,  # Total = 1.2 > 1.0
            )


@pytest.mark.unit
class TestImageCountValidation:
    """Test that count-mode split raises an error when images are insufficient."""

    def _make_images(self, n):
        images = []
        for i in range(n):
            img = Mock()
            img.getId.return_value = i + 1
            img.getName.return_value = f"img{i + 1}"
            img.getSizeT.return_value = 1
            img.getSizeZ.return_value = 1
            images.append(img)
        return images

    def _pipeline(self, train_n, validate_n, test_n):
        config = create_default_config()
        config.training.segment_all = False
        config.training.train_n = train_n
        config.training.validate_n = validate_n
        config.training.test_n = test_n
        return AnnotationPipeline(config, conn=Mock())

    def test_exact_count_succeeds(self):
        pipeline = self._pipeline(train_n=3, validate_n=2, test_n=0)
        pipeline._prepare_processing_units(self._make_images(5))
        categories = [a.category for a in pipeline.config.annotations]
        assert len(categories) == 5
        assert categories.count("training") == 3
        assert categories.count("validation") == 2

    def test_surplus_images_succeeds(self):
        pipeline = self._pipeline(train_n=3, validate_n=2, test_n=1)
        pipeline._prepare_processing_units(self._make_images(10))
        categories = [a.category for a in pipeline.config.annotations]
        assert len(categories) == 6
        assert categories.count("training") == 3
        assert categories.count("validation") == 2
        assert categories.count("test") == 1

    def test_insufficient_raises_error(self):
        pipeline = self._pipeline(train_n=3, validate_n=2, test_n=0)
        with pytest.raises(ValueError, match="Not enough images"):
            pipeline._prepare_processing_units(self._make_images(4))

    def test_train_only_no_val_succeeds(self):
        pipeline = self._pipeline(train_n=3, validate_n=0, test_n=0)
        pipeline._prepare_processing_units(self._make_images(3))
        categories = [a.category for a in pipeline.config.annotations]
        assert len(categories) == 3
        assert all(c == "training" for c in categories)

    def test_one_image_short_raises_error(self):
        pipeline = self._pipeline(train_n=3, validate_n=2, test_n=1)
        with pytest.raises(ValueError, match="Not enough images"):
            pipeline._prepare_processing_units(self._make_images(5))

    def test_error_message_content(self):
        pipeline = self._pipeline(train_n=2, validate_n=2, test_n=0)
        with pytest.raises(ValueError) as exc_info:
            pipeline._prepare_processing_units(self._make_images(2))
        msg = str(exc_info.value)
        assert "2 training" in msg
        assert "2 validation" in msg
        assert "4" in msg
        assert "2 images available" in msg


@pytest.mark.unit
class TestFractionSync:
    """Test that count-mode runs sync train_fraction/validation_fraction after the split."""

    def _make_images(self, n):
        images = []
        for i in range(n):
            img = Mock()
            img.getId.return_value = i + 1
            img.getName.return_value = f"img{i + 1}"
            img.getSizeT.return_value = 1
            img.getSizeZ.return_value = 1
            images.append(img)
        return images

    def _pipeline(self, train_n, validate_n, test_n):
        config = create_default_config()
        config.training.segment_all = False
        config.training.train_n = train_n
        config.training.validate_n = validate_n
        config.training.test_n = test_n
        return AnnotationPipeline(config, conn=Mock())

    def test_count_mode_syncs_fractions(self):
        pipeline = self._pipeline(train_n=3, validate_n=2, test_n=0)
        pipeline._prepare_processing_units(self._make_images(10))
        assert pipeline.config.training.train_fraction == pytest.approx(0.3)
        assert pipeline.config.training.validation_fraction == pytest.approx(0.2)
        assert pipeline.config.training.test_fraction == pytest.approx(0.0)

    def test_count_mode_fractions_sum_to_one_or_less(self):
        pipeline = self._pipeline(train_n=3, validate_n=2, test_n=1)
        pipeline._prepare_processing_units(self._make_images(10))
        total = (
            pipeline.config.training.train_fraction
            + pipeline.config.training.validation_fraction
            + pipeline.config.training.test_fraction
        )
        assert total <= 1.0

    def test_count_mode_no_val_fraction_is_zero(self):
        pipeline = self._pipeline(train_n=5, validate_n=0, test_n=0)
        pipeline._prepare_processing_units(self._make_images(10))
        assert pipeline.config.training.validation_fraction == pytest.approx(0.0)
        assert pipeline.config.training.test_fraction == pytest.approx(0.0)


@pytest.mark.unit
class TestPipelineIntegration:
    """Test pipeline integration scenarios."""

    @patch("omero_annotate_ai.core.annotation_pipeline.ezomero")
    def test_run_full_microsam_workflow_basic(self, mock_ezomero):
        """Test running the full micro-SAM workflow with mocked dependencies."""
        mock_ezomero.get_image_ids.return_value = [1, 2]
        config = create_default_config()
        config.omero.container_type = "dataset"
        config.omero.container_id = 123
        config.workflow.batch_size = 2
        config.training.train_n = 1
        config.training.validate_n = 1
        mock_images = []
        for i in range(2):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"image_{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)
        mock_conn = Mock()
        mock_conn.getObject = Mock(side_effect=mock_images)
        pipeline = AnnotationPipeline(config, conn=mock_conn)
        with (
            patch.object(pipeline, "_run_micro_sam_annotation") as mock_annotate,
            patch(
                "omero_annotate_ai.core.annotation_pipeline.create_omero_tracking_table",
                return_value=456,
            ),
            patch.object(pipeline, "_process_annotation_results"),
            patch.object(pipeline, "_write_omero_table"),
            patch.object(pipeline, "_auto_save_config"),
            patch.object(pipeline, "_upload_annotation_config_to_omero"),
        ):
            mock_annotate.return_value = {
                "metadata": [
                    ("ann1", {"image_id": 1}, 0),
                    ("ann2", {"image_id": 2}, 1),
                ],
                "annotations_path": Path(tempfile.mkdtemp()),
            }
            table_id, result_config = pipeline.run_full_micro_sam_workflow(mock_images)
            assert table_id == 456
            assert result_config == config
            mock_annotate.assert_called()

    def test_run_full_workflow_resume_mode(self):
        """Test running workflow in resume mode."""
        config = create_default_config()
        config.workflow.resume_from_table = True
        config.omero.container_id = 123
        config.training.train_n = 1
        config.training.validate_n = 1
        # Create mock images for the workflow
        mock_images = []
        for i in range(2):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"image_{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_img.getSizeY.return_value = 1000
            mock_img.getSizeX.return_value = 1000
            mock_images.append(mock_img)
        pipeline = AnnotationPipeline(config, conn=Mock())
        with (
            patch.object(pipeline, "_find_existing_table", return_value=789),
            patch(
                "omero_annotate_ai.core.annotation_pipeline.sync_omero_table_to_config"
            ),
            patch.object(
                pipeline, "get_images_from_container", return_value=mock_images
            ),
            patch.object(pipeline, "_auto_save_config"),
            patch.object(
                pipeline, "_upload_annotation_config_to_omero", return_value=789
            ),
            patch(
                "omero_annotate_ai.core.annotation_pipeline.create_omero_tracking_table",
                return_value=789,
            ),
        ):
            # Patch _prepare_processing_units only for the first call
            with patch.object(pipeline, "_prepare_processing_units"):
                config.annotations = []  # Start with empty annotations for resume mode
                pipeline.initialize_workflow(mock_images)
                # No schema to define, so run_micro_sam_annotation should raise ValueError
                with pytest.raises(ValueError):
                    pipeline.run_micro_sam_annotation()
            # Now define schema and mark all as processed (do not patch _prepare_processing_units)
            pipeline.define_annotation_schema(mock_images)
            for ann in config.annotations:
                ann.processed = True
            # Should print 'All annotations already processed!' and return table_id, config
            table_id, result_config = pipeline.run_micro_sam_annotation()
            assert table_id == 789
            assert pipeline.table_id == 789
            assert result_config == config

    @pytest.mark.parametrize(
        "processed_flags,expected_unprocessed",
        [
            ([False, False, False], 3),
            ([True, True, True], 0),
            ([True, False, True], 1),
            ([], 0),
        ],
    )
    def test_resume_logic_edge_cases(
        self, processed_flags, expected_unprocessed, tmp_path
    ):
        """Test resume logic for all/none/some processed and empty schema."""
        config = create_default_config()
        config.output.output_directory = tmp_path
        config.omero.container_id = 123
        config.training.train_n = len(processed_flags)
        config.training.validate_n = 0
        # Mock images
        mock_images = []
        for i in range(len(processed_flags)):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"image_{i + 1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)
        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline.initialize_workflow(mock_images)
        pipeline.define_annotation_schema(mock_images)
        # Set processed flags
        for ann, flag in zip(config.annotations, processed_flags, strict=True):
            ann.processed = flag
        # Patch annotation runner and OMERO table creation to avoid real OMERO logic
        with (
            patch.object(pipeline, "_run_micro_sam_annotation") as mock_annotate,
            patch(
                "omero_annotate_ai.core.annotation_pipeline.create_omero_tracking_table",
                return_value=12345,
            ),
            patch(
                "omero_annotate_ai.core.annotation_pipeline.replace_table_config",
                return_value=12345,
            ),
        ):
            mock_annotate.return_value = {
                "metadata": [],
                "annotations_path": Path(tempfile.mkdtemp()),
            }
            if len(processed_flags) == 0:
                # No schema defined, should raise ValueError
                with pytest.raises(ValueError):
                    pipeline.run_micro_sam_annotation()
            elif expected_unprocessed == 0:
                # Should print 'All annotations already processed!' and return
                table_id, result_config = pipeline.run_micro_sam_annotation()
                assert table_id == pipeline.table_id
                assert result_config == config
                mock_annotate.assert_not_called()
            else:
                table_id, result_config = pipeline.run_micro_sam_annotation()
                assert table_id == pipeline.table_id
                assert result_config == config
                mock_annotate.assert_called()


@pytest.mark.unit
class TestPipelineUtils:
    """Test pipeline utility functions."""

    def test_create_pipeline_function(self):
        """Test the create_pipeline helper function."""
        from omero_annotate_ai.core.annotation_pipeline import create_pipeline

        config = create_default_config()
        config.omero.container_id = 123

        pipeline = create_pipeline(config, conn=Mock())

        assert isinstance(pipeline, AnnotationPipeline)
        assert pipeline.config == config

    def test_pipeline_config_access(self):
        """Test accessing pipeline configuration."""
        config = create_default_config()
        config.ai_model.pretrained_from = "vit_h"
        config.workflow.batch_size = 5

        pipeline = AnnotationPipeline(config, conn=Mock())

        assert pipeline.config.ai_model.pretrained_from == "vit_h"
        assert pipeline.config.workflow.batch_size == 5

    def test_pipeline_with_custom_config(self):
        """Test pipeline with custom configuration."""
        config = create_default_config()
        config.omero.container_type = "plate"
        config.omero.container_id = 999
        config.ai_model.pretrained_from = "vit_l"
        config.spatial_coverage.three_d = True
        config.name = "custom_training_set"
        config.workflow.batch_size = 5

        pipeline = AnnotationPipeline(config, conn=Mock())

        assert pipeline.config.omero.container_type == "plate"
        assert pipeline.config.spatial_coverage.three_d is True
        assert pipeline.config.name == "custom_training_set"
        pipeline = AnnotationPipeline(config, conn=Mock())

        assert pipeline.config.ai_model.pretrained_from == "vit_l"
        assert pipeline.config.workflow.batch_size == 5


@pytest.mark.unit
class TestSpatialSampling:
    """Test spatial sampling with n_slices and n_timepoints."""

    def test_z_slices_random_with_n_slices(self):
        """Test that n_slices is used for random mode when specified."""
        config = create_default_config()
        config.omero.container_id = 1
        config.spatial_coverage.z_slices = [0]  # Dummy value
        config.spatial_coverage.z_slice_mode = "random"
        config.spatial_coverage.n_slices = 3  # Should select 3 random slices

        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        # Mock image with 10 z-slices
        mock_image = Mock()
        mock_image.getId.return_value = 1
        mock_image.getSizeZ.return_value = 10

        z_slices = pipeline._get_z_slices_for_image(mock_image)

        assert len(z_slices) == 3
        assert all(0 <= z < 10 for z in z_slices)
        assert len(set(z_slices)) == 3  # All unique

    def test_z_slices_random_backward_compatibility(self):
        """Test backward compatibility: uses list length when n_slices not specified."""
        config = create_default_config()
        config.omero.container_id = 1
        config.spatial_coverage.z_slices = [0, 1, 2, 3, 4]  # Length = 5
        config.spatial_coverage.z_slice_mode = "random"
        # n_slices not specified (None)

        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        mock_image = Mock()
        mock_image.getId.return_value = 1
        mock_image.getSizeZ.return_value = 10

        z_slices = pipeline._get_z_slices_for_image(mock_image)

        assert len(z_slices) == 5  # Should use list length

    def test_timepoints_random_with_n_timepoints(self):
        """Test that n_timepoints is used for random mode when specified."""
        config = create_default_config()
        config.omero.container_id = 1
        config.spatial_coverage.timepoints = [0]  # Dummy value
        config.spatial_coverage.timepoint_mode = "random"
        config.spatial_coverage.n_timepoints = 4  # Should select 4 random timepoints

        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        mock_image = Mock()
        mock_image.getId.return_value = 1
        mock_image.getSizeT.return_value = 12

        timepoints = pipeline._get_timepoints_for_image(mock_image)

        assert len(timepoints) == 4
        assert all(0 <= t < 12 for t in timepoints)
        assert len(set(timepoints)) == 4  # All unique


@pytest.mark.unit
class TestAnnotationTimestamps:
    """Tests for annotation timestamp field handling via mark_processed() method."""

    def test_mark_processed_sets_timestamps(self):
        """Test that mark_processed sets both timestamp fields correctly."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        annotation = ImageAnnotation(image_id=1, image_name="test.tif")
        assert annotation.annotation_created_at is None
        assert annotation.annotation_updated_at is None
        assert annotation.processed is False

        annotation.mark_processed()

        assert annotation.processed is True
        assert annotation.annotation_created_at is not None
        assert annotation.annotation_updated_at is not None
        # Both should be the same on first processing
        assert annotation.annotation_created_at == annotation.annotation_updated_at

    def test_mark_processed_preserves_created_at(self):
        """Test that mark_processed preserves existing created_at timestamp."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        annotation = ImageAnnotation(
            image_id=1,
            image_name="test.tif",
            annotation_created_at="2024-01-01T00:00:00",
        )
        original_created = annotation.annotation_created_at

        annotation.mark_processed()

        assert annotation.annotation_created_at == original_created
        assert annotation.annotation_updated_at != original_created

    def test_mark_processed_with_omero_ids(self):
        """Test that mark_processed correctly sets OMERO IDs."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        annotation = ImageAnnotation(image_id=1, image_name="test.tif")

        annotation.mark_processed(
            annotation_type="custom_type",
            roi_id=123,
            label_id=456,
        )

        assert annotation.roi_id == 123
        assert annotation.label_id == 456
        assert annotation.annotation_type == "custom_type"

    def test_mark_processed_default_annotation_type(self):
        """Test that mark_processed uses default annotation_type."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        annotation = ImageAnnotation(image_id=1, image_name="test.tif")

        annotation.mark_processed()

        assert annotation.annotation_type == "segmentation_mask"

    def test_mark_processed_without_optional_ids(self):
        """Test that mark_processed does not set IDs when not provided."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        annotation = ImageAnnotation(image_id=1, image_name="test.tif")
        assert annotation.roi_id is None
        assert annotation.label_id is None

        annotation.mark_processed()

        # IDs should remain None when not provided
        assert annotation.roi_id is None
        assert annotation.label_id is None


@pytest.mark.unit
class TestContextChannelHook:
    """Context channel images loaded for each batch item."""

    def test_build_context_images_empty_when_no_context_channels(self):
        """With no context_channels, context_images should be list of empty dicts."""
        import numpy as np

        from omero_annotate_ai.core.annotation_config import create_default_config
        from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

        source = FakeImageSource({1: np.zeros((1, 1, 1, 64, 64), dtype=np.uint16)})

        config = create_default_config()
        config.spatial_coverage.channels = [0]
        # No context_channels set
        config.workflow.read_only_mode = True
        pipeline = AnnotationPipeline(config, conn=None, image_source=source)

        meta = {"timepoint": 0, "z_slice": 0, "is_volumetric": False}
        batch_data = [(source.get_image(1), "ann1", meta, 0)]

        context_images = pipeline._build_context_images(batch_data)
        assert context_images == [{}]

    def test_build_context_images_loads_correct_channel(self):
        """With context_channels=[1], loads channel 1 for each batch item."""
        import numpy as np

        from omero_annotate_ai.core.annotation_config import create_default_config
        from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

        label_arr = np.zeros((64, 64), dtype=np.uint16)
        ctx_arr = np.ones((64, 64), dtype=np.uint16) * 100
        # Channel 0 is the label channel, channel 1 the context channel
        data = np.stack([label_arr, ctx_arr])[None, :, None]  # (t, c, z, y, x)
        source = FakeImageSource({1: data})

        config = create_default_config()
        config.spatial_coverage.channels = [0, 1]
        config.spatial_coverage.context_channels = [1]
        config.spatial_coverage.context_channel_names = ["DAPI"]
        config.workflow.read_only_mode = True
        pipeline = AnnotationPipeline(config, conn=None, image_source=source)

        meta = {"timepoint": 0, "z_slice": 0, "is_volumetric": False}
        batch_data = [(source.get_image(1), "ann1", meta, 0)]

        context_images = pipeline._build_context_images(batch_data)
        assert len(context_images) == 1
        assert "DAPI" in context_images[0]
        np.testing.assert_array_equal(context_images[0]["DAPI"], ctx_arr)


@pytest.mark.unit
class TestCellposeMultiChannelSaving:
    """Test that _save_images_for_cellpose saves channels-last multi-channel TIFFs."""

    def _make_pipeline(self, training_channels, label_channel=0):
        from omero_annotate_ai.core.annotation_config import create_default_config

        config = create_default_config()
        config.spatial_coverage.channels = list(
            set([label_channel] + training_channels)
        )
        config.spatial_coverage.label_channel = label_channel
        config.spatial_coverage.training_channels = training_channels
        return AnnotationPipeline(config, conn=Mock())

    def test_single_channel_saves_as_YXC(self, tmp_path):
        """Single training channel is saved as (Y, X, 1) channels-last."""
        pipeline = self._make_pipeline(training_channels=[0])
        pipeline.config.output.output_directory = str(tmp_path)

        h, w = 32, 32
        ch0 = np.zeros((h, w), dtype=np.uint16) + 10

        mock_image = Mock()
        pipeline._load_image_data = Mock(return_value=ch0)

        stacked = pipeline._load_and_stack_channels(
            mock_image, {"timepoint": 0, "z_slice": 0}, [0]
        )
        assert stacked.shape == (h, w, 1)
        np.testing.assert_array_equal(stacked[..., 0], ch0)

    def test_multi_channel_saves_as_YXC(self, tmp_path):
        """Two training channels are stacked as (Y, X, 2)."""
        pipeline = self._make_pipeline(training_channels=[0, 1])
        pipeline.config.output.output_directory = str(tmp_path)

        h, w = 32, 32
        ch0 = np.zeros((h, w), dtype=np.uint16) + 10
        ch1 = np.zeros((h, w), dtype=np.uint16) + 20

        def load_side_effect(image_obj, meta):
            ch = meta.get("channel_override", 0)
            return ch0 if ch == 0 else ch1

        mock_image = Mock()
        pipeline._load_image_data = Mock(side_effect=load_side_effect)

        stacked = pipeline._load_and_stack_channels(
            mock_image, {"timepoint": 0, "z_slice": 0}, [0, 1]
        )
        assert stacked.shape == (h, w, 2)
        np.testing.assert_array_equal(stacked[..., 0], ch0)
        np.testing.assert_array_equal(stacked[..., 1], ch1)

    def test_3d_multi_channel_saves_as_ZYXC(self, tmp_path):
        """3D images with two channels are stacked as (Z, Y, X, 2)."""
        pipeline = self._make_pipeline(training_channels=[0, 1])

        z, h, w = 5, 32, 32
        ch0 = np.zeros((z, h, w), dtype=np.uint16) + 10
        ch1 = np.zeros((z, h, w), dtype=np.uint16) + 20

        def load_side_effect(image_obj, meta):
            ch = meta.get("channel_override", 0)
            return ch0 if ch == 0 else ch1

        mock_image = Mock()
        pipeline._load_image_data = Mock(side_effect=load_side_effect)

        stacked = pipeline._load_and_stack_channels(
            mock_image, {"timepoint": 0, "is_volumetric": True}, [0, 1]
        )
        assert stacked.shape == (z, h, w, 2)

    def test_all_training_channels_in_output(self, tmp_path):
        """Every training channel appears in the stacked output; none is dropped."""
        pipeline = self._make_pipeline(training_channels=[0, 1, 2])

        h, w = 16, 16
        arrays = {i: np.full((h, w), i * 10, dtype=np.uint16) for i in range(3)}

        def load_side_effect(image_obj, meta):
            return arrays[meta.get("channel_override", 0)]

        mock_image = Mock()
        pipeline._load_image_data = Mock(side_effect=load_side_effect)

        stacked = pipeline._load_and_stack_channels(
            mock_image, {"timepoint": 0, "z_slice": 0}, [0, 1, 2]
        )
        assert stacked.shape == (h, w, 3)
        for i in range(3):
            np.testing.assert_array_equal(stacked[..., i], arrays[i])

    def test_annotation_input_stays_single_channel(self, tmp_path):
        """annotation_input/ TIFF is still saved as (Y, X) — not stacked."""
        pipeline = self._make_pipeline(training_channels=[1], label_channel=0)
        pipeline.config.output.output_directory = str(tmp_path)

        h, w = 16, 16
        label_arr = np.zeros((h, w), dtype=np.uint16) + 5
        train_arr = np.zeros((h, w), dtype=np.uint16) + 9

        def load_side_effect(image_obj, meta):
            ch = meta.get("channel_override", 0)
            return label_arr if ch == 0 else train_arr

        mock_conn = Mock()
        mock_image = Mock()
        mock_conn.getObject = Mock(return_value=mock_image)
        pipeline.conn = mock_conn
        pipeline._load_image_data = Mock(side_effect=load_side_effect)

        meta = {"timepoint": 0, "z_slice": 0, "channel_override": 0}
        processing_units = [(1, "img_001", meta, 0)]
        pipeline._save_images_for_cellpose(processing_units)

        import tifffile

        label_file = tmp_path / "annotation_input" / "img_001.tif"
        assert label_file.exists()
        saved = tifffile.imread(str(label_file))
        assert saved.ndim == 2, (
            f"annotation_input should be 2D, got shape {saved.shape}"
        )

        train_file = tmp_path / "model_input" / "img_001.tif"
        assert train_file.exists()
        saved_train = tifffile.imread(str(train_file))
        assert saved_train.ndim == 3, (
            f"model_input should be 3D (Y,X,C), got shape {saved_train.shape}"
        )
        assert saved_train.shape == (h, w, 1)


@pytest.mark.unit
class TestAnnotationFolderNames:
    """The annotation phase uses names that say what a file is for.

    'training_input' used to mean the source brightfield channel here AND the
    training split in the training phase. Cleaning one could delete the other.
    """

    def _pipeline(self, tmp_path, separate_channels):
        from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

        config = create_default_config()
        config.output.output_directory = str(tmp_path)
        if separate_channels:
            config.spatial_coverage.channels = [0, 1]
            config.spatial_coverage.label_channel = 0
            config.spatial_coverage.training_channels = [1]
        else:
            config.spatial_coverage.channels = [0]
            # Both must be None: uses_separate_channels() only short-circuits on None,
            # and an empty training_channels list falls through to "0 not in []" -> True.
            config.spatial_coverage.label_channel = None
            config.spatial_coverage.training_channels = None
        return AnnotationPipeline(config, conn=Mock())

    def test_single_channel_folders(self, tmp_path):
        pipeline = self._pipeline(tmp_path, separate_channels=False)

        assert pipeline._get_input_folders() == {
            "annotation_input": tmp_path / "annotation_input"
        }

    def test_separate_channel_folders(self, tmp_path):
        pipeline = self._pipeline(tmp_path, separate_channels=True)

        assert pipeline._get_input_folders() == {
            "annotation_input": tmp_path / "annotation_input",
            "model_input": tmp_path / "model_input",
        }


@pytest.mark.unit
class TestMicroSamSeparateChannelSaving:
    """The micro-SAM saver must write where reorganize_local_data_for_training reads.

    The saver used to write the training channel as annotation_input/{id}_train.tif
    (the legacy flat layout). reorganize now reads model_input/{id}.tif, so every
    record was dropped and the training set came back silently empty.
    """

    def _pipeline(self, tmp_path):
        from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

        config = create_default_config()
        config.output.output_directory = str(tmp_path)
        config.spatial_coverage.channels = [0, 1]
        config.spatial_coverage.label_channel = 0
        config.spatial_coverage.training_channels = [1]

        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline.conn.getObject = Mock(return_value=Mock())

        label_arr = np.full((8, 8), 3, dtype=np.uint16)
        train_arr = np.full((8, 8), 9, dtype=np.uint16)

        def load_side_effect(image_obj, meta):
            return train_arr if meta.get("channel_override") == 1 else label_arr

        pipeline._load_image_data = Mock(side_effect=load_side_effect)
        return pipeline

    def test_training_channel_goes_to_model_input(self, tmp_path):
        pipeline = self._pipeline(tmp_path)

        pipeline._save_original_image_for_annotation(
            {"image_id": 1, "timepoint": 0, "z_slice": 0}, "img_001"
        )

        assert (tmp_path / "annotation_input" / "img_001.tif").exists()
        assert (tmp_path / "model_input" / "img_001.tif").exists()
        # The legacy suffixed file must be gone: nothing reads it any more.
        assert not (tmp_path / "annotation_input" / "img_001_train.tif").exists()

    def test_saver_output_feeds_reorganize(self, tmp_path):
        """The real producer must satisfy the real consumer.

        This is the end-to-end check that was missing: every previous test
        hand-fabricated model_input/, so the saver could drift from the reader
        without anything failing.
        """
        from omero_annotate_ai.core.annotation_config import ImageAnnotation
        from omero_annotate_ai.processing.training_functions import (
            reorganize_local_data_for_training,
        )

        pipeline = self._pipeline(tmp_path)
        pipeline._save_original_image_for_annotation(
            {"image_id": 1, "timepoint": 0, "z_slice": 0}, "img_001"
        )
        # The annotation the user would have drawn.
        import tifffile

        (tmp_path / "annotation_output").mkdir()
        tifffile.imwrite(
            str(tmp_path / "annotation_output" / "img_001_mask.tif"),
            np.ones((8, 8), dtype=np.uint8),
        )

        pipeline.config.annotations = [
            ImageAnnotation(
                image_id=1,
                image_name="img",
                annotation_id="img_001",
                category="training",
                processed=True,
            )
        ]

        result = reorganize_local_data_for_training(
            config=pipeline.config,
            annotation_dir=tmp_path,
            output_dir=tmp_path.parent / "out",
        )

        assert result["stats"]["n_missing"] == 0
        assert result["stats"]["n_training_images"] == 1
        assert (result["train_input"] / "img_001.tif").exists()
        assert (result["train_annotation_input"] / "img_001.tif").exists()


def _zarr_config(tmp_path, uris, **zarr):
    config = AnnotationConfig(
        name="zarr_run", source="zarr", zarr={"uris": uris, **zarr}
    )
    config.output.output_directory = tmp_path / "out"
    config.training.segment_all = True
    config.spatial_coverage.use_patches = False
    return config


@pytest.mark.integration
class TestZarrWorkflow:
    def test_default_source_for_zarr_config(self, tmp_path, ngff_stores):
        from omero_annotate_ai.sources import ZarrImageSource

        pipeline = AnnotationPipeline(
            _zarr_config(tmp_path, [ngff_stores["plate"]], level=1)
        )
        assert isinstance(pipeline.image_source, ZarrImageSource)
        assert pipeline.image_source.level == 1

    def test_no_images_error_names_the_uris(self, tmp_path, ngff_stores):
        config = _zarr_config(tmp_path, [ngff_stores["plate"]], wells=["H12"])
        with (
            pytest.raises(ValueError, match="plate.zarr"),
            pytest.warns(UserWarning, match="H12"),
        ):
            AnnotationPipeline(config).initialize_workflow()

    def test_end_to_end_custom_annotation(self, tmp_path, ngff_stores):
        config = _zarr_config(
            tmp_path, [ngff_stores["plate"]], level=1, wells=["A1", "B3"]
        )
        pipeline = AnnotationPipeline(config)
        seg_dir = tmp_path / "seg"

        def fake_annotate(batch):
            seg_dir.mkdir(exist_ok=True)
            metadata = []
            for i, (image_obj, ann_id, meta, row_idx) in enumerate(batch):
                plane = pipeline._load_image_data(image_obj, meta)
                tifffile.imwrite(
                    seg_dir / f"seg_{i:05d}.tif",
                    (plane > plane.mean()).astype(np.uint32),
                )
                metadata.append((ann_id, meta, row_idx))
            return {"metadata": metadata, "annotations_path": seg_dir}

        pipeline.run_custom_annotation(fake_annotate)

        # Split planning shuffles unit order, so compare in a fixed order.
        anns = sorted(pipeline.config.annotations, key=lambda a: a.well)
        assert [(a.well, a.field, a.level, a.image_id) for a in anns] == [
            ("A1", "0", 1, None),
            ("B3", "0", 1, None),
        ]
        assert anns[0].image_name == "testplate_A1_0"
        assert anns[0].annotation_id == make_annotation_id(anns[0].image_uri, 0, 0)
        out = tmp_path / "out"
        for ann in anns:
            mask = tifffile.imread(
                out / "annotation_output" / f"{ann.annotation_id}_mask.tif"
            )
            assert mask.shape == (32, 40)  # level 1 grid
            assert (out / "annotation_input" / f"{ann.annotation_id}.tif").exists()
        table = pd.read_csv(out / "tracking_table.csv")
        assert sorted(table["image_uri"]) == sorted(a.image_uri for a in anns)
        assert table["processed"].all()
        assert (out / "well_matching.csv").exists()


@pytest.mark.unit
class TestFindExistingTable:
    """Which tracking table a resumed run picks up."""

    def _pipeline(self, table_id=None):
        config = create_default_config()
        config.name = "shared_name"
        config.omero.container_type = "plate"
        config.omero.container_id = 803
        config.omero.table_id = table_id
        return AnnotationPipeline(config, conn=Mock())

    def test_uses_selected_table_id(self):
        pipeline = self._pipeline(table_id=22159)
        assert pipeline._find_existing_table() == 22159
        pipeline.conn.getObject.assert_called_once_with("FileAnnotation", 22159)

    def test_missing_table_is_an_error_not_a_new_table(self):
        """A resume must never quietly start a fresh table."""
        pipeline = self._pipeline(table_id=1)
        pipeline.conn.getObject.return_value = None
        with pytest.raises(ValueError, match="not found"):
            pipeline._find_existing_table()

    def test_without_id_is_an_error(self):
        """Tables are never looked up by name."""
        with pytest.raises(ValueError, match="table_id"):
            self._pipeline()._find_existing_table()


@pytest.mark.unit
class TestExtendOnResume:
    """A resumed run with higher image counts adds unused images to the table."""

    N_IMAGES = 10

    def _pipeline(self, n_images=N_IMAGES, train_n=2, validate_n=1, test_n=0):
        source = FakeImageSource(
            {i: np.zeros((1, 1, 1, 8, 8), np.uint16) for i in range(1, n_images + 1)}
        )
        config = create_default_config()
        config.workflow.resume_from_table = True
        config.training.train_n = train_n
        config.training.validate_n = validate_n
        config.training.test_n = test_n
        # Existing table: images 1-2 training (1 done), image 3 validation
        config.annotations.extend(
            [
                ImageAnnotation(
                    image_id=1,
                    image_name="img1",
                    annotation_id="a1",
                    channel=0,
                    category="training",
                    processed=True,
                    roi_id=5,
                    label_id=6,
                ),
                ImageAnnotation(
                    image_id=2,
                    image_name="img2",
                    annotation_id="a2",
                    channel=0,
                    category="training",
                ),
                ImageAnnotation(
                    image_id=3,
                    image_name="img3",
                    annotation_id="a3",
                    channel=0,
                    category="validation",
                ),
            ]
        )
        pipeline = AnnotationPipeline(config, conn=Mock(), image_source=source)
        pipeline.get_image_uris = lambda: [omero_uri(i) for i in range(1, n_images + 1)]
        return pipeline

    @staticmethod
    def _images(config, category):
        return {a.image_uri for a in config.annotations if a.category == category}

    def test_adds_only_unused_images_per_category(self):
        pipeline = self._pipeline()
        pipeline.config.training.train_n = 4
        pipeline.config.training.validate_n = 2
        pipeline.config.training.test_n = 1

        assert pipeline._extend_annotations() == 4

        config = pipeline.config
        assert len(self._images(config, "training")) == 4
        assert len(self._images(config, "validation")) == 2
        assert len(self._images(config, "test")) == 1
        new = {a.image_uri for a in config.annotations[3:]}
        assert not new & {omero_uri(1), omero_uri(2), omero_uri(3)}
        assert len(new) == 4

    def test_keeps_existing_rows_and_ids(self):
        pipeline = self._pipeline()
        pipeline.config.training.train_n = 5
        before = [a.model_dump() for a in pipeline.config.annotations]

        pipeline._extend_annotations()

        assert [a.model_dump() for a in pipeline.config.annotations[:3]] == before
        done = pipeline.config.annotations[0]
        assert (done.processed, done.roi_id, done.label_id) == (True, 5, 6)
        assert not any(a.processed for a in pipeline.config.annotations[3:])

    def test_adds_what_is_left_when_too_few_images(self, capsys):
        pipeline = self._pipeline(n_images=4)
        pipeline.config.training.train_n = 5

        assert pipeline._extend_annotations() == 1
        assert self._images(pipeline.config, "training") == {
            omero_uri(1),
            omero_uri(2),
            omero_uri(4),
        }
        assert "Only 1 unused images left for training" in capsys.readouterr().out

    def test_lower_counts_never_drop_rows(self):
        pipeline = self._pipeline()
        pipeline.config.training.train_n = 1
        pipeline.config.training.validate_n = 0

        assert pipeline._extend_annotations() == 0
        assert [a.annotation_id for a in pipeline.config.annotations] == [
            "a1",
            "a2",
            "a3",
        ]

    def test_schema_updates_the_table_only_when_images_were_added(self):
        pipeline = self._pipeline()
        pipeline.table_id = 22159
        with patch.object(pipeline, "_write_omero_table") as replace:
            pipeline.define_annotation_schema([])
            replace.assert_not_called()

            pipeline.config.training.train_n = 3
            pipeline.define_annotation_schema([])
            replace.assert_called_once()
        assert len(self._images(pipeline.config, "training")) == 3
