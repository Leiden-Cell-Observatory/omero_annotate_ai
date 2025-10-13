"""Tests for the annotation pipeline."""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline
from omero_annotate_ai.core.annotation_config import create_default_config


@pytest.mark.unit
class TestAnnotationPipeline:
    """Test the main annotation pipeline class."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with config."""
        config = create_default_config()
        config.omero.container_id = 123
        
        # Should raise ValueError when no connection provided
        with pytest.raises(ValueError, match="OMERO connection is required"):
            pipeline = AnnotationPipeline(config, conn=None)
    
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
        assert config.processing.batch_size >= 0
    
    @patch('omero_annotate_ai.core.annotation_pipeline.ezomero')
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

    @patch('omero_annotate_ai.core.annotation_pipeline.ezomero')
    def test_get_images_from_container_project(self, mock_ezomero):
        """Test getting images from project (via datasets)."""
        mock_ezomero.get_dataset_ids.return_value = [10, 20]
        mock_ezomero.get_image_ids.side_effect = [
            [1, 2],    # Images in dataset 10
            [3, 4, 5]  # Images in dataset 20
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
    
    def test_setup_directories(self):
        """Test directory setup."""
        config = create_default_config()
        config.output.output_directory = Path(tempfile.mkdtemp())
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        output_path = pipeline._setup_directories()
        
        assert output_path.exists()
        assert (output_path / "embed").exists()
        assert (output_path / "annotations").exists()
    
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
        3. The table title used in create_or_replace_tracking_table matches config.name
        """
        from omero_annotate_ai.omero.omero_functions import generate_unique_table_name

        # Setup mock connection and container
        mock_conn = Mock()
        mock_container = Mock()
        mock_container.getName.return_value = "TestDataset"
        mock_conn.getObject.return_value = mock_container

        # Test 1: generate_unique_table_name with custom name
        unique_name = generate_unique_table_name(
            mock_conn,
            "Dataset",
            123,
            base_name="my_custom_name"
        )
        assert "my_custom_name" in unique_name

        # Test 2: generate_unique_table_name uses container name when no custom name
        unique_name_no_custom = generate_unique_table_name(
            mock_conn,
            "Dataset",
            123
        )
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

        # Test 4: Verify create_or_replace_tracking_table receives the exact config name
        mock_images = []
        for i in range(2):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"image_{i+1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)

        pipeline.define_annotation_schema(mock_images)

        with patch('omero_annotate_ai.core.annotation_pipeline.create_or_replace_tracking_table') as mock_create_table:
            mock_create_table.return_value = 999
            pipeline.create_tracking_table()

            # Verify the table was created with the exact name from config
            mock_create_table.assert_called_once()
            call_args = mock_create_table.call_args
            table_title_arg = call_args.kwargs.get('table_title')

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
            mock_img.getName.return_value = f"image_{i+1}"
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
            mock_img.getName.return_value = f"image_{i+1}"
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
        config.processing.use_patches = True
        config.processing.patches_per_image = 2
        config.training.train_n = 2
        config.training.validate_n = 1
        config.training.test_n = 0  # No test images for this test
        config.training.segment_all = False
        
        # Mock images - create distinct images
        mock_images = []
        for i in range(5):  # Provide more images than needed so selection works
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"test_image_{i+1}"
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
        """Test preparing processing units with patch processing. Test when patches too many patches to fit the image."""
        config = create_default_config()
        config.processing.use_patches = True
        config.processing.patches_per_image = 2
        config.training.train_n = 2
        config.processing.patch_size = [512, 512]
        config.training.validate_n = 1
        config.training.test_n = 0  # No test images for this test
        config.training.segment_all = False
        
        # Mock images - create distinct images
        mock_images = []
        for i in range(5):  # Provide more images than needed so selection works
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"test_image_{i+1}"
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
        """Test preparing processing units with patch processing. Test when patches are larger than image."""
        config = create_default_config()
        config.processing.use_patches = True
        config.processing.patches_per_image = 2
        config.training.train_n = 2
        config.processing.patch_size = [512, 512]
        config.training.validate_n = 1
        config.training.test_n = 0  # No test images for this test
        config.training.segment_all = False
        
        # Mock images - create distinct images
        mock_images = []
        for i in range(5):  # Provide more images than needed so selection works
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"test_image_{i+1}"
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
        #TODO actually they are not patches because patch size > image size
        
        # Check categories
        categories = [ann.category for ann in pipeline.config.annotations]
        assert categories.count("training") == 2  # 2 images × 1 patches each
        assert categories.count("validation") == 1  # 1 image × 1 patches
    
    @patch('omero_annotate_ai.core.annotation_pipeline.get_dask_image_single')
    def test_load_image_data_2d_with_single_function(self, mock_get_dask_single):
        """Test loading 2D image data using get_dask_image_single."""
        import numpy as np

        mock_image_data = np.random.rand(100, 100)
        mock_get_dask_single.return_value = mock_image_data

        config = create_default_config()
        pipeline = AnnotationPipeline(config, conn=Mock())

        mock_image = Mock()
        metadata = {
            "timepoint": 0,
            "z_slice": 5,
            "is_volumetric": False
        }

        result = pipeline._load_image_data(mock_image, metadata)

        assert result.shape == (100, 100)
        mock_get_dask_single.assert_called_once()

    @patch('omero_annotate_ai.core.annotation_pipeline.get_dask_image_single')
    def test_load_image_data_3d_with_single_function(self, mock_get_dask_single):
        """Test loading 3D volumetric image data using get_dask_image_single."""
        import numpy as np

        mock_image_data = np.random.rand(10, 100, 100)  # z, y, x
        mock_get_dask_single.return_value = mock_image_data

        config = create_default_config()
        pipeline = AnnotationPipeline(config, conn=Mock())

        mock_image = Mock()
        metadata = {
            "timepoint": 0,
            "z_start": 0,
            "z_end": 9,
            "z_length": 10,
            "is_volumetric": True
        }

        result = pipeline._load_image_data(mock_image, metadata)

        assert result.shape == (10, 100, 100)
        mock_get_dask_single.assert_called_once()

    # Keep the old test names for backward compatibility but mark them as aliases
    test_load_image_data_2d = test_load_image_data_2d_with_single_function
    test_load_image_data_3d = test_load_image_data_3d_with_single_function
    
    def test_config_annotation_management(self):
        """Test config annotation management methods."""
        config = create_default_config()
        pipeline = AnnotationPipeline(config, conn=Mock())
        
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
            mock_img.getName.return_value = f"img{i+1}"
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
            mock_img.getName.return_value = f"img{i+1}"
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
            mock_img.getName.return_value = f"img{i+1}"
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
            mock_img.getName.return_value = f"img{i+1}"
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
        """Test that fractions > 1.0 raise validation error."""
        config = create_default_config()
        with pytest.raises(ValueError, match="must sum to ≤ 1.0"):
            config.training.train_fraction = 0.6
            config.training.validation_fraction = 0.3
            config.training.test_fraction = 0.3  # Total = 1.2 > 1.0
            # Trigger validation
            config.training.validate_splits()


@pytest.mark.unit
class TestPipelineIntegration:
    """Test pipeline integration scenarios."""
    
    @patch('omero_annotate_ai.core.annotation_pipeline.ezomero')
    def test_run_full_microsam_workflow_basic(self, mock_ezomero):
        """Test running the full micro-SAM workflow with mocked dependencies."""
        mock_ezomero.get_image_ids.return_value = [1, 2]
        config = create_default_config()
        config.omero.container_type = "dataset"
        config.omero.container_id = 123
        config.processing.batch_size = 2
        config.training.train_n = 1
        config.training.validate_n = 1
        mock_images = []
        for i in range(2):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"image_{i+1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)
        mock_conn = Mock()
        mock_conn.getObject = Mock(side_effect=mock_images)
        pipeline = AnnotationPipeline(config, conn=mock_conn)
        with patch.object(pipeline, '_run_micro_sam_annotation') as mock_annotate, \
            patch('omero_annotate_ai.core.annotation_pipeline.create_or_replace_tracking_table', return_value=456), \
            patch.object(pipeline, '_process_annotation_results'), \
            patch.object(pipeline, '_replace_omero_table_from_config'), \
            patch.object(pipeline, '_auto_save_config'), \
            patch.object(pipeline, '_upload_annotation_config_to_omero'):
            mock_annotate.return_value = {
                "metadata": [("ann1", {"image_id": 1}, 0), ("ann2", {"image_id": 2}, 1)],
                "annotations_path": Path(tempfile.mkdtemp())
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
            mock_img.getName.return_value = f"image_{i+1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_img.getSizeY.return_value = 1000
            mock_img.getSizeX.return_value = 1000
            mock_images.append(mock_img)
        pipeline = AnnotationPipeline(config, conn=Mock())
        with patch.object(pipeline, '_find_existing_table', return_value=789), \
            patch('omero_annotate_ai.core.annotation_pipeline.sync_omero_table_to_config'), \
            patch.object(pipeline, 'get_images_from_container', return_value=mock_images), \
            patch.object(pipeline, '_setup_directories', return_value=Path(tempfile.mkdtemp())), \
            patch.object(pipeline, '_cleanup_embeddings'), \
            patch.object(pipeline, '_auto_save_config'), \
            patch.object(pipeline, '_upload_annotation_config_to_omero', return_value=789), \
            patch('omero_annotate_ai.core.annotation_pipeline.create_or_replace_tracking_table', return_value=789):
            # Patch _prepare_processing_units only for the first call
            with patch.object(pipeline, '_prepare_processing_units'):
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

    @pytest.mark.parametrize("processed_flags,expected_unprocessed", [
        ([False, False, False], 3),
        ([True, True, True], 0),
        ([True, False, True], 1),
        ([], 0),
    ])
    def test_resume_logic_edge_cases(self, processed_flags, expected_unprocessed):
        """Test resume logic for all/none/some processed and empty schema."""
        config = create_default_config()
        config.omero.container_id = 123
        config.training.train_n = len(processed_flags)
        config.training.validate_n = 0
        # Mock images
        mock_images = []
        for i in range(len(processed_flags)):
            mock_img = Mock()
            mock_img.getId.return_value = i + 1
            mock_img.getName.return_value = f"image_{i+1}"
            mock_img.getSizeT.return_value = 1
            mock_img.getSizeZ.return_value = 1
            mock_images.append(mock_img)
        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline.initialize_workflow(mock_images)
        pipeline.define_annotation_schema(mock_images)
        # Set processed flags
        for ann, flag in zip(config.annotations, processed_flags):
            ann.processed = flag
        # Patch annotation runner and OMERO table creation to avoid real OMERO logic
        with patch.object(pipeline, '_run_micro_sam_annotation') as mock_annotate, \
            patch('omero_annotate_ai.core.annotation_pipeline.create_or_replace_tracking_table', return_value=12345), \
            patch('omero_annotate_ai.core.annotation_pipeline.upload_annotation_config_to_omero', return_value=12345):
            mock_annotate.return_value = {"metadata": [], "annotations_path": Path(tempfile.mkdtemp())}
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
        config.ai_model.model_type = "vit_h"
        config.processing.batch_size = 5
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        assert pipeline.config.ai_model.model_type == "vit_h"
        assert pipeline.config.processing.batch_size == 5
    
    def test_pipeline_with_custom_config(self):
        """Test pipeline with custom configuration."""
        config = create_default_config()
        config.omero.container_type = "plate"
        config.omero.container_id = 999
        config.ai_model.model_type = "vit_l"
        config.spatial_coverage.three_d = True
        config.name = "custom_training_set"
        config.processing.batch_size = 5
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        assert pipeline.config.omero.container_type == "plate"
        assert pipeline.config.spatial_coverage.three_d is True
        assert pipeline.config.name == "custom_training_set"
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        assert pipeline.config.ai_model.model_type == "vit_l"
        assert pipeline.config.processing.batch_size == 5
