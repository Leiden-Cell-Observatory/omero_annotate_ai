"""Tests for the annotation pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from omero_annotate_ai.core.pipeline import AnnotationPipeline
from omero_annotate_ai.core.config import create_default_config


class TestAnnotationPipeline:
    """Test the main annotation pipeline class."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with config."""
        config = create_default_config()
        config.omero.container_id = 123
        
        pipeline = AnnotationPipeline(config, conn=None)
        
        assert pipeline.config == config
        assert pipeline.conn is None
        assert pipeline.table_id is None
    
    def test_pipeline_initialization_with_connection(self):
        """Test pipeline initialization with mock connection."""
        config = create_default_config()
        mock_conn = Mock()
        
        pipeline = AnnotationPipeline(config, conn=mock_conn)
        
        assert pipeline.config == config
        assert pipeline.conn == mock_conn
    
    def test_pipeline_validation_success(self):
        """Test successful pipeline validation."""
        config = create_default_config()
        config.omero.container_id = 123
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        # Should not raise exception
        pipeline._validate_config()
    
    def test_pipeline_validation_no_connection(self):
        """Test pipeline validation without connection."""
        config = create_default_config()
        config.omero.container_id = 123
        
        pipeline = AnnotationPipeline(config, conn=None)
        
        with pytest.raises(ValueError, match="OMERO connection is required"):
            pipeline._validate_config()
    
    def test_pipeline_validation_invalid_config(self):
        """Test pipeline validation with invalid config."""
        config = create_default_config()
        config.omero.container_id = -1  # Invalid ID
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        with pytest.raises(ValueError):
            pipeline._validate_config()
    
    @patch('omero_annotate_ai.core.pipeline.get_image_ids')
    def test_get_image_ids_dataset(self, mock_get_image_ids):
        """Test getting image IDs from dataset."""
        mock_get_image_ids.return_value = [1, 2, 3]
        
        config = create_default_config()
        config.omero.container_type = "dataset"
        config.omero.container_id = 123
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        image_ids = pipeline._get_image_ids()
        
        assert image_ids == [1, 2, 3]
        mock_get_image_ids.assert_called_once()
    
    @patch('omero_annotate_ai.core.pipeline.get_image_ids')
    def test_get_image_ids_single_image(self, mock_get_image_ids):
        """Test getting single image ID."""
        config = create_default_config()
        config.omero.container_type = "image"
        config.omero.container_id = 456
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        image_ids = pipeline._get_image_ids()
        
        assert image_ids == [456]
        mock_get_image_ids.assert_not_called()
    
    @patch('omero_annotate_ai.core.pipeline.ezomero')
    def test_get_image_ids_project(self, mock_ezomero):
        """Test getting image IDs from project (via datasets)."""
        # Mock dataset IDs in project
        mock_ezomero.get_dataset_ids.return_value = [10, 20]
        # Mock image IDs in each dataset
        mock_ezomero.get_image_ids.side_effect = [
            [1, 2],    # Images in dataset 10
            [3, 4, 5]  # Images in dataset 20
        ]
        
        config = create_default_config()
        config.omero.container_type = "project"
        config.omero.container_id = 789
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        image_ids = pipeline._get_image_ids()
        
        assert image_ids == [1, 2, 3, 4, 5]
        mock_ezomero.get_dataset_ids.assert_called_once_with(Mock(), project=789)
        assert mock_ezomero.get_image_ids.call_count == 2
    
    @patch('omero_annotate_ai.core.pipeline.initialize_tracking_table')
    @patch('omero_annotate_ai.core.pipeline._prepare_processing_units')
    def test_setup_tracking_table(self, mock_prepare_units, mock_init_table):
        """Test tracking table setup."""
        mock_prepare_units.return_value = [
            (1, 0, {"category": "training"}),
            (2, 1, {"category": "validation"})
        ]
        mock_init_table.return_value = 123
        
        config = create_default_config()
        config.omero.container_id = 456
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        with patch.object(pipeline, '_get_image_ids', return_value=[1, 2]):
            table_id = pipeline._setup_tracking_table()
        
        assert table_id == 123
        assert pipeline.table_id == 123
        mock_init_table.assert_called_once()
    
    @patch('omero_annotate_ai.core.pipeline.get_unprocessed_units')
    def test_get_unprocessed_units(self, mock_get_unprocessed):
        """Test getting unprocessed units from tracking table."""
        mock_get_unprocessed.return_value = [
            (1, 0, {"category": "training"}),
            (3, 2, {"category": "validation"})
        ]
        
        config = create_default_config()
        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline.table_id = 123
        
        units = pipeline._get_unprocessed_units()
        
        assert len(units) == 2
        assert units[0][0] == 1
        assert units[1][0] == 3
        mock_get_unprocessed.assert_called_once_with(Mock(), 123)
    
    @patch('omero_annotate_ai.core.pipeline.get_dask_image_multiple')
    def test_load_images_for_batch(self, mock_get_dask):
        """Test loading images for batch processing."""
        import numpy as np
        
        # Mock image loading
        mock_images = [np.random.rand(10, 100, 100) for _ in range(3)]
        mock_get_dask.return_value = mock_images
        
        config = create_default_config()
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        # Mock image objects
        mock_image_objs = [Mock() for _ in range(3)]
        for i, obj in enumerate(mock_image_objs):
            obj.getId.return_value = i + 1
        
        with patch.object(pipeline.conn, 'getObject', side_effect=mock_image_objs):
            images = pipeline._load_images_for_batch([1, 2, 3])
        
        assert len(images) == 3
        assert all(isinstance(img, np.ndarray) for img in images)
    
    def test_create_batch_info(self):
        """Test creating batch information."""
        processing_units = [
            (1, 0, {"category": "training"}),
            (2, 1, {"category": "validation"}),
            (3, 2, {"category": "training"})
        ]
        
        config = create_default_config()
        config.batch_processing.batch_size = 2
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        batches = pipeline._create_batch_info(processing_units)
        
        # Should create 2 batches: [0,1] and [2]
        assert len(batches) == 2
        assert len(batches[0]) == 2  # First batch has 2 items
        assert len(batches[1]) == 1  # Second batch has 1 item
    
    def test_create_batch_info_batch_size_zero(self):
        """Test creating batch info with batch_size=0 (all in one batch)."""
        processing_units = [
            (1, 0, {"category": "training"}),
            (2, 1, {"category": "validation"}),
            (3, 2, {"category": "training"}),
            (4, 3, {"category": "validation"})
        ]
        
        config = create_default_config()
        config.batch_processing.batch_size = 0  # Process all in one batch
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        batches = pipeline._create_batch_info(processing_units)
        
        # Should create 1 batch with all items
        assert len(batches) == 1
        assert len(batches[0]) == 4
    
    @patch('omero_annotate_ai.core.pipeline.image_series_annotator')
    def test_run_microsam_annotation(self, mock_annotator):
        """Test running micro-SAM annotation."""
        import numpy as np
        
        # Mock successful annotation
        mock_annotator.return_value = None
        
        config = create_default_config()
        config.microsam.model_type = "vit_b_lm"
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        # Mock images and metadata
        images = [np.random.rand(100, 100) for _ in range(2)]
        batch_metadata = [
            {"image_id": 1, "output_path": "output1.tif"},
            {"image_id": 2, "output_path": "output2.tif"}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.batch_processing.output_folder = temp_dir
            
            pipeline._run_microsam_annotation(images, batch_metadata)
            
            mock_annotator.assert_called_once()
            call_args = mock_annotator.call_args
            
            # Check that correct parameters were passed
            assert call_args[1]['model_type'] == "vit_b_lm"
            assert call_args[1]['output_folder'] == temp_dir
    
    @patch('omero_annotate_ai.core.pipeline.upload_rois_and_labels')
    @patch('omero_annotate_ai.core.pipeline.update_tracking_table_rows')
    def test_upload_and_update_results(self, mock_update_table, mock_upload):
        """Test uploading results and updating tracking table."""
        # Mock successful upload
        mock_upload.return_value = (789, 456)  # label_id, roi_id
        
        config = create_default_config()
        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline.table_id = 123
        
        batch_metadata = [
            {"image_id": 1, "output_path": "output1.tif", "row_index": 0},
            {"image_id": 2, "output_path": "output2.tif", "row_index": 1}
        ]
        
        pipeline._upload_and_update_results(batch_metadata)
        
        # Should upload for each image
        assert mock_upload.call_count == 2
        
        # Should update table once for the batch
        mock_update_table.assert_called_once_with(
            Mock(), 123, [0, 1], "completed", mock_upload.return_value
        )
    
    def test_process_batch_complete_workflow(self):
        """Test complete batch processing workflow."""
        config = create_default_config()
        config.omero.container_id = 123
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        pipeline.table_id = 456
        
        batch_units = [
            (1, 0, {"category": "training"}),
            (2, 1, {"category": "validation"})
        ]
        
        with patch.object(pipeline, '_load_images_for_batch') as mock_load, \
             patch.object(pipeline, '_run_microsam_annotation') as mock_annotate, \
             patch.object(pipeline, '_upload_and_update_results') as mock_upload:
            
            mock_load.return_value = [Mock(), Mock()]  # Mock images
            
            pipeline._process_batch(batch_units, 0)
            
            mock_load.assert_called_once()
            mock_annotate.assert_called_once()
            mock_upload.assert_called_once()


class TestPipelineIntegration:
    """Test pipeline integration scenarios."""
    
    @patch('omero_annotate_ai.core.pipeline.ezomero')
    def test_run_full_workflow_basic(self, mock_ezomero):
        """Test running the full workflow with mocked dependencies."""
        # Setup mocks
        mock_ezomero.get_image_ids.return_value = [1, 2, 3]
        
        config = create_default_config()
        config.omero.container_type = "dataset"
        config.omero.container_id = 123
        config.batch_processing.batch_size = 2
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        with patch.object(pipeline, '_setup_tracking_table', return_value=456), \
             patch.object(pipeline, '_get_unprocessed_units', return_value=[
                 (1, 0, {"category": "training"}),
                 (2, 1, {"category": "validation"})
             ]), \
             patch.object(pipeline, '_process_batch') as mock_process:
            
            pipeline.run_full_workflow()
            
            # Should process one batch
            mock_process.assert_called_once()
    
    def test_run_full_workflow_resume_mode(self):
        """Test running workflow in resume mode."""
        config = create_default_config()
        config.workflow.resume_from_table = True
        config.omero.container_id = 123
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        # Mock existing table
        with patch.object(pipeline, '_find_existing_table', return_value=789), \
             patch.object(pipeline, '_get_unprocessed_units', return_value=[]), \
             patch.object(pipeline, '_setup_tracking_table') as mock_setup:
            
            pipeline.run_full_workflow()
            
            # Should not create new table in resume mode
            mock_setup.assert_not_called()
            assert pipeline.table_id == 789
    
    def test_run_full_workflow_no_unprocessed_units(self):
        """Test workflow when no unprocessed units remain."""
        config = create_default_config()
        config.omero.container_id = 123
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        with patch.object(pipeline, '_setup_tracking_table', return_value=456), \
             patch.object(pipeline, '_get_unprocessed_units', return_value=[]), \
             patch.object(pipeline, '_process_batch') as mock_process:
            
            pipeline.run_full_workflow()
            
            # Should not process any batches
            mock_process.assert_not_called()
    
    @patch('omero_annotate_ai.core.pipeline.ezomero')
    def test_run_full_workflow_error_handling(self, mock_ezomero):
        """Test workflow error handling."""
        config = create_default_config()
        config.omero.container_id = 123
        
        # Mock error in image ID retrieval
        mock_ezomero.get_image_ids.side_effect = Exception("OMERO connection failed")
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        with pytest.raises(Exception, match="OMERO connection failed"):
            pipeline.run_full_workflow()


class TestPipelineUtils:
    """Test pipeline utility functions."""
    
    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        config = create_default_config()
        config.omero.container_id = 123
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        repr_str = repr(pipeline)
        assert "AnnotationPipeline" in repr_str
        assert "container_id=123" in repr_str
    
    def test_pipeline_config_access(self):
        """Test accessing pipeline configuration."""
        config = create_default_config()
        config.microsam.model_type = "vit_h"
        config.batch_processing.batch_size = 5
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        assert pipeline.config.microsam.model_type == "vit_h"
        assert pipeline.config.batch_processing.batch_size == 5
    
    def test_pipeline_with_custom_config(self):
        """Test pipeline with custom configuration."""
        config = create_default_config()
        config.omero.container_type = "plate"
        config.omero.container_id = 999
        config.microsam.model_type = "vit_l"
        config.microsam.three_d = True
        config.training.trainingset_name = "custom_training_set"
        
        pipeline = AnnotationPipeline(config, conn=Mock())
        
        assert pipeline.config.omero.container_type == "plate"
        assert pipeline.config.microsam.three_d is True
        assert pipeline.config.training.trainingset_name == "custom_training_set"