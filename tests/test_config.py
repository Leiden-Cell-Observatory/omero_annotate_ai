"""Modern tests for configuration management with updated package structure."""

import pytest
import yaml
import tempfile
from pathlib import Path

from omero_annotate_ai.core.config import (
    AnnotationConfig,
    BatchProcessingConfig,
    OMEROConfig,
    MicroSAMConfig,
    PatchConfig,
    TrainingConfig,
    WorkflowConfig,
    create_default_config,
    load_config,
    get_config_template
)


class TestAnnotationConfig:
    """Test the AnnotationConfig class with modern structure."""
    
    def test_default_config_creation(self):
        """Test creating a default configuration."""
        config = create_default_config()
        assert isinstance(config, AnnotationConfig)
        assert config.batch_processing.batch_size == 0  # New default
        assert config.omero.container_type == "dataset"
        assert config.microsam.model_type == "vit_b_lm"  # New default
        assert config.training.trainingset_name == "default_training_set"
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = create_default_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "batch_processing" in config_dict
        assert "omero" in config_dict
        assert "microsam" in config_dict  # Updated from image_processing
        assert "patches" in config_dict
        assert "training" in config_dict
        assert "workflow" in config_dict
    
    def test_config_to_yaml(self):
        """Test converting configuration to YAML."""
        config = create_default_config()
        yaml_str = config.to_yaml()
        
        assert isinstance(yaml_str, str)
        assert "batch_processing:" in yaml_str
        assert "omero:" in yaml_str
        assert "microsam:" in yaml_str  # Updated name
        assert "model_type: vit_b_lm" in yaml_str  # New default
        assert "batch_size: 0" in yaml_str  # New default
        
        # Test that it's valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
    
    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "batch_processing": {"batch_size": 5},
            "omero": {"container_type": "plate", "container_id": 123},
            "microsam": {"model_type": "vit_h", "three_d": True}
        }
        
        config = AnnotationConfig.from_dict(config_dict)
        
        assert config.batch_processing.batch_size == 5
        assert config.omero.container_type == "plate"
        assert config.omero.container_id == 123
        assert config.microsam.model_type == "vit_h"
        assert config.microsam.three_d is True
    
    def test_backward_compatibility_ai_model(self):
        """Test backward compatibility with old ai_model config."""
        config_dict = {
            "ai_model": {
                "model_type": "vit_l",
                "timepoints": [1, 2],
                "three_d": True
            }
        }
        
        config = AnnotationConfig.from_dict(config_dict)
        
        # Should be mapped to microsam config
        assert config.microsam.model_type == "vit_l"
        assert config.microsam.timepoints == [1, 2]
        assert config.microsam.three_d is True
    
    def test_backward_compatibility_image_processing(self):
        """Test backward compatibility with old image_processing config."""
        config_dict = {
            "image_processing": {
                "model_type": "vit_b",
                "z_slices": [0, 1, 2]
            }
        }
        
        config = AnnotationConfig.from_dict(config_dict)
        
        # Should be mapped to microsam config
        assert config.microsam.model_type == "vit_b"
        assert config.microsam.z_slices == [0, 1, 2]
    
    def test_config_from_yaml_string(self):
        """Test creating configuration from YAML string."""
        yaml_str = """
        batch_processing:
          batch_size: 0
        omero:
          container_type: project
          container_id: 456
        microsam:
          model_type: vit_b_lm
        """
        
        config = AnnotationConfig.from_yaml(yaml_str)
        
        assert config.batch_processing.batch_size == 0
        assert config.omero.container_type == "project"
        assert config.omero.container_id == 456
        assert config.microsam.model_type == "vit_b_lm"
    
    def test_config_from_yaml_file(self):
        """Test creating configuration from YAML file."""
        yaml_content = """
        batch_processing:
          batch_size: 2
        training:
          trainingset_name: "test_set"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = AnnotationConfig.from_yaml(f.name)
            
            assert config.batch_processing.batch_size == 2
            assert config.training.trainingset_name == "test_set"
        
        # Clean up
        Path(f.name).unlink()
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = create_default_config()
        config.omero.container_id = 123  # Set required field
        
        # Should not raise exception
        config.validate()
    
    def test_batch_size_zero_validation(self):
        """Test that batch_size=0 is valid (new behavior)."""
        config = create_default_config()
        config.omero.container_id = 123
        config.batch_processing.batch_size = 0  # Should be valid
        
        # Should not raise exception
        config.validate()
    
    def test_config_validation_failures(self):
        """Test configuration validation failures."""
        config = create_default_config()
        config.omero.container_id = 123  # Set valid container_id first
        
        # Test invalid batch size (negative)
        config.batch_processing.batch_size = -1
        with pytest.raises(ValueError, match="batch_size must be non-negative"):
            config.validate()
        
        # Reset and test invalid container type
        config = create_default_config()
        config.omero.container_id = 123
        config.omero.container_type = "invalid"
        with pytest.raises(ValueError, match="container_type must be one of"):
            config.validate()
        
        # Test invalid model type
        config = create_default_config()
        config.omero.container_id = 123
        config.microsam.model_type = "invalid_model"
        with pytest.raises(ValueError, match="model_type must be one of"):
            config.validate()
    
    def test_group_by_image_removed(self):
        """Test that group_by_image parameter has been removed."""
        config = create_default_config()
        
        # Should not have group_by_image attribute
        assert not hasattr(config.training, 'group_by_image')
        
        # Should not appear in legacy params
        legacy_params = config.get_legacy_params()
        assert 'group_by_image' not in legacy_params
    
    def test_training_set_name_required(self):
        """Test that training set name is included."""
        config = create_default_config()
        
        assert config.training.trainingset_name == "default_training_set"
        
        # Should appear in YAML
        yaml_str = config.to_yaml()
        assert "trainingset_name: default_training_set" in yaml_str
    
    def test_get_legacy_params(self):
        """Test conversion to legacy parameters."""
        config = create_default_config()
        config.omero.container_id = 123
        
        legacy_params = config.get_legacy_params()
        
        assert isinstance(legacy_params, dict)
        assert legacy_params["batch_size"] == 0  # Updated default
        assert legacy_params["container_type"] == "dataset"
        assert legacy_params["container_id"] == 123
        assert legacy_params["model_type"] == "vit_b_lm"  # Updated default
        assert legacy_params["use_patches"] is False
        assert legacy_params["trainingset_name"] == "default_training_set"
        
        # Ensure removed parameter is not present
        assert 'group_by_image' not in legacy_params
    
    def test_load_config_from_dict(self):
        """Test load_config function with dictionary."""
        config_dict = {"omero": {"container_id": 999}}
        config = load_config(config_dict)
        
        assert isinstance(config, AnnotationConfig)
        assert config.omero.container_id == 999
    
    def test_get_config_template(self):
        """Test getting configuration template."""
        template = get_config_template()
        
        assert isinstance(template, str)
        assert "batch_processing:" in template
        assert "model_type: \"vit_b_lm\"" in template  # New default
        assert "batch_size: 0" in template  # New default
        assert "trainingset_name:" in template  # Required field
        
        # Test that template is valid YAML
        parsed = yaml.safe_load(template)
        assert isinstance(parsed, dict)
    
    def test_microsam_params(self):
        """Test micro-SAM specific parameters extraction."""
        config = create_default_config()
        config.batch_processing.output_folder = "./test_output"
        
        microsam_params = config.get_microsam_params()
        
        assert microsam_params["model_type"] == "vit_b_lm"
        assert microsam_params["embedding_path"] == "./test_output/embed"
        assert microsam_params["is_volumetric"] is False


class TestConfigSubclasses:
    """Test individual configuration dataclasses."""
    
    def test_batch_processing_config(self):
        """Test BatchProcessingConfig defaults and behavior."""
        config = BatchProcessingConfig()
        
        assert config.batch_size == 0  # New default
        assert config.output_folder == "./output"
        
        # Test with custom values
        config = BatchProcessingConfig(batch_size=5, output_folder="./custom")
        assert config.batch_size == 5
        assert config.output_folder == "./custom"
    
    def test_microsam_config(self):
        """Test MicroSAMConfig defaults and behavior."""
        config = MicroSAMConfig()
        
        assert config.model_type == "vit_b_lm"  # New default
        assert config.timepoints == [0]
        assert config.z_slices == [0]
        assert config.three_d is False
        assert config.timepoint_mode == "specific"
        assert config.z_slice_mode == "specific"
    
    def test_training_config(self):
        """Test TrainingConfig defaults and behavior."""
        config = TrainingConfig()
        
        assert config.segment_all is True
        assert config.train_n == 3
        assert config.validate_n == 3
        assert config.trainingset_name == "default_training_set"
        
        # Ensure group_by_image is not present
        assert not hasattr(config, 'group_by_image')


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_patch_size_conversion(self):
        """Test patch size conversion from list to tuple."""
        config_dict = {
            "patches": {
                "patch_size": [256, 256]  # List instead of tuple
            }
        }
        
        config = AnnotationConfig.from_dict(config_dict)
        assert config.patches.patch_size == (256, 256)
        assert isinstance(config.patches.patch_size, tuple)
    
    def test_trainingset_name_persistence(self):
        """Test handling of trainingset_name in various scenarios."""
        # Test with None value (should use default)
        config_dict = {"training": {"trainingset_name": None}}
        config = AnnotationConfig.from_dict(config_dict)
        assert config.training.trainingset_name is None
        
        # Test with custom string value
        config_dict = {"training": {"trainingset_name": "my_custom_set"}}
        config = AnnotationConfig.from_dict(config_dict)
        assert config.training.trainingset_name == "my_custom_set"
    
    def test_invalid_config_source(self):
        """Test load_config with invalid source."""
        with pytest.raises(ValueError, match="config_source must be"):
            load_config(123)  # Invalid type
    
    def test_config_save_and_load_roundtrip(self):
        """Test saving and loading configuration preserves all data."""
        config = create_default_config()
        config.omero.container_id = 999
        config.microsam.model_type = "vit_h"
        config.training.trainingset_name = "test_roundtrip"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            
            # Load it back
            loaded_config = AnnotationConfig.from_yaml(f.name)
            
            assert loaded_config.omero.container_id == 999
            assert loaded_config.microsam.model_type == "vit_h"
            assert loaded_config.training.trainingset_name == "test_roundtrip"
            assert loaded_config.batch_processing.batch_size == 0  # Default preserved
        
        # Clean up
        Path(f.name).unlink()