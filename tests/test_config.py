"""Modern tests for configuration management with updated package structure."""

import pytest
import yaml
import tempfile
from pathlib import Path
from dataclasses import asdict

from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    create_default_config,
    load_config,
    get_config_template
)


@pytest.mark.unit
class TestAnnotationConfig:
    """Test the AnnotationConfig class with modern structure."""
    
    def test_default_config_creation(self):
        """
        Tests the creation of a default configuration object.
        This test ensures that the `create_default_config` function returns a valid
        `AnnotationConfig` object with the expected default values.
        """
        config = create_default_config()
        assert isinstance(config, AnnotationConfig)
        assert config.omero.container_type == "dataset"
    
    def test_config_to_dict(self):
        """
        Tests the conversion of a configuration object to a dictionary.
        This test ensures that the `to_dict` method correctly converts the `AnnotationConfig`
        object to a dictionary with the expected keys.
        """
        config = create_default_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "omero" in config_dict
        assert "training" in config_dict
        assert "workflow" in config_dict
    
    def test_config_to_yaml(self):
        """
        Tests the conversion of a configuration object to a YAML string.
        This test ensures that the `to_yaml` method correctly converts the `AnnotationConfig`
        object to a valid YAML string with the expected content.
        """
        config = create_default_config()
        yaml_str = config.to_yaml()
        
        assert isinstance(yaml_str, str)
        assert "omero:" in yaml_str
        
        # Test that it's valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)

    def test_yaml_key_order_deterministic(self):
        """
        Ensure YAML serialization preserves a deterministic, schema-defined key order.
        """
        config = create_default_config()
        yaml_str = config.to_yaml()

        # Extract the first-level keys order from the YAML text
        lines = [ln for ln in yaml_str.splitlines() if ln and not ln.startswith(' ') and ':' in ln]
        keys_in_yaml = [ln.split(':', 1)[0] for ln in lines]

        # Expected order follows field declaration order of AnnotationConfig
        expected_prefix_order = [
            'schema_version',
            'config_file_path',
            'name',
            'version',
            'authors',
            'created',
            'study',
            'dataset',
            'annotation_methodology',
            'spatial_coverage',
            'training',
            'ai_model',
            'processing',
            'workflow',
            'output',
            'omero',
            'annotations',
            'documentation',
            'repository',
            'tags',
        ]

        # Only compare until we reach a non-top-level sequence scalar line
        assert keys_in_yaml[: len(expected_prefix_order)] == expected_prefix_order
    
    def test_config_from_dict(self):
        """
        Tests the creation of a configuration object from a dictionary.
        This test ensures that the `from_dict` method correctly creates an `AnnotationConfig`
        object from a dictionary with the expected values.
        """
        config_dict = {
            "name": "test",
            "omero": {"container_type": "plate", "container_id": 123},
        }
        
        config = AnnotationConfig.from_dict(config_dict)
        
        assert config.omero.container_type == "plate"
        assert config.omero.container_id == 123
    
    def test_config_from_yaml_string(self):
        """
        Tests the creation of a configuration object from a YAML string.
        This test ensures that the `from_yaml` method correctly creates an `AnnotationConfig`
        object from a YAML string with the expected values.
        """
        yaml_str = """
        name: test
        omero:
          container_type: project
          container_id: 456
        """
        
        config = AnnotationConfig.from_yaml(yaml_str)
        
        assert config.omero.container_type == "project"
        assert config.omero.container_id == 456
    
    def test_config_from_yaml_file(self):
        """
        Tests the creation of a configuration object from a YAML file.
        This test ensures that the `from_yaml` method correctly creates an `AnnotationConfig`
        object from a YAML file with the expected values.
        """
        yaml_content = """
        name: test
        training:
          trainingset_name: "test_set"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = AnnotationConfig.from_yaml(f.name)
            
            assert config.name == "test"
        
        # Clean up
        Path(f.name).unlink()
    
    def test_config_structure(self):
        """
        Tests the overall structure and key parameters of the configuration.
        This test ensures that the default configuration has the expected structure
        and that the key parameters have the correct default values.
        """
        config = create_default_config()
        config.omero.container_id = 123
        
        # Test key configuration values
        assert config.omero.container_type == "dataset"
        assert config.omero.container_id == 123
    
    def test_load_config_from_dict(self):
        """
        Tests the `load_config` function with a dictionary as input.
        This test ensures that the `load_config` function correctly creates an
        `AnnotationConfig` object from a dictionary.
        """
        config_dict = {"name": "test", "omero": {"container_id": 999}}
        config = load_config(config_dict)
        
        assert isinstance(config, AnnotationConfig)
        assert config.omero.container_id == 999
    
    def test_get_config_template(self):
        """
        Tests the `get_config_template` function.
        This test ensures that the `get_config_template` function returns a valid
        YAML template with the expected content.
        """
        template = get_config_template()
        
        assert isinstance(template, str)
        assert "name:" in template
        
        # Test that template is valid YAML
        parsed = yaml.safe_load(template)
        assert isinstance(parsed, dict)


@pytest.mark.unit
class TestConfigEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_config_source(self):
        """
        Tests the `load_config` function with an invalid source.
        This test ensures that the `load_config` function raises a `ValueError`
        when it is called with an invalid source type.
        """
        with pytest.raises(ValueError, match="config_source must be"):
            load_config(123)  # Invalid type
    
    def test_config_save_and_load_roundtrip(self):
        """
        Tests that saving and loading a configuration preserves all data.
        This test ensures that a configuration object can be saved to a YAML file
        and then loaded back without any loss of data.
        """
        config = create_default_config()
        config.omero.container_id = 999
        config.name = "test_roundtrip"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)

            # Load it back
            loaded_config = AnnotationConfig.from_yaml(f.name)

            assert loaded_config.omero.container_id == 999
            assert loaded_config.name == "test_roundtrip"

        # Clean up
        Path(f.name).unlink()


@pytest.mark.unit
class TestMultiChannelSupport:
    """Test multi-channel support with separate label and training channels."""

    def test_get_label_channel_default(self):
        """Test that get_label_channel() defaults to primary_channel (channels[0])."""
        config = create_default_config()
        config.spatial_coverage.channels = [0, 1, 2]

        # No label_channel set, should default to channels[0]
        assert config.spatial_coverage.get_label_channel() == 0
        assert config.spatial_coverage.get_label_channel() == config.spatial_coverage.primary_channel

    def test_get_label_channel_explicit(self):
        """Test that explicit label_channel is used when set."""
        config = create_default_config()
        config.spatial_coverage.channels = [0, 1, 2]
        config.spatial_coverage.label_channel = 1

        assert config.spatial_coverage.get_label_channel() == 1

    def test_get_training_channels_default(self):
        """Test that get_training_channels() defaults to [get_label_channel()]."""
        config = create_default_config()
        config.spatial_coverage.channels = [0, 1, 2]

        # No training_channels set, should default to [get_label_channel()] = [0]
        assert config.spatial_coverage.get_training_channels() == [0]

    def test_get_training_channels_explicit(self):
        """Test that explicit training_channels is used when set."""
        config = create_default_config()
        config.spatial_coverage.channels = [0, 1, 2]
        config.spatial_coverage.training_channels = [1, 2]

        assert config.spatial_coverage.get_training_channels() == [1, 2]

    def test_uses_separate_channels_false_by_default(self):
        """Test that uses_separate_channels() is False when neither field is configured."""
        config = create_default_config()
        config.spatial_coverage.channels = [0, 1]

        assert config.spatial_coverage.uses_separate_channels() is False

    def test_uses_separate_channels_true_when_different(self):
        """Test that uses_separate_channels() is True when label and training channels differ."""
        config = create_default_config()
        config.spatial_coverage.channels = [0, 1]
        config.spatial_coverage.label_channel = 0
        config.spatial_coverage.training_channels = [1]

        assert config.spatial_coverage.uses_separate_channels() is True

    def test_uses_separate_channels_false_when_same(self):
        """Test that uses_separate_channels() is False when label channel is in training channels."""
        config = create_default_config()
        config.spatial_coverage.channels = [0, 1]
        config.spatial_coverage.label_channel = 0
        config.spatial_coverage.training_channels = [0, 1]

        assert config.spatial_coverage.uses_separate_channels() is False


@pytest.mark.unit
class TestMultiChannelValidation:
    """Test validation of channel configuration."""

    def test_label_channel_must_be_in_channels(self):
        """Test that label_channel must be in channels list."""
        from omero_annotate_ai.core.annotation_config import SpatialCoverage

        with pytest.raises(ValueError, match="label_channel"):
            SpatialCoverage(
                channels=[0, 1],
                label_channel=2,  # Not in channels list
                timepoints=[0],
                z_slices=[0]
            )

    def test_training_channels_must_be_in_channels(self):
        """Test that training_channels must be in channels list."""
        from omero_annotate_ai.core.annotation_config import SpatialCoverage

        with pytest.raises(ValueError, match="training_channel"):
            SpatialCoverage(
                channels=[0, 1],
                training_channels=[2],  # Not in channels list
                timepoints=[0],
                z_slices=[0]
            )

    def test_valid_channel_configuration(self):
        """Test that valid channel configuration passes validation."""
        from omero_annotate_ai.core.annotation_config import SpatialCoverage

        coverage = SpatialCoverage(
            channels=[0, 1, 2],
            label_channel=0,
            training_channels=[1, 2],
            timepoints=[0],
            z_slices=[0]
        )
        assert coverage.label_channel == 0
        assert coverage.training_channels == [1, 2]


@pytest.mark.unit
class TestMultiChannelYamlSerialization:
    """Test YAML serialization with channel fields."""

    def test_yaml_roundtrip_with_channels(self):
        """Test that channel fields survive YAML roundtrip."""
        config = create_default_config()
        config.spatial_coverage.channels = [0, 1, 2]
        config.spatial_coverage.label_channel = 0
        config.spatial_coverage.training_channels = [1, 2]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            loaded = AnnotationConfig.from_yaml(f.name)

            assert loaded.spatial_coverage.channels == [0, 1, 2]
            assert loaded.spatial_coverage.label_channel == 0
            assert loaded.spatial_coverage.training_channels == [1, 2]

        Path(f.name).unlink()

    def test_yaml_backward_compatibility(self):
        """Test that old YAML configs without new fields still work."""
        yaml_str = """
        name: test
        spatial_coverage:
          channels: [0]
          timepoints: [0]
          z_slices: [0]
        """

        config = AnnotationConfig.from_yaml(yaml_str)

        # New methods should work with defaults
        assert config.spatial_coverage.get_label_channel() == 0
        assert config.spatial_coverage.get_training_channels() == [0]
        assert config.spatial_coverage.uses_separate_channels() is False

    def test_channels_without_explicit_roles(self):
        """Test that channels list without explicit roles uses channels[0] for both."""
        yaml_str = """
        name: test
        spatial_coverage:
          channels: [0, 1]
          timepoints: [0]
          z_slices: [0]
        """

        config = AnnotationConfig.from_yaml(yaml_str)

        # Should use channels[0] for both label and training
        assert config.spatial_coverage.get_label_channel() == 0
        assert config.spatial_coverage.get_training_channels() == [0]
        assert config.spatial_coverage.uses_separate_channels() is False
