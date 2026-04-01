"""Modern tests for configuration management with updated package structure."""

import pytest
import yaml
import tempfile
from pathlib import Path
from dataclasses import asdict

from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    ImageAnnotation,
    create_default_config,
    load_config,
    get_config_template,
    validate_annotations_against_config,
    ValidationResult,
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
        # Note: processing was removed in schema v2.0.0
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

    def test_path_serialization_in_model_dump(self):
        """
        Tests that Path objects are serialized to strings in model_dump.
        This ensures OutputConfig and AnnotationConfig correctly handle
        Path to string conversion for JSON/YAML serialization.
        """
        from omero_annotate_ai.core.annotation_config import OutputConfig

        # Test OutputConfig standalone
        test_path = Path("/tmp/test_output")
        output = OutputConfig(output_directory=test_path)
        output_dict = output.model_dump()
        assert isinstance(output_dict["output_directory"], str)
        # Compare as Path objects to handle cross-platform path separators
        assert Path(output_dict["output_directory"]) == test_path

        # Test AnnotationConfig with nested OutputConfig
        config = create_default_config()
        config_path = Path("/tmp/test_config_output")
        config.output.output_directory = config_path
        config_dict = config.model_dump()
        assert isinstance(config_dict["output"]["output_directory"], str)
        # Compare as Path objects to handle cross-platform path separators
        assert Path(config_dict["output"]["output_directory"]) == config_path

        # Test to_dict (which uses model_dump)
        config_dict = config.to_dict()
        assert isinstance(config_dict["output"]["output_directory"], str)


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


@pytest.mark.unit
class TestOMEROConfigMultiContainer:
    """Test OMEROConfig multi-container fields and methods."""

    def test_get_all_container_ids_single(self):
        """Test get_all_container_ids with single container_id."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_id": 123}
        )
        assert config.omero.get_all_container_ids() == [123]

    def test_get_all_container_ids_multiple(self):
        """Test get_all_container_ids with container_ids list."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_ids": [1, 2, 3]}
        )
        assert config.omero.get_all_container_ids() == [1, 2, 3]

    def test_container_ids_precedence(self):
        """Test that container_ids takes precedence over container_id."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "dataset",
                "container_id": 999,  # Should be ignored
                "container_ids": [1, 2, 3]
            }
        )
        assert config.omero.get_all_container_ids() == [1, 2, 3]

    def test_get_all_container_ids_empty_list(self):
        """Test that empty container_ids falls back to container_id."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "dataset",
                "container_id": 456,
                "container_ids": []
            }
        )
        assert config.omero.get_all_container_ids() == [456]

    def test_get_all_container_ids_no_container(self):
        """Test get_all_container_ids with no containers configured."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_id": 0}
        )
        assert config.omero.get_all_container_ids() == []

    def test_get_primary_container_id_single(self):
        """Test get_primary_container_id with single container."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_id": 123}
        )
        assert config.omero.get_primary_container_id() == 123

    def test_get_primary_container_id_multiple(self):
        """Test get_primary_container_id with multiple containers."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_ids": [10, 20, 30]}
        )
        assert config.omero.get_primary_container_id() == 10

    def test_get_primary_container_id_empty(self):
        """Test get_primary_container_id with no containers."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_id": 0}
        )
        assert config.omero.get_primary_container_id() == 0

    def test_is_multi_container_false_single(self):
        """Test is_multi_container returns False for single container."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_id": 123}
        )
        assert config.omero.is_multi_container() is False

    def test_is_multi_container_false_one_in_list(self):
        """Test is_multi_container returns False for single item in list."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_ids": [123]}
        )
        assert config.omero.is_multi_container() is False

    def test_is_multi_container_true(self):
        """Test is_multi_container returns True for multiple containers."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_ids": [1, 2]}
        )
        assert config.omero.is_multi_container() is True

    def test_yaml_roundtrip_with_container_ids(self):
        """Test that container_ids survives YAML roundtrip."""
        config = AnnotationConfig(
            name="test_multi",
            omero={
                "container_type": "plate",
                "container_ids": [1, 2, 3]
            }
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            loaded = AnnotationConfig.from_yaml(f.name)

            assert loaded.omero.container_ids == [1, 2, 3]
            assert loaded.omero.get_all_container_ids() == [1, 2, 3]

        Path(f.name).unlink()

    def test_backward_compatible_single_container_yaml(self):
        """Test backward compatibility with existing single container configs."""
        yaml_str = """
        name: test
        omero:
          container_type: dataset
          container_id: 456
        """
        config = AnnotationConfig.from_yaml(yaml_str)
        assert config.omero.container_id == 456
        assert config.omero.get_all_container_ids() == [456]


@pytest.mark.unit
class TestAnnotationValidation:
    """Tests for validate_annotations_against_config()."""

    def _make_config(self, **spatial_kwargs):
        """Helper: default config with customisable spatial coverage fields."""
        config = create_default_config()
        for k, v in spatial_kwargs.items():
            setattr(config.spatial_coverage, k, v)
        return config

    def _make_ann(self, **kwargs):
        """Helper: ImageAnnotation with sensible defaults."""
        defaults = dict(
            image_id=1,
            image_name="test.tif",
            channel=0,
            timepoint=0,
            z_slice=0,
            is_volumetric=False,
            is_patch=False,
            category="training",
        )
        defaults.update(kwargs)
        return ImageAnnotation(**defaults)

    # ---- Valid cases ----

    def test_valid_2d_annotations_pass(self):
        """Consistent 2D annotations produce no errors and no warnings."""
        config = self._make_config(channels=[0], label_channel=0, three_d=False, use_patches=False)
        config.training.segment_all = True
        config.training.train_fraction = 0.7
        config.training.validation_fraction = 0.3
        config.annotations = [
            self._make_ann(image_id=i, channel=0, is_volumetric=False, is_patch=False,
                           category="training" if i < 7 else "validation")
            for i in range(10)
        ]
        result = validate_annotations_against_config(config)
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_valid_3d_annotations_pass(self):
        """Consistent 3D annotations produce no errors."""
        config = self._make_config(channels=[0], label_channel=0, three_d=True, use_patches=False)
        config.annotations = [
            self._make_ann(channel=0, is_volumetric=True, is_patch=False)
        ]
        result = validate_annotations_against_config(config)
        assert result.is_valid
        assert result.errors == []

    def test_valid_patch_annotations_pass(self):
        """Consistent patch annotations produce no errors."""
        config = self._make_config(
            channels=[0], label_channel=0, three_d=False,
            use_patches=True, patch_size=[512, 512]
        )
        config.annotations = [
            self._make_ann(channel=0, is_patch=True, patch_width=512, patch_height=512)
        ]
        result = validate_annotations_against_config(config)
        assert result.is_valid
        assert result.errors == []

    # ---- Error cases ----

    def test_channel_mismatch_is_error(self):
        """Annotations with wrong channel produce an error."""
        config = self._make_config(channels=[0, 1], label_channel=1)
        config.annotations = [self._make_ann(channel=0)]  # should be 1
        result = validate_annotations_against_config(config)
        assert not result.is_valid
        assert any(e.field == "channel" for e in result.errors)

    def test_volumetric_flag_mismatch_is_error(self):
        """Annotations with wrong is_volumetric produce an error."""
        config = self._make_config(channels=[0], three_d=True)
        config.annotations = [self._make_ann(is_volumetric=False)]  # should be True
        result = validate_annotations_against_config(config)
        assert not result.is_valid
        assert any(e.field == "is_volumetric" for e in result.errors)

    def test_patch_flag_mismatch_is_error(self):
        """Annotations with wrong is_patch produce an error."""
        config = self._make_config(channels=[0], use_patches=True, patch_size=[512, 512])
        config.annotations = [self._make_ann(is_patch=False)]  # should be True
        result = validate_annotations_against_config(config)
        assert not result.is_valid
        assert any(e.field == "is_patch" for e in result.errors)

    # ---- Warning cases ----

    def test_patch_size_mismatch_is_warning(self):
        """Patch dimension differences produce a warning, not an error."""
        config = self._make_config(channels=[0], use_patches=True, patch_size=[512, 512])
        config.annotations = [
            self._make_ann(is_patch=True, patch_width=256, patch_height=256)
        ]
        result = validate_annotations_against_config(config)
        assert result.is_valid  # warning only, not an error
        assert any(w.field == "patch_size" for w in result.warnings)

    def test_category_count_mismatch_is_warning(self):
        """Wrong annotation counts vs train_n/validate_n produce a warning."""
        config = create_default_config()
        config.training.segment_all = False
        config.training.train_n = 3
        config.training.validate_n = 2
        config.spatial_coverage.channels = [0]
        config.annotations = [
            self._make_ann(image_id=i, category="training") for i in range(5)
        ]  # 5 training, 0 validation — doesn't match train_n=3, validate_n=2
        result = validate_annotations_against_config(config)
        assert result.is_valid
        assert any(w.field == "category_counts" for w in result.warnings)

    def test_category_fraction_mismatch_is_warning(self):
        """A training fraction far from configured train_fraction produces a warning."""
        config = create_default_config()
        config.training.segment_all = True
        config.training.train_fraction = 0.7
        config.training.validation_fraction = 0.3
        config.spatial_coverage.channels = [0]
        # Only 2 training out of 10 → ratio = 0.2, far from 0.7
        config.annotations = (
            [self._make_ann(image_id=i, category="training") for i in range(2)]
            + [self._make_ann(image_id=i + 10, category="validation") for i in range(8)]
        )
        result = validate_annotations_against_config(config)
        assert result.is_valid
        assert any(w.field == "category_fractions" for w in result.warnings)

    # ---- is_valid semantics ----

    def test_is_valid_false_when_errors_present(self):
        """is_valid is False when the errors list is non-empty."""
        config = self._make_config(channels=[0, 1], label_channel=1)
        config.annotations = [self._make_ann(channel=0)]
        result = validate_annotations_against_config(config)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_is_valid_true_when_only_warnings(self):
        """is_valid is True even when warnings are present."""
        config = create_default_config()
        config.training.segment_all = False
        config.training.train_n = 3
        config.training.validate_n = 0
        config.spatial_coverage.channels = [0]
        config.annotations = [self._make_ann(image_id=i, category="training") for i in range(5)]
        result = validate_annotations_against_config(config)
        assert result.is_valid
        assert len(result.warnings) > 0

    def test_summary_valid(self):
        """summary property returns OK string for a passing result."""
        config = self._make_config(channels=[0], label_channel=0)
        config.annotations = [self._make_ann(channel=0)]
        result = validate_annotations_against_config(config)
        assert result.summary.startswith("OK:")

    def test_summary_invalid(self):
        """summary property returns INVALID string when errors exist."""
        config = self._make_config(channels=[0, 1], label_channel=1)
        config.annotations = [self._make_ann(channel=0)]
        result = validate_annotations_against_config(config)
        assert result.summary.startswith("INVALID:")

    def test_define_annotation_schema_raises_on_validation_error(self):
        """define_annotation_schema raises ValueError for invalid reused annotations."""
        from unittest.mock import Mock
        from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

        config = self._make_config(channels=[0, 1], label_channel=1)
        config.name = "test"
        config.output.output_directory = "/tmp/test_validation"
        config.annotations = [self._make_ann(channel=0)]  # wrong channel

        mock_conn = Mock()
        mock_conn.isConnected.return_value = True
        pipeline = AnnotationPipeline(config, mock_conn)

        with pytest.raises(ValueError, match="inconsistent with current config"):
            pipeline.define_annotation_schema(images_list=[])


@pytest.mark.unit
class TestChannelPresentation:
    """Tests for ChannelPresentation model."""

    def test_create_with_required_fields(self):
        from omero_annotate_ai.core.annotation_config import ChannelPresentation
        cp = ChannelPresentation(channel_index=0, contrast_start=100.0, contrast_end=4500.0)
        assert cp.channel_index == 0
        assert cp.visible is True
        assert cp.contrast_start == 100.0
        assert cp.contrast_end == 4500.0
        assert cp.color == "#FFFFFF"

    def test_create_with_all_fields(self):
        from omero_annotate_ai.core.annotation_config import ChannelPresentation
        cp = ChannelPresentation(
            channel_index=1, visible=False,
            contrast_start=0.0, contrast_end=255.0, color="#00FF00"
        )
        assert cp.visible is False
        assert cp.color == "#00FF00"

    def test_serialization_round_trip(self):
        from omero_annotate_ai.core.annotation_config import ChannelPresentation
        cp = ChannelPresentation(channel_index=0, contrast_start=100.0, contrast_end=4500.0, color="#FF0000")
        data = cp.model_dump()
        cp2 = ChannelPresentation(**data)
        assert cp == cp2


@pytest.mark.unit
class TestFeatureType:
    """Tests for FeatureType model."""

    def test_create(self):
        from omero_annotate_ai.core.annotation_config import FeatureType
        ft = FeatureType(name="cell", color="#FF0000")
        assert ft.name == "cell"
        assert ft.color == "#FF0000"

    def test_serialization_round_trip(self):
        from omero_annotate_ai.core.annotation_config import FeatureType
        ft = FeatureType(name="nucleus", color="#00FF00")
        data = ft.model_dump()
        ft2 = FeatureType(**data)
        assert ft == ft2
