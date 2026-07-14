"""Modern tests for configuration management with updated package structure."""

import pytest
import yaml
import tempfile
import pandas as pd
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
class TestContextChannelConfig:
    """Tests for context_channels fields on SpatialCoverage."""

    def test_defaults_empty(self):
        from omero_annotate_ai.core.annotation_config import SpatialCoverage
        sc = SpatialCoverage(channels=[0, 1, 2], timepoints=[0], z_slices=[0])
        assert sc.context_channels == []
        assert sc.context_channel_names is None
        assert sc.context_channel_colormaps is None
        assert sc.get_context_channel_specs() == []

    def test_uses_context_channels_false_by_default(self):
        from omero_annotate_ai.core.annotation_config import SpatialCoverage
        sc = SpatialCoverage(channels=[0], timepoints=[0], z_slices=[0])
        assert sc.uses_context_channels() is False

    def test_uses_context_channels_true_when_set(self):
        from omero_annotate_ai.core.annotation_config import SpatialCoverage
        sc = SpatialCoverage(channels=[0, 1], timepoints=[0], z_slices=[0],
                             context_channels=[1])
        assert sc.uses_context_channels() is True

    def test_get_context_channel_specs_defaults(self):
        from omero_annotate_ai.core.annotation_config import SpatialCoverage
        sc = SpatialCoverage(channels=[0, 1, 2], timepoints=[0], z_slices=[0],
                             context_channels=[1, 2])
        specs = sc.get_context_channel_specs()
        assert len(specs) == 2
        ch_idx0, name0, cmap0 = specs[0]
        ch_idx1, name1, cmap1 = specs[1]
        assert ch_idx0 == 1
        assert ch_idx1 == 2
        assert name0 == "ch1"
        assert name1 == "ch2"
        assert cmap0 == "blue"
        assert cmap1 == "magenta"

    def test_get_context_channel_specs_custom_names(self):
        from omero_annotate_ai.core.annotation_config import SpatialCoverage
        sc = SpatialCoverage(channels=[0, 1], timepoints=[0], z_slices=[0],
                             context_channels=[1],
                             context_channel_names=["DAPI"])
        specs = sc.get_context_channel_specs()
        assert specs[0][1] == "DAPI"

    def test_get_context_channel_specs_custom_colormaps(self):
        from omero_annotate_ai.core.annotation_config import SpatialCoverage
        sc = SpatialCoverage(channels=[0, 1], timepoints=[0], z_slices=[0],
                             context_channels=[1],
                             context_channel_colormaps=["magenta"])
        specs = sc.get_context_channel_specs()
        assert specs[0][2] == "magenta"

    def test_validation_rejects_overlap_with_label_channel(self):
        from omero_annotate_ai.core.annotation_config import SpatialCoverage
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="context_channels"):
            SpatialCoverage(channels=[0, 1], timepoints=[0], z_slices=[0],
                            label_channel=0, context_channels=[0])

    def test_validation_rejects_mismatched_names_length(self):
        from omero_annotate_ai.core.annotation_config import SpatialCoverage
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SpatialCoverage(channels=[0, 1, 2], timepoints=[0], z_slices=[0],
                            context_channels=[1, 2],
                            context_channel_names=["DAPI"])  # length 1, should be 2

    def test_validation_rejects_mismatched_colormaps_length(self):
        from omero_annotate_ai.core.annotation_config import SpatialCoverage
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SpatialCoverage(channels=[0, 1, 2], timepoints=[0], z_slices=[0],
                            context_channels=[1, 2],
                            context_channel_colormaps=["blue"])  # length 1, should be 2

    def test_yaml_round_trip(self):
        from omero_annotate_ai.core.annotation_config import AnnotationConfig
        config = AnnotationConfig(name="test")
        config.spatial_coverage.channels = [0, 1]
        config.spatial_coverage.context_channels = [1]
        config.spatial_coverage.context_channel_names = ["DAPI"]
        config.spatial_coverage.context_channel_colormaps = ["blue"]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            loaded = AnnotationConfig.from_yaml(f.name)
        Path(f.name).unlink()
        assert loaded.spatial_coverage.context_channels == [1]
        assert loaded.spatial_coverage.context_channel_names == ["DAPI"]
        assert loaded.spatial_coverage.context_channel_colormaps == ["blue"]


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


@pytest.mark.unit
class TestAnnotationConfigJSON:
    """Tests for JSON serialization of AnnotationConfig."""

    def _make_config(self):
        from omero_annotate_ai.core.annotation_config import (
            AnnotationConfig, FeatureType, ChannelPresentation,
            StudyContext, DatasetInfo, AnnotationMethodology,
            SpatialCoverage, TrainingConfig, AIModelConfig,
            WorkflowConfig, OutputConfig, OMEROConfig, ImageAnnotation,
        )
        config = AnnotationConfig(
            name="test_workflow",
            study=StudyContext(title="Test", description="Test study"),
            dataset=DatasetInfo(source_description="test"),
            annotation_methodology=AnnotationMethodology(annotation_criteria="test"),
            spatial_coverage=SpatialCoverage(channels=[0], timepoints=[0], z_slices=[0]),
            training=TrainingConfig(),
            ai_model=AIModelConfig(),
            workflow=WorkflowConfig(),
            output=OutputConfig(),
            omero=OMEROConfig(container_type="dataset", container_id=1),
            feature_types=[FeatureType(name="cell", color="#FF0000")],
        )
        ann = ImageAnnotation(image_id=1, image_name="img1")
        ann.channel_presentation = [
            ChannelPresentation(channel_index=0, contrast_start=100, contrast_end=4500, color="#00FF00")
        ]
        config.annotations.append(ann)
        return config

    def test_to_json_returns_valid_json(self):
        import json
        config = self._make_config()
        json_str = config.to_json()
        data = json.loads(json_str)
        assert data["name"] == "test_workflow"
        assert len(data["feature_types"]) == 1
        assert data["feature_types"][0]["name"] == "cell"

    def test_from_json_string(self):
        config = self._make_config()
        json_str = config.to_json()
        loaded = type(config).from_json(json_str)
        assert loaded.name == config.name
        assert len(loaded.feature_types) == 1
        assert loaded.feature_types[0].name == "cell"
        assert len(loaded.annotations) == 1
        assert loaded.annotations[0].channel_presentation[0].contrast_start == 100

    def test_from_json_dict(self):
        import json
        config = self._make_config()
        data = json.loads(config.to_json())
        loaded = type(config).from_json(data)
        assert loaded.name == config.name

    def test_from_json_file(self):
        import tempfile
        from pathlib import Path
        config = self._make_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(config.to_json())
            f.flush()
            loaded = type(config).from_json(Path(f.name))
        assert loaded.name == config.name

    def test_round_trip_preserves_all_fields(self):
        config = self._make_config()
        json_str = config.to_json()
        loaded = type(config).from_json(json_str)
        assert config.to_dict() == loaded.to_dict()


@pytest.mark.unit
class TestTrainingConfigWarnings:
    """Test that TrainingConfig warns when the inactive split set has non-default values."""

    def test_warns_when_fractions_set_but_count_mode(self):
        from omero_annotate_ai.core.annotation_config import TrainingConfig
        with pytest.warns(UserWarning, match="train_fraction.*ignored.*segment_all=False"):
            TrainingConfig(segment_all=False, train_n=3, validate_n=2, test_n=0, train_fraction=0.5)

    def test_warns_when_validation_fraction_set_but_count_mode(self):
        from omero_annotate_ai.core.annotation_config import TrainingConfig
        with pytest.warns(UserWarning, match="ignored.*segment_all=False"):
            TrainingConfig(segment_all=False, train_n=3, validate_n=2, test_n=0, validation_fraction=0.1)

    def test_warns_when_counts_set_but_fraction_mode(self):
        from omero_annotate_ai.core.annotation_config import TrainingConfig
        with pytest.warns(UserWarning, match="train_n.*ignored.*segment_all=True"):
            TrainingConfig(segment_all=True, train_n=10, validate_n=2, test_n=0)

    def test_no_warning_when_only_active_count_set_changed(self):
        from omero_annotate_ai.core.annotation_config import TrainingConfig
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TrainingConfig(segment_all=False, train_n=5, validate_n=2, test_n=0)

    def test_no_warning_when_only_active_fraction_set_changed(self):
        from omero_annotate_ai.core.annotation_config import TrainingConfig
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TrainingConfig(
                segment_all=True,
                train_fraction=0.6,
                validation_fraction=0.4,
                test_fraction=0.0,
            )


@pytest.mark.unit
class TestStudyContextFundingStatement:
    """The StudyContext gains an optional funding_statement field for MIFA export."""

    def test_funding_statement_defaults_to_none(self):
        from omero_annotate_ai.core.annotation_config import StudyContext
        study = StudyContext(title="t", description="d")
        assert study.funding_statement is None

    def test_funding_statement_is_settable(self):
        from omero_annotate_ai.core.annotation_config import StudyContext
        study = StudyContext(title="t", description="d", funding_statement="Funded by X")
        assert study.funding_statement == "Funded by X"

    def test_funding_statement_round_trips_through_yaml(self, tmp_path):
        config = create_default_config()
        config.study.funding_statement = "Grant ABC-123"
        yaml_path = tmp_path / "config.yaml"
        config.save_yaml(yaml_path)
        reloaded = AnnotationConfig.from_yaml(yaml_path)
        assert reloaded.study.funding_statement == "Grant ABC-123"


@pytest.mark.unit
class TestMIFAExportHelpers:
    """Pure mapping helpers in mifa_export (no upstream package needed)."""

    def test_map_license(self):
        from omero_annotate_ai.core.mifa_export import map_license
        assert map_license("CC-BY-4.0") == "CC_BY"
        assert map_license("CC-BY") == "CC_BY"
        assert map_license("CC_BY") == "CC_BY"
        assert map_license("CC0") == "CC0"
        assert map_license("CC0-1.0") == "CC0"

    def test_map_license_unknown_falls_back_with_warning(self):
        from omero_annotate_ai.core.mifa_export import map_license
        with pytest.warns(UserWarning, match="license"):
            assert map_license("MIT") == "CC_BY"

    @pytest.mark.parametrize(
        "ours,mifa",
        [
            ("segmentation_mask", "segmentation_mask"),
            ("semantic_segmentation", "segmentation_mask"),
            ("bounding_box", "bounding_boxes"),
            ("point", "point_annotations"),
            ("classification", "class_labels"),
        ],
    )
    def test_map_annotation_type(self, ours, mifa):
        from omero_annotate_ai.core.mifa_export import map_annotation_type
        assert map_annotation_type(ours) == mifa

    def test_map_annotation_type_unknown_falls_back_to_other(self):
        from omero_annotate_ai.core.mifa_export import map_annotation_type
        with pytest.warns(UserWarning, match="annotation type"):
            assert map_annotation_type("nonsense") == "other"

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Jane Doe", ("Jane", "Doe")),
            ("Jane", ("Jane", ".")),
            ("Jane van der Berg", ("Jane", "van der Berg")),
            ("  Marie  Curie  ", ("Marie", "Curie")),
            ("", None),
            ("   ", None),
            (None, None),
        ],
    )
    def test_split_author_name(self, name, expected):
        from omero_annotate_ai.core.mifa_export import split_author_name
        assert split_author_name(name) == expected

    def test_compose_overview_is_nonempty_and_mentions_name(self):
        from omero_annotate_ai.core.mifa_export import compose_overview
        config = create_default_config()
        config.name = "my_nuclei_workflow"
        overview = compose_overview(config)
        assert isinstance(overview, str) and overview.strip()
        assert "my_nuclei_workflow" in overview

    def test_compose_method_is_nonempty_and_mentions_framework(self):
        from omero_annotate_ai.core.mifa_export import compose_method
        config = create_default_config()
        method = compose_method(config)
        assert isinstance(method, str) and method.strip()
        assert "micro_sam" in method

    @pytest.mark.parametrize(
        "version,expected",
        [("1.0.0", 1.0), ("v1.1.0", 1.1), ("2", 2.0), ("3.4.5", 3.4)],
    )
    def test_version_to_float(self, version, expected):
        from omero_annotate_ai.core.mifa_export import version_to_float
        assert version_to_float(version) == expected

    def test_version_to_float_garbage_falls_back_with_warning(self):
        from omero_annotate_ai.core.mifa_export import version_to_float
        with pytest.warns(UserWarning, match="version"):
            assert version_to_float("not-a-version") == 1.0


@pytest.mark.unit
class TestMIFAExport:
    """Build MIFA documents from an AnnotationConfig (needs bia-mifa-models)."""

    @pytest.fixture(autouse=True)
    def _require_mifa(self):
        pytest.importorskip("bia_mifa_models")

    def _rich_config(self):
        from omero_annotate_ai.core.annotation_config import AuthorInfo
        config = create_default_config()
        config.name = "nuclei_seg"
        config.study.title = "Nuclei segmentation"
        config.study.description = "Annotated nuclei for training."
        config.study.keywords = ["nuclei", "segmentation"]
        config.dataset.source_dataset_id = "S-BIAD123"
        config.annotation_methodology.annotation_criteria = "In-focus nuclei only."
        config.authors = [
            AuthorInfo(
                name="Jane Doe",
                affiliation="EMBL-EBI",
                email="jane@example.org",
                orcid="https://orcid.org/0000-0002-1825-0097",
            ),
            AuthorInfo(name="", affiliation="Nowhere"),  # no name -> dropped
        ]
        config.annotations = [
            ImageAnnotation(
                image_id=1, image_name="a.tif", annotation_id="1_0_0",
                category="training", timepoint=0, z_slice=0, channel=0,
                annotation_type="segmentation_mask",
                annotation_created_at="2026-06-22T11:00:00+00:00",
            ),
            ImageAnnotation(
                image_id=2, image_name="b.tif", annotation_id="2_0_0",
                category="validation", timepoint=0, z_slice=0, channel=0,
                annotation_type="segmentation_mask",
            ),
            ImageAnnotation(
                image_id=3, image_name="c.tif", annotation_id="3_0_0",
                category="training", timepoint=0, z_slice=0, channel=0,
                is_patch=True, patch_x=10, patch_y=20, patch_width=128, patch_height=128,
                annotation_type="segmentation_mask",
            ),
        ]
        return config

    def test_to_mifa_returns_three_documents(self):
        from bia_mifa_models.datamodel.bia_mifa_models import Study, Annotations, Version
        docs = self._rich_config().to_mifa()
        assert isinstance(docs["study"], Study)
        assert isinstance(docs["annotations"], Annotations)
        assert isinstance(docs["version"], Version)

    def test_annotations_required_fields_present(self):
        ann = self._rich_config().to_mifa()["annotations"]
        assert ann.annotation_overview.strip()
        assert ann.annotation_method.strip()

    def test_study_required_fields_present_from_blank_config(self):
        # default config has blank study fields - fallbacks must satisfy MIFA requireds
        st = create_default_config().to_mifa()["study"]
        assert st.title and st.description
        assert st.keywords  # required, non-empty
        assert str(st.license) in ("CC_BY", "CC0")
        assert st.funding_statement
        assert st.link_url  # required by the Links mixin, non-empty

    def test_file_metadata_count_matches_annotations(self):
        ann = self._rich_config().to_mifa()["annotations"]
        assert len(ann.file_metadata) == 3

    def test_file_metadata_local_ids(self):
        ann = self._rich_config().to_mifa(file_id_source="local")["annotations"]
        flm = ann.file_metadata[0]
        assert flm.annotation_id == "output/1_0_0_mask.tif"
        assert flm.source_image_id == "input/1_0_0.tif"

    def test_file_metadata_omero_ids(self):
        config = self._rich_config()
        config.annotations[0].label_id = 555
        ann = config.to_mifa(file_id_source="omero")["annotations"]
        flm = ann.file_metadata[0]
        assert flm.annotation_id == "555"
        assert flm.source_image_id == "1"

    def test_file_id_source_auto_is_per_record(self):
        config = self._rich_config()
        config.annotations[0].label_id = 777  # uploaded -> omero ids
        ann = config.to_mifa(file_id_source="auto")["annotations"]
        assert ann.file_metadata[0].annotation_id == "777"
        # second record never uploaded -> local path
        assert ann.file_metadata[1].annotation_id == "output/2_0_0_mask.tif"

    def test_spatial_information_encodes_plane_and_patch(self):
        ann = self._rich_config().to_mifa(file_id_source="local")["annotations"]
        si0 = ann.file_metadata[0].spatial_information
        assert "t=0" in si0 and "z=0" in si0 and "c=0" in si0
        assert "patch" in ann.file_metadata[2].spatial_information.lower()

    def test_annotation_creation_time_propagates(self):
        ann = self._rich_config().to_mifa()["annotations"]
        assert ann.file_metadata[0].annotation_creation_time is not None
        assert ann.file_metadata[1].annotation_creation_time is None

    def test_authors_mapped_and_blank_dropped(self):
        ann = self._rich_config().to_mifa()["annotations"]
        assert len(ann.authors) == 1
        author = ann.authors[0]
        assert author.author_first_name == "Jane"
        assert author.author_last_name == "Doe"
        assert str(author.orcid_id).endswith("0000-0002-1825-0097")
        assert author.organisation[0].organisation_name == "EMBL-EBI"

    def test_license_maps_to_mifa_code(self):
        config = self._rich_config()
        config.dataset.license = "CC0"
        assert str(config.to_mifa()["study"].license) == "CC0"

    def test_version_doc_uses_float_safe_version(self):
        config = self._rich_config()
        config.version = "2.3.1"
        v = config.to_mifa()["version"]
        assert v.version == 2.3
        assert v.timestamp is not None

    def test_funding_statement_field_and_kwarg_override(self):
        config = self._rich_config()
        config.study.funding_statement = "Funded by EMBO."
        assert config.to_mifa()["study"].funding_statement == "Funded by EMBO."
        assert config.to_mifa(funding_statement="Override")["study"].funding_statement == "Override"

    def test_no_annotations_still_builds_valid_annotations_doc(self):
        ann = create_default_config().to_mifa()["annotations"]
        assert len(ann.file_metadata or []) == 0
        assert ann.annotation_overview.strip()
        assert ann.annotation_method.strip()

    def test_save_mifa_writes_three_yaml_files_that_round_trip(self, tmp_path):
        from linkml_runtime.loaders import yaml_loader
        from bia_mifa_models.datamodel.bia_mifa_models import Annotations
        paths = self._rich_config().save_mifa(tmp_path, accession="S-BIAD123")
        assert (tmp_path / "Study_S-BIAD123.yaml").exists()
        assert (tmp_path / "Annotations_S-BIAD123.yaml").exists()
        assert (tmp_path / "Version_S-BIAD123.yaml").exists()
        assert set(paths) == {"study", "annotations", "version"}
        loaded = yaml_loader.load(
            str(tmp_path / "Annotations_S-BIAD123.yaml"), target_class=Annotations
        )
        assert len(loaded.file_metadata) == 3

    def test_save_mifa_accession_fallback_naming(self, tmp_path):
        # no source_dataset_id -> accession falls back to a slug of the name
        create_default_config().save_mifa(tmp_path)
        assert (tmp_path / "Annotations_default_annotation_workflow.yaml").exists()

    def test_to_mifa_metadata_returns_mifa_dicts(self):
        result = create_default_config().to_mifa_metadata()
        assert set(result) == {"study", "annotations", "version"}
        assert isinstance(result["annotations"], dict)
        assert result["annotations"]["annotation_overview"]


@pytest.mark.unit
class TestBIAExport:
    """Build a BioImage Archive submission bundle (file lists + MIFA + data copy)."""

    @pytest.fixture(autouse=True)
    def _require_mifa(self):
        pytest.importorskip("bia_mifa_models")

    def _config_with_data(self, root):
        """Config with 3 annotations and matching dummy tifs on disk under ``root``."""
        config = create_default_config()
        config.name = "bia_demo"
        config.study.title = "BIA demo"
        config.study.description = "desc"
        config.study.keywords = ["demo"]
        config.dataset.source_dataset_id = "S-BIAD999"
        config.output.output_directory = root
        config.annotations = [
            ImageAnnotation(
                image_id=i, image_name=f"img{i}.tif", annotation_id=f"{i}_0_0",
                category=("training" if i % 2 else "validation"),
                timepoint=0, z_slice=0, channel=0, annotation_type="segmentation_mask",
            )
            for i in (1, 2, 3)
        ]
        (root / "input").mkdir(parents=True, exist_ok=True)
        (root / "output").mkdir(parents=True, exist_ok=True)
        for i in (1, 2, 3):
            (root / "input" / f"{i}_0_0.tif").write_bytes(b"img")
            (root / "output" / f"{i}_0_0_mask.tif").write_bytes(b"mask")
        return config

    def test_file_lists_headers_and_counts(self, tmp_path):
        from omero_annotate_ai.core.mifa_export import build_bia_file_lists
        images, annotations = build_bia_file_lists(self._config_with_data(tmp_path))
        assert list(annotations.columns)[0] == "Files"
        assert "source_image" in annotations.columns
        assert len(annotations) == 3
        assert annotations["Files"].iloc[0] == "output/1_0_0_mask.tif"
        assert annotations["source_image"].iloc[0] == "input/1_0_0.tif"
        assert list(images.columns)[0] == "Files"
        assert len(images) == 3

    def test_file_lists_drop_constant_optional_columns(self, tmp_path):
        from omero_annotate_ai.core.mifa_export import build_bia_file_lists
        _, annotations = build_bia_file_lists(self._config_with_data(tmp_path))
        # constant optional columns dropped, varying ones kept, required always kept
        assert "Channel" not in annotations.columns
        assert "Timepoint" not in annotations.columns
        assert "Category" in annotations.columns  # training/validation -> 2 distinct
        assert "Files" in annotations.columns and "source_image" in annotations.columns

    def test_separate_channel_uses_label_input(self, tmp_path):
        from omero_annotate_ai.core.mifa_export import build_bia_file_lists
        config = self._config_with_data(tmp_path)
        config.spatial_coverage.label_channel = 0
        config.spatial_coverage.training_channels = [1]
        _, annotations = build_bia_file_lists(config)
        assert annotations["source_image"].iloc[0] == "label_input/1_0_0.tif"

    def test_save_bia_package_copies_data_and_writes_lists(self, tmp_path):
        config = self._config_with_data(tmp_path / "store")
        dest = tmp_path / "bundle"
        result = config.save_bia_package(dest, accession="S-BIAD999")
        assert (dest / "file_list_images.tsv").exists()
        assert (dest / "file_list_annotations.tsv").exists()
        assert (dest / "metadata" / "Annotations_S-BIAD999.yaml").exists()
        assert (dest / "output" / "1_0_0_mask.tif").exists()
        assert (dest / "input" / "1_0_0.tif").exists()
        assert result["n_annotations"] == 3

    def test_save_bia_package_accession_fallback(self, tmp_path):
        config = self._config_with_data(tmp_path / "store")
        config.dataset.source_dataset_id = None
        config.save_bia_package(tmp_path / "bundle")
        assert (tmp_path / "bundle" / "metadata" / "Annotations_bia_demo.yaml").exists()


@pytest.mark.unit
class TestAnnotationIdPersistence:
    """annotation_id survives the OMERO table round-trip and is regenerated for legacy tables."""

    def _config_with(self, annotations):
        config = create_default_config()
        config.annotations = list(annotations)
        return config

    def _legacy_df(self, config) -> pd.DataFrame:
        """OMERO table as written before annotation_id was persisted."""
        return config.to_dataframe().drop(columns=["annotation_id"])

    def _reload(self, df) -> list:
        """Round-trip a DataFrame through from_dataframe, returning the ids."""
        config = create_default_config()
        config.from_dataframe(df)
        return [a.annotation_id for a in config.annotations]

    # --- to_dataframe -------------------------------------------------------

    def test_to_dataframe_has_annotation_id_column(self):
        config = self._config_with(
            [
                ImageAnnotation(
                    image_id=101, image_name="img.tif", annotation_id="101_0_0",
                    timepoint=0, z_slice=0, channel=0,
                )
            ]
        )
        df = config.to_dataframe()
        assert "annotation_id" in df.columns
        assert df["annotation_id"].iloc[0] == "101_0_0"

    def test_annotation_id_survives_round_trip(self):
        """Stored ids are read back verbatim, even when they don't follow the scheme."""
        config = self._config_with(
            [
                ImageAnnotation(
                    image_id=101, image_name="a.tif", annotation_id="101_0_0",
                    timepoint=0, z_slice=0, channel=0,
                ),
                ImageAnnotation(
                    image_id=101, image_name="a.tif", annotation_id="custom_name_xyz",
                    timepoint=0, z_slice=1, channel=0,
                ),
                ImageAnnotation(
                    image_id=202, image_name="b.tif", annotation_id="202_0_3d",
                    timepoint=0, z_slice=0, channel=0,
                    is_volumetric=True, z_start=0, z_end=4, z_length=5,
                ),
            ]
        )
        assert self._reload(config.to_dataframe()) == [
            "101_0_0",
            "custom_name_xyz",
            "202_0_3d",
        ]

    # --- legacy tables (no annotation_id column) ----------------------------

    def test_legacy_2d_ids_regenerated(self):
        config = self._config_with(
            [
                ImageAnnotation(
                    image_id=101, image_name="a.tif",
                    timepoint=t, z_slice=z, channel=0,
                )
                for t in (0, 1)
                for z in (0, 2)
            ]
        )
        df = self._legacy_df(config)
        assert "annotation_id" not in df.columns
        assert self._reload(df) == ["101_0_0", "101_0_2", "101_1_0", "101_1_2"]

    def test_legacy_patch_ids_indexed_by_sorted_coordinates(self):
        """Patches on one plane get distinct indices, ordered by (patch_x, patch_y)."""
        # Rows deliberately out of coordinate order in the table
        coords = [(256, 256), (0, 0), (256, 0), (0, 256)]
        config = self._config_with(
            [
                ImageAnnotation(
                    image_id=101, image_name="a.tif",
                    timepoint=0, z_slice=0, channel=0,
                    is_patch=True, patch_x=x, patch_y=y,
                    patch_width=256, patch_height=256,
                )
                for x, y in coords
            ]
        )
        ids = self._reload(self._legacy_df(config))
        # sorted on (x, y): (0,0) -> 0, (0,256) -> 1, (256,0) -> 2, (256,256) -> 3
        assert ids == ["101_0_0_3", "101_0_0_0", "101_0_0_2", "101_0_0_1"]
        assert len(set(ids)) == 4

    def test_legacy_patch_indices_restart_per_plane(self):
        config = self._config_with(
            [
                ImageAnnotation(
                    image_id=image_id, image_name="a.tif",
                    timepoint=0, z_slice=z, channel=0,
                    is_patch=True, patch_x=x, patch_y=0,
                    patch_width=256, patch_height=256,
                )
                for image_id in (101, 202)
                for z in (0, 1)
                for x in (0, 256)
            ]
        )
        ids = self._reload(self._legacy_df(config))
        assert ids == [
            "101_0_0_0", "101_0_0_1",
            "101_0_1_0", "101_0_1_1",
            "202_0_0_0", "202_0_0_1",
            "202_0_1_0", "202_0_1_1",
        ]
        assert len(set(ids)) == len(ids)

    def test_legacy_volumetric_ids_regenerated(self):
        config = self._config_with(
            [
                ImageAnnotation(
                    image_id=303, image_name="v.tif",
                    timepoint=t, z_slice=0, channel=0,
                    is_volumetric=True, z_start=0, z_end=4, z_length=5,
                )
                for t in (0, 1)
            ]
        )
        assert self._reload(self._legacy_df(config)) == ["303_0_3d", "303_1_3d"]

    def test_legacy_volumetric_patch_ids_regenerated(self):
        config = self._config_with(
            [
                ImageAnnotation(
                    image_id=303, image_name="v.tif",
                    timepoint=0, z_slice=0, channel=0,
                    is_volumetric=True, z_start=0, z_end=4, z_length=5,
                    is_patch=True, patch_x=x, patch_y=0,
                    patch_width=256, patch_height=256,
                )
                for x in (256, 0)
            ]
        )
        ids = self._reload(self._legacy_df(config))
        assert ids == ["303_0_3d_1", "303_0_3d_0"]

    # --- empty / stable / unique -------------------------------------------

    def test_empty_annotation_id_column_is_regenerated(self):
        """A table whose annotation_id column is blank does not stay blank."""
        config = self._config_with(
            [
                ImageAnnotation(
                    image_id=101, image_name="a.tif",
                    timepoint=0, z_slice=z, channel=0,
                )
                for z in (0, 1)
            ]
        )
        df = config.to_dataframe()
        assert list(df["annotation_id"]) == ["", ""]  # default annotation_id
        assert self._reload(df) == ["101_0_0", "101_0_1"]

    def test_regeneration_is_stable_across_calls(self):
        config = self._config_with(
            [
                ImageAnnotation(
                    image_id=101, image_name="a.tif",
                    timepoint=0, z_slice=0, channel=0,
                    is_patch=True, patch_x=x, patch_y=y,
                    patch_width=256, patch_height=256,
                )
                for x, y in [(256, 0), (0, 256), (0, 0)]
            ]
        )
        df = self._legacy_df(config)
        assert self._reload(df) == self._reload(df)

    def test_all_ids_non_empty_and_unique_for_mixed_legacy_table(self):
        """Post-condition: every reloaded annotation has a non-empty, unique id."""
        annotations = [
            ImageAnnotation(
                image_id=101, image_name="a.tif", timepoint=0, z_slice=0, channel=0,
            ),
            ImageAnnotation(
                image_id=101, image_name="a.tif", timepoint=0, z_slice=1, channel=0,
            ),
            ImageAnnotation(
                image_id=202, image_name="b.tif", timepoint=1, z_slice=0, channel=0,
                is_patch=True, patch_x=0, patch_y=0, patch_width=256, patch_height=256,
            ),
            ImageAnnotation(
                image_id=202, image_name="b.tif", timepoint=1, z_slice=0, channel=0,
                is_patch=True, patch_x=256, patch_y=0, patch_width=256, patch_height=256,
            ),
            ImageAnnotation(
                image_id=303, image_name="v.tif", timepoint=0, z_slice=0, channel=0,
                is_volumetric=True, z_start=0, z_end=4, z_length=5,
            ),
        ]
        ids = self._reload(self._legacy_df(self._config_with(annotations)))
        assert len(ids) == len(annotations)
        assert all(i for i in ids)
        assert len(set(ids)) == len(ids)

    def test_duplicate_rows_still_get_unique_ids(self):
        """Two identical rows cannot collapse onto the same id."""
        annotation = dict(
            image_id=101, image_name="a.tif", timepoint=0, z_slice=0, channel=0,
        )
        config = self._config_with(
            [ImageAnnotation(**annotation), ImageAnnotation(**annotation)]
        )
        ids = self._reload(self._legacy_df(config))
        assert ids[0] == "101_0_0"
        assert len(set(ids)) == 2
