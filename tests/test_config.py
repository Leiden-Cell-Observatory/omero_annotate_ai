"""Modern tests for configuration management with updated package structure."""

import tempfile
from pathlib import Path

import pytest
import yaml

from omero_annotate_ai.core.annotation_config import (
    AnnotationConfig,
    create_default_config,
    get_config_template,
    load_config,
    load_config_from_yaml,
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
        lines = [
            ln
            for ln in yaml_str.splitlines()
            if ln and not ln.startswith(" ") and ":" in ln
        ]
        keys_in_yaml = [ln.split(":", 1)[0] for ln in lines]

        # Expected order follows field declaration order of AnnotationConfig:
        # the schema sections first, then the tool sections (unset fields left out)
        expected_prefix_order = [
            "config_version",
            "name",
            "created",
            "license",
            "authors",
            "keywords",
            "links",
            "ai_models_trained",
            "methodology",
            "classes",
            "sampling",
            "split",
            "extensions",
            "training",
            "normalization",
            "ai_model",
            "display",
            "workflow",
            "omero",
            "source",
            "zarr",
            "storage",
            "identity",
            "sessions",
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
          train_n: 4
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = AnnotationConfig.from_yaml(f.name)

            assert config.name == "test"
            assert config.training.train_n == 4

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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.save_yaml(f.name)

            # Load it back
            loaded_config = AnnotationConfig.from_yaml(f.name)

            assert loaded_config.omero.container_id == 999
            assert loaded_config.name == "test_roundtrip"

        # Clean up
        Path(f.name).unlink()

    @pytest.mark.parametrize(
        ("data", "replacement"),
        [
            ({"annotations": []}, "TrainingSet.units"),
            ({"output": {"output_directory": "x"}}, "home"),
            ({"workflow": {"read_only_mode": True}}, "storage: local"),
            ({"workflow": {"resume_from_table": True}}, "TrainingSet.open"),
            ({"omero": {"table_id": 5}}, "identity.table_id"),
            ({"spatial_coverage": {"channels": [0]}}, "sampling"),
            ({"annotation_methodology": {}}, "methodology"),
            ({"dataset": {}}, "source_dataset"),
            ({"feature_types": []}, "classes"),
            ({"tags": []}, "keywords"),
            ({"custom_metadata": {}}, "extensions"),
            ({"training": {"seed": 1}}, "split.seed"),
            ({"training": {"split_unit": "well"}}, "split.unit"),
            ({"schema_version": 3}, "config_version: 4"),
        ],
    )
    def test_removed_keys_raise_naming_the_replacement(self, data, replacement):
        from omero_annotate_ai.errors import RemovedConfigKey

        with pytest.raises(RemovedConfigKey, match=replacement):
            AnnotationConfig.from_dict({"name": "x", **data})

    def test_old_yaml_file_raises_naming_the_replacement(self, tmp_path):
        from omero_annotate_ai.errors import RemovedConfigKey

        path = tmp_path / "old.yaml"
        path.write_text("name: x\nworkflow:\n  read_only_mode: true\n")
        with pytest.raises(RemovedConfigKey, match="storage: local"):
            load_config_from_yaml(str(path))

    def test_sampling_defaults_to_first_plane_of_channel_0(self):
        # As documented: a YAML may leave out the channels, timepoints and z-slices.
        config = AnnotationConfig.from_dict(
            {"name": "x", "sampling": {"patches": {"width": 64, "height": 64}}}
        )
        assert (
            config.label_channel(),
            config.timepoint_values(),
            config.z_values(),
        ) == (
            0,
            [0],
            [0],
        )

    def test_other_config_versions_are_refused(self):
        from omero_annotate_ai.errors import ConfigError

        with pytest.raises(ConfigError, match="config_version"):
            AnnotationConfig.from_dict({"name": "x", "config_version": 3})

    def test_unknown_keys_and_assignments_are_rejected(self):
        with pytest.raises(ValueError):
            AnnotationConfig.from_dict({"name": "x", "colour": "red"})
        with pytest.raises(ValueError, match="label_chanel"):  # a typo in a section
            AnnotationConfig.from_dict({"name": "x", "sampling": {"label_chanel": 1}})
        config = create_default_config()
        with pytest.raises(ValueError):
            config.storage = "cloud"

    def test_sessions_and_identity_round_trip(self):
        from datetime import datetime

        from omero_annotate_ai.core.annotation_config import SessionRecord

        config = create_default_config()
        config.identity.table_id = 12
        config.sessions.append(
            SessionRecord(id=1, started=datetime(2026, 1, 1), note="nuclei only")
        )
        again = AnnotationConfig.from_yaml(config.to_yaml())
        assert again.identity.table_id == 12
        assert again.sessions[0].note == "nuclei only"
        assert again.sessions[0].status == "open"


@pytest.mark.unit
class TestOMEROConfigMultiContainer:
    """Test OMEROConfig multi-container fields and methods."""

    def test_get_all_container_ids_single(self):
        """Test get_all_container_ids with single container_id."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_id": 123}
        )
        assert config.omero.get_all_container_ids() == [123]

    def test_get_all_container_ids_multiple(self):
        """Test get_all_container_ids with container_ids list."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_ids": [1, 2, 3]}
        )
        assert config.omero.get_all_container_ids() == [1, 2, 3]

    def test_container_ids_precedence(self):
        """Test that container_ids takes precedence over container_id."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "dataset",
                "container_id": 999,  # Should be ignored
                "container_ids": [1, 2, 3],
            },
        )
        assert config.omero.get_all_container_ids() == [1, 2, 3]

    def test_get_all_container_ids_empty_list(self):
        """Test that empty container_ids falls back to container_id."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "dataset",
                "container_id": 456,
                "container_ids": [],
            },
        )
        assert config.omero.get_all_container_ids() == [456]

    def test_get_all_container_ids_no_container(self):
        """Test get_all_container_ids with no containers configured."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_id": 0}
        )
        assert config.omero.get_all_container_ids() == []

    def test_get_primary_container_id_single(self):
        """Test get_primary_container_id with single container."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_id": 123}
        )
        assert config.omero.get_primary_container_id() == 123

    def test_get_primary_container_id_multiple(self):
        """Test get_primary_container_id with multiple containers."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_ids": [10, 20, 30]},
        )
        assert config.omero.get_primary_container_id() == 10

    def test_get_primary_container_id_empty(self):
        """Test get_primary_container_id with no containers."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_id": 0}
        )
        assert config.omero.get_primary_container_id() == 0

    def test_is_multi_container_false_single(self):
        """Test is_multi_container returns False for single container."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_id": 123}
        )
        assert config.omero.is_multi_container() is False

    def test_is_multi_container_false_one_in_list(self):
        """Test is_multi_container returns False for single item in list."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_ids": [123]}
        )
        assert config.omero.is_multi_container() is False

    def test_is_multi_container_true(self):
        """Test is_multi_container returns True for multiple containers."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_ids": [1, 2]}
        )
        assert config.omero.is_multi_container() is True

    def test_yaml_roundtrip_with_container_ids(self):
        """Test that container_ids survives YAML roundtrip."""
        config = AnnotationConfig(
            name="test_multi",
            omero={"container_type": "plate", "container_ids": [1, 2, 3]},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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
class TestAnnotationConfigJSON:
    """Tests for JSON serialization of AnnotationConfig."""

    def _make_config(self):
        from bioimage_annotation_schema import datamodel as bas

        from omero_annotate_ai.core.annotation_config import (
            AIModelConfig,
            AnnotationConfig,
            OMEROConfig,
            TrainingConfig,
            WorkflowConfig,
        )

        return AnnotationConfig(
            name="test_workflow",
            study=bas.Study(title="Test", description="Test study"),
            source_dataset=bas.SourceDataset(description="test"),
            methodology=bas.Methodology(
                annotation_types=["segmentation_mask"], criteria="test"
            ),
            sampling=bas.Sampling(label_channel=0),
            training=TrainingConfig(),
            ai_model=AIModelConfig(),
            workflow=WorkflowConfig(),
            omero=OMEROConfig(container_type="dataset", container_id=1),
            classes=[bas.AnnotationClass(name="cell", color="#FF0000")],
        )

    def test_to_json_returns_valid_json(self):
        import json

        config = self._make_config()
        json_str = config.to_json()
        data = json.loads(json_str)
        assert data["name"] == "test_workflow"
        assert len(data["classes"]) == 1
        assert data["classes"][0]["name"] == "cell"

    def test_from_json_string(self):
        config = self._make_config()
        json_str = config.to_json()
        loaded = type(config).from_json(json_str)
        assert loaded.name == config.name
        assert len(loaded.classes) == 1
        assert loaded.classes[0].name == "cell"

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

        with pytest.warns(
            UserWarning, match="train_fraction.*ignored.*segment_all=False"
        ):
            TrainingConfig(
                segment_all=False, train_n=3, validate_n=2, test_n=0, train_fraction=0.5
            )

    def test_warns_when_validation_fraction_set_but_count_mode(self):
        from omero_annotate_ai.core.annotation_config import TrainingConfig

        with pytest.warns(UserWarning, match="ignored.*segment_all=False"):
            TrainingConfig(
                segment_all=False,
                train_n=3,
                validate_n=2,
                test_n=0,
                validation_fraction=0.1,
            )

    def test_warns_when_counts_set_but_fraction_mode(self):
        from omero_annotate_ai.core.annotation_config import TrainingConfig

        with pytest.warns(UserWarning, match="train_n.*ignored.*segment_all=True"):
            TrainingConfig(segment_all=True, train_n=10, validate_n=2, test_n=0)

    def test_no_warning_when_only_active_count_set_changed(self):
        import warnings

        from omero_annotate_ai.core.annotation_config import TrainingConfig

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TrainingConfig(segment_all=False, train_n=5, validate_n=2, test_n=0)

    def test_no_warning_when_only_active_fraction_set_changed(self):
        import warnings

        from omero_annotate_ai.core.annotation_config import TrainingConfig

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TrainingConfig(
                segment_all=True,
                train_fraction=0.6,
                validation_fraction=0.4,
                test_fraction=0.0,
            )


@pytest.mark.unit
class TestZarrConfig:
    def test_default_source_is_omero(self):
        config = create_default_config()
        assert config.source == "omero"
        assert config.zarr.uris == []

    def test_zarr_requires_uris(self):
        with pytest.raises(ValueError, match="zarr.uris"):
            AnnotationConfig(name="z", source="zarr")

    def test_filters_require_well_metadata(self):
        with pytest.raises(ValueError, match="well_metadata"):
            AnnotationConfig(
                name="z",
                source="zarr",
                storage="local",
                zarr={"uris": ["a.zarr"], "well_filters": {"c": ["x"]}},
            )

    def test_zarr_needs_local_storage(self):
        with pytest.raises(ValueError, match="storage: local"):
            AnnotationConfig(name="z", source="zarr", zarr={"uris": ["a.zarr"]})

    def test_switching_an_existing_config_to_zarr_is_validated(self):
        """Assignment runs the same checks as construction (no silent OMERO storage)."""
        config = AnnotationConfig(name="z", zarr={"uris": ["a.zarr"]})
        with pytest.raises(ValueError, match="storage: local"):
            config.source = "zarr"
        assert config.source == "omero"

    def test_yaml_round_trip(self):
        config = AnnotationConfig(
            name="z",
            source="zarr",
            storage="local",
            zarr={"uris": ["a.zarr"], "level": 1, "fields": "all", "wells": ["B2"]},
        )
        again = AnnotationConfig.from_yaml(config.to_yaml())
        assert again.source == "zarr"
        assert (again.zarr.level, again.zarr.fields, again.zarr.wells) == (
            1,
            "all",
            ["B2"],
        )

    def test_bad_fields_value_rejected(self):
        with pytest.raises(ValueError):
            AnnotationConfig(name="z", source="zarr", zarr={"uris": ["a"], "fields": 3})

    def test_example_yaml_loads(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config_from_yaml(
            str(root / "notebooks/marimo/configs/zarr-s-biad855-example.yaml")
        )
        assert config.source == "zarr"
        assert config.zarr.level == 1
        assert config.storage == "local"

    def test_idr_example_yaml_loads(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config_from_yaml(
            str(root / "notebooks/marimo/configs/idr0013-mitocheck-zarr.yaml")
        )
        assert config.source == "zarr" and config.storage == "local"
        assert config.zarr.uris[0].startswith(
            "https://livingobjects.ebi.ac.uk/idr/zarr/"
        )
        assert config.sampling.timepoints.mode == "random"


@pytest.mark.unit
class TestOpenFields:
    """Free-text notes and custom key-value metadata on AnnotationConfig."""

    def test_defaults_are_empty(self):
        config = create_default_config()
        assert config.notes is None
        assert config.extensions == {}

    def test_yaml_round_trip(self, tmp_path):
        config = create_default_config()
        config.notes = "Some wells are out of focus."
        config.extensions = {"lab": {"instrument": "Incucyte SX5", "objective": "10x"}}
        path = tmp_path / "config.yaml"
        config.save_yaml(path)
        loaded = AnnotationConfig.from_yaml(path)
        assert loaded.notes == "Some wells are out of focus."
        assert loaded.extensions == {
            "lab": {"instrument": "Incucyte SX5", "objective": "10x"}
        }


@pytest.mark.unit
class TestSchemaSections:
    """The description sections are the Bioimage Annotation Schema's own models."""

    def test_template_sections_are_schema_models(self):
        from bioimage_annotation_schema import datamodel as bas

        config = AnnotationConfig.from_yaml(get_config_template())
        assert isinstance(config.methodology, bas.Methodology)
        assert isinstance(config.sampling, bas.Sampling)
        assert config.authors[0].last_name == "Example"
        assert config.methodology.annotation_types == ["segmentation_mask"]

    def test_annotation_types_use_mifa_names(self):
        with pytest.raises(ValueError):
            AnnotationConfig.from_dict(
                {
                    "name": "x",
                    "methodology": {"annotation_types": ["point"], "criteria": ""},
                }
            )
        config = AnnotationConfig.from_dict(
            {
                "name": "x",
                "methodology": {
                    "annotation_types": ["point_annotations"],
                    "criteria": "centre of each cell",
                },
            }
        )
        assert config.methodology.annotation_types == ["point_annotations"]

    def test_licence_is_one_the_archive_accepts(self):
        with pytest.raises(ValueError):
            AnnotationConfig.from_dict({"name": "x", "license": "MIT"})

    def test_yaml_leaves_out_unset_fields(self):
        text = create_default_config().to_yaml()
        assert "null" not in text


@pytest.mark.unit
class TestSampling:
    """What the tool reads from ``sampling``."""

    def test_training_channels_default_to_the_label_channel(self):
        config = AnnotationConfig.from_dict(
            {"name": "x", "sampling": {"label_channel": 2}}
        )
        assert config.training_channels() == [2]
        assert not config.uses_separate_channels()

    def test_separate_training_channel(self):
        config = AnnotationConfig.from_dict(
            {"name": "x", "sampling": {"label_channel": 0, "input_channels": [1]}}
        )
        assert config.uses_separate_channels()

    def test_volume_takes_every_z_slice_by_default_mode(self):
        config = AnnotationConfig.from_dict(
            {"name": "x", "sampling": {"volumetric": True, "z_slices": {"mode": "all"}}}
        )
        assert config.sampling.z_slices.mode == "all"


@pytest.mark.unit
class TestDisplayConfig:
    """Context channels shown next to the annotated one."""

    def test_specs_default_names_and_colormaps(self):
        config = AnnotationConfig.from_dict(
            {"name": "x", "display": {"context_channels": [1, 2]}}
        )
        assert config.display.specs() == [(1, "ch1", "blue"), (2, "ch2", "magenta")]

    def test_custom_names(self):
        config = AnnotationConfig.from_dict(
            {
                "name": "x",
                "display": {"context_channels": [1], "context_channel_names": ["DAPI"]},
            }
        )
        assert config.display.specs() == [(1, "DAPI", "blue")]

    def test_overlap_with_label_channel_is_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            AnnotationConfig.from_dict(
                {"name": "x", "display": {"context_channels": [0]}}
            )

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="length"):
            AnnotationConfig.from_dict(
                {
                    "name": "x",
                    "display": {
                        "context_channels": [1],
                        "context_channel_colormaps": ["red", "blue"],
                    },
                }
            )
