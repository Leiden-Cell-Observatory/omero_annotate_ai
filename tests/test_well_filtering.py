"""Tests for well filtering functionality."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline
from omero_annotate_ai.core.annotation_config import AnnotationConfig


@pytest.mark.unit
class TestWellFiltering:
    """Test well filtering functionality."""

    def test_check_well_filter_match(self):
        """Test that _check_well_filter correctly matches filter criteria."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "plate",
                "container_id": 1,
                "well_filters": {
                    "cellline": ["U2OS", "HeLa"],
                    "treatment": ["Control"]
                }
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        # Test matching case - all criteria met
        well_kv = {
            "cellline": "U2OS",
            "treatment": "Control"
        }
        assert pipeline._check_well_filter(well_kv, config.omero.well_filters) is True

        # Test matching case - HeLa instead of U2OS
        well_kv = {
            "cellline": "HeLa",
            "treatment": "Control"
        }
        assert pipeline._check_well_filter(well_kv, config.omero.well_filters) is True

    def test_check_well_filter_no_match(self):
        """Test that _check_well_filter correctly rejects non-matching wells."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "plate",
                "container_id": 1,
                "well_filters": {
                    "cellline": ["U2OS", "HeLa"],
                    "treatment": ["Control"]
                }
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        # Test non-matching cellline
        well_kv = {
            "cellline": "MCF7",
            "treatment": "Control"
        }
        assert pipeline._check_well_filter(well_kv, config.omero.well_filters) is False

        # Test non-matching treatment
        well_kv = {
            "cellline": "U2OS",
            "treatment": "Drug A"
        }
        assert pipeline._check_well_filter(well_kv, config.omero.well_filters) is False

        # Test missing key
        well_kv = {
            "cellline": "U2OS"
            # Missing "treatment" key
        }
        assert pipeline._check_well_filter(well_kv, config.omero.well_filters) is False

    def test_get_well_map_annotations(self):
        """Test that _get_well_map_annotations correctly retrieves map annotations."""
        config = AnnotationConfig(name="test", omero={"container_type": "plate", "container_id": 1})
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        # Mock well object with map annotations
        mock_well = Mock()
        mock_map_ann = Mock()
        mock_map_ann.OMERO_TYPE.__name__ = 'MapAnnotationI'

        # Mock key-value pairs
        mock_kv1 = Mock()
        mock_kv1.name = "cellline"
        mock_kv1.value = "U2OS"

        mock_kv2 = Mock()
        mock_kv2.name = "treatment"
        mock_kv2.value = "Control"

        mock_map_ann.getMapValue.return_value = [mock_kv1, mock_kv2]
        mock_well.listAnnotations.return_value = [mock_map_ann]

        mock_conn.getObject.return_value = mock_well

        # Test retrieval
        result = pipeline._get_well_map_annotations(123)

        assert result == {
            "cellline": "U2OS",
            "treatment": "Control"
        }

    def test_get_well_map_annotations_empty(self):
        """Test _get_well_map_annotations with no annotations."""
        config = AnnotationConfig(name="test", omero={"container_type": "plate", "container_id": 1})
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        # Mock well object with no annotations
        mock_well = Mock()
        mock_well.listAnnotations.return_value = []
        mock_conn.getObject.return_value = mock_well

        result = pipeline._get_well_map_annotations(123)
        assert result == {}

    def test_get_well_map_annotations_well_not_found(self):
        """Test _get_well_map_annotations when well doesn't exist."""
        config = AnnotationConfig(name="test", omero={"container_type": "plate", "container_id": 1})
        mock_conn = Mock()
        mock_conn.getObject.return_value = None
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        result = pipeline._get_well_map_annotations(999)
        assert result == {}

    @pytest.mark.parametrize("filter_mode,matches_filter,expected_include", [
        ("include", True, True),
        ("include", False, False),
        ("exclude", True, False),
        ("exclude", False, True),
    ])
    def test_well_filter_modes(self, filter_mode, matches_filter, expected_include):
        """Test include and exclude filter modes."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "plate",
                "container_id": 1,
                "well_filters": {"cellline": ["U2OS"]},
                "well_filter_mode": filter_mode
            }
        )

        # Calculate what should_include would be
        should_include = (
            (config.omero.well_filter_mode == "include" and matches_filter) or
            (config.omero.well_filter_mode == "exclude" and not matches_filter)
        )

        assert should_include == expected_include


@pytest.mark.integration
class TestWellFilteringIntegration:
    """Integration tests for well filtering with mocked OMERO connection."""

    def test_get_image_ids_with_well_filtering(self):
        """Test that get_image_ids_from_container applies well filtering."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "plate",
                "container_id": 1,
                "well_filters": {"cellline": ["U2OS"]},
                "well_filter_mode": "include"
            }
        )

        # Mock OMERO connection and ezomero functions
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        # Mock well IDs and well objects
        with patch('omero_annotate_ai.core.annotation_pipeline.ezomero') as mock_ezomero:
            mock_ezomero.get_well_ids.return_value = [101, 102, 103]

            # Mock get_image_ids to return different images per well
            def mock_get_image_ids(conn, well=None):
                if well == 101:
                    return [1, 2]
                elif well == 102:
                    return [3, 4]
                elif well == 103:
                    return [5, 6]
                return []

            mock_ezomero.get_image_ids.side_effect = mock_get_image_ids

            # Mock well map annotations
            def mock_get_well_annotations(well_id):
                if well_id == 101:
                    return {"cellline": "U2OS"}  # Match
                elif well_id == 102:
                    return {"cellline": "HeLa"}  # No match
                elif well_id == 103:
                    return {"cellline": "U2OS"}  # Match
                return {}

            pipeline._get_well_map_annotations = mock_get_well_annotations

            # Run the method
            image_ids = pipeline.get_image_ids_from_container()

            # Should only include images from wells 101 and 103 (U2OS)
            assert set(image_ids) == {1, 2, 5, 6}

    def test_get_image_ids_without_well_filtering(self):
        """Test that get_image_ids_from_container works without filtering."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "plate",
                "container_id": 1,
                # No well_filters specified
            }
        )

        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        with patch('omero_annotate_ai.core.annotation_pipeline.ezomero') as mock_ezomero:
            mock_ezomero.get_image_ids.return_value = [1, 2, 3, 4, 5, 6]

            image_ids = pipeline.get_image_ids_from_container()

            # Should include all images
            assert image_ids == [1, 2, 3, 4, 5, 6]


@pytest.mark.unit
class TestMultiContainerImageRetrieval:
    """Test multi-container image retrieval functionality."""

    def test_get_image_ids_multiple_datasets(self):
        """Test getting image IDs from multiple datasets."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "dataset",
                "container_ids": [1, 2]
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        with patch('omero_annotate_ai.core.annotation_pipeline.ezomero') as mock_ezomero:
            def mock_get_image_ids(conn, dataset=None, plate=None, well=None):
                if dataset == 1:
                    return [101, 102]
                elif dataset == 2:
                    return [201, 202]
                return []

            mock_ezomero.get_image_ids.side_effect = mock_get_image_ids

            image_ids = pipeline.get_image_ids_from_container()

            assert set(image_ids) == {101, 102, 201, 202}
            assert len(image_ids) == 4

    def test_get_image_ids_removes_duplicates(self):
        """Test that duplicate image IDs are removed."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "dataset",
                "container_ids": [1, 2]
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        with patch('omero_annotate_ai.core.annotation_pipeline.ezomero') as mock_ezomero:
            def mock_get_image_ids(conn, dataset=None, plate=None, well=None):
                if dataset == 1:
                    return [101, 102, 103]
                elif dataset == 2:
                    return [102, 103, 104]  # 102, 103 are duplicates
                return []

            mock_ezomero.get_image_ids.side_effect = mock_get_image_ids

            image_ids = pipeline.get_image_ids_from_container()

            # Should have 4 unique IDs: 101, 102, 103, 104
            assert len(image_ids) == 4
            assert set(image_ids) == {101, 102, 103, 104}
            # Order preserved: first occurrence wins
            assert image_ids == [101, 102, 103, 104]

    def test_get_image_ids_single_container_backward_compatible(self):
        """Test that single container_id still works (backward compatibility)."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "dataset",
                "container_id": 123
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        with patch('omero_annotate_ai.core.annotation_pipeline.ezomero') as mock_ezomero:
            mock_ezomero.get_image_ids.return_value = [1001, 1002, 1003]

            image_ids = pipeline.get_image_ids_from_container()

            assert image_ids == [1001, 1002, 1003]
            mock_ezomero.get_image_ids.assert_called_once_with(mock_conn, dataset=123)

    def test_get_image_ids_no_container_raises(self):
        """Test that no container configured raises ValueError."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "dataset",
                "container_id": 0
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        with pytest.raises(ValueError, match="No container IDs configured"):
            pipeline.get_image_ids_from_container()

    def test_get_image_ids_multiple_plates_with_well_filtering(self):
        """Test that well filters apply across multiple plates."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "plate",
                "container_ids": [1, 2],
                "well_filters": {"cellline": ["U2OS"]},
                "well_filter_mode": "include"
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        with patch('omero_annotate_ai.core.annotation_pipeline.ezomero') as mock_ezomero:
            # Mock well IDs for each plate
            def mock_get_well_ids(conn, plate=None):
                if plate == 1:
                    return [101, 102]  # Plate 1 wells
                elif plate == 2:
                    return [201, 202]  # Plate 2 wells
                return []

            mock_ezomero.get_well_ids.side_effect = mock_get_well_ids

            # Mock image IDs for each well
            def mock_get_image_ids(conn, dataset=None, plate=None, well=None):
                well_images = {
                    101: [1001],  # U2OS
                    102: [1002],  # HeLa
                    201: [2001],  # U2OS
                    202: [2002],  # HeLa
                }
                return well_images.get(well, [])

            mock_ezomero.get_image_ids.side_effect = mock_get_image_ids

            # Mock map annotations for wells
            def mock_annotations(well_id):
                annotations = {
                    101: {"cellline": "U2OS"},
                    102: {"cellline": "HeLa"},
                    201: {"cellline": "U2OS"},
                    202: {"cellline": "HeLa"},
                }
                return annotations.get(well_id, {})

            pipeline._get_well_map_annotations = mock_annotations

            image_ids = pipeline.get_image_ids_from_container()

            # Should only include U2OS wells from both plates
            assert set(image_ids) == {1001, 2001}


@pytest.mark.unit
class TestMultiContainerTableTitle:
    """Test table title generation for multi-container scenarios."""

    def test_table_title_single_container(self):
        """Test table title generation for single container."""
        config = AnnotationConfig(
            name="",  # Empty name triggers auto-generation
            omero={
                "container_type": "dataset",
                "container_id": 123
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        title = pipeline._get_table_title()

        assert "dataset_123" in title

    def test_table_title_multiple_containers(self):
        """Test table title generation for multiple containers."""
        config = AnnotationConfig(
            name="",  # Empty name triggers auto-generation
            omero={
                "container_type": "plate",
                "container_ids": [1, 2, 3]
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        title = pipeline._get_table_title()

        assert "plate_multi_3" in title

    def test_table_title_uses_config_name(self):
        """Test that explicit config name takes precedence."""
        config = AnnotationConfig(
            name="my_custom_table",
            omero={
                "container_type": "plate",
                "container_ids": [1, 2, 3]
            }
        )
        mock_conn = Mock()
        pipeline = AnnotationPipeline(config, conn=mock_conn)

        title = pipeline._get_table_title()

        assert title == "my_custom_table"


import warnings
from pathlib import Path

import pandas as pd

from omero_annotate_ai.core.annotation_config import ZarrConfig
from omero_annotate_ai.sources.wells import match_well_filter, normalize_well
from omero_annotate_ai.sources.zarr_discovery import discover_zarr_images


@pytest.mark.unit
class TestWellHelpers:
    @pytest.mark.parametrize("raw", ["B2", "b02", "B/2", "B_02", " B 2 "])
    def test_normalize_well(self, raw):
        assert normalize_well(raw) == "B2"

    def test_normalize_rejects_garbage(self):
        with pytest.raises(ValueError, match="well name"):
            normalize_well("field0")

    def test_match_well_filter_and_logic(self):
        kv = {"compound": "DMSO", "dose": "0"}
        assert match_well_filter(kv, {"compound": ["DMSO", "X"]})
        assert not match_well_filter(kv, {"compound": ["DMSO"], "dose": ["1"]})
        assert not match_well_filter(kv, {"cellline": ["U2OS"]})


@pytest.mark.unit
class TestZarrDiscovery:
    @pytest.fixture(autouse=True)
    def _stores(self, ngff_stores, tmp_path):
        self.s = ngff_stores
        self.out = tmp_path / "out"

    def _csv(self, tmp_path, rows):
        path = tmp_path / "layout.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_single_image(self):
        refs = discover_zarr_images(ZarrConfig(uris=[self.s["img04"]]))
        assert [(r.uri, r.well, r.name) for r in refs] == [(self.s["img04"], "", "")]

    def test_plate_first_field_in_row_column_order(self):
        refs = discover_zarr_images(ZarrConfig(uris=[self.s["plate"]]))
        plate = self.s["plate"]
        assert [r.uri for r in refs] == [f"{plate}/A/1/0", f"{plate}/A/2/0", f"{plate}/B/3/0"]
        assert refs[0].name == "testplate_A1_0"

    def test_plate_all_fields(self):
        assert len(discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], fields="all"))) == 6

    def test_well_uri(self):
        (ref,) = discover_zarr_images(ZarrConfig(uris=[self.s["plate"] + "/A/2"]))
        assert (ref.plate, ref.well, ref.field) == ("testplate", "A2", "0")

    def test_wells_list_normalized_and_unknown_ignored(self):
        with pytest.warns(UserWarning, match="H12"):
            refs = discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], wells=["a02", "H12"]))
        assert [r.well for r in refs] == ["A2"]

    def test_unknown_wells_list_entry_warns(self):
        with pytest.warns(UserWarning, match="H12"):
            discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], wells=["A2", "H12"]))

    def test_csv_filter_include_records_the_match(self, tmp_path):
        csv = self._csv(tmp_path, [{"well": "A1", "compound": "DMSO"},
                                   {"well": "A02", "compound": "X"},
                                   {"well": "B3", "compound": "DMSO"}])
        cfg = ZarrConfig(uris=[self.s["plate"]], well_metadata=csv,
                         well_filters={"compound": ["DMSO"]})
        refs = discover_zarr_images(cfg, output_dir=self.out)
        assert [r.well for r in refs] == ["A1", "B3"]
        assert refs[0].well_metadata == {"compound": "DMSO"}
        table = pd.read_csv(self.out / "well_matching.csv")
        assert list(table["well"]) == ["A1", "A2", "B3"]
        assert list(table["matched"]) == [True, True, True]
        assert list(table["included"]) == [True, False, True]
        assert list(table["compound"]) == ["DMSO", "X", "DMSO"]

    def test_csv_filter_exclude(self, tmp_path):
        csv = self._csv(tmp_path, [{"well": w, "compound": c}
                                   for w, c in [("A1", "DMSO"), ("A2", "X"), ("B3", "DMSO")]])
        cfg = ZarrConfig(uris=[self.s["plate"]], well_metadata=csv,
                         well_filters={"compound": ["DMSO"]}, well_filter_mode="exclude")
        assert [r.well for r in discover_zarr_images(cfg)] == ["A2"]

    def test_csv_with_plate_column(self, tmp_path):
        csv = self._csv(tmp_path, [{"plate": "testplate", "well": "A1", "compound": "DMSO"},
                                   {"plate": "other", "well": "A2", "compound": "DMSO"}])
        cfg = ZarrConfig(uris=[self.s["plate"]], well_metadata=csv,
                         well_filters={"compound": ["DMSO"]})
        with pytest.warns(UserWarning):
            refs = discover_zarr_images(cfg)
        assert [r.well for r in refs] == ["A1"]

    def test_mismatch_warnings(self, tmp_path):
        csv = self._csv(tmp_path, [{"well": "A1", "c": "1"}, {"well": "A2", "c": "1"},
                                   {"well": "H12", "c": "1"}])
        with pytest.warns(UserWarning) as record:
            discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], well_metadata=csv))
        messages = " ".join(str(w.message) for w in record)
        assert "B3" in messages and "H12" in messages

    def test_csv_without_well_column(self, tmp_path):
        csv = self._csv(tmp_path, [{"position": "A1"}])
        with pytest.raises(ValueError, match="'well' column"):
            discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], well_metadata=csv))

    def test_not_ngff(self, tmp_path):
        (tmp_path / "plain").mkdir()
        with pytest.raises(ValueError, match="OME-Zarr"):
            discover_zarr_images(ZarrConfig(uris=[str(tmp_path / "plain")]))
