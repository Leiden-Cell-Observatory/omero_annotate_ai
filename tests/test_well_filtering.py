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
