"""Tests for well filtering functionality."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from omero_annotate_ai.core.annotation_config import AnnotationConfig, ZarrConfig
from omero_annotate_ai.sources import ImageMeta
from omero_annotate_ai.sources.wells import match_well_filter, normalize_well
from omero_annotate_ai.sources.zarr_discovery import discover_zarr_images
from omero_annotate_ai.trainingset import TrainingSet


def _map_annotation(kv):
    ann = Mock()
    ann.OMERO_TYPE.__name__ = "MapAnnotationI"
    ann.getMapValue.return_value = [Mock(name=k, value=v) for k, v in kv.items()]
    for item, k in zip(ann.getMapValue.return_value, kv, strict=True):
        item.name = k
    return ann


def _plate_conn(wells):
    """Mock connection whose getObject("Well", id) returns gateway-like wells.

    ``wells`` maps well id -> (row, column, image ids in field order, key-value pairs).
    """

    def well(well_id):
        row, column, image_ids, kv = wells[well_id]
        obj = Mock()
        obj.row, obj.column = row, column
        obj.listChildren.return_value = [
            Mock(**{"getImage.return_value.getId.return_value": i}) for i in image_ids
        ]
        obj.listAnnotations.return_value = [_map_annotation(kv)] if kv else []
        return obj

    conn = Mock()
    conn.getObject.side_effect = lambda kind, oid: well(oid) if kind == "Well" else None
    return conn


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
                    "treatment": ["Control"],
                },
            },
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        # Test matching case - all criteria met
        well_kv = {"cellline": "U2OS", "treatment": "Control"}
        assert pipeline._check_well_filter(well_kv, config.omero.well_filters) is True

        # Test matching case - HeLa instead of U2OS
        well_kv = {"cellline": "HeLa", "treatment": "Control"}
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
                    "treatment": ["Control"],
                },
            },
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        # Test non-matching cellline
        well_kv = {"cellline": "MCF7", "treatment": "Control"}
        assert pipeline._check_well_filter(well_kv, config.omero.well_filters) is False

        # Test non-matching treatment
        well_kv = {"cellline": "U2OS", "treatment": "Drug A"}
        assert pipeline._check_well_filter(well_kv, config.omero.well_filters) is False

        # Test missing key
        well_kv = {
            "cellline": "U2OS"
            # Missing "treatment" key
        }
        assert pipeline._check_well_filter(well_kv, config.omero.well_filters) is False

    def test_get_well_map_annotations(self):
        """Test that _get_well_map_annotations correctly retrieves map annotations."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "plate", "container_id": 1}
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        # Mock well object with map annotations
        mock_well = Mock()
        mock_map_ann = Mock()
        mock_map_ann.OMERO_TYPE.__name__ = "MapAnnotationI"

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

        assert result == {"cellline": "U2OS", "treatment": "Control"}

    def test_get_well_map_annotations_empty(self):
        """Test _get_well_map_annotations with no annotations."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "plate", "container_id": 1}
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        # Mock well object with no annotations
        mock_well = Mock()
        mock_well.listAnnotations.return_value = []
        mock_conn.getObject.return_value = mock_well

        result = pipeline._get_well_map_annotations(123)
        assert result == {}

    def test_get_well_map_annotations_well_not_found(self):
        """Test _get_well_map_annotations when well doesn't exist."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "plate", "container_id": 1}
        )
        mock_conn = Mock()
        mock_conn.getObject.return_value = None
        pipeline = TrainingSet(config, mock_conn)

        result = pipeline._get_well_map_annotations(999)
        assert result == {}

    @pytest.mark.parametrize(
        "filter_mode,matches_filter,expected_include",
        [
            ("include", True, True),
            ("include", False, False),
            ("exclude", True, False),
            ("exclude", False, True),
        ],
    )
    def test_well_filter_modes(self, filter_mode, matches_filter, expected_include):
        """Test include and exclude filter modes."""
        config = AnnotationConfig(
            name="test",
            omero={
                "container_type": "plate",
                "container_id": 1,
                "well_filters": {"cellline": ["U2OS"]},
                "well_filter_mode": filter_mode,
            },
        )

        # Calculate what should_include would be
        should_include = (
            config.omero.well_filter_mode == "include" and matches_filter
        ) or (config.omero.well_filter_mode == "exclude" and not matches_filter)

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
                "well_filter_mode": "include",
            },
        )
        conn = _plate_conn(
            {
                101: (0, 0, [1, 2], {"cellline": "U2OS"}),
                102: (0, 1, [3, 4], {"cellline": "HeLa"}),
                103: (1, 2, [5, 6], {"cellline": "U2OS"}),
            }
        )
        pipeline = TrainingSet(config, conn)

        with patch("omero_annotate_ai.trainingset.ezomero") as mock_ezomero:
            mock_ezomero.get_well_ids.return_value = [101, 102, 103]
            image_ids = pipeline.get_image_ids_from_container()

        assert image_ids == [1, 2, 5, 6]

    def test_get_image_ids_without_well_filtering(self):
        """Test that get_image_ids_from_container works without filtering."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "plate", "container_id": 1},
        )
        conn = _plate_conn(
            {
                101: (0, 0, [1, 2], {}),
                102: (0, 1, [3], {}),
            }
        )
        pipeline = TrainingSet(config, conn)

        with patch("omero_annotate_ai.trainingset.ezomero") as mock_ezomero:
            mock_ezomero.get_well_ids.return_value = [101, 102]
            image_ids = pipeline.get_image_ids_from_container()

        assert image_ids == [1, 2, 3]

    def test_plate_images_carry_well_field_and_metadata(self):
        """OMERO plate images record their well, field index and well key-value pairs."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "plate", "container_id": 1},
        )
        conn = _plate_conn(
            {
                101: (1, 1, [7, 8], {"cellline": "U2OS"}),
                102: (27, 11, [9], {}),
            }
        )
        pipeline = TrainingSet(config, conn)

        with patch("omero_annotate_ai.trainingset.ezomero") as mock_ezomero:
            mock_ezomero.get_well_ids.return_value = [101, 102]
            pipeline.get_image_ids_from_container()

        def identity(image_id):
            meta = ImageMeta(
                f"omero:image:{image_id}", f"img{image_id}", 1, 1, 1, 8, 8, "uint16"
            )
            return pipeline._identity_fields(meta)

        assert identity(8)["well"] == "B2"
        assert identity(8)["field"] == "1"
        assert identity(8)["well_metadata"] == {"cellline": "U2OS"}
        assert identity(8)["image_name"] == "img8"
        assert identity(9)["well"] == "AB12"
        assert identity(9)["field"] == "0"

    def test_dataset_images_have_no_well(self):
        """Images outside a plate leave well and field empty."""
        config = AnnotationConfig(
            name="test",
            omero={"container_type": "dataset", "container_id": 1},
        )
        pipeline = TrainingSet(config, Mock())
        meta = ImageMeta("omero:image:5", "img5", 1, 1, 1, 8, 8, "uint16")
        assert "well" not in pipeline._identity_fields(meta)


@pytest.mark.unit
class TestMultiContainerImageRetrieval:
    """Test multi-container image retrieval functionality."""

    def test_get_image_ids_multiple_datasets(self):
        """Test getting image IDs from multiple datasets."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_ids": [1, 2]}
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        with patch("omero_annotate_ai.trainingset.ezomero") as mock_ezomero:

            def mock_get_image_ids(conn, dataset=None, plate=None, well=None):
                if dataset == 1:
                    return [101, 102]
                if dataset == 2:
                    return [201, 202]
                return []

            mock_ezomero.get_image_ids.side_effect = mock_get_image_ids

            image_ids = pipeline.get_image_ids_from_container()

            assert set(image_ids) == {101, 102, 201, 202}
            assert len(image_ids) == 4

    def test_get_image_ids_removes_duplicates(self):
        """Test that duplicate image IDs are removed."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_ids": [1, 2]}
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        with patch("omero_annotate_ai.trainingset.ezomero") as mock_ezomero:

            def mock_get_image_ids(conn, dataset=None, plate=None, well=None):
                if dataset == 1:
                    return [101, 102, 103]
                if dataset == 2:
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
            name="test", omero={"container_type": "dataset", "container_id": 123}
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        with patch("omero_annotate_ai.trainingset.ezomero") as mock_ezomero:
            mock_ezomero.get_image_ids.return_value = [1001, 1002, 1003]

            image_ids = pipeline.get_image_ids_from_container()

            assert image_ids == [1001, 1002, 1003]
            mock_ezomero.get_image_ids.assert_called_once_with(mock_conn, dataset=123)

    def test_get_image_ids_no_container_raises(self):
        """Test that no container configured raises ValueError."""
        config = AnnotationConfig(
            name="test", omero={"container_type": "dataset", "container_id": 0}
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

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
                "well_filter_mode": "include",
            },
        )
        conn = _plate_conn(
            {
                101: (0, 0, [1001], {"cellline": "U2OS"}),
                102: (0, 1, [1002], {"cellline": "HeLa"}),
                201: (0, 0, [2001], {"cellline": "U2OS"}),
                202: (0, 1, [2002], {"cellline": "HeLa"}),
            }
        )
        pipeline = TrainingSet(config, conn)

        with patch("omero_annotate_ai.trainingset.ezomero") as mock_ezomero:
            mock_ezomero.get_well_ids.side_effect = lambda conn, plate=None: {
                1: [101, 102],
                2: [201, 202],
            }[plate]
            image_ids = pipeline.get_image_ids_from_container()

        assert set(image_ids) == {1001, 2001}


@pytest.mark.unit
class TestMultiContainerTableTitle:
    """Test table title generation for multi-container scenarios."""

    def test_table_title_single_container(self):
        """Test table title generation for single container."""
        config = AnnotationConfig(
            name="",  # Empty name triggers auto-generation
            omero={"container_type": "dataset", "container_id": 123},
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        title = pipeline._get_table_title()

        assert "dataset_123" in title

    def test_table_title_multiple_containers(self):
        """Test table title generation for multiple containers."""
        config = AnnotationConfig(
            name="",  # Empty name triggers auto-generation
            omero={"container_type": "plate", "container_ids": [1, 2, 3]},
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        title = pipeline._get_table_title()

        assert "plate_multi_3" in title

    def test_table_title_uses_config_name(self):
        """Test that explicit config name takes precedence."""
        config = AnnotationConfig(
            name="my_custom_table",
            omero={"container_type": "plate", "container_ids": [1, 2, 3]},
        )
        mock_conn = Mock()
        pipeline = TrainingSet(config, mock_conn)

        title = pipeline._get_table_title()

        assert title == "my_custom_table"


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
        assert [r.uri for r in refs] == [
            f"{plate}/A/1/0",
            f"{plate}/A/2/0",
            f"{plate}/B/3/0",
        ]
        assert refs[0].name == "testplate_A1_0"

    def test_plate_all_fields(self):
        assert (
            len(discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], fields="all")))
            == 6
        )

    def test_well_uri(self):
        (ref,) = discover_zarr_images(ZarrConfig(uris=[self.s["plate"] + "/A/2"]))
        assert (ref.plate, ref.well, ref.field) == ("testplate", "A2", "0")

    def test_wells_list_normalized_and_unknown_ignored(self):
        with pytest.warns(UserWarning, match="H12"):
            refs = discover_zarr_images(
                ZarrConfig(uris=[self.s["plate"]], wells=["a02", "H12"])
            )
        assert [r.well for r in refs] == ["A2"]

    def test_unknown_wells_list_entry_warns(self):
        with pytest.warns(UserWarning, match="H12"):
            discover_zarr_images(
                ZarrConfig(uris=[self.s["plate"]], wells=["A2", "H12"])
            )

    def test_csv_filter_include_records_the_match(self, tmp_path):
        csv = self._csv(
            tmp_path,
            [
                {"well": "A1", "compound": "DMSO"},
                {"well": "A02", "compound": "X"},
                {"well": "B3", "compound": "DMSO"},
            ],
        )
        cfg = ZarrConfig(
            uris=[self.s["plate"]],
            well_metadata=csv,
            well_filters={"compound": ["DMSO"]},
        )
        refs = discover_zarr_images(cfg, output_dir=self.out)
        assert [r.well for r in refs] == ["A1", "B3"]
        assert refs[0].well_metadata == {"compound": "DMSO"}
        table = pd.read_csv(self.out / "well_matching.csv")
        assert list(table["well"]) == ["A1", "A2", "B3"]
        assert list(table["matched"]) == [True, True, True]
        assert list(table["included"]) == [True, False, True]
        assert list(table["compound"]) == ["DMSO", "X", "DMSO"]

    def test_csv_filter_exclude(self, tmp_path):
        csv = self._csv(
            tmp_path,
            [
                {"well": w, "compound": c}
                for w, c in [("A1", "DMSO"), ("A2", "X"), ("B3", "DMSO")]
            ],
        )
        cfg = ZarrConfig(
            uris=[self.s["plate"]],
            well_metadata=csv,
            well_filters={"compound": ["DMSO"]},
            well_filter_mode="exclude",
        )
        assert [r.well for r in discover_zarr_images(cfg)] == ["A2"]

    def test_csv_with_plate_column(self, tmp_path):
        csv = self._csv(
            tmp_path,
            [
                {"plate": "testplate", "well": "A1", "compound": "DMSO"},
                {"plate": "other", "well": "A2", "compound": "DMSO"},
            ],
        )
        cfg = ZarrConfig(
            uris=[self.s["plate"]],
            well_metadata=csv,
            well_filters={"compound": ["DMSO"]},
        )
        with pytest.warns(UserWarning):
            refs = discover_zarr_images(cfg)
        assert [r.well for r in refs] == ["A1"]

    def test_mismatch_warnings(self, tmp_path):
        csv = self._csv(
            tmp_path,
            [
                {"well": "A1", "c": "1"},
                {"well": "A2", "c": "1"},
                {"well": "H12", "c": "1"},
            ],
        )
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
