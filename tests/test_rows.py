"""Rows and their one table codec (omero_annotate_ai.rows)."""

import pandas as pd
import pytest

from omero_annotate_ai.planning import ImageIdentity, Unit
from omero_annotate_ai.rows import (
    OMERO_ID_UNSET,
    Row,
    progress_summary,
    rows_from_frame,
    rows_to_frame,
    split_report,
    split_summary,
)


def _zarr_row(**fields):
    return Row(
        image_uri="https://h/p.zarr/B/2/0",
        image_name="p_B2_0",
        annotation_id="abc_0_0",
        level=1,
        well="B2",
        field="0",
        well_metadata={"compound": "DMSO"},
        **fields,
    )


@pytest.mark.unit
class TestRow:
    def test_uri_is_derived_from_omero_id(self):
        assert Row(image_id=7, image_name="a", annotation_id="x").image_uri == (
            "omero:image:7"
        )

    def test_zarr_row_has_no_omero_id(self):
        assert _zarr_row().image_id is None

    def test_unit_round_trip(self):
        unit = Unit(
            annotation_id="u1",
            identity=ImageIdentity(
                image_uri="omero:image:3", image_name="i", image_id=3
            ),
            category="test",
            timepoint=1,
            z_start=2,
            z_end=4,
            z_length=3,
            channel=1,
            is_volumetric=True,
            patch=(5, 6, 32, 16),
        )
        row = Row.from_unit(unit)
        assert (row.is_patch, row.patch_width, row.z_slice) == (True, 32, 2)
        assert row.unit() == unit

    def test_mark_processed_keeps_first_time(self):
        row = Row(image_id=1, image_name="a", annotation_id="x")
        row.mark_processed()
        first = row.annotation_created_at
        row.mark_processed()
        assert row.processed and row.annotation_created_at == first


@pytest.mark.unit
class TestCodec:
    def test_round_trip_keeps_identity_state_and_category(self):
        row = _zarr_row(
            category="test",
            processed=True,
            session=2,
            mask_digest="d",
            input_iscc="ISCC:EEA4XSZUGQ2MXS6B",
            mask_iscc="ISCC:EEA6IF2FC2OREY73",
        )
        row.label_id, row.roi_id = 11, 12
        frame = rows_to_frame([row])
        assert frame.loc[0, "image_id"] == OMERO_ID_UNSET
        assert frame.loc[0, "well_metadata"] == '{"compound": "DMSO"}'
        (again,) = rows_from_frame(frame)
        assert again == row

    def test_unset_ids_are_zero_on_the_table_and_none_back(self):
        frame = rows_to_frame([Row(image_id=5, image_name="a", annotation_id="x")])
        assert (frame.loc[0, "label_id"], frame.loc[0, "roi_id"]) == (0, 0)
        (again,) = rows_from_frame(frame)
        assert (again.label_id, again.roi_id) == (None, None)

    def test_error_column_only_in_the_local_csv(self, tmp_path):
        row = Row(image_id=5, image_name="a", annotation_id="x", error="boom")
        assert "error" not in rows_to_frame([row]).columns
        path = tmp_path / "t.csv"
        rows_to_frame([row], local=True).to_csv(path, index=False)
        (again,) = rows_from_frame(pd.read_csv(path, keep_default_na=False))
        assert again.error == "boom"
        assert (again.well, again.field, again.well_metadata) == ("", "", {})

    def test_tables_without_annotation_id_are_refused(self):
        frame = pd.DataFrame([{"image_id": 101, "image_name": "img"}])
        with pytest.raises(ValueError, match="0.5.0"):
            rows_from_frame(frame)


@pytest.mark.unit
class TestReports:
    def _rows(self):
        return [
            Row(image_id=1, image_name="a", annotation_id="1a", category="training"),
            Row(image_id=1, image_name="a", annotation_id="1b", category="test"),
            Row(
                image_id=2,
                image_name="b",
                annotation_id="2a",
                category="validation",
                processed=True,
            ),
        ]

    def test_progress(self):
        assert progress_summary(self._rows())["completed_units"] == 1

    def test_excluded_units_count_as_done(self):
        rows = self._rows()
        rows[0].label_id = 7
        rows[0].claimed_by = "1@me/host"
        rows[0].mark_excluded(session=3)

        summary = progress_summary(rows)

        assert (summary["completed_units"], summary["excluded_units"]) == (2, 1)
        assert summary["pending_units"] == 1
        (again,) = rows_from_frame(rows_to_frame(rows[:1]))
        assert again.excluded and again.annotation_type == "excluded"
        assert (again.label_id, again.session, again.claimed_by) == (None, 3, "")
        assert not rows[2].excluded  # annotated, not excluded

    def test_split_report_lists_images_in_several_splits(self):
        report = split_report(self._rows(), "image")
        assert report["images"]["spanning_splits"] == ["omero:image:1"]
        assert "1 of 2 images" in split_summary(report)
