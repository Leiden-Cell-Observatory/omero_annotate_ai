"""A training set written by the QuPath extension loads here (formats stay in step).

``tests/data/qupath_set`` was written by the extension's own classes (see
``qupath-extension/README.md``): a local set with one annotated unit, one excluded
unit and one open unit, and one finished session. If a change to the row columns,
the config schema, the pixel digest or the folder layout breaks this test, the
extension must change with it.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from omero_annotate_ai import export_training_set
from omero_annotate_ai.bundle import file_digest, input_path, mask_path
from omero_annotate_ai.core.annotation_config import AnnotationConfig
from omero_annotate_ai.rows import COLUMNS, rows_from_frame
from omero_annotate_ai.trainingset import summarize

QUPATH_SET = Path(__file__).parent / "data" / "qupath_set"


@pytest.fixture
def qupath_set(tmp_path) -> Path:
    """A copy of the set, so nothing is written into the repository."""
    folder = tmp_path / "1603434b_qupath-fixture"
    shutil.copytree(QUPATH_SET, folder)
    return folder


def _rows(folder: Path):
    frame = pd.read_csv(folder / "tracking_table.csv", keep_default_na=False)
    return frame, rows_from_frame(frame)


@pytest.mark.unit
class TestQuPathSet:
    def test_config_is_a_valid_schema_3_config(self):
        config = AnnotationConfig.from_yaml(QUPATH_SET / "config.yaml")
        assert config.name == "QuPath fixture"
        assert config.storage == "local"
        assert config.identity.local_uuid == "1603434b"
        assert config.identity.table_id is None
        assert config.training.seed == 42
        assert config.spatial_coverage.channels == [0]
        assert not config.spatial_coverage.three_d
        assert config.annotation_methodology.annotation_tool.name == "QuPath"
        assert config.annotation_methodology.annotation_criteria == "every nucleus"

    def test_session_record_is_read(self):
        config = AnnotationConfig.from_yaml(QUPATH_SET / "config.yaml")
        (session,) = config.sessions
        assert (session.id, session.kind, session.status) == (1, "annotate", "finished")
        assert session.n_annotated == 1
        assert session.note == "nuclei only"
        assert session.started <= session.ended
        assert session.software_version.startswith("qupath-extension-omero-annotate-ai")

    def test_config_survives_a_python_round_trip(self):
        config = AnnotationConfig.from_yaml(QUPATH_SET / "config.yaml")
        again = AnnotationConfig.from_yaml(config.to_yaml())
        assert again == config

    def test_table_has_the_columns_of_rows_py(self):
        frame, _ = _rows(QUPATH_SET)
        assert list(frame.columns) == [*COLUMNS, "error"]

    def test_rows_are_read_with_their_state(self):
        _, rows = _rows(QUPATH_SET)
        annotated, excluded, pending = rows
        assert annotated.processed and not annotated.excluded
        assert annotated.category == "training"
        assert annotated.session == 1
        assert annotated.image_id is None  # a local image file, not on OMERO
        assert annotated.image_uri == "/data/img2.tif"
        assert annotated.label_id is None and annotated.roi_id is None
        assert (annotated.timepoint, annotated.z_slice, annotated.channel) == (0, 0, 0)
        assert excluded.excluded and excluded.mask_digest is None
        assert not pending.processed and pending.category == "validation"
        assert pending.annotation_created_at is None and pending.session == -1

    def test_pixel_digests_match(self):
        _, rows = _rows(QUPATH_SET)
        annotated = rows[0]
        mask_file = mask_path(QUPATH_SET, annotated.annotation_id)
        input_file = input_path(QUPATH_SET, annotated.annotation_id)
        assert file_digest(mask_file) == annotated.mask_digest
        assert file_digest(input_file) == annotated.input_digest

    def test_mask_is_a_2d_label_image(self):
        _, rows = _rows(QUPATH_SET)
        mask = tifffile.imread(mask_path(QUPATH_SET, rows[0].annotation_id))
        image = tifffile.imread(input_path(QUPATH_SET, rows[0].annotation_id))
        assert mask.shape == image.shape == (12, 16)
        assert mask.dtype == np.uint16 and image.dtype == np.uint16
        assert sorted(np.unique(mask)) == [0, 1, 2]
        assert (mask == 1).sum() == 24 and (mask == 2).sum() == 12

    def test_summarize_counts_done_and_excluded(self, qupath_set):
        summary = summarize(qupath_set)
        assert summary.name == "QuPath fixture"
        assert summary.storage == "local"
        assert summary.progress == {
            "training": (2, 2),
            "validation": (0, 1),
            "test": (0, 0),
        }
        assert summary.excluded["training"] == 1
        assert [s.note for s in summary.sessions] == ["nuclei only"]

    def test_export_writes_the_annotated_unit_only(self, qupath_set, tmp_path):
        out = tmp_path / "training_data"
        result = export_training_set(qupath_set, out)
        stats = result.get("stats", result)
        assert stats["n_training_images"] == 1
        assert stats["n_training_labels"] == 1
        assert stats["n_val_images"] == 0
        (image,) = (out / "train_input").glob("*.tif")
        (label,) = (out / "train_label").glob("*.tif")
        assert image.name == label.name
        assert np.array_equal(
            tifffile.imread(label),
            tifffile.imread(mask_path(qupath_set, image.stem)),
        )
