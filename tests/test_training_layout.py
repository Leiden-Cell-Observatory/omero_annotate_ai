"""Tests for the training layout writer."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from omero_annotate_ai.processing.training_layout import (
    AnnotationRecord,
    ArraySource,
    FileSource,
    assert_output_dir_is_separate,
    clean_layout,
    layout_folders,
)


@pytest.mark.unit
class TestLayoutFolders:
    """Folder names for each layout."""

    def test_split_layout_single_channel(self):
        assert layout_folders("split") == [
            "train_input",
            "train_label",
            "val_input",
            "val_label",
        ]

    def test_split_layout_separate_channels(self):
        folders = layout_folders("split", uses_separate_channels=True)

        assert "train_annotation_input" in folders
        assert "val_annotation_input" in folders

    def test_split_layout_with_test(self):
        folders = layout_folders("split", include_test=True)

        assert "test_input" in folders
        assert "test_label" in folders

    def test_cellpose_layout_not_implemented(self):
        with pytest.raises(NotImplementedError):
            layout_folders("cellpose")

    def test_unknown_layout_raises(self):
        with pytest.raises(ValueError, match="Unknown layout"):
            layout_folders("nonsense")


@pytest.mark.unit
class TestCleanLayout:
    """Cleaning removes exactly the folders the layout writes."""

    def test_removes_stale_files(self, tmp_path):
        stale = tmp_path / "train_input" / "old.tif"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        clean_layout(tmp_path, "split")

        assert not stale.exists()

    def test_safe_when_absent(self, tmp_path):
        clean_layout(tmp_path, "split", uses_separate_channels=True)


@pytest.mark.unit
class TestSources:
    """Sources materialize a file at a destination."""

    def test_file_source_copy(self, tmp_path):
        src = tmp_path / "src.tif"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.tif"

        action = FileSource(src).write_to(dst, file_mode="copy")

        assert action == "copy"
        assert dst.read_bytes() == b"data"
        assert src.exists()

    def test_file_source_move(self, tmp_path):
        src = tmp_path / "src.tif"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.tif"

        action = FileSource(src).write_to(dst, file_mode="move")

        assert action == "move"
        assert dst.read_bytes() == b"data"
        assert not src.exists()

    def test_file_source_symlink(self, tmp_path):
        src = tmp_path / "src.tif"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.tif"

        action = FileSource(src).write_to(dst, file_mode="symlink")

        assert action == "symlink"
        assert dst.is_symlink()
        assert dst.read_bytes() == b"data"

    def test_file_source_symlink_falls_back_to_copy(self, tmp_path):
        """Windows without Developer Mode raises OSError on symlink_to."""
        src = tmp_path / "src.tif"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.tif"

        with patch.object(Path, "symlink_to", side_effect=OSError("no privilege")):
            action = FileSource(src).write_to(dst, file_mode="symlink")

        assert action == "copy (symlink fallback)"
        assert not dst.is_symlink()
        assert dst.read_bytes() == b"data"

    def test_array_source_writes_tiff(self, tmp_path):
        from tifffile import imread

        array = np.arange(4, dtype=np.uint8).reshape(2, 2)
        dst = tmp_path / "dst.tif"

        action = ArraySource(lambda: array).write_to(dst)

        assert action == "write"
        assert np.array_equal(imread(str(dst)), array)

    def test_array_source_is_lazy(self, tmp_path):
        """The loader must not run until write_to is called."""
        calls = []

        def loader():
            calls.append(1)
            return np.zeros((2, 2), dtype=np.uint8)

        source = ArraySource(loader)
        assert calls == []

        source.write_to(tmp_path / "dst.tif")
        assert calls == [1]


@pytest.mark.unit
class TestAnnotationRecord:
    """Records pair an image with its label under a stable id."""

    def test_record_holds_id_and_category(self, tmp_path):
        src = tmp_path / "a.tif"
        src.write_bytes(b"x")

        record = AnnotationRecord(
            annotation_id=7,
            category="training",
            image=FileSource(src),
            label=FileSource(src),
        )

        assert record.annotation_id == 7
        assert record.category == "training"
        assert record.annotation_image is None

    def test_rejects_unknown_category(self, tmp_path):
        src = tmp_path / "a.tif"
        src.write_bytes(b"x")

        with pytest.raises(ValueError, match="Unknown category"):
            AnnotationRecord(
                annotation_id=1,
                category="bogus",
                image=FileSource(src),
                label=FileSource(src),
            )


@pytest.mark.unit
class TestOutputDirGuard:
    """Training output must not live inside the annotation directory."""

    def test_rejects_nested_output_dir(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()

        with pytest.raises(ValueError, match="must not be inside"):
            assert_output_dir_is_separate(annotation_dir / "training", annotation_dir)

    def test_rejects_identical_dirs(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()

        with pytest.raises(ValueError, match="must not be inside"):
            assert_output_dir_is_separate(annotation_dir, annotation_dir)

    def test_allows_sibling(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()

        assert_output_dir_is_separate(tmp_path / "project_training", annotation_dir)
