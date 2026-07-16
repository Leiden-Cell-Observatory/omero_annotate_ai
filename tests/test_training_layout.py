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


@pytest.mark.unit
class TestWriteTrainingLayout:
    """Records are written into the split layout, paired by annotation_id."""

    def _record(self, tmp_path, annotation_id, category, with_annotation_image=False):
        image = tmp_path / f"src_{annotation_id}_img.tif"
        label = tmp_path / f"src_{annotation_id}_lbl.tif"
        image.write_bytes(b"img")
        label.write_bytes(b"lbl")

        annotation_image = None
        if with_annotation_image:
            ann = tmp_path / f"src_{annotation_id}_ann.tif"
            ann.write_bytes(b"ann")
            annotation_image = FileSource(ann)

        return AnnotationRecord(
            annotation_id=annotation_id,
            category=category,
            image=FileSource(image),
            label=FileSource(label),
            annotation_image=annotation_image,
        )

    def test_writes_train_and_val_split(self, tmp_path):
        from omero_annotate_ai.processing.training_layout import write_training_layout

        out = tmp_path / "training"
        records = [
            self._record(tmp_path, 1, "training"),
            self._record(tmp_path, 2, "validation"),
        ]

        created_dirs, stats = write_training_layout(records, out)

        assert (out / "train_input" / "1.tif").read_bytes() == b"img"
        assert (out / "train_label" / "1.tif").read_bytes() == b"lbl"
        assert (out / "val_input" / "2.tif").read_bytes() == b"img"
        assert (out / "val_label" / "2.tif").read_bytes() == b"lbl"
        assert stats["n_training_images"] == 1
        assert stats["n_val_images"] == 1
        assert created_dirs["train_input"] == out / "train_input"

    def test_image_and_label_share_a_name(self, tmp_path):
        """Pairing is by annotation_id, so a gap cannot shift later pairs.

        The OMERO producer used to name files by loop index and write the image
        before the label. A missing label left an orphan image, and micro-SAM,
        which pairs raw_paths to label_paths by sorted filename, then mispaired
        every subsequent image.
        """
        from omero_annotate_ai.processing.training_layout import write_training_layout

        out = tmp_path / "training"
        records = [
            self._record(tmp_path, 10, "training"),
            self._record(tmp_path, 30, "training"),
        ]

        write_training_layout(records, out)

        images = sorted(p.name for p in (out / "train_input").glob("*.tif"))
        labels = sorted(p.name for p in (out / "train_label").glob("*.tif"))

        assert images == labels == ["10.tif", "30.tif"]

    def test_annotation_image_written_for_separate_channels(self, tmp_path):
        from omero_annotate_ai.processing.training_layout import write_training_layout

        out = tmp_path / "training"
        records = [self._record(tmp_path, 1, "training", with_annotation_image=True)]

        write_training_layout(records, out, uses_separate_channels=True)

        assert (out / "train_annotation_input" / "1.tif").read_bytes() == b"ann"

    def test_test_category_skipped_unless_included(self, tmp_path):
        from omero_annotate_ai.processing.training_layout import write_training_layout

        out = tmp_path / "training"
        records = [self._record(tmp_path, 1, "test")]

        _, stats = write_training_layout(records, out, include_test=False)

        assert not (out / "test_input").exists()
        assert stats["n_skipped"] == 1

    def test_test_category_written_when_included(self, tmp_path):
        from omero_annotate_ai.processing.training_layout import write_training_layout

        out = tmp_path / "training"
        records = [self._record(tmp_path, 1, "test")]

        _, stats = write_training_layout(records, out, include_test=True)

        assert (out / "test_input" / "1.tif").exists()
        assert stats["n_test_images"] == 1

    def test_clean_existing_removes_stale_data(self, tmp_path):
        from omero_annotate_ai.processing.training_layout import write_training_layout

        out = tmp_path / "training"
        stale = out / "train_input" / "999.tif"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        write_training_layout([self._record(tmp_path, 1, "training")], out, clean_existing=True)

        assert not stale.exists()
        assert (out / "train_input" / "1.tif").exists()

    def test_records_the_file_operation(self, tmp_path):
        from omero_annotate_ai.processing.training_layout import write_training_layout

        out = tmp_path / "training"
        records = [self._record(tmp_path, 1, "training")]

        _, stats = write_training_layout(records, out, file_mode="symlink")

        assert stats["file_operations"]["symlink"] == 2  # image + label
        # Compare as Paths: file_mapping stores str(Path), whose separator is
        # backslash on Windows.
        assert Path(stats["file_mapping"]["1"]["image"]) == out / "train_input" / "1.tif"

    def test_array_source_records_are_written(self, tmp_path):
        from tifffile import imread

        from omero_annotate_ai.processing.training_layout import write_training_layout

        out = tmp_path / "training"
        array = np.full((2, 2), 7, dtype=np.uint8)
        record = AnnotationRecord(
            annotation_id=5,
            category="training",
            image=ArraySource(lambda: array),
            label=ArraySource(lambda: array),
        )

        write_training_layout([record], out)

        assert np.array_equal(imread(str(out / "train_input" / "5.tif")), array)


@pytest.mark.unit
class TestOutputDirGuardAliases:
    """The guard asks the filesystem, so aliases of the annotation dir are caught.

    Plain path comparison is not enough: on case-insensitive filesystems (Windows,
    macOS) /x/Project and /x/project are one directory, and resolve() does not
    case-fold. samefile() gets that right without wrongly rejecting /x/Project vs
    /x/project on a case-SENSITIVE filesystem, where they really are different.
    """

    def test_rejects_output_under_an_aliased_annotation_dir(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(annotation_dir, target_is_directory=True)

        with pytest.raises(ValueError, match="must not be inside"):
            assert_output_dir_is_separate(alias / "out", annotation_dir)

    def test_rejects_the_aliased_dir_itself(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(annotation_dir, target_is_directory=True)

        with pytest.raises(ValueError, match="must not be inside"):
            assert_output_dir_is_separate(alias, annotation_dir)

    def test_allows_a_genuine_sibling(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()

        assert_output_dir_is_separate(tmp_path / "project_training", annotation_dir)

    def test_allows_a_sibling_sharing_a_name_prefix(self, tmp_path):
        """<annotation_dir>_training/ is the default target; it must not be rejected."""
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        sibling = tmp_path / "project_training"
        sibling.mkdir()

        assert_output_dir_is_separate(sibling, annotation_dir)
