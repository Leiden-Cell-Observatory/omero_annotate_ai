# Folder Layout Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the annotation and training phases one coherent folder layout, written by one piece of code, and delete the duplicate/dead code that the current three-layout mess accumulated.

**Architecture:** A new `processing/training_layout.py` owns the training layout: *sources* (how a file gets materialized — copied/symlinked from disk, or written from an OMERO plane) and `_write_training_layout()` (folder names, cleaning, splitting, stats). The two existing entry points keep their distinct signatures but stop building layouts themselves — they resolve their inputs into `AnnotationRecord`s and hand them to the writer.

**Tech Stack:** Python 3.11, pydantic, pandas, numpy, tifffile, ezomero, pytest, pixi.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-13-folder-layout-unification-design.md`.
- Run tests with `pixi run -e dev pytest tests/ -v`. Never plain `pytest`.
- Backwards compatibility for local folder names and the Python API is **explicitly waived**. Do not add shims.
- The **one** exception: tracking tables already written to OMERO must still load. The `label_input_id` column gets a read-side fallback (Task 7). Nothing else gets a fallback.
- Training files are named `{annotation_id}.tif`. Never by loop index.
- Annotation phase: `annotation_input/`, `model_input/`, `annotation_output/`, `sam_embeddings/`.
- Training phase: `train_input/`, `train_label/`, `val_input/`, `val_label/`, `test_*`, `*_annotation_input/`.
- Tests use `@pytest.mark.unit` and go in the test file that maps to the source module (see CLAUDE.md). Do not create new test files except where this plan says to.

---

### Task 1: Sources, records, and folder names

Creates the vocabulary everything else uses. Pure and fully unit-testable — no OMERO, no filesystem beyond `tmp_path`.

**Files:**
- Create: `src/omero_annotate_ai/processing/training_layout.py`
- Create: `tests/test_training_layout.py`
- Modify: `src/omero_annotate_ai/processing/training_functions.py` (re-export only, see Step 5)

**Interfaces:**
- Consumes: `_create_file_link_or_copy` — **moved** here from `processing/training_functions.py`. It is cut, not copied: `training_functions.py` imports it back from this module, so the existing `reorganize_local_data_for_training` keeps working through Tasks 2-3 without a second copy of the body existing anywhere.
- Produces: `FileSource`, `ArraySource`, `AnnotationRecord`, `layout_folders()`, `clean_layout()`, `split_folders_for()`, `assert_output_dir_is_separate()`, `CATEGORIES`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_layout.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_training_layout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'omero_annotate_ai.processing.training_layout'`

- [ ] **Step 3: Write the implementation**

Create `src/omero_annotate_ai/processing/training_layout.py`:

```python
"""On-disk layout for prepared training data.

This module owns every decision about where training files go: folder names,
cleaning, and how a file is materialized at its destination. Both training-data
producers (OMERO download and offline reorganization) route through it, so the
layout is defined exactly once.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from tifffile import imwrite

CATEGORIES = ("training", "validation", "test")

# Per-category folder names: (image, label, annotation-channel image)
_SPLIT_FOLDERS = {
    "training": ("train_input", "train_label", "train_annotation_input"),
    "validation": ("val_input", "val_label", "val_annotation_input"),
    "test": ("test_input", "test_label", "test_annotation_input"),
}


def split_folders_for(category: str) -> tuple:
    """Return (image, label, annotation_image) folder names for a category."""
    if category not in _SPLIT_FOLDERS:
        raise ValueError(f"Unknown category: {category}. Expected one of {CATEGORIES}")
    return _SPLIT_FOLDERS[category]


def layout_folders(
    layout: str = "split",
    uses_separate_channels: bool = False,
    include_test: bool = False,
) -> List[str]:
    """
    Return every folder name a layout writes, relative to the output directory.

    Args:
        layout: "split" (micro-SAM, BiaPy) or "cellpose" (not yet implemented)
        uses_separate_channels: Whether the annotation channel is carried along
        include_test: Whether a test split is written

    Returns:
        Folder names in a stable order

    Raises:
        NotImplementedError: For layout="cellpose"
        ValueError: For an unknown layout
    """
    if layout == "cellpose":
        raise NotImplementedError(
            "The cellpose layout (image and {id}_masks.tif in one folder) is "
            "specified but not implemented. Use layout='split'."
        )
    if layout != "split":
        raise ValueError(f"Unknown layout: {layout}. Expected 'split' or 'cellpose'.")

    categories = ["training", "validation"]
    if include_test:
        categories.append("test")

    folders = []
    for category in categories:
        image_dir, label_dir, annotation_dir = split_folders_for(category)
        folders += [image_dir, label_dir]
        if uses_separate_channels:
            folders.append(annotation_dir)

    return folders


def clean_layout(
    output_dir: Path,
    layout: str = "split",
    uses_separate_channels: bool = False,
    include_test: bool = False,
) -> None:
    """Remove the folders this layout writes, so a run cannot inherit stale data."""
    for folder_name in layout_folders(layout, uses_separate_channels, include_test):
        folder_path = output_dir / folder_name
        if folder_path.exists():
            shutil.rmtree(folder_path)


def assert_output_dir_is_separate(output_dir: Path, annotation_dir: Path) -> None:
    """
    Reject a training output directory that sits inside the annotation directory.

    The annotation phase writes source images the training phase must not touch.
    Cleaning the training layout inside the annotation directory would delete them.

    Raises:
        ValueError: If output_dir is annotation_dir or nested within it
    """
    output_resolved = Path(output_dir).resolve()
    annotation_resolved = Path(annotation_dir).resolve()

    if output_resolved == annotation_resolved or annotation_resolved in output_resolved.parents:
        raise ValueError(
            f"Training output directory must not be inside the annotation directory.\n"
            f"  annotation_dir: {annotation_resolved}\n"
            f"  output_dir:     {output_resolved}\n"
            f"Cleaning the training layout there would delete your source images. "
            f"Use a separate directory, e.g. {annotation_resolved.name}_training/."
        )


def _create_file_link_or_copy(src: Path, dst: Path, mode: str, logger=None) -> str:
    """
    Create a file at destination using the specified mode.

    Args:
        src: Source file path
        dst: Destination file path
        mode: One of "copy", "move", or "symlink"
        logger: Optional logger for messages

    Returns:
        String describing the action taken (e.g. "symlink", "copy (symlink fallback)")
    """
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return "symlink"
        except OSError as e:
            # Windows without developer mode or elevated privileges, or other OS issues
            if logger:
                logger.debug(f"Symlink failed ({e}), falling back to copy")
            shutil.copy2(src, dst)
            return "copy (symlink fallback)"
    elif mode == "move":
        shutil.move(str(src), str(dst))
        return "move"
    else:  # copy (default)
        shutil.copy2(src, dst)
        return "copy"


class FileSource:
    """An image that already exists on disk. Honours file_mode."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def write_to(self, dst: Path, file_mode: str = "copy", logger=None) -> str:
        return _create_file_link_or_copy(self.path, dst, file_mode, logger)


class ArraySource:
    """
    An image that must be fetched (e.g. an OMERO plane).

    The loader is called lazily, at write time, so preparing a large dataset does
    not hold every plane in memory at once. file_mode is meaningless here: there
    is no file on disk to link to.
    """

    def __init__(self, loader: Callable[[], np.ndarray]):
        self.loader = loader

    def exists(self) -> bool:
        return True

    def write_to(self, dst: Path, file_mode: str = "copy", logger=None) -> str:
        imwrite(str(dst), self.loader())
        return "write"


Source = object  # FileSource | ArraySource


@dataclass
class AnnotationRecord:
    """
    One annotated image, ready to be written into a training layout.

    Files are named by annotation_id, never by loop index, so an image and its
    label always pair by name. A record without a label is dropped by the caller
    rather than written, which is what keeps the pairing honest.
    """

    annotation_id: int
    category: str
    image: Source
    label: Source
    annotation_image: Optional[Source] = None

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(
                f"Unknown category: {self.category}. Expected one of {CATEGORIES}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_training_layout.py -v`
Expected: PASS — 16 passed

- [ ] **Step 5: Move `_create_file_link_or_copy` out of `training_functions.py`**

The function now lives in `training_layout.py` (you wrote it there in Step 3). Delete its
definition from `training_functions.py` and import it back instead, so exactly one copy of the
body exists. `reorganize_local_data_for_training` still calls it and must keep working until
Task 4 rewrites it.

In `src/omero_annotate_ai/processing/training_functions.py`, delete the whole
`def _create_file_link_or_copy(...)` block (line ~1170) and add to the imports:

```python
from .training_layout import _create_file_link_or_copy
```

Confirm exactly one definition remains:

```bash
grep -rn "def _create_file_link_or_copy" src/
```
Expected: one match, in `training_layout.py`.

- [ ] **Step 6: Run the full suite**

Run: `pixi run -e dev pytest tests/ -v`
Expected: PASS — the existing reorganize tests still pass, now calling the moved function.

- [ ] **Step 7: Commit**

```bash
git add src/omero_annotate_ai/processing/training_layout.py src/omero_annotate_ai/processing/training_functions.py tests/test_training_layout.py
git commit -m "feat: add training layout module with sources and records"
```

---

### Task 2: The layout writer

The single place that turns records into folders. This is what replaces the duplicated logic in both producers.

**Files:**
- Modify: `src/omero_annotate_ai/processing/training_layout.py`
- Modify: `tests/test_training_layout.py`

**Interfaces:**
- Consumes: `AnnotationRecord`, `FileSource`, `ArraySource`, `layout_folders`, `clean_layout`, `split_folders_for` (Task 1).
- Produces: `write_training_layout(records, output_dir, layout="split", file_mode="copy", clean_existing=True, include_test=False, uses_separate_channels=False, logger=None) -> tuple[dict[str, Path], dict[str, Any]]` returning `(created_dirs, stats)`. `created_dirs` keys are folder names (`train_input`, …) mapped to `Path`. `stats` keys: `n_training_images`, `n_training_labels`, `n_training_annotation_input`, `n_val_images`, `n_val_labels`, `n_val_annotation_input`, `n_test_images`, `n_test_labels`, `n_test_annotation_input`, `n_skipped`, `file_operations` (dict of action → count), `file_mapping` (dict of str(annotation_id) → dict of role → str path).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_layout.py`:

```python
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
        assert stats["file_mapping"]["1"]["image"].endswith("train_input/1.tif")

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_training_layout.py -k WriteTrainingLayout -v`
Expected: FAIL — `ImportError: cannot import name 'write_training_layout'`

- [ ] **Step 3: Write the implementation**

Append to `src/omero_annotate_ai/processing/training_layout.py`:

```python
def _empty_stats() -> dict:
    stats = {}
    for prefix in ("training", "val", "test"):
        stats[f"n_{prefix}_images"] = 0
        stats[f"n_{prefix}_labels"] = 0
        stats[f"n_{prefix}_annotation_input"] = 0
    stats["n_skipped"] = 0
    stats["file_operations"] = {}
    stats["file_mapping"] = {}
    return stats


# Category -> the prefix used in stats keys
_STATS_PREFIX = {"training": "training", "validation": "val", "test": "test"}


def write_training_layout(
    records: List[AnnotationRecord],
    output_dir: Path,
    layout: str = "split",
    file_mode: str = "copy",
    clean_existing: bool = True,
    include_test: bool = False,
    uses_separate_channels: bool = False,
    logger=None,
) -> tuple:
    """
    Write annotation records into a training layout.

    Files are named {annotation_id}.tif in every folder, so an image and its label
    always pair by name.

    Args:
        records: Annotation records to write
        output_dir: Target directory (must not be inside the annotation directory)
        layout: "split" or "cellpose"
        file_mode: "copy", "move" or "symlink" (ignored by ArraySource records)
        clean_existing: Remove the layout's folders before writing
        include_test: Write the test split; test records are skipped when False
        uses_separate_channels: Write the annotation-channel image alongside
        logger: Optional logger

    Returns:
        (created_dirs, stats)
    """
    output_dir = Path(output_dir)

    if clean_existing:
        clean_layout(output_dir, layout, uses_separate_channels, include_test)

    folder_names = layout_folders(layout, uses_separate_channels, include_test)
    created_dirs = {}
    for folder_name in folder_names:
        folder_path = output_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        created_dirs[folder_name] = folder_path

    stats = _empty_stats()

    def _record_operation(action: str):
        stats["file_operations"][action] = stats["file_operations"].get(action, 0) + 1

    for record in records:
        if record.category == "test" and not include_test:
            stats["n_skipped"] += 1
            continue

        image_folder, label_folder, annotation_folder = split_folders_for(record.category)
        prefix = _STATS_PREFIX[record.category]
        filename = f"{record.annotation_id}.tif"
        mapping = {}

        image_dst = created_dirs[image_folder] / filename
        _record_operation(record.image.write_to(image_dst, file_mode, logger))
        stats[f"n_{prefix}_images"] += 1
        mapping["image"] = str(image_dst)

        label_dst = created_dirs[label_folder] / filename
        _record_operation(record.label.write_to(label_dst, file_mode, logger))
        stats[f"n_{prefix}_labels"] += 1
        mapping["label"] = str(label_dst)

        if uses_separate_channels and record.annotation_image is not None:
            annotation_dst = created_dirs[annotation_folder] / filename
            _record_operation(
                record.annotation_image.write_to(annotation_dst, file_mode, logger)
            )
            stats[f"n_{prefix}_annotation_input"] += 1
            mapping["annotation_image"] = str(annotation_dst)

        stats["file_mapping"][str(record.annotation_id)] = mapping

        if logger:
            logger.debug(f"Wrote annotation {record.annotation_id} to {record.category}")

    return created_dirs, stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_training_layout.py -v`
Expected: PASS — 24 passed

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/processing/training_layout.py tests/test_training_layout.py
git commit -m "feat: add write_training_layout, the single layout writer"
```

---

### Task 3: Rename the annotation-phase folders

Must land before Task 4, because the offline producer reads these folders.

**Files:**
- Modify: `src/omero_annotate_ai/core/annotation_pipeline.py` (lines ~119-146, 789, 866, 1582-1625, 1685, 1771)
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: annotation directories `annotation_input/`, `model_input/`, `annotation_output/`, `sam_embeddings/`. `AnnotationPipeline._get_input_folders()` returns `{"annotation_input": Path}` (single channel) or `{"annotation_input": Path, "model_input": Path}` (separate channels).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
@pytest.mark.unit
class TestAnnotationFolderNames:
    """The annotation phase uses names that say what a file is for."""

    def test_single_channel_folders(self, sample_config, fake_omero_connection, tmp_path):
        from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

        sample_config.output.output_directory = str(tmp_path)
        sample_config.spatial_coverage.channels = [0]
        sample_config.spatial_coverage.label_channel = None
        sample_config.spatial_coverage.training_channels = []

        pipeline = AnnotationPipeline(sample_config, fake_omero_connection)
        folders = pipeline._get_input_folders(tmp_path)

        assert folders == {"annotation_input": tmp_path / "annotation_input"}

    def test_separate_channel_folders(self, sample_config, fake_omero_connection, tmp_path):
        from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

        sample_config.output.output_directory = str(tmp_path)
        sample_config.spatial_coverage.channels = [0, 1]
        sample_config.spatial_coverage.label_channel = 0
        sample_config.spatial_coverage.training_channels = [1]

        pipeline = AnnotationPipeline(sample_config, fake_omero_connection)
        folders = pipeline._get_input_folders(tmp_path)

        assert folders == {
            "annotation_input": tmp_path / "annotation_input",
            "model_input": tmp_path / "model_input",
        }

    def test_setup_creates_annotation_output(self, sample_config, fake_omero_connection, tmp_path):
        from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

        sample_config.output.output_directory = str(tmp_path)
        pipeline = AnnotationPipeline(sample_config, fake_omero_connection)
        pipeline._setup_directories()

        assert (tmp_path / "annotation_output").exists()
        assert not (tmp_path / "output").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pipeline.py -k AnnotationFolderNames -v`
Expected: FAIL — folders come back as `{"input": ...}` / `{"label_input": ..., "training_input": ...}`

- [ ] **Step 3: Write the implementation**

In `src/omero_annotate_ai/core/annotation_pipeline.py`, replace `_get_input_folders` (line ~119):

```python
    def _get_input_folders(self, output_path: Path) -> dict:
        """Return input folder paths for the current workflow configuration.

        Separate channels:
          - 'annotation_input': the channel you annotate (e.g. fluorescence)
          - 'model_input': the channel the model consumes (e.g. brightfield)
        Single channel:
          - 'annotation_input': serves both roles
        """
        folders = {"annotation_input": output_path / "annotation_input"}
        if self.config.spatial_coverage.uses_separate_channels():
            folders["model_input"] = output_path / "model_input"
        return folders
```

Replace `_setup_directories` (line ~135):

```python
    def _setup_directories(self):
        """Create output directories for the annotation workflow.

        Separate channels:  annotation_input/  model_input/  annotation_output/
        Single channel:     annotation_input/  annotation_output/
        micro-SAM adds:     sam_embeddings/
        """
        output_path = Path(self.config.output.output_directory)

        dirs = [output_path, output_path / "annotation_output"]
        dirs.extend(self._get_input_folders(output_path).values())
        if self.config.ai_model.framework == "micro_sam":
            dirs.append(output_path / "sam_embeddings")

        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
```

Then update every remaining hardcoded name in the same file:

- Line ~789: `output_dir = Path(self.config.output.output_directory) / "output"` → `/ "annotation_output"`
- Line ~866: `input_folder = output_path / "input"` → `output_path / "annotation_input"`
- Line ~1589: `if "label_input" in folders:` → `if "model_input" in folders:` (the separate-channel branch)
- Line ~1591: `folders["label_input"]` → `folders["annotation_input"]`
- Lines ~1598-1599: `folders["training_input"]` → `folders["model_input"]`
- Line ~1604: `folders["input"]` → `folders["annotation_input"]`
- Line ~1625: `/ "output"` → `/ "annotation_output"`
- Line ~1685: `def collect_annotations_from_disk(self, folder_pattern: str = "output")` → default `"annotation_output"`
- Line ~1771: `def get_annotation_status_from_disk(self, folder_name: str = "output")` → default `"annotation_output"`

Verify none are missed:

```bash
grep -nE '"(input|output|label_input|training_input)"' src/omero_annotate_ai/core/annotation_pipeline.py
```
Expected: no matches.

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_pipeline.py -v`
Expected: PASS. Any test asserting the old names must be updated to the new ones — that is an intended rename, not a regression.

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/core/annotation_pipeline.py tests/test_pipeline.py
git commit -m "refactor: rename annotation folders to annotation_input/model_input/annotation_output"
```

---

### Task 4: Rewire the offline producer onto records

**Files:**
- Modify: `src/omero_annotate_ai/processing/training_functions.py` (`reorganize_local_data_for_training`, line ~1201)
- Modify: `tests/test_training_functions.py`

**Interfaces:**
- Consumes: `AnnotationRecord`, `FileSource`, `write_training_layout`, `assert_output_dir_is_separate` (Tasks 1-2); annotation folders from Task 3.
- Produces: `reorganize_local_data_for_training(config, annotation_dir, output_dir=None, file_mode="copy", clean_existing=True, include_test=None, verbose=False) -> Dict[str, Any]` with result keys `base_dir`, `train_input`, `train_label`, `val_input`, `val_label`, `stats`, plus `test_*` / `*_annotation_input` when applicable.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_training_functions.py`:

```python
@pytest.mark.unit
class TestReorganizeOntoRecords:
    """The offline producer emits the unified layout."""

    def _config_with(self, tmp_path, categories):
        from omero_annotate_ai.core.annotation_config import create_default_config, ImageAnnotation

        config = create_default_config()
        config.output.output_directory = str(tmp_path)
        config.annotations = [
            ImageAnnotation(
                annotation_id=i,
                image_id=100 + i,
                image_name=f"img_{i}",
                category=category,
                processed=True,
            )
            for i, category in enumerate(categories)
        ]
        return config

    def _populate(self, annotation_dir, ids):
        (annotation_dir / "annotation_input").mkdir(parents=True, exist_ok=True)
        (annotation_dir / "annotation_output").mkdir(parents=True, exist_ok=True)
        for i in ids:
            (annotation_dir / "annotation_input" / f"{i}.tif").write_bytes(b"img")
            (annotation_dir / "annotation_output" / f"{i}_mask.tif").write_bytes(b"lbl")

    def test_writes_unified_layout(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0, 1])
        config = self._config_with(annotation_dir, ["training", "validation"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
        )

        assert (result["train_input"] / "0.tif").exists()
        assert (result["train_label"] / "0.tif").exists()
        assert (result["val_input"] / "1.tif").exists()
        assert "validation_input" not in result

    def test_rejects_output_inside_annotation_dir(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config_with(annotation_dir, ["training"])

        with pytest.raises(ValueError, match="must not be inside"):
            reorganize_local_data_for_training(
                config=config,
                annotation_dir=annotation_dir,
                output_dir=annotation_dir,
            )

    def test_defaults_to_sibling_training_dir(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config_with(annotation_dir, ["training"])

        result = reorganize_local_data_for_training(
            config=config, annotation_dir=annotation_dir
        )

        assert result["base_dir"] == tmp_path / "project_training"

    def test_symlink_mode(self, tmp_path):
        annotation_dir = tmp_path / "project"
        annotation_dir.mkdir()
        self._populate(annotation_dir, [0])
        config = self._config_with(annotation_dir, ["training"])

        result = reorganize_local_data_for_training(
            config=config,
            annotation_dir=annotation_dir,
            output_dir=tmp_path / "project_training",
            file_mode="symlink",
        )

        assert (result["train_input"] / "0.tif").is_symlink()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_training_functions.py -k ReorganizeOntoRecords -v`
Expected: FAIL — result still has `training_input`/`validation_input` keys, and no overlap guard.

- [ ] **Step 3: Write the implementation**

Replace the body of `reorganize_local_data_for_training` in `training_functions.py`. It keeps its signature and its validation, but resolves records and delegates:

```python
def reorganize_local_data_for_training(
    config: "AnnotationConfig",
    annotation_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    file_mode: Literal["copy", "move", "symlink"] = "copy",
    clean_existing: bool = True,
    include_test: Optional[bool] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Reorganize locally-stored annotation data into the training layout.

    Works entirely offline - no OMERO connection required. Reads the annotation
    phase's folders (annotation_input/, model_input/, annotation_output/) and writes
    the training layout (train_input/, train_label/, val_input/, val_label/).

    Args:
        config: AnnotationConfig with populated annotations (contains category info)
        annotation_dir: Directory containing annotation output
        output_dir: Target directory. Must not be inside annotation_dir.
            Defaults to the sibling <annotation_dir>_training/.
        file_mode: "copy" (default), "move", or "symlink" (falls back to copy where
            symlinks are unavailable, e.g. Windows without Developer Mode)
        clean_existing: Remove existing training folders before writing
        include_test: Write a test split. None (default) auto-detects from the annotations.
        verbose: Show detailed progress

    Returns:
        Dictionary with the created directories and statistics.

    Raises:
        ValueError: If config has no processed annotations, or output_dir is inside annotation_dir
        FileNotFoundError: If annotation_dir does not exist
    """
    annotation_dir = Path(annotation_dir)
    if output_dir is None:
        output_dir = annotation_dir.parent / f"{annotation_dir.name}_training"
    output_dir = Path(output_dir)

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    assert_output_dir_is_separate(output_dir, annotation_dir)

    logger = create_training_logger(output_dir, verbose=verbose)

    processed = [ann for ann in config.annotations if ann.processed]
    if not processed:
        raise ValueError("Config has no processed annotations to reorganize")

    uses_separate_channels = config.spatial_coverage.uses_separate_channels()

    if include_test is None:
        include_test = any(ann.category == "test" for ann in processed)

    annotation_input_dir = annotation_dir / "annotation_input"
    model_input_dir = annotation_dir / "model_input"
    output_source = annotation_dir / "annotation_output"

    records = []
    n_missing = 0
    for ann in processed:
        annotation_id = ann.annotation_id

        # The model's input: the model_input channel if there is one, else the
        # annotation channel (which serves both roles in single-channel mode).
        if uses_separate_channels:
            image_file = model_input_dir / f"{annotation_id}.tif"
            annotation_file = annotation_input_dir / f"{annotation_id}.tif"
        else:
            image_file = annotation_input_dir / f"{annotation_id}.tif"
            annotation_file = None

        label_file = output_source / f"{annotation_id}_mask.tif"

        if not image_file.exists() or not label_file.exists():
            n_missing += 1
            logger.warning(
                f"Skipping annotation {annotation_id}: "
                f"missing {'image' if not image_file.exists() else 'label'}"
            )
            continue

        annotation_source = None
        if annotation_file is not None and annotation_file.exists():
            annotation_source = FileSource(annotation_file)

        records.append(
            AnnotationRecord(
                annotation_id=annotation_id,
                category=ann.category,
                image=FileSource(image_file),
                label=FileSource(label_file),
                annotation_image=annotation_source,
            )
        )

    created_dirs, stats = write_training_layout(
        records,
        output_dir,
        layout="split",
        file_mode=file_mode,
        clean_existing=clean_existing,
        include_test=include_test,
        uses_separate_channels=uses_separate_channels,
        logger=logger,
    )
    stats["n_missing"] = n_missing

    logger.info(
        f"Reorganization complete: {len(records)} annotations written, {n_missing} skipped"
    )

    return _build_standard_result(base_dir=output_dir, created_dirs=created_dirs, stats=stats)
```

Add the imports at the top of `training_functions.py`:

```python
from .training_layout import (
    AnnotationRecord,
    ArraySource,
    FileSource,
    assert_output_dir_is_separate,
    write_training_layout,
)
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_training_functions.py -v`
Expected: PASS for the new class. Existing `TestReorganizeLocalDataForTraining` / `TestReorganizeSeparateChannels` tests assert the old `train_input`-from-`input/` layout and old result keys — update them to the new folder names and keys. Delete assertions on `validation_input`/`validation_label`.

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/processing/training_functions.py tests/test_training_functions.py
git commit -m "refactor: reorganize_local_data_for_training emits the unified layout"
```

---

### Task 5: Rewire the OMERO producer onto records

This is where the index-pairing bug dies. Extract the plane fetch into a function, then build records from it.

**Files:**
- Modify: `src/omero_annotate_ai/processing/training_functions.py` (`prepare_training_data_from_table`, `_prepare_dataset_from_table`)
- Modify: `tests/test_training_functions.py`

**Interfaces:**
- Consumes: `AnnotationRecord`, `ArraySource`, `FileSource`, `write_training_layout` (Tasks 1-2).
- Produces: `_fetch_plane(conn, row, channel, logger=None) -> np.ndarray` (8-bit, 2D or 3D); `_download_label(conn, label_id, tmp_dir, logger=None) -> Optional[Path]`; `prepare_training_data_from_table(...)` unchanged in signature, returning the unified result keys.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_training_functions.py`:

```python
@pytest.mark.unit
class TestPrepareFromTableRecords:
    """The OMERO producer pairs images and labels by annotation_id."""

    def _table(self):
        return pd.DataFrame(
            [
                {
                    "image_id": 100, "annotation_id": 0, "train": True, "validate": False,
                    "channel": 0, "z_slice": 0, "timepoint": 0, "label_id": 900,
                    "is_volumetric": False, "is_patch": False,
                    "patch_x": 0, "patch_y": 0, "patch_width": 0, "patch_height": 0,
                    "processed": True,
                },
                {
                    "image_id": 101, "annotation_id": 1, "train": True, "validate": False,
                    "channel": 0, "z_slice": 0, "timepoint": 0, "label_id": 901,
                    "is_volumetric": False, "is_patch": False,
                    "patch_x": 0, "patch_y": 0, "patch_width": 0, "patch_height": 0,
                    "processed": True,
                },
            ]
        )

    def test_a_missing_label_does_not_shift_later_pairs(self, tmp_path, fake_omero_connection):
        """The regression this refactor exists to kill.

        Files used to be named by loop index, and the image was written before the
        label download could fail. One missing label left an orphan image, and every
        subsequent image/label pair silently shifted by one.
        """
        from omero_annotate_ai.processing import training_functions as tf

        table = self._table()
        plane = np.ones((4, 4), dtype=np.uint8)

        def fake_download(conn, label_id, tmp_dir, logger=None):
            if label_id == 900:
                return None  # this label is gone
            path = Path(tmp_dir) / f"{label_id}.tif"
            imwrite(str(path), plane)
            return path

        with patch.object(tf, "_fetch_plane", return_value=plane), \
             patch.object(tf, "_download_label", side_effect=fake_download), \
             patch.object(tf, "_load_table", return_value=table):
            result = tf.prepare_training_data_from_table(
                conn=fake_omero_connection,
                table_id=1,
                output_dir=tmp_path / "training",
                validation_split=0.0,
            )

        images = sorted(p.name for p in result["train_input"].glob("*.tif"))
        labels = sorted(p.name for p in result["train_label"].glob("*.tif"))

        # Annotation 0 lost its label, so it is dropped whole - not left as an orphan.
        assert images == labels == ["1.tif"]

    def test_result_keys_feed_setup_training(self, tmp_path, fake_omero_connection):
        from omero_annotate_ai.processing import training_functions as tf

        table = self._table()
        plane = np.ones((4, 4), dtype=np.uint8)

        def fake_download(conn, label_id, tmp_dir, logger=None):
            path = Path(tmp_dir) / f"{label_id}.tif"
            imwrite(str(path), plane)
            return path

        with patch.object(tf, "_fetch_plane", return_value=plane), \
             patch.object(tf, "_download_label", side_effect=fake_download), \
             patch.object(tf, "_load_table", return_value=table):
            result = tf.prepare_training_data_from_table(
                conn=fake_omero_connection,
                table_id=1,
                output_dir=tmp_path / "training",
                validation_split=0.5,
            )

        for key in ("train_input", "train_label", "val_input", "val_label"):
            assert key in result, f"setup_training requires {key}"
```

Add `from tifffile import imwrite` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_training_functions.py -k PrepareFromTableRecords -v`
Expected: FAIL — `_fetch_plane`, `_download_label` and `_load_table` do not exist.

- [ ] **Step 3: Write the implementation**

In `training_functions.py`:

1. **Extract the table load.** Pull the existing `ezomero.get_table(...)` call out of `prepare_training_data_from_table` into:

```python
def _load_table(conn, table_id: int) -> pd.DataFrame:
    """Fetch the tracking table as a DataFrame."""
    return ezomero.get_table(conn, table_id)
```

2. **Extract the plane fetch.** This is a *move*, not a rewrite. The OMERO fetching logic
(patch geometry, volumetric z-stacks, XYZCT axis swaps) is correct and hard-won — reproducing
it from scratch would risk silent regressions. Cut it out and change only how it ends.

The source is the body of `_prepare_dataset_from_table` (`training_functions.py`, `def` at
line 687). Take these regions **verbatim**:

- **Lines 758-823** — per-row metadata extraction (`z_slice` parsing, `channel`, `timepoint`,
  `is_volumetric`, `is_patch`, `patch_x/y/width/height`). Change every `df.iloc[n]["col"]`
  to `row["col"]`, and drop the `image_id = int(df.iloc[n]["image_id"])` line in favour of
  `image_id = int(row["image_id"])`.
- **Lines 826-957** — the `if is_volumetric:` branch (z-slice loop, 3D array assembly,
  8-bit normalization).
- **Lines 959-1062** — the `else:` 2D branch (patch and full-plane cases).

Two changes, and nothing else:

- The `channel` local is no longer derived from the row when `train_channel` is set — it is now
  the `channel` **parameter**. Delete the `if train_channel is not None:` / `else:` block at
  lines 796-802; the caller has already resolved it.
- Both branches currently end by writing (`output_path = input_dir / f"input_{n:05d}.tif"` then
  `imwrite(...)`, lines 947-949 and the 2D equivalent near 1050). Delete those writes and
  `return img_8bit` instead.

The resulting signature:

```python
def _fetch_plane(conn, row: pd.Series, channel: int, logger=None) -> np.ndarray:
    """
    Fetch one annotated image from OMERO as an 8-bit array.

    Handles the 2D, 2D-patch and 3D-volumetric cases, and normalizes to 8-bit.
    Returns the array rather than writing it, so the caller decides where it lands —
    which is what lets images be named by annotation_id instead of loop index.
    """
```

After extracting, confirm no `{n:05d}` naming survives anywhere:

```bash
grep -n "05d" src/omero_annotate_ai/processing/training_functions.py
```
Expected: no matches.

3. **Extract the label download**, returning `None` instead of `continue`-ing:

```python
def _download_label(conn, label_id: int, tmp_dir: Path, logger=None) -> Optional[Path]:
    """
    Download a label file annotation. Returns None if it is missing or unreadable.

    Returning None rather than raising lets the caller drop the whole record, which
    is what keeps images and labels paired.
    """
    if label_id is None:
        return None
    try:
        file_ann = conn.getObject("FileAnnotation", label_id)
        if file_ann is None:
            if logger:
                logger.warning(f"File annotation {label_id} not found in OMERO")
            return None
        file_path = ezomero.get_file_annotation(conn, label_id, str(tmp_dir))
        return Path(file_path) if file_path else None
    except Exception as e:
        if logger:
            logger.error(f"Error downloading label {label_id}: {e}")
        return None
```

4. **Replace `_prepare_dataset_from_table` with a record builder.** Delete the function and add:

```python
def _records_from_table(
    conn,
    df: pd.DataFrame,
    tmp_dir: Path,
    uses_separate_channels: bool,
    label_channel: Optional[int],
    train_channel: Optional[int],
    logger=None,
) -> tuple:
    """
    Build annotation records from a tracking table.

    A row whose label cannot be downloaded is dropped entirely: writing its image
    without a label would leave an orphan, and consumers that pair images to labels
    by sorted filename would mispair everything after it.

    Returns:
        (records, n_missing)
    """
    records = []
    n_missing = 0

    for _, row in df.iterrows():
        annotation_id = int(row["annotation_id"])

        label_id = _optional_int(row.get("label_id"))
        label_path = _download_label(conn, label_id, tmp_dir, logger)
        if label_path is None:
            n_missing += 1
            logger.warning(f"Skipping annotation {annotation_id}: no label")
            continue

        category = "training" if bool(row["train"]) else "validation"

        image_channel = train_channel if train_channel is not None else _optional_int(row.get("channel")) or 0
        image_source = ArraySource(
            lambda row=row, ch=image_channel: _fetch_plane(conn, row, ch, logger)
        )

        annotation_source = None
        if uses_separate_channels and label_channel is not None:
            annotation_source = ArraySource(
                lambda row=row, ch=label_channel: _fetch_plane(conn, row, ch, logger)
            )

        records.append(
            AnnotationRecord(
                annotation_id=annotation_id,
                category=category,
                image=image_source,
                label=FileSource(label_path),
                annotation_image=annotation_source,
            )
        )

    return records, n_missing


def _optional_int(value) -> Optional[int]:
    """Parse a table cell that may be int, float, NaN, or the string 'None'."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text in ("None", "nan", ""):
        return None
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return None
```

Note the `row=row, ch=...` default-argument binding in the lambdas: without it every closure would capture the final loop values.

5. **Rewrite `prepare_training_data_from_table`'s body** to keep its signature, load the table via `_load_table`, apply the existing train/validate split (assigning `row["train"]`/`row["validate"]` exactly as it does today), then:

```python
    records, n_missing = _records_from_table(
        conn, df, tmp_dir, uses_separate_channels, label_channel, effective_train_channel, logger
    )

    created_dirs, stats = write_training_layout(
        records,
        output_dir,
        layout="split",
        clean_existing=clean_existing,
        uses_separate_channels=uses_separate_channels,
        logger=logger,
    )
    stats["n_missing"] = n_missing

    return _build_standard_result(base_dir=output_dir, created_dirs=created_dirs, stats=stats)
```

Keep the existing `upload_label_input` block, reading its paths from `created_dirs["train_annotation_input"]` and `created_dirs["val_annotation_input"]`.

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_training_functions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/processing/training_functions.py tests/test_training_functions.py
git commit -m "fix: pair training images and labels by annotation_id, not loop index

A row whose label failed to download previously left an orphan image, because
files were named by loop index and the image was written before the label was
fetched. Consumers pair raw_paths to label_paths by sorted filename, so one
missing label silently shifted every subsequent pair. Records are now dropped
whole when the label is missing, and files are named by annotation_id."
```

---

### Task 6: Delete the dead code

**Files:**
- Modify: `src/omero_annotate_ai/processing/training_functions.py`
- Modify: `tests/test_training_functions.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: nothing new. Removes `prepare_training_data_from_config`, `_get_standard_folder_structure`, `_create_training_directories`, and the module's now-unused `_create_file_link_or_copy` (it lives in `training_layout.py` since Task 1).

- [ ] **Step 1: Confirm each target is dead**

```bash
grep -rn --include='*.py' --include='*.ipynb' -E "prepare_training_data_from_config|_get_standard_folder_structure|_create_training_directories" src tests notebooks docs
```
Expected: matches only inside `training_functions.py` and the commented-out test blocks. If any live caller appears, stop and report it.

- [ ] **Step 2: Delete**

- `prepare_training_data_from_config()` (~210 lines)
- `_get_standard_folder_structure()`
- `_create_training_directories()`
- The `from .training_layout import _create_file_link_or_copy` re-export in `training_functions.py`, if nothing there still calls it (Task 4 rewrote `reorganize_local_data_for_training` to use `FileSource`, so it should be unused — check with `grep -n "_create_file_link_or_copy" src/omero_annotate_ai/processing/training_functions.py` before removing).
- In `tests/test_training_functions.py`: the 7 commented-out test blocks, and `TestConsistentFolderStructure` (it pins `_get_standard_folder_structure` / `_create_training_directories`, both gone). Its intent — "all producers return consistent keys" — is now covered by `test_result_keys_feed_setup_training` (Task 5) and the Task 4 tests.
- Update the `tests/test_training_functions.py` import block to drop the deleted names.

- [ ] **Step 3: Run the full suite**

Run: `pixi run -e dev pytest tests/ -v`
Expected: PASS, with no import errors.

- [ ] **Step 4: Commit**

```bash
git add src/omero_annotate_ai/processing/training_functions.py tests/test_training_functions.py
git commit -m "refactor: delete prepare_training_data_from_config and the superseded folder helpers"
```

---

### Task 7: Rename the OMERO identifiers

**Files:**
- Modify: `src/omero_annotate_ai/core/annotation_config.py` (line ~106, ~878, ~907, ~990)
- Modify: `src/omero_annotate_ai/omero/omero_functions.py` (line ~688-733)
- Modify: `src/omero_annotate_ai/omero/__init__.py` (lines ~18, ~37)
- Modify: `tests/test_config.py`, `tests/test_omero_functions.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ImageAnnotation.annotation_input_id`; `upload_annotation_input_image(...)`; namespace `openmicroscopy.org/omero/annotate/annotation_input`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
@pytest.mark.unit
class TestAnnotationInputIdRename:
    """annotation_input_id replaces label_input_id, and old tables still load."""

    def test_new_column_round_trips(self):
        from omero_annotate_ai.core.annotation_config import AnnotationConfig, create_default_config

        config = create_default_config()
        config.annotations = [
            ImageAnnotation(
                annotation_id=0, image_id=100, image_name="img",
                category="training", processed=True, annotation_input_id=555,
            )
        ]

        df = config.annotations_to_dataframe()
        assert "annotation_input_id" in df.columns
        assert "label_input_id" not in df.columns

        restored = AnnotationConfig.from_dataframe(df, config)
        assert restored.annotations[0].annotation_input_id == 555

    def test_legacy_label_input_id_column_still_loads(self):
        """Tables written by earlier versions must not break."""
        from omero_annotate_ai.core.annotation_config import AnnotationConfig, create_default_config

        config = create_default_config()
        df = config.annotations_to_dataframe()  # gives us the right columns
        df = df.rename(columns={"annotation_input_id": "label_input_id"})
        # one legacy row
        df.loc[0] = {**{c: "None" for c in df.columns}, "image_id": 100, "image_name": "img",
                     "label_input_id": "777", "train": True, "validate": False, "processed": True}

        restored = AnnotationConfig.from_dataframe(df, config)

        assert restored.annotations[0].annotation_input_id == 777
```

Match the exact constructor/serializer names used in `annotation_config.py` — if the round-trip helpers are named differently, use the real names.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_config.py -k AnnotationInputIdRename -v`
Expected: FAIL — `ImageAnnotation` has no field `annotation_input_id`.

- [ ] **Step 3: Write the implementation**

In `annotation_config.py`:

```python
    annotation_input_id: Optional[int] = Field(
        default=None,
        description="OMERO file annotation ID for the annotation-channel image",
    )
```

Serializer (line ~878): `"annotation_input_id": _optional_int_to_str(annotation.annotation_input_id),`
Column list (line ~907): `"label_input_id"` → `"annotation_input_id"`.

Reader (line ~990) — this is the only backwards-compatibility shim in the whole change:

```python
            # Tables written before the rename carry label_input_id. Read either.
            annotation_input_id = _str_to_optional_int(
                str(row.get("annotation_input_id", row.get("label_input_id", "None")))
            )
            if annotation_input_id is not None:
                annotation_data["annotation_input_id"] = annotation_input_id
```

In `omero_functions.py`, rename `upload_label_input_image` → `upload_annotation_input_image`, its `label_input_file` parameter → `annotation_input_file`, and the namespace:

```python
        ns="openmicroscopy.org/omero/annotate/annotation_input",
```

Update `omero/__init__.py` imports and `__all__`. Update the call site in `training_functions.py` (the `upload_label_input` block) and any test referencing the old name.

- [ ] **Step 4: Run the full suite**

Run: `pixi run -e dev pytest tests/ -v`
Expected: PASS

Verify nothing stale remains:

```bash
grep -rn --include='*.py' "label_input" src tests
```
Expected: only the legacy read fallback in `annotation_config.py`.

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/core/annotation_config.py src/omero_annotate_ai/omero/ tests/
git commit -m "refactor: rename label_input_id to annotation_input_id, keeping legacy tables readable"
```

---

### Task 8: Update consumers and docs

**Files:**
- Modify: `src/omero_annotate_ai/processing/training_utils.py` (line ~54, ~75, ~92-95, ~214-227)
- Modify: `src/omero_annotate_ai/core/annotation_pipeline.py` (`reorganize_for_training`, line ~1875)
- Modify: `notebooks/jupyter/training/omero-training-microsam.ipynb`, `omero-training_biapy.ipynb`, `omero-training_DL4mic.ipynb`, `notebooks/marimo/omero-training-microsam.py`, `notebooks/marimo/omero-idr-demo.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the unified result keys from Tasks 4-5.
- Produces: `setup_training()` requiring `train_input`, `train_label`, `val_input`, `val_label`.

- [ ] **Step 1: Update `setup_training`**

```python
    required_keys = ['train_input', 'train_label', 'val_input', 'val_label']
```

and the paths it builds:

```python
    training_config = {
        'train_input': Path(training_result['train_input']),
        'train_label': Path(training_result['train_label']),
        'val_input': Path(training_result['val_input']),
        'val_label': Path(training_result['val_label']),
        ...
    }
```

and `_run_microsam_training` (line ~214):

```python
        raw_paths=str(config["train_input"]),
        label_paths=str(config["train_label"]),
        ...
        raw_paths=str(config["val_input"]),
```

Also update the output-dir inference at line ~75: `Path(training_result['train_input'])`.

- [ ] **Step 2: Update the pipeline wrapper**

`AnnotationPipeline.reorganize_for_training()` currently defaults `output_dir` to the annotation dir, which the Task 1 guard now rejects. Change the default to `None` and let `reorganize_local_data_for_training` pick the sibling:

```python
        # Let reorganize_local_data_for_training default to <annotation_dir>_training/
        return reorganize_local_data_for_training(
            config=self.config,
            annotation_dir=annotation_dir,
            output_dir=output_dir,
            file_mode=file_mode,
            clean_existing=clean_existing,
            include_test=include_test,
            verbose=verbose,
        )
```

Delete the `if output_dir is None: output_dir = annotation_dir` lines. Update its docstring to say the default target is the sibling `<annotation_dir>_training/`.

- [ ] **Step 3: Update the notebooks**

In each, `training_result['training_input']` → `['train_input']`, `['training_label']` → `['train_label']`. `val_input` and `val_label` are unchanged.

```bash
grep -rn "training_input\|training_label" notebooks/
```
Expected after edits: no matches.

- [ ] **Step 4: Update CLAUDE.md**

Replace the "Cellpose Workflow Fixes" folder-design section with the unified layout:

```markdown
## Folder layout

Annotation phase (`AnnotationPipeline`), in `config.output.output_directory`:
- `annotation_input/{id}.tif` — the channel you annotate (e.g. fluorescence)
- `model_input/{id}.tif` — the channel the model consumes (e.g. brightfield); separate-channel workflows only
- `annotation_output/{id}_mask.tif` — the mask you produced
- `sam_embeddings/` — micro-SAM only

Training phase, in a **separate** directory (default `<annotation_dir>_training/`):
- `train_input/{id}.tif`, `train_label/{id}.tif`
- `val_input/{id}.tif`, `val_label/{id}.tif`
- `test_*` when `include_test=True`
- `train_annotation_input/`, `val_annotation_input/` for separate-channel workflows

Files are named by `annotation_id`, so an image and its label always pair by name.
The training directory must not be inside the annotation directory — `prepare_training_data*`
raises `ValueError` if it is, because cleaning the training layout there would delete the
source images.

`processing/training_layout.py` owns the layout. Both producers
(`prepare_training_data_from_table`, `reorganize_local_data_for_training`) resolve their
inputs into `AnnotationRecord`s and hand them to `write_training_layout()`. `file_mode`
(copy/move/symlink) applies only to the offline producer; the OMERO producer fetches planes,
so there is no file to link. `layout="cellpose"` is specified but raises `NotImplementedError`.
```

- [ ] **Step 5: Run the full suite and commit**

Run: `pixi run -e dev pytest tests/ -v`
Expected: PASS

```bash
git add src/omero_annotate_ai/processing/training_utils.py src/omero_annotate_ai/core/annotation_pipeline.py notebooks/ CLAUDE.md
git commit -m "refactor: point consumers at the unified train_* result keys"
```

---

### Task 9: Verify end to end

**Files:**
- None (verification only)

- [ ] **Step 1: Full suite**

Run: `pixi run -e dev pytest tests/ -v`
Expected: PASS, no skips beyond the pre-existing 5.

- [ ] **Step 2: No stale names anywhere**

```bash
grep -rn --include='*.py' --include='*.ipynb' --include='*.md' -E "training_input|training_label|train_label_input|validation_input|validation_label|label_input" src tests notebooks CLAUDE.md
```
Expected: only the legacy `label_input_id` read fallback in `annotation_config.py`, and the CLAUDE.md line documenting it.

- [ ] **Step 3: The round trip a user actually does**

Confirm the offline path produces something `setup_training` accepts:

```bash
pixi run -e dev python -c "
from omero_annotate_ai.processing.training_utils import setup_training
import inspect
src = inspect.getsource(setup_training)
assert \"'train_input'\" in src and \"'val_label'\" in src
print('setup_training requires the unified keys')
"
```

- [ ] **Step 4: Update pixi.lock and commit**

CLAUDE.md requires `pixi.lock` to be current before a PR. No dependencies changed, so this should be a no-op — confirm it:

```bash
pixi install
git status --porcelain pixi.lock
```
If `pixi.lock` changed, commit it. If not, nothing to do.

- [ ] **Step 5: Open the PR**

```bash
gh pr create --base main --title "refactor: unify the annotation and training folder layouts" --body "Implements docs/superpowers/specs/2026-07-13-folder-layout-unification-design.md

Also fixes a live pairing bug: training files were named by loop index and the image
was written before its label was fetched, so one missing label silently shifted every
subsequent image/label pair. Files are now named by annotation_id and label-less records
are dropped whole."
```
