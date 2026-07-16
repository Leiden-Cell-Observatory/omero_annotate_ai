# Folder Layout Unification

**Date**: 2026-07-13
**Status**: Draft, awaiting review

## Problem

The package has three folder layouts, and the same name means different things in different
phases. This is hard to explain to a user and it has already produced one data-corruption bug
(fixed in PR #46).

| Phase | Producer | Folders |
|---|---|---|
| Annotation | `AnnotationPipeline._setup_directories` | `input/` or `label_input/` + `training_input/`, `output/`, `sam_embeddings/` |
| OMERO → training | `prepare_training_data_from_table` | `training_input/`, `training_label/`, `val_input/`, `val_label/` |
| Local → training | `reorganize_local_data_for_training` | `train_input/`, `train_label/`, `val_input/`, `val_label/` |

Three specific defects follow from this:

1. **`training_input/` is overloaded.** In the annotation phase it holds *source* images (the
   brightfield channel pulled from OMERO). In the training phase it holds the *training split*
   of the prepared dataset. `AnnotationPipeline.reorganize_for_training()` defaults its
   `output_dir` to the annotation directory, so the two phases are expected to share a root.
   With `clean_existing=True`, preparing training data into the annotation directory now
   `rmtree`s the user's source images.

2. **`train_` vs `training_` is pure drift.** The two training-data producers build the same
   thing for the same consumer under different names. The result dicts diverge as a
   consequence: `reorganize_local_data_for_training()` returns `validation_input`/
   `validation_label`, while `setup_training()` requires `val_input`/`val_label`. Piping one
   into the other raises `ValueError`. Latent only because no caller currently does it.

3. **The layout logic is duplicated.** Folder naming, cleaning, train/val splitting and stats
   are implemented independently in each producer. PR #46 had to apply the same fix twice.

## Goals

- One canonical training layout, one place that writes it.
- Folder names that state what a file is *for*, so the annotation and training phases can never
  be confused.
- A seam for CellPose's flat layout, without building it yet.
- Delete dead code that the above makes obsolete.

## Non-goals

- CellPose *training* (`_run_cellpose_training`). Out of scope; only the layout seam is designed.
- Backwards compatibility for local folder names or the Python API. Explicitly waived.
- Migrating tracking tables already written to OMERO. They remain readable via a read-side
  fallback; they are not rewritten. See "OMERO identifiers".

## Design

### Layout

Annotation output and training data live in **separate roots**. Names can no longer collide,
and cleaning the training root can never touch source images.

```
ANNOTATION ROOT                      TRAINING ROOT
  annotation_input/                    train_input/
  model_input/          (sep. ch.)     train_label/
  annotation_output/                   val_input/
  sam_embeddings/       (micro-SAM)    val_label/
                                       test_input/           (include_test)
                                       test_label/           (include_test)
                                       train_annotation_input/  (sep. ch.)
                                       val_annotation_input/    (sep. ch.)
```

Annotation phase, separate-channel mode:

- `annotation_input/` — the channel you draw on (e.g. fluorescence). Replaces `label_input/`.
- `model_input/` — the channel the model consumes (e.g. brightfield). Replaces the annotation
  phase's `training_input/`, removing the word "training" from this phase entirely.
- `annotation_output/` — the masks you produced. Replaces `output/`.

Single-channel mode has no `model_input/`; `annotation_input/` serves both roles.

Training phase: `train_*` / `val_*` / `test_*`, matching the existing `val_` prefix.
`train_annotation_input/` carries the annotation channel into the training set (for reference
and optional upload to OMERO). It replaces `training_label_input/`, which read as though it
held labels — it does not; it holds the image labels were drawn *from*.

### Unified writer

The duplication is in the writer, not the entry points, so only the writer is unified. The two
entry points differ genuinely — one needs `conn` + `table_id` and downloads planes, the other
needs `config` + `annotation_dir` and reads files from disk. Merging their signatures would
produce a function where half the arguments are always `None`.

Both entry points resolve their inputs into a list of **annotation records** and hand them to
one writer:

```python
_write_training_layout(
    records: list[AnnotationRecord],
    output_dir: Path,
    layout: str = "split",
    file_mode: str = "copy",
    clean_existing: bool = True,
) -> tuple[dict[str, Path], dict[str, Any]]   # (created_dirs, stats)
```

A record knows its id, its category (`training` / `validation` / `test`), and how to
materialize itself at a destination:

- `LocalFileRecord.write_to(dst)` → `_create_file_link_or_copy(src, dst, file_mode)`
- `OMEROPlaneRecord.write_to(dst)` → `imwrite(dst, array)`

`file_mode` (`copy` / `move` / `symlink`) is therefore meaningful only for the local source.
This matches reality: `prepare_training_data_from_table` downloads planes from OMERO and
writes them directly, so there is no file to link. `symlink` falls back to `copy` on `OSError`,
which is what Windows raises without Developer Mode or elevation — the existing behaviour of
`_create_file_link_or_copy`, unchanged.

Splitting, cleaning, folder naming and stats live in `_write_training_layout` and nowhere else.

### Layout seam

`layout="split"` (default) builds the layout above. It serves micro-SAM and BiaPy, both of
which take configurable paths (`raw_paths`/`label_paths` and `DATA.TRAIN.PATH` respectively).

`layout="cellpose"` is specified but **not implemented** — it raises `NotImplementedError`.
CellPose wants images and masks in one folder distinguished by suffix:

```
train/001.tif
train/001_masks.tif
val/002.tif
val/002_masks.tif
```

Adding it later means adding one branch in `_write_training_layout`, with no change to callers.

### Overlap guard

`prepare_training_data*` raises `ValueError` if `output_dir` resolves inside the annotation
directory. This closes the "clean_existing deletes your source images" hole structurally,
rather than relying on names not colliding.

Consequence: `AnnotationPipeline.reorganize_for_training()` loses its current default of
`output_dir = annotation_dir`. It will require an explicit target, defaulting to the sibling
`<annotation_dir>_training/`.

### Result dict

Both producers return the same keys, so either can feed `setup_training()`:

`base_dir`, `train_input`, `train_label`, `val_input`, `val_label`, `stats`, plus
`test_*` and `*_annotation_input` when applicable. The redundant `validation_input` /
`validation_label` / `validation_label_input` keys are dropped.

`setup_training()` in `training_utils.py` is updated to require `train_input`, `train_label`,
`val_input`, `val_label`. This fixes defect 2 by construction.

## OMERO identifiers

`label_input` is not only a folder name. It is also persisted **on the OMERO server**:

- the file-annotation namespace `openmicroscopy.org/omero/annotate/label_input`
  (`omero_functions.py:733`)
- the `label_input_id` column in tracking tables (`annotation_config.py:106`)
- the public function `upload_label_input_image()`

These are renamed to match the disk layout, so there is no asymmetry at the OMERO boundary.
Renaming is safe because neither identifier is used to *find* existing data:

| Identifier | New name | Why renaming is safe |
|---|---|---|
| ns `.../annotate/label_input` | `.../annotate/annotation_input` | Write-only. Nothing queries by this namespace — unlike `CONFIG_NS` and `workflow_status`, which are read back. Old file annotations keep the old namespace and are still reachable, because they are found via the id stored in the tracking table, not by namespace lookup. |
| column `label_input_id` | `annotation_input_id` | Read via `row.get(...)`, so the reader accepts the legacy name as a fallback: `row.get("annotation_input_id", row.get("label_input_id", "None"))`. Tables written by earlier versions continue to load. |
| `upload_label_input_image()` | `upload_annotation_input_image()` | Python API. Backwards compatibility explicitly waived. |

New tables are written with `annotation_input_id` only. The legacy fallback is read-side
only, and is covered by a test that loads a table containing the old column name.

## Deletions

- **`prepare_training_data_from_config()`** — 210 lines, near-duplicate of
  `prepare_training_data_from_table()`. Not exported from `processing/__init__.py` or the
  top-level `__init__.py`; zero callers in `src/`, `notebooks/` or `docs/`; zero tests.
- **165 lines of commented-out tests** in `tests/test_training_functions.py` (7 blocks,
  all covering the OMERO-download path). These were commented out because the path could not
  be mocked. The record abstraction makes it injectable, so they are replaced by real tests
  rather than merely deleted.
- **`_get_standard_folder_structure()` / `_create_training_directories()`** — subsumed by
  `_write_training_layout`.

## Testing

- Layout: folder names per `layout` and per `uses_separate_channels`; `include_test`.
- Cleaning: stale files in every produced folder are removed; absent folders do not raise.
- Overlap guard: `output_dir` inside the annotation dir raises `ValueError`.
- Records: `LocalFileRecord` honours `copy` / `move` / `symlink`, and falls back to copy when
  `symlink_to` raises `OSError` (the Windows path — tested by patching `Path.symlink_to`).
- OMERO path: `OMEROPlaneRecord` is tested with a fake plane source, no `ezomero` mock. This is
  the coverage the deleted tests were meant to provide.
- Contract: the result dict of both producers satisfies `setup_training()`'s required keys.
- `layout="cellpose"` raises `NotImplementedError`.
- Legacy read: a tracking table containing `label_input_id` still loads, populating
  `annotation_input_id`.

## Files affected

| File | Change |
|---|---|
| `processing/training_functions.py` | Records + `_write_training_layout`; delete `prepare_training_data_from_config`, `_get_standard_folder_structure`, `_create_training_directories` |
| `processing/training_utils.py` | `setup_training` required keys → `train_*` |
| `core/annotation_pipeline.py` | `_get_input_folders`, `_setup_directories`, `_save_images_for_cellpose`, `collect_annotations_from_disk` / `get_annotation_status_from_disk` defaults (`output` → `annotation_output`), `reorganize_for_training` default target |
| `core/annotation_config.py` | `label_input_id` → `annotation_input_id`, with read-side fallback for legacy tables |
| `omero/omero_functions.py` | `upload_label_input_image` → `upload_annotation_input_image`; namespace → `.../annotate/annotation_input` |
| `omero/__init__.py` | Export rename |
| `tests/test_training_functions.py` | Rewrite; delete dead blocks |
| `tests/test_pipeline.py` | Folder-name assertions |
| `tests/test_config.py` | Legacy `label_input_id` read test |
| `tests/test_omero_functions.py` | Renamed upload function |
| `notebooks/` | Result-dict keys in the 3 training notebooks + idr demo |
| `CLAUDE.md` | Folder-design section |
