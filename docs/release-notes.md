# Release notes

Newest first. The full list of pull requests per version is on
[GitHub releases](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/releases).

## 0.5.2

- **Session numbers start at 1.** A new training set numbered its first session 0
  (unannotated rows carry session -1). Existing sets keep their numbers; the next
  session follows the highest one.
- **napari plugin:** while the next image's embedding is computed, that image is shown
  dimmed, the previous image's objects and prompts are hidden and micro-SAM's dock is
  disabled, so nothing is drawn against the wrong embedding. Each image gets its own
  contrast limits; the opacity you set on the image layer is kept.
- **The napari plugin no longer offers to store the OMERO password** in the system
  keychain; it is asked for at each connect and the field is cleared straight away.
- **napari plugin:** the dock can be made as short as the annotate page, so micro-SAM's
  dock below it gets the room (it kept the height of the start page).
- **napari plugin: Prev** goes back through every image of the session, also saved
  ones: they open with their objects, and Accept replaces the saved mask and upload
  (`AnnotationSession.revise`). The next image's embedding is computed while you
  annotate the current one.
- **One pixel reader.** The training export and the BioImage Archive export read
  planes, patches and volumes through the same image source as annotation
  (`omero_annotate_ai.sources.read_unit`) instead of two separate ezomero copies.
  This fixes the training export of 3D units, which wrote one plane instead of the
  annotated z range.
- **napari plugin: Exclude** leaves an image out for good (out of focus, empty well,
  artefact): it counts as done, is not shown again and is not exported. Prev back to
  it and Accept saves it after all. Skip still means "later".
- **Exclude:** `AnnotationSession.exclude(batch, ids)` marks units as excluded (bad,
  out of focus). An excluded unit is done: it is recorded as `annotation_type:
  excluded` with `processed` true and no mask, label file or ROI, is never handed out
  or reviewed again, and is left out of the training and BioImage Archive exports.
  Progress counts it as done and reports it separately (`summarize(...).excluded`,
  `progress_summary(...)["excluded_units"]`); a session's `n_annotated` does not count
  it. Committing a mask for it later in the same session makes it a normal annotation.
  Excluding a unit whose earlier upload failed halfway deletes that upload. Tables and
  configs stay readable by 0.5.0 and 0.5.1: no column or config key is added (those
  versions see an excluded unit as annotated without a mask). Resuming now compares
  against the local copy as the last run left it (it was rewritten when leftover
  claims were released first).

## 0.5.1

- **napari plugin** (experimental, `pip install "omero-annotate-ai[napari]"`): a dock in
  your own napari viewer to open a training set, annotate with micro-SAM image by image
  (Accept / Skip / Prev), review earlier annotations with their saved masks, and finish
  with a note. Closing the window or the dock ends the session and frees its claims; see
  [napari plugin](napari-plugin.md).
- **Review** counts a mask with the same labels in a wider data type (micro-SAM writes
  uint32) as unchanged, so it is not uploaded again. A batch that fails to load (or is
  cancelled while loading) releases its claims at once.
- **ISCC content codes** (optional `provenance` extra, `iscc-lib`): each unit records an
  ISCC Image-Code of its image and mask next to the SHA-256 digests. Resume and export
  warnings say whether a changed file is a near-duplicate or different content, and the
  BioImage Archive file lists carry `pixel_sha256` and `iscc` per file. Tables made by
  0.5.0 keep working without the new columns.
- **RO-Crate**: `ts.write_ro_crate()` describes a training set's folder as an RO-Crate
  1.1 (`ro-crate-metadata.json`): files with digests and codes, source images, and the
  sessions with their notes.
- **Training-data widget** rebuilt on the 0.5 API: pick an OMERO or local training set,
  see its progress, choose output folder, channels and normalisation, and export.
  The marimo training notebook offers the same choices; both call the new
  `export_training_set` (`get_selected_table_info` is gone; use `selected_config()`).
- **BioImage Archive export** from an OMERO training set works again (the rows had no
  image IDs), and a bundle holds only annotated units.
- **Public IDR data via OME-Zarr.** The IDR's OMERO API is no longer open for
  logins, so the [public-data tutorial](tutorials/public-data-idr.md) and the new
  `idr-zarr-demo.py` notebook (replacing `omero-idr-demo.py`) read IDR studies as
  OME-Zarr from its public S3 storage, with an example config
  (`notebooks/marimo/configs/idr0013-mitocheck-zarr.yaml`).

## 0.5.0

0.5.0 makes annotation runs reliable and traceable, adds OME-Zarr as an image source, and
reorganises the package around **training sets**. It is a breaking release; see
[migrating from 0.4](#migrating-from-04).

### Highlights

- **Training sets.** `TrainingSet.create(config, conn)` selects the images and starts
  tracking; `TrainingSet.open(table_id)` (or a folder) continues it. The tracking table
  is updated in place, so its ID identifies the set for good.
- **Crash-safe and resumable.** Each mask is saved locally, uploaded with a tag
  (`omero-annotate-ai:<table id>:<unit id>`) and then recorded. Opening a set repairs
  anything a crash left half-done, without duplicate uploads. A failed upload leaves the
  unit unannotated instead of marking it done.
- **Integrity checks.** Pixel checksums of every image and mask are stored in the table;
  a resume or an export reports a mask that changed after upload.
- **Several computers, one set.** Sessions claim the units they show, so two people can
  annotate the same OMERO set at the same time without doing an image twice. Your own
  claims (e.g. left by a dropped connection) never block you; other computers' claims
  expire after `workflow.claim_hours`, and a unit saved elsewhere is never uploaded again.
- **Review and correct.** `ts.review()` (widget: *Review annotated*) opens annotated
  units with their saved masks in napari; changed masks replace the uploads on OMERO.
- **Sessions and notes.** Every annotation session is recorded (who, when, model, how
  many units) with an optional note, e.g. "nuclei only, small debris skipped".
- **One folder per set** under `~/omero_annotate_ai/` (or `$OMERO_ANNOTATE_AI_HOME`), with
  the same layout whether masks go to OMERO or stay local.
- **OME-Zarr images** (local or remote, e.g. the BioImage Archive) with `source: zarr`,
  no OMERO needed. Images are identified by URI (`omero:image:<id>` or a Zarr URL).
- **Reproducible selection.** A stored `training.seed` drives every random choice;
  splits can keep wells together (`split_unit: well`) and be stratified by well metadata.
- **Training export** keeps native bit depth, includes the test split, records the
  normalisation in `normalization.json`, and micro-SAM training normalises at load.
- **New workflow widget**: *New training set* (timepoints and z-slices: specific, all or
  random) or *Open training set* with a summary of progress and earlier sessions, a
  progress bar and a note editor for any session.
- **Safer delete**: `delete_training_set` finds everything by table ID and tag, runs as a
  dry run first, waits for the server, and refuses objects owned by other users.
- **Cellpose route**: `ts.prepare_external()` exports images; masks saved in
  `external_output/` are collected with `ts.collect_masks()`.

### Migrating from 0.4

**Python and licence.** Python 3.11 or newer. The package is now licensed GPL-2.0-or-later
(it builds on omero-py and ezomero); 0.4.x releases stay Apache-2.0.

**Old tracking tables are not supported.** Tables written by 0.4 cannot be opened or
extended; finish them with 0.4 or start a new training set.

**Config (`schema_version: 3`).** Loading a config that still has a removed key raises
`RemovedConfigKey` naming its replacement:

| 0.4 | 0.5 |
|---|---|
| `annotations` | the rows live in the set's `tracking_table.csv` and OMERO table: `ts.units()` |
| `output.output_directory`, `config_file_path` | the set's folder: `home=` or `$OMERO_ANNOTATE_AI_HOME` |
| `workflow.read_only_mode: true` | `storage: local` |
| `workflow.resume_from_table`, `omero.table_id` | `TrainingSet.open(table_id)` (`identity.table_id` is written by the package) |
| `version` | `schema_version` |
| `spatial_coverage.normalization_method`, `percentile_*` | `normalization.*` |

Unknown keys are refused, in every section, instead of ignored.

**Code.**

| 0.4 | 0.5 |
|---|---|
| `create_pipeline(config, conn).run_full_micro_sam_workflow()` | `TrainingSet.create(config, conn).annotate()` (the old call still works, deprecated) |
| resume with `resume_from_table` | `TrainingSet.open(table_id, conn).annotate()` |
| `run_cp_workflow()` / `collect_annotations_from_disk()` | `ts.prepare_external()` / `ts.collect_masks()`; masks from Cellpose go to `external_output/` |
| `config.get_processed()`, `config.annotations` | `[r for r in ts.units() if r.processed]` |
| `config.save_bia_package(dest)` | `save_bia_package(ts.config, ts.units(), ts.bundle_dir, dest)` |
| `cleanup_project_annotations(...)` | `delete_training_set(conn, table_id)` |
| `prepare_training_data_from_table(..., validation_split=...)` | splits come from the set's categories |

**Notebooks.** Cellpose is a tool choice in the marimo annotation notebook, OMERO and
OME-Zarr YAML runs share `annotate-from-yaml.py`, and BiaPy/DL4MicEverywhere export is one
notebook; see [example notebooks](notebooks.md).

## 0.4.0

- MIFA / BioImage Archive export (#49).
- One folder layout for annotation and training (`annotation_input/`, `model_input/`,
  `annotation_output/`) and typed OMERO table columns (#50).
- numpy 2 support (#52); `__version__` matches the installed package (#57).

## 0.3.3 – 0.3.1

- Training folders cleaned correctly (#46).
- Workflow widget states what it creates; safer credential handling in the connection
  widget (#45, #44).
- Masks and ROIs uploaded to the right OMERO images (file-ordering bug, #42).

Older versions: see [GitHub releases](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/releases).
