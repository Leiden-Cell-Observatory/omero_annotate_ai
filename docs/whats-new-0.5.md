# What's new in 0.5.0

0.5.0 makes annotation runs reliable and traceable, adds OME-Zarr as an image source, and
reorganises the package around **training sets**. It is a breaking release; the
[migration notes](#migrating-from-04) below list what to change.

## Highlights

- **Training sets.** `TrainingSet.create(config, conn)` selects the images and starts
  tracking; `TrainingSet.open(table_id)` (or a folder) continues it. The tracking table
  is updated in place, so its ID identifies the set for good.
- **Crash-safe and resumable.** Each mask is saved locally, uploaded with a tag
  (`omero-annotate-ai:<table id>:<unit id>`) and then recorded. Opening a set repairs
  anything a crash left half-done, without duplicate uploads. A failed upload leaves the
  unit unannotated instead of marking it done.
- **Integrity checks.** Pixel checksums of every image and mask are stored in the table;
  a resume or an export reports a mask that changed after upload.
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
- **New workflow widget**: *New training set* or *Open training set* with a summary of
  progress and earlier sessions, progress bar and session-note box.
- **Safer delete**: `delete_training_set` finds everything by table ID and tag, runs as a
  dry run first, waits for the server, and refuses objects owned by other users.

## Migrating from 0.4

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

Unknown keys are refused instead of ignored.

**Code.**

| 0.4 | 0.5 |
|---|---|
| `create_pipeline(config, conn).run_full_micro_sam_workflow()` | `TrainingSet.create(config, conn).annotate()` (the old call still works, deprecated) |
| resume with `resume_from_table` | `TrainingSet.open(table_id, conn).annotate()` |
| `run_cp_workflow()` / `collect_annotations_from_disk()` | `ts.prepare_external()` / `ts.collect_masks()` |
| `config.get_processed()`, `config.annotations` | `[r for r in ts.units() if r.processed]` |
| `config.save_bia_package(dest)` | `save_bia_package(ts.config, ts.units(), ts.bundle_dir, dest)` |
| `cleanup_project_annotations(...)` | `delete_training_set(conn, table_id)` |
| `prepare_training_data_from_table(..., validation_split=...)` | splits come from the set's categories |

**Notebooks.** Cellpose is a tool choice in the marimo annotation notebook, OMERO and
OME-Zarr YAML runs share `annotate-from-yaml.py`, and BiaPy/DL4MicEverywhere export is one
notebook; see [example notebooks](notebooks.md).
