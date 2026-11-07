# Training-set bundle: reliable resume, provenance and cleanup (0.5.0)

Status: **draft for review**. Date: 2026-11-07.

## Why

Today one training set is spread over loosely joined pieces: a tracking table on OMERO that gets
a **new ID after every batch**, a growing pile of config YAMLs matched to it **by name**, ROIs
and label files that nothing traces back if a run crashes, local folders whose contents depend on
the mode (often a fresh `/tmp` folder per run), and an in-memory config that the widget shares
with the pipeline and rebuilds on resume by guessing from table rows.

Three reviews (widget, pipeline, OMERO data) found data-loss bugs (masks uploaded onto the wrong
images after a crash, table replaced by a random new one after a failed read, failed uploads
marked done, delete-before-create of the table) and many smaller holes. Most come from that
structure, not from individual lines.

## Decisions (agreed 2026-11-07)

- Everything ships in **0.5.0**, together with the OME-Zarr/URI work (#56). PRs #67 and #68 are
  folded in (their extend and delete-by-ID logic is reused).
- **Old tracking tables are not supported.** No conversion, no read-side shims (this also drops
  the `label_input_id` fallback).
- Each training set has a **fixed local home folder** (default under `~/omero_annotate_ai/`).
- One **simple note per annotation session**, no per-image notes.
- The local folder is an **RO-Crate** (generated metadata file; working files stay YAML + CSV).

## The model

A **training set** is identified by its **tracking table's ID**, which never changes.

```
OMERO                                   Local bundle (the working copy)
-----                                   ------------------------------
tracking table   (updated in place) <-> <home>/<server>/<table_id>_<name>/
one config YAML  (replaced per run) <->   config.yaml          (incl. sessions + notes)
ROIs, label files (tagged with          tracking_table.csv
  table ID + annotation_id)               annotation_input/{annotation_id}.tif
                                          annotation_output/{annotation_id}_mask.tif
                                          ro-crate-metadata.json   (generated)
```

- **OMERO is the published copy; the local folder is the working copy.** The folder can always be
  rebuilt from OMERO. Read-only mode and the OME-Zarr route have no table: their folder is
  `<home>/local/<uuid>_<name>/` and is the source of truth.
- **Resume** = "open training set <table ID>" (or a folder). It loads that set's one config as
  is. No name lookup, no guessing from rows, no lock rules.
- **Delete** = remove the set by table ID: rows give ROI/label IDs, tags catch anything a crash
  left untracked.

## Components

### 1. Tracking table (OMERO)

- Created once with explicit column types and generous string widths (URIs 1024, ids 256);
  lengths are checked before every write (OMERO rejects over-long strings with
  `ValidationException`, verified on a local server).
- Updated **in place**: `readCoordinates` + `update` for changed rows, `addData` for new rows.
  Verified on a local OMERO: the file ID stays the same.
- Schema (v2): `annotation_id, image_uri, image_id, category` (explicit
  `training|validation|test` string, fixes the lost test split), plane/patch columns
  (`timepoint, z_slice, z_start, z_end, is_volumetric, is_patch, patch_x, patch_y, patch_w,
  patch_h, channel`), state (`processed, roi_id, label_id, annotation_input_id, session`).
- The table's FileAnnotation description carries `omero-annotate-ai table v2`; tables without it
  are refused with a clear message.

### 2. Config and sessions

- One config YAML per set on OMERO, file `annotation_config_table_<id>.yaml`, ns `CONFIG_NS`,
  **replaced** (new one posted, old one deleted) after each batch, so there is exactly one.
- New `sessions` list in the config: `{id, started, ended, user, software_version, model,
  n_annotated, note}`. Rows record the `session` that annotated them.
- New `training.seed`, stored, so selections are reproducible.
- `get_config()` returns a **copy**; the pipeline never mutates the widget's object.

### 3. Pipeline write order (crash-safe)

Per annotated unit:

1. mask is renamed right away to `annotation_output/{annotation_id}_mask.tif`
   (micro-SAM writes into a fresh per-batch scratch folder; masks are matched by exact name
   `seg_{i:05}.tif`, never by position in a sorted list);
2. local CSV + YAML updated (`processed`, `session`);
3. ROI and label uploaded, each tagged `omero-annotate-ai:<table_id>:<annotation_id>`;
4. table row updated in place with the IDs; local state updated.

On resume a **reconcile** step fixes any interrupted unit: processed locally but no IDs on
OMERO → look up tagged ROI/label first (no duplicates), else upload. A failed upload leaves the
row unprocessed with the error recorded. A failed table read **aborts** the resume (never
"start fresh").

Also: SAM embeddings per batch folder (fixes batch-2 crash); `initialize_workflow` skips random
selection on resume and lets the extend step handle shortfalls; Cellpose marks rows processed
only when a mask is collected; upload mode writes the same local folders as read-only.

### 4. Session note

After napari (or Cellpose collection) finishes a batch, the user is asked for one short note
(widget text box / `input()` fallback / `note=` argument for scripts). Empty is fine. Stored in
the session record locally and on OMERO (in the config), and in the RO-Crate.

### 5. Widget

- Two clear paths: **New training set** (all settings) and **Open training set** (pick by
  name + ID; shows a summary: settings, counts, sessions with notes; editable only "add N
  training/validation/test images", model, batch size).
- Changing container or table resets the selection; `get_config()` syncs from the widgets.
- Read-only: a property of a set chosen at creation, not a per-run toggle on an OMERO set.

### 6. Delete

`delete_training_set(conn, table_id, dry_run=True)`: IDs from rows + tagged objects + the set's
config + the table, children first; refuses objects owned by someone else unless
`force_others=True`; waits for the server properly (`waitOnCmd`, fail on timeout).
`cleanup_project_annotations` (substring matching, no dry run) is removed.

### 7. RO-Crate

`ro-crate-metadata.json` generated from config + CSV after every save, following the Process Run
Crate profile: the set as `Dataset`, images and masks as `File`s, each session as a
`CreateAction` (agent, start/end, instrument = omero-annotate-ai + model, object = inputs,
result = masks, description = note). Hand-written JSON (no runtime dependency); validated in
tests with `rocrate-validator` in the dev environment.

### Removed

Name-based table/config lookup, "newest config wins", `infer_settings_from_table`, resume lock
rules, `label_input_id` fallback, flattened-config map annotation, workflow-status map
annotation, `_initialize_tracking_table`, `cleanup_project_annotations`.

## Testing

- Fix the Docker OMERO fixture (`docker compose`, pinned server/web/postgres images, seeded
  dataset/plate with small synthetic images). Marked `docker`; runnable locally with
  `pixi run start-omero`, optionally as a CI job.
- Integration tests: create → annotate (napari mocked) → resume → extend → crash at each step of
  the write order → reconcile → delete (dry run and real). Unit tests stay mocked and fast.

## Delivery (PRs, all into the 0.5.0 line)

1. **Test infrastructure**: working Docker OMERO fixture + seed data.
2. **Table layer**: schema v2, in-place updates, tagging, one config per set.
3. **Pipeline**: bundle folder, write order + reconcile, batch scratch/embeddings, sessions and
   notes, seed, Cellpose/upload-mode fixes.
4. **Widget**: New / Open training set, copy semantics, resets.
5. **Delete + removals**, docs and notebooks.
6. **RO-Crate** metadata.

Then merge into `main` with #56 and release 0.5.0.

## Settled follow-ups

- **Server in the path: yes.** Sites run several servers (e.g. production and test), and table
  IDs are only unique per server. `<server>` is a filesystem-safe slug of host and port
  (`host_port`, lower-case, non-alphanumerics replaced by `-`). Code and tests never hard-code
  a real server name.
- **Publishing read-only sets to OMERO later (`push`): maybe, not in 0.5.0.** The layout keeps
  it possible: a local set already has the full CSV, config, masks and inputs, and a uuid in
  its folder name. A later `push` would create the table, upload masks with tags, then rename
  the folder from `local/<uuid>_<name>` to `<server>/<table_id>_<name>`.
