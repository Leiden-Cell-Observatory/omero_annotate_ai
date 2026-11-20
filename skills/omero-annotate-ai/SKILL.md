---
name: omero-annotate-ai
description: Use when helping someone run omero-annotate-ai / omero_annotate_ai (0.5.x) - connecting to OMERO for annotation, writing or debugging a training-set config YAML (config_version 4, sampling, split, training, storage local, source zarr, RemovedConfigKey or unknown-key errors), creating or resuming a training set with TrainingSet.create / TrainingSet.open and ts.annotate (micro-SAM in napari), review, extend, describe, session notes, several computers on one set, Cellpose or other tools via prepare_external / collect_masks, OME-Zarr / IDR / BioImage Archive data without OMERO, finding a set's folder, masks, ROIs and tracking table, exporting training data (export_training_set, setup_training / run_training, BiaPy), BioImage Archive / MIFA / RO-Crate export (save_bia_package, write_ro_crate), deleting a set (delete_training_set), the napari plugin, or porting old notebooks (create_pipeline, run_full_micro_sam_workflow, read_only_mode, spatial_coverage are old or deprecated).
---

# omero-annotate-ai

## Overview

Builds a **training set**: picks images from an OMERO container (or OME-Zarr URIs), splits
them into training/validation/test, cuts them into annotation units (planes, volumes,
patches), opens them in napari with micro-SAM (or exports them for Cellpose), and keeps
masks, ROIs and a tracking table on OMERO plus a local folder. Exports training data and
BioImage Archive / RO-Crate bundles. Repo: Leiden-Cell-Observatory/omero_annotate_ai.
Changing the package itself: read the repo's `docs/architecture.md`, not this skill.

**Version first.** This skill follows `main` (config version 4, the Bioimage Annotation
Schema sections `sampling` / `split`). The PyPI release 0.5.2 is older: config
`schema_version: 3` with `spatial_coverage`. Check with
`from omero_annotate_ai.core import annotation_config as c; hasattr(c, "CONFIG_VERSION")`
(True = this skill; False = 0.5.2 from PyPI, see reference.md "Versions").

OMERO:
```yaml
# hela_nuclei_v1.yaml
config_version: 4
name: hela_nuclei_v1
omero: {container_type: dataset, container_id: 123}
sampling: {label_channel: 0}            # channel drawn on, 0-based
training: {train_n: 5, validate_n: 2}   # images, not units
workflow: {batch_size: 4}               # units per napari window
```
```python
import getpass
from omero_annotate_ai import TrainingSet, load_config_from_yaml
from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection

conn = SimpleOMEROConnection().connect(  # raises ConnectionFailed
    host="omero.example.org", username="me", password=getpass.getpass()
)
config = load_config_from_yaml("hela_nuclei_v1.yaml")  # the file is never changed
ts = TrainingSet.create(config, conn)  # selects images, creates table + folder
ts.annotate(note="nuclei only")  # blocks until each napari window closes
print(ts.table_id, ts.bundle_dir)
ts = TrainingSet.open(
    ts.table_id, conn
)  # later / elsewhere: repairs, then pending only
```

Local, OME-Zarr (no OMERO, no connection):
```yaml
name: idr0013-nuclei
source: zarr
storage: local                           # required with source: zarr
zarr:
  uris: [https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0013A/3451.zarr]
  wells: [A1, B2, C3]
sampling: {label_channel: 0, timepoints: {mode: random, n: 1}}
split: {seed: 42}
training: {train_n: 2, validate_n: 1}
```
```python
ts = TrainingSet.create(load_config_from_yaml("idr0013.yaml"))
ts.annotate()
ts = TrainingSet.open(ts.bundle_dir)  # resume a local set by its folder
```

Jupyter widgets (ipywidgets, not marimo): `create_omero_connection_widget()` ->
`.display()`, `.get_connection()`; `create_workflow_widget(connection=conn)` -> New/Open
set, `.training_set()`, `.annotate(note=...)`, `.get_config()`; its container picker has
group / owner / search / *Go to ID* (any group; switches the session group) and picks
several containers; folded sections for channels (train on, napari context channels,
normalisation) and plate/screen well filters (MIHCSME keys first).
`create_training_data_widget(connection=conn)` -> `.export()`. Marimo notebooks in
`notebooks/marimo/` build their own UI.

## Install

```bash
pixi init myproject && cd myproject
pixi add python=3.12 micro_sam jupyterlab         # micro-SAM from conda-forge
pixi add --pypi "omero-annotate-ai[notebooks]"
```
conda: `conda create -n oai -c conda-forge python=3.12 micro_sam jupyterlab`, then
`pip install "omero-annotate-ai[notebooks]"`. pip only:
`pip install "omero-annotate-ai[microsam,notebooks]"` (`zeroc-ice` may fail to build;
use pixi/conda or a Glencoe wheel). Python >= 3.11.

Extras: `notebooks` (ipywidgets, marimo), `microsam` (micro-sam>=1.8, napari), `napari`
(plugin; pins micro-sam>=1.8,<1.9), `zarr` (ome-zarr, needed for `source: zarr`),
`provenance` (ISCC codes), `all`. BioImage Archive export also needs
`pip install "git+https://github.com/BioImage-Archive/bia-mifa-models.git@v1.1.5"`.
Config version 4 (until released): `pip install
"git+https://github.com/Leiden-Cell-Observatory/bioimage-annotation-schema"
"git+https://github.com/Leiden-Cell-Observatory/omero_annotate_ai"`.

## Workflow at a glance

1. Connect: widget, `SimpleOMEROConnection().connect(...)`, `.join_session(host,
   session_key)`, `.connect_saved(host=...)` (a token saved by an earlier
   "Stay logged in" login), or any BlitzGateway. Not needed for `source: zarr`.
2. Config: YAML -> `load_config_from_yaml` / `load_config`, the widget's `get_config()`,
   or `create_default_config()` + attribute assignment. Only `name` is required.
3. New set: `TrainingSet.create(config, conn)`. Continue: `TrainingSet.open(table_id, conn)`
   (int) or `TrainingSet.open(folder)`.
4. micro-SAM: `ts.annotate(note=..., batch_size=..., model=...)`. Other tools:
   `ts.prepare_external()` -> tool writes `external_output/{annotation_id}_mask.tif` ->
   `ts.collect_masks()`.
5. Later: `ts.review(ids=[...])`, `ts.extend(train_n=10)`, `ts.set_session_note(id, text)`,
   `ts.describe(version="1", study={...})`. Progress (read-only):
   `summarize(ref, conn).progress` from `omero_annotate_ai.trainingset`.
6. Training data: `export_training_set(table_id or ts.bundle_dir, out_dir, conn)`, then
   `setup_training(result, model_name=...)` + `run_training(cfg, framework="microsam")`,
   or point BiaPy at `train_*`/`val_*`.
7. Share: `ts.write_ro_crate()`; BIA: `prepare_bia_data_from_table` +
   `omero_annotate_ai.schema_export.save_bia_package`. Delete:
   `omero_functions.delete_training_set(conn, table_id)` (dry run by default).

## Facts that bite

- **Typos are errors.** Every section forbids unknown keys (pydantic `extra_forbidden`).
  Old keys raise `RemovedConfigKey` naming the replacement (`read_only_mode` -> `storage:
  local`, `spatial_coverage` -> `sampling`, `training.seed` -> `split.seed`, ...);
  `config_version` other than 4 raises `ConfigError`. Mapping in reference.md.
- **The YAML is a template.** `create` never writes it; the set keeps its own
  `config.yaml` (identity, seed, sessions) in its folder. `ts.config` is a copy: changing
  it does nothing. Description fields change via `ts.describe(...)`; `sampling` and
  `split` are fixed once the set exists.
- **Resume = `TrainingSet.open`**, never `create` again (that makes a second table and
  folder). `create` on a config that has `identity` raises `ConfigError`; to reuse a
  set's `config.yaml` as a template remove `identity` and `sessions`. Pass the table ID
  as an int: a string is taken as a folder (`No training set in 12345`).
- **Opening repairs.** `open` compares the OMERO table, local CSV and mask files and
  adopts tagged uploads, uploads what is missing, marks pixel mismatches `needs review:`
  (skipped by sessions until you delete the wrong copy). It also marks this computer's
  unfinished sessions `interrupted` and releases its own claims.
- **When OMERO is written** (`storage: omero`): at `create` the tracking table (title =
  `name`, linked to every container) and `annotation_config_table_<id>.yaml` on the
  primary container. When a napari window closes, per unit: label TIFF on the image,
  then a polygon ROI `<name>_ROIs`, both tagged `omero-annotate-ai:<table_id>:<unit id>`
  in the description; then the table rows are updated in place (ID never changes). The
  config file is replaced at session start and end. `storage: local` writes nothing.
- **Masks are saved only when a batch's napari window closes.** Until then they are in
  `.scratch/`, which is not recovered after a crash. Keep `batch_size` small.
- **A failed upload leaves the unit pending** with the message in the `error` column of
  the local `tracking_table.csv`; the session carries on. Open the set again to retry.
- **Claims (OMERO sets only).** A batch claims its units in the table; others skip them.
  Your own claims (same user and computer) never block; others expire after
  `workflow.claim_hours` (6). A unit someone saved first is not uploaded again; your mask
  goes to `annotation_output/conflicts/`.
- **Folder:** `<home>/<server-slug>/<table_id>_<name>/` or `<home>/local/<8 hex>_<name>/`;
  `<home>` = `home=` > `$OMERO_ANNOTATE_AI_HOME` > `~/omero_annotate_ai`. Open an OMERO
  set with the same home, or it gets a fresh folder rebuilt from OMERO and the old local
  copy is ignored. Local sets are opened by their folder path.
- **Seed:** `split.seed` (generated and stored at `create` if empty) drives the split,
  random timepoints/z-slices and patch positions. Same seed + same image list = same
  selection. Counts are split units (images, or wells with `split.unit: well`); too few
  raises `SelectionError`. `extend` only raises counts and adds nothing with
  `segment_all: true`.
- **micro-SAM needs a display and wants a GPU.** napari opens a desktop window (on remote
  machines use a remote desktop, not SSH/JupyterHub). Embeddings are computed per batch
  in `.scratch/` and recomputed every session; first use downloads the model. On CPU:
  `vit_b_lm`, patches, small batches. No exclude button in `ts.annotate`; excluding
  (out of focus) is in the napari plugin or `AnnotationSession.exclude`.
- **Credentials are never stored.** `~/.omero-annotate-ai/connections.json` keeps host,
  port, user and group. A dropped connection is restored in memory; a session-key
  connection cannot log in again once its session ends.
- **OME-Zarr:** `source: zarr` needs `storage: local` set explicitly (else a validation
  error), `zarr.uris`, and the `zarr` extra. http(s) URLs or local paths (no `s3://`).
  One field per well by default (`fields: all`). Masks are on `zarr.level`'s grid.
  `zarr.well_filters` need a `well_metadata` CSV. Nothing is written to the Zarr;
  `well_matching.csv` lands in the set folder. Unit IDs hash the URI, so a moved local
  store becomes different images.
- **Training export** keeps native bit depth and writes `normalization.json`
  (`normalization.apply_at_export: true` writes float32); micro-SAM training scales to
  0-255 at load. Splits come from each row's `category`, test included when present.
  Unannotated and excluded units are skipped. The output folder must be outside the
  set folder. A label changed on OMERO after annotating raises `IntegrityError`.

## Common mistakes

- Old API from 0.3/0.4 notebooks: `create_pipeline` still works but is a deprecated
  facade (`config_file_path=` now a TypeError); `run_full_workflow`,
  `prepare_training_data_from_local`, `config.save_bia_package`, `save_password`,
  `output.output_directory`, `read_only_mode`, `resume_from_table` are gone. Rewrite to
  `TrainingSet`.
- `source: zarr` without `storage: local`; `zarr.well_filters` without `well_metadata`.
- Editing `ts.config` and expecting the set to change.
- Cellpose masks saved as `_seg.npy` / `_cp_masks.png`, named by image name, or written
  to `annotation_output/`: only `external_output/{annotation_id}_mask.tif` is collected.
- `validate_n: 0` then micro-SAM training: empty validation folder, training fails.
- `save_bia_package` without study description, funding statement, keywords, a link and
  `version`: raises `MifaNotReady`; fill them with `ts.describe(...)`.
- `delete_training_set(conn, table_id)` without `dry_run=False` deletes nothing.
- napari plugin to start a project: it only opens existing sets (2D only).

reference.md: config v4 schema with defaults, removed keys, folder layout, OMERO objects
and table columns, resume/repair, review/exclude, external tools, training and BIA /
RO-Crate export, deleting, connections, napari plugin, versions, doc/code disagreements.
