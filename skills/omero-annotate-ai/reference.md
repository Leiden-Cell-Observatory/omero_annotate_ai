# omero-annotate-ai reference

Read from the source on 2026-11-20 (origin/main 83aeb1b, version 0.5.2)

`pyproject.toml` on `main` says 0.5.2, but `main` is ahead of the 0.5.2 release on PyPI
(tag `v0.5.2`, 2026-11-08): it holds the changes the release notes list under "0.5.3" and
"Unreleased" (config version 4, no keychain, session keys). Everything below describes
`main`; the Versions table says how to recognise the others.

## Versions

| Install | Config | Recognise it by | Main API |
|---|---|---|---|
| 0.3.x (PyPI 0.3.3) | `schema_version` 2, `spatial_coverage`, `workflow.read_only_mode`, `output.output_directory`, `annotations:` list in the YAML | `seg_00000.tif` in `<output>/annotations/`, `input/`/`output/` folders, table deleted and re-created each batch | `create_pipeline(...).run_full_micro_sam_workflow()`, `run_cp_workflow()` |
| 0.4.0 | as 0.3 | `annotation_input/`, `annotation_output/`, `model_input/`; typed `Image`/`Roi` table columns | as 0.3; `config.save_bia_package` |
| 0.5.0 - 0.5.2 (PyPI) | `schema_version: 3`; `spatial_coverage` (`channels`, `label_channel`, `training_channels`, `context_channels`, `use_patches`, `patch_size`, `patches_per_image`, `timepoint_mode`, `three_d`), `training.seed` / `split_unit` / `stratify_by`, `dataset`, `study`, `annotation_methodology`, `storage`, `identity`, `sessions` | `TrainingSet`; no `CONFIG_VERSION` in `core.annotation_config`; `save_password` and `keyring` still there | as `main`, except `describe`, `omero_server`, split report; BIA: `omero_annotate_ai.core.mifa_export.save_bia_package(..., funding_statement=)` |
| `main` (this file) | `config_version: 4`, Bioimage Annotation Schema sections | `CONFIG_VERSION = 4`; depends on `bioimage-annotation-schema` | `TrainingSet`, `schema_export.save_bia_package` |

Tables written before 0.5.0 (no marker `omero-annotate-ai table v3` in the table's
description) are refused (`UnsupportedTableError`); there is no migration. A 0.5.0-0.5.2
config does not load on `main`: each moved key raises `RemovedConfigKey` (table below).
`create_pipeline` exists on 0.5.x only as a deprecated facade over `TrainingSet`.

## Config version 4 (`src/omero_annotate_ai/core/annotation_config.py`)

Only `name` is required. Unknown keys raise a pydantic `ValidationError` in every
section. `config_version` other than 4 raises `ConfigError`. `load_config(path | yaml
text | dict)`, `load_config_from_yaml(path)` (expands `~`), `create_default_config()`
(name `default_annotation_workflow`), `get_config_template()` (commented YAML),
`config.save_yaml(path)`, `config.to_yaml()`. Helpers: `config.label_channel()`,
`config.training_channels()`, `config.uses_separate_channels()`.

Tool sections:

`omero`:
| key | default | notes |
|---|---|---|
| `container_type` | `dataset` | `project`, `dataset`, `plate`, `screen`, `image` |
| `container_id` | 0 | |
| `container_ids` | null | same type; wins over `container_id`; first one is the primary container (config file) |
| `source_desc` | "" | |
| `well_filters` | null | `{key: [values]}` on well map annotations, AND over keys; plates, and the plates of screens |
| `well_filter_mode` | `include` | or `exclude` |

`zarr` (with `source: zarr`; needs the `zarr` extra):
| key | default | notes |
|---|---|---|
| `uris` | [] | plate, well or image; http(s) URL or local path; required |
| `level` | 0 | multiscale level; masks and coordinates on this grid |
| `fields` | `first` | or `all` |
| `wells` | null | e.g. `[B2]` |
| `well_metadata` | null | CSV with `well` (optional `plate`) and metadata columns |
| `well_filters` | null | on CSV columns; requires `well_metadata` |
| `well_filter_mode` | `include` | |

| key | default | notes |
|---|---|---|
| `source` | `omero` | or `zarr` |
| `storage` | `omero` | or `local` (nothing written to OMERO); `zarr` requires `local` |

`training`: count mode (`segment_all: false`, default): `train_n` 3 (>= 1), `validate_n` 2,
`test_n` 0. Fraction mode (`segment_all: true`, all images): `train_fraction` 0.7,
`validation_fraction` 0.3, `test_fraction` 0.0 (sum <= 1). Setting the unused mode's
values warns and does nothing. Counts are split units (images or wells).

`normalization`: `method` `percentile` (or `none`), `percentile_low` 1.0,
`percentile_high` 99.8, `clip` true, `apply_at_export` false.

`ai_model`: `framework` `micro_sam` (`cellpose`, `web_annotation`; metadata only, the
method you call picks the tool), `framework_version` null, `model_name` "",
`model_version` `latest`, `model_url`, `model_doi`, `training_mode` `inference`
(`finetuning`, `from_scratch`), `pretrained_from` `vit_b_lm` (micro-SAM model in napari).

`display`: `context_channels` [] (shown in napari only; must not include the label
channel), `context_channel_names`, `context_channel_colormaps` (same length).

`workflow`: `batch_size` 0 (= all units in one window), `claim_hours` 6.

Schema sections (models from `bioimage_annotation_schema.datamodel`):

`sampling` (fixed after `create`):
| key | default | notes |
|---|---|---|
| `label_channel` | 0 (null) | channel annotated |
| `input_channels` | `[label_channel]` | channels the model trains on; differing turns on `model_input/` |
| `timepoints` | `{mode: specific, values: [0]}` | modes `specific` (`values`), `all`, `random` (`n` per image) |
| `z_slices` | `{mode: specific, values: [0]}` | same |
| `volumetric` | false | one 3D unit per timepoint, first to last selected z |
| `patches` | absent = whole planes | `{width, height, per_image: 1, random: true}`; non-overlapping; image smaller than patch = one full unit |

`split` (fixed after `create`): `strategy` `random_split` (`expert_review`,
`cross_validation`; metadata), `unit` `image` (or `well`, plates only), `rationale`,
`stratify_by` (well keys or `well`; unknown key = error), `seed` (generated at `create`).

Description sections (change later with `ts.describe(**fields)`; allowed names in
`omero_annotate_ai.trainingset.DESCRIPTION_FIELDS`): `description`, `license`
(`CC-BY-4.0` default, or `CC0-1.0`), `version`, `url`, `changes`, `previous_version`,
`authors` (`first_name`, `last_name`, `email`, `orcid`, `affiliations: [{name, ror}]`,
`roles`), `keywords`, `links` (`url`, `description`), `ai_models_trained`, `study`
(`title` required, `description`, `organism`, `imaging_method`, `funding_statement`,
`publications`, `grants`, `acknowledgements`), `source_dataset` (`identifier`, `url`,
`description` required), `methodology` (default `annotation_types: [segmentation_mask]`,
`criteria: ""`; also `mask_kind` instance|semantic, `method`
manual|semi_automatic|automatic, `coverage` all|representative|partial,
`confidence_level`, `tools: [{name, version, mode}]`), `classes` (`name`, `color`,
`label_value`, `ontology_term`), `notes`, `extensions` (`{namespace: {...}}`).

Written by the tool: `config_version`, `created`, `identity` (`server`, `table_id`,
`local_uuid`), `omero_server` (`host`, `database_uuid`, optional `web_url`; filled from
the connection), `sessions` (`id`, `kind` annotate|review, `started`, `ended`, `status`
open|finished|interrupted, `user`, `host` (hashed), `software_version`, `model`,
`batch_size`, `n_annotated`, `note`).

Removed keys (each raises `RemovedConfigKey` with this hint):
| old | new |
|---|---|
| `annotations` | rows in `tracking_table.csv` / OMERO table; `ts.units()` |
| `output` | set folder under the home folder (`home=`, `$OMERO_ANNOTATE_AI_HOME`) |
| `config_file_path` | the set's own `config.yaml` |
| `schema_version` | `config_version: 4` |
| `workflow.resume_from_table` | `TrainingSet.open(table_id)` or `identity.table_id` |
| `workflow.read_only_mode` | `storage: local` |
| `omero.table_id` | `identity.table_id` |
| `spatial_coverage` | `sampling` (context channels -> `display`) |
| `annotation_methodology` | `methodology` |
| `dataset` | `source_dataset` and `license` |
| `feature_types` | `classes` |
| `tags` | `keywords` |
| `documentation`, `repository` | `links` |
| `custom_metadata` | `extensions` |
| `training.validation_strategy` / `split_unit` / `split_rationale` / `stratify_by` / `seed` | `split.strategy` / `unit` / `rationale` / `stratify_by` / `seed` |
| `training.quality_threshold` | `methodology.confidence_level` |
| `study.keywords` | top-level `keywords` |

Example plate config:
```yaml
config_version: 4
name: hela_plate_nuclei_v1
omero:
  container_type: plate
  container_id: 456
  well_filters: {treatment: [DMSO]}
sampling:
  label_channel: 0          # draw on nuclei
  input_channels: [1]       # train on brightfield -> model_input/
  patches: {width: 512, height: 512, per_image: 2}
split: {unit: well, stratify_by: [cell_line]}
training: {train_n: 6, validate_n: 2, test_n: 1}
display: {context_channels: [2], context_channel_names: [membrane]}
workflow: {batch_size: 4}
```

## TrainingSet API (`src/omero_annotate_ai/trainingset.py`)

| call | does |
|---|---|
| `TrainingSet.create(config, conn=None, *, home=None, images=None, events=None)` | plan units from the seed, create table (OMERO) or local ID, write folder, publish config; `config` untouched |
| `TrainingSet.open(table_id or folder, conn=None, *, home=None, prune_duplicates=False)` | rows from table (OMERO) or CSV (local), repair, close own open sessions |
| `ts.annotate(annotator=None, *, batch_size=None, model=None, note=None)` | pending units, batch by batch; default `MicroSamAnnotator()`; returns units saved |
| `ts.review(annotator=None, *, ids=None, batch_size=None, model=None, note=None)` | annotated units with their saved mask (downloaded if missing); returns units changed |
| `ts.extend(train_n=, validate_n=, test_n=)` | raise counts (lower raises `ConfigError`), add images; returns images added |
| `ts.prepare_external()` / `ts.collect_masks(folder="external_output")` | external tools, below |
| `ts.set_session_note(session_id, note)`, `ts.describe(**fields)` | locally and on OMERO |
| `ts.units()` | copies of the rows (`omero_annotate_ai.Row`) |
| `ts.start_session(...)` -> `AnnotationSession` | step by step: `next_batch()`, `commit(batch, {id: mask_path})`, `exclude(batch, ids)`, `revise(id, mask)`, `release(batch)`, `finish(note)` |
| `ts.reconcile(prune_duplicates=False)`, `ts.write_ro_crate()` | repair again; RO-Crate |
| `ts.table_id`, `ts.bundle_dir`, `ts.name`, `ts.is_local`, `ts.config` (copy) | |

Read-only helpers in `omero_annotate_ai.trainingset`: `summarize(ref, conn=None)`
(`.progress` = `{split: (done, total)}`, `.excluded`, `.sessions`),
`list_training_sets(conn, container_type, container_id)`, `training_set_choices(...)`,
`read_set_config(ref, conn=None)`. `ScriptedAnnotator(mask_fn)` in
`omero_annotate_ai.annotators` annotates without a display (scripts, tests). An annotator
returns `{annotation_id: mask file}`; micro-SAM writes `seg_{i:05}.tif` per batch item.

## Folder layout

`<home>/<server-slug>/<table_id>_<name-slug>/` (OMERO) or `<home>/local/<8 hex>_<name>/`;
server slug e.g. `omero.example.org-4064`, websocket hosts keep host and path.
```
config.yaml                    config + identity + sessions; source for resume
tracking_table.csv             rows, plus local-only column `error`
annotation_input/{id}.tif      image as annotated (native dtype)
model_input/{id}.tif           training channel(s), only if input_channels != label_channel
annotation_output/{id}_mask.tif accepted masks only
annotation_output/conflicts/{id}_session{n}.tif   masks that lost to another computer
external_output/               masks from external tools, until collect_masks
well_matching.csv              Zarr: wells seen, matched, metadata
.scratch/s{session}-b{batch}/  one batch's micro-SAM masks and embeddings; removed after
annotation-set.yaml, annotation-records.csv, split-report.yaml, ro-crate-metadata.json
                               written by ts.write_ro_crate()
```
Unit ID: `{sha1(image_uri)[:8]}_{t}_{z}`, `_{t}_3d` for volumes, plus `_{patch}`.
`image_uri` = `omero:image:<id>` or the normalised Zarr URL/absolute path. Files are
written atomically.

## What lands in OMERO (`storage: omero`)

| object | where | when |
|---|---|---|
| tracking table (title = `name`, ns bulk annotations, description marker `omero-annotate-ai table v3`) | primary container, linked to all `container_ids` | `create`; rows updated in place per batch / `extend` |
| config `annotation_config_table_<id>.yaml`, ns `openmicroscopy.org/omero/annotate/config` | primary container | `create`, session start/end, `describe`, notes; newest replaces older |
| label TIFF `<name>_<id>_mask.tif` (zlib), ns `openmicroscopy.org/omero/annotate/labels` | image | per unit at commit |
| polygon ROI `<name>_ROIs` (z/t/c, patch offset applied) | image | per unit, after the label |

Label and ROI descriptions carry `omero-annotate-ai:<table_id>:<annotation_id>`.
Table columns: `Image` (image link), `image_uri`, `image_name`, `annotation_id`, `level`,
`well`, `field`, `well_metadata`, `category` (training|validation|test), `timepoint`,
`z_slice`, `channel`, `is_volumetric`, `z_start`, `z_end`, `z_length`, `is_patch`,
`patch_x/y/width/height`, `processed`, `annotation_created_at`, `annotation_updated_at`,
`annotation_type` (`segmentation_mask` or `excluded`), `session`, `Roi` (ROI link),
`label_id`, `annotation_input_id`, `input_digest`, `mask_digest`, `input_iscc`,
`mask_iscc`, `claimed_by`, `claimed_at`. Unset IDs are 0, unset indices -1, unset
timestamps the string `None`.
Read with `omero_annotate_ai.omero.omero_functions.read_tracking_table(conn, table_id)`
(renames `Image`/`Roi` to `image_id`/`roi_id`).

## Resume and repair

`TrainingSet.open(table_id, conn)` uses the local `config.yaml` under
`<home>/<server>/<table_id>_*` if present, else the config on OMERO. Then per unit
(`src/omero_annotate_ai/reconcile.py`): adopt uploads carrying the unit's tag, upload what is missing, verify
IDs only the CSV has, reset units marked done with no mask and no IDs, finish an
interrupted review (keep the newer tagged upload). A local mask whose pixels differ from
the upload gets `error: needs review: ...` and is skipped; delete the local mask (keep
OMERO's) or the label file on OMERO (keep yours), then open again. Duplicate tagged
uploads are reported; `prune_duplicates=True` deletes the later copies. Idempotent.
Local sets have no repair step. Before 0.5 resume meant reloading the auto-saved YAML;
that no longer exists.

## Review and exclude

`ts.review()` reopens annotated (not excluded) units with their masks; unchanged masks
(also same labels in a wider dtype) keep their upload; a changed one is uploaded first,
then the old label and ROI deleted. Exclude: napari plugin button, or
`session.exclude(batch, ids)` (not in review sessions). An excluded unit has
`processed` true, `annotation_type: excluded`, no mask or uploads, is never shown again
and is left out of all exports; committing a mask for it in the same session undoes it.

## External tools (Cellpose, Fiji, QuPath, scripts)

`ts.prepare_external()` writes pending units to `annotation_input/{id}.tif` (and
`model_input/` stacked YXC/ZYXC for separate channels); units stay pending. Save label
TIFFs (0 = background) as `external_output/{annotation_id}_mask.tif` (or `.tiff`), then
`ts.collect_masks()` (folder relative to the set folder, glob allowed): one session,
same write order and claims as napari; rerunning handles only new masks. Cellpose is
not installed by this package.

## Training export

`export_training_set(ref, output_dir, conn=None, *, training_channels=None,
label_channel=None, apply_normalization=None, home=None)`: `ref` int = OMERO table
(`prepare_training_data_from_table`, downloads, writes `table_<id>.csv`, one training
channel only), folder = local set (`reorganize_local_data_for_training`, offline, only
the set's own channels; `file_mode` copy|move|symlink when called directly). Default
output used by the widget: `<home>/exports/<name>_<table_id>`.
```
train_input/ train_label/ val_input/ val_label/ [test_input/ test_label/]
[train_annotation_input/ val_annotation_input/]    separate channels
normalization.json
```
Image and label share `{annotation_id}.tif`; labels are instance masks. Returns a dict
with the folders and `stats`. micro-SAM: `setup_training(result, model_name="",
model_type="vit_b_lm", epochs=50, n_iterations=None, batch_size=2,
patch_shape=(512, 512), n_objects_per_batch=25)` then `run_training(cfg,
framework="microsam")`; checkpoints in `<output>/checkpoints/<model_name>/`, final
`<output>/<model_name>_final.pt`; CUDA if available; `learning_rate` is accepted but not
passed to micro-SAM. BiaPy: `DATA.TRAIN.PATH/GT_PATH` = `train_input/train_label`,
`DATA.VAL.FROM_TRAIN: False` + `val_*`; `processing.normalization.biapy_normalization(
config.normalization)` gives the matching block (or export with
`apply_normalization=True`). See the `biapy` skill.

## BioImage Archive (MIFA) and RO-Crate

Needs `bia-mifa-models` (git, see SKILL.md). MIFA needs `study.description`,
`study.funding_statement`, `keywords`, a link (or `source_dataset.url`) and `version`,
else `MifaNotReady` (from `bioimage_annotation_schema.mifa`) lists what is missing.
```python
from omero_annotate_ai import TrainingSet, prepare_bia_data_from_table
from omero_annotate_ai.schema_export import save_bia_package

ts = TrainingSet.open(table_id, conn)
ts.describe(version="1", keywords=["nuclei"], links=[{"url": "https://example.org"}],
            study={"title": "...", "description": "...", "funding_statement": "..."})
stats = prepare_bia_data_from_table(conn, table_id, "bia_staging")  # native dtype
save_bia_package(ts.config, stats["rows"], "bia_staging", "bia_bundle", move_data=True)
# local set: save_bia_package(ts.config, ts.units(), ts.bundle_dir, "bia_bundle")
```
Bundle: `metadata/Study_<acc>.yaml`, `Annotations_<acc>.yaml`, `Version_<acc>.yaml`,
`file_list_images.tsv`, `file_list_annotations.tsv` (with `pixel_sha256`, `iscc`),
`images/`, `annotations/`, `split-report.yaml`. Accession: `accession=` >
`source_dataset.identifier` > set name. Only annotated units. A mask changed on OMERO
raises `IntegrityError`. Exporting OMERO images needs `omero_server` in the config:
open an older set once with a connection. `ts.write_ro_crate()` writes the schema
documents and `ro-crate-metadata.json` into the set folder; `bas validate --mifa
<folder>` (schema library CLI) checks it.

## Deleting a set

```python
from omero_annotate_ai.omero.omero_functions import delete_training_set
report = delete_training_set(conn, table_id)               # dry run: lists IDs
delete_training_set(conn, table_id, dry_run=False)         # deletes, waits
```
Finds ROIs, labels, annotation inputs and configs by row ID and by tag (never by name),
deletes the table last. Objects owned by others raise `PermissionError` unless
`force_others=True`. The local folder is not touched. Notebook:
`notebooks/jupyter/other/cleanup_annotations.ipynb`.

## Connections (`omero/simple_connection.py`)

`SimpleOMEROConnection()`: `connect(host, username, password, group=None, secure=True,
verbose=True, port=None)` (host may be `wss://server/omero-ws`; raises
`ConnectionFailed`), `join_session(host, session_key, port=None, secure=True)`,
`create_connection_from_config({"host", "username", "password" | "session_key",
"group", "port", "secure"})` (saves history), `load_config_files()` (host/user/group/port
from `./.env` keys `HOST`, `USER_NAME`, `GROUP`, `PORT`, then history, then
`~/.ezomero`; never a password). History: `~/.omero-annotate-ai/connections.json`.
Remote calls first check the connection and rejoin or log in again with credentials
held in memory. IDR's OMERO API is closed to logins; use its OME-Zarr.

## napari plugin (experimental)

`pip install "omero-annotate-ai[napari]" "napari[all]"`; Plugins -> OMERO Annotate AI ->
Annotate training set. Lists local sets under the home folder, *Open folder...*, or
connect (password or session key, nothing stored) and *List sets* on a container. Keys:
Accept (`N`; empty image needs a second press), Skip (`Shift-N`, later), Exclude, Prev
(back through the session, Accept replaces the saved mask). *Review annotated images*
tickbox for review. *Finish session* with a note. Each accepted image is saved and
uploaded at once; the config on OMERO is updated at finish. Closing the window or dock
records the session as interrupted. Limits: cannot create or extend sets, 2D only,
needs micro-SAM 1.8.x (pinned), refuses viewers with layers named `image` or
`committed_objects`, do not open micro-SAM's own annotators at the same time.

## Doc / code disagreements (2026-11-20, origin/main)

- Version: `pyproject.toml` says 0.5.2, release notes list 0.5.3 and Unreleased above it,
  PyPI 0.5.2 is the older tag with config version 3.
- `docs/configuration.md` and `docs/concepts.md`: OME-Zarr sets "are always local". The
  code does not set it: `source: zarr` without `storage: local` is a validation error.
- `docs/configuration.md`: `container_ids` says "Set `container_id` too"; the code
  ignores `container_id` when `container_ids` is set (first ID is the primary).
- `docs/configuration.md` omits `ai_model.framework_version`. `workflow.batch_size` is
  described as images per session in the code and template; it counts units.
- `notebooks/marimo/annotate-from-yaml.py` tells users to save Cellpose masks as
  `external_output/<image name>_mask.tif`; `collect_masks` needs `<annotation_id>_mask.tif`.
- `pyproject.toml` calls the schema repository private; it is public (HTTP 200).
