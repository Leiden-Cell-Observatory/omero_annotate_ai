# Configuration (YAML)

Every run is described by one YAML file. The widgets write it for you; this page lists
every option for when you want to read, edit or write one by hand.

Only `name` is required. Everything else has a default.

!!! tip "Typos are errors"
    Unknown keys are refused when a config is loaded, in every section, so a misspelt
    setting never falls back to a default unnoticed. Keys from older versions name their
    replacement.

## A minimal example

```yaml
config_version: 4
name: hela_nuclei_v1

omero:
  container_type: dataset
  container_id: 123

sampling:
  label_channel: 0         # annotate channel 0 (e.g. DAPI)
  patches: {width: 512, height: 512, per_image: 2}

training:
  train_n: 5               # images for training
  validate_n: 2            # images for validation

ai_model:
  pretrained_from: vit_b_lm

workflow:
  batch_size: 4            # images per napari session

storage: omero             # "local" = write nothing to OMERO
```

Load and run it:

```python
from omero_annotate_ai import TrainingSet, load_config_from_yaml

config = load_config_from_yaml("hela_nuclei_v1.yaml")  # this file is not changed
ts = TrainingSet.create(config, conn)  # the set's own copy goes to its folder
ts.annotate()
```

A fully commented template is available from Python:

```python
from omero_annotate_ai.core.annotation_config import get_config_template

print(get_config_template())
```

## Two kinds of sections

A config (version 4) has two kinds of sections:

- **Schema sections** describe the annotation set: its name, licence, authors, study,
  source data, how it was annotated, the classes, what was sampled from each image and
  how the set is split. They are the models of the
  [Bioimage Annotation Schema](https://github.com/Leiden-Cell-Observatory/bioimage-annotation-schema)
  (`bioimage_annotation_schema.datamodel`), so a set's description is the same in
  every tool that uses the schema, and the RO-Crate, MIFA and BioImage Archive exports
  need no translation.
- **Tool sections** hold what omero-annotate-ai needs to run: where the images come
  from, how many go into each split, normalisation, the model, how sessions run and
  where results are kept.

| Kind | Sections |
|---|---|
| Schema | `name`, `description`, `version`, `url`, `license`, `changes`, `previous_version`, `authors`, `keywords`, `links`, `ai_models_trained`, `study`, `source_dataset`, `methodology`, `classes`, `sampling`, `split`, `notes`, `extensions` |
| Tool | `omero`, `zarr`, `source`, `storage`, `training`, `normalization`, `ai_model`, `display`, `workflow` |
| Written by omero-annotate-ai | `config_version`, `created`, `identity`, `omero_server`, `sessions` |

`sampling` and `split` decide which units a set has, so they are fixed once the set
exists. The other schema sections can be changed later with
[`TrainingSet.describe`](#changing-the-description-of-a-set).

## Tool sections

### `omero`: where the images come from

| Key | Default | Meaning |
|---|---|---|
| `container_type` | `dataset` | `project`, `dataset`, `plate`, `screen` or `image` |
| `container_id` | `0` | ID of the container. The tracking table and the set's config are attached here. |
| `container_ids` | – | List of several containers of the same type to draw images from. Set `container_id` too. |
| `source_desc` | `""` | Free-text description of the source. |
| `well_filters` | – | Plates and screens: keep wells by their key-value pairs (map annotations, e.g. MIHCSME assay conditions), e.g. `{treatment: [DMSO]}`. Several keys must all match. |
| `well_filter_mode` | `include` | `include` keeps matching wells, `exclude` drops them. |

```yaml
omero:
  container_type: plate
  container_id: 456
  well_filters:
    treatment: [DMSO, control]
    cell_line: [HeLa]
  well_filter_mode: include
```

### `zarr`: OME-Zarr images instead of OMERO

With `source: zarr` images are read from OME-Zarr files or URLs (for example the
BioImage Archive) and no OMERO connection is needed; such sets are always
`storage: local`.

| Key | Default | Meaning |
|---|---|---|
| `uris` | `[]` | OME-Zarr plates, wells or images (http(s) URL or local path). |
| `level` | `0` | Multiscale level to read. |
| `fields` | `first` | Fields per well: `first` or `all`. |
| `wells` | – | Only these wells, e.g. `[B2]`. |
| `well_metadata` | – | CSV with a `well` column (optional `plate`) and metadata columns, for filters and stratified splits. |
| `well_filters`, `well_filter_mode` | – , `include` | As for OMERO plates, on the `well_metadata` columns. |

An example is in
[`notebooks/marimo/configs/`](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/tree/main/notebooks/marimo/configs).

### `training`: how many units go to each split

There are two ways to split, chosen by `segment_all`. Which unit is split, the
stratification and the seed are part of the description, in
[`split`](#split-how-the-set-is-split).

=== "Pick a number of images (default)"

    ```yaml
    training:
      segment_all: false
      train_n: 10        # at least 1
      validate_n: 3
      test_n: 0
    ```
    Images are drawn at random from the container. Counts are split units (images, or
    wells with `split.unit: well`), not annotation units: with patches or several
    z-slices, one image gives several annotation units.

=== "Annotate everything"

    ```yaml
    training:
      segment_all: true
      train_fraction: 0.7
      validation_fraction: 0.3
      test_fraction: 0.0     # fractions add up to at most 1
    ```
    All images in the container are used and split by fraction.

Setting the counts in fraction mode (or the other way round) gives a warning and has no
effect.

### `normalization`: intensity scaling of model input

How raw intensities become model input is defined once, in the config, and travels with
the annotation set (OMERO, YAML, MIFA export):

```yaml
normalization:
  method: "percentile"      # "percentile" or "none"
  percentile_low: 1.0       # per-image percentile mapped to 0
  percentile_high: 99.8     # per-image percentile mapped to 1
  clip: true
  apply_at_export: false    # true: write float32 normalised images
```

Training data is exported at native bit depth unless `apply_at_export` is set. With
`apply_at_export: true` the model input images are written as float32 and the exact
bounds used for each image are recorded in `normalization.json` next to the data;
labels are never changed. Without it, `normalization.json` records what the trainer
must apply; micro-SAM training applies it when loading.

`omero_annotate_ai.processing.normalization.biapy_normalization(config.normalization)`
returns the matching BiaPy `DATA.NORMALIZATION` block. BiaPy's `div` and `scale_range`
types guess the bit depth from the image maximum, so exporting with
`apply_at_export: true` (BiaPy then applies the identity) is the most reliable route.

### `ai_model`: the model

| Key | Default | Meaning |
|---|---|---|
| `pretrained_from` | `vit_b_lm` | micro-SAM model opened in napari, e.g. `vit_b_lm`, `vit_l_lm` or `vit_b_em_organelles`. |
| `framework` | `micro_sam` | `micro_sam`, `cellpose` or `web_annotation`. Recorded as metadata; the workflow you call decides which tool is used. |
| `model_name`, `model_version`, `model_url`, `model_doi` | | Describe the model for the record. |
| `training_mode` | `inference` | `inference`, `finetuning` or `from_scratch`. |

When `methodology.tools` is empty, the exports describe the tool from this section.

### `display`: context channels in napari

| Key | Default | Meaning |
|---|---|---|
| `label_colormap` | `gray` | napari colormap of the annotated channel. |
| `context_channels` | `[]` | Extra channels shown in napari for orientation only; never used for training. Must differ from the label channel. |
| `context_channel_names` | – | Display names for the context channels (same length). |
| `context_channel_colormaps` | – | napari colormaps for the context channels, e.g. `[magenta, green]`. |

### `workflow`: how the run behaves

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | `0` | Units per napari session; a batch's masks are saved and uploaded when its napari window closes. `0` = all at once. |
| `claim_hours` | `6` | OMERO sets: how long a session's claim on the units it shows lasts before another computer may take them over. Your own claims on the same computer never block you. |

### `storage`: where rows and masks are kept

| Value | Meaning |
|---|---|
| `omero` (default) | A tracking table on OMERO (its ID identifies the set), masks uploaded as ROIs and label files, plus a local copy in the set's folder. |
| `local` | Nothing is written to OMERO; images, masks and `tracking_table.csv` stay in the set's folder. Use on servers you cannot write to. OME-Zarr sets (`source: zarr`) are always `local`. |

Fixed when the set is created. Each set has its own folder:
`<home>/<server>/<table id>_<name>/` (or `<home>/local/<id>_<name>/` for local sets),
where `<home>` is `home=`, else `$OMERO_ANNOTATE_AI_HOME`, else `~/omero_annotate_ai`.

## Schema sections

### `sampling`: what to annotate in each image

**Channels**

| Key | Default | Meaning |
|---|---|---|
| `label_channel` | `0` | Channel you annotate on (0-based). |
| `input_channels` | `[label_channel]` | Channel(s) the model is trained on. If different from `label_channel`, they are saved to `model_input/`. |

Channels shown only for orientation are set in
[`display`](#display-context-channels-in-napari).

**Timepoints and z-slices**

`timepoints` and `z_slices` each take a `mode` and, depending on it, `values` or `n`:

| Key | Default | Meaning |
|---|---|---|
| `timepoints` | `{mode: specific, values: [0]}` | `specific` (the `values` list), `all`, or `random` (`n` drawn per image with the seed). |
| `z_slices` | `{mode: specific, values: [0]}` | The same, for z-slices. |
| `volumetric` | `false` | Annotate each timepoint as one 3D volume, from the first to the last selected z-slice, instead of separate planes. |

**Patches**

Leave `patches` out to annotate whole planes. When it is present, crops are annotated:

| Key | Default | Meaning |
|---|---|---|
| `width`, `height` | required | Patch size in pixels. Square patches are safest. |
| `per_image` | `1` | Non-overlapping patches per plane. An image smaller than the patch gives one full-image unit. |
| `random` | `true` | Place patches at random positions. |

### `split`: how the set is split

| Key | Default | Meaning |
|---|---|---|
| `strategy` | `random_split` | `random_split`, `expert_review` or `cross_validation`. Recorded as metadata. |
| `unit` | `image` | Level at which data are assigned to a split: `image` or `well`. |
| `rationale` | – | Why this unit was chosen. |
| `stratify_by` | – | Well keys to balance the split on. |
| `seed` | – | Seed for every random choice (split, timepoints, z-slices, patches). Empty: one is picked and stored when the set is created, so the selection can be repeated. |

**Split unit.** All annotations of one unit end up in the same split, and
`train_n`/`validate_n`/`test_n` and the fractions count units:

- `image` (default): an image (a plate field) with all its timepoints and z-slices.
  Fields from one well may end up in different splits.
- `well`: all fields of a plate well. Needs plate data with well information
  (OMERO plates or OME-Zarr HCS plates).

Whether fields of one well are independent enough to be split apart depends on the
task, so the choice is recorded rather than imposed. Use `rationale` to say why, for
example:

```yaml
split:
  unit: image
  rationale: >
    Fields within a well do not overlap; for nucleus segmentation they are
    treated as independent.
```

The unit and rationale are added to the MIFA annotation method. Every export also
writes a split report, `split-report.yaml`: `ts.write_ro_crate()` puts it in the set's
folder and `save_bia_package` in the bundle. It lists per split the number of images,
wells and conditions, the images and wells with annotations in more than one split,
and the conditions that miss a split, followed by a summary in a few sentences. The
same report is available in Python as
`omero_annotate_ai.rows.split_report(rows, split_unit, stratify_by, split_rationale)`
and `split_summary(report)`.

**Stratified splits.** `stratify_by` divides the units over the splits per condition,
so that validation and test cover every condition instead of whatever a random draw
gives. Keys are well key-value pairs (from OMERO map annotations or OME-Zarr well
attributes), or `well` itself:

```yaml
split:
  stratify_by: [cell_line, treatment]
training:
  segment_all: true
  train_fraction: 0.6
  validation_fraction: 0.2
  test_fraction: 0.2
```

- Fraction mode: within each condition, validation and test get their rounded share
  and training the rest (4 fields at 0.6/0.2/0.2 become 2/1/1).
- Count mode: each split draws units in turn from all conditions, test and
  validation first.

When a condition has too few units to reach every split (for example one well per
condition with `unit: well`), the split still runs with a warning, and the split report
lists the conditions that miss a split. A key that no image has is an error, as it is
most likely a typo. Images lacking a key that others have form their own `(missing)`
group.

### Description: study, authors, methods

These sections describe your set for others and are used by the RO-Crate and the
[BioImage Archive export](tutorials/bioimage-archive-export.md). They do not change how
the run works.

```yaml
license: CC-BY-4.0                          # CC-BY-4.0 or CC0-1.0
version: "1"
url: https://doi.org/10.5281/zenodo.0000000   # where the set is published, once it is
authors:
  - first_name: Jane
    last_name: Doe
    email: j.doe@example.org
    orcid: https://orcid.org/0000-0000-0000-0000
    affiliations:
      - {name: Leiden University, ror: "https://ror.org/027bh9e22"}
    roles: [annotator]
keywords: [nuclei, segmentation]
links:
  - {url: "https://github.com/example/project", description: Analysis code}

study:
  title: "Nuclei annotations in HeLa cells"
  description: "Instance masks of nuclei for training segmentation models."
  organism: "Homo sapiens"
  imaging_method: "widefield fluorescence microscopy"
  funding_statement: "Funded by NWO."

source_dataset:
  identifier: "S-BIAD123"
  url: "https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD123"
  description: "HeLa cells, Hoechst channel"

methodology:
  annotation_types: [segmentation_mask]     # MIFA names: point_annotations, bounding_boxes, class_labels, ...
  mask_kind: instance                       # instance or semantic
  method: semi_automatic                    # manual, semi_automatic or automatic
  criteria: "Every nucleus fully inside the field."
  coverage: representative                  # all, representative or partial
  confidence_level: "One trained annotator, checked by a second."
  tools:
    - {name: micro-sam, version: "1.8.0", mode: interactive}   # automatic, interactive, human_in_the_loop

classes:
  - {name: nucleus, color: "#0FA3B1"}
```

- `study` also takes `publications`, `grants` and `acknowledgements`; only `title` is
  required.
- `source_dataset` needs a `description`.
- `methodology` needs `annotation_types` and `criteria`. A semantic segmentation is
  `segmentation_mask` with `mask_kind: semantic`.
- `classes` entries can also carry `label_value` (the pixel value in a semantic mask)
  and `ontology_term` (an IRI, e.g. a Cell Ontology term).

### Notes and extensions

Anything the schema does not cover can be recorded in two open fields. Both are stored
with the configuration and in `annotation-set.yaml`.

```yaml
notes: "Wells A1-B2 use the key 'celline' instead of 'cellline' in OMERO."
extensions:
  my_lab:                    # a namespace key
    instrument: "Incucyte SX5"
    objective: "10x"
```

### Changing the description of a set

`TrainingSet.config` is a copy; changing it does not change the set. To change the
description of an existing set, locally and on OMERO, use `describe`:

```python
ts = TrainingSet.open(table_id, conn)
ts.describe(
    version="1",
    keywords=["nuclei", "segmentation"],
    study={
        "title": "Nuclei annotations in HeLa cells",
        "description": "Instance masks of nuclei for training segmentation models.",
        "funding_statement": "Funded by NWO.",
    },
    links=[{"url": "https://github.com/example/project"}],
)
```

`describe` takes the schema sections except `sampling` and `split`
(`omero_annotate_ai.trainingset.DESCRIPTION_FIELDS`). Values can be plain dicts or the
schema's models (`bioimage_annotation_schema.datamodel`).

## Filled in by omero-annotate-ai

`config_version` (4), `created`, `split.seed` (if you left it empty), `identity` (the
set's table ID or local ID), `omero_server` and `sessions` (who annotated when, annotate
or review, how many units, and the session note) are written when a set is created and
annotated.

`omero_server` says which OMERO server the images are on: the `host` the connection
uses and the server's `database_uuid`, which stays the same if the server moves. The
exported set needs it, because its records name images only as `omero:image:<id>`.
It is taken from the connection when a set is created, and filled in when an older set,
made before it was recorded, is opened with a connection. You can add `web_url`, the address of the
server's OMERO.web, for readers of the exported set. The
rows (one per annotation unit, with its split and state) live in `tracking_table.csv`
and the OMERO table, not in the config. To start a new set from a saved config, remove
`identity` and `sessions`.

## Example recipes

??? example "Time-lapse: 5 random timepoints"
    ```yaml
    sampling:
      timepoints: {mode: random, n: 5}
    ```

??? example "Annotate fluorescence, train on brightfield"
    ```yaml
    sampling:
      label_channel: 0          # nuclei stain
      input_channels: [1]       # brightfield
    ```

??? example "Large images on a CPU"
    ```yaml
    sampling:
      patches: {width: 256, height: 256, per_image: 4}
    ai_model:
      pretrained_from: vit_b_lm
    workflow:
      batch_size: 5
    ```

??? example "3D volumes"
    ```yaml
    sampling:
      z_slices: {mode: specific, values: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}
      volumetric: true
    ```

## Working with configs in Python

```python
from omero_annotate_ai import load_config, create_default_config

config = load_config("my_run.yaml")  # also accepts a dict
config.save_yaml("copy.yaml")
print(config.to_yaml())

config = create_default_config()  # start from defaults
config.name = "new_run"
config.omero.container_id = 123
config.sampling.label_channel = 1
config.split.seed = 42
```

Full details: [API: Configuration](api/config.md).
