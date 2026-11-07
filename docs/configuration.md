# Configuration (YAML)

Every run is described by one YAML file. The widgets write it for you; this page lists
every option for when you want to read, edit or write one by hand.

Only `name` is required. Everything else has a default.

!!! warning "Misspelled keys are ignored without a warning"
    If a key is not recognised, it is silently dropped and the default is used. After
    loading a config, print the sections you changed to check that they were read:
    `print(config.spatial_coverage)`.

## A minimal example

```yaml
name: hela_nuclei_v1

omero:
  container_type: dataset
  container_id: 123

spatial_coverage:
  channels: [0]            # annotate channel 0 (e.g. DAPI)
  use_patches: true
  patch_size: [512, 512]
  patches_per_image: 2

training:
  train_n: 5               # images for training
  validate_n: 2            # images for validation

ai_model:
  pretrained_from: vit_b_lm

workflow:
  batch_size: 4            # images per napari session
  read_only_mode: false    # true = write nothing to OMERO

output:
  output_directory: ./annotations/hela_nuclei_v1
```

Load and run it:

```python
from omero_annotate_ai import load_config_from_yaml, create_pipeline

config = load_config_from_yaml("hela_nuclei_v1.yaml")
config.config_file_path = None     # keep this file unchanged; progress is saved in output_directory
pipeline = create_pipeline(config, conn)
table_id, config = pipeline.run_full_micro_sam_workflow()
```

A fully commented template is available from Python:

```python
from omero_annotate_ai.core.annotation_config import get_config_template
print(get_config_template())
```

## `omero`: where the images come from

| Key | Default | Meaning |
|---|---|---|
| `container_type` | `dataset` | `project`, `dataset`, `plate`, `screen` or `image` |
| `container_id` | `0` | ID of the container. The config is attached here at the end of a run. |
| `container_ids` | – | List of several containers of the same type to draw images from. Set `container_id` too. |
| `source_desc` | `""` | Free-text description of the source. |
| `well_filters` | – | Plates only: keep wells by their key-value pairs, e.g. `{treatment: [DMSO]}`. Several keys must all match. |
| `well_filter_mode` | `include` | `include` keeps matching wells, `exclude` drops them. |
| `table_id` | – | Filled in by the pipeline: the current tracking table. |

```yaml
omero:
  container_type: plate
  container_id: 456
  well_filters:
    treatment: [DMSO, control]
    cell_line: [HeLa]
  well_filter_mode: include
```

## `spatial_coverage`: what to annotate in each image

**Channels**

| Key | Default | Meaning |
|---|---|---|
| `channels` | `[0]` | Channels loaded from each image (0-based). |
| `label_channel` | first of `channels` | Channel you annotate on. |
| `training_channels` | `[label_channel]` | Channel(s) saved as model input. If different from `label_channel`, they are saved to `model_input/`. |
| `context_channels` | `[]` | Extra channels shown in napari for orientation only; never used for training. |
| `context_channel_names` | – | Display names for the context channels. |
| `context_channel_colormaps` | – | napari colormaps for the context channels, e.g. `[magenta, green]`. |

**Timepoints and z-slices**

| Key | Default | Meaning |
|---|---|---|
| `timepoints` | `[0]` | Timepoints to use (with `timepoint_mode: specific`). |
| `timepoint_mode` | `specific` | `specific` (the list above), `all`, or `random`. |
| `n_timepoints` | – | Number of random timepoints (with `random`). |
| `z_slices` | `[0]` | Z-slices to use (with `z_slice_mode: specific`). |
| `z_slice_mode` | `specific` | `specific`, `all`, or `random`. |
| `n_slices` | – | Number of random slices (with `random`). |
| `three_d` | `false` | Annotate each timepoint as one 3D volume instead of separate planes. |
| `z_range_start`, `z_range_end` | – | Volume range in 3D mode; filled from `z_slices` if not set. |

**Patches**

| Key | Default | Meaning |
|---|---|---|
| `use_patches` | `false` | Annotate crops instead of whole planes. |
| `patch_size` | `[512, 512]` | Patch size in pixels. Square patches are safest. |
| `patches_per_image` | `1` | Non-overlapping patches per plane. An image smaller than the patch gives one full-image unit. |
| `random_patches` | `true` | Place patches at random positions. |

## `training`: train, validation and test sets

There are two ways to split, chosen by `segment_all`.

=== "Pick a number of images (default)"

    ```yaml
    training:
      segment_all: false
      train_n: 10        # at least 1
      validate_n: 3
      test_n: 0
    ```
    Images are drawn at random from the container. Counts are images, not units: with
    patches or several z-slices, one image gives several units.

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

| Key | Default | Meaning |
|---|---|---|
| `validation_strategy` | `random_split` | Recorded as metadata: `random_split`, `expert_review` or `cross_validation`. |
| `quality_threshold` | – | Recorded as metadata. |

## `ai_model`: the model

| Key | Default | Meaning |
|---|---|---|
| `pretrained_from` | `vit_b_lm` | micro-SAM model opened in napari, e.g. `vit_b_lm`, `vit_l_lm` or `vit_b_em_organelles`. |
| `framework` | `micro_sam` | `micro_sam`, `cellpose` or `web_annotation`. Recorded as metadata; the workflow you call decides which tool is used. |
| `model_name`, `model_version`, `model_url`, `model_doi` | | Describe the model for the record. |
| `training_mode` | `inference` | `inference`, `finetuning` or `from_scratch`. |

## `workflow`: how the run behaves

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | `0` | Units per napari session; progress is saved after each batch. `0` = all at once. |
| `read_only_mode` | `false` | `true`: write nothing to OMERO; keep images, masks and `tracking_table.csv` locally. Use on public servers such as the IDR. |
| `resume_from_table` | `false` | Not reliable in this version; to resume, reload the saved YAML instead. See [Resume or repeat a run](tutorials/resume-and-repeat.md). |

## `output`

| Key | Default | Meaning |
|---|---|---|
| `output_directory` | `./annotations` | Working directory for this run. Progress is saved here as `annotation_config_<name>.yaml`. |

## Metadata for sharing

These sections describe your study for others and are used by the
[BioImage Archive export](tutorials/bioimage-archive-export.md). They do not change how
the run works.

```yaml
authors:
  - name: Jane Doe
    affiliation: Leiden University
    email: j.doe@example.org
    orcid: https://orcid.org/0000-0000-0000-0000

study:
  title: "Nuclei annotations in HeLa cells"
  description: "Instance masks of nuclei for training segmentation models."
  keywords: [nuclei, segmentation]
  organism: "Homo sapiens"
  imaging_method: "widefield fluorescence microscopy"
  funding_statement: "Funded by NWO."

dataset:
  source_dataset_id: "S-BIAD123"
  source_dataset_url: "https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD123"
  source_description: "HeLa cells, Hoechst channel"
  license: "CC-BY-4.0"

annotation_methodology:
  annotation_type: segmentation_mask        # bounding_box, point, classification, semantic_segmentation
  annotation_method: semi_automatic         # manual, automatic
  annotation_criteria: "Every nucleus fully inside the field."
  annotation_coverage: representative       # all, partial
  annotation_tool:
    name: micro-sam
    mode: interactive                       # automatic, human_in_the_loop

tags: [segmentation, nuclei]
```

Other top-level keys: `version` (your version of this config), `documentation` and
`repository` (URLs), and `feature_types` (list of `name` and `color`).

## Filled in by the pipeline

`schema_version`, `created`, `config_file_path` and the `annotations:` list are written
by the pipeline. `annotations:` holds one entry per annotation unit with its split,
position and `processed` flag; this is what makes a run resumable. Remove it when you
reuse a config for a new run.

## Example recipes

??? example "Time-lapse: 5 random timepoints"
    ```yaml
    spatial_coverage:
      channels: [0]
      timepoint_mode: random
      n_timepoints: 5
    ```

??? example "Annotate fluorescence, train on brightfield"
    ```yaml
    spatial_coverage:
      channels: [0, 1]
      label_channel: 0          # nuclei stain
      training_channels: [1]    # brightfield
    ```

??? example "Large images on a CPU"
    ```yaml
    spatial_coverage:
      use_patches: true
      patch_size: [256, 256]
      patches_per_image: 4
    ai_model:
      pretrained_from: vit_b_lm
    workflow:
      batch_size: 5
    ```

??? example "3D volumes"
    ```yaml
    spatial_coverage:
      channels: [0]
      three_d: true
      z_slices: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ```

## Working with configs in Python

```python
from omero_annotate_ai import load_config, create_default_config

config = load_config("my_run.yaml")       # also accepts a dict
config.save_yaml("copy.yaml")
print(config.to_yaml())

config = create_default_config()          # start from defaults
config.name = "new_run"
config.omero.container_id = 123
```

Full details: [API: Configuration](api/config.md).
