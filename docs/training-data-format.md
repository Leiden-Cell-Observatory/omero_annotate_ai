# Training data format

An exported training-data folder is a contract: any tool can read it, and
`omero_annotate_ai` trainers read nothing else. This page describes the folder, how to read
it from Python, and what to guarantee if you write one yourself.

## Folder layout

```text
my_training_data/
├── train_input/              4ba674d4_0_0.tif   ← image
├── train_label/              4ba674d4_0_0.tif   ← label mask with the same name
├── val_input/
├── val_label/
├── test_input/               ← optional, together with test_label/
├── test_label/
├── train_annotation_input/   ← optional, the annotation channel when it differs
├── val_annotation_input/
├── normalization.json        ← how intensities were or should be scaled
└── table_<id>.csv            ← provenance (tracking_table.csv for a local set)
```

`train_*` and `val_*` are required and must hold at least one pair. `test_input/` and
`test_label/` are both present or both absent. The `*_annotation_input/` folders are
optional and not read by the trainers.

## Files

- Every image and label is a TIFF named `<annotation_id>.tif`.
- An image and its label have the same file name, one in `*_input/`, one in `*_label/`.
  A file without a partner is an error.
- Labels are instance masks: `0` is background and each object has its own integer.
- A label has the axes of its image without the channel axis.

## Axes

The axes are guessed from the shape of the first training image. Channels are last.

| Shape | Axes |
|---|---|
| `(y, x)` | `YX` |
| `(y, x, c)`, last axis 4 or less | `YXC` |
| `(z, y, x)`, last axis more than 4 | `ZYX` |
| `(z, y, x, c)` | `ZYXC` |

A trailing axis of 4 or less is read as channels. All images in one folder are expected to
share these axes and channel count.

## Intensities and `normalization.json`

By default images keep their native bit depth and `normalization.json` says what the
trainer should apply. With `apply_normalization=True` at export the images are written
as float32 in [0, 1] and the file records the bounds that were used.

| Field | Meaning |
|---|---|
| `method` | `percentile` or `none` |
| `percentile_low`, `percentile_high` | Percentiles mapped to 0 and 1 (`1.0` and `99.8` by default) |
| `clip` | Clip the normalised values to [0, 1] |
| `applied_at_export` | `true`: images are already normalised float32 in [0, 1]. `false`: native dtype, the trainer applies `method` |
| `dtype` | `float32` or `native` |
| `bounds_per_image` | Only when applied at export: `{annotation_id: [[low, high], ...]}`, one pair per channel |

The file is optional. Without it a trainer assumes the defaults of the config
(`percentile`, not applied).

## Provenance

For an OMERO set the export contains `table_<id>.csv`, a copy of the tracking table. For a
local set it is `tracking_table.csv`. It links each file to its annotation unit, split and
source image. Trainers do not read it.

## Reading it

```python
from omero_annotate_ai import load_training_data

data = load_training_data("my_training_data")
print(data.axes, data.n_channels, data.dtype)
for image_path, label_path in data.pairs("train"):
    ...
```

`load_training_data` raises `InvalidTrainingData` if a required folder is missing, files
are unpaired, train or val is empty, a file cannot be read, or the first label does not
match the shape of the first image. It returns a `TrainingData` with:

| Field | Content |
|---|---|
| `root` | The folder |
| `train_input`, `train_label`, `val_input`, `val_label` | Folders (`Path`) |
| `test_input`, `test_label` | Folders, or `None` without a test split |
| `normalization` | Content of `normalization.json`, `{}` if absent |
| `axes`, `shape`, `n_channels`, `dtype` | From the first training image |

`data.pairs("train" | "val" | "test")` gives the sorted `(image, label)` paths of a split
(empty for a missing test split).

## Writing it from another tool

A folder written by another tool works with every trainer if it:

1. has `train_input/`, `train_label/`, `val_input/` and `val_label/`, each non-empty, and
   both test folders or neither;
2. names the label of every image identically and has no unpaired files;
3. uses TIFF with channels last and labels without a channel axis;
4. holds integer instance labels, `0` for background;
5. adds `normalization.json` when images are normalised, with `applied_at_export: true`
   only if they really are float32 in [0, 1].

Check it with `load_training_data(folder)`.

## Trainers

`train(data, trainer, output_dir, **params)` takes a folder or a `TrainingData` and
returns a `TrainingResult` (`trainer`, `package_version`, `model_path`, `output_dir`,
`params`, `data_root`, `extras`). The same call from a shell:

```bash
omero-annotate-ai train my_training_data --trainer cellpose --output models/hela \
    --set epochs=100 --set learning_rate=1e-5
```

`--params file.yaml` reads parameters from a YAML file; `--set` overrides them. Values
are read as JSON first, then YAML, so quote values that must stay text: `--set 'model_name="012"'`.
A dotted key goes into the trainer's `config`: `--set MODEL.ARCHITECTURE=resunet`
(BiaPy). The command prints the model path and writes `training_result.json` to the
output folder.

| Trainer | Install | Channels | 2D / 3D | Normalisation | Model |
|---|---|---|---|---|---|
| `microsam` | `omero-annotate-ai[microsam]` | 1 | 2D | Raw transform to [0, 255] when loading: from [0, 1] if applied at export, percentile bounds if `percentile`, none only accepts data already in [0, 255] | `<output>/<model_name>_final.pt`, checkpoints in `<output>/<model_name>/` |
| `cellpose` | `omero-annotate-ai[cellpose]` | up to 3 | 2D; `ZYX` volumes are trained plane by plane | Cellpose normalises itself unless the export already did | Under `<output>/models/` (path in `result.model_path`) |
| `biapy` | `omero-annotate-ai[biapy]` | any | 2D and 3D | Identity if applied at export or `none`; otherwise BiaPy clips at the percentiles and standardises per image | `<output>/<model_name>/checkpoints/<model_name>_1-checkpoint-best.pth`, logs in `<output>/<model_name>/logs/` |

Parameters (defaults in brackets):

- **microsam**: `model_name`, `model_type` (`vit_b_lm`), `epochs` (50), `n_iterations`
  (overrides `epochs`), `batch_size` (2), `learning_rate` (1e-5), `patch_shape`
  (`(512, 512)`), `n_objects_per_batch` (25), `train_instance_segmentation` (`True`).
- **cellpose**: `model_name`, `pretrained_model` (`cpsam`), `epochs` (100),
  `learning_rate` (1e-5), `batch_size` (1), `weight_decay` (0.1), `min_train_masks` (5).
- **biapy**: `model_name`, `epochs` (100), `patch_size` (spatial only; `(256, 256)` in
  2D, `(16, 128, 128)` in 3D), `architecture` (`unet`), `config` (a nested dict merged
  over the generated BiaPy config, which is saved as `<output>/biapy.yaml`).

When `model_name` is empty a timestamped name is used.

A trainer is any object with a `name` and a `train(data, output_dir, **params)` method
that returns a `TrainingResult`; pass it instead of the name.
