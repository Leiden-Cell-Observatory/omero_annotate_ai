# Prepare training data

Once you have annotations, turn them into a training dataset: pairs of images and label
masks, split into training, validation and (if you annotated one) test folders. The same layout works for micro-SAM,
BiaPy, DL4MicEverywhere and most other segmentation tools.

```text
my_training_data/
├── train_input/    4ba674d4_0_0.tif   ← image
├── train_label/    4ba674d4_0_0.tif   ← label mask with the same name
├── val_input/
├── val_label/
├── test_input/     ← only when the set has test units
└── test_label/
```

Labels are instance masks: 0 is background, and each object has its own number.

## From OMERO (`storage: omero`)

Use this when your annotations are stored in OMERO. It works from any computer that can
reach the server, and needs no display.

```python
from omero_annotate_ai import (
    create_omero_connection_widget,
    create_training_data_widget,
    prepare_training_data_from_table,
)

conn_widget = create_omero_connection_widget()
conn_widget.display()
conn = conn_widget.get_connection()

training_widget = create_training_data_widget(connection=conn)
training_widget.display()  # pick the container and the annotation table
```

```python
table_id = training_widget.get_selected_table_id()

result = prepare_training_data_from_table(
    conn=conn,
    table_id=table_id,
    output_dir="my_training_data",
)
print(result["stats"])
```

Only finished units with a mask are used. Each unit goes to the split recorded in the
table's `category` column (training, validation or test), the split chosen when the
training set was created. A copy of the tracking table is saved alongside
(`table_<id>.csv`).

!!! note "Things to know"
    - Images keep their original bit depth. `normalization.json` records how to scale
      them; micro-SAM training applies it when loading (see
      [normalisation settings](../configuration.md#normalization-intensity-scaling-of-model-input)).
    - A label that no longer matches its recorded pixel digest stops the export, so a
      mask changed after upload is never trained on unnoticed.
    - If you annotated one channel but want to train on another, pass
      `label_channel=` and `training_channels=`.

## From a local training set

When the images and masks are on your disk (`annotation_input/` and
`annotation_output/`), reorganise them without contacting OMERO:

```python
from omero_annotate_ai import TrainingSet, reorganize_local_data_for_training

ts = TrainingSet.open("~/omero_annotate_ai/local/<id>_hela_nuclei_v1")
result = reorganize_local_data_for_training(
    ts.config,
    annotation_dir=ts.bundle_dir,  # its tracking_table.csv says what is annotated
    include_test=True,
)
```

By default the data is copied to a folder next to the set's folder, named
`<set folder>_training`. Images keep their original bit depth, and
`include_test=True` also writes `test_input/` and `test_label/`.

## Train a model

### micro-SAM

Fine-tune micro-SAM on your annotations so its suggestions fit your images better. A
CUDA GPU is strongly recommended.

```python
from omero_annotate_ai import setup_training, run_training

training_config = setup_training(
    result,  # from prepare_training_data_from_table
    model_name="hela_nuclei_v1",
    model_type="vit_b_lm",  # start from this pretrained model
    epochs=50,
    batch_size=2,
    patch_shape=(512, 512),
)
training_results = run_training(training_config, framework="microsam")
```

`epochs` sets the training length. Pass `n_iterations=` instead for a fixed number of
steps, for example `n_iterations=10` for a quick check that everything runs; it
overrides `epochs`.

The trained model is saved as `<output_dir>/<model_name>_final.pt`, with checkpoints in
`<output_dir>/checkpoints/`. Load it in the micro-SAM napari plugin by choosing the
checkpoint as a custom model. Export to the [BioImage Model Zoo](https://bioimage.io) format with micro-SAM's own tools.

Notebook: [micro-SAM training (marimo)](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-training-microsam.py)

### BiaPy

[BiaPy](https://biapy.readthedocs.io) trains instance segmentation models (U-Net and
others) from a YAML file. Point its data paths at the prepared folders:

```yaml
DATA:
  TRAIN:
    PATH: my_training_data/train_input
    GT_PATH: my_training_data/train_label
  VAL:
    FROM_TRAIN: False
    PATH: my_training_data/val_input
    GT_PATH: my_training_data/val_label
```

Notebook: [export for BiaPy or DL4MicEverywhere](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/training/omero-training-export.ipynb)

### DL4MicEverywhere

[DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere) runs ZeroCostDL4Mic
notebooks (StarDist, Cellpose, U-Net, …) in containers. Give it the prepared folder as
its data directory.

Notebook: [export for BiaPy or DL4MicEverywhere](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/training/omero-training-export.ipynb)

## Tips

- Keep at least one validation image (`validate_n: 1` or more); micro-SAM training fails
  with an empty validation folder.
- A few well-annotated images go a long way when fine-tuning micro-SAM. Start with 5 to 10
  and add more where the model still fails.
