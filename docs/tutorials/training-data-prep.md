# Prepare training data

Once you have annotations, turn them into a training dataset: pairs of images and label
masks, split into training, validation and (if you annotated one) test folders. The same layout works for micro-SAM,
BiaPy, DL4MicEverywhere and most other segmentation tools.

The folder holds `train_input/`, `train_label/`, `val_input/`, `val_label/` and, when the
set has test units, `test_input/` and `test_label/`, plus `normalization.json`. Images and
labels have the same file name. The [training data format](../training-data-format.md)
describes it in full.

## Export a training set

One call exports any training set: by table ID for a set on OMERO (the data is
downloaded, from any computer that can reach the server, no display needed), or by
folder for a local set (offline).

```python
from omero_annotate_ai import export_training_set

export = export_training_set(table_id, "my_training_data", conn)  # OMERO set
export = export_training_set(ts.bundle_dir, "my_training_data")  # local set
print(export["stats"])
```

Only finished units with a mask are used. Each unit goes to the split recorded in the
table's `category` column (training, validation or test), the split chosen when the
training set was created. The output folder must be outside the set's folder.

Channels and normalisation come from the set's config. For an OMERO set you can train on
another channel than the annotated one with `training_channels=[c]` (one channel) and
`label_channel=`; a local set exports the channels it was annotated with. An output
folder inside the set's own folder is refused. `apply_normalization=True` writes
normalised float32 images instead of the native bit depth.

### In Jupyter: the training-data widget

```python
from omero_annotate_ai import (
    create_omero_connection_widget,
    create_training_data_widget,
)

conn_widget = create_omero_connection_widget()
conn_widget.display()
conn = conn_widget.get_connection()

training_widget = create_training_data_widget(connection=conn)
training_widget.display()
```

Pick the container and the training set (local sets are listed too); its progress per
split is shown. Check the output folder (default `~/omero_annotate_ai/exports/<name>_<table id>`, or
the set's folder name for a local set),
the channels and the normalisation, then click **Export**. The result is in
`training_widget.result`. The marimo training notebook offers the same choices.

To train on the export, pick the trainer, epochs and model name in the widget's **Train**
box and run `training_widget.train()` in the next cell (see [Train a model](#train-a-model)).

!!! note "Things to know"
    - Images keep their original bit depth unless you apply the normalisation at
      export. `normalization.json` records how to scale them; micro-SAM training applies
      it when loading (see
      [normalisation settings](../configuration.md#normalization-intensity-scaling-of-model-input)).
    - A label that no longer matches its recorded pixel digest stops the export, so a
      mask changed after upload is never trained on unnoticed.
    - For an OMERO set a copy of the tracking table is saved alongside (`table_<id>.csv`).
    - `export_training_set` calls `prepare_training_data_from_table` (OMERO) or
      `reorganize_local_data_for_training` (local); use those directly for more options,
      such as copying, moving or symlinking local files.

## Train a model

One call trains on the exported folder. micro-SAM, Cellpose and BiaPy are supported; each
needs its extra (`omero-annotate-ai[microsam]`, `[cellpose]` or `[biapy]`). A CUDA GPU is
strongly recommended.

```python
from omero_annotate_ai import train

result = train(
    export["base_dir"], "microsam", "models/hela", model_name="hela_v1", epochs=50
)
result = train(export["base_dir"], "cellpose", "models/hela", epochs=100)
result = train(
    export["base_dir"],
    "biapy",
    "models/hela",
    epochs=100,
    config={"MODEL": {"ARCHITECTURE": "resunet"}},
)
print(result.model_path)
```

Here `export` is the result of `export_training_set` above. From a shell:

```bash
omero-annotate-ai train my_training_data --trainer cellpose --output models/hela --set epochs=100
```

Parameters, what each trainer does with the normalisation and where the model is saved
are in the [training data format](../training-data-format.md#trainers) page.

- micro-SAM: pass `n_iterations=` instead of `epochs` for a fixed number of steps, for
  example `n_iterations=10` for a quick check. The model is saved as
  `<output>/<model_name>_final.pt`; load it in the micro-SAM napari plugin as a custom
  model, or export to the [BioImage Model Zoo](https://bioimage.io) format with
  micro-SAM's own tools.
- BiaPy writes its log to `<output>/<model_name>/logs/`.

Notebooks: [export and train (Jupyter)](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/training/omero-training-export.ipynb), [micro-SAM training (marimo)](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-training-microsam.py)

### DL4MicEverywhere

[DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere) runs ZeroCostDL4Mic
notebooks (StarDist, Cellpose, U-Net, ...) in containers. Give it the export folder as
its data directory.

Notebook: [export and train](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/training/omero-training-export.ipynb)

## Tips

- Keep at least one validation image (`validate_n: 1` or more); micro-SAM training fails
  with an empty validation folder.
- A few well-annotated images go a long way when fine-tuning micro-SAM. Start with 5 to 10
  and add more where the model still fails.
