# Resume or repeat a run

Every run is described by its YAML configuration. That file is how you continue an
interrupted run, repeat a run on new data, or let a colleague reproduce your annotation
set.

## Resume an interrupted run

A training set tracked in OMERO is identified by its tracking table's ID, which never
changes. To continue it, point the config at that table:

```python
from omero_annotate_ai import load_config_from_yaml, create_pipeline

config = load_config_from_yaml(
    "~/omero_annotate_ai/<server>/<table id>_hela_nuclei_v1/config.yaml"
)
config.workflow.resume_from_table = True
config.omero.table_id = 12345  # the table ID, shown in OMERO.web on the container
pipeline = create_pipeline(config, conn)
table_id, config = pipeline.run_full_micro_sam_workflow()  # or run_cp_workflow()
```

The units are read back from the table and only the unfinished ones are opened, with
the same images and patches as before. In the widget, choose *Continue Existing* on the
table instead.

### What resuming repairs

Each mask is saved in a fixed order: moved into the set's folder, recorded in the local
`tracking_table.csv`, then uploaded (label file, then ROI) with a tag naming the table
and unit, and finally written to the OMERO table. If a run stops halfway - napari
closed, kernel killed, network gone - resuming compares the OMERO table, the local copy
and the mask files for every unit and finishes the job:

- uploads the table never recorded are found by their tag and adopted, never uploaded
  a second time;
- masks saved locally but not uploaded are uploaded;
- a unit whose upload failed is tried again (the error is kept in the local
  `tracking_table.csv`, column `error`).

The local folder can even be lost: the uploads carry their tag, so resuming still
finds them.

### Pixel checks

Every unit records a SHA-256 digest of its pixels: `input_digest` for the image you
annotated and `mask_digest` for your mask. Resuming warns when a local file no longer
matches (for example a mask edited after upload), and
`prepare_training_data_from_table` stops with an `IntegrityError` if a label file on
OMERO changed after it was annotated. Nothing is overwritten silently.

If a mask in your folder and its upload on OMERO differ, the unit is marked
`needs review` and left out of new sessions. Decide which one is right:

- delete the local mask (`annotation_output/<unit id>_mask.tif`) to accept the one on
  OMERO, or
- delete the label file on OMERO to keep your local mask,

then resume the set again.

## Repeat a run from a YAML file

To annotate a different dataset with the same settings, or to run without widgets (for
example from a script), start from a YAML file. Take one you saved from the widget, or
generate a commented template:

```python
from omero_annotate_ai.core.annotation_config import get_config_template

print(get_config_template())
```

Remove the `annotations:` section if you copy a saved progress file, change `name` and
the `omero` container, then:

```python
config = load_config_from_yaml("my_run.yaml")
config.config_file_path = (
    None  # keep my_run.yaml unchanged; progress goes to the working directory
)
pipeline = create_pipeline(config, conn)
table_id, config = pipeline.run_full_micro_sam_workflow()
```

!!! tip "Check what was read"
    Keys that the package does not know are silently ignored, so a typo means a setting
    is not applied. After loading, print what matters:
    ```python
    print(config.spatial_coverage)
    print(config.training)
    ```

Notebooks: [Jupyter](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/annotation/omero-annotate-ai-from-yaml.ipynb)
· [marimo](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-annotate-ai-from-yaml.py)

## Check progress

```python
print(config.get_progress_summary())
done = config.get_processed()
todo = config.get_unprocessed()
df = config.to_dataframe()  # one row per unit, as a pandas table
```
