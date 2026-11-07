# Resume or repeat a run

Every run is described by its YAML configuration. That file is how you continue an
interrupted run, repeat a run on new data, or let a colleague reproduce your annotation
set.

## Resume an interrupted run

After every batch the pipeline saves its progress to

```text
<working directory>/annotation_config_<name>.yaml
```

This copy contains the config **and** the list of annotation units with a `processed`
flag for each. Load it and start the same workflow again; only the unfinished units are
opened, with the same images and patches as before:

```python
from omero_annotate_ai import load_config_from_yaml, create_pipeline

config = load_config_from_yaml("annotations/annotation_config_hela_nuclei_v1.yaml")
pipeline = create_pipeline(config, conn)
table_id, config = pipeline.run_full_micro_sam_workflow()  # or run_cp_workflow()
```

!!! warning "Upload mode: clear micro-SAM's working folder first"
    Before resuming in upload mode, empty the `annotations/` subfolder inside your
    working directory. micro-SAM skips images whose `seg_*.tif` file already exists, so
    leftovers from an earlier session can end up attached to the wrong images.

!!! note "Use the saved file, not the 'resume from table' switch"
    The *Resume from existing table* option in the widget and `resume_from_table: true`
    in the YAML do not reliably find the existing table in this version; you may get a
    fresh random selection and a second table with the same name. Reloading the saved
    YAML is the dependable way to continue.

### Resume on another computer

The YAML is also attached to the OMERO container, one per tracking table. To continue
elsewhere, fetch it by the table's ID and sync the progress from the tracking table:

```python
from omero_annotate_ai.omero.omero_functions import (
    find_annotation_config_for_table,
    sync_omero_table_to_config,
)

config = find_annotation_config_for_table(conn, table_id)
config.config_file_path = None  # save progress in the working directory instead
pipeline = create_pipeline(config, conn)
sync_omero_table_to_config(conn, table_id, pipeline.config)
pipeline.set_table_id(table_id)
pipeline.run_full_micro_sam_workflow()
```

The table ID is shown in OMERO.web on the container's attachments.

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
