# Annotate with micro-SAM

This is the main tutorial. You will connect to OMERO, choose images, annotate them in
napari with micro-SAM, and find the results back in OMERO. It takes about 20 minutes plus
the time you spend annotating.

!!! note "Before you start"
    - OMERO Annotate AI is [installed](../installation.md) and Jupyter is running.
    - You can log in to an OMERO server and know the ID of a dataset, project, plate or
      screen with images (visible in OMERO.web next to the name).
    - You are on a machine with a screen: napari opens a separate window.

The steps below follow the notebook
[`omero-annotate-ai-annotation-widget.ipynb`](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/annotation/omero-annotate-ai-annotation-widget.ipynb).
Download it, or copy the code blocks into a new notebook.

## 1. Connect to OMERO

```python
from omero_annotate_ai import (
    create_omero_connection_widget,
    create_workflow_widget,
    create_pipeline,
)

conn_widget = create_omero_connection_widget()
conn_widget.display()
```

Fill in the server address, your username and password, and click **Connect**.

![OMERO connection widget](../images/widget_connect.png)

The widget remembers the server and username for next time (in
`~/.omero-annotate-ai/connections.json`), but never your password unless you tick the
option to store it in your system keychain.

Then, in the next cell:

```python
conn = conn_widget.get_connection()
```

## 2. Configure the run

```python
workflow_widget = create_workflow_widget(connection=conn)
workflow_widget.display()
```

The widget walks you through a few tabs. Use **Next** to move on.

**Working directory.** Where local files go. Pick a folder with some free space; every
run gets its own subfolder.

**Container.** Choose the type (project, dataset, plate, screen) and the container
itself from the list.

![Choosing an OMERO container](../images/widget_container.png)

**Tables.** If annotations already exist for this container, they are listed here. For a
first run, choose **Name New Table** and give the run a short, descriptive name such as
`hela_nuclei_v1`. This name is used for the OMERO table and the files.

**Configuration.** The scientific choices:

| Setting | What to choose |
|---|---|
| Training / validation / test images | How many images to annotate for each set. Start small, e.g. 5 training and 2 validation. |
| Channel | The channel you will draw on, e.g. the DAPI channel for nuclei. Channel numbers start at 0. |
| Timepoints, Z-slices | Which planes to annotate. For 2D images leave them at 0. |
| 3D processing | Annotate whole z-stacks as volumes instead of single planes. |
| Use patches | Annotate crops instead of whole images. Recommended for large images or a CPU-only computer. |
| SAM model | `vit_b_lm` is a good default for light microscopy; `vit_b_em_organelles` for electron microscopy. |
| Batch size | Units per napari session; progress is saved after each batch. 0 = all at once. |
| Read-only mode | Save results only on your computer and write nothing to OMERO. |

**Save.** Shows the YAML configuration that will be used. Click **Save Configuration**
to keep a copy.

![Reviewing the generated YAML](../images/widget_config.png)

Then collect the configuration:

```python
config = workflow_widget.get_config()
```

!!! tip "Not sure what a setting means?"
    [How it works](../concepts.md) explains annotation units, patches and the two modes.
    The [configuration reference](../configuration.md) lists every option.

## 3. Annotate in napari

```python
pipeline = create_pipeline(config, conn)
table_id, config = pipeline.run_full_micro_sam_workflow()
```

The pipeline now:

1. picks images at random and assigns them to training and validation,
2. creates the tracking table in OMERO (not in read-only mode),
3. downloads the first batch and opens it in napari with the micro-SAM annotator.

The first time, micro-SAM downloads its model (a few hundred MB). For each image it then
computes an *embedding*; this is what makes the clicks fast afterwards, and takes a few
seconds on a GPU or longer on a CPU.

### In napari

The micro-SAM image series annotator shows one image at a time.

1. With the `point_prompts` layer selected, click on an object to place a **positive
   point**, or use the `prompts` layer to draw a box around it. Press ++t++ to switch to
   **negative** points, which tell SAM what is *not* part of the object.
2. Press ++s++ to segment the object from your prompts. Correct it by adding more points
   and pressing ++s++ again.
3. Press ++c++ to **commit** the object. It is added to the label layer and gets its own
   colour. (++shift+c++ clears the current prompts and segmentation.)
4. Repeat for every object. To speed things up, use **Automatic Segmentation** in the side
   panel, then fix mistakes with napari's label tools.
5. Click **Next Image** (or press ++n++) to save the mask and go to the next image.
   In 3D mode, ++shift+s++ extends the segmentation through the volume.

When the last image of the batch is done, close the napari window. The masks are then
uploaded and the next batch, if any, opens.

See the [micro-SAM documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html)
for all annotator features and shortcuts.

!!! warning "The notebook cell waits for napari"
    The cell keeps running until you close the last napari window. That is normal.

## 4. Check the results

In the notebook:

```python
print(config.get_progress_summary())
```

In OMERO.web (upload mode):

- Each annotated image now has **ROIs** (open it in OMERO.iviewer) and a **mask file**
  in its attachments.
- The container has a **table** named after your run, listing every unit and whether it
  is done, plus the **configuration file** as an attachment.

In read-only mode, the images and masks are in `annotation_input/` and
`annotation_output/` in your working directory, with a `tracking_table.csv`.

## 5. What next?

- Stopped half-way? [Resume the run](resume-and-repeat.md).
- Want to repeat it on another dataset? Save the config and
  [run it from YAML](resume-and-repeat.md#repeat-a-run-from-a-yaml-file).
- Enough annotations? [Prepare training data](training-data-prep.md) and fine-tune a
  model on your own images.

## Variations

??? example "Annotate one channel, train on another"
    Draw nuclei on a fluorescent channel but train a model that predicts them from
    brightfield: set `label_channel` to the fluorescent channel and `training_channels`
    to the brightfield channel. The brightfield images are saved in `model_input/`.

    ```yaml
    spatial_coverage:
      channels: [0, 1]
      label_channel: 0
      training_channels: [1]
    ```

??? example "Show extra channels for context"
    `context_channels` are shown in napari next to the channel you annotate (e.g. a
    membrane stain while you draw cytoplasm) but never used for training.

    ```yaml
    spatial_coverage:
      channels: [0]
      context_channels: [1, 2]
      context_channel_names: [membrane, actin]
    ```

??? example "Only some wells of a plate"
    Keep only wells whose key-value pairs match:

    ```yaml
    omero:
      container_type: plate
      container_id: 456
      well_filters:
        treatment: [DMSO, control]
      well_filter_mode: include    # or exclude
    ```

??? example "3D volumes"
    ```yaml
    spatial_coverage:
      three_d: true
      z_slices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # the range of the volume
    ```
    Each timepoint becomes one volumetric unit and opens in micro-SAM's 3D annotator.

??? example "Without the widgets"
    ```python
    import getpass
    from omero_annotate_ai import load_config_from_yaml, create_pipeline
    from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection

    conn = SimpleOMEROConnection().connect(
        host="omero.example.org", username="me", password=getpass.getpass()
    )
    config = load_config_from_yaml("annotation_config.yaml")
    pipeline = create_pipeline(config, conn)
    table_id, config = pipeline.run_full_micro_sam_workflow()
    ```
