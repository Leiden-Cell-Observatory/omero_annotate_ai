# Annotate with Cellpose

Prefer drawing in the [Cellpose](https://www.cellpose.org) GUI, or want to correct
Cellpose predictions instead of using micro-SAM? This workflow exports the selected
images from OMERO, lets you annotate them in any tool, and then collects the masks and
uploads them to OMERO.

```mermaid
flowchart LR
    A[OMERO] -->|1. export| B[annotation_input/<br/>unit_id.tif]
    B -->|2. annotate in Cellpose| C[external_output/<br/>unit_id_mask.tif]
    C -->|3. collect| D[OMERO<br/>masks · ROIs · table]
```

Notebook: [annotation (marimo)](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-annotate-ai-annotation.py), with
*Annotation tool: Cellpose GUI*.

Cellpose itself is not installed with this package. Add it to your environment with
`pixi add cellpose` (or `pip install cellpose`), or use an existing Cellpose installation.

## 1. Connect, configure and export

Connect and fill in the *New training set* form exactly as in the
[micro-SAM tutorial](microsam-annotation-pipeline.md) (steps 1 and 2), without clicking
**Create**; the SAM model setting is ignored. Then create the set from the form (or from a
YAML file with `load_config_from_yaml`) and export the images:

```python
from omero_annotate_ai import TrainingSet

config = workflow_widget.get_config()
ts = TrainingSet.create(config, conn)
ts.prepare_external()
```

This creates the tracking table in OMERO and writes one TIFF per annotation unit to
`annotation_input/` in the training set's folder (`ts.bundle_dir`), named after the unit, for example
`4ba674d4_0_0.tif`. No napari window opens and no display is needed.

!!! note
    A unit counts as annotated only once its mask is collected (step 3), so the progress
    summary shows which masks are still missing.

## 2. Annotate

Open the images in `annotation_input/` in the Cellpose GUI (`cellpose` on the command
line). Run a model, correct the outlines, and **save each result as a label image** in
`external_output/` with the same name plus `_mask.tif`:

```text
annotation_input/4ba674d4_0_0.tif   →   external_output/4ba674d4_0_0_mask.tif
```

!!! warning "Use the `_mask.tif` naming"
    Cellpose's own `_seg.npy` and `_cp_masks.png` files are not picked up. Export or
    convert the masks to a label TIFF (0 = background, 1, 2, 3 … = objects) with the
    name above. Any tool that writes label TIFFs works: napari, QuPath, Fiji or a script.

You can annotate in several sittings. Check what is still missing with:

```python
from omero_annotate_ai.trainingset import summarize

print(summarize(ts.bundle_dir).progress)  # per split: (annotated, total)
```

## 3. Collect and upload

```python
n_done = ts.collect_masks()
```

Each mask is matched to its unit by name, moved to `annotation_output/` (the folder of
accepted masks) and uploaded as a mask attachment plus ROIs, and the tracking table is
updated; the collection is recorded as one session. Running it again only handles new
masks, and units someone else saved in the meantime are skipped. Masks waiting in
`external_output/` are never uploaded until you collect them. For local sets (`storage: local`) the masks stay local and
`tracking_table.csv` is updated instead. In a later sitting, reopen the set with
`TrainingSet.open(table_id, conn)` first.

## Next

[Prepare training data](training-data-prep.md) from the collected annotations, for
example to fine-tune a Cellpose model on your own images.
