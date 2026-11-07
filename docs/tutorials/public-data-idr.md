# Try it on public data

The [Image Data Resource (IDR)](https://idr.openmicroscopy.org) is a public OMERO server
with reference image sets from published studies. Anyone can log in, which makes it a
good place to try OMERO Annotate AI before using your own data.

Because you cannot write to the IDR, the run uses **read-only mode**: everything is
saved on your computer and nothing is uploaded.

## With the demo notebook

The [IDR demo notebook](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-idr-demo.py)
(marimo) takes you through connection, annotation and a short training run with
everything pre-filled.

```bash
pixi add marimo                        # once
pixi run marimo edit omero-idr-demo.py
```

## With the Jupyter widgets

Follow the [micro-SAM tutorial](microsam-annotation-pipeline.md) with these values:

| Setting | Value |
|---|---|
| Server | `ws://idr.openmicroscopy.org/omero-ws` |
| Username / password | `public` / `public` |
| Container | screen `1203` (idr0019, NF-κB signalling) or dataset `369` (idr0015, endosomes) |
| Training / validation images | 3 / 2 |
| Read-only mode | **on** |

The IDR server is connected through a websocket (`ws://…/omero-ws`), which works through
most institutional firewalls.

## What you get

After annotating, your working directory holds:

```text
annotation_input/     the images you annotated, as TIFF
annotation_output/    your masks, <unit id>_mask.tif
tracking_table.csv    which units were done, and their train/validation split
annotation_config_<name>.yaml
```

These can go straight into [training](training-data-prep.md#from-a-local-run-read-only-mode-or-cellpose).

## Finding other data

Browse [idr.openmicroscopy.org](https://idr.openmicroscopy.org). The ID of a screen,
plate, project or dataset is shown in the right-hand panel when you select it. Large
screens hold thousands of images; the pipeline only downloads the ones it picks.
