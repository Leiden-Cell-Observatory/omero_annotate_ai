# Notebooks

Example notebooks for the omero-annotate-ai workflows. The Jupyter widget notebook is the
easiest start; the marimo notebooks cover everything else. Committed notebooks carry no
outputs (`pixi run -e dev install-hooks` sets up the hook that strips them).

## marimo (`marimo/`)

Reactive notebooks stored as plain `.py` files. Run with `marimo edit notebooks/marimo/<notebook>.py`.

| Notebook | What it does |
|---|---|
| `omero-annotate-ai-annotation.py` | Create or open a training set on an OMERO container and annotate it with micro-SAM (napari) or the Cellpose GUI; session notes |
| `annotate-from-yaml.py` | Create a training set from a YAML config (OMERO or OME-Zarr, `source: zarr` needs no OMERO) or open a set's folder, then annotate |
| `omero-training-microsam.py` | Export a training set from OMERO (train/validation/test) and fine-tune micro-SAM |
| `omero-idr-demo.py` | Annotate and train on public IDR data as a local training set |
| `configs/zarr-s-biad855-example.yaml` | Example OME-Zarr config (a BioImage Archive plate) for `annotate-from-yaml.py` |

## Jupyter (`jupyter/`)

Notebooks with ipywidgets forms. Run with `jupyter lab`.

| Notebook | What it does |
|---|---|
| `annotation/omero-annotate-ai-annotation-widget.ipynb` | The widget workflow: connect, create or open a training set, annotate with micro-SAM, add a session note |
| `training/omero-training-export.ipynb` | Export a training set for BiaPy or DL4MicEverywhere |
| `inference/omero-test_model-microsam.ipynb` | Run a fine-tuned micro-SAM model on OMERO images |
| `export_annotations/omero-table-to-bioimage-archive.ipynb` | Build a BioImage Archive submission from a training set on OMERO |
| `export_annotations/prepare_bioimage_archive.ipynb` | Build a BioImage Archive submission from a local training set (runs offline on demo data) |
| `other/cleanup_annotations.ipynb` | Delete a training set from OMERO (dry run first) |

## Prerequisites

- An OMERO server, or OME-Zarr images for `annotate-from-yaml.py` with `source: zarr`
- `pixi install` in this repository (or `pip install "omero-annotate-ai[microsam,notebooks]"`)
- A GPU is recommended for micro-SAM
