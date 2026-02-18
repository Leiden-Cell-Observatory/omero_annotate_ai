# Notebooks

Example notebooks demonstrating the omero-annotate-ai package workflows.

## Marimo Notebooks (`marimo/`)

[Marimo](https://marimo.io) notebooks are reactive Python notebooks that are easier to maintain and version control. They are stored as plain `.py` files.

**To run:** `marimo edit notebooks/marimo/<notebook>.py`

| Notebook | Description |
|----------|-------------|
| `omero-annotate-ai-annotation.py` | Micro-SAM annotation workflow — connect to OMERO, configure parameters, run interactive annotation via Napari |
| `omero-annotate-ai-cellpose.py` | CellPose annotation workflow — export images for CellPose annotation, then collect and upload masks |
| `omero-annotate-ai-from-yaml.py` | Load an existing YAML configuration and run the pipeline without configuration widgets |
| `omero-training-microsam.py` | Micro-SAM training — select an annotation table, download training data, fine-tune a micro-SAM model |
| `omero-idr-demo.py` | Full demo using public IDR data — no local OMERO server needed. Annotation + training in read-only mode |

## Jupyter Notebooks (`jupyter/`)

Classic Jupyter notebooks using ipywidgets for the UI.

**To run:** `jupyter notebook notebooks/jupyter/<path>/<notebook>.ipynb`

### Annotation (`jupyter/annotation/`)

| Notebook | Description |
|----------|-------------|
| `omero-annotate-ai-annotation-widget.ipynb` | Widget-based micro-SAM annotation workflow |
| `omero-annotate-ai-cellpose.ipynb` | CellPose annotation workflow — export, annotate, upload |
| `omero-annotate-ai-from-yaml.ipynb` | Run annotation from a saved YAML configuration |

### Training (`jupyter/training/`)

| Notebook | Description |
|----------|-------------|
| `omero-training-microsam.ipynb` | Full micro-SAM training pipeline with BioImage.IO model export |
| `omero-training_biapy.ipynb` | Prepare OMERO annotation data for BiaPy training |
| `omero-training_DL4mic.ipynb` | Prepare OMERO annotation data for DL4MicEverywhere training |

### Inference (`jupyter/inference/`)

| Notebook | Description |
|----------|-------------|
| `omero-test_model-microsam.ipynb` | Run a fine-tuned micro-SAM model on OMERO images for inference |

### Utilities (`jupyter/other/`)

| Notebook | Description |
|----------|-------------|
| `cleanup_annotation.ipynb` | List, delete, and bulk-clean annotation tables from OMERO containers |

## Developer Notebooks (`dev/`)

Notebooks for package development and debugging. Not intended for end users.

| Notebook | Description |
|----------|-------------|
| `omero-annotate-ai-annotation_dev.ipynb` | Development version with autoreload and widget reload helpers |

## Prerequisites

- A running OMERO server (or use IDR for the demo notebook)
- The `omero-annotate-ai` package installed: `pip install omero-annotate-ai`
- For micro-SAM workflows: a GPU is recommended
- For Jupyter notebooks: `jupyter notebook` or `jupyter lab`
- For Marimo notebooks: `pip install marimo` then `marimo edit <notebook>.py`
