# OMERO Annotate.AI

[![PyPI version](https://img.shields.io/pypi/v/omero-annotate-ai)](https://pypi.org/project/omero-annotate-ai/)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/omero-annotate-ai/README.html)
[![Python versions](https://img.shields.io/pypi/pyversions/omero-annotate-ai)](https://pypi.org/project/omero-annotate-ai/)
[![CI/CD](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/actions)
[![Documentation](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/actions/workflows/docs.yml/badge.svg)](https://leiden-cell-observatory.github.io/omero_annotate_ai/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![License](https://img.shields.io/github/license/Leiden-Cell-Observatory/omero_annotate_ai)](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/LICENSE)

**Reproducible AI-assisted annotation of OMERO data, from image selection to training-ready datasets.**

Pick images from an OMERO project, dataset, plate or screen. Annotate them in napari with
[micro-SAM](https://github.com/computational-cell-analytics/micro-sam) or prepare them for
[Cellpose](https://www.cellpose.org). The masks, ROIs and a tracking table go back to OMERO.
Every run is described by one YAML file, so you can resume it, repeat it or share it.

![micro-SAM segmentation of a Cell Painting field in napari](docs/images/napari_microsam.jpg)
<sub>micro-SAM (`vit_b_lm`) instance segmentation of nuclei in a Cell Painting field from the BioImage Archive (S-BIAD855), shown in napari.</sub>

## How it works

```mermaid
flowchart LR
    A[OMERO<br/>project · dataset · plate · screen] --> B[Select images<br/>train / val / test split]
    B --> C[Annotate in napari<br/>micro-SAM or Cellpose]
    C --> D[Back to OMERO<br/>masks · ROIs · tracking table]
    D --> E[Training data<br/>micro-SAM · BiaPy · DL4MicEverywhere]
    D --> F[BioImage Archive<br/>MIFA export]
```

- **Jupyter widgets** for connecting to OMERO and configuring a run step by step, no code needed.
- **Resumable:** the tracking table records which images and patches are done, so an interrupted run continues where it stopped.
- **Flexible sampling:** timepoints, z-slices, 3D volumes, random patches, and filtering plate wells on their key-value pairs.
- **Training-ready output:** one folder layout for micro-SAM, BiaPy and DL4MicEverywhere, with images and labels paired by name.
- **FAIR by default:** the config carries study and method metadata and exports to the BioImage Archive (MIFA).
- **Read-only mode** for public servers such as the [IDR](https://idr.openmicroscopy.org).

## Jupyter widgets

<table>
  <tr>
    <td width="45%"><img src="docs/images/widget_connect.png" alt="OMERO connection widget"></td>
    <td width="55%"><img src="docs/images/widget_container.png" alt="Choosing an OMERO container in the workflow widget"><br><br>
    <img src="docs/images/widget_config.png" alt="Reviewing the generated YAML configuration"></td>
  </tr>
  <tr>
    <td><sub>Connect to any OMERO server, here the public IDR.</sub></td>
    <td><sub>Choose a container, set the parameters, and review the YAML config the run will use.</sub></td>
  </tr>
</table>

## Installation

```bash
# Recommended: pixi
pixi init myproject && cd myproject
pixi add micro_sam
pixi add --pypi omero-annotate-ai

# Or: conda + pip
conda install -c conda-forge micro_sam
pip install omero-annotate-ai
```

See the [installation guide](https://leiden-cell-observatory.github.io/omero_annotate_ai/installation/) for details and troubleshooting.

## Quick start

**With widgets** (in Jupyter):

```python
from omero_annotate_ai import create_omero_connection_widget, create_workflow_widget, create_pipeline

conn_widget = create_omero_connection_widget()
conn_widget.display()
conn = conn_widget.get_connection()

workflow_widget = create_workflow_widget(connection=conn)
workflow_widget.display()

config = workflow_widget.get_config()
pipeline = create_pipeline(config, conn)
table_id, config = pipeline.run_full_micro_sam_workflow()
```

**From a YAML config** (scripts, batch runs, or repeating a run):

```python
from omero_annotate_ai import load_config, create_pipeline

config = load_config("annotation_config.yaml")
pipeline = create_pipeline(config, conn)
table_id, config = pipeline.run_full_micro_sam_workflow()
```

The [configuration guide](https://leiden-cell-observatory.github.io/omero_annotate_ai/configuration/) lists every option.

## Example notebooks

| | Jupyter | marimo |
|---|---|---|
| Annotate with micro-SAM | [widget workflow](notebooks/jupyter/annotation/omero-annotate-ai-annotation-widget.ipynb) | [annotation](notebooks/marimo/omero-annotate-ai-annotation.py) |
| Prepare for Cellpose | [cellpose](notebooks/jupyter/annotation/omero-annotate-ai-cellpose.ipynb) | [cellpose](notebooks/marimo/omero-annotate-ai-cellpose.py) |
| Run from YAML | [from YAML](notebooks/jupyter/annotation/omero-annotate-ai-from-yaml.ipynb) | [from YAML](notebooks/marimo/omero-annotate-ai-from-yaml.py) |
| Train | [micro-SAM](notebooks/jupyter/training/omero-training-microsam.ipynb) · [BiaPy](notebooks/jupyter/training/omero-training_biapy.ipynb) · [DL4MicEverywhere](notebooks/jupyter/training/omero-training_DL4mic.ipynb) | [micro-SAM](notebooks/marimo/omero-training-microsam.py) |
| Export to the BioImage Archive | [BIA export](notebooks/jupyter/export_annotations/omero-table-to-bioimage-archive.ipynb) | |
| Try it on public IDR data | | [IDR demo](notebooks/marimo/omero-idr-demo.py) |

## Documentation

Full documentation: **https://leiden-cell-observatory.github.io/omero_annotate_ai/**
([micro-SAM tutorial](https://leiden-cell-observatory.github.io/omero_annotate_ai/tutorials/microsam-annotation-pipeline/),
[API reference](https://leiden-cell-observatory.github.io/omero_annotate_ai/api/)).

## Contributing

```bash
git clone https://github.com/Leiden-Cell-Observatory/omero_annotate_ai.git
cd omero_annotate_ai
pixi install
pixi run -e dev pytest tests/
```

Issues and pull requests are welcome on [GitHub](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/issues).

## Contact and acknowledgements

**Maarten Paul**, m.w.paul@lacdr.leidenuniv.nl

Developed at the [Leiden Cell Observatory](https://github.com/Leiden-Cell-Observatory) within the
[NL-BioImaging](https://github.com/NL-BioImaging) infrastructure, funded by NWO.
