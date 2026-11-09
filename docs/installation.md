# Installation

OMERO Annotate AI is a Python package. It needs Python 3.11 or newer and works on Linux,
macOS and Windows. micro-SAM and napari are installed alongside it.

!!! tip "Never installed Python software before?"
    Use **pixi**. It creates a self-contained folder with Python and every dependency, and
    does not touch the rest of your system. Install it once by following the
    [pixi instructions](https://pixi.sh/latest/#installation).

## Install with pixi (recommended)

Open a terminal and run:

```bash
pixi init my-annotation-project
cd my-annotation-project
pixi add python=3.12 micro_sam jupyterlab
pixi add --pypi "omero-annotate-ai[notebooks]"
```

This creates the folder `my-annotation-project` with everything inside it. To start
Jupyter in that environment:

```bash
pixi run jupyter lab
```

The `notebooks` extra adds the Jupyter widgets and [marimo](https://marimo.io). Start a marimo notebook with `pixi run marimo edit <notebook>.py`.

## Install with conda and pip

If you already use conda (or mamba):

```bash
conda create -n omero-annotate-ai -c conda-forge python=3.12 micro_sam jupyterlab
conda activate omero-annotate-ai
pip install "omero-annotate-ai[notebooks]"
```

The OMERO libraries (`omero-py`, `zeroc-ice`) are pulled in automatically.

## Install with pip only

micro-SAM is also [on PyPI](https://pypi.org/project/micro-sam/); the `microsam` extra installs it
together with napari:

```bash
python -m venv omero-annotate-ai
source omero-annotate-ai/bin/activate      # Windows: omero-annotate-ai\Scripts\activate
pip install "omero-annotate-ai[microsam,notebooks]" jupyterlab
```

With plain pip, `zeroc-ice` (needed by `omero-py`) may have no ready-made build for your
platform; see [Troubleshooting](troubleshooting.md#installation). pixi and conda avoid
this. For a CUDA GPU, install the matching PyTorch build first, following the
[PyTorch instructions](https://pytorch.org/get-started/locally/).

## Check the installation

In a notebook cell or Python prompt:

```python
from importlib.metadata import version

print(version("omero-annotate-ai"))
```

If this prints a version number, you are ready. Continue with
[Try it on public data](tutorials/public-data-idr.md) or the
[micro-SAM tutorial](tutorials/microsam-annotation-pipeline.md).

## Optional extras

| You want to | Install |
|---|---|
| Export annotations to the BioImage Archive | `pip install "git+https://github.com/BioImage-Archive/bia-mifa-models.git@v1.1.5"` |
| Record ISCC content codes (how much an image or mask changed) | `pip install "omero-annotate-ai[provenance]"` (included in the pixi environments) |
| Train micro-SAM models | nothing extra; a CUDA GPU is strongly recommended |
| Train with BiaPy or DL4MicEverywhere | install those tools separately; this package only prepares the data |

## GPU or CPU?

micro-SAM runs on a CPU, but computing the image embeddings that make interactive
segmentation fast takes tens of seconds per image without a GPU. On a CPU, keep the
default `vit_b_lm` model and use patches (smaller crops) for large images. Training a
model really needs a GPU.

Check whether a GPU is visible:

```python
import torch

print(torch.cuda.is_available())
```

## Working on a remote machine

napari opens a desktop window, so micro-SAM annotation needs a graphical session. On a
remote server or cloud workspace, use its remote desktop (for example VNC or the desktop
tile on SURF Research Cloud), not a plain SSH or JupyterHub tab. Preparing images for
Cellpose and preparing training data work without a display.

## For developers

```bash
git clone https://github.com/Leiden-Cell-Observatory/omero_annotate_ai.git
cd omero_annotate_ai
pixi install
pixi run -e dev pytest tests/      # tests
pixi run -e docs docs-serve        # this documentation, live at http://localhost:8000
```

Problems? See [Troubleshooting](troubleshooting.md).
