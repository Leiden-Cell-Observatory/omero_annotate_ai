# Troubleshooting

## Installation

**`pip install micro-sam` fails.**
The PyPI package needs Python 3.11 or newer; check with `python --version`. If it still
fails, install micro-SAM from conda-forge instead (`pixi add micro_sam` or
`conda install -c conda-forge micro_sam`), then `pip install omero-annotate-ai`.

**`zeroc-ice` fails to build with pip.**
Use pixi or conda, which provide ready-made builds. With plain pip, install a wheel for
your platform from Glencoe Software first:
[Linux](https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases) ·
[Windows](https://github.com/glencoesoftware/zeroc-ice-py-win-x86_64/releases) ·
[macOS](https://github.com/glencoesoftware/zeroc-ice-py-macos-universal2/releases).

**The widgets do not appear in Jupyter.**
Use JupyterLab 4 or Notebook 7 with `ipywidgets` 8 installed in the same environment
(`pixi add ipywidgets`). Restart the kernel after installing. The Jupyter widgets do not
work in marimo; use the marimo notebooks there.

## Connecting

**The connection widget says it cannot connect.**
Check the server address in OMERO.insight or with your OMERO admin. Websocket addresses
(`ws://…/omero-ws` or `wss://…`) work through most firewalls; the classic port 4064 is
often blocked outside the institute network.

**My password is not remembered.**
By design. Tick the option to store it in your system keychain if you want this; on a
Linux server without a desktop there may be no keychain available.

## Annotating

**No napari window opens, or the kernel crashes when it should.**
napari needs a graphical desktop. On a remote machine, run Jupyter from inside a remote
desktop session (VNC or similar), not via SSH or JupyterHub only.

**The notebook cell keeps running.**
It waits until you close the last napari window of the batch. That is normal.

**micro-SAM is very slow.**
Each image needs an embedding before you can click. Without a GPU this takes tens of
seconds per image. Use `vit_b_lm`, switch on patches, and keep batches small.

**An upload failed.**
The run carries on with the next unit; the failed one stays unannotated and its error is
in the `error` column of the set's `tracking_table.csv`. Resume the set to try again.

**Resuming says a unit "needs review".**
Its mask in your folder and its label file on OMERO differ. Delete the one that is wrong
and resume again; see
[Pixel checks](tutorials/resume-and-repeat.md#pixel-checks).

**"Not enough images" when starting.**
`train_n + validate_n + test_n` is larger than the number of images in the container
(after well filtering). Lower the counts or use `segment_all: true`.

## Config files

**Loading my YAML fails with "unknown key" or `RemovedConfigKey`.**
Unknown keys are refused, so a typo never goes unnoticed; check the name against the
[configuration reference](configuration.md). Keys from 0.4 that moved name their
replacement in the error, see [migrating from 0.4](whats-new-0.5.md#migrating-from-04).

**Does running a training set change my YAML file?**
No. A loaded YAML is a template; the set keeps its own `config.yaml` (with sessions and
progress) in its folder under `~/omero_annotate_ai/`.

## Still stuck?

Open an [issue on GitHub](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/issues)
with the error message, your config (without passwords) and the package version:

```python
from importlib.metadata import version

print(version("omero-annotate-ai"))
```
