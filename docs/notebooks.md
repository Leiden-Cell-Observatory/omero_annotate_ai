# Example notebooks

Ready-made notebooks for every workflow are in the
[`notebooks/` folder](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/tree/main/notebooks)
on GitHub. Download the one you need, or clone the repository.

- **Jupyter** notebooks use ipywidgets for the forms. Open with `jupyter lab`.
- **marimo** notebooks are reactive and stored as plain Python files. Open with
  `marimo edit <notebook>.py`.

| Task | Jupyter | marimo | Guide |
|---|---|---|---|
| Annotate with micro-SAM | [widget workflow][gh-ann] | [annotation][mo-ann] | [guide](tutorials/microsam-annotation-pipeline.md) |
| Annotate with Cellpose | | [annotation][mo-ann] (tool: Cellpose GUI) | [guide](tutorials/cellpose-integration.md) |
| Run from a YAML file (OMERO or OME-Zarr) | | [from YAML][mo-yaml] | [guide](tutorials/resume-and-repeat.md) |
| Try on public data | | [IDR demo][mo-idr], [OME-Zarr from YAML][mo-yaml] | [guide](tutorials/public-data-idr.md) |
| Train micro-SAM | | [micro-SAM][mo-tr] | [guide](tutorials/training-data-prep.md) |
| Export for BiaPy or DL4MicEverywhere | [export][gh-export] | | [guide](tutorials/training-data-prep.md#biapy) |
| Apply a trained micro-SAM model | [inference][gh-inf] | | |
| Export to the BioImage Archive | [from OMERO][gh-bia], [from a local set][gh-bia-local] | | [guide](tutorials/bioimage-archive-export.md) |
| Delete a training set from OMERO | [cleanup][gh-clean] | | |

[gh-ann]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/annotation/omero-annotate-ai-annotation-widget.ipynb
[gh-export]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/training/omero-training-export.ipynb
[gh-inf]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/inference/omero-test_model-microsam.ipynb
[gh-bia]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/export_annotations/omero-table-to-bioimage-archive.ipynb
[gh-bia-local]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/export_annotations/prepare_bioimage_archive.ipynb
[gh-clean]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/other/cleanup_annotations.ipynb
[mo-ann]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-annotate-ai-annotation.py
[mo-yaml]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/annotate-from-yaml.py
[mo-idr]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-idr-demo.py
[mo-tr]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-training-microsam.py
