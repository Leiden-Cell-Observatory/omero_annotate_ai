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
| Annotate with Cellpose | [cellpose][gh-cp] | [cellpose][mo-cp] | [guide](tutorials/cellpose-integration.md) |
| Run from a YAML file | [from YAML][gh-yaml] | [from YAML][mo-yaml] | [guide](tutorials/resume-and-repeat.md) |
| Try on public IDR data | | [IDR demo][mo-idr], [OME-Zarr][mo-zarr] | [guide](tutorials/public-data-idr.md) |
| Train micro-SAM | [micro-SAM][gh-tr] | [micro-SAM][mo-tr] | [guide](tutorials/training-data-prep.md) |
| Prepare data for BiaPy | [BiaPy][gh-biapy] | | [guide](tutorials/training-data-prep.md#biapy) |
| Prepare data for DL4MicEverywhere | [DL4Mic][gh-dl4mic] | | [guide](tutorials/training-data-prep.md#dl4miceverywhere) |
| Apply a trained micro-SAM model | [inference][gh-inf] | | |
| Export to the BioImage Archive | [BIA export][gh-bia] | | [guide](tutorials/bioimage-archive-export.md) |
| Delete a training set from OMERO | [cleanup][gh-clean] | | |

[gh-ann]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/annotation/omero-annotate-ai-annotation-widget.ipynb
[gh-cp]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/annotation/omero-annotate-ai-cellpose.ipynb
[gh-yaml]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/annotation/omero-annotate-ai-from-yaml.ipynb
[gh-tr]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/training/omero-training-microsam.ipynb
[gh-biapy]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/training/omero-training_biapy.ipynb
[gh-dl4mic]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/training/omero-training_DL4mic.ipynb
[gh-inf]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/inference/omero-test_model-microsam.ipynb
[gh-bia]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/export_annotations/omero-table-to-bioimage-archive.ipynb
[gh-clean]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/other/cleanup_annotations.ipynb
[mo-ann]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-annotate-ai-annotation.py
[mo-cp]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-annotate-ai-cellpose.py
[mo-yaml]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-annotate-ai-from-yaml.py
[mo-idr]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-idr-demo.py
[mo-tr]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/omero-training-microsam.py
[mo-zarr]: https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/zarr-annotate-ai-from-yaml.py
