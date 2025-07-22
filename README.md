# OMERO AI Annotation tools

This package provides tools for image analysis and AI training with OMERO data. It is specifically designed to work with the [micro-SAM](https://github.com/computational-cell-analytics/micro-sam) annotator plugin in napari. It simplifies the process of connecting to an OMERO server, configuring annotation workflows, and running AI models on image data using jupyter noteobook widgets. 

## Installation
It is best to install this package in a conda environment (or alternatively even more convenient use [pixi](https://pixi.sh/latest/). You can create a new conda environment with the following command:

```bash
conda create -n omero_annotate_ai python=3.12
conda activate omero_annotate_ai
```

1. To use the OMERO annotator tool micro_sam is required as a dependencies. However micro-sam is only installable via conda. So first run:
```bash
conda install -c conda-forge micro-sam
```

2. This package also depends on `ezomero` and `omero-py` , which again depend on `zeroc-ice` , which is much easier to install using a wheel from Glencoe software, which can be found [here](https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases}, find the right wheel for your python version and OS. For example, for Python 3.12 on Linux x86_64, you would download `zeroc_ice-3.7.0-py3-none-any.whl`. Install it with:

```bash
pip install https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp312-cp312-manylinux_2_28_x86_64.whl
```

The package has not been released on PyPI yet, so you need to install it directly from the GitHub repository. You can do this with pip:

```bash
pip install git+https://github.com/Leiden-Cell-Observatory/omero_annotate_ai.git
```
## Usage

### OMERO Connection widget
![alt text](images/omero_connect_widget.png)   

This widget allows you to connect to an OMERO server by providing the server URL, username, and password. It provides a simple interface to authenticate and to store credentials locally.

### Annotation Pipeline widget
![alt text](images/omero_annotation_widget.png "Annotation Pipeline Widget") . This widget allows you to define the annotation pipeline and configure the parameters for the micro-SAM annotator.

### Annotation Config .YAML file
The Annotation pipeline widget uses a configuration file to define the annotation workflow. The configuration file is a YAML file that defines the parameters for the annotation pipeline, including the Dataset ID in OMERO, the number of images to process and the dimensions of the images. 

### Running the Annotation Pipeline
The annotation pipeline takes the configuration file and runs the annotation workflow. It connects to the OMERO server, retrieves the images, and runs the annotation pipeline on the images. It will run the micro-sam series annotator in OMERO and automatically stores the annotations back into OMERO as label images. The results are stored in the OMERO server as an annotation table. 

## Example notebooks

### omero-annotate-ai-pipeline.ipynb
This example notebook demonstrates how to use the OMERO Annotate AI tools to run the annotation pipeline. It includes steps to connect to the OMERO server, configure the annotation pipeline, and run the full annotation workflow.

### omero-annotate-ai-from-yaml.ipynb
This examples notebook demonstrates how to run the annotation pipeline using a yaml file directly and run the annotation pipeline.

### cleanup_annotation.ipynb
This example notebook demonstrates how to clean up annotations in an OMERO project or dataset. It includes steps to list all annotation tables in a project or dataset, delete specific annotation tables, and confirm the deletion. This is useful for managing annotations and ensuring that only relevant annotations are kept in the OMERO server.

## Contact
For questions, reach out to Maarten Paul (m.w.paul@lacdr.leidenuniv.nl). For issues or suggestions, please use the Issues section of the GitHub repository.

This repository is developed with the NL-BioImaging intrastructure, funded by NWO (National Roadmap for Large-Scale Research Facilities).
