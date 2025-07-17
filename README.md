# OMERO Annotate AI tools

This package provides a streamlined interface for annotating images in OMERO using AI models, specifically designed to work with the micro-SAM annotator plugin in napari. It simplifies the process of connecting to an OMERO server, configuring annotation workflows, and running AI models on image data using jupyter noteobook widgets. 

## Installation
For running the OMERO annotator tool micro_sam is required as a dependencies. However micro-sam is only installable via conda.   
So run `conda install -c conda-forge micro-sam` separately.

This package also depends on `ezomero` and `omero-py` , which again depend on `zeroc-ice` , which is easier to install using a wheel from Glencoe software, which can be found [here](https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases}, find the right wheel for your python version and OS.

## Usage



## Contact
For questions, reach out to Maarten Paul (m.w.paul@lacdr.leidenuniv.nl). For issues or suggestions, please use the Issues section of the GitHub repository.

This repository is developed with the NL-BioImaging intrastructure, funded by NWO (National Roadmap for Large-Scale Research Facilities).
