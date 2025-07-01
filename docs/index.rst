.. omero-annotate-ai documentation master file

Welcome to omero-annotate-ai's documentation!
==============================================

omero-annotate-ai is a Python package that provides micro-SAM annotation workflows for OMERO (Open Microscopy Environment) data repositories. The package enables automated image processing workflows that connect OMERO data with micro-SAM for generating training datasets and annotations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/modules
   examples
   contributing

Key Features
------------

- **Micro-SAM Integration**: Focused exclusively on micro-SAM for image segmentation
- **OMERO Connectivity**: Professional-grade connection management with secure password storage
- **Modern Architecture**: Clean, pip-installable solution with src layout
- **Jupyter Widgets**: Interactive configuration and connection widgets
- **Batch Processing**: Efficient processing of large image datasets
- **Training Data Generation**: Automated creation of training/validation datasets

Quick Start
-----------

Installation requires conda for micro-SAM dependency:

.. code-block:: bash

   conda activate micro-sam
   pip install omero-annotate-ai

Basic usage:

.. code-block:: python

   from omero_annotate_ai import create_config_widget, create_pipeline

   # Create configuration widget for Jupyter
   config_widget = create_config_widget()
   
   # Create and run annotation pipeline
   pipeline = create_pipeline(config, conn)
   results = pipeline.run()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`