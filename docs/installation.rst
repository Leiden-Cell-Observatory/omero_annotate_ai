Installation
============

Prerequisites
-------------

omero-annotate-ai requires Python 3.8 or higher and conda for the micro-SAM dependency.

Required Environment Setup
---------------------------

The package requires the micro-SAM conda environment to be activated before installation:

.. code-block:: bash

   # Install micro-SAM via conda (required dependency)
   conda install -c conda-forge micro_sam
   
   # Activate the environment containing micro-SAM
   conda activate micro-sam

Standard Installation
---------------------

Install from PyPI:

.. code-block:: bash

   pip install omero-annotate-ai

Development Installation
------------------------

For development or latest features:

.. code-block:: bash

   git clone https://github.com/Leiden-Cell-Observatory/omero_annotate_ai.git
   cd omero_annotate_ai
   pip install -e .[dev,test,docs]

Optional Dependencies
---------------------

The package includes several optional dependency groups:

.. code-block:: bash

   # For Napari integration
   pip install omero-annotate-ai[napari]
   
   # For development (includes testing and linting tools)
   pip install omero-annotate-ai[dev]
   
   # For testing
   pip install omero-annotate-ai[test]
   
   # For documentation building
   pip install omero-annotate-ai[docs]
   
   # Install all optional dependencies
   pip install omero-annotate-ai[all]

Verification
------------

Test your installation:

.. code-block:: python

   import omero_annotate_ai
   from omero_annotate_ai.core.config import create_default_config
   from omero_annotate_ai.widgets import create_config_widget
   
   print("✅ omero-annotate-ai installed successfully!")

Troubleshooting
---------------

**Import Errors with micro-SAM**
   Ensure micro-SAM is installed via conda and the environment is activated.

**OMERO Connection Issues**
   Install OMERO dependencies: ``pip install omero-py ezomero``

**Widget Display Issues**
   Install Jupyter widgets: ``pip install ipywidgets`` and enable the extension.