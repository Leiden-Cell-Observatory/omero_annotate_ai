# OMERO Annotate AI

omero-annotate-ai is a Python package that provides tools for reproducible AI
workflows (annotation, training and inference) using OMERO (Open Microscopy Environment) data repositories. The package directly connects OMERO datasets with AI dataset annotation using micro-sam assisted annotation of images.

## Key Features

* OMERO connection and annotation workflow widgets within Jupyter notebooks
* Using Pydantic model validated YAML configuration files to track the annotation and training workflow.
* Direct integration of micro-SAM annotation of OMERO data
* Saving annotations and annotation configuration back into OMERO (OMERO.table, YAML).
* Preparation of training data for Biapy and DL4MicEverywhere

## Quick Start

Installation requires conda for micro-SAM dependency:

```bash
conda activate micro-sam
pip install -e .
```

Basic usage with 2-widget workflow:

```python
from omero_annotate_ai import create_omero_connection_widget, create_workflow_widget, create_pipeline

# Step 1: Connect to OMERO
conn_widget = create_omero_connection_widget()
conn_widget.display()
conn = conn_widget.get_connection()

# Step 2: Configure workflow
workflow_widget = create_workflow_widget(connection=conn)
workflow_widget.display()
config = workflow_widget.get_config()

# Step 3: Run pipeline
pipeline = create_pipeline(config, conn)
table_id, processed_images = pipeline.run_full_workflow()
```

## Documentation Structure

- [Installation Guide](installation.md) - Setup instructions and requirements
- [API Reference](api/index.md) - Complete API documentation with docstrings
- [Examples](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/tree/main/examples) - Jupyter notebook examples

## Links

- [GitHub Repository](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai)
- [PyPI Package](https://pypi.org/project/omero-annotate-ai/)
- [Issues](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/issues)
