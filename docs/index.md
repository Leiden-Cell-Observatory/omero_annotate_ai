# OMERO Annotate AI

omero-annotate-ai is a Python package that provides micro-SAM annotation 
workflows for OMERO (Open Microscopy Environment) data repositories. The package enables 
automated image processing workflows that connect OMERO data with micro-SAM for generating 
training datasets and annotations.

## Key Features

* **Streamlined 2-Widget Workflow**: Simplified interface with connection and workflow widgets
* **MIFA-Compliant Tracking**: Enhanced annotation ID tracking compatible with BioImage Archive standards  
* **Enhanced Resume Functionality**: Robust workflow resumption with status tracking
* **Comprehensive Configuration**: YAML-based configuration with validation
* **Full Pipeline Integration**: Direct micro-SAM processing with OMERO integration

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
