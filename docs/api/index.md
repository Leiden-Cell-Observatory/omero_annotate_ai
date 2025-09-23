# API Reference

The OMERO Annotate AI API documentation is automatically generated from source code docstrings.

## Core Modules

### Configuration and Pipeline Management
- **annotation_config** - YAML configuration management and validation
- **annotation_pipeline** - Main pipeline execution and workflow management

### OMERO Integration  
- **omero_functions** - Core OMERO server interaction functions
- **omero_utils** - Utility functions for OMERO data handling
- **simple_connection** - Simplified OMERO connection management

### Processing Modules
- **image_functions** - Image processing and ROI conversion utilities
- **training_functions** - Training data preparation and validation
- **file_io_functions** - File input/output operations

### Jupyter Widgets
- **connection_widget** - OMERO connection interface
- **workflow_widget** - Annotation pipeline configuration interface

## Usage Examples

Most API functions are designed to work together in pipelines:

```python
from omero_annotate_ai.core.annotation_config import load_config
from omero_annotate_ai.core.annotation_pipeline import create_pipeline
from omero_annotate_ai.omero.simple_connection import create_connection

# Load configuration
config = load_config("config.yaml")

# Connect to OMERO
conn = create_connection(host="omero.server.com", user="username")

# Create and run pipeline
pipeline = create_pipeline(config, conn)
results = pipeline.run_full_workflow()
```

## Getting Started

For practical usage examples, see:
- [micro-SAM Tutorial](../tutorials/microsam-annotation-pipeline.md) - Complete workflow walkthrough
- [Configuration Guide](../configuration.md) - Parameter reference
- [Example Notebooks](../../notebooks/) - Working code examples
