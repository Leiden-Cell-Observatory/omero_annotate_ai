# OMERO Annotate.AI

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Leiden-Cell-Observatory/omero_annotate_ai/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fannotation%2Fomero-annotate-ai-annotation-widget.ipynb)
[![CI/CD](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/actions)
[![Documentation](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/actions/workflows/docs.yml/badge.svg)](https://leiden-cell-observatory.github.io/omero_annotate_ai/)
[![PyPI version](https://img.shields.io/pypi/v/omero-annotate-ai)](https://pypi.org/project/omero-annotate-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/omero-annotate-ai)](https://pypi.org/project/omero-annotate-ai/)
[![License](https://img.shields.io/github/license/Leiden-Cell-Observatory/omero_annotate_ai)](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/LICENSE)

Package to support reproducible image annotation workflows for AI training using OMERO data repositories. This Python package provides Jupyter widgets and tools for reproducible annotation, training, and inference using micro-SAM, Cellpose, and other AI models directly with OMERO datasets.

## Key Features

- **Interactive Jupyter widgets** for OMERO connection and workflow configuration
- **AI-assisted annotation** using micro-SAM integration 
- **Reproducible workflows** with YAML configuration tracking
- **Training data preparation** for BiaPy and DL4MicEverywhere
- **Direct OMERO integration** with automatic result storage

## Quick Start

### Installation

```bash
# Recommended: Using pixi
pixi init myproject && cd myproject
pixi add micro-sam
pixi add --pypi omero-annotate-ai
pixi shell

# Alternative: Conda + pip
conda install -c conda-forge micro-sam
pip install omero-annotate-ai
```

📖 See [Installation Guide](https://leiden-cell-observatory.github.io/omero_annotate_ai/installation/) for detailed instructions and troubleshooting.

### Basic Usage

**OMERO Connection Widget**
![OMERO Connection Widget](images/omero_connect_widget.png)

**Annotation Pipeline Widget** 
![Annotation Pipeline Widget](images/omero_annotation_widget.png)

```python
from omero_annotate_ai import create_omero_connection_widget, create_workflow_widget, create_pipeline

# Connect to OMERO
conn_widget = create_omero_connection_widget()
conn_widget.display()
conn = conn_widget.get_connection()

# Configure annotation workflow  
workflow_widget = create_workflow_widget(connection=conn)
workflow_widget.display()
config = workflow_widget.get_config()

# Run annotation pipeline
pipeline = create_pipeline(config, conn)
table_id, processed_images = pipeline.run_full_workflow()
```

### Example Notebooks

Try these example notebooks to get started:
- [Widget-based annotation workflow](notebooks/annotation/omero-annotate-ai-annotation-widget.ipynb)
- [YAML-based configuration](notebooks/annotation/omero-annotate-ai-from-yaml.ipynb)
- [Training with BiaPy](notebooks/training/omero-training_biapy.ipynb)

### Alternative: YAML Configuration

For batch processing and reproducible workflows, you can also use YAML configuration files:

```python
from omero_annotate_ai.core.annotation_config import load_config
from omero_annotate_ai.core.annotation_pipeline import create_pipeline

# Load configuration from YAML
config = load_config("annotation_config.yaml")
conn = create_connection(host="omero.server.com", user="username")

# Run annotation pipeline
pipeline = create_pipeline(config, conn)
results = pipeline.run_full_workflow()
```

See the [YAML Configuration Guide](docs/configuration.md) for complete documentation.

## Documentation

📚 **[Complete Documentation](docs/index.md)**

- **[Installation Guide](docs/installation.md)** - Detailed installation instructions and troubleshooting
- **[micro-SAM Tutorial](docs/tutorials/microsam-annotation-pipeline.md)** - Step-by-step annotation workflow tutorial
- **[YAML Configuration Guide](docs/configuration.md)** - Complete YAML configuration reference and examples
- **[API Reference](docs/api/index.md)** - Complete API documentation
- **[Examples](notebooks/)** - Jupyter notebook tutorials

## Links

- **[GitHub Repository](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai)**
- **[PyPI Package](https://pypi.org/project/omero-annotate-ai/)**
- **[Issues & Support](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/issues)**

## Contributing

We welcome contributions! For development setup:

1. Fork the repository
2. Clone and set up development environment:
   ```bash
   git clone https://github.com/YOUR_USERNAME/omero_annotate_ai.git
   cd omero_annotate_ai
   pixi install
   ```
3. Make changes and run tests: `pixi run pytest`
4. Submit a pull request

See [Installation Guide - Development Setup](https://leiden-cell-observatory.github.io/omero_annotate_ai/installation#development-setup) for detailed instructions.

## Contact

**Maarten Paul** - m.w.paul@lacdr.leidenuniv.nl

**Acknowledgments**: Developed within the [NL-BioImaging](https://github.com/NL-BioImaging) infrastructure, funded by NWO.
