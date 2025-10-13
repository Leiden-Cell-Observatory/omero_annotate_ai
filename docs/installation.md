# Installation Guide

## Prerequisites

- **Python 3.8 or higher**
- **Pixi** (recommended, install pixi following the [official instructions](https://pixi.sh/latest/)
) or **Conda** for dependency management

## Quick Install (Recommended)

Using **Pixi** - automatically handles all dependencies including conda packages:

```bash
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```


```bash
# Create a new project
pixi init omero-annotate-ai
cd omero-annotate-ai

# Add dependencies
pixi add python=3.12
pixi add micro-sam                      # AI model
pixi add --pypi omero-annotate-ai      # Main package

# Activate environment
pixi shell
```

### Verify Installation
Run `pixi run python`, then:   

```python
import omero_annotate_ai
print(f"✅ Version: {omero_annotate_ai.__version__}")
```

## Alternative: Conda + Pip

For users who prefer manual conda environment management:

```bash
# Create and activate environment
conda create -n omero-ai python=3.11
conda activate omero-ai

# Install conda dependencies
conda install -c conda-forge micro-sam napari zeroc-ice

# Install package from PyPI
pip install omero-annotate-ai
```

## Quick Start

After installation, try connecting to OMERO:

```python
from omero_annotate_ai import create_omero_connection_widget

# Display connection widget in Jupyter
conn_widget = create_omero_connection_widget()
conn_widget.display()

# Get connection
conn = conn_widget.get_connection()
```

## Development Setup

For contributors who want to modify the code:

### Clone and Install

```bash
git clone https://github.com/Leiden-Cell-Observatory/omero_annotate_ai.git
cd omero_annotate_ai
pixi install              # Installs dev dependencies + package in editable mode
```

### Run Tests

```bash
pixi run pytest           # Run all tests
pixi run test-unit        # Run unit tests only
pixi run test-cov         # Run with coverage report
```

### Development Environments

Pixi provides multiple environments configured in `pyproject.toml`:

```bash
pixi shell                # Default environment (user)
pixi shell -e dev         # Development tools (pytest, black, isort)
pixi shell -e docs        # Documentation building (mkdocs)
```

### Code Quality

```bash
pixi run format           # Format code with black and isort
pixi run lint             # Check code quality
```

## Troubleshooting

### Pixi not found

Install pixi following the [official instructions](https://pixi.sh/latest/):

```bash
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

### Import Errors

**micro-SAM not found:**

```bash
# With pixi
pixi add micro-sam

# With conda
conda install -c conda-forge micro-sam
```

**OMERO connection issues:**

```bash
# Ensure OMERO dependencies are installed
pixi add --pypi omero-py ezomero
```

### Widget Display Issues

**Widgets not showing in Jupyter:**

```bash
# Install and enable widget extension
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Platform-Specific Issues

**zeroc-ice installation:**

For pip-only installations (without conda), `zeroc-ice` requires platform-specific wheels:

- **Linux x86_64**: [zeroc-ice-py-linux-x86_64](https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases)
- **Windows x86_64**: [zeroc-ice-py-win-x86_64](https://github.com/glencoesoftware/zeroc-ice-py-win-x86_64/releases)
- **macOS Universal**: [zeroc-ice-py-macos-universal2](https://github.com/glencoesoftware/zeroc-ice-py-macos-universal2/releases)

**Note:** Both pixi and conda handle zeroc-ice automatically - no manual wheel installation needed.

## Next Steps

- 📚 Try the [micro-SAM Tutorial](tutorials/microsam-annotation-pipeline.md)
- 📖 Read the [Configuration Guide](configuration.md)
- 💻 Explore [example notebooks](../notebooks/)
