# Installation Guide

## Prerequisites

- **Python 3.8 or higher**
- **conda** for micro-SAM dependency (AI features)
- **Git** for development installation

## Installation Methods

Choose the installation method that best fits your needs:

=== "🚀 Pixi (Recommended)"

    [Pixi](https://pixi.sh/latest/) automatically handles all dependencies including conda packages and platform-specific wheels.

    **1. Install pixi** (if not already installed):
    ```bash
    # Windows (PowerShell)
    iwr -useb https://pixi.sh/install.ps1 | iex

    # macOS/Linux  
    curl -fsSL https://pixi.sh/install.sh | bash
    ```

    **2. Clone and install**:
    ```bash
    git clone https://github.com/Leiden-Cell-Observatory/omero_annotate_ai.git
    cd omero_annotate_ai
    pixi install
    ```

    **3. Activate environment**:
    ```bash
    pixi shell
    ```

    That's it! All dependencies (micro-sam, napari, zeroc-ice, etc.) are automatically configured.

=== "📦 PyPI (Core Only)"

    For basic OMERO functionality without AI features:

    ```bash
    pip install omero-annotate-ai
    ```

    !!! warning "Limited Features"
        This installs core functionality only. For AI annotation features, you'll need micro-SAM which requires conda or pixi installation.

=== "🐍 Conda + Pip"

    If you prefer manual dependency management:

    **1. Create conda environment**:
    ```bash
    conda create -n omero_annotate_ai python=3.12
    conda activate omero_annotate_ai
    ```

    **2. Install conda-only dependencies**:
    ```bash
    conda install -c conda-forge micro-sam napari
    ```

    **3. Install platform-specific zeroc-ice wheel**:

    === "Linux x86_64"

        ```bash
        pip install https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp312-cp312-manylinux_2_28_x86_64.whl
        ```

    === "Windows x86_64"

        ```bash
        pip install https://github.com/glencoesoftware/zeroc-ice-py-win-x86_64/releases/download/20240325/zeroc_ice-3.6.5-cp312-cp312-win_amd64.whl
        ```

    === "macOS Universal"

        ```bash
        pip install https://github.com/glencoesoftware/zeroc-ice-py-macos-universal2/releases/download/20240131/zeroc_ice-3.6.5-cp312-cp312-macosx_11_0_universal2.whl
        ```

    **4. Install the package**:
    ```bash
    # From PyPI 
    pip install omero-annotate-ai[microsam]

    # Or from source
    git clone https://github.com/Leiden-Cell-Observatory/omero_annotate_ai.git
    cd omero_annotate_ai
    pip install -e .[microsam]
    ```

=== "🔧 Development"

    For contributors and developers:

    === "Pixi (Recommended)"

        ```bash
        git clone https://github.com/Leiden-Cell-Observatory/omero_annotate_ai.git
        cd omero_annotate_ai
        pixi install
        pixi run pytest  # Run tests
        ```

        **Development environments:**
        ```bash
        pixi shell -e dev     # Development environment
        pixi shell -e docs    # Documentation environment  
        pixi shell -e test    # Testing environment
        ```

    === "Conda"

        ```bash
        conda create -n omero_annotate_ai-dev python=3.12
        conda activate omero_annotate_ai-dev  
        conda install -c conda-forge micro-sam napari
        pip install -e .[dev,microsam]
        ```

## Optional Dependencies

The package includes several optional dependency groups:

=== "Feature Sets"

    ```bash
    # AI annotation features (includes micro-sam)
    pip install omero-annotate-ai[microsam]

    # Napari integration
    pip install omero-annotate-ai[napari]

    # Install all optional dependencies
    pip install omero-annotate-ai[all]
    ```

=== "Development Tools"

    ```bash
    # Development tools (testing, linting, formatting)
    pip install omero-annotate-ai[dev]

    # Testing framework
    pip install omero-annotate-ai[test]

    # Documentation building
    pip install omero-annotate-ai[docs]
    ```

## Quick Start After Installation

Start a Jupyter environment:

=== "Pixi"

    ```bash
    pixi run jupyter lab
    ```

=== "Conda/Pip"

    ```bash
    conda activate omero_annotate_ai
    jupyter lab
    ```

## Verification

Test your installation:

```python
import omero_annotate_ai
from omero_annotate_ai.core.annotation_config import create_default_config
from omero_annotate_ai.widgets import create_omero_connection_widget

print("✅ omero-annotate-ai installed successfully!")
```

## Troubleshooting

=== "Import Errors"

    **micro-SAM not found:**
    ```bash
    # Ensure micro-SAM is installed via conda
    conda install -c conda-forge micro-sam
    ```

    **OMERO connection issues:**
    ```bash
    pip install omero-py ezomero
    ```

=== "Widget Issues"

    **Widgets not displaying:**
    ```bash
    pip install ipywidgets
    jupyter nbextension enable --py widgetsnbextension
    ```

=== "Environment Issues"

    **Pixi not found:**
    ```bash
    # Install pixi
    curl -fsSL https://pixi.sh/install.sh | bash
    ```

    **Conda environment issues:**
    ```bash
    conda clean --all
    conda update conda
    ```
