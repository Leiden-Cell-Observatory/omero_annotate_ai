# QuPath Extension: OMERO Annotate AI

A QuPath extension for creating AI training data from microscopy image annotations, with optional OMERO integration.

Part of the [omero_annotate_ai](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai) ecosystem.

## Features

- **Draw annotations** in QuPath using its powerful annotation tools (brush, polygon, wand, etc.)
- **Extract training patches** centered on annotations with configurable patch size
- **Generate label masks** from annotation geometries (16-bit TIFF, one label per class)
- **Train/validation split** with configurable ratios
- **Two operating modes**:
  - **Local-only**: Works with any image, no OMERO required
  - **OMERO mode**: Full round-trip with OMERO server (tracking table, config sync, ROI upload)
- **Cross-tool compatibility**: Workflows started in QuPath can be continued in Jupyter/napari and vice versa

## Requirements

- **QuPath 0.6.0+** (also compatible with 0.7.x)
- **Java 21+**
- For OMERO mode: [qupath-extension-omero](https://github.com/qupath/qupath-extension-omero) installed

## Installation

1. Download the latest JAR from [Releases](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/releases)
2. Drag the JAR file onto the QuPath main window
3. Restart QuPath

The extension appears under **Extensions > OMERO Annotate AI**.

## Quick Start

### Local-Only Mode (no OMERO)

1. Open an image in QuPath
2. Go to **Extensions > OMERO Annotate AI > Open Annotate Dialog**
3. In the **Connection** tab, check **"Work without OMERO (local only)"**
4. In the **Configure** tab:
   - Set patch size (default 256x256)
   - Adjust train/val split (default 80/20)
   - Set output directory
5. Draw annotations on the image using QuPath's tools
6. In the **Export & Upload** tab, click **"Export Training Data"**

Output structure:
```
output_dir/
  input/                    # Source image patches (TIFF)
  output/                   # Label masks (16-bit TIFF)
  tracking_table.csv        # Local progress tracking
  annotation_config.yaml    # Workflow configuration
```

### OMERO Mode

1. Connect to OMERO (via qupath-extension-omero or the built-in login)
2. Open an image from OMERO
3. Configure the workflow (set OMERO container ID)
4. Draw annotations and export
5. Click **"Upload to OMERO"** to push results to the server

The OMERO tracking table is fully compatible with `omero_annotate_ai` Python workflows.

## Interoperability

This extension uses the same data formats as the Python `omero_annotate_ai` package:

- **YAML config**: `AnnotationConfig` schema v2.0.0
- **OMERO tracking table**: Same 23-column schema
- **Annotation type**: `"qupath_manual"` (vs `"micro_sam"` for Python workflows)
- **Namespace**: `openmicroscopy.org/omero/annotate/config`

Workflows can be started in any tool (QuPath, napari, Jupyter) and continued in another.

## Building from Source

```bash
cd qupath-extension
./gradlew build
```

The JAR is generated at `build/libs/qupath-extension-omero-annotate-*.jar`.

## License

Apache-2.0 (same as omero_annotate_ai)
