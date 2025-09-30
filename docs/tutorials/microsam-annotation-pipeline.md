# Tutorial: micro-SAM Annotation Pipeline

This tutorial walks you through the complete micro-SAM annotation workflow using OMERO Annotate AI. You'll learn how to set up an annotation project, configure the pipeline, and process images with AI-assisted segmentation.

## Overview

The micro-SAM annotation pipeline provides automated segmentation of biological structures in microscopy images. This tutorial covers:

1. **OMERO connection setup** - Connect to your OMERO server
2. **Data selection** - Choose datasets and configure spatial parameters  
3. **Pipeline configuration** - Set up micro-SAM model parameters
4. **Annotation execution** - Run the annotation workflow
5. **Result review** - Examine and validate annotations

## Prerequisites

!!! note "Before You Start"
    - OMERO Annotate AI installed with micro-SAM support
    - Access to an OMERO server with image data
    - Jupyter notebook environment

!!! tip "Installation Check"
    Verify your installation by running:
    ```python
    import omero_annotate_ai
    print("✅ OMERO Annotate AI is ready!")
    ```

## Tutorial Setup

### 1. Environment Preparation

Start your environment and launch Jupyter:

```bash
# With pixi
pixi shell
pixi run jupyter lab

# With conda
conda activate omero_annotate_ai  
jupyter lab
```

### 2. Import Required Modules

!!! info "Required Imports"
    These imports provide all the functionality needed for the tutorial:

```python
import numpy as np
import matplotlib.pyplot as plt
from omero_annotate_ai import (
    create_omero_connection_widget,
    create_workflow_widget, 
    create_pipeline
)
from omero_annotate_ai.core.annotation_config import load_config
```

## Step 1: OMERO Connection

### Using the Connection Widget

The OMERO connection widget provides a secure interface for server authentication:

```python
# Create and display connection widget
conn_widget = create_omero_connection_widget()
conn_widget.display()

# Wait for user to enter credentials and connect
# Widget will validate connection and store credentials securely
```

![OMERO Connection Widget](../images/omero_connect_widget.png)

**Widget features:**
- Server URL validation
- Secure credential storage with keyring
- Connection status indicator
- Group selection support

### Programmatic Connection (Alternative)

For automated workflows, you can connect programmatically:

```python
from omero_annotate_ai.omero.simple_connection import create_connection

conn = create_connection(
    host="omero.example.com",
    user="your_username",
    password="your_password",  # Use environment variables in production
    port=4064,
    secure=True
)

print(f"Connected to OMERO as {conn.getUser().getName()}")
```

## Step 2: Workflow Configuration

### Using the Workflow Widget

The annotation pipeline widget provides visual configuration:

```python
# Get connection from widget
conn = conn_widget.get_connection()

# Create workflow configuration widget
workflow_widget = create_workflow_widget(connection=conn)
workflow_widget.display()

# Widget allows you to:
# - Browse and select OMERO datasets/projects
# - Configure spatial parameters (channels, z-slices, timepoints)
# - Set micro-SAM model parameters
# - Define training/validation splits
```

![Annotation Pipeline Widget](../images/omero_annotation_widget.png)

**Configuration options:**
- **Data source**: Select OMERO dataset or project
- **Spatial coverage**: Choose channels, z-slices, timepoints
- **Model settings**: micro-SAM model type and parameters
- **Processing mode**: Patch-based or full image processing
- **Training split**: Define training vs validation data

### YAML Configuration (Alternative)

For reproducible workflows, use YAML configuration:

```python
# Create configuration template
from omero_annotate_ai.core.annotation_config import get_config_template

template = get_config_template()
print(template)
```

Example configuration:

```yaml
name: "nuclei_segmentation_tutorial"
omero:
  container_type: "dataset"
  container_id: 123
  
spatial_coverage:
  channels: [0]           # DAPI channel
  timepoints: [0]         # Single timepoint
  z_slices: [0, 1, 2]     # Three z-slices
  three_d: false          # Process slice-by-slice

ai_model:
  model_type: "vit_b_lm"  # Balanced performance/speed

processing:
  batch_size: 0           # Process all images
  use_patches: true
  patch_size: [512, 512]
  patches_per_image: 4

training:
  train_n: 5
  validate_n: 2
```

## Step 3: Pipeline Execution

### Running the Annotation Pipeline

```python
# Get configuration from widget
config = workflow_widget.get_config()

# Or load from YAML
# config = load_config("tutorial_config.yaml")

# Create and run pipeline
pipeline = create_pipeline(config, conn)
table_id, processed_images = pipeline.run_full_workflow()

print(f"✅ Annotation complete!")
print(f"📊 Results table ID: {table_id}")
print(f"🖼️ Processed {len(processed_images)} images")
```

### Understanding the Pipeline Process

The pipeline executes these steps automatically:

1. **Image retrieval** - Downloads images from OMERO
2. **Preprocessing** - Applies any configured image adjustments  
3. **Segmentation** - Runs micro-SAM on each image/patch
4. **Post-processing** - Converts masks to ROIs and labels
5. **Upload** - Stores results back to OMERO
6. **Tracking** - Updates progress table

### Monitoring Progress

Track pipeline progress in real-time:

```python
# Get progress summary
progress = pipeline.get_progress_summary()
print(f"Progress: {progress['completed_units']}/{progress['total_units']} "
      f"({progress['progress_percent']:.1f}%)")

# View detailed status
from omero_annotate_ai.omero.omero_functions import get_table_progress_summary
status = get_table_progress_summary(conn, table_id)
print(status)
```

## Step 4: Result Review

### Viewing Annotations in OMERO

Results are automatically stored in OMERO:

- **ROI annotations** - Segmentation boundaries as vector shapes
- **Label images** - Pixel masks as image attachments  
- **Progress table** - Detailed tracking information
- **Configuration** - YAML config saved as attachment

### Programmatic Result Access

```python
# Load results table
import pandas as pd
from omero_annotate_ai.omero.omero_functions import load_annotation_table

df = load_annotation_table(conn, table_id)
print(f"Results shape: {df.shape}")
print(df.head())

# Filter completed annotations
completed = df[df['processed'] == True]
print(f"Completed annotations: {len(completed)}")

# Access specific results
for idx, row in completed.iterrows():
    image_id = row['image_id']
    roi_id = row['roi_id'] 
    label_id = row['label_id']
    
    print(f"Image {image_id}: ROI {roi_id}, Label {label_id}")
```

### Quality Assessment

Evaluate annotation quality:

```python
# Check annotation statistics
stats = pipeline.get_annotation_statistics()
print(f"Average objects per image: {stats['mean_objects']:.1f}")
print(f"Object size range: {stats['size_range']}")

# Visualize sample results
import matplotlib.pyplot as plt

sample_image_id = completed.iloc[0]['image_id']
image, mask = pipeline.load_annotation_result(sample_image_id)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(mask)
axes[1].set_title('Segmentation Mask')
axes[2].imshow(image, cmap='gray', alpha=0.7)
axes[2].imshow(mask, alpha=0.3)
axes[2].set_title('Overlay')
plt.tight_layout()
plt.show()
```

## Step 5: Advanced Configuration

### 3D Volumetric Processing

For z-stack data, enable 3D processing:

```python
# Configure 3D processing
config.spatial_coverage.three_d = True
config.spatial_coverage.z_range_start = 0
config.spatial_coverage.z_range_end = 10

# Run 3D pipeline
pipeline_3d = create_pipeline(config, conn)
results_3d = pipeline_3d.run_full_workflow()
```

### Multi-channel Processing

Process multiple channels simultaneously:

```python
# Multi-channel configuration
config.spatial_coverage.channels = [0, 1, 2]  # DAPI, GFP, RFP

# Channel-specific model parameters
config.ai_model.model_type = "vit_l_lm"  # More powerful model for multi-channel
```

### Custom Model Parameters

Fine-tune micro-SAM behavior:

```python
# Advanced model configuration
config.ai_model.model_type = "vit_h_lm"  # Highest accuracy
config.processing.patch_size = [1024, 1024]  # Larger patches
config.processing.patches_per_image = 1  # Single patch per image
```

## Troubleshooting

### Troubleshooting

!!! warning "Common Issues and Solutions"

    **Connection Problems:**
    ```python
    # Test OMERO connection
    try:
        user = conn.getUser()
        print(f"✅ Connected as: {user.getName()}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        # Check server URL, credentials, network access
    ```

    **Memory Issues:**
    ```python
    # Reduce memory usage
    config.processing.batch_size = 1  # Process one image at a time
    config.processing.patch_size = [256, 256]  # Smaller patches
    ```

    **Slow Processing:**
    ```python
    # Speed optimization
    config.ai_model.model_type = "vit_b_lm"  # Faster model
    config.processing.patches_per_image = 2  # Fewer patches
    ```

!!! tip "Getting Help"
    - Check the [Configuration Guide](../configuration.md) for detailed parameter explanations
    - Review the other tutorials for more specialized use cases
    - Open [GitHub issues](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/issues) for bugs

## Next Steps

After completing this tutorial, explore:

- [Cellpose Integration Tutorial](cellpose-integration.md) - Alternative segmentation models
- [Training Data Preparation](training-data-prep.md) - Prepare data for model training
- [BiaPy Integration Tutorial](biapy-integration.md) - Train custom models
- [Batch Processing Guide](batch-processing.md) - Large-scale annotation workflows

## Summary

You've learned to:

✅ **Connect to OMERO** using widgets or programmatic methods  
✅ **Configure annotation workflows** with visual tools or YAML  
✅ **Execute micro-SAM pipelines** with progress monitoring  
✅ **Review and access results** stored in OMERO  
✅ **Troubleshoot common issues** and optimize performance

The micro-SAM annotation pipeline provides a powerful foundation for automated biological image analysis. Experiment with different configurations to find the optimal settings for your specific use case.
