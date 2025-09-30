# YAML Configuration Guide

The OMERO Annotate AI package uses structured YAML configuration files to define annotation workflows. This ensures reproducibility and allows fine-grained control over all aspects of the annotation process.

## Configuration Overview

The configuration system is built around a robust Pydantic model that validates all settings and provides sensible defaults. You can use configurations in two ways:

1. **Interactive widgets** - Generate configurations through Jupyter notebook widgets
2. **YAML files** - Define configurations directly in YAML for batch processing

## Quick Start Example

Here's a minimal working configuration:

```yaml
# Basic OMERO micro-SAM Configuration
schema_version: "1.0.0"
name: "nuclei_segmentation_demo"

# OMERO data source
omero:
  container_type: "dataset"
  container_id: 123
  source_desc: "HeLa cells for nuclei segmentation"

# Spatial processing settings
spatial_coverage:
  channels: [0]           # DAPI channel
  timepoints: [0]         # Single timepoint
  z_slices: [0, 1, 2]     # Process 3 z-slices
  three_d: false          # Process slice-by-slice

# AI model configuration
ai_model:
  name: "micro-sam"
  model_type: "vit_b_lm"

# Processing parameters
processing:
  batch_size: 0           # Process all images
  use_patches: true
  patch_size: [512, 512]
  patches_per_image: 4

# Training data split
training:
  train_n: 5
  validate_n: 2
```

## Complete Configuration Schema

### Core Identification

```yaml
schema_version: "1.0.0"     # Configuration schema version
name: "workflow_name"       # Unique workflow identifier  
version: "1.0.0"           # Your workflow version
created: "2025-01-14T10:30:00Z"  # Auto-generated timestamp

authors:                   # Optional author information
  - name: "Your Name"
    email: "your.email@institution.edu" 
    affiliation: "Your Institution"
```

### OMERO Connection

```yaml
omero:
  container_type: "dataset"    # "dataset", "project", or "plate"
  container_id: 123           # OMERO container ID
  source_desc: "Description"  # Human-readable description
```

### Study Context (MIFA Compatible)

```yaml
study:
  title: "Study title"
  description: "Detailed study description" 
  keywords: ["nuclei", "segmentation", "fluorescence"]
  organism: "Homo sapiens"
  imaging_method: "fluorescence microscopy"

dataset:
  source_dataset_id: "S-BIAD123"  # BioImage Archive accession
  source_dataset_url: "https://www.ebi.ac.uk/bioimaging/studies/S-BIAD123"
  source_description: "Dataset description"
  license: "CC-BY-4.0"
```

### Spatial Coverage Settings

The spatial coverage section defines which parts of your images to process:

```yaml
spatial_coverage:
  # Basic spatial selection
  channels: [0, 1]              # Channel indices to process
  timepoints: [0]               # Timepoint indices  
  z_slices: [0, 1, 2, 3, 4]     # Z-slice indices
  
  # Selection modes
  timepoint_mode: "specific"     # "all", "random", "specific"
  z_slice_mode: "specific"       # "all", "random", "specific"
  
  # 3D processing
  three_d: false                 # Enable 3D volumetric processing
  z_range_start: 0               # Start z-slice for 3D (when three_d=true)
  z_range_end: 10                # End z-slice for 3D (when three_d=true)
  
  spatial_units: "pixels"        # Spatial measurement units
```

#### 2D vs 3D Processing

**2D Mode (Default)**:
```yaml
spatial_coverage:
  three_d: false
  z_slices: [0, 1, 2, 3, 4]   # Each slice processed individually
```

**3D Volumetric Mode**:
```yaml
spatial_coverage:
  three_d: true
  z_range_start: 0             # Process z-slices 0-10 as one volume
  z_range_end: 10
  # OR alternatively:
  z_slices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### Annotation Methodology

```yaml
annotation_methodology:
  annotation_type: "segmentation_mask"      # "segmentation_mask", "bounding_box", "point", "classification"
  annotation_method: "automatic"            # "manual", "semi_automatic", "automatic"  
  annotation_criteria: "Complete nuclei boundaries based on DAPI staining"
  annotation_coverage: "representative"     # "all", "representative", "partial"
```

### AI Model Configuration

```yaml
ai_model:
  name: "micro-sam"            # Model identifier
  version: "latest"            # Model version
  model_type: "vit_b_lm"       # "vit_b_lm", "vit_l_lm", "vit_h_lm"
  framework: "pytorch"         # AI framework
```

### Processing Parameters

```yaml
processing:
  batch_size: 0                # Number of images to process (0 = all)
  use_patches: true            # Extract patches vs full images
  patch_size: [512, 512]       # Patch dimensions [width, height]
  patches_per_image: 4         # Number of patches per image
  random_patches: true         # Use random vs systematic patch extraction
```

### Training Configuration

```yaml
training:
  validation_strategy: "random_split"    # "random_split", "expert_review", "cross_validation"
  train_fraction: 0.7                   # Training data fraction (auto-calculated if train_n specified)
  train_n: 10                          # Explicit number of training images
  validation_fraction: 0.3              # Validation data fraction  
  validate_n: 5                        # Explicit number of validation images
  segment_all: false                   # Segment all objects vs sample
  quality_threshold: 0.8               # Minimum quality score (optional)
```

### Workflow Control

```yaml
workflow:
  resume_from_table: false      # Resume from existing annotation table
  read_only_mode: false         # Read-only mode for viewing results
```

### Output Configuration

```yaml
output:
  output_directory: "./annotations"    # Output directory path
  format: "ome_tiff"                  # "ome_tiff", "png", "numpy"
  compression: null                   # Compression method (optional)
  resume_from_checkpoint: false       # Resume interrupted workflow
```

### Metadata and Tracking

```yaml
# Workflow metadata (bioimage.io compatible)
documentation: "https://github.com/your-org/your-repo/docs"
repository: "https://github.com/your-org/your-repo"  
tags: ["segmentation", "nuclei", "micro-sam", "AI-ready"]

# Annotation tracking (auto-populated during processing)
annotations: []  # List of ImageAnnotation records
```

## Working with Configuration Files

### Loading Configurations

```python
from omero_annotate_ai.core.annotation_config import load_config

# From YAML file
config = load_config("my_config.yaml")

# From dictionary  
config_dict = {...}
config = load_config(config_dict)

# Create default configuration
config = create_default_config()
```

### Saving Configurations

```python
# Save to YAML file
config.save_yaml("my_config.yaml")

# Export as dictionary
config_dict = config.to_dict()

# Export as YAML string
yaml_string = config.to_yaml()
```

### Configuration Templates

Generate a complete template with all options:

```python
from omero_annotate_ai.core.annotation_config import get_config_template

template = get_config_template()
print(template)
```

## Advanced Configuration Examples

### Multi-channel Time Series

```yaml
name: "multi_channel_timeseries"
spatial_coverage:
  channels: [0, 1, 2]         # DAPI, GFP, RFP
  timepoints: [0, 5, 10, 15]  # Every 5th timepoint
  z_slices: [2]               # Middle focal plane
  timepoint_mode: "specific"
```

### 3D Volumetric Processing

```yaml
name: "3d_organoid_segmentation" 
spatial_coverage:
  channels: [0]
  timepoints: [0]
  three_d: true
  z_range_start: 5
  z_range_end: 25             # Process 20 z-slices as volume
```

### Large-scale Patch-based Processing

```yaml
name: "large_image_patches"
processing:
  use_patches: true
  patch_size: [1024, 1024]    # Larger patches
  patches_per_image: 16       # More patches per image
  random_patches: false       # Systematic grid sampling
training:
  train_n: 100               # Process many images
  validate_n: 20
```

### High-throughput Screening

```yaml
name: "hts_plate_processing"
omero:
  container_type: "plate"
  container_id: 456
spatial_coverage:
  channels: [0, 1]            # Nuclei + marker
  timepoints: [0]
  z_slices: [0]               # Single focal plane
processing:
  batch_size: 50              # Process in batches
  use_patches: false          # Full images only
```

## Configuration Validation

The configuration system provides thorough validation:

- **Type checking** - All fields are validated against expected types
- **Range validation** - Numeric fields are checked against valid ranges  
- **Dependency validation** - Related fields are checked for consistency
- **3D configuration validation** - 3D settings are validated for completeness

## Integration with Widgets

Widget-generated configurations are fully compatible with YAML configurations:

```python
# Generate config with widgets
workflow_widget = create_workflow_widget(connection=conn)
workflow_widget.display()
config = workflow_widget.get_config()

# Save widget config as YAML
config.save_yaml("widget_generated_config.yaml")

# Load and modify YAML config
config = load_config("widget_generated_config.yaml")
config.processing.batch_size = 10
config.save_yaml("modified_config.yaml")
```

## Best Practices

1. **Use descriptive names** - Make workflow names meaningful for tracking
2. **Document your criteria** - Specify clear annotation criteria
3. **Version your configs** - Update version numbers when making changes
4. **Validate before running** - Test configurations on small datasets first
5. **Keep configs with data** - Store configuration files alongside results
6. **Use templates** - Start from the provided template for completeness

## Troubleshooting

Common configuration issues and solutions:

**3D Configuration Errors**:
```yaml
# ❌ Incorrect - missing z_range when three_d=true
spatial_coverage:
  three_d: true
  z_slices: [0, 1, 2]

# ✅ Correct - specify z_range for 3D
spatial_coverage:
  three_d: true  
  z_range_start: 0
  z_range_end: 2
```

**Training Split Validation**:
```yaml
# ❌ Incorrect - fractions don't sum to 1.0
training:
  train_fraction: 0.8
  validation_fraction: 0.3

# ✅ Correct - use explicit counts or proper fractions
training:
  train_n: 8
  validate_n: 2
```

For more examples and tutorials, see the [tutorial section](tutorials/index.md) for detailed workflow guides.
