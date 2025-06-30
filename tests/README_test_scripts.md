# Testing the OMERO Micro-SAM Annotation Pipeline

This directory contains test scripts and configurations for the `omero-annotate-ai` package.

## Files

### Test Scripts
- **`test_annotation_pipeline.py`** - Standalone Python script that replicates the Jupyter notebook functionality
- **`simple_test_config.yaml`** - Simple configuration for quick testing  
- **`test_config.yaml`** - Full configuration example with all options

### Original Implementation
- **`notebooks/omero-annotate-ai-batch.ipynb`** - Original Jupyter notebook (for reference)

## Quick Start

### 1. Dry Run Test (No OMERO Connection)
```bash
# Test configuration validation without running the pipeline
python test_annotation_pipeline.py --dry-run simple_test_config.yaml
```

### 2. Full Pipeline Test
```bash
# Run the complete annotation pipeline  
python test_annotation_pipeline.py simple_test_config.yaml
```

### 3. Custom Configuration
```bash
# Use your own configuration file
python test_annotation_pipeline.py my_config.yaml
```

### 4. Custom Output Directory
```bash
# Specify output directory
python test_annotation_pipeline.py --output-dir ./my_output simple_test_config.yaml
```

## Configuration Files

### Simple Test Configuration (`simple_test_config.yaml`)
- Uses fastest model (`vit_b`) for quick testing
- Small batch size (2 images)
- Processes all images in container
- No patch extraction
- Uploads results to OMERO

### Full Test Configuration (`test_config.yaml`)
- Balanced model (`vit_l`)
- Standard batch size (3 images)
- Configurable patch extraction
- All micro-SAM options available

## Environment Setup

**Important**: This test script must be run in the `micro-sam` conda environment where micro-SAM is installed.

### 1. Activate Environment
```bash
# Activate the micro-sam conda environment
conda activate micro-sam
```

### 2. Install Package
```bash
# Core package
pip install -e .

# With OMERO functionality
pip install -e .[omero]

# Development mode with all dependencies
pip install -e .[dev]
```

### 2. Environment Variables
Create a `.env` file (copy from `.env.example`):
```
HOST=your.omero.server
USER_NAME=your_username
GROUP=your_group
# PASSWORD will be prompted interactively
```

### 3. Micro-SAM Dependencies
```bash
# Install micro-SAM via conda (required)
conda install -c conda-forge micro_sam
```

## Configuration Options

### Micro-SAM Models
- `vit_b` - Fast, lightweight (recommended for testing)
- `vit_l` - Balanced performance and speed
- `vit_h` - High accuracy, slower processing
- `vit_b_lm` - Specialized model variant

### Container Types
- `project` - Process all images in all datasets within a project
- `dataset` - Process all images in a specific dataset
- `plate` - Process all images in a screening plate
- `screen` - Process all images in all plates within a screen
- `image` - Process a single image

### Processing Modes
- **2D Mode**: Process individual z-slices and timepoints
- **3D Mode**: Process volumetric data (set `three_d: true`)
- **Patch Mode**: Extract and process image patches (set `use_patches: true`)

## Troubleshooting

### Common Issues

1. **Package Import Error**
   ```
   ModuleNotFoundError: No module named 'omero_annotate_ai'
   ```
   **Solution**: Install the package with `pip install -e .[omero]`

2. **OMERO Connection Failed**
   ```
   ❌ Connection to OMERO Server Failed
   ```
   **Solution**: Check your `.env` file and OMERO server credentials

3. **Micro-SAM Not Available**
   ```
   ImportError: micro-sam and napari are required
   ```
   **Solution**: Install micro-SAM via conda: `conda install -c conda-forge micro_sam`

4. **Container Not Found**
   ```
   ValueError: Project with ID 101 not found
   ```
   **Solution**: Update the `container_id` in your YAML config to a valid container ID

### Debug Mode
Add print statements or use the `--dry-run` flag to validate configuration without running the full pipeline.

## Expected Output

### Successful Run
```
🚀 OMERO Micro-SAM Annotation Pipeline Test
✅ Package version: 0.1.0
✅ OMERO functionality available
✅ Configuration loaded successfully
✅ Configuration is valid
✅ Connected to OMERO Server
📊 Found 6 images to process
🚀 Starting annotation pipeline...
📋 Created tracking table 'microsam_training_quick_test' with 6 units
✅ Annotation pipeline completed successfully!
📊 Processed 6 images
📋 Tracking table ID: 12346
☁️ Annotations uploaded to OMERO
✨ Test completed successfully!
```

### Key Features Tested
- ✅ Package installation and imports
- ✅ YAML configuration loading and validation
- ✅ OMERO connection and authentication
- ✅ Container validation and image enumeration
- ✅ Tracking table creation and management
- ✅ Image data loading with dask
- ✅ Micro-SAM annotation pipeline execution
- ✅ Results upload to OMERO
- ✅ Progress tracking and error handling

## Next Steps

1. **Customize Configuration**: Modify the YAML files for your specific use case
2. **Run on Real Data**: Test with your own OMERO containers
3. **Model Comparison**: Try different micro-SAM models to compare results
4. **Batch Processing**: Process multiple datasets using different configurations
5. **Integration**: Integrate the pipeline into your existing workflows

## Support

For issues and questions:
- Check the main package documentation in `CLAUDE.md`
- Review the original notebook: `notebooks/omero-annotate-ai-batch.ipynb`
- Examine the source code in `src/omero_annotate_ai/`