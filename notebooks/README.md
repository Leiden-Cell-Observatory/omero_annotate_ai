# Example Tests

This folder contains example test scripts that demonstrate real-world usage of the omero-annotate-ai package.

## test_omero_connection_widget.py

A test script for the new OMERO connection widget with keychain support:

- Demonstrates secure password storage using OS keychain
- Tests configuration loading from .env and .ezomero files  
- Shows password expiration functionality
- Validates widget creation and integration
- Provides usage examples for Jupyter notebooks

### Usage

```bash
python example_test/test_omero_connection_widget.py
```

### Widget Features Demonstrated

1. **Secure Password Storage**: Passwords stored in OS-native keychain (Windows Credential Manager, macOS Keychain, Linux GNOME Keyring)
2. **Password Expiration**: Custom expiration times from 1 hour to permanent
3. **Configuration Auto-loading**: Automatically loads from .env and .ezomero files
4. **Connection Testing**: Test connection before saving credentials
5. **User-Friendly Interface**: Show/hide password, clear status messages

## test_annotation_pipeline.py

A comprehensive end-to-end test that demonstrates the complete micro-SAM annotation workflow:

- Creates a configuration with batch_size=0 and vit_b_lm model
- Connects to a real OMERO server
- Runs the full annotation pipeline on a test dataset
- Uploads ROIs and creates tracking tables
- Demonstrates proper table ID management

### Requirements

- Active OMERO server connection
- Valid `.env` file with OMERO credentials
- Test dataset (Dataset ID 351 by default)
- micro-sam installed via conda

### Usage

```bash
# Ensure you have the package installed
pip install -e .

# Run the test
python example_test/test_annotation_pipeline.py
```

### Key Features Demonstrated

1. **Configuration Creation**: Shows how to create and customize annotation configs
2. **OMERO Integration**: Demonstrates connection and dataset loading
3. **Pipeline Execution**: Full micro-SAM processing workflow
4. **Table Management**: Proper handling of OMERO table updates and ID changes
5. **Error Handling**: Robust error handling and cleanup

This test serves as both a validation tool and a reference implementation for users wanting to understand how to use the package in their own workflows.