# Testing Guide

## Running Tests

### Quick Commands
- **All tests**: `make test`
- **Fast tests only**: `make test-unit` 
- **OMERO tests**: `make test-integration` (requires Docker)

### Manual Commands
- **Unit tests**: `python -m pytest tests/ -v -m "unit"`
- **OMERO tests**: `python -m pytest tests/ -v -m "omero"`
- **Single test**: `python -m pytest tests/test_config.py::TestAnnotationConfig::test_default_config_creation -v`

## Test Files

| File | Purpose | Type |
|------|---------|------|
| `test_config.py` | Configuration system tests | Unit |
| `test_pipeline.py` | Main workflow tests | Unit |
| `test_omero_functions.py` | OMERO integration functions | Unit |
| `test_omero_integration.py` | Real OMERO server tests | Integration |
| `test_widgets.py` | UI component tests | Unit |
| `test_file_io_functions.py` | File operations tests | Unit |
| `test_image_functions.py` | Image processing tests | Unit |
| `test_utils.py` | Utility function tests | Unit |
| `test_installation.py` | Package import tests | Unit |

## Test Markers

- `@pytest.mark.unit` - Fast tests with mocks (no external dependencies)
- `@pytest.mark.omero` - Tests requiring OMERO server connection
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take longer than 5 seconds

## Common Test Patterns

### Using Shared Fixtures
```python
def test_something(sample_config, fake_omero_connection):
    # Use the shared config and connection
    pipeline = AnnotationPipeline(sample_config, fake_omero_connection)
    assert pipeline is not None
```

### Using Helper Functions
```python
def test_image_processing():
    # Create test objects easily
    mock_image = create_mock_image(image_id=123, name="test.tiff")
    mock_dataset = create_mock_dataset(dataset_id=456, name="test_dataset")
    
    # Check config is valid
    config = create_default_config()
    assert_config_has_required_fields(config)
```

### Working with Temporary Files
```python
def test_file_operations(temp_work_dir):
    # temp_work_dir is automatically cleaned up
    test_file = temp_work_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
```

## OMERO Integration Tests

Integration tests require a real OMERO server running via Docker:

```bash
# Start OMERO server
make start-omero
# Wait ~60 seconds for startup

# Run integration tests
make test-integration

# Stop OMERO server
make stop-omero
```

## Adding New Tests

1. **Create test file**: `test_new_feature.py`
2. **Add test class**: 
   ```python
   @pytest.mark.unit
   class TestNewFeature:
       def test_basic_functionality(self):
           # Your test here
           pass
   ```
3. **Use shared fixtures**: Import from `conftest.py`
4. **Run your tests**: `python -m pytest tests/test_new_feature.py -v`

## Tips for Basic Python Programmers

- **Start simple**: Write basic tests that check if functions work
- **Use mocks**: Replace complex dependencies with `Mock()` objects
- **Test edge cases**: What happens with empty inputs, None values, etc.
- **Keep tests focused**: One test should check one specific thing
- **Use descriptive names**: `test_config_creation_with_valid_data()`

## Troubleshooting

### Import Errors
```bash
# Make sure package is installed
pip install -e .
```

### OMERO Tests Failing
```bash
# Check if OMERO server is running
make start-omero
# Wait for startup, then try again
```

### Tests Taking Too Long
```bash
# Run only fast unit tests
make test-unit
```

This testing approach keeps things simple while ensuring good coverage and reliability.