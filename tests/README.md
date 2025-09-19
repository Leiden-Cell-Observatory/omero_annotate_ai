# Testing Guide

This guide provides instructions for running the tests for the `omero-annotate-ai` package. The test suite is built with `pytest` and managed with `pixi`.

## Test Environments

All test commands should be run from the root of the project using the `dev` environment in `pixi`.

## Running Tests

### Running All Tests

To run all tests, including unit and integration tests, use the following command. This requires Docker to be running for the integration tests.

```bash
pixi run -e dev pytest
```

### Running Unit Tests

To run only the unit tests, which are fast and do not require any external dependencies, use the `-m "unit"` marker:

```bash
pixi run -e dev pytest -m "unit"
```

### Running Integration Tests

To run the integration tests, which require a running OMERO server via Docker, use the `-m "integration"` marker:

```bash
pixi run -e dev pytest -m "integration"
```

**Note:** The integration tests use `docker-compose` to automatically start and stop an OMERO server. Ensure Docker is installed and running before executing these tests.

## Test Markers

The test suite uses `pytest` markers to categorize tests:

-   `@pytest.mark.unit`: For fast, self-contained unit tests.
-   `@pytest.mark.integration`: For tests that require external services, like a running OMERO server.
-   `@pytest.mark.omero`: A specific marker for tests that interact with an OMERO server.
-   `@pytest.mark.docker`: For tests that require Docker.

## Test Suite Status (As of last update)

The test suite has recently undergone a cleanup to remove obsolete tests and improve clarity. As a result, some tests may be temporarily failing while they are being updated to match the latest codebase.

| File | Purpose | Type | Status |
|---|---|---|---|
| `test_config.py` | Tests the configuration management system. | Unit | Passing |
| `test_file_io_functions.py` | Obsolete; functions removed. | - | No Tests |
| `test_image_functions.py` | Tests the image processing functions. | Unit | Passing |
| `test_installation.py` | Smoke tests for package installation and imports. | Unit | Passing |
| `test_omero_functions.py` | Tests the OMERO utility functions. | Unit | Passing |
| `test_omero_integration.py` | Tests the OMERO integration with a real server. | Integration/Unit | In Progress |
| `test_pipeline.py` | Tests the annotation pipeline. | Unit | In Progress |
| `test_training_functions.py` | Tests the training data preparation functions. | Unit | In Progress |
| `test_utils.py` | Obsolete; functions removed. | - | No Tests |
| `test_widgets.py` | Tests the widget classes. | Unit | In Progress |